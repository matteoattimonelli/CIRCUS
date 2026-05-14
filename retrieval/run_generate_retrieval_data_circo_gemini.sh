#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$REPO_ROOT/retrieval/generate_retrieval_data_circo.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
RETRIEVER_MODULE="${RETRIEVER_MODULE:-retrievers.gemini_embedding_2_retriever}"
RETRIEVER_CLASS="${RETRIEVER_CLASS:-Retriever}"

MBEIR_ROOT="${MBEIR_ROOT:-$REPO_ROOT/../circo_mbeir}"
OUTDIR="${OUTDIR:-$REPO_ROOT/retrieval/retrieval_results_circo_gemini}"
LOGDIR="${LOGDIR:-$REPO_ROOT/retrieval/logs_circo_gemini}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/retrieval/embedding_cache_circo_gemini}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
K="${K:-10}"
TOP_K="${TOP_K:-20}"
QRELS_PATH="${QRELS_PATH:-}"
NORMALIZE=1
FORCE=0

DEFAULT_EXPERIMENTS=(
  "circo:7:test:circo_test.jsonl"
)

EXPERIMENTS=()

usage() {
  cat <<'EOF'
Usage:
  export GEMINI_API_KEY=...
  bash retrieval/run_generate_retrieval_data_circo_gemini.sh

By default this runs:
  --mbeir_root <path_to_circo_mbeir>
  --experiment circo:7:test:circo_test.jsonl

Options:
  --mbeir_root PATH
  --experiment DATASET:TASK_ID:QUERY_FILE
  --experiment DATASET:TASK_ID:QUERY_SPLIT:QUERY_FILE
  --qrels_path PATH
  --output_dir PATH
  --log_dir PATH
  --cache_root PATH
  --device DEVICE
  --batch_size INT
  --num_workers INT
  --k INT
  --top_k INT
  --python BIN
  --force
  --no-normalize

Relevant environment variables:
  GEMINI_API_KEY or GOOGLE_API_KEY
  GEMINI_EMBED_MODEL
  GEMINI_EMBED_BATCH_SIZE
  GEMINI_OUTPUT_DIMENSIONALITY
  GEMINI_EMBED_MAX_RETRIES
  GEMINI_EMBED_TIMEOUT_S
EOF
}

dataset_instruction() {
  local dataset="${1,,}"
  case "$dataset" in
    circo|cirr)
      printf '%s' "Retrieve a day-to-day image that aligns with the modification instructions of the provided image."
      ;;
    fashioniq|fiq|fashion200k)
      printf '%s' "Find a fashion image that aligns with the reference image and style note."
      ;;
    lasco)
      printf '%s' ""
      ;;
    *)
      printf '%s' "Retrieve the target image that best matches the reference image and the textual modification."
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mbeir_root)
      MBEIR_ROOT="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENTS+=("$2")
      shift 2
      ;;
    --qrels_path)
      QRELS_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTDIR="$2"
      shift 2
      ;;
    --log_dir)
      LOGDIR="$2"
      shift 2
      ;;
    --cache_root)
      CACHE_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-normalize)
      NORMALIZE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Set GEMINI_API_KEY or GOOGLE_API_KEY before running this script." >&2
  exit 1
fi

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

mkdir -p "$OUTDIR" "$LOGDIR" "$CACHE_ROOT"
cd "$REPO_ROOT"

passed=0
failed=0
skipped=0
fail_list=()

total=${#EXPERIMENTS[@]}
current=0

for experiment in "${EXPERIMENTS[@]}"; do
  dataset=""
  task_id=""
  query_split=""
  query_file=""
  IFS=':' read -r field1 field2 field3 field4 extra <<< "$experiment"
  if [[ -n "${extra:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    echo "Expected DATASET:TASK_ID:QUERY_FILE or DATASET:TASK_ID:QUERY_SPLIT:QUERY_FILE" >&2
    exit 1
  fi
  if [[ -n "${field4:-}" ]]; then
    dataset="$field1"
    task_id="$field2"
    query_split="$field3"
    query_file="$field4"
  else
    dataset="$field1"
    task_id="$field2"
    query_file="$field3"
  fi
  if [[ -z "${dataset:-}" || -z "${task_id:-}" || -z "${query_file:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    echo "Expected DATASET:TASK_ID:QUERY_FILE or DATASET:TASK_ID:QUERY_SPLIT:QUERY_FILE" >&2
    exit 1
  fi

  current=$((current + 1))
  instruction="$(dataset_instruction "$dataset")"

  short_retriever="${RETRIEVER_MODULE##retrievers.}"
  short_retriever="${short_retriever%_retriever}"
  model_tag="${GEMINI_EMBED_MODEL:-models/gemini-embedding-2-preview}"
  model_tag="${model_tag//\//__}"
  dim_tag="${GEMINI_OUTPUT_DIMENSIONALITY:-default}"
  split_tag="${query_split:-auto}"
  outfile="${OUTDIR}/retrieval_data_${dataset}_task${task_id}_${short_retriever}.json"
  logfile="${LOGDIR}/${dataset}_task${task_id}_${short_retriever}.log"
  cache_dir="${CACHE_ROOT}/${dataset}_split-${split_tag}_task${task_id}_${short_retriever}_${model_tag}_dim-${dim_tag}"

  echo ""
  echo "================================================================"
  echo "[$current/$total] dataset=$dataset task=$task_id retriever=$short_retriever"
  if [[ -n "$query_split" ]]; then
    echo "query_split=$query_split"
  fi
  echo "query_file=$query_file"
  echo "cache_dir=$cache_dir"
  if [[ -n "$instruction" ]]; then
    echo "query_instruction=$instruction"
  else
    echo "query_instruction=<empty>"
  fi
  if [[ -n "$QRELS_PATH" ]]; then
    echo "qrels_path=$QRELS_PATH"
  fi
  echo "================================================================"

  if [[ "$FORCE" -eq 0 && -f "$outfile" ]]; then
    echo "SKIP: output already exists at $outfile"
    skipped=$((skipped + 1))
    continue
  fi

  gemini_batch_size="${GEMINI_EMBED_BATCH_SIZE:-$BATCH_SIZE}"
  run_env_vars=(
    "TOKENIZERS_PARALLELISM=false"
    "GEMINI_EMBED_BATCH_SIZE=$gemini_batch_size"
  )

  cmd=(
    "$PYTHON_BIN"
    "$PY_SCRIPT"
    --mbeir_root "$MBEIR_ROOT"
    --dataset "$dataset"
    --task_id "$task_id"
    --query_file "$query_file"
    --retriever_module "$RETRIEVER_MODULE"
    --retriever_class "$RETRIEVER_CLASS"
    --device "$DEVICE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --k "$K"
    --top_k "$TOP_K"
    --output "$outfile"
    --cache_dir "$cache_dir"
    --query_instruction "$instruction"
  )

  if [[ -n "$query_split" ]]; then
    cmd+=(--query_split "$query_split")
  fi
  if [[ -n "$QRELS_PATH" ]]; then
    cmd+=(--qrels_path "$QRELS_PATH")
  fi
  if [[ "$NORMALIZE" -eq 1 ]]; then
    cmd+=(--normalize)
  fi

  env "${run_env_vars[@]}" "${cmd[@]}" 2>&1 | tee "$logfile"

  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo "PASS: $dataset / task$task_id / $short_retriever"
    passed=$((passed + 1))
  else
    echo "FAIL: $dataset / task$task_id / $short_retriever  (see $logfile)"
    failed=$((failed + 1))
    fail_list+=("$dataset/task$task_id/$short_retriever")
  fi
done

echo ""
echo "================================================================"
echo "SUMMARY: $passed passed, $failed failed, $skipped skipped (of $total total)"
echo "================================================================"
if [[ ${#fail_list[@]} -gt 0 ]]; then
  echo "Failed runs:"
  for item in "${fail_list[@]}"; do
    echo "  - $item"
  done
fi
