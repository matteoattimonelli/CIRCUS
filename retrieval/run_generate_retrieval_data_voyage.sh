#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$REPO_ROOT/retrieval/generate_retrieval_data.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
RETRIEVER_MODULE="${RETRIEVER_MODULE:-retrievers.voyage_multimodal_35_retriever}"
RETRIEVER_CLASS="${RETRIEVER_CLASS:-Retriever}"

MBEIR_ROOT="${MBEIR_ROOT:-$REPO_ROOT/../M-BEIR}"
OUTDIR="${OUTDIR:-$REPO_ROOT/retrieval/retrieval_results_voyage}"
LOGDIR="${LOGDIR:-$REPO_ROOT/retrieval/logs_voyage}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/retrieval/embedding_cache_voyage}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
K="${K:-10}"
TOP_K="${TOP_K:-20}"
MAX_QUERIES="${MAX_QUERIES:-}"
SMOKE_GALLERY_SIZE="${SMOKE_GALLERY_SIZE:-}"
NORMALIZE=1
FORCE=0

DEFAULT_EXPERIMENTS=(
  "fashioniq:7:mbeir_fashioniq_task7_test.jsonl"
  "cirr:7:mbeir_cirr_task7_test.jsonl"
)

EXPERIMENTS=()

usage() {
  cat <<'EOF'
Usage:
  export VOYAGE_API_KEY=...
  bash retrieval/run_generate_retrieval_data_voyage.sh \
    --mbeir_root /data/M-BEIR \
    --experiment fashioniq:7:mbeir_fashioniq_task7_test.jsonl \
    --experiment cirr:7:mbeir_cirr_task7_test.jsonl

If no --experiment is provided, the script runs:
  - fashioniq:7:mbeir_fashioniq_task7_test.jsonl
  - cirr:7:mbeir_cirr_task7_test.jsonl

Options:
  --mbeir_root PATH
  --experiment DATASET:TASK_ID:QUERY_FILE
  --output_dir PATH
  --log_dir PATH
  --cache_root PATH
  --device DEVICE
  --batch_size INT
  --k INT
  --top_k INT
  --max_queries INT
  --smoke_gallery_size INT
  --python BIN
  --force
  --no-normalize

Relevant environment variables:
  VOYAGE_API_KEY
  VOYAGE_MODEL
  VOYAGE_BATCH_SIZE
  VOYAGE_OUTPUT_DIMENSION
  VOYAGE_MAX_RETRIES
  VOYAGE_TIMEOUT_S
EOF
}

dataset_instruction() {
  local dataset="${1,,}"
  case "$dataset" in
    fashioniq|fiq|fashion200k)
      printf '%s' "Find a fashion image that aligns with the reference image and style note."
      ;;
    cirr)
      printf '%s' "Retrieve a day-to-day image that aligns with the modification instructions of the provided image."
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
    --k)
      K="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --max_queries)
      MAX_QUERIES="$2"
      shift 2
      ;;
    --smoke_gallery_size)
      SMOKE_GALLERY_SIZE="$2"
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

if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "Set VOYAGE_API_KEY before running this script." >&2
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
  IFS=':' read -r dataset task_id query_file <<< "$experiment"
  if [[ -z "${dataset:-}" || -z "${task_id:-}" || -z "${query_file:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    echo "Expected DATASET:TASK_ID:QUERY_FILE" >&2
    exit 1
  fi

  current=$((current + 1))
  instruction="$(dataset_instruction "$dataset")"

  short_retriever="${RETRIEVER_MODULE##retrievers.}"
  short_retriever="${short_retriever%_retriever}"
  model_tag="${VOYAGE_MODEL:-voyage-multimodal-3.5}"
  model_tag="${model_tag//\//__}"
  dim_tag="${VOYAGE_OUTPUT_DIMENSION:-default}"
  run_suffix=""
  if [[ -n "$MAX_QUERIES" ]]; then
    run_suffix="_maxq-${MAX_QUERIES}"
  fi
  if [[ -n "$SMOKE_GALLERY_SIZE" ]]; then
    run_suffix="${run_suffix}_maxg-${SMOKE_GALLERY_SIZE}"
  fi
  outfile="${OUTDIR}/retrieval_data_${dataset}_task${task_id}_${short_retriever}${run_suffix}.json"
  logfile="${LOGDIR}/${dataset}_task${task_id}_${short_retriever}${run_suffix}.log"
  cache_dir="${CACHE_ROOT}/${dataset}_task${task_id}_${short_retriever}_${model_tag}_dim-${dim_tag}${run_suffix}"

  echo ""
  echo "================================================================"
  echo "[$current/$total] dataset=$dataset task=$task_id retriever=$short_retriever"
  echo "query_file=$query_file"
  echo "cache_dir=$cache_dir"
  if [[ -n "$instruction" ]]; then
    echo "query_instruction=$instruction"
  else
    echo "query_instruction=<empty>"
  fi
  echo "================================================================"

  if [[ "$FORCE" -eq 0 && -f "$outfile" ]]; then
    echo "SKIP: output already exists at $outfile"
    skipped=$((skipped + 1))
    continue
  fi

  voyage_batch_size="${VOYAGE_BATCH_SIZE:-$BATCH_SIZE}"
  run_env_vars=(
    "TOKENIZERS_PARALLELISM=false"
    "VOYAGE_BATCH_SIZE=$voyage_batch_size"
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
    --k "$K"
    --top_k "$TOP_K"
    --output "$outfile"
    --cache_dir "$cache_dir"
    --query_instruction "$instruction"
  )

  if [[ -n "$MAX_QUERIES" ]]; then
    cmd+=(--max_queries "$MAX_QUERIES")
  fi
  if [[ -n "$SMOKE_GALLERY_SIZE" ]]; then
    cmd+=(--smoke_gallery_size "$SMOKE_GALLERY_SIZE")
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
