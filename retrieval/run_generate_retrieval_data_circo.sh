#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/generate_retrieval_data_circo.py"

MBEIR_ROOT="${MBEIR_ROOT:-$REPO_ROOT/../M-BEIR}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/retrieval_results_circo}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_circo}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
K="${K:-10}"
TOP_K="${TOP_K:-20}"
QRELS_PATH="${QRELS_PATH:-}"
ENV_PREFIX="${ENV_PREFIX:-}"
NORMALIZE=1
FORCE=0

DEFAULT_RETRIEVERS=(
  "retrievers.e5_omni_retriever"
  "retrievers.gme_qwen2vl_retriever"
  "retrievers.lamra_retriever"
  "retrievers.lamra_qwen25vl_retriever"
  "retrievers.mmembed_retriever"
  "retrievers.qwen3vl2b_vllm_retriever"
  "retrievers.qwen3vl8b_vllm_retriever"
  "retrievers.rzen_embed_retriever"
  "retrievers.vlm2vec_v2_retriever"
)

DEFAULT_EXPERIMENTS=(
  "circo:7:mbeir_circo_task7_test.jsonl"
)

RETRIEVERS=()
EXPERIMENTS=()

usage() {
  cat <<'EOF'
Usage:
  bash retrieval/run_generate_retrieval_data_circo.sh \
    --mbeir_root /data/M-BEIR \
    --experiment circo:7:mbeir_circo_task7_test.jsonl

If no --experiment is provided, the script runs:
  - circo:7:mbeir_circo_task7_test.jsonl

If no --retriever is provided, it runs the open-source retrievers from the paper.

Options:
  --mbeir_root PATH
  --experiment DATASET:TASK_ID:QUERY_FILE
  --experiment DATASET:TASK_ID:QUERY_SPLIT:QUERY_FILE
  --retriever MODULE
  --env_prefix PREFIX
  --qrels_path PATH
  --output_dir PATH
  --log_dir PATH
  --device DEVICE
  --batch_size INT
  --num_workers INT
  --k INT
  --top_k INT
  --force
  --no-normalize
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

retriever_conda_env() {
  local retriever="$1"
  local base_env=""
  case "$retriever" in
      base_env="omni"
      ;;
    retrievers.gme_qwen2vl_retriever)
      base_env="gme"
      ;;
    retrievers.lamra_retriever)
      base_env="lamra"
      ;;
    retrievers.lamra_qwen25vl_retriever)
      base_env="lamra2"
      ;;
    retrievers.mmembed_retriever)
      base_env="mmemb"
      ;;
    retrievers.qwen3vl2b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever|retrievers.rzen_embed_retriever|retrievers.vlm2vec_v2_retriever)
      base_env="qwen3emb"
      ;;
    *)
      echo "Unsupported retriever module: $retriever" >&2
      exit 1
      ;;
  esac

  if [[ -n "$ENV_PREFIX" ]]; then
    printf '%s-%s' "$ENV_PREFIX" "$base_env"
  else
    printf '%s' "$base_env"
  fi
}

build_env_vars() {
  local retriever="$1"
  local instruction="$2"

  RUN_ENV_VARS=(
    "HF_HUB_DISABLE_PROGRESS_BARS=1"
    "TOKENIZERS_PARALLELISM=false"
    "Q_INSTRUCTION=$instruction"
    "GME_Q_INSTR=$instruction"
  )

  case "$retriever" in
    retrievers.gme_qwen2vl_retriever)
      RUN_ENV_VARS+=(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
        "GME_USE_FUSED=1"
        "GME_BATCH_SIZE=32"
        "GME_T_IS_QUERY=0"
        "GME_DEVICE_MAP=cuda"
        "GME_DTYPE=bfloat16"
      )
      ;;
    retrievers.mmembed_retriever)
      RUN_ENV_VARS+=(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
      )
      ;;
    retrievers.qwen3vl2b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever)
      RUN_ENV_VARS+=(
        "VLLM_WORKER_MULTIPROC_METHOD=spawn"
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
        "QWEN3VL_T_INSTR=Represent the user's input."
      )
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mbeir_root) MBEIR_ROOT="$2"; shift 2 ;;
    --experiment) EXPERIMENTS+=("$2"); shift 2 ;;
    --retriever|--retriever) RETRIEVERS+=("$2"); shift 2 ;;
    --env_prefix) ENV_PREFIX="${2%-}"; shift 2 ;;
    --qrels_path) QRELS_PATH="$2"; shift 2 ;;
    --output_dir) OUTDIR="$2"; shift 2 ;;
    --log_dir) LOGDIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-normalize) NORMALIZE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

ENV_PREFIX="${ENV_PREFIX%-}"

if [[ ${#RETRIEVERS[@]} -eq 0 ]]; then
  RETRIEVERS=("${DEFAULT_RETRIEVERS[@]}")
fi
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

mkdir -p "$OUTDIR" "$LOGDIR"
cd "$REPO_ROOT"

passed=0
failed=0
skipped=0
fail_list=()

total=$(( ${#EXPERIMENTS[@]} * ${#RETRIEVERS[@]} ))
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
    dataset="$field1"; task_id="$field2"; query_split="$field3"; query_file="$field4"
  else
    dataset="$field1"; task_id="$field2"; query_file="$field3"
  fi
  if [[ -z "${dataset:-}" || -z "${task_id:-}" || -z "${query_file:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    exit 1
  fi

  instruction="$(dataset_instruction "$dataset")"

  for retriever in "${RETRIEVERS[@]}"; do
    current=$((current + 1))

    conda_env="$(retriever_conda_env "$retriever")"
    build_env_vars "$retriever" "$instruction"

    short_retriever="${retriever##retrievers.}"
    short_retriever="${short_retriever%_retriever}"
    outfile="${OUTDIR}/retrieval_data_${dataset}_task${task_id}_${short_retriever}.json"
    logfile="${LOGDIR}/${dataset}_task${task_id}_${short_retriever}.log"

    echo ""
    echo "================================================================"
    echo "[$current/$total] dataset=$dataset task=$task_id retriever=$short_retriever env=$conda_env"
    if [[ -n "$query_split" ]]; then echo "query_split=$query_split"; fi
    echo "query_file=$query_file"
    if [[ -n "$instruction" ]]; then echo "query_instruction=$instruction"; else echo "query_instruction=<empty>"; fi
    if [[ -n "$QRELS_PATH" ]]; then echo "qrels_path=$QRELS_PATH"; fi
    echo "================================================================"

    if [[ "$FORCE" -eq 0 && -f "$outfile" ]]; then
      echo "SKIP: output already exists at $outfile"
      skipped=$((skipped + 1))
      continue
    fi

    cmd=(
      conda run --no-capture-output -n "$conda_env" python "$PY_SCRIPT"
      --mbeir_root "$MBEIR_ROOT"
      --dataset "$dataset"
      --task_id "$task_id"
      --query_file "$query_file"
      --retriever_module "$retriever"
      --retriever_class Retriever
      --device "$DEVICE"
      --batch_size "$BATCH_SIZE"
      --num_workers "$NUM_WORKERS"
      --k "$K"
      --top_k "$TOP_K"
      --output "$outfile"
      --query_instruction "$instruction"
    )

    if [[ -n "$query_split" ]]; then cmd+=(--query_split "$query_split"); fi
    if [[ -n "$QRELS_PATH" ]]; then cmd+=(--qrels_path "$QRELS_PATH"); fi
    if [[ "$NORMALIZE" -eq 1 ]]; then cmd+=(--normalize); fi

    env "${RUN_ENV_VARS[@]}" "${cmd[@]}" 2>&1 | tee "$logfile"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
      echo "PASS: $dataset / task$task_id / $short_retriever"
      passed=$((passed + 1))
    else
      echo "FAIL: $dataset / task$task_id / $short_retriever  (see $logfile)"
      failed=$((failed + 1))
      fail_list+=("$dataset/task$task_id/$short_retriever")
    fi
  done
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
