#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/generate_retrieval_data.py"

MBEIR_ROOT="${MBEIR_ROOT:-$REPO_ROOT/../M-BEIR}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/retrieval_results}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
K="${K:-10}"
TOP_K="${TOP_K:-20}"
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
  "fashioniq:7:mbeir_fashioniq_task7_test.jsonl"
  "cirr:7:mbeir_cirr_task7_test.jsonl"
  "lasco:7:lasco_test_task7_test.jsonl"
)

RETRIEVERS=()
EXPERIMENTS=()

usage() {
  cat <<'EOF'
Usage:
  bash retrieval/run_generate_retrieval_data.sh \
    --mbeir_root /data/M-BEIR \
    --experiment fashioniq:7:mbeir_fashioniq_task7_test.jsonl \
    --experiment cirr:7:mbeir_cirr_task7_test.jsonl \
    --experiment lasco:7:lasco_test_task7_test.jsonl

If no --experiment is provided, the script runs:
  - fashioniq:7:mbeir_fashioniq_task7_test.jsonl
  - cirr:7:mbeir_cirr_task7_test.jsonl
  - lasco:7:lasco_test_task7_test.jsonl

If no --retriever is provided, it runs the open-source retrievers from the paper:
  - retrievers.e5_omni_retriever
  - retrievers.gme_qwen2vl_retriever
  - retrievers.lamra_retriever
  - retrievers.lamra_qwen25vl_retriever
  - retrievers.mmembed_retriever
  - retrievers.qwen3vl2b_vllm_retriever
  - retrievers.qwen3vl8b_vllm_retriever
  - retrievers.rzen_embed_retriever
  - retrievers.vlm2vec_v2_retriever

For Gemini Embedding 2 and Voyage MM-3.5 (commercial APIs) use the dedicated
launchers: run_generate_retrieval_data_gemini.sh and run_generate_retrieval_data_voyage.sh.
For CIRCO (different evaluator), use run_generate_retrieval_data_circo.sh.

Options:
  --mbeir_root PATH
  --experiment DATASET:TASK_ID:QUERY_FILE
  --retriever MODULE
  --output_dir PATH
  --log_dir PATH
  --device DEVICE
  --batch_size INT
  --k INT
  --top_k INT
  --force
  --no-normalize
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

retriever_conda_env() {
  local retriever="$1"
  case "$retriever" in
    retrievers.e5_omni_retriever)
      printf '%s' "omni"
      ;;
    retrievers.gme_qwen2vl_retriever)
      printf '%s' "gme"
      ;;
    retrievers.lamra_retriever)
      printf '%s' "lamra"
      ;;
    retrievers.lamra_qwen25vl_retriever)
      printf '%s' "lamra2"
      ;;
    retrievers.mmembed_retriever)
      printf '%s' "mmemb"
      ;;
    retrievers.qwen3vl2b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever|retrievers.rzen_embed_retriever|retrievers.vlm2vec_v2_retriever)
      printf '%s' "qwen3emb"
      ;;
    *)
      echo "Unsupported retriever module: $retriever" >&2
      exit 1
      ;;
  esac
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
    --mbeir_root)
      MBEIR_ROOT="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENTS+=("$2")
      shift 2
      ;;
    --retriever|--retriever)
      RETRIEVERS+=("$2")
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
  IFS=':' read -r dataset task_id query_file <<< "$experiment"
  if [[ -z "${dataset:-}" || -z "${task_id:-}" || -z "${query_file:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    echo "Expected DATASET:TASK_ID:QUERY_FILE" >&2
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
    echo "query_file=$query_file"
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

    cmd=(
      conda run
      --no-capture-output
      -n "$conda_env"
      python
      "$PY_SCRIPT"
      --mbeir_root "$MBEIR_ROOT"
      --dataset "$dataset"
      --task_id "$task_id"
      --query_file "$query_file"
      --retriever_module "$retriever"
      --retriever_class Retriever
      --device "$DEVICE"
      --batch_size "$BATCH_SIZE"
      --k "$K"
      --top_k "$TOP_K"
      --output "$outfile"
      --query_instruction "$instruction"
    )

    if [[ "$NORMALIZE" -eq 1 ]]; then
      cmd+=(--normalize)
    fi

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
