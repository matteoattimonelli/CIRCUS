#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PY_EVAL="$SCRIPT_DIR/eval_subset.py"
PY_CIRCO="$SCRIPT_DIR/eval_subset_circo.py"
PY_EXPORT="$SCRIPT_DIR/export_embedding_cache_safetensors.py"
PY_APPEND="$SCRIPT_DIR/append_metrics_row.py"

ALLOWED_ABLATION_IMG="ablation_configs/image_only_zero_image.json"
ALLOWED_ABLATION_TXT="ablation_configs/text_only_drop_text.json"

MBEIR_ROOT="${MBEIR_ROOT:-$REPO_ROOT/../M-BEIR}"
SUBSETS_ROOT="${SUBSETS_ROOT:-$REPO_ROOT/final_dataset/query_jsonl}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
RAW_CACHE_ROOT="${RAW_CACHE_ROOT:-$SCRIPT_DIR/cache_subset_eval_raw}"
EXPORT_ROOT="${EXPORT_ROOT:-$SCRIPT_DIR/cache_subset_eval_export}"
METRICS_JSONL="${METRICS_JSONL:-}"
METRICS_DIR="${METRICS_DIR:-}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ENV_PREFIX="${ENV_PREFIX:-}"
EXPORT_DTYPE="${EXPORT_DTYPE:-float16}"
NORMALIZE=1
FORCE=0
EXPORT_ENABLED=1

DEFAULT_SUBSETS=(
  "composition_required"
  "shortcut_free"
)

DEFAULT_RETRIEVERS=(
  "retrievers.e5_omni_retriever"
  "retrievers.gme_qwen2vl_retriever"
  "retrievers.lamra_retriever"
  "retrievers.lamra_qwen25vl_retriever"
  "retrievers.mmembed_retriever"
  "retrievers.qwen3vl2b_vllm_retriever"
  "retrievers.qwen3vl8b_vllm_retriever"
  "retrievers.qwen3vl8b_vllm_retriever"
  "retrievers.rzen_embed_retriever"
  "retrievers.vlm2vec_v2_retriever"
)

DEFAULT_EXPERIMENTS=(
  "fashioniq:7:mbeir_fashioniq_task7_test.jsonl"
  "cirr:7:mbeir_cirr_task7_test.jsonl"
  "lasco:7:mbeir_lasco_task7_test.jsonl"
  "circo:7:mbeir_circo_task7_test.jsonl"
)

SUBSETS=()
RETRIEVERS=()
EXPERIMENTS=()
DATASETS=()
ABLATION_JSONS=()
SKIP_ABLATIONS=0

usage() {
  cat <<'EOF'
Usage:
  bash subset_evaluation/run_subset_eval.sh \
    --mbeir_root /data/M-BEIR \
    --subsets_root /path/to/subsets \
    --dataset cirr \
    --subset composition_required \
    --subset shortcut_free \
    --retriever retrievers.qwen3vl8b_vllm_retriever

Defaults:
  subsets:
    - composition_required
    - shortcut_free
  experiments:
    - fashioniq:7:mbeir_fashioniq_task7_test.jsonl
    - cirr:7:mbeir_cirr_task7_test.jsonl
    - lasco:7:mbeir_lasco_task7_test.jsonl
    - circo:7:mbeir_circo_task7_test.jsonl

Notes:
  - You can select datasets either with repeated --dataset flags or repeated
    --experiment DATASET:TASK_ID:QUERY_FILE flags. --dataset is the simpler path.
  - If you do not pass any --ablation_json, the launcher runs three variants:
    no ablation, ablation_configs/image_only_zero_image.json, and
    ablation_configs/text_only_drop_text.json.
  - If you do pass one or more --ablation_json flags, only those configs are run.
    ONLY the two shipped configs are accepted; any other path is rejected.
  - Regular datasets (fashioniq, cirr, lasco) use eval_subset.py
    with --query_source pointing at subsets/<dataset>/<subset>/*.jsonl.
  - CIRCO uses eval_subset_circo.py with --query_source so subset queries can
    live outside mbeir_root/query/test.
  - Raw cache tensors stay in RAW_CACHE_ROOT for eval reuse.
  - Float16 safetensors exports plus manifests are written to EXPORT_ROOT.
  - A structured metrics summary is appended to LOGDIR/results.jsonl by default.
  - Pass --no-export if you only want raw caches plus metrics JSON/JSONL and
    do not want safetensors exports.
EOF
}

dataset_experiment() {
  local dataset="${1,,}"
  case "$dataset" in
    fashioniq|fiq)
      printf '%s' "fashioniq:7:mbeir_fashioniq_task7_test.jsonl"
      ;;
    cirr)
      printf '%s' "cirr:7:mbeir_cirr_task7_test.jsonl"
      ;;
    lasco)
      printf '%s' "lasco:7:mbeir_lasco_task7_test.jsonl"
      ;;
    circo)
      printf '%s' "circo:7:mbeir_circo_task7_test.jsonl"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

# Validate that an ablation JSON path is one of the two shipped whitelisted
# configs. Resolves the path relative to REPO_ROOT and compares to the
# canonical absolute paths.
validate_ablation_json() {
  local raw="$1"
  local resolved=""

  if [[ -z "$raw" ]]; then
    return 0
  fi

  if [[ "$raw" == /* ]]; then
    resolved="$raw"
  else
    resolved="$REPO_ROOT/$raw"
  fi

  local allowed_img="$REPO_ROOT/$ALLOWED_ABLATION_IMG"
  local allowed_txt="$REPO_ROOT/$ALLOWED_ABLATION_TXT"

  if [[ "$resolved" != "$allowed_img" && "$resolved" != "$allowed_txt" ]]; then
    echo "Refusing --ablation_json $raw" >&2
    echo "Only these two ablation configs are allowed:" >&2
    echo "  - $ALLOWED_ABLATION_IMG" >&2
    echo "  - $ALLOWED_ABLATION_TXT" >&2
    exit 1
  fi
}

dataset_instruction() {
  local dataset="${1,,}"
  case "$dataset" in
    fashioniq|fiq|fashion200k)
      printf '%s' "Find a fashion image that aligns with the reference image and style note."
      ;;
    cirr|circo)
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
  local base_env=""
  case "$retriever" in
    retrievers.e5_omni_retriever)
      base_env="omni"
      ;;
    retrievers.gme_qwen2vl_retriever|retrievers.gme_qwen2vl_retriever)
      base_env="gme"
      ;;
    retrievers.lamra_retriever|retrievers.lamra_retriever)
      base_env="lamra"
      ;;
    retrievers.lamra_qwen25vl_retriever)
      base_env="lamra2"
      ;;
    retrievers.mmembed_retriever)
      base_env="mmemb"
      ;;
    retrievers.qwen3vl2b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever)
      base_env="qwen3emb"
      ;;
    retrievers.rzen_embed_retriever|retrievers.vlm2vec_v2_retriever)
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
    "UME_Q_INSTR=$instruction"
  )

  case "$retriever" in
    retrievers.gme_qwen2vl_retriever|retrievers.gme_qwen2vl_retriever)
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
    retrievers.qwen3vl2b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever|retrievers.qwen3vl8b_vllm_retriever)
      RUN_ENV_VARS+=(
        "VLLM_WORKER_MULTIPROC_METHOD=spawn"
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
        "QWEN3VL_T_INSTR=Represent the user's input."
      )
      ;;
    retrievers.lamra_retriever)
      RUN_ENV_VARS+=(
        "UNIME_BATCH_SIZE=8"
        "UNIME_DTYPE=float16"
        "UNIME_DEVICE_MAP=cuda"
        "UNIME_ATTN_IMPL=sdpa"
      )
      ;;
  esac
}

find_subset_jsonl() {
  local dataset="$1"
  local subset="$2"
  local subset_dir="$SUBSETS_ROOT/$dataset/$subset"

  if [[ ! -d "$subset_dir" ]]; then
    echo "Missing subset directory: $subset_dir" >&2
    return 1
  fi

  mapfile -t subset_files < <(
    find "$subset_dir" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.trace.jsonl' | sort
  )
  if [[ ${#subset_files[@]} -eq 0 ]]; then
    echo "No subset jsonl found under: $subset_dir" >&2
    return 1
  fi
  if [[ ${#subset_files[@]} -gt 1 ]]; then
    echo "Multiple subset jsonl files found under: $subset_dir" >&2
    printf '  %s\n' "${subset_files[@]}" >&2
    return 1
  fi

  printf '%s' "${subset_files[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mbeir_root)
      MBEIR_ROOT="$2"
      shift 2
      ;;
    --subsets_root)
      SUBSETS_ROOT="$2"
      shift 2
      ;;
    --subset)
      SUBSETS+=("$2")
      shift 2
      ;;
    --dataset)
      DATASETS+=("$2")
      shift 2
      ;;
    --experiment)
      EXPERIMENTS+=("$2")
      shift 2
      ;;
    --retriever)
      RETRIEVERS+=("$2")
      shift 2
      ;;
    --log_dir)
      LOGDIR="$2"
      shift 2
      ;;
    --raw_cache_root)
      RAW_CACHE_ROOT="$2"
      shift 2
      ;;
    --export_root)
      EXPORT_ROOT="$2"
      shift 2
      ;;
    --metrics_jsonl)
      METRICS_JSONL="$2"
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
    --ablation_json)
      validate_ablation_json "$2"
      ABLATION_JSONS+=("$2")
      shift 2
      ;;
    --no-ablations|--no_ablations|--skip-ablations)
      SKIP_ABLATIONS=1
      shift 1
      ;;
    --export_dtype)
      EXPORT_DTYPE="$2"
      shift 2
      ;;
    --env_prefix)
      ENV_PREFIX="${2%-}"
      shift 2
      ;;
    --no-export)
      EXPORT_ENABLED=0
      shift
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

ENV_PREFIX="${ENV_PREFIX%-}"

if [[ -z "$METRICS_JSONL" ]]; then
  METRICS_JSONL="$LOGDIR/results.jsonl"
fi
if [[ -z "$METRICS_DIR" ]]; then
  METRICS_DIR="$LOGDIR/metrics_artifacts"
fi

if [[ ${#SUBSETS[@]} -eq 0 ]]; then
  SUBSETS=("${DEFAULT_SUBSETS[@]}")
fi
if [[ ${#RETRIEVERS[@]} -eq 0 ]]; then
  RETRIEVERS=("${DEFAULT_RETRIEVERS[@]}")
fi
if [[ "$SKIP_ABLATIONS" -eq 1 ]]; then
  ABLATION_JSONS=("")
elif [[ ${#ABLATION_JSONS[@]} -eq 0 ]]; then
  ABLATION_JSONS=(
    ""
    "$ALLOWED_ABLATION_IMG"
    "$ALLOWED_ABLATION_TXT"
  )
fi
if [[ ${#EXPERIMENTS[@]} -eq 0 && ${#DATASETS[@]} -gt 0 ]]; then
  for dataset in "${DATASETS[@]}"; do
    EXPERIMENTS+=("$(dataset_experiment "$dataset")")
  done
fi
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
  mkdir -p "$LOGDIR" "$RAW_CACHE_ROOT" "$EXPORT_ROOT" "$METRICS_DIR"
else
  mkdir -p "$LOGDIR" "$RAW_CACHE_ROOT" "$METRICS_DIR"
fi
cd "$REPO_ROOT"

passed=0
failed=0
skipped=0
fail_list=()

total=$(( ${#EXPERIMENTS[@]} * ${#RETRIEVERS[@]} * ${#SUBSETS[@]} * ${#ABLATION_JSONS[@]} ))
current=0

for experiment in "${EXPERIMENTS[@]}"; do
  IFS=':' read -r dataset task_id base_query_file <<< "$experiment"
  if [[ -z "${dataset:-}" || -z "${task_id:-}" || -z "${base_query_file:-}" ]]; then
    echo "Bad --experiment value: $experiment" >&2
    echo "Expected DATASET:TASK_ID:QUERY_FILE" >&2
    exit 1
  fi

  instruction="$(dataset_instruction "$dataset")"
  dataset_l="${dataset,,}"

  for subset in "${SUBSETS[@]}"; do
    subset_src="$(find_subset_jsonl "$dataset_l" "$subset")" || exit 1
    subset_base="$(basename "$subset_src")"

    for retriever in "${RETRIEVERS[@]}"; do
      for ablation_json in "${ABLATION_JSONS[@]}"; do
        current=$((current + 1))

        conda_env="$(retriever_conda_env "$retriever")"
        short_retriever="${retriever##retrievers.}"
        short_retriever="${short_retriever%_retriever}"
        if [[ -n "$ablation_json" ]]; then
          ablation_tag="$(basename "${ablation_json%.json}")"
          ablation_json_label="$ablation_json"
        else
          ablation_tag="noablation"
          ablation_json_label="noablation"
        fi
        if [[ "$NORMALIZE" -eq 1 ]]; then
          norm_tag="norm"
        else
          norm_tag="nonorm"
        fi
        if [[ "$dataset_l" == "circo" ]]; then
          cache_variant_tag="${norm_tag}"
        else
          cache_variant_tag="${norm_tag}_qmode-both_qpart-full"
        fi
        if [[ "$dataset_l" == "circo" ]]; then
          variant_tag="${norm_tag}_atk-${ablation_tag}"
        else
          variant_tag="${norm_tag}_qmode-both_qpart-full_atk-${ablation_tag}"
        fi
        # gallery is never ablated by the released evaluator
        gallery_variant_tag="${norm_tag}_catalog"
        gallery_ablation_label="noablation"

        run_tag="${dataset_l}_task${task_id}_${subset}_${short_retriever}_${variant_tag}"
        log_path="$LOGDIR/${run_tag}.log"
        metrics_json_path="$METRICS_DIR/${run_tag}.json"
        run_cache_dir="$RAW_CACHE_ROOT/$dataset_l/$short_retriever/$cache_variant_tag"
        run_query_export_dir="$EXPORT_ROOT/$dataset_l/$subset/$short_retriever/$variant_tag"
        shared_gallery_export_dir="$EXPORT_ROOT/$dataset_l/_catalog/$short_retriever/$gallery_variant_tag"

        if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
          mkdir -p "$run_cache_dir" "$run_query_export_dir" "$shared_gallery_export_dir"
        else
          mkdir -p "$run_cache_dir"
        fi

        echo ""
        echo "================================================================"
        echo "[$current/$total] dataset=$dataset_l task=$task_id subset=$subset retriever=$short_retriever env=$conda_env"
        echo "ablation=${ablation_json_label}"
        echo "subset_src=$subset_src"
        echo "raw_cache_dir=$run_cache_dir"
        echo "metrics_json=$metrics_json_path"
        if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
          echo "query_export_dir=$run_query_export_dir"
          echo "gallery_export_dir=$shared_gallery_export_dir"
        else
          echo "export=disabled"
        fi
        echo "================================================================"

        if [[ "$FORCE" -eq 0 ]]; then
          if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
            if [[ -f "$metrics_json_path" ]] \
              && compgen -G "$shared_gallery_export_dir/gallery_*.safetensors" > /dev/null \
              && compgen -G "$run_query_export_dir/queries_*.safetensors" > /dev/null; then
              echo "SKIP: metrics and exported gallery/query safetensors already exist"
              skipped=$((skipped + 1))
              continue
            fi
          else
            if [[ -f "$metrics_json_path" ]]; then
              echo "SKIP: metrics json already exists"
              skipped=$((skipped + 1))
              continue
            fi
          fi
        fi

        build_env_vars "$retriever" "$instruction"

        if [[ "$dataset_l" == "circo" ]]; then
          eval_py="$PY_CIRCO"
        else
          eval_py="$PY_EVAL"
        fi

        query_file_arg="$subset_base"
        cmd=(
          conda run
          --no-capture-output
          -n "$conda_env"
          python
          "$eval_py"
          --mbeir_root "$MBEIR_ROOT"
          --dataset "$dataset_l"
          --query_file "$query_file_arg"
          --query_source "$subset_src"
          --task_id "$task_id"
          --retriever_module "$retriever"
          --retriever_class Retriever
          --device "$DEVICE"
          --batch_size "$BATCH_SIZE"
          --cache_dir "$run_cache_dir"
          --num_workers "$NUM_WORKERS"
          --metrics_out "$metrics_json_path"
        )
        if [[ -n "$ablation_json" ]]; then
          cmd+=(--ablation_json "$ablation_json")
        fi
        if [[ "$NORMALIZE" -eq 1 ]]; then
          cmd+=(--normalize)
        fi

        export_cmd=(
          conda run
          --no-capture-output
          -n "$conda_env"
          python
          "$PY_EXPORT"
          --cache_dir "$run_cache_dir"
          --output_dir "$shared_gallery_export_dir"
          --dataset "$dataset_l"
          --task_id "$task_id"
          --subset_name "_catalog"
          --retriever_module "$retriever"
          --retriever_class Retriever
          --query_file "$query_file_arg"
          --query_source "$subset_src"
          --eval_script "$(basename "$eval_py")"
          --ablation_json "$gallery_ablation_label"
          --export_dtype "$EXPORT_DTYPE"
          --kind gallery
        )
        export_queries_cmd=(
          conda run
          --no-capture-output
          -n "$conda_env"
          python
          "$PY_EXPORT"
          --cache_dir "$run_cache_dir"
          --output_dir "$run_query_export_dir"
          --dataset "$dataset_l"
          --task_id "$task_id"
          --subset_name "$subset"
          --retriever_module "$retriever"
          --retriever_class Retriever
          --query_file "$query_file_arg"
          --query_source "$subset_src"
          --eval_script "$(basename "$eval_py")"
          --ablation_json "$ablation_json_label"
          --export_dtype "$EXPORT_DTYPE"
          --kind queries
        )
        if [[ "$dataset_l" == "circo" ]]; then
          export_cmd+=(--multi_positive)
          export_queries_cmd+=(--multi_positive)
        fi

        append_cmd=(
          conda run
          --no-capture-output
          -n "$conda_env"
          python
          "$PY_APPEND"
          --metrics_json "$metrics_json_path"
          --output_jsonl "$METRICS_JSONL"
          --dataset "$dataset_l"
          --task_id "$task_id"
          --subset_name "$subset"
          --retriever_module "$retriever"
          --retriever_class Retriever
          --ablation_json "$ablation_json_label"
          --ablation_tag "$ablation_tag"
          --query_file "$query_file_arg"
          --query_source "$subset_src"
          --eval_script "$(basename "$eval_py")"
          --log_path "$log_path"
          --raw_cache_dir "$run_cache_dir"
          --variant_tag "$variant_tag"
        )
        if [[ "$dataset_l" == "circo" ]]; then
          append_cmd+=(--multi_positive)
        fi
        if [[ "$NORMALIZE" -eq 1 ]]; then
          append_cmd+=(--normalize)
        fi
        if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
          append_cmd+=(
            --export_enabled
            --gallery_export_dir "$shared_gallery_export_dir"
            --query_export_dir "$run_query_export_dir"
          )
        fi

        if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
          if [[ "$NORMALIZE" -eq 1 ]]; then
            export_cmd+=(--normalize)
            export_queries_cmd+=(--normalize)
          fi
          if [[ "$FORCE" -eq 1 ]]; then
            export_cmd+=(--force)
            export_queries_cmd+=(--force)
          fi
        fi

        if env "${RUN_ENV_VARS[@]}" "${cmd[@]}" 2>&1 | tee "$log_path"; then
          if [[ "$EXPORT_ENABLED" -eq 1 ]]; then
            if env "${RUN_ENV_VARS[@]}" "${export_cmd[@]}" 2>&1 | tee -a "$log_path" \
              && env "${RUN_ENV_VARS[@]}" "${export_queries_cmd[@]}" 2>&1 | tee -a "$log_path" \
              && env "${RUN_ENV_VARS[@]}" "${append_cmd[@]}" 2>&1 | tee -a "$log_path"; then
              echo "PASS: $dataset_l / task$task_id / $subset / $short_retriever / $ablation_tag"
              passed=$((passed + 1))
            else
              echo "FAIL postprocess: $dataset_l / task$task_id / $subset / $short_retriever / $ablation_tag  (see $log_path)"
              failed=$((failed + 1))
              fail_list+=("$dataset_l/task$task_id/$subset/$short_retriever/$ablation_tag/postprocess")
            fi
          else
            if env "${RUN_ENV_VARS[@]}" "${append_cmd[@]}" 2>&1 | tee -a "$log_path"; then
              echo "PASS: $dataset_l / task$task_id / $subset / $short_retriever / $ablation_tag"
              passed=$((passed + 1))
            else
              echo "FAIL metrics: $dataset_l / task$task_id / $subset / $short_retriever / $ablation_tag  (see $log_path)"
              failed=$((failed + 1))
              fail_list+=("$dataset_l/task$task_id/$subset/$short_retriever/$ablation_tag/metrics")
            fi
          fi
        else
          echo "FAIL eval: $dataset_l / task$task_id / $subset / $short_retriever / $ablation_tag  (see $log_path)"
          failed=$((failed + 1))
          fail_list+=("$dataset_l/task$task_id/$subset/$short_retriever/$ablation_tag/eval")
        fi
      done
    done
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
