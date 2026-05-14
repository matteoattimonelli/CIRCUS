#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_RUNNER="$REPO_ROOT/retrieval/run_generate_retrieval_data_voyage.sh"

DEFAULT_MBEIR_ROOT="${REPO_ROOT}/../lasco_mbeir"
DEFAULT_OUTDIR="${REPO_ROOT}/retrieval/retrieval_results_lasco_voyage"
DEFAULT_LOGDIR="${REPO_ROOT}/retrieval/logs_lasco_voyage"
DEFAULT_CACHE_ROOT="${REPO_ROOT}/retrieval/embedding_cache_lasco_voyage"
DEFAULT_EXPERIMENT="lasco:7:lasco_test_task7_test.jsonl"

usage() {
  cat <<'EOF'
Usage:
  export VOYAGE_API_KEY=...
  bash retrieval/run_generate_retrieval_data_lasco_voyage.sh

By default this runs:
  --mbeir_root <path_to_lasco_mbeir>
  --experiment lasco:7:lasco_test_task7_test.jsonl

Any extra arguments are forwarded to:
  bash retrieval/run_generate_retrieval_data_voyage.sh

Examples:
  bash retrieval/run_generate_retrieval_data_lasco_voyage.sh

  BATCH_SIZE=4 VOYAGE_BATCH_SIZE=4 \
  bash retrieval/run_generate_retrieval_data_lasco_voyage.sh \
    --max_queries 20 \
    --smoke_gallery_size 512
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

has_experiment=0
for arg in "$@"; do
  if [[ "$arg" == "--experiment" ]]; then
    has_experiment=1
    break
  fi
done

cmd=(
  bash
  "$BASE_RUNNER"
  --mbeir_root "${MBEIR_ROOT:-$DEFAULT_MBEIR_ROOT}"
  --output_dir "${OUTDIR:-$DEFAULT_OUTDIR}"
  --log_dir "${LOGDIR:-$DEFAULT_LOGDIR}"
  --cache_root "${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
)

if [[ "$has_experiment" -eq 0 ]]; then
  cmd+=(--experiment "$DEFAULT_EXPERIMENT")
fi

cmd+=("$@")
"${cmd[@]}"
