#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-qwen3emb}"
PY_SCRIPT="$SCRIPT_DIR/run_final_evaluation.py"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"

mkdir -p "$LOGDIR" "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$LOGDIR/final_evaluation_${timestamp}.log"

if [[ -n "$CONDA_ENV" ]]; then
  CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  CMD=("$PYTHON_BIN")
fi

echo "Running final_evaluation -> $log_path"
echo "Python command: ${CMD[*]}"
PYTHONUNBUFFERED=1 "${CMD[@]}" "$PY_SCRIPT" --output_dir "$OUTPUT_DIR" "$@" 2>&1 | tee "$log_path"
