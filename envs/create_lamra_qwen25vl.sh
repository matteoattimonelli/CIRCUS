#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-lamra2}"

exec "$SCRIPT_DIR/create_conda_env.sh" lamra_qwen25vl "$ENV_NAME"
