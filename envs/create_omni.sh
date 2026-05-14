#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-omni}"

exec "$SCRIPT_DIR/create_conda_env.sh" omni "$ENV_NAME"
