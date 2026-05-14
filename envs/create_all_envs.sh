#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACKS=(gme lamra lamra_qwen25vl mmemb omni qwen3emb)
NAME_PREFIX="${1:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

env_exists() {
  local env_name="$1"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"
}

default_env_name() {
  local stack="$1"
  case "$stack" in
    lamra_qwen25vl) printf '%s' "lamra2" ;;
    *) printf '%s' "$stack" ;;
  esac
}

require_cmd conda

for stack in "${STACKS[@]}"; do
  base_env_name="$(default_env_name "$stack")"
  if [[ -n "$NAME_PREFIX" ]]; then
    env_name="${NAME_PREFIX}-${base_env_name}"
  else
    env_name="$base_env_name"
  fi

  if env_exists "$env_name"; then
    echo "Skipping existing environment: $env_name"
    continue
  fi

  echo "Creating environment '$env_name' for stack '$stack'"
  "$SCRIPT_DIR/create_conda_env.sh" "$stack" "$env_name"
done

echo "Finished creating all requested environments."
