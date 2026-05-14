#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.6}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"

usage() {
  cat <<'EOF'
Usage: create_conda_env.sh STACK [ENV_NAME]

STACK values:
  gme           — for GME-Qwen2VL
  lamra         — for LamRA
  lamra_qwen25vl (alias: lamra2) — for LamRA-Qwen2.5VL
  mmemb         — for MM-Embed
  omni          — for E5-Omni
  qwen3emb      — for Qwen3-VL-Embedding (2B/8B), Rzen-Embed, VLM2Vec-V2 (vLLM stack)

ENV_NAME defaults to STACK.
Override the Python version with PYTHON_VERSION=3.11.6.
Override the Conda channel with CONDA_CHANNEL=conda-forge.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

STACK="$1"
ENV_NAME="${2:-$STACK}"
REQ_FILE=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
TORCHAUDIO_VERSION=""
TORCH_INDEX_URL=""
INSTALL_VLLM=0
INSTALL_TORCH=1

case "$STACK" in
  gme)
    REQ_FILE="$SCRIPT_DIR/envs_files/gme.txt"
    TORCH_VERSION="2.9.0"
    TORCHVISION_VERSION="0.24.0"
    TORCHAUDIO_VERSION="2.9.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  lamra)
    REQ_FILE="$SCRIPT_DIR/envs_files/lamra.txt"
    TORCH_VERSION="2.3.1"
    TORCHVISION_VERSION="0.18.1"
    TORCHAUDIO_VERSION="2.3.1"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    ;;
  lamra_qwen25vl|lamra-qwen25vl|lamra2)
    REQ_FILE="$SCRIPT_DIR/envs_files/lamra_qwen25vl.txt"
    TORCH_VERSION="2.9.0"
    TORCHVISION_VERSION="0.24.0"
    TORCHAUDIO_VERSION="2.9.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  mmemb)
    REQ_FILE="$SCRIPT_DIR/envs_files/mmemb.txt"
    TORCH_VERSION="2.2.2"
    TORCHVISION_VERSION="0.17.2"
    TORCHAUDIO_VERSION="2.2.2"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  omni)
    REQ_FILE="$SCRIPT_DIR/envs_files/omni.txt"
    TORCH_VERSION="2.9.0"
    TORCHVISION_VERSION="0.24.0"
    TORCHAUDIO_VERSION="2.9.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  qwen3emb|qwen3|qwen-embedding)
    REQ_FILE="$SCRIPT_DIR/envs_files/qwen-embedding.txt"
    TORCH_VERSION="2.9.0"
    TORCHVISION_VERSION="0.24.0"
    TORCHAUDIO_VERSION="2.9.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    INSTALL_VLLM=1
    ;;
  *)
    echo "Unknown stack: $STACK" >&2
    usage >&2
    exit 1
    ;;
esac

require_cmd conda

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment already exists: $ENV_NAME" >&2
  exit 1
fi

conda create -y -n "$ENV_NAME" -c "$CONDA_CHANNEL" "python=${PYTHON_VERSION}" pip
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
if [[ "$INSTALL_TORCH" -eq 1 ]]; then
  conda run -n "$ENV_NAME" python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
fi

if [[ "$INSTALL_VLLM" -eq 1 ]]; then
  conda run -n "$ENV_NAME" python -m pip install vllm==0.15 \
    --extra-index-url "$TORCH_INDEX_URL"
fi

conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"
conda run -n "$ENV_NAME" python -m pip install -U wandb weave

cat <<EOF
Created conda environment '$ENV_NAME' for stack '$STACK'.

Activate it with:
  conda activate $ENV_NAME
EOF
