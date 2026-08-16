#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FASTWAM_VENV:-/root/wuqingman/.venv-fastwam}"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is required for experiment 37." >&2
  exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Training environment not found: ${VENV_DIR}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="/root/wuqingman/models/wan"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export RUN_ID="${RUN_ID:-experiment37_$(date +%Y-%m-%d_%H-%M-%S)}"

exec bash scripts/train_zero2.sh 8 \
  task=libero_h3_uncond_2cam224_1e-4 \
  "$@"
