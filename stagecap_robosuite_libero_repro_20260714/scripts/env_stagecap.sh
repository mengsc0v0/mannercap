#!/usr/bin/env bash
# Source this file before running Stage-CAP experiments:
#   source scripts/env_stagecap.sh

set -euo pipefail

STAGECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGECAP_ROOT}"

if [[ -f "${STAGECAP_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${STAGECAP_ROOT}/.env"
  set +a
fi

export PYTHONPATH="${STAGECAP_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MIMO_BASE_URL="${MIMO_BASE_URL:-http://127.0.0.1:8110/v1}"
export MIMO_MODEL="${MIMO_MODEL:-mimo-v2.5}"

export SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:8114}"
export PYROKI_SERVICE_URL="${PYROKI_SERVICE_URL:-http://127.0.0.1:8116}"
export GRASPNET_SERVICE_URL="${GRASPNET_SERVICE_URL:-http://127.0.0.1:8115}"

export MOLMO_BASE_URL="${MOLMO_BASE_URL:-${MIMO_BASE_URL}}"
export MOLMO_API_KEY="${MOLMO_API_KEY:-${MIMO_API_KEY:-}}"
export MOLMO_MODEL="${MOLMO_MODEL:-mimo-v2.5}"

export PYTHON_BIN="${PYTHON_BIN:-/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python}"

echo "Stage-CAP environment loaded from ${STAGECAP_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "MIMO endpoint: ${MIMO_BASE_URL}"
