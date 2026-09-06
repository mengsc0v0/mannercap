#!/usr/bin/env bash
# Source this file for LIBERO-PRO with its RoboSuite 1.4.0 fork.

set -euo pipefail

STAGECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGECAP_ROOT}"

if [[ -f "${STAGECAP_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${STAGECAP_ROOT}/.env"
  set +a
fi

LIBERO_ROOT="${STAGECAP_ROOT}/capx/third_party/LIBERO-PRO"
LIBERO_ROBOSUITE_SRC="${STAGECAP_ROOT}/capx/third_party/libero_dependencies/robosuite"
if [[ ! -f "${LIBERO_ROOT}/libero/libero/__init__.py" ]]; then
  echo "Bundled LIBERO-PRO source not found: ${LIBERO_ROOT}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -f "${LIBERO_ROBOSUITE_SRC}/robosuite/__init__.py" ]]; then
  echo "Bundled LIBERO RoboSuite 1.4.0 source not found: ${LIBERO_ROBOSUITE_SRC}" >&2
  return 2 2>/dev/null || exit 2
fi

# LIBERO's Python package root is LIBERO-PRO/libero. Keep its RoboSuite fork
# ahead of site-packages and exclude the standalone RoboSuite 1.5.1 source.
export PYTHONPATH="${STAGECAP_ROOT}:${LIBERO_ROBOSUITE_SRC}:${LIBERO_ROOT}/libero${STAGECAP_EXTRA_PYTHONPATH:+:${STAGECAP_EXTRA_PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYROKI_SERVICE_URL="${PYROKI_SERVICE_URL:-http://127.0.0.1:8116}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${STAGECAP_ROOT}/.venv-stagecap/bin/python" ]]; then
    export PYTHON_BIN="${STAGECAP_ROOT}/.venv-stagecap/bin/python"
  else
    export PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "StageCaP LIBERO runtime: ${LIBERO_ROOT}"
echo "LIBERO RoboSuite runtime: ${LIBERO_ROBOSUITE_SRC}"
echo "Python: ${PYTHON_BIN}"

