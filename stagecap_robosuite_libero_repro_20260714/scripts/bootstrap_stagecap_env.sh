#!/usr/bin/env bash
# Create a local Python environment and install the lightweight runtime deps.
# Large simulator/model assets still need to be installed or mounted separately;
# see ENVIRONMENT.md.

set -euo pipefail

STAGECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGECAP_ROOT}"

VENV_DIR="${VENV_DIR:-.venv-libero}"
PYTHON_EXE="${PYTHON_EXE:-python3.12}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_EXE}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements-stagecap-runtime.txt
python -m pip install -r requirements-kite-vendored.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill MIMO_API_KEY before running experiments."
fi

echo "Environment bootstrap finished: ${STAGECAP_ROOT}/${VENV_DIR}"
echo "Next: source scripts/env_stagecap.sh && python scripts/check_environment.py"
