#!/usr/bin/env bash
# Install only into the project-owned environment, from a local wheelhouse.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${MANNERCAP_ENV:-/data/gaochuanhao/mannercap_env}"
wheel_dir="${1:?usage: setup_gate0_offline.sh WHEELHOUSE}"
python3 -m venv --without-pip "$env_dir"
PYTHONPATH="$wheel_dir/pip-25.3-py3-none-any.whl" "$env_dir/bin/python" -m pip install --no-index "$wheel_dir/pip-25.3-py3-none-any.whl"
"$env_dir/bin/python" -m pip install --no-index --find-links "$wheel_dir" setuptools==80.9.0 wheel==0.45.1
"$env_dir/bin/python" -m pip install --no-index --find-links "$wheel_dir" --no-build-isolation evdev==1.9.2
"$env_dir/bin/python" -m pip install --no-index --find-links "$wheel_dir" -r "$root/scripts/requirements-gate0.txt"
"$env_dir/bin/python" -m pip check
mkdir -p "$root/reports/gate0"
"$env_dir/bin/python" -m pip freeze > "$root/reports/gate0/environment-lock.txt"
