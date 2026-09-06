#!/usr/bin/env bash
# Launch local services used by Stage-CAP. This command blocks until stopped.

set -euo pipefail

STAGECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGECAP_ROOT}"

if [[ -f scripts/env_stagecap.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/env_stagecap.sh
fi

PROFILE="${1:-default}"
TIMEOUT="${SERVICE_READY_TIMEOUT:-180}"

"${PYTHON_BIN:-python}" capx/serving/launch_servers.py \
  --profile "${PROFILE}" \
  --timeout "${TIMEOUT}"
