#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_libero_vifailback_gpu_pool.sh [ViFailback args...]

Environment:
  VIFAILBACK_GPU_POOL          Comma-separated GPU ids. Default: 3,4,7
  VIFAILBACK_MIN_FREE_MIB      Minimum free memory required. Default: 18000
  VIFAILBACK_GPU_WAIT_SECONDS  Seconds to wait between retries. Default: 30
  VIFAILBACK_GPU_MAX_WAITS     Max retries before using fallback. Default: 120
  VIFAILBACK_FALLBACK_GPU      GPU used if no GPU satisfies the threshold.
  VIFAILBACK_LOCK_DIR          Lock directory. Default: /tmp/capx_vifailback_gpu_locks
  AUTOROB_ROOT                 Autorob repo root.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GPU_POOL="${VIFAILBACK_GPU_POOL:-3,4,7}"
MIN_FREE_MIB="${VIFAILBACK_MIN_FREE_MIB:-18000}"
WAIT_SECONDS="${VIFAILBACK_GPU_WAIT_SECONDS:-30}"
MAX_WAITS="${VIFAILBACK_GPU_MAX_WAITS:-120}"
FALLBACK_GPU="${VIFAILBACK_FALLBACK_GPU:-3}"
LOCK_DIR="${VIFAILBACK_LOCK_DIR:-/tmp/capx_vifailback_gpu_locks}"
AUTOROB_ROOT="${AUTOROB_ROOT:-/home/lifuchun/snap/snapd-desktop-integration/autorob}"
REAL_SCRIPT="$AUTOROB_ROOT/scripts/run_libero_vifailback_experiment.sh"

mkdir -p "$LOCK_DIR"

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v target="$gpu" '$1 ~ "^[[:space:]]*" target "[[:space:]]*$" {gsub(/ /, "", $2); print $2; exit}'
}

choose_gpu() {
  local best_gpu=""
  local best_free=-1
  local gpu free
  IFS=',' read -ra gpus <<< "$GPU_POOL"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "$gpu" ]] && continue
    free="$(gpu_free_mib "$gpu" || true)"
    [[ -z "$free" ]] && continue
    if (( free >= MIN_FREE_MIB && free > best_free )); then
      best_gpu="$gpu"
      best_free="$free"
    fi
  done
  if [[ -n "$best_gpu" ]]; then
    echo "$best_gpu"
  else
    echo ""
  fi
}

run_with_gpu() {
  local gpu="$1"
  shift
  echo "[vifailback-gpu-pool] using gpu=$gpu free_mib=$(gpu_free_mib "$gpu" || echo unknown)" >&2
  exec "$REAL_SCRIPT" "$@" --gpu "$gpu"
}

wait_count=0
while (( wait_count <= MAX_WAITS )); do
  selected="$(choose_gpu)"
  if [[ -n "$selected" ]]; then
    lock_path="$LOCK_DIR/gpu_${selected}.lock"
    exec 9>"$lock_path"
    if flock -n 9; then
      run_with_gpu "$selected" "$@"
    fi
  fi
  wait_count=$((wait_count + 1))
  sleep "$WAIT_SECONDS"
done

lock_path="$LOCK_DIR/gpu_${FALLBACK_GPU}.lock"
exec 9>"$lock_path"
flock 9
run_with_gpu "$FALLBACK_GPU" "$@"
