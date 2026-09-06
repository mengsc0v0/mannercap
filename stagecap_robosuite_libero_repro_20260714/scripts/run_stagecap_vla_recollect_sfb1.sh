#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-.}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export MIMO_BASE_URL="${MIMO_BASE_URL:-http://127.0.0.1:8110}"

PYTHON_BIN="${PYTHON_BIN:-/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python}"
MANIFEST="${MANIFEST:-outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stagecap_vla_recollect_sfb1_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-mimo-v2.5}"
TASK_IDS="${TASK_IDS:-}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-0}"
ROUND_TIMEOUT_SEC="${ROUND_TIMEOUT_SEC:-420}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
FEEDBACK_ROUNDS="${FEEDBACK_ROUNDS:-1}"
MAX_TOTAL_STAGE_ATTEMPTS="${MAX_TOTAL_STAGE_ATTEMPTS:-10}"
ORACLE_READONLY="${ORACLE_READONLY:-1}"

TASK_ID_ARGS=()
if [[ -n "${TASK_IDS}" ]]; then
  read -r -a TASK_ID_ARGS <<< "${TASK_IDS//,/ }"
fi

OPTIONAL_TASK_ARGS=()
if (( ${#TASK_ID_ARGS[@]} > 0 )); then
  OPTIONAL_TASK_ARGS+=(--task-ids "${TASK_ID_ARGS[@]}")
fi

OPTIONAL_ORACLE_ARGS=()
if [[ "${ORACLE_READONLY}" == "1" || "${ORACLE_READONLY}" == "true" || "${ORACLE_READONLY}" == "True" ]]; then
  OPTIONAL_ORACLE_ARGS+=(--oracle-readonly)
else
  OPTIONAL_ORACLE_ARGS+=(--no-oracle-readonly)
fi

"${PYTHON_BIN}" scripts/run_libero_stage_planner_batch.py \
  --custom-task-manifest "${MANIFEST}" \
  "${OPTIONAL_TASK_ARGS[@]}" \
  --models "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-label stagecap_vla_recollect_sfb1 \
  --use-stage-verifiable-runner \
  --record-video True \
  --use-wrist-camera True \
  --feedback-rounds "${FEEDBACK_ROUNDS}" \
  --total-trials "${TOTAL_TRIALS}" \
  --num-workers "${NUM_WORKERS}" \
  --round-timeout-sec "${ROUND_TIMEOUT_SEC}" \
  --max-total-stage-attempts "${MAX_TOTAL_STAGE_ATTEMPTS}" \
  --use-kite-feedback \
  --use-online-stage-kite-feedback \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_ARG}" \
  "${OPTIONAL_ORACLE_ARGS[@]}" \
  "$@"

echo "Output: ${OUTPUT_DIR}"
echo "Inspect with:"
echo "  python3 scripts/inspect_stagecap_vla_trials.py ${OUTPUT_DIR} --output-jsonl ${OUTPUT_DIR}/vla_trials_manifest.jsonl --output-csv ${OUTPUT_DIR}/vla_trials_manifest.csv"
