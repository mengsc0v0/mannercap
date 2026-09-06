#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-.}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MIMO_BASE_URL="${MIMO_BASE_URL:-http://127.0.0.1:8110}"

PYTHON_BIN="${PYTHON_BIN:-/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero40_codex_boundaries_mimo_kite_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-mimo-v2.5}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-0}"
FEEDBACK_ROUNDS="${FEEDBACK_ROUNDS:-2}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TOTAL_STAGE_ATTEMPTS="${MAX_TOTAL_STAGE_ATTEMPTS:-10}"
ROUND_TIMEOUT_SEC="${ROUND_TIMEOUT_SEC:-1200}"
SUITES="${SUITES:-libero_10 libero_object libero_spatial libero_goal}"
TASK_IDS="${TASK_IDS:-}"
USE_SAM3_PERCEPTION="${USE_SAM3_PERCEPTION:-0}"

read -r -a SUITE_ARGS <<< "${SUITES}"
TASK_ID_ARGS=()
if [[ -n "${TASK_IDS}" ]]; then
  # Accept both "0 5 7" and "0,5,7" so parallel eval jobs do not silently
  # create configs without running the intended task ids.
  read -r -a TASK_ID_ARGS <<< "${TASK_IDS//,/ }"
fi

OPTIONAL_TASK_ARGS=()
if (( ${#TASK_ID_ARGS[@]} > 0 )); then
  OPTIONAL_TASK_ARGS+=(--task-ids "${TASK_ID_ARGS[@]}")
fi
OPTIONAL_SAM3_ARGS=()
if [[ "${USE_SAM3_PERCEPTION}" == "1" || "${USE_SAM3_PERCEPTION}" == "true" || "${USE_SAM3_PERCEPTION}" == "True" ]]; then
  OPTIONAL_SAM3_ARGS+=(--use-sam3-perception)
fi

"${PYTHON_BIN}" scripts/run_libero_stage_planner_batch.py \
  --experiment-label codex_boundaries_mimo_kite_feedback \
  --suites "${SUITE_ARGS[@]}" \
  "${OPTIONAL_TASK_ARGS[@]}" \
  --models "${MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --total-trials "${TOTAL_TRIALS}" \
  --feedback-rounds "${FEEDBACK_ROUNDS}" \
  --num-workers "${NUM_WORKERS}" \
  --record-video True \
  --oracle-readonly \
  --use-stage-verifiable-runner \
  --use-kite-feedback \
  --max-total-stage-attempts "${MAX_TOTAL_STAGE_ATTEMPTS}" \
  --round-timeout-sec "${ROUND_TIMEOUT_SEC}" \
  --stage-snapshot-chars 2200 \
  --save-stage-snapshots \
  --no-allow-point-prompt-molmo \
  "${OPTIONAL_SAM3_ARGS[@]}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_ARG}" \
  "$@"

echo "Report: ${OUTPUT_DIR}/libero_stage_planner_report.json"
