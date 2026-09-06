#!/usr/bin/env bash
# Unified reproduction entry point for Stage-CAP data-generation experiments.
#
# Usage examples:
#   bash scripts/reproduce_stagecap.sh check
#   bash scripts/reproduce_stagecap.sh dry-run
#   bash scripts/reproduce_stagecap.sh smoke-libero
#   bash scripts/reproduce_stagecap.sh libero40
#   bash scripts/reproduce_stagecap.sh sfb1-smoke
#   bash scripts/reproduce_stagecap.sh sfb1
#   bash scripts/reproduce_stagecap.sh inspect-sfb1 outputs/stagecap_vla_recollect_sfb1_xxx

set -euo pipefail

STAGECAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STAGECAP_ROOT}"

if [[ -f "${STAGECAP_ROOT}/scripts/env_stagecap.sh" ]]; then
  # shellcheck disable=SC1091
  source "${STAGECAP_ROOT}/scripts/env_stagecap.sh"
fi

MODE="${1:-help}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-${MIMO_MODEL:-mimo-v2.5}}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/reproduce_stagecap}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROUND_TIMEOUT_SEC="${ROUND_TIMEOUT_SEC:-420}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
FEEDBACK_ROUNDS="${FEEDBACK_ROUNDS:-2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TOTAL_STAGE_ATTEMPTS="${MAX_TOTAL_STAGE_ATTEMPTS:-10}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-${CUDA_VISIBLE_DEVICES:-0}}"

usage() {
  cat <<'EOF'
Stage-CAP reproduction script

Modes:
  help
      Show this message.

  check
      Check Python imports, required environment variables, and service ports.
      Set STRICT_SERVICES=1 to require PyRoKi/SAM3/GraspNet ports.

  dry-run
      Generate stage plans and task configs only; no robot execution.
      Default task: libero_object task 0.

  smoke-libero
      Run one small LIBERO task with Stage-CAP + Online KITE.
      Default task: libero_object task 0.

  libero40
      Run all 40 base LIBERO tasks in privileged data-construction mode.
      This can take many hours.

  sfb1-smoke
      Run two SFB1 scene-edited tasks for a quick data-construction smoke test.
      Default task ids: 6 9.

  sfb1
      Run the full SFB1 scene-edited task batch.

  inspect-sfb1 <output_dir>
      Inspect a completed SFB1 recollection output directory and write VLA
      trial manifests.

Common environment overrides:
  PYTHON_BIN=/path/to/python
  MIMO_BASE_URL=https://...
  MIMO_API_KEY=...
  MODEL=mimo-v2.5
  OUTPUT_BASE=outputs/reproduce_stagecap
  SUITES="libero_object"
  TASK_IDS="0 1"
  ROUND_TIMEOUT_SEC=420
  FEEDBACK_ROUNDS=2
  MAX_TOTAL_STAGE_ATTEMPTS=10
  CUDA_VISIBLE_DEVICES_ARG=0
  STRICT_SERVICES=1

Before real execution:
  1. cp .env.example .env
  2. Fill MIMO_API_KEY and PYTHON_BIN in .env
  3. source scripts/env_stagecap.sh
  4. Start required services if needed:
       bash scripts/launch_stagecap_services.sh minimal
EOF
}

require_mimo_key() {
  if [[ -z "${MIMO_API_KEY:-}" || "${MIMO_API_KEY:-}" == "replace_with_your_key" || "${MIMO_API_KEY:-}" == "<your_key>" ]]; then
    echo "MIMO_API_KEY is not set. Copy .env.example to .env and fill the key before running experiments." >&2
    exit 2
  fi
}

run_check() {
  local args=("--require-mimo-key")
  if [[ "${STRICT_SERVICES:-0}" == "1" || "${STRICT_SERVICES:-0}" == "true" || "${STRICT_SERVICES:-0}" == "True" ]]; then
    args+=("--strict-services")
  fi
  "${PYTHON_BIN}" scripts/check_environment.py "${args[@]}" "$@"
}

summarize_report_if_present() {
  local output_dir="$1"
  local report_path="${output_dir}/libero_stage_planner_report.json"
  if [[ -f "${report_path}" ]]; then
    echo ""
    echo "Summary:"
    "${PYTHON_BIN}" scripts/summarize_stage_planner_report.py --report-path "${report_path}" || true
  fi
}

run_stagecap_batch() {
  local output_dir="$1"
  shift
  require_mimo_key
  "${PYTHON_BIN}" scripts/run_libero_stage_planner_batch.py \
    --models "${MODEL}" \
    --output-dir "${output_dir}" \
    --total-trials "${TOTAL_TRIALS}" \
    --feedback-rounds "${FEEDBACK_ROUNDS}" \
    --num-workers "${NUM_WORKERS}" \
    --record-video True \
    --use-wrist-camera "${USE_WRIST_CAMERA:-True}" \
    --oracle-readonly \
    --use-stage-verifiable-runner \
    --use-kite-feedback \
    --use-online-stage-kite-feedback \
    --max-total-stage-attempts "${MAX_TOTAL_STAGE_ATTEMPTS}" \
    --round-timeout-sec "${ROUND_TIMEOUT_SEC}" \
    --stage-snapshot-chars 2200 \
    --save-stage-snapshots \
    --no-allow-point-prompt-molmo \
    --skip-api-servers \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_ARG}" \
    "$@"
  summarize_report_if_present "${output_dir}"
}

run_libero_mode() {
  local mode_name="$1"
  local output_dir="$2"
  shift 2
  local suites="${SUITES:-libero_object}"
  local task_ids="${TASK_IDS-0}"
  read -r -a suite_args <<< "${suites}"
  local task_args=()
  if [[ -n "${task_ids}" ]]; then
    read -r -a task_args <<< "${task_ids//,/ }"
  fi

  local extra_args=(
    --experiment-label "${mode_name}"
    --suites "${suite_args[@]}"
  )
  if (( ${#task_args[@]} > 0 )); then
    extra_args+=(--task-ids "${task_args[@]}")
  fi
  run_stagecap_batch "${output_dir}" "${extra_args[@]}" "$@"
}

run_sfb1_mode() {
  local mode_name="$1"
  local output_dir="$2"
  local default_task_ids="$3"
  shift 3
  local manifest="${MANIFEST:-outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json}"
  if [[ ! -f "${manifest}" ]]; then
    echo "SFB1 manifest not found: ${manifest}" >&2
    echo "Regenerate it with: ${PYTHON_BIN} scripts/prepare_sfb1_scene_batch.py" >&2
    exit 2
  fi
  local task_ids="${TASK_IDS:-${default_task_ids}}"
  local task_args=()
  if [[ -n "${task_ids}" ]]; then
    read -r -a task_args <<< "${task_ids//,/ }"
  fi
  local extra_args=(
    --experiment-label "${mode_name}"
    --custom-task-manifest "${manifest}"
    --use-wrist-camera True
  )
  if (( ${#task_args[@]} > 0 )); then
    extra_args+=(--task-ids "${task_args[@]}")
  fi
  run_stagecap_batch "${output_dir}" "${extra_args[@]}" "$@"
}

case "${MODE}" in
  help|-h|--help)
    usage
    ;;
  check)
    run_check "$@"
    ;;
  dry-run)
    out="${OUTPUT_DIR:-${OUTPUT_BASE}/dry_run_${RUN_STAMP}}"
    suites="${SUITES:-libero_object}"
    task_ids="${TASK_IDS:-0}"
    read -r -a suite_args <<< "${suites}"
    read -r -a task_args <<< "${task_ids//,/ }"
    "${PYTHON_BIN}" scripts/run_libero_stage_planner_batch.py \
      --experiment-label reproduce_stagecap_dry_run \
      --suites "${suite_args[@]}" \
      --task-ids "${task_args[@]}" \
      --models "${MODEL}" \
      --output-dir "${out}" \
      --oracle-readonly \
      --use-stage-verifiable-runner \
      --use-kite-feedback \
      --dry-run \
      --skip-api-servers \
      "$@"
    echo "Dry-run output: ${out}"
    ;;
  smoke-libero)
    out="${OUTPUT_DIR:-${OUTPUT_BASE}/smoke_libero_${RUN_STAMP}}"
    run_libero_mode reproduce_stagecap_smoke_libero "${out}" "$@"
    echo "Output: ${out}"
    ;;
  libero40)
    out="${OUTPUT_DIR:-${OUTPUT_BASE}/libero40_${RUN_STAMP}}"
    SUITES="${SUITES:-libero_10 libero_object libero_spatial libero_goal}"
    TASK_IDS="${TASK_IDS:-}"
    run_libero_mode reproduce_stagecap_libero40 "${out}" "$@"
    echo "Output: ${out}"
    ;;
  sfb1-smoke)
    out="${OUTPUT_DIR:-${OUTPUT_BASE}/sfb1_smoke_${RUN_STAMP}}"
    run_sfb1_mode reproduce_stagecap_sfb1_smoke "${out}" "6 9" "$@"
    echo "Output: ${out}"
    ;;
  sfb1)
    out="${OUTPUT_DIR:-${OUTPUT_BASE}/sfb1_full_${RUN_STAMP}}"
    TASK_IDS="${TASK_IDS:-}"
    run_sfb1_mode reproduce_stagecap_sfb1_full "${out}" "" "$@"
    echo "Output: ${out}"
    ;;
  inspect-sfb1)
    if [[ "$#" -lt 1 ]]; then
      echo "Usage: bash scripts/reproduce_stagecap.sh inspect-sfb1 <output_dir>" >&2
      exit 2
    fi
    target="$1"
    shift
    "${PYTHON_BIN}" scripts/inspect_stagecap_vla_trials.py "${target}" \
      --output-jsonl "${target}/vla_trials_manifest.jsonl" \
      --output-csv "${target}/vla_trials_manifest.csv" \
      "$@"
    echo "Wrote: ${target}/vla_trials_manifest.jsonl"
    echo "Wrote: ${target}/vla_trials_manifest.csv"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
