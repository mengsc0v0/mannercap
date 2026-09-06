#!/usr/bin/env bash
# ============================================================
# 一键启动所有待跑对比实验（在已有 tmux cap_exp 会话中新增窗口）
# 使用：bash scripts/run_all_pending_experiments.sh
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SESSION="cap_exp"

# 确保 tmux 会话存在
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}"
  echo "[info] 新建 tmux 会话: ${SESSION}"
else
  echo "[info] 复用已有 tmux 会话: ${SESSION}"
fi

# ── 窗口 1：LIBERO-40  Stage-CAP + KITE ────────────────────
tmux new-window -t "${SESSION}" -n "libero_kite"
tmux send-keys -t "${SESSION}:libero_kite" "
cd ${ROOT}
source scripts/env_stagecap.sh
echo '[libero_kite] 启动 Stage-CAP+KITE ...'
ROUND_TIMEOUT_SEC=1200 \
OUTPUT_DIR=outputs/libero40_stagecap_kite_\$(date +%Y%m%d_%H%M%S) \
bash scripts/run_libero40_codex_boundaries_mimo_kite.sh
echo '[libero_kite] 完成'
" Enter

# ── 窗口 2：RoboSuite  CAP agent0 基线（无特权状态）──────────
tmux new-window -t "${SESSION}" -n "robosuite_baseline"
PYTHON_BIN="/workdata/cap-x/.venv-libero/bin/python"
RS_OUT_BASELINE="outputs/robosuite_cap_agent0_baseline_\$(date +%Y%m%d_%H%M%S)"

tmux send-keys -t "${SESSION}:robosuite_baseline" "
cd ${ROOT}
source scripts/env_stagecap.sh
OUTDIR=outputs/robosuite_cap_agent0_baseline_\$(date +%Y%m%d_%H%M%S)
echo '[robosuite_baseline] NutAssembly ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly_reduced_api.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/nut_assembly
echo '[robosuite_baseline] StackCube ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_reduced_api.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/cube_stack
echo '[robosuite_baseline] TwoArmLift ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/two_arm_lift/franka_robosuite_two_arm_lift_reduced.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/two_arm_lift
echo '[robosuite_baseline] 全部完成，结果在:' \${OUTDIR}
" Enter

# ── 窗口 3：RoboSuite  特权状态（我们的方法等效） ────────────
tmux new-window -t "${SESSION}" -n "robosuite_privileged"
tmux send-keys -t "${SESSION}:robosuite_privileged" "
cd ${ROOT}
source scripts/env_stagecap.sh
OUTDIR=outputs/robosuite_privileged_\$(date +%Y%m%d_%H%M%S)
echo '[robosuite_privileged] NutAssembly ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/nut_assembly
echo '[robosuite_privileged] StackCube ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/cube_stack
echo '[robosuite_privileged] TwoArmLift ...'
${PYTHON_BIN} capx/envs/launch.py \
  --config-path env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml \
  --total-trials 10 --num-workers 2 \
  --output-dir \${OUTDIR}/two_arm_lift
echo '[robosuite_privileged] 全部完成，结果在:' \${OUTDIR}
" Enter

echo ""
echo "======================================================"
echo "  已在 tmux 会话 '${SESSION}' 中启动："
echo "  window 0 (bash)              : LIBERO baseline（已在跑）"
echo "  window libero_kite           : LIBERO-40 Stage-CAP+KITE"
echo "  window robosuite_baseline    : RoboSuite CAP agent0 基线"
echo "  window robosuite_privileged  : RoboSuite 特权状态方法"
echo ""
echo "  查看进度："
echo "  tmux attach -t ${SESSION}     # 进入会话"
echo "  Ctrl+B 然后 数字/名字          # 切换窗口"
echo "  Ctrl+B 然后 D                  # 退出但保持运行"
echo ""
echo "  明天查结果："
echo "  bash scripts/check_all_results.sh"
echo "======================================================"
