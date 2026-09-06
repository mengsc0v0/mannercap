#!/usr/bin/env bash
# ================================================================
# 在 tmux cap_exp 中启动 RoboSuite 对比实验
#   Window robosuite_agent0   : CAP agent0 基线（无Stage-CAP/KITE）
#   Window robosuite_kite     : Stage-CAP + KITE（我们的方法）
#
# 使用：bash scripts/run_robosuite_comparison_tmux.sh
# ================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="cap_exp"

# RoboSuite 必须用 .venv (Python 3.10)，不能用 .venv-libero
PYTHON_BIN="/workdata/cap-x/.venv/bin/python"

# API 配置（从 .env 读取，或手动设置）
MIMO_BASE_URL="${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}"
: "${MIMO_API_KEY:?Set MIMO_API_KEY in the environment before launching experiments}"
MIMO_MODEL="${MIMO_MODEL:-mimo-v2.5}"

TOTAL_TRIALS="${TOTAL_TRIALS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_AGENT0="outputs/robosuite_cap_agent0_baseline_${TS}"
OUT_KITE="outputs/robosuite_stagecap_kite_${TS}"

# 确保 tmux 会话存在
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}"
  echo "[info] 新建 tmux 会话: ${SESSION}"
else
  echo "[info] 复用已有 tmux 会话: ${SESSION}"
fi

# 清理旧的 robosuite 窗口（如果存在）
for win in robosuite_agent0 robosuite_kite robosuite_baseline robosuite_privileged; do
  if tmux list-windows -t "${SESSION}" | grep -q "${win}"; then
    tmux kill-window -t "${SESSION}:${win}" 2>/dev/null || true
    echo "[info] 已关闭旧窗口: ${win}"
  fi
done

# ── Window 1: RoboSuite CAP agent0 基线 ──────────────────────────────────────
tmux new-window -t "${SESSION}" -n "robosuite_agent0"
tmux send-keys -t "${SESSION}:robosuite_agent0" "
cd ${ROOT}
export PYTHONPATH=\${PYTHONPATH:-.}
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MIMO_BASE_URL=${MIMO_BASE_URL}
export MIMO_API_KEY=${MIMO_API_KEY}
export MIMO_MODEL=${MIMO_MODEL}
export http_proxy=http://127.0.0.1:18110
export https_proxy=http://127.0.0.1:18110
echo '[robosuite_agent0] 开始 CAP agent0 基线...'
${PYTHON_BIN} scripts/run_robosuite_stagecap_batch.py \\
  --condition agent0 \\
  --tasks nut_assembly cube_stack two_arm_lift \\
  --total-trials ${TOTAL_TRIALS} \\
  --num-workers ${NUM_WORKERS} \\
  --output-dir ${OUT_AGENT0}
echo '[robosuite_agent0] 完成'
" Enter

# ── Window 2: RoboSuite Stage-CAP + KITE ─────────────────────────────────────
tmux new-window -t "${SESSION}" -n "robosuite_kite"
tmux send-keys -t "${SESSION}:robosuite_kite" "
cd ${ROOT}
export PYTHONPATH=\${PYTHONPATH:-.}
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MIMO_BASE_URL=${MIMO_BASE_URL}
export MIMO_API_KEY=${MIMO_API_KEY}
export MIMO_MODEL=${MIMO_MODEL}
export http_proxy=http://127.0.0.1:18110
export https_proxy=http://127.0.0.1:18110
echo '[robosuite_kite] 开始 Stage-CAP+KITE ...'
${PYTHON_BIN} scripts/run_robosuite_stagecap_batch.py \\
  --condition stagecap_kite \\
  --tasks nut_assembly cube_stack two_arm_lift \\
  --total-trials ${TOTAL_TRIALS} \\
  --num-workers ${NUM_WORKERS} \\
  --output-dir ${OUT_KITE}
echo '[robosuite_kite] 完成'
" Enter

echo ""
echo "======================================================"
echo "  RoboSuite 对比实验已在 tmux '${SESSION}' 中启动："
echo "  window robosuite_agent0  : CAP agent0 基线"
echo "  window robosuite_kite    : Stage-CAP + KITE"
echo ""
echo "  查看进度：tmux attach -t ${SESSION}"
echo "  明天查结果：bash scripts/check_all_results.sh"
echo "======================================================"
