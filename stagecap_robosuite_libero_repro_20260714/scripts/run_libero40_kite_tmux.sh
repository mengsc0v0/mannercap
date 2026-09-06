#!/usr/bin/env bash
# ================================================================
# 在 tmux cap_exp 中启动 LIBERO-40 Stage-CAP+KITE 实验（新窗口）
# 用途：libero agent0 跑完后，手动执行此脚本继续跑 KITE 对比
#
# 使用：bash scripts/run_libero40_kite_tmux.sh
# ================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="cap_exp"
WIN_NAME="libero_kite"

PYTHON_BIN="${PYTHON_BIN:-/workdata/cap-x/.venv-libero/bin/python}"
MIMO_BASE_URL="${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}"
: "${MIMO_API_KEY:?Set MIMO_API_KEY in the environment before launching experiments}"
MIMO_MODEL="${MIMO_MODEL:-mimo-v2.5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero40_stagecap_kite_$(date +%Y%m%d_%H%M%S)}"
FEEDBACK_ROUNDS="${FEEDBACK_ROUNDS:-2}"
TOTAL_TRIALS="${TOTAL_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ROUND_TIMEOUT_SEC="${ROUND_TIMEOUT_SEC:-1200}"

# 确保 tmux 会话存在
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}"
fi

# 关闭同名旧窗口
if tmux list-windows -t "${SESSION}" | grep -q "${WIN_NAME}"; then
  tmux kill-window -t "${SESSION}:${WIN_NAME}" 2>/dev/null || true
  echo "[info] 已关闭旧 ${WIN_NAME} 窗口"
fi

tmux new-window -t "${SESSION}" -n "${WIN_NAME}"
tmux send-keys -t "${SESSION}:${WIN_NAME}" "
cd ${ROOT}
export PYTHONPATH=\${PYTHONPATH:-.}
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MIMO_BASE_URL=${MIMO_BASE_URL}
export MIMO_API_KEY=${MIMO_API_KEY}
export MIMO_MODEL=${MIMO_MODEL}
export http_proxy=http://127.0.0.1:18110
export https_proxy=http://127.0.0.1:18110
echo '[libero_kite] 启动 LIBERO-40 Stage-CAP+KITE...'
ROUND_TIMEOUT_SEC=${ROUND_TIMEOUT_SEC} \\
OUTPUT_DIR=${OUTPUT_DIR} \\
PYTHON_BIN=${PYTHON_BIN} \\
FEEDBACK_ROUNDS=${FEEDBACK_ROUNDS} \\
TOTAL_TRIALS=${TOTAL_TRIALS} \\
NUM_WORKERS=${NUM_WORKERS} \\
bash scripts/run_libero40_codex_boundaries_mimo_kite.sh
echo '[libero_kite] 完成'
" Enter

echo ""
echo "======================================================"
echo "  LIBERO-40 Stage-CAP+KITE 已启动"
echo "  tmux 窗口: ${SESSION}:${WIN_NAME}"
echo "  输出目录:  ${OUTPUT_DIR}"
echo "  查看进度:  tmux attach -t ${SESSION}"
echo "======================================================"
