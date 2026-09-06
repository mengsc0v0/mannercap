#!/usr/bin/env bash
# 查看所有实验结果
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/workdata/cap-x/.venv-libero/bin/python

echo "======================================================"
echo "  实验结果汇总  $(date '+%Y-%m-%d %H:%M')"
echo "======================================================"

# ── LIBERO baseline (cap_agent0) ──────────────────────────
echo ""
echo "【LIBERO-40  CAP agent0 基线】"
BASELINE_DIR=$(ls -dt "${ROOT}/outputs/libero40_cap_agent0_baseline_"* 2>/dev/null | head -1)
if [[ -n "${BASELINE_DIR}" ]]; then
  echo "  目录: $(basename ${BASELINE_DIR})"
  python3 << PYEOF
import glob, os, json
base='${BASELINE_DIR}'
results=[]
for suite_dir in sorted(glob.glob(f'{base}/libero_*')):
    suite=os.path.basename(suite_dir)
    for task_dir in sorted(glob.glob(f'{suite_dir}/*')):
        rounds=sorted(glob.glob(f'{task_dir}/round_*'))
        if not rounds: continue
        all_trials=[]
        for r in rounds:
            all_trials += glob.glob(f'{r}/*/run/trial_*')
        success=any('sandboxrc_0' in t and 'reward_1.' in t for t in all_trials)
        done=(len(rounds)>=2 or success)
        results.append((suite, success, done))
succ=sum(1 for r in results if r[1])
done=sum(1 for r in results if r[2])
print(f'  总体: {len(results)}/40  完成: {done}/40  TSR: {succ}/40 = {succ/40:.0%}')
for suite in ['libero_10','libero_object','libero_spatial','libero_goal']:
    tasks=[r for r in results if r[0]==suite]
    if not tasks: continue
    s=sum(1 for t in tasks if t[1])
    print(f'    {suite}: {s}/10 = {s/10:.0%}')
PYEOF
else
  echo "  未找到结果目录"
fi

# ── LIBERO KITE ───────────────────────────────────────────
echo ""
echo "【LIBERO-40  Stage-CAP + KITE】"
KITE_DIR=$(ls -dt "${ROOT}/outputs/libero40_stagecap_kite_"* 2>/dev/null | head -1)
if [[ -n "${KITE_DIR}" ]]; then
  echo "  目录: $(basename ${KITE_DIR})"
  python3 << PYEOF
import glob, os
base='${KITE_DIR}'
results=[]
for suite_dir in sorted(glob.glob(f'{base}/libero_*')):
    suite=os.path.basename(suite_dir)
    for task_dir in sorted(glob.glob(f'{suite_dir}/*')):
        rounds=sorted(glob.glob(f'{task_dir}/round_*'))
        if not rounds: continue
        all_trials=[]
        for r in rounds:
            all_trials += glob.glob(f'{r}/*/run/trial_*')
        success=any('sandboxrc_0' in t and 'reward_1.' in t for t in all_trials)
        done=(len(rounds)>=2 or success)
        results.append((suite, success, done))
succ=sum(1 for r in results if r[1])
done=sum(1 for r in results if r[2])
print(f'  总体: {len(results)}/40  完成: {done}/40  TSR: {succ}/40 = {succ/40:.0%}')
for suite in ['libero_10','libero_object','libero_spatial','libero_goal']:
    tasks=[r for r in results if r[0]==suite]
    if not tasks: continue
    s=sum(1 for t in tasks if t[1])
    print(f'    {suite}: {s}/10 = {s/10:.0%}')
PYEOF
else
  echo "  尚未开始或未找到结果"
fi

# ── RoboSuite agent0 基线 ────────────────────────────────
echo ""
echo "【RoboSuite  CAP agent0 基线】"
RS_BASE_DIR=$(ls -dt "${ROOT}/outputs/robosuite_cap_agent0_baseline_"* 2>/dev/null | head -1)
if [[ -n "${RS_BASE_DIR}" ]]; then
  echo "  目录: $(basename ${RS_BASE_DIR})"
  python3 << PYEOF2
import glob, os, re
trial_re = re.compile(r'^trial_\d+_sandboxrc_-?\d+_reward_(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<done>[01])$')
base='${RS_BASE_DIR}'
for task in ['nut_assembly','cube_stack','two_arm_lift']:
    run_dir=os.path.join(base, task, 'run')
    total,succ=0,0
    if os.path.isdir(run_dir):
        for e in os.listdir(run_dir):
            m=trial_re.match(e)
            if m:
                total+=1
                if int(m.group('done'))==1 or float(m.group('reward'))>=1.0:
                    succ+=1
    status='未完成' if total==0 else f'{succ}/{total} ({succ/total:.0%})'
    print(f'    {task}: {status}')
PYEOF2
else
  echo "  尚未开始或未找到结果"
fi

# ── RoboSuite Stage-CAP + KITE ────────────────────────────
echo ""
echo "【RoboSuite  Stage-CAP + KITE（我们的方法）】"
RS_KITE_DIR=$(ls -dt "${ROOT}/outputs/robosuite_stagecap_kite_"* 2>/dev/null | head -1)
if [[ -n "${RS_KITE_DIR}" ]]; then
  echo "  目录: $(basename ${RS_KITE_DIR})"
  python3 << PYEOF3
import glob, os, re
trial_re = re.compile(r'^trial_\d+_sandboxrc_-?\d+_reward_(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<done>[01])$')
base='${RS_KITE_DIR}'
for task in ['nut_assembly','cube_stack','two_arm_lift']:
    run_dir=os.path.join(base, task, 'run')
    total,succ=0,0
    if os.path.isdir(run_dir):
        for e in os.listdir(run_dir):
            m=trial_re.match(e)
            if m:
                total+=1
                if int(m.group('done'))==1 or float(m.group('reward'))>=1.0:
                    succ+=1
    status='未完成' if total==0 else f'{succ}/{total} ({succ/total:.0%})'
    print(f'    {task}: {status}')
PYEOF3
else
  echo "  尚未开始或未找到结果"
fi

echo ""
echo "======================================================"
