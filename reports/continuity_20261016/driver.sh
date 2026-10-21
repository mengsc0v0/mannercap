#!/bin/bash
# continuity sweeps on dev seeds 9201-9206 (disjoint from calibration 9001-9024/9101-9108 and production)
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/workdata/cap-x/.venv/bin/python3
D=reports/continuity_20261016
S=9201,9202,9203,9204,9205,9206
SPD=0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.60
WSPD=0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.18,0.20,0.25,0.30,0.40,0.60,0.90
WPR=5,10,15,20,25,30,35,40,45,50,60,70
LG=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42
SG=4,8,12,16,20,24,28,32,36,40
N=6
for i in $(seq 0 $((N-1))); do
  $PY scripts/p11_continuity_sweep.py lift_grip $D/lift_grip.jsonl $S $LG $i/$N > $D/lift_grip.$i.log 2>&1 &
  $PY scripts/p11_continuity_sweep.py stack_grip $D/stack_grip.jsonl $S $SG $i/$N > $D/stack_grip.$i.log 2>&1 &
  $PY scripts/p11_continuity_sweep.py lift_speed $D/lift_speed.jsonl $S $SPD $i/$N > $D/lift_speed.$i.log 2>&1 &
  $PY scripts/p11_continuity_sweep.py stack_speed $D/stack_speed.jsonl $S $SPD $i/$N > $D/stack_speed.$i.log 2>&1 &
  $PY scripts/p11_continuity_sweep.py wipe_speed $D/wipe_speed.jsonl $S $WSPD $i/$N > $D/wipe_speed.$i.log 2>&1 &
  $PY scripts/p11_continuity_sweep.py wipe_press $D/wipe_press.jsonl $S $WPR $i/$N > $D/wipe_press.$i.log 2>&1 &
done
wait
echo "all sweeps done $(date)" > $D/driver.done
