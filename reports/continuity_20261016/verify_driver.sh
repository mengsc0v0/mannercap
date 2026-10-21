#!/bin/bash
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/workdata/cap-x/.venv/bin/python3
D=reports/continuity_20261016
N=12
for i in $(seq 0 $((N-1))); do
  $PY scripts/p11_continuity_verify.py $D/response_curves.json $D/verify.jsonl 9301,9302,9303,9304,9305,9306 18 $i/$N > $D/verify.$i.log 2>&1 &
done
wait
echo "verify done $(date)" > $D/verify.done
