#!/bin/bash
# usage: run_jobs.sh OUT JOBS LOG [extra env assignments via env]
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=.:scripts/mt
export S30_OUT=$1
cat $2 | xargs -P ${P:-16} -L 1 sh -c '/workdata/cap-x/.venv-libero/bin/python scripts/mt/s30_manner_gen.py run $0 $1 $2 2>/dev/null | grep "^T"' >> $3
date -u +%FT%TZ > $1/end_utc.txt
