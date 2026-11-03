#!/bin/bash
# usage: s27_round.sh ROUND PANEL_ANGLE DEPTH INITS...
R=$1; A=$2; D=$3; shift 3
cd /workdata/stage_cap_privileged_data_generation_repro_20260629/robosuite/stagecap_robosuite_libero_repro_20260714/new
O=reports/step27_t6_repair/round$R; mkdir -p $O
python3 scripts/mt/s27_t6_mkprop.py $R $A $D $O
for i in "$@"; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=.:scripts/mt timeout 1500 /workdata/cap-x/.venv-libero/bin/python scripts/mt/s27_contact_diag.py 6 $O/proposal.json $i $O/init$i.json > $O/init$i.log 2>&1
  python3 -c "
import json;d=json.load(open('$O/init$i.json'))
u=[(c['robot_body'],c['v1_reasons'],c['control_steps'],round(c['detail'].get('min_inner_cos',9),2)) for c in d['contacts'] if not c['v1_excused']]
print('init',$i,d['status'],'success',d['official_success_final'],'clean',d['clean_v1'],u)" 2>&1 | tail -1
done
