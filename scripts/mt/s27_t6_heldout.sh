#!/bin/bash
cd /workdata/stage_cap_privileged_data_generation_repro_20260629/robosuite/stagecap_robosuite_libero_repro_20260714/new
O=reports/step27_t6_repair/heldout
for i in $(seq 30 49); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=.:scripts/mt timeout 1500 /workdata/cap-x/.venv-libero/bin/python scripts/mt/s27_contact_diag.py 6 reports/step27_t6_repair/frozen_proposal.json $i $O/init$i.json > $O/init$i.log 2>&1
  python3 -c "
import json;d=json.load(open('$O/init$i.json'))
u=[(c['robot_body'],c['other'],c['v1_reasons'],c['control_steps']) for c in d['contacts'] if not c['v1_excused']]+[(x['robot_body'],x['other'],x['max_n'],x['stages']) for x in d['support_unexplained']]
print('init',$i,d['status'],'success',d['official_success_final'],'clean',d['clean_v1'],u)" 2>&1 | tail -1
done
echo HELDOUT_DONE
