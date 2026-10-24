"""Grasp/lift diagnosis for one picked object (dev, read-only): hand pose, finger opening and object pose through grasp+lift.
Usage: s7_lift_diag.py TASK INIT ROLE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from s7_task_config import SPECS, make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode
T, I, ROLE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
prog = make_program(T); s = make_session(T, I, prog)
ep = produce_program_episode(prog, I, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
a = ep.arrays; st = np.array([str(x) for x in a["stage"]]); key = dict(SPECS[T].objects)[ROLE]
P, S = a["pos_" + key], a["state"]
print(f"T{T} init{I} {ROLE} ok={ep.task_ok} status={ep.status} state_dim={S.shape[1]}", {k: v for k, v in list(ep.meta["approach_plan"].items())[0 if ROLE == SPECS[T].objects[0][0] else 1][1]["chosen"].items() if k in ("lateral_m", "dz_m")})
for stg in ("descend", "grasp", "lift"):
    idx = np.where(st == f"{ROLE}.pick_1.{stg}")[0]
    if not len(idx): continue
    for j in (idx[0], idx[len(idx) // 2], idx[-1]):
        print(f"  {stg:7s} t={j:4d} eef_xyz={np.round(S[j, :3], 3).tolist()} grip_state={np.round(S[j, 6:8], 4).tolist()} obj_xyz={np.round(P[j], 3).tolist()} "
              f"eef-obj={np.round(S[j, :3] - P[j], 3).tolist()} cfL={a['cf_L_' + key][j]:.1f} cfR={a['cf_R_' + key][j]:.1f} act_g={a['actions'][j, 6]:+.2f}")
