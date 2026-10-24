"""Per-control-step grasp trace (dev, read-only): gripper command, finger positions, left/right force, object z.
Usage: s7_grasp_trace.py TASK INIT ROLE"""
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
idx = np.where(np.isin(st, [f"{ROLE}.pick_1.grasp", f"{ROLE}.pick_1.lift"]))[0]
print(f"T{T} i{I} {ROLE} ok={ep.task_ok}")
for j in idx[:14]:
    print(f"  t={j:4d} {st[j].split('.')[-1]:5s} act_g={a['actions'][j,6]:+.1f} fingers=({a['state'][j,6]*1000:5.2f},{a['state'][j,7]*1000:6.2f})mm cfL={a['cf_L_'+key][j]:5.1f} cfR={a['cf_R_'+key][j]:5.1f} sum={a['cf_L_'+key][j]+a['cf_R_'+key][j]:5.1f} obj_z={a['pos_'+key][j][2]:.4f}")
