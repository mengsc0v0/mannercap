"""Per-pick grasp policy outcome + lift result (dev, read-only). Usage: s7_grasp_policy_diag.py TASK INIT..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from s7_task_config import SPECS, make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode
T = int(sys.argv[1])
for I in map(int, sys.argv[2:]):
    prog = make_program(T); s = make_session(T, I, prog)
    ep = produce_program_episode(prog, I, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
    a = ep.arrays; st = np.array([str(x) for x in a["stage"]])
    for role, key in SPECS[T].objects:
        l = np.where(st == f"{role}.pick_1.lift")[0]; P = a["pos_" + key]
        rise = (P[l[-1], 2] - P[l[0], 2]) * 100 if len(l) else float("nan")
        pol = {k: v for k, v in ep.meta["grasp_policy_by_instance"].items() if k.startswith(role)}
        print(f"G T{T} i{I} {role:14s} task_ok={ep.task_ok} lift_rise={rise:5.1f}cm policy={pol}", flush=True)
