"""Hold-force diagnosis (dev, read-only): per pick, peak grasp force, force at lift start, finger gap, object rise.
Usage: s7_hold_diag.py TASK INIT..."""
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
        g = np.where(st == f"{role}.pick_1.grasp")[0]; l = np.where(st == f"{role}.pick_1.lift")[0]
        if not len(g) or not len(l): print(f"T{T} i{I} {role}: no grasp/lift"); continue
        F = a["cf_L_" + key] + a["cf_R_" + key]; S = a["state"]; P = a["pos_" + key]
        print(f"T{T} i{I} {role:15s} ok={ep.task_ok} peakF={F[g].max():5.1f} grasp_end_F={F[g[-1]]:4.1f} lift_start_F={F[l[0]]:4.1f} lift_mid_F={F[l[len(l)//2]]:4.1f} "
              f"gap_end={(S[g[-1],6]-S[g[-1],7])*100:4.2f}cm gap_liftend={(S[l[-1],6]-S[l[-1],7])*100:4.2f}cm obj_rise={(P[l[-1],2]-P[g[0],2])*100:5.1f}cm", flush=True)
