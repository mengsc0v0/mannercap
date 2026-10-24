"""Aligned grasp curve (dev): gripper action, left/right finger force on the target, target height, per control step of
grasp+lift, same initial state, old (baseline) vs new (settle) release policy.  Usage: s7_grasp_curve.py TASK INIT ROLE"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from s7_task_config import GRIP_CONFIGS, SPECS, make_plan, make_program, make_session
from synthesis_v2.program_executor import produce_program_episode

T, I, ROLE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
out = Path("reports/step7_grasp_closure"); rows = {}
for cfg in ("baseline", "settle"):
    prog = make_program(T); s = make_session(T, I, prog)
    ep = produce_program_episode(prog, I, make_plan(T, prog, grip=GRIP_CONFIGS[cfg], approach_check=True), keep_images=False, session=s)
    a = ep.arrays; st = np.array([str(x) for x in a["stage"]]); key = dict(SPECS[T].objects)[ROLE]
    idx = np.where(np.isin(st, [f"{ROLE}.pick_1.grasp", f"{ROLE}.pick_1.lift"]))[0]
    z0 = float(a["pos_" + key][idx[0]][2])
    rows[cfg] = {"task_ok": bool(ep.task_ok), "t": idx.tolist(), "stage": [st[j].split(".")[-1] for j in idx],
                 "grip_action": [float(a["actions"][j, 6]) for j in idx], "left_n": [float(a["cf_L_" + key][j]) for j in idx],
                 "right_n": [float(a["cf_R_" + key][j]) for j in idx], "obj_rise_mm": [1000 * (float(a["pos_" + key][j][2]) - z0) for j in idx]}
(out / f"curve_t{T}_i{I}_{ROLE}.json").write_text(json.dumps(rows))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex="col")
    for c, cfg in enumerate(rows):
        r = rows[cfg]; t = np.array(r["t"]) - r["t"][0]
        ax[0, c].step(t, r["grip_action"], where="post"); ax[0, c].set_ylabel("gripper action")
        ax[1, c].plot(t, r["left_n"], label="left"); ax[1, c].plot(t, r["right_n"], label="right"); ax[1, c].set_ylabel("finger force (N)"); ax[1, c].legend()
        ax[2, c].plot(t, r["obj_rise_mm"]); ax[2, c].set_ylabel("object rise (mm)"); ax[2, c].set_xlabel("control step since grasp start")
        ax[0, c].set_title(f"{cfg}  task_ok={r['task_ok']}")
    fig.tight_layout(); fig.savefig(out / f"curve_t{T}_i{I}_{ROLE}.png", dpi=110)
except Exception as e:
    print("plot skipped", e)
for cfg, r in rows.items():
    print("C", cfg, "ok", r["task_ok"], "final rise mm %.0f" % r["obj_rise_mm"][-1], "L/R at grasp end", round(r["left_n"][r["stage"].index("lift") - 1], 1), round(r["right_n"][r["stage"].index("lift") - 1], 1))
