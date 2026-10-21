"""P0-3 probe in the deployment controller: contact-feedback native-7D grasp vs target force.

usage: p03_native_force_probe.py TASK SEEDS(comma) TARGETS(comma 'lo:hi' pairs) OUT.json [v_max]
"""
import json, sys, time
import numpy as np
from synthesis_v2.native_producer import GripPlan, ProductionPlan, default_stage_plans, produce_episode

task = sys.argv[1]
seeds = [int(x) for x in sys.argv[2].split(",")]
bands = [tuple(float(y) for y in x.split(":")) for x in sys.argv[3].split(",")]
out = sys.argv[4]
vmax = float(sys.argv[5]) if len(sys.argv) > 5 else 0.12
rows = []
for lo, hi in bands:
    for seed in seeds:
        plan = ProductionPlan(task, default_stage_plans(task, vmax), GripPlan(lo, hi))
        t0 = time.time()
        ep = produce_episode(task, seed, plan, keep_images=False)
        a = ep.arrays
        st = a["stage"]
        m = np.isin(st, ["lift", "transport"]) if task == "stack" else (st == "lift")
        tot = a["left_mean_n"] + a["right_mean_n"]
        row = dict(task=task, seed=seed, lo=lo, hi=hi, status=ep.status, ok=ep.task_ok, n=ep.n_steps,
                   grasp=ep.meta.get("grasp_policy"),
                   hold_med=float(np.nanmedian(tot[m])) if m.any() else None,
                   hold_min=float(np.nanmin(tot[m])) if m.any() else None,
                   hold_max=float(np.nanmax(tot[m])) if m.any() else None,
                   peak=float(np.nanmax(a["left_max_n"] + a["right_max_n"])) if ep.n_steps else None,
                   js_peak={s: float(np.nanmax(a["joint_speed_max"][st == s])) for s in set(st.tolist())},
                   sec=round(time.time() - t0, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"))
