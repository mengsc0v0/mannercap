"""Single-episode probe for the 'object_is_not_a_bound_grasp_target' contacts (not a batch re-run).

Question: is the hand-vs-tomato_sauce contact in T7 a real contact with a non-target object, or a missing binding/record?
We re-run ONE episode through the same code path as the 2x2 (s12.run_guarded, same guard, same candidate) and print, for
every contact key that involves the object in question: the substep count, the peak force, the geoms, where on the geom it
touches, the stages it occurs in, and how far the object ended up from where it started.
"""
import json
import sys

import numpy as np

import s12_recovery_ablation as A

task, init, cand = int(sys.argv[1]), int(sys.argv[2]), (float(sys.argv[3]), float(sys.argv[4]))
who = sys.argv[5]
roles = [r for r, _ in A.SPECS[task].objects]
req_role = [r for r, n in A.SPECS[task].objects if n.startswith(A.E1["scope"]["requested_object"][str(task)])][0]
grips = {r: A.grip_for(cand if r == req_role else (A.OTHER["target_lo_n"], A.OTHER["target_hi_n"]), A.SETTLE) for r in roles}
guard = {**A.GUARD, "on_refusal": "restore", "pre_rejected": {}}
prog, ep, rec, crec, det, dt = A.run_guarded(task, init, grips, guard)
stages = [str(x) for x in ep.arrays["stage"]]
wins = A.windows_from(ep, task)
print("status", ep.status, "n_steps", ep.n_steps, "task_ok", ep.task_ok)
print("windows (bound grasp targets):", json.dumps([{k: w.get(k) for k in ("instance_id", "object", "close_step", "release_command_step", "release_step")} for w in wins]))
exp = det.export()
for k, row in exp.items():
    if who in k:
        steps = sorted({s for (rb, ob), (mx, st) in crec.events.items() if f"{rb}||{ob}" == k for s in st})
        print("KEY", k)
        print("  steps", steps[:3], "..", steps[-3:], "n_ctrl_steps", len(steps), "stages", sorted({stages[s] for s in steps if s < len(stages)}))
        print("  ", json.dumps({kk: row[kk] for kk in row if kk in ("n_substeps", "peak_n", "peak_step", "peak_geom", "peak_other_geom",
              "peak_contact_point_body_local_m", "peak_eef_pos", "side_role")}, default=float))
