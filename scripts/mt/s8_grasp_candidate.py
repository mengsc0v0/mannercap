"""Step 8: widened grasp-candidate test on one development case.

  s8_grasp_candidate.py enum  TAG TASK INIT ROLE   -> geometry only: every candidate of the widened grid is evaluated with the
                                                      unchanged clearance criterion (no episode is executed)
  s8_grasp_candidate.py run   TAG TASK INIT ROLE   -> executes ONE candidate chosen by the rule below, from the snapshot taken
                                                      before the approach instance; force/motion parameters are the fixed ones

Selection rule (fixed before any outcome was looked at): among candidates of ``extended_grid`` that pass the unchanged
geometry check (min whole-hand clearance over the path > 0 and open-hand clearance to the target at the grasp pose > 0) and
differ from the candidate the default grid chose, take the first in the grid's preference order (smallest departure from the
centred grasp).  Nothing is tuned after the outcome; if that candidate fails, it is reported as failed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program, make_session  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for  # noqa: E402
from s8_transport_repair import compare, run  # noqa: E402
from s8_shared_repair import full_audit, save_traj, view  # noqa: E402
from s8_diagnose_hold import per_stage  # noqa: E402
from synthesis_v2.approach_planner import extended_grid, plan_approach, scene_from_libero  # noqa: E402
from synthesis_v2.program_executor import APPROACH_DZ, _Geometry, _subtask_roles  # noqa: E402

OUT = ROOT / "reports" / "step8_grasp_candidates"


def enumerate_candidates(t, init, role):
    prog = make_program(t)
    s = make_session(t, init, prog)
    obs = s.reset()
    meta = {"down_quat_xyzw": np.asarray(obs["robot0_eef_quat"], float).tolist(), "subtasks": {}}
    it = next(i for i in prog.instances if i.object_role == role and i.primitive == "approach")
    geo = _Geometry(s, prog, True, meta)
    rec = geo.capture(it.subtask, *_subtask_roles(prog, it.subtask))
    nm = prog.objects[role]
    m, d, site, env, obst, tgt, look = scene_from_libero(s, nm, return_steps=True)

    def plan(grid, all_):
        return plan_approach(m, d, site, env, obst, start_pos=np.asarray(s.obs["robot0_eef_pos"], float),
                             start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"], float),
                             grasp_pos=np.asarray(rec["obj_pos0"], float), grasp_quat_xyzw=np.asarray(rec["grasp_quat_xyzw"], float),
                             hover_dz=APPROACH_DZ, target=tgt, grid=grid, evaluate_all=all_).to_dict()
    default = plan(None, False)
    ext = plan(extended_grid(), True)
    s.close()
    return prog, it, default, ext


def key(c):
    return (c["yaw_deg"], c["lateral_m"], c["dz_m"], c.get("longitudinal_m", 0.0))


def select(default, ext):
    cur = key(default["chosen"]) if default["chosen"] else None
    for c in ext["candidates"]:
        if c["feasible"] and key(c) != cur:
            return c
    return None


def main():
    mode, tag, t, init, role = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    OUT.mkdir(parents=True, exist_ok=True)
    prog0, it0, default, ext = enumerate_candidates(t, init, role)
    feas = [c for c in ext["candidates"] if c["feasible"]]
    summ = {"request": f"T{t}_init{init}_{role}", "default_grid": {"chosen": default["chosen"], "n_evaluated": len(default["candidates"])},
            "extended_grid": {"n_candidates": len(ext["candidates"]), "n_feasible": len(feas),
                              "feasible_by_yaw_deg": {str(y): sum(1 for c in feas if c["yaw_deg"] == y) for y in sorted({c["yaw_deg"] for c in feas})},
                              "first_feasible_in_preference_order": feas[:12]},
            "criterion": {"margin_m": ext["margin_m"], "rule": "worst_path_clearance > margin and at_grasp_clearance > 0 (unchanged)"}}
    # consistency: the widened grid evaluated with evaluate_all must contain the default candidate set's verdicts
    sel = select(default, ext)
    summ["selected"] = sel
    if mode == "enum":
        json.dump({"summary": summ, "all_candidates": ext["candidates"]}, open(OUT / f"{tag}_enumeration.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in summ.items() if k != "extended_grid"}, indent=1)[:1500])
        print(json.dumps({k: v for k, v in summ["extended_grid"].items() if k != "first_feasible_in_preference_order"}))
        return
    assert sel is not None, "no new geometry-passing candidate"
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    snaps = {}
    A = run(t, init, prog0, grips, snapshots=snaps)
    gc = {"yaw_deg": sel["yaw_deg"], "lateral_m": sel["lateral_m"], "dz_m": sel["dz_m"], "longitudinal_m": sel.get("longitudinal_m", 0.0)}
    prog1 = replace(prog0, instances=tuple(replace(i, params={**i.params, "grasp_candidate": gc}) if i.label == it0.label else i
                                           for i in prog0.instances))
    snap = snaps[it0.label]
    k0 = int(snap.session["control_step_index"])
    C = run(t, init, prog1, grips, resume=snap, seed={"sub": A["sub"], "crec": A["crec"], "car": A["car"]})
    B = run(t, init, prog1, grips)
    tdir = ROOT / "reports" / "step8_shared_repair" / "trajectories"
    name = prog0.objects[role]
    out = {"request": summ["request"], "selected_candidate": gc, "selection": summ["selected"], "default_chosen": default["chosen"],
           "resume_point": {"stage_instance": snap.stage, "control_step_index": k0, "prefix_steps": int(snap.prefix_steps)},
           "executor": EXECUTOR, "plan": "gentle", "plan_range_n": list(PLANS["gentle"]), "trajectories": {}}
    for c, R in (("A", A), ("B", B), ("C", C)):
        v = view(R, prog0 if c == "A" else prog1)
        a_ = {k: x for k, x in v.items() if not k.startswith("_")}
        a_["full_audit"] = full_audit(t, R, prog0 if c == "A" else prog1)
        z_path = tdir / f"{tag}_T{t}_init{init}_{c}.npz"
        out["trajectories"][c] = save_traj(z_path, R)
        z = np.load(z_path, allow_pickle=False)
        a_["per_stage"], _, _ = per_stage(z, prog0 if c == "A" else prog1, role, name)
        out[c] = a_
    ncols = 4 * len(A["names"])
    out["B_vs_C"] = {"from_k0": compare(B, C, k0, ncols), "all_substeps_bitwise_equal": bool(B["arr"].shape == C["arr"].shape and np.array_equal(B["arr"], C["arr"]))}
    out["cost"] = {"A": out["A"]["cost_executed_substeps"], "B_full_rerun": out["B"]["cost_executed_substeps"], "C_stage_resume": out["C"]["cost_executed_substeps"]}
    json.dump(out, open(OUT / f"{tag}_T{t}_init{init}.run.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for c in "ABC":
        ck = out[c]["checks"]
        print(c, "task_ok", ck["task_ok"], "local_checks", ck["task_and_hold_checks_passed"], "hold_states", ck["hold_state_counts_by_object"],
              "choice", out[c]["approach_choice"])
    print("B_vs_C bitwise", out["B_vs_C"]["all_substeps_bitwise_equal"], "cost", out["cost"])


main()
