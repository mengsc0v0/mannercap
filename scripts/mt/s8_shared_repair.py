"""Step 8: shared repair v0 on a list of requests.  Per request: run the request, let the rule (synthesis_v2/transport_repair_rule.py)
diagnose / propose from the run itself, and only for a proposal: stage resume (C) from the start of the modified transport instance and,
for the comparison, a full re-run (B) of the same modified program.  Requests without a trigger are kept and reported.
The rule never receives the task id or the init index; they are used here only to build the session and to name the output.

  s8_shared_repair.py OUT_TAG T:INIT [T:INIT ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, cfg_path, grip_for  # noqa: E402
from s8_transport_repair import compare, run, summarize  # noqa: E402
import synthesis_v2.transport_repair_rule as RULE  # noqa: E402

_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731


def view(R, prog):
    a = R["ep"].arrays
    stages = [str(x) for x in a["stage"]]
    L, Rr = np.asarray(a["left_mean_n"], float), np.asarray(a["right_mean_n"], float)
    ev = {k: [v[0], sorted(v[1])] for k, v in R["car"].events.items()}
    chk = RULE.request_checks(prog, stages, L, Rr, {k: (v[0], set(v[1])) for k, v in ev.items()}, R["ep"].status, R["ep"].task_ok, valid=a["valid_fraction"])
    c = R["sub"].cost()
    return {"checks": chk, "n_steps": len(stages), "program_hash": prog.program_hash(), "plan_hash": R["plan"].plan_hash(),
            "cost_executed_substeps": c["executed_substeps"], "wall_s": round(R["wall"], 1),
            "object_contact_events_all_stages": [{"object": k[0], "other_body": k[1], "peak_n": round(v[0], 1), "control_steps": len(v[1])} for k, v in sorted(ev.items()) if v[0] >= RULE.CONTACT_THR_N],
            "approach_choice": {k: {"chosen": v.get("chosen"), "default_feasible": v.get("default_feasible"), "n_candidates_evaluated": len(v.get("candidates", []))}
                                for k, v in R["ep"].meta.get("approach_plan", {}).items()},
            "_stages": stages, "_L": L, "_R": Rr, "_V": np.asarray(a["valid_fraction"], float)}


def save_traj(path: Path, R) -> dict:
    """Raw trajectory of one condition: every control-step array of the episode, per-physics-substep force columns
    (L, R, palm/other-gripper, basket per object), their control-step index and the object contact events."""
    ev = {f"{k[0]}|{k[1]}": [float(v[0]), sorted(int(x) for x in v[1])] for k, v in R["car"].events.items()}
    np.savez_compressed(path, **{f"ep_{k}": np.asarray(v) for k, v in R["ep"].arrays.items()}, sub_ctrl=R["ctrl"], sub_arr=R["arr"],
                        sub_columns=np.asarray([f"{n}.{c}" for n in R["names"] for c in ("L", "R", "palm_other_gripper", "basket")]),
                        car_events_json=np.asarray(json.dumps(ev)))
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def full_audit(t, R, prog):
    a = summarize(t, None, R, prog)
    return {k: a[k] for k in ("status", "task_ok", "objects", "contact_policy", "cost")}


def one_request(t, init, tag=None):
    prog0 = make_program(t)
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    snaps = {}
    A = run(t, init, prog0, grips, snapshots=snaps)
    va = view(A, prog0)
    trig = RULE.diagnose(prog0, va["_stages"], va["_L"], va["_R"], {k: (v[0], set(v[1])) for k, v in A["car"].events.items()}, valid=va["_V"])
    prop, decision = RULE.propose(prog0, trig)
    rec = {"request": f"T{t}_init{init}", "rule": RULE.RULE_VERSION, "A": {k: v for k, v in va.items() if not k.startswith("_")},
           "decision": decision, "proposal": None if prop is None else {"instance": prop.instance_label, "old_dz": prop.old_dz, "new_dz": prop.new_dz, "failure_step": prop.failure_step, "trigger": prop.trigger.__dict__}}
    out = {"A": A}
    tdir = ROOT / "reports" / "step8_shared_repair" / "trajectories"
    tdir.mkdir(parents=True, exist_ok=True)
    rec["trajectories"] = {}
    rec["A"]["full_audit"] = full_audit(t, A, prog0)
    if tag:
        rec["trajectories"]["A"] = save_traj(tdir / f"{tag}_{rec['request']}_A.npz", A)
    if prop is not None:
        prog1 = RULE.apply(prog0, prop)
        snap = snaps[prop.instance_label]
        k0 = int(snap.session["control_step_index"])
        C = run(t, init, prog1, grips, resume=snap, seed={"sub": A["sub"], "crec": A["crec"], "car": A["car"]})
        B = run(t, init, prog1, grips)
        vb, vc = view(B, prog1), view(C, prog1)
        ncols = 4 * len(A["names"])
        rec["resume_point"] = {"stage_instance": snap.stage, "control_step_index": k0, "prefix_steps": int(snap.prefix_steps)}
        rec["B_full_rerun"] = {k: v for k, v in vb.items() if not k.startswith("_")}
        rec["C_stage_resume"] = {k: v for k, v in vc.items() if not k.startswith("_")}
        rec["B_vs_C"] = {"from_k0": compare(B, C, k0, ncols),
                         "all_substeps_bitwise_equal": bool(B["arr"].shape == C["arr"].shape and np.array_equal(B["arr"], C["arr"])),
                         "ctrl_index_equal": bool(np.array_equal(B["ctrl"], C["ctrl"]))}
        a_full, b_x, c_x = va["cost_executed_substeps"], vb["cost_executed_substeps"], vc["cost_executed_substeps"]
        rec["cost"] = {"A_full": a_full, "B_extra_full_rerun": b_x, "C_extra_stage_resume": c_x,
                       "first_run_plus_one_repair_attempt_via_B_rerun": a_full + b_x, "first_run_plus_one_repair_attempt_via_C_resume": a_full + c_x}
        rec["B_full_rerun"]["full_audit"], rec["C_stage_resume"]["full_audit"] = full_audit(t, B, prog1), full_audit(t, C, prog1)
        if tag:
            rec["trajectories"]["B"], rec["trajectories"]["C"] = (save_traj(tdir / f"{tag}_{rec['request']}_{c}.npz", R) for c, R in (("B", B), ("C", C)))
        rec["outcome"] = {"local_checks_passed_after_resume": vc["checks"]["task_and_hold_checks_passed"], "local_checks_passed_after_full_rerun": vb["checks"]["task_and_hold_checks_passed"],
                          "task_ok_A_to_C": [va["checks"]["task_ok"], vc["checks"]["task_ok"]]}
        out.update(B=B, C=C)
    rec["case"] = case_record(t, init, tag, prog0, rec)
    return rec


def case_record(t, init, tag, prog0, rec) -> dict:
    """Minimal case record (traceability now, experience-library input later): request, objects/stages, executor and calibration
    versions, failure state, candidate change, resume point, measured values before/after, full audit, cost, raw trajectories.
    No benefit claim is attached to it."""
    cal = Path(cfg_path(t))
    a = rec["A"]["checks"]
    fin = rec.get("C_stage_resume", rec["A"])
    return {"schema": "mannercap.repair_case.v1", "request": {"task": t, "init": init, "spec_program_hash": prog0.program_hash(),
            "objects": dict(prog0.objects), "stage_instances": [it.label for it in prog0.instances]},
            "versions": {"rule": RULE.RULE_VERSION, "checks": RULE.CHECKS_VERSION, "executor": EXECUTOR, "plan": "gentle", "plan_range_n": list(PLANS["gentle"]),
                         "calibration_file": str(cal.relative_to(ROOT)) if cal.is_absolute() and ROOT in cal.parents else str(cal),
                         "calibration_sha256": hashlib.sha256(cal.read_bytes()).hexdigest() if cal.exists() else None},
            "failure_state": {"task_ok": a["task_ok"], "hold_state_counts_by_object": a["hold_state_counts_by_object"],
                              "unknown_objects": a["unknown_objects"], "residual_triggers": a["residual_triggers"],
                              "object_contact_events_all_stages": rec["A"]["object_contact_events_all_stages"]},
            "candidate": rec["proposal"], "resume_point": rec.get("resume_point"),
            "before": {"checks": a, "audit_contact_policy": rec["A"]["full_audit"]["contact_policy"]},
            "after": None if "C_stage_resume" not in rec else {"checks": fin["checks"], "audit_contact_policy": fin["full_audit"]["contact_policy"],
                                                               "B_vs_C_bitwise": rec["B_vs_C"]["all_substeps_bitwise_equal"]},
            "cost": rec.get("cost", {"A_full": rec["A"]["cost_executed_substeps"]}), "trajectories": rec["trajectories"], "tag": tag}


def aggregate(recs) -> dict:
    """Corrected whole-list accounting (denominators stated).  Abnormal grasp = any non-bilateral or invalid hold step in A."""
    n = len(recs)
    ab = lambda r: sum(r["A"]["checks"]["non_bilateral_steps_by_object"].values()) > 0 or bool(r["A"]["checks"]["unknown_objects"])  # noqa: E731
    fail = lambda r: ab(r) or not r["A"]["checks"]["task_ok"]  # noqa: E731
    trig = [r for r in recs if r["decision"]["n_triggers"] > 0]
    prop = [r for r in recs if r["proposal"]]
    neg = [r for r in recs if not ab(r)]
    unc = [r["request"] for r in recs if fail(r) and r["decision"]["n_triggers"] == 0]
    a_all = sum(r["A"]["cost_executed_substeps"] for r in recs)
    b_x = sum(r["cost"]["B_extra_full_rerun"] for r in prop)
    c_x = sum(r["cost"]["C_extra_stage_resume"] for r in prop)
    fin_task = sum(1 for r in recs if (r["C_stage_resume"]["checks"]["task_ok"] if r["proposal"] else r["A"]["checks"]["task_ok"]))
    fin_local = sum(1 for r in recs if (r["C_stage_resume"]["checks"]["task_and_hold_checks_passed"] if r["proposal"] else r["A"]["checks"]["task_and_hold_checks_passed"]))
    return {"n_requests": n, "abnormal_grasp_in_A": [sum(ab(r) for r in recs), n], "triggered": [len(trig), n], "proposed": [len(prop), n],
            "local_checks_passed_after_resume": [sum(r["outcome"]["local_checks_passed_after_resume"] for r in prop), len(prop)],
            "local_checks_passed_after_full_rerun": [sum(r["outcome"]["local_checks_passed_after_full_rerun"] for r in prop), len(prop)],
            "B_equals_C_bitwise": [sum(r["B_vs_C"]["all_substeps_bitwise_equal"] for r in prop), len(prop)],
            "false_triggers_among_no_abnormal_grasp": [sum(r["decision"]["n_triggers"] > 0 for r in neg), len(neg)],
            "uncovered_failures": unc, "local_checks_passed": {"A": [sum(r["A"]["checks"]["task_and_hold_checks_passed"] for r in recs), n], "after_repair": [fin_local, n]},
            "task_ok": {"A": [sum(r["A"]["checks"]["task_ok"] for r in recs), n], "after_repair": [fin_task, n]},
            "cost_substeps": {"first_runs_all": a_all, "repair_extra_full_rerun_triggered_only": b_x, "repair_extra_stage_resume_triggered_only": c_x,
                              "list_total_via_full_rerun": a_all + b_x, "list_total_via_stage_resume": a_all + c_x,
                              "list_reduction": None if a_all + b_x == 0 else round(1 - (a_all + c_x) / (a_all + b_x), 4),
                              "triggered_extra_reduction": None if b_x == 0 else round(1 - c_x / b_x, 4)}}


def main(tag, reqs):
    d = ROOT / "reports" / "step8_shared_repair"
    d.mkdir(exist_ok=True)
    allr = []
    for r in reqs:
        t, i = (int(x) for x in r.split(":"))
        rec = one_request(t, i, tag)
        allr.append(rec)
        (d / f"{tag}_{rec['request']}.json").write_text(json.dumps(rec, indent=1, default=_J))
        cd = ROOT / "reports" / "step8_cases"
        cd.mkdir(exist_ok=True)
        (cd / f"{tag}_{rec['request']}.case.json").write_text(json.dumps(rec["case"], indent=1, default=_J))
        a = rec["A"]["checks"]
        print(f"{rec['request']}: A local_ok={a['task_and_hold_checks_passed']} task_ok={a['task_ok']} non_bilateral={a['non_bilateral_steps_by_object']} triggers={rec['decision']['n_triggers']} "
              f"proposal={rec['proposal'] and (rec['proposal']['instance'], rec['proposal']['new_dz'])} "
              f"{'' if 'outcome' not in rec else rec['outcome']} {'' if 'cost' not in rec else rec['cost']}", flush=True)
    (d / f"{tag}_summary.json").write_text(json.dumps(allr, indent=1, default=_J))
    (d / f"{tag}_aggregate.json").write_text(json.dumps(aggregate(allr), indent=1, default=_J))
    print(json.dumps(aggregate(allr), default=_J), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
