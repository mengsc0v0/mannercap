"""Step 8: shared repair v0 on a list of requests.  Per request: run the request, let the rule (synthesis_v2/transport_repair_rule.py)
diagnose / propose from the run itself, and only for a proposal: stage resume (C) from the start of the modified transport instance and,
for the comparison, a full re-run (B) of the same modified program.  Requests without a trigger are kept and reported.
The rule never receives the task id or the init index; they are used here only to build the session and to name the output.

  s8_shared_repair.py OUT_TAG T:INIT [T:INIT ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for  # noqa: E402
from s8_transport_repair import compare, run  # noqa: E402
import synthesis_v2.transport_repair_rule as RULE  # noqa: E402

_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731


def view(R, prog):
    a = R["ep"].arrays
    stages = [str(x) for x in a["stage"]]
    L, Rr = np.asarray(a["left_mean_n"], float), np.asarray(a["right_mean_n"], float)
    ev = {k: [v[0], sorted(v[1])] for k, v in R["car"].events.items()}
    chk = RULE.request_checks(prog, stages, L, Rr, {k: (v[0], set(v[1])) for k, v in ev.items()}, R["ep"].status, R["ep"].task_ok)
    c = R["sub"].cost()
    return {"checks": chk, "n_steps": len(stages), "program_hash": prog.program_hash(), "plan_hash": R["plan"].plan_hash(),
            "cost_executed_substeps": c["executed_substeps"], "wall_s": round(R["wall"], 1),
            "object_contact_events_all_stages": [{"object": k[0], "other_body": k[1], "peak_n": round(v[0], 1), "control_steps": len(v[1])} for k, v in sorted(ev.items()) if v[0] >= RULE.CONTACT_THR_N],
            "_stages": stages, "_L": L, "_R": Rr}


def one_request(t, init):
    prog0 = make_program(t)
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    snaps = {}
    A = run(t, init, prog0, grips, snapshots=snaps)
    va = view(A, prog0)
    trig = RULE.diagnose(prog0, va["_stages"], va["_L"], va["_R"], {k: (v[0], set(v[1])) for k, v in A["car"].events.items()})
    prop, decision = RULE.propose(prog0, trig)
    rec = {"request": f"T{t}_init{init}", "rule": RULE.RULE_VERSION, "A": {k: v for k, v in va.items() if not k.startswith("_")},
           "decision": decision, "proposal": None if prop is None else {"instance": prop.instance_label, "old_dz": prop.old_dz, "new_dz": prop.new_dz, "failure_step": prop.failure_step, "trigger": prop.trigger.__dict__}}
    out = {"A": A}
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
        rec["outcome"] = {"repaired_by_resume": vc["checks"]["request_ok"], "repaired_by_full_rerun": vb["checks"]["request_ok"],
                          "task_ok_A_to_C": [va["checks"]["task_ok"], vc["checks"]["task_ok"]]}
        out.update(B=B, C=C)
    return rec


def main(tag, reqs):
    d = ROOT / "reports" / "step8_shared_repair"
    d.mkdir(exist_ok=True)
    allr = []
    for r in reqs:
        t, i = (int(x) for x in r.split(":"))
        rec = one_request(t, i)
        allr.append(rec)
        (d / f"{tag}_{rec['request']}.json").write_text(json.dumps(rec, indent=1, default=_J))
        a = rec["A"]["checks"]
        print(f"{rec['request']}: A ok={a['request_ok']} task_ok={a['task_ok']} one_sided={a['one_sided_steps_by_object']} triggers={rec['decision']['n_triggers']} "
              f"proposal={rec['proposal'] and (rec['proposal']['instance'], rec['proposal']['new_dz'])} "
              f"{'' if 'outcome' not in rec else rec['outcome']} {'' if 'cost' not in rec else rec['cost']}", flush=True)
    (d / f"{tag}_summary.json").write_text(json.dumps(allr, indent=1, default=_J))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
