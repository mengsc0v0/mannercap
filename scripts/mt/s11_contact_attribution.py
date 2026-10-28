"""Step 11.1: attribute the 0/32 contact-policy failures of the E3 retrieval comparison.

Why this exists: the E3 trial records kept only CONTACT COUNTS, not the contact identities, so the existing artifacts
cannot say WHICH body touched WHAT in WHICH stage.  The full detail is already produced by contact_audit.report(); it
was simply not written out.  Rather than re-running all 32 requests, this re-executes ONE representative episode per
task and dumps the complete report.

Representative case = the episode group B accepted for request R1 on init 26, i.e. an episode that passed the
development criterion and would have been "shipped" if the contact policy were ignored.
One case per task, not one overall: T0 and T7 are different scenes with different requested objects, and the saved
counts already differ in shape (T0 has >=5 N forbidden contacts, T7 has none), so a single case cannot stand for both.

Usage: s11_contact_attribution.py   ->  reports/step11_contact/case_t{T}_init{I}.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS  # noqa: E402
from s8_task_force import grip_for, per_object, run_episode  # noqa: E402

E1 = json.loads((ROOT / "configs" / "experience_retrieval_e1.json").read_text())
OUT = ROOT / "reports" / "step11_contact"
OTHER = E1["scope"]["other_object_fixed_plan"]
SETTLE = E1["executor"]["settle_after_open"]

# (task, init, request id) -> the candidate group B accepted, read from the committed E3 trial records
CASES = [(0, 26, "R1"), (7, 26, "R1")]


def accepted_candidate(task, init, rid):
    f = ROOT / "reports" / "step10_experience" / "trials" / "B" / f"t{task}_init{init}_{rid}.json"
    r = json.loads(f.read_text())
    a = r["attempts"][-1]
    assert a["task_and_force_band_checks_passed"], "representative case must be an accepted episode"
    return tuple(a["candidate"]), r["object"], r["request"]["band_n"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for task, init, rid in CASES:
        cand, obj, band = accepted_candidate(task, init, rid)
        roles = [r for r, _ in SPECS[task].objects]
        req_role = [r for r, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
        grips = {r: grip_for((cand[0], cand[1]) if r == req_role else (OTHER["target_lo_n"], OTHER["target_hi_n"]), SETTLE)
                 for r in roles}
        prog, ep, rec, dt = run_episode(task, init, grips, None)
        o = per_object(ep, prog, task, rec)[req_role]
        ca = rec["contact_audit"]
        # which contacts involve the requested object, and which involve the non-requested object held at a fixed plan
        other_names = {n for _, n in SPECS[task].objects if n != obj}
        for c in ca["forbidden_contacts"]:
            tgt = c["other"].replace("_main", "")
            c["involves_requested_object"] = bool(tgt == obj or obj.startswith(tgt))
            c["involves_non_requested_fixed_object"] = bool(any(tgt == n or n.startswith(tgt) for n in other_names))
        out = {"schema": "mannercap.contact_attribution.v1", "e1_hash": E1["content_hash"],
               "task": task, "init": init, "request": rid, "band_n": band, "candidate": list(cand),
               "requested_object": obj, "other_objects": sorted(other_names), "role": req_role,
               "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "wall_s": round(dt, 2),
               "measured": {k: o.get(k) for k in ("hold_min", "hold_mean", "hold_max")},
               "development_criterion_passed": True,
               "contact_audit": ca,
               "note": "re-execution of a single committed E3 episode; the simulator is deterministic for a fixed "
                       "init and plan, so the counts below must match the counts stored in the E3 trial record."}
        f = OUT / f"case_t{task}_init{init}.json"
        f.write_text(json.dumps(out, indent=1, default=float))
        print(f"\n=== T{task} i{init} {rid} band={band} cand={list(cand)} obj={obj} task_ok={ep.task_ok} "
              f"hold={o.get('hold_min')}..{o.get('hold_max')}")
        print(f"    clean={ca['clean']} max_forbidden_n={ca['max_forbidden_n']} by_class={ca['forbidden_by_class']} "
              f"preview_if_finger_link_allowed={ca['clean_if_finger_link_on_target_allowed__preview_not_verdict']}")
        for c in ca["forbidden_contacts"]:
            print(f"    FORBIDDEN {c['robot_body']:44s} vs {c['other']:24s} max={c['max_n']:6.1f}N "
                  f"steps={c['control_steps']:4d} class={c['class']:22s} req_obj={c['involves_requested_object']} "
                  f"stages={c.get('stages')}")
        for c in ca["support_contacts_observed"]:
            print(f"    support   {c['robot_body']:44s} vs {c['other']:24s} max={c['max_n']:6.1f}N steps={c['control_steps']}")
        if ca.get("rolled_back_not_counted"):
            print(f"    rolled_back_not_counted: {ca['rolled_back_not_counted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
