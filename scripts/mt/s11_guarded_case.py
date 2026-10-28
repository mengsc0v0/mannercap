"""Step 11.2: run one representative case per task with the execution guard on, and judge it under BOTH contact
policies.  Usage: s11_guarded_case.py TASK INIT [REQUEST]   -> reports/step11_contact/guarded_t{T}_init{I}_{R}.json

The case is the request/candidate group B accepted in the committed E3 trial, so the force request is unchanged: the
only difference from step 11.1 is the guard and the extra recording.  The development criterion is RECOMPUTED from this
episode by the same function the trials used; it is never copied from the earlier result.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from s7_task_config import ROOT, SPECS, make_program, make_session  # noqa: E402
import s8_task_force as S8  # noqa: E402
from s10_retrieval_trial import passed as dev_criterion  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

E1 = json.loads((ROOT / "configs" / "experience_retrieval_e1.json").read_text())
OUT = ROOT / "reports" / "step11_contact"
OTHER = E1["scope"]["other_object_fixed_plan"]
SETTLE = E1["executor"]["settle_after_open"]
GUARD = {"enabled": True, "min_clearance_m": 0.0, "contact_threshold_n": 0.1}


def accepted(task, init, rid):
    r = json.loads((ROOT / "reports" / "step10_experience" / "trials" / "B" / f"t{task}_init{init}_{rid}.json").read_text())
    a = r["attempts"][-1]
    return tuple(a["candidate"]), r["object"], r["request"]["band_n"], a


def run(task, init, grips, guard):
    """Same episode construction as s8_task_force.run_episode, plus the v1 contact detail hook and the guard."""
    prog = make_program(task)
    s = make_session(task, init, prog)
    names = [n for _, n in SPECS[task].objects]
    sub = S8.install_recorder(s, names)
    box = {}
    t0 = time.time()
    plan = S8.make_plan(task, prog, grip=grips, approach_check=S8.EXECUTOR["approach_check"])

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))

    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s,
                                 on_session=on_session, execution_guard=guard)
    crec, det = box["c"], box["det"]
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 2 * len(names))
    rec = {"ctrl": ctrl, "L": {n: arr[:, 2 * i] for i, n in enumerate(names)}, "R": {n: arr[:, 2 * i + 1] for i, n in enumerate(names)},
           "rolled_back_cost": sub.rolled_back_cost(),
           "contact_audit": crec.report([str(x) for x in ep.arrays["stage"]])}
    return prog, ep, rec, crec, det, time.time() - t0


def windows_from(ep, task):
    """[close_step, release_step) per grasp instance, from the executor's own record, with the bound force target.

    release_step is the step at which the fingers have FINISHED opening, not the step at which the open command is
    issued: the opening motion is still authorised contact with the held object.
    """
    m = ep.meta
    out = []
    for iid, cs in sorted(m.get("close_start_step_by_instance", {}).items()):
        rs = m.get("release_end_step_by_instance", {}).get(iid)
        rs0 = m.get("release_start_step_by_instance", {}).get(iid)
        obj = m.get("force_target_by_instance", {}).get(iid)
        out.append({"instance_id": iid, "object": None if obj is None else f"{obj}_main".replace("_1_1", "_1"),
                    "env_object": obj, "close_step": int(cs),
                    "release_step": int(ep.n_steps if rs is None else rs),
                    "release_command_step": None if rs0 is None else int(rs0)})
    return out


def main() -> int:
    task, init = int(sys.argv[1]), int(sys.argv[2])
    rid = sys.argv[3] if len(sys.argv) > 3 else "R1"
    cand, obj, band, prev = accepted(task, init, rid)
    roles = [r for r, _ in SPECS[task].objects]
    req_role = [r for r, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    grips = {r: S8.grip_for((cand[0], cand[1]) if r == req_role else (OTHER["target_lo_n"], OTHER["target_hi_n"]), SETTLE)
             for r in roles}
    prog, ep, rec, crec, det, dt = run(task, init, grips, GUARD)
    o = S8.per_object(ep, prog, task, rec)[req_role]
    v0 = rec["contact_audit"]
    wins = windows_from(ep, task)
    v1 = eval_v1(v0, crec.events, det.detail, wins, [str(x) for x in ep.arrays["stage"]])
    # development criterion recomputed here from THIS episode, never copied from the earlier trial
    shaped = {"task_ok": bool(ep.task_ok),
              "measured": {k: o.get(k) for k in ("hold_n", "hold_min", "hold_mean", "hold_max", "lr_imbalance",
                                                 "rel_z_spread_m", "peak_simultaneous_n")},
              "validity": {"status": o.get("status"), "measurement_valid": o.get("measurement_valid"),
                           "grasp_stable": o.get("grasp_stable")}}
    dev_ok = dev_criterion(shaped, band)
    g = ep.meta.get("execution_guard", {})
    out = {"schema": "mannercap.guarded_case.v1", "e1_hash": E1["content_hash"],
           "task": task, "init": init, "request": rid, "band_n": band, "candidate": list(cand),
           "requested_object": obj, "role": req_role, "guard_config": GUARD,
           "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "wall_s": round(dt, 2),
           "measured": shaped["measured"], "validity": shaped["validity"],
           "development_criterion_passed": bool(dev_ok),
           "development_criterion_source": "recomputed by s10_retrieval_trial.passed() on this episode",
           "unguarded_reference": {"development_criterion_passed": prev["task_and_force_band_checks_passed"],
                                   "n_steps": prev["n_steps"], "hold_min": prev["measured"]["hold_min"],
                                   "hold_max": prev["measured"]["hold_max"],
                                   "contact_policy_passed_v0": prev["contact_policy_passed"]},
           "contact_policy_v0": v0, "contact_policy_v1": v1,
           "hold_windows": wins,
           "guard": {"n_checks": len(g.get("checks", [])), "n_retries": len(g.get("retries", [])),
                     "retries": g.get("retries", []), "chosen_candidate": g.get("chosen_candidate", {}),
                     "checks": g.get("checks", [])},
           "contact_detail": det.export(),
           "rolled_back_cost": rec["rolled_back_cost"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"guarded_t{task}_init{init}_{rid}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n=== T{task} i{init} {rid} band={band} cand={list(cand)} obj={obj}")
    print(f"  status={ep.status} task_ok={ep.task_ok} steps={ep.n_steps} (unguarded {prev['n_steps']}) wall={dt:.1f}s")
    print(f"  hold {o.get('hold_min')} .. {o.get('hold_max')}  dev_criterion={dev_ok} (unguarded {prev['task_and_force_band_checks_passed']})")
    print(f"  guard: checks={len(g.get('checks', []))} retries={len(g.get('retries', []))} chosen={g.get('chosen_candidate')}")
    for r in g.get("retries", []):
        print(f"    RETRY {r['check']} at step {r['control_step']} refused={r['refused_candidate']} -> {r.get('next_candidate')} "
              f"discarded_steps={r['steps_discarded']} resume_prefix={r['resume_prefix_steps']} viol={r['violations']}")
    print(f"  v0 clean={v0['clean']} by_class={v0['forbidden_by_class']}")
    for c in v0["forbidden_contacts"]:
        print(f"     v0 {c['robot_body']:42s} vs {c['other']:22s} {c['max_n']:7.1f}N steps={c['control_steps']:4d} {c['stages']}")
    print(f"  v1 clean_v1={v1['clean_v1']} (ignoring support {v1['clean_v1_ignoring_support']}) excused={v1['n_excused']}/{len(v1['contacts'])}")
    for c in v1["contacts"]:
        print(f"     v1 {c['robot_body']:42s} excused={str(c['v1_excused']):5s} cos={c['peak_normal_inward_cos']} "
              f"inward={c['peak_inward_offset_m']} inner={c['n_substeps_inner_region']}/{c['n_substeps']} steps={c['control_step_range']} {c['v1_reasons']}")
    for sct in v1["support_contacts_unexplained"]:
        print(f"     SUPPORT {sct['robot_body']:38s} vs {sct['other']:22s} {sct['max_n']:8.1f}N steps={sct['control_steps']} "
              f"range={sct['control_step_range']} stages={sct['stages']} peak_step={sct['peak_step']} geom={sct['peak_geom']}")
    print(f"  windows: {wins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
