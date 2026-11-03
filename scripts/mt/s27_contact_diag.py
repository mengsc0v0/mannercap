"""Step 27 contact diagnosis: re-run ONE onboarding episode with the s26 wiring and dump the full contact-policy-v1.1 verdict.

For every forbidden contact: v1 reasons, matched window, control-step range, stage labels at first/last step, peak geometry
(geom, contact point in geom frame, normal, inward cos/offset), plus the permission windows and stage step ranges.
Usage: s27_contact_diag.py TASK PROPOSAL_JSON INIT OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s26_onboard as S26  # noqa: E402


def _js(x):
    if isinstance(x, (set, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> int:
    t, prop_path, init, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    p = json.loads(Path(prop_path).read_text())
    proposal = p.get("proposal", p)
    import s8_task_force as S8
    from s11_guarded_case import GUARD, windows_from
    from synthesis_v2.contact_audit import ContactRecorder
    from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1, POLICY_VERSION
    from synthesis_v2.libero_backend import LiberoSession
    from synthesis_v2.native_producer import StagePlan
    from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
    prog, refused = S26.compile_program(t, proposal)
    names = [o["object"] for o in proposal["objects"]]
    s = LiberoSession(t, init, render=False, default_target=names[0])
    box = {}

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))
    grip = S8.grip_for((1.0, 100.0), S8.EXECUTOR["settle_after_open"])
    plan = ProgramPlan(prog, tuple(StagePlan(it.label, 0.12) for it in prog.instances),
                       {f"obj{k}.pick_1.grasp": grip for k in range(len(names))}, approach_check=True)
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s, on_session=on_session,
                                 execution_guard={**GUARD, "on_refusal": "restore", "pre_rejected": {}})
    st = [str(x) for x in ep.arrays["stage"]]
    crec, det = box["c"], box["det"]
    wins = windows_from(ep, t)
    v1 = eval_v1(crec.report(st), crec.events, det.detail, wins, st)
    stage_ranges = {}
    for k, lab in enumerate(st):
        r = stage_ranges.setdefault(lab, [k, k]); r[1] = k
    rows = []
    for c in v1["contacts"]:
        key = (c["robot_body"], c["other"])
        steps = sorted(S26_steps(crec.events, key))
        dd = det.detail.get(key, {})
        rows.append({**{k: c.get(k) for k in ("robot_body", "other", "class", "max_n", "v1_excused", "v1_reasons",
                                               "matched_instance")},
                     "control_steps": [steps[0], steps[-1]] if steps else None, "n_steps": len(steps),
                     "stage_first": st[steps[0]] if steps and steps[0] < len(st) else None,
                     "stage_last": st[steps[-1]] if steps and steps[-1] < len(st) else None,
                     "detail": {k: v for k, v in dd.items() if k != "region_fail_samples"},
                     "region_fail_samples_first5": list(dd.get("region_fail_samples", []))[:5]})
    rec = {"task": t, "init": init, "proposal": prop_path, "policy_version": POLICY_VERSION,
           "status": ep.status, "official_success_final": bool(ep.task_ok), "clean_v1": v1["clean_v1"],
           "windows": wins, "stage_ranges": stage_ranges, "contacts": rows,
           "support_unexplained": v1["support_contacts_unexplained"],
           "arrays_keys": sorted(ep.arrays.keys()),
           "grasp_policy": ep.meta.get("grasp_policy_by_instance"),
           "force_target_by_instance": ep.meta.get("force_target_by_instance"),
           "guard": ep.meta.get("execution_guard")}
    s.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_js(rec), indent=1, default=str))
    print(json.dumps({"status": rec["status"], "success": rec["official_success_final"], "clean": rec["clean_v1"],
                      "unexcused": [(r["robot_body"], r["other"], r["v1_reasons"], r["control_steps"], r["stage_first"],
                                     r["stage_last"]) for r in rows if not r["v1_excused"]],
                      "windows": wins}, default=str))
    return 0


def S26_steps(events, key):
    from synthesis_v2.contact_policy_v1 import _steps_of
    return _steps_of(events, key)


if __name__ == "__main__":
    raise SystemExit(main())
