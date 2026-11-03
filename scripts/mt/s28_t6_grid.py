"""Step 28: force x carry-speed response grid on the newly onboarded T6 (frozen step-27 program).

Same contract as s25_grid.py (one deterministic execution per (init, force candidate, carry speed), guard on, contact
policy v1.1 judged afterwards, nothing selected or tuned), but the program is the frozen T6 proposal and the requested
object is porcelain_mug_1 (role obj1).  Output schema matches s25 so s25_policies can replay it.
Usage: s28_t6_grid.py INIT [INIT ...]  -> reports/step28_t6_manner/grid/t6_init{I}.json
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import s8_task_force as S8  # noqa: E402
import s26_onboard as S26  # noqa: E402
from s25_grid import FORCES, SPEEDS, BASE_V, carry_speed  # noqa: E402
from s7_task_config import ROOT  # noqa: E402

T = 6
PROPOSAL = ROOT / "reports" / "step27_t6_repair" / "frozen_proposal.json"
OUT = ROOT / "reports" / "step28_t6_manner" / "grid"
REQ = "obj1"
OTHER = (1.0, 100.0)
VERSION = "s28_t6_grid.v1"


def run_one(init, force, speed):
    from s11_guarded_case import GUARD, windows_from
    from synthesis_v2.contact_audit import ContactRecorder
    from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1
    from synthesis_v2.libero_backend import LiberoSession
    from synthesis_v2.native_producer import StagePlan
    from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
    proposal = json.loads(PROPOSAL.read_text())["proposal"]
    prog, refused = S26.compile_program(T, proposal)
    assert not refused, refused
    names = [o["object"] for o in proposal["objects"]]
    roles = [f"obj{k}" for k in range(len(names))]
    S8.SPECS = {**S8.SPECS, T: SimpleNamespace(objects=tuple(zip(roles, names)))}   # per_object() reads S8.SPECS[t]
    s = LiberoSession(T, init, render=False, default_target=names[0])
    sub = S8.install_recorder(s, names)
    box = {}
    t0 = time.time()

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))
    settle = S8.EXECUTOR["settle_after_open"]
    grips = {f"{r}.pick_1.grasp": S8.grip_for(force if r == REQ else OTHER, settle) for r in roles}
    stages = tuple(StagePlan(it.label, speed if (it.label.startswith(REQ + ".") and it.primitive in ("lift", "transport"))
                             else BASE_V) for it in prog.instances)
    plan = ProgramPlan(prog, stages, grips, approach_check=True)
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s, on_session=on_session,
                                 execution_guard={**GUARD, "on_refusal": "restore", "pre_rejected": {}})
    crec, det = box["c"], box["det"]
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 2 * len(names))
    st = [str(x) for x in ep.arrays["stage"]]
    rec = {"ctrl": ctrl, "L": {n: arr[:, 2 * i] for i, n in enumerate(names)},
           "R": {n: arr[:, 2 * i + 1] for i, n in enumerate(names)},
           "substep_cost": sub.cost(), "contact_audit": crec.report(st)}
    po = S8.per_object(ep, prog, T, rec)
    v1 = eval_v1(rec["contact_audit"], crec.events, det.detail, windows_from(ep, T), st)
    vmed, vmax = carry_speed(ep, REQ)
    g = ep.meta.get("execution_guard", {})
    objs = {role: {k: o.get(k) for k in ("target", "status", "measurement_valid", "grasp_stable", "hold_n", "hold_min",
                                          "hold_mean", "hold_max", "lr_imbalance", "rel_z_spread_m", "peak_simultaneous_n",
                                          "peak_upper_bound_n")} for role, o in po.items()}
    return {"version": VERSION, "task": T, "init": init, "force": list(force), "speed": speed,
            "requested_role": REQ, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps),
            "wall_s": round(time.time() - t0, 2), "objects": objs,
            "contact_v1_clean": bool(v1["clean_v1"]),
            "v1_unexcused": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"],
                              "stages": c.get("stages")} for c in v1["unexcused_contacts"]],
            "support_unexplained": len(v1["support_contacts_unexplained"]),
            "carry_speed_med_mps": vmed, "carry_speed_max_mps": vmax,
            "guard_retries": len(g.get("retries", [])),
            "substeps_executed": int(rec["substep_cost"]["executed_substeps"]),
            "stage_failure": ep.meta.get("stage_failure"), "grasp_failure": ep.meta.get("grasp_failure")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in [int(x) for x in sys.argv[1:]]:
        f = OUT / f"t{T}_init{i}.json"
        have = json.loads(f.read_text())["episodes"] if f.exists() else {}
        for force in FORCES:
            for speed in SPEEDS:
                k = f"{force[0]:g}_{force[1]:g}@{speed:g}"
                if k in have and "error" not in have[k]:
                    continue
                try:
                    have[k] = run_one(i, force, speed)
                except Exception as e:
                    have[k] = {"version": VERSION, "task": T, "init": i, "force": list(force), "speed": speed,
                               "error": repr(e)[:400]}
                r = have[k]
                o = r.get("objects", {}).get(REQ, {})
                print(f"T6 i{i} {k} ok={r.get('task_ok')} hold={o.get('hold_mean')} [{o.get('hold_min')},{o.get('hold_max')}] "
                      f"stable={o.get('grasp_stable')} v={r.get('carry_speed_med_mps')} c1={r.get('contact_v1_clean')} "
                      f"err={r.get('error')}", flush=True)
                f.write_text(json.dumps({"task": T, "init": i, "version": VERSION, "episodes": have}, indent=1,
                                        default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
