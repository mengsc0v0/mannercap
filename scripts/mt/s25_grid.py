"""Step 25: force x carry-speed response grid on LIBERO-10 basket tasks (T0, T1, T7).

Every (task, init, force candidate, carry speed) is executed ONCE with the current shared executor (guard on, recover
from the approach snapshot, contact policy v1.1 judged afterwards).  The simulator is deterministic for a fixed init and
plan, so ANY selection / repair / library policy can later be replayed exactly on this grid (s25_stream.py) without new
simulation.  Nothing here selects or tunes anything.

Usage: s25_grid.py TASK INIT [INIT ...]   -> reports/step25_response_grid/t{T}_init{I}.json (one entry per config)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from s7_task_config import ROOT, SPECS, make_program, make_session  # noqa: E402
import s8_task_force as S8  # noqa: E402
from s11_guarded_case import GUARD, windows_from  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1  # noqa: E402
from synthesis_v2.native_producer import StagePlan  # noqa: E402
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode  # noqa: E402

OUT = ROOT / "reports" / "step25_response_grid"
FORCES = [(1.0, 22.0), (1.0, 100.0), (5.0, 100.0), (10.0, 22.0), (10.0, 100.0), (15.0, 100.0), (20.0, 100.0),
          (25.0, 100.0), (30.0, 100.0), (40.0, 100.0)]
SPEEDS = [0.06, 0.12, 0.24]          # commanded v_max (m/s) of the requested object's lift + transport stages
BASE_V = 0.12                        # every other stage (unchanged executor default)
OTHER = (1.0, 22.0)                  # non-requested object: fixed gentlest plan (as E1)
SETTLE = S8.EXECUTOR["settle_after_open"]
REQUESTED = {0: "tomato_sauce", 7: "cream_cheese", 1: "cream_cheese"}
VERSION = "s25_grid.v1"


def build_plan(task, prog, force, speed):
    req = REQUESTED[task]
    grips = {r: S8.grip_for(force if r == req else OTHER, SETTLE) for r, _ in SPECS[task].objects}
    stages = tuple(StagePlan(it.label, speed if (it.object_role == req and it.primitive in ("lift", "transport")) else BASE_V)
                   for it in prog.instances)
    return ProgramPlan(prog, stages, {f"{r}.pick_1.grasp": grips[r] for r, _ in SPECS[task].objects},
                       approach_check=S8.EXECUTOR["approach_check"])


def carry_speed(ep, req):
    st = [str(x) for x in ep.arrays["stage"]]
    pos = np.asarray(ep.arrays["state"])[:, :3]
    idx = [k for k, s in enumerate(st) if s.startswith(req + ".") and (s.endswith(".transport") or s.endswith(".lift"))]
    if len(idx) < 3:
        return None, None
    v = np.linalg.norm(np.diff(pos[idx], axis=0), axis=1) * 20.0
    return float(np.median(v)), float(np.max(v))


def run_one(task, init, force, speed):
    prog = make_program(task)
    s = make_session(task, init, prog)
    names = [n for _, n in SPECS[task].objects]
    sub = S8.install_recorder(s, names)
    box = {}
    t0 = time.time()

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))

    guard = {**GUARD, "on_refusal": "restore", "pre_rejected": {}}
    ep = produce_program_episode(prog, init, build_plan(task, prog, force, speed), keep_images=False, session=s,
                                 on_session=on_session, execution_guard=guard)
    crec, det = box["c"], box["det"]
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 2 * len(names))
    rec = {"ctrl": ctrl, "L": {n: arr[:, 2 * i] for i, n in enumerate(names)},
           "R": {n: arr[:, 2 * i + 1] for i, n in enumerate(names)},
           "rolled_back_cost": sub.rolled_back_cost(), "substep_cost": sub.cost(),
           "contact_audit": crec.report([str(x) for x in ep.arrays["stage"]])}
    req = REQUESTED[task]
    po = S8.per_object(ep, prog, task, rec)
    stages = [str(x) for x in ep.arrays["stage"]]
    v1 = eval_v1(rec["contact_audit"], crec.events, det.detail, windows_from(ep, task), stages)
    vmed, vmax = carry_speed(ep, req)
    g = ep.meta.get("execution_guard", {})
    objs = {}
    for role, o in po.items():
        objs[role] = {k: o.get(k) for k in ("target", "status", "measurement_valid", "grasp_stable", "hold_n", "hold_min",
                                              "hold_mean", "hold_max", "lr_imbalance", "rel_z_spread_m",
                                              "peak_simultaneous_n")}
    return {"version": VERSION, "task": task, "init": init, "force": list(force), "speed": speed,
            "requested_role": req, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps),
            "wall_s": round(time.time() - t0, 2), "objects": objs,
            "contact_v1_clean": bool(v1["clean_v1"]),
            "v1_unexcused": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"],
                              "stages": c.get("stages")} for c in v1["unexcused_contacts"]],
            "carry_speed_med_mps": vmed, "carry_speed_max_mps": vmax,
            "guard_retries": len(g.get("retries", [])),
            "substeps_executed": int(rec["substep_cost"]["executed_substeps"]),
            "stage_failure": ep.meta.get("stage_failure"), "grasp_failure": ep.meta.get("grasp_failure")}


def main() -> int:
    task = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for i in [int(x) for x in sys.argv[2:]]:
        f = OUT / f"t{task}_init{i}.json"
        have = json.loads(f.read_text())["episodes"] if f.exists() else {}
        for force in FORCES:
            for speed in SPEEDS:
                k = f"{force[0]:g}_{force[1]:g}@{speed:g}"
                if k in have and "error" not in have[k]:  # infra errors (EGL) are re-run, never kept as outcomes
                    continue
                try:
                    have[k] = run_one(task, i, force, speed)
                except Exception as e:  # keep going; the failure is recorded, never silently dropped
                    have[k] = {"version": VERSION, "task": task, "init": i, "force": list(force), "speed": speed,
                               "error": repr(e)[:400]}
                r = have[k]
                o = r.get("objects", {}).get(REQUESTED[task], {})
                print(f"T{task} i{i} {k} task_ok={r.get('task_ok')} hold={o.get('hold_mean')} "
                      f"stable={o.get('grasp_stable')} v={r.get('carry_speed_med_mps')} c1={r.get('contact_v1_clean')} "
                      f"err={r.get('error')}", flush=True)
                f.write_text(json.dumps({"task": task, "init": i, "version": VERSION, "episodes": have}, indent=1,
                                        default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
