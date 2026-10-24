"""Step 7.0 validation on dev inits: same grip plan, approach_check off (before) vs on (after).
Per episode: status, task success, grasp band reached, non-target robot contacts (body pair, max N) and the plan used.
Usage: s7_approach_validate.py TAG INIT... ; writes reports/step7_approach_<TAG>/init<i>_{before,after}.json"""
import json, sys, dataclasses
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import ROOT, OBJECTS, make_program, make_plan, make_session  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

TAG = sys.argv[1]
OUT = ROOT / "reports" / f"step7_approach_{TAG}"; OUT.mkdir(parents=True, exist_ok=True)
GENTLE = GripPlan(1.0, 22.0)


def episode(i, check):
    prog = make_program(2)
    plan = dataclasses.replace(make_plan(prog, {r: GENTLE for r, _ in OBJECTS}), approach_check=check)
    s = make_session(i, prog)
    ev, cur = {}, {"k": -1}

    def hook(ss):
        m, d = ss._mj(); orig = ss.env.sim.step; buf = np.zeros(6)
        nb = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
        isr = lambda b: b.startswith(("robot0", "gripper0"))  # noqa: E731

        def st(*a, **kw):
            o = orig(*a, **kw)
            for ci in range(int(d.ncon)):
                c = d.contact[ci]; b1, b2 = nb(int(c.geom1)), nb(int(c.geom2))
                if isr(b1) == isr(b2):
                    continue
                rb, ob = (b1, b2) if isr(b1) else (b2, b1)
                if "finger_joint" in rb and "tip" in rb and ob == f"{ss.force_target_name}_main".replace("_1_1", "_1"):
                    continue
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if f > 0:
                    v = ev.setdefault((rb.replace("gripper0_", ""), ob), [0.0, set()]); v[0] = max(v[0], f); v[1].add(cur["k"])
            return o
        ss.env.sim.step = st
        orig_step = ss.step

        def step(*a, **kw):
            cur["k"] = ss.control_step_index
            return orig_step(*a, **kw)
        ss.step = step
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, on_session=hook)
    a = ep.arrays
    stages = [str(x) for x in a["stage"]] if "stage" in a else []
    by = {}
    for (rb, ob), (f, ks) in ev.items():
        if ob.startswith(("living_room_table", "floor")):  # support surfaces are reported separately below
            by.setdefault("support:" + ob, []).append({"robot_body": rb, "max_n": round(f, 1), "steps": len(ks)})
            continue
        by.setdefault(ob, []).append({"robot_body": rb, "max_n": round(f, 1), "steps": len(ks),
                                      "stages": sorted({stages[k].split(".")[-1] + "@" + stages[k].split(".")[0] for k in ks if k < len(stages)})})
    rec = {"init": i, "approach_check": check, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(len(a["actions"])),
           "grasp_reached": {k: v["reached_band"] for k, v in (ep.meta.get("grasp_policy_by_instance") or {}).items()},
           "nontarget_contacts": by, "approach_plan": ep.meta.get("approach_plan")}
    s.close()
    return rec


for i in [int(x) for x in sys.argv[2:]]:
    for check in (False, True):
        rec = episode(i, check)
        (OUT / f"init{i}_{'after' if check else 'before'}.json").write_text(json.dumps(rec))
        nt = {k: max(x["max_n"] for x in v) for k, v in rec["nontarget_contacts"].items() if not k.startswith("support")}
        print(f"init{i} {'after ' if check else 'before'} status={rec['status']} ok={rec['task_ok']} reached={list(rec['grasp_reached'].values())} nontarget(maxN)={nt}", flush=True)
