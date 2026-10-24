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
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402

TAG = sys.argv[1]
OUT = ROOT / "reports" / f"step7_approach_{TAG}"; OUT.mkdir(parents=True, exist_ok=True)
GENTLE = GripPlan(1.0, 22.0)


def episode(i, check):
    prog = make_program(2)
    plan = dataclasses.replace(make_plan(prog, {r: GENTLE for r, _ in OBJECTS}), approach_check=check)
    s = make_session(i, prog)
    rec_ = {}

    def hook(ss):
        rec_["r"] = ContactRecorder(ss).install()

    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, on_session=hook)
    a = ep.arrays
    stages = [str(x) for x in a["stage"]] if "stage" in a else []
    audit = rec_["r"].report(stages)
    rec = {"init": i, "approach_check": check, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(len(a["actions"])),
           "grasp_reached": {k: v["reached_band"] for k, v in (ep.meta.get("grasp_policy_by_instance") or {}).items()},
           "contact_audit": audit, "approach_plan": ep.meta.get("approach_plan")}
    s.close()
    return rec


for i in [int(x) for x in sys.argv[2:]]:
    for check in (False, True):
        rec = episode(i, check)
        (OUT / f"init{i}_{'after' if check else 'before'}.json").write_text(json.dumps(rec))
        nt = {x["other"] + "/" + x["robot_body"]: x["max_n"] for x in rec["contact_audit"]["forbidden_contacts"]}
        print(f"init{i} {'after ' if check else 'before'} status={rec['status']} ok={rec['task_ok']} reached={list(rec['grasp_reached'].values())} forbidden={nt} clean={rec['contact_audit']['clean']} lookahead={ {k: v.get('lookahead_control_steps') for k, v in (rec['approach_plan'] or {}).items()} }", flush=True)
