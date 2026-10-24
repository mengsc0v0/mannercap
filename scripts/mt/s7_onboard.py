"""Step 7 onboarding run: a task from s7_task_config through the shared executor, default grip plan, approach_check off/on,
contact audit on every episode.  Usage: s7_onboard.py TAG TASK INIT...   -> reports/step7_onboard_<TAG>/t<TASK>_init<i>_{before,after}.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program, make_plan, make_session  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

TAG, T = sys.argv[1], int(sys.argv[2])
OUT = ROOT / "reports" / f"step7_onboard_{TAG}"; OUT.mkdir(parents=True, exist_ok=True)


def episode(i, check):
    prog = make_program(T); plan = make_plan(T, prog, approach_check=check); s = make_session(T, i, prog)
    box = {}
    t0 = time.time()
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    a = ep.arrays; stages = [str(x) for x in a["stage"]] if "stage" in a else []
    rec = {"task": T, "init": i, "approach_check": check, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(len(a["actions"])),
           "wall_s": round(time.time() - t0, 1),
           "grasp_reached": {k: v["reached_band"] for k, v in (ep.meta.get("grasp_policy_by_instance") or {}).items()},
           "contact_audit": box["r"].report(stages), "approach_plan": ep.meta.get("approach_plan")}
    s.close(); return rec


for i in [int(x) for x in sys.argv[3:]]:
    for check in (False, True):
        r = episode(i, check)
        (OUT / f"t{T}_init{i}_{'after' if check else 'before'}.json").write_text(json.dumps(r))
        fb = {x["other"] + "/" + x["robot_body"]: x["max_n"] for x in r["contact_audit"]["forbidden_contacts"] if x["max_n"] >= 5}
        print(f"t{T} init{i} {'after ' if check else 'before'} status={r['status']} ok={r['task_ok']} reached={list(r['grasp_reached'].values())} "
              f"forbidden>=5N={fb} wall={r['wall_s']}s", flush=True)
