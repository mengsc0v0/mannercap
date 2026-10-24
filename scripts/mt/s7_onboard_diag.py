"""Per-predicate terminal-state diagnosis of an onboarding episode (read-only; same executor, approach_check on).
Usage: s7_onboard_diag.py TASK INIT...  ->  reports/step7_onboard_diag/t<T>_init<i>.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from s7_task_config import ROOT, make_program, make_plan, make_session  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

T = int(sys.argv[1]); OUT = ROOT / "reports" / "step7_onboard_diag"; OUT.mkdir(parents=True, exist_ok=True)
for i in [int(x) for x in sys.argv[2:]]:
    prog = make_program(T); s = make_session(T, i, prog); box = {}
    ep = produce_program_episode(prog, i, make_plan(T, prog, approach_check=True), keep_images=False, session=s,
                                 on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    e = s.env; inner = getattr(e, "env", e)
    preds = []
    try:
        for st in inner.parsed_problem["goal_state"]:
            preds.append({"predicate": " ".join(map(str, st)), "true": bool(inner._eval_predicate(st))})
    except Exception as exc:
        preds.append({"error": repr(exc)[:200]})
    obs = s.obs
    pos = {k[:-4]: np.round(np.asarray(v), 3).tolist() for k, v in obs.items() if k.endswith("_pos") and not k.startswith("robot")}
    rec = {"task": T, "init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "predicates": preds, "final_pos": pos,
           "subtask_obj_pos0": {k: v.get("obj_pos0") for k, v in ep.meta["subtasks"].items()},
           "stage_steps": ep.stage_steps if hasattr(ep, "stage_steps") else None,
           "contact_audit": box["r"].report([str(x) for x in ep.arrays["stage"]])}
    (OUT / f"t{T}_init{i}.json").write_text(json.dumps(rec, default=str))
    print(f"t{T} init{i} ok={rec['task_ok']} preds={[(p.get('predicate'), p.get('true')) for p in preds]}", flush=True)
    s.close()
