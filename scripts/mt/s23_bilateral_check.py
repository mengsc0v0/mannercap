"""Step 23: does require_bilateral_grasp wrongly refuse grasps that already work?  T5 (official demo reference, frozen GUARD,
frozen GRIP) on a few dev inits, with the two new opt-in flags added to the T5 plan.  Read-only check, no tuning.
usage: s23_bilateral_check.py INIT..."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_plan, make_program, make_session  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

GUARD = {"enabled": True, "min_clearance_m": 0.0, "contact_threshold_n": 0.1, "max_retries": 3}
GRIP = GripPlan(1.0, 100.0, settle_after_open=1)
rows = []
for i in [int(x) for x in sys.argv[1:]]:
    prog = make_program(5)
    plan = make_plan(5, prog, grip=GRIP, approach_check=True, stop_on_grasp_failure=True, stop_on_place_failure=True,
                     stop_on_stage_timeout=True, require_bilateral_grasp=True)
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=make_session(5, i, prog), execution_guard=GUARD)
    g = ep.meta.get("grasp_policy_by_instance") or {}
    rows.append({"init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": len(ep.arrays["stage"]),
                 "bilateral": {k: v.get("bilateral") for k, v in g.items()}, "stage_failure": ep.meta.get("stage_failure"),
                 "grasp_failure": ep.meta.get("grasp_failure")})
    print(json.dumps(rows[-1], default=str))
p = ROOT / "reports" / "step23_shared_fixes"; p.mkdir(parents=True, exist_ok=True)
(p / "bilateral_check_T5.json").write_text(json.dumps(rows, indent=1, default=str))
