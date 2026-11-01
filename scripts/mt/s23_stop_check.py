"""Step 23: targeted check of the opt-in stop flags (stop_on_stage_timeout, require_bilateral_grasp) on the SAME frozen T8 ordinary
episode that round 1 ran to the end (T8 init INIT, no guard).  Not an onboarding round: it only shows where the branch now ends.
usage: s23_stop_check.py INIT [bilateral_only]   (bilateral_only: only require_bilateral_grasp, no stage-timeout stop)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
import s8_task_force as S8  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

T, init = 8, int(sys.argv[1])
GRIP = S8.grip_for((1.0, 100.0), 1)
prog = make_program(T)
grips = {r: GRIP for r, _ in SPECS[T].objects}
base = make_plan(T, prog, grip=grips, approach_check=True)
ONLY = len(sys.argv) > 2 and sys.argv[2] == "bilateral_only"
plan = make_plan(T, prog, grip=grips, approach_check=True, stop_on_stage_timeout=not ONLY, require_bilateral_grasp=True)
ep = produce_program_episode(prog, init, plan, keep_images=False, session=make_session(T, init, prog))
st = [str(x) for x in ep.arrays["stage"]]
seq = []
for i, x in enumerate(st):
    if not seq or seq[-1][0] != x: seq.append([x, i, i])
    else: seq[-1][2] = i
out = {"init": init, "default_plan_hash_unchanged_by_new_fields": base.plan_hash() == make_plan(T, prog, grip=grips, approach_check=True).plan_hash(),
       "plan_hash_with_flags_differs": base.plan_hash() != plan.plan_hash(), "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": len(st),
       "stage_sequence": seq, "stage_failure": ep.meta.get("stage_failure"), "grasp_failure": ep.meta.get("grasp_failure"),
       "bilateral": {k: v.get("bilateral") for k, v in (ep.meta.get("grasp_policy_by_instance") or {}).items()}}
print(json.dumps(out, indent=1, default=str))
p = ROOT / "reports" / "step23_shared_fixes"; p.mkdir(parents=True, exist_ok=True)
(p / f"stop_check_T8_init{init}{'_bilateral_only' if ONLY else ''}.json").write_text(json.dumps(out, indent=1, default=str))
