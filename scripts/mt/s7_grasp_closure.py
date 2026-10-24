"""Grasp-closure comparison (dev only).  Usage: s7_grasp_closure.py TAG TASK CONFIG INIT...
Writes reports/step7_grasp_closure/<TAG>_t<TASK>_<CONFIG>.json (one record per init)."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import GRIP_CONFIGS, SPECS, make_plan, make_program, make_session
from synthesis_v2.grasp_audit import audit_episode
from synthesis_v2.program_executor import produce_program_episode

TAG, T, CFG, INITS = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(x) for x in sys.argv[4:]]
assert all(i in set(range(0, 8)) | set(range(14, 22)) for i in INITS), "dev split only (0-7, 14-21)"
out = Path("reports/step7_grasp_closure"); out.mkdir(parents=True, exist_ok=True)
recs = []
for I in INITS:
    prog = make_program(T); s = make_session(T, I, prog); t0 = time.time()
    ep = produce_program_episode(prog, I, make_plan(T, prog, grip=GRIP_CONFIGS[CFG], approach_check=True), keep_images=False, session=s)
    audit = audit_episode(ep, prog.objects)
    pol = ep.meta["grasp_policy_by_instance"]
    rec = {"init": I, "status": ep.status, "task_ok": bool(ep.task_ok), "control_steps": int(len(ep.arrays["actions"])),
           "lookahead_steps": sum(v["lookahead_control_steps"] for v in ep.meta.get("approach_plan", {}).values()),
           "wall_s": round(time.time() - t0, 1), "grasps": audit,
           "peak_total_n": {k: max([r["total_n"] for r in v["trace"] if r["total_n"] == r["total_n"]] + [0.0]) for k, v in pol.items()},
           "trace": {k: v["trace"] for k, v in pol.items()}}
    recs.append(rec)
    print(f"R {CFG} T{T} i{I} ok={rec['task_ok']} " + " ".join(
        f"{k.split('.')[0][:5]}:end={int(v['contact_at_end'])},stable={int(v['prelift_stable'])},inlift={int(v['contact_in_lift'])},lift={int(v['lift_supported'])},rev={v['reverse_pulses']},rc={v['reclose_pulses']}"
        for k, v in audit.items()), flush=True)
(out / f"{TAG}_t{T}_{CFG}.json").write_text(json.dumps(recs, indent=1))
