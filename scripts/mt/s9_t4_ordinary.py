"""Step 9.3: T4 ordinary operation (no manner request) with the step-8 'normal' plan (GripPlan(1,100,settle_after_open=1)),
approach check on, grasp reference from the task adapter, contact audit on.  Usage: s9_t4_ordinary.py TAG INIT...
-> reports/step9_t4_onboard/<TAG>/init{N}.json.  Per-mug: carried = final xy within 3 cm of its own plate, max rise of the mug."""
import json, sys, time, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_plan, make_session
from synthesis_v2.contact_audit import ContactRecorder
from synthesis_v2.native_producer import GripPlan
from synthesis_v2.program_executor import produce_program_episode

TAG = sys.argv[1]; OUT = ROOT / "reports" / "step9_t4_onboard" / TAG; OUT.mkdir(parents=True, exist_ok=True)
GRIP = GripPlan(1.0, 100.0, settle_after_open=1)
PAIRS = {"white_mug": ("porcelain_mug_1", "plate_1"), "yellow_mug": ("white_yellow_mug_1", "plate_2")}
for i in [int(x) for x in sys.argv[2:]]:
    prog = make_program(4); plan = make_plan(4, prog, grip=GRIP, approach_check=True); s = make_session(4, i, prog)
    box = {}; t0 = time.time()
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    a = ep.arrays; stages = [str(x) for x in a["stage"]]
    mugs = {}
    for role, (m, p) in PAIRS.items():
        pm = np.asarray(a["pos_" + m]); pp = np.asarray(a["pos_" + p])
        mugs[role] = {"rise_max_m": float(pm[:, 2].max() - pm[0, 2]), "final_xy_to_plate_m": float(np.hypot(*(pm[-1, :2] - pp[-1, :2]))), "final_minus_plate_xy": (pm[-1, :2] - pp[-1, :2]).round(4).tolist(),
                      "init_minus_plate_xy": (pm[0, :2] - pp[0, :2]).round(4).tolist(), "final_z": float(pm[-1, 2]), "plate_final_shift_m": float(np.hypot(*(pp[-1, :2] - pp[0, :2]))),
                      "carried": bool(np.hypot(*(pm[-1, :2] - pp[-1, :2])) < 0.03 and pm[:, 2].max() - pm[0, 2] > 0.05)}
    rep = box["r"].report(stages)
    gp = ep.meta.get("grasp_policy_by_instance") or {}
    rec = {"init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "mugs": mugs, "grasp": {k: {"reached": v["reached_band"], "reverse_pulses": v["reverse_pulses"], "reclose": v["reclose_pulses"], "closing_steps": v["closing_steps"]} for k, v in gp.items()},
           "forbidden_ge5N": {x["other"] + "/" + x["robot_body"]: x["max_n"] for x in rep["forbidden_contacts"] if x["max_n"] >= 5},
           "executed_candidates": {k: {"source": v["grid_source"], "yaw_deg": v["chosen"]["yaw_deg"], "lat": v["chosen"]["lateral_m"], "lon": v["chosen"].get("longitudinal_m", 0.0), "dz": v["chosen"]["dz_m"]} for k, v in (ep.meta.get("approach_plan") or {}).items()},
           "wall_s": round(time.time() - t0, 1), "contact_audit": rep}
    s.close()
    (OUT / f"init{i}.json").write_text(json.dumps(rec, default=str))
    print(f"INIT {i} status={rec['status']} ok={rec['task_ok']} carried={[m['carried'] for m in mugs.values()]} rise={[round(m['rise_max_m'],3) for m in mugs.values()]} "
          f"to_plate={[round(m['final_xy_to_plate_m'],3) for m in mugs.values()]} rev={[g['reverse_pulses'] for g in rec['grasp'].values()]} forb>=5N={rec['forbidden_ge5N']} wall={rec['wall_s']}s", flush=True)
