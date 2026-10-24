"""Dev probe (T0): force a single grasp-height offset dz for every pick (approach_check on) and report outcome + palm contact
with the grasped object.  Usage: s7_dz_sweep.py TASK DZ_CM INIT...   (dz in cm)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import synthesis_v2.approach_planner as ap
from s7_task_config import ROOT, SPECS, make_program, make_plan, make_session  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

ASC = sys.argv[2] == "asc"      # "asc": lateral fixed at 0, dz tried in ascending order (strict-clearance rule decides)
T, DZ = int(sys.argv[1]), (0.0 if ASC else float(sys.argv[2]) / 100.0)
ap.candidate_grid = (lambda *a, **k: [(0, 0.0, z) for z in (0.0, 0.01, 0.02, 0.03, 0.04)]) if ASC else (lambda *a, **k: [(0, 0.0, DZ)])
OUT = ROOT / "reports" / "step7_dz_sweep"; OUT.mkdir(parents=True, exist_ok=True)
for i in [int(x) for x in sys.argv[3:]]:
    prog = make_program(T); s = make_session(T, i, prog); box = {}
    ep = produce_program_episode(prog, i, make_plan(T, prog, approach_check=True), keep_images=False, session=s,
                                 on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    a = ep.arrays; palm = {}
    for c in box["r"].report([str(x) for x in a["stage"]])["forbidden_contacts"]:
        if c["robot_body"] == "right_gripper": palm[c["other"][:-5]] = c["max_n"]
    asym = {}
    for r, nm in SPECS[T].objects:
        m = np.array([x.startswith(r) and x.endswith("grasp") for x in a["stage"].astype(str).tolist()]) if False else np.array([f"{r}.pick_1.grasp" == str(x) for x in a["stage"]])
        if m.any(): asym[r[:5]] = (round(float(a["cf_L_" + nm][m].max()), 1), round(float(a["cf_R_" + nm][m].max()), 1))
    rec = {"task": T, "init": i, "dz": DZ, "status": ep.status, "task_ok": bool(ep.task_ok), "palm_n": palm, "grasp_LR_n": asym}
    (OUT / f"t{T}_dz{int(round(DZ*1000)):03d}_init{i}.json").write_text(json.dumps(rec))
    print(f"t{T} dz={DZ*100:.0f}cm init{i} {ep.status.split(':')[0]} ok={ep.task_ok} palm={palm} graspLR={asym}", flush=True)
    s.close()
