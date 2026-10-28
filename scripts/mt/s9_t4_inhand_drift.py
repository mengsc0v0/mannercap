"""Step 9.3 diagnosis: in-hand pose drift of each mug between the end of its grasp stage and the end of its place stage.

Reports, per mug, the hand-minus-object vector (world xy) at both moments and the drift between them, plus where the mug
lands relative to its plate.  Usage: s9_t4_inhand_drift.py INIT..."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_plan, make_session
from synthesis_v2.native_producer import GripPlan
from synthesis_v2.program_executor import produce_program_episode

GRIP = GripPlan(1.0, 100.0, settle_after_open=1)
PAIRS = {"white_mug": ("porcelain_mug_1", "plate_1"), "yellow_mug": ("white_yellow_mug_1", "plate_2")}
OUT = ROOT / "reports" / "step9_t4_onboard" / "inhand_drift"; OUT.mkdir(parents=True, exist_ok=True)
for i in [int(x) for x in sys.argv[1:]]:
    prog = make_program(4); plan = make_plan(4, prog, grip=GRIP, approach_check=True); s = make_session(4, i, prog)
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s); s.close()
    a = ep.arrays; st = [str(x) for x in a["stage"]]
    rec = {"init": i, "task_ok": bool(ep.task_ok), "mugs": {}}
    for role, (m, p) in PAIRS.items():
        pm, pp, eef = np.asarray(a["pos_" + m]), np.asarray(a["pos_" + p]), np.asarray(a["state"])[:, :3]
        g = [k for k, x in enumerate(st) if x == f"{role}.pick_1.grasp"][-1]
        pl = [k for k, x in enumerate(st) if x == f"{role}.pick_1.place"][-1]
        hand_minus_obj_grasp, hand_minus_obj_place = (eef[g] - pm[g])[:2], (eef[pl] - pm[pl])[:2]
        rec["mugs"][role] = {"hand_minus_obj_at_grasp_xy": hand_minus_obj_grasp.round(4).tolist(),
                             "hand_minus_obj_at_place_xy": hand_minus_obj_place.round(4).tolist(),
                             "in_hand_drift_xy_m": float(np.hypot(*(hand_minus_obj_place - hand_minus_obj_grasp))),
                             "mug_minus_plate_xy_at_place": (pm[pl, :2] - pp[pl, :2]).round(4).tolist(),
                             "hand_minus_plate_xy_at_place": (eef[pl, :2] - pp[pl, :2]).round(4).tolist()}
    (OUT / f"init{i}.json").write_text(json.dumps(rec))
    for role, v in rec["mugs"].items():
        print(f"init{i} ok={rec['task_ok']} {role:11s} hand-obj grasp {v['hand_minus_obj_at_grasp_xy']} place {v['hand_minus_obj_at_place_xy']} "
              f"drift {v['in_hand_drift_xy_m']:.4f} mug-plate {v['mug_minus_plate_xy_at_place']} hand-plate {v['hand_minus_plate_xy_at_place']}", flush=True)
