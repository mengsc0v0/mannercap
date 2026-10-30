"""Step 14: T5 ordinary operation (no manner request), same plan as T4 ordinary (GripPlan(1,100,settle_after_open=1),
approach check on, contact audit on).  Records where the book ends up relative to the BACK compartment, in the caddy frame.
usage: s14_t5_ordinary.py TAG INIT...  -> reports/step14_t5_precheck/<TAG>/init{N}.json"""
import json, sys, time, math
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_plan, make_session
from synthesis_v2.contact_audit import ContactRecorder
from synthesis_v2.native_producer import GripPlan
from synthesis_v2.program_executor import produce_program_episode

TAG = sys.argv[1]; OUT = ROOT / "reports" / "step14_t5_precheck" / TAG; OUT.mkdir(parents=True, exist_ok=True)
GRIP = GripPlan(1.0, 100.0, settle_after_open=1)
for i in [int(x) for x in sys.argv[2:]]:
    prog = make_program(5); plan = make_plan(5, prog, grip=GRIP, approach_check=True); s = make_session(5, i, prog)
    box = {}; t0 = time.time()
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    a = ep.arrays; stages = [str(x) for x in a["stage"]]
    m, d = s._mj()
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "desk_caddy_1_back_contain_region")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "black_book_1_main")
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "desk_caddy_1_main")
    site_c, Rs = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    book = d.xpos[bid].copy(); w, x, y, z = d.xquat[bid]
    byaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    print("array keys:", sorted(k for k in a.keys() if "pos" in k or "stage" in k)[:20], flush=True)
    pk = lambda n: next((np.asarray(a[k]) for k in (f"pos_{n}", f"{n}_pos", f"obj_pos_{n}") if k in a), None)
    pb = pk("black_book_1"); pc = pk("desk_caddy_1")
    if pb is None:   # object not recorded per step: fall back to the episode's start/end only
        pb = np.stack([s.initial["black_book_1_pos"], book])
    rel = Rs.T @ (book - site_c)            # book centre in the back-region frame; |rel| must stay inside half_size
    rec = {"init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "success_now": s.success(),
           "book_rise_max_m": float(pb[:, 2].max() - pb[0, 2]), "book_final_pos": book.round(4).tolist(),
           "book_in_back_region_frame_m": rel.round(4).tolist(), "back_region_half_size": m.site_size[sid].round(4).tolist(),
           "book_yaw_init_final_deg": [None, round(byaw, 1)], "book_init_pos": pb[0].round(4).tolist(),
           "caddy_shift_m": None if pc is None else float(np.hypot(*(pc[-1, :2] - pc[0, :2]))),
           "grasp": {k: {"reached": v["reached_band"], "reverse_pulses": v["reverse_pulses"]} for k, v in (ep.meta.get("grasp_policy_by_instance") or {}).items()},
           "executed_candidates": ep.meta.get("approach_plan"),
           "forbidden_ge5N": {x["other"] + "/" + x["robot_body"]: x["max_n"] for x in box["r"].report(stages)["forbidden_contacts"] if x["max_n"] >= 5},
           "n_steps": int(len(a["actions"])), "wall_s": round(time.time() - t0, 1), "contact_audit": box["r"].report(stages)}
    s.close()
    (OUT / f"init{i}.json").write_text(json.dumps(rec, default=str))
    print(f"INIT {i} status={rec['status']} task_ok={rec['task_ok']} rise={rec['book_rise_max_m']:.3f} rel_in_back={rec['book_in_back_region_frame_m']} "
          f"half={rec['back_region_half_size']} caddy_shift={rec['caddy_shift_m']} forb>=5N={rec['forbidden_ge5N']} wall={rec['wall_s']}", flush=True)
