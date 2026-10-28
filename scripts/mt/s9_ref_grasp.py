"""Step 9.1/9.2: turn replayed official demonstrations into a reference grasp in the executor's candidate frame.

usage: s9_ref_grasp.py DEMO_DIR OUT_JSON [INIT ...]
 - per mug: last close event of each *replay-successful* demo (the grasp that carried the mug), eef pose in the object yaw
   frame + hand yaw relative to the executor's yaw-aligned grasp quaternion;
 - median / spread over demos;
 - per given init: the reference expressed as one executor candidate (yaw_deg, lateral_m, longitudinal_m, dz_m) and the
   planner's clearance for that single candidate (no tuning: the median is used as is).
Needs no force labels."""
import sys, json, glob, math
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "scripts/mt"); sys.path.insert(0, ".")

demo_dir, out = sys.argv[1], sys.argv[2]
inits = [int(x) for x in sys.argv[3:]]
MUGS = {"porcelain_mug_1": "white_mug", "white_yellow_mug_1": "yellow_mug"}

def wrap_yaw(obj_yaw):  # same 4-fold wrap as native_producer._yaw_aligned_quat
    return (obj_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4

def hand_yaw_rel(q_eef, down, obj_yaw):
    """Hand yaw (about world z) of the demo grasp relative to the executor's yaw-aligned quat for this object yaw."""
    q0 = R.from_euler("z", wrap_yaw(obj_yaw)) * R.from_quat(down)
    rel = (R.from_quat(q_eef) * q0.inv()).as_matrix()
    return math.degrees(math.atan2(rel[1, 0], rel[0, 0]))

def main():
    from s7_task_config import make_program, make_session
    prog = make_program(4)
    s = make_session(4, 0, prog); o = s.reset()
    down = np.asarray(o["robot0_eef_quat"], float)
    s.close()
    demos = [json.load(open(p)) for p in sorted(glob.glob(f"{demo_dir}/demo_*.json"))]
    n_total, n_ok = len(demos), sum(bool(d["replay_success"]) for d in demos)
    rows = {m: [] for m in MUGS}
    for d in demos:
        if not d["replay_success"]:
            continue
        for m in MUGS:
            ev = [e for e in d["events"] if e["object"] == m]
            if not ev:
                continue
            e = ev[-1]
            rows[m].append({"demo": d["demo"], "close_step": e["close_step"], "rel_obj": e["eef_minus_obj_in_obj_yaw_frame"],
                            "obj_yaw_deg": math.degrees(e["obj_yaw"]),
                            "hand_yaw_rel_deg": hand_yaw_rel(np.asarray(e["eef_quat_xyzw"]), down, e["obj_yaw"]),
                            "n_close_events_for_object": len(ev)})
    res = {"replay": {"n_demos": n_total, "n_replay_success": n_ok}, "per_mug": {}, "candidates": {}}
    for m, r in rows.items():
        A = np.array([x["rel_obj"] for x in r]); Y = np.array([x["hand_yaw_rel_deg"] for x in r])
        res["per_mug"][m] = {"n": len(r), "median_rel_obj": np.median(A, 0).round(4).tolist(), "std_rel_obj": A.std(0).round(4).tolist(),
                             "median_hand_yaw_rel_deg": float(np.median(Y)), "std_hand_yaw_rel_deg": float(Y.std()),
                             "rows": r}
    if inits:
        from synthesis_v2.program_executor import _Geometry, _subtask_roles, APPROACH_DZ
        from synthesis_v2.approach_planner import plan_approach, scene_from_libero, yaw_quat, quat_to_mat
        for init in inits:
            s = make_session(4, init, prog); obs = s.reset()
            meta = {"down_quat_xyzw": np.asarray(obs["robot0_eef_quat"], float).tolist(), "subtasks": {}}
            geo = _Geometry(s, prog, True, meta)
            for sub, role in (("white_mug.pick_1", "white_mug"), ("yellow_mug.pick_1", "yellow_mug")):
                bk = prog.objects[role]
                rec = geo.capture(sub, *_subtask_roles(prog, sub))
                st = res["per_mug"][bk]
                oyaw = R.from_quat(obs[f"{bk}_quat"]).as_euler("xyz")[2]
                off_w = R.from_euler("z", oyaw).apply(np.asarray(st["median_rel_obj"]))  # object-frame offset -> world
                m_, d_, site, env, ob, tg = scene_from_libero(s, bk, settled=True)
                q0 = np.asarray(rec["grasp_quat_xyzw"], float)
                yaw = st["median_hand_yaw_rel_deg"]
                Rq = quat_to_mat(yaw_quat(q0, np.deg2rad(yaw))) @ env.obs_to_site
                lat, lon = float(off_w @ Rq[:, 1]), float(off_w @ Rq[:, 0])
                cand = (yaw, lat, off_w[2], lon)
                resid = float(np.linalg.norm(Rq[:, 1] * lat + Rq[:, 0] * lon + np.array([0, 0, off_w[2]]) - off_w))
                pl = plan_approach(m_, d_, site, env, ob, start_pos=np.asarray(s.obs["robot0_eef_pos"], float),
                                   start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"], float), grasp_pos=np.asarray(rec["obj_pos0"], float),
                                   grasp_quat_xyzw=q0, hover_dz=APPROACH_DZ, target=tg, grid=[cand], evaluate_all=True)
                c = pl.candidates[0]
                res["candidates"][f"{init}:{role}"] = {"obj_yaw_deg": math.degrees(oyaw), "yaw_deg": yaw, "lateral_m": lat, "longitudinal_m": lon,
                                                      "dz_m": float(off_w[2]), "decomposition_residual_m": resid,
                                                      "feasible": c.feasible, "min_clearance_m": c.min_clearance_m,
                                                      "at_grasp_clearance_m": c.at_grasp_clearance_m}
                print(init, role, "yaw %.1f lat %.4f lon %.4f dz %.4f resid %.4f" % (yaw, lat, lon, off_w[2], resid),
                      "feasible", c.feasible, "min", round(c.min_clearance_m, 4), "at_grasp", round(c.at_grasp_clearance_m, 4), flush=True)
            s.close()
    json.dump(res, open(out, "w"), indent=1)
    for m in MUGS:
        p = res["per_mug"][m]
        print(m, "n", p["n"], "of", n_ok, "ok-replays; median rel_obj", p["median_rel_obj"], "std", p["std_rel_obj"],
              "hand yaw rel %.1f +- %.1f" % (p["median_hand_yaw_rel_deg"], p["std_hand_yaw_rel_deg"]))

main()
