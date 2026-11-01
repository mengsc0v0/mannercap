"""Step 23: full-pose grasp reference for T8 (offline; no episode is run).
(1) derive hand_quat_obj_xyzw (hand orientation in the object's yaw frame) from the SELECTED demo's close events (demo_0, replay A;
    rule unchanged), check rel_obj / hand_yaw_rel_deg against the frozen TaskSpec 8 values;
(2) round trip: reference -> planner candidate -> hand pose in the demo's own pot frame must reproduce the demo's hand pose;
(3) the four dev inits: the live-scene candidate keeps the demo tilt, the hand POSITION equals the old yaw-only one.
usage: s23_full_pose_ref.py [derive|check]"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from s7_task_config import ROOT, SPECS, make_program, make_session  # noqa: E402
from synthesis_v2.approach_planner import quat_to_mat, scene_from_libero  # noqa: E402
from synthesis_v2.program_executor import reference_to_candidate  # noqa: E402

T = 8
OUT = ROOT / "reports" / "step23_shared_fixes"; OUT.mkdir(parents=True, exist_ok=True)
demo = json.load(open(ROOT / "reports/step22_t8_onboard/demo_replay_A/demo_0.json"))
ang = lambda a, b: float(np.degrees((R.from_quat(a) * R.from_quat(b).inv()).magnitude()))  # noqa: E731
tilt = lambda q: float(np.degrees(np.arccos(np.clip(-R.from_quat(q).apply([0, 0, 1])[2], -1, 1))))  # noqa: E731
wrap = lambda y: (y + math.pi / 4) % (math.pi / 2) - math.pi / 4  # noqa: E731
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
ev = {e["lifted"]: e for e in demo["close_events"]}
down = np.asarray(demo["demo_start_down_quat_xyzw"], float)
refs = {}
for k, (role, name) in enumerate(SPECS[T].objects):
    e = ev[name]; by = math.radians(e["pot_yaw_deg"])
    hq = (R.from_euler("z", by).inv() * R.from_quat(e["eef_quat_xyzw"])).as_quat()
    if hq[3] < 0: hq = -hq
    old = SPECS[T].grasp_reference[k]
    refs[role] = {"object": name, "close_step": e["close_step"], "hand_quat_obj_xyzw": [round(float(x), 6) for x in hq],
                  "tilt_deg": tilt(e["eef_quat_xyzw"]),
                  "frozen_rel_obj": old["rel_obj"], "demo_rel_obj": [round(x, 4) for x in e["rel_obj"]],
                  "frozen_hand_yaw_rel_deg": old["hand_yaw_rel_deg"], "demo_hand_yaw_rel_deg": round(e["hand_yaw_rel_deg"], 1)}
if mode == "derive":
    print(json.dumps(refs, indent=1)); sys.exit()
res = {"refs": refs, "roundtrip": {}, "dev_inits": {}}
for k, (role, name) in enumerate(SPECS[T].objects):
    e = ev[name]; old = SPECS[T].grasp_reference[k]
    ref = {**old}                                   # SPECS must already carry hand_quat_obj_xyzw after the freeze edit
    assert "hand_quat_obj_xyzw" in ref, "SPECS[8] has no full-pose key yet"
    s = make_session(T, 0, make_program(T)); s.reset() if hasattr(s, "reset") else None
    env = scene_from_libero(s, name)[3]; s.close()
    by = math.radians(e["pot_yaw_deg"]); pq = R.from_euler("z", by).as_quat()
    q0 = (R.from_euler("z", wrap(by)) * R.from_quat(down)).as_quat()
    cand = reference_to_candidate(ref, pq, q0, env.obs_to_site)
    Rq = quat_to_mat(np.asarray(cand[4])) @ env.obs_to_site
    off = Rq[:, 1] * cand[1] + Rq[:, 0] * cand[3] + np.array([0, 0, cand[2]])
    hand = np.asarray(e["pot_pos"]) + off
    res["roundtrip"][role] = {"hand_pos_err_m": float(np.linalg.norm(hand - np.asarray(e["eef_pos"]))),
                              "hand_rot_err_deg": ang(cand[4], e["eef_quat_xyzw"]), "tilt_deg": tilt(cand[4])}
for i in range(4):
    s = make_session(T, i, make_program(T)); s.reset(); row = {}
    for k, (role, name) in enumerate(SPECS[T].objects):
        old = SPECS[T].grasp_reference[k]
        s2 = scene_from_libero(s, name, settled=False)[3]
        pq = np.asarray(s.obs[f"{name}_quat"], float); pp = np.asarray(s.obs[f"{name}_pos"], float)
        oyaw = R.from_quat(pq).as_euler("xyz")[2]
        down_i = np.asarray(s.obs["robot0_eef_quat"], float)
        q0 = (R.from_euler("z", wrap(oyaw)) * R.from_quat(down_i)).as_quat()
        c_new = reference_to_candidate(old, pq, q0, s2.obs_to_site)
        c_old = reference_to_candidate({k_: v for k_, v in old.items() if k_ != "hand_quat_obj_xyzw"}, pq, q0, s2.obs_to_site)
        pos = lambda c, qq: pp + (lambda Rq: Rq[:, 1] * c[1] + Rq[:, 0] * c[3] + np.array([0, 0, c[2]]))(quat_to_mat(qq) @ s2.obs_to_site)  # noqa: E731
        from synthesis_v2.approach_planner import yaw_quat
        q_old = yaw_quat(q0, np.deg2rad(c_old[0]))
        row[role] = {"pot_yaw_deg": math.degrees(oyaw), "tilt_deg_new": tilt(c_new[4]), "tilt_deg_old": tilt(q_old),
                     "hand_pos_diff_new_vs_old_m": float(np.linalg.norm(pos(c_new, c_new[4]) - pos(c_old, q_old))),
                     "rot_diff_new_vs_old_deg": ang(c_new[4], q_old)}
    s.close(); res["dev_inits"][i] = row
(OUT / "t8_full_pose_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
