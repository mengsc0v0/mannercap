"""Step 14 (C): replay ordinary official T5 demonstrations in the matched env and extract, per demo and WITHOUT averaging,
the real 3-D poses of the successful grasp / carry / release: hand pose relative to the book at the last close, the approach
path before it, and the release pose relative to the official target site.
usage: s14_t5_demo_extract.py DEMO_FILE OUT_DIR DEMO_INDEX...   (one JSON per demo)"""
import json, math, os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO")
sys.path.append(LIBERO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
import h5py  # noqa: E402
from libero import benchmark  # noqa: E402
from libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.utils import get_libero_path  # noqa: E402

FILE, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
T, BOOK, SITE = 5, "black_book_1", "desk_caddy_1_back_contain_region"
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(T)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)


def yaw_of(q):
    return float(R.from_quat(q).as_euler("xyz")[2])


def rot_z(a):
    return R.from_euler("z", a).as_matrix()


def wrap_yaw(y):  # same 4-fold wrap as native_producer._yaw_aligned_quat
    return (y + math.pi / 4) % (math.pi / 2) - math.pi / 4


with h5py.File(FILE, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for di in [int(x) for x in sys.argv[3:]]:
        g = f["data"][keys[di]]
        acts, states = g["actions"][:], g["states"][:]
        t0 = time.time()
        env.reset(); obs = env.set_init_state(states[0])
        sim = env.env.sim
        down = np.asarray(obs["robot0_eef_quat"], float).copy()   # the executor's "down" quaternion = reset pose
        site_p = sim.data.get_site_xpos(SITE).copy(); site_m = sim.data.get_site_xmat(SITE).reshape(3, 3).copy()
        tr = []
        for a in acts:
            tr.append({"g": float(a[-1]), "ep": np.asarray(obs["robot0_eef_pos"]).copy(), "eq": np.asarray(obs["robot0_eef_quat"]).copy(),
                       "bp": np.asarray(obs[BOOK + "_pos"]).copy(), "bq": np.asarray(obs[BOOK + "_quat"]).copy()})
            obs, r, done, info = env.step(a)
        ok = bool(env.env._check_success())
        tr.append({"g": 0.0, "ep": np.asarray(obs["robot0_eef_pos"]).copy(), "eq": np.asarray(obs["robot0_eef_quat"]).copy(),
                   "bp": np.asarray(obs[BOOK + "_pos"]).copy(), "bq": np.asarray(obs[BOOK + "_quat"]).copy()})
        closes = [k for k in range(1, len(acts)) if tr[k]["g"] > 0 and tr[k - 1]["g"] <= 0]
        opens = [k for k in range(1, len(acts)) if tr[k]["g"] <= 0 and tr[k - 1]["g"] > 0]
        rec = {"demo": keys[di], "n_actions": int(len(acts)), "replay_success": ok, "wall_s": round(time.time() - t0, 1),
               "close_steps": closes, "open_steps": opens,
               "init_book": {"pos": tr[0]["bp"].tolist(), "yaw_deg": math.degrees(yaw_of(tr[0]["bq"]))},
               "site": {"pos": site_p.tolist(), "yaw_deg": math.degrees(math.atan2(site_m[1, 0], site_m[0, 0]))}}
        if closes:
            k = closes[-1]
            by = yaw_of(tr[k]["bq"])
            q0 = R.from_euler("z", wrap_yaw(by)) * R.from_quat(down)
            rel = (R.from_quat(tr[k]["eq"]) * q0.inv()).as_matrix()
            kk = min(len(tr) - 1, k + 80)
            rec["grasp"] = {
                "close_step": k, "book_pos": tr[k]["bp"].tolist(), "book_yaw_deg": math.degrees(by),
                "eef_pos": tr[k]["ep"].tolist(), "eef_quat_xyzw": tr[k]["eq"].tolist(),
                "eef_minus_book_world": (tr[k]["ep"] - tr[k]["bp"]).tolist(),
                "rel_obj": (rot_z(-by) @ (tr[k]["ep"] - tr[k]["bp"])).tolist(),        # hand minus book in the book yaw frame
                "hand_yaw_rel_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),     # vs the executor's yaw-aligned quaternion
                "hand_yaw_deg": math.degrees(yaw_of(tr[k]["eq"])),
                "book_rise_80": float(tr[kk]["bp"][2] - tr[k]["bp"][2]),
                "approach_path_in_book_frame": {str(k - d): (rot_z(-by) @ (tr[k - d]["ep"] - tr[k]["bp"])).round(4).tolist()
                                                for d in (60, 45, 30, 20, 10, 5, 0) if k - d >= 0}}
        if opens:
            k = opens[-1]
            by = yaw_of(tr[k]["bq"])
            seg = tr[closes[-1]:k + 1] if closes else tr[:k + 1]
            rec["release"] = {
                "open_step": k, "book_pos": tr[k]["bp"].tolist(), "book_pos_in_site_frame": (site_m.T @ (tr[k]["bp"] - site_p)).round(4).tolist(),
                "book_yaw_rel_site_deg": math.degrees((by - math.atan2(site_m[1, 0], site_m[0, 0]) + math.pi / 2) % math.pi - math.pi / 2),
                "eef_pos": tr[k]["ep"].tolist(), "eef_pos_in_site_frame": (site_m.T @ (tr[k]["ep"] - site_p)).round(4).tolist(),
                "eef_minus_book_in_book_yaw_frame": (rot_z(-by) @ (tr[k]["ep"] - tr[k]["bp"])).round(4).tolist(),
                "carry_peak_eef_z_above_site": float(max(x["ep"][2] for x in seg) - site_p[2]),
                "carry_peak_book_z_above_site": float(max(x["bp"][2] for x in seg) - site_p[2]),
                "book_final_in_site_frame": (site_m.T @ (tr[-1]["bp"] - site_p)).round(4).tolist()}
        k = len(tr) - 1   # the moment the dataset sequence ends (= official success for replay-successful demos)
        by = yaw_of(tr[k]["bq"])
        rec["demo_start_down_quat_xyzw"] = down.tolist()
        rec["success_moment"] = {
            "step": k, "gripper_cmd": tr[k - 1]["g"], "book_pos": tr[k]["bp"].tolist(),
            "book_pos_in_site_frame": (site_m.T @ (tr[k]["bp"] - site_p)).round(4).tolist(),
            "book_yaw_rel_site_deg": math.degrees((by - math.atan2(site_m[1, 0], site_m[0, 0]) + math.pi / 2) % math.pi - math.pi / 2),
            "book_rp_deg": R.from_quat(tr[k]["bq"]).as_euler("xyz", degrees=True)[:2].round(1).tolist(),
            "eef_pos_in_site_frame": (site_m.T @ (tr[k]["ep"] - site_p)).round(4).tolist(),
            "eef_minus_book_in_book_yaw_frame": (rot_z(-by) @ (tr[k]["ep"] - tr[k]["bp"])).round(4).tolist(),
            "peak_book_z_above_site_after_grasp": float(max(x["bp"][2] for x in tr[closes[-1]:]) - site_p[2]) if closes else None}
        json.dump(rec, open(os.path.join(OUT, f"{keys[di]}.json"), "w"), indent=1)
        print(keys[di], "ok", ok, "closes", closes, "opens", opens, f"{rec['wall_s']}s", flush=True)
