"""Step 22 (T8 onboarding): replay ordinary official T8 demonstrations in the matched env and extract, per demo and WITHOUT
averaging, for each moka pot the hand pose relative to the pot at the gripper close that lifted it (same conventions as
s14_t5_demo_extract.py: rel_obj = hand minus object in the object yaw frame, hand_yaw_rel_deg vs the executor's yaw-aligned
quaternion) plus the pot pose at release relative to the official target site.
Pre-registered selection rule (written before any T8 demo was replayed): demos 0..7 are replayed TWICE (independent
processes, dirs A and B); the chosen demo is the LOWEST index that is replay-successful in BOTH replays and in which each of
the two pots has at least one close event after which it rises >= 0.05 m within 80 steps; per pot the LAST such close is used.
No averaging, no per-init choice.
usage: s22_t8_demo_extract.py DEMO_FILE OUT_DIR DEMO_INDEX..."""
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
T, POTS, SITE = 8, ("moka_pot_1", "moka_pot_2"), "flat_stove_1_cook_region"
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
        down = np.asarray(obs["robot0_eef_quat"], float).copy()
        site_p = sim.data.get_site_xpos(SITE).copy(); site_m = sim.data.get_site_xmat(SITE).reshape(3, 3).copy()

        def snap(o):
            d = {"g": 0.0, "ep": np.asarray(o["robot0_eef_pos"]).copy(), "eq": np.asarray(o["robot0_eef_quat"]).copy()}
            for p in POTS:
                d[p + "_p"] = np.asarray(o[p + "_pos"]).copy(); d[p + "_q"] = np.asarray(o[p + "_quat"]).copy()
            return d
        tr = []
        for a in acts:
            d = snap(obs); d["g"] = float(a[-1]); tr.append(d)
            obs, r, done, info = env.step(a)
        ok = bool(env.env._check_success())
        tr.append(snap(obs))
        closes = [k for k in range(1, len(acts)) if tr[k]["g"] > 0 and tr[k - 1]["g"] <= 0]
        opens = [k for k in range(1, len(acts)) if tr[k]["g"] <= 0 and tr[k - 1]["g"] > 0]
        rec = {"demo": keys[di], "n_actions": int(len(acts)), "replay_success": ok, "wall_s": round(time.time() - t0, 1),
               "close_steps": closes, "open_steps": opens,
               "init_pots": {p: {"pos": tr[0][p + "_p"].tolist(), "yaw_deg": math.degrees(yaw_of(tr[0][p + "_q"]))} for p in POTS},
               "site": {"pos": site_p.tolist(), "yaw_deg": math.degrees(math.atan2(site_m[1, 0], site_m[0, 0]))},
               "demo_start_down_quat_xyzw": down.tolist(), "close_events": []}
        for k in closes:
            kk = min(len(tr) - 1, k + 80)
            rise = {p: float(tr[kk][p + "_p"][2] - tr[k][p + "_p"][2]) for p in POTS}
            who = max(rise, key=rise.get)
            by = yaw_of(tr[k][who + "_q"])
            q0 = R.from_euler("z", wrap_yaw(by)) * R.from_quat(down)
            rel = (R.from_quat(tr[k]["eq"]) * q0.inv()).as_matrix()
            rec["close_events"].append({
                "close_step": k, "lifted": who, "rise_80": rise[who], "lifted_ge_5cm": bool(rise[who] >= 0.05),
                "pot_pos": tr[k][who + "_p"].tolist(), "pot_yaw_deg": math.degrees(by),
                "eef_pos": tr[k]["ep"].tolist(), "eef_quat_xyzw": tr[k]["eq"].tolist(),
                "rel_obj": (rot_z(-by) @ (tr[k]["ep"] - tr[k][who + "_p"])).tolist(),
                "hand_yaw_rel_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
                "hand_yaw_deg": math.degrees(yaw_of(tr[k]["eq"]))})
        rec["releases"] = []
        for k in opens:
            rec["releases"].append({"open_step": k, "pots_in_site_frame": {p: (site_m.T @ (tr[k][p + "_p"] - site_p)).round(4).tolist() for p in POTS},
                                    "eef_in_site_frame": (site_m.T @ (tr[k]["ep"] - site_p)).round(4).tolist()})
        rec["final_pots_in_site_frame"] = {p: (site_m.T @ (tr[-1][p + "_p"] - site_p)).round(4).tolist() for p in POTS}
        rec["peak_pot_z_above_site"] = {p: float(max(x[p + "_p"][2] for x in tr) - site_p[2]) for p in POTS}
        json.dump(rec, open(os.path.join(OUT, f"{keys[di]}.json"), "w"), indent=1)
        print(keys[di], "ok", ok, "closes", closes, "opens", opens, [(e["lifted"], round(e["rise_80"], 3)) for e in rec["close_events"]], f"{rec['wall_s']}s", flush=True)
