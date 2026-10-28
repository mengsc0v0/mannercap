"""Step 9.1: replay official LIBERO demonstrations of one task in the matched env (same bddl, OSC_POSE, 20 Hz) and extract
object-frame grasp references.  Usage: s9_demo_replay.py TASK DEMO_FILE OUT_DIR DEMO_INDEX...
Per demo: replay success (does the dataset action sequence reproduce task success in THIS env), every gripper-close event
(step, eef pose, which object it lifts, eef pose in that object's yaw frame, finger-pad contacts on that object)."""
import json, os, sys, time
import numpy as np

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO")
sys.path.append(LIBERO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, mujoco  # noqa: E402
from libero import benchmark  # noqa: E402
from libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.utils import get_libero_path  # noqa: E402

T, FILE, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(T)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)
OBJS = ("porcelain_mug_1", "white_yellow_mug_1")


def yaw_of(q):  # xyzw
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pad_contacts(sim, obj_name):
    m, d = sim.model._model, sim.data._data
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = (m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)
        g1, g2 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}")
        if (obj_name in b1) != (obj_name in b2):
            rb = b2 if obj_name in b1 else b1
            if "gripper" in rb or "finger" in rb or "hand" in rb:
                f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6)
                out.append({"robot_body": rb, "obj_geom": g1 if obj_name in b1 else g2, "pos": c.pos.copy(), "n": float(abs(f6[0]))})
    return out


with h5py.File(FILE, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for di in [int(x) for x in sys.argv[4:]]:
        g = f["data"][keys[di]]
        acts, states = g["actions"][:], g["states"][:]
        t0 = time.time()
        env.reset(); obs = env.set_init_state(states[0])
        sim = env.env.sim
        traj = []
        for k, a in enumerate(acts):
            pre = {o: (np.asarray(obs[o + "_pos"]).copy(), np.asarray(obs[o + "_quat"]).copy()) for o in OBJS}
            eef = (np.asarray(obs["robot0_eef_pos"]).copy(), np.asarray(obs["robot0_eef_quat"]).copy())
            pc = {o: pad_contacts(sim, o) for o in OBJS} if a[-1] > 0 else {o: [] for o in OBJS}
            traj.append((float(a[-1]), pre, eef, pc))
            obs, r, done, info = env.step(a)
        ok = bool(env.env._check_success())
        # close events: gripper command goes -1 -> +1
        ev, k = [], 1
        for k in range(1, len(traj)):
            if traj[k][0] > 0 and traj[k - 1][0] <= 0:
                ev.append(k)
        events = []
        for k in ev:
            # object lifted after this close: the one whose z rises most within the next 80 steps
            kk = min(len(traj) - 1, k + 80)
            rise = {o: float(traj[kk][1][o][0][2] - traj[k][1][o][0][2]) for o in OBJS}
            who = max(rise, key=rise.get)
            kh = min(len(traj) - 1, k + 10)  # contacts a few steps into the close
            opos, oquat = traj[k][1][who]
            epos, equat = traj[k][2]
            ya = yaw_of(oquat)
            rel_world = (epos - opos)
            rel_obj = rotz(-ya) @ rel_world
            cts = traj[kh][3][who]
            events.append({"close_step": k, "object": who, "rise_80": rise[who], "obj_pos": opos.tolist(), "obj_yaw": ya,
                           "eef_pos": epos.tolist(), "eef_quat_xyzw": equat.tolist(), "eef_yaw": yaw_of(equat),
                           "eef_minus_obj_world": rel_world.tolist(), "eef_minus_obj_in_obj_yaw_frame": rel_obj.tolist(),
                           "eef_yaw_minus_obj_yaw": float((yaw_of(equat) - ya + np.pi) % (2 * np.pi) - np.pi),
                           "pad_contacts_k+10": [{"robot_body": c["robot_body"], "obj_geom": c["obj_geom"], "n": round(c["n"], 2),
                                                  "pos_in_obj_yaw_frame": (rotz(-ya) @ (c["pos"] - opos)).round(4).tolist()} for c in cts]})
        rec = {"demo": keys[di], "n_actions": int(len(acts)), "replay_success": ok, "wall_s": round(time.time() - t0, 1), "events": events,
               "init_obj": {o: {"pos": traj[0][1][o][0].tolist(), "yaw": yaw_of(traj[0][1][o][1])} for o in OBJS}}
        json.dump(rec, open(os.path.join(OUT, f"{keys[di]}.json"), "w"))
        print(keys[di], "ok", ok, "events", [(e["object"], e["close_step"]) for e in events], f"{rec['wall_s']}s", flush=True)
