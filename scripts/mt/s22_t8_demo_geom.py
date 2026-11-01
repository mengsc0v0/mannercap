"""Step 22 diagnostic (offline, read-only): (a) hand tilt at the official demos' grasps for T4/T5/T8 (angle between hand z-axis and
world down), (b) which pot geoms the fingers touch at the T8 demo_0 closes."""
import json, os, sys
import numpy as np, mujoco, h5py
sys.path.append("/workdata/cap-x/capx/third_party/LIBERO-PRO"); os.environ.setdefault("MUJOCO_GL", "egl")
from scipy.spatial.transform import Rotation as R
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path
tilt = lambda q: float(np.degrees(np.arccos(np.clip(-R.from_quat(q).apply([0, 0, 1])[2], -1, 1))))  # hand z vs world down
base = "reports/step"
for name, p in (("T5 demo_0", "reports/step14_t5_onboard/demo_replay/demo_0.json"), ("T8 demo_0", "reports/step22_t8_onboard/demo_replay_A/demo_0.json")):
    try:
        r = json.load(open(p)); ev = r.get("close_events") or r.get("grasps") or []
        print("A", name, [round(tilt(e["eef_quat_xyzw"]), 1) for e in ev if "eef_quat_xyzw" in e] or list(r.keys())[:12])
    except Exception as ex:
        print("A", name, "n/a", repr(ex)[:80])
for f in sorted(__import__("glob").glob("reports/step9_t4_onboard/*.json"))[:6]:
    try:
        r = json.load(open(f)); print("A T4", f.split("/")[-1], type(r).__name__, list(r)[:8] if isinstance(r, dict) else len(r))
    except Exception:
        pass
T = 8
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(T)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file), camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)
D = "/workdata/cap-x/capx/third_party/LIBERO-PRO/libero/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
with h5py.File(D, "r") as f:
    g = f["data"]["demo_0"]; st = g["states"][:]
    for k, who in ((111, "moka_pot_2"), (349, "moka_pot_1")):
        env.reset(); env.set_init_state(st[0])
        for a in g["actions"][:k + 8]:
            env.step(a)
        m, d = env.env.sim.model._model, env.env.sim.data._data
        rows = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            n1, n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)
            if who + "_main" in (n1, n2):
                gg = c.geom1 if n1 == who + "_main" else c.geom2; og = c.geom2 if gg == c.geom1 else c.geom1
                rows[(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, og), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gg))] = rows.get((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, og), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gg)), 0) + 1
        o = env.env._get_observations() if hasattr(env.env, "_get_observations") else {}
        print("B", who, "step", k + 8, "hand tilt", round(tilt(o["robot0_eef_quat"]), 1) if "robot0_eef_quat" in o else None, "contacts (other_geom, pot_geom):count", {f"{a}|{b}": n for (a, b), n in rows.items()})
