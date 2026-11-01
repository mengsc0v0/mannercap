"""Step 23 (attribution, offline): at the instant demo_0 starts to close on each pot, how far is the (still open) hand from the pot?
Min geometric distance between every gripper collision geom and every pot collision geom (the planner's feasibility criterion needs
this to be > 0), plus any hand-pot contacts with their signed distance and force.  Read-only replay of the official demo."""
import json, os, sys
import numpy as np, mujoco, h5py
sys.path.append("/workdata/cap-x/capx/third_party/LIBERO-PRO"); os.environ.setdefault("MUJOCO_GL", "egl")
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path
demo = json.load(open("reports/step22_t8_onboard/demo_replay_A/demo_0.json"))
closes = {e["lifted"]: e["close_step"] for e in demo["close_events"]}
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(8)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file), camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)
D = "/workdata/cap-x/capx/third_party/LIBERO-PRO/libero/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
nm = lambda m, t, i: mujoco.mj_id2name(m, t, int(i)) or ""  # noqa: E731
out = {}
with h5py.File(D, "r") as f:
    g = f["data"]["demo_0"]; st = g["states"][:]
    for pot, k in closes.items():
        env.reset(); env.set_init_state(st[0])
        for a in g["actions"][:k]:
            env.step(a)
        m, d = env.env.sim.model._model, env.env.sim.data._data
        hand = [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and nm(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]).startswith("gripper0")]
        potg = [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and nm(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) == pot + "_main"]
        fromto = np.zeros(6); rows = []
        for hg in hand:
            for pg in potg:
                dist = mujoco.mj_geomDistance(m, d, hg, pg, 0.05, fromto)
                if dist < 0.05: rows.append((float(dist), nm(m, mujoco.mjtObj.mjOBJ_GEOM, hg), nm(m, mujoco.mjtObj.mjOBJ_GEOM, pg)))
        rows.sort()
        cons = []
        for c in d.contact[:d.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if (a in hand and b in potg) or (b in hand and a in potg):
                fb = np.zeros(6); mujoco.mj_contactForce(m, d, list(d.contact).index(c) if False else 0, fb)
                cons.append({"geoms": [nm(m, mujoco.mjtObj.mjOBJ_GEOM, a), nm(m, mujoco.mjtObj.mjOBJ_GEOM, b)], "dist_m": float(c.dist)})
        out[pot] = {"close_step": k, "gripper_ctrl_open_width_m": float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint1")]]),
                    "n_hand_pot_geom_pairs_within_5cm": len(rows), "nearest": [{"dist_m": round(r[0], 5), "hand_geom": r[1], "pot_geom": r[2]} for r in rows[:6]],
                    "contacts_at_close_start": cons}
        print(pot, json.dumps(out[pot])[:900])
json.dump(out, open("reports/step23_shared_fixes/demo_close_clearance.json", "w"), indent=1)
