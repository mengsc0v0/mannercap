import os, sys, json, numpy as np
sys.path.append("/workdata/cap-x/capx/third_party/LIBERO-PRO")
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path
R = {}
b = benchmark.get_benchmark_dict()["libero_10"]()
TID = int(sys.argv[1]) if len(sys.argv) > 1 else 4
t = b.get_task(TID)
R["task"] = t.name; R["language"] = t.language
f = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=f, camera_heights=224, camera_widths=224, controller="OSC_POSE", control_freq=20)
env.seed(0)
init = b.get_task_init_states(TID)
e = env.env
def reset0():
    env.reset(); return env.set_init_state(init[0])
obs = reset0()
R["objects"] = list(e.objects_dict.keys())
R["fixtures"] = list(getattr(e, "fixtures_dict", {}).keys())
R["obs_keys"] = sorted(obs.keys())
R["control_freq"] = e.control_freq; R["model_timestep"] = float(e.sim.model.opt.timestep)
c = e.robots[0].controller
R["controller"] = {"type": type(c).__name__, "kp": np.asarray(c.kp).tolist(), "input_max": np.asarray(c.input_max).tolist(),
                   "output_max": np.asarray(c.output_max).tolist(), "use_delta": getattr(c, "use_delta", None)}
R["gripper"] = type(e.robots[0].gripper).__name__
# action sign test: +x, +y, +z, and gripper close/hold/open
signs = {}
for ax in range(3):
    obs = reset0()
    for _ in range(5): obs, *_ = env.step(np.zeros(7))
    p0 = obs["robot0_eef_pos"].copy()
    a = np.zeros(7); a[ax] = 1.0; a[6] = -1
    for _ in range(10): obs, *_ = env.step(a)
    d = obs["robot0_eef_pos"] - p0
    signs["xyz"[ax]] = {"delta_m": d.tolist(), "mean_speed_mps_at_cmd1": float(np.linalg.norm(d) / (10 / e.control_freq))}
R["action_sign"] = signs
obs = reset0()
g = {}
for cmd, name in ((1, "close"), (0, "hold"), (-1, "open")):
    q0 = obs["robot0_gripper_qpos"].copy()
    for _ in range(10): obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, cmd], float))
    g[name] = {"qpos_before": q0.tolist(), "qpos_after": obs["robot0_gripper_qpos"].tolist()}
R["gripper_sign"] = g
# camera orientation: does agentview need vertical flip (robosuite convention)?
img = obs["agentview_image"]; R["image_shape"] = list(img.shape)
R["image_top_vs_bottom_mean"] = [float(img[:20].mean()), float(img[-20:].mean())]
# contact geoms of fingers
m = e.sim.model
fing = [i for i in range(m.ngeom) if m.geom_bodyid[i] >= 0 and ("finger" in (m.body_id2name(m.geom_bodyid[i]) or ""))]
R["n_finger_geoms"] = len(fing)
# state get/set repeatability: from same flattened state, same actions -> same result?
obs = reset0()
for _ in range(5): obs, *_ = env.step(np.zeros(7))
st = e.sim.get_state().flatten().copy()
rng = np.random.default_rng(0); acts = rng.uniform(-0.5, 0.5, (30, 7)); acts[:, 6] = np.sign(acts[:, 6])
def roll():
    out = []
    for a in acts:
        o, *_ = env.step(a); out.append(np.concatenate([o["robot0_eef_pos"], o["robot0_gripper_qpos"]]))
    return np.array(out), e.sim.get_state().flatten().copy()
A, sA = roll()
env.set_init_state(st)  # LIBERO wrapper: set_state + forward
B, sB = roll()
R["restore_repeat"] = {"traj_max_abs_diff": float(np.abs(A - B).max()), "final_state_max_abs_diff": float(np.abs(sA - sB).max()),
                       "note": "sim state only; controller internal state / gripper / done caches not restored"}
R["state_dim"] = int(st.shape[0]); R["init_states_shape"] = list(init.shape)
print(json.dumps(R, indent=1, default=str))
env.close()
