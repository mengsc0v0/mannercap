"""Step 0 read-only probe: can a LIBERO (robosuite 1.4) rollout be restored bit-identically?

Compares (a) LIBERO's own set_init_state (flattened qpos/qvel only) against
(b) a full snapshot: mj integration state + OSC controller internals + gripper
command memory + env clocks.  Same action suffix is replayed after restore.
"""
import copy, json, os, sys
import numpy as np
sys.path.append("/workdata/cap-x/capx/third_party/LIBERO-PRO")
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path

TID = int(sys.argv[1]) if len(sys.argv) > 1 else 4
USE_OBSV = True
b = benchmark.get_benchmark_dict()["libero_10"]()
t = b.get_task(TID)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                         camera_heights=128, camera_widths=128, controller="OSC_POSE", control_freq=20)
env.seed(0)
env.reset(); env.set_init_state(b.get_task_init_states(TID)[0])
e = env.env
CT = (np.ndarray, float, int, bool, list, tuple, type(None))

def mj():
    return e.sim.model._model, e.sim.data._data

def save():
    m, d = mj(); spec = mujoco.mjtState.mjSTATE_INTEGRATION
    phys = np.empty(mujoco.mj_stateSize(m, spec)); mujoco.mj_getState(m, d, phys, spec)
    r = e.robots[0]
    return {"phys": phys, "ctrl": {k: copy.deepcopy(v) for k, v in vars(r.controller).items() if isinstance(v, CT)},
            "grip": copy.deepcopy(np.asarray(r.gripper.current_action)) if hasattr(r.gripper, "current_action") else None,
            "env": (e.timestep, e.cur_time, e.done),
            "obsv": {n: {k: copy.deepcopy(v) for k, v in vars(o).items() if k.startswith("_") and isinstance(v, CT)}
                     for n, o in e._observables.items()}}

def restore(st):
    m, d = mj(); mujoco.mj_setState(m, d, st["phys"], mujoco.mjtState.mjSTATE_INTEGRATION); e.sim.forward()
    r = e.robots[0]
    for k, v in st["ctrl"].items(): setattr(r.controller, k, copy.deepcopy(v))
    if st["grip"] is not None: r.gripper.current_action = copy.deepcopy(st["grip"])
    e.timestep, e.cur_time, e.done = st["env"]
    if USE_OBSV:
        for n, vals in st["obsv"].items():
            for k, v in vals.items(): setattr(e._observables[n], k, copy.deepcopy(v))

rng = np.random.default_rng(0)
pre = rng.uniform(-0.5, 0.5, (20, 7)); pre[:, 6] = np.sign(pre[:, 6])
suf = rng.uniform(-0.5, 0.5, (40, 7)); suf[:, 6] = np.sign(suf[:, 6])
for a in pre: env.step(a)
flat = e.sim.get_state().flatten().copy(); full = save()

def roll():
    out = []
    for a in suf:
        o, *_ = env.step(a)
        out.append(np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]]))
    return np.array(out), e.sim.get_state().flatten().copy()

A, sA = roll()
env.set_init_state(flat); B, sB = roll()
restore(full); C, sC = roll()
restore(full); D, sD = roll()
USE_OBSV = False
restore(full); E_, sE = roll()
R = {"task_id": TID, "gripper_has_current_action": full["grip"] is not None,
     "ctrl_fields": sorted(full["ctrl"].keys()),
     "libero_set_init_state": {"obs_max_abs": float(np.abs(A - B).max()), "state_max_abs": float(np.abs(sA - sB).max())},
     "full_snapshot": {"obs_max_abs": float(np.abs(A - C).max()), "state_max_abs": float(np.abs(sA - sC).max()),
                       "bit_identical": bool(np.array_equal(A, C) and np.array_equal(sA, sC))},
     "snapshot_without_observables": {"obs_max_abs": float(np.abs(A - E_).max()), "bit_identical": bool(np.array_equal(A, E_))},
     "full_snapshot_repeat": {"bit_identical": bool(np.array_equal(C, D) and np.array_equal(sC, sD))},
     "full_vs_continuous_per_step_max_abs": [float(x) for x in np.abs(A - C).max(axis=1)[:6]],
     "full_vs_continuous_per_dim_max_abs": [float(x) for x in np.abs(A - C).max(axis=0)]}
print("RESULT " + json.dumps(R))
env.close()
