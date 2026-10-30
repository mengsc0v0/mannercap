"""LIBERO (LIBERO-PRO, robosuite 1.4) backend for the shared primitives (design 8.4 step 3).

``LiberoSession`` exposes the same session API as ``NativeSession`` so ``produce_program_episode`` /
``_Runner`` / ``_Recorder`` run unchanged.  Everything LIBERO-specific lives here:
  * env construction + deterministic reset from a benchmark initial state (+ fixed settle steps)
  * contact-geom lookup and gripper-force binding for LIBERO objects
  * full snapshot/restore (mj integration state + OSC internals + gripper command + env clocks +
    observable filter state; ``set_init_state`` alone is NOT bit-identical on resume)
"""
from __future__ import annotations

import copy
import math
import os
import sys
from typing import Any, Optional

import numpy as np

from synthesis_v2.native_osc_env import NativeSession

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO")
SETTLE_STEPS = 10  # zero-motion, gripper-open steps after set_init_state (objects/controller settle)


class LiberoSession(NativeSession):
    _CT = (np.ndarray, float, int, bool, list, tuple, type(None))
    # Action contract measured on LIBERO OSC_POSE (scripts/mt: constant-action response, 30 steps, x/y/z/yaw,
    # a in 0.05..1.0): realised steady per-step motion = 0.25 * (a * 0.05 m) and 0.24 * (a * 0.5 rad) for small a.
    pos_scale = 0.05 * 0.25
    rot_scale = 0.5 * 0.24

    def __init__(self, task_id: int, init_index: int, *, suite: str = "libero_10", render: bool = True,
                 image_size: int = 128, default_target: Optional[str] = None) -> None:
        if LIBERO_ROOT not in sys.path:
            sys.path.append(LIBERO_ROOT)
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero import benchmark
        from libero.envs import OffScreenRenderEnv
        from libero.utils import get_libero_path

        self.bench = benchmark.get_benchmark_dict()[suite]()
        self.libero_task = self.bench.get_task(int(task_id))
        self.task = f"libero:{suite}:{int(task_id)}"
        self.task_name = self.libero_task.name
        self.seed = int(init_index)
        self.init_index = int(init_index)
        self.render = bool(render)
        self.default_target = default_target
        self._init_state = self.bench.get_task_init_states(int(task_id))[int(init_index)]
        bddl = os.path.join(get_libero_path("bddl_files"), self.libero_task.problem_folder, self.libero_task.bddl_file)
        self.wrapper = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=image_size, camera_widths=image_size,
                                          controller="OSC_POSE", control_freq=20)
        self.env = self.wrapper.env
        self.env.horizon, self.env.ignore_done = 10**6, True  # LIBERO default horizon (1000) would end long programs
        self._binding = None
        self.force_target_name = None
        self._acc = self._new_acc()
        self._hooked = False
        self.obs = None
        self.sim_step_index = 0
        self.control_step_index = 0
        self.initial = {}

    # --- scene ----------------------------------------------------------
    def _target_geoms(self) -> tuple[str, tuple[int, ...]]:
        if not self.default_target:
            raise ValueError("LiberoSession needs default_target (object name for the initial force binding)")
        return self.default_target, self.geom_ids_of(self.default_target)

    def geom_ids_of(self, env_name: str) -> tuple[int, ...]:
        obj = getattr(self.env, "objects_dict", {}).get(env_name)
        if obj is None:
            return ()
        m = self._mj()[0]
        import mujoco
        ids = tuple(int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in (obj.contact_geoms or ()))
        if any(i < 0 for i in ids):
            raise ValueError(f"unresolved contact geom for {env_name!r}")
        return ids

    # --- episode --------------------------------------------------------
    def reset(self) -> dict:
        self.wrapper.seed(0)
        self.wrapper.reset()
        obs = self.wrapper.set_init_state(self._init_state)
        for _ in range(SETTLE_STEPS):
            obs, *_ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
        self.obs = obs = self._with_fixtures(obs)
        self.sim_step_index = 0
        self.control_step_index = 0
        if "step" in vars(self.env.sim):  # drop a hook left by an earlier reset
            del self.env.sim.step
        self._hook()
        self._hooked = True
        self._acc = self._new_acc()
        self.initial = self.snapshot_initial()
        for nm in getattr(self.env, "objects_dict", {}):
            for suf in ("pos", "quat"):
                if f"{nm}_{suf}" in obs:
                    self.initial[f"{nm}_{suf}"] = np.asarray(obs[f"{nm}_{suf}"], dtype=np.float64).copy()
        return obs

    def _with_fixtures(self, obs: dict) -> dict:
        """env.step() publishes poses for objects only.  A task whose placement target is a BDDL *fixture* (T5: the desk
        caddy) therefore has no ``<name>_pos`` for it, and the shared executor cannot name it as a target.  Publish
        ``<name>_pos`` / ``<name>_quat`` (xyzw, as for objects) for every entry of ``env.fixtures_dict``.  Tasks whose
        BDDL lists no fixture (T0/T1/T4/T7) get an unchanged observation dict."""
        import mujoco
        m, d = self._mj()
        for nm in getattr(self.env, "fixtures_dict", {}):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm + "_main")
            if bid < 0:
                continue
            w, x, y, z = d.xquat[bid]
            obs[f"{nm}_pos"] = np.asarray(d.xpos[bid], dtype=np.float64).copy()
            obs[f"{nm}_quat"] = np.array([x, y, z, w], dtype=np.float64)
        return obs

    def step(self, action7):
        obs, tel = super().step(action7)
        self.obs = obs = self._with_fixtures(obs)
        return obs, tel

    def success(self) -> bool:
        try:
            return bool(self.env._check_success())
        except Exception:
            return False

    # --- full snapshot --------------------------------------------------
    def save_state(self) -> dict[str, Any]:
        import mujoco

        mj_m, mj_d = self._mj()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        phys = np.empty(mujoco.mj_stateSize(mj_m, spec), dtype=np.float64)
        mujoco.mj_getState(mj_m, mj_d, phys, spec)
        r = self.env.robots[0]
        ctrl = {k: copy.deepcopy(v) for k, v in vars(r.controller).items() if isinstance(v, self._CT)}
        obsv = {n: {k: copy.deepcopy(v) for k, v in vars(o).items() if k.startswith("_") and isinstance(v, self._CT)}
                for n, o in self.env._observables.items()}
        return {"phys": phys, "ctrl": ctrl, "grip": copy.deepcopy(np.asarray(r.gripper.current_action)),
                "obsv": obsv,
                "env": {"timestep": int(self.env.timestep), "cur_time": float(self.env.cur_time),
                        "done": bool(self.env.done)},
                "obs": dict(self.obs), "sim_step_index": int(self.sim_step_index),
                "control_step_index": int(self.control_step_index), "force_target": self.force_target_name}

    def restore_state(self, st: dict[str, Any]) -> None:
        import mujoco

        mj_m, mj_d = self._mj()
        mujoco.mj_setState(mj_m, mj_d, np.asarray(st["phys"], dtype=np.float64), mujoco.mjtState.mjSTATE_INTEGRATION)
        self.env.sim.forward()
        r = self.env.robots[0]
        for k, v in st["ctrl"].items():
            setattr(r.controller, k, copy.deepcopy(v))
        r.gripper.current_action = copy.deepcopy(st["grip"])
        for n, vals in st["obsv"].items():
            for k, v in vals.items():
                setattr(self.env._observables[n], k, copy.deepcopy(v))
        self.env.timestep, self.env.cur_time, self.env.done = (
            st["env"]["timestep"], st["env"]["cur_time"], st["env"]["done"])
        self.obs = dict(st["obs"])
        self.sim_step_index = st["sim_step_index"]
        self.control_step_index = st["control_step_index"]
        self._acc = self._new_acc()
        if st.get("force_target"):
            self.set_force_target(st["force_target"])
