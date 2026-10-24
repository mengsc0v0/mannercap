"""Deployment-controller RoboSuite environment for MannerCAP data production.

The environment uses the *student-visible* control interface only: fixed-gain
world-frame delta ``OSC_POSE`` (kp=150, 20 Hz) with the native Panda ``GRIP``
channel as the 7th action.  Every action that is recorded is the exact vector
passed to ``env.step``; nothing is re-derived from measured motion.

Physics-rate telemetry (per-finger target-object force and arm joint speed) is
sampled at **every MuJoCo substep** and reduced to mean/max per control step, so
a spike inside a 50 ms control interval is not lost.  Sensor failures become
NaN with ``valid=False``; they are never filled with 0.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

CONTROL_FREQ = 20
DT = 1.0 / CONTROL_FREQ
POS_SCALE = 0.05
ROT_SCALE = 0.5
IMG_SIZE = 224
CAM_MAIN = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
SUPPORTED_TASKS = ("lift", "stack", "wipe", "nut_assembly")
GRIP_CLOSE = 1.0
GRIP_HOLD = 0.0
GRIP_OPEN = -1.0

CONTROLLER_CONFIG: dict[str, Any] = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "input_type": "delta",
            "input_ref_frame": "world",
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


CONTROLLER_CONFIG_SHA256 = hashlib.sha256(_canonical(CONTROLLER_CONFIG).encode()).hexdigest()


def controller_config_path() -> str:
    path = os.path.join(tempfile.gettempdir(), f"mannercap_osc_{CONTROLLER_CONFIG_SHA256[:12]}.json")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(CONTROLLER_CONFIG, fh)
        os.replace(tmp, path)
    return path


def environment_config(task: str, render: bool = True) -> dict[str, Any]:
    """The full reproducible environment parameter record (handoff appendix A)."""
    return {
        "task": task,
        "robots": ["Panda"],
        "has_renderer": False,
        "has_offscreen_renderer": bool(render),
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": bool(render),
        "control_freq": CONTROL_FREQ,
        "camera_names": [CAM_MAIN, CAM_WRIST],
        "camera_depths": False,
        "camera_heights": IMG_SIZE,
        "camera_widths": IMG_SIZE,
        "lite_physics": False,
        "reward_shaping": False,
        "controller_configs": CONTROLLER_CONFIG,
        "controller_config_sha256": CONTROLLER_CONFIG_SHA256,
        "image_flip": "vertical flip applied to raw robosuite sim image (opengl convention)",
    }


def make_env(task: str, seed: int, *, render: bool = True) -> Any:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {task!r}; expected one of {SUPPORTED_TASKS}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )

    cc = load_composite_controller_config(controller=controller_config_path())
    common = dict(
        robots=["Panda"],
        use_camera_obs=bool(render),
        has_renderer=False,
        has_offscreen_renderer=bool(render),
        ignore_done=True,
        control_freq=CONTROL_FREQ,
        controller_configs=cc,
        horizon=100000,
        reward_shaping=False,
        lite_physics=False,
        renderer="mujoco",
    )
    if render:
        common.update(
            camera_names=[CAM_MAIN, CAM_WRIST],
            camera_depths=False,
            camera_heights=IMG_SIZE,
            camera_widths=IMG_SIZE,
        )
    if task == "lift":
        env = suite.environments.manipulation.lift.Lift(**common)
    elif task == "nut_assembly":
        from robosuite.environments.manipulation import nut_assembly as _nut_assembly

        env = _nut_assembly.NutAssemblySquare(**common)
    elif task == "wipe":
        # The wiping tool is a 0-dof gripper: the arm channel is 6D, the 7th (GRIP) action
        # is a recorded constant 0 so that the student interface stays 7D.
        from robosuite.environments.manipulation import wipe as _wipe

        env = _wipe.Wipe(**common, task_config=dict(_wipe.DEFAULT_WIPE_CONFIG, early_terminations=False))
    else:
        from robosuite.utils.placement_samplers import UniformRandomSampler

        env = suite.environments.manipulation.stack.Stack(**common)
        env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[env.cubeA, env.cubeB],
            x_range=[-0.18, 0.18],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=env.table_offset,
            z_offset=0.01,
            rng=env.rng,
        )
    return env


def rotvec_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_rotvec()


def rotvec_delta(cur_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return (R.from_quat(target_xyzw) * R.from_quat(cur_xyzw).inv()).as_rotvec()


def state8(obs: dict) -> np.ndarray:
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    if grip.size == 0:  # wiping tool has no finger joints: constant 0 padding
        grip = np.zeros(2, dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            rotvec_from_quat_xyzw(obs["robot0_eef_quat"]),
            grip,
        ]
    ).astype(np.float32)


@dataclass
class StepTelemetry:
    """Physics-rate measurements reduced over one control step."""

    n_substeps: int
    left_mean_n: float
    right_mean_n: float
    left_max_n: float
    right_max_n: float
    other_max_n: float
    valid_fraction: float
    joint_speed_mean: float
    joint_speed_max: float
    tool_mean_n: float = math.nan   # wipe: tool-vs-world normal force, mean / max over substeps
    tool_max_n: float = math.nan


def _seed_samplers(sampler: Any, rng: Any) -> None:
    """Point ``sampler`` and every nested sub-sampler at ``rng`` (robosuite samplers cache it)."""
    if sampler is None:
        return
    sampler.rng = rng
    for sub in getattr(sampler, "samplers", {}).values():
        _seed_samplers(sub, rng)


class NativeSession:
    """One reset-to-end episode in the deployment controller."""

    def __init__(self, task: str, seed: int, *, render: bool = True) -> None:
        self.task = task
        self.seed = int(seed)
        self.render = bool(render)
        self.env = make_env(task, self.seed, render=render)
        self._binding = None
        self.force_target_name: Optional[str] = None  # object the force binding currently measures
        self._acc = self._new_acc()
        self._hooked = False
        self.obs: Optional[dict] = None
        self.sim_step_index = 0
        self.control_step_index = 0
        self.initial: dict[str, Any] = {}

    # --- sampling -------------------------------------------------------
    @staticmethod
    def _new_acc() -> dict:
        return dict(n=0, nv=0, ls=0.0, rs=0.0, lm=0.0, rm=0.0, om=0.0, js=0.0, jm=0.0, ts=0.0, tm=0.0, tv=0)

    def _mj(self):
        sim = self.env.sim
        return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)

    def _target_geoms(self) -> tuple[str, tuple[int, ...]]:
        if self.task == "nut_assembly":
            nut = self.env.nuts[self.env.nut_id]
            geoms = getattr(nut, "contact_geoms", None)
            if geoms:
                return nut.name, tuple(int(self.env.sim.model.geom_name2id(n)) for n in geoms)
            return "", ()
        for attr in ("cubeA", "cube"):
            obj = getattr(self.env, attr, None)
            geoms = getattr(obj, "contact_geoms", None)
            if obj is not None and geoms:
                return attr, tuple(int(self.env.sim.model.geom_name2id(n)) for n in geoms)
        return "", ()

    def _mujoco_objects(self) -> tuple:
        objs = getattr(getattr(self.env, "model", None), "mujoco_objects", None)
        return tuple(objs) if objs else ()

    def geom_ids_of(self, env_name: str) -> tuple[int, ...]:
        """Contact geom ids of the scene object named ``env_name`` (empty if it has none)."""
        for obj in self._mujoco_objects():
            if getattr(obj, "name", None) != env_name:
                continue
            geoms = getattr(obj, "contact_geoms", None) or ()
            return tuple(int(self.env.sim.model.geom_name2id(n)) for n in geoms)
        return ()

    def object_contacts(self, names) -> dict[str, tuple[float, float]]:
        """Instantaneous (post-step) left/right finger normal force on each named object; NaN = unmeasured."""
        import mujoco
        from synthesis_v2.grasp_force_contract import read_grasp_contact_sample, resolve_grasp_binding

        mj_m, mj_d = self._mj()
        cache = self.__dict__.setdefault("_obj_bindings", {})
        out: dict[str, tuple[float, float]] = {}
        for nm in names:
            if nm not in cache:
                ids = self.geom_ids_of(nm)
                cache[nm] = resolve_grasp_binding(mj_m, mujoco, ids, nm) if ids else None
            smp = read_grasp_contact_sample(mj_m, mj_d, cache[nm]) if cache[nm] is not None else None
            out[nm] = (float(smp.left_force_n), float(smp.right_force_n)) if smp is not None and smp.valid else (math.nan, math.nan)
        return out

    def set_force_target(self, env_name: str) -> None:
        """Point the grasp-force measurement at ``env_name``.

        The binding is re-resolved per grasp *instance*, so a program that grasps two different
        objects measures each hold against its own object instead of a single task-wide target.
        """
        import mujoco
        from synthesis_v2.grasp_force_contract import resolve_grasp_binding

        ids = self.geom_ids_of(env_name)
        if not ids:
            raise ValueError(f"no contact geoms for object {env_name!r}")
        self._binding = resolve_grasp_binding(self._mj()[0], mujoco, ids, env_name)
        self.force_target_name = env_name

    def _hook(self) -> None:
        import mujoco
        from synthesis_v2.grasp_force_contract import read_grasp_contact_sample, resolve_grasp_binding

        mj_m, mj_d = self._mj()
        wipe = self.task == "wipe"
        if wipe:
            tool_geoms = {i for i in range(mj_m.ngeom)
                          if (mujoco.mj_id2name(mj_m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("gripper0")}
            fbuf = np.zeros(6)
        else:
            name, ids = self._target_geoms()
            self._binding = resolve_grasp_binding(mj_m, mujoco, ids, name)
        joint_idx = list(self.env.robots[0]._ref_joint_vel_indexes)
        orig_step = self.env.sim.step

        def stepped(*args, **kwargs):
            out = orig_step(*args, **kwargs)
            a = self._acc
            a["n"] += 1
            if wipe:
                tot = 0.0
                for ci in range(mj_d.ncon):
                    c = mj_d.contact[ci]
                    if (int(c.geom1) in tool_geoms) != (int(c.geom2) in tool_geoms):
                        mujoco.mj_contactForce(mj_m, mj_d, ci, fbuf)
                        tot += abs(float(fbuf[0]))
                a["tv"] += 1
                a["ts"] += tot
                a["tm"] = max(a["tm"], tot)
                s = None
            else:
                s = read_grasp_contact_sample(mj_m, mj_d, self._binding)
            if s is not None and s.valid:
                a["nv"] += 1
                a["ls"] += s.left_force_n
                a["rs"] += s.right_force_n
                a["lm"] = max(a["lm"], s.left_force_n)
                a["rm"] = max(a["rm"], s.right_force_n)
                a["om"] = max(a["om"], s.other_force_n)
            speed = float(np.linalg.norm(np.asarray(mj_d.qvel)[joint_idx]))
            a["js"] += speed
            a["jm"] = max(a["jm"], speed)
            return out

        self.env.sim.step = stepped
        self._hooked = True

    def _drain(self) -> StepTelemetry:
        a, self._acc = self._acc, self._new_acc()
        n = max(a["n"], 1)
        nv = a["nv"]
        nan = math.nan
        return StepTelemetry(
            n_substeps=a["n"],
            left_mean_n=a["ls"] / nv if nv else nan,
            right_mean_n=a["rs"] / nv if nv else nan,
            left_max_n=a["lm"] if nv else nan,
            right_max_n=a["rm"] if nv else nan,
            other_max_n=a["om"] if nv else nan,
            valid_fraction=nv / n if a["n"] else 0.0,
            joint_speed_mean=a["js"] / n if a["n"] else nan,
            joint_speed_max=a["jm"] if a["n"] else nan,
            tool_mean_n=a["ts"] / a["tv"] if a["tv"] else nan,
            tool_max_n=a["tm"] if a["tv"] else nan,
        )

    # --- episode API ----------------------------------------------------
    def reset(self) -> dict:
        np.random.seed(self.seed)
        self.env.rng = np.random.default_rng(self.seed)
        # Composite samplers (nut_assembly) build their sub-samplers once, during env construction,
        # each holding a reference to the *original* env.rng.  Re-seeding only the top-level object
        # leaves those sub-samplers on the construction-time stream, which makes the sampled object
        # pose depend on how much randomness the process consumed earlier (observed as different
        # nut poses for the same seed in different processes).  Seed every sampler in the tree.
        _seed_samplers(getattr(self.env, "placement_initializer", None), self.env.rng)
        self.obs = self.env.reset()
        self.sim_step_index = 0
        self.control_step_index = 0
        if not self._hooked:
            self._hook()
        self._acc = self._new_acc()
        self.initial = self.snapshot_initial()
        return self.obs

    def snapshot_initial(self) -> dict[str, Any]:
        sim = self.env.sim
        mj_m, mj_d = self._mj()
        out: dict[str, Any] = {
            "qpos": np.asarray(mj_d.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(mj_d.qvel, dtype=np.float64).copy(),
            "time": float(mj_d.time),
        }
        if self.task == "wipe":
            out["marker_pos"] = np.stack([
                np.asarray(sim.data.body_xpos[sim.model.body_name2id(m.root_body)], dtype=np.float64)
                for m in self.env.model.mujoco_arena.markers])
        for key in ("cube_pos", "cube_quat", "cubeA_pos", "cubeB_pos", "cubeA_quat", "cubeB_quat",
                    "SquareNut_pos", "SquareNut_quat"):
            if self.obs is not None and key in self.obs:
                out[key] = np.asarray(self.obs[key], dtype=np.float64).copy()
        return out

    def step(self, action7: np.ndarray) -> tuple[dict, StepTelemetry]:
        a = np.asarray(action7, dtype=np.float32)
        if a.shape != (7,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1.0 + 1e-6):
            raise ValueError(f"invalid 7D action {a!r}")
        self._acc = self._new_acc()
        if self.task == "wipe":
            if a[6] != 0.0:
                raise ValueError("wipe has no gripper: the 7th action must be 0")
            obs, _rew, _done, _info = self.env.step(a[:6])
        else:
            obs, _rew, _done, _info = self.env.step(a)
        tel = self._drain()
        self.obs = obs
        self.sim_step_index += tel.n_substeps
        self.control_step_index += 1
        return obs, tel

    # --- stage-level snapshot / restore ---------------------------------
    _CTRL_TYPES = (np.ndarray, float, int, bool, list, tuple, type(None))

    def save_state(self) -> dict[str, Any]:
        """Everything that influences the future of a rollout: MuJoCo integration state, controller
        internals, gripper command memory, env clocks and the current observation."""
        import copy

        import mujoco

        mj_m, mj_d = self._mj()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        phys = np.empty(mujoco.mj_stateSize(mj_m, spec), dtype=np.float64)
        mujoco.mj_getState(mj_m, mj_d, phys, spec)
        robot = self.env.robots[0]
        ctrl: dict[str, dict[str, Any]] = {}
        for part, c in robot.composite_controller.part_controllers.items():
            for k, v in vars(c).items():
                if "interpolator" in k and v is not None:
                    raise RuntimeError(f"stateful interpolator {part}.{k} is not snapshot-safe")
            ctrl[part] = {k: copy.deepcopy(v) for k, v in vars(c).items() if isinstance(v, self._CTRL_TYPES)}
        grippers = {arm: copy.deepcopy(np.asarray(g.current_action)) for arm, g in robot.gripper.items()}
        extra: dict[str, Any] = {}
        if self.task == "wipe":
            names = {id(m): i for i, m in enumerate(self.env.model.mujoco_arena.markers)}
            extra["wiped"] = [names[id(m)] for m in self.env.wiped_markers]
            extra["force_bias"] = copy.deepcopy(self.env.ee_force_bias)
            extra["torque_bias"] = copy.deepcopy(self.env.ee_torque_bias)
        return {"phys": phys, "ctrl": ctrl, "grippers": grippers, "extra": extra,
                "env": {"timestep": int(self.env.timestep), "cur_time": float(self.env.cur_time),
                        "done": bool(self.env.done)},
                "obs": dict(self.obs), "sim_step_index": int(self.sim_step_index),
                "control_step_index": int(self.control_step_index),
                "force_target": self.force_target_name}

    def restore_state(self, st: dict[str, Any]) -> None:
        import copy

        import mujoco

        mj_m, mj_d = self._mj()
        mujoco.mj_setState(mj_m, mj_d, np.asarray(st["phys"], dtype=np.float64), mujoco.mjtState.mjSTATE_INTEGRATION)
        self.env.sim.forward()
        robot = self.env.robots[0]
        for part, vals in st["ctrl"].items():
            c = robot.composite_controller.part_controllers[part]
            for k, v in vals.items():
                setattr(c, k, copy.deepcopy(v))
        for arm, cur in st["grippers"].items():
            robot.gripper[arm].current_action = copy.deepcopy(cur)
        if self.task == "wipe":
            markers = list(self.env.model.mujoco_arena.markers)
            self.env.wiped_markers = [markers[i] for i in st["extra"]["wiped"]]
            gid = self.env.sim.model.geom_name2id
            for i, m in enumerate(markers):
                self.env.sim.model.geom_rgba[gid(m.visual_geoms[0])][3] = 0 if i in st["extra"]["wiped"] else 1
            self.env.ee_force_bias = copy.deepcopy(st["extra"]["force_bias"])
            self.env.ee_torque_bias = copy.deepcopy(st["extra"]["torque_bias"])
        self.env.timestep, self.env.cur_time, self.env.done = (
            st["env"]["timestep"], st["env"]["cur_time"], st["env"]["done"])
        self.obs = dict(st["obs"])
        self.sim_step_index = st["sim_step_index"]
        self.control_step_index = st["control_step_index"]
        self._acc = self._new_acc()
        # the measurement binding is part of the rollout state: a restore into a fresh session must
        # measure the object that was being held at snapshot time, not the construction-time default
        if st.get("force_target"):
            self.set_force_target(st["force_target"])

    def success(self) -> bool:
        try:
            return bool(self.env._check_success())
        except Exception:
            return False

    def current_sim_time(self) -> float:
        return float(self._mj()[1].time)

    def model_xml(self) -> str:
        return self.env.sim.model.get_xml()

    def camera_parameters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        mj_m, _ = self._mj()
        import mujoco

        for cam in (CAM_MAIN, CAM_WRIST):
            cid = mujoco.mj_name2id(mj_m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            out[cam] = {
                "id": int(cid),
                "fovy_deg": float(mj_m.cam_fovy[cid]),
                "pos": [float(x) for x in mj_m.cam_pos[cid]],
                "quat_wxyz": [float(x) for x in mj_m.cam_quat[cid]],
                "body_id": int(mj_m.cam_bodyid[cid]),
                "width": IMG_SIZE,
                "height": IMG_SIZE,
            }
        return out

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
