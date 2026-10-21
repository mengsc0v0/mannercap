"""Shared base class for single-arm Robosuite Franka environments.

Eliminates code duplication across robosuite_cubes.py, robosuite_cube_lift.py,
robosuite_cubes_restack.py, robosuite_spill_wipe.py, and robosuite_nut_assembly.py.
"""

from __future__ import annotations

import os
import inspect
from typing import Any

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf
from robosuite.utils.camera_utils import get_real_depth_map

from capx.envs.base import BaseEnv
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud

os.environ.setdefault("MUJOCO_GL", "egl")



# ---------------------------------------------------------------------------
# Manner-aware trajectory helpers (module-level, used by profiled methods)
# ---------------------------------------------------------------------------

def _trapezoid_waypoints(q_start, q_end, distance, T, t_acc, v_max, a_max, dt):
    import numpy as _np
    direction = (q_end - q_start) / distance
    n_steps = max(1, int(_np.ceil(T / dt)))
    waypoints = []
    for i in range(1, n_steps + 1):
        t = min(i * dt, T)
        if t <= t_acc:
            s = 0.5 * a_max * t ** 2
        elif t <= T - t_acc:
            s = 0.5 * a_max * t_acc ** 2 + v_max * (t - t_acc)
        else:
            t_dec = T - t
            s = distance - 0.5 * a_max * t_dec ** 2
        s = float(_np.clip(s, 0.0, distance))
        waypoints.append(q_start + direction * s)
    return waypoints


def _quintic_waypoints(q_start, q_end, T, dt):
    import numpy as _np
    n_steps = max(1, int(_np.ceil(T / dt)))
    waypoints = []
    for i in range(1, n_steps + 1):
        s = min(i * dt / T, 1.0)
        blend = 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5
        waypoints.append(q_start + (q_end - q_start) * blend)
    return waypoints


class RobosuiteBaseEnv(BaseEnv):
    """Base class for single-arm Robosuite Franka environments.

    Subclasses must implement:
        - _create_robosuite_env() to create the specific robosuite environment
        - _post_env_init() for any task-specific post-initialization (optional)
        - reset() for task-specific reset logic
        - get_observation() for task-specific observation processing
        - compute_reward() for task-specific reward
        - task_completed() for task-specific completion check
    """

    # Subclasses can override these defaults
    _SUBSAMPLE_RATE: int = 5
    _ACTION_SLICE: int = -1  # action[:-1] for most envs, action[:-2] for spill_wipe

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__()
        self.controller_cfg = controller_cfg
        self.max_steps = max_steps
        self.save_camera_name = "robot0_robotview"
        self.render_camera_names = [self.save_camera_name]
        self.segmentation_level = "instance"

        self._render_width = 512
        self._render_height = 512

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._rng = np.random.default_rng(seed)

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = self._SUBSAMPLE_RATE

        # Control state
        self._current_joints = np.zeros(7, dtype=np.float64)
        self._gripper_fraction = 1.0  # 1.0 = open, 0.0 = closed

        # Manner telemetry recording (MannerValidator integration)
        self._manner_recording: bool = False
        self._manner_telemetry: list = []          # list[MannerStepSnapshot]
        self._manner_phase: str = "other"          # "move" | "grip" | "stiff" | "other"
        self._manner_prev_vel: np.ndarray = np.zeros(7, dtype=np.float64)

    def _init_robot_links(self) -> None:
        """Initialize robot link indices and base transforms. Call after robosuite_env is created."""
        self.gripper_metric_length = 0.04
        self.base_link_idx = self.robosuite_env.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx = self.robosuite_env.sim.model.body_name2id("gripper0_right_eef")

        self.base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

    def _init_viser_debug(self, viser_debug: bool) -> None:
        """Initialize viser debug state. Call at end of subclass __init__."""
        if viser_debug:
            self.viser_server = viser.ViserServer()
            self.pyroki_ee_frame_handle = None
            self.mjcf_ee_frame_handle = None
            self.urdf_vis = None
            self.viser_img_handle = None
            self.image_frustum_handle = None
            self.gripper_metric_length = 0.0584
            self.cube_points = None
            self.cube_color = None
            self.cube_center = None
            self.cube_rot = None

    # ----------------------- Shared Control Interface -----------------------

    def _set_gripper(self, fraction: float) -> None:
        """Set gripper opening fraction.

        Args:
            fraction: 0.0 (closed) to 1.0 (open)
        """
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _build_action(self) -> np.ndarray:
        """Build a robosuite action array from current joints and gripper state."""
        action = np.concatenate(
            [self._current_joints, [self._gripper_fraction, self._gripper_fraction]]
        )
        # Map gripper: 1.0 (open) -> -1.0, 0.0 (closed) -> 1.0
        action[-2:] = 1.0 - action[-2:] * 2.0
        return action

    def _do_robosuite_step(self, action: np.ndarray) -> None:
        """Step robosuite with the given action, handling render skipping."""
        sliced = action[:self._ACTION_SLICE] if self._ACTION_SLICE != 0 else action
        need_render = (self._record_frames and self._sim_step_count % self._subsample_rate == 0) or hasattr(self, "viser_server")
        # The original machine used a patched robosuite step API. Upstream
        # 1.5.2 has no skip_render_images argument; inspect once rather than
        # catching TypeError and possibly executing a physical step twice.
        if not hasattr(self, "_supports_skip_render_images"):
            self._supports_skip_render_images = (
                "skip_render_images" in inspect.signature(self.robosuite_env.step).parameters
            )
        if need_render or not self._supports_skip_render_images:
            self.robosuite_env.step(sliced)
        else:
            self.robosuite_env.step(sliced, skip_render_images=True)

    def _step_once(self) -> None:
        """Execute one simulation step with current control state."""
        action = self._build_action()
        self._do_robosuite_step(action)

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        if self._manner_recording:
            self._capture_manner_snapshot()
        self._sim_step_count += 1

    def move_to_joints_non_blocking(self, joints: np.ndarray) -> None:
        """Move to target joint positions using Robosuite's controller (non-blocking)."""
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
        action[-2:] = 1.0 - action[-2:] * 2.0

        self._do_robosuite_step(action)

        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()

        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

        self._sim_step_count += 1

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)

            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0

            self._do_robosuite_step(action)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )


    def move_to_joints_profiled(
        self,
        joints,
        *,
        v_max=0.15,
        a_max=0.10,
        jerk_limit=None,
        tolerance=0.005,
        phase="move",
    ):
        """Move joints with a velocity profile (Manner A+D: 速度/平稳约束).

        Args:
            joints:     (7,) target joint positions in radians.
            v_max:      Max joint-space speed rad/s.  缓缓 e.g. 0.03, 迅速 e.g. 0.25.
            a_max:      Max joint-space acceleration rad/s^2.
            jerk_limit: When set, minimum-jerk quintic interpolation (Manner D: 平稳).
                        None -> trapezoidal profile. Typical: 0.5 (smooth) – 5.0.
            tolerance:  Final convergence tolerance (rad).
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        robosuite_obs = self.robosuite_env._get_observations()
        current = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        distance = float(np.linalg.norm(target - current))
        if distance < 1e-4:
            self._current_joints = target
            return

        sim_freq = float(getattr(self, "_sim_freq", 20.0))
        dt = 1.0 / sim_freq

        if jerk_limit is not None:
            t_jerk = (60.0 * distance / max(jerk_limit, 1e-6)) ** (1.0 / 3.0)
            t_vel = 15.0 * distance / (8.0 * max(v_max, 1e-6))
            T = max(t_jerk, t_vel, dt)
            waypoints = _quintic_waypoints(current, target, T, dt)
        else:
            t_acc = max(v_max, 1e-6) / max(a_max, 1e-6)
            d_acc = 0.5 * a_max * t_acc ** 2
            if 2.0 * d_acc >= distance:
                t_acc = np.sqrt(distance / max(a_max, 1e-6))
                T = 2.0 * t_acc
            else:
                T = 2.0 * t_acc + (distance - 2.0 * d_acc) / max(v_max, 1e-6)
            T = max(T, dt)
            waypoints = _trapezoid_waypoints(current, target, distance, T, t_acc, v_max, a_max, dt)

        self._current_joints = target
        _prev_phase = self._manner_phase
        self._manner_phase = phase
        for q_cmd in waypoints:
            self._current_joints = q_cmd
            self._step_once()
        self._manner_phase = _prev_phase

        self.move_to_joints_blocking(target, tolerance=tolerance, max_steps=40)

    def set_joint_stiffness(self, kp):
        """Adjust joint position controller gain (Manner B: 阻抗刚度).

        Lower kp -> compliant (轻轻/柔和); higher kp -> stiff (用力/牢牢).
        Default kp in panda_joint_ctrl.json is 1500.

        Args:
            kp: Joint position gain.  Range 300-1500.
                300-600  -> low stiffness (轻轻, 柔和接触)
                800-1200 -> medium (正常)
                1300-1500-> high stiffness (用力, 牢牢按压)
        """
        try:
            robot = self.robosuite_env.robots[0]
            # Composite controller path: part_controllers["right"]
            comp = getattr(robot, "composite_controller", None)
            if comp is not None and hasattr(comp, "part_controllers"):
                pc = comp.part_controllers.get("right") or comp.part_controllers.get("arm")
                if pc is not None and hasattr(pc, "kp"):
                    if isinstance(pc.kp, np.ndarray):
                        pc.kp[:] = float(kp)
                    else:
                        pc.kp = float(kp)
            else:
                # Fallback: legacy single-controller path
                ctrl = getattr(robot, "controller", None)
                if ctrl is not None:
                    if hasattr(ctrl, "kp"):
                        if isinstance(ctrl.kp, np.ndarray):
                            ctrl.kp[:] = float(kp)
                        else:
                            ctrl.kp = float(kp)
                    elif hasattr(ctrl, "_kp"):
                        ctrl._kp = float(kp)
        except Exception:
            pass
        if self._manner_recording:
            _prev_phase = self._manner_phase
            self._manner_phase = "stiff"
            self._capture_manner_snapshot()
            self._manner_phase = _prev_phase

    def close_gripper_partial(self, *, force=None, width=None, steps=60):
        """Close gripper to partial position (Manner C: 夹持力矩).

        Args:
            force:  Normalised grip force [0,1].  0->虚握, 1->紧握.
                    Overridden by width when both are given.
            width:  Finger-gap target metres.  Panda max ~2*gripper_metric_length.
                    e.g. width=0.01 -> lightly pinched (轻捏).
            steps:  Simulation steps to hold at target.
        """
        max_opening = self.gripper_metric_length * 2.0
        if width is not None:
            fraction = float(np.clip(width / max(max_opening, 1e-6), 0.0, 1.0))
        elif force is not None:
            fraction = float(np.clip(1.0 - force, 0.0, 1.0))
        else:
            fraction = 0.0
        self._set_gripper(fraction)
        _prev_phase = self._manner_phase
        self._manner_phase = "grip"
        for _ in range(steps):
            self._step_once()
        self._manner_phase = _prev_phase

    # ── Manner Telemetry Recording ──────────────────────────────────────────

    def begin_manner_recording(self, *, clear: bool = True) -> None:
        """Start capturing MannerStepSnapshot per simulation step.

        Call this before a manner-conditioned motion block, then call
        ``end_manner_recording()`` and ``get_manner_telemetry()`` to retrieve
        the recorded data for ``MannerValidator.validate()``.
        """
        if clear:
            self._manner_telemetry = []
        self._manner_recording = True

    def end_manner_recording(self) -> None:
        """Stop capturing telemetry (buffer is retained until next begin)."""
        self._manner_recording = False
        self._manner_phase = "other"

    def get_manner_telemetry(self) -> list:
        """Return a copy of the recorded MannerStepSnapshot list."""
        return list(self._manner_telemetry)

    # ── Stage-checkpoint rollback (privileged repair loop) ──────────────────
    def capture_checkpoint(self) -> dict:
        """Snapshot the full physics + controller/recorder state.

        ``sim.set_state`` only restores MuJoCo physics (qpos/qvel incl. object
        free-joints). The controller target (``_current_joints``), gripper
        fraction, manner phase, step counter, and telemetry length are Python
        side state that must be captured separately so a rollback lands on a
        byte-for-byte reproducible point. Impedance kp is intentionally NOT
        stored: every runner stage re-issues set_(joint_)stiffness, so it is
        re-established on replay.
        """
        return {
            "mj_state": self.robosuite_env.sim.get_state(),
            "current_joints": self._current_joints.copy(),
            "gripper_fraction": float(self._gripper_fraction),
            "manner_phase": self._manner_phase,
            "sim_step_count": int(self._sim_step_count),
            "telemetry_len": len(self._manner_telemetry),
        }

    def restore_checkpoint(self, ckpt: dict) -> None:
        """Restore state saved by ``capture_checkpoint``.

        ``sim.forward()`` is mandatory after ``set_state`` so derived
        quantities (xpos/xquat/contact forces read by the privileged API) are
        recomputed before the next stage runs. The manner telemetry buffer is
        truncated back to the checkpoint length, discarding any samples that a
        now-rolled-back failed stage appended.
        """
        sim = self.robosuite_env.sim
        sim.set_state(ckpt["mj_state"])
        sim.forward()
        self._current_joints = ckpt["current_joints"].copy()
        self._gripper_fraction = ckpt["gripper_fraction"]
        self._manner_phase = ckpt["manner_phase"]
        self._sim_step_count = ckpt["sim_step_count"]
        del self._manner_telemetry[ckpt["telemetry_len"]:]

    def _compute_grasp_contact_force(self) -> float:
        """Sum of normal contact forces (N) between gripper finger geoms and
        manipulated-object geoms.

        ``cfrc_ext`` (per-body external force) does not register fingertip grasp
        force in this MuJoCo model, so grasp force is read per-contact via
        ``mujoco.mj_contactForce`` over (finger geom × object geom) pairs. An
        object geom is any geom that is neither a robot/gripper geom nor a
        static arena geom (table/wall/mount/bin/floor/peg).
        """
        import mujoco
        sim = self.robosuite_env.sim
        mj_m = getattr(sim.model, "_model", sim.model)
        mj_d = getattr(sim.data, "_data", sim.data)

        fg = getattr(self, "_finger_geom_ids", None)
        if fg is None:
            fg = set()
            arena = set()
            for gi in range(mj_m.ngeom):
                nm = (mujoco.mj_id2name(mj_m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "").lower()
                if ("finger" in nm or "pad" in nm) and "gripper0" in nm:
                    fg.add(gi)
                if any(k in nm for k in ("table", "wall", "mount", "bin", "floor", "peg_")):
                    arena.add(gi)
            self._finger_geom_ids = fg
            self._arena_geom_ids = arena
        arena = getattr(self, "_arena_geom_ids", set())
        if not fg:
            return 0.0

        buf = np.zeros(6)
        total = 0.0
        for ci in range(mj_d.ncon):
            c = mj_d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            in1, in2 = g1 in fg, g2 in fg
            if not (in1 or in2):
                continue
            other = g2 if in1 else g1
            if other in fg:
                continue  # finger-finger, ignore
            nm = (mujoco.mj_id2name(mj_m, mujoco.mjtObj.mjOBJ_GEOM, other) or "").lower()
            if other in arena or nm.startswith("robot0") or nm.startswith("gripper0"):
                continue
            mujoco.mj_contactForce(mj_m, mj_d, ci, buf)
            total += abs(float(buf[0]))
        return float(total)

    def _grasp_target_geom_ids(self):
        env = self.robosuite_env
        for attr in ("cubeA", "cube", "cup"):
            obj = getattr(env, attr, None)
            geoms = getattr(obj, "contact_geoms", None)
            if obj is not None and geoms:
                model = env.sim.model
                return attr, tuple(int(model.geom_name2id(n)) for n in geoms)
        return "", ()

    def _read_grasp_sample(self):
        """P0-1: per-finger, target-object-only normal force (N) + validity."""
        import mujoco
        from synthesis_v2.grasp_force_contract import (
            read_grasp_contact_sample, resolve_grasp_binding,
        )
        sim = self.robosuite_env.sim
        mj_m = getattr(sim.model, "_model", sim.model)
        mj_d = getattr(sim.data, "_data", sim.data)
        cached = getattr(self, "_grasp_binding_cache", None)
        if cached is None or cached[0] != id(mj_m):
            name, ids = self._grasp_target_geom_ids()
            cached = (id(mj_m), resolve_grasp_binding(mj_m, mujoco, ids, name))
            self._grasp_binding_cache = cached
        return read_grasp_contact_sample(mj_m, mj_d, cached[1])

    def _capture_manner_snapshot(self) -> None:
        """Capture one telemetry snapshot from the current sim state."""
        try:
            from capx.envs.manner_validator import MannerStepSnapshot
        except ImportError:
            return

        try:
            robs = self.robosuite_env._get_observations()
            joint_vel = np.array(robs.get("robot0_joint_vel", np.zeros(7)), dtype=np.float64)
            joint_acc = np.array(robs.get("robot0_joint_acc", np.zeros(7)), dtype=np.float64)
            joint_pos = np.array(robs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float64)
            gripper_qpos = np.array(robs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float64)
            eef_wxyz_xyz = getattr(self, "gripper_link_wxyz_xyz", None)
            if eef_wxyz_xyz is not None:
                eef_quat = np.asarray(eef_wxyz_xyz[:4], dtype=np.float64)
                eef_pos = np.asarray(eef_wxyz_xyz[4:], dtype=np.float64)
            else:
                eef_quat = np.zeros(4)
                eef_pos = np.zeros(3)

            # kp: read from composite part_controllers["right"]
            try:
                robot = self.robosuite_env.robots[0]
                comp = getattr(robot, "composite_controller", None)
                kp_val = 1500.0
                if comp is not None and hasattr(comp, "part_controllers"):
                    pc = comp.part_controllers.get("right") or comp.part_controllers.get("arm")
                    if pc is not None and hasattr(pc, "kp"):
                        kp_raw = pc.kp
                        kp_val = float(np.mean(kp_raw)) if isinstance(kp_raw, np.ndarray) else float(kp_raw)
                else:
                    ctrl = getattr(robot, "controller", None)
                    if ctrl is not None:
                        kp_raw = getattr(ctrl, "kp", 1500.0)
                        kp_val = float(np.mean(kp_raw)) if isinstance(kp_raw, np.ndarray) else float(kp_raw)
            except Exception:
                kp_val = 1500.0

            # contact force norm from cfrc_ext (N, skip internal/world bodies)
            try:
                cfrc = self.robosuite_env.sim.data.cfrc_ext  # (nbody, 6)
                cfrc_norm = float(np.max(np.linalg.norm(cfrc, axis=1)))
            except Exception:
                cfrc_norm = 0.0

            # grasp contact force: TRUE finger↔object normal force (N).
            # cfrc_ext (per-body external force) does NOT register fingertip
            # grasp force in this MuJoCo model, so it is read per-contact via
            # mujoco.mj_contactForce over (finger geom × object geom) pairs.
            try:
                grasp_contact_force = self._compute_grasp_contact_force()
            except Exception:
                grasp_contact_force = 0.0

            # P0-1: auditable per-finger target-object grasp force.
            try:
                gs = self._read_grasp_sample()
                grasp_kwargs = dict(
                    grasp_left_force_n=float(gs.left_force_n),
                    grasp_right_force_n=float(gs.right_force_n),
                    grasp_other_force_n=float(gs.other_force_n),
                    grasp_left_contacts=int(gs.left_contacts),
                    grasp_right_contacts=int(gs.right_contacts),
                    grasp_left_geom_id=int(gs.left_geom_id),
                    grasp_right_geom_id=int(gs.right_geom_id),
                    grasp_force_valid=bool(gs.valid),
                    grasp_force_reason=str(gs.reason),
                )
            except Exception as _exc:
                grasp_kwargs = dict(
                    grasp_force_valid=False,
                    grasp_force_reason="contact_read_error:" + type(_exc).__name__,
                )

            # object ground-truth poses (privileged): generically probe raw
            # observables for "<name>_pos"/"<name>_quat" pairs that are NOT the
            # robot itself and NOT relative (gripper_to_*) quantities.
            try:
                obj_names = []
                obj_pos_list = []
                obj_quat_list = []
                for key in sorted(robs.keys()):
                    if not key.endswith("_pos") or key.startswith("robot0_") or "gripper_to_" in key:
                        continue
                    quat_key = key[:-4] + "_quat"
                    if quat_key not in robs:
                        continue
                    pos_val = np.asarray(robs[key], dtype=np.float64)
                    quat_val = np.asarray(robs[quat_key], dtype=np.float64)
                    if pos_val.shape != (3,) or quat_val.shape != (4,):
                        continue
                    obj_names.append(key[:-4])
                    obj_pos_list.append(pos_val)
                    obj_quat_list.append(quat_val)
                object_names = tuple(obj_names)
                object_pos = np.stack(obj_pos_list) if obj_pos_list else np.zeros((0, 3))
                object_quat = np.stack(obj_quat_list) if obj_quat_list else np.zeros((0, 4))
            except Exception:
                object_names = ()
                object_pos = np.zeros((0, 3))
                object_quat = np.zeros((0, 4))

            snap = MannerStepSnapshot(
                t=round(self._sim_step_count * (1.0 / float(getattr(self, "_sim_freq", 20.0))), 4),
                phase=self._manner_phase,
                joint_vel=joint_vel,
                joint_acc=joint_acc,
                kp=kp_val,
                gripper_fraction=float(self._gripper_fraction),
                gripper_qpos=gripper_qpos,
                cfrc_norm=cfrc_norm,
                grasp_contact_force=grasp_contact_force,
                **grasp_kwargs,
                joint_pos=joint_pos,
                eef_pos=eef_pos,
                eef_quat=eef_quat,
                object_names=object_names,
                object_pos=object_pos,
                object_quat=object_quat,
            )
            self._manner_telemetry.append(snap)
        except Exception:
            pass  # never crash the simulation

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._step_count += 1
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    # ----------------------- Camera / Observation Helpers -----------------------

    def _process_camera_observations(
        self, robosuite_obs: dict[str, Any], *, base_wxyz_xyz: np.ndarray | None = None
    ) -> None:
        """Process camera observations in-place on robosuite_obs.

        Adds pose, pose_mat, intrinsics, and images for each camera in render_camera_names.
        """
        if base_wxyz_xyz is None:
            base_wxyz_xyz = self.base_link_wxyz_xyz

        for camera_name in self.render_camera_names:
            if camera_name not in robosuite_obs:
                robosuite_obs[camera_name] = {}

            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(
                        self.robosuite_env.sim.data.get_camera_xmat(camera_name)
                    ).wxyz,
                    self.robosuite_env.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (
                    vtf.SE3(wxyz_xyz=base_wxyz_xyz).inverse()
                    @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz)
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )

            robosuite_obs[camera_name]["pose"] = np.concatenate(
                [
                    cam_robot_tf.translation(),
                    cam_robot_tf.rotation().wxyz,
                ]
            )
            robosuite_obs[camera_name]["pose_mat"] = cam_robot_tf.as_matrix()

            cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)

            K = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            robosuite_obs[camera_name]["intrinsics"] = K

            robosuite_obs[camera_name]["images"] = {}
            if camera_name + "_image" in robosuite_obs:
                robosuite_obs[camera_name]["images"]["rgb"] = robosuite_obs[camera_name + "_image"][
                    ::-1
                ]
            if camera_name + "_depth" in robosuite_obs:
                depth_metric = get_real_depth_map(
                    self.robosuite_env.sim, robosuite_obs[camera_name + "_depth"][::-1]
                )
                robosuite_obs[camera_name]["images"]["depth"] = depth_metric
            if camera_name + "_segmentation_" + self.segmentation_level in robosuite_obs:
                robosuite_obs[camera_name]["images"]["segmentation"] = robosuite_obs[
                    camera_name + "_segmentation_" + self.segmentation_level
                ][::-1]

    def _compute_gripper_obs(self, robosuite_obs: dict[str, Any]) -> None:
        """Compute gripper pose and add robot_joint_pos / robot_cartesian_pos to obs."""
        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )

        robosuite_obs["robot_joint_pos"] = np.concatenate(
            [
                robosuite_obs["robot0_joint_pos"],
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )
        robosuite_obs["robot_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base.translation(),
                gripper_robot_base.rotation().wxyz,
                [robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length],
            ]
        )

    # ------------------------- Video Capture -------------------------

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def _record_frame(self) -> None:
        if not self._record_frames:
            return

        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        self._frame_buffer.append(frame[::-1])  # Flip vertically

        if self._record_wrist_camera:
            wrist_frame = self.robosuite_env.sim.render(
                camera_name=self._wrist_camera_name,
                width=self._render_width,
                height=self._render_height,
                depth=False,
            )
            self._wrist_frame_buffer.append(wrist_frame[::-1])

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.robosuite_env.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    def render_wrist(self) -> np.ndarray:
        """Render the current frame from the wrist (eye-in-hand) camera."""
        frame = self.robosuite_env.sim.render(
            camera_name=self._wrist_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    # ------------------------- Viser Debugging -------------------------

    def _update_viser_server(self) -> None:
        """Default viser update. Subclasses with different viser logic should override."""
        obs = self.get_observation()
        if self.viser_server is not None:
            self._viser_init_check()

            obs_cartesian = obs["robot_cartesian_pos"][:-1]

            self.mjcf_ee_frame_handle.position = obs_cartesian[:3]
            self.mjcf_ee_frame_handle.wxyz = obs_cartesian[3:]

            rbg_imgs = obs_get_rgb(obs)
            for image_key in rbg_imgs:
                self.viser_img_handle.image = rbg_imgs[image_key]

                if "pose" in obs[image_key]:
                    self.image_frustum_handle.position = obs[image_key]["pose"][:3]
                    self.image_frustum_handle.wxyz = obs[image_key]["pose"][3:]
                    self.image_frustum_handle.image = rbg_imgs[image_key]
                else:
                    self.image_frustum_handle.visible = False

                # Point cloud from depth (use whichever camera has depth)
                if "depth" in obs[image_key].get("images", {}):
                    points, colors = depth_color_to_pointcloud(
                        obs[image_key]["images"]["depth"][:, :, 0],
                        rbg_imgs[image_key],
                        obs[image_key]["intrinsics"],
                    )
                    self.viser_server.scene.add_point_cloud(
                        f"{image_key}/point_cloud",
                        points,
                        colors,
                        point_size=0.001,
                        point_shape="square",
                    )

            if self.cube_center is not None and self.cube_rot is not None:
                self.viser_server.scene.add_frame(
                    "robot0_robotview/cube_frame",
                    position=self.cube_center,
                    wxyz=vtf.SO3.from_matrix(self.cube_rot).wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )

            if self.cube_points is not None and self.cube_color is not None:
                self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/cube_point_cloud",
                    self.cube_points,
                    self.cube_color,
                    point_size=0.001,
                    point_shape="square",
                )

            if hasattr(self, "grasp_sample") and self.grasp_sample is not None:
                grasp = self.grasp_sample[np.argmax(self.grasp_scores)]

                grasp_tf = vtf.SE3.from_matrix(grasp) @ vtf.SE3.from_translation(
                    np.array([0, 0, 0.1])
                )
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=grasp_tf.wxyz_xyz[-3:],
                    wxyz=grasp_tf.wxyz_xyz[:4],
                    axes_length=0.05,
                    axes_radius=0.0015,
                )

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return

        if self.mjcf_ee_frame_handle is None:
            self.mjcf_ee_frame_handle = self.viser_server.scene.add_frame(
                "/panda_ee_target_mjcf", axes_length=0.15, axes_radius=0.005
            )

        if self.viser_img_handle is None:
            img_init = np.zeros((480, 640, 3), dtype=np.uint8)
            self.viser_img_handle = self.viser_server.gui.add_image(img_init, label="Mujoco render")

        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=self._render_width / self._render_height,
                scale=0.05,
            )


__all__ = ["RobosuiteBaseEnv"]
