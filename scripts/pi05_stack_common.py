"""Shared OSC_POSE world-frame stack-task capture utilities for pi0.5 deliverable.

Controller: fixed-stiffness OSC_POSE, kp=150, damping_ratio=1, input_type=delta,
input_ref_frame=world, output_max=[0.05,0.05,0.05,0.5,0.5,0.5] (per agent_handoff.md).
Action 7D = [dx,dy,dz,drx,dry,drz,gripper]; gripper sign verified empirically:
+1 = close, -1 = open (see /tmp/osc_probe probe results, 2026-09-30).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite as suite  # noqa: E402
from robosuite.controllers.composite.composite_controller_factory import (  # noqa: E402
    load_composite_controller_config,
)
from robosuite.utils.placement_samplers import UniformRandomSampler  # noqa: E402

CONTROL_FREQ = 20
DT = 1.0 / CONTROL_FREQ
POS_SCALE = 0.05
ROT_SCALE = 0.5
CAM_MAIN = "agentview"
CAM_WRIST = "robot0_eye_in_hand"
IMG_SIZE = 224

OSC_CONTROLLER_CFG = {
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

# Manner realization for the demo narrative:
# "先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开"
PHASE_SPEEDS = {
    "approach":  dict(v_max=0.40, w_max=2.0, max_steps=120),
    "descend":   dict(v_max=0.05, w_max=1.0, max_steps=250),
    "lift":      dict(v_max=0.10, w_max=1.0, max_steps=150),
    "transport": dict(v_max=0.12, w_max=1.0, max_steps=150),
    "final":     dict(v_max=0.04, w_max=0.5, max_steps=150),
    "retreat":   dict(v_max=0.40, w_max=2.0, max_steps=120),
}
GRASP_STEPS = 15
OPEN_STEPS = 15
GRIP_CLOSE = 1.0
GRIP_OPEN = -1.0

PROMPT_ZH = "先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开"
MANNER_TAG = "phased_5stage"
TASK_NAME = "stack"
TASK_PROMPT = "Stack the red cube on top of the green cube."


_CONTROLLER_CFG_PATH = "/tmp/pi05_osc_pose_world.json"
with open(_CONTROLLER_CFG_PATH, "w") as _f:
    json.dump(OSC_CONTROLLER_CFG, _f)


def make_stack_env(controller_cfg_path: str = _CONTROLLER_CFG_PATH):
    cc = load_composite_controller_config(controller=controller_cfg_path)
    env = suite.environments.manipulation.stack.Stack(
        robots=["Panda"],
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        camera_names=[CAM_MAIN, CAM_WRIST],
        camera_depths=False,
        camera_heights=IMG_SIZE,
        camera_widths=IMG_SIZE,
        renderer="mujoco",
        controller_configs=cc,
        control_freq=CONTROL_FREQ,
        horizon=100000,
        reward_shaping=True,
    )
    cubes = [env.cubeA, env.cubeB]
    env.placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=cubes,
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


def rotvec_delta(cur_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    r_cur = R.from_quat(cur_xyzw)
    r_tgt = R.from_quat(target_xyzw)
    return (r_tgt * r_cur.inv()).as_rotvec()


def eef_state8(obs: dict) -> np.ndarray:
    pos = obs["robot0_eef_pos"]
    quat_xyzw = obs["robot0_eef_quat"]
    rotvec = R.from_quat(quat_xyzw).as_rotvec()
    grip = obs["robot0_gripper_qpos"]
    return np.concatenate([pos, rotvec, grip]).astype(np.float32)


@dataclass
class EpisodeRecorder:
    images: list = field(default_factory=list)          # each (2,224,224,3) uint8
    state: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    timestamp: list = field(default_factory=list)
    dt: list = field(default_factory=list)
    frame_index: list = field(default_factory=list)
    sim_step_index: list = field(default_factory=list)
    terminated: list = field(default_factory=list)
    truncated: list = field(default_factory=list)
    success: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    t_counter: int = 0

    def record_step(self, env, obs_before, action, obs_after, phase_name, sim_step_idx):
        # robosuite's default macros.IMAGE_CONVENTION="opengl" means the
        # observable-layer image is the RAW sim.render() output, unflipped.
        # Empirically confirmed obs["agentview_image"] == env.sim.render(...)
        # bit-for-bit. This codebase's own legacy video pipeline
        # (robosuite_base.py) explicitly does frame[::-1] on raw sim.render()
        # output to produce upright video frames, proving raw render is
        # upside-down. Flip vertically here so images.npy is positively
        # (upright) displayed per agent_handoff.md section 3.
        img_main = obs_before[f"{CAM_MAIN}_image"][::-1]
        img_wrist = obs_before[f"{CAM_WRIST}_image"][::-1]
        self.images.append(np.stack([img_main, img_wrist], axis=0).astype(np.uint8))
        self.state.append(eef_state8(obs_before))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.next_state.append(eef_state8(obs_after))
        self.timestamp.append(self.t_counter * DT)
        self.dt.append(DT)
        self.frame_index.append(self.t_counter)
        self.sim_step_index.append(sim_step_idx)
        self.terminated.append(False)
        self.truncated.append(False)
        self.success.append(bool(env._check_success()))
        self.phase.append(phase_name)
        self.t_counter += 1

    def finalize_terminated(self):
        if self.terminated:
            self.terminated[-1] = True

    def to_arrays(self):
        return dict(
            images=np.asarray(self.images, dtype=np.uint8),
            state=np.asarray(self.state, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            next_state=np.asarray(self.next_state, dtype=np.float32),
            timestamp=np.asarray(self.timestamp, dtype=np.float64),
            dt=np.asarray(self.dt, dtype=np.float64),
            frame_index=np.asarray(self.frame_index, dtype=np.int64),
            sim_step_index=np.asarray(self.sim_step_index, dtype=np.int64),
            terminated=np.asarray(self.terminated, dtype=bool),
            truncated=np.asarray(self.truncated, dtype=bool),
            success=np.asarray(self.success, dtype=bool),
            phase=np.asarray(self.phase, dtype="<U16"),
        )


def servo_to(env, obs, rec: EpisodeRecorder, sim_step_ref, target_pos, target_quat_xyzw,
             v_max, w_max, gripper_cmd, phase_name, tol_pos=0.004, tol_rot=0.03, max_steps=300):
    steps = 0
    while steps < max_steps:
        cur_pos = obs["robot0_eef_pos"]
        cur_quat = obs["robot0_eef_quat"]
        pos_err = target_pos - cur_pos
        rot_err = rotvec_delta(cur_quat, target_quat_xyzw)
        pos_norm = float(np.linalg.norm(pos_err))
        rot_norm = float(np.linalg.norm(rot_err))
        if pos_norm < tol_pos and rot_norm < tol_rot:
            break
        max_pos_step = v_max * DT
        pos_step = pos_err / pos_norm * max_pos_step if pos_norm > max_pos_step else pos_err
        max_rot_step = w_max * DT
        rot_step = rot_err / rot_norm * max_rot_step if rot_norm > max_rot_step else rot_err
        a_pos = np.clip(pos_step / POS_SCALE, -1, 1)
        a_rot = np.clip(rot_step / ROT_SCALE, -1, 1)
        action = np.concatenate([a_pos, a_rot, [gripper_cmd]]).astype(np.float32)
        obs_before = obs
        obs, rew, done, info = env.step(action)
        rec.record_step(env, obs_before, action, obs, phase_name, sim_step_ref[0])
        sim_step_ref[0] += 1
        steps += 1
    return obs, steps


def hold_gripper(env, obs, rec: EpisodeRecorder, sim_step_ref, gripper_cmd, n_steps, phase_name):
    for _ in range(n_steps):
        action = np.concatenate([[0, 0, 0], [0, 0, 0], [gripper_cmd]]).astype(np.float32)
        obs_before = obs
        obs, rew, done, info = env.step(action)
        rec.record_step(env, obs_before, action, obs, phase_name, sim_step_ref[0])
        sim_step_ref[0] += 1
    return obs


def run_one_episode(seed: int):
    """Run the full stack manner-demo episode; returns (rec, obs_final, env, meta)."""
    env = make_stack_env()
    env.rng = np.random.default_rng(seed)
    env.placement_initializer.rng = env.rng
    obs = env.reset()
    down_quat_xyzw = obs["robot0_eef_quat"].copy()
    cubeA_pos0 = obs["cubeA_pos"].copy()
    cubeB_pos0 = obs["cubeB_pos"].copy()

    rec = EpisodeRecorder()
    sim_step_ref = [0]

    above_A = cubeA_pos0 + np.array([0, 0, 0.12])
    obs, n1 = servo_to(env, obs, rec, sim_step_ref, above_A, down_quat_xyzw,
                        gripper_cmd=GRIP_OPEN, phase_name="approach", **PHASE_SPEEDS["approach"])

    grasp_pos = cubeA_pos0.copy()
    obs, n2 = servo_to(env, obs, rec, sim_step_ref, grasp_pos, down_quat_xyzw,
                        gripper_cmd=GRIP_OPEN, phase_name="descend", **PHASE_SPEEDS["descend"])

    obs = hold_gripper(env, obs, rec, sim_step_ref, GRIP_CLOSE, GRASP_STEPS, "grasp")

    lift_pos = grasp_pos + np.array([0, 0, 0.15])
    obs, n3 = servo_to(env, obs, rec, sim_step_ref, lift_pos, down_quat_xyzw,
                        gripper_cmd=GRIP_CLOSE, phase_name="lift", **PHASE_SPEEDS["lift"])

    above_B = cubeB_pos0 + np.array([0, 0, 0.15])
    obs, n4 = servo_to(env, obs, rec, sim_step_ref, above_B, down_quat_xyzw,
                        gripper_cmd=GRIP_CLOSE, phase_name="transport", **PHASE_SPEEDS["transport"])

    place_pos = cubeB_pos0 + np.array([0, 0, 0.041])
    obs, n5 = servo_to(env, obs, rec, sim_step_ref, place_pos, down_quat_xyzw,
                        gripper_cmd=GRIP_CLOSE, phase_name="final", tol_pos=0.008,
                        **{k: v for k, v in PHASE_SPEEDS["final"].items() if k != "max_steps"},
                        max_steps=PHASE_SPEEDS["final"]["max_steps"])

    obs = hold_gripper(env, obs, rec, sim_step_ref, GRIP_OPEN, OPEN_STEPS, "final")

    retreat_pos = obs["robot0_eef_pos"] + np.array([0, 0, 0.15])
    obs, n6 = servo_to(env, obs, rec, sim_step_ref, retreat_pos, down_quat_xyzw,
                        gripper_cmd=GRIP_OPEN, phase_name="retreat", **PHASE_SPEEDS["retreat"])

    rec.finalize_terminated()
    task_ok = bool(env._check_success())
    meta = dict(
        seed=seed, task_ok=task_ok,
        cubeA_pos0=cubeA_pos0.tolist(), cubeB_pos0=cubeB_pos0.tolist(),
        cubeA_pos_final=obs["cubeA_pos"].tolist(), cubeB_pos_final=obs["cubeB_pos"].tolist(),
        phase_steps=dict(approach=n1, descend=n2, grasp=GRASP_STEPS, lift=n3,
                          transport=n4, final=n5, open=OPEN_STEPS, retreat=n6),
        n_steps=rec.t_counter,
        gripper_sign_convention="action[6]=+1 closes gripper (finger separation decreases); -1 opens.",
        controller="OSC_POSE world-frame delta, kp=150 fixed, damping_ratio=1, "
                   "output_max=[0.05]*3+[0.5]*3, control_freq=20Hz",
    )
    return rec, obs, env, meta
