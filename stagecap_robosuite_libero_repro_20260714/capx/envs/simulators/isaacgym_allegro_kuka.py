from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv


class IsaacGymAllegroKukaEnv(BaseEnv):
    """IsaacGymEnvs AllegroKuka wrapper for CaP code execution.

    This wrapper intentionally starts as a headless state/action integration.
    Rendering for URDF-backed Isaac Gym tasks is unstable in the current CPU
    PyTorch smoke setup, so `agentview.images` is left empty.
    """

    def __init__(
        self,
        task: str = "AllegroKuka",
        subtask: str = "reorientation",
        num_envs: int = 1,
        seed: int = 0,
        sim_device: str = "cpu:0",
        rl_device: str = "cpu",
        graphics_device_id: int = -1,
        max_steps: int = 300,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        del viser_debug
        os.environ.setdefault(
            "LD_LIBRARY_PATH",
            "/home/lifuchun/miniconda3/envs/isaacgym-preview4/lib",
        )
        self.task = task
        self.subtask = subtask
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.sim_device = sim_device
        self.rl_device = rl_device
        self.graphics_device_id = int(graphics_device_id)
        self.max_steps = int(max_steps)
        self.enable_render = bool(enable_render)
        self.viser_debug = False
        self._step_count = 0
        self._sim_step_count = 0
        self._trajectory_steps: list[dict[str, Any]] = []
        self._frame_buffer: list[np.ndarray] = []

        self._torch = None
        self._env = None
        self._last_obs: dict[str, Any] | None = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_info: dict[str, Any] = {}
        self._last_action: np.ndarray | None = None
        self._create_env()

    @property
    def action_dim(self) -> int:
        assert self._env is not None
        return int(np.prod(self._env.action_space.shape))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            self.close()
            self._create_env()

        assert self._env is not None
        obs = self._env.reset()
        self._last_obs = obs
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._step_count = 0
        self._sim_step_count = 0
        self._trajectory_steps.clear()
        return self.get_observation(), {"task_prompt": self.task_prompt}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.apply_action(action, steps=1)
        truncated = self._sim_step_count >= self.max_steps
        return (
            self.get_observation(),
            float(self._last_reward),
            bool(self._last_done),
            bool(truncated),
            dict(self._last_info),
        )

    def apply_action(self, action: Any, *, steps: int = 1) -> dict[str, Any]:
        assert self._env is not None
        assert self._torch is not None
        command = self._normalize_action(action)
        tensor = self._torch.as_tensor(command.reshape(1, -1), device=self.rl_device, dtype=self._torch.float32)
        for _ in range(max(1, int(steps))):
            state_before = self._trajectory_state()
            obs, reward, done, info = self._env.step(tensor)
            self._last_obs = obs
            self._last_reward = float(reward.detach().cpu().reshape(-1)[0])
            self._last_done = bool(done.detach().cpu().reshape(-1)[0])
            self._last_info = dict(info)
            self._last_action = command.copy()
            self._sim_step_count += 1
            self._step_count += 1
            self._record_trajectory_step(command, state_before=state_before)
            if self._sim_step_count >= self.max_steps:
                break
        return self.get_task_state()

    def get_observation(self) -> dict[str, Any]:
        state = self.get_task_state()
        return {
            "task": {
                "env_name": self.task,
                "subtask": self.subtask,
                "prompt": self.task_prompt,
                **state,
            },
            "robot_joint_pos": state["arm_hand_joint_pos"],
            "robot_joint_vel": state["arm_hand_joint_vel"],
            "hand_joint_pos": state["hand_joint_pos"],
            "hand_joint_vel": state["hand_joint_vel"],
            "object_pose": state["object_pose"],
            "goal_pose": state["goal_pose"],
            "palm_pose": state["palm_pose"],
            "fingertip_poses": state["fingertip_poses"],
            "last_action": None if self._last_action is None else self._last_action.copy(),
            "agentview": {"images": {}},
        }

    def get_task_state(self) -> dict[str, Any]:
        assert self._env is not None
        self._refresh_observations()

        arm_hand_pos = _to_numpy(self._env.arm_hand_dof_pos[0, : self._env.num_hand_arm_dofs])
        arm_hand_vel = _to_numpy(self._env.arm_hand_dof_vel[0, : self._env.num_hand_arm_dofs])
        keypoint_dist = _to_numpy(self._env.keypoint_distances_l2[0])
        fingertip_dist = _to_numpy(self._env.curr_fingertip_distances[0])
        object_pos = _to_numpy(self._env.object_pos[0])
        object_rot = _to_numpy(self._env.object_rot[0])
        goal_pos = _to_numpy(self._env.goal_pos[0])
        goal_rot = _to_numpy(self._env.goal_rot[0])
        palm_pos = _to_numpy(self._env.palm_center_pos[0])
        palm_rot = _to_numpy(self._env._palm_rot[0])
        fingertip_pos = _to_numpy(self._env.fingertip_pos_offset[0])
        fingertip_rot = _to_numpy(self._env.fingertip_rot[0])
        object_linvel = _to_numpy(self._env.object_linvel[0])
        object_angvel = _to_numpy(self._env.object_angvel[0])
        max_keypoint_dist = float(np.max(keypoint_dist))
        lifted_object = bool(float(_to_numpy(self._env.lifted_object[0])) > 0.5)
        success_tolerance = float(self._env.success_tolerance * self._env.keypoint_scale)
        task_success = bool(max_keypoint_dist <= success_tolerance)

        return {
            "arm_joint_pos": arm_hand_pos[:7].copy(),
            "arm_joint_vel": arm_hand_vel[:7].copy(),
            "hand_joint_pos": arm_hand_pos[7:].copy(),
            "hand_joint_vel": arm_hand_vel[7:].copy(),
            "arm_hand_joint_pos": arm_hand_pos.copy(),
            "arm_hand_joint_vel": arm_hand_vel.copy(),
            "object_pose": np.concatenate([object_pos, object_rot]).copy(),
            "object_pos": object_pos.copy(),
            "object_rot_xyzw": object_rot.copy(),
            "object_linvel": object_linvel.copy(),
            "object_angvel": object_angvel.copy(),
            "goal_pose": np.concatenate([goal_pos, goal_rot]).copy(),
            "goal_pos": goal_pos.copy(),
            "goal_rot_xyzw": goal_rot.copy(),
            "palm_pose": np.concatenate([palm_pos, palm_rot]).copy(),
            "palm_pos": palm_pos.copy(),
            "palm_rot_xyzw": palm_rot.copy(),
            "fingertip_poses": np.concatenate([fingertip_pos, fingertip_rot], axis=1).copy(),
            "fingertip_pos": fingertip_pos.copy(),
            "fingertip_distances": fingertip_dist.copy(),
            "keypoint_distances": keypoint_dist.copy(),
            "max_keypoint_dist": max_keypoint_dist,
            "success_tolerance": success_tolerance,
            "lifted_object": lifted_object,
            "last_reward": float(self._last_reward),
            "task_success": task_success,
            "sim_step": int(self._sim_step_count),
        }

    def compute_reward(self) -> float:
        return float(self._last_reward)

    def task_completed(self) -> bool:
        return bool(self.get_task_state()["task_success"])

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        raise RuntimeError(
            "AllegroKuka task rendering is disabled in the current Isaac Gym CPU smoke setup."
        )

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        del enabled
        if clear:
            self._frame_buffer.clear()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_trajectory_steps(self, *, clear: bool = False) -> list[dict[str, Any]]:
        steps = [dict(step) for step in self._trajectory_steps]
        if clear:
            self._trajectory_steps.clear()
        return steps

    def close(self) -> None:
        # Explicit destroy_sim can hang in this Isaac Gym Preview 4 setup; rely
        # on the short-lived process to release native resources.
        self._env = None

    @property
    def task_prompt(self) -> str:
        return (
            "Use a Kuka arm with an Allegro dexterous hand to lift the block and "
            "rotate it so object keypoints align with the goal keypoints."
        )

    def _create_env(self) -> None:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        from isaacgym import gymapi  # noqa: F401
        import isaacgymenvs
        import torch

        cfg_dir = Path(isaacgymenvs.__file__).resolve().parent / "cfg"
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"task={self.task}",
                    "pipeline=cpu",
                    f"sim_device={self.sim_device}",
                    f"rl_device={self.rl_device}",
                    f"graphics_device_id={self.graphics_device_id}",
                    "headless=True",
                    f"num_envs={self.num_envs}",
                ],
            )
            OmegaConf.set_struct(cfg.task.env, False)
            cfg.task.env.numEnvs = self.num_envs
            cfg.task.env.subtask = self.subtask

        self._torch = torch
        self._env = isaacgymenvs.make(
            seed=self.seed,
            task=self.task,
            num_envs=self.num_envs,
            sim_device=self.sim_device,
            rl_device=self.rl_device,
            graphics_device_id=self.graphics_device_id,
            headless=True,
            force_render=False,
            cfg=cfg,
        )
        self._last_obs = self._env.reset()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._refresh_observations()

    def _refresh_observations(self) -> None:
        assert self._env is not None
        self._env.compute_observations()

    def _normalize_action(self, action: Any) -> np.ndarray:
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.size != self.action_dim:
            raise ValueError(f"AllegroKuka action must have {self.action_dim} values, got {command.size}")
        return np.clip(command, -1.0, 1.0).astype(np.float32)

    def _trajectory_state(self) -> list[float]:
        state = self.get_task_state()
        return np.concatenate(
            [
                np.asarray(state["arm_hand_joint_pos"], dtype=np.float64).reshape(-1),
                np.asarray(state["object_pos"], dtype=np.float64).reshape(3),
                np.asarray(state["goal_pos"], dtype=np.float64).reshape(3),
                [float(state["max_keypoint_dist"])],
            ]
        ).astype(float).tolist()

    def _record_trajectory_step(self, action: np.ndarray, *, state_before: list[float]) -> None:
        state_after = self._trajectory_state()
        frame_index = len(self._trajectory_steps)
        self._trajectory_steps.append(
            {
                "observation.state": state_before,
                "next_observation.state": state_after,
                "action": np.asarray(action, dtype=np.float64).astype(float).tolist(),
                "timestamp": time.time(),
                "frame_index": frame_index,
                "sim_step_index": self._sim_step_count,
                "episode_index": 0,
                "task_index": 0,
                "reward": float(self._last_reward),
                "done": bool(self._last_done),
                "task_completed": bool(self.task_completed()),
                "wall_time": time.time(),
            }
        )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()
