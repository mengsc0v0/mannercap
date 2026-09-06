from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from gymnasium import spaces

from capx.envs.base import BaseEnv


FINGER_SEMANTIC_ORDER = {"index": 0, "middle": 1, "ring": 2, "thumb": 3}


class MfrScrewdriverTurningEnv(BaseEnv):
    """MFR_benchmark Victor/Allegro screwdriver-turning wrapper for CaP-X.

    MFR's screwdriver task is an arm + Allegro hand task, but the upstream
    environment starts near a pregrasp. This wrapper makes the phase boundary
    explicit so generated CaP code can first command an approach/pregrasp stage
    and only then attempt the screwdriver-turning stage.
    """

    def __init__(
        self,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cpu",
        arm_type: str = "robot",
        fingers: list[str] | None = None,
        steps_per_action: int = 8,
        max_steps: int = 160,
        control_mode: str = "joint_impedance",
        graphics_device_id: int | None = None,
        gradual_control: bool = True,
        friction_coefficient: float = 1.0,
        gravity: bool = True,
        success_angle_delta: float = 0.12,
        max_screwdriver_top_error: float = 0.035,
        enable_render: bool = False,
        viewer: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        del enable_render, viser_debug
        if int(num_envs) != 1:
            raise ValueError("MFR screwdriver wrapper currently supports num_envs=1")

        os.environ.setdefault(
            "LD_LIBRARY_PATH",
            "/home/lifuchun/miniconda3/envs/isaacgym-preview4/lib",
        )
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.device = device
        self.arm_type = arm_type
        self.fingers = list(fingers or ["index", "middle", "thumb"])
        self.steps_per_action = int(steps_per_action)
        self.max_steps = int(max_steps)
        self.control_mode = control_mode
        self.graphics_device_id = None if graphics_device_id is None else int(graphics_device_id)
        self.gradual_control = bool(gradual_control)
        self.friction_coefficient = float(friction_coefficient)
        self.gravity = bool(gravity)
        self.success_angle_delta = float(success_angle_delta)
        self.max_screwdriver_top_error = float(max_screwdriver_top_error)
        self.viewer = bool(viewer)
        self.viser_debug = False

        self._torch = None
        self._env = None
        self._sim_step_count = 0
        self._trajectory_steps: list[dict[str, Any]] = []
        self._frame_buffer: list[np.ndarray] = []
        self._last_action: np.ndarray | None = None
        self._default_action: np.ndarray | None = None
        self._initial_screwdriver_angle = 0.0
        self._initial_screwdriver_top_pos = np.zeros(3, dtype=np.float64)
        self._phase = "created"

        self._create_env()
        self.action_space = spaces.Box(
            low=-4.0,
            high=4.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({})

    @property
    def action_dim(self) -> int:
        assert self._env is not None
        return int(self._env.robot_dof)

    @property
    def arm_dof(self) -> int:
        assert self._env is not None
        return int(self._env.arm_dof)

    @property
    def task_prompt(self) -> str:
        return (
            "Use the Victor/iiwa-style 7-DoF arm with an Allegro dexterous hand "
            "to approach the table-mounted screwdriver, grasp around the handle, "
            "then turn it while keeping the screwdriver top close to its nominal axis."
        )

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
        else:
            assert self._env is not None
            self._env.reset()
            self._refresh_tensors()

        self._sim_step_count = 0
        self._trajectory_steps.clear()
        self._default_action = self._semantic_default_action()
        self._last_action = self._default_action.copy()
        self._initial_screwdriver_angle = self._read_screwdriver_angle()
        self._initial_screwdriver_top_pos = self._read_screwdriver_top_pos()
        self._phase = "reset_pregrasp"
        return self.get_observation(), {"task_prompt": self.task_prompt}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.apply_action(action, steps=1)
        truncated = self._sim_step_count >= self.max_steps
        return (
            self.get_observation(),
            float(self.compute_reward()),
            bool(self.task_completed()),
            bool(truncated),
            self.get_task_state(),
        )

    def apply_action(
        self,
        action: Any,
        *,
        steps: int = 1,
        phase: str | None = None,
    ) -> dict[str, Any]:
        assert self._env is not None
        assert self._torch is not None
        command = self._normalize_action(action)
        tensor = self._torch.as_tensor(
            command.reshape(1, -1),
            device=self._env.device,
            dtype=self._torch.float32,
        )
        for _ in range(max(1, int(steps))):
            state_before = self._trajectory_state()
            self._env.step(tensor, ignore_img=True)
            self._sim_step_count += 1
            self._last_action = command.copy()
            if phase is not None:
                self._phase = phase
            self._record_trajectory_step(command, state_before=state_before)
            if self._sim_step_count >= self.max_steps:
                break
        return self.get_task_state()

    def get_observation(self) -> dict[str, Any]:
        state = self.get_task_state()
        return {
            "task": {
                "env_name": "MFR_benchmark",
                "subtask": "allegro_screwdriver_turning",
                "prompt": self.task_prompt,
                **state,
            },
            "robot_joint_pos": state["arm_hand_joint_pos"],
            "arm_joint_pos": state["arm_joint_pos"],
            "hand_joint_pos": state["selected_hand_joint_pos"],
            "screwdriver_angle": state["screwdriver_angle"],
            "screwdriver_angle_delta": state["screwdriver_angle_delta"],
            "screwdriver_top_pos": state["screwdriver_top_pos"],
            "fingertip_pos": state["fingertip_pos"],
            "last_action": None if self._last_action is None else self._last_action.copy(),
            "agentview": {"images": {}},
        }

    def get_task_state(self) -> dict[str, Any]:
        assert self._env is not None
        self._refresh_tensors()
        full_q = self._read_full_q()
        state_q = self._read_state_q()
        screwdriver_ori_euler = full_q[-4:-1].copy()
        screwdriver_angle = float(full_q[-1])
        screwdriver_angle_delta = float(screwdriver_angle - self._initial_screwdriver_angle)
        screwdriver_top_pos = self._read_screwdriver_top_pos()
        nominal_top = np.asarray(self._env.nominal_screwdriver_top, dtype=np.float64)
        top_error = float(np.linalg.norm(screwdriver_top_pos - nominal_top))
        valid_axis = bool(top_error <= self.max_screwdriver_top_error)
        fingertip_pos = self._read_fingertip_positions()
        handle_pos = self._read_screwdriver_handle_pos()
        fingertip_distances = np.linalg.norm(fingertip_pos - handle_pos.reshape(1, 3), axis=1)
        min_fingertip_distance = float(np.min(fingertip_distances)) if fingertip_distances.size else float("inf")
        task_success = bool(valid_axis and abs(screwdriver_angle_delta) >= self.success_angle_delta)

        return {
            "arm_joint_pos": full_q[: self.arm_dof].copy(),
            "arm_hand_joint_pos": state_q[: self.action_dim].copy(),
            "selected_hand_joint_pos": state_q[self.arm_dof : self.action_dim].copy(),
            "full_sim_joint_pos": full_q.copy(),
            "default_action": self.get_default_action(),
            "screwdriver_root_pos": np.asarray(self._env.obj_pose, dtype=np.float64).copy(),
            "screwdriver_handle_pos": handle_pos.copy(),
            "screwdriver_top_pos": screwdriver_top_pos.copy(),
            "nominal_screwdriver_top_pos": nominal_top.copy(),
            "screwdriver_top_error": top_error,
            "valid_screwdriver_axis": valid_axis,
            "screwdriver_ori_euler": screwdriver_ori_euler,
            "screwdriver_angle": screwdriver_angle,
            "initial_screwdriver_angle": float(self._initial_screwdriver_angle),
            "screwdriver_angle_delta": screwdriver_angle_delta,
            "fingertip_pos": fingertip_pos.copy(),
            "fingertip_distances_to_handle": fingertip_distances.copy(),
            "min_fingertip_distance_to_handle": min_fingertip_distance,
            "phase": self._phase,
            "sim_step": int(self._sim_step_count),
            "success_angle_delta": float(self.success_angle_delta),
            "max_screwdriver_top_error": float(self.max_screwdriver_top_error),
            "task_success": task_success,
        }

    def get_default_action(self) -> np.ndarray:
        if self._default_action is None:
            self._default_action = self._semantic_default_action()
        return self._default_action.copy()

    def compute_reward(self) -> float:
        state = self.get_task_state()
        turn_score = min(
            abs(float(state["screwdriver_angle_delta"])) / max(self.success_angle_delta, 1e-6),
            1.0,
        )
        axis_score = 1.0 if bool(state["valid_screwdriver_axis"]) else 0.0
        return float(turn_score * axis_score)

    def task_completed(self) -> bool:
        return bool(self.get_task_state()["task_success"])

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        raise RuntimeError(
            "MFR screwdriver rendering is not enabled in the current headless Isaac Gym setup."
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
        # Explicit destroy_sim can segfault in this MFR/Isaac Gym CPU setup.
        self._env = None

    def _create_env(self) -> None:
        import random

        random.seed(self.seed)
        np.random.seed(self.seed)

        from MFR_benchmark.tasks.allegro import AllegroScrewdriverTurningEnv
        import torch

        torch.manual_seed(self.seed)
        self._torch = torch
        self._env = AllegroScrewdriverTurningEnv(
            self.num_envs,
            steps_per_action=self.steps_per_action,
            control_mode=self.control_mode,
            viewer=self.viewer,
            device=self.device,
            friction_coefficient=self.friction_coefficient,
            fingers=self.fingers,
            gravity=self.gravity,
            gradual_control=self.gradual_control,
            arm_type=self.arm_type,
            graphics_device_id=self.graphics_device_id,
        )
        self._refresh_tensors()
        self._default_action = self._semantic_default_action()
        self._last_action = self._default_action.copy()
        self._initial_screwdriver_angle = self._read_screwdriver_angle()
        self._initial_screwdriver_top_pos = self._read_screwdriver_top_pos()
        self._phase = "created_pregrasp"

    def _refresh_tensors(self) -> None:
        assert self._env is not None
        self._env._refresh_tensors()

    def _normalize_action(self, action: Any) -> np.ndarray:
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.size != self.action_dim:
            raise ValueError(f"MFR screwdriver action must have {self.action_dim} values, got {command.size}")
        return np.clip(command, -4.0, 4.0).astype(np.float32)

    def _semantic_default_action(self) -> np.ndarray:
        assert self._env is not None
        default_q = self._env.default_dof_pos[0].detach().cpu().numpy().astype(np.float32)
        parts = [default_q[: self.arm_dof]]
        for finger in self.fingers:
            if finger not in FINGER_SEMANTIC_ORDER:
                raise ValueError(f"Unknown Allegro finger: {finger}")
            offset = self.arm_dof + FINGER_SEMANTIC_ORDER[finger] * 4
            parts.append(default_q[offset : offset + 4])
        return np.concatenate(parts).astype(np.float32)

    def _read_full_q(self) -> np.ndarray:
        assert self._env is not None
        return self._env._q[0].detach().cpu().numpy().astype(np.float64).copy()

    def _read_state_q(self) -> np.ndarray:
        assert self._env is not None
        return self._env.get_state()[0].detach().cpu().numpy().astype(np.float64).copy()

    def _read_screwdriver_angle(self) -> float:
        return float(self._read_full_q()[-1])

    def _read_screwdriver_top_pos(self) -> np.ndarray:
        assert self._env is not None
        assert self._torch is not None
        env_q = self._torch.as_tensor(
            self._read_full_q()[-4:-1],
            device=self._env.device,
            dtype=self._torch.float32,
        )
        return (
            self._env.get_screwdriver_top_in_world(env_q)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        )

    def _read_screwdriver_handle_pos(self) -> np.ndarray:
        assert self._env is not None
        # In screwdriver.urdf the handle/body center is about 15 cm above the
        # root joint and the cap/top is about 20 cm above it.
        return np.asarray(self._env.obj_pose, dtype=np.float64) + np.array([0.0, 0.0, 0.15])

    def _read_fingertip_positions(self) -> np.ndarray:
        assert self._env is not None
        points = []
        for finger in self.fingers:
            rb_index = self._env.finger_ee_index[finger]
            points.append(
                self._env.rb_states[rb_index, :3]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return np.stack(points, axis=0) if points else np.zeros((0, 3), dtype=np.float64)

    def _trajectory_state(self) -> list[float]:
        state = self.get_task_state()
        return np.concatenate(
            [
                np.asarray(state["arm_hand_joint_pos"], dtype=np.float64).reshape(-1),
                np.asarray(state["screwdriver_top_pos"], dtype=np.float64).reshape(3),
                [
                    float(state["screwdriver_angle_delta"]),
                    float(state["screwdriver_top_error"]),
                    float(state["min_fingertip_distance_to_handle"]),
                ],
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
                "reward": float(self.compute_reward()),
                "done": bool(self.task_completed()),
                "phase": self._phase,
                "wall_time": time.time(),
            }
        )
