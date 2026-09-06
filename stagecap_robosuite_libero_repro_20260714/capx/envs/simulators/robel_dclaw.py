from __future__ import annotations

import importlib
import os
import time
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv


TASK_CLASS_PATHS: dict[str, str] = {
    "DClawTurnFixed-v0": "robel.dclaw.turn:DClawTurnFixed",
    "DClawTurnRandom-v0": "robel.dclaw.turn:DClawTurnRandom",
    "DClawScrewFixed-v0": "robel.dclaw.screw:DClawScrewFixed",
    "DClawScrewRandom-v0": "robel.dclaw.screw:DClawScrewRandom",
}


class RobelDClawEnv(BaseEnv):
    """Low-level ROBEL D'Claw wrapper for CaP code execution."""

    def __init__(
        self,
        env_id: str = "DClawScrewFixed-v0",
        class_path: str | None = None,
        privileged: bool = False,
        max_steps: int = 400,
        seed: int | None = None,
        enable_render: bool = True,
        render_width: int = 480,
        render_height: int = 480,
        camera_id: int = -1,
        sim_backend: str = "dm_control",
        record_subsample_rate: int = 2,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        self.env_id = env_id
        self.class_path = class_path or TASK_CLASS_PATHS[env_id]
        self.privileged = privileged
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self._render_width = render_width
        self._render_height = render_height
        self._camera_id = camera_id
        self._record_subsample_rate = max(1, int(record_subsample_rate))
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._trajectory_steps: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)
        self._sim_step_count = 0
        self._step_count = 0

        env_cls = _load_class(self.class_path)
        sim_backend_obj = _load_sim_backend(sim_backend)
        self.handle = env_cls(sim_backend=sim_backend_obj)
        if seed is not None and hasattr(self.handle, "seed"):
            self.handle.seed(seed)

        self._current_obs_array: np.ndarray | None = None
        self._current_obs_dict: dict[str, Any] = {}
        self._current_reward = 0.0
        self._current_done = False
        self._current_info: dict[str, Any] = {}
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.reset(seed=seed)

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.handle.action_space.shape))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
            if hasattr(self.handle, "seed"):
                self.handle.seed(seed)

        obs_array = self.handle.reset()
        self._current_obs_array = np.asarray(obs_array)
        self._current_obs_dict = _array_dict(self.handle.get_obs_dict())
        self._current_reward = 0.0
        self._current_done = False
        self._current_info = self._build_info()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._sim_step_count = 0
        self._step_count = 0
        self._trajectory_steps.clear()
        if self._record_frames:
            self._record_frame()
        return self.get_observation(), {"task_prompt": self.task_prompt}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs_array, reward, done, info = self.apply_action(action, steps=1)
        del obs_array
        truncated = self._sim_step_count >= self.max_steps
        return self.get_observation(), float(reward), bool(done), bool(truncated), dict(info)

    def apply_action(self, action: Any, *, steps: int = 1) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply a normalized 9-DoF D'Claw command for one or more ROBEL steps."""
        command = self._normalize_action(action)
        last_obs = self._current_obs_array
        last_reward = self._current_reward
        last_done = self._current_done
        last_info = self._current_info

        for _ in range(max(1, int(steps))):
            state_before = self._trajectory_state()
            obs_array, reward, done, info = self.handle.step(command)
            self._current_obs_array = np.asarray(obs_array)
            self._current_obs_dict = _array_dict(self.handle.get_obs_dict())
            self._current_reward = float(reward)
            self._current_done = bool(done)
            self._current_info = dict(info)
            self._last_action = command.copy()
            self._sim_step_count += 1
            self._step_count += 1
            self._record_trajectory_step(command, state_before=state_before)
            if self._record_frames and self._sim_step_count % self._record_subsample_rate == 0:
                self._record_frame()
            last_obs = self._current_obs_array
            last_reward = self._current_reward
            last_done = self._current_done
            last_info = self._current_info
            if self._sim_step_count >= self.max_steps:
                break

        assert last_obs is not None
        return last_obs, float(last_reward), bool(last_done), dict(last_info)

    def get_observation(self) -> dict[str, Any]:
        state = self.get_task_state()
        obs: dict[str, Any] = {
            "task": {
                "env_id": self.env_id,
                "prompt": self.task_prompt,
                **state,
            },
            "hand_joint_pos": state["hand_joint_pos"],
            "hand_joint_vel": state["hand_joint_vel"],
            "robot_joint_pos": state["hand_joint_pos"],
            "robot_joint_vel": state["hand_joint_vel"],
            "object_angle": state["object_angle"],
            "object_velocity": state["object_velocity"],
            "target_angle": state["target_angle"],
            "target_velocity": state["target_velocity"],
            "target_error": state["target_error"],
            "last_action": self._last_action.copy(),
            "agentview": {"images": {}},
        }
        if self.enable_render:
            obs["agentview"]["images"]["rgb"] = self.render()
        return obs

    def get_task_state(self) -> dict[str, Any]:
        obs_dict = self._current_obs_dict or _array_dict(self.handle.get_obs_dict())
        object_x = _scalar(obs_dict.get("object_x", 1.0))
        object_y = _scalar(obs_dict.get("object_y", 0.0))
        object_angle = float(np.arctan2(object_y, object_x))
        target_error = _scalar(obs_dict.get("target_error", 0.0))
        target_angle = float(getattr(self.handle, "_target_object_pos", object_angle + target_error))
        target_velocity = float(getattr(self.handle, "_target_object_vel", 0.0))
        hand_joint_pos = np.asarray(obs_dict.get("claw_qpos", np.zeros(9)), dtype=np.float64)
        hand_joint_vel = np.asarray(obs_dict.get("claw_qvel", np.zeros(9)), dtype=np.float64)
        object_velocity = _scalar(obs_dict.get("object_qvel", 0.0))
        abs_target_error = abs(target_error)
        score_success = bool(self._current_info.get("score/success", abs_target_error < 0.2))
        return {
            "hand_joint_pos": hand_joint_pos.copy(),
            "hand_joint_vel": hand_joint_vel.copy(),
            "object_angle": object_angle,
            "object_velocity": object_velocity,
            "target_angle": target_angle,
            "target_velocity": target_velocity,
            "target_error": target_error,
            "abs_target_error": abs_target_error,
            "last_reward": float(self._current_reward),
            "score_success": score_success,
            "sim_step": int(self._sim_step_count),
        }

    def get_flat_observation(self) -> np.ndarray:
        """Return the flattened ROBEL observation used by bundled policies."""
        if self._current_obs_array is not None:
            return np.asarray(self._current_obs_array, dtype=np.float32).copy()
        return np.asarray(self.handle._get_obs(self.handle.get_obs_dict()), dtype=np.float32).copy()

    def compute_reward(self) -> float:
        return float(self._current_reward)

    def task_completed(self) -> bool:
        if self._current_info:
            return bool(self._current_info.get("score/success", False))
        score_dict = getattr(self.handle, "last_score_dict", None)
        if score_dict is not None and "success" in score_dict:
            return bool(score_dict["success"])
        return self.get_task_state()["abs_target_error"] < 0.2

    def get_current_time_s(self) -> float:
        dt = float(getattr(self.handle, "dt", 0.0))
        return float(self._sim_step_count * dt)

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.handle.render(
            mode="rgb_array",
            width=self._render_width,
            height=self._render_height,
            camera_id=self._camera_id,
        )
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        if hasattr(self.handle, "close"):
            self.handle.close()

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = bool(enabled)
        if clear:
            self._frame_buffer.clear()
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

    def get_trajectory_steps(self, *, clear: bool = False) -> list[dict[str, Any]]:
        steps = [dict(step) for step in self._trajectory_steps]
        if clear:
            self._trajectory_steps.clear()
        return steps

    def clear_trajectory_steps(self) -> None:
        self._trajectory_steps.clear()

    def checkpoint_state(self) -> dict[str, Any]:
        """Capture simulator and wrapper state for short rollout search."""
        state = self.handle.get_state()
        return {
            "handle_state": {key: np.asarray(value).copy() for key, value in state.items()},
            "target_object_pos": np.asarray(getattr(self.handle, "_target_object_pos", 0.0)).copy(),
            "target_object_vel": float(getattr(self.handle, "_target_object_vel", 0.0)),
            "desired_target_pos": np.asarray(getattr(self.handle, "_desired_target_pos", 0.0)).copy(),
            "handle_last_action": (
                None if self.handle.last_action is None else np.asarray(self.handle.last_action).copy()
            ),
            "handle_last_obs_dict": _array_dict(self.handle.last_obs_dict or {}),
            "handle_last_reward_dict": dict(self.handle.last_reward_dict or {}),
            "handle_last_score_dict": dict(self.handle.last_score_dict or {}),
            "handle_is_done": bool(self.handle.is_done),
            "handle_step_count": int(self.handle.step_count),
            "current_obs_array": (
                None if self._current_obs_array is None else np.asarray(self._current_obs_array).copy()
            ),
            "current_obs_dict": _array_dict(self._current_obs_dict),
            "current_reward": float(self._current_reward),
            "current_done": bool(self._current_done),
            "current_info": dict(self._current_info),
            "last_action": self._last_action.copy(),
            "sim_step_count": int(self._sim_step_count),
            "step_count": int(self._step_count),
            "frame_buffer_len": len(self._frame_buffer),
            "trajectory_len": len(self._trajectory_steps),
        }

    def restore_state(self, checkpoint: dict[str, Any], *, restore_buffers: bool = True) -> None:
        """Restore a checkpoint made by `checkpoint_state`."""
        self.handle.set_state(checkpoint["handle_state"])
        if hasattr(self.handle, "_set_target_object_pos"):
            self.handle._set_target_object_pos(
                float(np.asarray(checkpoint["target_object_pos"]).reshape(-1)[0]),
                unbounded=True,
            )
        if hasattr(self.handle, "_target_object_vel"):
            self.handle._target_object_vel = float(checkpoint["target_object_vel"])
        if hasattr(self.handle, "_desired_target_pos"):
            self.handle._desired_target_pos = np.asarray(
                checkpoint["desired_target_pos"],
                dtype=np.float32,
            ).copy()

        self.handle.last_action = (
            None
            if checkpoint["handle_last_action"] is None
            else np.asarray(checkpoint["handle_last_action"]).copy()
        )
        self.handle.last_obs_dict = _array_dict(checkpoint["handle_last_obs_dict"])
        self.handle.last_reward_dict = dict(checkpoint["handle_last_reward_dict"])
        self.handle.last_score_dict = dict(checkpoint["handle_last_score_dict"])
        self.handle.is_done = bool(checkpoint["handle_is_done"])
        self.handle.step_count = int(checkpoint["handle_step_count"])

        self._current_obs_array = (
            None
            if checkpoint["current_obs_array"] is None
            else np.asarray(checkpoint["current_obs_array"]).copy()
        )
        self._current_obs_dict = _array_dict(checkpoint["current_obs_dict"])
        self._current_reward = float(checkpoint["current_reward"])
        self._current_done = bool(checkpoint["current_done"])
        self._current_info = dict(checkpoint["current_info"])
        self._last_action = np.asarray(checkpoint["last_action"], dtype=np.float32).copy()
        self._sim_step_count = int(checkpoint["sim_step_count"])
        self._step_count = int(checkpoint["step_count"])

        if restore_buffers:
            del self._frame_buffer[int(checkpoint["frame_buffer_len"]) :]
            del self._trajectory_steps[int(checkpoint["trajectory_len"]) :]

    @property
    def task_prompt(self) -> str:
        if "Screw" in self.env_id:
            return (
                "Continuously rotate the D'Claw screw object so its angle tracks "
                "the moving target with small target_error."
            )
        return "Turn the D'Claw object until target_error is close to zero."

    def _normalize_action(self, action: Any) -> np.ndarray:
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.size != self.action_dim:
            raise ValueError(f"DClaw action must have {self.action_dim} values, got {command.size}")
        return np.clip(command, -1.0, 1.0).astype(np.float32)

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        obs_dict = self._current_obs_dict or {}
        for key, value in obs_dict.items():
            info[f"obs/{key}"] = value
        state = self.get_task_state()
        info["reward/total"] = float(self._current_reward)
        info["score/success"] = bool(state["abs_target_error"] < 0.2)
        return info

    def _record_frame(self) -> None:
        self._frame_buffer.append(self.render())

    def _trajectory_state(self) -> list[float]:
        state = self.get_task_state()
        return np.concatenate(
            [
                np.asarray(state["hand_joint_pos"], dtype=np.float64).reshape(9),
                [float(state["object_angle"])],
                [float(state["target_angle"])],
                [float(state["target_error"])],
            ]
        ).astype(float).tolist()

    def _record_trajectory_step(
        self,
        action: np.ndarray,
        *,
        state_before: list[float],
    ) -> None:
        state_after = self._trajectory_state()
        frame_index = len(self._trajectory_steps)
        self._trajectory_steps.append(
            {
                "observation.state": state_before,
                "next_observation.state": state_after,
                "action": np.asarray(action, dtype=np.float64).astype(float).tolist(),
                "timestamp": self.get_current_time_s(),
                "frame_index": frame_index,
                "sim_step_index": self._sim_step_count,
                "episode_index": 0,
                "task_index": 0,
                "reward": float(self._current_reward),
                "done": bool(self._current_done),
                "task_completed": bool(self.task_completed()),
                "wall_time": time.time(),
            }
        )


def _load_class(class_path: str) -> type[Any]:
    module_name, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_sim_backend(sim_backend: str) -> Any:
    from robel.simulation.sim_scene import SimBackend

    normalized = sim_backend.lower()
    if normalized in {"dm_control", "dm-control", "dm"}:
        return SimBackend.DM_CONTROL
    if normalized in {"mujoco_py", "mujoco-py", "mjpy"}:
        return SimBackend.MUJOCO_PY
    raise ValueError(f"Unsupported ROBEL sim backend: {sim_backend}")


def _array_dict(obs_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value).copy() for key, value in obs_dict.items()}


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
