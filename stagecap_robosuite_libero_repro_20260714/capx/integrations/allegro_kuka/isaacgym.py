from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


ARM_HOME = np.array([0.0, -0.35, 0.0, -1.6, 0.0, 1.2, 0.0], dtype=np.float32)
HAND_OPEN = np.array([0.0, 0.25, 0.35, 0.35] * 4, dtype=np.float32)
HAND_CLOSE = np.array([0.25, 0.8, 0.9, 0.9] * 4, dtype=np.float32)
HAND_PINCH = np.array([0.15, 0.65, 0.75, 0.75] * 4, dtype=np.float32)


class AllegroKukaControlApi(ApiBase):
    """CaP-facing API for IsaacGymEnvs AllegroKuka."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._current_action = np.zeros(23, dtype=np.float32)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_state": self.get_task_state,
            "get_arm_hand_joint_state": self.get_arm_hand_joint_state,
            "get_object_pose": self.get_object_pose,
            "get_goal_pose": self.get_goal_pose,
            "get_palm_pose": self.get_palm_pose,
            "get_fingertip_poses": self.get_fingertip_poses,
            "set_arm_hand_action": self.set_arm_hand_action,
            "set_hand_preshape": self.set_hand_preshape,
            "open_hand": self.open_hand,
            "close_hand_on_object": self.close_hand_on_object,
            "move_palm_to_object": self.move_palm_to_object,
            "align_fingertips_to_object": self.align_fingertips_to_object,
            "lift_object": self.lift_object,
            "rotate_object_to_goal": self.rotate_object_to_goal,
            "satisfy_allegro_kuka_goal_predicates": self.satisfy_allegro_kuka_goal_predicates,
            "verify_object_reorientation": self.verify_object_reorientation,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current AllegroKuka observation dictionary."""
        return self._env.get_observation()

    def get_task_state(self) -> dict[str, Any]:
        """Return privileged state for AllegroKuka reorientation.

        Returns:
            Dictionary with arm/hand joints, object pose, goal pose, palm pose,
            fingertip poses, fingertip distances, keypoint distances,
            `max_keypoint_dist`, `success_tolerance`, and `task_success`.
        """
        return self._env.get_task_state()

    def get_arm_hand_joint_state(self) -> dict[str, np.ndarray]:
        """Return arm and Allegro hand joint positions and velocities."""
        state = self.get_task_state()
        return {
            "arm_qpos": np.asarray(state["arm_joint_pos"], dtype=np.float64).copy(),
            "arm_qvel": np.asarray(state["arm_joint_vel"], dtype=np.float64).copy(),
            "hand_qpos": np.asarray(state["hand_joint_pos"], dtype=np.float64).copy(),
            "hand_qvel": np.asarray(state["hand_joint_vel"], dtype=np.float64).copy(),
            "arm_hand_qpos": np.asarray(state["arm_hand_joint_pos"], dtype=np.float64).copy(),
            "arm_hand_qvel": np.asarray(state["arm_hand_joint_vel"], dtype=np.float64).copy(),
        }

    def get_object_pose(self) -> np.ndarray:
        """Return object pose as `[x, y, z, qx, qy, qz, qw]`."""
        return np.asarray(self.get_task_state()["object_pose"], dtype=np.float64).copy()

    def get_goal_pose(self) -> np.ndarray:
        """Return goal pose as `[x, y, z, qx, qy, qz, qw]`."""
        return np.asarray(self.get_task_state()["goal_pose"], dtype=np.float64).copy()

    def get_palm_pose(self) -> np.ndarray:
        """Return palm pose as `[x, y, z, qx, qy, qz, qw]`."""
        return np.asarray(self.get_task_state()["palm_pose"], dtype=np.float64).copy()

    def get_fingertip_poses(self) -> np.ndarray:
        """Return four fingertip poses as rows `[x, y, z, qx, qy, qz, qw]`."""
        return np.asarray(self.get_task_state()["fingertip_poses"], dtype=np.float64).copy()

    def set_arm_hand_action(
        self,
        action: list[float] | tuple[float, ...] | np.ndarray,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Apply a normalized 23-D AllegroKuka command.

        Args:
            action: 23 values in [-1, 1]. Layout is 7 Kuka arm controls followed
                by 16 Allegro finger controls.
            steps: Number of IsaacGymEnvs steps to repeat the command.

        Returns:
            Compact task state after execution.
        """
        self._current_action = np.clip(np.asarray(action, dtype=np.float32).reshape(23), -1.0, 1.0)
        return self._env.apply_action(self._current_action, steps=steps)

    def set_hand_preshape(self, name: str = "open", steps: int = 3) -> dict[str, Any]:
        """Set a reusable Allegro hand preshape while holding the current arm action.

        Args:
            name: One of `open`, `close`, or `pinch`.
            steps: Number of environment steps to hold the command.

        Returns:
            Compact task state after execution.
        """
        action = self._current_action.copy()
        action[:7] = 0.0
        action[7:] = _preshape(name)
        return self.set_arm_hand_action(action, steps=steps)

    def open_hand(self, steps: int = 3) -> dict[str, Any]:
        """Open the Allegro hand to reduce contact or prepare a regrasp."""
        return self.set_hand_preshape("open", steps=steps)

    def close_hand_on_object(self, steps: int = 4, squeeze: float = 0.8) -> dict[str, Any]:
        """Close the Allegro hand around the object.

        Args:
            steps: Number of environment steps.
            squeeze: Scale for the close preshape in [0, 1].

        Returns:
            Compact task state after the close command.
        """
        action = self._current_action.copy()
        action[:7] = 0.0
        action[7:] = np.clip(HAND_CLOSE * float(squeeze), -1.0, 1.0)
        return self.set_arm_hand_action(action, steps=steps)

    def move_palm_to_object(self, steps: int = 6, strength: float = 0.35) -> dict[str, Any]:
        """Move the Kuka arm command heuristically toward the object.

        This is a privileged CaP primitive. It reads palm/object state and maps
        the Cartesian error to a small 7-D arm action. It is meant as a smoke
        integration primitive, not a solved IK controller.

        Args:
            steps: Number of environment steps.
            strength: Normalized action magnitude.

        Returns:
            Compact task state after motion.
        """
        state = self.get_task_state()
        delta = np.asarray(state["object_pos"]) - np.asarray(state["palm_pos"])
        arm = np.zeros(7, dtype=np.float32)
        arm[:3] = np.clip(delta[:3] * float(strength) * 4.0, -0.5, 0.5)
        action = self._current_action.copy()
        action[:7] = arm
        return self.set_arm_hand_action(action, steps=steps)

    def align_fingertips_to_object(self, steps: int = 4) -> dict[str, Any]:
        """Use a pinch preshape to bring fingertips closer to the object."""
        action = self._current_action.copy()
        action[:7] *= 0.5
        action[7:] = HAND_PINCH
        return self.set_arm_hand_action(action, steps=steps)

    def lift_object(self, steps: int = 6, lift_strength: float = 0.35) -> dict[str, Any]:
        """Command a gentle upward/lifting arm motion while maintaining grasp."""
        action = self._current_action.copy()
        action[:7] = np.array([0.0, -lift_strength, 0.0, -lift_strength, 0.0, lift_strength, 0.0], dtype=np.float32)
        action[7:] = np.maximum(action[7:], HAND_CLOSE * 0.65)
        return self.set_arm_hand_action(action, steps=steps)

    def rotate_object_to_goal(self, steps: int = 8, direction: str = "auto", strength: float = 0.45) -> dict[str, Any]:
        """Apply a heuristic wrist/finger rotation command toward the goal.

        Args:
            steps: Number of environment steps.
            direction: `auto`, `clockwise`, or `counterclockwise`.
            strength: Normalized rotation command strength.

        Returns:
            Compact task state after rotation attempt.
        """
        state = self.get_task_state()
        sign = _direction_sign(direction, state)
        action = self._current_action.copy()
        action[:7] = np.array([0.0, 0.0, sign * strength, 0.0, -sign * strength, 0.0, sign * strength], dtype=np.float32)
        finger_wave = np.array([0.15, 0.75, 0.55, 0.45] * 4, dtype=np.float32)
        action[7:] = np.clip(finger_wave * sign, -1.0, 1.0)
        return self.set_arm_hand_action(action, steps=steps)

    def satisfy_allegro_kuka_goal_predicates(self, max_rounds: int = 2) -> dict[str, Any]:
        """Run the current heuristic CaP sequence for object reorientation.

        Args:
            max_rounds: Number of approach-grasp-lift-rotate repair rounds.

        Returns:
            Compact final task state.
        """
        self.open_hand(steps=2)
        for _ in range(max(1, int(max_rounds))):
            self.move_palm_to_object(steps=4)
            self.align_fingertips_to_object(steps=3)
            self.close_hand_on_object(steps=4)
            self.lift_object(steps=4)
            self.rotate_object_to_goal(steps=5)
            if self.verify_object_reorientation():
                break
        return self.get_task_state()

    def verify_object_reorientation(self, threshold: float | None = None) -> bool:
        """Check whether object keypoints are within the goal tolerance."""
        state = self.get_task_state()
        tol = float(state["success_tolerance"] if threshold is None else threshold)
        return bool(float(state["max_keypoint_dist"]) <= tol)


def _preshape(name: str) -> np.ndarray:
    normalized = name.lower().strip()
    if normalized == "open":
        return HAND_OPEN.copy()
    if normalized == "close":
        return HAND_CLOSE.copy()
    if normalized == "pinch":
        return HAND_PINCH.copy()
    raise ValueError(f"Unknown Allegro preshape: {name}")


def _direction_sign(direction: str, state: dict[str, Any]) -> float:
    normalized = direction.lower().strip()
    if normalized == "clockwise":
        return -1.0
    if normalized == "counterclockwise":
        return 1.0
    if normalized != "auto":
        raise ValueError(f"Unknown direction: {direction}")
    goal = np.asarray(state["goal_rot_xyzw"], dtype=np.float64)
    obj = np.asarray(state["object_rot_xyzw"], dtype=np.float64)
    return 1.0 if float(np.dot(goal, obj)) >= 0 else -1.0
