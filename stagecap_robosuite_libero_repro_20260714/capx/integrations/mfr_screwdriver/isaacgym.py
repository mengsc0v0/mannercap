from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


ARM_STANDOFF_DELTA = np.array([0.04, -0.03, -0.03, 0.05, 0.02, -0.04, 0.0], dtype=np.float32)
TURN_ARM_DELTA = np.array([0.0, 0.0, 0.12, 0.0, -0.10, 0.0, 0.18], dtype=np.float32)
TURN_FINGER_DELTA = np.array([0.02, -0.08, 0.10, -0.06], dtype=np.float32)
GRASP_FINGER_DELTA = {
    "index": np.array([0.06, 0.20, 0.18, 0.04], dtype=np.float32),
    "middle": np.array([0.04, 0.18, 0.14, 0.02], dtype=np.float32),
    "ring": np.array([0.04, 0.16, 0.12, 0.02], dtype=np.float32),
    "thumb": np.array([0.00, 0.22, 0.18, 0.00], dtype=np.float32),
}


class MfrScrewdriverControlApi(ApiBase):
    """CaP-facing API for MFR_benchmark arm + Allegro screwdriver turning."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._current_action = self._env.get_default_action()

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_state": self.get_task_state,
            "get_arm_hand_joint_state": self.get_arm_hand_joint_state,
            "get_screwdriver_pose": self.get_screwdriver_pose,
            "get_fingertip_positions": self.get_fingertip_positions,
            "get_default_screwdriver_action": self.get_default_screwdriver_action,
            "set_arm_hand_target": self.set_arm_hand_target,
            "hold_measured_pose": self.hold_measured_pose,
            "hold_current_pose": self.hold_current_pose,
            "move_to_standoff_pose": self.move_to_standoff_pose,
            "open_hand": self.open_hand,
            "move_hand_to_screwdriver_pose": self.move_hand_to_screwdriver_pose,
            "align_fingertips_to_screwdriver": self.align_fingertips_to_screwdriver,
            "grasp_screwdriver": self.grasp_screwdriver,
            "move_tool_tip_to_screw_slot": self.move_tool_tip_to_screw_slot,
            "insert_tool_tip": self.insert_tool_tip,
            "turn_screwdriver": self.turn_screwdriver,
            "axis_preserving_finger_pronation": self.axis_preserving_finger_pronation,
            "verify_screwdriver_turning": self.verify_screwdriver_turning,
            "satisfy_mfr_screwdriver_goal_predicates": self.satisfy_mfr_screwdriver_goal_predicates,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current MFR screwdriver observation dictionary."""
        return self._env.get_observation()

    def get_task_state(self) -> dict[str, Any]:
        """Return privileged numeric state for screwdriver turning.

        Returns:
            Dictionary with arm/hand joints, screwdriver root/handle/top
            positions, screwdriver angle and angle delta, fingertip-to-handle
            distances, axis validity, phase label, and task_success.
        """
        return self._env.get_task_state()

    def get_arm_hand_joint_state(self) -> dict[str, np.ndarray]:
        """Return arm and selected Allegro finger joint positions."""
        state = self.get_task_state()
        return {
            "arm_qpos": np.asarray(state["arm_joint_pos"], dtype=np.float64).copy(),
            "selected_hand_qpos": np.asarray(state["selected_hand_joint_pos"], dtype=np.float64).copy(),
            "arm_hand_qpos": np.asarray(state["arm_hand_joint_pos"], dtype=np.float64).copy(),
            "full_sim_qpos": np.asarray(state["full_sim_joint_pos"], dtype=np.float64).copy(),
        }

    def get_screwdriver_pose(self) -> dict[str, Any]:
        """Return screwdriver root, handle, top, orientation, and turn angle."""
        state = self.get_task_state()
        return {
            "root_pos": np.asarray(state["screwdriver_root_pos"], dtype=np.float64).copy(),
            "handle_pos": np.asarray(state["screwdriver_handle_pos"], dtype=np.float64).copy(),
            "top_pos": np.asarray(state["screwdriver_top_pos"], dtype=np.float64).copy(),
            "nominal_top_pos": np.asarray(state["nominal_screwdriver_top_pos"], dtype=np.float64).copy(),
            "ori_euler": np.asarray(state["screwdriver_ori_euler"], dtype=np.float64).copy(),
            "angle": float(state["screwdriver_angle"]),
            "angle_delta": float(state["screwdriver_angle_delta"]),
            "top_error": float(state["screwdriver_top_error"]),
        }

    def get_fingertip_positions(self) -> np.ndarray:
        """Return selected Allegro fingertip positions as rows `[x, y, z]`."""
        return np.asarray(self.get_task_state()["fingertip_pos"], dtype=np.float64).copy()

    def get_default_screwdriver_action(self) -> np.ndarray:
        """Return the upstream MFR pregrasp action target for this task."""
        return self._env.get_default_action()

    def set_arm_hand_target(
        self,
        action: list[float] | tuple[float, ...] | np.ndarray,
        steps: int = 1,
        phase: str = "custom_target",
    ) -> dict[str, Any]:
        """Apply an absolute arm+selected-finger joint target.

        Args:
            action: Absolute target with layout `[7 arm joints, selected finger
                joints]`. The current screwdriver task uses 19 values:
                7 arm values plus 3 selected Allegro fingers x 4 joints.
            steps: Number of MFR high-level action steps to repeat.
            phase: Label stored in trajectory/state for feedback diagnosis.

        Returns:
            Compact task state after execution.
        """
        self._current_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        return self._env.apply_action(self._current_action, steps=steps, phase=phase)

    def hold_current_pose(self, steps: int = 2) -> dict[str, Any]:
        """Hold the current arm+hand target for a few steps."""
        return self.set_arm_hand_target(self._current_action, steps=steps, phase="hold")

    def hold_measured_pose(self, steps: int = 2) -> dict[str, Any]:
        """Hold the measured arm+hand joint positions from the current simulator state.

        This is useful when the pregrasp has already settled into contact and
        moving back to the upstream nominal target would disturb the screwdriver
        axis. ViFailback should prefer this primitive when diagnosis says the
        turn is strong enough but the screwdriver top drifts laterally.
        """
        state = self.get_task_state()
        action = np.asarray(state["arm_hand_joint_pos"], dtype=np.float32).copy()
        return self.set_arm_hand_target(action, steps=steps, phase="hold_measured_pose")

    def move_to_standoff_pose(self, steps: int = 3, distance_scale: float = 1.0) -> dict[str, Any]:
        """Move slightly away from the screwdriver before the approach phase.

        Args:
            steps: Number of action steps.
            distance_scale: Multiplier for the small joint-space standoff.

        Returns:
            Compact task state after the standoff command.
        """
        action = self._env.get_default_action()
        action[: self._env.arm_dof] += ARM_STANDOFF_DELTA * float(distance_scale)
        action[self._env.arm_dof :] = _open_selected_fingers(action[self._env.arm_dof :])
        return self.set_arm_hand_target(action, steps=steps, phase="approach_standoff")

    def open_hand(self, steps: int = 2) -> dict[str, Any]:
        """Open selected Allegro fingers while holding the current arm target."""
        action = self._current_action.copy()
        action[self._env.arm_dof :] = _open_selected_fingers(action[self._env.arm_dof :])
        return self.set_arm_hand_target(action, steps=steps, phase="open_hand")

    def move_hand_to_screwdriver_pose(self, steps: int = 4) -> dict[str, Any]:
        """Move the arm from standoff/pregrasp toward the screwdriver handle.

        MFR's upstream task starts close to a pregrasp. This primitive exposes
        that approach/pregrasp target explicitly so ViFailback can distinguish
        approach failures from turning failures.
        """
        action = self._env.get_default_action()
        action[self._env.arm_dof :] = _open_selected_fingers(action[self._env.arm_dof :])
        return self.set_arm_hand_target(action, steps=steps, phase="approach_to_screwdriver")

    def align_fingertips_to_screwdriver(self, steps: int = 3) -> dict[str, Any]:
        """Move selected fingertips into a light pinch around the screwdriver handle."""
        action = self._env.get_default_action()
        action[self._env.arm_dof :] = _add_per_finger_delta(
            action[self._env.arm_dof :],
            self._env.fingers,
            scale=0.45,
        )
        return self.set_arm_hand_target(action, steps=steps, phase="align_fingertips")

    def grasp_screwdriver(self, steps: int = 4, squeeze: float = 1.0) -> dict[str, Any]:
        """Close the Allegro fingers around the screwdriver handle.

        Args:
            steps: Number of action steps.
            squeeze: Grasp strength multiplier in joint-target space.

        Returns:
            Compact task state after grasping.
        """
        action = self._env.get_default_action()
        action[self._env.arm_dof :] = _add_per_finger_delta(
            action[self._env.arm_dof :],
            self._env.fingers,
            scale=float(squeeze),
        )
        return self.set_arm_hand_target(action, steps=steps, phase="grasp_screwdriver")

    def move_tool_tip_to_screw_slot(self, steps: int = 2) -> dict[str, Any]:
        """Hold the grasp and settle the tool tip on the screwdriver/screw axis."""
        action = self._current_action.copy()
        return self.set_arm_hand_target(action, steps=steps, phase="tip_to_slot")

    def insert_tool_tip(self, steps: int = 2) -> dict[str, Any]:
        """Apply a small settling hold for the insertion/contact phase."""
        action = self._current_action.copy()
        return self.set_arm_hand_target(action, steps=steps, phase="insert_tip")

    def turn_screwdriver(
        self,
        direction: str = "clockwise",
        steps: int = 6,
        strength: float = 1.0,
    ) -> dict[str, Any]:
        """Turn the screwdriver after approach and grasp.

        Args:
            direction: `clockwise` or `counterclockwise`.
            steps: Number of action steps.
            strength: Multiplier for the arm wrist and finger rolling deltas.

        Returns:
            Compact task state after turning.
        """
        sign = _direction_sign(direction)
        action = self._current_action.copy()
        action[: self._env.arm_dof] += TURN_ARM_DELTA * sign * float(strength)
        for finger_index, _finger in enumerate(self._env.fingers):
            start = self._env.arm_dof + finger_index * 4
            action[start : start + 4] += TURN_FINGER_DELTA * sign * float(strength)
        return self.set_arm_hand_target(action, steps=steps, phase="turn_screwdriver")

    def axis_preserving_finger_pronation(
        self,
        steps: int = 1,
        joint_delta: float = 0.35,
    ) -> dict[str, Any]:
        """Turn the screwdriver with a short axis-preserving Allegro finger motion.

        This primitive is the repaired CAP action found by the ViFailback-style
        search: arm/wrist turning moved the screwdriver top off axis, while a
        synchronized positive delta on the first joint of each selected Allegro
        finger produced enough rotation with much smaller top drift.

        Args:
            steps: Number of high-level MFR action steps. Use `steps=1` with
                `steps_per_action=2` for the validated strategy.
            joint_delta: Positive delta added to joint 0 of each selected
                finger in the upstream pregrasp target.

        Returns:
            Compact task state after the guarded finger-pronation command.
        """
        action = self._env.get_default_action()
        for finger_index, _finger in enumerate(self._env.fingers):
            start = self._env.arm_dof + finger_index * 4
            action[start] += float(joint_delta)
        return self.set_arm_hand_target(
            action,
            steps=steps,
            phase="axis_preserving_finger_pronation",
        )

    def verify_screwdriver_turning(
        self,
        min_angle_delta: float | None = None,
        max_top_error: float | None = None,
    ) -> bool:
        """Check whether the screwdriver turned while staying near its axis."""
        state = self.get_task_state()
        angle_threshold = float(
            state["success_angle_delta"] if min_angle_delta is None else min_angle_delta
        )
        top_threshold = float(
            state["max_screwdriver_top_error"] if max_top_error is None else max_top_error
        )
        return bool(
            abs(float(state["screwdriver_angle_delta"])) >= angle_threshold
            and float(state["screwdriver_top_error"]) <= top_threshold
        )

    def satisfy_mfr_screwdriver_goal_predicates(self, max_turn_rounds: int = 3) -> dict[str, Any]:
        """Run the default two-phase CaP sequence for screwdriver turning.

        Phase 1 approaches and grasps the screwdriver. Phase 2 applies repeated
        turning motions and verifies angle change plus axis validity.
        """
        self.move_to_standoff_pose(steps=2)
        self.open_hand(steps=1)
        self.move_hand_to_screwdriver_pose(steps=3)
        self.align_fingertips_to_screwdriver(steps=2)
        self.grasp_screwdriver(steps=3)
        self.move_tool_tip_to_screw_slot(steps=1)
        self.insert_tool_tip(steps=1)
        self.axis_preserving_finger_pronation(steps=1, joint_delta=0.35)
        if self.verify_screwdriver_turning():
            return self.get_task_state()
        for _ in range(max(1, int(max_turn_rounds))):
            self.turn_screwdriver(direction="clockwise", steps=4, strength=0.8)
            if self.verify_screwdriver_turning():
                break
        return self.get_task_state()


def _open_selected_fingers(finger_action: np.ndarray) -> np.ndarray:
    action = np.asarray(finger_action, dtype=np.float32).copy()
    for start in range(0, action.size, 4):
        action[start : start + 4] -= np.array([0.04, 0.18, 0.18, 0.04], dtype=np.float32)
    return np.clip(action, -2.2, 2.2)


def _add_per_finger_delta(
    finger_action: np.ndarray,
    fingers: list[str],
    *,
    scale: float,
) -> np.ndarray:
    action = np.asarray(finger_action, dtype=np.float32).copy()
    for finger_index, finger in enumerate(fingers):
        start = finger_index * 4
        action[start : start + 4] += GRASP_FINGER_DELTA.get(
            finger,
            np.array([0.04, 0.16, 0.12, 0.02], dtype=np.float32),
        ) * float(scale)
    return np.clip(action, -2.2, 2.2)


def _direction_sign(direction: str) -> float:
    normalized = direction.lower().strip()
    if normalized in {"clockwise", "cw"}:
        return -1.0
    if normalized in {"counterclockwise", "ccw", "anticlockwise"}:
        return 1.0
    raise ValueError(f"Unknown screwdriver turn direction: {direction}")
