from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class KukaAllegroNutBoltControlApi(ApiBase):
    """CaP-facing API for the Kuka+Allegro nut-bolt screw demo."""

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_state": self.get_task_state,
            "get_arm_hand_joint_state": self.get_arm_hand_joint_state,
            "plan_nut_contact_approach": self.plan_nut_contact_approach,
            "plan_multi_fingertip_nut_contact": self.plan_multi_fingertip_nut_contact,
            "approach_nut_contact_pose": self.approach_nut_contact_pose,
            "execute_approach_waypoint": self.execute_approach_waypoint,
            "move_to_standoff": self.move_to_standoff,
            "move_to_nut": self.move_to_nut,
            "close_on_nut": self.close_on_nut,
            "screw_turn": self.screw_turn,
            "regrasp": self.regrasp,
            "run_reach_grasp_screw_sequence": self.run_reach_grasp_screw_sequence,
            "verify_screwed": self.verify_screwed,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current Kuka+Allegro nut-bolt observation dictionary."""
        return self._env.get_observation()

    def get_task_state(self) -> dict[str, Any]:
        """Return compact state for feedback and verification.

        Returns:
            Dictionary with arm/hand joint positions, current phase,
            unwrapped nut_joint command in radians, wrapped raw-equivalent
            angle, nut progress in turns, fingertip distance to the nut,
            regrasp cycle count, success threshold, task_success, and a
            physics note describing the DexScrew joint-model limitation.
        """
        return self._env.get_task_state()

    def get_arm_hand_joint_state(self) -> dict[str, np.ndarray]:
        """Return current Kuka arm and Allegro hand joint positions."""
        state = self.get_task_state()
        return {
            "arm_qpos": np.asarray(state["arm_joint_pos"], dtype=np.float64).copy(),
            "hand_qpos": np.asarray(state["hand_joint_pos"], dtype=np.float64).copy(),
        }

    def plan_nut_contact_approach(
        self,
        side: str = "front",
        clearance_m: float = 0.075,
        contact_offset_m: float = 0.018,
        press_offset_m: float = 0.008,
        wrist_sweep_rad: float = 0.95,
        close_for_contact: bool = False,
    ) -> dict[str, Any]:
        """Plan fine-grained approach waypoints for the already-sleeved nut.

        Args:
            side: Approach side around the nut. Use `front`, `back`, `left`, or `right`.
            clearance_m: Distance from the nut for the pre-contact waypoint.
            contact_offset_m: Lateral distance from the nut at the contact waypoint.
            press_offset_m: Additional inward offset for the press/contact waypoint.
            wrist_sweep_rad: Wrist sweep magnitude used by the later turn phase.
            close_for_contact: If true, the contact waypoints include a pinch/tight hand preshape.

        Returns:
            A dictionary containing Cartesian target points, empirical arm/hand
            waypoints, and reachability metadata for CAP/ViFailback.
        """
        return self._env.plan_nut_contact_approach(
            side=side,
            clearance_m=clearance_m,
            contact_offset_m=contact_offset_m,
            press_offset_m=press_offset_m,
            wrist_sweep_rad=wrist_sweep_rad,
            close_for_contact=close_for_contact,
        )

    def execute_approach_waypoint(self, waypoint_name: str, phase: str | None = None) -> dict[str, Any]:
        """Execute one waypoint returned by `plan_nut_contact_approach`.

        Args:
            waypoint_name: One of `standoff`, `precontact`, `contact`, `press`, or `turn_ready`.
            phase: Optional phase label for logging and feedback.

        Returns:
            Compact task state after moving to that waypoint.
        """
        return self._env.execute_approach_waypoint(waypoint_name, phase=phase)

    def plan_multi_fingertip_nut_contact(
        self,
        fingertips: list[str] | tuple[str, ...] = ("index_link_3", "middle_link_3", "thumb_link_3"),
        surface_offset_m: float = 0.018,
        z_offset_m: float = 0.0,
        layout: str = "front_arc",
        approach_side: str = "front",
        press_in_m: float = 0.0,
    ) -> dict[str, Any]:
        """Plan three fingertip targets around the already-sleeved nut.

        This is the CAP-facing description consumed by the physical GIF runner:
        index, middle, and thumb each get a world-frame contact target, then the
        low-level controller solves a stacked fingertip Jacobian closed loop.
        """
        return self._env.plan_multi_fingertip_nut_contact(
            fingertips=fingertips,
            surface_offset_m=surface_offset_m,
            z_offset_m=z_offset_m,
            layout=layout,
            approach_side=approach_side,
            press_in_m=press_in_m,
        )

    def approach_nut_contact_pose(
        self,
        side: str = "front",
        close_for_contact: bool = False,
        include_standoff: bool = False,
    ) -> dict[str, Any]:
        """Move through pre-contact, contact, and press waypoints near the nut.

        This is the preferred fine-grained replacement for the coarse
        `move_to_nut()` primitive when the task needs a physical contact turn.

        Args:
            side: Approach side around the nut. Use `front` first unless feedback says otherwise.
            close_for_contact: If true, close the hand while approaching the final contact pose.
            include_standoff: If true, begin from the high standoff waypoint.

        Returns:
            Compact task state with `approach_feedback` and the executed plan.
        """
        return self._env.approach_nut_contact_pose(
            side=side,
            close_for_contact=close_for_contact,
            include_standoff=include_standoff,
        )

    def move_to_standoff(self) -> dict[str, Any]:
        """Move the fixed-base arm to the high standoff pose before approach."""
        return self._env.move_to_standoff()

    def move_to_nut(self) -> dict[str, Any]:
        """Move the Kuka+Allegro hand from standoff to the nut-bolt work area."""
        return self._env.move_to_nut()

    def close_on_nut(self) -> dict[str, Any]:
        """Close the Allegro hand around the nut after the arm reaches it."""
        return self._env.close_on_nut()

    def screw_turn(self, angle_delta_rad: float = 3.0) -> dict[str, Any]:
        """Rotate the nut_joint forward by an unwrapped angle increment.

        Args:
            angle_delta_rad: Positive commanded rotation in radians. Values
                around 2.8-3.2 rad match the validated regrasp cadence.

        Returns:
            Compact task state after the turn.
        """
        return self._env.screw_turn(angle_delta_rad=angle_delta_rad)

    def regrasp(self) -> dict[str, Any]:
        """Open and reset the Allegro hand/wrist for another screw turn."""
        return self._env.regrasp()

    def run_reach_grasp_screw_sequence(
        self,
        turns: int = 3,
        angle_delta_rad: float = 3.1,
    ) -> dict[str, Any]:
        """Run the validated reach, grasp, screw, and regrasp sequence.

        Args:
            turns: Number of screw-turn phases.
            angle_delta_rad: Nut rotation command per turn phase in radians.

        Returns:
            Compact final task state.
        """
        self.move_to_standoff()
        self.move_to_nut()
        self.close_on_nut()
        for turn_idx in range(max(1, int(turns))):
            self.screw_turn(angle_delta_rad=angle_delta_rad)
            if turn_idx != max(1, int(turns)) - 1:
                self.regrasp()
                self.close_on_nut()
        return self.get_task_state()

    def verify_screwed(self, min_turns: float | None = None) -> bool:
        """Check whether nut rotation and hand proximity satisfy the task."""
        state = self.get_task_state()
        threshold = float(state["success_turns"] if min_turns is None else min_turns)
        return bool(
            float(state["nut_joint_progress_turns"]) >= threshold
            and float(state["min_fingertip_nut_distance_m"]) < 0.08
        )
