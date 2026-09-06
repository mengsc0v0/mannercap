from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class DexScrewXHandNutBoltControlApi(ApiBase):
    """CaP-facing API for DexScrew XHand nut-bolt contact manipulation."""

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_state": self.get_task_state,
            "get_hand_joint_state": self.get_hand_joint_state,
            "get_nutbolt_state": self.get_nutbolt_state,
            "apply_hand_action": self.apply_hand_action,
            "run_contact_policy": self.run_contact_policy,
            "verify_contact_driven_rotation": self.verify_contact_driven_rotation,
            "diagnose_nutbolt_failure": self.diagnose_nutbolt_failure,
            "get_rollout_trace": self.get_rollout_trace,
            "satisfy_dexscrew_nutbolt_goal_predicates": self.satisfy_dexscrew_nutbolt_goal_predicates,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current DexScrew XHand nut-bolt observation dictionary."""
        return self._env.get_observation()

    def get_task_state(self) -> dict[str, Any]:
        """Return privileged numeric state for contact-driven feedback.

        Returns:
            Dictionary with selected hand joint state, nut pose, nut_joint
            position/velocity, nut angle delta, contact force metrics,
            success thresholds, rollout phase, and audit booleans proving that
            the CAP program did not directly script object or nut_joint state.
        """
        return self._env.get_task_state()

    def get_hand_joint_state(self) -> dict[str, np.ndarray]:
        """Return measured and target XHand joint positions.

        Returns:
            Dictionary with `hand_qpos` and `target_qpos`, each 12-D arrays for
            the selected DexScrew XHand action space.
        """
        state = self.get_task_state()
        return {
            "hand_qpos": np.asarray(state["hand_q_sample"], dtype=np.float64).copy(),
            "target_qpos": np.asarray(state["target_q_sample"], dtype=np.float64).copy(),
        }

    def get_nutbolt_state(self) -> dict[str, Any]:
        """Return nut-bolt state and contact metrics for failure diagnosis."""
        state = self.get_task_state()
        return {
            "nut_pos": np.asarray(state["nut_pos"], dtype=np.float64).copy(),
            "object_pos": np.asarray(state["object_pos"], dtype=np.float64).copy(),
            "nut_dof_pos": float(state["nut_dof_pos"]),
            "nut_dof_vel": float(state["nut_dof_vel"]),
            "nut_dof_initial_rad": float(state["nut_dof_initial_rad"]),
            "nut_dof_delta_rad": float(state["nut_dof_delta_rad"]),
            "nut_dof_delta_turns": float(state["nut_dof_delta_turns"]),
            "nut_contact_force_norm": float(state["nut_contact_force_norm"]),
            "nut_contact_force_max": float(state["nut_contact_force_max"]),
            "fingertip_contact_force_max": float(state["fingertip_contact_force_max"]),
            "fingertip_contact_force_norms": list(state["fingertip_contact_force_norms"]),
            "task_success": bool(state["task_success"]),
        }

    def get_rollout_trace(self) -> list[dict[str, Any]]:
        """Return the per-step physical state trace from the current rollout."""
        return self._env.get_rollout_trace(clear=False)

    def apply_hand_action(
        self,
        action: list[float] | tuple[float, ...] | np.ndarray,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Apply only a 12-D XHand action.

        Args:
            action: 12 hand joint action values in [-1, 1]. This API never
                accepts or forwards a nut/object joint command.
            steps: Number of DexScrew control steps to repeat.

        Returns:
            Compact task state after the action chunk.
        """
        return self._env.apply_hand_action(action, steps=steps, phase="cap_hand_action")

    def run_contact_policy(
        self,
        steps: int = 120,
        stop_on_success: bool = False,
    ) -> dict[str, Any]:
        """Run the bundled DexScrew XHand policy as a CAP primitive.

        Args:
            steps: Number of policy control steps to execute.
            stop_on_success: If true, stop once contact-driven rotation passes
                the current threshold.

        Returns:
            Compact task state after the policy rollout.
        """
        return self._env.run_contact_policy(steps=steps, stop_on_success=stop_on_success)

    def verify_contact_driven_rotation(
        self,
        angle_threshold: float | None = None,
        min_nut_contact: float | None = None,
        min_fingertip_contact: float | None = None,
    ) -> bool:
        """Check whether nut rotation is physically contact-driven.

        The check requires nut_joint angle change, nut contact force, fingertip
        contact force, no direct object pose scripting, and zero commanded nut
        joint action.
        """
        return self._env.verify_contact_driven_rotation(
            angle_threshold=angle_threshold,
            min_nut_contact=min_nut_contact,
            min_fingertip_contact=min_fingertip_contact,
        )

    def diagnose_nutbolt_failure(self) -> dict[str, Any]:
        """Return ViFailback-style numeric diagnosis and repair suggestions."""
        return self._env.diagnose_nutbolt_failure()

    def satisfy_dexscrew_nutbolt_goal_predicates(
        self,
        max_rounds: int = 3,
        steps_per_round: int = 60,
    ) -> dict[str, Any]:
        """Run a guarded contact-policy loop until the nut-bolt goal is met.

        Args:
            max_rounds: Maximum feedback rounds. This mirrors the project rule
                that a task should run at most three correction rounds.
            steps_per_round: Policy steps per round.

        Returns:
            Final task state, with diagnosis stored under `feedback` if the
            goal is still not satisfied.
        """
        final_state: dict[str, Any] = self.get_task_state()
        for _ in range(max(1, int(max_rounds))):
            final_state = self.run_contact_policy(
                steps=max(1, int(steps_per_round)),
                stop_on_success=True,
            )
            if self.verify_contact_driven_rotation():
                return final_state
        final_state = dict(final_state)
        final_state["feedback"] = self.diagnose_nutbolt_failure()
        return final_state
