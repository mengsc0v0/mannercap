from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.dclaw.reference_policy import DClawReferencePolicy


RESET_POSE = np.array([0.0, -1.0, 0.67] * 3, dtype=np.float32)
OPEN_POSE = np.array([0.0, -1.0, 1.0] * 3, dtype=np.float32)
CLOSE_POSE = np.array([0.0, 1.0, -1.0] * 3, dtype=np.float32)
PINCH_POSE = np.array([0.0, 0.65, -0.75] * 3, dtype=np.float32)


class DClawControlApi(ApiBase):
    """CaP-facing API for ROBEL D'Claw screw/turn tasks."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        self._last_roll_direction = "counterclockwise"
        self._reference_policies: dict[tuple[str, str, float], DClawReferencePolicy] = {}

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_state": self.get_task_state,
            "get_hand_joint_state": self.get_hand_joint_state,
            "get_object_angle": self.get_object_angle,
            "get_object_velocity": self.get_object_velocity,
            "get_target_angle": self.get_target_angle,
            "get_target_error": self.get_target_error,
            "set_joint_positions": self.set_joint_positions,
            "set_hand_preshape": self.set_hand_preshape,
            "open_hand": self.open_hand,
            "close_hand": self.close_hand,
            "increase_normal_force": self.increase_normal_force,
            "apply_synchronous_sweep": self.apply_synchronous_sweep,
            "apply_tripod_roll": self.apply_tripod_roll,
            "roll_object": self.roll_object,
            "search_and_roll_object": self.search_and_roll_object,
            "track_target_error_until": self.track_target_error_until,
            "run_reference_screw_policy": self.run_reference_screw_policy,
            "recover_slip": self.recover_slip,
            "stabilize_object_spin": self.stabilize_object_spin,
            "verify_screw_rotation": self.verify_screw_rotation,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current D'Claw observation.

        Returns:
            A dictionary with an `agentview` RGB image and task state fields:
            `hand_joint_pos`, `hand_joint_vel`, `object_angle`,
            `object_velocity`, `target_angle`, `target_velocity`,
            `target_error`, and `last_action`.
        """
        return self._env.get_observation()

    def get_task_state(self) -> dict[str, Any]:
        """Return compact privileged task state for feedback and verification.

        Returns:
            Dictionary containing hand joints, object angle/velocity, target
            angle/velocity, signed target error in radians, absolute target
            error, reward, success, and simulation step.
        """
        return self._env.get_task_state()

    def get_hand_joint_state(self) -> dict[str, np.ndarray]:
        """Return the current 9-DoF D'Claw joint positions and velocities.

        Returns:
            Dict with:
            - `qpos`: 9 joint positions in radians.
            - `qvel`: 9 joint velocities in radians/second.
        """
        state = self.get_task_state()
        return {
            "qpos": np.asarray(state["hand_joint_pos"], dtype=np.float64).copy(),
            "qvel": np.asarray(state["hand_joint_vel"], dtype=np.float64).copy(),
        }

    def get_object_angle(self) -> float:
        """Return the current screw/object angle in radians."""
        return float(self.get_task_state()["object_angle"])

    def get_object_velocity(self) -> float:
        """Return the current screw/object angular velocity in radians/second."""
        return float(self.get_task_state()["object_velocity"])

    def get_target_angle(self) -> float:
        """Return the current target screw/object angle in radians."""
        return float(self.get_task_state()["target_angle"])

    def get_target_error(self) -> float:
        """Return signed target error in radians.

        Positive values mean the target is ahead of the object angle. The usual
        repair is to roll counterclockwise for positive error and clockwise for
        negative error.
        """
        return float(self.get_task_state()["target_error"])

    def set_joint_positions(
        self,
        action: list[float] | tuple[float, ...] | np.ndarray,
        steps: int = 20,
    ) -> dict[str, Any]:
        """Apply a normalized 9-value D'Claw joint command for several steps.

        Args:
            action: 9 values in [-1, 1]. The layout is three joints per finger:
                [finger0_base, finger0_middle, finger0_tip, finger1_base, ...].
            steps: Number of ROBEL environment steps to repeat this command.

        Returns:
            Compact task state after the command.
        """
        self._env.apply_action(np.asarray(action, dtype=np.float32), steps=steps)
        return self.get_task_state()

    def set_hand_preshape(self, name: str = "reset", steps: int = 30) -> dict[str, Any]:
        """Move the hand to a reusable preshape.

        Args:
            name: One of `reset`, `open`, `close`, or `pinch`.
            steps: Number of ROBEL steps to hold the preshape.

        Returns:
            Compact task state after moving to the preshape.
        """
        pose = _preshape(name)
        return self.set_joint_positions(pose, steps=steps)

    def open_hand(self, steps: int = 25) -> dict[str, Any]:
        """Open all three fingers to reduce contact before regrasping."""
        return self.set_hand_preshape("open", steps=steps)

    def close_hand(self, steps: int = 25) -> dict[str, Any]:
        """Close all three fingers to increase contact with the screw/object."""
        return self.set_hand_preshape("close", steps=steps)

    def increase_normal_force(self, amount: float = 0.2, steps: int = 20) -> dict[str, Any]:
        """Increase fingertip contact pressure while preserving the last command.

        Args:
            amount: Extra normalized closing amount applied mainly to middle
                and fingertip joints. Use 0.1-0.35 for gentle repairs.
            steps: Number of ROBEL steps to hold the contact command.

        Returns:
            Compact task state after the contact-pressure repair.
        """
        last_action = np.asarray(self.get_observation().get("last_action", RESET_POSE), dtype=np.float32)
        repair = last_action.copy()
        repair[1::3] += float(amount)
        repair[2::3] -= float(amount)
        repair = np.clip(repair, -1.0, 1.0)
        return self.set_joint_positions(repair, steps=steps)

    def apply_synchronous_sweep(
        self,
        steps: int = 80,
        amplitude: float = 0.65,
        period: int = 80,
        bias: float = 0.0,
    ) -> dict[str, Any]:
        """Apply the same sinusoidal sweep to all fingers.

        This is useful as a diagnostic primitive: if the object barely moves,
        contact is likely insufficient; if it moves but target_error grows, the
        phase/direction is wrong.

        Args:
            steps: Number of ROBEL steps.
            amplitude: Normalized joint oscillation amplitude.
            period: Oscillation period in ROBEL steps.
            bias: Bias added to middle joints to increase contact.

        Returns:
            Compact task state after the sweep.
        """
        for t in range(max(1, int(steps))):
            phase = 2.0 * np.pi * t / max(1, int(period))
            action = np.array(
                [
                    amplitude * np.sin(phase),
                    bias + amplitude * np.sin(phase + 1.2),
                    amplitude * np.cos(phase),
                ]
                * 3,
                dtype=np.float32,
            )
            self._env.apply_action(np.clip(action, -1.0, 1.0), steps=1)
        return self.get_task_state()

    def apply_tripod_roll(
        self,
        steps: int = 80,
        direction: str = "counterclockwise",
        amplitude: float = 0.8,
        contact_bias: float = 0.15,
        period: int = 80,
    ) -> dict[str, Any]:
        """Roll the object with phase-shifted tripod finger motion.

        Args:
            steps: Number of ROBEL steps to execute.
            direction: `counterclockwise` or `clockwise`.
            amplitude: Normalized rolling amplitude for finger joints.
            contact_bias: Bias on middle joints; larger values press harder.
            period: Oscillation period in ROBEL steps.

        Returns:
            Compact task state after rolling.
        """
        sign = _direction_sign(direction)
        self._last_roll_direction = "counterclockwise" if sign > 0 else "clockwise"
        for t in range(max(1, int(steps))):
            phase = sign * 2.0 * np.pi * t / max(1, int(period))
            values: list[float] = []
            for finger in range(3):
                ph = phase + finger * 2.0 * np.pi / 3.0
                values.extend(
                    [
                        amplitude * np.sin(ph),
                        contact_bias + amplitude * np.sin(ph + 1.2),
                        amplitude * np.cos(ph),
                    ]
                )
            self._env.apply_action(np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0), steps=1)
        return self.get_task_state()

    def roll_object(
        self,
        direction: str = "auto",
        steps: int = 80,
        amplitude: float = 0.8,
        contact_bias: float = 0.15,
    ) -> dict[str, Any]:
        """Roll the object toward the target angle.

        Args:
            direction: `auto`, `counterclockwise`, or `clockwise`. In auto mode,
                positive target_error rolls counterclockwise and negative
                target_error rolls clockwise.
            steps: Number of ROBEL steps.
            amplitude: Normalized rolling amplitude.
            contact_bias: Fingertip contact bias.

        Returns:
            Compact task state after the roll.
        """
        if direction == "auto":
            direction = "counterclockwise" if self.get_target_error() >= 0 else "clockwise"
        return self.apply_tripod_roll(
            steps=steps,
            direction=direction,
            amplitude=amplitude,
            contact_bias=contact_bias,
        )

    def search_and_roll_object(
        self,
        steps: int = 40,
        amplitudes: list[float] | tuple[float, ...] = (0.45, 0.65, 0.85),
        contact_biases: list[float] | tuple[float, ...] = (0.0, 0.18, 0.35),
        directions: list[str] | tuple[str, ...] = ("auto", "clockwise", "counterclockwise"),
    ) -> dict[str, Any]:
        """Try short rollouts from the current state and execute the best one.

        This helper is a lightweight automatic calculation tool. It checkpoints
        the simulator, tries several direction/amplitude/contact settings,
        scores them by target_error and real object motion, restores the
        original state, then runs the best candidate for real.

        Args:
            steps: Number of ROBEL steps per candidate and final execution.
            amplitudes: Candidate normalized rolling amplitudes.
            contact_biases: Candidate fingertip contact biases.
            directions: Candidate directions; `auto` is resolved from target_error.

        Returns:
            Compact task state with `selected_roll` metadata describing the
            chosen direction, amplitude, contact_bias, and predicted error.
        """
        if not hasattr(self._env, "checkpoint_state") or not hasattr(self._env, "restore_state"):
            return self.roll_object(direction="auto", steps=steps)

        base_checkpoint = self._env.checkpoint_state()
        base_angle = self.get_object_angle()
        candidates: list[dict[str, Any]] = []
        was_recording = bool(getattr(self._env, "_record_frames", False))

        try:
            self._env._record_frames = False
            for direction in directions:
                resolved = self._resolve_direction(direction)
                for amplitude in amplitudes:
                    for contact_bias in contact_biases:
                        self._env.restore_state(base_checkpoint)
                        state = self.apply_tripod_roll(
                            steps=steps,
                            direction=resolved,
                            amplitude=float(amplitude),
                            contact_bias=float(contact_bias),
                        )
                        moved = abs(_angle_delta(float(state["object_angle"]), base_angle))
                        score = (
                            abs(float(state["target_error"]))
                            - 0.15 * moved
                            - 0.01 * float(state["last_reward"])
                        )
                        candidates.append(
                            {
                                "score": float(score),
                                "direction": resolved,
                                "amplitude": float(amplitude),
                                "contact_bias": float(contact_bias),
                                "predicted_target_error": float(state["target_error"]),
                                "predicted_abs_error": abs(float(state["target_error"])),
                                "object_motion": float(moved),
                            }
                        )
        finally:
            self._env.restore_state(base_checkpoint)
            self._env._record_frames = was_recording

        best = min(candidates, key=lambda item: item["score"])
        state = self.apply_tripod_roll(
            steps=steps,
            direction=str(best["direction"]),
            amplitude=float(best["amplitude"]),
            contact_bias=float(best["contact_bias"]),
        )
        state["selected_roll"] = best
        return state

    def track_target_error_until(
        self,
        threshold: float = 0.2,
        max_steps: int = 240,
        chunk_steps: int = 30,
        amplitude: float = 0.8,
        contact_bias: float = 0.18,
        use_search: bool = True,
    ) -> dict[str, Any]:
        """Closed-loop roll until target_error is small or the step budget ends.

        Args:
            threshold: Stop when abs(target_error) is below this many radians.
            max_steps: Maximum ROBEL steps available to this helper.
            chunk_steps: Roll in chunks and re-check target_error after each.
            amplitude: Tripod roll amplitude.
            contact_bias: Fingertip contact pressure bias.
            use_search: If True, run short parameter search before each roll
                chunk, then execute the best candidate.

        Returns:
            Compact task state after tracking.
        """
        used_steps = 0
        last_abs_error = abs(self.get_target_error())
        while used_steps < max_steps:
            if last_abs_error <= threshold:
                break
            before_angle = self.get_object_angle()
            step_budget = min(chunk_steps, max_steps - used_steps)
            if use_search:
                state = self.search_and_roll_object(
                    steps=step_budget,
                    amplitudes=(max(0.25, amplitude - 0.25), amplitude, min(1.0, amplitude + 0.15)),
                    contact_biases=(0.0, contact_bias, min(0.45, contact_bias + 0.17)),
                )
            else:
                state = self.roll_object(
                    direction="auto",
                    steps=step_budget,
                    amplitude=amplitude,
                    contact_bias=contact_bias,
                )
            used_steps += min(chunk_steps, max_steps - used_steps)
            moved = abs(_angle_delta(float(state["object_angle"]), before_angle))
            current_abs_error = abs(float(state["target_error"]))
            if moved < 0.02:
                self.increase_normal_force(amount=0.15, steps=10)
            elif current_abs_error > last_abs_error + 0.05:
                reverse = "clockwise" if self._last_roll_direction == "counterclockwise" else "counterclockwise"
                self.apply_tripod_roll(
                    steps=min(chunk_steps, max_steps - used_steps),
                    direction=reverse,
                    amplitude=amplitude,
                    contact_bias=contact_bias,
                )
                used_steps += min(chunk_steps, max_steps - used_steps)
            last_abs_error = abs(self.get_target_error())
        return self.get_task_state()

    def recover_slip(self, steps: int = 50) -> dict[str, Any]:
        """Recover when fingers visually slip or object angle barely changes.

        The helper resets the hand, closes gently, increases normal force, then
        resumes rolling in the direction implied by target_error.
        """
        self.set_hand_preshape("reset", steps=max(5, steps // 4))
        self.close_hand(steps=max(5, steps // 4))
        self.increase_normal_force(amount=0.2, steps=max(5, steps // 4))
        return self.roll_object(direction="auto", steps=max(5, steps // 4))

    def stabilize_object_spin(self, steps: int = 30) -> dict[str, Any]:
        """Hold a stable closing pose to damp object spin after rolling."""
        return self.set_joint_positions(CLOSE_POSE, steps=steps)

    def verify_screw_rotation(self, threshold: float = 0.2) -> bool:
        """Check whether the screw/object is within target-error threshold.

        Args:
            threshold: Maximum absolute target_error in radians.

        Returns:
            True if abs(target_error) <= threshold or the ROBEL score reports success.
        """
        state = self.get_task_state()
        return bool(state.get("score_success", False) or abs(float(state["target_error"])) <= threshold)

    def _resolve_direction(self, direction: str) -> str:
        if direction == "auto":
            return "counterclockwise" if self.get_target_error() >= 0 else "clockwise"
        _direction_sign(direction)
        return direction

    def run_reference_screw_policy(
        self,
        steps: int = 160,
        action_mode: str = "mean_scaled_tanh",
        scale: float = 0.5,
        stop_on_success: bool = False,
        success_threshold: float = 0.2,
    ) -> dict[str, Any]:
        """Run the bundled ROBEL softlearning screw policy through a numpy MLP.

        This is a reference/oracle primitive for DClaw screw tracking. It does
        not require installing `softlearning` or TensorFlow; the bundled ROBEL
        pickle is decoded into numpy weights and executed directly.

        Args:
            steps: Maximum ROBEL steps to execute.
            action_mode: Deterministic action head interpretation. The default
                `mean_scaled_tanh` is the validated mode for DClawScrewFixed-v0.
            scale: Scale applied before tanh for `mean_scaled_tanh`.
            stop_on_success: If True, stop as soon as success is observed. For
                continuous screw tracking, keep this False when generating a
                longer successful trajectory.
            success_threshold: Absolute target_error threshold in radians.

        Returns:
            Compact task state plus `reference_policy` metadata with final
            success, ever success, max consecutive success steps, and error
            statistics.
        """
        policy = self._get_reference_policy(action_mode=action_mode, scale=scale)
        errors: list[float] = []
        successes: list[bool] = []
        rewards: list[float] = []

        for _ in range(max(1, int(steps))):
            obs = self._env.get_flat_observation()
            action = policy.action(obs)
            self._env.apply_action(action, steps=1)
            state = self.get_task_state()
            error = float(state["target_error"])
            success = bool(state.get("score_success", False) or abs(error) <= success_threshold)
            errors.append(error)
            successes.append(success)
            rewards.append(float(state["last_reward"]))
            if stop_on_success and success:
                break

        final_state = self.get_task_state()
        max_consecutive = _max_consecutive_true(successes)
        final_state["reference_policy"] = {
            "env_id": getattr(self._env, "env_id", "unknown"),
            "steps_requested": int(steps),
            "steps_executed": len(errors),
            "action_mode": action_mode,
            "scale": float(scale),
            "success_threshold": float(success_threshold),
            "final_success": bool(successes[-1]) if successes else False,
            "ever_success": bool(any(successes)),
            "max_consecutive_success_steps": int(max_consecutive),
            "final_abs_error": abs(errors[-1]) if errors else None,
            "best_abs_error": min((abs(error) for error in errors), default=None),
            "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        }
        return final_state

    def _get_reference_policy(
        self,
        *,
        action_mode: str,
        scale: float,
    ) -> DClawReferencePolicy:
        env_id = getattr(self._env, "env_id", "DClawScrewFixed-v0")
        key = (env_id, action_mode, float(scale))
        if key not in self._reference_policies:
            self._reference_policies[key] = DClawReferencePolicy.from_robel_pickle(
                env_id,
                action_mode=action_mode,
                scale=scale,
            )
        return self._reference_policies[key]


def _preshape(name: str) -> np.ndarray:
    normalized = name.strip().lower()
    poses = {
        "reset": RESET_POSE,
        "home": RESET_POSE,
        "open": OPEN_POSE,
        "close": CLOSE_POSE,
        "closed": CLOSE_POSE,
        "pinch": PINCH_POSE,
    }
    if normalized not in poses:
        raise ValueError(f"Unknown DClaw preshape: {name}. Use reset/open/close/pinch.")
    return poses[normalized].copy()


def _direction_sign(direction: str) -> float:
    normalized = direction.strip().lower()
    if normalized in {"counterclockwise", "ccw", "positive", "+"}:
        return 1.0
    if normalized in {"clockwise", "cw", "negative", "-"}:
        return -1.0
    raise ValueError("direction must be counterclockwise/ccw or clockwise/cw")


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _max_consecutive_true(values: list[bool]) -> int:
    current = 0
    best = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
