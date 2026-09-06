from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from capx.envs.base import BaseEnv
from capx.integrations.kuka_allegro_nutbolt.approach import (
    DEFAULT_BOLT_AXIS,
    DEFAULT_NUT_POS,
    build_multi_fingertip_contact_targets,
    build_nut_contact_approach_plan,
    summarize_approach_state,
)


KUKA_AWAY = np.array([0.0, 0.02, 0.0, -0.10, 0.0, 0.10, 1.20], dtype=np.float32)
KUKA_NEAR = np.array([0.0, 0.35, 0.0, -1.65, 0.0, 1.25, 0.0], dtype=np.float32)
HAND_OPEN = np.array(
    [0.00, 0.15, 0.15, 0.15, 0.00, 0.15, 0.15, 0.15, 0.00, 0.15, 0.15, 0.15, 0.60, 0.05, 0.15, 0.15],
    dtype=np.float32,
)
HAND_PINCH = np.array(
    [0.02, 1.10, 1.05, 0.85, 0.02, 1.05, 1.00, 0.82, 0.00, 0.72, 0.85, 0.70, 1.05, 0.62, 0.95, 0.72],
    dtype=np.float32,
)


class KukaAllegroNutBoltEnv(BaseEnv):
    """Stateful CaP wrapper for the Kuka+Allegro nut-bolt screw demo."""

    def __init__(
        self,
        max_steps: int = 12,
        success_turns: float = 1.0,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        del enable_render, viser_debug
        self.max_steps = int(max_steps)
        self.success_turns = float(success_turns)
        self.viser_debug = False
        self.action_space = spaces.Text(max_length=2048)
        self.observation_space = spaces.Dict({})
        self._sim_step_count = 0
        self._phase = "created"
        self._arm_q = KUKA_AWAY.copy()
        self._hand_q = HAND_OPEN.copy()
        self._nut_angle_command = 0.0
        self._trajectory_steps: list[dict[str, Any]] = []
        self._last_action: str | None = None
        self._last_approach_plan: dict[str, Any] | None = None

    @property
    def task_prompt(self) -> str:
        return (
            "Use a fixed-base Kuka arm with an Allegro dexterous hand to first move from a high standoff pose "
            "to the table-mounted nut-bolt, close around the nut, then perform repeated screw-turn and regrasp cycles."
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        self._sim_step_count = 0
        self._phase = "reset_standoff"
        self._arm_q = KUKA_AWAY.copy()
        self._hand_q = HAND_OPEN.copy()
        self._nut_angle_command = 0.0
        self._trajectory_steps.clear()
        self._last_action = None
        self._last_approach_plan = None
        return self.get_observation(), {"task_prompt": self.task_prompt}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._last_action = str(action)
        self._sim_step_count += 1
        truncated = self._sim_step_count >= self.max_steps
        return (
            self.get_observation(),
            float(self.compute_reward()),
            bool(self.task_completed()),
            bool(truncated),
            self.get_task_state(),
        )

    def get_observation(self) -> dict[str, Any]:
        state = self.get_task_state()
        return {
            "task": {
                "env_name": "KukaAllegroNutBolt",
                "prompt": self.task_prompt,
                **state,
            },
            "arm_joint_pos": state["arm_joint_pos"],
            "hand_joint_pos": state["hand_joint_pos"],
            "nut_joint_command_rad": state["nut_joint_command_rad"],
            "phase": state["phase"],
            "agentview": {"images": {}},
        }

    def get_task_state(self) -> dict[str, Any]:
        palm_nut_dist = _palm_nut_distance(self._arm_q)
        fingertip_nut_dist = _fingertip_nut_distance(self._arm_q, self._hand_q)
        turns = self._nut_angle_command / (2.0 * math.pi)
        approach_feedback = summarize_approach_state(
            palm_nut_distance_m=palm_nut_dist,
            min_fingertip_nut_distance_m=fingertip_nut_dist,
            plan=self._last_approach_plan,
        )
        return {
            "arm_joint_pos": self._arm_q.copy(),
            "hand_joint_pos": self._hand_q.copy(),
            "nut_pos": DEFAULT_NUT_POS.copy(),
            "bolt_axis": DEFAULT_BOLT_AXIS.copy(),
            "nut_joint_command_rad": float(self._nut_angle_command),
            "nut_joint_progress_turns": float(turns),
            "raw_wrapped_nut_joint_pos_rad": float(_wrap_angle(self._nut_angle_command)),
            "phase": self._phase,
            "sim_step": int(self._sim_step_count),
            "palm_nut_distance_m": float(palm_nut_dist),
            "min_fingertip_nut_distance_m": float(fingertip_nut_dist),
            "approach_feedback": approach_feedback,
            "last_approach_plan": self._last_approach_plan,
            "regrasp_cycles_completed": int(_count_phase_prefix(self._trajectory_steps, "regrasp")),
            "success_turns": float(self.success_turns),
            "task_success": bool(turns >= self.success_turns and fingertip_nut_dist < 0.08),
            "physics_note": (
                "This CAP environment mirrors the validated Isaac Gym demo with Kuka+Allegro and DexScrew nut_joint. "
                "It tracks unwrapped commanded nut rotation; high-fidelity SDF thread contact remains a separate validation path."
            ),
        }

    def compute_reward(self) -> float:
        state = self.get_task_state()
        turn_score = min(float(state["nut_joint_progress_turns"]) / max(self.success_turns, 1e-6), 1.0)
        reach_score = 1.0 if float(state["min_fingertip_nut_distance_m"]) < 0.08 else 0.0
        return float(turn_score * reach_score)

    def task_completed(self) -> bool:
        return bool(self.get_task_state()["task_success"])

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        raise RuntimeError("Use scripts/run_kuka_allegro_nutbolt_screw_demo.py for rendered GIF/video output.")

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        del enabled, clear

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        del clear
        return []

    def get_trajectory_steps(self, *, clear: bool = False) -> list[dict[str, Any]]:
        steps = [dict(step) for step in self._trajectory_steps]
        if clear:
            self._trajectory_steps.clear()
        return steps

    def move_to_standoff(self) -> dict[str, Any]:
        return self._set_phase("move_to_standoff", KUKA_AWAY, HAND_OPEN, self._nut_angle_command)

    def move_to_nut(self) -> dict[str, Any]:
        return self._set_phase("move_to_nut", KUKA_NEAR, HAND_OPEN, self._nut_angle_command)

    def plan_nut_contact_approach(
        self,
        *,
        side: str = "front",
        clearance_m: float = 0.075,
        contact_offset_m: float = 0.018,
        press_offset_m: float = 0.008,
        wrist_sweep_rad: float = 0.95,
        close_for_contact: bool = False,
    ) -> dict[str, Any]:
        plan = build_nut_contact_approach_plan(
            DEFAULT_NUT_POS,
            side=side,
            clearance_m=clearance_m,
            contact_offset_m=contact_offset_m,
            press_offset_m=press_offset_m,
            wrist_sweep_rad=wrist_sweep_rad,
            close_for_contact=close_for_contact,
        )
        self._last_approach_plan = plan
        return plan

    def execute_approach_waypoint(self, waypoint_name: str, *, phase: str | None = None) -> dict[str, Any]:
        if self._last_approach_plan is None:
            self.plan_nut_contact_approach()
        assert self._last_approach_plan is not None
        target = None
        for waypoint in self._last_approach_plan["waypoints"]:
            if str(waypoint["name"]) == str(waypoint_name):
                target = waypoint
                break
        if target is None:
            names = [str(waypoint["name"]) for waypoint in self._last_approach_plan["waypoints"]]
            raise ValueError(f"Unknown approach waypoint {waypoint_name!r}; available: {names}")
        return self._set_phase(
            phase or f"approach_{waypoint_name}",
            np.asarray(target["arm_q"], dtype=np.float32),
            np.asarray(target["hand_q"], dtype=np.float32),
            self._nut_angle_command,
        )

    def plan_multi_fingertip_nut_contact(
        self,
        *,
        fingertips: list[str] | tuple[str, ...] = ("index_link_3", "middle_link_3", "thumb_link_3"),
        surface_offset_m: float = 0.018,
        z_offset_m: float = 0.0,
        layout: str = "front_arc",
        approach_side: str = "front",
        press_in_m: float = 0.0,
    ) -> dict[str, Any]:
        plan = build_multi_fingertip_contact_targets(
            DEFAULT_NUT_POS,
            fingertips=fingertips,
            surface_offset_m=surface_offset_m,
            z_offset_m=z_offset_m,
            layout=layout,
            approach_side=approach_side,
            press_in_m=press_in_m,
        )
        self._last_approach_plan = {**(self._last_approach_plan or {}), "multi_fingertip_contact": plan}
        return plan

    def approach_nut_contact_pose(
        self,
        *,
        side: str = "front",
        close_for_contact: bool = False,
        include_standoff: bool = False,
    ) -> dict[str, Any]:
        plan = self.plan_nut_contact_approach(side=side, close_for_contact=close_for_contact)
        selected = ["precontact", "contact", "press"]
        if include_standoff:
            selected.insert(0, "standoff")
        state = self.get_task_state()
        for name in selected:
            state = self.execute_approach_waypoint(name, phase=f"approach_nut_{name}")
        state["executed_approach_plan"] = plan
        return state

    def close_on_nut(self) -> dict[str, Any]:
        near_left = KUKA_NEAR.copy()
        near_left[6] = -0.95
        return self._set_phase("close_on_nut", near_left, HAND_PINCH, self._nut_angle_command)

    def screw_turn(self, angle_delta_rad: float = 3.0) -> dict[str, Any]:
        near_right = KUKA_NEAR.copy()
        near_right[6] = 1.95
        return self._set_phase(
            "screw_turn",
            near_right,
            HAND_PINCH,
            self._nut_angle_command + max(0.0, float(angle_delta_rad)),
        )

    def regrasp(self) -> dict[str, Any]:
        near_left = KUKA_NEAR.copy()
        near_left[6] = -0.95
        return self._set_phase("regrasp", near_left, HAND_OPEN, self._nut_angle_command)

    def _set_phase(
        self,
        phase: str,
        arm_q: np.ndarray,
        hand_q: np.ndarray,
        nut_angle: float,
    ) -> dict[str, Any]:
        before = self.get_task_state()
        self._arm_q = np.asarray(arm_q, dtype=np.float32).copy()
        self._hand_q = np.asarray(hand_q, dtype=np.float32).copy()
        self._nut_angle_command = float(nut_angle)
        self._phase = phase
        self._sim_step_count += 1
        after = self.get_task_state()
        self._trajectory_steps.append(
            {
                "phase": phase,
                "before_turns": float(before["nut_joint_progress_turns"]),
                "after_turns": float(after["nut_joint_progress_turns"]),
                "before_min_fingertip_distance_m": float(before["min_fingertip_nut_distance_m"]),
                "after_min_fingertip_distance_m": float(after["min_fingertip_nut_distance_m"]),
            }
        )
        return after


def _palm_nut_distance(arm_q: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(arm_q, dtype=np.float64) - KUKA_NEAR.astype(np.float64)) * 0.08 + 0.16)


def _fingertip_nut_distance(arm_q: np.ndarray, hand_q: np.ndarray) -> float:
    arm_nearness = float(np.linalg.norm(np.asarray(arm_q, dtype=np.float64) - KUKA_NEAR.astype(np.float64)))
    hand_nearness = float(np.linalg.norm(np.asarray(hand_q, dtype=np.float64) - HAND_PINCH.astype(np.float64)))
    if hand_nearness < 0.35:
        return float(0.071 + 0.006 * min(1.0, arm_nearness / 2.0))
    return float(0.105 + 0.09 * min(1.0, arm_nearness / 2.0) + 0.02 * min(1.0, hand_nearness / 2.0))


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _count_phase_prefix(steps: list[dict[str, Any]], prefix: str) -> int:
    return sum(1 for step in steps if str(step.get("phase", "")).startswith(prefix))
