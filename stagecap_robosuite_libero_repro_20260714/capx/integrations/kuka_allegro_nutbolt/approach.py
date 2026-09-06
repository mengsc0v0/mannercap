from __future__ import annotations

from typing import Any

import numpy as np


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
HAND_TIGHT = np.array(
    [0.05, 1.28, 1.18, 0.98, 0.05, 1.22, 1.14, 0.96, 0.02, 0.95, 1.02, 0.86, 1.18, 0.78, 1.04, 0.86],
    dtype=np.float32,
)

DEFAULT_NUT_POS = np.array([0.47, -0.19, 0.255], dtype=np.float64)
DEFAULT_BOLT_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_CONTACT_FINGERTIPS = ("index_link_3", "middle_link_3", "thumb_link_3")


def build_nut_contact_approach_plan(
    nut_pos: list[float] | tuple[float, ...] | np.ndarray | None = None,
    *,
    side: str = "front",
    clearance_m: float = 0.075,
    contact_offset_m: float = 0.018,
    press_offset_m: float = 0.008,
    wrist_sweep_rad: float = 0.95,
    close_for_contact: bool = False,
) -> dict[str, Any]:
    """Build a CAP-readable approach plan for a nut already sleeved on a bolt.

    The plan has two parts: Cartesian target points around the nut and an
    empirical joint-space fallback for the current Kuka+Allegro asset. Future
    Jacobian/IK controllers can consume the Cartesian targets directly while
    existing CAP code can execute the joint waypoints.
    """
    nut = _as_vec3(DEFAULT_NUT_POS if nut_pos is None else nut_pos)
    side_name, side_vec = _side_vector(side)
    clearance = max(0.0, float(clearance_m))
    contact_offset = max(0.0, float(contact_offset_m))
    press_offset = max(0.0, float(press_offset_m))
    wrist = abs(float(wrist_sweep_rad))

    precontact_pos = nut + side_vec * clearance + np.array([0.0, 0.0, 0.040], dtype=np.float64)
    contact_pos = nut + side_vec * contact_offset + np.array([0.0, 0.0, 0.026], dtype=np.float64)
    press_pos = nut + side_vec * max(contact_offset - press_offset, 0.003) + np.array([0.0, 0.0, 0.020], dtype=np.float64)

    pre_q = _joint_bias_from_nut(KUKA_NEAR, nut)
    pre_q[6] = 0.0
    contact_press = max(0.0, (0.018 - contact_offset) / 0.018) * 0.5
    press_level = max(0.0, press_offset / 0.008)

    contact_q = _side_contact_bias(pre_q, side_name, press=contact_press)
    contact_q[6] = -wrist
    press_q = _side_contact_bias(pre_q, side_name, press=press_level)
    press_q[6] = -wrist
    turn_ready_q = press_q.copy()
    turn_ready_q[6] = wrist

    contact_hand = HAND_PINCH if close_for_contact else HAND_OPEN
    press_hand = HAND_TIGHT if close_for_contact else HAND_OPEN
    waypoints = [
        _waypoint("standoff", "start away from the nut", KUKA_AWAY, HAND_OPEN, precontact_pos),
        _waypoint("precontact", "move near the nut without touching it", pre_q, HAND_OPEN, precontact_pos),
        _waypoint("contact", "place fingertips beside the nut", contact_q, contact_hand, contact_pos),
        _waypoint("press", "slightly over-close the path to create contact", press_q, press_hand, press_pos),
        _waypoint("turn_ready", "wrist endpoint for a contact-driven turn", turn_ready_q, press_hand, press_pos),
    ]

    reachability = _reachability_score(nut)
    return {
        "planner_version": "kuka_allegro_nut_contact_approach_v0.2",
        "control_mode": "cartesian_targets_plus_empirical_joint_waypoints",
        "assumption": "nut is already sleeved on the fixed bolt before approach starts",
        "nut_pos": nut.astype(float).tolist(),
        "bolt_axis": DEFAULT_BOLT_AXIS.astype(float).tolist(),
        "approach_side": side_name,
        "side_unit_vector": side_vec.astype(float).tolist(),
        "precontact_pos": precontact_pos.astype(float).tolist(),
        "contact_pos": contact_pos.astype(float).tolist(),
        "press_pos": press_pos.astype(float).tolist(),
        "clearance_m": float(clearance),
        "contact_offset_m": float(contact_offset),
        "press_offset_m": float(press_offset),
        "contact_press_level": float(contact_press),
        "press_level": float(press_level),
        "wrist_sweep_rad": float(wrist),
        "estimated_reachability_score": float(reachability),
        "waypoints": waypoints,
    }


def build_multi_fingertip_contact_targets(
    nut_pos: list[float] | tuple[float, ...] | np.ndarray | None = None,
    *,
    fingertips: list[str] | tuple[str, ...] = DEFAULT_CONTACT_FINGERTIPS,
    surface_offset_m: float = 0.018,
    z_offset_m: float = 0.0,
    layout: str = "front_arc",
    approach_side: str = "front",
    press_in_m: float = 0.0,
) -> dict[str, Any]:
    """Assign index/middle/thumb tips to three contact points around the nut.

    The targets are expressed in the same world frame as the SDF nut. They are
    intentionally CAP-readable so feedback can say which fingertip missed which
    part of the nut instead of only reporting one aggregate distance.
    """
    nut = _as_vec3(DEFAULT_NUT_POS if nut_pos is None else nut_pos)
    names = [str(name) for name in fingertips]
    side_name, side_vec = _side_vector(approach_side)
    offsets = _multi_fingertip_offsets(
        surface_offset_m,
        z_offset_m,
        layout=layout,
        approach_side=side_name,
        side_vec=side_vec,
        press_in_m=press_in_m,
    )
    targets = []
    for idx, name in enumerate(names):
        offset = offsets[idx % len(offsets)]
        targets.append(
            {
                "fingertip": name,
                "target_pos": (nut + offset).astype(float).tolist(),
                "target_offset": offset.astype(float).tolist(),
                "role": _fingertip_contact_role(idx, name),
            }
        )
    return {
        "planner_version": "kuka_allegro_multi_fingertip_contact_v0.1",
        "control_mode": "multi_fingertip_jacobian_closed_loop",
        "assumption": "nut is already sleeved on the fixed bolt before approach starts",
        "nut_pos": nut.astype(float).tolist(),
        "bolt_axis": DEFAULT_BOLT_AXIS.astype(float).tolist(),
        "fingertips": names,
        "surface_offset_m": float(max(0.006, surface_offset_m)),
        "effective_surface_offset_m": float(max(0.0, surface_offset_m - max(0.0, press_in_m))),
        "press_in_m": float(max(0.0, press_in_m)),
        "z_offset_m": float(z_offset_m),
        "layout": _contact_layout_name(layout),
        "approach_side": side_name,
        "side_unit_vector": side_vec.astype(float).tolist(),
        "fingertip_targets": targets,
        "success_criteria": {
            "target_error_all_tips_max_m": 0.055,
            "target_error_any_tip_max_m": 0.035,
            "nut_precontact_translation_max_m": 0.010,
            "nut_contact_force_min": 0.1,
        },
    }


def summarize_approach_state(
    *,
    palm_nut_distance_m: float,
    min_fingertip_nut_distance_m: float,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return compact feedback fields for ViFailback/CAP diagnosis."""
    fingertip_distance = float(min_fingertip_nut_distance_m)
    palm_distance = float(palm_nut_distance_m)
    ready_for_close = fingertip_distance <= 0.095
    ready_for_contact_turn = fingertip_distance <= 0.080
    if plan is None:
        stage = "no_plan"
    elif ready_for_contact_turn:
        stage = "contact_ready"
    elif ready_for_close:
        stage = "close_ready"
    else:
        stage = "approach_not_close_enough"
    return {
        "stage": stage,
        "ready_for_close": bool(ready_for_close),
        "ready_for_contact_turn": bool(ready_for_contact_turn),
        "palm_nut_distance_m": palm_distance,
        "min_fingertip_nut_distance_m": fingertip_distance,
        "active_plan_version": None if plan is None else str(plan.get("planner_version")),
        "active_approach_side": None if plan is None else str(plan.get("approach_side")),
    }


def _as_vec3(value: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3)


def _side_vector(side: str) -> tuple[str, np.ndarray]:
    normalized = str(side).strip().lower()
    aliases = {
        "front": "front",
        "+y": "front",
        "near": "front",
        "back": "back",
        "-y": "back",
        "left": "left",
        "-x": "left",
        "right": "right",
        "+x": "right",
    }
    name = aliases.get(normalized, "front")
    vectors = {
        "front": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "back": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
    return name, vectors[name]


def _multi_fingertip_offsets(
    surface_offset: float,
    z_offset: float,
    *,
    layout: str,
    approach_side: str,
    side_vec: np.ndarray,
    press_in_m: float,
) -> list[np.ndarray]:
    nominal_radius = max(0.006, float(surface_offset))
    radius = max(0.0, nominal_radius - max(0.0, float(press_in_m)))
    z = float(z_offset)
    name = _contact_layout_name(layout)
    side = np.asarray(side_vec, dtype=np.float64).reshape(3)
    tangent = _planar_tangent(side)
    if name in {"side_arc", "wide_side_arc"}:
        spread = 0.85 if name == "side_arc" else 1.20
        center_gain = 1.0
        shoulder_gain = 0.48 if name == "side_arc" else 0.30
        return [
            shoulder_gain * radius * side + spread * nominal_radius * tangent + np.array([0.0, 0.0, z], dtype=np.float64),
            center_gain * radius * side + np.array([0.0, 0.0, z], dtype=np.float64),
            shoulder_gain * radius * side - spread * nominal_radius * tangent + np.array([0.0, 0.0, z], dtype=np.float64),
        ]
    if name == "opposed_tripod":
        return [
            0.65 * radius * side + 0.75 * nominal_radius * tangent + np.array([0.0, 0.0, z], dtype=np.float64),
            0.65 * radius * side - 0.75 * nominal_radius * tangent + np.array([0.0, 0.0, z], dtype=np.float64),
            -0.85 * radius * side + np.array([0.0, 0.0, z], dtype=np.float64),
        ]
    if name == "legacy_wrap":
        return [
            np.array([nominal_radius, -0.55 * radius, z], dtype=np.float64),
            np.array([0.0, radius, z], dtype=np.float64),
            np.array([-nominal_radius, -0.55 * radius, z], dtype=np.float64),
        ]
    if name == "wide_front_arc":
        return [
            np.array([nominal_radius, 0.25 * radius, z], dtype=np.float64),
            np.array([0.0, radius, z], dtype=np.float64),
            np.array([-nominal_radius, 0.25 * radius, z], dtype=np.float64),
        ]
    return [
        np.array([0.85 * nominal_radius, 0.45 * radius, z], dtype=np.float64),
        np.array([0.0, radius, z], dtype=np.float64),
        np.array([-0.85 * nominal_radius, 0.45 * radius, z], dtype=np.float64),
    ]


def _contact_layout_name(layout: str) -> str:
    normalized = str(layout).strip().lower().replace("-", "_")
    aliases = {
        "front": "front_arc",
        "frontarc": "front_arc",
        "front_arc": "front_arc",
        "tripod_front": "front_arc",
        "wide": "wide_front_arc",
        "wide_front": "wide_front_arc",
        "wide_front_arc": "wide_front_arc",
        "side": "side_arc",
        "side_arc": "side_arc",
        "approach_side": "side_arc",
        "tripod_side": "side_arc",
        "wide_side": "wide_side_arc",
        "wide_side_arc": "wide_side_arc",
        "opposed": "opposed_tripod",
        "opposed_tripod": "opposed_tripod",
        "wrap": "legacy_wrap",
        "legacy": "legacy_wrap",
        "legacy_wrap": "legacy_wrap",
    }
    return aliases.get(normalized, "front_arc")


def _planar_tangent(side_vec: np.ndarray) -> np.ndarray:
    side = np.asarray(side_vec, dtype=np.float64).reshape(3)
    tangent = np.array([side[1], -side[0], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return tangent / norm


def _fingertip_contact_role(idx: int, name: str) -> str:
    lower = name.lower()
    if "thumb" in lower:
        return "opposing-left/front contact"
    if "middle" in lower:
        return "front-center stabilizing contact"
    if "index" in lower:
        return "right/front contact"
    return f"contact point {idx + 1}"


def _joint_bias_from_nut(base_q: np.ndarray, nut: np.ndarray) -> np.ndarray:
    q = np.asarray(base_q, dtype=np.float32).copy()
    delta = nut - DEFAULT_NUT_POS
    q[0] += float(np.clip(-1.4 * delta[1], -0.35, 0.35))
    q[1] += float(np.clip(1.2 * delta[0] + 0.7 * delta[2], -0.25, 0.25))
    q[3] += float(np.clip(-2.0 * delta[2], -0.30, 0.30))
    q[5] += float(np.clip(0.8 * delta[0], -0.20, 0.20))
    return q


def _side_contact_bias(base_q: np.ndarray, side: str, *, press: float) -> np.ndarray:
    q = np.asarray(base_q, dtype=np.float32).copy()
    p = float(np.clip(press, 0.0, 3.0))
    if side == "front":
        q[1] += 0.04 + 0.05 * p
        q[3] -= 0.04 + 0.05 * p
    elif side == "back":
        q[1] -= 0.04 + 0.05 * p
        q[3] += 0.04 + 0.05 * p
    elif side == "left":
        q[0] += 0.10 + 0.06 * p
        q[5] -= 0.04 + 0.03 * p
    elif side == "right":
        q[0] -= 0.10 + 0.06 * p
        q[5] += 0.04 + 0.03 * p
    return q


def _waypoint(name: str, role: str, arm_q: np.ndarray, hand_q: np.ndarray, target_pos: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "arm_q": np.asarray(arm_q, dtype=np.float32).astype(float).tolist(),
        "hand_q": np.asarray(hand_q, dtype=np.float32).astype(float).tolist(),
        "target_pos": np.asarray(target_pos, dtype=np.float64).astype(float).tolist(),
    }


def _reachability_score(nut: np.ndarray) -> float:
    lower = np.array([0.30, -0.32, 0.18], dtype=np.float64)
    upper = np.array([0.66, 0.08, 0.38], dtype=np.float64)
    margin = np.minimum(nut - lower, upper - nut)
    normalized = np.min(margin / np.array([0.18, 0.20, 0.10], dtype=np.float64))
    return float(np.clip(normalized, 0.0, 1.0))
