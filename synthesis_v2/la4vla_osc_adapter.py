"""Fail-closed MannerCAP pose transition to LA4VLA OSC_POSE adapter.

LA4VLA's evaluated policy sends a normalized 7-D command directly to
RoboSuite's delta OSC_POSE controller.  The controller scales translation by
0.05 m and its axis-angle rotation vector by 0.5 rad, then pre-multiplies the
rotation delta with the current end-effector orientation.  This differs from
the old componentwise difference of two axis-angle state vectors.

This module is pure Python.  It proves the pose-command algebra and records
clipping explicitly; it does not claim physical replay success.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


SCHEMA_VERSION = "synthesis_v2.la4vla_osc_pose_adapter.v1"
TRANSLATION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5


class OscAdapterError(ValueError):
    def __init__(self, message: str, *, code: str, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


def _finite_vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OscAdapterError(
            f"{field} must be a sequence of length {size}",
            code="invalid_vector",
            field=field,
        )
    if len(value) != size:
        raise OscAdapterError(
            f"{field} must have length {size}",
            code="dimension_mismatch",
            field=field,
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise OscAdapterError(
                f"{field}[{index}] must be finite",
                code="non_finite",
                field=field,
            )
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise OscAdapterError(
                f"{field}[{index}] must be finite",
                code="non_finite",
                field=field,
            ) from error
        if not math.isfinite(number):
            raise OscAdapterError(
                f"{field}[{index}] must be finite",
                code="non_finite",
                field=field,
            )
        result.append(number)
    return tuple(result)


def _normalize_quat_wxyz(value: Any, field: str) -> tuple[float, float, float, float]:
    quat = _finite_vector(value, 4, field)
    norm = math.sqrt(sum(component * component for component in quat))
    if norm <= 1e-12:
        raise OscAdapterError(
            f"{field} has zero norm", code="invalid_quaternion", field=field
        )
    return tuple(component / norm for component in quat)  # type: ignore[return-value]


def _quat_multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_inverse_wxyz(quat: Sequence[float]) -> tuple[float, float, float, float]:
    return (quat[0], -quat[1], -quat[2], -quat[3])


def _quat_to_rotvec_wxyz(quat: Sequence[float]) -> tuple[float, float, float]:
    normalized = _normalize_quat_wxyz(quat, "relative_quaternion")
    if normalized[0] < 0.0:
        normalized = tuple(-value for value in normalized)  # shortest representation
    w, x, y, z = normalized
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, min(1.0, w)))
    scale = angle / vector_norm
    return (x * scale, y * scale, z * scale)


def _rotvec_to_quat_wxyz(rotvec: Sequence[float]) -> tuple[float, float, float, float]:
    vector = _finite_vector(rotvec, 3, "rotation_delta_rad")
    angle = math.sqrt(sum(component * component for component in vector))
    if angle <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return (math.cos(half), vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _orientation_error_rad(left: Sequence[float], right: Sequence[float]) -> float:
    relative = _quat_multiply_wxyz(left, _quat_inverse_wxyz(right))
    return math.sqrt(sum(value * value for value in _quat_to_rotvec_wxyz(relative)))


def orientation_error_rad(
    left_wxyz: Sequence[Any], right_wxyz: Sequence[Any]
) -> float:
    """Return the shortest angular distance between two orientations."""

    left = _normalize_quat_wxyz(left_wxyz, "left_wxyz")
    right = _normalize_quat_wxyz(right_wxyz, "right_wxyz")
    return _orientation_error_rad(left, right)


def quaternion_to_rotvec_wxyz(value: Sequence[Any]) -> tuple[float, float, float]:
    """Convert a quaternion to the shortest axis-angle vector."""

    return _quat_to_rotvec_wxyz(_normalize_quat_wxyz(value, "quaternion_wxyz"))


@dataclass(frozen=True)
class OscPoseAdapterResult:
    translation_delta_m: tuple[float, float, float]
    rotation_delta_rad: tuple[float, float, float]
    gripper_command: float
    normalized_action: tuple[float, float, float, float, float, float, float]
    clipped_dimensions: tuple[int, ...]
    algebraic_position_error_m: float
    algebraic_orientation_error_rad: float

    @property
    def clipped(self) -> bool:
        return bool(self.clipped_dimensions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "measured_eef_pose_transition",
            "target_interface": "la4vla_libero_osc_pose",
            "translation_delta_m": list(self.translation_delta_m),
            "rotation_delta_rad": list(self.rotation_delta_rad),
            "gripper_command": self.gripper_command,
            "normalized_action": list(self.normalized_action),
            "scales": {
                "translation_m_per_unit": TRANSLATION_SCALE_M,
                "rotation_rad_per_unit": ROTATION_SCALE_RAD,
            },
            "rotation_composition": "R_after_times_transpose_R_before",
            "clipped": self.clipped,
            "clipped_dimensions": list(self.clipped_dimensions),
            "algebraic_round_trip": {
                "position_error_m": self.algebraic_position_error_m,
                "orientation_error_rad": self.algebraic_orientation_error_rad,
            },
            "physical_replay_status": "not_run",
            "replay_verified": False,
        }


def adapt_pose_transition(
    *,
    position_before_m: Sequence[Any],
    quaternion_before_wxyz: Sequence[Any],
    position_after_m: Sequence[Any],
    quaternion_after_wxyz: Sequence[Any],
    gripper_command: Any,
    allow_clipping: bool = False,
) -> OscPoseAdapterResult:
    """Convert one measured EEF transition to a normalized OSC command.

    By default an out-of-range action is rejected.  Explicit clipping keeps
    the altered dimensions in the result and therefore cannot be mistaken for
    the original request or for a verified physical replay.
    """

    before_pos = _finite_vector(position_before_m, 3, "position_before_m")
    after_pos = _finite_vector(position_after_m, 3, "position_after_m")
    before_quat = _normalize_quat_wxyz(
        quaternion_before_wxyz, "quaternion_before_wxyz"
    )
    after_quat = _normalize_quat_wxyz(
        quaternion_after_wxyz, "quaternion_after_wxyz"
    )
    if isinstance(gripper_command, bool):
        raise OscAdapterError(
            "gripper_command must be finite", code="non_finite", field="gripper_command"
        )
    try:
        gripper = float(gripper_command)
    except (TypeError, ValueError, OverflowError) as error:
        raise OscAdapterError(
            "gripper_command must be finite", code="non_finite", field="gripper_command"
        ) from error
    if not math.isfinite(gripper):
        raise OscAdapterError(
            "gripper_command must be finite", code="non_finite", field="gripper_command"
        )
    if not -1.0 <= gripper <= 1.0:
        raise OscAdapterError(
            "gripper_command must be in [-1, 1]",
            code="gripper_range",
            field="gripper_command",
        )

    translation = tuple(after - before for before, after in zip(before_pos, after_pos))
    # RoboSuite applies R_error @ R_current, hence R_error = R_after @ R_before.T.
    relative_quat = _quat_multiply_wxyz(after_quat, _quat_inverse_wxyz(before_quat))
    rotation = _quat_to_rotvec_wxyz(relative_quat)
    raw = tuple(value / TRANSLATION_SCALE_M for value in translation) + tuple(
        value / ROTATION_SCALE_RAD for value in rotation
    ) + (gripper,)
    clipped_dimensions = tuple(index for index, value in enumerate(raw[:6]) if abs(value) > 1.0)
    if clipped_dimensions and not allow_clipping:
        raise OscAdapterError(
            f"normalized OSC command exceeds [-1, 1] in dimensions {clipped_dimensions}",
            code="command_out_of_range",
            field="normalized_action",
        )
    normalized = tuple(max(-1.0, min(1.0, value)) for value in raw[:6]) + (gripper,)
    applied_translation = tuple(value * TRANSLATION_SCALE_M for value in normalized[:3])
    applied_rotation = tuple(value * ROTATION_SCALE_RAD for value in normalized[3:6])
    reconstructed_pos = tuple(before + delta for before, delta in zip(before_pos, applied_translation))
    reconstructed_quat = _quat_multiply_wxyz(
        _rotvec_to_quat_wxyz(applied_rotation), before_quat
    )
    position_error = math.sqrt(
        sum((actual - expected) ** 2 for actual, expected in zip(reconstructed_pos, after_pos))
    )
    orientation_error = _orientation_error_rad(reconstructed_quat, after_quat)
    return OscPoseAdapterResult(
        translation_delta_m=translation,  # type: ignore[arg-type]
        rotation_delta_rad=rotation,
        gripper_command=gripper,
        normalized_action=normalized,  # type: ignore[arg-type]
        clipped_dimensions=clipped_dimensions,
        algebraic_position_error_m=position_error,
        algebraic_orientation_error_rad=orientation_error,
    )


__all__ = [
    "OscAdapterError",
    "OscPoseAdapterResult",
    "ROTATION_SCALE_RAD",
    "SCHEMA_VERSION",
    "TRANSLATION_SCALE_M",
    "adapt_pose_transition",
    "orientation_error_rad",
    "quaternion_to_rotvec_wxyz",
]
