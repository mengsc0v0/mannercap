"""Versioned RoboSuite/LIBERO/LeRobot action-interface contract.

The two action vectors used by the current code have the same width but a
different meaning:

* the RoboSuite teacher sends seven *absolute* Panda joint targets and one
  normalized gripper command;
* the LIBERO low-level controller receives seven joint displacements for one
  control command and one normalized gripper command;
* the existing LeRobot export candidate stores six EEF state differences and
  one raw gripper value.  That 7-D candidate is explicitly unverified and is
  not interchangeable with the 8-D low-level controller input.

This module is deliberately a pure Python boundary.  It does not import
NumPy, RoboSuite, LIBERO, or a controller.  In particular, it never derives a
missing ``dt``, current joint state, joint order, or gripper sign convention.
The strict conversion helper therefore fails unless all of those facts are
provided explicitly.  A caller may opt into a lossy conversion, but the
unexpressed teacher fields remain visible in the returned object.

The round-trip candidate schema is a frozen *proposal* for a later
export/import check.  Constructing a candidate does not execute an environment
and does not establish replay success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Final, Optional


ACTION_INTERFACE_SCHEMA_VERSION: Final[str] = "synthesis_v2.action_interface.v1"
ROUND_TRIP_CANDIDATE_SCHEMA_VERSION: Final[str] = (
    "synthesis_v2.action_round_trip_candidate.v1"
)
# Convenient aliases used by the other v2 contracts.
SCHEMA_VERSION: Final[str] = ACTION_INTERFACE_SCHEMA_VERSION

JOINT_COUNT: Final[int] = 7
TEACHER_ACTION_DIM: Final[int] = 8
STUDENT_ACTION_DIM: Final[int] = 8
LEROBOT_EXPORT_ACTION_DIM: Final[int] = 7
LA4VLA_OSC_POSE_ACTION_DIM: Final[int] = 7
JOINT_ORDER: Final[tuple[str, ...]] = tuple(f"robot0_joint{i}" for i in range(1, 8))
JOINT_FRAME: Final[str] = "robot_joint_coordinates"
GRIPPER_SIGN: Final[str] = "normalized_command:-1=open,+1=close"
TIMESTAMP_SEMANTICS: Final[str] = "transition_start_seconds"
DT_SEMANTICS: Final[str] = "measured_transition_duration_seconds"


class InterfaceId(str, Enum):
    """Stable identifiers for the source and target action interfaces."""

    ROBOSUITE_TEACHER = "robosuite_teacher"
    LIBERO_LOW_LEVEL_CONTROLLER = "libero_low_level_controller"
    LEROBOT_EXPORT_CANDIDATE = "lerobot_export_candidate"
    LA4VLA_LIBERO_OSC_POSE = "la4vla_libero_osc_pose"
    # Accepted as a legacy lookup spelling only.  It resolves to the
    # low-level controller contract, not to the current 7-D training export.
    LIBERO_STUDENT = "libero_student"
    ROUND_TRIP_CANDIDATE = "round_trip_candidate"


class ActionInterfaceError(ValueError):
    """Base error for an invalid action contract or action value."""

    def __init__(self, message: str, *, code: str = "invalid_action", field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


class ActionConversionError(ActionInterfaceError):
    """Raised when an action cannot be converted without an unsafe guess."""


@dataclass(frozen=True)
class ActionIssue:
    """One structural issue found while checking an action or candidate."""

    code: str
    field: str | None
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class ActionValidationReport:
    """Pure validation result; no simulator call is implied by ``valid``."""

    valid: bool
    interface_id: str | None
    action_dimension: int | None
    issues: tuple[ActionIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def has_code(self, code: str) -> bool:
        return code in self.codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "interface_id": self.interface_id,
            "action_dimension": self.action_dimension,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def raise_if_invalid(self) -> "ActionValidationReport":
        if not self.valid:
            issue = self.issues[0]
            raise ActionInterfaceError(issue.message, code=issue.code, field=issue.field)
        return self


def _interface_id(value: InterfaceId | str) -> str:
    if isinstance(value, InterfaceId):
        return value.value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ActionInterfaceError("interface_id must be a non-empty string", field="interface_id")


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ActionInterfaceError(f"{field} must be a finite number", code="non_numeric", field=field)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ActionInterfaceError(
            f"{field} must be a finite number", code="non_numeric", field=field
        ) from exc
    if not math.isfinite(number):
        raise ActionInterfaceError(f"{field} must be finite", code="non_finite", field=field)
    return number


def _vector(value: Any, *, dimension: int, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ActionInterfaceError(
            f"{field} must be a sequence of length {dimension}",
            code="invalid_vector",
            field=field,
        )
    if len(value) != dimension:
        raise ActionInterfaceError(
            f"{field} must have dimension {dimension}, got {len(value)}",
            code="dimension_mismatch",
            field=field,
        )
    return tuple(_finite(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _optional_time(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field=field)


@dataclass(frozen=True)
class ChannelSpec:
    """Description of one vector or side-channel with explicit semantics."""

    name: str
    dimension: int
    unit: str
    coordinate_frame: str
    timing: str
    sign_convention: str
    expressibility: str = "expressible"
    required: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ActionInterfaceError("channel name must be non-empty", field="name")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ActionInterfaceError("channel dimension must be a positive integer", field=self.name)
        for field_name, value in (
            ("unit", self.unit),
            ("coordinate_frame", self.coordinate_frame),
            ("timing", self.timing),
            ("sign_convention", self.sign_convention),
            ("expressibility", self.expressibility),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ActionInterfaceError(
                    f"channel {field_name} must be non-empty", field=self.name
                )
        if not isinstance(self.required, bool):
            raise ActionInterfaceError("channel required must be boolean", field=self.name)

    # Short aliases make schema inspection convenient without changing the
    # canonical serialized names.
    @property
    def dim(self) -> int:
        return self.dimension

    @property
    def frame(self) -> str:
        return self.coordinate_frame

    @property
    def time(self) -> str:
        return self.timing

    @property
    def is_expressible(self) -> bool:
        return self.expressibility == "expressible"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "unit": self.unit,
            "coordinate_frame": self.coordinate_frame,
            "timing": self.timing,
            "sign_convention": self.sign_convention,
            "expressibility": self.expressibility,
            "required": self.required,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ActionInterfaceSpec:
    """Versioned description of a concrete action vector."""

    schema_version: str
    interface_id: InterfaceId | str
    controller: str
    action_channels: tuple[ChannelSpec, ...]
    auxiliary_channels: tuple[ChannelSpec, ...] = ()
    command_timing: str = TIMESTAMP_SEMANTICS
    dt_semantics: str = DT_SEMANTICS
    dt_required: bool = True
    nominal_control_frequency_hz: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ActionInterfaceError("schema_version must be non-empty", field="schema_version")
        if not isinstance(self.controller, str) or not self.controller.strip():
            raise ActionInterfaceError("controller must be non-empty", field="controller")
        if not isinstance(self.dt_required, bool):
            raise ActionInterfaceError("dt_required must be boolean", field="dt_required")
        channels = tuple(self.action_channels)
        auxiliary = tuple(self.auxiliary_channels)
        if not channels:
            raise ActionInterfaceError("an action interface needs at least one channel", field="action_channels")
        names = [channel.name for channel in channels]
        if len(names) != len(set(names)):
            raise ActionInterfaceError("action channel names must be unique", field="action_channels")
        aux_names = [channel.name for channel in auxiliary]
        if len(aux_names) != len(set(aux_names)):
            raise ActionInterfaceError("auxiliary channel names must be unique", field="auxiliary_channels")
        if set(names) & set(aux_names):
            raise ActionInterfaceError("action and auxiliary channel names must be disjoint", field="channels")
        object.__setattr__(self, "action_channels", channels)
        object.__setattr__(self, "auxiliary_channels", auxiliary)
        if self.nominal_control_frequency_hz is not None:
            frequency = _finite(self.nominal_control_frequency_hz, field="nominal_control_frequency_hz")
            if frequency <= 0:
                raise ActionInterfaceError(
                    "nominal control frequency must be > 0", code="invalid_frequency", field="nominal_control_frequency_hz"
                )
            object.__setattr__(self, "nominal_control_frequency_hz", frequency)

    @property
    def action_dimension(self) -> int:
        return sum(channel.dimension for channel in self.action_channels)

    @property
    def action_dim(self) -> int:
        return self.action_dimension

    @property
    def dimension(self) -> int:
        return self.action_dimension

    @property
    def channels(self) -> tuple[ChannelSpec, ...]:
        return self.action_channels

    def channel(self, name: str) -> ChannelSpec:
        for channel in self.action_channels + self.auxiliary_channels:
            if channel.name == name:
                return channel
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface_id": _interface_id(self.interface_id),
            "controller": self.controller,
            "action_dimension": self.action_dimension,
            "action_channels": [channel.as_dict() for channel in self.action_channels],
            "auxiliary_channels": [channel.as_dict() for channel in self.auxiliary_channels],
            "timing": {
                "command": self.command_timing,
                "dt": self.dt_semantics,
                "dt_required": self.dt_required,
            },
            # This is documentation only.  It is intentionally never used as
            # a conversion default because the source recorder has no measured
            # dt in its current rows.
            "nominal_control_frequency_hz": self.nominal_control_frequency_hz,
            "notes": self.notes,
        }


_PROFILE_CHANNELS: Final[tuple[ChannelSpec, ...]] = (
    ChannelSpec(
        "v_max",
        1,
        "rad/s",
        JOINT_FRAME,
        "profile_for_following_waypoints",
        "positive_speed_bound",
        "teacher_only",
        False,
        "RoboSuite profile argument; not a per-step student action value.",
    ),
    ChannelSpec(
        "a_max",
        1,
        "rad/s^2",
        JOINT_FRAME,
        "profile_for_following_waypoints",
        "positive_acceleration_bound",
        "teacher_only",
        False,
        "RoboSuite profile argument; not represented in LIBERO's action vector.",
    ),
    ChannelSpec(
        "jerk_limit",
        1,
        "rad/s^3",
        JOINT_FRAME,
        "profile_for_following_waypoints",
        "positive_jerk_bound",
        "teacher_only",
        False,
        "Optional minimum-jerk profile selector/bound.",
    ),
    ChannelSpec(
        "kp_pos",
        1,
        "controller_gain",
        JOINT_FRAME,
        "controller_state_at_command",
        "nonnegative_controller_gain",
        "teacher_only",
        False,
        "Hidden joint position controller gain; physical impedance units are not inferred.",
    ),
    ChannelSpec(
        "kp_ori",
        1,
        "controller_gain",
        "controller_orientation_coordinates",
        "controller_state_at_command",
        "nonnegative_controller_gain",
        "teacher_only",
        False,
        "Accepted by the API for symmetry but not separately applied by the current joint controller.",
    ),
    ChannelSpec(
        "grip_force",
        1,
        "fraction",
        "gripper_controller",
        "controller_state_at_command",
        "0=no_force,1=maximum_requested_force",
        "teacher_only",
        False,
        "Normalized request; no calibrated Newton value is implied.",
    ),
    ChannelSpec(
        "grip_width",
        1,
        "m",
        "gripper_controller",
        "controller_state_at_command",
        "nonnegative_gap",
        "teacher_only",
        False,
        "Width overrides force in the current close_gripper_partial implementation.",
    ),
    ChannelSpec(
        "contact_force",
        3,
        "N?",
        "simulator_body_frame",
        "observation_diagnostic",
        "vector_direction_preserved",
        "diagnostic_only",
        False,
        "Not an action; cfrc_ext is not independently calibrated here.",
    ),
    ChannelSpec(
        "contact_torque",
        3,
        "N*m?",
        "simulator_body_frame",
        "observation_diagnostic",
        "vector_direction_preserved",
        "diagnostic_only",
        False,
        "Not an action; must remain separate from force.",
    ),
)


ROBOSUITE_TEACHER_INTERFACE: Final[ActionInterfaceSpec] = ActionInterfaceSpec(
    schema_version=ACTION_INTERFACE_SCHEMA_VERSION,
    interface_id=InterfaceId.ROBOSUITE_TEACHER,
    controller="RoboSuite JOINT_POSITION (input_type=absolute)",
    action_channels=(
        ChannelSpec(
            "joint_target",
            JOINT_COUNT,
            "rad",
            JOINT_FRAME,
            "transition_start_seconds",
            "signed_target_about_joint_zero",
            notes="Seven absolute target positions in JOINT_ORDER.",
        ),
        ChannelSpec(
            "gripper_command",
            1,
            "normalized_command",
            "gripper_controller",
            "transition_start_seconds",
            GRIPPER_SIGN,
            notes="External action value; internal opening fraction is 1=open and 0=closed.",
        ),
    ),
    auxiliary_channels=_PROFILE_CHANNELS,
    nominal_control_frequency_hz=20.0,
    notes=(
        "Teacher calls may generate profiled waypoints. v_max/a_max/jerk_limit, "
        "kp_pos, grip_force, and grip_width are side channels and are not in the vector."
    ),
)


LIBERO_LOW_LEVEL_INTERFACE: Final[ActionInterfaceSpec] = ActionInterfaceSpec(
    schema_version=ACTION_INTERFACE_SCHEMA_VERSION,
    interface_id=InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER,
    controller="LIBERO JOINT_POSITION wrapper with joint-delta command construction",
    action_channels=(
        ChannelSpec(
            "joint_delta",
            JOINT_COUNT,
            "rad",
            JOINT_FRAME,
            "transition_start_seconds",
            "signed_displacement_from_measured_joint_state",
            notes=(
                "One control-command displacement. The current wrapper forms "
                "(target-current)*control_freq; this is not an absolute target."
            ),
        ),
        ChannelSpec(
            "gripper_command",
            1,
            "normalized_command",
            "gripper_controller",
            "transition_start_seconds",
            GRIPPER_SIGN,
            notes="-1=open and +1=close, matching the documented current wrapper mapping.",
        ),
    ),
    auxiliary_channels=(),
    nominal_control_frequency_hz=20.0,
    notes=(
        "Low-level vector cannot express RoboSuite waypoint profile, controller kp, "
        "gripper force, or gripper width."
    ),
)


LEROBOT_EXPORT_CANDIDATE_INTERFACE: Final[ActionInterfaceSpec] = ActionInterfaceSpec(
    schema_version=ACTION_INTERFACE_SCHEMA_VERSION,
    interface_id=InterfaceId.LEROBOT_EXPORT_CANDIDATE,
    controller="none; existing unverified LeRobot export candidate",
    action_channels=(
        ChannelSpec(
            "eef_translation_state_difference",
            3,
            "m",
            "robot_base_frame",
            "difference_between_recorded_transition_endpoints",
            "end_minus_start",
            notes="Componentwise difference of recorded EEF positions; not a verified controller command.",
        ),
        ChannelSpec(
            "eef_axis_angle_state_difference",
            3,
            "rad",
            "robot_base_frame_axis_angle_coordinates",
            "difference_between_recorded_transition_endpoints",
            "componentwise_end_minus_start",
            notes=(
                "Componentwise difference of axis-angle state coordinates. It is not a "
                "validated relative-rotation action or a joint-space command."
            ),
        ),
        ChannelSpec(
            "raw_gripper_value",
            1,
            "source_value_unverified",
            "recorded_source_action",
            "transition_start_seconds",
            "preserved_without_semantic_promotion",
            notes="Copied from raw_action[-1]; ground-truth action semantics remain unverified.",
        ),
    ),
    auxiliary_channels=(),
    nominal_control_frequency_hz=None,
    notes=(
        "This describes the current 7-D lerobot_action_candidate export only. "
        "validated_as_ground_truth is false in the source exporter."
    ),
)


LA4VLA_LIBERO_OSC_POSE_INTERFACE: Final[ActionInterfaceSpec] = ActionInterfaceSpec(
    schema_version=ACTION_INTERFACE_SCHEMA_VERSION,
    interface_id=InterfaceId.LA4VLA_LIBERO_OSC_POSE,
    controller="LIBERO RoboSuite OSC_POSE (control_delta=true)",
    action_channels=(
        ChannelSpec(
            "osc_translation_delta_normalized",
            3,
            "normalized_command",
            "OSC_POSE_controller_frame",
            "transition_start_seconds",
            "[-1,+1] scaled componentwise to [-0.05,+0.05] m",
            notes="The evaluated LA4VLA policy emits controller commands, not measured EEF state differences.",
        ),
        ChannelSpec(
            "osc_rotation_delta_normalized",
            3,
            "normalized_command",
            "OSC_POSE_controller_axis_angle_frame",
            "transition_start_seconds",
            "[-1,+1] scaled componentwise to [-0.5,+0.5] rad",
            notes="Exact dataset-to-controller rotation semantics remain pending a physical adapter replay.",
        ),
        ChannelSpec(
            "gripper_command",
            1,
            "normalized_command",
            "Panda_gripper_controller",
            "transition_start_seconds",
            GRIPPER_SIGN,
            notes="The installed PandaGripper maps -1 to open and +1 to closed.",
        ),
    ),
    auxiliary_channels=(),
    nominal_control_frequency_hz=20.0,
    notes=(
        "Audited fixed-student candidate used by eval_libero_goal.py. Its 7-D vector "
        "is not interchangeable with the 7-D MannerCAP LeRobot state-difference export."
    ),
)

# Spelling and role aliases kept intentionally small for callers/readers.
ROBOSUITE_TEACHER = ROBOSUITE_TEACHER_INTERFACE
ROBO_SUITE_TEACHER_INTERFACE = ROBOSUITE_TEACHER_INTERFACE
TEACHER_INTERFACE = ROBOSUITE_TEACHER_INTERFACE
LIBERO_STUDENT_INTERFACE = LIBERO_LOW_LEVEL_INTERFACE
LIBERO_STUDENT = LIBERO_LOW_LEVEL_INTERFACE
STUDENT_INTERFACE = LIBERO_LOW_LEVEL_INTERFACE


def get_action_interface(interface: ActionInterfaceSpec | InterfaceId | str) -> ActionInterfaceSpec:
    """Resolve a concrete interface specification.

    ``libero_student`` remains accepted as a compatibility spelling for the
    8-D low-level controller input.  It must not be interpreted as the 7-D
    LeRobot export candidate used by the current dataset converter.
    """

    if isinstance(interface, ActionInterfaceSpec):
        return interface
    key = _interface_id(interface)
    known = {
        InterfaceId.ROBOSUITE_TEACHER.value: ROBOSUITE_TEACHER_INTERFACE,
        InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value: LIBERO_LOW_LEVEL_INTERFACE,
        InterfaceId.LIBERO_STUDENT.value: LIBERO_LOW_LEVEL_INTERFACE,
        InterfaceId.LEROBOT_EXPORT_CANDIDATE.value: LEROBOT_EXPORT_CANDIDATE_INTERFACE,
        InterfaceId.LA4VLA_LIBERO_OSC_POSE.value: LA4VLA_LIBERO_OSC_POSE_INTERFACE,
    }
    try:
        return known[key]
    except KeyError as exc:
        raise ActionInterfaceError(
            f"unknown action interface {key!r}", code="unknown_interface", field="interface_id"
        ) from exc


def validate_action(
    values: Sequence[Any],
    interface: ActionInterfaceSpec | InterfaceId | str,
    *,
    timestamp_s: Any = None,
    dt_s: Any = None,
    require_timestamp: bool = True,
    require_dt: bool | None = None,
) -> ActionValidationReport:
    """Validate a numeric action vector and its timing metadata.

    ``require_dt`` defaults to the selected interface's contract.  Callers
    may set it to ``False`` only for an explicitly incomplete source record;
    conversion still rejects that record unless a real ``dt_s`` is supplied.
    """

    issues: list[ActionIssue] = []
    try:
        spec = get_action_interface(interface)
    except ActionInterfaceError as exc:
        return ActionValidationReport(
            False,
            str(interface) if isinstance(interface, str) else None,
            None,
            (ActionIssue(exc.code, exc.field, str(exc)),),
        )

    expected_dt = spec.dt_required if require_dt is None else require_dt
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        issues.append(ActionIssue("invalid_vector", "action", "action must be a numeric sequence"))
    elif len(values) != spec.action_dimension:
        issues.append(
            ActionIssue(
                "dimension_mismatch",
                "action",
                f"{_interface_id(spec.interface_id)} action must have dimension {spec.action_dimension}, got {len(values)}",
            )
        )
    else:
        for index, value in enumerate(values):
            try:
                _finite(value, field=f"action[{index}]")
            except ActionInterfaceError as exc:
                issues.append(ActionIssue(exc.code, exc.field, str(exc)))

        gripper = float(values[-1]) if not issues else None
        if gripper is not None and not -1.0 <= gripper <= 1.0:
            issues.append(
                ActionIssue(
                    "gripper_range",
                    "action[-1]",
                    f"gripper command must be in [-1, 1], got {gripper}",
                )
            )

    if require_timestamp and timestamp_s is None:
        issues.append(ActionIssue("missing_timestamp", "timestamp_s", "action timestamp is required"))
    elif timestamp_s is not None:
        try:
            _finite(timestamp_s, field="timestamp_s")
        except ActionInterfaceError as exc:
            issues.append(ActionIssue(exc.code, exc.field, str(exc)))

    if expected_dt and dt_s is None:
        issues.append(ActionIssue("missing_dt", "dt_s", "measured action dt is required; no nominal frequency is substituted"))
    elif dt_s is not None:
        try:
            dt = _finite(dt_s, field="dt_s")
            if dt <= 0:
                issues.append(ActionIssue("invalid_dt", "dt_s", f"dt_s must be > 0, got {dt}"))
        except ActionInterfaceError as exc:
            issues.append(ActionIssue(exc.code, exc.field, str(exc)))

    return ActionValidationReport(
        not issues,
        _interface_id(spec.interface_id),
        spec.action_dimension,
        tuple(issues),
    )


validate_action_vector = validate_action


@dataclass(frozen=True)
class HiddenTeacherControls:
    """Teacher side channels that the student action vector cannot encode."""

    v_max_rad_s: float | None = None
    a_max_rad_s2: float | None = None
    jerk_limit_rad_s3: float | None = None
    kp_pos: float | None = None
    kp_ori: float | None = None
    grip_force_fraction: float | None = None
    grip_width_m: float | None = None

    def __post_init__(self) -> None:
        positive = (
            ("v_max_rad_s", self.v_max_rad_s),
            ("a_max_rad_s2", self.a_max_rad_s2),
            ("jerk_limit_rad_s3", self.jerk_limit_rad_s3),
            ("kp_pos", self.kp_pos),
            ("kp_ori", self.kp_ori),
            ("grip_width_m", self.grip_width_m),
        )
        for field, value in positive:
            if value is not None:
                number = _finite(value, field=field)
                if number < 0:
                    raise ActionInterfaceError(f"{field} must be non-negative", code="negative_value", field=field)
                object.__setattr__(self, field, number)
        if self.grip_force_fraction is not None:
            force = _finite(self.grip_force_fraction, field="grip_force_fraction")
            if not 0.0 <= force <= 1.0:
                raise ActionInterfaceError(
                    "grip_force_fraction must be in [0, 1]", code="force_range", field="grip_force_fraction"
                )
            object.__setattr__(self, "grip_force_fraction", force)

    @property
    def force(self) -> float | None:
        return self.grip_force_fraction

    @property
    def width(self) -> float | None:
        return self.grip_width_m

    @property
    def has_hidden_controls(self) -> bool:
        return bool(self.unexpressed_fields)

    @property
    def unexpressed_fields(self) -> tuple[str, ...]:
        fields: list[str] = []
        for name, value in (
            ("v_max", self.v_max_rad_s),
            ("a_max", self.a_max_rad_s2),
            ("jerk_limit", self.jerk_limit_rad_s3),
            ("kp_pos", self.kp_pos),
            ("kp_ori", self.kp_ori),
            ("grip_force", self.grip_force_fraction),
            ("grip_width", self.grip_width_m),
        ):
            if value is not None:
                fields.append(name)
        return tuple(fields)

    def as_dict(self) -> dict[str, Any]:
        return {
            "v_max": self.v_max_rad_s,
            "a_max": self.a_max_rad_s2,
            "jerk_limit": self.jerk_limit_rad_s3,
            "kp_pos": self.kp_pos,
            "kp_ori": self.kp_ori,
            "grip_force": self.grip_force_fraction,
            "grip_width": self.grip_width_m,
            "units": {
                "v_max": "rad/s",
                "a_max": "rad/s^2",
                "jerk_limit": "rad/s^3",
                "kp_pos": "controller_gain",
                "kp_ori": "controller_gain",
                "grip_force": "fraction",
                "grip_width": "m",
            },
            "unexpressed_fields": list(self.unexpressed_fields),
        }


@dataclass(frozen=True)
class TeacherAction:
    """One RoboSuite teacher command plus optional measured context."""

    joint_target_rad: Sequence[Any]
    gripper_command: Any
    timestamp_s: float | None = None
    dt_s: float | None = None
    current_joint_rad: Sequence[Any] | None = None
    hidden_controls: HiddenTeacherControls | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_target_rad",
            _vector(self.joint_target_rad, dimension=JOINT_COUNT, field="joint_target_rad"),
        )
        grip = _finite(self.gripper_command, field="gripper_command")
        if not -1.0 <= grip <= 1.0:
            raise ActionInterfaceError(
                "gripper_command must be in [-1, 1]", code="gripper_range", field="gripper_command"
            )
        object.__setattr__(self, "gripper_command", grip)
        object.__setattr__(self, "timestamp_s", _optional_time(self.timestamp_s, field="timestamp_s"))
        dt = _optional_time(self.dt_s, field="dt_s")
        if dt is not None and dt <= 0:
            raise ActionInterfaceError("dt_s must be > 0", code="invalid_dt", field="dt_s")
        object.__setattr__(self, "dt_s", dt)
        if self.current_joint_rad is not None:
            object.__setattr__(
                self,
                "current_joint_rad",
                _vector(self.current_joint_rad, dimension=JOINT_COUNT, field="current_joint_rad"),
            )
        if isinstance(self.hidden_controls, Mapping):
            object.__setattr__(self, "hidden_controls", HiddenTeacherControls(**dict(self.hidden_controls)))
        elif self.hidden_controls is not None and not isinstance(self.hidden_controls, HiddenTeacherControls):
            raise ActionInterfaceError(
                "hidden_controls must be HiddenTeacherControls or a mapping",
                code="invalid_hidden_controls",
                field="hidden_controls",
            )

    @property
    def vector(self) -> tuple[float, ...]:
        return self.joint_target_rad + (self.gripper_command,)

    @property
    def values(self) -> tuple[float, ...]:
        return self.vector

    @property
    def joint_targets(self) -> tuple[float, ...]:
        return self.joint_target_rad

    @property
    def gripper(self) -> float:
        return self.gripper_command

    @property
    def profile(self) -> HiddenTeacherControls | None:
        return self.hidden_controls

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_INTERFACE_SCHEMA_VERSION,
            "interface_id": InterfaceId.ROBOSUITE_TEACHER.value,
            "action": list(self.vector),
            "action_dimension": TEACHER_ACTION_DIM,
            "channels": {
                "joint_target": {"values": list(self.joint_target_rad), "unit": "rad", "frame": JOINT_FRAME},
                "gripper_command": {
                    "value": self.gripper_command,
                    "unit": "normalized_command",
                    "sign": GRIPPER_SIGN,
                },
            },
            "timing": {"timestamp_s": self.timestamp_s, "dt_s": self.dt_s},
            "current_joint_rad": list(self.current_joint_rad) if self.current_joint_rad is not None else None,
            "hidden_controls": self.hidden_controls.as_dict() if self.hidden_controls is not None else None,
        }

    @classmethod
    def from_vector(cls, values: Sequence[Any], **kwargs: Any) -> "TeacherAction":
        vector = _vector(values, dimension=TEACHER_ACTION_DIM, field="teacher_action")
        return cls(vector[:JOINT_COUNT], vector[-1], **kwargs)


RobosuiteTeacherAction = TeacherAction
RoboSuiteTeacherAction = TeacherAction


@dataclass(frozen=True)
class StudentAction:
    """One candidate LIBERO low-level command: seven joint deltas plus gripper."""

    joint_delta_rad: Sequence[Any]
    gripper_command: Any
    timestamp_s: float | None = None
    dt_s: float | None = None
    unexpressed_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_delta_rad",
            _vector(self.joint_delta_rad, dimension=JOINT_COUNT, field="joint_delta_rad"),
        )
        grip = _finite(self.gripper_command, field="gripper_command")
        if not -1.0 <= grip <= 1.0:
            raise ActionInterfaceError(
                "gripper_command must be in [-1, 1]", code="gripper_range", field="gripper_command"
            )
        object.__setattr__(self, "gripper_command", grip)
        object.__setattr__(self, "timestamp_s", _optional_time(self.timestamp_s, field="timestamp_s"))
        dt = _optional_time(self.dt_s, field="dt_s")
        if dt is not None and dt <= 0:
            raise ActionInterfaceError("dt_s must be > 0", code="invalid_dt", field="dt_s")
        object.__setattr__(self, "dt_s", dt)
        if isinstance(self.unexpressed_fields, str) or not isinstance(self.unexpressed_fields, Sequence):
            raise ActionInterfaceError(
                "unexpressed_fields must be a sequence", code="invalid_metadata", field="unexpressed_fields"
            )
        object.__setattr__(self, "unexpressed_fields", tuple(str(item) for item in self.unexpressed_fields))

    @property
    def vector(self) -> tuple[float, ...]:
        return self.joint_delta_rad + (self.gripper_command,)

    @property
    def values(self) -> tuple[float, ...]:
        return self.vector

    @property
    def joint_delta(self) -> tuple[float, ...]:
        return self.joint_delta_rad

    @property
    def gripper(self) -> float:
        return self.gripper_command

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_INTERFACE_SCHEMA_VERSION,
            "interface_id": InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value,
            "action": list(self.vector),
            "action_dimension": STUDENT_ACTION_DIM,
            "channels": {
                "joint_delta": {"values": list(self.joint_delta_rad), "unit": "rad", "frame": JOINT_FRAME},
                "gripper_command": {
                    "value": self.gripper_command,
                    "unit": "normalized_command",
                    "sign": GRIPPER_SIGN,
                },
            },
            "timing": {"timestamp_s": self.timestamp_s, "dt_s": self.dt_s},
            "unexpressed_fields": list(self.unexpressed_fields),
        }

    @classmethod
    def from_vector(cls, values: Sequence[Any], **kwargs: Any) -> "StudentAction":
        vector = _vector(values, dimension=STUDENT_ACTION_DIM, field="student_action")
        return cls(vector[:JOINT_COUNT], vector[-1], **kwargs)


LiberoStudentAction = StudentAction
TeacherProfile = HiddenTeacherControls


def _coalesce(value: Any, alias: Any, *, field: str) -> Any:
    if value is not None and alias is not None:
        try:
            if value != alias:
                raise ActionConversionError(
                    f"conflicting values supplied for {field}", code="conflicting_argument", field=field
                )
        except ValueError:
            # Sequence-like values are compared by the caller's explicit
            # vector validator; keep this helper scalar-only.
            pass
    return value if value is not None else alias


def convert_teacher_to_student(
    teacher: TeacherAction | Sequence[Any],
    *,
    current_joint_rad: Sequence[Any] | None = None,
    current_joints_rad: Sequence[Any] | None = None,
    timestamp_s: Any = None,
    dt_s: Any = None,
    dt: Any = None,
    source_interface: ActionInterfaceSpec | InterfaceId | str = ROBOSUITE_TEACHER_INTERFACE,
    target_interface: ActionInterfaceSpec | InterfaceId | str = LIBERO_STUDENT_INTERFACE,
    allow_unexpressed: bool = False,
) -> StudentAction:
    """Convert one absolute teacher target to one student joint delta.

    Strict prerequisites are intentional:

    * current measured seven-joint state;
    * transition ``timestamp_s`` and measured positive ``dt_s``;
    * matching Panda joint order and gripper sign convention;
    * no teacher-only profile/kp/force/width side channel unless the caller
      explicitly opts into a visibly lossy result.

    ``dt`` is a spelling alias for ``dt_s``.  It is still required explicitly;
    no 20 Hz/0.05 s default is used.
    """

    source = get_action_interface(source_interface)
    target = get_action_interface(target_interface)
    if _interface_id(source.interface_id) != InterfaceId.ROBOSUITE_TEACHER.value:
        raise ActionConversionError(
            "teacher-to-student conversion requires the RoboSuite teacher source interface",
            code="interface_mismatch",
            field="source_interface",
        )
    if _interface_id(target.interface_id) == InterfaceId.LEROBOT_EXPORT_CANDIDATE.value:
        raise ActionConversionError(
            "the 7-D LeRobot EEF state-difference export is not a verified controller action",
            code="unverified_export_semantics",
            field="target_interface",
        )
    if _interface_id(target.interface_id) == InterfaceId.LA4VLA_LIBERO_OSC_POSE.value:
        raise ActionConversionError(
            "the LA4VLA OSC_POSE adapter has not passed physical controller replay",
            code="adapter_not_verified",
            field="target_interface",
        )
    if _interface_id(target.interface_id) != InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value:
        raise ActionConversionError(
            "teacher-to-student conversion requires the LIBERO student target interface",
            code="interface_mismatch",
            field="target_interface",
        )
    if source.schema_version != target.schema_version:
        raise ActionConversionError(
            "source and target action schema versions differ",
            code="schema_version_mismatch",
            field="schema_version",
        )

    if isinstance(teacher, TeacherAction):
        teacher_action = teacher
    else:
        teacher_action = TeacherAction.from_vector(teacher)

    current = _coalesce(current_joint_rad, current_joints_rad, field="current_joint_rad")
    if current is None:
        current = teacher_action.current_joint_rad
    # Resolve timing before state conversion so an absent dt gets a stable,
    # actionable error instead of a later arithmetic error.
    timestamp = timestamp_s if timestamp_s is not None else teacher_action.timestamp_s
    duration = _coalesce(dt_s, dt, field="dt_s")
    if timestamp is None:
        raise ActionConversionError(
            "teacher action has no transition timestamp; provide timestamp_s explicitly",
            code="missing_timestamp",
            field="timestamp_s",
        )
    timestamp = _finite(timestamp, field="timestamp_s")
    if duration is None:
        duration = teacher_action.dt_s
    if duration is None:
        raise ActionConversionError(
            "teacher action has no measured dt; provide dt_s explicitly; nominal 20 Hz is not substituted",
            code="missing_dt",
            field="dt_s",
        )
    duration = _finite(duration, field="dt_s")
    if duration <= 0:
        raise ActionConversionError(
            f"dt_s must be > 0, got {duration}", code="invalid_dt", field="dt_s"
        )
    if current is None:
        raise ActionConversionError(
            "absolute teacher target cannot become a delta without measured current_joint_rad",
            code="missing_state",
            field="current_joint_rad",
        )
    current_vector = _vector(current, dimension=JOINT_COUNT, field="current_joint_rad")

    hidden = teacher_action.hidden_controls
    omitted = hidden.unexpressed_fields if hidden is not None else ()
    if omitted and not allow_unexpressed:
        raise ActionConversionError(
            "teacher-only controls are not expressible by the LIBERO student vector: "
            + ", ".join(omitted),
            code="unexpressed_teacher_field",
            field=omitted[0],
        )

    # The two interfaces document the same order and sign.  The checks are
    # kept explicit so a future adapter cannot silently reverse or reorder it.
    source_joint = source.channel("joint_target")
    target_joint = target.channel("joint_delta")
    if source_joint.dimension != JOINT_COUNT or target_joint.dimension != JOINT_COUNT:
        raise ActionConversionError(
            "joint action channel dimensions must both be seven",
            code="dimension_mismatch",
            field="joint_target",
        )
    if source_joint.unit != target_joint.unit:
        raise ActionConversionError(
            "joint action units differ; no implicit unit conversion is allowed",
            code="unit_mismatch",
            field="joint_target",
        )
    if source_joint.coordinate_frame != target_joint.coordinate_frame:
        raise ActionConversionError(
            "joint coordinate frames differ; provide an explicit joint transform",
            code="coordinate_mismatch",
            field="joint_target",
        )
    source_grip = source.channel("gripper_command")
    target_grip = target.channel("gripper_command")
    if source_grip.dimension != 1 or target_grip.dimension != 1:
        raise ActionConversionError(
            "gripper action channel dimensions must both be one",
            code="dimension_mismatch",
            field="gripper_command",
        )
    if source_grip.unit != target_grip.unit:
        raise ActionConversionError(
            "gripper action units differ; no implicit unit conversion is allowed",
            code="unit_mismatch",
            field="gripper_command",
        )
    if source_grip.sign_convention != target_grip.sign_convention:
        raise ActionConversionError(
            "gripper sign conventions differ; no implicit sign flip is allowed",
            code="sign_mismatch",
            field="gripper_command",
        )
    delta = tuple(target_value - current_value for target_value, current_value in zip(teacher_action.joint_target_rad, current_vector))
    return StudentAction(
        delta,
        teacher_action.gripper_command,
        timestamp_s=timestamp,
        dt_s=duration,
        unexpressed_fields=omitted,
    )


teacher_to_student = convert_teacher_to_student
convert_teacher_action_to_student = convert_teacher_to_student


def convert_student_to_teacher(*args: Any, **kwargs: Any) -> TeacherAction:
    """Reject the reverse conversion unless a future explicit adapter exists.

    A student delta alone lacks the controller state and teacher profile that
    define a RoboSuite absolute command.  Keeping this direction fail-closed
    prevents a caller from mistaking a mathematically reconstructed target for
    a verified teacher replay.
    """

    raise ActionConversionError(
        "LIBERO student deltas are not an invertible RoboSuite teacher command; "
        "an explicit controller/state/profile adapter is required",
        code="non_invertible_conversion",
        field="target_interface",
    )


def convert_action(
    action: Any,
    source_interface: ActionInterfaceSpec | InterfaceId | str,
    target_interface: ActionInterfaceSpec | InterfaceId | str,
    **kwargs: Any,
) -> StudentAction | TeacherAction:
    """Dispatch only the one currently specified direction; fail otherwise."""

    source = get_action_interface(source_interface)
    target = get_action_interface(target_interface)
    pair = (_interface_id(source.interface_id), _interface_id(target.interface_id))
    if pair == (InterfaceId.ROBOSUITE_TEACHER.value, InterfaceId.LEROBOT_EXPORT_CANDIDATE.value):
        raise ActionConversionError(
            "the 7-D LeRobot EEF state-difference export is not a verified controller action",
            code="unverified_export_semantics",
            field="target_interface",
        )
    if pair == (InterfaceId.ROBOSUITE_TEACHER.value, InterfaceId.LA4VLA_LIBERO_OSC_POSE.value):
        raise ActionConversionError(
            "the LA4VLA OSC_POSE adapter has not passed physical controller replay",
            code="adapter_not_verified",
            field="target_interface",
        )
    if pair == (InterfaceId.ROBOSUITE_TEACHER.value, InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value):
        return convert_teacher_to_student(
            action,
            source_interface=source,
            target_interface=target,
            **kwargs,
        )
    if pair == (InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value, InterfaceId.ROBOSUITE_TEACHER.value):
        return convert_student_to_teacher(action, source_interface=source, target_interface=target, **kwargs)
    raise ActionConversionError(
        f"unsupported action-interface conversion {pair[0]!r} -> {pair[1]!r}",
        code="interface_mismatch",
        field="target_interface",
    )


@dataclass(frozen=True)
class RoundTripCandidateSchema:
    """Frozen record schema reserved for a later export/import round-trip."""

    schema_version: str = ROUND_TRIP_CANDIDATE_SCHEMA_VERSION
    source_interface: str = InterfaceId.ROBOSUITE_TEACHER.value
    target_interface: str = InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value
    action_dimension: int = STUDENT_ACTION_DIM
    required_fields: tuple[str, ...] = (
        "timestamp_s",
        "dt_s",
        "state_before.joint_position_rad",
        "teacher.action",
        "student.action",
        "source_interface_schema_version",
        "target_interface_schema_version",
        "unexpressed_fields",
        "replay_status",
    )
    replay_status_on_creation: str = "not_run"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_interface": self.source_interface,
            "target_interface": self.target_interface,
            "action_dimension": self.action_dimension,
            "required_fields": list(self.required_fields),
            "replay_status_on_creation": self.replay_status_on_creation,
            "purpose": "future round-trip fixture only",
        }

    to_dict = as_dict


ROUND_TRIP_CANDIDATE_SCHEMA: Final[RoundTripCandidateSchema] = RoundTripCandidateSchema()
ROUND_TRIP_SCHEMA = ROUND_TRIP_CANDIDATE_SCHEMA


@dataclass(frozen=True)
class RoundTripCandidate:
    """One candidate pair; construction never claims that replay was run."""

    timestamp_s: Any
    dt_s: Any
    state_before_joint_rad: Sequence[Any]
    teacher_action: TeacherAction | Sequence[Any]
    student_action: StudentAction | Sequence[Any]
    unexpressed_fields: tuple[str, ...] = ()
    replay_status: str = "not_run"
    source_interface_schema_version: str = ACTION_INTERFACE_SCHEMA_VERSION
    target_interface_schema_version: str = ACTION_INTERFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp_s, field="timestamp_s")
        duration = _finite(self.dt_s, field="dt_s")
        if duration <= 0:
            raise ActionInterfaceError("dt_s must be > 0", code="invalid_dt", field="dt_s")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "dt_s", duration)
        object.__setattr__(
            self,
            "state_before_joint_rad",
            _vector(self.state_before_joint_rad, dimension=JOINT_COUNT, field="state_before_joint_rad"),
        )
        teacher = self.teacher_action if isinstance(self.teacher_action, TeacherAction) else TeacherAction.from_vector(self.teacher_action)
        student = self.student_action if isinstance(self.student_action, StudentAction) else StudentAction.from_vector(self.student_action)
        if teacher.timestamp_s is not None and not math.isclose(teacher.timestamp_s, timestamp, abs_tol=1e-12):
            raise ActionInterfaceError("teacher timestamp differs from candidate timestamp", code="timestamp_mismatch", field="teacher_action")
        if student.timestamp_s is not None and not math.isclose(student.timestamp_s, timestamp, abs_tol=1e-12):
            raise ActionInterfaceError("student timestamp differs from candidate timestamp", code="timestamp_mismatch", field="student_action")
        if teacher.dt_s is not None and not math.isclose(teacher.dt_s, duration, abs_tol=1e-12):
            raise ActionInterfaceError("teacher dt differs from candidate dt", code="dt_mismatch", field="teacher_action")
        if student.dt_s is not None and not math.isclose(student.dt_s, duration, abs_tol=1e-12):
            raise ActionInterfaceError("student dt differs from candidate dt", code="dt_mismatch", field="student_action")
        expected = tuple(target - current for target, current in zip(teacher.joint_target_rad, self.state_before_joint_rad))
        if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(expected, student.joint_delta_rad)):
            raise ActionInterfaceError(
                "student delta does not equal teacher target minus recorded state",
                code="conversion_mismatch",
                field="student_action",
            )
        if teacher.gripper_command != student.gripper_command:
            raise ActionInterfaceError(
                "teacher and student gripper command signs/values differ",
                code="sign_mismatch",
                field="gripper_command",
            )
        if self.replay_status != "not_run":
            raise ActionInterfaceError(
                "new candidates must remain marked not_run until an external replay audit records evidence",
                code="candidate_status",
                field="replay_status",
            )
        fields = tuple(self.unexpressed_fields)
        object.__setattr__(self, "teacher_action", teacher)
        object.__setattr__(self, "student_action", student)
        object.__setattr__(self, "unexpressed_fields", fields)

    @property
    def schema_version(self) -> str:
        return ROUND_TRIP_CANDIDATE_SCHEMA_VERSION

    @property
    def candidate_only(self) -> bool:
        return True

    @property
    def replay_verified(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_only": True,
            "replay_verified": False,
            "replay_status": self.replay_status,
            "timestamp_s": self.timestamp_s,
            "dt_s": self.dt_s,
            "timing": {"timestamp_semantics": TIMESTAMP_SEMANTICS, "dt_semantics": DT_SEMANTICS},
            "state_before": {
                "joint_position_rad": list(self.state_before_joint_rad),
                "unit": "rad",
                "frame": JOINT_FRAME,
                "joint_order": list(JOINT_ORDER),
            },
            "teacher": self.teacher_action.as_dict(),
            "student": self.student_action.as_dict(),
            "source_interface_schema_version": self.source_interface_schema_version,
            "target_interface_schema_version": self.target_interface_schema_version,
            "unexpressed_fields": list(self.unexpressed_fields),
        }

    to_dict = as_dict


def build_round_trip_candidate(
    teacher: TeacherAction | Sequence[Any],
    *,
    current_joint_rad: Sequence[Any] | None = None,
    current_joints_rad: Sequence[Any] | None = None,
    timestamp_s: Any = None,
    dt_s: Any = None,
    dt: Any = None,
    allow_unexpressed: bool = False,
) -> RoundTripCandidate:
    """Build the frozen candidate record after strict conversion checks."""

    student = convert_teacher_to_student(
        teacher,
        current_joint_rad=current_joint_rad,
        current_joints_rad=current_joints_rad,
        timestamp_s=timestamp_s,
        dt_s=dt_s,
        dt=dt,
        allow_unexpressed=allow_unexpressed,
    )
    teacher_action = teacher if isinstance(teacher, TeacherAction) else TeacherAction.from_vector(teacher)
    current = current_joint_rad if current_joint_rad is not None else current_joints_rad
    if current is None:
        current = teacher_action.current_joint_rad
    if current is None:
        # Conversion above already raises; this guard keeps type checkers and
        # future changes from allowing a partially formed candidate.
        raise ActionConversionError("candidate needs current joint state", code="missing_state", field="current_joint_rad")
    timestamp_value = timestamp_s if timestamp_s is not None else teacher_action.timestamp_s
    duration_value = dt_s if dt_s is not None else dt
    if duration_value is None:
        duration_value = teacher_action.dt_s
    return RoundTripCandidate(
        timestamp_s=timestamp_value,
        dt_s=duration_value,
        state_before_joint_rad=current,
        teacher_action=teacher_action,
        student_action=student,
        unexpressed_fields=student.unexpressed_fields,
    )


def validate_candidate(candidate: RoundTripCandidate | Mapping[str, Any]) -> ActionValidationReport:
    """Validate a candidate object; serialized mappings are checked minimally."""

    if isinstance(candidate, RoundTripCandidate):
        return ActionValidationReport(True, InterfaceId.ROUND_TRIP_CANDIDATE.value, STUDENT_ACTION_DIM)
    if not isinstance(candidate, Mapping):
        return ActionValidationReport(
            False,
            InterfaceId.ROUND_TRIP_CANDIDATE.value,
            None,
            (ActionIssue("invalid_candidate", None, "candidate must be a RoundTripCandidate or mapping"),),
        )
    issues: list[ActionIssue] = []
    # ``as_dict`` uses nested objects for readability while the frozen schema
    # names the semantic paths.  Accept exactly those nested representations;
    # do not accept an arbitrary flattened substitute.
    present = set(candidate)
    if "state_before" in candidate and isinstance(candidate["state_before"], Mapping):
        present.add("state_before.joint_position_rad")
    if "teacher" in candidate and isinstance(candidate["teacher"], Mapping):
        present.add("teacher.action")
    if "student" in candidate and isinstance(candidate["student"], Mapping):
        present.add("student.action")
    for field in ROUND_TRIP_CANDIDATE_SCHEMA.required_fields:
        if field not in present:
            issues.append(ActionIssue("missing_field", field, f"candidate is missing {field!r}"))
    if candidate.get("schema_version") != ROUND_TRIP_CANDIDATE_SCHEMA_VERSION:
        issues.append(ActionIssue("schema_version_mismatch", "schema_version", "candidate schema version is not the frozen v1 value"))
    if candidate.get("replay_status") != "not_run":
        issues.append(ActionIssue("candidate_status", "replay_status", "candidate must remain not_run until external replay evidence exists"))
    return ActionValidationReport(
        not issues,
        InterfaceId.ROUND_TRIP_CANDIDATE.value,
        STUDENT_ACTION_DIM,
        tuple(issues),
    )


__all__ = [
    "ACTION_INTERFACE_SCHEMA_VERSION",
    "ActionConversionError",
    "ActionInterfaceError",
    "ActionInterfaceSpec",
    "ActionIssue",
    "ActionValidationReport",
    "ChannelSpec",
    "DT_SEMANTICS",
    "GRIPPER_SIGN",
    "HiddenTeacherControls",
    "InterfaceId",
    "JOINT_COUNT",
    "JOINT_FRAME",
    "JOINT_ORDER",
    "LA4VLA_LIBERO_OSC_POSE_INTERFACE",
    "LA4VLA_OSC_POSE_ACTION_DIM",
    "LEROBOT_EXPORT_ACTION_DIM",
    "LEROBOT_EXPORT_CANDIDATE_INTERFACE",
    "LIBERO_LOW_LEVEL_INTERFACE",
    "LIBERO_STUDENT",
    "LIBERO_STUDENT_INTERFACE",
    "LiberoStudentAction",
    "ROBO_SUITE_TEACHER_INTERFACE",
    "ROBOSUITE_TEACHER",
    "ROBOSUITE_TEACHER_INTERFACE",
    "RoboSuiteTeacherAction",
    "RobosuiteTeacherAction",
    "ROUND_TRIP_CANDIDATE_SCHEMA",
    "ROUND_TRIP_CANDIDATE_SCHEMA_VERSION",
    "ROUND_TRIP_SCHEMA",
    "RoundTripCandidate",
    "RoundTripCandidateSchema",
    "SCHEMA_VERSION",
    "STUDENT_ACTION_DIM",
    "STUDENT_INTERFACE",
    "StudentAction",
    "TEACHER_ACTION_DIM",
    "TEACHER_INTERFACE",
    "TIMESTAMP_SEMANTICS",
    "TeacherAction",
    "TeacherProfile",
    "build_round_trip_candidate",
    "convert_action",
    "convert_teacher_action_to_student",
    "convert_teacher_to_student",
    "convert_student_to_teacher",
    "get_action_interface",
    "teacher_to_student",
    "validate_action",
    "validate_action_vector",
    "validate_candidate",
]
