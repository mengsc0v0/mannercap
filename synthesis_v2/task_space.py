"""Versioned task, phase, property, and request-cell contracts.

This module describes the request space before any candidate is generated.  It
is intentionally declarative and calibration-neutral: it names the task
events, object-property sources, and legal control channels, but it does not
invent physical values or thresholds.  Lift and Stack are available for
development scheduling.  Wipe is retained only as a ``candidate`` task with
``pending_calibration`` status and cannot produce an applicable request cell.

The contract is independent of the simulator and model services.  A future
adapter may use these records to populate ``request_manifest.py`` after
calibration, while preserving the split and pair rules here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, TypeAlias

from .request_manifest import (
    DEVELOPMENT_SPLIT,
    PairType,
    SUPPORTED_SPLITS,
)


JSONValue: TypeAlias = Any
TASK_SPACE_SCHEMA_VERSION: Final[str] = "synthesis_v2.task_space.v1"


class TaskSpaceError(ValueError):
    """Raised when a task-space record violates the versioned contract."""


class TaskStatus(str, Enum):
    SUPPORTED = "supported"
    CANDIDATE = "candidate"
    DISABLED = "disabled"


class CalibrationStatus(str, Enum):
    READY_FOR_DEVELOPMENT = "ready_for_development"
    PENDING_CALIBRATION = "pending_calibration"
    CALIBRATED = "calibrated"


class Applicability(str, Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    PENDING_CALIBRATION = "pending_calibration"


_SPLIT_ALIASES: Final[dict[str, str]] = {
    "dev": DEVELOPMENT_SPLIT,
    "development": DEVELOPMENT_SPLIT,
    "generator_test": "generator_test",
    "generator-test": "generator_test",
    "student_train": "student_train",
    "student-train": "student_train",
    "student_validation": "student_validation",
    "student-validation": "student_validation",
    "student_test": "student_test",
    "student-test": "student_test",
}
_PAIR_TYPE_ALIASES: Final[dict[str, str]] = {
    "modifier_word": PairType.MODIFIER_WORD.value,
    "modifier-word": PairType.MODIFIER_WORD.value,
    "modifier": PairType.MODIFIER_WORD.value,
    "manner_word": PairType.MODIFIER_WORD.value,
    "manner-word": PairType.MODIFIER_WORD.value,
    "manner": PairType.MODIFIER_WORD.value,
    "physical_attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "physical-attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "stage_scope": PairType.STAGE_SCOPE.value,
    "stage-scope": PairType.STAGE_SCOPE.value,
    "stage": PairType.STAGE_SCOPE.value,
    "scope": PairType.STAGE_SCOPE.value,
}


def _json_copy(value: JSONValue, *, field_name: str) -> JSONValue:
    try:
        copied = deepcopy(value)
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TaskSpaceError(f"{field_name} must be JSON-compatible") from exc
    return copied


def _canonical_json(value: JSONValue) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TaskSpaceError("task-space values must be JSON-compatible") from exc


def _content_hash(value: JSONValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSpaceError(f"{field_name} must be a non-empty string")
    return value


def _normalize_split(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSpaceError("split must be an explicit non-empty name")
    normalized = _SPLIT_ALIASES.get(value.strip().lower())
    if normalized is None or normalized not in SUPPORTED_SPLITS:
        raise TaskSpaceError(f"unknown split {value!r}")
    return normalized


def _normalize_pair_type(value: str | PairType, field_name: str = "pair_type") -> str:
    if isinstance(value, PairType):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise TaskSpaceError(f"{field_name} must be an explicit non-empty value")
    normalized = _PAIR_TYPE_ALIASES.get(value.strip().lower())
    if normalized is None:
        allowed = ", ".join(member.value for member in PairType)
        raise TaskSpaceError(f"unknown {field_name} {value!r}; expected one of {allowed}")
    return normalized


def _normalize_enum(value: str | Enum, enum_type: type[Enum], field_name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise TaskSpaceError(f"{field_name} must be an explicit non-empty value")
    for member in enum_type:
        if value.strip().lower() == member.value.lower():
            return member.value
    allowed = ", ".join(member.value for member in enum_type)
    raise TaskSpaceError(f"unknown {field_name} {value!r}; expected one of {allowed}")


def _strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TaskSpaceError(f"{field_name} must be a sequence of names")
    try:
        result = tuple(_non_empty_string(value, field_name) for value in values)
    except TypeError as exc:
        raise TaskSpaceError(f"{field_name} must be a sequence of names") from exc
    return result


@dataclass(frozen=True)
class EventSpec:
    """Named phase boundary event; its adapter condition is kept opaque."""

    name: str
    source: str
    description: str = ""
    object_roles: Sequence[str] = field(default_factory=tuple)
    condition_ref: str | None = None
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "event.name"))
        object.__setattr__(self, "source", _non_empty_string(self.source, "event.source"))
        if not isinstance(self.description, str):
            raise TaskSpaceError("event.description must be a string")
        if self.condition_ref is not None:
            _non_empty_string(self.condition_ref, "event.condition_ref")
        object.__setattr__(self, "object_roles", _strings(self.object_roles, "event.object_roles"))
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="event.metadata"))

    @classmethod
    def from_value(cls, value: "EventSpec | Mapping[str, JSONValue]") -> "EventSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TaskSpaceError("event must be EventSpec or mapping")
        return cls(
            name=value.get("name", value.get("event_name")),
            source=value.get("source", value.get("event_source")),
            description=value.get("description", ""),
            object_roles=value.get("object_roles", ()),
            condition_ref=value.get("condition_ref"),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "object_roles": list(self.object_roles),
            "condition_ref": self.condition_ref,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class PhaseSpec:
    """One executable phase with explicit entry and exit events."""

    stage_instance_id: str
    name: str
    entry_event: EventSpec | Mapping[str, JSONValue]
    exit_event: EventSpec | Mapping[str, JSONValue]
    object_roles: Sequence[str] = field(default_factory=tuple)
    allowed_control_channels: Sequence[str] = field(default_factory=tuple)
    manner_scope_allowed: bool = True
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_instance_id",
            _non_empty_string(self.stage_instance_id, "phase.stage_instance_id"),
        )
        object.__setattr__(self, "name", _non_empty_string(self.name, "phase.name"))
        object.__setattr__(self, "entry_event", EventSpec.from_value(self.entry_event))
        object.__setattr__(self, "exit_event", EventSpec.from_value(self.exit_event))
        object.__setattr__(self, "object_roles", _strings(self.object_roles, "phase.object_roles"))
        object.__setattr__(
            self,
            "allowed_control_channels",
            _strings(self.allowed_control_channels, "phase.allowed_control_channels"),
        )
        if not isinstance(self.manner_scope_allowed, bool):
            raise TaskSpaceError("phase.manner_scope_allowed must be boolean")
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="phase.metadata"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "stage_instance_id": self.stage_instance_id,
            "name": self.name,
            "entry_event": self.entry_event.to_dict(),
            "exit_event": self.exit_event.to_dict(),
            "object_roles": list(self.object_roles),
            "allowed_control_channels": list(self.allowed_control_channels),
            "manner_scope_allowed": self.manner_scope_allowed,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ObjectPropertySpec:
    """An object property declaration with explicit provenance and unit.

    ``value`` remains optional.  A missing value means calibration or an
    adapter must provide it; it is never filled with a guessed physical value.
    """

    property_name: str
    unit: str
    source: str
    value: JSONValue = None
    availability: str | CalibrationStatus = CalibrationStatus.PENDING_CALIBRATION.value
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_name", _non_empty_string(self.property_name, "property_name"))
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "property.unit"))
        object.__setattr__(self, "source", _non_empty_string(self.source, "property.source"))
        object.__setattr__(
            self,
            "availability",
            _normalize_enum(self.availability, CalibrationStatus, "property.availability"),
        )
        object.__setattr__(self, "value", _json_copy(self.value, field_name="property.value"))
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="property.metadata"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "property_name": self.property_name,
            "unit": self.unit,
            "source": self.source,
            "value": deepcopy(self.value),
            "availability": self.availability,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ObjectSpec:
    """Object role and its source-tagged property declarations."""

    object_id: str
    role: str
    properties: Sequence[ObjectPropertySpec | Mapping[str, JSONValue]] = field(default_factory=tuple)
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _non_empty_string(self.object_id, "object.object_id"))
        object.__setattr__(self, "role", _non_empty_string(self.role, "object.role"))
        object.__setattr__(
            self,
            "properties",
            tuple(
                item if isinstance(item, ObjectPropertySpec) else ObjectPropertySpec(**item)
                for item in self.properties
            ),
        )
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="object.metadata"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "object_id": self.object_id,
            "role": self.role,
            "properties": [item.to_dict() for item in self.properties],
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ControlChannelSpec:
    """Legal control-channel declaration without an invented numeric bound."""

    name: str
    unit: str
    source: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    domain_ref: str | None = None
    enabled: bool = True
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "control.name"))
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "control.unit"))
        object.__setattr__(self, "source", _non_empty_string(self.source, "control.source"))
        if self.domain_ref is None:
            raise TaskSpaceError("control.domain_ref is required to name the legal control domain")
        object.__setattr__(self, "domain_ref", _non_empty_string(self.domain_ref, "control.domain_ref"))
        for field_name, value in (
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
        ):
            if value is not None:
                if isinstance(value, bool):
                    raise TaskSpaceError(f"control.{field_name} must be finite numeric or null")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TaskSpaceError(
                        f"control.{field_name} must be finite numeric or null"
                    ) from exc
                if not math.isfinite(number):
                    raise TaskSpaceError(f"control.{field_name} must be finite")
                object.__setattr__(self, field_name, number)
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise TaskSpaceError("control lower_bound cannot exceed upper_bound")
        if not isinstance(self.enabled, bool):
            raise TaskSpaceError("control.enabled must be boolean")
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="control.metadata"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "domain_ref": self.domain_ref,
            "enabled": self.enabled,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class TaskSpec:
    """Versioned task definition used by request-space validation."""

    task_id: str
    display_name: str
    status: str | TaskStatus
    calibration_status: str | CalibrationStatus
    phases: Sequence[PhaseSpec | Mapping[str, JSONValue]] = field(default_factory=tuple)
    objects: Sequence[ObjectSpec | Mapping[str, JSONValue]] = field(default_factory=tuple)
    control_channels: Sequence[ControlChannelSpec | Mapping[str, JSONValue]] = field(default_factory=tuple)
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = _non_empty_string(self.task_id, "task_id").strip().lower()
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "display_name", _non_empty_string(self.display_name, "display_name"))
        object.__setattr__(self, "status", _normalize_enum(self.status, TaskStatus, "status"))
        object.__setattr__(
            self,
            "calibration_status",
            _normalize_enum(self.calibration_status, CalibrationStatus, "calibration_status"),
        )
        object.__setattr__(
            self,
            "phases",
            tuple(item if isinstance(item, PhaseSpec) else PhaseSpec(**item) for item in self.phases),
        )
        object.__setattr__(
            self,
            "objects",
            tuple(item if isinstance(item, ObjectSpec) else ObjectSpec(**item) for item in self.objects),
        )
        object.__setattr__(
            self,
            "control_channels",
            tuple(
                item if isinstance(item, ControlChannelSpec) else ControlChannelSpec(**item)
                for item in self.control_channels
            ),
        )
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="task.metadata"))
        if self.status == TaskStatus.CANDIDATE.value and self.calibration_status != CalibrationStatus.PENDING_CALIBRATION.value:
            raise TaskSpaceError("candidate task must have pending_calibration status")
        if self.status == TaskStatus.DISABLED.value and self.phases:
            raise TaskSpaceError("disabled task cannot declare executable phases")
        phase_ids = [phase.stage_instance_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise TaskSpaceError("phase stage_instance_id values must be unique within a task")
        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise TaskSpaceError("object IDs must be unique within a task")
        control_names = [channel.name for channel in self.control_channels]
        if len(control_names) != len(set(control_names)):
            raise TaskSpaceError("control channel names must be unique within a task")

    @property
    def enabled(self) -> bool:
        return self.status == TaskStatus.SUPPORTED.value

    def phase_ids(self) -> tuple[str, ...]:
        return tuple(phase.stage_instance_id for phase in self.phases)

    def control_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.control_channels if channel.enabled)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "task_id": self.task_id,
            "display_name": self.display_name,
            "status": self.status,
            "calibration_status": self.calibration_status,
            "enabled": self.enabled,
            "phases": [phase.to_dict() for phase in self.phases],
            "objects": [obj.to_dict() for obj in self.objects],
            "control_channels": [channel.to_dict() for channel in self.control_channels],
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class RequestCell:
    """One task-stage-property-initial-state request-space unit."""

    task_id: str
    stage_instance_ids: Sequence[str] = field(default_factory=tuple)
    property_config_id: str | None = None
    initial_state_id: str | None = None
    split: str = DEVELOPMENT_SPLIT
    applicability: str | Applicability = Applicability.APPLICABLE.value
    reason: str | None = None
    pair_group_key: str | None = None
    pair_type: str | PairType | None = None
    cell_id: str | None = None
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = _non_empty_string(self.task_id, "cell.task_id").strip().lower()
        object.__setattr__(self, "task_id", task_id)
        stage_ids = _strings(self.stage_instance_ids, "cell.stage_instance_ids")
        object.__setattr__(self, "stage_instance_ids", stage_ids)
        for field_name, value in (
            ("property_config_id", self.property_config_id),
            ("initial_state_id", self.initial_state_id),
            ("pair_group_key", self.pair_group_key),
            ("cell_id", self.cell_id),
        ):
            if value is not None:
                _non_empty_string(value, f"cell.{field_name}")
        object.__setattr__(self, "split", _normalize_split(self.split))
        applicability = _normalize_enum(self.applicability, Applicability, "applicability")
        object.__setattr__(self, "applicability", applicability)
        if applicability != Applicability.APPLICABLE.value:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise TaskSpaceError("non-applicable request cell requires reason")
        elif self.reason is not None and not isinstance(self.reason, str):
            raise TaskSpaceError("cell.reason must be a string or null")
        if self.pair_type is not None:
            object.__setattr__(self, "pair_type", _normalize_pair_type(self.pair_type, "cell.pair_type"))
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="cell.metadata"))
        computed_id = "cell_" + _content_hash(self.payload())[:24]
        if self.cell_id is not None and self.cell_id != computed_id:
            raise TaskSpaceError(f"cell_id {self.cell_id!r} does not match deterministic ID {computed_id!r}")
        object.__setattr__(self, "cell_id", computed_id)

    def payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_SPACE_SCHEMA_VERSION,
            "task_id": self.task_id,
            "stage_instance_ids": list(self.stage_instance_ids),
            "property_config_id": self.property_config_id,
            "initial_state_id": self.initial_state_id,
            "split": self.split,
            "applicability": self.applicability,
            "reason": self.reason,
            "pair_group_key": self.pair_group_key,
            "pair_type": self.pair_type,
            "metadata": deepcopy(self.metadata),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        result = self.payload()
        result["cell_id"] = self.cell_id
        return result


@dataclass(frozen=True)
class PairConstraint:
    """Explicit structural rule for one of the three pair types."""

    pair_type: str | PairType
    same_fields: Sequence[str]
    varying_fields: Sequence[str]
    same_split: bool = True
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_type", _normalize_pair_type(self.pair_type, "pair constraint type"))
        object.__setattr__(self, "same_fields", _strings(self.same_fields, "pair.same_fields"))
        object.__setattr__(self, "varying_fields", _strings(self.varying_fields, "pair.varying_fields"))
        if not isinstance(self.same_split, bool):
            raise TaskSpaceError("pair.same_split must be boolean")
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="pair.metadata"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "pair_type": self.pair_type,
            "same_fields": list(self.same_fields),
            "varying_fields": list(self.varying_fields),
            "same_split": self.same_split,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class PairCell:
    """A two-cell planned relation checked against the split policy."""

    pair_type: str | PairType
    member_cell_ids: Sequence[str]
    split: str
    pair_id: str | None = None
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        pair_type = _normalize_pair_type(self.pair_type, "pair cell type")
        object.__setattr__(self, "pair_type", pair_type)
        if isinstance(self.member_cell_ids, (str, bytes)):
            raise TaskSpaceError("pair.member_cell_ids must be a sequence of cell IDs")
        ids = tuple(_non_empty_string(item, "pair.member_cell_ids") for item in self.member_cell_ids)
        if len(ids) != 2 or ids[0] == ids[1]:
            raise TaskSpaceError("pair cell requires two distinct members")
        object.__setattr__(self, "member_cell_ids", ids)
        object.__setattr__(self, "split", _normalize_split(self.split))
        if self.pair_id is not None:
            _non_empty_string(self.pair_id, "pair_id")
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="pair.metadata"))
        computed_id = "paircell_" + _content_hash(
            {"pair_type": pair_type, "member_cell_ids": sorted(ids), "split": self.split}
        )[:24]
        if self.pair_id is not None and self.pair_id != computed_id:
            raise TaskSpaceError(f"pair_id {self.pair_id!r} does not match deterministic ID {computed_id!r}")
        object.__setattr__(self, "pair_id", computed_id)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "pair_id": self.pair_id,
            "pair_type": self.pair_type,
            "member_cell_ids": list(self.member_cell_ids),
            "split": self.split,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class SplitPolicy:
    """Explicit split names and the no-cross-split pairing rule."""

    allowed_splits: Sequence[str] = field(default_factory=lambda: tuple(SUPPORTED_SPLITS))
    development_split: str = DEVELOPMENT_SPLIT
    pair_members_same_split: bool = True

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(_normalize_split(item) for item in self.allowed_splits))
        if not normalized:
            raise TaskSpaceError("split policy must name at least one split")
        development = _normalize_split(self.development_split)
        if development not in normalized:
            raise TaskSpaceError("development_split must be in allowed_splits")
        if not isinstance(self.pair_members_same_split, bool):
            raise TaskSpaceError("pair_members_same_split must be boolean")
        object.__setattr__(self, "allowed_splits", normalized)
        object.__setattr__(self, "development_split", development)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "allowed_splits": list(self.allowed_splits),
            "development_split": self.development_split,
            "pair_members_same_split": self.pair_members_same_split,
        }


def default_pair_constraints() -> tuple[PairConstraint, ...]:
    """Return the three structural pair rules used by the request manifest."""

    return (
        PairConstraint(
            PairType.MODIFIER_WORD,
            same_fields=(
                "task",
                "scene_id",
                "initial_state",
                "target_object",
                "physical_config",
                "stage_scope",
                "allowed_control_channels",
                "split",
                "budget",
            ),
            varying_fields=("raw_language", "requested_constraints"),
        ),
        PairConstraint(
            PairType.PHYSICAL_ATTRIBUTE,
            same_fields=(
                "task",
                "scene_id",
                "raw_language",
                "requested_constraints",
                "stage_scope",
                "geometry_key",
                "split",
            ),
            varying_fields=("physical_config", "initial_state", "attribute_intervention"),
        ),
        PairConstraint(
            PairType.STAGE_SCOPE,
            same_fields=(
                "task",
                "scene_id",
                "initial_state",
                "target_object",
                "physical_config",
                "manner_key",
                "allowed_control_channels",
                "split",
                "budget",
            ),
            varying_fields=("raw_language", "requested_constraints", "stage_scope"),
        ),
    )


@dataclass(frozen=True)
class TaskSpace:
    """Complete versioned task registry and request-cell policy."""

    tasks: Sequence[TaskSpec | Mapping[str, JSONValue]]
    request_cells: Sequence[RequestCell | Mapping[str, JSONValue]] = field(default_factory=tuple)
    pair_cells: Sequence[PairCell | Mapping[str, JSONValue]] = field(default_factory=tuple)
    split_policy: SplitPolicy = field(default_factory=SplitPolicy)
    pair_constraints: Sequence[PairConstraint | Mapping[str, JSONValue]] = field(
        default_factory=default_pair_constraints
    )
    status: str = "draft"
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        tasks = tuple(item if isinstance(item, TaskSpec) else TaskSpec(**item) for item in self.tasks)
        cells = tuple(item if isinstance(item, RequestCell) else RequestCell(**item) for item in self.request_cells)
        pair_cells = tuple(item if isinstance(item, PairCell) else PairCell(**item) for item in self.pair_cells)
        policy = self.split_policy if isinstance(self.split_policy, SplitPolicy) else SplitPolicy(**self.split_policy)
        constraints = tuple(
            item if isinstance(item, PairConstraint) else PairConstraint(**item)
            for item in self.pair_constraints
        )
        if not isinstance(self.status, str) or not self.status.strip():
            raise TaskSpaceError("task-space status must be a non-empty string")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "request_cells", cells)
        object.__setattr__(self, "pair_cells", pair_cells)
        object.__setattr__(self, "split_policy", policy)
        object.__setattr__(self, "pair_constraints", constraints)
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="task_space.metadata"))
        self.validate()

    def get_task(self, task_id: str) -> TaskSpec:
        task_id = _non_empty_string(task_id, "task_id").strip().lower()
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise TaskSpaceError(f"unknown task_id {task_id!r}")

    def validate(self) -> None:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise TaskSpaceError("task IDs must be unique")
        cell_ids = [cell.cell_id for cell in self.request_cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise TaskSpaceError("request cell IDs must be unique")
        pair_types = {constraint.pair_type for constraint in self.pair_constraints}
        expected_pair_types = {member.value for member in PairType}
        if len(self.pair_constraints) != len(expected_pair_types) or pair_types != expected_pair_types:
            raise TaskSpaceError(
                "pair_constraints must explicitly cover modifier_word, physical_attribute, and stage_scope"
            )
        if self.split_policy.pair_members_same_split and any(
            not constraint.same_split for constraint in self.pair_constraints
        ):
            raise TaskSpaceError("every pair constraint must preserve split membership")

        task_by_id = {task.task_id: task for task in self.tasks}
        cell_by_id = {cell.cell_id: cell for cell in self.request_cells}
        for task in self.tasks:
            known_controls = set(task.control_names())
            known_roles = {obj.role for obj in task.objects}
            for phase in task.phases:
                unknown_controls = set(phase.allowed_control_channels) - known_controls
                if unknown_controls:
                    raise TaskSpaceError(
                        f"task {task.task_id!r} phase {phase.stage_instance_id!r} "
                        f"uses undeclared control channel(s): {sorted(unknown_controls)}"
                    )
                unknown_roles = set(phase.object_roles) - known_roles
                if unknown_roles:
                    raise TaskSpaceError(
                        f"task {task.task_id!r} phase {phase.stage_instance_id!r} "
                        f"uses undeclared object role(s): {sorted(unknown_roles)}"
                    )
        for cell in self.request_cells:
            task = task_by_id.get(cell.task_id)
            if task is None:
                raise TaskSpaceError(f"request cell {cell.cell_id!r} refers to unknown task")
            if cell.split not in self.split_policy.allowed_splits:
                raise TaskSpaceError(f"request cell {cell.cell_id!r} uses split outside policy")
            if cell.applicability == Applicability.APPLICABLE.value:
                if not task.enabled:
                    raise TaskSpaceError(
                        f"candidate/disabled task {task.task_id!r} cannot have applicable request cell"
                    )
                if cell.property_config_id is None or cell.initial_state_id is None:
                    raise TaskSpaceError(
                        "applicable request cell requires explicit property_config_id and initial_state_id"
                    )
                known_phases = set(task.phase_ids())
                unknown_phases = set(cell.stage_instance_ids) - known_phases
                if unknown_phases:
                    raise TaskSpaceError(
                        f"request cell {cell.cell_id!r} names unknown phase(s): {sorted(unknown_phases)}"
                    )
                if not cell.stage_instance_ids:
                    raise TaskSpaceError("applicable request cell requires at least one stage_instance_id")
            elif not cell.reason:
                raise TaskSpaceError("non-applicable request cell requires reason")

        for pair in self.pair_cells:
            if pair.pair_type not in pair_types:
                raise TaskSpaceError(f"pair cell {pair.pair_id!r} uses undeclared pair type")
            members = [cell_by_id.get(cell_id) for cell_id in pair.member_cell_ids]
            if any(member is None for member in members):
                raise TaskSpaceError(f"pair cell {pair.pair_id!r} refers to unknown request cell")
            if members[0] is None or members[1] is None:
                raise TaskSpaceError(f"pair cell {pair.pair_id!r} has missing member cells")
            if self.split_policy.pair_members_same_split:
                if members[0].split != members[1].split or pair.split != members[0].split:
                    raise TaskSpaceError(
                        f"pair cell {pair.pair_id!r} crosses split; pair members must share split"
                    )
            if members[0].task_id != members[1].task_id:
                raise TaskSpaceError(f"pair cell {pair.pair_id!r} must use one task")
            if members[0].applicability != Applicability.APPLICABLE.value or members[1].applicability != Applicability.APPLICABLE.value:
                raise TaskSpaceError("non-applicable request cells cannot form an executable pair")
        pair_ids = [pair.pair_id for pair in self.pair_cells]
        if len(pair_ids) != len(set(pair_ids)):
            raise TaskSpaceError("pair IDs must be unique")

    @property
    def development_tasks(self) -> tuple[TaskSpec, ...]:
        return tuple(
            task
            for task in self.tasks
            if task.enabled and task.task_id in {"lift", "stack"}
        )

    @property
    def pending_tasks(self) -> tuple[TaskSpec, ...]:
        return tuple(task for task in self.tasks if task.calibration_status == CalibrationStatus.PENDING_CALIBRATION.value)

    @property
    def inapplicable_cells(self) -> tuple[RequestCell, ...]:
        return tuple(
            cell
            for cell in self.request_cells
            if cell.applicability != Applicability.APPLICABLE.value
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_SPACE_SCHEMA_VERSION,
            "status": self.status,
            "tasks": [task.to_dict() for task in self.tasks],
            "request_cells": [cell.to_dict() for cell in self.request_cells],
            "pair_cells": [pair.to_dict() for pair in self.pair_cells],
            "split_policy": self.split_policy.to_dict(),
            "pair_constraints": [constraint.to_dict() for constraint in self.pair_constraints],
            "metadata": deepcopy(self.metadata),
        }


def _event(name: str, source: str, description: str, roles: Sequence[str] = ()) -> EventSpec:
    return EventSpec(name=name, source=source, description=description, object_roles=roles)


def _default_controls() -> tuple[ControlChannelSpec, ...]:
    # Units are interface labels; numeric domains intentionally remain null
    # until A/B calibration and action-interface checks are complete.
    return (
        ControlChannelSpec(
            name="joint_velocity",
            unit="rad/s",
            source="existing executor interface; calibration registry pending",
            domain_ref="calibration_registry:joint_velocity",
        ),
        ControlChannelSpec(
            name="acceleration",
            unit="rad/s^2",
            source="existing executor interface; calibration registry pending",
            domain_ref="calibration_registry:acceleration",
        ),
        ControlChannelSpec(
            name="gripper",
            unit="adapter-defined",
            source="existing executor interface; calibration registry pending",
            domain_ref="calibration_registry:gripper",
        ),
    )


def _default_properties() -> tuple[ObjectPropertySpec, ...]:
    # These are schema-level declarations only.  No mass, friction, force, or
    # threshold is supplied before calibration.
    return (
        ObjectPropertySpec(
            property_name="geometry",
            unit="adapter-defined",
            source="calibration_registry:pending",
        ),
        ObjectPropertySpec(
            property_name="physical_configuration",
            unit="adapter-defined",
            source="calibration_registry:pending",
        ),
    )


def _lift_task() -> TaskSpec:
    controls = _default_controls()
    target = "target_object"
    return TaskSpec(
        task_id="lift",
        display_name="Lift",
        status=TaskStatus.SUPPORTED,
        calibration_status=CalibrationStatus.READY_FOR_DEVELOPMENT,
        phases=(
            PhaseSpec(
                "approach",
                "Approach",
                _event("episode_start", "episode", "request execution begins", (target,)),
                _event("approach_target_reached", "task_adapter", "approach endpoint event", (target,)),
                (target,),
                ("joint_velocity",),
            ),
            PhaseSpec(
                "descend",
                "Descend",
                _event("approach_target_reached", "task_adapter", "approach endpoint event", (target,)),
                _event("grasp_pose_reached", "task_adapter", "grasp pose endpoint event", (target,)),
                (target,),
                ("joint_velocity",),
            ),
            PhaseSpec(
                "grasp",
                "Grasp",
                _event("grasp_pose_reached", "task_adapter", "grasp pose endpoint event", (target,)),
                _event("gripper_close_complete", "control_trace", "gripper command completion event", (target,)),
                (target,),
                ("gripper",),
            ),
            PhaseSpec(
                "lift",
                "Lift",
                _event("gripper_close_complete", "control_trace", "gripper command completion event", (target,)),
                _event("lift_target_reached", "task_adapter", "lift endpoint event", (target,)),
                (target,),
                ("joint_velocity", "acceleration"),
            ),
        ),
        objects=(ObjectSpec("red_cube", "target_object", _default_properties()),),
        control_channels=controls,
    )


def _stack_task() -> TaskSpec:
    controls = _default_controls()
    moving = "moving_object"
    support = "support_object"
    return TaskSpec(
        task_id="stack",
        display_name="Stack",
        status=TaskStatus.SUPPORTED,
        calibration_status=CalibrationStatus.READY_FOR_DEVELOPMENT,
        phases=(
            PhaseSpec(
                "approach",
                "Approach",
                _event("episode_start", "episode", "request execution begins", (moving,)),
                _event("approach_target_reached", "task_adapter", "approach endpoint event", (moving,)),
                (moving,),
                ("joint_velocity",),
            ),
            PhaseSpec(
                "descend",
                "Descend",
                _event("approach_target_reached", "task_adapter", "approach endpoint event", (moving,)),
                _event("grasp_pose_reached", "task_adapter", "grasp pose endpoint event", (moving,)),
                (moving,),
                ("joint_velocity",),
            ),
            PhaseSpec(
                "grasp",
                "Grasp",
                _event("grasp_pose_reached", "task_adapter", "grasp pose endpoint event", (moving,)),
                _event("gripper_close_complete", "control_trace", "gripper command completion event", (moving,)),
                (moving,),
                ("gripper",),
            ),
            PhaseSpec(
                "lift",
                "Lift",
                _event("gripper_close_complete", "control_trace", "gripper command completion event", (moving,)),
                _event("lift_target_reached", "task_adapter", "lift endpoint event", (moving,)),
                (moving,),
                ("joint_velocity", "acceleration"),
            ),
            PhaseSpec(
                "transport",
                "Transport",
                _event("lift_target_reached", "task_adapter", "lift endpoint event", (moving,)),
                _event("transport_target_reached", "task_adapter", "transport endpoint event", (moving, support)),
                (moving, support),
                ("joint_velocity", "acceleration"),
            ),
            PhaseSpec(
                "place",
                "Place",
                _event("transport_target_reached", "task_adapter", "transport endpoint event", (moving, support)),
                _event("placement_target_reached", "task_adapter", "placement endpoint event", (moving, support)),
                (moving, support),
                ("joint_velocity", "acceleration"),
            ),
            PhaseSpec(
                "retreat",
                "Retreat",
                _event("release_complete", "control_trace", "release command completion event", (moving,)),
                _event("retreat_target_reached", "task_adapter", "retreat endpoint event", (moving,)),
                (moving,),
                ("joint_velocity",),
            ),
        ),
        objects=(
            ObjectSpec("red_cube", "moving_object", _default_properties()),
            ObjectSpec("green_cube", "support_object", _default_properties()),
        ),
        control_channels=controls,
    )


def _wipe_candidate() -> TaskSpec:
    return TaskSpec(
        task_id="wipe",
        display_name="Wipe (candidate)",
        status=TaskStatus.CANDIDATE,
        calibration_status=CalibrationStatus.PENDING_CALIBRATION,
        phases=(),
        objects=(),
        control_channels=(),
        metadata={
            "reason": "contact metric and task-result definition are pending calibration",
            "requests_enabled": False,
        },
    )


def build_default_task_space() -> TaskSpace:
    """Return the initial Lift/Stack development registry plus pending Wipe."""

    pending_wipe = RequestCell(
        task_id="wipe",
        stage_instance_ids=(),
        property_config_id=None,
        initial_state_id=None,
        split=DEVELOPMENT_SPLIT,
        applicability=Applicability.PENDING_CALIBRATION,
        reason="Wipe contact metric and task-result definition are not calibrated",
        metadata={"requests_enabled": False},
    )
    return TaskSpace(
        tasks=(_lift_task(), _stack_task(), _wipe_candidate()),
        request_cells=(pending_wipe,),
        pair_cells=(),
        status="draft",
        metadata={
            "default_development_tasks": ["lift", "stack"],
            "third_task": {"task_id": "wipe", "status": "candidate", "calibration_status": "pending_calibration"},
            "physical_values_and_thresholds": "not supplied before calibration",
        },
    )


default_task_space = build_default_task_space


def build_request_cell(
    task_id: str,
    *,
    stage_instance_ids: Sequence[str],
    property_config_id: str | None,
    initial_state_id: str | None,
    split: str = DEVELOPMENT_SPLIT,
    applicability: str | Applicability = Applicability.APPLICABLE,
    reason: str | None = None,
    pair_group_key: str | None = None,
    pair_type: str | PairType | None = None,
    metadata: JSONValue = None,
) -> RequestCell:
    """Create one explicit request cell; callers provide all calibrated refs."""

    return RequestCell(
        task_id=task_id,
        stage_instance_ids=stage_instance_ids,
        property_config_id=property_config_id,
        initial_state_id=initial_state_id,
        split=split,
        applicability=applicability,
        reason=reason,
        pair_group_key=pair_group_key,
        pair_type=pair_type,
        metadata={} if metadata is None else metadata,
    )


def write_task_space(path: str | Path, task_space: TaskSpace) -> None:
    """Write the complete versioned task-space object as stable JSON."""

    if not isinstance(task_space, TaskSpace):
        raise TaskSpaceError("write_task_space expects a TaskSpace")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            task_space.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Applicability",
    "CalibrationStatus",
    "ControlChannelSpec",
    "EventSpec",
    "ObjectPropertySpec",
    "ObjectSpec",
    "PairCell",
    "PairConstraint",
    "PhaseSpec",
    "RequestCell",
    "SplitPolicy",
    "TASK_SPACE_SCHEMA_VERSION",
    "TaskSpace",
    "TaskSpaceError",
    "TaskSpec",
    "TaskStatus",
    "build_default_task_space",
    "build_request_cell",
    "default_pair_constraints",
    "default_task_space",
    "write_task_space",
]
