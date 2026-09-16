"""Pure object-level contact-wrench recording and evaluation helpers.

The legacy RoboSuite/MuJoCo recorder stores ``sim.data.cfrc_ext`` as one
six-dimensional body row and then takes a norm over all six coordinates.  The
first three coordinates are a force vector (N) and the last three are a torque
vector (N*m); combining them into one scalar has no physical unit.  This module
keeps those quantities separate and requires an explicit object/body/geom
binding before a row can be treated as object evidence.

This file deliberately has no MuJoCo or NumPy dependency.  It is a pure
boundary: a caller supplies an immutable snapshot of the simulator row and a
registry binding, and receives an immutable record or a strict three-state
evaluation.  It never reads or mutates simulator state.  ``cfrc_ext`` is a
body-level external wrench; it does not identify an individual contact pair.
The record therefore says ``body_aggregate`` for its geometry attribution.

The numeric decision is useful for contract fixtures, but the physical
qualification remains ``unknown`` until an independent calibration is supplied
(``calibration_verified=True``).  The helper intentionally cannot turn an
uncalibrated simulator quantity into a calibrated real-grip-force claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


class Decision(str, Enum):
    """Strict three-state result used by the Gate1 force contract."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ContactState(str, Enum):
    """Contact state derived from a valid, object-bound force vector."""

    CONTACT = "contact"
    NO_CONTACT = "no_contact"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ContactIssue:
    """One reason why an observation cannot be qualified."""

    code: str
    message: str
    field: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class ObjectBodyBinding:
    """Explicit mapping from an object to one MuJoCo body and its geoms.

    ``cfrc_ext`` has one row per body, so a row cannot identify the active
    contact geom by itself.  ``geom_ids`` and ``geom_names`` are the declared
    geoms belonging to this object body and are retained as a body aggregate.
    A future per-contact recorder must add the actual contact pair separately;
    it must not silently reinterpret this aggregate as a single geom force.
    """

    object_id: str
    body_id: int
    body_name: str
    geom_ids: Tuple[int, ...]
    geom_names: Tuple[str, ...]
    frame: str = "world"
    geom_attribution: str = "body_aggregate"


@dataclass(frozen=True)
class ContactForceSpec:
    """Bounds for the norm of the *3-D force* vector, in N.

    At least one bound or ``require_contact`` is required.  A free-motion
    stage should use a different contract and should not accidentally pass by
    constructing an empty force spec.
    """

    min_force_n: Optional[float] = None
    max_force_n: Optional[float] = None
    require_contact: bool = False
    contact_epsilon_n: float = 1e-9


@dataclass(frozen=True)
class ObjectContactWrench:
    """One immutable object-bound snapshot of a MuJoCo body external wrench."""

    timestamp_s: Optional[float]
    object_id: Optional[str]
    body_id: Optional[int]
    body_name: Optional[str]
    geom_ids: Tuple[int, ...]
    geom_names: Tuple[str, ...]
    frame: Optional[str]
    geom_attribution: Optional[str]
    force_n: Optional[Tuple[float, ...]]
    torque_nm: Optional[Tuple[float, ...]]
    sensor_valid: Optional[bool]
    source: str
    issues: Tuple[ContactIssue, ...] = ()
    calibration_status: str = "unverified"

    @property
    def force_norm_n(self) -> Optional[float]:
        """Return the norm of the 3-D force only, never a mixed wrench norm."""

        return _norm3(self.force_n)

    @property
    def torque_norm_nm(self) -> Optional[float]:
        """Return the norm of the 3-D torque only."""

        return _norm3(self.torque_nm)

    @property
    def valid_record(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with separate force/torque keys."""

        return {
            "timestamp_s": self.timestamp_s,
            "object_id": self.object_id,
            "body": {"id": self.body_id, "name": self.body_name},
            "geoms": {
                "ids": list(self.geom_ids),
                "names": list(self.geom_names),
                "attribution": self.geom_attribution,
            },
            "frame": self.frame,
            "force": {
                "vector": list(self.force_n) if self.force_n is not None else None,
                "unit": "N",
                "norm_n": self.force_norm_n,
            },
            "torque": {
                "vector": list(self.torque_nm) if self.torque_nm is not None else None,
                "unit": "N*m",
                "norm_nm": self.torque_norm_nm,
            },
            "sensor_valid": self.sensor_valid,
            "source": self.source,
            "calibration_status": self.calibration_status,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class ContactForceEvaluation:
    """Numeric force-bound decision plus contact and calibration provenance."""

    decision: Decision
    contact_state: ContactState
    object_id: Optional[str]
    body_id: Optional[int]
    body_name: Optional[str]
    force_n: Optional[Tuple[float, ...]]
    torque_nm: Optional[Tuple[float, ...]]
    force_norm_n: Optional[float]
    torque_norm_nm: Optional[float]
    issues: Tuple[ContactIssue, ...] = ()
    calibration_verified: bool = False

    @property
    def physical_qualification(self) -> Decision:
        """A calibrated qualification; uncalibrated numeric results stay unknown."""

        if not self.calibration_verified:
            return Decision.UNKNOWN
        return self.decision

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.PASS

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def has_code(self, code: str) -> bool:
        return code in self.codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "physical_qualification": self.physical_qualification.value,
            "contact_state": self.contact_state.value,
            "object_id": self.object_id,
            "body": {"id": self.body_id, "name": self.body_name},
            "force": {
                "vector": list(self.force_n) if self.force_n is not None else None,
                "unit": "N",
                "norm_n": self.force_norm_n,
            },
            "torque": {
                "vector": list(self.torque_nm) if self.torque_nm is not None else None,
                "unit": "N*m",
                "norm_nm": self.torque_norm_nm,
            },
            "calibration_verified": self.calibration_verified,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def record_object_contact_wrench(
    cfrc_ext: Any,
    *,
    binding: ObjectBodyBinding | Mapping[str, Any] | None,
    timestamp_s: Any,
    sensor_valid: Any,
    source: str = "mujoco.data.cfrc_ext",
) -> ObjectContactWrench:
    """Extract one body row and split it into force (N) and torque (N*m).

    ``cfrc_ext`` must be a body-row collection: either a sequence indexed by
    ``body_id`` or a mapping keyed by integer ``body_id``.  A direct six-vector
    is deliberately not accepted here because it has no object/body identity.
    The function returns a record with issues instead of filling missing data
    with zero; callers can then route the record to ``unknown``.
    """

    issues: list[ContactIssue] = []
    normalized_binding, binding_issues = _coerce_binding(binding)
    issues.extend(binding_issues)

    object_id = normalized_binding.object_id if normalized_binding else None
    body_id = normalized_binding.body_id if normalized_binding else None
    body_name = normalized_binding.body_name if normalized_binding else None
    geom_ids = normalized_binding.geom_ids if normalized_binding else ()
    geom_names = normalized_binding.geom_names if normalized_binding else ()
    frame = normalized_binding.frame if normalized_binding else None
    geom_attribution = normalized_binding.geom_attribution if normalized_binding else None

    time_value = _finite_scalar(timestamp_s)
    if time_value is None:
        issues.append(
            ContactIssue("missing_or_invalid_timestamp", "timestamp_s must be a finite scalar", "timestamp_s")
        )

    validity: Optional[bool]
    if isinstance(sensor_valid, bool):
        validity = sensor_valid
        if not sensor_valid:
            issues.append(ContactIssue("sensor_invalid", "sensor_valid is false", "sensor_valid"))
    elif sensor_valid is None:
        validity = None
        issues.append(ContactIssue("missing_sensor_validity", "sensor_valid must be explicitly true or false", "sensor_valid"))
    else:
        validity = None
        issues.append(ContactIssue("invalid_sensor_validity", "sensor_valid must be a boolean", "sensor_valid"))

    row: Any = None
    if cfrc_ext is None:
        issues.append(ContactIssue("missing_cfrc_ext", "cfrc_ext body rows are missing", "cfrc_ext"))
    elif body_id is None:
        issues.append(ContactIssue("missing_body_id", "cannot select a cfrc_ext row without body_id", "body_id"))
    else:
        row, row_issue = _select_body_row(cfrc_ext, body_id)
        if row_issue is not None:
            issues.append(row_issue)

    force: Optional[Tuple[float, ...]] = None
    torque: Optional[Tuple[float, ...]] = None
    if row is not None:
        vector, vector_issue = _finite_vector(row, expected=6, field="cfrc_ext[body_id]")
        if vector_issue is not None:
            issues.append(vector_issue)
        elif vector is not None:
            force = vector[:3]
            torque = vector[3:]

    return ObjectContactWrench(
        timestamp_s=time_value,
        object_id=object_id,
        body_id=body_id,
        body_name=body_name,
        geom_ids=geom_ids,
        geom_names=geom_names,
        frame=frame,
        geom_attribution=geom_attribution,
        force_n=force,
        torque_nm=torque,
        sensor_valid=validity,
        source=source if isinstance(source, str) and source else "",
        issues=tuple(issues),
    )


def evaluate_object_contact_force(
    record: ObjectContactWrench,
    spec: ContactForceSpec,
    *,
    calibration_verified: bool = False,
) -> ContactForceEvaluation:
    """Evaluate only the 3-D force norm for one object-bound record.

    Any missing/invalid evidence, including a six-dimensional force input,
    returns ``unknown``.  A valid zero force is different: it is an explicit
    ``no_contact`` observation and fails a spec that requires contact.  Force
    and torque are reported independently; this function has no combined
    wrench norm by design.
    """

    if not isinstance(record, ObjectContactWrench):
        return _unknown_evaluation(
            object_id=None,
            body_id=None,
            body_name=None,
            issue=ContactIssue("invalid_record", "record must be an ObjectContactWrench"),
            calibration_verified=calibration_verified,
        )

    spec_issues = _validate_spec(spec)
    issues = list(record.issues)
    issues.extend(spec_issues)

    metadata_issues = _validate_record_metadata(record)
    issues.extend(metadata_issues)

    if record.sensor_valid is not True:
        if record.sensor_valid is None and not any(i.code == "missing_sensor_validity" for i in issues):
            issues.append(ContactIssue("missing_sensor_validity", "sensor_valid must be true to qualify", "sensor_valid"))
        elif record.sensor_valid is False and not any(i.code == "sensor_invalid" for i in issues):
            issues.append(ContactIssue("sensor_invalid", "sensor_valid is false", "sensor_valid"))
        else:
            issues.append(ContactIssue("invalid_sensor_validity", "sensor_valid must be true to qualify", "sensor_valid"))

    force = record.force_n
    torque = record.torque_nm
    force_issue = _vector_issue(force, field="force_n", mixed_code="mixed_wrench_not_force")
    torque_issue = _vector_issue(torque, field="torque_nm", mixed_code="mixed_wrench_not_torque")
    if force_issue is not None and not any(i.code == force_issue.code for i in issues):
        issues.append(force_issue)
    if torque_issue is not None and not any(i.code == torque_issue.code for i in issues):
        issues.append(torque_issue)

    if issues:
        return ContactForceEvaluation(
            decision=Decision.UNKNOWN,
            contact_state=ContactState.UNKNOWN,
            object_id=record.object_id,
            body_id=record.body_id,
            body_name=record.body_name,
            force_n=force,
            torque_nm=torque,
            force_norm_n=None,
            torque_norm_nm=None,
            issues=tuple(issues),
            calibration_verified=calibration_verified,
        )

    assert force is not None
    assert torque is not None
    force_norm = _norm3(force)
    torque_norm = _norm3(torque)
    assert force_norm is not None
    assert torque_norm is not None
    contact_state = (
        ContactState.CONTACT
        if force_norm > spec.contact_epsilon_n
        else ContactState.NO_CONTACT
    )

    decision = Decision.PASS
    decision_issues: list[ContactIssue] = []
    if spec.min_force_n is None and spec.max_force_n is None and not spec.require_contact:
        decision = Decision.UNKNOWN
        decision_issues.append(
            ContactIssue("no_force_constraint", "force spec has no bound and does not require contact")
        )
    else:
        if spec.require_contact and contact_state is ContactState.NO_CONTACT:
            decision = Decision.FAIL
            decision_issues.append(
                ContactIssue("no_contact", "valid force is zero; required contact was not observed")
            )
        if spec.min_force_n is not None and force_norm < spec.min_force_n:
            decision = Decision.FAIL
            decision_issues.append(
                ContactIssue(
                    "below_min_force",
                    f"force norm {force_norm:.9g} N is below minimum {spec.min_force_n:.9g} N",
                    "force_norm_n",
                )
            )
        if spec.max_force_n is not None and force_norm > spec.max_force_n:
            decision = Decision.FAIL
            decision_issues.append(
                ContactIssue(
                    "above_max_force",
                    f"force norm {force_norm:.9g} N is above maximum {spec.max_force_n:.9g} N",
                    "force_norm_n",
                )
            )

    return ContactForceEvaluation(
        decision=decision,
        contact_state=contact_state,
        object_id=record.object_id,
        body_id=record.body_id,
        body_name=record.body_name,
        force_n=force,
        torque_nm=torque,
        force_norm_n=force_norm,
        torque_norm_nm=torque_norm,
        issues=tuple(decision_issues),
        calibration_verified=calibration_verified,
    )


def _unknown_evaluation(
    *,
    object_id: Optional[str],
    body_id: Optional[int],
    body_name: Optional[str],
    issue: ContactIssue,
    calibration_verified: bool,
) -> ContactForceEvaluation:
    return ContactForceEvaluation(
        decision=Decision.UNKNOWN,
        contact_state=ContactState.UNKNOWN,
        object_id=object_id,
        body_id=body_id,
        body_name=body_name,
        force_n=None,
        torque_nm=None,
        force_norm_n=None,
        torque_norm_nm=None,
        issues=(issue,),
        calibration_verified=calibration_verified,
    )


def _validate_spec(spec: Any) -> list[ContactIssue]:
    if not isinstance(spec, ContactForceSpec):
        return [ContactIssue("invalid_force_spec", "spec must be a ContactForceSpec")]
    issues: list[ContactIssue] = []
    for name in ("min_force_n", "max_force_n"):
        value = getattr(spec, name)
        if value is not None and (_finite_scalar(value) is None or float(value) < 0.0):
            issues.append(ContactIssue("invalid_force_bound", f"{name} must be a finite non-negative scalar", name))
    epsilon = _finite_scalar(spec.contact_epsilon_n)
    if epsilon is None or epsilon < 0.0:
        issues.append(ContactIssue("invalid_contact_epsilon", "contact_epsilon_n must be finite and non-negative", "contact_epsilon_n"))
    if (
        spec.min_force_n is not None
        and spec.max_force_n is not None
        and _finite_scalar(spec.min_force_n) is not None
        and _finite_scalar(spec.max_force_n) is not None
        and float(spec.min_force_n) > float(spec.max_force_n)
    ):
        issues.append(ContactIssue("reversed_force_bounds", "min_force_n must not exceed max_force_n"))
    return issues


def _validate_record_metadata(record: ObjectContactWrench) -> list[ContactIssue]:
    issues: list[ContactIssue] = []
    if not record.object_id:
        issues.append(ContactIssue("missing_object_id", "object_id is required", "object_id"))
    if record.body_id is None or isinstance(record.body_id, bool) or not isinstance(record.body_id, int) or record.body_id < 0:
        issues.append(ContactIssue("invalid_body_id", "body_id must be a non-negative integer", "body_id"))
    if not record.body_name:
        issues.append(ContactIssue("missing_body_name", "body_name is required", "body_name"))
    if not record.geom_ids or len(record.geom_ids) != len(record.geom_names):
        issues.append(ContactIssue("missing_geom_binding", "object geom ids and names must be a non-empty aligned mapping", "geom_ids"))
    if record.frame != "world":
        issues.append(ContactIssue("unsupported_or_missing_frame", "cfrc_ext records must declare the world frame", "frame"))
    if record.geom_attribution != "body_aggregate":
        issues.append(ContactIssue("ambiguous_geom_attribution", "cfrc_ext geometry attribution must be body_aggregate", "geom_attribution"))
    if not record.source:
        issues.append(ContactIssue("missing_source", "source is required", "source"))
    return issues


def _coerce_binding(
    binding: ObjectBodyBinding | Mapping[str, Any] | None,
) -> tuple[Optional[ObjectBodyBinding], list[ContactIssue]]:
    if binding is None:
        return None, [ContactIssue("missing_object_binding", "object/body/geom binding is required", "binding")]
    if isinstance(binding, ObjectBodyBinding):
        raw = {
            "object_id": binding.object_id,
            "body_id": binding.body_id,
            "body_name": binding.body_name,
            "geom_ids": binding.geom_ids,
            "geom_names": binding.geom_names,
            "frame": binding.frame,
            "geom_attribution": binding.geom_attribution,
        }
    elif isinstance(binding, Mapping):
        raw = binding
    else:
        return None, [ContactIssue("invalid_object_binding", "binding must be ObjectBodyBinding or a mapping", "binding")]

    issues: list[ContactIssue] = []
    object_id = raw.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        issues.append(ContactIssue("missing_object_id", "object_id must be a non-empty string", "object_id"))
        object_id = None

    body_id = raw.get("body_id")
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        issues.append(ContactIssue("invalid_body_id", "body_id must be a non-negative integer", "body_id"))
        body_id = None

    body_name = raw.get("body_name")
    if not isinstance(body_name, str) or not body_name:
        issues.append(ContactIssue("missing_body_name", "body_name must be a non-empty string", "body_name"))
        body_name = None

    geom_ids = _integer_tuple(raw.get("geom_ids"))
    if geom_ids is None or not geom_ids or any(value < 0 for value in geom_ids):
        issues.append(ContactIssue("missing_geom_ids", "geom_ids must be a non-empty sequence of non-negative integers", "geom_ids"))
        geom_ids = ()

    geom_names = _string_tuple(raw.get("geom_names"))
    if geom_names is None or not geom_names or any(not value for value in geom_names):
        issues.append(ContactIssue("missing_geom_names", "geom_names must be a non-empty sequence of names", "geom_names"))
        geom_names = ()
    if geom_ids and geom_names and len(geom_ids) != len(geom_names):
        issues.append(ContactIssue("geom_mapping_length_mismatch", "geom_ids and geom_names must have equal length", "geom_names"))

    frame = raw.get("frame")
    if frame != "world":
        issues.append(ContactIssue("unsupported_or_missing_frame", "frame must be explicitly declared as world", "frame"))
        frame = None if frame is None else str(frame)

    geom_attribution = raw.get("geom_attribution")
    if geom_attribution != "body_aggregate":
        issues.append(ContactIssue("ambiguous_geom_attribution", "cfrc_ext can only be recorded as body_aggregate", "geom_attribution"))
        geom_attribution = None if geom_attribution is None else str(geom_attribution)

    normalized = ObjectBodyBinding(
        object_id=object_id or "",
        body_id=body_id if body_id is not None else -1,
        body_name=body_name or "",
        geom_ids=geom_ids,
        geom_names=geom_names,
        frame=frame or "",
        geom_attribution=geom_attribution or "",
    )
    return normalized, issues


def _select_body_row(cfrc_ext: Any, body_id: int) -> tuple[Any, Optional[ContactIssue]]:
    if isinstance(cfrc_ext, Mapping):
        if body_id not in cfrc_ext:
            return None, ContactIssue("missing_body_row", f"cfrc_ext has no row for body_id {body_id}", "cfrc_ext")
        return cfrc_ext[body_id], None
    try:
        # A row collection is required.  A flat six-vector would return a
        # scalar here and is rejected by _finite_vector below.
        row = cfrc_ext[body_id]
    except (IndexError, KeyError, TypeError):
        return None, ContactIssue("missing_body_row", f"cfrc_ext has no row for body_id {body_id}", "cfrc_ext")
    return row, None


def _finite_vector(
    value: Any,
    *,
    expected: int,
    field: str,
) -> tuple[Optional[Tuple[float, ...]], Optional[ContactIssue]]:
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None, ContactIssue("invalid_vector", f"{field} must be an iterable numeric vector", field)
    if len(values) != expected:
        return None, ContactIssue("wrong_vector_shape", f"{field} must have exactly {expected} coordinates", field)
    converted: list[float] = []
    for item in values:
        number = _finite_scalar(item)
        if number is None:
            return None, ContactIssue("nonfinite_vector", f"{field} must contain only finite numbers", field)
        converted.append(number)
    return tuple(converted), None


def _vector_issue(
    value: Optional[Tuple[float, ...]],
    *,
    field: str,
    mixed_code: str,
) -> Optional[ContactIssue]:
    if value is None:
        return ContactIssue("missing_vector", f"{field} is missing", field)
    try:
        length = len(value)
    except TypeError:
        return ContactIssue("invalid_vector", f"{field} must be a vector", field)
    if length == 6:
        return ContactIssue(
            mixed_code,
            f"{field} contains a six-dimensional force/torque wrench; mixed norms are rejected",
            field,
        )
    if length != 3:
        return ContactIssue("wrong_vector_shape", f"{field} must contain exactly three coordinates", field)
    if any(_finite_scalar(item) is None for item in value):
        return ContactIssue("nonfinite_vector", f"{field} must contain only finite numbers", field)
    return None


def _norm3(value: Optional[Sequence[float]]) -> Optional[float]:
    if value is None:
        return None
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    numbers = [_finite_scalar(item) for item in values]
    if any(item is None for item in numbers):
        return None
    return math.sqrt(sum(item * item for item in numbers if item is not None))


def _finite_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        return None
    return values


def _string_tuple(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None
    if any(not isinstance(item, str) for item in values):
        return None
    return values


__all__ = [
    "ContactForceEvaluation",
    "ContactForceSpec",
    "ContactIssue",
    "ContactState",
    "Decision",
    "ObjectBodyBinding",
    "ObjectContactWrench",
    "evaluate_object_contact_force",
    "record_object_contact_wrench",
]
