"""Pure structural audit for object-contact force calibration fixtures.

This module describes the evidence needed before a MuJoCo ``cfrc_ext`` stream
can be used for Gate1 force calibration.  It does not run MuJoCo, infer a
calibration constant, or alter a simulator.  A case can therefore pass the
numeric/structural checks while its ``physical_qualification`` remains
``unknown`` until an independent calibration run and sign-off exist.

The case vocabulary intentionally includes nominal controls and failure modes:
no contact, controlled load response, missing sensor, wrong units, wrong
object/body/geom, a mixed six-dimensional wrench, timestamp/reset gaps, and
under/over-action.  Invalid or incomplete evidence is never zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .contact_force_contract import Decision


class CalibrationCaseType(str, Enum):
    """Registered calibration and anomaly case identifiers."""

    NO_CONTACT = "no_contact"
    CONTROLLED_LOAD_RESPONSE = "controlled_load_response"
    MISSING_SENSOR = "missing_sensor"
    WRONG_FORCE_UNITS = "wrong_force_units"
    WRONG_TORQUE_UNITS = "wrong_torque_units"
    WRONG_OBJECT = "wrong_object"
    WRONG_BODY = "wrong_body"
    WRONG_GEOM = "wrong_geom"
    MIXED_WRENCH = "mixed_wrench"
    TIMESTAMP_GAP = "timestamp_gap"
    RESET_GAP = "reset_gap"
    UNDER_ACTION = "under_action"
    OVER_ACTION = "over_action"


# Thirteen entries keep under-action and over-action independently auditable.
REQUIRED_CALIBRATION_CASE_TYPES: Tuple[str, ...] = tuple(
    case_type.value for case_type in CalibrationCaseType
)


@dataclass(frozen=True)
class CalibrationSample:
    """One immutable snapshot from a prospective calibration run.

    ``force_n`` and ``torque_nm`` are deliberately separate three-vectors.  A
    six-vector in either field is retained as invalid evidence and causes an
    ``unknown`` case result rather than a mixed-unit norm.
    """

    timestamp_s: Optional[float]
    reset_id: Optional[str]
    object_id: Optional[str]
    body_id: Optional[int]
    body_name: Optional[str]
    geom_ids: Tuple[int, ...]
    geom_names: Tuple[str, ...]
    force_n: Optional[Tuple[float, ...]]
    torque_nm: Optional[Tuple[float, ...]]
    sensor_valid: Optional[bool]
    units_force: Optional[str] = "N"
    units_torque: Optional[str] = "N*m"
    frame: Optional[str] = "mujoco_subtree_com"
    geom_attribution: Optional[str] = "body_aggregate"
    known_load_n: Optional[float] = None


@dataclass(frozen=True)
class CalibrationCase:
    """A named calibration fixture or anomaly sample set.

    ``executed`` is explicit because a planned fixture must not be counted as
    an observed calibration result.  Expected identity fields are optional at
    construction time so missing registry metadata can be reported as
    ``unknown`` by the audit.
    """

    case_id: str
    case_type: CalibrationCaseType | str
    samples: Tuple[CalibrationSample, ...]
    executed: bool = False
    expected_object_id: Optional[str] = "cube_a"
    expected_body_id: Optional[int] = 4
    expected_body_name: Optional[str] = "cube_a_body"
    expected_geom_ids: Tuple[int, ...] = (17,)
    expected_geom_names: Tuple[str, ...] = ("cube_a_geom",)
    expected_frame: str = "mujoco_subtree_com"
    expected_geom_attribution: str = "body_aggregate"
    min_force_n: Optional[float] = None
    max_force_n: Optional[float] = None
    max_dt_s: Optional[float] = 0.2
    contact_epsilon_n: float = 1e-9
    response_tolerance_n: float = 1e-9


@dataclass(frozen=True)
class CalibrationIssue:
    """One structural or response issue found in a calibration case."""

    code: str
    message: str
    sample_index: Optional[int] = None
    field: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sample_index": self.sample_index,
            "field": self.field,
        }


@dataclass(frozen=True)
class CalibrationCaseReport:
    """Strict result for one case, with no claim of physical calibration."""

    case_id: str
    case_type: str
    decision: Decision
    executed: bool
    sample_count: int
    force_norms_n: Tuple[float, ...]
    torque_norms_nm: Tuple[float, ...]
    dt_s: Tuple[float, ...]
    reset_ids: Tuple[str, ...]
    issues: Tuple[CalibrationIssue, ...] = ()

    @property
    def physical_qualification(self) -> Decision:
        """Always unknown until an external calibration sign-off is supplied."""

        return Decision.UNKNOWN

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def has_code(self, code: str) -> bool:
        return code in self.codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "decision": self.decision.value,
            "physical_qualification": self.physical_qualification.value,
            "executed": self.executed,
            "sample_count": self.sample_count,
            "force_norms_n": list(self.force_norms_n),
            "torque_norms_nm": list(self.torque_norms_nm),
            "dt_s": list(self.dt_s),
            "reset_ids": list(self.reset_ids),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CalibrationSuiteReport:
    """Coverage report for a registered set of calibration/anomaly cases."""

    decision: Decision
    case_reports: Tuple[CalibrationCaseReport, ...]
    missing_case_types: Tuple[str, ...]
    duplicate_case_ids: Tuple[str, ...]
    unknown_case_types: Tuple[str, ...]
    physical_qualification: Decision = Decision.UNKNOWN

    @property
    def case_count(self) -> int:
        return len(self.case_reports)

    @property
    def observed_case_count(self) -> int:
        return sum(report.executed for report in self.case_reports)

    @property
    def coverage_complete(self) -> bool:
        return not self.missing_case_types and not self.unknown_case_types and not self.duplicate_case_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "physical_qualification": self.physical_qualification.value,
            "case_count": self.case_count,
            "observed_case_count": self.observed_case_count,
            "coverage_complete": self.coverage_complete,
            "missing_case_types": list(self.missing_case_types),
            "duplicate_case_ids": list(self.duplicate_case_ids),
            "unknown_case_types": list(self.unknown_case_types),
            "cases": [report.as_dict() for report in self.case_reports],
        }


def audit_calibration_case(case: CalibrationCase) -> CalibrationCaseReport:
    """Audit one case without executing or mutating its samples.

    A valid sample set produces a numeric ``pass``/``fail`` result only when
    all required provenance and units are present.  Missing, invalid, stale,
    or ambiguous evidence produces ``unknown``.  An anomaly case is useful
    precisely because it is rejected or marked unknown; it is not expected to
    pass the nominal contact contract.
    """

    if not isinstance(case, CalibrationCase):
        return CalibrationCaseReport(
            case_id="",
            case_type="unknown",
            decision=Decision.UNKNOWN,
            executed=False,
            sample_count=0,
            force_norms_n=(),
            torque_norms_nm=(),
            dt_s=(),
            reset_ids=(),
            issues=(CalibrationIssue("invalid_case", "case must be a CalibrationCase"),),
        )

    case_type = _case_type_value(case.case_type)
    issues: list[CalibrationIssue] = []
    if case_type is None:
        case_type = str(case.case_type)
        issues.append(CalibrationIssue("unknown_case_type", f"unregistered case type {case_type!r}"))
    if not isinstance(case.case_id, str) or not case.case_id:
        issues.append(CalibrationIssue("missing_case_id", "case_id must be non-empty", field="case_id"))
    if not isinstance(case.executed, bool):
        issues.append(CalibrationIssue("invalid_execution_marker", "executed must be boolean", field="executed"))
    elif not case.executed:
        issues.append(CalibrationIssue("case_not_executed", "planned/unrun case cannot qualify calibration", field="executed"))

    try:
        samples = case.samples if isinstance(case.samples, tuple) else tuple(case.samples or ())
    except (TypeError, ValueError):
        samples = ()
        issues.append(CalibrationIssue("invalid_samples_container", "samples must be an iterable of CalibrationSample", field="samples"))
    if not samples:
        issues.append(CalibrationIssue("empty_case", "case has no calibration samples", field="samples"))

    if _finite_nonnegative(case.contact_epsilon_n) is None:
        issues.append(CalibrationIssue("invalid_contact_epsilon", "contact_epsilon_n must be finite and non-negative", field="contact_epsilon_n"))
    if _finite_nonnegative(case.max_dt_s) is None and case.max_dt_s is not None:
        issues.append(CalibrationIssue("invalid_max_dt", "max_dt_s must be finite and non-negative", field="max_dt_s"))
    if _finite_nonnegative(case.response_tolerance_n) is None:
        issues.append(CalibrationIssue("invalid_response_tolerance", "response_tolerance_n must be finite and non-negative", field="response_tolerance_n"))
    for name in ("min_force_n", "max_force_n"):
        value = getattr(case, name)
        if value is not None and _finite_nonnegative(value) is None:
            issues.append(CalibrationIssue("invalid_force_bound", f"{name} must be finite and non-negative", field=name))
    if (
        case.min_force_n is not None
        and case.max_force_n is not None
        and _finite_nonnegative(case.min_force_n) is not None
        and _finite_nonnegative(case.max_force_n) is not None
        and float(case.min_force_n) > float(case.max_force_n)
    ):
        issues.append(CalibrationIssue("reversed_force_bounds", "min_force_n must not exceed max_force_n"))

    force_norms: list[float] = []
    torque_norms: list[float] = []
    times: list[float] = []
    reset_ids: list[str] = []
    valid_indices: list[int] = []

    for index, sample in enumerate(samples):
        if not isinstance(sample, CalibrationSample):
            issues.append(CalibrationIssue("invalid_sample", "sample must be a CalibrationSample", index))
            continue
        sample_ok = True
        sample_issues = _audit_sample_metadata(sample, case, index)
        if sample_issues:
            issues.extend(sample_issues)
            sample_ok = False

        timestamp = _finite_scalar(sample.timestamp_s)
        if timestamp is None:
            issues.append(CalibrationIssue("missing_or_invalid_timestamp", "timestamp_s must be finite", index, "timestamp_s"))
            sample_ok = False
        else:
            times.append(timestamp)

        if sample.reset_id is None or not isinstance(sample.reset_id, str) or not sample.reset_id:
            issues.append(CalibrationIssue("missing_reset_id", "reset_id must be non-empty", index, "reset_id"))
            sample_ok = False
        else:
            reset_ids.append(sample.reset_id)

        if sample.sensor_valid is not True:
            code = "missing_sensor_validity" if sample.sensor_valid is None else "sensor_invalid"
            issues.append(CalibrationIssue(code, "sensor_valid must be true for calibration evidence", index, "sensor_valid"))
            sample_ok = False

        force, force_issue = _vector(sample.force_n, field="force_n", allow_mixed=True)
        if force_issue is not None:
            issues.append(CalibrationIssue(force_issue[0], force_issue[1], index, "force_n"))
            sample_ok = False
        torque, torque_issue = _vector(sample.torque_nm, field="torque_nm", allow_mixed=True)
        if torque_issue is not None:
            issues.append(CalibrationIssue(torque_issue[0], torque_issue[1], index, "torque_nm"))
            sample_ok = False
        if force is not None:
            force_norms.append(_norm3(force))
        if torque is not None:
            torque_norms.append(_norm3(torque))
        if sample_ok and force is not None and torque is not None:
            valid_indices.append(index)

    _audit_timing(times, reset_ids, case, issues)

    # Any structural problem prevents a nominal numeric verdict.  Anomaly
    # cases intentionally use these issues as the reason for ``unknown``.
    structural_unknown = bool(issues)
    if structural_unknown:
        decision = Decision.UNKNOWN
    elif case_type == CalibrationCaseType.NO_CONTACT.value:
        decision = _decision_no_contact(force_norms, case, issues)
    elif case_type == CalibrationCaseType.CONTROLLED_LOAD_RESPONSE.value:
        decision = _decision_controlled_response(samples, valid_indices, force_norms, case, issues)
    elif case_type == CalibrationCaseType.UNDER_ACTION.value:
        decision = _decision_under_action(force_norms, case, issues)
    elif case_type == CalibrationCaseType.OVER_ACTION.value:
        decision = _decision_over_action(force_norms, case, issues)
    elif case_type in {
        CalibrationCaseType.MISSING_SENSOR.value,
        CalibrationCaseType.WRONG_FORCE_UNITS.value,
        CalibrationCaseType.WRONG_TORQUE_UNITS.value,
        CalibrationCaseType.WRONG_OBJECT.value,
        CalibrationCaseType.WRONG_BODY.value,
        CalibrationCaseType.WRONG_GEOM.value,
        CalibrationCaseType.MIXED_WRENCH.value,
        CalibrationCaseType.TIMESTAMP_GAP.value,
        CalibrationCaseType.RESET_GAP.value,
    }:
        # With no issue, the named anomaly was not reproduced in the fixture.
        issues.append(CalibrationIssue("anomaly_not_reproduced", f"case type {case_type} has no detectable anomaly"))
        decision = Decision.FAIL
    else:
        decision = Decision.UNKNOWN

    return CalibrationCaseReport(
        case_id=case.case_id if isinstance(case.case_id, str) else "",
        case_type=case_type,
        decision=decision,
        executed=case.executed is True,
        sample_count=len(samples),
        force_norms_n=tuple(force_norms),
        torque_norms_nm=tuple(torque_norms),
        dt_s=tuple(_deltas(times)),
        reset_ids=tuple(reset_ids),
        issues=tuple(issues),
    )


def audit_calibration_suite(
    cases: Iterable[CalibrationCase],
    *,
    required_case_types: Sequence[str] = REQUIRED_CALIBRATION_CASE_TYPES,
) -> CalibrationSuiteReport:
    """Audit coverage and all cases without claiming the suite calibrated.

    The suite decision describes coverage/evidence only.  Its
    ``physical_qualification`` is deliberately fixed at ``unknown`` even
    when every registered fixture is present and numerically well formed.
    """

    try:
        materialized = tuple(cases)
    except TypeError:
        materialized = ()
    reports = tuple(audit_calibration_case(case) for case in materialized)
    observed_types = {report.case_type for report in reports}
    if isinstance(required_case_types, (str, CalibrationCaseType)):
        required_case_types = (required_case_types,)
    required = tuple(str(value.value if isinstance(value, Enum) else value) for value in required_case_types)
    missing = tuple(case_type for case_type in required if case_type not in observed_types)
    unknown_types = tuple(sorted({report.case_type for report in reports if report.case_type not in REQUIRED_CALIBRATION_CASE_TYPES}))

    counts: dict[str, int] = {}
    for report in reports:
        if report.case_id:
            counts[report.case_id] = counts.get(report.case_id, 0) + 1
    duplicate_ids = tuple(sorted(case_id for case_id, count in counts.items() if count > 1))

    if not materialized or missing or unknown_types or duplicate_ids:
        decision = Decision.FAIL
    elif any(report.decision is Decision.UNKNOWN for report in reports):
        decision = Decision.UNKNOWN
    elif any(report.decision is Decision.FAIL for report in reports):
        decision = Decision.FAIL
    else:
        decision = Decision.PASS

    return CalibrationSuiteReport(
        decision=decision,
        case_reports=reports,
        missing_case_types=missing,
        duplicate_case_ids=duplicate_ids,
        unknown_case_types=unknown_types,
    )


def _audit_sample_metadata(
    sample: CalibrationSample,
    case: CalibrationCase,
    index: int,
) -> list[CalibrationIssue]:
    issues: list[CalibrationIssue] = []
    if sample.object_id is None:
        issues.append(CalibrationIssue("missing_object_id", "object_id is required", index, "object_id"))
    elif case.expected_object_id is not None and sample.object_id != case.expected_object_id:
        issues.append(CalibrationIssue("wrong_object", f"expected object {case.expected_object_id!r}, got {sample.object_id!r}", index, "object_id"))
    if sample.body_id is None:
        issues.append(CalibrationIssue("missing_body_id", "body_id is required", index, "body_id"))
    elif case.expected_body_id is not None and sample.body_id != case.expected_body_id:
        issues.append(CalibrationIssue("wrong_body", f"expected body_id {case.expected_body_id}, got {sample.body_id}", index, "body_id"))
    if sample.body_name is None:
        issues.append(CalibrationIssue("missing_body_name", "body_name is required", index, "body_name"))
    elif case.expected_body_name is not None and sample.body_name != case.expected_body_name:
        issues.append(CalibrationIssue("wrong_body", f"expected body_name {case.expected_body_name!r}, got {sample.body_name!r}", index, "body_name"))

    if not sample.geom_ids or not sample.geom_names:
        issues.append(CalibrationIssue("missing_geom_binding", "geom ids/names must be non-empty", index, "geom_ids"))
    elif len(sample.geom_ids) != len(sample.geom_names):
        issues.append(CalibrationIssue("geom_mapping_length_mismatch", "geom ids/names must be aligned", index, "geom_names"))
    elif case.expected_geom_ids and tuple(sample.geom_ids) != tuple(case.expected_geom_ids):
        issues.append(CalibrationIssue("wrong_geom", "sample geom ids do not match expected object mapping", index, "geom_ids"))
    elif case.expected_geom_names and tuple(sample.geom_names) != tuple(case.expected_geom_names):
        issues.append(CalibrationIssue("wrong_geom", "sample geom names do not match expected object mapping", index, "geom_names"))

    if sample.units_force != "N":
        issues.append(CalibrationIssue("wrong_force_units" if sample.units_force is not None else "missing_force_units", "force units must be explicitly N", index, "units_force"))
    if sample.units_torque != "N*m":
        issues.append(CalibrationIssue("wrong_torque_units" if sample.units_torque is not None else "missing_torque_units", "torque units must be explicitly N*m", index, "units_torque"))
    if sample.frame != case.expected_frame:
        issues.append(CalibrationIssue("wrong_frame", f"frame must be {case.expected_frame!r}", index, "frame"))
    if sample.geom_attribution != case.expected_geom_attribution:
        issues.append(CalibrationIssue("ambiguous_geom_attribution", "cfrc_ext geometry must remain body_aggregate", index, "geom_attribution"))
    return issues


def _audit_timing(
    times: Sequence[float],
    reset_ids: Sequence[str],
    case: CalibrationCase,
    issues: list[CalibrationIssue],
) -> None:
    if len(reset_ids) > 1 and len(set(reset_ids)) > 1:
        issues.append(CalibrationIssue("reset_gap", "sample window crosses reset_id", field="reset_id"))
    deltas = _deltas(times)
    for index, delta in enumerate(deltas, start=1):
        if delta <= 0.0:
            issues.append(CalibrationIssue("timestamp_nonpositive", "timestamps must increase strictly", index, "timestamp_s"))
        if case.max_dt_s is not None and delta > float(case.max_dt_s):
            issues.append(CalibrationIssue("timestamp_gap", f"dt {delta:.9g}s exceeds max_dt_s {float(case.max_dt_s):.9g}s", index, "timestamp_s"))


def _decision_no_contact(
    force_norms: Sequence[float],
    case: CalibrationCase,
    issues: list[CalibrationIssue],
) -> Decision:
    if not force_norms:
        issues.append(CalibrationIssue("missing_force_evidence", "no-contact case needs a valid force vector"))
        return Decision.UNKNOWN
    if any(value > float(case.contact_epsilon_n) for value in force_norms):
        issues.append(CalibrationIssue("contact_present", "non-zero force was observed in no-contact case"))
        return Decision.FAIL
    return Decision.PASS


def _decision_controlled_response(
    samples: Sequence[CalibrationSample],
    valid_indices: Sequence[int],
    force_norms: Sequence[float],
    case: CalibrationCase,
    issues: list[CalibrationIssue],
) -> Decision:
    if len(valid_indices) < 2:
        issues.append(CalibrationIssue("insufficient_controlled_samples", "controlled response needs at least two valid samples"))
        return Decision.UNKNOWN
    selected = [samples[index] for index in valid_indices]
    if any(sample.known_load_n is None or _finite_nonnegative(sample.known_load_n) is None for sample in selected):
        issues.append(CalibrationIssue("missing_known_load", "controlled response needs finite known_load_n for every valid sample"))
        return Decision.UNKNOWN
    pairs = sorted(
        ((float(sample.known_load_n), force_norms[pos]) for pos, sample in enumerate(selected)),
        key=lambda pair: pair[0],
    )
    distinct_loads = {load for load, _ in pairs}
    if len(distinct_loads) < 2:
        issues.append(CalibrationIssue("constant_controlled_load", "controlled response needs at least two distinct known loads"))
        return Decision.UNKNOWN
    for index in range(1, len(pairs)):
        previous = pairs[index - 1][1]
        current = pairs[index][1]
        if current + float(case.response_tolerance_n) < previous:
            issues.append(CalibrationIssue("non_monotonic_load_response", "measured force decreases as known load increases", index, "force_n"))
            return Decision.FAIL
    if max(value for _, value in pairs) - min(value for _, value in pairs) <= float(case.response_tolerance_n):
        issues.append(CalibrationIssue("constant_force_response", "measured force did not respond to controlled load change", field="force_n"))
        return Decision.FAIL
    return Decision.PASS


def _decision_under_action(
    force_norms: Sequence[float],
    case: CalibrationCase,
    issues: list[CalibrationIssue],
) -> Decision:
    if case.min_force_n is None or _finite_nonnegative(case.min_force_n) is None:
        issues.append(CalibrationIssue("missing_under_action_bound", "under_action needs min_force_n"))
        return Decision.UNKNOWN
    if not force_norms:
        issues.append(CalibrationIssue("missing_force_evidence", "under_action needs a valid force vector"))
        return Decision.UNKNOWN
    if all(value < float(case.min_force_n) for value in force_norms):
        issues.append(CalibrationIssue("under_action", "force stayed below the required lower bound", field="force_norm_n"))
        return Decision.FAIL
    issues.append(CalibrationIssue("anomaly_not_reproduced", "force was not consistently under the lower bound"))
    return Decision.FAIL


def _decision_over_action(
    force_norms: Sequence[float],
    case: CalibrationCase,
    issues: list[CalibrationIssue],
) -> Decision:
    if case.max_force_n is None or _finite_nonnegative(case.max_force_n) is None:
        issues.append(CalibrationIssue("missing_over_action_bound", "over_action needs max_force_n"))
        return Decision.UNKNOWN
    if not force_norms:
        issues.append(CalibrationIssue("missing_force_evidence", "over_action needs a valid force vector"))
        return Decision.UNKNOWN
    if all(value > float(case.max_force_n) for value in force_norms):
        issues.append(CalibrationIssue("over_action", "force stayed above the allowed upper bound", field="force_norm_n"))
        return Decision.FAIL
    issues.append(CalibrationIssue("anomaly_not_reproduced", "force was not consistently over the upper bound"))
    return Decision.FAIL


def _case_type_value(value: Any) -> Optional[str]:
    if isinstance(value, CalibrationCaseType):
        return value.value
    if isinstance(value, str) and value in REQUIRED_CALIBRATION_CASE_TYPES:
        return value
    return None


def _vector(
    value: Any,
    *,
    field: str,
    allow_mixed: bool,
) -> tuple[Optional[Tuple[float, ...]], Optional[tuple[str, str]]]:
    if value is None:
        return None, ("missing_vector", f"{field} is missing")
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None, ("invalid_vector", f"{field} must be an iterable numeric vector")
    if len(values) == 6 and allow_mixed:
        return None, ("mixed_wrench", f"{field} contains a mixed six-dimensional force/torque wrench")
    if len(values) != 3:
        return None, ("wrong_vector_shape", f"{field} must contain exactly three coordinates")
    converted: list[float] = []
    for item in values:
        number = _finite_scalar(item)
        if number is None:
            return None, ("nonfinite_vector", f"{field} must contain only finite numbers")
        converted.append(number)
    return tuple(converted), None


def _norm3(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _deltas(times: Sequence[float]) -> list[float]:
    return [times[index] - times[index - 1] for index in range(1, len(times))]


def _finite_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_nonnegative(value: Any) -> Optional[float]:
    number = _finite_scalar(value)
    return number if number is not None and number >= 0.0 else None


__all__ = [
    "CalibrationCase",
    "CalibrationCaseReport",
    "CalibrationCaseType",
    "CalibrationIssue",
    "CalibrationSample",
    "CalibrationSuiteReport",
    "REQUIRED_CALIBRATION_CASE_TYPES",
    "audit_calibration_case",
    "audit_calibration_suite",
]
