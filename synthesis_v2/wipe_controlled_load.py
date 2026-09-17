"""Pure-Python audit for the Wipe controlled-load sensor-channel probe.

The runtime probe applies a known world-frame wrench through MuJoCo's
``xfrc_applied`` to the Wipe wiping-gripper body.  This module audits the
resulting JSON evidence without importing MuJoCo, RoboSuite, NumPy, or a
controller.  It keeps the raw ``sensordata`` channel separate from
RoboSuite's filtered ``recent_ee_forcetorques`` channel.

Passing this audit means that the recorded simulator channel responded to the
known input within the supplied tolerances.  ``physical_qualification`` is
always ``unknown``: a MuJoCo channel identity check is not contact-model
validation, real hardware calibration, or a Wipe safety-threshold approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence


class Decision(str, Enum):
    """Strict three-state result used by case and suite audits."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


SENSOR_TRANSFORM = "-R_site.T @ applied_world"
RAW_CHANNEL = "raw_sensordata"
RECENT_CHANNEL = "recent_ee_forcetorques"
AUDITED_CHANNELS = (RAW_CHANNEL, RECENT_CHANNEL)


@dataclass(frozen=True)
class ControlledLoadSpec:
    """Expected Wipe binding and conservative evidence thresholds."""

    expected_body_id: int | None = 114
    expected_body_name: str = "gripper0_right_wiping_gripper"
    expected_site_name: str = "gripper0_right_ft_frame"
    expected_force_sensor_name: str = "gripper0_right_force_ee"
    expected_torque_sensor_name: str = "gripper0_right_torque_ee"
    expected_force_unit: str = "N"
    expected_torque_unit: str = "N*m"
    expected_application_frame: str = "world"
    expected_sensor_frame: str = "site"
    min_window_samples: int = 20
    max_gap_s: float = 0.1
    force_tolerance_n: float = 0.01
    torque_tolerance_nm: float = 1e-4
    cross_force_tolerance_n: float = 0.01
    cross_torque_tolerance_nm: float = 1e-4
    force_levels_n: tuple[float, ...] = (2.0, 5.0, 10.0)
    force_directions_world: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    torque_levels_nm: tuple[float, ...] = (0.05, 0.1)
    torque_directions_world: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class AuditIssue:
    """One evidence or response issue."""

    code: str
    message: str
    field: str | None = None
    sample_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "sample_index": self.sample_index,
        }


@dataclass(frozen=True)
class ControlledLoadCaseReport:
    """Audited case; physical qualification remains unknown by contract."""

    case_id: str
    case_type: str
    decision: Decision
    executed: bool
    sample_count: int
    force_norm_n: float | None
    torque_norm_nm: float | None
    channel_results: Mapping[str, Mapping[str, Any]]
    issues: tuple[AuditIssue, ...] = ()

    @property
    def physical_qualification(self) -> Decision:
        return Decision.UNKNOWN

    @property
    def codes(self) -> tuple[str, ...]:
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
            "force_norm_n": self.force_norm_n,
            "torque_norm_nm": self.torque_norm_nm,
            "channels": _json_value(self.channel_results),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class ControlledLoadSuiteReport:
    """Coverage and response result for a set of Wipe load cases."""

    decision: Decision
    case_reports: tuple[ControlledLoadCaseReport, ...]
    missing_cases: tuple[str, ...]
    duplicate_case_ids: tuple[str, ...]
    monotonicity_issues: tuple[AuditIssue, ...]
    physical_qualification: Decision = Decision.UNKNOWN
    promotion_allowed: bool = False

    @property
    def response_characterized(self) -> bool:
        return self.decision is Decision.PASS

    @property
    def coverage_complete(self) -> bool:
        return not self.missing_cases and not self.duplicate_case_ids

    @property
    def case_count(self) -> int:
        return len(self.case_reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "physical_qualification": self.physical_qualification.value,
            "promotion_allowed": self.promotion_allowed,
            "response_characterized": self.response_characterized,
            "coverage_complete": self.coverage_complete,
            "case_count": self.case_count,
            "missing_cases": list(self.missing_cases),
            "duplicate_case_ids": list(self.duplicate_case_ids),
            "monotonicity_issues": [issue.as_dict() for issue in self.monotonicity_issues],
            "cases": [case.as_dict() for case in self.case_reports],
        }


def expected_sensor_delta(
    force_world_n: Sequence[float],
    torque_world_nm: Sequence[float],
    site_rotation_world: Sequence[float],
) -> dict[str, tuple[float, float, float]]:
    """Compute the documented site-frame response ``-R_site.T @ wrench``.

    ``site_rotation_world`` is a row-major 3x3 rotation taking site-frame
    vectors to world coordinates.  The function is intentionally free of
    NumPy so tests can exercise the transform with identity and tilted frames.
    """

    force = _vector3(force_world_n, "force_world_n")
    torque = _vector3(torque_world_nm, "torque_world_nm")
    rotation = _matrix3(site_rotation_world, "site_rotation_world")
    return {
        "force_n": _negative_transpose_matvec(rotation, force),
        "torque_nm": _negative_transpose_matvec(rotation, torque),
    }


def audit_controlled_load_case(
    case: Mapping[str, Any],
    *,
    spec: ControlledLoadSpec | None = None,
) -> ControlledLoadCaseReport:
    """Audit one recorded no-contact/known-load case.

    The audit recomputes expected sensor deltas from each loaded sample's site
    rotation and compares each channel's loaded-window mean against its paired
    no-load baseline mean.  It never trusts a caller-provided error field as
    the source of truth.
    """

    if spec is None:
        spec = ControlledLoadSpec()
    issues: list[AuditIssue] = []
    if not isinstance(case, Mapping):
        return _case_report(
            "",
            "unknown",
            Decision.UNKNOWN,
            False,
            0,
            None,
            None,
            {},
            (AuditIssue("invalid_case", "case must be a mapping"),),
        )

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        issues.append(AuditIssue("missing_case_id", "case_id must be a non-empty string", "case_id"))
        case_id = ""
    case_type = case.get("case_type")
    if not isinstance(case_type, str) or not case_type:
        issues.append(AuditIssue("missing_case_type", "case_type must be a non-empty string", "case_type"))
        case_type = "unknown"

    executed = case.get("executed")
    if executed is not True:
        code = "case_not_executed" if executed is False else "missing_execution_marker"
        issues.append(AuditIssue(code, "only an explicitly executed case can be audited", "executed"))
    executed_bool = executed is True

    _audit_binding(case.get("binding"), spec, issues)
    _audit_load_metadata(case.get("known_load"), case_type, spec, issues)
    if case.get("sensor_transform") != SENSOR_TRANSFORM:
        issues.append(
            AuditIssue(
                "unexpected_sensor_transform",
                f"sensor_transform must be exactly {SENSOR_TRANSFORM!r}",
                "sensor_transform",
            )
        )

    top_reset = case.get("reset_id")
    if not isinstance(top_reset, str) or not top_reset:
        issues.append(AuditIssue("missing_reset_id", "reset_id must be explicit and non-empty", "reset_id"))
        top_reset = None
    if case.get("state_restored") is not True:
        issues.append(
            AuditIssue(
                "state_not_restored",
                "each load case must restore qpos/qvel/act/ctrl/time before the next case",
                "state_restored",
            )
        )

    load = case.get("known_load")
    force_world: tuple[float, float, float] | None = None
    torque_world: tuple[float, float, float] | None = None
    if isinstance(load, Mapping):
        try:
            force_world = _vector3(load.get("force_world_n"), "force_world_n")
        except (TypeError, ValueError):
            pass
        try:
            torque_world = _vector3(load.get("torque_world_nm"), "torque_world_nm")
        except (TypeError, ValueError):
            pass

    baseline = _samples(case.get("baseline_samples"), "baseline_samples", spec, top_reset, issues)
    loaded = _samples(case.get("loaded_samples"), "loaded_samples", spec, top_reset, issues)
    sample_count = min(len(baseline), len(loaded))
    channel_results: dict[str, Mapping[str, Any]] = {}

    if force_world is None or torque_world is None:
        force_norm = None
        torque_norm = None
    else:
        force_norm = _norm(force_world)
        torque_norm = _norm(torque_world)

    if not baseline or not loaded:
        issues.append(AuditIssue("missing_window", "both baseline_samples and loaded_samples are required"))
    elif force_world is not None and torque_world is not None:
        for channel in AUDITED_CHANNELS:
            channel_result = _audit_channel(
                channel,
                baseline,
                loaded,
                force_world,
                torque_world,
                spec,
                issues,
            )
            channel_results[channel] = channel_result

    structural_codes = {
        "invalid_case",
        "missing_case_id",
        "missing_case_type",
        "case_not_executed",
        "missing_execution_marker",
        "missing_binding",
        "missing_load",
        "missing_reset_id",
        "state_not_restored",
        "missing_window",
        "window_too_short",
        "invalid_sample",
        "invalid_timestamp",
        "non_monotonic_timestamp",
        "timestamp_gap",
        "reset_mismatch",
        "sensor_invalid",
        "missing_sensor_validity",
        "missing_channel",
        "invalid_sensor_vector",
        "mixed_wrench",
        "invalid_site_rotation",
        "contact_precondition_failed",
        "wrong_body",
        "wrong_site",
        "wrong_force_sensor",
        "wrong_torque_sensor",
        "wrong_geom_binding",
        "wrong_force_units",
        "wrong_torque_units",
        "wrong_application_frame",
        "wrong_sensor_frame",
        "unexpected_sensor_transform",
        "invalid_known_load",
        "missing_known_load",
        "invalid_load_case",
    }
    if any(issue.code in structural_codes for issue in issues):
        decision = Decision.UNKNOWN
    elif any(
        result.get("force_error_n", math.inf) > spec.force_tolerance_n
        or result.get("torque_error_nm", math.inf) > spec.torque_tolerance_nm
        or (
            result.get("cross_torque_nm") is not None
            and result["cross_torque_nm"] > spec.cross_torque_tolerance_nm
        )
        or (
            result.get("cross_force_n") is not None
            and result["cross_force_n"] > spec.cross_force_tolerance_n
        )
        for result in channel_results.values()
    ):
        decision = Decision.FAIL
        for channel, result in channel_results.items():
            if result.get("force_error_n", math.inf) > spec.force_tolerance_n:
                issues.append(AuditIssue("force_response_error", f"{channel} force response exceeds tolerance", channel))
            if result.get("torque_error_nm", math.inf) > spec.torque_tolerance_nm:
                issues.append(AuditIssue("torque_response_error", f"{channel} torque response exceeds tolerance", channel))
            if result.get("cross_torque_nm") is not None and result["cross_torque_nm"] > spec.cross_torque_tolerance_nm:
                issues.append(AuditIssue("cross_coupling_torque", f"{channel} force-only torque cross-coupling exceeds tolerance", channel))
            if result.get("cross_force_n") is not None and result["cross_force_n"] > spec.cross_force_tolerance_n:
                issues.append(AuditIssue("cross_coupling_force", f"{channel} torque-only force cross-coupling exceeds tolerance", channel))
    elif not channel_results:
        decision = Decision.UNKNOWN
    else:
        decision = Decision.PASS

    return _case_report(
        case_id,
        case_type,
        decision,
        executed_bool,
        sample_count,
        force_norm,
        torque_norm,
        channel_results,
        tuple(dict.fromkeys(issues)),
    )


def audit_controlled_load_suite(
    cases: Iterable[Mapping[str, Any]],
    *,
    spec: ControlledLoadSpec | None = None,
) -> ControlledLoadSuiteReport:
    """Audit coverage, case responses, and monotonic multi-level response."""

    if spec is None:
        spec = ControlledLoadSpec()
    try:
        materialized = tuple(cases)
    except (TypeError, ValueError):
        materialized = ()
    reports = tuple(audit_controlled_load_case(case, spec=spec) for case in materialized)
    ids = [report.case_id for report in reports if report.case_id]
    counts: dict[str, int] = {}
    for case_id in ids:
        counts[case_id] = counts.get(case_id, 0) + 1
    duplicate_ids = tuple(sorted(case_id for case_id, count in counts.items() if count > 1))

    missing: list[str] = []
    if not any(report.case_type == "no_contact" for report in reports):
        missing.append("no_contact")

    def has_level(case_type: str, direction: Sequence[float], level: float) -> bool:
        direction_tuple = tuple(float(value) for value in direction)
        for case, report in zip(materialized, reports):
            if report.case_type != case_type or report.decision is not Decision.PASS:
                continue
            load = case.get("known_load") if isinstance(case, Mapping) else None
            if not isinstance(load, Mapping):
                continue
            raw_direction = load.get("direction_world")
            raw_level = load.get("level")
            try:
                candidate_direction = _vector3(raw_direction, "direction_world")
                candidate_level = float(raw_level)
            except (TypeError, ValueError):
                continue
            if _vector_close(candidate_direction, direction_tuple) and math.isclose(candidate_level, level, rel_tol=0.0, abs_tol=1e-12):
                return True
        return False

    for direction in spec.force_directions_world:
        for level in spec.force_levels_n:
            key = f"force:{_vector_label(direction)}:{float(level):g}N"
            if not has_level("force_level", direction, level):
                missing.append(key)
    for direction in spec.torque_directions_world:
        for level in spec.torque_levels_nm:
            key = f"torque:{_vector_label(direction)}:{float(level):g}N*m"
            if not has_level("torque_level", direction, level):
                missing.append(key)

    monotonic_issues: list[AuditIssue] = []
    for case_type, directions, value_name, tolerance in (
        ("force_level", spec.force_directions_world, "response_norm_n", spec.force_tolerance_n),
        ("torque_level", spec.torque_directions_world, "response_norm_nm", spec.torque_tolerance_nm),
    ):
        for direction in directions:
            grouped: list[tuple[float, float, str]] = []
            for case, report in zip(materialized, reports):
                if report.case_type != case_type or report.decision is not Decision.PASS:
                    continue
                load = case.get("known_load") if isinstance(case, Mapping) else None
                if not isinstance(load, Mapping):
                    continue
                try:
                    candidate_direction = _vector3(load.get("direction_world"), "direction_world")
                    level = float(load.get("level"))
                    response = float(report.channel_results[RECENT_CHANNEL][value_name])
                except (KeyError, TypeError, ValueError):
                    continue
                if _vector_close(candidate_direction, direction):
                    grouped.append((level, response, report.case_id))
            grouped.sort(key=lambda item: item[0])
            for previous, current in zip(grouped, grouped[1:]):
                if current[1] + tolerance < previous[1]:
                    monotonic_issues.append(
                        AuditIssue(
                            "non_monotonic_response",
                            f"{case_type} response decreases from {previous[1]:.9g} to {current[1]:.9g}",
                            field=f"{previous[2]}->{current[2]}",
                        )
                    )

    if duplicate_ids:
        decision = Decision.FAIL
    elif any(report.decision is Decision.FAIL for report in reports):
        decision = Decision.FAIL
    elif monotonic_issues:
        decision = Decision.FAIL
    elif missing or any(report.decision is Decision.UNKNOWN for report in reports):
        decision = Decision.UNKNOWN
    elif not reports:
        decision = Decision.UNKNOWN
    else:
        decision = Decision.PASS

    return ControlledLoadSuiteReport(
        decision=decision,
        case_reports=reports,
        missing_cases=tuple(dict.fromkeys(missing)),
        duplicate_case_ids=duplicate_ids,
        monotonicity_issues=tuple(monotonic_issues),
    )


def _case_report(
    case_id: str,
    case_type: str,
    decision: Decision,
    executed: bool,
    sample_count: int,
    force_norm: float | None,
    torque_norm: float | None,
    channel_results: Mapping[str, Mapping[str, Any]],
    issues: Sequence[AuditIssue],
) -> ControlledLoadCaseReport:
    return ControlledLoadCaseReport(
        case_id=case_id,
        case_type=case_type,
        decision=decision,
        executed=executed,
        sample_count=sample_count,
        force_norm_n=force_norm,
        torque_norm_nm=torque_norm,
        channel_results=channel_results,
        issues=tuple(dict.fromkeys(issues)),
    )


def _audit_binding(binding: Any, spec: ControlledLoadSpec, issues: list[AuditIssue]) -> None:
    if not isinstance(binding, Mapping):
        issues.append(AuditIssue("missing_binding", "body/site/sensor/geom binding is required", "binding"))
        return
    body = binding.get("body")
    sensor = binding.get("sensor")
    if not isinstance(body, Mapping):
        issues.append(AuditIssue("missing_binding", "binding.body is required", "binding.body"))
    else:
        body_id = body.get("id")
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
            issues.append(AuditIssue("wrong_body", "body.id must be a non-negative integer", "binding.body.id"))
        elif spec.expected_body_id is not None and body_id != spec.expected_body_id:
            issues.append(AuditIssue("wrong_body", f"body id {body_id} != expected {spec.expected_body_id}", "binding.body.id"))
        if body.get("name") != spec.expected_body_name:
            issues.append(AuditIssue("wrong_body", f"body name must be {spec.expected_body_name!r}", "binding.body.name"))
        geom_ids = body.get("geom_ids")
        geom_names = body.get("geom_names")
        if not _nonempty_sequence(geom_ids) or not _nonempty_sequence(geom_names) or len(tuple(geom_ids)) != len(tuple(geom_names)):
            issues.append(AuditIssue("wrong_geom_binding", "body geom ids/names must be explicit, non-empty, and aligned", "binding.body"))
    if not isinstance(sensor, Mapping):
        issues.append(AuditIssue("missing_binding", "binding.sensor is required", "binding.sensor"))
    else:
        if sensor.get("site_name") != spec.expected_site_name:
            issues.append(AuditIssue("wrong_site", f"site name must be {spec.expected_site_name!r}", "binding.sensor.site_name"))
        if sensor.get("force_sensor_name") != spec.expected_force_sensor_name:
            issues.append(AuditIssue("wrong_force_sensor", f"force sensor must be {spec.expected_force_sensor_name!r}", "binding.sensor.force_sensor_name"))
        if sensor.get("torque_sensor_name") != spec.expected_torque_sensor_name:
            issues.append(AuditIssue("wrong_torque_sensor", f"torque sensor must be {spec.expected_torque_sensor_name!r}", "binding.sensor.torque_sensor_name"))
        recent = sensor.get("recent_channel")
        if not isinstance(recent, Mapping) or recent.get("source") != "robot.recent_ee_forcetorques" or recent.get("arm") != "right":
            issues.append(AuditIssue("missing_channel", "recent channel must identify robot.recent_ee_forcetorques arm=right", "binding.sensor.recent_channel"))


def _audit_load_metadata(
    load: Any,
    case_type: str,
    spec: ControlledLoadSpec,
    issues: list[AuditIssue],
) -> None:
    if not isinstance(load, Mapping):
        issues.append(AuditIssue("missing_load", "known_load is required", "known_load"))
        return
    if load.get("force_unit") != spec.expected_force_unit:
        issues.append(AuditIssue("wrong_force_units", "force_unit must be N", "known_load.force_unit"))
    if load.get("torque_unit") != spec.expected_torque_unit:
        issues.append(AuditIssue("wrong_torque_units", "torque_unit must be N*m", "known_load.torque_unit"))
    if load.get("application_frame") != spec.expected_application_frame:
        issues.append(AuditIssue("wrong_application_frame", "xfrc_applied must be declared in world frame", "known_load.application_frame"))
    if load.get("sensor_frame") != spec.expected_sensor_frame:
        issues.append(AuditIssue("wrong_sensor_frame", "sensor values must be declared in site frame", "known_load.sensor_frame"))
    try:
        force = _vector3(load.get("force_world_n"), "force_world_n")
        torque = _vector3(load.get("torque_world_nm"), "torque_world_nm")
    except (TypeError, ValueError) as exc:
        issues.append(AuditIssue("invalid_known_load", str(exc), "known_load"))
        return
    if case_type == "force_level":
        _audit_level_vector(load, force, torque, "force", issues)
    elif case_type == "torque_level":
        _audit_level_vector(load, force, torque, "torque", issues)
    elif case_type == "no_contact":
        if _norm(force) > 1e-12 or _norm(torque) > 1e-12:
            issues.append(AuditIssue("invalid_load_case", "no_contact case must apply zero force and torque", "known_load"))
    elif case_type not in {"combined", "no_contact"}:
        issues.append(AuditIssue("invalid_load_case", f"unrecognized case_type {case_type!r}", "case_type"))


def _audit_level_vector(
    load: Mapping[str, Any],
    force: Sequence[float],
    torque: Sequence[float],
    kind: str,
    issues: list[AuditIssue],
) -> None:
    try:
        direction = _vector3(load.get("direction_world"), "direction_world")
        level = float(load.get("level"))
    except (TypeError, ValueError) as exc:
        issues.append(AuditIssue("invalid_load_case", str(exc), "known_load"))
        return
    direction_norm = _norm(direction)
    if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        issues.append(AuditIssue("invalid_load_case", "direction_world must be a unit vector", "known_load.direction_world"))
    if not math.isfinite(level) or level <= 0.0:
        issues.append(AuditIssue("invalid_load_case", "non-zero load level must be positive and finite", "known_load.level"))
    expected_force = _scale(direction, level) if kind == "force" else (0.0, 0.0, 0.0)
    expected_torque = _scale(direction, level) if kind == "torque" else (0.0, 0.0, 0.0)
    if not _vector_close(force, expected_force, tolerance=1e-9) or not _vector_close(torque, expected_torque, tolerance=1e-9):
        issues.append(AuditIssue("invalid_load_case", f"{kind} vector does not equal direction_world*level", "known_load"))


def _samples(
    raw: Any,
    field: str,
    spec: ControlledLoadSpec,
    top_reset: str | None,
    issues: list[AuditIssue],
) -> list[Mapping[str, Any]]:
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        issues.append(AuditIssue("missing_window", f"{field} must be a sequence of samples", field))
        return []
    try:
        samples = list(raw or ())
    except (TypeError, ValueError):
        issues.append(AuditIssue("missing_window", f"{field} must be iterable", field))
        return []
    if len(samples) < spec.min_window_samples:
        issues.append(AuditIssue("window_too_short", f"{field} has {len(samples)} samples; need {spec.min_window_samples}", field))
    previous_time: float | None = None
    previous_reset: str | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            issues.append(AuditIssue("invalid_sample", "sample must be a mapping", field, index))
            continue
        try:
            timestamp = float(sample.get("timestamp_s"))
        except (TypeError, ValueError):
            timestamp = math.nan
        if not math.isfinite(timestamp):
            issues.append(AuditIssue("invalid_timestamp", "timestamp_s must be finite", field, index))
        elif previous_time is not None:
            delta = timestamp - previous_time
            if delta <= 0.0:
                issues.append(AuditIssue("non_monotonic_timestamp", "window timestamps must increase strictly", field, index))
            elif delta > spec.max_gap_s:
                issues.append(AuditIssue("timestamp_gap", f"sample gap {delta:g}s exceeds max_gap_s", field, index))
        previous_time = timestamp if math.isfinite(timestamp) else previous_time

        reset = sample.get("reset_id")
        if not isinstance(reset, str) or not reset:
            issues.append(AuditIssue("missing_reset_id", "sample reset_id must be explicit", field, index))
        else:
            if top_reset is not None and reset != top_reset:
                issues.append(AuditIssue("reset_mismatch", "sample reset_id differs from case reset_id", field, index))
            if previous_reset is not None and reset != previous_reset:
                issues.append(AuditIssue("reset_mismatch", "window crosses reset_id", field, index))
            previous_reset = reset

        if sample.get("sensor_valid") is not True:
            code = "sensor_invalid" if sample.get("sensor_valid") is False else "missing_sensor_validity"
            issues.append(AuditIssue(code, "sensor_valid must be explicitly true", field, index))
        contact_count = sample.get("selected_body_contact_count")
        if isinstance(contact_count, bool) or not isinstance(contact_count, int) or contact_count < 0:
            issues.append(AuditIssue("contact_precondition_failed", "selected_body_contact_count must be a non-negative integer", field, index))
        elif contact_count != 0:
            issues.append(AuditIssue("contact_precondition_failed", "known-load channel case requires zero selected-body contacts", field, index))
        for channel in AUDITED_CHANNELS:
            values = sample.get("channels")
            channel_values = values.get(channel) if isinstance(values, Mapping) else None
            if not isinstance(channel_values, Mapping):
                issues.append(AuditIssue("missing_channel", f"sample lacks {channel}", field, index))
                continue
            try:
                _vector3(channel_values.get("force_n"), f"{channel}.force_n")
                _vector3(channel_values.get("torque_nm"), f"{channel}.torque_nm")
            except (TypeError, ValueError) as exc:
                code = "mixed_wrench" if "six" in str(exc) else "invalid_sensor_vector"
                issues.append(AuditIssue(code, str(exc), f"{field}.{channel}", index))
            try:
                _matrix3(sample.get("site_rotation_world"), "site_rotation_world")
            except (TypeError, ValueError) as exc:
                issues.append(AuditIssue("invalid_site_rotation", str(exc), f"{field}.site_rotation_world", index))
    return [sample for sample in samples if isinstance(sample, Mapping)]


def _audit_channel(
    channel: str,
    baseline: Sequence[Mapping[str, Any]],
    loaded: Sequence[Mapping[str, Any]],
    force_world: Sequence[float],
    torque_world: Sequence[float],
    spec: ControlledLoadSpec,
    issues: list[AuditIssue],
) -> dict[str, Any]:
    try:
        baseline_force = [_vector3(sample["channels"][channel].get("force_n"), "force_n") for sample in baseline]
        baseline_torque = [_vector3(sample["channels"][channel].get("torque_nm"), "torque_nm") for sample in baseline]
        loaded_force = [_vector3(sample["channels"][channel].get("force_n"), "force_n") for sample in loaded]
        loaded_torque = [_vector3(sample["channels"][channel].get("torque_nm"), "torque_nm") for sample in loaded]
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(AuditIssue("missing_channel", str(exc), channel))
        return {}
    baseline_force_mean = _mean_vector(baseline_force)
    baseline_torque_mean = _mean_vector(baseline_torque)
    loaded_force_mean = _mean_vector(loaded_force)
    loaded_torque_mean = _mean_vector(loaded_torque)
    measured_force_delta = _subtract(loaded_force_mean, baseline_force_mean)
    measured_torque_delta = _subtract(loaded_torque_mean, baseline_torque_mean)

    expected_force_values: list[tuple[float, float, float]] = []
    expected_torque_values: list[tuple[float, float, float]] = []
    for sample in loaded:
        expected = expected_sensor_delta(
            force_world,
            torque_world,
            sample.get("site_rotation_world"),
        )
        expected_force_values.append(expected["force_n"])
        expected_torque_values.append(expected["torque_nm"])
    expected_force_delta = _mean_vector(expected_force_values)
    expected_torque_delta = _mean_vector(expected_torque_values)
    force_error = _norm(_subtract(measured_force_delta, expected_force_delta))
    torque_error = _norm(_subtract(measured_torque_delta, expected_torque_delta))
    force_only = _norm(force_world) > 1e-12 and _norm(torque_world) <= 1e-12
    torque_only = _norm(torque_world) > 1e-12 and _norm(force_world) <= 1e-12
    cross_torque = _norm(measured_torque_delta) if force_only else None
    cross_force = _norm(measured_force_delta) if torque_only else None
    return {
        "baseline_force_n": baseline_force_mean,
        "baseline_torque_nm": baseline_torque_mean,
        "measured_force_delta_n": measured_force_delta,
        "measured_torque_delta_nm": measured_torque_delta,
        "expected_force_delta_n": expected_force_delta,
        "expected_torque_delta_nm": expected_torque_delta,
        "force_error_n": force_error,
        "torque_error_nm": torque_error,
        "cross_torque_nm": cross_torque,
        "cross_force_n": cross_force,
        "response_norm_n": _norm(measured_force_delta),
        "response_norm_nm": _norm(measured_torque_delta),
        "sample_count_baseline": len(baseline),
        "sample_count_loaded": len(loaded),
    }


def _vector3(value: Any, field: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field} must be a numeric 3-vector, not a mapping/string")
    try:
        values = tuple(value)
    except TypeError:
        raise TypeError(f"{field} must be a numeric 3-vector") from None
    if len(values) == 6:
        raise ValueError(f"{field} is a mixed six-dimensional wrench")
    if len(values) != 3:
        raise ValueError(f"{field} must contain exactly three coordinates")
    converted = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{field} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{field} must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain finite numbers")
        converted.append(number)
    return tuple(converted)  # type: ignore[return-value]


def _matrix3(value: Any, field: str) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field} must be a finite 3x3 matrix")
    try:
        values = tuple(value)
    except TypeError:
        raise TypeError(f"{field} must be a finite 3x3 matrix") from None
    if len(values) == 9 and not any(_is_iterable(value) for value in values):
        flat = values
    else:
        try:
            rows = tuple(tuple(row) for row in values)
        except TypeError:
            raise TypeError(f"{field} must be a finite 3x3 matrix") from None
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError(f"{field} must be a finite 3x3 matrix")
        flat = tuple(item for row in rows for item in row)
    converted = []
    for item in flat:
        if isinstance(item, bool):
            raise ValueError(f"{field} must contain finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{field} must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain finite numbers")
        converted.append(number)
    return tuple(tuple(converted[row * 3 + column] for column in range(3)) for row in range(3))


def _negative_transpose_matvec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        -sum(float(matrix[row][column]) * float(vector[row]) for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _mean_vector(values: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot average an empty vector sequence")
    return tuple(sum(float(value[index]) for value in values) / len(values) for index in range(3))  # type: ignore[return-value]


def _subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def _scale(vector: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(value) * float(scalar) for value in vector)  # type: ignore[return-value]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _vector_close(left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-12) -> bool:
    return len(tuple(left)) == len(tuple(right)) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _vector_label(vector: Sequence[float]) -> str:
    return ",".join(f"{float(value):g}" for value in vector)


def _nonempty_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or value is None:
        return False
    try:
        return len(tuple(value)) > 0
    except (TypeError, ValueError):
        return False


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    try:
        iter(value)
        return True
    except TypeError:
        return False


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "AUDITED_CHANNELS",
    "AuditIssue",
    "ControlledLoadCaseReport",
    "ControlledLoadSpec",
    "ControlledLoadSuiteReport",
    "Decision",
    "RAW_CHANNEL",
    "RECENT_CHANNEL",
    "SENSOR_TRANSFORM",
    "audit_controlled_load_case",
    "audit_controlled_load_suite",
    "expected_sensor_delta",
]
