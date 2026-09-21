"""Offline audits for planned versus executed stage semantics.

The execution runner records three different things that are easy to confuse:
the request, the candidate proposal, and the parameters actually handed to the
controller.  This module checks the last two without turning a mismatch into a
physical manner verdict.  A physical verdict still requires the independent
calibration and final-audit contracts.
"""
from __future__ import annotations

from math import isclose
from typing import Any, Mapping


class SemanticAuditError(ValueError):
    """Raised when an execution-semantic record is malformed or inconsistent."""


# These are the ratios used by ``run_manner_ablation.spec_to_exec_params``.
# Keeping the public audit copy here makes the check usable without importing
# the simulator or the optional robosuite stack.
PHASE_DEFAULT_RATIOS: dict[str, tuple[float, float]] = {
    "approach": (1.0, 1.0),
    "descend": (0.6, 0.5),
    "lift": (1.0, 1.0),
    "transport": (0.5, 0.4),
    "final": (0.5, 0.4),
    "retreat": (1.0, 1.0),
}
V_MIN_DEFAULT = 0.02


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticAuditError(f"{path} must be numeric")
    return float(value)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticAuditError(f"{path} must be an object")
    return value


def _derive_velocity(max_bound: Any, min_bound: Any, default: float) -> float:
    max_value = None if max_bound is None else _number(max_bound, "max_joint_vel")
    min_value = None if min_bound is None else _number(min_bound, "min_joint_vel")
    if max_value is not None and min_value is not None:
        value = (max_value * 0.85 + min_value * 1.05) / 2
    elif max_value is not None:
        value = max_value * 0.85
    elif min_value is not None:
        value = min_value * 1.10
    else:
        value = default
    return max(V_MIN_DEFAULT, value)


def _close(actual: Any, expected: float, *, path: str) -> bool:
    try:
        value = _number(actual, path)
    except SemanticAuditError:
        return False
    return isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def _global_expected(candidate: Mapping[str, Any]) -> dict[str, float | None]:
    global_spec = _mapping(candidate.get("global", {}), "candidate_preset.global")
    v_max = _derive_velocity(
        global_spec.get("max_joint_vel"),
        global_spec.get("min_joint_vel"),
        0.12,
    )
    return {
        "v_max": v_max,
        "a_max": max(0.01, v_max * 0.7),
        "jerk_limit": (
            None
            if global_spec.get("max_jerk") is None
            else _number(global_spec["max_jerk"], "candidate_preset.global.max_jerk")
        ),
    }


def _phase_expected(
    candidate: Mapping[str, Any],
    phase: str,
    actual: Mapping[str, Any],
    neutral: Mapping[str, Any] | None,
) -> tuple[str, float, float]:
    phase_spec = _mapping(candidate.get("phase_velocity", {}), "candidate_preset.phase_velocity")
    explicit = phase in phase_spec
    bounds = phase_spec.get(phase)
    if explicit and not isinstance(bounds, Mapping):
        raise SemanticAuditError(f"candidate_preset.phase_velocity.{phase} must be an object")

    if explicit and not bounds:
        if neutral is None:
            raise SemanticAuditError(
                f"{phase}: explicit neutral requires a same-task neutral baseline"
            )
        neutral_v = _mapping(neutral.get("phase_v_max"), "neutral.phase_v_max")
        neutral_a = _mapping(neutral.get("phase_a_max"), "neutral.phase_a_max")
        try:
            return (
                "explicit_neutral",
                _number(neutral_v[phase], f"neutral.phase_v_max.{phase}"),
                _number(neutral_a[phase], f"neutral.phase_a_max.{phase}"),
            )
        except KeyError as exc:
            raise SemanticAuditError(f"neutral baseline lacks phase {phase!r}") from exc

    if explicit:
        actual_v = _number(actual.get("v_max"), "executed_parameters.v_max")
        bounds = dict(bounds)
        value = _derive_velocity(
            bounds.get("max_joint_vel"),
            bounds.get("min_joint_vel"),
            actual_v,
        )
        return "explicit_override", value, max(0.01, value * 0.7)

    actual_v = _number(actual.get("v_max"), "executed_parameters.v_max")
    actual_a = _number(actual.get("a_max"), "executed_parameters.a_max")
    ratio_v, ratio_a = PHASE_DEFAULT_RATIOS[phase]
    return "omitted_inherits_global", max(0.02, actual_v * ratio_v), max(0.01, actual_a * ratio_a)


def _phase_observation(phase_metrics: Any, phase: str) -> dict[str, Any]:
    """Return descriptive metrics for a phase without assigning pass/fail."""
    metrics = phase_metrics if isinstance(phase_metrics, Mapping) else {}
    value = metrics.get(phase)
    if value is None:
        value = metrics.get(f"move:{phase}")
    if not isinstance(value, Mapping):
        return {"status": "unknown", "reason": "phase_metrics_missing"}
    return {"status": "observed", "metrics": dict(value)}


def _request_semantic_diff(
    request: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe request-to-candidate changes without judging physical quality."""
    if not isinstance(request, Mapping) or not isinstance(request.get("requested_manner"), Mapping):
        return {
            "status": "unknown",
            "reason": "request.requested_manner_missing",
            "request_spec": None,
            "requested_spec": None,
            "candidate_spec": dict(candidate),
            "candidate_matches_request": None,
            "differences": [],
        }

    requested = request["requested_manner"]
    requested_global = _mapping(requested.get("global", {}), "request.requested_manner.global")
    requested_phases = _mapping(
        requested.get("phase_velocity", {}), "request.requested_manner.phase_velocity"
    )
    candidate_global = _mapping(candidate.get("global", {}), "candidate_preset.global")
    candidate_phases = _mapping(
        candidate.get("phase_velocity", {}), "candidate_preset.phase_velocity"
    )
    differences: list[dict[str, Any]] = []

    for field in sorted(set(requested_global) | set(candidate_global)):
        request_value = requested_global.get(field)
        candidate_value = candidate_global.get(field)
        if field not in requested_global:
            kind = "candidate_global_extra"
        elif field not in candidate_global:
            kind = "requested_global_missing"
        elif request_value != candidate_value:
            kind = "global_constraint_changed"
        else:
            continue
        differences.append(
            {"kind": kind, "path": f"global.{field}", "requested": request_value, "candidate": candidate_value}
        )

    for phase in sorted(set(requested_phases) | set(candidate_phases)):
        request_value = requested_phases.get(phase)
        candidate_value = candidate_phases.get(phase)
        if phase not in requested_phases:
            kind = "candidate_phase_extra"
        elif phase not in candidate_phases:
            kind = "requested_phase_missing"
        elif request_value != candidate_value:
            kind = "phase_constraint_changed"
        else:
            continue
        differences.append(
            {
                "kind": kind,
                "path": f"phase_velocity.{phase}",
                "requested": request_value,
                "candidate": candidate_value,
            }
        )

    requested_is_neutral = not requested_global and not requested_phases
    candidate_has_constraints = bool(candidate_global) or any(
        bool(value) for value in candidate_phases.values()
    )
    if requested_is_neutral and candidate_has_constraints:
        differences.insert(
            0,
            {
                "kind": "neutral_extra_constraints",
                "path": "candidate_preset",
                "requested": {"global": {}, "phase_velocity": {}},
                "candidate": {
                    "global": dict(candidate_global),
                    "phase_velocity": dict(candidate_phases),
                },
            },
        )

    requested_spec = {
        "global": dict(requested_global),
        "phase_velocity": dict(requested_phases),
    }
    candidate_spec = {
        "global": dict(candidate_global),
        "phase_velocity": dict(candidate_phases),
    }
    return {
        "status": "available",
        "request_spec": requested_spec,
        "requested_spec": requested_spec,
        "candidate_spec": candidate_spec,
        "candidate_preset": candidate_spec,
        "requested_target_phases": sorted(str(phase) for phase in requested_phases),
        "candidate_target_phases": sorted(str(phase) for phase in candidate_phases),
        "candidate_non_target_phases": [
            phase for phase in PHASE_DEFAULT_RATIOS if phase not in candidate_phases
        ],
        "candidate_matches_request": not differences,
        "differences": differences,
    }


def audit_phase_semantics(
    candidate_preset: Mapping[str, Any],
    executed_parameters: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
    neutral_parameters: Mapping[str, Any] | None = None,
    phase_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit target/non-target phase scope and applied execution parameters.

    A phase listed in the candidate's ``phase_velocity`` is a candidate target
    phase.  An empty mapping is an explicit neutral target and must match the
    same-task neutral baseline.  A phase absent from that mapping is a candidate
    non-target phase and must inherit the candidate's global profile.  When the
    original request is supplied, request-to-candidate differences are reported
    separately.  Those differences do not change the execution check or infer a
    physical manner verdict.
    """
    candidate = _mapping(candidate_preset, "candidate_preset")
    actual = _mapping(executed_parameters, "executed_parameters")
    phase_spec = _mapping(candidate.get("phase_velocity", {}), "candidate_preset.phase_velocity")
    actual_v = _mapping(actual.get("phase_v_max"), "executed_parameters.phase_v_max")
    actual_a = _mapping(actual.get("phase_a_max"), "executed_parameters.phase_a_max")

    unknown_targets = set(phase_spec) - set(PHASE_DEFAULT_RATIOS)
    if unknown_targets:
        raise SemanticAuditError(f"candidate_preset contains unknown phases: {sorted(unknown_targets)}")
    unknown_phases = (set(actual_v) | set(actual_a)) - set(PHASE_DEFAULT_RATIOS)
    if unknown_phases:
        raise SemanticAuditError(f"executed_parameters contains unknown phases: {sorted(unknown_phases)}")
    missing_v = set(PHASE_DEFAULT_RATIOS) - set(actual_v)
    missing_a = set(PHASE_DEFAULT_RATIOS) - set(actual_a)
    if missing_v or missing_a:
        raise SemanticAuditError(
            "executed_parameters lacks phase fields: "
            f"phase_v_max={sorted(missing_v)}, phase_a_max={sorted(missing_a)}"
        )

    errors: list[str] = []
    global_expected = _global_expected(candidate)
    global_checks: dict[str, Any] = {}
    for field, expected in global_expected.items():
        if expected is None:
            continue
        passed = _close(actual.get(field), expected, path=f"executed_parameters.{field}")
        global_checks[field] = {"expected": expected, "actual": actual.get(field), "passed": passed}
        if not passed:
            errors.append(f"global {field}: expected {expected!r}, got {actual.get(field)!r}")

    phase_reports: dict[str, Any] = {}
    target_phases = sorted(str(phase) for phase in phase_spec)
    for phase in PHASE_DEFAULT_RATIOS:
        try:
            scope, expected_v, expected_a = _phase_expected(candidate, phase, actual, neutral_parameters)
        except SemanticAuditError as exc:
            errors.append(str(exc))
            phase_reports[phase] = {"scope": "invalid", "parameter_check": "fail", "error": str(exc)}
            continue
        passed_v = _close(actual_v.get(phase), expected_v, path=f"executed_parameters.phase_v_max.{phase}")
        passed_a = _close(actual_a.get(phase), expected_a, path=f"executed_parameters.phase_a_max.{phase}")
        passed = passed_v and passed_a
        if not passed:
            errors.append(
                f"{phase} ({scope}): expected v/a=({expected_v!r}, {expected_a!r}), "
                f"got ({actual_v.get(phase)!r}, {actual_a.get(phase)!r})"
            )
        phase_reports[phase] = {
            "scope": scope,
            "parameter_check": "pass" if passed else "fail",
            "expected": {"phase_v_max": expected_v, "phase_a_max": expected_a},
            "actual": {"phase_v_max": actual_v.get(phase), "phase_a_max": actual_a.get(phase)},
            "behavior_observation": _phase_observation(phase_metrics, phase),
        }

    request_diff = _request_semantic_diff(request, candidate)
    return {
        "schema": "mannercap-phase-semantic-audit-v1",
        "valid": not errors,
        "requested_target_phases": request_diff.get("requested_target_phases", []),
        "candidate_target_phases": target_phases,
        "candidate_non_target_phases": [
            phase for phase in PHASE_DEFAULT_RATIOS if phase not in phase_spec
        ],
        "request_semantic_diff": request_diff,
        "global_parameters": global_checks,
        "phases": phase_reports,
        "errors": errors,
        "manner_qualification": "unknown",
        "physical_interpretation": "descriptive_execution_semantics_only",
    }


def require_valid_phase_semantics(report: Mapping[str, Any], *, context: str = "semantic audit") -> None:
    """Raise an explicit error for a failed semantic audit.

    This helper deliberately uses a normal branch rather than ``assert`` so
    optimized Python runs cannot disable an execution-integrity check.
    """
    if report.get("valid") is not True:
        errors = report.get("errors")
        raise SemanticAuditError(f"{context} semantic audit failed: {errors!r}")


__all__ = [
    "PHASE_DEFAULT_RATIOS",
    "SemanticAuditError",
    "audit_phase_semantics",
    "require_valid_phase_semantics",
]
