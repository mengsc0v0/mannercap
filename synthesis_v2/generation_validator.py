"""Strict generation-time validation for the MannerCAP v2 contract.

This module is a small, dependency-free *public* generator gate.  It checks
the evidence emitted by a trajectory generator and returns independent
``task``, ``manner`` and ``safety`` decisions.  It deliberately has no
knowledge of a controller, solver, method name, or the sealed final-audit
dataset.  The final audit may consume an exported trace, but it is a separate
consumer and must not be replaced by this generation-time feedback path.

The contract is conservative by design:

* each decision is ``pass``, ``fail`` or ``unknown``;
* missing evidence, bad units, failed sensors, a reset boundary, a repeated
  timestamp, or a sampling gap cannot be promoted to ``pass``;
* target phases form one continuous interval and every target sample is
  checked, so a brief threshold hit cannot hide sustained under-action; and
* the immutable requested constraints are compared with the executed
  constraints.  Removing a bound by clipping is a known failure, not a new
  unconstrained pass.

Only Python's standard library is used.  Threshold values here are protocol
inputs; this module does not claim that they have been physically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Iterable, Mapping, Sequence


class Decision(str, Enum):
    """The only admissible state for each evaluation dimension."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationSample:
    """One measured sample in a generated trajectory.

    ``None`` is an absent measurement and is never converted into a benign
    default.  ``sensor_valid`` and ``reset_id`` are intentionally nullable so
    that callers cannot accidentally omit evidence while receiving a passing
    default.  ``t`` is measured wall/simulation time in seconds; no fixed
    frame period is assumed.
    """

    t: float | None
    phase: str | None
    velocity: float | Sequence[float] | None = None
    acceleration: float | Sequence[float] | None = None
    kp: float | None = None
    gripper_fraction: float | None = None
    contact_force_n: float | Sequence[float] | None = None
    sensor_valid: bool | None = None
    reset_id: str | int | None = None
    # Optional measured interval.  When present it must agree with adjacent
    # timestamps; when absent, the validator computes the interval from t.
    dt_s: float | None = None


@dataclass(frozen=True)
class GenerationTrace:
    """Immutable samples plus exact unit declarations."""

    samples: tuple[GenerationSample, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[GenerationSample],
        *,
        units: Mapping[str, str] | None = None,
    ) -> "GenerationTrace":
        """Materialize an iterable without mutating the caller's container."""

        return cls(tuple(samples), dict(units or {}))


@dataclass(frozen=True)
class ConstraintSpec:
    """Stage-scoped generation constraints.

    ``target_phases`` identifies the one interval where manner rules apply.
    ``safety_max_force_n`` is independent and applies to every sample.  A
    caller should construct one instance from the original request and one
    from the executed plan when clipping or adaptation is possible.
    """

    stage_id: str = "generation"
    target_phases: tuple[str, ...] = ()

    max_velocity: float | None = None
    min_velocity: float | None = None
    max_kp: float | None = None
    min_kp: float | None = None
    max_gripper_fraction: float | None = None
    min_gripper_fraction: float | None = None
    max_jerk: float | None = None

    # Hard safety cap.  It has no manner tolerance.
    safety_max_force_n: float | None = None

    # Protocol values are part of the request and therefore are also
    # protected from silent relaxation by clipping.
    tolerance: float = 0.0
    min_samples: int = 1
    min_duration_s: float = 0.0
    max_gap_s: float = 0.1

    def has_manner_rule(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_velocity,
                self.min_velocity,
                self.max_kp,
                self.min_kp,
                self.max_gripper_fraction,
                self.min_gripper_fraction,
                self.max_jerk,
            )
        )

    def has_safety_rule(self) -> bool:
        return self.safety_max_force_n is not None

    def has_any_rule(self) -> bool:
        return self.has_manner_rule() or self.has_safety_rule()


@dataclass(frozen=True)
class GenerationCase:
    """Inputs to :func:`validate_generation`.

    ``effective_constraints=None`` means that the requested spec was used as
    is.  To represent an execution-side clipped spec, pass an explicit
    ``ConstraintSpec`` with the removed fields set to ``None``.  This
    distinction prevents a missing override argument from being confused with
    deliberate removal of a requested bound.
    """

    trace: GenerationTrace | None
    requested_constraints: ConstraintSpec
    effective_constraints: ConstraintSpec | None = None
    task: Decision | bool | str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Per-dimension and joint result returned by the generator gate."""

    task: Decision
    manner: Decision
    safety: Decision
    overall: Decision
    reasons: tuple[str, ...] = ()

    @property
    def publishable(self) -> bool:
        """Whether all three dimensions are known to pass."""

        return self.overall is Decision.PASS

    @property
    def qualifies(self) -> bool:
        """Alias used by generator callers that prefer qualification wording."""

        return self.publishable

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result without adding a serialization dependency."""

        return {
            "task": self.task.value,
            "manner": self.manner.value,
            "safety": self.safety.value,
            "overall": self.overall.value,
            "publishable": self.publishable,
            "reasons": list(self.reasons),
        }


# Public aliases make the contract easy to discover beside the final-audit
# module, while keeping the generation boundary explicit in the primary names.
TraceSample = GenerationSample
Trace = GenerationTrace
GenerationConstraintSpec = ConstraintSpec


_UNIT_CONTRACT = {
    "time": "s",
    "velocity": "rad/s",
    "acceleration": "rad/s^2",
    "kp": "gain",
    "gripper_fraction": "fraction",
    "contact_force": "N",
}

_MANNER_UPPER_FIELDS = (
    "max_velocity",
    "max_kp",
    "max_gripper_fraction",
    "max_jerk",
)
_MANNER_LOWER_FIELDS = (
    "min_velocity",
    "min_kp",
    "min_gripper_fraction",
)
def validate_generation(case: GenerationCase) -> GenerationResult:
    """Validate one generated trace with conservative three-state semantics.

    The function does not execute a controller and does not consult a method
    name.  It is safe to call on development traces, including failed or
    partially recorded attempts.  A final-audit implementation may reuse the
    data model, but its sealed reference and acceptance set are intentionally
    outside this module.
    """

    reasons: list[str] = []
    task, task_reason = _coerce_decision(case.task)
    if task_reason is not None:
        reasons.append(task_reason)

    if not isinstance(case.requested_constraints, ConstraintSpec):
        reasons.append("invalid_requested_constraint_spec")
        manner = Decision.UNKNOWN
        safety = Decision.UNKNOWN
        if task is Decision.UNKNOWN:
            reasons.append("task_status_unknown")
        elif task is Decision.FAIL:
            reasons.append("task_failed")
        return _result(task, manner, safety, reasons)

    requested = case.requested_constraints
    effective = case.effective_constraints
    if effective is None:
        effective = requested

    requested_issues = _constraint_issues(requested)
    effective_issues = _constraint_issues(effective)
    if requested_issues:
        reasons.extend(f"requested:{issue}" for issue in requested_issues)
    if effective_issues:
        reasons.extend(f"effective:{issue}" for issue in effective_issues)

    # A no-log attempt must remain unknown even when a side-channel says the
    # request was clipped.  There is no evidence on which to publish a result.
    trace = case.trace
    if not isinstance(trace, GenerationTrace) or not trace.samples:
        reasons.append("missing_trace" if trace is None else "empty_trace")
        manner = Decision.UNKNOWN
        safety = Decision.UNKNOWN
    elif requested_issues or effective_issues:
        # Invalid rules cannot be interpreted as either success or violation.
        # A valid requested rule may still be known to have been clipped; the
        # clipping check below is intentionally kept for that case.
        (manner, manner_reasons), (safety, safety_reasons) = _unknown_for_invalid_specs(
            requested, effective, requested_issues, effective_issues
        )
        reasons.extend(manner_reasons)
        reasons.extend(safety_reasons)
    else:
        manner, safety, structural_reasons = _evaluate_dimensions(
            trace, requested, effective
        )
        reasons.extend(structural_reasons)

    if task is Decision.UNKNOWN:
        reasons.append("task_status_unknown")
    elif task is Decision.FAIL:
        reasons.append("task_failed")

    return _result(task, manner, safety, reasons)


def evaluate_generation(case: GenerationCase) -> GenerationResult:
    """Readable alias for :func:`validate_generation`."""

    return validate_generation(case)


def _result(
    task: Decision,
    manner: Decision,
    safety: Decision,
    reasons: Iterable[str],
) -> GenerationResult:
    reason_tuple = tuple(dict.fromkeys(str(reason) for reason in reasons))
    return GenerationResult(
        task=task,
        manner=manner,
        safety=safety,
        overall=_joint_decision(task, manner, safety),
        reasons=reason_tuple,
    )


def _coerce_decision(value: Decision | bool | str | None) -> tuple[Decision, str | None]:
    if value is None:
        return Decision.UNKNOWN, None
    if isinstance(value, Decision):
        return value, None
    if isinstance(value, bool):
        return (Decision.PASS if value else Decision.FAIL), None
    if isinstance(value, str):
        normalized = value.strip().lower()
        for decision in Decision:
            if normalized == decision.value:
                return decision, None
    return Decision.UNKNOWN, "invalid_task_status"


def _joint_decision(*decisions: Decision) -> Decision:
    """Apply the fixed joint rule: fail dominates, all pass is required."""

    if any(decision is Decision.FAIL for decision in decisions):
        return Decision.FAIL
    if all(decision is Decision.PASS for decision in decisions):
        return Decision.PASS
    return Decision.UNKNOWN


def _constraint_issues(spec: ConstraintSpec) -> list[str]:
    """Return configuration defects; defects are unknown, not violations."""

    if not isinstance(spec, ConstraintSpec):
        return ["not_constraint_spec"]

    issues: list[str] = []
    if not spec.has_any_rule():
        issues.append("no_constraint")
    if not isinstance(spec.stage_id, str) or not spec.stage_id.strip():
        issues.append("missing_stage_id")

    phases = spec.target_phases
    if not isinstance(phases, (tuple, list)) or not phases:
        issues.append("missing_target_phases")
    else:
        seen: set[str] = set()
        for phase in phases:
            if not isinstance(phase, str) or not phase.strip():
                issues.append("invalid_target_phase")
            elif phase in seen:
                issues.append("duplicate_target_phase")
            seen.add(phase)

    for field_name in (
        "max_velocity",
        "min_velocity",
        "max_kp",
        "min_kp",
        "max_gripper_fraction",
        "min_gripper_fraction",
        "max_jerk",
        "safety_max_force_n",
        "tolerance",
        "min_duration_s",
        "max_gap_s",
    ):
        value = getattr(spec, field_name)
        if value is not None and not _finite_scalar(value):
            issues.append(f"non_finite:{field_name}")

    if isinstance(spec.min_samples, bool) or not isinstance(spec.min_samples, int):
        issues.append("invalid_min_samples")
    elif spec.min_samples < 1:
        issues.append("invalid_min_samples")

    if _finite_scalar(spec.tolerance) and not 0.0 <= float(spec.tolerance) < 1.0:
        issues.append("invalid_tolerance")
    if _finite_scalar(spec.min_duration_s) and spec.min_duration_s < 0:
        issues.append("invalid_min_duration")
    if _finite_scalar(spec.max_gap_s) and spec.max_gap_s <= 0:
        issues.append("invalid_max_gap")

    for field_name in (
        "max_velocity",
        "min_velocity",
        "max_kp",
        "min_kp",
        "max_gripper_fraction",
        "min_gripper_fraction",
        "max_jerk",
        "safety_max_force_n",
    ):
        value = getattr(spec, field_name)
        if value is not None and _finite_scalar(value) and float(value) < 0:
            issues.append(f"negative_bound:{field_name}")

    for low_name, high_name in (
        ("min_velocity", "max_velocity"),
        ("min_kp", "max_kp"),
        ("min_gripper_fraction", "max_gripper_fraction"),
    ):
        low = getattr(spec, low_name)
        high = getattr(spec, high_name)
        if (
            low is not None
            and high is not None
            and _finite_scalar(low)
            and _finite_scalar(high)
            and float(low) > float(high)
        ):
            issues.append(f"inverted_bounds:{low_name}:{high_name}")

    for field_name in ("min_gripper_fraction", "max_gripper_fraction"):
        value = getattr(spec, field_name)
        if value is not None and _finite_scalar(value) and not 0.0 <= float(value) <= 1.0:
            issues.append(f"fraction_out_of_range:{field_name}")

    return list(dict.fromkeys(issues))


def _unknown_for_invalid_specs(
    requested: ConstraintSpec,
    effective: ConstraintSpec,
    requested_issues: Sequence[str],
    effective_issues: Sequence[str],
) -> tuple[tuple[Decision, tuple[str, ...]], tuple[Decision, tuple[str, ...]]]:
    """Classify malformed configs while preserving a known clipping failure."""

    # If the requested contract itself is malformed, no dimension can be
    # judged.  This includes an actually unconstrained request.
    if requested_issues:
        return (
            Decision.UNKNOWN,
            tuple(f"manner_config_unknown:{issue}" for issue in requested_issues),
        ), (
            Decision.UNKNOWN,
            tuple(f"safety_config_unknown:{issue}" for issue in requested_issues),
        )

    clipping = _clipping_reasons(requested, effective)
    manner_clip = tuple(
        reason for reason in clipping if reason.startswith("manner_clip:")
    )
    safety_clip = tuple(
        reason for reason in clipping if reason.startswith("safety_clip:")
    )
    manner = (
        Decision.FAIL if manner_clip else Decision.UNKNOWN,
        manner_clip
        + tuple(f"manner_config_unknown:{issue}" for issue in effective_issues),
    )
    safety = (
        Decision.FAIL if safety_clip else Decision.UNKNOWN,
        safety_clip
        + tuple(f"safety_config_unknown:{issue}" for issue in effective_issues),
    )
    return manner, safety


def _evaluate_dimensions(
    trace: GenerationTrace,
    requested: ConstraintSpec,
    effective: ConstraintSpec,
) -> tuple[Decision, Decision, tuple[str, ...]]:
    """Evaluate both dimensions after configuration validation."""

    structural = _trace_structure(trace, requested)
    reasons: list[str] = []
    reasons.extend(structural.common)
    reasons.extend(structural.manner)
    reasons.extend(structural.safety)

    clipping = _clipping_reasons(requested, effective)
    manner_clip = [r for r in clipping if r.startswith("manner_clip:")]
    safety_clip = [r for r in clipping if r.startswith("safety_clip:")]

    # Common evidence defects affect both dimensions.  A phase defect only
    # affects manner; a missing force unit/value only affects safety.
    manner_unknown = bool(structural.common or structural.manner)
    safety_unknown = bool(structural.common or structural.safety)

    if manner_clip:
        manner = Decision.FAIL
        reasons.extend(manner_clip)
    elif manner_unknown:
        manner = Decision.UNKNOWN
    else:
        manner, manner_reasons = _evaluate_manner(trace, effective)
        reasons.extend(manner_reasons)

    if safety_clip:
        safety = Decision.FAIL
        reasons.extend(safety_clip)
    elif safety_unknown:
        safety = Decision.UNKNOWN
    else:
        safety, safety_reasons = _evaluate_safety(trace, effective)
        reasons.extend(safety_reasons)

    return manner, safety, tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class _Structure:
    common: tuple[str, ...] = ()
    manner: tuple[str, ...] = ()
    safety: tuple[str, ...] = ()


def _trace_structure(trace: GenerationTrace, spec: ConstraintSpec) -> _Structure:
    """Check shared time/reset evidence and dimension-specific metadata."""

    common: list[str] = []
    manner: list[str] = []
    safety: list[str] = []
    if isinstance(trace.units, Mapping):
        units = trace.units
    else:
        units = {}
        common.append("invalid_units")

    required_manner_units: set[str] = set()
    if spec.max_velocity is not None or spec.min_velocity is not None:
        required_manner_units.add("velocity")
    if spec.max_jerk is not None:
        required_manner_units.update(("velocity", "acceleration"))
    if spec.max_kp is not None or spec.min_kp is not None:
        required_manner_units.add("kp")
    if (
        spec.max_gripper_fraction is not None
        or spec.min_gripper_fraction is not None
    ):
        required_manner_units.add("gripper_fraction")

    # Time and force are independently required by the relevant dimensions.
    if units.get("time") != _UNIT_CONTRACT["time"]:
        common.append(
            f"unit_mismatch:time:{units.get('time')!r}!={_UNIT_CONTRACT['time']!r}"
        )
    for field_name in sorted(required_manner_units):
        expected = _UNIT_CONTRACT[field_name]
        actual = units.get(field_name)
        if actual != expected:
            manner.append(f"unit_mismatch:{field_name}:{actual!r}!={expected!r}")
    if spec.safety_max_force_n is not None:
        expected = _UNIT_CONTRACT["contact_force"]
        actual = units.get("contact_force")
        if actual != expected:
            safety.append(f"unit_mismatch:contact_force:{actual!r}!={expected!r}")

    previous: GenerationSample | None = None
    for index, sample in enumerate(trace.samples):
        if not isinstance(sample, GenerationSample):
            common.append(f"invalid_sample:index={index}")
            previous = None
            continue

        if not _finite(sample.t):
            common.append(f"invalid_time:index={index}")
        if sample.reset_id is None:
            common.append(f"missing_reset_id:index={index}")
        if sample.sensor_valid is not True:
            common.append(
                f"sensor_invalid:index={index}"
                if sample.sensor_valid is False
                else f"sensor_validity_missing:index={index}"
            )
        if sample.phase is None or not isinstance(sample.phase, str) or not sample.phase.strip():
            manner.append(f"missing_phase:index={index}")

        if previous is not None and _finite(previous.t) and _finite(sample.t):
            if sample.reset_id != previous.reset_id:
                common.append(f"reset_boundary:index={index}")
            else:
                dt = float(sample.t) - float(previous.t)
                if dt <= 0:
                    common.append(f"non_monotonic_time:index={index}")
                elif dt > spec.max_gap_s:
                    common.append(f"sampling_gap:index={index}:dt={dt:g}")
                if sample.dt_s is not None:
                    if not _finite(sample.dt_s) or float(sample.dt_s) <= 0:
                        common.append(f"invalid_dt:index={index}")
                    elif abs(float(sample.dt_s) - dt) > max(1e-9, 1e-6 * abs(dt)):
                        common.append(f"dt_mismatch:index={index}")
        elif previous is not None and sample.reset_id != previous.reset_id:
            common.append(f"reset_boundary:index={index}")

        previous = sample

    # The target interval is checked here even when no manner value is
    # configured.  An unconstrained request still returns unknown; this reason
    # makes the missing scope inspectable in logs.
    if not spec.target_phases:
        manner.append("missing_target_phase")
    else:
        target_indices = [
            index
            for index, sample in enumerate(trace.samples)
            if isinstance(sample, GenerationSample)
            and sample.phase in spec.target_phases
        ]
        if not target_indices:
            manner.append("missing_target_phase")
        else:
            start, end = target_indices[0], target_indices[-1]
            if end - start + 1 != len(target_indices):
                manner.append("phase_interval_not_contiguous")
            else:
                order = {phase: index for index, phase in enumerate(spec.target_phases)}
                target_phases = [trace.samples[index].phase for index in target_indices]
                numeric_order = [order.get(phase, -1) for phase in target_phases]
                if any(right < left for left, right in zip(numeric_order, numeric_order[1:])):
                    manner.append("phase_order_invalid")
            if len(target_indices) < spec.min_samples:
                manner.append(
                    f"insufficient_target_samples:{len(target_indices)}<{spec.min_samples}"
                )
            else:
                first = trace.samples[target_indices[0]]
                last = trace.samples[target_indices[-1]]
                if _finite(first.t) and _finite(last.t):
                    duration = float(last.t) - float(first.t)
                    if duration + 1e-9 < spec.min_duration_s:
                        manner.append(
                            f"insufficient_target_duration:{duration:g}<{spec.min_duration_s:g}"
                        )

    # Numeric shape/finite checks are dimension-specific.  A missing velocity
    # does not erase a valid force measurement, and vice versa.
    manner_field_names: set[str] = set()
    if spec.max_velocity is not None or spec.min_velocity is not None:
        manner_field_names.add("velocity")
    if spec.max_jerk is not None:
        manner_field_names.add("acceleration")
    if spec.max_kp is not None or spec.min_kp is not None:
        manner_field_names.add("kp")
    if spec.max_gripper_fraction is not None or spec.min_gripper_fraction is not None:
        manner_field_names.add("gripper_fraction")

    target_indices = [
        index
        for index, sample in enumerate(trace.samples)
        if isinstance(sample, GenerationSample)
        and sample.phase in spec.target_phases
    ]
    for index in target_indices:
        sample = trace.samples[index]
        for field_name in sorted(manner_field_names):
            value = getattr(sample, field_name)
            if value is None:
                manner.append(f"missing_field:{field_name}:index={index}")
            elif not _finite_value(value):
                manner.append(f"non_finite:{field_name}:index={index}")

    if spec.safety_max_force_n is not None:
        for index, sample in enumerate(trace.samples):
            if not isinstance(sample, GenerationSample):
                continue
            value = sample.contact_force_n
            if value is None:
                safety.append(f"missing_field:contact_force_n:index={index}")
            elif not _finite_value(value):
                safety.append(f"non_finite:contact_force_n:index={index}")

    return _Structure(
        common=tuple(dict.fromkeys(common)),
        manner=tuple(dict.fromkeys(manner)),
        safety=tuple(dict.fromkeys(safety)),
    )


def _evaluate_manner(
    trace: GenerationTrace, spec: ConstraintSpec
) -> tuple[Decision, tuple[str, ...]]:
    if not spec.has_manner_rule():
        return Decision.UNKNOWN, ("no_manner_constraint",)

    target = [
        sample
        for sample in trace.samples
        if isinstance(sample, GenerationSample) and sample.phase in spec.target_phases
    ]
    if not target:
        return Decision.UNKNOWN, ("missing_target_phase",)

    reasons: list[str] = []

    if spec.max_velocity is not None or spec.min_velocity is not None:
        values, issue = _values(target, "velocity", norm=True)
        if issue is not None:
            reasons.append(issue)
        elif values is not None:
            if spec.max_velocity is not None and any(
                value > _upper_limit(spec.max_velocity, spec.tolerance) for value in values
            ):
                reasons.append("velocity_upper_bound_violated")
            if spec.min_velocity is not None and any(
                value < _lower_limit(spec.min_velocity, spec.tolerance) for value in values
            ):
                reasons.append("velocity_lower_bound_violated")

    if spec.max_kp is not None or spec.min_kp is not None:
        values, issue = _values(target, "kp", norm=False)
        if issue is not None:
            reasons.append(issue)
        elif values is not None:
            if spec.max_kp is not None and any(
                value > _upper_limit(spec.max_kp, spec.tolerance) for value in values
            ):
                reasons.append("kp_upper_bound_violated")
            if spec.min_kp is not None and any(
                value < _lower_limit(spec.min_kp, spec.tolerance) for value in values
            ):
                reasons.append("kp_lower_bound_violated")

    if spec.max_gripper_fraction is not None or spec.min_gripper_fraction is not None:
        values, issue = _values(target, "gripper_fraction", norm=False)
        if issue is not None:
            reasons.append(issue)
        elif values is not None:
            if spec.max_gripper_fraction is not None and any(
                value > _upper_limit(spec.max_gripper_fraction, spec.tolerance)
                for value in values
            ):
                reasons.append("gripper_upper_bound_violated")
            if spec.min_gripper_fraction is not None and any(
                value < _lower_limit(spec.min_gripper_fraction, spec.tolerance)
                for value in values
            ):
                reasons.append("gripper_lower_bound_violated")

    if spec.max_jerk is not None:
        jerk_values: list[float] = []
        for left, right in zip(target, target[1:]):
            if left.phase != right.phase or left.reset_id != right.reset_id:
                continue
            if left.acceleration is None or right.acceleration is None:
                reasons.append("missing_field:acceleration")
                continue
            if not _finite(left.t) or not _finite(right.t):
                reasons.append("invalid_jerk_time")
                continue
            dt = float(right.t) - float(left.t)
            if dt <= 0 or dt > spec.max_gap_s:
                reasons.append("invalid_jerk_interval")
                continue
            try:
                jerk_values.append(_norm(_subtract(right.acceleration, left.acceleration)) / dt)
            except (TypeError, ValueError, ZeroDivisionError):
                reasons.append("invalid_acceleration_shape")
        if not jerk_values and not any(
            reason.startswith("missing_field:acceleration")
            or reason == "invalid_jerk_interval"
            or reason == "invalid_acceleration_shape"
            for reason in reasons
        ):
            reasons.append("no_valid_jerk_interval")
        if any(
            value > _upper_limit(spec.max_jerk, spec.tolerance)
            for value in jerk_values
        ):
            reasons.append("jerk_upper_bound_violated")

    if any(
        reason.startswith("missing_field")
        or reason.startswith("invalid_")
        or reason.startswith("non_finite")
        for reason in reasons
    ):
        return Decision.UNKNOWN, tuple(dict.fromkeys(reasons))
    if reasons:
        return Decision.FAIL, tuple(dict.fromkeys(reasons))
    return Decision.PASS, ("manner_constraints_satisfied",)


def _evaluate_safety(
    trace: GenerationTrace, spec: ConstraintSpec
) -> tuple[Decision, tuple[str, ...]]:
    if spec.safety_max_force_n is None:
        return Decision.UNKNOWN, ("safety_limit_unset",)

    values: list[float] = []
    for sample in trace.samples:
        if not isinstance(sample, GenerationSample) or sample.contact_force_n is None:
            return Decision.UNKNOWN, ("missing_field:contact_force_n",)
        try:
            value = _norm(sample.contact_force_n)
        except (TypeError, ValueError):
            return Decision.UNKNOWN, ("invalid_field:contact_force_n",)
        if not isfinite(value) or value < 0:
            return Decision.UNKNOWN, ("invalid_contact_force",)
        values.append(value)

    if any(value > spec.safety_max_force_n for value in values):
        return Decision.FAIL, ("safety_force_limit_violated",)
    return Decision.PASS, ("safety_limit_satisfied",)


def _clipping_reasons(
    requested: ConstraintSpec, effective: ConstraintSpec
) -> tuple[str, ...]:
    """Identify only relaxations/removals; stricter execution is permitted."""

    reasons: list[str] = []
    for field_name in _MANNER_UPPER_FIELDS:
        requested_value = getattr(requested, field_name)
        effective_value = getattr(effective, field_name)
        if requested_value is not None and (
            effective_value is None or float(effective_value) > float(requested_value)
        ):
            reasons.append(f"manner_clip:{field_name}")
    for field_name in _MANNER_LOWER_FIELDS:
        requested_value = getattr(requested, field_name)
        effective_value = getattr(effective, field_name)
        if requested_value is not None and (
            effective_value is None or float(effective_value) < float(requested_value)
        ):
            reasons.append(f"manner_clip:{field_name}")

    if requested.has_manner_rule():
        if requested.target_phases and tuple(requested.target_phases) != tuple(
            effective.target_phases
        ):
            reasons.append("manner_clip:target_phases")
        if requested.stage_id != effective.stage_id:
            reasons.append("manner_clip:stage_id")

    if requested.safety_max_force_n is not None and (
        effective.safety_max_force_n is None
        or float(effective.safety_max_force_n) > float(requested.safety_max_force_n)
    ):
        reasons.append("safety_clip:safety_max_force_n")

    # Protocol relaxation can hide a gap or a short stage even when all
    # physical bounds are copied.  A changed tolerance also changes the
    # meaning of every manner bound.
    if requested.has_manner_rule() and effective.min_samples < requested.min_samples:
        reasons.append("manner_clip:min_samples")
    if requested.has_manner_rule() and effective.min_duration_s < requested.min_duration_s:
        reasons.append("manner_clip:min_duration_s")
    if effective.max_gap_s > requested.max_gap_s:
        if requested.has_manner_rule():
            reasons.append("manner_clip:max_gap_s")
        if requested.has_safety_rule():
            reasons.append("safety_clip:max_gap_s")
    if requested.has_manner_rule() and effective.tolerance > requested.tolerance:
        reasons.append("manner_clip:tolerance")

    return tuple(dict.fromkeys(reasons))


def _values(
    samples: Sequence[GenerationSample], field_name: str, *, norm: bool
) -> tuple[list[float] | None, str | None]:
    values: list[float] = []
    for sample in samples:
        raw = getattr(sample, field_name)
        if raw is None:
            return None, f"missing_field:{field_name}"
        try:
            value = _norm(raw) if norm else _scalar(raw)
        except (TypeError, ValueError):
            return None, f"invalid_field:{field_name}"
        if not isfinite(value):
            return None, f"non_finite:{field_name}"
        if field_name == "gripper_fraction" and not 0.0 <= value <= 1.0:
            return None, "invalid_gripper_fraction"
        if field_name == "kp" and value < 0:
            return None, "invalid_kp"
        values.append(value)
    return values, None


def _upper_limit(value: float, tolerance: float) -> float:
    return float(value) * (1.0 + float(tolerance))


def _lower_limit(value: float, tolerance: float) -> float:
    return float(value) * (1.0 - float(tolerance))


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_scalar(value: Any) -> bool:
    return _finite(value)


def _finite_value(value: Any) -> bool:
    try:
        _norm(value)
    except (TypeError, ValueError):
        return False
    return True


def _as_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("numeric value must not be a string, bool, or mapping")
    try:
        iterator = iter(value)
    except TypeError:
        converted = float(value)
        if not isfinite(converted):
            raise ValueError("non-finite numeric value")
        return (converted,)
    converted_values = tuple(float(item) for item in iterator)
    if not converted_values or not all(isfinite(item) for item in converted_values):
        raise ValueError("empty or non-finite numeric vector")
    return converted_values


def _norm(value: Any) -> float:
    vector = _as_vector(value)
    return sqrt(sum(item * item for item in vector))


def _scalar(value: Any) -> float:
    vector = _as_vector(value)
    if len(vector) != 1:
        raise ValueError("expected a scalar")
    return vector[0]


def _subtract(right: Any, left: Any) -> float | tuple[float, ...]:
    right_vector = _as_vector(right)
    left_vector = _as_vector(left)
    if len(right_vector) != len(left_vector):
        raise ValueError("numeric vector shape mismatch")
    delta = tuple(right_item - left_item for right_item, left_item in zip(right_vector, left_vector))
    return delta[0] if len(delta) == 1 else delta


__all__ = [
    "ConstraintSpec",
    "Decision",
    "GenerationCase",
    "GenerationConstraintSpec",
    "GenerationResult",
    "GenerationSample",
    "GenerationTrace",
    "Trace",
    "TraceSample",
    "evaluate_generation",
    "validate_generation",
]
