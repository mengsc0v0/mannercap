"""Independent Gate-1 evaluation contract for MannerCAP v2.

This module is deliberately separate from ``capx.envs.manner_validator``.
The latter is the generation-time validator used by the existing driver.  The
contract here is a small, conservative final-audit kernel that can be used to
exercise the acceptance semantics before the production recorder and the
threshold registry are frozen.

The important property is that missing or unverifiable evidence is never
treated as a pass.  Every dimension has exactly three states:

``pass``
    All required observations and the frozen rule are available and the rule
    is satisfied.
``fail``
    Required observations are valid and a frozen rule is violated.
``unknown``
    The decision cannot be made because evidence, units, time continuity,
    stage coverage, or the rule itself is missing.

The implementation intentionally uses only the Python standard library.  It
accepts scalar values or short numeric sequences for velocity and acceleration
so that the same checks can be used with a reduced fixture and with recorded
joint vectors.  It does not claim that the thresholds below are calibrated;
callers must provide thresholds from the independent calibration package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Iterable, Mapping, Sequence


class Decision(str, Enum):
    """The only admissible state for task, manner, safety, and joint results."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TraceSample:
    """One recorded sample.

    ``None`` means that the field was not recorded.  It is intentionally not
    converted to zero, a default gain, or an open gripper: doing so would turn
    missing evidence into a favourable measurement.
    """

    t: float
    phase: str
    velocity: float | Sequence[float] | None = None
    acceleration: float | Sequence[float] | None = None
    kp: float | None = None
    gripper_fraction: float | None = None
    contact_force_n: float | None = None
    sensor_valid: bool = True
    reset_id: int = 0


@dataclass(frozen=True)
class Trace:
    """Immutable evidence buffer and its unit declarations.

    The unit keys are fixed by the contract: ``time``, ``velocity``,
    ``acceleration``, ``kp``, ``gripper_fraction``, and ``contact_force``.
    A final audit must reject a trace whose declared unit is absent or does not
    match the contract for a field it uses.
    """

    samples: tuple[TraceSample, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[TraceSample],
        *,
        units: Mapping[str, str] | None = None,
    ) -> "Trace":
        return cls(tuple(samples), dict(units or {}))


@dataclass(frozen=True)
class ConstraintSpec:
    """A frozen, stage-scoped physical contract.

    ``target_phases`` is required for every constrained stage.  Hard safety
    is represented separately by ``safety_max_force_n`` and is evaluated on
    every recorded sample.  Manner bounds are evaluated only inside the
    target phases.  This keeps a velocity requirement for an approach stage
    from silently becoming a requirement during release or idle motion.
    """

    stage_id: str = "unknown_stage"
    target_phases: tuple[str, ...] = ()

    max_velocity: float | None = None
    min_velocity: float | None = None
    max_kp: float | None = None
    min_kp: float | None = None
    max_gripper_fraction: float | None = None
    min_gripper_fraction: float | None = None
    max_jerk: float | None = None

    # Hard safety cap.  It has no soft tolerance by design.
    safety_max_force_n: float | None = None

    # These are protocol parameters, not fitted from the evaluated result.
    tolerance: float = 0.0
    min_samples: int = 2
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

    def has_any_rule(self) -> bool:
        return self.has_manner_rule() or self.safety_max_force_n is not None


@dataclass(frozen=True)
class EvaluationCase:
    """Inputs to the final-audit kernel.

    ``requested_constraints`` is the immutable request before execution.  If
    an execution-side clipping step removes or relaxes a requested bound, the
    manner decision is ``fail`` and the original request is retained in the
    reason.  A clipped bound is not allowed to disappear and become a pass.
    """

    trace: Trace | None
    constraints: ConstraintSpec
    task: Decision | None = None
    requested_constraints: ConstraintSpec | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Per-dimension decisions and the conservative joint decision."""

    task: Decision
    manner: Decision
    safety: Decision
    overall: Decision
    reasons: tuple[str, ...] = ()

    @property
    def publishable(self) -> bool:
        """Whether this episode may be counted as a final-audit success."""

        return self.overall is Decision.PASS


_UNIT_CONTRACT = {
    "time": "s",
    "velocity": "rad/s",
    "acceleration": "rad/s^2",
    "kp": "gain",
    "gripper_fraction": "fraction",
    "contact_force": "N",
}


def evaluate(case: EvaluationCase) -> EvaluationResult:
    """Evaluate one episode with strict pass/fail/unknown semantics.

    The function does not mutate the input and never infers values for absent
    fields.  Structural evidence failures make both physical dimensions
    unknown because an invalid time axis or failed sensor cannot support a
    safety or manner claim.  Task status remains independently reportable.
    """

    reasons: list[str] = []
    task = _coerce_decision(case.task)

    if case.trace is None or not case.trace.samples:
        reasons.append("missing_trace")
        manner = Decision.UNKNOWN
        safety = Decision.UNKNOWN
    else:
        structural = _structural_reasons(case.trace, case.constraints)
        if structural:
            reasons.extend(structural)
            manner = Decision.UNKNOWN
            safety = Decision.UNKNOWN
        else:
            clipping = _clipping_reasons(
                case.requested_constraints, case.constraints
            )
            if clipping:
                reasons.extend(clipping)
                manner = Decision.FAIL
            else:
                manner, manner_reasons = _evaluate_manner(
                    case.trace, case.constraints
                )
                reasons.extend(manner_reasons)
            safety, safety_reasons = _evaluate_safety(case.trace, case.constraints)
            reasons.extend(safety_reasons)

    if task is Decision.UNKNOWN:
        reasons.append("task_status_unknown")
    elif task is Decision.FAIL:
        reasons.append("task_failed")

    overall = _joint_decision(task, manner, safety)
    return EvaluationResult(
        task=task,
        manner=manner,
        safety=safety,
        overall=overall,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _coerce_decision(value: Decision | bool | None) -> Decision:
    if value is None:
        return Decision.UNKNOWN
    if isinstance(value, Decision):
        return value
    if isinstance(value, bool):
        return Decision.PASS if value else Decision.FAIL
    raise TypeError(f"unsupported decision value: {value!r}")


def _joint_decision(*decisions: Decision) -> Decision:
    # A known failure must remain a failure even if another dimension is
    # unknown.  All dimensions must pass before an episode is publishable.
    if any(decision is Decision.FAIL for decision in decisions):
        return Decision.FAIL
    if all(decision is Decision.PASS for decision in decisions):
        return Decision.PASS
    return Decision.UNKNOWN


def _structural_reasons(
    trace: Trace, spec: ConstraintSpec
) -> list[str]:
    reasons: list[str] = []

    if not spec.has_any_rule():
        reasons.append("no_frozen_constraint")

    if not spec.stage_id or not spec.target_phases:
        reasons.append("missing_stage_scope")

    if spec.min_samples < 1 or spec.min_duration_s < 0 or spec.max_gap_s <= 0:
        reasons.append("invalid_protocol_parameters")

    required_units = {"time"}
    if spec.max_velocity is not None or spec.min_velocity is not None:
        required_units.add("velocity")
    if spec.max_jerk is not None:
        required_units.update(("velocity", "acceleration"))
    if spec.max_kp is not None or spec.min_kp is not None:
        required_units.add("kp")
    if (
        spec.max_gripper_fraction is not None
        or spec.min_gripper_fraction is not None
    ):
        required_units.add("gripper_fraction")
    if spec.safety_max_force_n is not None:
        required_units.add("contact_force")

    for field_name in sorted(required_units):
        expected = _UNIT_CONTRACT[field_name]
        actual = trace.units.get(field_name)
        if actual != expected:
            reasons.append(f"unit_mismatch:{field_name}:{actual!r}!={expected!r}")

    previous: TraceSample | None = None
    for index, sample in enumerate(trace.samples):
        if not _finite(sample.t):
            reasons.append(f"invalid_time:index={index}")
        if not sample.phase:
            reasons.append(f"missing_phase:index={index}")
        if not sample.sensor_valid:
            reasons.append(f"sensor_invalid:index={index}")
        if previous is not None:
            if sample.reset_id != previous.reset_id:
                reasons.append(f"reset_boundary:index={index}")
            else:
                dt = sample.t - previous.t
                if not _finite(dt) or dt <= 0:
                    reasons.append(f"non_monotonic_time:index={index}")
                elif dt > spec.max_gap_s:
                    reasons.append(f"sampling_gap:index={index}:dt={dt:g}")
        previous = sample

    target = [s for s in trace.samples if s.phase in spec.target_phases]
    if not target:
        reasons.append("missing_target_phase")
    elif len(target) < spec.min_samples:
        reasons.append(
            f"insufficient_target_samples:{len(target)}<{spec.min_samples}"
        )
    else:
        duration = target[-1].t - target[0].t
        # Decimal-looking simulator timestamps (for example 0.15 - 0.05)
        # can be one ulp below the protocol value.  This epsilon only avoids
        # rejecting an exactly-on-boundary duration; it does not waive a real
        # shortfall.
        if duration + 1e-9 < spec.min_duration_s:
            reasons.append(
                f"insufficient_target_duration:{duration:g}<{spec.min_duration_s:g}"
            )

    # Any non-finite numeric field that a rule could consume is unknown, not a
    # violation.  The field-specific checks below also catch None values.
    for index, sample in enumerate(trace.samples):
        for name, value in (
            ("velocity", sample.velocity),
            ("acceleration", sample.acceleration),
            ("kp", sample.kp),
            ("gripper_fraction", sample.gripper_fraction),
            ("contact_force", sample.contact_force_n),
        ):
            if value is not None and not _finite_value(value):
                reasons.append(f"non_finite:{name}:index={index}")

    return list(dict.fromkeys(reasons))


def _clipping_reasons(
    requested: ConstraintSpec | None, effective: ConstraintSpec
) -> list[str]:
    if requested is None:
        return []
    reasons: list[str] = []
    upper_fields = (
        "max_velocity",
        "max_kp",
        "max_gripper_fraction",
        "max_jerk",
        "safety_max_force_n",
    )
    lower_fields = (
        "min_velocity",
        "min_kp",
        "min_gripper_fraction",
    )
    for field_name in upper_fields:
        requested_value = getattr(requested, field_name)
        effective_value = getattr(effective, field_name)
        if requested_value is not None and (
            effective_value is None or effective_value > requested_value
        ):
            reasons.append(f"constraint_relaxed_or_clipped:{field_name}")
    for field_name in lower_fields:
        requested_value = getattr(requested, field_name)
        effective_value = getattr(effective, field_name)
        if requested_value is not None and (
            effective_value is None or effective_value < requested_value
        ):
            reasons.append(f"constraint_relaxed_or_clipped:{field_name}")
    if requested.target_phases and tuple(requested.target_phases) != tuple(
        effective.target_phases
    ):
        reasons.append("constraint_scope_changed")
    return reasons


def _evaluate_manner(
    trace: Trace, spec: ConstraintSpec
) -> tuple[Decision, list[str]]:
    if not spec.has_manner_rule():
        return Decision.UNKNOWN, ["no_manner_constraint"]

    target = [s for s in trace.samples if s.phase in spec.target_phases]
    reasons: list[str] = []
    # ``_structural_reasons`` normally catches this; keeping it here makes the
    # helper safe when called independently in a debugger.
    if not target:
        return Decision.UNKNOWN, ["missing_target_phase"]

    def values(name: str) -> list[float] | None:
        raw = [getattr(sample, name) for sample in target]
        if any(value is None for value in raw):
            reasons.append(f"missing_field:{name}")
            return None
        try:
            converted = [_finite_scalar(value, name) for value in raw]
        except (TypeError, ValueError):
            reasons.append(f"invalid_field:{name}")
            return None
        return converted

    if spec.max_velocity is not None or spec.min_velocity is not None:
        raw = [sample.velocity for sample in target]
        if any(value is None for value in raw):
            reasons.append("missing_field:velocity")
        else:
            velocities = [_norm(value) for value in raw]
            if spec.max_velocity is not None:
                limit = spec.max_velocity * (1.0 + spec.tolerance)
                if any(value > limit for value in velocities):
                    reasons.append("velocity_upper_bound_violated")
            if spec.min_velocity is not None:
                limit = spec.min_velocity * (1.0 - spec.tolerance)
                if any(value < limit for value in velocities):
                    reasons.append("velocity_lower_bound_violated")

    if spec.max_kp is not None or spec.min_kp is not None:
        kps = values("kp")
        if kps is not None:
            if spec.max_kp is not None:
                limit = spec.max_kp * (1.0 + spec.tolerance)
                if any(value > limit for value in kps):
                    reasons.append("kp_upper_bound_violated")
            if spec.min_kp is not None:
                limit = spec.min_kp * (1.0 - spec.tolerance)
                if any(value < limit for value in kps):
                    reasons.append("kp_lower_bound_violated")

    if (
        spec.max_gripper_fraction is not None
        or spec.min_gripper_fraction is not None
    ):
        fractions = values("gripper_fraction")
        if fractions is not None:
            if spec.max_gripper_fraction is not None:
                limit = spec.max_gripper_fraction * (1.0 + spec.tolerance)
                if any(value > limit for value in fractions):
                    reasons.append("gripper_upper_bound_violated")
            if spec.min_gripper_fraction is not None:
                limit = spec.min_gripper_fraction * (1.0 - spec.tolerance)
                if any(value < limit for value in fractions):
                    reasons.append("gripper_lower_bound_violated")

    if spec.max_jerk is not None:
        jerk_values: list[float] = []
        for left, right in zip(target, target[1:]):
            # Do not differentiate across a phase boundary.  It is a control
            # discontinuity, not a measured in-phase jerk sample.
            if left.phase != right.phase or left.reset_id != right.reset_id:
                continue
            if left.acceleration is None or right.acceleration is None:
                reasons.append("missing_field:acceleration")
                continue
            dt = right.t - left.t
            if dt <= 0 or dt > spec.max_gap_s:
                reasons.append("invalid_jerk_interval")
                continue
            delta = _sub(right.acceleration, left.acceleration)
            jerk_values.append(_norm(delta) / dt)
        if not jerk_values:
            reasons.append("no_valid_jerk_interval")
        elif any(
            value > spec.max_jerk * (1.0 + spec.tolerance)
            for value in jerk_values
        ):
            reasons.append("jerk_upper_bound_violated")

    if any(reason.startswith("missing_") or reason.startswith("invalid_") for reason in reasons):
        return Decision.UNKNOWN, reasons
    if reasons:
        return Decision.FAIL, reasons
    return Decision.PASS, ["manner_constraints_satisfied"]


def _evaluate_safety(
    trace: Trace, spec: ConstraintSpec
) -> tuple[Decision, list[str]]:
    if spec.safety_max_force_n is None:
        return Decision.UNKNOWN, ["safety_limit_unset"]
    forces = [sample.contact_force_n for sample in trace.samples]
    if any(value is None for value in forces):
        return Decision.UNKNOWN, ["missing_field:contact_force_n"]
    values = [_finite_scalar(value, "contact_force_n") for value in forces]
    if any(value < 0 for value in values):
        return Decision.UNKNOWN, ["invalid_negative_contact_force"]
    if any(value > spec.safety_max_force_n for value in values):
        return Decision.FAIL, ["safety_force_limit_violated"]
    return Decision.PASS, ["safety_limit_satisfied"]


def _finite(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_value(value: Any) -> bool:
    if _is_vector(value):
        return all(_finite(item) for item in value)
    return _finite(value)


def _finite_scalar(value: Any, field_name: str) -> float:
    if _is_vector(value):
        if len(value) != 1:
            raise ValueError(f"{field_name} must be scalar here")
        value = value[0]
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} is not finite")
    return result


def _norm(value: float | Sequence[float]) -> float:
    if _is_vector(value):
        return sqrt(sum(float(item) ** 2 for item in value))
    return abs(float(value))


def _sub(
    right: float | Sequence[float], left: float | Sequence[float]
) -> float | tuple[float, ...]:
    if _is_vector(right):
        if not _is_vector(left) or len(right) != len(left):
            raise ValueError("acceleration vector shape mismatch")
        return tuple(float(r) - float(l) for r, l in zip(right, left))
    return float(right) - float(left)  # type: ignore[arg-type]


def _is_vector(value: Any) -> bool:
    """Return whether ``value`` is a numeric vector, including a NumPy array.

    The contract kernel itself has no NumPy dependency, but production
    telemetry commonly carries NumPy arrays.  Checking iterability instead of
    ``isinstance(value, Sequence)`` keeps that boundary dependency-free.
    Strings and mappings are deliberately treated as scalar-invalid values.
    """

    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


__all__ = [
    "ConstraintSpec",
    "Decision",
    "EvaluationCase",
    "EvaluationResult",
    "Trace",
    "TraceSample",
    "evaluate",
]
