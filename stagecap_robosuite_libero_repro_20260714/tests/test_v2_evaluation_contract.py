"""Contract tests for the independent MannerCAP v2 final-audit kernel.

These cases are intentionally small hand-built traces.  They exercise the
acceptance semantics that must be frozen before a large 4090 run; they do not
stand in for physics calibration or simulator integration tests.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from capx.evaluation_contract_v2 import (
    ConstraintSpec,
    Decision,
    EvaluationCase,
    Trace,
    TraceSample,
    evaluate,
)


UNITS = {
    "time": "s",
    "velocity": "rad/s",
    "acceleration": "rad/s^2",
    "kp": "gain",
    "gripper_fraction": "fraction",
    "contact_force": "N",
}


def _sample(
    t: float,
    *,
    phase: str = "contact",
    velocity: float = 0.2,
    acceleration: float = 0.0,
    kp: float = 500.0,
    gripper_fraction: float = 0.5,
    contact_force_n: float = 1.0,
    sensor_valid: bool = True,
    reset_id: int = 0,
) -> TraceSample:
    return TraceSample(
        t=t,
        phase=phase,
        velocity=velocity,
        acceleration=acceleration,
        kp=kp,
        gripper_fraction=gripper_fraction,
        contact_force_n=contact_force_n,
        sensor_valid=sensor_valid,
        reset_id=reset_id,
    )


def _trace(samples: list[TraceSample]) -> Trace:
    return Trace.from_samples(samples, units=UNITS)


def _spec(**changes) -> ConstraintSpec:
    base = ConstraintSpec(
        stage_id="lift_contact",
        target_phases=("contact",),
        max_velocity=0.5,
        min_velocity=0.1,
        max_kp=1000.0,
        min_kp=100.0,
        max_gripper_fraction=0.8,
        min_gripper_fraction=0.1,
        safety_max_force_n=5.0,
        min_samples=3,
        min_duration_s=0.1,
        max_gap_s=0.1,
    )
    return replace(base, **changes)


def _clean_case(*, task: Decision | None = Decision.PASS, **changes) -> EvaluationCase:
    samples = [_sample(0.0), _sample(0.05), _sample(0.1)]
    return EvaluationCase(
        trace=_trace(samples),
        constraints=_spec(**changes),
        task=task,
    )


def test_clean_trace_is_publishable_only_when_all_three_dimensions_pass():
    result = evaluate(_clean_case())

    assert result.task is Decision.PASS
    assert result.manner is Decision.PASS
    assert result.safety is Decision.PASS
    assert result.overall is Decision.PASS
    assert result.publishable


def test_clipped_requested_bound_cannot_disappear_into_a_pass():
    requested = _spec(max_velocity=0.2)
    effective = replace(requested, max_velocity=None)
    case = _clean_case()
    case = replace(
        case,
        constraints=effective,
        requested_constraints=requested,
    )

    result = evaluate(case)

    assert result.manner is Decision.FAIL
    assert result.overall is Decision.FAIL
    assert "constraint_relaxed_or_clipped:max_velocity" in result.reasons


def test_no_log_is_unknown_and_never_an_automatic_pass():
    result = evaluate(
        EvaluationCase(trace=None, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.UNKNOWN
    assert result.overall is Decision.UNKNOWN
    assert "missing_trace" in result.reasons


def test_missing_target_stage_is_unknown():
    trace = _trace([_sample(0.0, phase="approach"), _sample(0.05, phase="approach")])

    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(min_duration_s=0.0), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert "missing_target_phase" in result.reasons


def test_sensor_failure_is_unknown_instead_of_zero_filled_pass():
    trace = _trace(
        [
            _sample(0.0, sensor_valid=False),
            _sample(0.05),
            _sample(0.1),
        ]
    )

    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.UNKNOWN
    assert "sensor_invalid:index=0" in result.reasons


def test_one_instant_high_speed_does_not_satisfy_a_stage_lower_bound():
    trace = _trace(
        [
            _sample(0.0, velocity=1.0),
            _sample(0.05, velocity=0.05),
            _sample(0.1, velocity=0.05),
        ]
    )
    result = evaluate(
        EvaluationCase(
            trace=trace,
            constraints=_spec(
                max_velocity=None,
                min_velocity=0.5,
                max_kp=None,
                min_kp=None,
                max_gripper_fraction=None,
                min_gripper_fraction=None,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.FAIL
    assert result.overall is Decision.FAIL
    assert "velocity_lower_bound_violated" in result.reasons


def test_sustained_below_lower_bound_is_a_known_failure():
    trace = _trace(
        [_sample(0.0, velocity=0.05), _sample(0.05, velocity=0.05), _sample(0.1, velocity=0.05)]
    )
    result = evaluate(
        EvaluationCase(
            trace=trace,
            constraints=_spec(
                max_velocity=None,
                min_velocity=0.5,
                max_kp=None,
                min_kp=None,
                max_gripper_fraction=None,
                min_gripper_fraction=None,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.FAIL
    assert "velocity_lower_bound_violated" in result.reasons


def test_safety_violation_fails_joint_result_even_when_manner_passes():
    trace = _trace(
        [_sample(0.0, contact_force_n=6.0), _sample(0.05), _sample(0.1)]
    )
    result = evaluate(
        EvaluationCase(
            trace=trace,
            constraints=_spec(
                max_velocity=None,
                min_velocity=None,
                max_kp=None,
                min_kp=None,
                max_gripper_fraction=None,
                min_gripper_fraction=None,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.FAIL
    assert result.overall is Decision.FAIL
    assert "safety_force_limit_violated" in result.reasons


def test_duplicate_timestamps_are_unknown():
    trace = _trace([_sample(0.0), _sample(0.0), _sample(0.1)])
    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.UNKNOWN
    assert "non_monotonic_time:index=1" in result.reasons


def test_sampling_gap_is_unknown_and_cannot_be_hidden_by_fixed_dt():
    trace = _trace([_sample(0.0), _sample(0.05), _sample(0.5)])
    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.UNKNOWN
    assert any(reason.startswith("sampling_gap:index=2") for reason in result.reasons)


def test_wrong_phase_effect_does_not_contaminate_target_phase_decision():
    trace = _trace(
        [
            _sample(0.0, phase="approach", velocity=5.0),
            _sample(0.05, phase="contact", velocity=0.2),
            _sample(0.1, phase="contact", velocity=0.2),
            _sample(0.15, phase="contact", velocity=0.2),
        ]
    )
    result = evaluate(
        EvaluationCase(
            trace=trace,
            constraints=_spec(
                max_velocity=0.5,
                min_velocity=None,
                max_kp=None,
                min_kp=None,
                max_gripper_fraction=None,
                min_gripper_fraction=None,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.PASS
    assert result.safety is Decision.PASS
    assert result.overall is Decision.PASS


def test_unit_mismatch_is_unknown_instead_of_treating_meters_as_radians():
    trace = Trace.from_samples(
        [_sample(0.0), _sample(0.05), _sample(0.1)],
        units={**UNITS, "velocity": "m/s"},
    )
    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.UNKNOWN
    assert any(reason.startswith("unit_mismatch:velocity") for reason in result.reasons)


def test_task_failure_does_not_get_relabelled_as_manner_infeasible():
    result = evaluate(_clean_case(task=Decision.FAIL))

    assert result.task is Decision.FAIL
    assert result.manner is Decision.PASS
    assert result.safety is Decision.PASS
    assert result.overall is Decision.FAIL
    assert "task_failed" in result.reasons
    assert all("infeasible" not in reason for reason in result.reasons)


def test_unconstrained_trace_is_unknown_even_when_telemetry_looks_good():
    result = evaluate(
        EvaluationCase(
            trace=_trace([_sample(0.0), _sample(0.05), _sample(0.1)]),
            constraints=ConstraintSpec(
                stage_id="lift_contact",
                target_phases=("contact",),
                safety_max_force_n=5.0,
                min_samples=3,
                min_duration_s=0.1,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.UNKNOWN
    assert result.safety is Decision.PASS
    assert result.overall is Decision.UNKNOWN
    assert "no_manner_constraint" in result.reasons


def test_missing_acceleration_is_unknown_when_jerk_is_required():
    trace = _trace(
        [_sample(0.0, acceleration=None), _sample(0.05), _sample(0.1)]
    )
    result = evaluate(
        EvaluationCase(
            trace=trace,
            constraints=_spec(
                max_velocity=None,
                min_velocity=None,
                max_kp=None,
                min_kp=None,
                max_gripper_fraction=None,
                min_gripper_fraction=None,
                max_jerk=1.0,
            ),
            task=Decision.PASS,
        )
    )

    assert result.manner is Decision.UNKNOWN


def test_missing_velocity_is_unknown_not_default_value():
    trace = _trace([_sample(0.0, velocity=None), _sample(0.05), _sample(0.1)])
    result = evaluate(
        EvaluationCase(trace=trace, constraints=_spec(), task=Decision.PASS)
    )

    assert result.manner is Decision.UNKNOWN
