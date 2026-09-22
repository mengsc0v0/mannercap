"""Executable contract tests for the independent generation validator.

These fixtures are deliberately small synthetic records.  They verify the
decision and evidence semantics; they are not simulator runs or physical
calibration data.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

from synthesis_v2.generation_validator import (
    ConstraintSpec,
    Decision,
    GenerationCase,
    GenerationSample,
    GenerationTrace,
    validate_generation,
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
    velocity: float | tuple[float, ...] | None = 0.2,
    acceleration: float | tuple[float, ...] | None = 0.0,
    kp: float | None = 100.0,
    gripper_fraction: float | None = 0.5,
    contact_force_n: float | tuple[float, ...] | None = 1.0,
    sensor_valid: bool | None = True,
    reset_id: int | None = 0,
    dt_s: float | None = None,
) -> GenerationSample:
    return GenerationSample(
        t=t,
        phase=phase,
        velocity=velocity,
        acceleration=acceleration,
        kp=kp,
        gripper_fraction=gripper_fraction,
        contact_force_n=contact_force_n,
        sensor_valid=sensor_valid,
        reset_id=reset_id,
        dt_s=dt_s,
    )


def _trace(
    samples: tuple[GenerationSample, ...] | list[GenerationSample] | None = None,
    *,
    units: dict[str, str] | None = None,
) -> GenerationTrace:
    if samples is None:
        samples = [_sample(0.0), _sample(0.05), _sample(0.1)]
    return GenerationTrace.from_samples(samples, units=UNITS if units is None else units)


def _spec(**overrides: object) -> ConstraintSpec:
    base = ConstraintSpec(
        stage_id="lift.contact",
        target_phases=("contact",),
        max_velocity=0.5,
        safety_max_force_n=5.0,
        min_samples=3,
        min_duration_s=0.1,
        max_gap_s=0.1,
    )
    return replace(base, **overrides)


def _case(
    *,
    trace: GenerationTrace | None = None,
    requested: ConstraintSpec | None = None,
    effective: ConstraintSpec | None = None,
    task: Decision | bool | str | None = True,
) -> GenerationCase:
    return GenerationCase(
        trace=_trace() if trace is None else trace,
        requested_constraints=_spec() if requested is None else requested,
        effective_constraints=effective,
        task=task,
    )


class GenerationValidatorContractTests(unittest.TestCase):
    def test_valid_three_dimension_trace_passes(self) -> None:
        result = validate_generation(_case())
        self.assertEqual(
            (result.task, result.manner, result.safety, result.overall),
            (Decision.PASS, Decision.PASS, Decision.PASS, Decision.PASS),
        )
        self.assertTrue(result.publishable)

    def test_clipped_requested_bound_cannot_disappear_into_a_pass(self) -> None:
        requested = _spec(max_velocity=0.5)
        effective = replace(requested, max_velocity=None)
        result = validate_generation(_case(requested=requested, effective=effective))
        self.assertEqual(result.manner, Decision.FAIL)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertEqual(result.overall, Decision.FAIL)
        self.assertIn("manner_clip:max_velocity", result.reasons)

    def test_no_log_is_unknown_and_never_an_automatic_pass(self) -> None:
        result = validate_generation(
            GenerationCase(trace=None, requested_constraints=_spec(), task=True)
        )
        self.assertEqual(result.task, Decision.PASS)
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertEqual(result.overall, Decision.UNKNOWN)
        self.assertFalse(result.publishable)

    def test_missing_target_stage_is_unknown(self) -> None:
        trace = _trace([_sample(0.0, phase="approach"), _sample(0.05, phase="approach"), _sample(0.1, phase="approach")])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertEqual(result.overall, Decision.UNKNOWN)
        self.assertIn("missing_target_phase", result.reasons)

    def test_sensor_failure_is_unknown_instead_of_zero_filled_pass(self) -> None:
        samples = [_sample(0.0), _sample(0.05, sensor_valid=False), _sample(0.1)]
        result = validate_generation(_case(trace=_trace(samples)))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertEqual(result.overall, Decision.UNKNOWN)
        self.assertIn("sensor_invalid:index=1", result.reasons)

    def test_one_instant_high_speed_does_not_satisfy_a_stage_lower_bound(self) -> None:
        requested = _spec(max_velocity=None, min_velocity=0.5)
        trace = _trace([_sample(0.0, velocity=0.6), _sample(0.05, velocity=0.2), _sample(0.1, velocity=0.2)])
        result = validate_generation(_case(trace=trace, requested=requested))
        self.assertEqual(result.manner, Decision.FAIL)
        self.assertIn("velocity_lower_bound_violated", result.reasons)

    def test_sustained_below_lower_bound_is_a_known_failure(self) -> None:
        requested = _spec(max_velocity=None, min_velocity=0.5)
        trace = _trace([_sample(0.0, velocity=0.2), _sample(0.05, velocity=0.2), _sample(0.1, velocity=0.2)])
        result = validate_generation(_case(trace=trace, requested=requested))
        self.assertEqual(result.manner, Decision.FAIL)
        self.assertEqual(result.safety, Decision.PASS)

    def test_safety_violation_fails_joint_result_even_when_manner_passes(self) -> None:
        trace = _trace([_sample(0.0), _sample(0.05, contact_force_n=6.0), _sample(0.1)])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.PASS)
        self.assertEqual(result.safety, Decision.FAIL)
        self.assertEqual(result.overall, Decision.FAIL)
        self.assertIn("safety_force_limit_violated", result.reasons)

    def test_duplicate_timestamps_are_unknown(self) -> None:
        trace = _trace([_sample(0.0), _sample(0.05), _sample(0.05)])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertIn("non_monotonic_time:index=2", result.reasons)

    def test_sampling_gap_is_unknown_and_cannot_be_hidden_by_fixed_dt(self) -> None:
        trace = _trace([_sample(0.0), _sample(0.05), _sample(0.5)])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertTrue(any(reason.startswith("sampling_gap:index=2") for reason in result.reasons))

    def test_unit_mismatch_is_unknown_instead_of_treating_meters_as_radians(self) -> None:
        units = dict(UNITS)
        units["velocity"] = "m/s"
        result = validate_generation(_case(trace=_trace(units=units)))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertEqual(result.overall, Decision.UNKNOWN)
        self.assertTrue(any(reason.startswith("unit_mismatch:velocity") for reason in result.reasons))

    def test_wrong_phase_effect_does_not_contaminate_target_phase_decision(self) -> None:
        trace = _trace(
            [
                _sample(0.0, phase="approach", velocity=99.0),
                _sample(0.05, phase="contact", velocity=0.2),
                _sample(0.1, phase="contact", velocity=0.2),
                _sample(0.15, phase="contact", velocity=0.2),
            ]
        )
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.PASS)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertEqual(result.overall, Decision.PASS)

    def test_task_failure_does_not_get_relabelled_as_manner_infeasible(self) -> None:
        result = validate_generation(_case(task=False))
        self.assertEqual(result.task, Decision.FAIL)
        self.assertEqual(result.manner, Decision.PASS)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertEqual(result.overall, Decision.FAIL)
        self.assertNotIn("manner_infeasible", " ".join(result.reasons))

    def test_unconstrained_trace_is_unknown_even_when_telemetry_looks_good(self) -> None:
        requested = ConstraintSpec(
            stage_id="lift.contact",
            target_phases=("contact",),
            min_samples=3,
            min_duration_s=0.1,
            max_gap_s=0.1,
        )
        result = validate_generation(_case(requested=requested))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertFalse(result.publishable)
        self.assertIn("requested:no_constraint", result.reasons)

    def test_missing_velocity_is_unknown_not_default_value(self) -> None:
        trace = _trace([_sample(0.0, velocity=None), _sample(0.05, velocity=None), _sample(0.1, velocity=None)])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertTrue(any(reason.startswith("missing_field:velocity") for reason in result.reasons))

    def test_missing_acceleration_is_unknown_when_jerk_is_required(self) -> None:
        requested = _spec(max_velocity=None, max_jerk=1.0)
        trace = _trace([_sample(0.0, acceleration=None), _sample(0.05, acceleration=None), _sample(0.1, acceleration=None)])
        result = validate_generation(_case(trace=trace, requested=requested))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.PASS)
        self.assertIn("missing_field:acceleration:index=0", result.reasons)

    def test_interleaved_target_phase_interval_is_unknown(self) -> None:
        trace = _trace(
            [
                _sample(0.0, phase="contact"),
                _sample(0.05, phase="approach"),
                _sample(0.1, phase="contact"),
            ]
        )
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertIn("phase_interval_not_contiguous", result.reasons)

    def test_reset_boundary_is_unknown_even_with_increasing_time(self) -> None:
        trace = _trace([_sample(0.0), _sample(0.05, reset_id=1), _sample(0.1, reset_id=1)])
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertIn("reset_boundary:index=1", result.reasons)

    def test_measured_dt_must_match_timestamp_axis(self) -> None:
        trace = _trace(
            [
                _sample(0.0),
                _sample(0.05, dt_s=0.01),
                _sample(0.1, dt_s=0.05),
            ]
        )
        result = validate_generation(_case(trace=trace))
        self.assertEqual(result.manner, Decision.UNKNOWN)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertIn("dt_mismatch:index=1", result.reasons)

    def test_safety_unit_mismatch_is_scoped_to_safety(self) -> None:
        units = dict(UNITS)
        units["contact_force"] = "N*m"
        result = validate_generation(_case(trace=_trace(units=units)))
        self.assertEqual(result.manner, Decision.PASS)
        self.assertEqual(result.safety, Decision.UNKNOWN)
        self.assertEqual(result.overall, Decision.UNKNOWN)

    def test_original_request_is_not_mutated_by_validation(self) -> None:
        requested = _spec(max_velocity=0.5)
        effective = replace(requested, max_velocity=None)
        before = requested
        validate_generation(_case(requested=requested, effective=effective))
        self.assertEqual(requested, before)
        self.assertEqual(requested.max_velocity, 0.5)


if __name__ == "__main__":
    unittest.main()
