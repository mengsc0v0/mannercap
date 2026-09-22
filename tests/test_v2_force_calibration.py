"""Tests for the pure Gate1 calibration/anomaly case audit."""

from __future__ import annotations

import json
import unittest

from synthesis_v2.contact_force_contract import Decision
from synthesis_v2.force_calibration import (
    CalibrationCase,
    CalibrationCaseType,
    CalibrationSample,
    REQUIRED_CALIBRATION_CASE_TYPES,
    audit_calibration_case,
    audit_calibration_suite,
)


class ForceCalibrationContractTests(unittest.TestCase):
    def sample(
        self,
        *,
        force=(0.0, 0.0, 0.0),
        torque=(0.0, 0.0, 0.0),
        timestamp=0.0,
        reset_id="reset-1",
        sensor_valid=True,
        object_id="cube_a",
        body_id=4,
        body_name="cube_a_body",
        geom_ids=(17,),
        geom_names=("cube_a_geom",),
        units_force="N",
        units_torque="N*m",
        frame="mujoco_subtree_com",
        geom_attribution="body_aggregate",
        known_load_n=None,
    ):
        return CalibrationSample(
            timestamp_s=timestamp,
            reset_id=reset_id,
            object_id=object_id,
            body_id=body_id,
            body_name=body_name,
            geom_ids=geom_ids,
            geom_names=geom_names,
            force_n=force,
            torque_nm=torque,
            sensor_valid=sensor_valid,
            units_force=units_force,
            units_torque=units_torque,
            frame=frame,
            geom_attribution=geom_attribution,
            known_load_n=known_load_n,
        )

    def case(self, case_type, samples, **kwargs):
        return CalibrationCase(
            case_id=kwargs.pop("case_id", f"case-{case_type}"),
            case_type=case_type,
            samples=tuple(samples),
            executed=kwargs.pop("executed", True),
            **kwargs,
        )

    def test_no_contact_is_valid_observation_but_not_contact_success(self):
        report = audit_calibration_case(
            self.case(CalibrationCaseType.NO_CONTACT, [self.sample(force=(0.0, 0.0, 0.0))])
        )
        self.assertEqual(report.decision, Decision.PASS)
        self.assertEqual(report.force_norms_n, (0.0,))

    def test_controlled_load_response_requires_increasing_measured_response(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.CONTROLLED_LOAD_RESPONSE,
                [
                    self.sample(force=(1.0, 0.0, 0.0), timestamp=0.0, known_load_n=1.0),
                    self.sample(force=(3.0, 0.0, 0.0), timestamp=0.1, known_load_n=3.0),
                ],
            )
        )
        self.assertEqual(report.decision, Decision.PASS)
        self.assertEqual(report.dt_s, (0.1,))

    def test_controlled_load_response_rejects_flat_sensor(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.CONTROLLED_LOAD_RESPONSE,
                [
                    self.sample(force=(2.0, 0.0, 0.0), timestamp=0.0, known_load_n=1.0),
                    self.sample(force=(2.0, 0.0, 0.0), timestamp=0.1, known_load_n=3.0),
                ],
            )
        )
        self.assertEqual(report.decision, Decision.FAIL)
        self.assertIn("constant_force_response", report.codes)

    def test_missing_sensor_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.MISSING_SENSOR,
                [self.sample(force=None, sensor_valid=None)],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("missing_sensor_validity", report.codes)

    def test_wrong_force_units_are_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.WRONG_FORCE_UNITS,
                [self.sample(units_force="kgf")],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("wrong_force_units", report.codes)

    def test_wrong_torque_units_are_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.WRONG_TORQUE_UNITS,
                [self.sample(units_torque="N")],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("wrong_torque_units", report.codes)

    def test_wrong_object_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.WRONG_OBJECT,
                [self.sample(object_id="other_object")],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("wrong_object", report.codes)

    def test_wrong_body_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.WRONG_BODY,
                [self.sample(body_id=5, body_name="other_body")],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("wrong_body", report.codes)

    def test_wrong_geom_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.WRONG_GEOM,
                [self.sample(geom_ids=(99,), geom_names=("other_geom",))],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("wrong_geom", report.codes)

    def test_mixed_wrench_is_rejected_without_a_norm(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.MIXED_WRENCH,
                [self.sample(force=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("mixed_wrench", report.codes)
        self.assertEqual(report.force_norms_n, ())

    def test_timestamp_gap_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.TIMESTAMP_GAP,
                [
                    self.sample(timestamp=0.0),
                    self.sample(timestamp=0.5),
                ],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("timestamp_gap", report.codes)

    def test_reset_gap_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.RESET_GAP,
                [
                    self.sample(timestamp=0.0, reset_id="reset-1"),
                    self.sample(timestamp=0.1, reset_id="reset-2"),
                ],
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("reset_gap", report.codes)

    def test_under_action_is_explicit_fail(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.UNDER_ACTION,
                [self.sample(force=(0.2, 0.0, 0.0))],
                min_force_n=1.0,
            )
        )
        self.assertEqual(report.decision, Decision.FAIL)
        self.assertIn("under_action", report.codes)

    def test_over_action_is_explicit_fail(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.OVER_ACTION,
                [self.sample(force=(2.0, 0.0, 0.0))],
                max_force_n=1.0,
            )
        )
        self.assertEqual(report.decision, Decision.FAIL)
        self.assertIn("over_action", report.codes)

    def test_unexecuted_nominal_case_is_unknown(self):
        report = audit_calibration_case(
            self.case(
                CalibrationCaseType.NO_CONTACT,
                [self.sample()],
                executed=False,
            )
        )
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertIn("case_not_executed", report.codes)
        self.assertEqual(report.physical_qualification, Decision.UNKNOWN)

    def test_suite_tracks_all_registered_case_types_without_claiming_calibration(self):
        cases = []
        for case_type in REQUIRED_CALIBRATION_CASE_TYPES:
            if case_type == CalibrationCaseType.CONTROLLED_LOAD_RESPONSE.value:
                samples = [
                    self.sample(force=(1.0, 0.0, 0.0), timestamp=0.0, known_load_n=1.0),
                    self.sample(force=(2.0, 0.0, 0.0), timestamp=0.1, known_load_n=2.0),
                ]
            elif case_type == CalibrationCaseType.MISSING_SENSOR.value:
                samples = [self.sample(force=None, sensor_valid=None)]
            elif case_type == CalibrationCaseType.WRONG_FORCE_UNITS.value:
                samples = [self.sample(units_force="kgf")]
            elif case_type == CalibrationCaseType.WRONG_TORQUE_UNITS.value:
                samples = [self.sample(units_torque="N")]
            elif case_type == CalibrationCaseType.WRONG_OBJECT.value:
                samples = [self.sample(object_id="other_object")]
            elif case_type == CalibrationCaseType.WRONG_BODY.value:
                samples = [self.sample(body_id=5, body_name="other_body")]
            elif case_type == CalibrationCaseType.WRONG_GEOM.value:
                samples = [self.sample(geom_ids=(99,), geom_names=("other_geom",))]
            elif case_type == CalibrationCaseType.MIXED_WRENCH.value:
                samples = [self.sample(force=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))]
            elif case_type == CalibrationCaseType.TIMESTAMP_GAP.value:
                samples = [self.sample(timestamp=0.0), self.sample(timestamp=0.5)]
            elif case_type == CalibrationCaseType.RESET_GAP.value:
                samples = [self.sample(timestamp=0.0, reset_id="a"), self.sample(timestamp=0.1, reset_id="b")]
            elif case_type == CalibrationCaseType.UNDER_ACTION.value:
                samples = [self.sample(force=(0.2, 0.0, 0.0))]
            elif case_type == CalibrationCaseType.OVER_ACTION.value:
                samples = [self.sample(force=(2.0, 0.0, 0.0))]
            else:
                samples = [self.sample(force=(0.0, 0.0, 0.0))]
            bounds = {}
            if case_type == CalibrationCaseType.UNDER_ACTION.value:
                bounds["min_force_n"] = 1.0
            if case_type == CalibrationCaseType.OVER_ACTION.value:
                bounds["max_force_n"] = 1.0
            cases.append(self.case(case_type, samples, case_id=f"suite-{case_type}", **bounds))

        suite = audit_calibration_suite(cases)
        self.assertTrue(suite.coverage_complete)
        self.assertEqual(suite.case_count, len(REQUIRED_CALIBRATION_CASE_TYPES))
        self.assertEqual(suite.physical_qualification, Decision.UNKNOWN)
        self.assertNotEqual(suite.decision, Decision.PASS)

    def test_suite_missing_case_type_is_not_complete(self):
        suite = audit_calibration_suite(
            [self.case(CalibrationCaseType.NO_CONTACT, [self.sample()])]
        )
        self.assertFalse(suite.coverage_complete)
        self.assertIn(CalibrationCaseType.CONTROLLED_LOAD_RESPONSE.value, suite.missing_case_types)
        self.assertEqual(suite.decision, Decision.FAIL)

    def test_reports_are_json_serializable_and_immutable_input_is_untouched(self):
        sample = self.sample(force=(3.0, 4.0, 0.0))
        case = self.case(CalibrationCaseType.NO_CONTACT, [sample])
        original = case.samples
        report = audit_calibration_case(case)
        json.dumps(report.as_dict(), sort_keys=True)
        self.assertIs(case.samples, original)
        self.assertEqual(case.samples[0].force_n, (3.0, 4.0, 0.0))
        self.assertEqual(report.decision, Decision.FAIL)


if __name__ == "__main__":
    unittest.main()
