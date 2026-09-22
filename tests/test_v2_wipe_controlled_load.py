"""Pure-stdlib tests for the Wipe controlled-load response audit.

No RoboSuite, MuJoCo, NumPy, controller, or action path is imported here.
Fixtures model the JSON evidence emitted by the runtime probe.
"""

from __future__ import annotations

import copy
import unittest

from synthesis_v2.wipe_controlled_load import (
    RAW_CHANNEL,
    RECENT_CHANNEL,
    SENSOR_TRANSFORM,
    ControlledLoadSpec,
    Decision,
    audit_controlled_load_case,
    audit_controlled_load_suite,
    expected_sensor_delta,
)


IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _add(left: tuple[float, float, float], right: tuple[float, float, float]):
    return tuple(left[index] + right[index] for index in range(3))


def _channel(force, torque):
    return {"force_n": list(force), "torque_nm": list(torque)}


def _sample(timestamp, *, force, torque, reset_id="r0", contact_count=0, rotation=IDENTITY):
    return {
        "timestamp_s": timestamp,
        "reset_id": reset_id,
        "sensor_valid": True,
        "site_rotation_world": list(rotation),
        "selected_body_contact_count": contact_count,
        "channels": {
            RAW_CHANNEL: _channel(force, torque),
            RECENT_CHANNEL: _channel(force, torque),
        },
    }


def _binding():
    return {
        "body": {
            "id": 114,
            "name": "gripper0_right_wiping_gripper",
            "geom_ids": [21, 22],
            "geom_names": ["gripper0_right_wiping_corner1", "gripper0_right_wiping_corner2"],
        },
        "sensor": {
            "site_id": 8,
            "site_name": "gripper0_right_ft_frame",
            "force_sensor_id": 5,
            "force_sensor_name": "gripper0_right_force_ee",
            "torque_sensor_id": 6,
            "torque_sensor_name": "gripper0_right_torque_ee",
            "recent_channel": {
                "source": "robot.recent_ee_forcetorques",
                "arm": "right",
                "layout": "force_n_then_torque_nm",
            },
        },
    }


def _case(
    *,
    case_id="force_1n_x",
    case_type="force_level",
    force=(1.0, 0.0, 0.0),
    torque=(0.0, 0.0, 0.0),
    level=1.0,
    direction=(1.0, 0.0, 0.0),
    baseline_force=(0.2, -0.1, 0.05),
    baseline_torque=(0.01, -0.02, 0.03),
    loaded_force=None,
    loaded_torque=None,
    **extra,
):
    expected = expected_sensor_delta(force, torque, IDENTITY)
    if loaded_force is None:
        loaded_force = _add(baseline_force, expected["force_n"])
    if loaded_torque is None:
        loaded_torque = _add(baseline_torque, expected["torque_nm"])
    baseline = [
        _sample(0.0, force=baseline_force, torque=baseline_torque),
        _sample(0.01, force=baseline_force, torque=baseline_torque),
    ]
    loaded = [
        _sample(1.0, force=loaded_force, torque=loaded_torque),
        _sample(1.01, force=loaded_force, torque=loaded_torque),
    ]
    result = {
        "case_id": case_id,
        "case_type": case_type,
        "executed": True,
        "reset_id": "r0",
        "known_load": {
            "force_world_n": list(force),
            "torque_world_nm": list(torque),
            "force_unit": "N",
            "torque_unit": "N*m",
            "application_frame": "world",
            "sensor_frame": "site",
        },
        "sensor_transform": SENSOR_TRANSFORM,
        "binding": _binding(),
        "baseline_samples": baseline,
        "loaded_samples": loaded,
        "state_restored": True,
    }
    if case_type in {"force_level", "torque_level"}:
        result["known_load"]["level"] = level
        result["known_load"]["direction_world"] = list(direction)
    result.update(extra)
    return result


class WipeControlledLoadAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ControlledLoadSpec(
            min_window_samples=2,
            force_levels_n=(1.0, 2.0),
            force_directions_world=((1.0, 0.0, 0.0),),
            torque_levels_nm=(0.1,),
            torque_directions_world=((0.0, 0.0, 1.0),),
        )

    def test_expected_site_transform_is_explicit_and_sign_correct(self) -> None:
        result = expected_sensor_delta((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), IDENTITY)
        self.assertEqual(result["force_n"], (-1.0, -2.0, -3.0))
        self.assertEqual(result["torque_nm"], (-0.1, -0.2, -0.3))

    def test_no_contact_and_known_force_are_audited_for_both_channels(self) -> None:
        report = audit_controlled_load_case(_case(), spec=self.spec)
        self.assertEqual(report.decision, Decision.PASS)
        self.assertEqual(set(report.channel_results), {RAW_CHANNEL, RECENT_CHANNEL})
        self.assertEqual(report.channel_results[RAW_CHANNEL]["force_error_n"], 0.0)
        self.assertEqual(report.channel_results[RECENT_CHANNEL]["force_error_n"], 0.0)
        self.assertEqual(report.physical_qualification, Decision.UNKNOWN)

    def test_raw_and_recent_channels_are_not_silently_merged(self) -> None:
        case = _case()
        for sample in case["loaded_samples"]:
            sample["channels"][RECENT_CHANNEL]["force_n"][0] -= 0.02
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.FAIL)
        self.assertEqual(report.channel_results[RAW_CHANNEL]["force_error_n"], 0.0)
        self.assertGreater(report.channel_results[RECENT_CHANNEL]["force_error_n"], 0.01)
        self.assertTrue(report.has_code("force_response_error"))

    def test_missing_recent_channel_is_unknown(self) -> None:
        case = _case()
        for sample in case["loaded_samples"]:
            del sample["channels"][RECENT_CHANNEL]
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("missing_channel"))

    def test_mixed_six_vector_is_rejected(self) -> None:
        case = _case()
        case["loaded_samples"][0]["channels"][RAW_CHANNEL]["force_n"] = [1, 2, 3, 4, 5, 6]
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("mixed_wrench"))

    def test_wrong_body_site_sensor_or_geom_identity_is_unknown(self) -> None:
        case = _case()
        case["binding"]["body"]["id"] = 115
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("wrong_body"))

    def test_wrong_units_and_frame_are_unknown(self) -> None:
        case = _case()
        case["known_load"]["force_unit"] = "N*m"
        case["known_load"]["application_frame"] = "site"
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("wrong_force_units"))
        self.assertTrue(report.has_code("wrong_application_frame"))

    def test_timestamp_gap_and_reset_gap_are_unknown(self) -> None:
        case = _case()
        case["loaded_samples"][1]["timestamp_s"] = 2.0
        case["loaded_samples"][1]["reset_id"] = "r1"
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("timestamp_gap"))
        self.assertTrue(report.has_code("reset_mismatch"))

    def test_sensor_failure_and_contact_precondition_are_unknown(self) -> None:
        case = _case()
        case["loaded_samples"][0]["sensor_valid"] = False
        case["loaded_samples"][1]["selected_body_contact_count"] = 1
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("sensor_invalid"))
        self.assertTrue(report.has_code("contact_precondition_failed"))

    def test_unrestored_state_is_unknown(self) -> None:
        case = _case(state_restored=False)
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("state_not_restored"))

    def test_wrong_direction_level_vector_is_unknown(self) -> None:
        case = _case(force=(2.0, 0.0, 0.0), level=1.0)
        report = audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertTrue(report.has_code("invalid_load_case"))

    def test_suite_requires_all_declared_levels_and_tracks_monotonic_response(self) -> None:
        cases = [
            _case(case_id="zero", case_type="no_contact", force=(0.0, 0.0, 0.0), level=None),
            _case(case_id="f1", case_type="force_level", force=(1.0, 0.0, 0.0), level=1.0),
            _case(case_id="f2", case_type="force_level", force=(2.0, 0.0, 0.0), level=2.0),
            _case(
                case_id="t1",
                case_type="torque_level",
                force=(0.0, 0.0, 0.0),
                torque=(0.0, 0.0, 0.1),
                level=0.1,
                direction=(0.0, 0.0, 1.0),
            ),
        ]
        suite = audit_controlled_load_suite(cases, spec=self.spec)
        self.assertEqual(suite.decision, Decision.PASS)
        self.assertTrue(suite.coverage_complete)
        self.assertTrue(suite.response_characterized)
        self.assertFalse(suite.promotion_allowed)
        self.assertEqual(suite.physical_qualification, Decision.UNKNOWN)

    def test_suite_missing_level_is_unknown(self) -> None:
        cases = [
            _case(case_id="zero", case_type="no_contact", force=(0.0, 0.0, 0.0), level=None),
            _case(case_id="f1", case_type="force_level", force=(1.0, 0.0, 0.0), level=1.0),
        ]
        suite = audit_controlled_load_suite(cases, spec=self.spec)
        self.assertEqual(suite.decision, Decision.UNKNOWN)
        self.assertIn("force:1,0,0:2N", suite.missing_cases)
        self.assertIn("torque:0,0,1:0.1N*m", suite.missing_cases)

    def test_input_case_is_not_mutated(self) -> None:
        case = _case()
        before = copy.deepcopy(case)
        audit_controlled_load_case(case, spec=self.spec)
        self.assertEqual(case, before)


if __name__ == "__main__":
    unittest.main()
