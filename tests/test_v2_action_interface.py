"""Pure structural tests for the v2 RoboSuite/LIBERO action contract.

These tests do not import either simulator and do not perform a controller
step.  A successful conversion here only means that an explicitly supplied
joint state and timing record can form a candidate for a future round-trip.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from synthesis_v2.action_interface import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    ActionConversionError,
    ActionInterfaceError,
    ActionInterfaceSpec,
    ChannelSpec,
    HiddenTeacherControls,
    JOINT_FRAME,
    LA4VLA_LIBERO_OSC_POSE_INTERFACE,
    LEROBOT_EXPORT_CANDIDATE_INTERFACE,
    LIBERO_LOW_LEVEL_INTERFACE,
    LIBERO_STUDENT_INTERFACE,
    ROBOSUITE_TEACHER_INTERFACE,
    ROUND_TRIP_CANDIDATE_SCHEMA,
    ROUND_TRIP_CANDIDATE_SCHEMA_VERSION,
    RoundTripCandidate,
    StudentAction,
    TeacherAction,
    build_round_trip_candidate,
    convert_action,
    convert_teacher_to_student,
    validate_action,
    validate_candidate,
)


class ActionInterfaceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = (0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70)
        self.target = (0.11, -0.18, 0.27, -0.40, 0.54, -0.65, 0.71)

    def teacher(self, **kwargs) -> TeacherAction:
        return TeacherAction(
            self.target,
            -1.0,
            timestamp_s=1.25,
            dt_s=0.05,
            current_joint_rad=self.current,
            **kwargs,
        )

    def test_interfaces_have_same_width_but_different_joint_semantics(self) -> None:
        self.assertEqual(ROBOSUITE_TEACHER_INTERFACE.action_dimension, 8)
        self.assertEqual(LIBERO_STUDENT_INTERFACE.action_dimension, 8)
        self.assertEqual(ROBOSUITE_TEACHER_INTERFACE.channel("joint_target").unit, "rad")
        self.assertEqual(LIBERO_STUDENT_INTERFACE.channel("joint_delta").unit, "rad")
        self.assertIn("absolute", ROBOSUITE_TEACHER_INTERFACE.controller)
        self.assertIn("delta", LIBERO_STUDENT_INTERFACE.controller)
        self.assertNotIn("v_max", {channel.name for channel in LIBERO_STUDENT_INTERFACE.channels})
        self.assertIn("v_max", {channel.name for channel in ROBOSUITE_TEACHER_INTERFACE.auxiliary_channels})

    def test_existing_lerobot_export_is_a_distinct_unverified_7d_candidate(self) -> None:
        self.assertIs(LIBERO_STUDENT_INTERFACE, LIBERO_LOW_LEVEL_INTERFACE)
        self.assertEqual(LEROBOT_EXPORT_CANDIDATE_INTERFACE.action_dimension, 7)
        self.assertEqual(
            [channel.dimension for channel in LEROBOT_EXPORT_CANDIDATE_INTERFACE.action_channels],
            [3, 3, 1],
        )
        self.assertIn("unverified", LEROBOT_EXPORT_CANDIDATE_INTERFACE.controller)
        report = validate_action(
            [0.0] * 7,
            "lerobot_export_candidate",
            timestamp_s=0.0,
            dt_s=0.05,
        )
        self.assertTrue(report.valid)

        with self.assertRaises(ActionConversionError) as caught:
            convert_teacher_to_student(
                self.teacher(), target_interface=LEROBOT_EXPORT_CANDIDATE_INTERFACE
            )
        self.assertEqual(caught.exception.code, "unverified_export_semantics")

    def test_la4vla_osc_pose_is_distinct_from_same_width_lerobot_export(self) -> None:
        self.assertEqual(LA4VLA_LIBERO_OSC_POSE_INTERFACE.action_dimension, 7)
        self.assertEqual(LEROBOT_EXPORT_CANDIDATE_INTERFACE.action_dimension, 7)
        self.assertEqual(
            LA4VLA_LIBERO_OSC_POSE_INTERFACE.channel(
                "osc_translation_delta_normalized"
            ).unit,
            "normalized_command",
        )
        self.assertEqual(
            LEROBOT_EXPORT_CANDIDATE_INTERFACE.channel(
                "eef_translation_state_difference"
            ).unit,
            "m",
        )
        self.assertEqual(
            LA4VLA_LIBERO_OSC_POSE_INTERFACE.nominal_control_frequency_hz,
            20.0,
        )
        with self.assertRaises(ActionConversionError) as caught:
            convert_teacher_to_student(
                self.teacher(), target_interface=LA4VLA_LIBERO_OSC_POSE_INTERFACE
            )
        self.assertEqual(caught.exception.code, "adapter_not_verified")

    def test_versioned_channel_schema_records_units_frames_sign_and_time(self) -> None:
        payload = ROBOSUITE_TEACHER_INTERFACE.as_dict()
        self.assertEqual(payload["schema_version"], ACTION_INTERFACE_SCHEMA_VERSION)
        self.assertEqual(payload["action_dimension"], 8)
        joint = payload["action_channels"][0]
        grip = payload["action_channels"][1]
        self.assertEqual(joint["coordinate_frame"], JOINT_FRAME)
        self.assertEqual(joint["unit"], "rad")
        self.assertEqual(grip["sign_convention"], "normalized_command:-1=open,+1=close")
        self.assertTrue(payload["timing"]["dt_required"])

    def test_absolute_target_converts_only_with_explicit_state_and_dt(self) -> None:
        converted = convert_teacher_to_student(self.teacher())
        for actual, expected in zip(
            converted.joint_delta_rad,
            (0.01, 0.02, -0.03, 0.0, 0.04, -0.05, 0.01),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(converted.gripper_command, -1.0)
        self.assertEqual(converted.timestamp_s, 1.25)
        self.assertEqual(converted.dt_s, 0.05)

    def test_missing_dt_fails_and_does_not_use_nominal_20_hz(self) -> None:
        teacher = TeacherAction(
            self.target,
            -1.0,
            timestamp_s=1.25,
            current_joint_rad=self.current,
        )
        with self.assertRaises(ActionConversionError) as caught:
            convert_teacher_to_student(teacher)
        self.assertEqual(caught.exception.code, "missing_dt")

    def test_missing_current_state_fails_for_absolute_to_delta(self) -> None:
        teacher = TeacherAction(self.target, -1.0, timestamp_s=1.25, dt_s=0.05)
        with self.assertRaises(ActionConversionError) as caught:
            convert_teacher_to_student(teacher)
        self.assertEqual(caught.exception.code, "missing_state")

    def test_hidden_profile_kp_force_and_width_are_not_silently_dropped(self) -> None:
        teacher = self.teacher(
            hidden_controls=HiddenTeacherControls(
                v_max_rad_s=0.03,
                a_max_rad_s2=0.01,
                jerk_limit_rad_s3=0.8,
                kp_pos=500.0,
                grip_force_fraction=0.2,
                grip_width_m=0.008,
            )
        )
        with self.assertRaises(ActionConversionError) as caught:
            convert_teacher_to_student(teacher)
        self.assertEqual(caught.exception.code, "unexpressed_teacher_field")
        self.assertEqual(caught.exception.field, "v_max")

        converted = convert_teacher_to_student(teacher, allow_unexpressed=True)
        self.assertEqual(
            converted.unexpressed_fields,
            ("v_max", "a_max", "jerk_limit", "kp_pos", "grip_force", "grip_width"),
        )

    def test_dimension_unit_sign_and_coordinate_mismatches_fail(self) -> None:
        with self.assertRaises(ActionInterfaceError) as wrong_dim:
            StudentAction((0.0,) * 6, -1.0, timestamp_s=1.0, dt_s=0.05)
        self.assertEqual(wrong_dim.exception.code, "dimension_mismatch")

        wrong_unit = replace(
            ROBOSUITE_TEACHER_INTERFACE,
            action_channels=(
                replace(ROBOSUITE_TEACHER_INTERFACE.channel("joint_target"), unit="deg"),
                ROBOSUITE_TEACHER_INTERFACE.channel("gripper_command"),
            ),
        )
        with self.assertRaises(ActionConversionError) as unit_error:
            convert_teacher_to_student(self.teacher(), source_interface=wrong_unit)
        self.assertEqual(unit_error.exception.code, "unit_mismatch")

        wrong_sign = replace(
            LIBERO_STUDENT_INTERFACE,
            action_channels=(
                LIBERO_STUDENT_INTERFACE.channel("joint_delta"),
                replace(
                    LIBERO_STUDENT_INTERFACE.channel("gripper_command"),
                    sign_convention="normalized_command:-1=close,+1=open",
                ),
            ),
        )
        with self.assertRaises(ActionConversionError) as sign_error:
            convert_teacher_to_student(self.teacher(), target_interface=wrong_sign)
        self.assertEqual(sign_error.exception.code, "sign_mismatch")

        wrong_frame = replace(
            LIBERO_STUDENT_INTERFACE,
            action_channels=(
                replace(LIBERO_STUDENT_INTERFACE.channel("joint_delta"), coordinate_frame="world_xyz"),
                LIBERO_STUDENT_INTERFACE.channel("gripper_command"),
            ),
        )
        with self.assertRaises(ActionConversionError) as frame_error:
            convert_teacher_to_student(self.teacher(), target_interface=wrong_frame)
        self.assertEqual(frame_error.exception.code, "coordinate_mismatch")

    def test_interface_pair_mismatch_and_noninvertible_direction_fail(self) -> None:
        with self.assertRaises(ActionConversionError) as caught:
            convert_action(
                self.teacher(),
                "libero_student",
                "libero_student",
                current_joint_rad=self.current,
                dt_s=0.05,
                timestamp_s=1.25,
            )
        self.assertEqual(caught.exception.code, "interface_mismatch")

        student = StudentAction((0.01,) * 7, -1.0, timestamp_s=1.25, dt_s=0.05)
        with self.assertRaises(ActionConversionError) as reverse:
            convert_action(student, "libero_student", "robosuite_teacher")
        self.assertEqual(reverse.exception.code, "non_invertible_conversion")

    def test_invalid_gripper_and_vector_shapes_are_rejected(self) -> None:
        with self.assertRaises(ActionInterfaceError) as caught:
            TeacherAction((0.0,) * 7, 2.0, timestamp_s=0.0, dt_s=0.05)
        self.assertEqual(caught.exception.code, "gripper_range")

        report = validate_action(
            [0.0] * 7 + [-1.0],
            "libero_student",
            timestamp_s=0.0,
            dt_s=None,
        )
        self.assertFalse(report.valid)
        self.assertTrue(report.has_code("missing_dt"))

    def test_round_trip_candidate_is_frozen_and_explicitly_unverified(self) -> None:
        candidate = build_round_trip_candidate(self.teacher())
        self.assertIsInstance(candidate, RoundTripCandidate)
        self.assertEqual(candidate.schema_version, ROUND_TRIP_CANDIDATE_SCHEMA_VERSION)
        self.assertTrue(candidate.candidate_only)
        self.assertFalse(candidate.replay_verified)
        self.assertEqual(candidate.replay_status, "not_run")
        self.assertEqual(candidate.student_action.vector, candidate.student_action.values)
        self.assertTrue(validate_candidate(candidate).valid)
        encoded = json.dumps(candidate.as_dict(), ensure_ascii=False, sort_keys=True)
        self.assertIn("candidate_only", encoded)
        self.assertIn("not_run", encoded)
        self.assertEqual(ROUND_TRIP_CANDIDATE_SCHEMA.action_dimension, 8)

    def test_candidate_rejects_wrong_delta_or_claimed_replay(self) -> None:
        teacher = self.teacher()
        student = StudentAction((0.0,) * 7, -1.0, timestamp_s=1.25, dt_s=0.05)
        with self.assertRaises(ActionInterfaceError) as mismatch:
            RoundTripCandidate(
                timestamp_s=1.25,
                dt_s=0.05,
                state_before_joint_rad=self.current,
                teacher_action=teacher,
                student_action=student,
            )
        self.assertEqual(mismatch.exception.code, "conversion_mismatch")

        with self.assertRaises(ActionInterfaceError) as status:
            RoundTripCandidate(
                timestamp_s=1.25,
                dt_s=0.05,
                state_before_joint_rad=self.current,
                teacher_action=teacher,
                student_action=convert_teacher_to_student(teacher),
                replay_status="passed",
            )
        self.assertEqual(status.exception.code, "candidate_status")

    def test_mapping_serialization_is_versioned(self) -> None:
        payload = self.teacher().as_dict()
        self.assertEqual(payload["schema_version"], ACTION_INTERFACE_SCHEMA_VERSION)
        self.assertEqual(payload["action_dimension"], 8)
        self.assertEqual(payload["channels"]["joint_target"]["unit"], "rad")
        self.assertEqual(payload["channels"]["gripper_command"]["sign"], "normalized_command:-1=open,+1=close")
        json.dumps(payload, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
