import math
import unittest

from synthesis_v2.la4vla_osc_adapter import (
    OscAdapterError,
    adapt_pose_transition,
    quaternion_to_rotvec_wxyz,
)


class La4vlaOscAdapterTests(unittest.TestCase):
    def test_identity_transition_is_zero_command(self):
        result = adapt_pose_transition(
            position_before_m=[0.1, 0.2, 0.3],
            quaternion_before_wxyz=[1, 0, 0, 0],
            position_after_m=[0.1, 0.2, 0.3],
            quaternion_after_wxyz=[1, 0, 0, 0],
            gripper_command=-1,
        )
        self.assertEqual(result.normalized_action, (0, 0, 0, 0, 0, 0, -1))
        self.assertEqual(result.algebraic_position_error_m, 0)
        self.assertEqual(result.algebraic_orientation_error_rad, 0)

    def test_translation_and_pre_multiplied_axis_angle_round_trip(self):
        angle = 0.25
        result = adapt_pose_transition(
            position_before_m=[0.4, -0.1, 0.2],
            quaternion_before_wxyz=[1, 0, 0, 0],
            position_after_m=[0.41, -0.12, 0.205],
            quaternion_after_wxyz=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
            gripper_command=1,
        )
        expected = (0.2, -0.4, 0.1, 0.0, 0.0, 0.5, 1.0)
        for actual, target in zip(result.normalized_action, expected):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertLess(result.algebraic_position_error_m, 1e-15)
        self.assertLess(result.algebraic_orientation_error_rad, 1e-15)
        self.assertEqual(
            result.as_dict()["rotation_composition"],
            "R_after_times_transpose_R_before",
        )

    def test_quaternion_sign_change_does_not_create_rotation(self):
        result = adapt_pose_transition(
            position_before_m=[0, 0, 0],
            quaternion_before_wxyz=[0.5, 0.5, 0.5, 0.5],
            position_after_m=[0, 0, 0],
            quaternion_after_wxyz=[-0.5, -0.5, -0.5, -0.5],
            gripper_command=0,
        )
        self.assertEqual(result.rotation_delta_rad, (0, 0, 0))

    def test_public_quaternion_to_rotvec_uses_shortest_representation(self):
        angle = 0.2
        rotvec = quaternion_to_rotvec_wxyz(
            [-math.cos(angle / 2), 0, 0, -math.sin(angle / 2)]
        )
        self.assertAlmostEqual(rotvec[2], angle, places=12)

    def test_out_of_range_fails_closed_or_records_explicit_clipping(self):
        kwargs = dict(
            position_before_m=[0, 0, 0],
            quaternion_before_wxyz=[1, 0, 0, 0],
            position_after_m=[0.06, 0, 0],
            quaternion_after_wxyz=[1, 0, 0, 0],
            gripper_command=-1,
        )
        with self.assertRaises(OscAdapterError) as caught:
            adapt_pose_transition(**kwargs)
        self.assertEqual(caught.exception.code, "command_out_of_range")
        clipped = adapt_pose_transition(**kwargs, allow_clipping=True)
        self.assertTrue(clipped.clipped)
        self.assertEqual(clipped.clipped_dimensions, (0,))
        self.assertAlmostEqual(clipped.normalized_action[0], 1.0)
        self.assertAlmostEqual(clipped.algebraic_position_error_m, 0.01)

    def test_invalid_quaternion_and_gripper_are_rejected(self):
        base = dict(
            position_before_m=[0, 0, 0],
            quaternion_before_wxyz=[1, 0, 0, 0],
            position_after_m=[0, 0, 0],
            quaternion_after_wxyz=[1, 0, 0, 0],
            gripper_command=-1,
        )
        with self.assertRaises(OscAdapterError) as quat_error:
            adapt_pose_transition(**{**base, "quaternion_after_wxyz": [0, 0, 0, 0]})
        self.assertEqual(quat_error.exception.code, "invalid_quaternion")
        with self.assertRaises(OscAdapterError) as grip_error:
            adapt_pose_transition(**{**base, "gripper_command": 2})
        self.assertEqual(grip_error.exception.code, "gripper_range")


if __name__ == "__main__":
    unittest.main()
