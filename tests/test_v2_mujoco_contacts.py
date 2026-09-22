import unittest

from synthesis_v2.mujoco_contacts import split_body_spatial_wrench, split_contact_wrench


class WrenchLayoutTests(unittest.TestCase):
    def test_cfrc_ext_is_torque_then_force(self):
        result = split_body_spatial_wrench((0.1, 0.2, 0.3, 3.0, 4.0, 0.0))
        self.assertEqual(result["torque_nm"], [0.1, 0.2, 0.3])
        self.assertEqual(result["force_n"], [3.0, 4.0, 0.0])

    def test_contact_force_is_force_then_torque(self):
        result = split_contact_wrench((3.0, 4.0, 0.0, 0.1, 0.2, 0.3))
        self.assertEqual(result["force_n"], [3.0, 4.0, 0.0])
        self.assertEqual(result["torque_nm"], [0.1, 0.2, 0.3])

    def test_wrong_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            split_body_spatial_wrench((1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            split_contact_wrench((1.0,) * 7)


if __name__ == "__main__":
    unittest.main()
