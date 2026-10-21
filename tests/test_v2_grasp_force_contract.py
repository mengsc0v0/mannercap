import math
import unittest
from types import SimpleNamespace

import numpy as np

from synthesis_v2.grasp_force_contract import (
    GraspBinding, GraspState, read_grasp_contact_sample,
    summarize_grasp_force, summarize_telemetry_grasp,
)

L, R, CUBE, TABLE = 10, 11, 20, 30
BINDING = GraspBinding((L,), (R,), (CUBE,), "cube")


def _data(pairs):
    contacts = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
    return SimpleNamespace(ncon=len(contacts), contact=contacts)


def _fn(forces):
    def fn(model, data, ci, buf):
        buf[0] = forces[ci]
    return fn


def _read(pairs, forces):
    return read_grasp_contact_sample(
        None, _data(pairs), BINDING, contact_force_fn=_fn(forces))


class ReadSampleTests(unittest.TestCase):
    def test_no_contact_is_valid_zero(self):
        s = _read([], [])
        self.assertTrue(s.valid)
        self.assertEqual((s.left_force_n, s.right_force_n), (0.0, 0.0))

    def test_single_side(self):
        s = _read([(L, CUBE)], [5.0])
        self.assertTrue(s.valid)
        self.assertEqual((s.left_force_n, s.right_force_n), (5.0, 0.0))
        self.assertEqual((s.left_contacts, s.left_geom_id), (1, CUBE))

    def test_bilateral_either_geom_order(self):
        s = _read([(L, CUBE), (CUBE, R)], [4.0, 6.0])
        self.assertEqual((s.left_force_n, s.right_force_n), (4.0, 6.0))

    def test_non_target_force_kept_separate(self):
        s = _read([(L, TABLE), (R, CUBE)], [9.0, 2.0])
        self.assertEqual(s.left_force_n, 0.0)
        self.assertEqual(s.other_force_n, 9.0)
        self.assertEqual(s.right_force_n, 2.0)

    def test_finger_finger_ignored(self):
        s = _read([(L, R)], [3.0])
        self.assertEqual((s.left_force_n, s.right_force_n, s.other_force_n), (0, 0, 0))

    def test_injected_read_error_is_invalid_not_zero(self):
        def boom(*a):
            raise RuntimeError("x")
        s = read_grasp_contact_sample(
            None, _data([(L, CUBE)]), BINDING, contact_force_fn=boom)
        self.assertFalse(s.valid)
        self.assertTrue(math.isnan(s.left_force_n))

    def test_nan_force_is_invalid(self):
        s = _read([(L, CUBE)], [float("nan")])
        self.assertFalse(s.valid)

    def test_empty_binding_is_invalid(self):
        s = read_grasp_contact_sample(
            None, _data([]), GraspBinding((), (R,), (CUBE,)),
            contact_force_fn=_fn([]))
        self.assertFalse(s.valid)


class SummaryTests(unittest.TestCase):
    def test_no_contact(self):
        s = summarize_grasp_force([0, 0, 0], [0, 0, 0], [True] * 3)
        self.assertEqual(s.state, GraspState.NO_CONTACT)
        self.assertEqual(s.peak_total_n, 0.0)

    def test_single_side(self):
        s = summarize_grasp_force([0, 3, 0], [0, 0, 0], [True] * 3)
        self.assertEqual(s.state, GraspState.SINGLE_SIDE)

    def test_bilateral_and_recompute(self):
        left = [0, 4, 6, 0]
        right = [0, 2, 8, 0]
        s = summarize_grasp_force(left, right, [True] * 4, dt_s=0.05)
        self.assertEqual(s.state, GraspState.BILATERAL)
        self.assertEqual(s.peak_left_n, 6.0)
        self.assertEqual(s.peak_total_n, 14.0)
        self.assertEqual(s.peak_bilateral_n, 6.0)
        self.assertEqual(s.bilateral_steps, 2)
        self.assertAlmostEqual(s.bilateral_duration_s, 0.1)

    def test_missing_channel_is_unknown(self):
        self.assertEqual(
            summarize_grasp_force(None, None, None).state, GraspState.UNKNOWN)

    def test_invalid_step_makes_unknown_by_default(self):
        s = summarize_grasp_force(
            [0, 5, 5], [0, 5, float("nan")], [True, True, False])
        self.assertEqual(s.state, GraspState.UNKNOWN)

    def test_all_invalid_never_zero_newton_pass(self):
        s = summarize_grasp_force(
            [float("nan")] * 3, [float("nan")] * 3, [False] * 3)
        self.assertEqual(s.state, GraspState.UNKNOWN)
        self.assertIsNone(s.peak_total_n)

    def test_valid_flag_with_nan_is_rejected(self):
        s = summarize_grasp_force([float("nan")], [1.0], [True])
        self.assertEqual(s.state, GraspState.UNKNOWN)

    def test_threshold_is_explicit(self):
        s = summarize_grasp_force([0.2], [0.2], [True], contact_threshold_n=0.5)
        self.assertEqual(s.state, GraspState.NO_CONTACT)

    def test_phase_window(self):
        tele = [
            SimpleNamespace(phase=p, grasp_left_force_n=l, grasp_right_force_n=r,
                            grasp_force_valid=True, grasp_other_force_n=0.0)
            for p, l, r in [("approach", 0, 0), ("grip", 3, 3), ("lift", 0, 0)]
        ]
        self.assertEqual(
            summarize_telemetry_grasp(tele, phases=["grip"]).state,
            GraspState.BILATERAL)
        self.assertEqual(
            summarize_telemetry_grasp(tele, phases=["approach"]).state,
            GraspState.NO_CONTACT)

    def test_legacy_telemetry_is_unknown(self):
        legacy = [SimpleNamespace(phase="grip", grasp_contact_force=0.0)]
        self.assertEqual(summarize_telemetry_grasp(legacy).state, GraspState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
