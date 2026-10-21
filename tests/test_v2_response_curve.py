import math
import unittest

import numpy as np

from synthesis_v2 import response_curve as rc


def _sweep(levels, fn, reps=5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.repeat(levels, reps).astype(float)
    y = np.array([fn(v) for v in x]) + rng.normal(0, noise, len(x))
    return x, y


class TestStats(unittest.TestCase):
    def test_spearman_ties_and_monotone(self):
        self.assertAlmostEqual(rc.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)
        self.assertAlmostEqual(rc.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(rc.spearman([1, 1, 1], [1, 2, 3])))

    def test_isotonic_pav(self):
        np.testing.assert_allclose(rc.isotonic_increasing([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
        np.testing.assert_allclose(rc.isotonic_increasing([3, 2, 1]), [2, 2, 2])

    def test_auc(self):
        self.assertEqual(rc.auc_separation([1, 2], [3, 4]), 1.0)
        self.assertEqual(rc.auc_separation([3, 4], [1, 2]), 0.0)
        self.assertEqual(rc.auc_separation([1, 2], [1, 2]), 0.5)

    def test_linear_fit(self):
        f = rc.linear_fit([0, 1, 2, 3], [1, 3, 5, 7])
        self.assertAlmostEqual(f["slope"], 2.0)
        self.assertAlmostEqual(f["r2"], 1.0)


class TestAnalyse(unittest.TestCase):
    def test_linear_response_is_continuous(self):
        lv = np.linspace(0.04, 0.24, 11)
        x, y = _sweep(lv, lambda v: 0.9 * v, noise=0.002)
        s = rc.analyse_response(x, y)
        self.assertEqual(s["classification"]["label"], "continuous")
        self.assertGreaterEqual(s["monotone_range"]["spearman"], 0.95)
        self.assertAlmostEqual(s["monotone_range"]["command_hi"], 0.24)

    def test_saturation_is_excluded_from_monotone_range(self):
        lv = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.2, 0.4, 0.6, 0.9])
        x, y = _sweep(lv, lambda v: min(v, 0.14), noise=0.001)
        s = rc.analyse_response(x, y)
        self.assertTrue(s["monotone_range"]["found"])
        self.assertLessEqual(s["monotone_range"]["command_hi"], 0.14 + 1e-9)
        self.assertLess(s["spearman_all"], 1.0)
        self.assertAlmostEqual(s["reachable_realised"][1], 0.14, places=2)

    def test_staircase_is_graded_not_continuous(self):
        lv = np.arange(1.0, 13.0)
        x, y = _sweep(lv, lambda v: float(np.floor((v - 1) / 4)) * 10.0 + 3.0, noise=0.3)
        s = rc.analyse_response(x, y)
        self.assertTrue(s["classification"]["label"].startswith("graded") or
                        s["classification"]["label"] == "not_controllable")
        self.assertEqual(s["distinguishable"]["n_distinguishable"], 3)

    def test_single_plateau_point_trimmed(self):
        lv = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70], dtype=float)
        x, y = _sweep(lv, lambda v: min(v, 60.0), noise=0.4)
        s = rc.analyse_response(x, y)
        self.assertEqual(s["monotone_range"]["command_hi"], 60.0)
        self.assertEqual(s["classification"]["label"], "continuous")

    def test_quantised_response_is_graded_even_when_correlated(self):
        # integer number of "close steps" sets the value: many commands share one realised value
        lv = np.arange(2.0, 42.0, 2.0)
        x, y = _sweep(lv, lambda v: 4.5 * math.floor(v / 8.0) + 2.0, noise=0.05)
        s = rc.analyse_response(x, y)
        self.assertGreaterEqual(s["monotone_range"]["spearman"], 0.95)
        self.assertTrue(s["classification"]["label"].startswith("graded"))
        self.assertIn("staircase", s["classification"]["failed_criteria"])

    def test_compressed_top_excluded_from_continuous_range(self):
        lv = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70], dtype=float)
        # linear to 50, then compressed (+2 N per 10 N command) and finally saturated
        x, y = _sweep(lv, lambda v: v if v <= 50 else 50 + 0.2 * (min(v, 60) - 50), noise=0.4)
        s = rc.analyse_response(x, y)
        self.assertEqual(s["classification"]["label"], "continuous")
        self.assertLessEqual(s["continuous_range"]["command_hi"], 60.0)
        self.assertGreaterEqual(s["continuous_range"]["command_lo"], 10.0)

    def test_staircase_has_no_continuous_range(self):
        lv = np.arange(2.0, 42.0, 2.0)
        x, y = _sweep(lv, lambda v: 4.5 * math.floor(v / 8.0) + 2.0, noise=0.05)
        self.assertFalse(rc.analyse_response(x, y)["continuous_range"]["found"])

    def test_lucky_subinterval_of_staircase_not_continuous(self):
        # staircase whose steps are uneven: most levels distinct in the middle, many ties overall
        lv = np.arange(2.0, 44.0, 2.0)
        vals = {2: 2, 4: 2, 6: 6, 8: 9, 10: 9, 12: 9, 14: 16, 16: 17, 18: 19, 20: 24, 22: 25, 24: 25, 26: 30, 28: 32,
                30: 33, 32: 33, 34: 40, 36: 40, 38: 40, 40: 40, 42: 40}
        lv = np.array(sorted(vals), dtype=float)
        x, y = _sweep(lv, lambda v: vals[int(v)], noise=0.05)
        s = rc.analyse_response(x, y)
        self.assertTrue(s["classification"]["label"].startswith("graded"))

    def test_noise_only_not_controllable(self):
        lv = np.linspace(1, 10, 10)
        x, y = _sweep(lv, lambda v: 5.0, noise=1.0, seed=3)
        s = rc.analyse_response(x, y)
        self.assertEqual(s["classification"]["label"], "not_controllable")

    def test_invalid_episodes_excluded_and_counted(self):
        lv = np.linspace(1, 8, 8)
        x, y = _sweep(lv, lambda v: 2 * v, noise=0.01)
        y[::7] = np.nan
        s = rc.analyse_response(x, y)
        self.assertEqual(s["n_invalid"], int(np.isnan(y).sum()))
        self.assertEqual(s["classification"]["label"], "continuous")

    def test_inverse_map_roundtrip(self):
        lv = np.linspace(1, 8, 8)
        x, y = _sweep(lv, lambda v: 3 * v + 1, noise=0.01)
        s = rc.analyse_response(x, y)
        r = rc.command_for(s["inverse"], 13.0)
        self.assertTrue(r["in_range"])
        self.assertAlmostEqual(r["command"], 4.0, places=1)
        self.assertFalse(rc.command_for(s["inverse"], 100.0)["in_range"])


if __name__ == "__main__":
    unittest.main()
