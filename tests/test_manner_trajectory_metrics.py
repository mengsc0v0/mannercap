"""Unit tests for synthesis_v2.manner_trajectory_metrics (pure numpy)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synthesis_v2 import manner_trajectory_metrics as M


class TestSmoothness(unittest.TestCase):
    def test_sparc_smooth_greater_than_jerky(self):
        fs = 100.0
        t = np.arange(0, 2, 1.0 / fs)
        # smooth bell-shaped speed vs noisy speed
        smooth = np.exp(-((t - 1.0) ** 2) / 0.05)
        rng = np.random.default_rng(0)
        jerky = smooth + 0.3 * rng.standard_normal(t.size)
        jerky = np.abs(jerky)
        s_smooth = M.spectral_arc_length(smooth, fs)
        s_jerky = M.spectral_arc_length(jerky, fs)
        # SPARC is <= 0; smoother profile is closer to 0 (greater).
        self.assertGreater(s_smooth, s_jerky)
        self.assertLessEqual(s_smooth, 0.0)

    def test_sparc_degenerate_returns_zero(self):
        self.assertEqual(M.spectral_arc_length(np.zeros(50), 100.0), 0.0)
        self.assertEqual(M.spectral_arc_length([1.0, 1.0], 100.0), 0.0)

    def test_ldlj_smooth_greater_than_jerky(self):
        fs = 100.0
        t = np.arange(0, 2, 1.0 / fs)
        smooth = np.exp(-((t - 1.0) ** 2) / 0.05)
        rng = np.random.default_rng(1)
        jerky = np.abs(smooth + 0.3 * rng.standard_normal(t.size))
        self.assertGreater(
            M.log_dimensionless_jerk(smooth, fs),
            M.log_dimensionless_jerk(jerky, fs),
        )


class TestPhaseRatios(unittest.TestCase):
    def test_phase_mean_speeds_order_and_values(self):
        phases = ["approach"] * 3 + ["descend"] * 3
        speed = [0.30, 0.30, 0.30, 0.01, 0.01, 0.01]
        pms = M.phase_mean_speeds(phases, speed)
        self.assertEqual(list(pms.keys()), ["approach", "descend"])
        self.assertAlmostEqual(pms["approach"], 0.30)
        self.assertAlmostEqual(pms["descend"], 0.01)

    def test_step_ratio_matches_manual(self):
        pms = {"approach": 0.335, "descend": 0.010, "transport": 0.05, "final": 0.02}
        ratios = M.phase_step_ratios(pms)
        self.assertAlmostEqual(ratios["approach_descend_ratio"], 33.5, places=3)
        self.assertAlmostEqual(ratios["max_min_ratio"], 33.5, places=3)
        self.assertAlmostEqual(ratios["transport_final_ratio"], 2.5, places=3)


class TestPathEnergy(unittest.TestCase):
    def test_path_length_straight_line(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
        self.assertAlmostEqual(M.path_length(pos), 2.0)

    def test_integral_energy_constant(self):
        jv = np.ones((10, 7))
        # sum sq speed (=7 per row) * 10 rows * dt(0.1) = 7*10*0.1 = 7.0
        self.assertAlmostEqual(M.integral_kinetic_energy(jv, 0.1), 7.0)


class TestContact(unittest.TestCase):
    def test_contact_onset_and_approach_velocity(self):
        cfrc = np.array([0.0, 0.0, 0.0, 5.0, 6.0, 7.0])
        speed = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
        onset = M.contact_onset_index(cfrc, threshold=1.0)
        self.assertEqual(onset, 3)
        # approach velocity is speed at onset-1 == index 2 == 0.1
        self.assertAlmostEqual(
            M.contact_approach_velocity(cfrc, speed, threshold=1.0), 0.1
        )

    def test_post_contact_slope_positive(self):
        cfrc = np.concatenate([np.zeros(5), np.arange(0, 40, 2.0)])
        slope = M.post_contact_force_slope(cfrc, dt=0.01, threshold=1.0, window=10)
        self.assertGreater(slope, 0.0)

    def test_no_contact_returns_defaults(self):
        cfrc = np.zeros(50)
        self.assertIsNone(M.contact_onset_index(cfrc))
        self.assertEqual(M.contact_approach_velocity(cfrc, np.ones(50)), 0.0)
        self.assertEqual(M.post_contact_force_slope(cfrc, 0.01), 0.0)


class TestSeparability(unittest.TestCase):
    def test_cohens_d_large_effect(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, 50)
        b = rng.normal(3.0, 1.0, 50)
        d = M.cohens_d(a, b)
        self.assertGreater(abs(d), 0.8)

    def test_cohens_d_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(M.cohens_d(a, a), 0.0)

    def test_group_separability_structure(self):
        g = {
            "fast": [{"peak_joint_speed": v} for v in [1.0, 1.1, 0.9, 1.05]],
            "gentle": [{"peak_joint_speed": v} for v in [0.2, 0.25, 0.18, 0.22]],
        }
        out = M.group_separability(g, ["peak_joint_speed"])
        self.assertIn("peak_joint_speed", out)
        self.assertIn("fast|gentle", out["peak_joint_speed"])
        self.assertGreater(abs(out["peak_joint_speed"]["fast|gentle"]), 0.8)


class TestWholeTrajectory(unittest.TestCase):
    def _make_traj(self):
        fs = 100.0
        n = 300
        t = np.arange(n) / fs
        phase = np.array(["approach"] * 100 + ["descend"] * 100 + ["lift"] * 100)
        # eef moves fast in approach, slow in descend, medium in lift
        eef = np.zeros((n, 3))
        eef[:100, 0] = np.linspace(0, 0.30, 100)
        eef[100:200, 0] = 0.30 + np.linspace(0, 0.01, 100)
        eef[200:, 0] = 0.31 + np.linspace(0, 0.10, 100)
        joint_vel = np.tile(np.linspace(0.3, 0.05, n)[:, None], (1, 7))
        cfrc = np.concatenate([np.zeros(150), np.linspace(0, 20, 150)])
        return {
            "t": t, "phase": phase, "eef_pos": eef,
            "joint_vel": joint_vel, "cfrc_norm": cfrc,
        }

    def test_compute_trajectory_metrics_keys_and_ratio(self):
        m = M.compute_trajectory_metrics(self._make_traj())
        for k in ("sparc", "ldlj", "phase_mean_speeds", "phase_step_ratios",
                  "eef_path_length", "integral_kinetic_energy", "peak_cfrc",
                  "contact_approach_velocity", "post_contact_force_slope",
                  "mean_joint_speed", "peak_joint_speed", "peak_joint_speed_p95",
                  "fs_hz", "n_samples"):
            self.assertIn(k, m)
        # approach should be much faster than descend
        pms = m["phase_mean_speeds"]
        self.assertGreater(pms["approach"], pms["descend"])
        self.assertGreater(m["phase_step_ratios"]["approach_descend_ratio"], 5.0)
        self.assertAlmostEqual(m["fs_hz"], 100.0, places=3)
        self.assertGreater(m["peak_cfrc"], 15.0)

    def test_peak_joint_speed_p95_robust_to_settle_spike(self):
        # Simulate the known move_to_joints_profiled() artifact: a genuine
        # profiled speed (v_max ~ 0.3 rad/s) plus one single-step settle-catch-up
        # spike (~5.5 rad/s, independent of v_max) injected at the tail, as
        # produced by the un-profiled move_to_joints_blocking() call.
        traj = self._make_traj()
        joint_vel = traj["joint_vel"].copy()
        joint_vel[-1, :] = 5.5 / np.sqrt(7)  # norm ~= 5.5, single-step artifact
        traj["joint_vel"] = joint_vel
        m = M.compute_trajectory_metrics(traj)
        self.assertGreater(m["peak_joint_speed"], 5.0)
        # p95 should ignore the lone spike and stay close to the real profile speed
        self.assertLess(m["peak_joint_speed_p95"], 1.0)

    def test_grasp_force_metrics_present_and_grip_phase_selected(self):
        traj = self._make_traj()
        # grasp force spikes only during the grip phase samples
        n = traj["t"].size
        phase = np.array(["approach"] * 100 + ["grip"] * 100 + ["lift"] * 100)
        grasp = np.zeros(n)
        grasp[100:200] = 40.0
        traj["phase"] = phase
        traj["grasp_contact_force"] = grasp
        m = M.compute_trajectory_metrics(traj)
        for k in ("peak_grasp_force", "mean_grasp_force",
                  "peak_grasp_force_grip", "mean_grasp_force_grip"):
            self.assertIn(k, m)
        self.assertAlmostEqual(m["peak_grasp_force"], 40.0, places=6)
        # grip-phase mean should equal 40 (only grip samples counted)
        self.assertAlmostEqual(m["mean_grasp_force_grip"], 40.0, places=6)
        # whole-trajectory mean is diluted by zero samples
        self.assertLess(m["mean_grasp_force"], m["mean_grasp_force_grip"])

    def test_grasp_force_defaults_when_absent(self):
        m = M.compute_trajectory_metrics(self._make_traj())
        self.assertEqual(m["peak_grasp_force"], 0.0)
        self.assertEqual(m["mean_grasp_force_grip"], 0.0)


class TestClassifierAUC(unittest.TestCase):
    def test_roc_auc_perfect_and_chance(self):
        # perfectly separable
        self.assertAlmostEqual(
            M.roc_auc([3.0, 4.0, 5.0, 0.0, 1.0, 2.0],
                      [1, 1, 1, 0, 0, 0]), 1.0, places=6)
        # reversed => 0
        self.assertAlmostEqual(
            M.roc_auc([0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                      [1, 1, 1, 0, 0, 0]), 0.0, places=6)
        # identical scores => chance
        self.assertAlmostEqual(
            M.roc_auc([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0]), 0.5, places=6)

    def test_manner_classifier_loo_auc_separable(self):
        rng = np.random.RandomState(0)
        a = [{"sparc": -3.0 + 0.1 * rng.randn(),
              "mean_joint_speed": 0.30 + 0.01 * rng.randn()} for _ in range(8)]
        b = [{"sparc": -6.0 + 0.1 * rng.randn(),
              "mean_joint_speed": 0.05 + 0.01 * rng.randn()} for _ in range(8)]
        res = M.manner_classifier_auc(a, b, ["sparc", "mean_joint_speed"])
        self.assertEqual(res["n_a"], 8)
        self.assertGreater(res["loo_auc"], 0.9)
        self.assertGreater(res["univariate_auc"]["mean_joint_speed"], 0.9)

    def test_manner_classifier_auc_chance_when_identical(self):
        # single small-n LOO AUC has high variance under H0; the *average* over
        # many identically-distributed draws must sit near chance (0.5).
        aucs = []
        for s in range(40):
            rng = np.random.RandomState(s)
            a = [{"x": float(rng.randn())} for _ in range(8)]
            b = [{"x": float(rng.randn())} for _ in range(8)]
            aucs.append(M.manner_classifier_auc(a, b, ["x"])["loo_auc"])
        self.assertLess(abs(float(np.mean(aucs)) - 0.5), 0.15)

    def test_group_classifier_auc_structure(self):
        a = [{"v": 1.0 + 0.01 * i} for i in range(4)]
        b = [{"v": 5.0 + 0.01 * i} for i in range(4)]
        out = M.group_classifier_auc({"A": a, "B": b}, ["v"])
        self.assertIn("A|B", out)
        self.assertIn("loo_auc", out["A|B"])


class TestSimilarity(unittest.TestCase):
    def test_dtw_zero_for_identical(self):
        a = np.linspace(0, 1, 20)[:, None]
        self.assertAlmostEqual(M.dtw_distance(a, a), 0.0, places=9)

    def test_dtw_time_warp_invariance(self):
        # same shape, different speed -> DTW small; unrelated -> larger
        t = np.linspace(0, 1, 50)
        a = np.stack([np.sin(2 * np.pi * t)], axis=1)
        b = np.stack([np.sin(2 * np.pi * np.linspace(0, 1, 80))], axis=1)
        c = np.stack([np.cos(4 * np.pi * t)], axis=1)
        self.assertLess(M.dtw_distance(a, b), M.dtw_distance(a, c))

    def test_frechet_zero_for_identical_and_positive(self):
        a = np.stack([np.linspace(0, 1, 15), np.zeros(15)], axis=1)
        self.assertAlmostEqual(M.discrete_frechet_distance(a, a), 0.0, places=9)
        b = a + np.array([0.0, 0.5])
        self.assertAlmostEqual(M.discrete_frechet_distance(a, b), 0.5, places=6)

    def test_similarity_to_reference_keys(self):
        traj = {"eef_pos": np.random.RandomState(0).randn(30, 3)}
        ref = {"eef_pos": np.random.RandomState(1).randn(40, 3)}
        r = M.similarity_to_reference(traj, ref)
        self.assertIn("dtw_to_reference", r)
        self.assertIn("frechet_to_reference", r)
        self.assertGreater(r["dtw_to_reference"], 0.0)

    def test_silhouette_separated_vs_mixed(self):
        rng = np.random.RandomState(0)
        # two well-separated clusters
        A = np.concatenate([rng.randn(6, 4) + 10.0, rng.randn(6, 4) - 10.0])
        labA = np.array([0] * 6 + [1] * 6)
        sep = M.silhouette_score(A, labA)
        self.assertGreater(sep, 0.7)
        # random labels on one blob -> near zero / negative
        B = rng.randn(12, 4)
        labB = np.array([0, 1] * 6)
        mix = M.silhouette_score(B, labB)
        self.assertLess(mix, sep)

    def test_trajectory_signature_shape(self):
        traj = {"eef_pos": np.random.RandomState(0).randn(37, 3)}
        sig = M.trajectory_signature(traj, num=50)
        self.assertEqual(sig.shape, (150,))


if __name__ == "__main__":
    unittest.main()
