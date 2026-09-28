"""Unit tests for synthesis_v2.experience_library (pure numpy, no capx deps)."""
import os
import tempfile
import unittest

from synthesis_v2.experience_library import (
    ExperienceLibrary,
    flatten_trial_params,
)


class TestCalibrateColdStart(unittest.TestCase):
    def test_empty_library_returns_fallback(self):
        lib = ExperienceLibrary()
        value, used = lib.calibrate("lift", "approach", "v_max", "fast", fallback=0.33)
        self.assertEqual(value, 0.33)
        self.assertFalse(used)

    def test_below_min_samples_returns_fallback(self):
        lib = ExperienceLibrary()
        lib.add_experience("lift", "approach", "v_max", "fast", 0.30, "both_pass")
        lib.add_experience("lift", "approach", "v_max", "fast", 0.34, "both_pass")
        value, used = lib.calibrate("lift", "approach", "v_max", "fast", fallback=0.33, min_samples=3)
        self.assertEqual(value, 0.33)
        self.assertFalse(used)


class TestCalibrateWithSamples(unittest.TestCase):
    def test_median_of_matching_samples(self):
        lib = ExperienceLibrary()
        for v in (0.30, 0.32, 0.34, 0.36, 0.38):
            lib.add_experience("lift", "approach", "v_max", "fast", v, "both_pass")
        value, used = lib.calibrate("lift", "approach", "v_max", "fast", fallback=0.10, min_samples=3)
        self.assertTrue(used)
        self.assertAlmostEqual(value, 0.34, places=6)

    def test_only_both_pass_outcomes_counted(self):
        lib = ExperienceLibrary()
        lib.add_experience("lift", "approach", "v_max", "fast", 0.99, "task_only")
        lib.add_experience("lift", "approach", "v_max", "fast", 0.30, "both_pass")
        lib.add_experience("lift", "approach", "v_max", "fast", 0.32, "both_pass")
        lib.add_experience("lift", "approach", "v_max", "fast", 0.34, "both_pass")
        value, used = lib.calibrate("lift", "approach", "v_max", "fast", fallback=0.10, min_samples=3)
        self.assertTrue(used)
        self.assertNotAlmostEqual(value, 0.99, places=2)

    def test_bucket_isolation_by_task_phase_dimension_level(self):
        lib = ExperienceLibrary()
        for v in (0.30, 0.32, 0.34):
            lib.add_experience("lift", "approach", "v_max", "fast", v, "both_pass")
        # different phase -> should not leak into approach bucket
        for v in (0.01, 0.02, 0.03):
            lib.add_experience("lift", "descend", "v_max", "fast", v, "both_pass")
        # different task -> should not leak
        for v in (0.50, 0.55, 0.60):
            lib.add_experience("stack", "approach", "v_max", "fast", v, "both_pass")
        value, used = lib.calibrate("lift", "approach", "v_max", "fast", fallback=0.10, min_samples=3)
        self.assertTrue(used)
        self.assertAlmostEqual(value, 0.32, places=6)


class TestRepairRatio(unittest.TestCase):
    def test_repair_ratio_cold_start_fallback(self):
        lib = ExperienceLibrary()
        ratio, used = lib.calibrate_repair_ratio("pour", "transport", "spill", "v_max", fallback_ratio=0.75)
        self.assertEqual(ratio, 0.75)
        self.assertFalse(used)

    def test_repair_ratio_from_history(self):
        lib = ExperienceLibrary()
        # three historical successful repairs: before -> after
        lib.add_repair("pour", "transport", "spill", "v_max", 0.10, 0.070, "both_pass")
        lib.add_repair("pour", "transport", "spill", "v_max", 0.20, 0.150, "both_pass")
        lib.add_repair("pour", "transport", "spill", "v_max", 0.30, 0.210, "both_pass")
        ratio, used = lib.calibrate_repair_ratio("pour", "transport", "spill", "v_max", fallback_ratio=0.75, min_samples=3)
        self.assertTrue(used)
        self.assertAlmostEqual(ratio, 0.7, places=6)

    def test_repair_ratio_ignores_failed_retries(self):
        lib = ExperienceLibrary()
        lib.add_repair("pour", "transport", "spill", "v_max", 0.10, 0.09, "manner_infeasible")
        lib.add_repair("pour", "transport", "spill", "v_max", 0.20, 0.15, "both_pass")
        lib.add_repair("pour", "transport", "spill", "v_max", 0.30, 0.21, "both_pass")
        ratio, used = lib.calibrate_repair_ratio("pour", "transport", "spill", "v_max", fallback_ratio=0.75, min_samples=3)
        self.assertFalse(used)  # only 2 both_pass samples < min_samples=3


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        lib = ExperienceLibrary()
        lib.add_experience("lift", "approach", "v_max", "fast", 0.33, "both_pass", source_run_stamp="s1", seed=42)
        lib.add_repair("pour", "transport", "spill", "v_max", 0.10, 0.075, "both_pass", source_run_stamp="s1")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            lib.save(path)
            reloaded = ExperienceLibrary()
            reloaded.load(path)
            self.assertEqual(len(reloaded.experiences), 1)
            self.assertEqual(len(reloaded.repairs), 1)
            self.assertEqual(reloaded.experiences[0].value, 0.33)
            self.assertEqual(reloaded.repairs[0].after_value, 0.075)

    def test_append_record_incremental(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            lib = ExperienceLibrary()
            lib.add_experience("lift", "approach", "v_max", "fast", 0.30, "both_pass")
            lib.append_record(path, lib.experiences[0])
            lib.add_experience("lift", "approach", "v_max", "fast", 0.34, "both_pass")
            lib.append_record(path, lib.experiences[1])
            reloaded = ExperienceLibrary()
            reloaded.load(path)
            self.assertEqual(len(reloaded.experiences), 2)
            self.assertEqual({r.value for r in reloaded.experiences}, {0.30, 0.34})

    def test_load_missing_file_is_noop(self):
        lib = ExperienceLibrary()
        lib.load("/tmp/definitely_does_not_exist_experience_lib.jsonl")
        self.assertEqual(lib.experiences, [])
        self.assertEqual(lib.repairs, [])


class TestFlattenTrialParams(unittest.TestCase):
    def test_flatten_phase_and_flat_dims(self):
        params = {
            "v_max": 0.12,
            "a_max": 0.084,
            "jerk_limit": 1.0,
            "kp_pos": 1000.0,
            "grip": 0.6,
            "phase_v_max": {"approach": 0.33, "descend": 0.02},
            "phase_a_max": {"approach": 0.231, "descend": 0.014},
        }
        flat = flatten_trial_params(params)
        self.assertEqual(flat["v_max"]["approach"], 0.33)
        self.assertEqual(flat["v_max"]["descend"], 0.02)
        self.assertEqual(flat["a_max"]["approach"], 0.231)
        self.assertEqual(flat["grip"]["_global"], 0.6)
        self.assertEqual(flat["kp_pos"]["_global"], 1000.0)
        self.assertEqual(flat["jerk_limit"]["_global"], 1.0)

    def test_flatten_skips_none_values(self):
        params = {"phase_v_max": {"approach": None, "descend": 0.02}}
        flat = flatten_trial_params(params)
        self.assertNotIn("approach", flat["v_max"])
        self.assertEqual(flat["v_max"]["descend"], 0.02)


if __name__ == "__main__":
    unittest.main()
