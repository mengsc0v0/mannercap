"""Offline timing recomputation must reject incomplete or invalid recordings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("gate0_smoke", Path(__file__).resolve().parents[1] / "scripts/gate0_smoke.py")
gate0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate0)


class AuditTest(unittest.TestCase):
    def audit(self, times):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text("".join(json.dumps({"state_t": {"time_s": a},
                "state_t_plus_1": {"time_s": b}, "task_success_after": True,
                "action_t": [0.0]}) + "\n"
                for a, b in times), encoding="utf-8")
            return gate0.audit_transitions(path)

    def test_valid_timing_is_not_physics_qualification(self):
        result = self.audit([(0, .05), (.05, .1)])
        self.assertTrue(result["timing_check_passed"])
        self.assertEqual(result["physics_qualification"], "unknown")

    def test_empty_and_repeated_time_fail(self):
        self.assertFalse(self.audit([])["timing_check_passed"])
        self.assertFalse(self.audit([(0, 0)])["timing_check_passed"])

    def test_unrecorded_gap_fails(self):
        self.assertFalse(self.audit([(0, .05), (.15, .2)])["timing_check_passed"])

    def test_nonfinite_time_fails(self):
        self.assertFalse(self.audit([(0, float("nan"))])["timing_check_passed"])

    def test_integration_count_must_match_elapsed_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "initial_state.json").write_text(json.dumps({"physics_timestep_s": .002}))
            row = {"state_t": {"time_s": 0}, "state_t_plus_1": {"time_s": .05},
                   "physics_substeps": 24, "task_success_after": True, "action_t": [0.0]}
            path = root / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(gate0.audit_transitions(path)["integration_check"], "fail")
            row["physics_substeps"] = 25
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(gate0.audit_transitions(path)["integration_check"], "pass")

    def test_nonfinite_transition_vector_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            row = {"state_t": {"time_s": 0, "qpos": [float("nan")], "qvel": [0.0]},
                   "state_t_plus_1": {"time_s": .05, "qpos": [0.0], "qvel": [0.0]},
                   "action_t": [0.0], "task_success_after": True}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(gate0.audit_transitions(path)["invalid_transition_vectors"], 1)


if __name__ == "__main__":
    unittest.main()
