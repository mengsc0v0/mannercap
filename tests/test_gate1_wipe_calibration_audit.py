from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate1_wipe_calibration_audit",
    ROOT / "scripts" / "gate1_wipe_calibration_audit.py",
)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class WipeCalibrationAuditTests(unittest.TestCase):
    def write_case(
        self,
        root: Path,
        name: str,
        penetration: float,
        completed: bool,
        markers: int,
        reward: float,
        eef_force: float,
        error: bool = False,
    ) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "initial_state.json").write_text(
            json.dumps({"model_sha256": "a" * 64}), encoding="utf-8"
        )
        attempt_result = {"skill": "probe", "success": completed}
        if error:
            attempt_result["error"] = "bounded controller failure"
        probe = {
            "schema": "mannercap-gate1-wipe-probe-v1",
            "surface_penetration_m": penetration,
            "status": "executed",
            "task_operable": completed,
            "wipe_attempt": {
                "final_marker_status": {
                    "task_completed": completed,
                    "wiped_markers": markers,
                    "total_markers": 100,
                    "reward": reward,
                },
                "attempt_results": [attempt_result],
            },
            "trace_summary": {
                "control_transitions": 10,
                "max_named_contact_force_n": eef_force / 10,
                "max_eef_force_n": eef_force,
                "max_eef_torque_nm": 0.1,
                "eef_wrench_qualification": "uncalibrated_robosuite_sensor",
            },
        }
        (directory / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
        return directory

    def cases(self, root: Path):
        return (
            self.write_case(root, "under", 0.0, False, 97, 0.02, 60.0),
            self.write_case(root, "nominal", 0.002, True, 100, 0.95, 140.0),
            self.write_case(root, "over", 0.01, False, 24, -0.2, 170.0, error=True),
        )

    def test_expected_three_cases_are_characterized_without_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_module.audit(*self.cases(Path(temporary)))
        self.assertTrue(result["task_outcome_cases_verified"])
        self.assertTrue(result["penetration_response_characterized"])
        self.assertFalse(result["wrench_calibration_verified"])
        self.assertFalse(result["promotion_allowed"])

    def test_missing_over_action_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            under, nominal, _over = self.cases(root)
            replacement = self.write_case(root, "over2", 0.01, True, 100, 0.8, 170.0)
            result = audit_module.audit(under, nominal, replacement)
        self.assertFalse(result["task_outcome_cases_verified"])

    def test_model_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            under, nominal, over = self.cases(Path(temporary))
            initial = json.loads((over / "initial_state.json").read_text())
            initial["model_sha256"] = "b" * 64
            (over / "initial_state.json").write_text(json.dumps(initial))
            result = audit_module.audit(under, nominal, over)
        self.assertFalse(result["task_outcome_cases_verified"])


if __name__ == "__main__":
    unittest.main()
