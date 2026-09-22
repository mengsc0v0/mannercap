import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.gate2_audit_e0_observation import audit, digest


PHASE_V = {
    "approach": 0.12,
    "descend": 0.072,
    "lift": 0.12,
    "transport": 0.06,
    "final": 0.06,
    "retreat": 0.12,
}
PHASE_A = {
    "approach": 0.084,
    "descend": 0.042,
    "lift": 0.084,
    "transport": 0.0336,
    "final": 0.0336,
    "retreat": 0.084,
}


def _parameters(v_max, a_max, phase_v, phase_a):
    return {
        "v_max": v_max,
        "a_max": a_max,
        "jerk_limit": None,
        "phase_v_max": dict(phase_v),
        "phase_a_max": dict(phase_a),
    }


def _fixture(root: Path, *, corrupt_neutral: bool = False) -> None:
    request = {
        "request_id": "e0-dev-lift-50",
        "episode_id": "d0-lift-50",
        "task": "lift",
        "requested_manner": {"global": {}, "phase_velocity": {}},
    }
    g0 = {
        "candidate_preset": {"global": {}, "phase_velocity": {}},
        "proposal_source": "test",
    }
    m = {
        "candidate_preset": {
            "global": {"max_joint_vel": 0.08},
            "phase_velocity": {"approach": {}},
        },
        "proposal_source": "test",
    }
    candidates = [
        {"request": request, "method": "G0", "index": 1, "proposal": g0},
        {"request": request, "method": "M", "index": 1, "proposal": m},
    ]
    plan = {
        "schema": "test",
        "requests": [request],
        "candidates": candidates,
        "max_attempts": 1,
        "physics_substep_cap_per_attempt": 150000,
        "phase_semantics": "explicit_neutral_v2",
    }
    plan["plan_sha256"] = digest(plan)
    (root / "episodes").mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    report_results = []
    for item in candidates:
        key = f'{request["request_id"]}-{item["method"]}-{item["index"]}'
        actual = _parameters(0.12, 0.084, PHASE_V, PHASE_A)
        if item["method"] == "M":
            actual = _parameters(
                0.068,
                0.0476,
                {"approach": 0.12, "descend": 0.0408, "lift": 0.068,
                 "transport": 0.034, "final": 0.034, "retreat": 0.068},
                {"approach": 0.084, "descend": 0.0238, "lift": 0.0476,
                 "transport": 0.01904, "final": 0.01904, "retreat": 0.0476},
            )
            if corrupt_neutral:
                actual["phase_v_max"]["approach"] = 0.068
                actual["phase_a_max"]["approach"] = 0.0476
        raw = {
            **request,
            "episode_id": key,
            "method_id": item["method"],
            "candidate_preset": item["proposal"]["candidate_preset"],
            "task": "lift",
            "status": "executed",
            "task_completed": True,
            "observed_execution_physics_substeps": 1,
            "wall_s": 0.1,
            "cpu_time_s": 0.1,
            "initial_state_sha256": "a" * 64,
            "compiled_model_sha256": "b" * 64,
            "executed_parameters": actual,
            "descriptive_phase_metrics": {},
            "offline_recomputation": {
                "integration_check": "pass",
                "timing_check_passed": True,
            },
        }
        episode_dir = root / "episodes" / key
        episode_dir.mkdir()
        (episode_dir / "result.json").write_text(json.dumps(raw), encoding="utf-8")
        row = {
            "request_id": request["request_id"],
            "request_snapshot": request,
            "request_snapshot_hash": digest(request),
            "candidate_plan": item["proposal"],
            "attempt_index": 1,
            "request_state_after": "budget_exhausted",
            "metadata": {
                "method_attempt_id": key,
                "validation": {"publishable": False, "manner": "unknown", "safety": "unknown"},
            },
            "resources": {
                "physics_substeps": 1,
                "api_calls": 0,
                "api_tokens": 0,
                "wall_time_s": 0.1,
                "cpu_time_s": 0.1,
            },
        }
        with (root / f'{item["method"]}_attempts.jsonl').open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        report_results.append({"episode_id": key})
    report = {
        "results": report_results,
        "api_calls": 0,
        "accepted": 0,
        "cost_limitations": "test fixture",
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")


class ObservationAuditTests(unittest.TestCase):
    def test_audit_checks_applied_neutral_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root)
            result = audit(root)
            self.assertTrue(result["valid"])
            self.assertGreater(result["explicit_neutral_parameter_checks"], 0)
            self.assertEqual(
                result["phase_semantic_audits"]["e0-dev-lift-50-M-1"]["manner_qualification"],
                "unknown",
            )
            self.assertEqual(
                result["phase_semantic_audits"]["e0-dev-lift-50-M-1"]["request_semantic_diff"]
                ["differences"][0]["kind"],
                "neutral_extra_constraints",
            )

    def test_optimized_python_cannot_disable_neutral_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root, corrupt_neutral=True)
            code = (
                "from pathlib import Path; "
                "from scripts.gate2_audit_e0_observation import audit; "
                f"audit(Path({str(root)!r}))"
            )
            proc = subprocess.run(
                [sys.executable, "-O", "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("semantic audit failed", proc.stderr)


if __name__ == "__main__":
    unittest.main()
