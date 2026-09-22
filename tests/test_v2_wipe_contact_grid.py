from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from io import StringIO


ROOT = Path(__file__).resolve().parents[1]
from synthesis_v2 import wipe_contact_grid as grid  # noqa: E402


SPEC = importlib.util.spec_from_file_location(
    "gate1_wipe_contact_grid_cli", ROOT / "scripts" / "gate1_wipe_contact_grid.py"
)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_probe_case(
    root: Path,
    plan: dict,
    *,
    status: str = "executed",
    task_completed: bool = True,
    probe_error: str | None = None,
    attempts: list[dict] | None = None,
    include_transitions: bool = True,
) -> Path:
    case = plan["cases"][0]
    case_dir = root / case["case_directory"]
    probe_dir = root / case["run_directory"]
    _write_json(
        probe_dir / "initial_state.json",
        {
            "model_sha256": "a" * 64,
            "marker_status": {
                "total_markers": 100,
                "wiped_markers": 0,
                "remaining_markers": 100,
                "wiped_fraction": 0.0,
                "reward": 0.0,
                "task_completed": False,
            },
        },
    )
    marker_status = {
        "total_markers": 100,
        "wiped_markers": 100 if task_completed else 93,
        "remaining_markers": 0 if task_completed else 7,
        "wiped_fraction": 1.0 if task_completed else 0.93,
        "reward": 0.95 if task_completed else 0.2,
        "task_completed": task_completed,
    }
    probe = {
        "schema": grid.PROBE_SCHEMA,
        "task": "wipe",
        "seed": case["seed"],
        "surface_penetration_m": case["surface_penetration_m"],
        "surface_penetration_semantics": grid.PARAMETER_SEMANTICS,
        "budget": case["budget"],
        "model_sha256": "a" * 64,
        "status": status,
        "task_operable": task_completed,
        "wall_s": 12.5,
        "wipe_attempt": {
            "attempt_results": attempts or [
                {"skill": "approach_spill_surface", "success": True, "metrics": {"contact_count": 2}},
                {"skill": "wipe_spill_and_verify", "success": task_completed, "metrics": {"stage_attempt": 1}},
            ],
            "final_marker_status": marker_status,
        },
        "trace_summary": {
            "control_transitions": 2,
            "max_named_contact_force_n": 0.5,
            "max_eef_force_n": 12.0,
            "max_eef_torque_nm": 2.0,
            "eef_wrench_qualification": "uncalibrated_robosuite_sensor",
        },
    }
    if probe_error:
        probe["error"] = probe_error
    _write_json(probe_dir / "probe.json", probe)
    if include_transitions:
        transitions = [
            {
                "transition": 1,
                "contacts": [
                    {
                        "geom1": "table_collision",
                        "geom2": "wipe_corner_0",
                        "distance_m": -0.001,
                        "force_contact_local_n": [0.3, 0.4, 0.0],
                        "force_norm_n": 0.5,
                    }
                ],
                "eef_wrenches": [
                    {
                        "arm": "right",
                        "force_n": [3.0, 4.0, 0.0],
                        "torque_nm": [0.0, 0.0, 1.0],
                        "force_norm_n": 5.0,
                        "torque_norm_nm": 1.0,
                        "qualification": "uncalibrated_robosuite_sensor",
                    }
                ],
            },
            {
                "transition": 2,
                "contacts": [
                    {
                        "geom1": "wipe_corner_0",
                        "geom2": "table_collision",
                        "distance_m": -0.0005,
                        "force_contact_local_n": [0.2, 0.0, 0.0],
                        "force_norm_n": 0.2,
                    }
                ],
                "eef_wrenches": [
                    {
                        "arm": "right",
                        "force_n": [0.0, 0.0, 12.0],
                        "torque_nm": [0.0, 0.0, 2.0],
                        "force_norm_n": 12.0,
                        "torque_norm_nm": 2.0,
                        "qualification": "uncalibrated_robosuite_sensor",
                    }
                ],
            },
        ]
        path = probe_dir / "transitions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in transitions),
            encoding="utf-8",
        )
    return case_dir


class WipeContactGridTests(unittest.TestCase):
    def test_default_plan_is_nine_cells_and_keeps_parameter_semantics(self):
        plan = grid.build_grid_plan()
        self.assertEqual(len(plan["cases"]), 9)
        self.assertEqual(plan["seeds"], [42, 43, 44])
        self.assertEqual(plan["surface_penetrations_m"], [0.0, 0.002, 0.01])
        self.assertTrue(plan["parameter"]["known_applied_load"] is False)
        self.assertTrue(plan["parameter"]["approved_safety_threshold"] is False)
        self.assertIn("not a known", plan["parameter"]["semantics"])
        self.assertEqual(plan["budget"]["environment_max_steps"], 4000)

    def test_grid_rejects_values_outside_probe_geometry_domain_without_calling_them_safe(self):
        with self.assertRaisesRegex(ValueError, "not a safety limit"):
            grid.build_grid_plan([42], [0.02])
        with self.assertRaisesRegex(ValueError, "finite"):
            grid.build_grid_plan([42], [float("nan")])
        with self.assertRaisesRegex(ValueError, "unique"):
            grid.build_grid_plan([42, 42], [0.002])

    def test_audit_reads_success_markers_reward_contact_pairs_eef_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = grid.build_grid_plan([43], [0.002], max_steps=2500, wipe_attempts=2)
            _write_probe_case(
                root,
                plan,
                attempts=[
                    {
                        "skill": "approach_spill_surface",
                        "success": False,
                        "error": "first contact approach failed",
                    },
                    {"skill": "wipe_spill_and_verify", "success": True},
                ],
            )
            result = grid.audit_grid(root, plan)
        case = result["cases"][0]
        self.assertEqual(result["counts"]["task_success"], 1)
        self.assertEqual(case["task_outcome"], "success")
        self.assertIs(case["final_marker_status"]["task_completed"], True)
        self.assertEqual(case["final_marker_status"]["wiped_markers"], 100)
        self.assertEqual(case["final_reward"], 0.95)
        self.assertEqual(case["model_sha256"], "a" * 64)
        self.assertEqual(case["requested_budget"]["environment_max_steps"], 2500)
        self.assertEqual(case["observed_budget_use"]["wipe_spill_and_verify_calls"], 1)
        self.assertEqual(case["contact_response"]["maximum_contact_pair_force_n"], 0.5)
        self.assertEqual(
            case["contact_response"]["pairs"][0]["geom_pair"],
            ["table_collision", "wipe_corner_0"],
        )
        self.assertEqual(case["contact_response"]["pairs"][0]["observations"], 2)
        self.assertEqual(case["recent_eef_response"]["maximum_force_n"], 12.0)
        self.assertEqual(case["recent_eef_response"]["maximum_torque_nm"], 2.0)
        self.assertEqual(case["recent_eef_response"]["qualification"], "uncalibrated_robosuite_sensor")
        grouped = result["by_surface_penetration"]
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["surface_penetration_m"], 0.002)
        self.assertEqual(grouped[0]["task_outcomes"]["success_rate_among_known"], 1.0)
        self.assertEqual(grouped[0]["maximum_contact_pair_force_n"]["median"], 0.5)
        self.assertEqual(grouped[0]["maximum_recent_eef_force_n"]["maximum"], 12.0)
        self.assertEqual(len(case["failure_events"]), 1)
        self.assertEqual(case["execution_errors"], ["first contact approach failed"])
        self.assertFalse(result["contact_model_calibration_verified"])
        self.assertFalse(result["thresholds_frozen"])
        self.assertFalse(result["promotion_allowed"])

    def test_execution_error_preserves_probe_traceback_and_attempt_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = grid.build_grid_plan([42], [0.01])
            _write_probe_case(
                root,
                plan,
                status="execution_error",
                task_completed=False,
                probe_error="RuntimeError: controller target failed",
                attempts=[
                    {
                        "skill": "bounded_wipe_probe",
                        "success": False,
                        "error": "full attempt traceback",
                    }
                ],
            )
            result = grid.audit_grid(root, plan)
        case = result["cases"][0]
        self.assertEqual(case["run_state"], "execution_error")
        self.assertEqual(case["task_outcome"], "failure")
        self.assertIn("RuntimeError: controller target failed", case["execution_errors"])
        self.assertIn("full attempt traceback", case["execution_errors"])
        self.assertEqual(case["failure_events"][0]["result"]["error"], "full attempt traceback")
        self.assertEqual(case["final_reward"], 0.2)

    def test_process_failure_without_probe_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = grid.build_grid_plan([42], [0.002])
            case_dir = root / plan["cases"][0]["case_directory"]
            _write_json(
                case_dir / "process.json",
                {
                    "status": "execution_error",
                    "exit_code": None,
                    "error": "child process could not start",
                    "stderr": "spawn failed",
                },
            )
            result = grid.audit_grid(root, plan)
        case = result["cases"][0]
        self.assertEqual(case["run_state"], "execution_error")
        self.assertIn("child process could not start", case["execution_errors"])
        self.assertIn("probe.json is missing", case["execution_errors"][-1])
        self.assertEqual(case["task_outcome"], "unknown")

    def test_missing_cell_remains_planned(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = grid.audit_grid(
                Path(temporary), grid.build_grid_plan([42], [0.002])
            )
        self.assertEqual(result["counts"]["planned"], 1)
        self.assertEqual(result["cases"][0]["run_state"], "planned")

    def test_existing_probe_runs_are_audited_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = grid.build_grid_plan([42], [0.002])
            _write_probe_case(root, plan)
            source = root / plan["cases"][0]["run_directory"]
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.iterdir()
            }
            imported_plan = grid.build_existing_runs_plan([source])
            result = grid.audit_grid(root / "audit", imported_plan)
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.iterdir()
            }
        self.assertEqual(before, after)
        self.assertEqual(imported_plan["mode"], "existing_runs_audit")
        self.assertEqual(result["cases"][0]["task_outcome"], "success")
        self.assertEqual(result["cases"][0]["seed"], 42)

    def test_plan_hash_detects_changes(self):
        plan = grid.build_grid_plan([42], [0.002])
        plan["cases"][0]["seed"] = 45
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            grid.validate_plan(plan)

    def test_default_cli_only_writes_plan_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "grid"
            stdout = StringIO()
            with contextlib.redirect_stdout(stdout):
                status = cli.main(["--output", str(output)])
            summary = json.loads(stdout.getvalue())
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(summary["counts"]["planned"], 9)
        self.assertFalse(summary["execution_requested"])
        self.assertFalse((output / "cases").exists())
        self.assertEqual(len(plan["cases"]), 9)

    def test_report_spells_out_non_calibration_interpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = grid.build_grid_plan([42], [0.002])
            _write_probe_case(root, plan)
            rendered = grid.render_audit_markdown(grid.audit_grid(root, plan))
        self.assertIn("不是已知外加载荷", rendered)
        self.assertIn("不构成接触力校准或安全判定", rendered)
        self.assertIn("## 按压入量聚合", rendered)
        self.assertIn("contact max min/median/max", rendered)
        self.assertIn("model SHA-256", rendered)


if __name__ == "__main__":
    unittest.main()
