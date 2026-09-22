"""Contract tests for the offline Gate 2/E0 preflight runner."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from synthesis_v2.preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    PreflightError,
    canonical_preflight_hash,
    load_draft_manifest,
    preflight_attempt_id_for,
    run_preflight,
    write_preflight_report,
)
from synthesis_v2.request_manifest import (
    PairType,
    RequestDraft,
    build_development_manifest,
    make_pair,
    write_request_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = ROOT / "configs" / "generation_methods"
REGISTRY_PATH = ROOT / "configs" / "constraint_registry.yaml"
SCRIPT_PATH = ROOT / "scripts" / "gate2_e0_preflight.py"
_SPEC = importlib.util.spec_from_file_location("gate2_e0_preflight", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate2)


def _request(
    *,
    raw_language: str,
    requested_constraints: dict[str, object],
    physical_config: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    stage_scope: list[dict[str, object]] | None = None,
) -> RequestDraft:
    return RequestDraft(
        task="lift",
        scene_id="e0-scene",
        initial_state={"initial_state_id": "initial_dev", "state": {"qpos": [0.0]}},
        target_object="cube",
        physical_config={"configuration_ref": "lift:pending-a"}
        if physical_config is None
        else physical_config,
        raw_language=raw_language,
        requested_constraints=requested_constraints,
        stage_scope=[{"stage_instance_id": "approach"}]
        if stage_scope is None
        else stage_scope,
        allowed_control_channels=("joint_velocity", "gripper"),
        budget={"max_attempts": 3},
        split="development",
        geometry_key="lift:geometry:pending",
        manner_key="manner-neutral",
        metadata={} if metadata is None else metadata,
    )


def _manifest_records(*, include_pending: bool = True) -> list[dict[str, object]]:
    modifier_a = _request(
        raw_language="lift the cube",
        requested_constraints={"manner": "neutral", "raw_marker": "a"},
    )
    modifier_b = _request(
        raw_language="lift the cube carefully",
        requested_constraints={"manner": "careful", "raw_marker": "b"},
    )
    requests: list[RequestDraft] = [modifier_a, modifier_b]
    if include_pending:
        requests.append(
            _request(
                raw_language="pending lift",
                requested_constraints={},
                physical_config={},
                stage_scope=[],
                metadata={
                    "lifecycle_status": "pending_calibration",
                    "applicability": "pending_calibration",
                    "incomplete": True,
                    "requests_enabled": False,
                },
            )
        )
    return build_development_manifest(
        requests,
        [make_pair(modifier_a, modifier_b, PairType.MODIFIER_WORD)],
    )


def _write_manifest(path: Path, records: list[dict[str, object]], *, reverse: bool = False) -> None:
    ordered = list(reversed(records)) if reverse else records
    write_request_manifest(path, ordered)


class Gate2E0PreflightTests(unittest.TestCase):
    def test_load_draft_manifest_orders_and_validates_request_ids(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records, reverse=True)
            loaded = load_draft_manifest(path)
            self.assertEqual(
                [row["request_id"] for row in loaded],
                sorted(row["request_id"] for row in records),
            )

            tampered = list(records)
            tampered[0] = dict(tampered[0])
            tampered[0]["requested_constraints"] = {"tampered": True}
            _write_manifest(path, tampered)
            with self.assertRaisesRegex(PreflightError, "request_id"):
                load_draft_manifest(path)

    def test_runner_creates_pending_g0_g3_lanes_and_explicit_m_skip(self) -> None:
        records = _manifest_records()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records)
            report = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)

        self.assertEqual(report["schema_version"], PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["manifest"]["manifest_status"], "draft")
        self.assertEqual(report["manifest"]["request_count"], len(records))
        self.assertEqual(len(report["records"]), len(records) * 5)
        by_method: dict[str, list[dict[str, object]]] = {method_id: [] for method_id in ("G0", "G1", "G2", "G3", "M")}
        for row in report["records"]:
            by_method[row["method_id"]].append(row)
        for method_id in ("G0", "G1", "G2", "G3"):
            self.assertEqual(len(by_method[method_id]), len(records))
            self.assertTrue(all(row["preflight_status"] == "pending" for row in by_method[method_id]))
            self.assertTrue(all(len(row["candidate_attempts"]) == 1 for row in by_method[method_id]))
            self.assertTrue(all(row["budget"]["max_attempts"] == 3 for row in by_method[method_id]))
            self.assertTrue(all(row["budget"]["actual_attempts_charged"] == 0 for row in by_method[method_id]))
        self.assertEqual(len(by_method["M"]), len(records))
        self.assertTrue(all(row["preflight_status"] == "skipped" for row in by_method["M"]))
        self.assertTrue(all(row["reason_code"] == "api_not_configured" for row in by_method["M"]))
        self.assertTrue(all(row["budget"]["planned_candidate_slots"] == 0 for row in by_method["M"]))
        self.assertEqual(report["method_suite"]["preflight_methods"], ["G0", "G1", "G2", "G3"])
        self.assertEqual(report["method_suite"]["skipped"][0]["reason_code"], "api_not_configured")

    def test_no_execution_success_or_audit_is_fabricated(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records)
            report = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)

        guards = report["execution_guards"]
        self.assertTrue(all(value is False for value in guards.values()))
        self.assertEqual(report["attempt_ledger"]["actual_attempt_record_count"], 0)
        self.assertEqual(report["attempt_ledger"]["charged_attempts"], 0)
        self.assertEqual(report["unexecuted_result_policy"]["accepted_count"], 0)
        for row in report["records"]:
            self.assertEqual(row["validation"], {"task": "unknown", "manner": "unknown", "safety": "unknown", "overall": "unknown"})
            self.assertEqual(row["final_audit"], {"status": "unknown", "checked": False})
            attempt = row["candidate_attempts"][0]
            self.assertIsNone(attempt["ledger_outcome"])
            self.assertFalse(attempt["ledger_recorded"])
            self.assertEqual(attempt["validation"]["overall"], "unknown")
            self.assertEqual(attempt["final_audit_status"], "unknown")
            if row["method_id"] != "M":
                plan = attempt["candidate_plan"]
                self.assertIsNone(plan["control_plan"])
                self.assertIsNone(plan["predicted_plan"])
                self.assertIsNone(plan["executed_plan"])
                self.assertFalse(plan["sampling"]["sampled"])

    def test_pending_request_is_retained_without_becoming_executable(self) -> None:
        records = _manifest_records()
        pending_ids = {
            row["request_id"] for row in records if row["metadata"].get("incomplete") is True
        }
        self.assertEqual(len(pending_ids), 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records)
            report = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
        pending_rows = [row for row in report["records"] if row["request_id"] in pending_ids]
        self.assertEqual(len(pending_rows), 5)
        for row in pending_rows:
            self.assertEqual(row["request_lifecycle"], "pending_calibration")
            if row["method_id"] == "M":
                self.assertEqual(row["reason_code"], "api_not_configured")
            else:
                self.assertEqual(row["reason_code"], "request_pending_calibration")
                self.assertEqual(row["candidate_attempts"][0]["reason_code"], "request_pending_calibration")

    def test_report_is_deterministic_under_manifest_line_reordering(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records, reverse=True)
            first = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
            _write_manifest(path, records, reverse=False)
            second = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
        self.assertEqual(first, second)
        self.assertEqual(first["report_hash"], canonical_preflight_hash({key: value for key, value in first.items() if key != "report_hash"}))

    def test_attempt_identity_and_raw_constraints_are_stable(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records)
            report = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
        expected_ids = set()
        for row in report["records"]:
            request_id = row["request_id"]
            method_id = row["method_id"]
            attempt = row["candidate_attempts"][0]
            self.assertEqual(attempt["attempt_id"], "attempt_" + attempt["attempt_id"].removeprefix("attempt_"))
            self.assertEqual(attempt["method_attempt_id"], preflight_attempt_id_for(method_id, request_id, 1))
            expected_ids.add(attempt["method_attempt_id"])
            self.assertEqual(row["requested_constraints"], attempt["candidate_plan"]["requested_constraints"] if attempt["candidate_plan"] else row["requested_constraints"])
            self.assertEqual(row["request_snapshot"], attempt["request_snapshot"])
            self.assertEqual(row["request_snapshot_hash"], attempt["request_snapshot_hash"])
        self.assertEqual(len(expected_ids), len(report["records"]))

    def test_registry_pending_state_is_reported_as_unknown(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            _write_manifest(path, records)
            report = run_preflight(path, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
        registry = report["constraint_registry"]
        self.assertTrue(registry["validation_valid"])
        self.assertFalse(registry["promotion_allowed"])
        self.assertEqual(registry["status"], "development_pending")
        self.assertEqual(registry["decision"], "unknown")
        self.assertTrue(registry["public_view_only"])

    def test_forbidden_manifest_states_are_rejected(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            bad_status = dict(records[0])
            bad_status["manifest_status"] = "frozen"
            _write_manifest(path, [bad_status, *records[1:]])
            with self.assertRaisesRegex(PreflightError, "manifest_status"):
                load_draft_manifest(path)

            bad_task = dict(records[0])
            bad_task["task"] = "wipe"
            _write_manifest(path, [bad_task, *records[1:]])
            with self.assertRaisesRegex(PreflightError, "request_id|unsupported"):
                load_draft_manifest(path)

            bad_audit = dict(records[0])
            bad_audit["metadata"] = {"sealed_final_audit": "sealed://not-for-generation"}
            _write_manifest(path, [bad_audit, *records[1:]])
            with self.assertRaisesRegex(PreflightError, "final-audit|public preflight"):
                load_draft_manifest(path)

    def test_writer_and_cli_emit_one_stable_json_report(self) -> None:
        records = _manifest_records(include_pending=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            manifest = directory / "manifest.jsonl"
            output = directory / "preflight.json"
            _write_manifest(manifest, records)
            report = run_preflight(manifest, method_config_dir=METHOD_DIR, registry_path=REGISTRY_PATH)
            write_preflight_report(output, report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

            cli_output = directory / "cli.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = gate2.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(cli_output),
                        "--method-config-dir",
                        str(METHOD_DIR),
                        "--constraint-registry",
                        str(REGISTRY_PATH),
                    ]
                )
            self.assertEqual(result, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["charged_attempts"], 0)
            self.assertEqual(json.loads(cli_output.read_text(encoding="utf-8"))["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
