"""Tests for the local-only, calibration-gated candidate package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from synthesis_v2.constraint_registry import validate_registry


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate2_build_calibration_candidate",
    ROOT / "scripts" / "gate2_build_calibration_candidate.py",
)
assert SPEC and SPEC.loader
candidate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate_module
SPEC.loader.exec_module(candidate_module)


class CalibrationCandidateTests(unittest.TestCase):
    def test_existing_workspace_does_not_promote_from_reports_or_hash_indexes(self):
        manifest = candidate_module.build_manifest(ROOT, stamp="test")

        self.assertFalse(manifest["candidate_ready"])
        self.assertEqual(manifest["physical_qualification"], "unknown")
        self.assertFalse(manifest["promotion_allowed"])
        self.assertEqual(
            sum(item["eligible_for_measurement"] for item in manifest["raw_artifact_inventory"]),
            0,
        )
        self.assertEqual(manifest["conditions"]["local_raw_measurements_present"]["status"], "unknown")
        self.assertEqual(manifest["conditions"]["known_loads_and_sensor_vectors"]["status"], "unknown")

        summary = next(
            item for item in manifest["source_artifacts"] if item["artifact_id"] == "d0_development_summary"
        )
        self.assertEqual(summary["kind"], "json_summary")
        self.assertFalse(summary["raw_measurement"])
        self.assertFalse(summary["eligible_for_measurement"])

    def test_raw_payload_presence_alone_cannot_make_candidate_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            required = {
                "reports/gate1/force_calibration_protocol.md",
                "reports/gate1/force_measurement_spec.md",
                "reports/gate1/wipe_controlled_load_spec.md",
                "synthesis_v2/force_calibration.py",
                "synthesis_v2/contact_force_contract.py",
                "synthesis_v2/trajectory_contract.py",
            }
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("contract fixture\n", encoding="utf-8")

            payload = root / "raw" / "transitions.jsonl"
            payload.parent.mkdir(parents=True)
            payload.write_text('{"summary": "not a validated transition"}\n', encoding="utf-8")
            digest = candidate_module.sha256_file(payload)
            index = root / "reports/gate1/wipe_controlled_load_20260917.sha256"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(f"{digest}  raw/transitions.jsonl\n", encoding="utf-8")

            manifest = candidate_module.build_manifest(root, stamp="fixture")

        self.assertEqual(
            sum(item["eligible_for_measurement"] for item in manifest["raw_artifact_inventory"]),
            1,
        )
        self.assertFalse(manifest["candidate_ready"])
        self.assertFalse(manifest["conditions"]["known_loads_and_sensor_vectors"]["satisfied"])
        self.assertEqual(manifest["conditions"]["timestamp_dt_resolution"]["status"], "unknown")

    def test_complete_body_fixture_still_cannot_promote_contact_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            required = {
                "reports/gate1/force_calibration_protocol.md",
                "reports/gate1/force_measurement_spec.md",
                "reports/gate1/wipe_controlled_load_spec.md",
                "synthesis_v2/force_calibration.py",
                "synthesis_v2/contact_force_contract.py",
                "synthesis_v2/trajectory_contract.py",
            }
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("contract fixture\n", encoding="utf-8")

            raw_root = root / "reports/gate2/calibration_raw_fixture"
            raw_root.mkdir(parents=True)
            payload = {
                "independent_reference": True,
                "measurement_resolution_n": 0.001,
                "noise_std_n": 0.0001,
                "samples": [
                    {
                        "force_n": [0.0, 0.0, 0.0],
                        "torque_nm": [0.0, 0.0, 0.0],
                        "known_load_n": 0.0,
                        "sensor_valid": True,
                        "frame": "world",
                        "units_force": "N",
                        "units_torque": "N*m",
                        "sign_convention": "+x/+y/+z",
                        "timestamp_s": 0.0,
                        "dt": 0.1,
                        "reset_id": "r0",
                        "object_id": "obj",
                        "body_id": 1,
                        "body_name": "body",
                        "geom_ids": [2],
                        "geom_names": ["geom"],
                    },
                    {
                        "force_n": [1.0, 0.0, 0.0],
                        "torque_nm": [0.0, 0.0, 0.0],
                        "known_load_n": 1.0,
                        "sensor_valid": True,
                        "frame": "world",
                        "units_force": "N",
                        "units_torque": "N*m",
                        "sign_convention": "+x/+y/+z",
                        "timestamp_s": 0.1,
                        "dt": 0.1,
                        "reset_id": "r0",
                        "object_id": "obj",
                        "body_id": 1,
                        "body_name": "body",
                        "geom_ids": [2],
                        "geom_names": ["geom"],
                    },
                ],
            }
            payload_path = raw_root / "controlled_load_calibration.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest_path = root / "reports/gate1/evidence_manifest_attempt03.sha256"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                f"{candidate_module.sha256_file(payload_path)}  controlled_load_calibration.json\n",
                encoding="utf-8",
            )

            manifest = candidate_module.build_manifest(root, stamp="fixture", raw_root=raw_root)

        self.assertTrue(manifest["conditions"]["known_loads_and_sensor_vectors"]["satisfied"])
        self.assertTrue(manifest["conditions"]["frame_sign_and_units"]["satisfied"])
        self.assertTrue(manifest["conditions"]["timestamp_dt_resolution"]["satisfied"])
        self.assertTrue(manifest["conditions"]["object_body_geom_attribution"]["satisfied"])
        self.assertTrue(manifest["conditions"]["independent_calibration_input"]["satisfied"])
        self.assertFalse(manifest["conditions"]["contact_pair_calibration_verified"]["satisfied"])
        self.assertFalse(manifest["candidate_ready"])

    def test_candidate_registry_is_pending_null_threshold_and_structurally_valid(self):
        manifest = candidate_module.build_manifest(ROOT, stamp="test")
        registry = candidate_module.build_registry(manifest, stamp="test")

        self.assertEqual(registry["status"], "development_pending")
        self.assertEqual(registry["calibration"]["status"], "pending_calibration")
        self.assertFalse(registry["candidate_evidence"]["candidate_ready"])
        for task in registry["tasks"]:
            self.assertEqual(task["calibration_status"], "pending_calibration")
            for constraint in task["constraints"]:
                self.assertEqual(constraint["status"], "pending_calibration")
                self.assertEqual(
                    constraint["thresholds"],
                    {"lower": None, "upper": None, "tolerance": None},
                )
                self.assertIsNone(constraint["calibration"]["evidence_sha256"])

        report = validate_registry(registry)
        self.assertTrue(report.valid, report.as_dict())
        self.assertFalse(report.promotion_allowed)
        self.assertEqual(report.decision.value, "unknown")
        self.assertEqual(report.declared_hash, report.computed_hash)

    def test_report_links_unknown_findings_to_hashed_sources_and_missing_inputs(self):
        manifest = candidate_module.build_manifest(ROOT, stamp="test")
        registry = candidate_module.build_registry(manifest, stamp="test")
        report = candidate_module.build_report(manifest, registry, stamp="test")

        known_load = next(item for item in report["findings"] if item["finding_id"] == "known_load_response")
        self.assertEqual(known_load["decision"], "unknown")
        self.assertTrue(known_load["evidence_refs"])
        self.assertTrue(all(ref.get("sha256") for ref in known_load["evidence_refs"]))
        required = {item["input_id"] for item in report["next_experiment_inputs"]}
        self.assertIn("raw_controlled_load_payloads", required)
        self.assertIn("raw_contact_rows", required)
        self.assertIn("timing_and_validity", required)

    def test_package_inputs_are_deterministic(self):
        manifest_a = candidate_module.build_manifest(ROOT, stamp="test")
        manifest_b = candidate_module.build_manifest(ROOT, stamp="test")
        self.assertEqual(
            json.dumps(manifest_a, sort_keys=True),
            json.dumps(manifest_b, sort_keys=True),
        )
        registry_a = candidate_module.build_registry(manifest_a, stamp="test")
        registry_b = candidate_module.build_registry(manifest_b, stamp="test")
        self.assertEqual(
            json.dumps(registry_a, sort_keys=True),
            json.dumps(registry_b, sort_keys=True),
        )

    def test_retrieved_payload_recomputes_scoped_time_response_and_contact_observation(self):
        raw_root = ROOT / "reports" / "gate2" / "calibration_raw_20260921"
        controlled = raw_root / "gate1_wipe_controlled_load_20260917_attempt03" / "controlled_load_calibration.json"
        if not controlled.is_file():
            self.skipTest("retrieved calibration payload is not present in this checkout")

        manifest = candidate_module.build_manifest(ROOT, stamp="retrieved", raw_root=raw_root)
        audit = manifest["raw_measurement_audit"]

        self.assertFalse(manifest["candidate_ready"])
        self.assertEqual(audit["record_count"], 480)
        self.assertEqual(audit["distinct_known_load_n"], [0.0, 2.0, 5.0, 10.0])
        self.assertEqual(audit["timestamp_dt_audit"]["status"], "pass")
        self.assertEqual(audit["timestamp_dt_audit"]["stream_count"], 24)
        self.assertEqual(audit["timestamp_dt_audit"]["derived_dt_count"], 456)
        self.assertEqual(audit["timestamp_dt_audit"]["first_row_dt_unknown_stream_count"], 24)
        self.assertEqual(audit["response_recompute"]["status"], "pass")
        self.assertEqual(audit["response_recompute"]["paired_row_count"], 240)
        self.assertEqual(audit["body_binding_audit"]["status"], "pass")

        contact = audit["contact_pair_observation_audit"]
        self.assertEqual(contact["status"], "partial")
        self.assertEqual(contact["contact_row_count"], 92)
        self.assertEqual(contact["valid_contact_row_count"], 92)
        self.assertEqual(contact["frame_status"], "pass")
        self.assertEqual(contact["sign_status"], "unknown")
        self.assertFalse(contact["calibration_verified"])
        self.assertTrue(manifest["conditions"]["contact_pair_rows_observed"]["satisfied"])
        self.assertFalse(manifest["conditions"]["contact_pair_calibration_verified"]["satisfied"])


if __name__ == "__main__":
    unittest.main()
