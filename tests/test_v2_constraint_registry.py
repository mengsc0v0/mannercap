"""Tests for the versioned, calibration-gated constraint registry."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from synthesis_v2.contact_force_contract import Decision
from synthesis_v2.constraint_registry import (
    REQUIRED_TASKS,
    SCHEMA_VERSION,
    canonical_registry_hash,
    load_registry,
    public_generation_view,
    sealed_final_audit_reference,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "constraint_registry.yaml"


class ConstraintRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        loaded = load_registry(REGISTRY_PATH)
        self.document = deepcopy(dict(loaded.document))
        self.assertTrue(loaded.report.valid, loaded.report.as_dict())
        self.assertEqual(loaded.report.decision, Decision.UNKNOWN)
        self.assertFalse(loaded.report.promotion_allowed)

    def constraints(self, task_index=0):
        return self.document["tasks"][task_index]["constraints"]

    def test_checked_in_registry_has_version_hash_and_pending_boundary(self):
        report = validate_registry(self.document)
        self.assertEqual(self.document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(len(self.document["registry_version_hash"]), 64)
        self.assertEqual(report.declared_hash, report.computed_hash)
        self.assertTrue(report.valid)
        self.assertEqual(report.decision, Decision.UNKNOWN)
        self.assertFalse(report.promotion_allowed)

    def test_public_generation_and_sealed_audit_are_distinct_references(self):
        public = self.document["public_generation"]
        sealed = self.document["sealed_final_audit"]
        self.assertEqual(public["visibility"], "public")
        self.assertEqual(sealed["visibility"], "sealed")
        self.assertNotEqual(public["config_id"], sealed["config_id"])
        self.assertEqual(public["final_audit_ref"]["config_id"], sealed["config_id"])
        self.assertNotIn("thresholds", public)
        self.assertNotIn("constraints", public)
        self.assertNotIn("thresholds", sealed)
        self.assertNotIn("constraints", sealed)
        self.assertTrue(report := validate_registry(self.document).valid)

    def test_boundary_views_do_not_expose_the_other_section(self):
        public = public_generation_view(self.document)
        sealed = sealed_final_audit_reference(self.document)
        self.assertNotIn("tasks", public)
        self.assertNotIn("sealed_final_audit", public)
        self.assertNotIn("constraints", public)
        self.assertIn("config_ref", sealed)
        self.assertNotIn("constraints", sealed)
        self.assertNotIn("public_generation", sealed)

    def test_all_development_tasks_and_declared_phases_have_constraints(self):
        tasks = {task["task_id"]: task for task in self.document["tasks"]}
        self.assertEqual(set(tasks), set(REQUIRED_TASKS))
        for task in tasks.values():
            phase_ids = set(task["phases"])
            constrained = {constraint["phase"] for constraint in task["constraints"]}
            self.assertTrue(phase_ids <= constrained)
            self.assertTrue(all(constraint["thresholds"] == {"lower": None, "upper": None, "tolerance": None} for constraint in task["constraints"]))

    def test_wrong_unit_is_rejected(self):
        self.constraints()[0]["unit"] = "m/s"
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("wrong_unit", report.codes)

    def test_missing_phase_reference_is_rejected(self):
        self.constraints()[0]["phase"] = "not_declared"
        self.constraints()[0]["scope"]["stage_instance_id"] = "not_declared"
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("invalid_constraint_phase", report.codes)

    def test_empty_constraint_list_is_rejected(self):
        self.document["tasks"][0]["constraints"] = []
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("missing_constraints", report.codes)

    def test_pending_numeric_threshold_cannot_be_promoted(self):
        self.constraints()[0]["thresholds"]["upper"] = 0.25
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertFalse(report.promotion_allowed)
        self.assertIn("pending_threshold_value", report.codes)
        self.assertIn("threshold_before_calibration", report.codes)

    def test_future_calibrated_registry_has_an_explicit_promotion_path(self):
        # This is an in-memory future-state fixture only.  It does not alter
        # the checked-in development registry or claim that these values were
        # measured.
        self.document["status"] = "frozen"
        self.document["registry_version"] = "2026-09-16.frozen.fixture"
        self.document["calibration"]["status"] = "calibrated"
        self.document["calibration"]["evidence_sha256"] = "a" * 64
        self.document["sealed_final_audit"]["status"] = "sealed_ready"
        self.document["sealed_final_audit"]["config_sha256"] = "b" * 64
        for task in self.document["tasks"]:
            task["calibration_status"] = "calibrated"
            for constraint in task["constraints"]:
                constraint["status"] = "calibrated"
                constraint["thresholds"] = {"lower": 0.0, "upper": 1.0, "tolerance": 0.05}
                constraint["calibration"]["status"] = "calibrated"
                constraint["calibration"]["evidence_sha256"] = "c" * 64
        self.document["registry_version_hash"] = canonical_registry_hash(self.document)

        report = validate_registry(self.document)
        self.assertTrue(report.valid, report.as_dict())
        self.assertTrue(report.promotion_allowed)
        self.assertEqual(report.decision, Decision.PASS)

    def test_missing_calibration_source_is_rejected(self):
        self.document["calibration"]["source_ref"] = None
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("missing_calibration_source", report.codes)

    def test_missing_constraint_calibration_source_is_rejected(self):
        self.constraints()[0]["calibration"]["source_ref"] = ""
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("missing_calibration_source", report.codes)

    def test_task_manner_safety_and_unknown_policy_are_required(self):
        self.document["decision_contract"]["dimensions"].pop("safety")
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("missing_decision_dimension", report.codes)

        self.setUp()
        self.document["decision_contract"]["dimensions"]["task"]["states"] = ["pass", "fail"]
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("invalid_decision_states", report.codes)

    def test_unknown_cannot_be_declared_qualified(self):
        self.document["decision_contract"]["dimensions"]["manner"]["unknown_is_qualified"] = True
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("unknown_policy_allows_qualification", report.codes)

    def test_constraint_unknown_policy_is_required(self):
        self.constraints()[0].pop("unknown_policy")
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("invalid_constraint_unknown_policy", report.codes)

    def test_sealed_final_audit_cannot_inline_constraints(self):
        self.document["sealed_final_audit"]["constraints"] = []
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("sealed_inline_data", report.codes)

    def test_public_generation_cannot_inline_thresholds(self):
        self.document["public_generation"]["thresholds"] = {}
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("public_inline_audit_data", report.codes)

    def test_hash_mismatch_is_rejected(self):
        self.document["registry_version_hash"] = "0" * 64
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("registry_hash_mismatch", report.codes)

    def test_wrong_scope_phase_is_rejected(self):
        self.constraints()[0]["scope"]["stage_instance_id"] = "descend"
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("scope_phase_mismatch", report.codes)

    def test_calibration_version_is_required(self):
        self.constraints()[0]["calibration"]["version"] = None
        report = validate_registry(self.document)
        self.assertFalse(report.valid)
        self.assertIn("missing_calibration_version", report.codes)

    def test_hash_function_is_deterministic_and_does_not_mutate(self):
        before = deepcopy(self.document)
        digest_a = canonical_registry_hash(self.document)
        reordered = {key: self.document[key] for key in reversed(tuple(self.document.keys()))}
        digest_b = canonical_registry_hash(reordered)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(self.document, before)

    def test_report_is_json_serializable(self):
        payload = validate_registry(self.document).as_dict()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self.assertIn('"decision": "unknown"', encoded)
        self.assertIn('"promotion_allowed": false', encoded)


if __name__ == "__main__":
    unittest.main()
