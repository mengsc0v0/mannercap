"""Contract tests for the five Gate 1 synthesis method configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from synthesis_v2.constraint_registry import load_registry
from synthesis_v2.method_config import (
    ALLOWED_ABLATIONS,
    ATTEMPT_OUTCOMES,
    METHOD_IDS,
    RESOURCE_DIMENSIONS,
    MethodConfig,
    MethodConfigError,
    apply_ablation,
    canonical_config_hash,
    load_method_config,
    load_method_suite,
    validate_method_config,
    validate_method_suite,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = ROOT / "configs" / "generation_methods"
REGISTRY_PATH = ROOT / "configs" / "constraint_registry.yaml"


class MethodConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configs = load_method_suite(METHOD_DIR)
        self.by_id = {config.method_id: config for config in self.configs}

    def test_checked_in_suite_has_exactly_five_correct_method_profiles(self) -> None:
        self.assertEqual(tuple(config.method_id for config in self.configs), METHOD_IDS)
        self.assertEqual(
            {config.method["planner_kind"] for config in self.configs},
            {"fixed", "global_random", "stage_random", "context_rules", "context_stage_plan"},
        )
        self.assertEqual(self.by_id["G0"].method["proposal_source"], "fixed_script")
        self.assertEqual(self.by_id["G1"].method["proposal_source"], "global_random")
        self.assertEqual(self.by_id["G2"].method["proposal_source"], "stage_random")
        self.assertEqual(self.by_id["G3"].method["proposal_source"], "context_rules")
        self.assertEqual(self.by_id["M"].method["proposal_source"], "context_stage_plan")
        self.assertFalse(self.by_id["G1"].method["stage_local_control"])
        self.assertTrue(self.by_id["G2"].method["stage_local_control"])
        self.assertTrue(self.by_id["G3"].method["stage_local_control"])
        self.assertTrue(self.by_id["G3"].method["uses_object_attributes"])
        self.assertTrue(self.by_id["M"].method["uses_object_attributes"])

    def test_all_methods_share_executor_public_constraints_attempts_and_budget(self) -> None:
        shared = [config.shared for config in self.configs]
        self.assertEqual({canonical_config_hash(item) for item in shared}, {canonical_config_hash(shared[0])})
        common = self.configs[0].shared
        self.assertEqual(common["executor"]["id"], "shared_executor.v1")
        self.assertEqual(common["request_contract"], "synthesis_v2.request.v1")
        self.assertEqual(common["task_space_contract"], "synthesis_v2.task_space.v1")
        self.assertEqual(tuple(common["task_scope"]), ("lift", "stack"))
        self.assertEqual(common["constraint_source"]["visibility"], "public")
        self.assertEqual(common["constraint_source"]["unknown_policy"], "not_qualified")
        self.assertEqual(
            common["constraint_source"]["thresholds"],
            {"lower": None, "upper": None, "tolerance": None},
        )
        attempts = common["attempt_policy"]
        self.assertEqual(attempts["max_attempts"], 3)
        self.assertEqual(tuple(attempts["outcomes"]), ATTEMPT_OUTCOMES)
        self.assertFalse(attempts["hidden_retries"])
        self.assertTrue(attempts["stop_on_first_accepted"])
        self.assertTrue(attempts["acceptance_terminal"])
        self.assertTrue(attempts["preserve_original_request"])
        self.assertEqual(tuple(common["budget"]["resource_dimensions"]), RESOURCE_DIMENSIONS)
        self.assertEqual(
            set(common["budget"]["caps"]),
            set(RESOURCE_DIMENSIONS),
        )
        self.assertTrue(all(value is None for value in common["budget"]["caps"].values()))
        self.assertFalse(common["validation"]["final_audit_feedback"])
        self.assertFalse(common["audit_access"]["artifact_loading"])

    def test_public_rule_boundary_has_no_inline_or_hidden_rules(self) -> None:
        self.assertEqual(self.by_id["G0"].method["rules"]["rule_refs"], [])
        self.assertEqual(self.by_id["G1"].method["rules"]["rule_refs"], [])
        self.assertEqual(self.by_id["G2"].method["rules"]["rule_refs"], [])
        for method_id in ("G3", "M"):
            rules = self.by_id[method_id].method["rules"]
            self.assertEqual(rules["visibility"], "public")
            self.assertEqual(
                tuple(rules["rule_refs"]),
                ("task_space.public.v1", "constraint_registry.public_generation.v1"),
            )
            self.assertEqual(rules["inline_rules"], [])
            self.assertFalse(rules["hidden_rules"])

        registry = load_registry(REGISTRY_PATH)
        self.assertTrue(registry.report.valid, registry.report.as_dict())
        report = validate_method_config(self.by_id["M"], registry_document=registry.document)
        self.assertTrue(report.valid, report.as_dict())

    def test_unknown_thresholds_and_resource_caps_cannot_be_filled_early(self) -> None:
        document = self.by_id["G2"].to_dict()
        document["shared"]["constraint_source"]["thresholds"]["upper"] = 0.2
        report = validate_method_config(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("threshold" in issue for issue in report.issues))

        document = self.by_id["G2"].to_dict()
        document["method"]["sampling"]["numeric_domain"]["lower"] = 0.1
        report = validate_method_config(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("remain null" in issue for issue in report.issues))

        document = self.by_id["G2"].to_dict()
        document["shared"]["budget"]["caps"]["cpu_time_s"] = 10.0
        report = validate_method_config(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("caps" in issue for issue in report.issues))

    def test_sealed_or_final_audit_references_are_rejected(self) -> None:
        for mutation in (
            lambda document: document["shared"]["constraint_source"].update(
                registry_ref="sealed://audit/config.yaml"
            ),
            lambda document: document["method"]["rules"].update(
                rule_refs=["sealed_rule_table.v1"]
            ),
            lambda document: document["metadata"].update(
                final_audit_results={"pass": True}
            ),
        ):
            document = self.by_id["M"].to_dict()
            mutation(document)
            report = validate_method_config(document)
            self.assertFalse(report.valid, report.as_dict())
            self.assertTrue(
                any("final-audit" in issue or "sealed" in issue or "outside" in issue for issue in report.issues),
                report.as_dict(),
            )

    def test_hidden_rules_and_unknown_fields_are_rejected(self) -> None:
        document = self.by_id["G3"].to_dict()
        document["method"]["rules"]["hidden_rules"] = True
        report = validate_method_config(document)
        self.assertFalse(report.valid)

        document = self.by_id["G3"].to_dict()
        document["method"]["rules"]["inline_rules"] = [{"when": "opaque", "then": "hidden"}]
        report = validate_method_config(document)
        self.assertFalse(report.valid)

        document = self.by_id["G3"].to_dict()
        document["method"]["unregistered_rule"] = "implicit"
        report = validate_method_config(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("unknown field" in issue for issue in report.issues))

    def test_method_difference_outside_declared_profile_is_rejected(self) -> None:
        mutations = (
            ("G1", "planner_kind", "language_direct"),
            ("G2", "uses_object_attributes", True),
            ("G3", "stage_local_control", False),
            ("M", "proposal_source", "private_model"),
        )
        for method_id, field, value in mutations:
            document = self.by_id[method_id].to_dict()
            document["method"][field] = value
            report = validate_method_config(document)
            self.assertFalse(report.valid, (method_id, field, report.as_dict()))

    def test_invalid_retry_and_stop_contract_is_rejected(self) -> None:
        for field, value in (
            ("max_attempts", 4),
            ("hidden_retries", True),
            ("stop_on_first_accepted", False),
            ("acceptance_terminal", False),
            ("preserve_original_request", False),
        ):
            document = self.by_id["G0"].to_dict()
            document["shared"]["attempt_policy"][field] = value
            report = validate_method_config(document)
            self.assertFalse(report.valid, (field, report.as_dict()))

        document = self.by_id["G0"].to_dict()
        document["ablations"]["switches"]["pair_orchestration"] = False
        report = validate_method_config(document)
        self.assertFalse(report.valid)

    def test_ablations_are_explicit_and_keep_shared_contract(self) -> None:
        no_stage = apply_ablation(self.by_id["M"], "no_phase_local_control")
        self.assertFalse(no_stage.method["stage_local_control"])
        self.assertIn("no_phase_local_control", no_stage.ablations["active"])
        self.assertFalse(no_stage.ablations["switches"]["phase_local_control"])
        self.assertEqual(canonical_config_hash(no_stage.shared), canonical_config_hash(self.by_id["M"].shared))
        self.assertTrue(validate_method_config(no_stage).valid)

        no_context = apply_ablation(self.by_id["G3"], "no_context_attributes")
        self.assertFalse(no_context.method["uses_object_attributes"])
        self.assertTrue(validate_method_config(no_context).valid)

        no_filter = apply_ablation(self.by_id["G0"], "no_generation_manner_filter")
        self.assertFalse(no_filter.method["generation_manner_filter"])
        self.assertTrue(validate_method_config(no_filter).valid)

        no_pair = apply_ablation(self.by_id["G1"], "no_pair_orchestration")
        self.assertFalse(no_pair.method["pair_orchestration"])
        self.assertTrue(validate_method_config(no_pair).valid)

        no_feedback = apply_ablation(self.by_id["G2"], "no_failure_feedback")
        self.assertFalse(no_feedback.method["failure_feedback"])
        self.assertTrue(validate_method_config(no_feedback).valid)

        with self.assertRaises(MethodConfigError):
            apply_ablation(self.by_id["M"], "not_a_declared_ablation")
        self.assertEqual(len(ALLOWED_ABLATIONS), 5)

    def test_ablation_switches_cannot_change_without_a_declared_active_name(self) -> None:
        document = self.by_id["M"].to_dict()
        document["ablations"]["switches"]["phase_local_control"] = False
        report = validate_method_config(document)
        self.assertFalse(report.valid)

        document = self.by_id["M"].to_dict()
        document["ablations"]["active"] = ["no_phase_local_control"]
        document["ablations"]["switches"]["phase_local_control"] = True
        report = validate_method_config(document)
        self.assertFalse(report.valid)

    def test_suite_validation_rejects_missing_duplicate_or_mismatched_shared_config(self) -> None:
        report = validate_method_suite(self.configs[:-1])
        self.assertFalse(report.valid)
        self.assertTrue(any("exactly" in issue for issue in report.issues))

        report = validate_method_suite(self.configs + (self.configs[0],))
        self.assertFalse(report.valid)
        self.assertTrue(any("unique" in issue for issue in report.issues))

        altered = self.configs[1].to_dict()
        altered["shared"]["budget"]["max_attempts"] = 2
        report = validate_method_suite((self.configs[0], MethodConfig.from_mapping(altered), *self.configs[2:]))
        self.assertFalse(report.valid)
        self.assertTrue(any("max_attempts" in issue for issue in report.issues))

    def test_config_copy_and_hash_do_not_mutate_input(self) -> None:
        document = self.by_id["G0"].to_dict()
        before = deepcopy(document)
        config = MethodConfig.from_mapping(document)
        document["shared"]["budget"]["max_attempts"] = 99
        self.assertEqual(config.shared["budget"]["max_attempts"], 3)
        self.assertEqual(config.to_dict()["method_id"], "G0")
        self.assertEqual(canonical_config_hash(before), canonical_config_hash(before))


if __name__ == "__main__":  # pragma: no cover - convenient local invocation
    unittest.main()
