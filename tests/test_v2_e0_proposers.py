from copy import deepcopy
import unittest

from synthesis_v2.d0_manner import PRESETS, build_plan
from synthesis_v2.e0_proposers import METHODS, SPEED_CHOICES, propose


class E0ProposerTests(unittest.TestCase):
    def setUp(self):
        self.requests = build_plan(tasks=("lift",), seeds=(42,))["episodes"]

    def test_deterministic_and_does_not_mutate_request_or_presets(self):
        original = deepcopy(self.requests)
        presets = deepcopy(PRESETS)
        for request in self.requests:
            for method in METHODS:
                result = propose(request, method, 2)
                self.assertEqual(result, propose(request, method, 2))
                self.assertFalse(result["promotion_allowed"])
                self.assertEqual(result["call_record"]["api_calls"], 0)
                if result["candidate_preset"] is not None:
                    result["candidate_preset"]["global"]["max_joint_vel"] = 99
        self.assertEqual(self.requests, original)
        self.assertEqual(PRESETS, presets)

    def test_fixed_is_neutral_and_global_random_is_instruction_blind(self):
        for attempt in range(1, 9):
            outputs = [propose(row, "G1", attempt) for row in self.requests]
            self.assertTrue(all(out["candidate_preset"] == outputs[0]["candidate_preset"] for out in outputs))
            self.assertEqual(len({out["proposal_seed"] for out in outputs}), 1)
        for row in self.requests:
            self.assertEqual(propose(row, "G0")["candidate_preset"], PRESETS["neutral"])

    def test_stage_random_respects_public_scopes(self):
        for row in self.requests:
            out = propose(row, "G2")
            scopes = set(row["requested_manner"]["phase_velocity"]) or {"global"}
            self.assertEqual(set(out["selected_speed_choices"]), scopes)
            self.assertTrue(set(out["selected_speed_choices"].values()) <= set(SPEED_CHOICES))
            self.assertEqual(set(out["candidate_preset"]["phase_velocity"]), scopes - {"global"})

    def test_public_rule_has_no_physical_context_claim(self):
        for row in self.requests:
            out = propose(row, "G3")
            self.assertEqual(out["candidate_preset"], row["requested_manner"])
            self.assertFalse(out["physical_context_variation"])
            self.assertFalse(out["context_effect_claim_allowed"])

    def test_original_request_provenance_survives_candidate_changes(self):
        row = self.requests[-1]
        outputs = [propose(row, method) for method in METHODS]
        self.assertEqual(len({out["request_sha256"] for out in outputs}), 1)
        self.assertEqual(len({out["requested_manner_sha256"] for out in outputs}), 1)
        changed = deepcopy(row)
        changed["raw_language"] += " Please."
        self.assertNotEqual(propose(changed, "G0")["request_sha256"], outputs[0]["request_sha256"])
        self.assertNotEqual(propose(row, "G1", 2)["proposal_seed"], propose(row, "G1", 1)["proposal_seed"])

    def test_llm_method_is_explicitly_skipped(self):
        out = propose(self.requests[0], "M")
        self.assertEqual(out["status"], "skipped")
        self.assertIsNone(out["candidate_preset"])
        self.assertEqual(out["call_record"]["reason"], "api_not_configured")

    def test_invalid_or_ambiguous_requests_fail_closed(self):
        for method, attempt in [("bad", 1), ("G0", 0), ("G0", -1), ("G0", True)]:
            with self.assertRaises(ValueError):
                propose(self.requests[0], method, attempt)
        bad = deepcopy(self.requests[0])
        bad["requested_manner"] = PRESETS["fast"]
        with self.assertRaises(ValueError):
            propose(bad, "G3")


if __name__ == "__main__":
    unittest.main()
