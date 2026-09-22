import unittest

from synthesis_v2.llm_proposal_contract import (
    build_prompt,
    build_prompt_v2,
    expected_v2_unsupported_constraints,
    parse_response,
    parse_response_v2,
    validate_v2_coverage,
)


class LLMContractTests(unittest.TestCase):
    def test_explicit_neutral_resets_but_omitted_phase_inherits(self):
        from synthesis_v2.llm_proposal_contract import apply_explicit_neutral_phases
        params = {'phase_v_max': {'approach': .068, 'lift': .068},
                  'phase_a_max': {'approach': .0476, 'lift': .0476}}
        neutral = {'phase_v_max': {'approach': .12, 'lift': .12},
                   'phase_a_max': {'approach': .084, 'lift': .084}}
        actual = apply_explicit_neutral_phases(params, {'phase_velocity': {'approach': {}}}, neutral)
        self.assertEqual(actual['phase_v_max'], {'approach': .12, 'lift': .068})
        self.assertEqual(actual['phase_a_max']['approach'], .084)
        self.assertEqual(params['phase_v_max']['approach'], .068)

    def test_prompt_whitelists_public_fields(self):
        prompt = build_prompt({"task": "lift", "raw_language": "Lift slowly",
                              "api_key": "secret", "final_audit": "answer",
                              "requested_manner": "teacher answer"})
        self.assertNotIn("secret", str(prompt))
        self.assertNotIn("answer", str(prompt))
        self.assertEqual(prompt["instruction"], "Lift slowly")

    def test_valid_phase_plan(self):
        result = parse_response('{"global_speed":"slow","phase_speeds":{"approach":"fast"}}', "lift")
        self.assertEqual(result["global"]["max_joint_vel"], 0.08)
        self.assertEqual(result["phase_velocity"]["approach"]["min_joint_vel"], 0.18)

    def test_invalid_outputs_rejected(self):
        for value in ('{}', '[]', '{"global_speed":"slow","phase_speeds":{"transport":"fast"}}',
                      '{"global_speed":"slow","global_speed":"fast","phase_speeds":{}}',
                      '{"global_speed":"slow","phase_speeds":{},"api_key":"bad"}',
                      '{"global_speed":"slow","phase_speeds":{"lift":2}}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(value, "lift")

    def test_v2_prompt_makes_omission_and_explicit_neutral_distinct(self):
        prompt = build_prompt_v2({"task": "lift", "raw_language": "Approach fast, then neutral contact."})
        self.assertEqual(prompt["contract_version"], "e0-m-velocity-plan-v2")
        self.assertEqual(prompt["phase_scope_semantics"]["omitted_phase"], "inherit_global_speed")
        self.assertEqual(prompt["phase_scope_semantics"]["neutral_phase"], "explicit_neutral_override")
        self.assertIn("unsupported_request_constraints", prompt["required_response"])
        self.assertIn("max_jerk", prompt["unsupported_constraint_domain"])
        self.assertIn("omit unmentioned phases", prompt["instructions"])

    def test_v2_parser_requires_unsupported_constraint_declaration(self):
        result = parse_response_v2(
            '{"global_speed":"slow","phase_speeds":{"approach":"neutral"},'
            '"unsupported_request_constraints":["max_jerk"]}',
            "lift",
        )
        self.assertEqual(result["contract_version"], "e0-m-velocity-plan-v2")
        self.assertEqual(result["phase_velocity"]["approach"], {})
        self.assertEqual(result["unsupported_request_constraints"], ["max_jerk"])
        invalid = (
            '{"global_speed":"slow","phase_speeds":{},'
            '"unsupported_request_constraints":[]}'
        )
        self.assertEqual(parse_response_v2(invalid, "lift")["phase_velocity"], {})
        for value in (
            '{"global_speed":"slow","phase_speeds":{}}',
            '{"global_speed":"slow","phase_speeds":{},"unsupported_request_constraints":["jerk"]}',
            '{"global_speed":"slow","phase_speeds":{},"unsupported_request_constraints":["max_jerk","max_jerk"]}',
            '{"global_speed":"slow","phase_speeds":{"transport":"fast"},"unsupported_request_constraints":[]}',
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response_v2(value, "lift")

    def test_v2_coverage_validation_rejects_silent_or_invented_constraints(self):
        request = {
            "task": "lift",
            "requested_manner": {
                "global": {"max_joint_vel": 0.08, "max_jerk": 2.0},
                "phase_velocity": {},
            },
        }
        self.assertEqual(expected_v2_unsupported_constraints(request), ["max_jerk"])
        candidate = parse_response_v2(
            '{"global_speed":"slow","phase_speeds":{},'
            '"unsupported_request_constraints":["max_jerk"]}',
            "lift",
        )
        self.assertIs(validate_v2_coverage(candidate, request), candidate)
        candidate["unsupported_request_constraints"] = []
        with self.assertRaises(ValueError):
            validate_v2_coverage(candidate, request)
        candidate["unsupported_request_constraints"] = ["max_jerk", "stiffness"]
        with self.assertRaises(ValueError):
            validate_v2_coverage(candidate, request)


if __name__ == "__main__":
    unittest.main()
