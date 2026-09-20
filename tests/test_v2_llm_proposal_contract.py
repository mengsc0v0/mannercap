import unittest

from synthesis_v2.llm_proposal_contract import build_prompt, parse_response


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


if __name__ == "__main__":
    unittest.main()
