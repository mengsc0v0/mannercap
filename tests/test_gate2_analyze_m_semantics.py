import unittest

from scripts.gate2_analyze_m_semantics import SemanticAnalysisError, analyze


def _record(requested, candidate):
    return {
        "valid": True,
        "manner_qualification": "unknown",
        "physical_interpretation": "descriptive_execution_semantics_only",
        "phases": {},
        "request_semantic_diff": {
            "requested_spec": requested,
            "candidate_spec": candidate,
        },
    }


class MSemanticAnalysisTests(unittest.TestCase):
    def test_eight_cached_records_are_classified_without_physical_verdict(self):
        records = {
            f"req-{index}-M-1": _record(
                {"global": {}, "phase_velocity": {}},
                {"global": {}, "phase_velocity": {"approach": {}}}
                if index == 0
                else {"global": {"max_joint_vel": 0.08}, "phase_velocity": {}},
            )
            for index in range(8)
        }
        result = analyze(
            {
                "schema": "mannercap-e0-reconciliation-v1",
                "valid": True,
                "plan_sha256": "plan",
                "phase_semantic_audits": records,
            }
        )
        self.assertEqual(result["cached_m_records"], 8)
        self.assertEqual(result["overall_counts"]["representation_difference"], 1)
        self.assertEqual(result["overall_counts"]["substantive_conflict"], 7)
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["manner_qualification"], "unknown")
        self.assertIn("no physical failure inferred", result["physical_interpretation"])

    def test_analysis_requires_exact_cached_m_cardinality(self):
        with self.assertRaises(SemanticAnalysisError):
            analyze(
                {
                    "schema": "mannercap-e0-reconciliation-v1",
                    "phase_semantic_audits": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
