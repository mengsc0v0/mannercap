import unittest

from synthesis_v2.semantic_audit import (
    audit_phase_semantics,
    classify_request_candidate_semantics,
)


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


def _neutral_parameters():
    return {
        "v_max": 0.12,
        "a_max": 0.084,
        "jerk_limit": None,
        "phase_v_max": dict(PHASE_V),
        "phase_a_max": dict(PHASE_A),
    }


class SemanticAuditTests(unittest.TestCase):
    def test_explicit_neutral_is_distinct_from_omitted_phase(self):
        actual = {
            "v_max": 0.068,
            "a_max": 0.0476,
            "jerk_limit": None,
            "phase_v_max": {
                "approach": 0.12, "descend": 0.0408, "lift": 0.068,
                "transport": 0.034, "final": 0.034, "retreat": 0.068,
            },
            "phase_a_max": {
                "approach": 0.084, "descend": 0.0238, "lift": 0.0476,
                "transport": 0.01904, "final": 0.01904, "retreat": 0.0476,
            },
        }
        report = audit_phase_semantics(
            {"global": {"max_joint_vel": 0.08}, "phase_velocity": {"approach": {}}},
            actual,
            neutral_parameters=_neutral_parameters(),
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["phases"]["approach"]["scope"], "explicit_neutral")
        self.assertEqual(report["phases"]["lift"]["scope"], "omitted_inherits_global")
        self.assertIn("approach", report["candidate_target_phases"])
        self.assertIn("lift", report["candidate_non_target_phases"])

    def test_omitted_phase_must_follow_global_profile(self):
        actual = _neutral_parameters() | {"v_max": 0.068, "a_max": 0.0476}
        report = audit_phase_semantics(
            {"global": {"max_joint_vel": 0.08}, "phase_velocity": {}},
            actual,
            neutral_parameters=_neutral_parameters(),
        )
        self.assertFalse(report["valid"])
        self.assertTrue(any("approach" in error for error in report["errors"]))
        self.assertEqual(report["manner_qualification"], "unknown")

    def test_target_and_non_target_behavior_are_descriptive_only(self):
        report = audit_phase_semantics(
            {"global": {}, "phase_velocity": {"approach": {"min_joint_vel": 0.18}}},
            _neutral_parameters()
            | {
                "phase_v_max": {**PHASE_V, "approach": 0.198},
                "phase_a_max": {**PHASE_A, "approach": 0.1386},
            },
            neutral_parameters=_neutral_parameters(),
            phase_metrics={"move:approach": {"sample_count": 2, "peak_joint_speed_rad_s": 0.2}},
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["phases"]["approach"]["scope"], "explicit_override")
        self.assertEqual(report["phases"]["approach"]["behavior_observation"]["status"], "observed")
        self.assertEqual(report["phases"]["lift"]["behavior_observation"]["status"], "unknown")
        self.assertEqual(report["physical_interpretation"], "descriptive_execution_semantics_only")

    def test_request_neutral_extra_constraints_are_reported_separately(self):
        report = audit_phase_semantics(
            {"global": {"max_joint_vel": 0.08}, "phase_velocity": {"approach": {}}},
            {
                "v_max": 0.068,
                "a_max": 0.0476,
                "jerk_limit": None,
                "phase_v_max": {
                    "approach": 0.12, "descend": 0.0408, "lift": 0.068,
                    "transport": 0.034, "final": 0.034, "retreat": 0.068,
                },
                "phase_a_max": {
                    "approach": 0.084, "descend": 0.0238, "lift": 0.0476,
                    "transport": 0.01904, "final": 0.01904, "retreat": 0.0476,
                },
            },
            request={
                "requested_manner": {"global": {}, "phase_velocity": {}},
            },
            neutral_parameters=_neutral_parameters(),
        )
        self.assertTrue(report["valid"], report["errors"])
        diff = report["request_semantic_diff"]
        self.assertFalse(diff["candidate_matches_request"])
        self.assertEqual(diff["differences"][0]["kind"], "neutral_extra_constraints")
        self.assertEqual(report["manner_qualification"], "unknown")

    def test_requested_combination_missing_stage_is_reported(self):
        report = audit_phase_semantics(
            {"global": {}, "phase_velocity": {"approach": {"min_joint_vel": 0.18}}},
            _neutral_parameters()
            | {
                "phase_v_max": {**PHASE_V, "approach": 0.198},
                "phase_a_max": {**PHASE_A, "approach": 0.1386},
            },
            request={
                "requested_manner": {
                    "global": {},
                    "phase_velocity": {
                        "approach": {"min_joint_vel": 0.18},
                        "descend": {"max_joint_vel": 0.06},
                    },
                },
            },
            neutral_parameters=_neutral_parameters(),
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertIn(
            "requested_phase_missing",
            {item["kind"] for item in report["request_semantic_diff"]["differences"]},
        )

    def test_semantic_classifier_separates_representation_unknown_and_conflict(self):
        result = classify_request_candidate_semantics(
            {"requested_manner": {"global": {}, "phase_velocity": {}}},
            {"global": {}, "phase_velocity": {"approach": {}}},
        )
        self.assertEqual(result["overall"], "representation_difference")
        self.assertEqual(
            result["differences"][0]["classification"], "representation_difference"
        )

        result = classify_request_candidate_semantics(
            {"requested_manner": {"global": {}, "phase_velocity": {}}},
            {"global": {}, "phase_velocity": {"descend": {}}},
        )
        self.assertEqual(result["overall"], "substantive_conflict")
        self.assertEqual(
            result["differences"][0]["classification"], "substantive_conflict"
        )

        result = classify_request_candidate_semantics(
            {
                "requested_manner": {
                    "global": {"max_joint_vel": 0.08, "max_jerk": 2.0},
                    "phase_velocity": {},
                }
            },
            {"global": {"max_joint_vel": 0.08}, "phase_velocity": {}},
        )
        self.assertEqual(result["overall"], "unknown")
        self.assertEqual(result["differences"][0]["classification"], "unknown")

        result = classify_request_candidate_semantics(
            {"requested_manner": {"global": {}, "phase_velocity": {}}},
            {"global": {"max_joint_vel": 0.08}, "phase_velocity": {}},
        )
        self.assertEqual(result["overall"], "substantive_conflict")
        self.assertEqual(result["differences"][0]["classification"], "substantive_conflict")
        self.assertEqual(result["manner_qualification"], "unknown")


if __name__ == "__main__":
    unittest.main()
