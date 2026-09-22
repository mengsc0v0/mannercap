import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "gate2_summarize_d0_development.py"
SPEC = importlib.util.spec_from_file_location(PATH.stem, PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(task, seed, tag, overall, approach, descend, lift):
    return {
        "task": task,
        "seed": seed,
        "manner_tag": tag,
        "task_completed": True,
        "api_calls": 0,
        "descriptive_metrics": {"mean_joint_speed": overall},
        "descriptive_phase_metrics": {
            "approach": {"mean_joint_speed_rad_s": approach},
            "descend": {"mean_joint_speed_rad_s": descend},
            "lift": {"mean_joint_speed_rad_s": lift},
        },
    }


class D0DevelopmentSummaryTests(unittest.TestCase):
    def test_paired_direction_summary(self):
        rows = [
            result("lift", 1, "neutral", 1.0, 1.0, 1.0, 1.0),
            result("lift", 1, "slow", 0.5, 0.5, 0.5, 0.5),
            result("lift", 1, "fast", 1.5, 1.5, 1.5, 1.5),
            result("lift", 1, "fast_approach_slow_contact", 0.8, 1.2, 0.4, 0.4),
        ]
        report = MODULE.summarize(rows)
        self.assertTrue(report["all_direction_checks_pass"])
        self.assertEqual(report["episode_count"], 4)
        self.assertEqual(
            report["paired_ratio_summary"]["fast_overall_vs_neutral"]["mean"],
            1.5,
        )
        self.assertFalse(report["formal_thresholds_frozen"])

    def test_empty_numeric_summary_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.numeric_summary([])


if __name__ == "__main__":
    unittest.main()
