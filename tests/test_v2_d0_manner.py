from __future__ import annotations

import unittest
import contextlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile

from synthesis_v2.d0_manner import audit_results, build_plan, validate_plan


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate2_capture_manner_d0", ROOT / "scripts" / "gate2_capture_manner_d0.py"
)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class D0MannerTests(unittest.TestCase):
    def test_plan_crosses_tasks_seeds_and_student_expressible_tags(self) -> None:
        plan = build_plan(tasks=("lift", "stack"), seeds=(42, 43))
        validate_plan(plan)
        self.assertEqual(plan["episode_count"], 16)
        self.assertEqual(plan["api_budget"], 0)
        self.assertEqual(
            plan["manner_tags"],
            ["neutral", "slow", "fast", "fast_approach_slow_contact"],
        )
        self.assertIn("gentle", plan["excluded_conditions"])
        self.assertTrue(all(row["api_calls"] == 0 for row in plan["episodes"]))
        self.assertEqual(len({row["pair_id"] for row in plan["episodes"]}), 4)

    def test_plan_rejects_unknown_or_duplicate_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported manner"):
            build_plan(tags=("neutral", "gentle"))
        with self.assertRaisesRegex(ValueError, "unique"):
            build_plan(seeds=(42, 42))

    def test_hash_detects_plan_mutation(self) -> None:
        plan = build_plan(tasks=("lift",), seeds=(42,), tags=("neutral",))
        plan["episodes"][0]["raw_language"] = "changed"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_plan(plan)

    def test_audit_requires_exact_initial_state_within_modifier_pair(self) -> None:
        plan = build_plan(
            tasks=("lift",), seeds=(42,), tags=("neutral", "slow")
        )
        results = []
        for episode in plan["episodes"]:
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "status": "executed",
                    "task_completed": True,
                    "initial_state_sha256": "a" * 64,
                    "compiled_model_sha256": "b" * 64,
                    "api_calls": 0,
                }
            )
        audit = audit_results(plan, results)
        self.assertEqual(audit["executed_count"], 2)
        self.assertEqual(audit["task_success_count"], 2)
        self.assertEqual(audit["same_initial_state_pair_count"], 1)
        self.assertEqual(audit["api_calls"], 0)
        self.assertFalse(audit["promotion_allowed"])

        results[1]["initial_state_sha256"] = "c" * 64
        audit = audit_results(plan, results)
        self.assertEqual(audit["same_initial_state_pair_count"], 0)

    def test_cli_defaults_to_plan_only_and_can_reopen_immutable_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "d0"
            with contextlib.redirect_stdout(StringIO()):
                status = cli.main(
                    [
                        "--output",
                        str(output),
                        "--tasks",
                        "lift",
                        "--seed",
                        "42",
                        "--tags",
                        "neutral",
                        "slow",
                    ]
                )
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            with contextlib.redirect_stdout(StringIO()):
                second_status = cli.main(
                    [
                        "--output",
                        str(output),
                        "--tasks",
                        "lift",
                        "--seed",
                        "42",
                        "--tags",
                        "neutral",
                        "slow",
                    ]
                )
        self.assertEqual(status, 0)
        self.assertEqual(second_status, 0)
        self.assertEqual(plan["episode_count"], 2)
        self.assertEqual(manifest["api_budget"], 0)


if __name__ == "__main__":
    unittest.main()
