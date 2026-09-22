import json
import tempfile
import unittest
from pathlib import Path

from scripts.gate2_build_human_gold_manifest import (
    HIDDEN_EXECUTOR_FIELDS,
    build_manifest,
)


class HumanGoldManifestTest(unittest.TestCase):
    def _episode(self, root: Path, episode_id: str, task: str, manner: str, seed: int) -> None:
        target = root / "episodes" / episode_id
        target.mkdir(parents=True)
        result = {
            "episode_id": episode_id,
            "task": task,
            "seed": seed,
            "manner_tag": manner,
            "raw_language": f"{task} {manner}",
            "requested_manner": {"description": manner},
            "pair_id": f"{task}-{seed}",
            "task_completed": True,
            "formal_manner_status": "unknown",
            "formal_safety_status": "unknown",
            "promotion_allowed": False,
            "descriptive_metrics": {"mean_joint_speed": 1.0},
        }
        (target / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (target / "transitions.jsonl").write_text('{"step": 0}\n', encoding="utf-8")

    def test_stratified_double_annotation_and_blinding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            for task in ("lift", "stack"):
                for manner in ("neutral", "slow"):
                    for seed in range(3):
                        self._episode(run, f"{task}-{manner}-{seed}", task, manner, seed)

            manifest = build_manifest([run], double_per_cell=2, selection_salt="test")
            self.assertEqual(manifest["candidate_count"], 12)
            self.assertEqual(manifest["cell_count"], 4)
            self.assertEqual(manifest["double_annotation_count"], 8)
            self.assertFalse(manifest["formal_thresholds_frozen"])
            self.assertFalse(manifest["promotion_allowed"])

            encoded_records = json.dumps(manifest["records"])
            for hidden in HIDDEN_EXECUTOR_FIELDS:
                self.assertNotIn(f'"{hidden}"', encoded_records)

            modes = [record["annotation_mode"] for record in manifest["records"]]
            self.assertEqual(modes.count("double_independent_then_adjudicate"), 8)
            self.assertEqual(modes.count("single_independent"), 4)

            repeated = build_manifest([run], double_per_cell=2, selection_salt="test")
            self.assertEqual(manifest, repeated)

    def test_duplicate_episode_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            self._episode(first, "duplicate", "lift", "neutral", 1)
            self._episode(second, "duplicate", "lift", "neutral", 1)
            with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
                build_manifest([first, second])


if __name__ == "__main__":
    unittest.main()
