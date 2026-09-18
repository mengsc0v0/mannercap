import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CAPTURE = load_script("gate2_capture_la4vla_microset.py")
PACK = load_script("gate2_pack_la4vla_microset.py")


class La4vlaMicrosetTests(unittest.TestCase):
    def test_window_starts_include_full_range_without_duplicates(self):
        starts = CAPTURE.evenly_spaced_window_starts(100, 50, 4)
        self.assertEqual(starts, [0, 16, 33, 50])
        self.assertEqual(len(starts), len(set(starts)))

    def test_short_episode_and_invalid_count_fail(self):
        with self.assertRaises(ValueError):
            CAPTURE.evenly_spaced_window_starts(49, 50, 1)
        with self.assertRaises(ValueError):
            CAPTURE.evenly_spaced_window_starts(100, 50, 0)

    def test_staging_validation_requires_exact_state_action_and_horizon(self):
        manifest = {
            "schema": "mannercap-la4vla-microset-staging-v1",
            "sample_count": 1,
            "horizon": 1,
            "windows": [{
                "task_index": 0,
                "prompt": "Lift the cube.",
                "rows": [{"observation_state": [0] * 8, "action": [0] * 7}],
            }],
        }
        PACK.validate_staging(manifest)
        with self.assertRaises(ValueError):
            PACK.validate_staging({**manifest, "sample_count": 2})
        bad = json_clone(manifest)
        bad["windows"][0]["rows"][0]["action"] = [0] * 6
        with self.assertRaises(ValueError):
            PACK.validate_staging(bad)

    def test_staging_validation_rejects_task_index_prompt_collision(self):
        manifest = {
            "schema": "mannercap-la4vla-microset-staging-v1",
            "sample_count": 2,
            "horizon": 1,
            "windows": [
                {
                    "task_index": 0,
                    "prompt": prompt,
                    "rows": [{"observation_state": [0] * 8, "action": [0] * 7}],
                }
                for prompt in ("Lift the cube.", "Lift the cube quickly.")
            ],
        }
        with self.assertRaisesRegex(ValueError, "multiple prompts"):
            PACK.validate_staging(manifest)

    def test_episode_metadata_uses_manner_prompt_and_rejects_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = {
                "status": "executed",
                "task": "lift",
                "task_completed": False,
                "episode_id": "lift/slow",
                "raw_language": "Lift the cube slowly.",
                "manner_tag": "slow",
            }
            (directory / "result.json").write_text(json.dumps(result))
            (directory / "transitions.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "did not complete"):
                CAPTURE.episode_metadata(directory)
            metadata = CAPTURE.episode_metadata(
                directory, include_task_failures=True
            )
            self.assertEqual(metadata["prompt"], "Lift the cube slowly.")
            self.assertEqual(metadata["file_stem"], "lift_slow")
            self.assertFalse(metadata["task_completed"])

    def test_array_statistics_uses_population_std_and_quantiles(self):
        stats = PACK.array_statistics([[0.0, 2.0], [2.0, 4.0]])
        self.assertEqual(stats["mean"], [1.0, 3.0])
        self.assertEqual(stats["std"], [1.0, 1.0])
        self.assertEqual(stats["min"], [0.0, 2.0])
        self.assertEqual(stats["max"], [2.0, 4.0])
        self.assertEqual(stats["q01"], [0.02, 2.02])
        self.assertEqual(stats["q99"], [1.98, 3.98])


def json_clone(value):
    import json
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
