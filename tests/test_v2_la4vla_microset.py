import importlib.util
from pathlib import Path
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
            "windows": [{"rows": [{"observation_state": [0] * 8, "action": [0] * 7}]}],
        }
        PACK.validate_staging(manifest)
        with self.assertRaises(ValueError):
            PACK.validate_staging({**manifest, "sample_count": 2})
        bad = json_clone(manifest)
        bad["windows"][0]["rows"][0]["action"] = [0] * 6
        with self.assertRaises(ValueError):
            PACK.validate_staging(bad)


def json_clone(value):
    import json
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
