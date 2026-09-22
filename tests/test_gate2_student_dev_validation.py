import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "gate2_student_dev_validation.py"
SPEC = importlib.util.spec_from_file_location(PATH.stem, PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StudentDevValidationTests(unittest.TestCase):
    def test_canonical_hash_is_order_independent(self):
        left = {"b": [2, 1], "a": {"z": 0}}
        right = {"a": {"z": 0}, "b": [2, 1]}
        self.assertEqual(MODULE.canonical_sha256(left), MODULE.canonical_sha256(right))

    def test_array_statistics_is_train_only_shape(self):
        stats = MODULE.array_statistics([[0.0, 2.0], [2.0, 4.0]], 2)
        self.assertEqual(stats["mean"], [1.0, 3.0])
        self.assertEqual(stats["std"], [1.0, 1.0])
        self.assertEqual(stats["min"], [0.0, 2.0])
        self.assertEqual(stats["max"], [2.0, 4.0])
        self.assertEqual(stats["q01"], [0.02, 2.02])
        self.assertEqual(stats["q99"], [1.98, 3.98])

    def test_finite_vector_rejects_wrong_width_and_nan(self):
        with self.assertRaises(ValueError):
            MODULE._finite_vector([0.0], 2, "state")
        with self.assertRaises(ValueError):
            MODULE._finite_vector([0.0, float("nan")], 2, "state")


if __name__ == "__main__":
    unittest.main()
