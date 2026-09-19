import unittest

from scripts.gate2_e0_observe import plan


class ObservationPlanTests(unittest.TestCase):
    def test_same_requests_and_bounded_attempts_for_every_method(self):
        value = plan(50, 3)
        self.assertEqual(len(value["requests"]), 8)
        self.assertEqual(len(value["candidates"]), 96)
        for method in ("G0", "G1", "G2", "G3"):
            candidates = [c for c in value["candidates"] if c["method"] == method]
            self.assertEqual(len(candidates), 24)
            self.assertEqual({c["index"] for c in candidates}, {1, 2, 3})
            self.assertEqual({c["request"]["request_id"] for c in candidates},
                             {r["request_id"] for r in value["requests"]})
        self.assertEqual(value["M"]["charged_attempts"], 0)
        self.assertFalse(value["promotion_allowed"])
        self.assertEqual(value, plan(50, 3))


if __name__ == "__main__":
    unittest.main()
