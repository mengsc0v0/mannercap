import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.gate2_e0_observe import plan


class ObservationPlanTests(unittest.TestCase):
    def test_cached_five_method_plan_has_equal_request_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cached = root / 'cache.json'
            cached.write_text(json.dumps([
                {'request_id': r['request_id'], 'candidate_preset':
                 {'description': 'test', 'global': {}, 'phase_velocity': {}}}
                for r in plan(50, 1)['requests']]))
            proc = subprocess.run([sys.executable, 'scripts/gate2_e0_observe.py',
                '--output', str(root/'out'), '--attempts', '1',
                '--cached-m-proposals', str(cached), '--include-baselines'],
                capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            generated = json.loads((root/'out'/'plan.json').read_text())
            self.assertEqual(len(generated['candidates']), 40)
            self.assertEqual(generated['phase_semantics'], 'explicit_neutral_v2')
            for method in ('G0', 'G1', 'G2', 'G3', 'M'):
                self.assertEqual({x['request']['request_id'] for x in generated['candidates']
                                  if x['method'] == method},
                                 {x['request_id'] for x in generated['requests']})

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
