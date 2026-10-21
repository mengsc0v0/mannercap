import os
import unittest

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from synthesis_v2.native_osc_env import NativeSession
from synthesis_v2.physical_context import ContextSession, PhysicalContext


class TestPhysicalContext(unittest.TestCase):
    def test_noop_is_identical(self):
        a = NativeSession("lift", 3, render=False)
        b = ContextSession("lift", 3, PhysicalContext(), render=False)
        oa, ob = a.reset(), b.reset()
        self.assertTrue(np.array_equal(oa["cube_pos"], ob["cube_pos"]))
        act = np.zeros(7)
        for _ in range(5):
            oa, _ = a.step(act)
            ob, _ = b.step(act)
        self.assertTrue(np.array_equal(oa["cube_pos"], ob["cube_pos"]))
        a.close(); b.close()

    def test_overrides_are_applied(self):
        s = ContextSession("lift", 3, PhysicalContext(mass_kg=0.5, friction=0.2, size_scale=1.2), render=False)
        obs = s.reset()
        rec = s.context_record
        self.assertTrue(rec["applied"])
        self.assertAlmostEqual(rec["mass_kg"], 0.5)
        self.assertAlmostEqual(rec["friction"], 0.2)
        m, _ = s._mj()
        _, geoms = s._target_geoms()
        self.assertEqual(int(m.geom_priority[geoms[0]]), 1)
        base = NativeSession("lift", 3, render=False)
        base.reset()
        mb, _ = base._mj()
        self.assertAlmostEqual(float(m.geom_size[geoms[0]][2]) / float(mb.geom_size[geoms[0]][2]), 1.2)
        # observation reflects the lifted centre of the larger object (resting on the table)
        self.assertGreater(float(obs["cube_pos"][2]), float(base.obs["cube_pos"][2]))
        s.close(); base.close()

    def test_bad_values_rejected(self):
        with self.assertRaises(ValueError):
            ContextSession("lift", 1, PhysicalContext(mass_kg=50.0), render=False)
        with self.assertRaises(ValueError):
            ContextSession("wipe", 1, PhysicalContext(mass_kg=0.2), render=False)


if __name__ == "__main__":
    unittest.main()
