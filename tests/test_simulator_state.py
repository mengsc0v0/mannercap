import random
from types import SimpleNamespace
import unittest

from synthesis_v2.simulator_state import capture_simulator_snapshot, restore_simulator_snapshot


class FakeArray(list):
    @property
    def size(self):
        return len(self)

    def tolist(self):
        return list(self)


class FakeData:
    def __init__(self):
        self.time = 1.25
        for name in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl",
                     "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
                     "userdata", "plugin_state", "eq_active"):
            setattr(self, name, FakeArray([1.0, 2.0]))


class FakeSim:
    def __init__(self):
        self.data = FakeData()

    def forward(self):
        pass


class FakeInner:
    def __init__(self):
        self.sim = FakeSim()
        self.timestep = 3
        self.cur_time = 1.25
        self.done = False
        interpolator = SimpleNamespace(step=2, start=FakeArray([0.1, 0.2]), goal=FakeArray([0.3, 0.4]))
        controller = SimpleNamespace(
            goal_qpos=FakeArray([0.3, 0.4]),
            joint_pos=FakeArray([0.1, 0.2]),
            new_update=False,
            interpolator=interpolator,
        )
        composite = SimpleNamespace(part_controllers={"right": controller})
        self.robots = [SimpleNamespace(torques=FakeArray([1.0, 2.0]), composite_controller=composite)]


class FakeBitGenerator:
    state = {"state": 7}


class FakeRng:
    bit_generator = FakeBitGenerator()


class FakeEnv:
    def __init__(self):
        self.robosuite_env = FakeInner()
        self._step_count = 4
        self._sim_step_count = 5
        self._gripper_fraction = .75
        self._manner_phase = "approach"
        self._current_joints = FakeArray([0.1, 0.2])
        self._rng = FakeRng()


class SnapshotTests(unittest.TestCase):
    def test_capture_and_restore_mutable_state(self):
        random.seed(123)
        env = FakeEnv()
        snapshot = capture_simulator_snapshot(env)
        env.robosuite_env.sim.data.time = 9
        env.robosuite_env.sim.data.qpos[:] = [8, 8]
        env._step_count = 99
        part = env.robosuite_env.robots[0].composite_controller.part_controllers["right"]
        part.goal_qpos[:] = [8, 8]
        part.interpolator.step = 99
        restore_simulator_snapshot(env, snapshot)
        self.assertEqual(env.robosuite_env.sim.data.time, 1.25)
        self.assertEqual(env.robosuite_env.sim.data.qpos, [1.0, 2.0])
        self.assertEqual(env._step_count, 4)
        self.assertEqual(part.goal_qpos, [0.3, 0.4])
        self.assertEqual(part.interpolator.step, 2)
        self.assertTrue(snapshot["controller_object_state_captured"])

    def test_rejects_unknown_schema(self):
        with self.assertRaises(ValueError):
            restore_simulator_snapshot(FakeEnv(), {"schema": "other"})


if __name__ == "__main__":
    unittest.main()
