import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gate2_la4vla_osc_roundtrip.py"
SPEC = importlib.util.spec_from_file_location("gate2_la4vla_osc_roundtrip", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class La4vlaOscRoundTripTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_ends_and_are_unique(self):
        self.assertEqual(MODULE.evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(MODULE.evenly_spaced_indices(2, 30), [0, 1])

    def test_allocate_samples_is_fair_and_respects_capacity(self):
        self.assertEqual(MODULE.allocate_samples([100, 200], 30), [15, 15])
        self.assertEqual(MODULE.allocate_samples([1, 5], 5), [1, 4])

    def test_summary_preserves_failures_and_bias(self):
        rows = [
            {
                "status": "executed",
                "position_error_vector_m": [0.1, -0.2, 0.3],
                "position_error_norm_m": 0.4,
                "orientation_error_rad": 0.05,
                "serialization_round_trip_exact": True,
                "phase_program_marker": "move",
                "adapter": {"clipped": False},
            },
            {
                "status": "adapter_error",
                "phase_program_marker": "grip",
                "error_code": "command_out_of_range",
            },
        ]
        summary = MODULE.summarize_samples(rows)
        self.assertEqual(summary["physical_controller_steps"], 1)
        self.assertEqual(summary["adapter_errors"], 1)
        self.assertEqual(summary["position_error_bias_m"], [0.1, -0.2, 0.3])
        self.assertEqual(summary["phase_program_markers_covered"], ["grip", "move"])

    def test_controller_reinitialization_uses_restored_joint_state(self):
        class Controller:
            def __init__(self):
                self.joint_pos = None
                self.updated = []
                self.initial = None

            def update(self, force=False):
                self.updated.append(force)
                self.joint_pos = [1.0, 2.0]

            def update_initial_joints(self, value):
                self.initial = list(value)

        controller = Controller()
        env = SimpleNamespace(
            robosuite_env=SimpleNamespace(
                robots=[SimpleNamespace(
                    composite_controller=SimpleNamespace(
                        part_controllers={"right": controller}
                    )
                )]
            )
        )
        initialized = MODULE._reinitialize_controllers_at_restored_state(env)
        self.assertEqual(initialized, ["robot0.right"])
        self.assertEqual(controller.updated, [True])
        self.assertEqual(controller.initial, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
