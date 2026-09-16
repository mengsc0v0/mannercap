"""Contract tests for the versioned Gate 1 task/request space."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from synthesis_v2.request_manifest import DEVELOPMENT_SPLIT, PairType
from synthesis_v2.task_space import (
    Applicability,
    CalibrationStatus,
    EventSpec,
    PairCell,
    PairConstraint,
    RequestCell,
    TaskSpace,
    TaskSpaceError,
    TaskStatus,
    build_default_task_space,
    build_request_cell,
    write_task_space,
)


class TaskSpaceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.default = build_default_task_space()

    def _cell(
        self,
        *,
        task_id: str = "lift",
        stages: tuple[str, ...] = ("approach",),
        property_config_id: str | None = "property-neutral",
        initial_state_id: str | None = "reset-0",
        split: str = DEVELOPMENT_SPLIT,
        **kwargs: object,
    ) -> RequestCell:
        return build_request_cell(
            task_id,
            stage_instance_ids=stages,
            property_config_id=property_config_id,
            initial_state_id=initial_state_id,
            split=split,
            **kwargs,
        )

    def test_schema_and_development_scope_are_explicit(self) -> None:
        self.assertEqual(self.default.to_dict()["schema_version"], "synthesis_v2.task_space.v1")
        self.assertEqual(self.default.status, "draft")
        self.assertEqual({task.task_id for task in self.default.development_tasks}, {"lift", "stack"})
        self.assertEqual(
            {task.task_id for task in self.default.pending_tasks},
            {"wipe"},
        )
        wipe = self.default.get_task("wipe")
        self.assertEqual(wipe.status, TaskStatus.CANDIDATE.value)
        self.assertEqual(wipe.calibration_status, CalibrationStatus.PENDING_CALIBRATION.value)
        self.assertEqual(len(wipe.phases), 0)

    def test_phases_have_entry_exit_events_and_declared_controls(self) -> None:
        for task in self.default.development_tasks:
            self.assertGreater(len(task.phases), 0)
            object_roles = {obj.role for obj in task.objects}
            controls = set(task.control_names())
            self.assertTrue(controls)
            for phase in task.phases:
                self.assertTrue(phase.stage_instance_id)
                self.assertTrue(phase.entry_event.name)
                self.assertTrue(phase.entry_event.source)
                self.assertTrue(phase.exit_event.name)
                self.assertTrue(phase.exit_event.source)
                self.assertTrue(set(phase.object_roles) <= object_roles)
                self.assertTrue(set(phase.allowed_control_channels) <= controls)

    def test_properties_have_source_unit_and_no_guessed_values(self) -> None:
        for task in self.default.development_tasks:
            self.assertTrue(task.objects)
            for obj in task.objects:
                self.assertTrue(obj.role)
                self.assertTrue(obj.properties)
                for prop in obj.properties:
                    self.assertTrue(prop.property_name)
                    self.assertTrue(prop.source)
                    self.assertTrue(prop.unit)
                    self.assertIsNone(prop.value)
            for channel in task.control_channels:
                self.assertTrue(channel.name)
                self.assertTrue(channel.source)
                self.assertTrue(channel.unit)
                self.assertTrue(channel.domain_ref)
                self.assertIsNone(channel.lower_bound)
                self.assertIsNone(channel.upper_bound)

    def test_pair_constraints_explicitly_cover_all_three_types(self) -> None:
        constraints = {constraint.pair_type: constraint for constraint in self.default.pair_constraints}
        self.assertEqual(
            set(constraints),
            {"modifier_word", "physical_attribute", "stage_scope"},
        )
        self.assertEqual(len(self.default.pair_constraints), 3)
        for constraint in constraints.values():
            self.assertTrue(constraint.same_fields)
            self.assertTrue(constraint.varying_fields)
            self.assertTrue(constraint.same_split)
            self.assertIn("split", constraint.same_fields)
        self.assertTrue(self.default.split_policy.pair_members_same_split)
        self.assertEqual(self.default.split_policy.development_split, DEVELOPMENT_SPLIT)

    def test_applicable_cell_requires_explicit_stage_property_and_initial_axes(self) -> None:
        cell = self._cell(task_id="lift", stages=("approach", "lift"))
        space = TaskSpace(tasks=self.default.tasks, request_cells=(cell,))
        self.assertEqual(space.request_cells[0].cell_id, cell.cell_id)
        self.assertEqual(space.request_cells[0].split, DEVELOPMENT_SPLIT)

        missing_property = self._cell(property_config_id=None)
        with self.assertRaisesRegex(TaskSpaceError, "property_config_id and initial_state_id"):
            TaskSpace(tasks=self.default.tasks, request_cells=(missing_property,))

        missing_stage = self._cell(stages=())
        with self.assertRaisesRegex(TaskSpaceError, "at least one stage"):
            TaskSpace(tasks=self.default.tasks, request_cells=(missing_stage,))

    def test_request_cell_id_is_stable_for_mapping_order(self) -> None:
        left = RequestCell(
            task_id="lift",
            stage_instance_ids=("approach",),
            property_config_id="p0",
            initial_state_id="s0",
            metadata={"z": 1, "a": {"b": 2, "a": 1}},
        )
        right = RequestCell(
            task_id="lift",
            stage_instance_ids=("approach",),
            property_config_id="p0",
            initial_state_id="s0",
            metadata={"a": {"a": 1, "b": 2}, "z": 1},
        )
        self.assertEqual(left.cell_id, right.cell_id)
        self.assertEqual(left.payload(), right.payload())
        self.assertTrue(left.cell_id.startswith("cell_"))

    def test_non_applicable_cell_is_explicit_and_wipe_is_pending(self) -> None:
        pending = self.default.request_cells[0]
        self.assertEqual(pending.task_id, "wipe")
        self.assertEqual(pending.applicability, Applicability.PENDING_CALIBRATION.value)
        self.assertTrue(pending.reason)
        self.assertEqual(self.default.inapplicable_cells, (pending,))

        with self.assertRaisesRegex(TaskSpaceError, "requires reason"):
            RequestCell(
                task_id="wipe",
                applicability=Applicability.INAPPLICABLE,
                reason=None,
            )

        applicable_wipe = self._cell(task_id="wipe", stages=("approach",))
        with self.assertRaisesRegex(TaskSpaceError, "candidate/disabled task"):
            TaskSpace(tasks=self.default.tasks, request_cells=(applicable_wipe,))

    def test_candidate_task_cannot_be_promoted_by_populating_phases(self) -> None:
        wipe = self.default.get_task("wipe")
        promoted = replace(
            wipe,
            phases=(
                # A candidate must not acquire an executable phase by accident.
                {
                    "stage_instance_id": "candidate_phase",
                    "name": "Candidate phase",
                    "entry_event": {"name": "start", "source": "pending"},
                    "exit_event": {"name": "end", "source": "pending"},
                },
            ),
        )
        with self.assertRaisesRegex(TaskSpaceError, "candidate task must have pending_calibration status"):
            # The phase itself is accepted as a record; the task remains visibly candidate.
            type(wipe)(
                task_id=promoted.task_id,
                display_name=promoted.display_name,
                status=promoted.status,
                calibration_status=CalibrationStatus.READY_FOR_DEVELOPMENT,
                phases=promoted.phases,
            )

    def test_phase_control_and_object_role_references_are_checked(self) -> None:
        lift = self.default.get_task("lift")
        bad_control = replace(
            lift.phases[0],
            allowed_control_channels=("joint_velocity", "undeclared_control"),
        )
        with self.assertRaisesRegex(TaskSpaceError, "undeclared control channel"):
            TaskSpace(tasks=(replace(lift, phases=(bad_control,) + lift.phases[1:]),))

        bad_role = replace(lift.phases[0], object_roles=("undeclared_role",))
        with self.assertRaisesRegex(TaskSpaceError, "undeclared object role"):
            TaskSpace(tasks=(replace(lift, phases=(bad_role,) + lift.phases[1:]),))

    def test_pair_cells_are_deterministic_and_cannot_cross_split(self) -> None:
        first = self._cell(pair_type=PairType.MODIFIER_WORD)
        second = self._cell(
            property_config_id="property-neutral",
            initial_state_id="reset-0",
            pair_type=PairType.MODIFIER_WORD,
            metadata={"raw_language": "slowly"},
        )
        pair = PairCell(PairType.MODIFIER_WORD, (first.cell_id, second.cell_id), DEVELOPMENT_SPLIT)
        reversed_pair = PairCell(PairType.MODIFIER_WORD, (second.cell_id, first.cell_id), DEVELOPMENT_SPLIT)
        self.assertEqual(pair.pair_id, reversed_pair.pair_id)
        self.assertTrue(pair.pair_id.startswith("paircell_"))
        space = TaskSpace(tasks=self.default.tasks, request_cells=(first, second), pair_cells=(pair,))
        self.assertEqual(space.pair_cells[0].pair_type, PairType.MODIFIER_WORD.value)

        test_cell = self._cell(split="student_test", pair_type=PairType.MODIFIER_WORD)
        cross_split_pair = PairCell(
            PairType.MODIFIER_WORD,
            (first.cell_id, test_cell.cell_id),
            DEVELOPMENT_SPLIT,
        )
        with self.assertRaisesRegex(TaskSpaceError, "crosses split"):
            TaskSpace(
                tasks=self.default.tasks,
                request_cells=(first, test_cell),
                pair_cells=(cross_split_pair,),
            )

    def test_pair_cells_require_two_distinct_members_and_same_task(self) -> None:
        first = self._cell(task_id="lift")
        with self.assertRaisesRegex(TaskSpaceError, "two distinct"):
            PairCell(PairType.MODIFIER_WORD, (first.cell_id, first.cell_id), DEVELOPMENT_SPLIT)

        stack = self._cell(task_id="stack", stages=("approach",))
        pair = PairCell(PairType.MODIFIER_WORD, (first.cell_id, stack.cell_id), DEVELOPMENT_SPLIT)
        with self.assertRaisesRegex(TaskSpaceError, "one task"):
            TaskSpace(tasks=self.default.tasks, request_cells=(first, stack), pair_cells=(pair,))

    def test_writer_emits_versioned_stable_json(self) -> None:
        cell = self._cell(task_id="stack", stages=("transport", "place"))
        space = TaskSpace(tasks=self.default.tasks, request_cells=(cell,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_space.json"
            write_task_space(path, space)
            first = path.read_text(encoding="utf-8")
            write_task_space(path, space)
            self.assertEqual(first, path.read_text(encoding="utf-8"))
            decoded = json.loads(first)
        self.assertEqual(decoded["schema_version"], "synthesis_v2.task_space.v1")
        self.assertEqual(decoded["status"], "draft")
        self.assertEqual(decoded["request_cells"][0]["cell_id"], cell.cell_id)
        self.assertEqual(decoded["split_policy"]["development_split"], DEVELOPMENT_SPLIT)


if __name__ == "__main__":
    unittest.main()
