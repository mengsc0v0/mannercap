"""Contract tests for the first request-manifest orchestration pass."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from synthesis_v2.request_manifest import (
    DEVELOPMENT_SPLIT,
    ManifestError,
    PairType,
    RequestDraft,
    build_development_manifest,
    build_request_manifest,
    make_pair,
    pair_id_for,
    request_id_for,
    write_request_manifest,
)


def _draft(
    *,
    language: str = "place the red cube",
    constraints: object | None = None,
    initial_state: object | None = None,
    physical_config: object | None = None,
    stage_scope: object | None = None,
    split: str = DEVELOPMENT_SPLIT,
    geometry_key: str | None = "geometry-01",
    manner_key: str | None = None,
    metadata: object | None = None,
) -> RequestDraft:
    return RequestDraft(
        task="lift",
        scene_id="scene-01",
        initial_state={"snapshot": "s0"} if initial_state is None else initial_state,
        target_object="red cube",
        physical_config={"config_ref": "calibration-pending"}
        if physical_config is None
        else physical_config,
        raw_language=language,
        requested_constraints={} if constraints is None else constraints,
        stage_scope=[{"stage_instance_id": "lift"}]
        if stage_scope is None
        else stage_scope,
        allowed_control_channels=("joint_velocity", "gripper"),
        budget={"max_attempts": 3},
        split=split,
        geometry_key=geometry_key,
        manner_key=manner_key,
        metadata={} if metadata is None else metadata,
    )


class RequestManifestContractTests(unittest.TestCase):
    def test_ids_are_stable_under_mapping_order_and_pair_member_order(self):
        first = _draft(
            constraints={"speed": {"max": "raw-a", "min": "raw-b"}},
            initial_state={"b": 2, "a": 1},
        )
        second = _draft(
            language="place the red cube quickly",
            constraints={"speed": {"min": "raw-b", "max": "raw-a"}},
            initial_state={"a": 1, "b": 2},
        )

        self.assertEqual(
            request_id_for(first),
            request_id_for(
                {
                    "task": "lift",
                    "scene_id": "scene-01",
                    "initial_state": {"a": 1, "b": 2},
                    "target_object": "red cube",
                    "physical_config": {"config_ref": "calibration-pending"},
                    "raw_language": "place the red cube",
                    "requested_constraints": {
                        "speed": {"max": "raw-a", "min": "raw-b"}
                    },
                    "stage_scope": [{"stage_instance_id": "lift"}],
                    "allowed_control_channels": ["gripper", "joint_velocity"],
                    "budget": {"max_attempts": 3},
                    "split": "development",
                    "geometry_key": "geometry-01",
                    "manner_key": None,
                    "attribute_intervention": None,
                    "metadata": {},
                }
            ),
        )

        pair_one = make_pair(first, second, PairType.MODIFIER_WORD)
        pair_two = make_pair(second, first, "modifier_word")
        records_one = build_development_manifest([first, second], [pair_one])
        records_two = build_development_manifest([second, first], [pair_two])

        self.assertEqual(
            [record["request_id"] for record in records_one],
            [record["request_id"] for record in records_two],
        )
        self.assertEqual(records_one[0]["pair_id"], records_two[0]["pair_id"])
        self.assertTrue(records_one[0]["pair_id"].startswith("pair_"))
        self.assertEqual(
            pair_id_for(PairType.MODIFIER_WORD, [first.request_id, second.request_id]),
            pair_id_for(PairType.MODIFIER_WORD, [second.request_id, first.request_id]),
        )

    def test_development_manifest_is_explicit_and_default_tasks_are_lift_and_stack(self):
        lift = _draft()
        stack = RequestDraft(
            task="stack",
            scene_id="scene-02",
            initial_state={"snapshot": "stack-s0"},
            target_object="red cube",
            raw_language="stack the red cube",
        )
        records = build_development_manifest([lift, stack])

        self.assertEqual({record["task"] for record in records}, {"lift", "stack"})
        self.assertEqual({record["split"] for record in records}, {DEVELOPMENT_SPLIT})
        self.assertTrue(all(record["request_status"] == "planned" for record in records))
        self.assertTrue(all(record["manifest_status"] == "draft" for record in records))

        wipe = RequestDraft(
            task="wipe",
            scene_id="scene-wipe",
            initial_state={"snapshot": "un-calibrated"},
            target_object="surface",
            raw_language="wipe the surface",
        )
        with self.assertRaisesRegex(ManifestError, "calibrated request space"):
            build_development_manifest([wipe])

    def test_pair_members_cannot_cross_splits(self):
        development = _draft(split="development")
        generator_test = _draft(
            language="place the red cube quickly", split="generator_test"
        )

        with self.assertRaisesRegex(ManifestError, "share a split"):
            build_request_manifest(
                [development, generator_test],
                [make_pair(development, generator_test, "modifier_word")],
            )

    def test_three_pair_types_are_structurally_checked_and_recorded(self):
        # Type 1: same initial state, modifier wording/declared constraint changes.
        modifier_a = _draft(
            language="place the red cube slowly", constraints={"manner": "slow"}
        )
        modifier_b = _draft(
            language="place the red cube quickly", constraints={"manner": "fast"}
        )
        modifier_records = build_development_manifest(
            [modifier_a, modifier_b],
            [make_pair(modifier_a, modifier_b, "modifier_word")],
        )
        self.assertEqual(
            {record["pair_type"] for record in modifier_records}, {"modifier_word"}
        )
        self.assertTrue(all(record["pair_member_ids"] for record in modifier_records))

        # Type 2: neutral language and common geometry are fixed; the opaque
        # full physical state/configuration changes.  No physical value is interpreted.
        attribute_a = _draft(
            initial_state={"geometry": "same", "state_hash_input": "a"},
            physical_config={"calibration_ref": "config-a"},
            language="place the red cube",
        )
        attribute_b = _draft(
            initial_state={"geometry": "same", "state_hash_input": "b"},
            physical_config={"calibration_ref": "config-b"},
            language="place the red cube",
        )
        attribute_records = build_development_manifest(
            [attribute_a, attribute_b],
            [make_pair(attribute_a, attribute_b, PairType.PHYSICAL_ATTRIBUTE)],
        )
        self.assertEqual(
            {record["pair_type"] for record in attribute_records}, {"physical_attribute"}
        )
        self.assertEqual(
            {record["pair_metadata"]["common_geometry_id"] for record in attribute_records},
            {"geometry-01"},
        )
        self.assertIn("attribute_intervention", attribute_records[0]["pair_metadata"])
        self.assertNotEqual(
            attribute_records[0]["physical_state_hash"],
            attribute_records[1]["physical_state_hash"],
        )

        # Type 3: same task/manner and initial state, only the active stage scope
        # (and its preserved language/constraint declaration) changes.
        stage_a = _draft(
            language="approach slowly, then lift normally",
            constraints={"active_stage": "approach", "manner": "slow"},
            stage_scope=[{"stage_instance_id": "approach"}],
            manner_key="slow",
        )
        stage_b = _draft(
            language="approach normally, then lift slowly",
            constraints={"active_stage": "lift", "manner": "slow"},
            stage_scope=[{"stage_instance_id": "lift"}],
            manner_key="slow",
        )
        stage_records = build_development_manifest(
            [stage_a, stage_b], [make_pair(stage_a, stage_b, PairType.STAGE_SCOPE)]
        )
        self.assertEqual(
            {record["pair_type"] for record in stage_records}, {"stage_scope"}
        )
        self.assertNotEqual(stage_records[0]["stage_scope"], stage_records[1]["stage_scope"])

    def test_original_constraints_are_copied_and_never_clipped_or_rewritten(self):
        raw_constraints = {
            "speed": {"max": "as-requested"},
            "scope": [{"stage_instance_id": "lift", "raw_text": "慢慢"}],
            "unresolved": True,
        }
        original = copy.deepcopy(raw_constraints)
        draft = _draft(constraints=raw_constraints)
        raw_constraints["speed"]["max"] = "caller-mutated"
        raw_constraints["extra"] = "caller-mutated"

        record = build_development_manifest([draft])[0]
        self.assertEqual(record["requested_constraints"], original)
        self.assertEqual(draft.raw_constraints, original)
        self.assertTrue(record["requested_constraints_hash"])
        self.assertNotIn("clipped_constraints", record)
        self.assertNotIn("threshold", record["requested_constraints"])

    def test_manifest_keeps_unpaired_requests_and_writes_one_json_object_per_line(self):
        records = build_development_manifest(
            [_draft(), _draft(language="place the red cube fast")]
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["pair_id"] is None for record in records))

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "generation" / "request_manifest.jsonl"
            write_request_manifest(output, records)
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(
            all(json.loads(line)["request_status"] == "planned" for line in lines)
        )

    def test_invalid_modifier_pair_cannot_change_initial_state(self):
        first = _draft(language="place slowly")
        second = _draft(
            language="place quickly", initial_state={"snapshot": "different"}
        )
        with self.assertRaisesRegex(ManifestError, "may vary only raw_language"):
            build_development_manifest(
                [first, second], [make_pair(first, second, "modifier")]
            )


if __name__ == "__main__":  # pragma: no cover - convenient local invocation
    unittest.main()
