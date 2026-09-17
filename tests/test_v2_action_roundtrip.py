"""Pure tests for the Gate 1 action export/import preflight."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synthesis_v2.action_roundtrip import (
    ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION,
    ROUND_TRIP_RECORD_SCHEMA_VERSION,
    build_round_trip_manifest,
    compare_round_trip_records,
    export_round_trip_records,
    import_round_trip_records,
    inspect_action_record,
    read_json_records,
    read_round_trip_manifest,
)
from synthesis_v2.action_interface import InterfaceId


def _raw_row(index: int, *, with_dt: bool = True) -> dict:
    state = [0.01 * index, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, 1.0]
    raw = [0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -1.0]
    row = {
        "observation.state": state,
        "next_observation.state": state,
        "observation.state.raw_joint_gripper": state,
        "next_observation.state.raw_joint_gripper": state,
        "action": raw,
        "action.raw_joint_gripper": list(raw),
        "lerobot_action_candidate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "timestamp": 0.05 * index,
        "frame_index": index,
        "sim_step_index": index + 1,
        "episode_index": 0,
        "task_index": 7,
        "nested": {"source": "fixture", "values": [index, True, None]},
    }
    if with_dt:
        row["dt"] = 0.05
    return row


class ActionRoundTripTests(unittest.TestCase):
    def test_classifier_keeps_8d_low_level_and_7d_eef_candidate_distinct(self) -> None:
        raw_info = inspect_action_record(_raw_row(0))
        self.assertEqual(raw_info["interface_id"], InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value)
        self.assertEqual(raw_info["fields"]["action.raw_joint_gripper"]["dimension"], 8)
        self.assertFalse(raw_info["verified"])
        self.assertEqual(
            raw_info["fields"]["lerobot_action_candidate"]["interface_id"],
            InterfaceId.LEROBOT_EXPORT_CANDIDATE.value,
        )
        candidate_only = {"action": _raw_row(0)["lerobot_action_candidate"]}
        candidate_info = inspect_action_record(candidate_only)
        self.assertEqual(candidate_info["interface_id"], InterfaceId.LEROBOT_EXPORT_CANDIDATE.value)
        self.assertEqual(candidate_info["semantic"], "eef_position_axis_angle_state_difference_plus_raw_gripper")

    def test_jsonl_export_import_preserves_every_field(self) -> None:
        records = [_raw_row(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.records.jsonl"
            returned = export_round_trip_records(records, path, source_path="steps.jsonl")
            self.assertEqual(returned, path)
            imported = import_round_trip_records(path)
            report = compare_round_trip_records(records, imported)
            self.assertTrue(report.valid)
            self.assertEqual(report.source_count, 3)
            self.assertEqual(report.imported_count, 3)
            self.assertGreater(report.checked_field_count, 3)
            envelope = import_round_trip_records(path, include_metadata=True)
            self.assertEqual(envelope[0]["schema_version"], ROUND_TRIP_RECORD_SCHEMA_VERSION)
            self.assertEqual(envelope[0]["source_record_sha256"], envelope[0]["source_record_sha256"])

    def test_field_comparison_reports_nested_mutation(self) -> None:
        records = [_raw_row(0), _raw_row(1)]
        altered = json.loads(json.dumps(records))
        altered[1]["nested"]["values"][0] = 99
        report = compare_round_trip_records(records, altered)
        self.assertFalse(report.valid)
        self.assertTrue(report.has_code("value_mismatch"))
        self.assertEqual(report.mismatches[0].field_path, "nested.values[0]")

    def test_manifest_defaults_to_30_without_duplicating_short_input(self) -> None:
        records = [_raw_row(i, with_dt=(i % 2 == 0)) for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "steps.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            output = Path(directory) / "out"
            manifest = build_round_trip_manifest(source, output)
            self.assertEqual(manifest["schema_version"], ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION)
            self.assertEqual(manifest["requested_count"], 30)
            self.assertEqual(manifest["selected_count"], 4)
            self.assertFalse(manifest["count_satisfied"])
            self.assertEqual(manifest["status"], "preflight_incomplete")
            self.assertEqual(manifest["field_consistency"]["mismatch_count"], 0)
            self.assertEqual(manifest["timing"]["missing_dt_count"], 2)
            self.assertEqual(manifest["controller_execution"]["status"], "pending")
            self.assertEqual(manifest["controller_execution"]["decision"], "unknown")
            self.assertFalse(manifest["controller_execution"]["replay_verified"])
            self.assertEqual(len(import_round_trip_records(output / "action_round_trip_records.jsonl")), 4)
            loaded = read_round_trip_manifest(output / "action_round_trip_manifest.json")
            self.assertEqual(loaded["schema_version"], ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION)

    def test_manifest_can_select_exactly_30_real_rows(self) -> None:
        records = [_raw_row(i) for i in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "steps.jsonl"
            source.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                encoding="utf-8",
            )
            manifest = build_round_trip_manifest(source, Path(directory) / "manifest.json", count=30)
            self.assertTrue(manifest.preflight_passed)
            self.assertEqual(manifest["selected_count"], 30)
            self.assertEqual(manifest["field_consistency"]["checked_record_count"], 30)
            self.assertEqual(manifest["controller_status"], "pending")
            self.assertEqual(manifest["controller_decision"], "unknown")
            self.assertEqual(read_json_records(source), records)

    def test_stagecap_manifest_rows_expand_steps_without_injecting_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = root / "trial" / "steps.jsonl"
            steps.parent.mkdir()
            steps.write_text(json.dumps(_raw_row(0)) + "\n", encoding="utf-8")
            source_manifest = root / "source_manifest.jsonl"
            source_manifest.write_text(
                json.dumps({"trial_dir": "trial", "steps_path": "trial/steps.jsonl", "quality_tier": "gold"}) + "\n",
                encoding="utf-8",
            )
            manifest = build_round_trip_manifest(source_manifest, root / "out", count=1)
            self.assertEqual(manifest["source"]["format"], "stagecap_manifest+jsonl")
            self.assertEqual(manifest["source"]["manifest_rows"], 1)
            self.assertEqual(manifest["selected_count"], 1)
            imported = import_round_trip_records(root / "out" / "action_round_trip_records.jsonl")
            self.assertNotIn("quality_tier", imported[0])
            self.assertIn("action.raw_joint_gripper", imported[0])

    def test_parquet_is_explicitly_rejected_by_stdlib_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            path.write_bytes(b"PAR1")
            with self.assertRaises(Exception) as caught:
                read_json_records(path)
            self.assertEqual(getattr(caught.exception, "code", None), "unsupported_format")

    def test_bundle_hash_tampering_is_rejected(self) -> None:
        records = [_raw_row(0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            export_round_trip_records(records, path)
            entry = json.loads(path.read_text(encoding="utf-8"))
            entry["record"]["action"][0] = 123.0
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with self.assertRaises(Exception) as caught:
                import_round_trip_records(path)
            self.assertEqual(getattr(caught.exception, "code", None), "record_hash_mismatch")


if __name__ == "__main__":
    unittest.main()
