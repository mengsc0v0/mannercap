"""Tests for the Gate 1 development-manifest entry point."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "gate1_build_dev_manifest",
    ROOT / "scripts" / "gate1_build_dev_manifest.py",
)
assert _SPEC is not None and _SPEC.loader is not None
gate1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate1)


class DevelopmentManifestEntryPointTests(unittest.TestCase):
    def _write_initial_state(
        self,
        directory: Path,
        name: str,
        *,
        model_sha256: str = "a" * 64,
        state: object | None = None,
        snapshot: object | None = None,
    ) -> Path:
        payload = {
            "model_sha256": model_sha256,
            "state": {"time_s": 0.0, "qpos": [0.0], "qvel": [0.0]} if state is None else state,
            "snapshot": {"controller": "opaque"} if snapshot is None else snapshot,
            "restoration_verified": False,
        }
        path = directory / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _inputs(self, directory: Path) -> tuple[Path, Path]:
        return (
            self._write_initial_state(directory, "lift-initial_state.json"),
            self._write_initial_state(directory, "stack-initial_state.json"),
        )

    def test_reference_requires_actual_model_hash_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            good = self._write_initial_state(directory, "initial_state.json")
            reference = gate1.load_initial_state_reference(good, "lift")
            self.assertEqual(reference.task_id, "lift")
            self.assertEqual(reference.model_sha256, "a" * 64)
            self.assertEqual(len(reference.initial_state_hash), 64)
            self.assertEqual(len(reference.state_sha256), 64)
            self.assertEqual(len(reference.file_sha256), 64)
            self.assertTrue(reference.initial_state_id.startswith("initial_"))
            self.assertEqual(
                reference.initial_state_id,
                gate1.initial_state_id_for(
                    "lift", reference.model_sha256, reference.initial_state_hash
                ),
            )

            missing_model = directory / "missing-model.json"
            missing_model.write_text(json.dumps({"state": {"qpos": [0.0]}}), encoding="utf-8")
            with self.assertRaisesRegex(gate1.DevelopmentManifestError, "model_sha256"):
                gate1.load_initial_state_reference(missing_model, "lift")

            bad_state = directory / "bad-state.json"
            bad_state.write_text(
                json.dumps({"model_sha256": "a" * 64, "state": [0.0]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate1.DevelopmentManifestError, "under 'state'"):
                gate1.load_initial_state_reference(bad_state, "lift")

            bad_hash = directory / "bad-hash.json"
            bad_hash.write_text(
                json.dumps({"model_sha256": "short", "state": {"qpos": [0.0]}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(gate1.DevelopmentManifestError, "64-character SHA-256"):
                gate1.load_initial_state_reference(bad_hash, "lift")

    def test_initial_state_id_uses_model_and_canonical_content_not_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            first = self._write_initial_state(
                directory,
                "first.json",
                model_sha256="b" * 64,
                state={"qpos": [1.0], "qvel": [0.0]},
                snapshot={"cache": [1, 2]},
            )
            second = directory / "nested" / "second.json"
            second.parent.mkdir()
            second.write_text(
                json.dumps(
                    {
                        "restoration_verified": False,
                        "snapshot": {"cache": [1, 2]},
                        "state": {"qvel": [0.0], "qpos": [1.0]},
                        "model_sha256": "b" * 64,
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            first_ref = gate1.load_initial_state_reference(first, "lift")
            second_ref = gate1.load_initial_state_reference(second, "lift")
            self.assertNotEqual(first_ref.file_sha256, second_ref.file_sha256)
            self.assertEqual(first_ref.initial_state_hash, second_ref.initial_state_hash)
            self.assertEqual(first_ref.state_sha256, second_ref.state_sha256)
            self.assertEqual(first_ref.initial_state_id, second_ref.initial_state_id)

            changed_model = self._write_initial_state(
                directory,
                "changed-model.json",
                model_sha256="c" * 64,
                state={"qpos": [1.0], "qvel": [0.0]},
                snapshot={"cache": [1, 2]},
            )
            changed_state = self._write_initial_state(
                directory,
                "changed-state.json",
                model_sha256="b" * 64,
                state={"qpos": [1.1], "qvel": [0.0]},
                snapshot={"cache": [1, 2]},
            )
            self.assertNotEqual(
                first_ref.initial_state_id,
                gate1.load_initial_state_reference(changed_model, "lift").initial_state_id,
            )
            self.assertNotEqual(
                first_ref.initial_state_id,
                gate1.load_initial_state_reference(changed_state, "lift").initial_state_id,
            )
            self.assertNotEqual(
                first_ref.initial_state_id,
                gate1.load_initial_state_reference(first, "stack").initial_state_id,
            )

    def test_manifest_has_lift_stack_three_pair_types_and_initial_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lift_path, stack_path = self._inputs(Path(temp_dir))
            records = gate1.build_development_manifest_from_initial_states(lift_path, stack_path)

        self.assertEqual(len(records), 14)
        self.assertEqual({record["task"] for record in records}, {"lift", "stack"})
        self.assertEqual({record["split"] for record in records}, {"development"})
        self.assertEqual(
            {record["pair_type"] for record in records if record["pair_type"]},
            {"modifier_word", "physical_attribute", "stage_scope"},
        )
        pair_ids = {record["pair_id"] for record in records if record["pair_id"]}
        self.assertEqual(len(pair_ids), 6)
        for record in records:
            initial_state = record["initial_state"]
            metadata = record["metadata"]
            self.assertTrue(initial_state["initial_state_id"].startswith("initial_"))
            self.assertEqual(initial_state["initial_state_id"], metadata["initial_state_id"])
            self.assertEqual(initial_state["model_sha256"], metadata["model_sha256"])
            self.assertEqual(initial_state["initial_state_hash"], metadata["initial_state_hash"])
            self.assertEqual(record["manifest_status"], "draft")
            self.assertEqual(record["request_status"], "planned")

    def test_pending_incomplete_entries_are_retained_and_not_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lift_path, stack_path = self._inputs(Path(temp_dir))
            records = gate1.build_development_manifest_from_initial_states(lift_path, stack_path)
            no_pending = gate1.build_development_manifest_from_initial_states(
                lift_path, stack_path, include_pending=False
            )

        pending = [record for record in records if record["metadata"].get("incomplete")]
        self.assertEqual({record["task"] for record in pending}, {"lift", "stack"})
        self.assertTrue(all(record["metadata"]["lifecycle_status"] == "pending_calibration" for record in pending))
        self.assertTrue(all(record["metadata"]["requests_enabled"] is False for record in pending))
        self.assertTrue(all(record["stage_scope"] == [] for record in pending))
        self.assertTrue(all(record["allowed_control_channels"] == [] for record in pending))
        self.assertTrue(all(record["pair_id"] is None for record in pending))
        self.assertEqual(len(no_pending), 12)
        self.assertFalse(any(record["metadata"].get("incomplete") for record in no_pending))

    def test_original_constraints_are_emitted_without_rewriting(self) -> None:
        expected = {
            ("modifier_word", "neutral", "keep-a"),
            ("modifier_word", "careful", "keep-b"),
            ("physical_attribute", "neutral", "attribute"),
            ("stage_scope", "neutral", "stage"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            lift_path, stack_path = self._inputs(Path(temp_dir))
            records = gate1.build_development_manifest_from_initial_states(lift_path, stack_path)

        observed = {
            (
                record["metadata"].get("pair_example"),
                record["requested_constraints"].get("manner"),
                record["requested_constraints"].get("raw_marker"),
            )
            for record in records
            if record["metadata"].get("pair_example")
        }
        self.assertEqual(observed, expected)
        self.assertTrue(all("threshold" not in record["requested_constraints"] for record in records))
        self.assertTrue(all("clipped_constraints" not in record for record in records))

    def test_cli_writes_jsonl_without_external_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            lift_path, stack_path = self._inputs(directory)
            output = directory / "out" / "request_manifest.jsonl"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = gate1.main(
                    [
                        "--lift-initial-state",
                        str(lift_path),
                        "--stack-initial-state",
                        str(stack_path),
                        "--output",
                        str(output),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result, 0)
        self.assertEqual(summary["schema_version"], "synthesis_v2.dev_manifest.v1")
        self.assertEqual(summary["records"], 14)
        self.assertEqual(summary["external_api_calls"], 0)
        self.assertEqual(len(lines), 14)
        self.assertTrue(all(json.loads(line)["split"] == "development" for line in lines))


if __name__ == "__main__":  # pragma: no cover - convenient local invocation
    unittest.main()
