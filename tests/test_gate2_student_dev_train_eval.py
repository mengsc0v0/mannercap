import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "gate2_student_dev_train_eval.py"
SPEC = importlib.util.spec_from_file_location(PATH.stem, PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        steps=100,
        max_wall_seconds=1200.0,
        train_cache=root / "train-cache",
        validation_cache=root / "validation-cache",
        checkpoint_output=root / "attempt02" / "checkpoint.pt",
        output=root / "attempt02" / "report.json",
    )


class StudentDevTrainEvalTests(unittest.TestCase):
    def test_progress_keeps_actual_phase_and_completed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = MODULE._new_progress(_args(Path(directory)))
            MODULE._set_phase(progress, "dataset_loader")
            progress["completed_steps"] = 3
            MODULE._refresh_progress(progress)
            self.assertEqual(progress["phase"], "dataset_loader")
            self.assertEqual(progress["completed_steps"], 3)
            self.assertGreaterEqual(progress["elapsed_seconds"], 0.0)

    def test_action_mask_requires_front_seven_of_twenty_four(self):
        cache = {
            "sample_count": 2,
            "action_mask": torch.cat(
                [
                    torch.ones(2, MODULE.HORIZON, MODULE.ACTION_DIM, dtype=torch.bool),
                    torch.zeros(
                        2,
                        MODULE.HORIZON,
                        MODULE.MODEL_ACTION_DIM - MODULE.ACTION_DIM,
                        dtype=torch.bool,
                    ),
                ],
                dim=-1,
            ),
        }
        MODULE.validate_action_mask(cache)
        cache["action_mask"][0, 0, MODULE.ACTION_DIM] = True
        with self.assertRaises(RuntimeError):
            MODULE.validate_action_mask(cache)

    def test_train_only_stats_are_applied_to_every_group(self):
        class Dataset:
            arm2stats_dict = {"franka_eef": {"mannercap": {"action": {"min": [0]}}}}

        stats = {"action": {"min": [1]}, "observation.state": {"min": [2]}}
        dataset = Dataset()
        MODULE.apply_train_only_stats(dataset, stats)
        self.assertEqual(dataset.arm2stats_dict["franka_eef"]["mannercap"], stats)
        self.assertIsNot(dataset.arm2stats_dict["franka_eef"]["mannercap"], stats)

    def test_loader_fails_closed_for_non_bounds_normalization(self):
        class Yaml:
            @staticmethod
            def safe_load(stream):
                return {"normalization_type": "normal"}

        class Dataset:
            def __init__(self, **kwargs):
                raise AssertionError("dataset constructor must not run")

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "dataset_config.yaml"
            config.write_text("normalization_type: normal\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.load_dataset(
                    config_path=config,
                    cache_dir=Path(directory) / "cache",
                    yaml_module=Yaml,
                    dataset_class=Dataset,
                )

    def test_dataset_binding_checks_source_ids_and_task_manner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "meta").mkdir(parents=True)
            config_path = root / "dataset_config.yaml"
            config_path.write_text("normalization_type: bounds\n", encoding="utf-8")
            source_id = "d0-lift-seed-42-neutral"
            manifest = {
                "schema": "mannercap-la4vla-microset-staging-v1",
                "sample_count": 2,
                "horizon": 50,
                "windows": [
                    {
                        "task": "lift",
                        "task_index": 0,
                        "source_episode_id": source_id,
                        "manner_tag": "neutral",
                        "source_initial_state_sha256": "initial",
                        "source_result_sha256": "result",
                    },
                    {
                        "task": "lift",
                        "task_index": 0,
                        "source_episode_id": source_id,
                        "manner_tag": "neutral",
                        "source_initial_state_sha256": "initial",
                        "source_result_sha256": "result",
                    },
                ],
            }
            manifest_path = root / "source_staging_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "independent_source_episode_count": 1,
                        "total_episodes": 2,
                        "total_tasks": 1,
                    }
                ),
                encoding="utf-8",
            )
            (root / "meta" / "tasks.jsonl").write_text(
                '{"task_index": 0, "task": "Lift the cube."}\n', encoding="utf-8"
            )
            qc_path = Path(directory) / "qc.json"
            qc = {
                "train": {
                    "seed": 42,
                    "manifest": str(manifest_path),
                    "windows": 2,
                    "source_episode_count": 1,
                    "source_episode_ids": [source_id],
                    "source_initial_state_hashes": ["initial"],
                    "source_result_hash_count": 1,
                }
            }
            qc_path.write_text(json.dumps(qc), encoding="utf-8")

            class Yaml:
                @staticmethod
                def safe_load(stream):
                    return {
                        "normalization_type": "bounds",
                        "data_groups": {"franka_eef": {"mannercap": {"path": str(root)}}},
                    }

            binding = MODULE.validate_dataset_binding(
                config_path=config_path,
                qc_report_path=qc_path,
                split="train",
                yaml_module=Yaml,
            )
            self.assertEqual(binding["status"], "pass")
            qc["train"]["source_episode_ids"] = ["d0-stack-seed-42-neutral"]
            qc_path.write_text(json.dumps(qc), encoding="utf-8")
            with self.assertRaises(MODULE.DatasetBindingError) as context:
                MODULE.validate_dataset_binding(
                    config_path=config_path,
                    qc_report_path=qc_path,
                    split="train",
                    yaml_module=Yaml,
                )
            self.assertTrue(
                any("source_episode_ids" in item for item in context.exception.details["mismatches"])
            )

    def test_qc_binding_requires_expected_hash_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "qc.json"
            report.write_text(json.dumps({"plan_sha256": "plan"}), encoding="utf-8")
            digest = MODULE.file_sha256(report)
            self.assertEqual(
                MODULE.validate_qc_binding(
                    qc_report_path=report,
                    expected_qc_sha256=digest,
                    plan_sha256="plan",
                ),
                digest,
            )
            with self.assertRaises(RuntimeError):
                MODULE.validate_qc_binding(
                    qc_report_path=report,
                    expected_qc_sha256="0" * 64,
                    plan_sha256="plan",
                )
            with self.assertRaises(RuntimeError):
                MODULE.validate_qc_binding(
                    qc_report_path=report,
                    expected_qc_sha256=digest,
                    plan_sha256="other",
                )


if __name__ == "__main__":
    unittest.main()
