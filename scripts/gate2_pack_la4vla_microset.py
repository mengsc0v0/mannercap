"""Pack a captured MannerCAP microset into the LeRobot layout LA4VLA reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def validate_staging(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "mannercap-la4vla-microset-staging-v1":
        raise ValueError("unsupported staging schema")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != manifest.get("sample_count"):
        raise ValueError("staging sample count mismatch")
    horizon = manifest.get("horizon")
    task_prompts: dict[int, str] = {}
    for index, window in enumerate(windows):
        if "task_index" not in window or "prompt" not in window:
            raise ValueError(f"window {index} is missing task_index or prompt")
        task_index = int(window["task_index"])
        prompt = str(window["prompt"])
        previous = task_prompts.setdefault(task_index, prompt)
        if previous != prompt:
            raise ValueError(
                f"task index {task_index} maps to multiple prompts: "
                f"{previous!r} and {prompt!r}"
            )
        if len(window.get("rows", [])) != horizon:
            raise ValueError(f"window {index} does not have exactly {horizon} rows")
        for row in window["rows"]:
            if len(row.get("observation_state", [])) != 8:
                raise ValueError(f"window {index} has a non-8D state")
            if len(row.get("action", [])) != 7:
                raise ValueError(f"window {index} has a non-7D action")


def array_statistics(values: Any) -> dict[str, list[float]]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not np.isfinite(array).all():
        raise ValueError("statistics input must be a non-empty finite 2D array")
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def pack(staging_dir: Path, output: Path) -> dict[str, Any]:
    import pandas as pd

    manifest = json.loads((staging_dir / "staging_manifest.json").read_text(encoding="utf-8"))
    validate_staging(manifest)
    if output.exists():
        raise FileExistsError(output)
    data_dir = output / "data" / "chunk-000"
    meta_dir = output / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    tasks: dict[int, str] = {}
    all_states: list[list[float]] = []
    all_actions: list[list[float]] = []
    for episode_index, window in enumerate(manifest["windows"]):
        task_index = int(window["task_index"])
        tasks[task_index] = str(window["prompt"])
        image_bytes = {
            key: (staging_dir / relative).read_bytes()
            for key, relative in window["images"].items()
        }
        records = []
        base_time = float(window["rows"][0]["timestamp"])
        for frame_index, row in enumerate(window["rows"]):
            all_states.append(row["observation_state"])
            all_actions.append(row["action"])
            records.append({
                "observation.state": row["observation_state"],
                "action": row["action"],
                "observation.images.image": {"bytes": image_bytes["image"], "path": None},
                "observation.images.wrist_image": {"bytes": image_bytes["wrist_image"], "path": None},
                "timestamp": float(row["timestamp"]) - base_time,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "task_index": task_index,
            })
        pd.DataFrame.from_records(records).to_parquet(
            data_dir / f"episode_{episode_index:06d}.parquet", index=False
        )

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task_index, prompt in sorted(tasks.items()):
            stream.write(json.dumps({"task_index": task_index, "task": prompt}) + "\n")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "fps": 20,
        "total_episodes": len(manifest["windows"]),
        "total_frames": len(manifest["windows"]) * manifest["horizon"],
        "total_tasks": len(tasks),
        "synthetic_window_episodes": True,
        "independent_source_episode_count": len({
            window["source_episode"] for window in manifest["windows"]
        }),
        "source_schema": manifest["schema"],
        "source_manner_tags": sorted({
            str(window.get("manner_tag", "unspecified"))
            for window in manifest["windows"]
        }),
        "includes_task_failures": bool(manifest.get("includes_task_failures", False)),
        "generalization_claim_allowed": False,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    stats = {
        "action": array_statistics(all_actions),
        "observation.state": array_statistics(all_states),
    }
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(staging_dir / "staging_manifest.json", output / "source_staging_manifest.json")

    config = f"""max_action_dim: 24
max_state_dim: 24
max_views: 2
normalization_type: bounds
datasets_manifest: mannercap_microset_64.pkl
decoder_cache_size: 16
cache_shard_size_bytes: 268435456
data_groups:
  franka_eef:
    mannercap_microset:
      path: {output.as_posix()}
      view_map:
        image_1: observation.images.image
        image_2: observation.images.wrist_image
      use_delta_action: false
      process_suite: franka_ee_pose
      suite_config:
        state_key: observation.state
        action_key: action
        ee_pose_dims: [0, 6]
        gripper_indices: [6]
"""
    (output / "dataset_config.yaml").write_text(config, encoding="utf-8")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = pack(args.staging_dir.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), **info}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

