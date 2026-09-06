#!/usr/bin/env python3
"""Convert recorded Stage-CAP LIBERO steps into a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [8],
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["actions"],
    },
}

_TASK_DIR_RE = re.compile(r"^(?P<task_id>\d{2})_(?P<name>.+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Local/HF-style repo id, e.g. local/stagecap_libero_sfb1")
    parser.add_argument("--root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--lerobot-repo", type=Path, default=Path("/home/lifuchun/data/lerobot/repos/lerobot"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--action-field", default="lerobot_action_candidate")
    parser.add_argument("--include-quality", nargs="*", default=["gold"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _prepare_lerobot_import(args.lerobot_repo)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = _load_manifest(args.manifest_jsonl, set(args.include_quality))
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    if not rows:
        raise SystemExit("No manifest rows matched --include-quality.")

    if args.root.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.root)
    elif args.root.exists():
        raise SystemExit(f"Output root already exists. Use --overwrite to replace: {args.root}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=DEFAULT_FEATURES,
        root=args.root,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    conversion_rows = []
    for episode_idx, row in enumerate(rows):
        task = _task_text_for_row(row)
        steps = list(_iter_steps(Path(row["steps_path"])))
        if not steps:
            raise RuntimeError(f"No steps found for {row['steps_path']}")

        added = 0
        for step in steps:
            frame = _frame_from_step(
                step,
                trial_dir=Path(row["trial_dir"]),
                task=task,
                action_field=args.action_field,
            )
            dataset.add_frame(frame)
            added += 1

        dataset.save_episode(parallel_encoding=True)
        conversion_rows.append(
            {
                "episode_index": episode_idx,
                "task": task,
                "source_trial_dir": row["trial_dir"],
                "source_steps_path": row["steps_path"],
                "source_stage_trace_path": row.get("stage_trace_path", ""),
                "quality_tier": row.get("quality_tier", ""),
                "has_privileged_finalize": row.get("has_privileged_finalize", False),
                "frame_count": added,
            }
        )
        print(f"Converted episode {episode_idx}: {added} frames | {task}")

    manifest_path = args.root / "stagecap_conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stagecap_to_lerobot_conversion_v0.1",
                "repo_id": args.repo_id,
                "root": str(args.root.resolve()),
                "fps": args.fps,
                "action_field": args.action_field,
                "include_quality": args.include_quality,
                "episodes": conversion_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote conversion manifest: {manifest_path.resolve()}")
    print(f"LeRobot dataset root: {args.root.resolve()}")


def _prepare_lerobot_import(lerobot_repo: Path) -> None:
    src = lerobot_repo / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    try:
        import lerobot  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Could not import LeRobot. Run this in an environment with LeRobot dependencies, "
            f"or pass --lerobot-repo. Original error: {exc!r}"
        ) from exc


def _load_manifest(path: Path, include_quality: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("quality_tier") in include_quality:
                if not row.get("steps_path"):
                    continue
                rows.append(row)
    return rows


def _iter_steps(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _task_text_for_row(row: dict[str, Any]) -> str:
    stage_trace_path = row.get("stage_trace_path")
    if stage_trace_path:
        try:
            stage_trace = json.loads(Path(stage_trace_path).read_text(encoding="utf-8"))
            for item in stage_trace:
                task = item.get("task") or item.get("task_language")
                if task:
                    return str(task)
        except Exception:
            pass
    trial_dir = Path(row["trial_dir"])
    for parent in trial_dir.parents:
        if re_match := _TASK_DIR_RE.match(parent.name):
            return re_match.group("name").replace("_", " ")
    return trial_dir.parent.name


def _frame_from_step(
    step: dict[str, Any],
    *,
    trial_dir: Path,
    task: str,
    action_field: str,
) -> dict[str, Any]:
    state = np.asarray(step["observation.state"], dtype=np.float32)
    action = np.asarray(step[action_field], dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"Bad observation.state shape: {state.shape}")
    if action.shape != (7,):
        raise ValueError(f"Bad action shape for {action_field}: {action.shape}")

    return {
        "task": task,
        "observation.images.image": _load_image(trial_dir, step, "observation.images.image"),
        "observation.images.image2": _load_image(trial_dir, step, "observation.images.image2"),
        "observation.state": state,
        "action": action,
    }


def _load_image(trial_dir: Path, step: dict[str, Any], key: str) -> np.ndarray:
    ref = step.get(key) or {}
    rel_path = ref.get("path") if isinstance(ref, dict) else None
    if not rel_path:
        raise ValueError(f"Missing image path for {key}")
    path = trial_dir / rel_path
    image = Image.open(path).convert("RGB")
    if image.size != (256, 256):
        image = image.resize((256, 256), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


if __name__ == "__main__":
    main()
