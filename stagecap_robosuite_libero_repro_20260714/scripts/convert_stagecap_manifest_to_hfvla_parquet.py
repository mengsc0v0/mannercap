#!/usr/bin/env python3
"""Convert Stage-CAP recorded steps into a LeRobot/HuggingFaceVLA-style parquet dataset.

This converter intentionally avoids importing LeRobot.  It writes the same
high-level v3 dataset layout used by the local HuggingFaceVLA/libero dataset:

  data/chunk-000/file-XXX.parquet
  meta/info.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/stats.json

The image columns are embedded as Hugging Face Image structs with PNG bytes and a
frame path string, which keeps the dataset self-contained.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


TASK_DIR_RE = re.compile(r"^(?P<task_id>\d{2})_(?P<name>.+)$")

IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
DATA_SCHEMA = pa.schema(
    [
        ("observation.images.image", IMAGE_STRUCT),
        ("observation.images.image2", IMAGE_STRUCT),
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ]
)


FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "fps": 10.0,
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "fps": 10.0,
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [8],
        "names": ["state"],
        "fps": 10.0,
    },
    "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["actions"],
        "fps": 10.0,
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": 10.0},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
    "index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
    "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": 10.0},
}


class RunningVectorStats:
    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, value: Iterable[float]) -> None:
        arr = np.asarray(list(value), dtype=np.float64)
        if arr.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim,)}, got {arr.shape}")
        self.count += 1
        self.min = np.minimum(self.min, arr)
        self.max = np.maximum(self.max, arr)
        self.sum += arr
        self.sumsq += arr * arr

    def merge(self, other: "RunningVectorStats") -> None:
        if other.count == 0:
            return
        if self.dim != other.dim:
            raise ValueError("Cannot merge stats with different dims")
        if self.count == 0:
            self.min = other.min.copy()
            self.max = other.max.copy()
        else:
            self.min = np.minimum(self.min, other.min)
            self.max = np.maximum(self.max, other.max)
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq

    def as_dict(self) -> dict[str, list[Any]]:
        if self.count == 0:
            zeros = [0.0] * self.dim
            return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [self.count],
        }


class RunningImageStats:
    def __init__(self):
        self.count = 0
        self.pixel_count = 0
        self.min = np.full(3, np.inf, dtype=np.float64)
        self.max = np.full(3, -np.inf, dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.sumsq = np.zeros(3, dtype=np.float64)

    def update(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        if image.size != (256, 256):
            image = image.resize((256, 256), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3).astype(np.float64)
        self.count += 1
        self.pixel_count += flat.shape[0]
        self.min = np.minimum(self.min, flat.min(axis=0))
        self.max = np.maximum(self.max, flat.max(axis=0))
        self.sum += flat.sum(axis=0)
        self.sumsq += (flat * flat).sum(axis=0)

    def merge(self, other: "RunningImageStats") -> None:
        if other.pixel_count == 0:
            return
        if self.pixel_count == 0:
            self.min = other.min.copy()
            self.max = other.max.copy()
        else:
            self.min = np.minimum(self.min, other.min)
            self.max = np.maximum(self.max, other.max)
        self.count += other.count
        self.pixel_count += other.pixel_count
        self.sum += other.sum
        self.sumsq += other.sumsq

    def as_dict(self, frame_count: int | None = None) -> dict[str, list[Any]]:
        reported_count = frame_count if frame_count is not None else self.count
        if self.pixel_count == 0:
            min_v = [0.0, 0.0, 0.0]
            max_v = [1.0, 1.0, 1.0]
            mean_v = [0.5, 0.5, 0.5]
            std_v = [0.25, 0.25, 0.25]
        else:
            mean = self.sum / self.pixel_count
            var = np.maximum(self.sumsq / self.pixel_count - mean * mean, 0.0)
            min_v = self.min.tolist()
            max_v = self.max.tolist()
            mean_v = mean.tolist()
            std_v = np.sqrt(var).tolist()
        return {
            "min": _image_stat_shape(min_v),
            "max": _image_stat_shape(max_v),
            "mean": _image_stat_shape(mean_v),
            "std": _image_stat_shape(std_v),
            "count": [reported_count],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-quality", nargs="*", default=["gold"])
    parser.add_argument("--action-field", default="lerobot_action_candidate")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=2,
        help=(
            "Keep every Nth recorded step. Stage-CAP records at 20Hz; the default "
            "stride 2 yields 10Hz and recomputes action from retained state deltas."
        ),
    )
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--repo-id", default="local/stagecap_libero_sfb1")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-stats-stride", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest_jsonl, set(args.include_quality))
    if args.max_episodes is not None:
        rows = rows[: args.max_episodes]
    if not rows:
        raise SystemExit("No manifest rows matched --include-quality.")

    if args.root.exists() and args.overwrite:
        shutil.rmtree(args.root)
    elif args.root.exists():
        raise SystemExit(f"Output root already exists: {args.root}")

    data_dir = args.root / "data" / "chunk-000"
    meta_episode_dir = args.root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_episode_dir.mkdir(parents=True)

    tasks = _task_index(rows)
    global_stats = _new_stats()
    episode_rows = []
    conversion_rows = []
    dataset_index = 0

    for episode_index, row in enumerate(rows):
        task = _task_text_for_row(row)
        task_index = tasks[task]
        file_index = episode_index
        data_path = data_dir / f"file-{file_index:03d}.parquet"
        episode_stats, length = _write_episode_data(
            row=row,
            data_path=data_path,
            episode_index=episode_index,
            task_index=task_index,
            dataset_start_index=dataset_index,
            action_field=args.action_field,
            fps=args.fps,
            sample_stride=args.sample_stride,
            batch_size=args.batch_size,
            image_stats_stride=args.image_stats_stride,
        )
        _merge_stats(global_stats, episode_stats)
        episode_rows.append(
            _episode_meta_row(
                episode_index=episode_index,
                file_index=file_index,
                dataset_from_index=dataset_index,
                dataset_to_index=dataset_index + length,
                task=task,
                length=length,
                stats=episode_stats,
            )
        )
        conversion_rows.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task": task,
                "quality_tier": row.get("quality_tier", ""),
                "has_privileged_finalize": row.get("has_privileged_finalize", False),
                "frame_count": length,
                "source_trial_dir": row["trial_dir"],
                "source_steps_path": row["steps_path"],
                "source_stage_trace_path": row.get("stage_trace_path", ""),
                "data_file": str(data_path.relative_to(args.root)),
            }
        )
        dataset_index += length
        print(f"Converted episode {episode_index:03d}: {length} frames | {task}")

    _write_tasks(args.root / "meta" / "tasks.parquet", tasks)
    _write_episode_meta(meta_episode_dir / "file-000.parquet", episode_rows)
    _write_info(args.root, args, rows, tasks, total_frames=dataset_index)
    _write_stats(args.root / "meta" / "stats.json", global_stats, total_frames=dataset_index)
    _write_readme(args.root, args, len(rows), dataset_index)
    (args.root / "stagecap_conversion_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "stagecap_hfvla_parquet_v0.1",
                "repo_id": args.repo_id,
                "root": str(args.root.resolve()),
                "source_manifest": str(args.manifest_jsonl.resolve()),
                "include_quality": args.include_quality,
                "action_field": args.action_field,
                "fps": args.fps,
                "sample_stride": args.sample_stride,
                "episodes": conversion_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote dataset: {args.root.resolve()}")


def _load_manifest(path: Path, include_quality: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("quality_tier") in include_quality:
                rows.append(row)
    rows.sort(key=lambda row: str(row["trial_dir"]))
    return rows


def _task_index(rows: list[dict[str, Any]]) -> dict[str, int]:
    tasks: dict[str, int] = {}
    for row in rows:
        task = _task_text_for_row(row)
        if task not in tasks:
            tasks[task] = len(tasks)
    return tasks


def _task_text_for_row(row: dict[str, Any]) -> str:
    stage_trace_path = row.get("stage_trace_path")
    if stage_trace_path:
        try:
            stage_trace = json.loads(Path(stage_trace_path).read_text(encoding="utf-8"))
            items = stage_trace if isinstance(stage_trace, list) else [stage_trace]
            for item in items:
                if isinstance(item, dict):
                    task = item.get("task") or item.get("task_language")
                    if task:
                        return str(task)
        except Exception:
            pass
    trial_dir = Path(row["trial_dir"])
    for parent in trial_dir.parents:
        match = TASK_DIR_RE.match(parent.name)
        if match:
            return match.group("name").replace("_", " ")
    return trial_dir.parent.name


def _write_episode_data(
    *,
    row: dict[str, Any],
    data_path: Path,
    episode_index: int,
    task_index: int,
    dataset_start_index: int,
    action_field: str,
    fps: float,
    sample_stride: int,
    batch_size: int,
    image_stats_stride: int,
) -> tuple[dict[str, Any], int]:
    if sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    trial_dir = Path(row["trial_dir"])
    steps_path = Path(row["steps_path"])
    stats = _new_stats()
    writer = pq.ParquetWriter(data_path, DATA_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    length = 0

    try:
        sampled_steps = _sample_steps(steps_path, sample_stride=sample_stride)
        for sample_idx, step in enumerate(sampled_steps):
            state = _as_float_list(step.get("observation.state"), 8, "observation.state")
            action = _action_for_sampled_step(
                step,
                next_step=sampled_steps[sample_idx + 1] if sample_idx + 1 < len(sampled_steps) else None,
                action_field=action_field,
            )
            agent_path = _image_path(trial_dir, step, "observation.images.image")
            wrist_path = _image_path(trial_dir, step, "observation.images.image2")
            frame_index = length
            global_index = dataset_start_index + frame_index
            timestamp = frame_index / fps
            record = {
                "observation.images.image": {
                    "bytes": _png_bytes(agent_path),
                    "path": f"frame_{frame_index:06d}.png",
                },
                "observation.images.image2": {
                    "bytes": _png_bytes(wrist_path),
                    "path": f"frame_{frame_index:06d}.png",
                },
                "observation.state": state,
                "action": action,
                "timestamp": float(timestamp),
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": global_index,
                "task_index": task_index,
            }
            batch.append(record)
            _update_numeric_stats(
                stats,
                state=state,
                action=action,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index,
                global_index=global_index,
                task_index=task_index,
            )
            if image_stats_stride > 0 and frame_index % image_stats_stride == 0:
                stats["observation.images.image"].update(agent_path)
                stats["observation.images.image2"].update(wrist_path)
            length += 1
            if len(batch) >= batch_size:
                writer.write_table(_records_to_table(batch))
                batch.clear()
        if batch:
            writer.write_table(_records_to_table(batch))
    finally:
        writer.close()

    return stats, length


def _sample_steps(path: Path, *, sample_stride: int) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            if not line.strip():
                continue
            if source_index % sample_stride == 0:
                steps.append(json.loads(line))
    if not steps:
        raise RuntimeError(f"No sampled steps from {path}")
    return steps


def _action_for_sampled_step(
    step: dict[str, Any],
    *,
    next_step: dict[str, Any] | None,
    action_field: str,
) -> list[float]:
    if next_step is None:
        return _as_float_list(step.get(action_field), 7, action_field)
    state = _as_float_list(step.get("observation.state"), 8, "observation.state")
    next_state = _as_float_list(next_step.get("observation.state"), 8, "next sampled observation.state")
    raw_action = step.get(action_field)
    gripper = float(raw_action[-1]) if isinstance(raw_action, list) and raw_action else 0.0
    return [float(next_state[i] - state[i]) for i in range(6)] + [gripper]


def _records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    columns = {name: [record[name] for record in records] for name in DATA_SCHEMA.names}
    return pa.Table.from_pydict(columns, schema=DATA_SCHEMA)


def _as_float_list(value: Any, dim: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dim:
        raise ValueError(f"Bad {name}: expected {dim} values, got {value!r}")
    return [float(x) for x in value]


def _image_path(trial_dir: Path, step: dict[str, Any], key: str) -> Path:
    ref = step.get(key)
    rel_path = ref.get("path") if isinstance(ref, dict) else None
    if not rel_path:
        raise ValueError(f"Missing image reference for {key}")
    path = trial_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _png_bytes(path: Path) -> bytes:
    # Fast path: Stage-CAP records PNG frames at 256x256 already.
    if path.suffix.lower() == ".png":
        return path.read_bytes()
    image = Image.open(path).convert("RGB")
    if image.size != (256, 256):
        image = image.resize((256, 256), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _new_stats() -> dict[str, Any]:
    return {
        "observation.images.image": RunningImageStats(),
        "observation.images.image2": RunningImageStats(),
        "observation.state": RunningVectorStats(8),
        "action": RunningVectorStats(7),
        "timestamp": RunningVectorStats(1),
        "frame_index": RunningVectorStats(1),
        "episode_index": RunningVectorStats(1),
        "index": RunningVectorStats(1),
        "task_index": RunningVectorStats(1),
    }


def _update_numeric_stats(
    stats: dict[str, Any],
    *,
    state: list[float],
    action: list[float],
    timestamp: float,
    frame_index: int,
    episode_index: int,
    global_index: int,
    task_index: int,
) -> None:
    stats["observation.state"].update(state)
    stats["action"].update(action)
    stats["timestamp"].update([timestamp])
    stats["frame_index"].update([frame_index])
    stats["episode_index"].update([episode_index])
    stats["index"].update([global_index])
    stats["task_index"].update([task_index])


def _merge_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[key].merge(value)


def _episode_meta_row(
    *,
    episode_index: int,
    file_index: int,
    dataset_from_index: int,
    dataset_to_index: int,
    task: str,
    length: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "data/chunk_index": 0,
        "data/file_index": file_index,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_to_index,
        "tasks": [task],
        "length": length,
    }
    row.update(_flatten_stats(stats, frame_count=length))
    row["meta/episodes/chunk_index"] = 0
    row["meta/episodes/file_index"] = 0
    return row


def _flatten_stats(stats: dict[str, Any], *, frame_count: int | None) -> dict[str, Any]:
    flat = {}
    for name, stat in stats.items():
        stat_dict = stat.as_dict(frame_count=frame_count) if isinstance(stat, RunningImageStats) else stat.as_dict()
        for key, value in stat_dict.items():
            flat[f"stats/{name}/{key}"] = value
    return flat


def _write_tasks(path: Path, tasks: dict[str, int]) -> None:
    ordered = sorted(tasks.items(), key=lambda item: item[1])
    table = pa.table(
        {
            "task_index": [idx for _, idx in ordered],
            "__index_level_0__": [task for task, _ in ordered],
        }
    )
    pq.write_table(table, path, compression="zstd")


def _write_episode_meta(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_info(
    root: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    tasks: dict[str, int],
    *,
    total_frames: int,
) -> None:
    features = json.loads(json.dumps(FEATURES))
    for feature in features.values():
        feature["fps"] = float(args.fps)
    info = {
        "codebase_version": "v3.0",
        "robot_type": args.robot_type,
        "total_episodes": len(rows),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": float(args.fps),
        "splits": {"train": f"0:{len(rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": _size_mb(root / "data"),
        "video_files_size_in_mb": 0,
    }
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _write_stats(path: Path, stats: dict[str, Any], *, total_frames: int) -> None:
    serializable = {}
    for name, stat in stats.items():
        if isinstance(stat, RunningImageStats):
            serializable[name] = stat.as_dict(frame_count=total_frames)
        else:
            serializable[name] = stat.as_dict()
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _write_readme(root: Path, args: argparse.Namespace, episodes: int, frames: int) -> None:
    text = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- LIBERO
- Stage-CAP
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Stage-CAP LIBERO VLA Dataset

This dataset was converted from Stage-CAP recorded trajectories.

- Source manifest: `{args.manifest_jsonl.resolve()}`
- Episodes: {episodes}
- Frames: {frames}
- FPS: {args.fps}
- Sample stride: {args.sample_stride}
- Action field: `{args.action_field}`
- Included quality tiers: `{', '.join(args.include_quality)}`

The layout follows the local `HuggingFaceVLA/libero` LeRobot v3 parquet schema.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _image_stat_shape(values: list[float]) -> list[list[list[float]]]:
    return [[[float(v)]] for v in values]


def _size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return int(math.ceil(total / 1024 / 1024))


if __name__ == "__main__":
    main()
