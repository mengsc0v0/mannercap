#!/usr/bin/env python3
"""Inspect Stage-CAP trials for VLA dataset conversion readiness."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TRIAL_DIR_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_reward_"
    r"(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Stage-CAP output roots or trial dirs.")
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--allow-privileged-finalize", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for trial_dir in _iter_trial_dirs(root):
            row = inspect_trial(trial_dir, min_steps=args.min_steps)
            row["quality_tier"] = quality_tier(
                row,
                allow_privileged_finalize=args.allow_privileged_finalize,
            )
            rows.append(row)

    rows.sort(key=lambda row: str(row["trial_dir"]))
    print_summary(rows)

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL manifest: {args.output_jsonl.resolve()}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV manifest: {args.output_csv.resolve()}")


def _iter_trial_dirs(root: Path):
    root = root.resolve()
    if TRIAL_DIR_RE.match(root.name):
        yield root
        return
    for path in root.rglob("trial_*_sandboxrc_*_reward_*_taskcompleted_*"):
        if path.is_dir() and TRIAL_DIR_RE.match(path.name):
            yield path


def inspect_trial(trial_dir: Path, *, min_steps: int) -> dict[str, Any]:
    match = TRIAL_DIR_RE.match(trial_dir.name)
    parsed = match.groupdict() if match else {}
    steps_path = trial_dir / "steps.jsonl"
    stage_trace_path = trial_dir / "stage_trace.json"
    code_path = trial_dir / "code.py"
    video_path = trial_dir / "video_combined.mp4"
    metadata_path = trial_dir / "steps_metadata.json"

    step_count = 0
    valid_step_count = 0
    missing_agent_images = 0
    missing_wrist_images = 0
    bad_state_dim = 0
    bad_action_dim = 0
    action_field = "lerobot_action_candidate"

    if steps_path.exists():
        with steps_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                step_count += 1
                try:
                    step = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = step.get("observation.state")
                action = step.get(action_field)
                if not isinstance(state, list) or len(state) != 8:
                    bad_state_dim += 1
                if not isinstance(action, list) or len(action) != 7:
                    bad_action_dim += 1
                if not _image_exists(trial_dir, step, "observation.images.image"):
                    missing_agent_images += 1
                if not _image_exists(trial_dir, step, "observation.images.image2"):
                    missing_wrist_images += 1
                if (
                    isinstance(state, list)
                    and len(state) == 8
                    and isinstance(action, list)
                    and len(action) == 7
                    and _image_exists(trial_dir, step, "observation.images.image")
                    and _image_exists(trial_dir, step, "observation.images.image2")
                ):
                    valid_step_count += 1

    stage_trace = _read_json(stage_trace_path)
    has_privileged_finalize = _contains_stage(stage_trace, "privileged_data_generation_finalize")

    reasons = []
    if not steps_path.exists():
        reasons.append("missing_steps_jsonl")
    if step_count < min_steps:
        reasons.append(f"too_few_steps<{min_steps}")
    if missing_agent_images:
        reasons.append("missing_agent_images")
    if missing_wrist_images:
        reasons.append("missing_wrist_images")
    if bad_state_dim:
        reasons.append("bad_state_dim")
    if bad_action_dim:
        reasons.append("bad_action_dim")

    return {
        "trial_dir": str(trial_dir),
        "steps_path": str(steps_path) if steps_path.exists() else "",
        "steps_metadata_path": str(metadata_path) if metadata_path.exists() else "",
        "stage_trace_path": str(stage_trace_path) if stage_trace_path.exists() else "",
        "code_path": str(code_path) if code_path.exists() else "",
        "video_path": str(video_path) if video_path.exists() else "",
        "trial": int(parsed.get("trial", -1)),
        "sandbox_rc": int(parsed.get("sandbox_rc", -1)),
        "reward": float(parsed.get("reward", 0.0)),
        "task_completed": bool(int(parsed.get("task_completed", 0))),
        "step_count": step_count,
        "valid_step_count": valid_step_count,
        "missing_agent_images": missing_agent_images,
        "missing_wrist_images": missing_wrist_images,
        "bad_state_dim": bad_state_dim,
        "bad_action_dim": bad_action_dim,
        "has_privileged_finalize": has_privileged_finalize,
        "candidate_action_field": action_field,
        "ready_reasons": ";".join(reasons),
    }


def quality_tier(row: dict[str, Any], *, allow_privileged_finalize: bool) -> str:
    if row["sandbox_rc"] != 0 or row["reward"] < 1.0 or not row["task_completed"]:
        return "reject_failed_task"
    if row["ready_reasons"]:
        return "reject_incomplete_recording"
    if row["has_privileged_finalize"] and not allow_privileged_finalize:
        return "weak_privileged_finalize"
    if row["has_privileged_finalize"]:
        return "silver_privileged_finalize"
    return "gold"


def _image_exists(trial_dir: Path, step: dict[str, Any], key: str) -> bool:
    ref = step.get(key) or {}
    rel_path = ref.get("path") if isinstance(ref, dict) else None
    return bool(rel_path and (trial_dir / rel_path).exists())


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _contains_stage(value: Any, stage_id: str) -> bool:
    if isinstance(value, dict):
        if value.get("stage_id") == stage_id:
            return True
        return any(_contains_stage(item, stage_id) for item in value.values())
    if isinstance(value, list):
        return any(_contains_stage(item, stage_id) for item in value)
    return False


def print_summary(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["quality_tier"]] = counts.get(row["quality_tier"], 0) + 1
    print(f"Scanned trials: {len(rows)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    ready = [row for row in rows if row["quality_tier"] in {"gold", "silver_privileged_finalize"}]
    print(f"Ready for conversion: {len(ready)}")


if __name__ == "__main__":
    main()
