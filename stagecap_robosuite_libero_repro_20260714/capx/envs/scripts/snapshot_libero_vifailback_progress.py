"""Snapshot an in-progress LIBERO CaP-X + ViFailback output directory."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRIAL_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_reward_"
    r"(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)


def main(output_dir: str) -> None:
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    report_path = root / "libero_cap_vifailback_report.json"
    records = sorted(root.glob("*/*/*_vifailback_capx_record.json"))
    rounds = sorted(root.glob("*/*/round_[0-9][0-9]"))
    trial_dirs = _find_trial_dirs(root)
    feedback_files = sorted(root.glob("**/vifailback_feedback.json"))

    completed_tasks = _completed_tasks_from_report(report_path)
    active_tasks = _active_task_paths(rounds)
    trial_counts = _trial_counts(trial_dirs)
    feedback_counts = _feedback_counts(feedback_files)
    commands = _low_level_command_counts(feedback_files)

    print(f"Output: {root.resolve()}")
    print(f"Report exists: {'yes' if report_path.exists() else 'no'}")
    print(f"Completed task records: {len(records)}")
    if completed_tasks is not None:
        print(
            "Report runs: "
            f"{completed_tasks['runs']}/40, "
            f"task success {completed_tasks['task_success']}, "
            f"max-round exhausted {completed_tasks['max_rounds_exhausted']}"
        )
    print(f"Active/generated task dirs: {len(active_tasks)}")
    print(f"Generated rounds: {len(rounds)}")
    print(
        "Trials: "
        f"{trial_counts['success']} success / "
        f"{trial_counts['total']} generated / "
        f"{trial_counts['failed']} failed"
    )
    print(
        "ViFailback feedback files: "
        f"{feedback_counts['completed']} completed / "
        f"{feedback_counts['total']} total / "
        f"{feedback_counts['failed']} failed"
    )

    if active_tasks:
        print("\nCurrent task snapshots:")
        for task_path in active_tasks[-5:]:
            print(f"  {_task_label(root, task_path)}")
            for round_path in sorted(task_path.glob("round_[0-9][0-9]")):
                run_paths = list(round_path.glob("*/run"))
                if not run_paths:
                    print(f"    {round_path.name}: no run dir yet")
                    continue
                run_path = run_paths[0]
                c = _trial_counts(_find_trial_dirs(run_path))
                fb = _feedback_counts(sorted(run_path.glob("**/vifailback_feedback.json")))
                done = "done" if (run_path / "aaa_done_flag" / "aaa_done_flag.txt").exists() else "running"
                print(
                    f"    {round_path.name}: {done}, "
                    f"trials {c['success']}/{c['total']} success, "
                    f"feedback {fb['completed']}/{fb['total']}"
                )

    if commands:
        print("\nTop ViFailback low-level commands:")
        for command, count in commands.most_common(10):
            print(f"  {count}x {command}")


def _completed_tasks_from_report(report_path: Path) -> dict[str, int] | None:
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs", [])
    summary = report.get("summary", {})
    return {
        "runs": len(runs),
        "task_success": int(summary.get("task_success_count", 0)),
        "max_rounds_exhausted": int(summary.get("max_rounds_exhausted_count", 0)),
    }


def _active_task_paths(round_paths: list[Path]) -> list[Path]:
    tasks = {round_path.parent for round_path in round_paths}
    return sorted(tasks)


def _find_trial_dirs(root: Path) -> list[Path]:
    return [path for path in root.glob("**/trial_*") if path.is_dir() and TRIAL_RE.match(path.name)]


def _trial_counts(trial_dirs: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["total"] = len(trial_dirs)
    for trial_dir in trial_dirs:
        match = TRIAL_RE.match(trial_dir.name)
        if not match:
            continue
        if match.group("task_completed") == "1":
            counts["success"] += 1
        else:
            counts["failed"] += 1
    return counts


def _feedback_counts(feedback_files: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["total"] = len(feedback_files)
    for feedback_file in feedback_files:
        try:
            data = json.loads(feedback_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counts["failed"] += 1
            continue
        status = str(data.get("status", "unknown"))
        counts[status] += 1
        if status != "completed":
            counts["failed"] += 1
    return counts


def _low_level_command_counts(feedback_files: list[Path]) -> Counter[str]:
    commands: Counter[str] = Counter()
    for feedback_file in feedback_files:
        try:
            data: dict[str, Any] = json.loads(feedback_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        command = _extract_low_level_command(str(data.get("feedback_text", "")))
        if command:
            commands[command] += 1
    return commands


def _extract_low_level_command(feedback_text: str) -> str:
    marker = '"low level commands":'
    if marker not in feedback_text:
        return ""
    tail = feedback_text.split(marker, 1)[1].strip()
    if tail.startswith('"'):
        tail = tail[1:]
    return tail.split('",', 1)[0].split('"\n', 1)[0].strip()


def _task_label(root: Path, task_path: Path) -> str:
    try:
        rel = task_path.relative_to(root)
    except ValueError:
        rel = task_path
    parts = rel.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return str(rel)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python capx/envs/scripts/snapshot_libero_vifailback_progress.py "
            "<output_dir>"
        )
    main(sys.argv[1])
