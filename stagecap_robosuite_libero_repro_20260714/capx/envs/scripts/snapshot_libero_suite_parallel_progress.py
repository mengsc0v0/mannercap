"""Snapshot a suite-parallel LIBERO CaP-X + ViFailback run."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRIAL_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_reward_"
    r"(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)
ACTION_API_RE = re.compile(
    r"\b("
    r"goto_pose|goto_pose_interactive_cartesian|open_gripper|close_gripper|"
    r"open_cabinet_drawer|close_cabinet_drawer|pick_and_place_object|pick_and_place_objects|"
    r"put_object_in_container|put_object_on_target|put_object_in_goal_region|"
    r"put_object_in_microwave_and_close|"
    r"push_object_to_goal_region|"
    r"satisfy_goal_predicates|"
    r"move_to_joints|execute_joint_trajectory|goto_home_joint_position|"
    r"plan_grasp_trajectory|plan_with_grasped_object"
    r")\s*\("
)
PREDICATE_ALIGNED_API_RE = re.compile(r"\bsatisfy_goal_predicates\s*\(")
HIGH_LEVEL_PLACE_API_RE = re.compile(
    r"\b("
    r"pick_and_place_object|pick_and_place_objects|"
    r"put_object_in_container|put_object_on_target"
    r")\s*\("
)
COMPOUND_DRAWER_API_RE = re.compile(r"\b(open_cabinet_drawer|close_cabinet_drawer)\s*\(")


def main(root_dir: str) -> None:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    workers = _read_workers(root / "workers.tsv")
    records = sorted(root.glob("**/*_vifailback_capx_record.json"))
    trial_dirs = _find_trial_dirs(root)
    feedback_files = sorted(root.glob("**/vifailback_feedback.json"))

    print(f"Root: {root.resolve()}")
    print(f"Workers: {sum(_pid_running(w['pid']) for w in workers)}/{len(workers)} running")
    for worker in workers:
        status = "RUNNING" if _pid_running(worker["pid"]) else "STOPPED"
        print(f"  {worker['suite']}: {status} pid={worker['pid']} {worker['gpu']}")

    trial_counts = _trial_counts(trial_dirs)
    feedback_counts = _feedback_counts(feedback_files)
    record_counts = _record_counts(records)
    program_counts = _program_counts(root)
    print("")
    print(
        "Tasks: "
        f"{record_counts['records']} records / "
        f"{record_counts['success']} success / "
        f"{record_counts['max_rounds_exhausted']} max-round exhausted / "
        f"{record_counts['launch_failed']} launch failed"
    )
    print(
        "Trials: "
        f"{trial_counts['success']} success / "
        f"{trial_counts['total']} generated / "
        f"{trial_counts['failed']} failed"
    )
    print(
        "ViFailback: "
        f"{feedback_counts['completed']} completed / "
        f"{feedback_counts['total']} files / "
        f"{feedback_counts['skipped_low_evidence']} skipped low-evidence / "
        f"{feedback_counts['false_success']} false-success / "
        f"{feedback_counts['failed']} failed"
    )
    print(
        "Program critic: "
        f"{program_counts['actionful_code']} actionful code / "
        f"{program_counts['code_files']} code files, "
        f"{program_counts['one_frame_videos']} one-frame videos, "
        f"{program_counts['predicate_alignment_needed']} predicate-align candidates, "
        f"{program_counts['compound_drawer_alignment_needed']} compound-drawer candidates, "
        f"{program_counts['program_feedback_prompts']} retry prompts with program feedback"
    )

    print("\nBy suite:")
    for suite_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if suite_dir.name == "logs":
            continue
        _print_suite_line(root, suite_dir)

    commands = _low_level_command_counts(feedback_files)
    if commands:
        print("\nTop ViFailback low-level commands:")
        for command, count in commands.most_common(10):
            print(f"  {count}x {command}")


def _read_workers(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    workers: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4:
            workers.append(
                {
                    "suite": parts[0],
                    "pid": parts[1],
                    "gpu": parts[2],
                    "log": " ".join(parts[3:]),
                }
            )
    return workers


def _pid_running(pid: str) -> bool:
    return subprocess.run(
        ["kill", "-0", pid],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _print_suite_line(root: Path, suite_dir: Path) -> None:
    records = sorted(suite_dir.glob("*/*/*_vifailback_capx_record.json"))
    trial_dirs = _find_trial_dirs(suite_dir)
    feedback_files = sorted(suite_dir.glob("**/vifailback_feedback.json"))
    trials = _trial_counts(trial_dirs)
    feedback = _feedback_counts(feedback_files)
    record_counts = _record_counts(records)
    program = _program_counts(suite_dir)
    active_tasks = sorted({p.parent for p in suite_dir.glob("*/*/round_[0-9][0-9]")})
    current = ""
    if active_tasks:
        try:
            current = str(active_tasks[-1].relative_to(root))
        except ValueError:
            current = str(active_tasks[-1])
    print(
        f"  {suite_dir.name}: tasks {record_counts['records']}/10, "
        f"success {record_counts['success']}, "
        f"trials {trials['success']}/{trials['total']} success, "
        f"feedback {feedback['completed']}/{feedback['total']}, "
        f"low-evidence skipped {feedback['skipped_low_evidence']}, "
        f"false-success {feedback['false_success']}, "
        f"actionful {program['actionful_code']}/{program['code_files']}, "
        f"1-frame {program['one_frame_videos']}, "
        f"predicate-align {program['predicate_alignment_needed']}, "
        f"compound-drawer {program['compound_drawer_alignment_needed']}, "
        f"current {current or '-'}"
    )


def _record_counts(records: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["records"] = len(records)
    for record in records:
        try:
            data: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("task_success"):
            counts["success"] += 1
        if data.get("max_rounds_exhausted"):
            counts["max_rounds_exhausted"] += 1
        if data.get("status") == "launch_failed":
            counts["launch_failed"] += 1
    return counts


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
        if status == "skipped_insufficient_rollout_evidence":
            counts["skipped_low_evidence"] += 1
            continue
        if status in {"completed", "skipped_existing"}:
            command_text = str(data.get("feedback_text", ""))
            if re.search(r'"success detection"\s*:\s*"Successful"', command_text, flags=re.IGNORECASE):
                counts["false_success"] += 1
        if status != "completed":
            counts["failed"] += 1
    return counts


def _program_counts(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    code_files = sorted(root.glob("**/code.py"))
    counts["code_files"] = len(code_files)
    for code_file in code_files:
        try:
            code = code_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ACTION_API_RE.search(code):
            counts["actionful_code"] += 1
        if HIGH_LEVEL_PLACE_API_RE.search(code) and not PREDICATE_ALIGNED_API_RE.search(code):
            summary_path = code_file.with_name("summary.txt")
            try:
                summary = summary_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                summary = ""
            if "Task Completed: False" in summary:
                counts["predicate_alignment_needed"] += 1
        if (
            COMPOUND_DRAWER_API_RE.search(code)
            and (HIGH_LEVEL_PLACE_API_RE.search(code) or "drawer" in code.lower())
            and not PREDICATE_ALIGNED_API_RE.search(code)
        ):
            summary_path = code_file.with_name("summary.txt")
            try:
                summary = summary_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                summary = ""
            task_text = " ".join(str(part) for part in code_file.parts).replace("_", " ").lower()
            if (
                "Task Completed: False" in summary
                and "drawer" in task_text
                and any(word in task_text for word in ("put", "place", "inside", "on the"))
            ):
                counts["compound_drawer_alignment_needed"] += 1

    for text_file in list(root.glob("**/summary.txt")) + list(root.glob("**/logs/*.log")):
        try:
            text = text_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        counts["one_frame_videos"] += len(
            re.findall(r"video_combined\.mp4 \(1 frames\)", text)
        )

    for config_file in root.glob("**/round_[0-9][0-9]/config.yaml"):
        try:
            text = config_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "Program-level feedback from CaP-X" in text:
            counts["program_feedback_prompts"] += 1
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python capx/envs/scripts/snapshot_libero_suite_parallel_progress.py "
            "<suite_parallel_root>"
        )
    main(sys.argv[1])
