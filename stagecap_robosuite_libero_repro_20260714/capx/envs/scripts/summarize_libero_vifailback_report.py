"""Summarize a LIBERO CaP-X + ViFailback batch report."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main(report_path: str) -> None:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(path)

    report = json.loads(path.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = report.get("runs", [])
    planned = _planned_task_count(report, runs)
    summary = report.get("summary", {})

    print(f"Report: {path.resolve()}")
    print(f"Runs: {len(runs)}/{planned}")
    print(
        "Tasks succeeded: "
        f"{summary.get('task_success_count', 0)}/{len(runs)} "
        f"(max-round exhausted: {summary.get('max_rounds_exhausted_count', 0)})"
    )
    print(
        "Trials: "
        f"{summary.get('successful_trial_count', 0)} success / "
        f"{summary.get('executed_trial_count', 0)} executed / "
        f"{summary.get('failed_trial_count', 0)} failed"
    )
    print(
        "ViFailback: "
        f"{summary.get('vifailback_completed_count', 0)} completed / "
        f"{summary.get('vifailback_failed_count', 0)} failed"
    )

    by_suite: dict[str, Counter[str]] = defaultdict(Counter)
    success_rounds: Counter[int] = Counter()
    feedback_phrases: Counter[str] = Counter()
    for run in runs:
        suite = str(run.get("suite_name", "unknown"))
        by_suite[suite]["runs"] += 1
        if run.get("task_success"):
            by_suite[suite]["success"] += 1
            if run.get("success_round") is not None:
                success_rounds[int(run["success_round"])] += 1
        if run.get("max_rounds_exhausted"):
            by_suite[suite]["max_rounds_exhausted"] += 1
        for fb in run.get("feedback_history", []):
            phrase = _extract_low_level_command(str(fb.get("feedback_text", "")))
            if phrase:
                feedback_phrases[phrase] += 1

    if by_suite:
        print("\nBy suite:")
        for suite in sorted(by_suite):
            c = by_suite[suite]
            print(
                f"  {suite}: success {c['success']}/{c['runs']}, "
                f"max-round exhausted {c['max_rounds_exhausted']}"
            )

    if success_rounds:
        print("\nSuccess round distribution:")
        for round_idx in sorted(success_rounds):
            print(f"  round {round_idx}: {success_rounds[round_idx]}")

    if feedback_phrases:
        print("\nTop ViFailback low-level commands:")
        for phrase, count in feedback_phrases.most_common(10):
            print(f"  {count}x {phrase}")


def _planned_task_count(report: dict[str, Any], runs: list[dict[str, Any]]) -> int:
    planned = report.get("planned_task_count")
    if isinstance(planned, int) and planned > 0:
        return planned
    suites = report.get("suites") or []
    if suites:
        return max(len(runs), len(suites) * 10)
    return len(runs)


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
            "Usage: python capx/envs/scripts/summarize_libero_vifailback_report.py "
            "<libero_cap_vifailback_report.json>"
        )
    main(sys.argv[1])
