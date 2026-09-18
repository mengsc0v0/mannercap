"""Summarize paired D0 development trajectories without freezing thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "min": min(items),
        "max": max(items),
    }


def load_results(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    keys: set[tuple[str, int, str]] = set()
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("episodes/*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "executed":
                raise ValueError(f"non-executed result in development input: {path}")
            key = (str(result["task"]), int(result["seed"]), str(result["manner_tag"]))
            if key in keys:
                raise ValueError(f"duplicate D0 condition: {key}")
            keys.add(key)
            results.append(result)
    if not results:
        raise ValueError("no D0 results found")
    return results


def _metric(result: dict[str, Any], phase: str | None = None) -> float:
    if phase is None:
        return float(result["descriptive_metrics"]["mean_joint_speed"])
    return float(result["descriptive_phase_metrics"][phase]["mean_joint_speed_rad_s"])


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(item["task"]), int(item["seed"]), str(item["manner_tag"])): item
        for item in results
    }
    cells: list[dict[str, Any]] = []
    for task, tag in sorted({(key[0], key[2]) for key in by_key}):
        group = [
            item for (item_task, _, item_tag), item in by_key.items()
            if item_task == task and item_tag == tag
        ]
        phases = sorted(set.intersection(*[
            set(item["descriptive_phase_metrics"]) for item in group
        ]))
        cells.append({
            "task": task,
            "manner_tag": tag,
            "episode_count": len(group),
            "task_success_count": sum(bool(item["task_completed"]) for item in group),
            "mean_joint_speed_rad_s": numeric_summary(_metric(item) for item in group),
            "phase_mean_joint_speed_rad_s": {
                phase: numeric_summary(_metric(item, phase) for item in group)
                for phase in phases
            },
        })

    comparisons = {
        "slow_overall_vs_neutral": [],
        "fast_overall_vs_neutral": [],
        "mixed_approach_vs_neutral": [],
        "mixed_descend_vs_neutral": [],
        "mixed_lift_vs_neutral": [],
    }
    pairs: list[dict[str, Any]] = []
    for task, seed in sorted({(key[0], key[1]) for key in by_key}):
        neutral = by_key[(task, seed, "neutral")]
        slow = by_key[(task, seed, "slow")]
        fast = by_key[(task, seed, "fast")]
        mixed = by_key[(task, seed, "fast_approach_slow_contact")]
        ratios = {
            "slow_overall_vs_neutral": _metric(slow) / _metric(neutral),
            "fast_overall_vs_neutral": _metric(fast) / _metric(neutral),
            "mixed_approach_vs_neutral": _metric(mixed, "approach") / _metric(neutral, "approach"),
            "mixed_descend_vs_neutral": _metric(mixed, "descend") / _metric(neutral, "descend"),
            "mixed_lift_vs_neutral": _metric(mixed, "lift") / _metric(neutral, "lift"),
        }
        for name, value in ratios.items():
            comparisons[name].append(value)
        pairs.append({"task": task, "seed": seed, "ratios": ratios})

    direction_checks = {
        "slow_overall_below_neutral": all(value < 1.0 for value in comparisons["slow_overall_vs_neutral"]),
        "fast_overall_above_neutral": all(value > 1.0 for value in comparisons["fast_overall_vs_neutral"]),
        "mixed_approach_above_neutral": all(value > 1.0 for value in comparisons["mixed_approach_vs_neutral"]),
        "mixed_descend_below_neutral": all(value < 1.0 for value in comparisons["mixed_descend_vs_neutral"]),
        "mixed_lift_below_neutral": all(value < 1.0 for value in comparisons["mixed_lift_vs_neutral"]),
    }
    return {
        "schema": "mannercap-d0-development-summary-v1",
        "episode_count": len(results),
        "task_success_count": sum(bool(item["task_completed"]) for item in results),
        "api_calls": sum(int(item.get("api_calls", 0)) for item in results),
        "cells": cells,
        "paired_ratios": pairs,
        "paired_ratio_summary": {
            name: numeric_summary(values) for name, values in comparisons.items()
        },
        "direction_checks": direction_checks,
        "all_direction_checks_pass": all(direction_checks.values()),
        "formal_thresholds_frozen": False,
        "promotion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(load_results(path.resolve() for path in args.run_dirs))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "episode_count": report["episode_count"],
        "all_direction_checks_pass": report["all_direction_checks_pass"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
