#!/usr/bin/env python3
"""Populate synthesis_v2.experience_library from historical trial jsonl files.

Scans manner_ablation_*.jsonl (and any other trial-record jsonl) produced by
run_manner_ablation.py, and for every trial with outcome == "both_pass" (the
task succeeded AND the recorded manner constraint was satisfied), flattens
its `params` dict into per-(task, phase, dimension, level) experience
samples that the calibration layer (Phase 3) can later query.

Known, deliberate scope limitation: this script does NOT reconstruct
MANNER_REPAIR_LIBRARY "repair" records (before/after value pairs) from
historical data, because run_single_trial() only persists the *final*
(post-adjustment) params to jsonl -- the pre-adjustment params are never
saved, so a historical "before" value would have to be reverse-engineered
from the hardcoded _adjust_* ratio constants, which is fragile and would
duplicate logic that may drift. Repair records are instead populated going
forward via live write-back inside run_single_trial (Phase 5), where both
the before and after params are directly available in the same process.

Pure post-processing: never runs a simulator, never calls an LLM.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthesis_v2.experience_library import ExperienceLibrary, flatten_trial_params


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", default=None,
                    help="Glob pattern(s) for trial jsonl files. Default: "
                         "repro_results/robosuite/*.jsonl")
    p.add_argument("--out", type=str,
                    default="repro_results/robosuite/experience_library.jsonl",
                    help="Output experience-library jsonl path (overwritten).")
    return p.parse_args()


def _iter_trial_records(paths: list[str]):
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"    [skip] {path}:{line_no} malformed json: {e}")
        except OSError as e:
            print(f"    [skip] cannot read {path}: {e}")


def build(paths: list[str]) -> tuple[ExperienceLibrary, dict]:
    lib = ExperienceLibrary()
    stats = {"scanned": 0, "both_pass": 0, "records_written": 0,
              "per_task": {}}
    for path, rec in _iter_trial_records(paths):
        stats["scanned"] += 1
        if rec.get("outcome") != "both_pass":
            continue
        task = rec.get("task")
        tag = rec.get("tag")
        seed = rec.get("seed", -1)
        run_stamp = rec.get("run_stamp", "")
        params = rec.get("params") or {}
        if not task or not tag or not params:
            continue
        stats["both_pass"] += 1
        flat = flatten_trial_params(params)
        for dimension, per_phase in flat.items():
            for phase, value in per_phase.items():
                lib.add_experience(
                    task=task, phase=phase, dimension=dimension, level=tag,
                    value=value, outcome="both_pass",
                    source_run_stamp=run_stamp, seed=seed,
                )
                stats["records_written"] += 1
                stats["per_task"][task] = stats["per_task"].get(task, 0) + 1
    return lib, stats


def main() -> int:
    args = parse_args()
    patterns = args.inputs or ["repro_results/robosuite/*.jsonl"]
    paths = sorted({
        p for pat in patterns for p in glob.glob(pat)
    })
    # never re-ingest the experience library's own output file if it lives
    # inside the same glob pattern
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(args.out)]
    if not paths:
        print("No input files matched; nothing to do.")
        return 1
    print(f"Scanning {len(paths)} jsonl file(s)...")
    lib, stats = build(paths)
    lib.save(args.out)
    print(f"Scanned {stats['scanned']} trial records, "
          f"{stats['both_pass']} both_pass, "
          f"wrote {stats['records_written']} experience records "
          f"-> {args.out}")
    print("Per-task record counts:", stats["per_task"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
