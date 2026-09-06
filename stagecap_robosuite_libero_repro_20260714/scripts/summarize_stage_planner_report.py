"""Summarize a LIBERO stage-planner experiment report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro


@dataclass
class Args:
    report_path: str


def main(args: Args) -> None:
    report = json.loads(Path(args.report_path).read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = report.get("runs", [])
    print(f"Report: {Path(args.report_path).resolve()}")
    print(f"Summary: {json.dumps(report.get('summary', {}), ensure_ascii=False)}")
    print("")
    for run in runs:
        rounds = run.get("rounds", [])
        local_signal_count = sum(
            len(item.get("local_success_signals", [])) for item in rounds
        )
        local_success_env_fail = bool(
            run.get("local_success_but_env_failed")
            or (local_signal_count > 0 and not run.get("task_success"))
        )
        round_status = ", ".join(
            f"r{item.get('round')}:{item.get('stop_reason') or item.get('status')}"
            for item in rounds
        )
        print(
            f"{run.get('suite_name')}/{run.get('task_id'):02d} "
            f"success={bool(run.get('task_success'))} "
            f"local_success_env_fail={local_success_env_fail} "
            f"forbidden={run.get('forbidden_symbol_violation_count', 0)} "
            f"trials={run.get('trial_count', 0)} "
            f"rounds=[{round_status}]"
        )
        refs = run.get("success_code_references", [])
        if refs:
            clean_refs = sum(1 for ref in refs if ref.get("clean_reference"))
            print(f"  historical_refs={len(refs)} clean_refs={clean_refs}")
        for item in rounds:
            skills = item.get("skill_usage", [])
            local = item.get("local_success_signals", [])
            if skills or local:
                print(
                    f"  round {item.get('round')}: "
                    f"skills={len(skills)} local_success_signals={len(local)}"
                )


if __name__ == "__main__":
    main(tyro.cli(Args))
