"""Audit deterministic Wipe under/nominal/over-action probe directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
    initial = json.loads((directory / "initial_state.json").read_text(encoding="utf-8"))
    if probe.get("schema") != "mannercap-gate1-wipe-probe-v1":
        raise ValueError(f"unexpected probe schema: {directory}")
    return probe, initial


def _case(case_id: str, directory: Path) -> dict[str, Any]:
    probe, initial = _load(directory)
    attempt = probe.get("wipe_attempt", {})
    final = attempt.get("final_marker_status", {})
    trace = probe.get("trace_summary", {})
    attempt_results = attempt.get("attempt_results", [])
    return {
        "case_id": case_id,
        "source_directory": str(directory.resolve()),
        "model_sha256": initial.get("model_sha256"),
        "surface_penetration_m": probe.get("surface_penetration_m"),
        "status": probe.get("status"),
        "task_operable": probe.get("task_operable"),
        "task_completed": final.get("task_completed"),
        "wiped_markers": final.get("wiped_markers"),
        "total_markers": final.get("total_markers"),
        "final_reward": final.get("reward"),
        "control_transitions": trace.get("control_transitions"),
        "max_contact_pair_force_n": trace.get("max_named_contact_force_n"),
        "max_eef_force_n": trace.get("max_eef_force_n"),
        "max_eef_torque_nm": trace.get("max_eef_torque_nm"),
        "eef_wrench_qualification": trace.get("eef_wrench_qualification"),
        "execution_error_recorded": any(bool(item.get("error")) for item in attempt_results),
    }


def audit(under_dir: Path, nominal_dir: Path, over_dir: Path) -> dict[str, Any]:
    cases = [
        _case("under_action_zero_penetration", under_dir),
        _case("nominal_two_mm_penetration", nominal_dir),
        _case("over_action_ten_mm_penetration", over_dir),
    ]
    under, nominal, over = cases
    same_model = len({case["model_sha256"] for case in cases}) == 1
    penetration_ordered = [case["surface_penetration_m"] for case in cases] == [0.0, 0.002, 0.01]
    eef_force_ordered = (
        under["max_eef_force_n"] < nominal["max_eef_force_n"] < over["max_eef_force_n"]
    )
    expected_outcomes = (
        under["task_completed"] is False
        and nominal["task_completed"] is True
        and over["task_completed"] is False
        and over["execution_error_recorded"] is True
        and float(over["final_reward"]) < 0.0
    )
    return {
        "schema": "mannercap-gate1-wipe-calibration-audit-v1",
        "qualification": "characterized_not_calibrated",
        "task_outcome_cases_verified": bool(same_model and penetration_ordered and expected_outcomes),
        "penetration_response_characterized": bool(same_model and penetration_ordered and eef_force_ordered),
        "wrench_calibration_verified": False,
        "promotion_allowed": False,
        "checks": {
            "same_model": same_model,
            "penetration_ordered": penetration_ordered,
            "eef_force_monotonic_for_three_probes": eef_force_ordered,
            "expected_under_nominal_over_outcomes": expected_outcomes,
        },
        "cases": cases,
        "limitations": [
            "RoboSuite end-effector wrench is not calibrated against an independent controlled load.",
            "Per-contact maxima come from different trajectories and are not a penetration response curve.",
            "The 2 mm nominal penetration is a development probe parameter, not an approved safety threshold.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--under", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--over", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.under, args.nominal, args.over)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "task_outcome_cases_verified": result["task_outcome_cases_verified"],
        "penetration_response_characterized": result["penetration_response_characterized"],
        "wrench_calibration_verified": result["wrench_calibration_verified"],
        "promotion_allowed": result["promotion_allowed"],
    }))
    return 0 if result["task_outcome_cases_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

