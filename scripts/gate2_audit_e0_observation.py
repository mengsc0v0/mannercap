"""Reconcile an E0 observation plan, raw results, and method ledgers offline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthesis_v2.semantic_audit import (
    audit_phase_semantics,
    require_valid_phase_semantics,
)


class ObservationAuditError(ValueError):
    """Raised when an E0 observation artifact violates its audit contract."""


def _require(condition, message: str) -> None:
    """Enforce an audit invariant even when Python runs with ``-O``."""
    if not condition:
        raise ObservationAuditError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit(root):
    plan = json.loads((root / "plan.json").read_text())
    _require(
        digest({k: v for k, v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"],
        "plan_sha256 does not match plan contents",
    )
    report = json.loads((root / "report.json").read_text())
    expected = {}
    for item in plan["candidates"]:
        key = f'{item["request"]["request_id"]}-{item["method"]}-{item["index"]}'
        _require(key not in expected, f"duplicate candidate key: {key}")
        expected[key] = item
    observed = {p.parent.name: json.loads(p.read_text()) for p in root.glob("episodes/*/result.json")}
    _require(set(expected) == set(observed), "candidate/result episode set mismatch")
    _require(
        {r["episode_id"] for r in report["results"]} == set(expected),
        "report/result episode set mismatch",
    )
    groups = defaultdict(set)
    rows_seen = set()
    totals = {}
    for method in sorted({item['method'] for item in plan['candidates']}):
        rows = [json.loads(line) for line in (root / f"{method}_attempts.jsonl").read_text().splitlines()]
        request_attempts = defaultdict(list)
        resources = Counter()
        success = 0
        for row in rows:
            key = row["metadata"]["method_attempt_id"]
            _require(key not in rows_seen, f"duplicate ledger method_attempt_id: {key}")
            rows_seen.add(key)
            _require(key in expected and key in observed, f"unknown ledger key: {key}")
            item, raw = expected[key], observed[key]
            _require(item["method"] == method, f"{key}: ledger method mismatch")
            _require(row["request_snapshot"] == item["request"], f"{key}: request snapshot changed")
            _require(row["candidate_plan"] == item["proposal"], f"{key}: candidate plan changed")
            for field, expected_value in item["request"].items():
                _require(
                    raw.get(field) == expected_value,
                    f"{key}: executed record changed request field {field!r}",
                )
            _require(
                raw.get("candidate_preset") == item["proposal"].get("candidate_preset"),
                f"{key}: executed candidate preset changed",
            )
            _require(
                digest(row["request_snapshot"]) == row["request_snapshot_hash"],
                f"{key}: request snapshot hash mismatch",
            )
            _require(row["attempt_index"] == item["index"], f"{key}: attempt index mismatch")
            request_attempts[row["request_id"]].append(row["attempt_index"])
            _require(
                row["resources"]["physics_substeps"] == raw["observed_execution_physics_substeps"],
                f"{key}: physics substep accounting mismatch",
            )
            _require(
                row["resources"]["physics_substeps"] <= plan["physics_substep_cap_per_attempt"],
                f"{key}: physics substep cap exceeded",
            )
            _require(
                row["resources"]["api_calls"] == row["resources"]["api_tokens"] == 0,
                f"{key}: unexpected API resource charge",
            )
            _require(row["resources"]["wall_time_s"] == raw["wall_s"], f"{key}: wall-time mismatch")
            _require(row["resources"]["cpu_time_s"] == raw["cpu_time_s"], f"{key}: CPU-time mismatch")
            _require(raw["status"] == "executed", f"{key}: candidate did not execute")
            _require(
                raw["offline_recomputation"]["integration_check"] == "pass",
                f"{key}: offline integration check failed",
            )
            _require(raw["offline_recomputation"]["timing_check_passed"], f"{key}: timing check failed")
            _require(
                row["metadata"]["validation"]["publishable"] is False,
                f"{key}: development observation marked publishable",
            )
            _require(
                row["metadata"]["validation"]["manner"] == "unknown",
                f"{key}: manner status was promoted from unknown",
            )
            _require(
                row["metadata"]["validation"]["safety"] == "unknown",
                f"{key}: safety status was promoted from unknown",
            )
            if row["attempt_index"] == plan["max_attempts"]:
                _require(
                    row["request_state_after"] == "budget_exhausted",
                    f"{key}: final attempt did not exhaust budget",
                )
            groups[raw["task"]].add((raw["initial_state_sha256"], raw["compiled_model_sha256"]))
            resources.update(row["resources"])
            success += raw["task_completed"] is True
        _require(
            set(request_attempts) == {r["request_id"] for r in plan["requests"]},
            f"{method}: request coverage mismatch",
        )
        _require(
            all(
                sorted(indices) == list(range(1, plan["max_attempts"] + 1))
                for indices in request_attempts.values()
            ),
            f"{method}: attempt indices are incomplete or duplicated",
        )
        totals[method] = {"attempts": len(rows), "teacher_task_successes": success, "resources": dict(resources)}
    _require(rows_seen == set(expected), "ledger rows do not cover all planned candidates")
    neutral_phase_checks = 0
    semantic_audits = {}
    if plan.get('phase_semantics') == 'explicit_neutral_v2':
        # Compare applied commands with the same-task fixed neutral baseline,
        # rather than merely checking the requested constraint representation.
        neutral = {}
        for key, item in expected.items():
            if item['method'] == 'G0':
                neutral[item['request']['task']] = observed[key]['executed_parameters']
        task_names = {item['request']['task'] for item in expected.values()}
        _require(set(neutral) == task_names, "explicit neutral semantics lacks a G0 baseline per task")
        for key, item in expected.items():
            semantic = audit_phase_semantics(
                item['proposal']['candidate_preset'],
                observed[key]['executed_parameters'],
                request=item['request'],
                neutral_parameters=neutral[item['request']['task']],
                phase_metrics=observed[key].get('descriptive_phase_metrics'),
            )
            require_valid_phase_semantics(semantic, context=key)
            semantic_audits[key] = semantic
            neutral_phase_checks += sum(
                phase.get('scope') == 'explicit_neutral'
                for phase in semantic['phases'].values()
            )
    _require(
        all(
            len(hashes) == 1 and all(all(value for value in pair) for pair in hashes)
            for hashes in groups.values()
        ),
        "initial-state/model hash mismatch within a task group",
    )
    _require(report["api_calls"] == report["accepted"] == 0, "unexpected API call or accepted candidate")
    return {"schema": "mannercap-e0-reconciliation-v1", "valid": True,
            "candidate_count": len(expected), "plan_sha256": plan["plan_sha256"],
            "methods": totals, "exact_initial_state_groups": len(groups),
            "explicit_neutral_parameter_checks": neutral_phase_checks,
            "phase_semantic_audits": semantic_audits,
            "api_calls": 0, "accepted": 0, "promotion_allowed": False,
            "scope": "development_execution_and_accounting_only",
            "limitations": report["cost_limitations"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
