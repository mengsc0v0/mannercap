"""Reconcile an E0 observation plan, raw results, and method ledgers offline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def audit(root):
    plan = json.loads((root / "plan.json").read_text())
    assert digest({k: v for k, v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"]
    report = json.loads((root / "report.json").read_text())
    expected = {}
    for item in plan["candidates"]:
        key = f'{item["request"]["request_id"]}-{item["method"]}-{item["index"]}'
        assert key not in expected
        expected[key] = item
    observed = {p.parent.name: json.loads(p.read_text()) for p in root.glob("episodes/*/result.json")}
    assert set(expected) == set(observed)
    assert {r["episode_id"] for r in report["results"]} == set(expected)
    groups = defaultdict(set)
    rows_seen = set()
    totals = {}
    for method in ("G0", "G1", "G2", "G3"):
        rows = [json.loads(line) for line in (root / f"{method}_attempts.jsonl").read_text().splitlines()]
        request_attempts = defaultdict(list)
        resources = Counter()
        success = 0
        for row in rows:
            key = row["metadata"]["method_attempt_id"]
            assert key not in rows_seen
            rows_seen.add(key)
            item, raw = expected[key], observed[key]
            assert item["method"] == method
            assert row["request_snapshot"] == item["request"]
            assert row["candidate_plan"] == item["proposal"]
            assert digest(row["request_snapshot"]) == row["request_snapshot_hash"]
            assert row["attempt_index"] == item["index"]
            request_attempts[row["request_id"]].append(row["attempt_index"])
            assert row["resources"]["physics_substeps"] == raw["observed_execution_physics_substeps"]
            assert row["resources"]["physics_substeps"] <= plan["physics_substep_cap_per_attempt"]
            assert row["resources"]["api_calls"] == row["resources"]["api_tokens"] == 0
            assert row["resources"]["wall_time_s"] == raw["wall_s"]
            assert row["resources"]["cpu_time_s"] == raw["cpu_time_s"]
            assert raw["status"] == "executed"
            assert raw["offline_recomputation"]["integration_check"] == "pass"
            assert raw["offline_recomputation"]["timing_check_passed"]
            assert row["metadata"]["validation"]["publishable"] is False
            assert row["metadata"]["validation"]["manner"] == "unknown"
            assert row["metadata"]["validation"]["safety"] == "unknown"
            if row["attempt_index"] == plan["max_attempts"]:
                assert row["request_state_after"] == "budget_exhausted"
            groups[raw["task"]].add((raw["initial_state_sha256"], raw["compiled_model_sha256"]))
            resources.update(row["resources"])
            success += raw["task_completed"] is True
        assert set(request_attempts) == {r["request_id"] for r in plan["requests"]}
        assert all(sorted(indices) == list(range(1, plan["max_attempts"] + 1)) for indices in request_attempts.values())
        totals[method] = {"attempts": len(rows), "teacher_task_successes": success, "resources": dict(resources)}
    assert rows_seen == set(expected)
    assert all(len(hashes) == 1 and all(all(v for v in pair) for pair in hashes) for hashes in groups.values())
    assert report["api_calls"] == report["accepted"] == 0
    return {"schema": "mannercap-e0-reconciliation-v1", "valid": True,
            "candidate_count": len(expected), "plan_sha256": plan["plan_sha256"],
            "methods": totals, "exact_initial_state_groups": len(groups),
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
