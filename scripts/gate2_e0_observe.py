"""Execute preregistered G0-G3 development candidates; M is explicitly skipped.

This observation runner cannot publish data: independent calibration is pending.
It uses a separate AttemptLedger per method and forks every candidate from an
unstepped environment shared across all methods for each task/seed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gate2_capture_manner_d0 import _create_pair_environment, _run_episode, _write_json
from synthesis_v2.d0_manner import build_plan
from synthesis_v2.attempt_ledger import AttemptLedger
from synthesis_v2.e0_proposers import propose


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def plan(seed, attempts):
    requests = build_plan(seeds=[seed])["episodes"]
    for request in requests:
        request["request_id"] = "e0-dev-" + request["episode_id"]
    candidates = []
    for request in requests:
        for method in ("G0", "G1", "G2", "G3"):
            for index in range(1, attempts + 1):
                proposal = propose(request, method, index)
                candidates.append({"request": request, "method": method, "index": index,
                                   "proposal": proposal})
    result = {"schema": "mannercap-e0-development-observation-plan-v1",
              "seed": seed, "requests": requests, "candidates": candidates,
              "max_attempts": attempts, "physics_substep_cap_per_attempt": 150000,
              "wall_cap_per_attempt_s": 180, "api_budget": 0,
              "purpose": "execution_and_accounting_engineering_observation",
              "formal_thresholds_frozen": False, "promotion_allowed": False,
              "context_attribute_interventions": False,
              "M": {"status": "skipped", "reason": "api_not_configured",
                    "requests": len(requests), "charged_attempts": 0}}
    result["plan_sha256"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cached-m-proposals", type=Path)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    planned = plan(args.seed, args.attempts)
    if args.include_baselines and not args.cached_m_proposals:
        raise ValueError("include-baselines requires cached M proposals")
    methods = ("G0", "G1", "G2", "G3")
    if args.cached_m_proposals:
        if args.attempts != 1:
            raise ValueError("cached M smoke requires exactly one attempt")
        cached = json.loads(args.cached_m_proposals.read_text())
        by_id = {item["request_id"]: item for item in cached}
        if len(by_id) != len(cached) or set(by_id) != {r["request_id"] for r in planned["requests"]}:
            raise ValueError("cached proposals must cover all requests exactly once")
        methods = ("G0", "G1", "G2", "G3", "M") if args.include_baselines else ("M",)
        if not args.include_baselines:
            planned["candidates"] = []
        for request in planned["requests"]:
            item = by_id[request["request_id"]]
            planned["candidates"].append({"request": request, "method": "M", "index": 1,
                "proposal": {**item, "proposal_source": "cached_external_llm",
                             "request_sha256": digest(request)}})
        planned["M"] = {"status": "cached_proposals", "online_calls_this_execution": 0,
                        "prior_api_calls": len(cached)}
        planned["cached_proposals_sha256"] = digest(cached)
        planned["phase_semantics"] = "explicit_neutral_v2"
        planned.pop("plan_sha256")
        planned["plan_sha256"] = digest(planned)
    if (output / "plan.json").exists():
        if json.loads((output / "plan.json").read_text()) != planned:
            raise ValueError("existing plan differs from requested plan")
        if (output / "episodes").exists():
            raise ValueError("execution output exists; partial runs cannot resume")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty")
        _write_json(output / "plan.json", planned)
    if not args.run:
        print(json.dumps({"planned_candidates": len(planned["candidates"]), "plan_sha256": planned["plan_sha256"]}))
        return
    if not hasattr(os, "fork"):
        raise RuntimeError("real execution requires POSIX fork")
    _write_json(output / "run_manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "plan_sha256": planned["plan_sha256"], "api_calls": 0})
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")
    ledgers = {method: AttemptLedger(planned["requests"], max_attempts=args.attempts)
               for method in methods}
    results = []
    groups = defaultdict(list)
    for candidate in planned["candidates"]:
        groups[candidate["request"]["task"]].append(candidate)
    for task, candidates in groups.items():
        base = _create_pair_environment(candidates[0]["request"])
        reference = None
        try:
            for candidate in candidates:
                request, method, index = candidate["request"], candidate["method"], candidate["index"]
                proposal = candidate["proposal"]
                episode_id = f'{request["request_id"]}-{method}-{index}'
                record = {**request, "episode_id": episode_id, "method_id": method,
                          "phase_semantics": planned.get("phase_semantics"),
                          "candidate_preset": proposal["candidate_preset"],
                          "physics_substep_cap": planned["physics_substep_cap_per_attempt"]}
                result_path = output / "episodes" / episode_id / "result.json"
                sys.stdout.flush()
                pid = os.fork()
                if pid == 0:
                    with (output / f"{episode_id}.log").open("w") as log:
                        os.dup2(log.fileno(), 1)
                        os.dup2(log.fileno(), 2)
                        _run_episode(record, output, base_env=base, pair_reference=reference)
                    os._exit(0)
                started = time.monotonic()
                while True:
                    waited, status = os.waitpid(pid, os.WNOHANG)
                    if waited:
                        break
                    if time.monotonic() - started > planned["wall_cap_per_attempt_s"]:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                        break
                    time.sleep(0.1)
                if not result_path.exists():
                    # Preserve partial evidence and refuse to invent unmeasured cost.
                    _write_json(output / "interruption.json", {"episode_id": episode_id,
                        "reason": "child_timeout_or_crash", "wall_s": time.monotonic() - started,
                        "resource_measurement_incomplete": True})
                    raise RuntimeError("child produced no result; partial evidence preserved")
                result = json.loads(result_path.read_text())
                if reference is None and (result_path.parent / "model.mjb").exists():
                    reference = result_path.parent
                decision = {"task": "pass" if result.get("task_completed") is True else
                            "fail" if result.get("task_completed") is False else "unknown",
                            "manner": "unknown", "safety": "unknown", "overall": "unknown",
                            "publishable": False, "reason": "independent_calibration_pending"}
                if decision["task"] == "fail":
                    decision["overall"] = "fail"
                size = sum(p.stat().st_size for p in result_path.parent.rglob("*") if p.is_file())
                usage = {"physics_substeps": result["observed_execution_physics_substeps"],
                         "api_calls": 0, "api_tokens": 0, "gpu_time_s": 0,
                         "cpu_time_s": result["cpu_time_s"], "wall_time_s": result["wall_s"],
                         "storage_bytes": size}
                item = ledgers[method].record_attempt(request["request_id"],
                    outcome="execution_failure" if result["status"] != "executed" else "validation_failure",
                    proposal_source=proposal["proposal_source"], candidate_plan=proposal,
                    call_record={"called": False, "api_calls": 0}, resources=usage,
                    reason="execution_error" if result["status"] != "executed" else "independent_calibration_pending",
                    request_snapshot=request, metadata={"method_id": method,
                        "method_attempt_id": episode_id, "validation": decision,
                        "result_ref": str(result_path), "resource_scope": "candidate_execution_only"})
                with (output / f"{method}_attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(item.to_dict()) + "\n")
                results.append({"episode_id": episode_id, "method": method,
                    "status": result["status"], "task_completed": result["task_completed"],
                    "initial_state_sha256": result.get("initial_state_sha256"),
                    "compiled_model_sha256": result.get("compiled_model_sha256"),
                    "task": task, "resources": usage, "validation": decision})
                _write_json(output / "progress.json", {"completed": len(results), "planned": len(planned["candidates"])})
                print(f"completed {len(results)}/{len(planned['candidates'])}: {episode_id}", flush=True)
        finally:
            base.robosuite_env.close()
    pairs = {}
    for task in groups:
        subset = [r for r in results if r["task"] == task]
        hashes = {(r["initial_state_sha256"], r["compiled_model_sha256"]) for r in subset}
        pairs[task] = len(hashes) == 1 and all(None not in pair for pair in hashes)
    report = {"schema": "mannercap-e0-development-observation-report-v1", "results": results,
        "executed_candidates": len(results), "execution_errors": sum(r["status"] != "executed" for r in results),
        "teacher_task_successes": sum(r["task_completed"] is True for r in results),
        "exact_initial_state_across_methods_and_attempts": pairs,
        "accepted": 0, "api_calls": 0, "M": planned["M"],
        "formal_thresholds_frozen": False, "promotion_allowed": False,
        "cost_limitations": "setup, parent orchestration, and log storage excluded from per-candidate vectors"}
    _write_json(output / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
