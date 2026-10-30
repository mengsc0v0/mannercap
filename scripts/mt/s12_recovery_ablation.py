"""Minimal 2x2: {no-experience direct rule, experience retrieval} x {whole re-run, stage recovery}.

Same requests, same candidate set, same budget, same executor version, same audit for all four cells.

FACTOR 1 -- how the force candidate is chosen
  rule       no experience is read.  The contact-feedback grip's lower target is taken AT FACE VALUE as the
             expected hold force, so candidates are ordered by |target_lo_n - centre of the requested band|,
             preferring a target_hi_n that brackets the band.  This is a direct request -> parameter rule; it is
             stronger than a fixed ascending order because it already uses the request.
  retrieval  the frozen library's measured response (s10_retrieval_trial.order_B), unchanged.

FACTOR 2 -- what happens when the execution guard refuses a grasp candidate
  rerun      guard on_refusal="abort": the episode stops and the CALLER re-executes the WHOLE episode from step 0
             with the refused geometry candidate excluded.  No snapshot recovery is used inside the episode.
  recover    guard on_refusal="restore": the episode rolls back to the approach snapshot of that subtask and
             continues.  The discarded prefix is counted as cost.

BUDGET (frozen, identical in all four cells): at most ATTEMPT_BUDGET force candidates per request, at most
GUARD["max_retries"] geometry refusals per episode, at most MAX_EXECUTIONS physical episode executions per request.

COST: every physical execution is counted -- failed force candidates, geometry retries, rolled-back prefixes and
planner probes alike.  Cache hits are counted separately and never presented as executed cost.  An episode is cached
on (task, init, candidate, recovery, frozenset(pre-rejected geometry keys)): the simulator is deterministic, so two
cells that ask for exactly the same execution get the same trajectory.

Usage: s12_recovery_ablation.py TASK INIT...   -> reports/step12_ablation/t{T}_init{I}.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS  # noqa: E402
from s8_task_force import grip_for, per_object  # noqa: E402
from s10_retrieval_trial import E1, LIB, order_B, passed as force_criterion  # noqa: E402
from s11_guarded_case import GUARD, run as run_guarded, windows_from  # noqa: E402
from synthesis_v2.contact_policy_v1 import POLICY_VERSION, evaluate as eval_v1  # noqa: E402

OUT = ROOT / "reports" / "step12_ablation"
CAND = [tuple(c) for c in E1["candidates"]]
OTHER = E1["scope"]["other_object_fixed_plan"]
SETTLE = E1["executor"]["settle_after_open"]
ATTEMPT_BUDGET = int(E1["attempt_budget"])
MAX_EXECUTIONS = ATTEMPT_BUDGET * (1 + int(GUARD["max_retries"]))

CACHE: dict = {}
STATS = {"executed": 0, "cache_hits": 0}


CACHE_IDENTITY = (E1["content_hash"], LIB["content_hash"], POLICY_VERSION,
                  tuple(sorted((k, str(v)) for k, v in GUARD.items())), ATTEMPT_BUDGET)


def order_rule(band):
    """No-experience rule: take the commanded lower target as the expected hold force."""
    centre = 0.5 * (band[0] + band[1])
    rows = [{"cand": c, "fixed_rank": i, "brackets": bool(c[1] >= band[1]),
             "distance": abs(c[0] - centre)} for i, c in enumerate(CAND)]
    rows.sort(key=lambda r: (0 if r["brackets"] else 1, r["distance"], r["fixed_rank"]))
    return rows


def execute(task, init, cand, recovery, pre_rejected):
    """One physical episode (or a cache hit), judged by the force criterion and contact policy v1."""
    # Everything the episode outcome depends on belongs in the key.  The frozen identity (library, request set,
    # guard, contact policy) is part of it so an entry can never outlive the configuration that produced it;
    # ``pre_rejected`` is part of it because it changes which geometry candidate the approach actually picks.
    key = (CACHE_IDENTITY, task, init, cand, recovery,
           tuple(sorted((k, tuple(tuple(x) for x in v)) for k, v in pre_rejected.items())))
    if key in CACHE:
        STATS["cache_hits"] += 1
        return {**CACHE[key], "reused_cached_episode": True}
    roles = [r for r, _ in SPECS[task].objects]
    req_role = [r for r, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    grips = {r: grip_for((cand[0], cand[1]) if r == req_role else (OTHER["target_lo_n"], OTHER["target_hi_n"]), SETTLE)
             for r in roles}
    guard = {**GUARD, "on_refusal": ("abort" if recovery == "rerun" else "restore"), "pre_rejected": pre_rejected}
    prog, ep, rec, crec, det, dt = run_guarded(task, init, grips, guard)
    o = per_object(ep, prog, task, rec)[req_role]
    stages = [str(x) for x in ep.arrays["stage"]]
    v0 = rec["contact_audit"]
    v1 = eval_v1(v0, crec.events, det.detail, windows_from(ep, task), stages)
    g = ep.meta.get("execution_guard", {})
    shaped = {"task_ok": bool(ep.task_ok),
              "measured": {k: o.get(k) for k in ("hold_n", "hold_min", "hold_mean", "hold_max", "lr_imbalance",
                                                 "rel_z_spread_m", "peak_simultaneous_n")},
              "validity": {"status": o.get("status"), "measurement_valid": o.get("measurement_valid"),
                           "grasp_stable": o.get("grasp_stable")}}
    out = {"candidate": list(cand), "status": ep.status, "task_ok": bool(ep.task_ok),
           "n_steps": int(ep.n_steps), "wall_s": round(dt, 2), **shaped,
           "contact_v0_clean": bool(v0["clean"]), "contact_v1_clean": bool(v1["clean_v1"]),
           "v1_unexcused": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"],
                             "stages": c.get("stages"), "reasons": c["v1_reasons"]} for c in v1["unexcused_contacts"]],
           "v1_support_unexplained": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"]}
                                      for c in v1["support_contacts_unexplained"]],
           "guard_refused": [{"check": r["check"], "control_step": r["control_step"],
                              "refused_candidate": r.get("refused_candidate"),
                              "approach_instance": r.get("approach_instance"),
                              "aborted_episode": bool(r.get("aborted_episode"))} for r in g.get("retries", [])],
           "guard_chosen": g.get("chosen_candidate", {}),
           "rolled_back_cost": rec["rolled_back_cost"],
           # the one ground truth for physical cost: every substep this call simulated, the planner's settled
           # look-ahead and every branch a guard retry threw away included.  ``n_steps`` is neither -- it omits the
           # discarded branches and, after a recovery, includes a prefix that was copied in, not re-simulated.
           "executed_substeps": int(rec["substep_cost"]["executed_substeps"]),
           "formal_rows": int(rec["substep_cost"]["formal_rows"]),
           "discarded_contact_rows": det.discarded_report()["branches"],
           "reused_cached_episode": False}
    CACHE[key] = out
    STATS["executed"] += 1
    return out


def run_request(task, init, req, selection, recovery):
    band = req["band_n"]
    obj = [n for _, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    t0 = time.time()
    ranking = order_B(obj, task, band) if selection == "retrieval" else order_rule(band)
    order = [tuple(r["cand"]) for r in ranking]
    select_s = round(time.time() - t0, 4)

    executions, n_exec = [], 0
    joint_ok_at = None
    for k, cand in enumerate(order[:ATTEMPT_BUDGET]):
        pre_rejected: dict = {}
        while True:
            if n_exec >= MAX_EXECUTIONS:
                break
            e = execute(task, init, cand, recovery, pre_rejected)
            e["force_attempt"] = k + 1
            e["force_band_passed"] = force_criterion(e, band)
            e["joint_passed"] = bool(e["force_band_passed"] and e["contact_v1_clean"])
            executions.append(e)
            n_exec += 1
            if e["status"].startswith("guard_refused_abort"):
                # whole re-run arm: carry the refusal forward and execute the WHOLE episode again
                r = e["guard_refused"][-1]
                owner = r["approach_instance"]
                pre_rejected = {kk: list(vv) for kk, vv in pre_rejected.items()}
                pre_rejected.setdefault(owner, []).append(list(r["refused_candidate"]))
                continue
            break
        if executions and executions[-1]["joint_passed"]:
            joint_ok_at = k + 1
            break
        if n_exec >= MAX_EXECUTIONS:
            break
    last = executions[-1] if executions else {}
    return {"schema": "mannercap.ablation_2x2.v1", "e1_hash": E1["content_hash"], "library_hash": LIB["content_hash"],
            "policy_version": POLICY_VERSION, "guard": GUARD,
            "selection": selection, "recovery": recovery, "task": task, "init": init, "request": req, "object": obj,
            "budget": {"force_candidates": ATTEMPT_BUDGET, "geometry_retries_per_episode": GUARD["max_retries"],
                       "max_physical_executions": MAX_EXECUTIONS},
            "order": [list(c) for c in order], "selection_seconds": select_s,
            "executions": executions, "n_physical_executions": len(executions),
            "n_actually_executed": sum(0 if e["reused_cached_episode"] else 1 for e in executions),
            "n_cache_hits": sum(1 for e in executions if e["reused_cached_episode"]),
            "joint_passed_at_force_attempt": joint_ok_at,
            "task_passed": bool(last.get("task_ok")), "force_band_passed": bool(last.get("force_band_passed")),
            "contact_v1_passed": bool(last.get("contact_v1_clean")), "joint_passed": bool(joint_ok_at is not None),
            "control_steps_used": sum(e["n_steps"] for e in executions),
            "discarded_substeps": sum(e["rolled_back_cost"]["substeps"] for e in executions),
            # two books.  "spent" = what this experiment actually paid, a cache hit costing nothing.
            # "method" = what this configuration costs on its own, charging every cache hit its recorded cost.
            "substeps_spent": sum(0 if e["reused_cached_episode"] else e["executed_substeps"] for e in executions),
            "substeps_method": sum(e["executed_substeps"] for e in executions),
            "wall_spent_s": round(sum(0.0 if e["reused_cached_episode"] else e["wall_s"] for e in executions), 2),
            "wall_method_s": round(sum(e["wall_s"] for e in executions), 2)}


def main() -> int:
    task = int(sys.argv[1])
    inits = [int(x) for x in sys.argv[2:]]
    assert all(i in E1["scope"]["inits"] for i in inits), "inits are frozen in E1"
    OUT.mkdir(parents=True, exist_ok=True)
    for i in inits:
        cells = []
        for selection in ("rule", "retrieval"):
            for recovery in ("rerun", "recover"):
                for req in E1["manner_requests"]:
                    r = run_request(task, i, req, selection, recovery)
                    cells.append(r)
                    print(f"T{task} i{i} {selection:9s}/{recovery:7s} {req['id']} band={req['band_n']} "
                          f"joint={r['joint_passed']} task={r['task_passed']} force={r['force_band_passed']} "
                          f"contact={r['contact_v1_passed']} exec={r['n_physical_executions']} "
                          f"(real {r['n_actually_executed']}, cached {r['n_cache_hits']}) "
                          f"sub_spent={r['substeps_spent']} sub_method={r['substeps_method']} "
                          f"wall_spent={r['wall_spent_s']}s wall_method={r['wall_method_s']}s", flush=True)
        (OUT / f"t{task}_init{i}.json").write_text(json.dumps(
            {"schema": "mannercap.ablation_2x2_grid.v1", "task": task, "init": i,
             "executed_episodes_total": STATS["executed"], "cache_hits_total": STATS["cache_hits"],
             "cells": cells}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
