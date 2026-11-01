"""Step 3 (2026-11-01): three-arm selection evaluation on fresh initial states (frozen by configs/selection_eval_e2.json).

arms (all share ONE budget implementation, ``run_request``; only ``recovery`` changes what the executor does on a refusal):
  rule+recover       order_rule      / in-episode restore to the approach snapshot
  retrieval+recover  order_B         / in-episode restore
  retrieval+rerun    order_B         / abort the episode, re-execute the WHOLE episode with the refusal carried forward
budget b2 (identical for every arm): at most ``attempt_budget`` force candidates per request; within one force candidate
the guard may refuse at most ``max_retries`` times and still be retried -- the (max_retries+1)-th refusal ends the
candidate as ``guard_retry_budget_exhausted``.  The count is cumulative over reruns and restarts per force candidate.
usage: s20_selection_eval.py TASK INIT"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from s7_task_config import ROOT, SPECS  # noqa: E402
from s8_task_force import grip_for, per_object  # noqa: E402
from s10_retrieval_trial import E1, LIB, order_B, predict, passed as force_criterion  # noqa: E402
from s11_guarded_case import GUARD, run as run_guarded, windows_from  # noqa: E402
from s12_recovery_ablation import order_rule  # noqa: E402
from synthesis_v2.contact_policy_v1 import POLICY_VERSION, evaluate as eval_v1  # noqa: E402

CFG_PATH = ROOT / "configs" / "selection_eval_e2.json"
CFG = json.loads(CFG_PATH.read_text())
OUT = ROOT / "reports" / "step20_selection_eval"
OTHER = E1["scope"]["other_object_fixed_plan"]
SETTLE = E1["executor"]["settle_after_open"]
ATTEMPT_BUDGET = int(CFG["budget"]["force_candidates_per_request"])
MAX_RETRIES = int(CFG["budget"]["guard_retries_per_force_candidate"])
MAX_EXEC_DERIVED = ATTEMPT_BUDGET * (1 + MAX_RETRIES)
CACHE: dict = {}
STATS = {"executed": 0, "cache_hits": 0}
CACHE_IDENTITY = (E1["content_hash"], LIB["content_hash"], POLICY_VERSION,
                  tuple(sorted((k, str(v)) for k, v in GUARD.items())))


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg_hash():
    return hashlib.sha256(json.dumps({k: v for k, v in CFG.items() if k != "content_hash"}, sort_keys=True).encode()).hexdigest()


def verify_freeze():
    """Refuse to run if any frozen file differs from the committed freeze manifest."""
    man = json.loads((OUT / "FREEZE_MANIFEST.json").read_text())
    bad = [f for f, h in man["code_sha256"].items() if sha256_file(ROOT / f) != h]
    if cfg_hash() != CFG["content_hash"]:
        bad.append("configs/selection_eval_e2.json(content_hash)")
    if LIB["content_hash"] != man["experience_library"]["content_hash"]:
        bad.append("library content_hash")
    assert GUARD["max_retries"] == MAX_RETRIES and POLICY_VERSION == man["contact_policy_version"]
    if bad:
        raise SystemExit(f"FREEZE VIOLATED, refusing to run: {bad}")
    return man


def candidate_source(obj, task, cand):
    """Where the library's prediction for this candidate comes from: a directly measured (lo, hi) pair, an interpolation
    between measured lo values, or a clamped extrapolation / hi-unmatched fallback.  Informational for every arm."""
    lo, hi = cand
    p = predict(obj, task, lo, hi)
    grid, same_hi = p["lo_grid"], p["basis"].endswith("+same_hi")
    if same_hi and lo in grid:
        kind = "direct_measured"
    elif len(grid) >= 2 and grid[0] < lo < grid[-1]:
        kind = "interpolated"
    else:
        kind = "clamped_extrapolated"
    return {"support": kind, "pred_min": p["pred_min"], "pred_max": p["pred_max"], "basis": p["basis"],
            "n_records_used": p["n_records_used"], "lo_grid": grid, "hi_matched": same_hi}


def ranking_rows(obj, task, band, selection):
    rows = order_B(obj, task, band) if selection == "retrieval" else order_rule(band)
    out = []
    for rank, r in enumerate(rows):
        c = tuple(r["cand"])
        out.append({"rank": rank, "candidate": list(c), "fixed_rank": r["fixed_rank"],
                    **{k: r[k] for k in ("inside", "margin", "centre_gap", "brackets", "distance") if k in r},
                    "library_prediction": candidate_source(obj, task, c)})
    return out


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def _save_traj(path, ep, rec):
    arrs = {}
    for k, v in ep.arrays.items():
        a = np.asarray(v)
        if a.dtype == object or "image" in k or "rgb" in k or "depth" in k or a.ndim >= 3:
            if a.dtype == object:
                arrs["arr_" + k] = a.astype(str)
            continue
        arrs["arr_" + k] = a
    arrs["substep_ctrl"] = np.asarray(rec["ctrl"])
    for n in sorted(rec["L"]):
        arrs[f"force_L_{n}"] = np.asarray(rec["L"][n]); arrs[f"force_R_{n}"] = np.asarray(rec["R"][n])
    np.savez_compressed(path, **arrs)


def execute(task, init, cand, recovery, pre_rejected):
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
    retries = json.loads(json.dumps(g.get("retries", []), default=float))
    sc = rec["substep_cost"]
    disc = int(rec["rolled_back_cost"]["substeps"])
    guard_disc = int(sum(r.get("discarded_substeps_by_this_restore", 0) for r in retries))
    last_restore_cum = max([r.get("executed_substeps_before_restore", 0) for r in retries], default=0)
    force_stage = next((i.instance_id for i in prog.instances if i.primitive == "grasp" and i.object_role == req_role), None)
    n_pre = sum(len(v) for v in pre_rejected.values())
    phys_id = f"T{task}-i{init}-c{cand[0]:g}_{cand[1]:g}-{recovery}-pre{n_pre}"
    traj = f"traj/{phys_id}.npz"
    (OUT / "traj").mkdir(parents=True, exist_ok=True)
    _save_traj(OUT / traj, ep, rec)
    out = {"physical_run_id": phys_id, "trajectory_file": traj, "candidate": list(cand), "status": ep.status,
           "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "wall_s": round(dt, 2),
           "measured": {k: o.get(k) for k in ("hold_n", "hold_min", "hold_mean", "hold_max", "lr_imbalance",
                                              "rel_z_spread_m", "peak_simultaneous_n")},
           "validity": {"status": o.get("status"), "measurement_valid": o.get("measurement_valid"),
                        "grasp_stable": o.get("grasp_stable")},
           "contact_v0_clean": bool(v0["clean"]), "contact_v1_clean": bool(v1["clean_v1"]),
           "v1_unexcused": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"],
                             "stages": c.get("stages"), "reasons": c["v1_reasons"]} for c in v1["unexcused_contacts"]],
           "v1_support_unexplained": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": c["max_n"]}
                                      for c in v1["support_contacts_unexplained"]],
           "guard_events": retries, "guard_chosen": g.get("chosen_candidate", {}),
           "initial_state_digest": _digest(ep.meta.get("initial_state", {})),
           "parameter_effective_stage": {"force_candidate": force_stage,
                                         "geometry_rejection": sorted({r.get("approach_instance") for r in retries if r.get("approach_instance")})},
           "substeps": {"executed_total": int(sc["executed_substeps"]), "formal_rows": int(sc["formal_rows"]),
                        "discarded_total_listed": disc, "discarded_by_guard_restore": guard_disc,
                        "discarded_planner_lookahead": disc - guard_disc,
                        "executed_after_last_restore": int(sc["executed_substeps"]) - int(last_restore_cum) if retries and last_restore_cum else None,
                        "prefix_rows_not_resimulated": int(sum(r.get("prefix_rows_kept", 0) for r in retries)),
                        "identity_executed_eq_formal_plus_discarded": bool(int(sc["executed_substeps"]) == int(sc["formal_rows"]) + disc)},
           "control_steps": {"n_steps": int(ep.n_steps)},
           "discarded_contact_rows": det.discarded_report()["branches"], "reused_cached_episode": False}
    CACHE[key] = out
    STATS["executed"] += 1
    return out


def run_request(task, init, req, selection, recovery):
    arm = f"{selection}+{recovery}"
    band = req["band_n"]
    obj = [n for _, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    rid = f"T{task}-i{init}-{req['id']}"
    t0 = time.time()
    ranking = ranking_rows(obj, task, band, selection)
    order = [tuple(r["candidate"]) for r in ranking]
    select_s = round(time.time() - t0, 4)
    attempts, n_exec, joint_ok_at = [], 0, None
    for k, cand in enumerate(order[:ATTEMPT_BUDGET]):
        src = next(r["library_prediction"] for r in ranking if tuple(r["candidate"]) == cand)
        at = {"attempt_id": f"{rid}/{arm}/a{k + 1}", "force_attempt": k + 1, "candidate": list(cand),
              "library_prediction": src, "executions": [], "refusals_in_attempt": 0, "end_reason": None}
        pre_rejected: dict = {}
        while True:
            e = execute(task, init, cand, recovery, pre_rejected)
            e = {**e, "execution_id": f"{at['attempt_id']}/x{len(at['executions']) + 1}",
                 "attempt_id": at["attempt_id"], "request_id": rid, "arm": arm, "force_attempt": k + 1}
            e["force_band_passed"] = bool(force_criterion(e, band))
            e["joint_passed"] = bool(e["force_band_passed"] and e["contact_v1_clean"])
            m = e["measured"]
            e["prediction_vs_actual"] = {"pred_min": src["pred_min"], "pred_max": src["pred_max"],
                                         "actual_hold_min": m["hold_min"], "actual_hold_max": m["hold_max"],
                                         "predicted_inside_band": bool(src["pred_min"] >= band[0] and src["pred_max"] <= band[1])}
            at["executions"].append(e)
            n_exec += 1
            if e["status"].startswith("guard_refused_abort"):
                at["refusals_in_attempt"] += 1
                if at["refusals_in_attempt"] > MAX_RETRIES:
                    at["end_reason"] = "guard_retry_budget_exhausted"
                    break
                r = e["guard_events"][-1]
                pre_rejected = {kk: list(vv) for kk, vv in pre_rejected.items()}
                pre_rejected.setdefault(r["approach_instance"], []).append(list(r["refused_candidate"]))
                continue
            if recovery == "recover":                      # restores of the one episode; the rerun counter already holds its refusals
                at["refusals_in_attempt"] = len(e["guard_events"])
            at["end_reason"] = "guard_retry_budget_exhausted" if e["status"].startswith("guard_retry_budget_exhausted") else e["status"]
            break
        last = at["executions"][-1]
        at["joint_passed"] = bool(last["joint_passed"])
        at["force_band_passed"] = bool(last["force_band_passed"])
        at["task_ok"] = bool(last["task_ok"])
        at["substeps_method"] = sum(x["substeps"]["executed_total"] for x in at["executions"])
        at["wall_method_s"] = round(sum(x["wall_s"] for x in at["executions"]), 2)
        attempts.append(at)
        if at["joint_passed"]:
            joint_ok_at = k + 1
            break
    execs = [x for a in attempts for x in a["executions"]]
    audit = {"force_candidates": len(attempts), "executions": len(execs),
             "max_refusals_in_one_attempt": max(a["refusals_in_attempt"] for a in attempts),
             "within_bounds": bool(len(attempts) <= ATTEMPT_BUDGET and len(execs) <= MAX_EXEC_DERIVED
                                   and all(a["refusals_in_attempt"] <= MAX_RETRIES + 1 for a in attempts))}
    return {"request_id": rid, "arm": arm, "selection": selection, "recovery": recovery, "task": task, "init": init,
            "request": req, "object": obj, "ranking": ranking, "selection_seconds": select_s, "attempts": attempts,
            "budget_audit": audit, "joint_passed": joint_ok_at is not None, "joint_passed_at_force_attempt": joint_ok_at,
            "first_candidate_joint_passed": bool(attempts[0]["joint_passed"]),
            "first_candidate_force_passed": bool(attempts[0]["force_band_passed"]),
            "n_physical_executions": len(execs), "n_actually_executed": sum(0 if x["reused_cached_episode"] else 1 for x in execs),
            "n_cache_hits": sum(1 for x in execs if x["reused_cached_episode"]),
            "substeps_method": sum(x["substeps"]["executed_total"] for x in execs),
            "substeps_spent": sum(0 if x["reused_cached_episode"] else x["substeps"]["executed_total"] for x in execs),
            "wall_method_s": round(sum(x["wall_s"] for x in execs), 2)}


def main() -> int:
    task, init = int(sys.argv[1]), int(sys.argv[2])
    assert task in CFG["scope"]["tasks"] and init in CFG["scope"]["inits"][str(task)], "task/init are frozen in the config"
    man = verify_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    launch = {"git_head": head, "freeze_code_commit": man.get("code_commit"), "config_hash": CFG["content_hash"],
              "started_unix": time.time(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cells = []
    for phase in CFG["phases"]:
        for req in E1["manner_requests"]:
            for selection, recovery in phase["arms"]:
                r = run_request(task, init, req, selection, recovery)
                cells.append(r)
                print(f"T{task} i{init} {selection}+{recovery} {req['id']} first={r['first_candidate_joint_passed']} "
                      f"joint={r['joint_passed']} exec={r['n_physical_executions']} (real {r['n_actually_executed']}) "
                      f"sub_method={r['substeps_method']} wall_method={r['wall_method_s']}s", flush=True)
        (OUT / f"t{task}_init{init}.json").write_text(json.dumps(
            {"schema": "mannercap.selection_eval.v2", "launch": launch, "task": task, "init": init,
             "phases_done": phase["name"], "executed_episodes_total": STATS["executed"],
             "cache_hits_total": STATS["cache_hits"], "finished_unix": time.time(), "cells": cells}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
