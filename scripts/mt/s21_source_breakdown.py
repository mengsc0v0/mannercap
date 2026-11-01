"""Offline split of the step-20 results by candidate source (library support type); no simulation, nothing tuned.
Arms: rule+recover and retrieval+recover (retrieval+rerun repeats the retrieval+recover decisions 32/32 and is not counted twice).
'support' = scripts/mt/s20_selection_eval.candidate_source: direct_measured (lo on a measured grid point with matching hi),
interpolated (lo strictly between measured grid points), clamped_extrapolated (outside the grid or hi not matched).
The predicted interval [pred_min, pred_max] is NOT a statistical interval: it is the min of recorded hold_min / max of recorded
hold_max over the library records at the neighbouring lo grid points (linearly interpolated per end point).  'covered' below means the
WHOLE measured hold range [hold_min, hold_max] lies inside it.
Run: PYTHONPATH=.:scripts/mt python scripts/mt/s21_source_breakdown.py"""
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s20_analyze as A  # noqa: E402

ARMS2 = ["rule+recover", "retrieval+recover"]
SUP = ["direct_measured", "interpolated", "clamped_extrapolated"]


def mean(x):
    return round(float(np.mean(x)), 3) if len(x) else None


def med(x):
    return round(float(np.median(x)), 3) if len(x) else None


def block(rows):
    """rows: list of dict(att, req, measured, pred, first, passed_final, pid)"""
    meas = [r for r in rows if r["m"]["hold_min"] is not None]
    e_min = [r["m"]["hold_min"] - r["p"]["pred_min"] for r in meas]
    e_max = [r["m"]["hold_max"] - r["p"]["pred_max"] for r in meas]
    cov_all = [r["m"]["hold_min"] >= r["p"]["pred_min"] and r["m"]["hold_max"] <= r["p"]["pred_max"] for r in meas]
    cov_mean = [r["p"]["pred_min"] <= r["m"]["hold_mean"] <= r["p"]["pred_max"] for r in meas]
    pred_in = [r["p"]["pred_min"] >= r["lo_b"] and r["p"]["pred_max"] <= r["hi_b"] for r in rows]
    return {
        "attempts": len(rows), "unique_physical_runs": len({r["pid"] for r in rows}),
        "attempts_with_valid_measurement": len(meas),
        "force_band_passed": sum(r["force"] for r in rows), "joint_passed": sum(r["joint"] for r in rows),
        "first_candidate_attempts": sum(r["first"] for r in rows),
        "first_candidate_force_passed": sum(r["first"] and r["force"] for r in rows),
        "first_candidate_joint_passed": sum(r["first"] and r["joint"] for r in rows),
        "final_qualifying_attempt_of_a_request": sum(r["final_ok"] for r in rows),
        "pred_inside_request_band": sum(pred_in),
        "pred_inside_and_force_passed": sum(p and r["force"] for p, r in zip(pred_in, rows)),
        "pred_outside_but_force_passed": sum((not p) and r["force"] for p, r in zip(pred_in, rows)),
        "whole_hold_range_covered": f"{sum(cov_all)}/{len(meas)}", "hold_mean_inside_pred_interval": f"{sum(cov_mean)}/{len(meas)}",
        "signed_err_hold_min_n_mean": mean(e_min), "signed_err_hold_max_n_mean": mean(e_max),
        "abs_err_hold_min_n_mean_median": [mean(np.abs(e_min)), med(np.abs(e_min))],
        "abs_err_hold_max_n_mean_median": [mean(np.abs(e_max)), med(np.abs(e_max))],
    }


allrows = {a: [] for a in ARMS2}
for a in ARMS2:
    for k in A.KEYS:
        c = A.C[(a,) + k]
        lo_b, hi_b = c["request"]["band_n"]
        passed_idx = next((i for i, at in enumerate(c["attempts"]) if at["joint_passed"]), None)
        for i, at in enumerate(c["attempts"]):
            ex = at["executions"][-1]
            allrows[a].append({"arm": a, "key": k, "sup": at["library_prediction"]["support"], "p": at["library_prediction"],
                               "m": ex["measured"], "first": at["force_attempt"] == 1, "force": bool(at["force_band_passed"]),
                               "joint": bool(at["joint_passed"]), "final_ok": passed_idx == i, "pid": ex["physical_run_id"].rsplit("-", 2)[0],
                               "cand": tuple(at["candidate"]), "lo_b": lo_b, "hi_b": hi_b, "cluster": (k[0], k[1])})
# unique physical-run id: task-init-candidate (drop arm / preset suffix)
for a in ARMS2:
    for r in allrows[a]:
        r["pid"] = (r["key"][0], r["key"][1], r["cand"])

res = {"schema": "mannercap.source_breakdown.v1", "arms_counted": ARMS2,
       "denominator_note": ("unit = one force attempt of one request in one arm; the same physical run (task, init, candidate) can be "
                            "counted several times (R1..R4 share cached executions, and the two arms can pick the same candidate); "
                            "'unique_physical_runs' dedupes by (task, init, candidate) within the row group"),
       "interval_note": "pred interval = library envelope (min hold_min / max hold_max at neighbouring grid lo values); not calibrated, not a confidence interval",
       "by_arm_and_support": {}, "pooled_over_arms_by_support": {}, "unique_runs_pooled_by_support": {}}
for a in ARMS2:
    res["by_arm_and_support"][a] = {s: block([r for r in allrows[a] if r["sup"] == s]) for s in SUP}
pooled = allrows[ARMS2[0]] + allrows[ARMS2[1]]
for s in SUP:
    res["pooled_over_arms_by_support"][s] = block([r for r in pooled if r["sup"] == s])
    # dedupe by physical run (prediction depends only on object/task/lo/hi, so one row per run is well defined)
    seen, ded = set(), []
    for r in pooled:
        if r["sup"] == s and r["pid"] not in seen:
            seen.add(r["pid"]); ded.append(r)
    res["unique_runs_pooled_by_support"][s] = block(ded)
# per-cluster coverage of interpolated attempts (8 clusters) to show it is not one cluster driving the number
cl = collections.defaultdict(lambda: [0, 0])
for r in pooled:
    if r["sup"] == "interpolated" and r["m"]["hold_min"] is not None:
        cl[f"T{r['cluster'][0]}-i{r['cluster'][1]}"][1] += 1
        cl[f"T{r['cluster'][0]}-i{r['cluster'][1]}"][0] += int(r["m"]["hold_min"] >= r["p"]["pred_min"] and r["m"]["hold_max"] <= r["p"]["pred_max"])
res["interpolated_coverage_by_cluster"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(cl.items())}
# does ranking help beyond the prediction accuracy?  attempts needed to first FORCE pass, per arm (requests that ever passed)
att = {}
for a in ARMS2:
    n_pass = [next(i + 1 for i, x in enumerate(A.C[(a,) + k]["attempts"]) if x["force_band_passed"])
              for k in A.KEYS if any(x["force_band_passed"] for x in A.C[(a,) + k]["attempts"])]
    att[a] = {"requests_with_force_pass": len(n_pass), "mean_attempts_to_first_force_pass": mean(n_pass),
              "distribution": dict(sorted(collections.Counter(n_pass).items())),
              "requests_never_force_passed": 32 - len(n_pass)}
res["attempts_to_first_force_pass"] = att
# source mix of what each arm tried FIRST
res["first_candidate_source_mix"] = {a: dict(collections.Counter(r["sup"] for r in allrows[a] if r["first"])) for a in ARMS2}
# order evidence that does not need a prediction to be right: the same 32 requests, the first candidate's outcome
res["first_candidate_outcome_by_arm"] = {a: {"force_passed": sum(r["force"] for r in allrows[a] if r["first"]),
                                             "joint_passed": sum(r["joint"] for r in allrows[a] if r["first"]), "n": 32} for a in ARMS2}
out = A.OUT / "source_breakdown.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("attempts_to_first_force_pass", "first_candidate_source_mix", "first_candidate_outcome_by_arm", "interpolated_coverage_by_cluster")}))
for grp in ("by_arm_and_support",):
    for a in ARMS2:
        for s in SUP:
            b = res[grp][a][s]
            print(a, s, {k: b[k] for k in ("attempts", "unique_physical_runs", "first_candidate_attempts", "first_candidate_joint_passed", "final_qualifying_attempt_of_a_request", "force_band_passed", "whole_hold_range_covered", "hold_mean_inside_pred_interval", "abs_err_hold_min_n_mean_median", "abs_err_hold_max_n_mean_median", "pred_inside_request_band", "pred_inside_and_force_passed", "pred_outside_but_force_passed")})
for s in SUP:
    print("POOLED", s, {k: res["pooled_over_arms_by_support"][s][k] for k in ("attempts", "unique_physical_runs", "whole_hold_range_covered", "hold_mean_inside_pred_interval", "signed_err_hold_min_n_mean", "signed_err_hold_max_n_mean")})
    print("UNIQUE", s, {k: res["unique_runs_pooled_by_support"][s][k] for k in ("attempts", "whole_hold_range_covered", "hold_mean_inside_pred_interval", "abs_err_hold_min_n_mean_median", "abs_err_hold_max_n_mean_median")})
