"""Offline analysis of the step-20 three-arm run (no simulation, nothing is tuned).
Pre-declared in configs/selection_eval_e2.json: pair-level and cluster-level reported separately, cluster bootstrap
(8 clusters, 10000 resamples, seed 20261101), method-book cost, recover vs rerun agreement incl. full trajectory comparison,
relative-budget analysis, interpolation reliability, median-ranking sensitivity (offline only).
Run with the libero venv:  PYTHONPATH=.:scripts/mt python scripts/mt/s20_analyze.py
"""
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mt"))
OUT = ROOT / "reports" / "step20_selection_eval"
ARMS = ["rule+recover", "retrieval+recover", "retrieval+rerun"]
SEED, NBOOT = 20261101, 10000

cells = []
for f in sorted(glob.glob(str(OUT / "t*_init*.json"))):
    cells += json.loads(Path(f).read_text())["cells"]
C = {(c["arm"], c["task"], c["init"], c["request"]["id"]): c for c in cells}
KEYS = sorted({(c["task"], c["init"], c["request"]["id"]) for c in cells})
CLUSTERS = sorted({(c["task"], c["init"]) for c in cells})
assert len(KEYS) == 32 and len(CLUSTERS) == 8 and all((a,) + k in C for a in ARMS for k in KEYS)


def last_exec(c):
    return c["attempts"][-1]["executions"][-1]


def cluster_vals(arm, fn):
    out = []
    for cl in CLUSTERS:
        v = [fn(C[(arm,) + cl + (r,)]) for r in ("R1", "R2", "R3", "R4")]
        out.append(sum(v) / len(v))
    return out


def boot(stat_fn):
    rng = np.random.default_rng(SEED)
    st = [stat_fn(rng.integers(0, len(CLUSTERS), len(CLUSTERS))) for _ in range(NBOOT)]
    return [round(float(np.percentile(st, 2.5)), 4), round(float(np.percentile(st, 97.5)), 4)]


def rate_block(fn):
    res = {}
    for a in ARMS:
        pair = [int(fn(C[(a,) + k])) for k in KEYS]
        cl = np.array(cluster_vals(a, lambda c: int(fn(c))))
        res[a] = {"pair_level": f"{sum(pair)}/{len(pair)}", "pair_rate": round(sum(pair) / len(pair), 4),
                  "cluster_rates": [round(float(x), 3) for x in cl], "cluster_mean": round(float(cl.mean()), 4),
                  "cluster_bootstrap_95ci": boot(lambda i, cl=cl: cl[i].mean())}
    return res


def diff_block(fn, a, b):
    ca = np.array(cluster_vals(a, lambda c: int(fn(c)))); cb = np.array(cluster_vals(b, lambda c: int(fn(c))))
    d = ca - cb
    pa = [int(fn(C[(a,) + k])) for k in KEYS]; pb = [int(fn(C[(b,) + k])) for k in KEYS]
    disc = collections.Counter(zip(pa, pb))
    return {"a_minus_b": f"{a} - {b}", "pair_diff": sum(pa) - sum(pb), "pair_discordant": {"a_only": disc[(1, 0)], "b_only": disc[(0, 1)]},
            "cluster_mean_diff": round(float(d.mean()), 4), "cluster_bootstrap_95ci": boot(lambda i: d[i].mean()),
            "clusters_a_better_b_better_same": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]}


res = {"schema": "mannercap.selection_eval_analysis.v1", "n_pairs_per_arm": len(KEYS), "n_clusters": len(CLUSTERS),
       "clusters": [f"T{t}-i{i}" for t, i in CLUSTERS],
       "limits": ["8 clusters (2 tasks x 4 inits) only; cluster CIs are wide",
                  "only 2 of 8 pool candidates are directly measured in the library; the rest are interpolated / clamped",
                  "same task and same requested object as the library only; one stage window",
                  "guard and contact policy were tuned on inits 26-27 (development); this run uses fresh inits 30-33",
                  "the rule arm is a face-value rule, not a calibrated lookup; the median lookup is only an offline sensitivity",
                  "no claim of 'better than calibration': same type of information, not algorithmically equivalent"]}

res["first_candidate_force_pass"] = rate_block(lambda c: c["first_candidate_force_passed"])
res["first_candidate_joint_pass"] = rate_block(lambda c: c["first_candidate_joint_passed"])
res["final_joint_pass_within_budget"] = rate_block(lambda c: c["joint_passed"])
res["diff_first_candidate_force_retrieval_minus_rule"] = diff_block(lambda c: c["first_candidate_force_passed"], "retrieval+recover", "rule+recover")
res["diff_first_candidate_joint_retrieval_minus_rule"] = diff_block(lambda c: c["first_candidate_joint_passed"], "retrieval+recover", "rule+recover")
res["diff_final_joint_retrieval_minus_rule"] = diff_block(lambda c: c["joint_passed"], "retrieval+recover", "rule+recover")
res["diff_final_joint_recover_minus_rerun_retrieval"] = diff_block(lambda c: c["joint_passed"], "retrieval+recover", "retrieval+rerun")
res["per_task"] = {f"T{t}": {a: {"first_force": sum(C[(a,) + k]["first_candidate_force_passed"] for k in KEYS if k[0] == t),
                                 "first_joint": sum(C[(a,) + k]["first_candidate_joint_passed"] for k in KEYS if k[0] == t),
                                 "final_joint": sum(C[(a,) + k]["joint_passed"] for k in KEYS if k[0] == t),
                                 "n": sum(1 for k in KEYS if k[0] == t)} for a in ARMS} for t in (0, 7)}
res["per_request_id"] = {r: {a: {"first_force": sum(C[(a,) + k]["first_candidate_force_passed"] for k in KEYS if k[2] == r),
                                 "final_joint": sum(C[(a,) + k]["joint_passed"] for k in KEYS if k[2] == r),
                                 "n": sum(1 for k in KEYS if k[2] == r)} for a in ARMS} for r in ("R1", "R2", "R3", "R4")}
why = {}
for a in ARMS:
    w = collections.Counter()
    for k in KEYS:
        c = C[(a,) + k]
        if c["joint_passed"]:
            continue
        e = last_exec(c)
        w["force_band_not_reached" if not c["attempts"][-1]["force_band_passed"] else
          ("contact_v1_not_clean" if not e["contact_v1_clean"] else "other")] += 1
    why[a] = dict(w)
res["non_pass_reason_of_last_attempt"] = why

ctc = {}
for a in ARMS:
    w = collections.Counter()
    for k in KEYS:
        c = C[(a,) + k]
        e = last_exec(c)
        if not c["joint_passed"] and c["attempts"][-1]["force_band_passed"] is not None and not e["contact_v1_clean"]:
            for u in e["v1_unexcused"]:
                w[f'{u["robot_body"]}|{u["other"]}|{"/".join(u["stages"])}'] += 1
    ctc[a] = dict(w)
res["unexcused_contact_in_last_execution_of_unpassed_requests"] = ctc
res["joint_zero_clusters"] = [f"T{t}-i{i}" for t, i in CLUSTERS if all(not C[("retrieval+recover", t, i, r)]["joint_passed"] for r in ("R1", "R2", "R3", "R4"))]

cost = {}
for a in ARMS:
    cs = [C[(a,) + k] for k in KEYS]
    sub = [c["substeps_method"] for c in cs]
    wl = [c["wall_method_s"] for c in cs]
    ok = sum(c["joint_passed"] for c in cs)
    cost[a] = {"method_substeps_total": sum(sub), "method_substeps_mean_per_request": round(statistics.mean(sub)),
               "method_substeps_median_per_request": statistics.median(sub), "wall_method_s_total": round(sum(wl), 1),
               "executions_total": sum(c["n_physical_executions"] for c in cs), "force_candidates_total": sum(len(c["attempts"]) for c in cs),
               "joint_passes": ok, "method_substeps_per_joint_pass": round(sum(sub) / ok) if ok else None,
               "spent_substeps_total_order_dependent": sum(c["substeps_spent"] for c in cs),
               "cache_hits": sum(c["n_cache_hits"] for c in cs), "includes_failed_requests": True}
res["cost_method_book"] = cost
rc, rr = cost["retrieval+recover"]["method_substeps_total"], cost["retrieval+rerun"]["method_substeps_total"]
cl_rc = np.array([sum(C[("retrieval+recover",) + cl + (r,)]["substeps_method"] for r in ("R1", "R2", "R3", "R4")) for cl in CLUSTERS], float)
cl_rr = np.array([sum(C[("retrieval+rerun",) + cl + (r,)]["substeps_method"] for r in ("R1", "R2", "R3", "R4")) for cl in CLUSTERS], float)
res["recover_vs_rerun_cost"] = {"saving_fraction_pooled": round(1 - rc / rr, 4),
                                "cluster_savings": [round(float(x), 3) for x in 1 - cl_rc / cl_rr],
                                "cluster_bootstrap_95ci_of_pooled_saving": boot(lambda i: 1 - cl_rc[i].sum() / cl_rr[i].sum()),
                                "wall_saving_fraction_pooled": round(1 - cost["retrieval+recover"]["wall_method_s_total"] / cost["retrieval+rerun"]["wall_method_s_total"], 4)}
res["retrieval_vs_rule_cost_recover"] = {"retrieval_over_rule_ratio": round(rc / cost["rule+recover"]["method_substeps_total"], 4)}

acc = {}
for a in ARMS:
    refus = collections.Counter(at["refusals_in_attempt"] for k in KEYS for at in C[(a,) + k]["attempts"])
    end = collections.Counter(at["end_reason"] for k in KEYS for at in C[(a,) + k]["attempts"])
    acc[a] = {"refusals_per_force_candidate_histogram": dict(sorted(refus.items())), "end_reason": dict(end),
              "requests_within_bounds": sum(C[(a,) + k]["budget_audit"]["within_bounds"] for k in KEYS),
              "max_force_candidates_in_a_request": max(len(C[(a,) + k]["attempts"]) for k in KEYS),
              "max_executions_in_a_request": max(C[(a,) + k]["n_physical_executions"] for k in KEYS),
              "force_candidates_per_request_histogram": dict(sorted(collections.Counter(len(C[(a,) + k]["attempts"]) for k in KEYS).items()))}
res["budget_and_refusals"] = acc
ids_ok = [x["substeps"]["identity_executed_eq_formal_plus_discarded"] for c in cells for at in c["attempts"] for x in at["executions"] if x.get("substeps")]
res["substep_identity_executed_eq_formal_plus_discarded"] = {"checked": len(ids_ok), "all_true": all(ids_ok)}
sim = [ev.get("state_identity_match") for c in cells for at in c["attempts"] for x in at["executions"] for ev in x["guard_events"] if "state_identity_match" in ev]
res["restore_state_identity_match"] = {"checked": len(sim), "all_true": all(sim) if sim else None}


def cand_seq(c):
    return [tuple(at["candidate"]) for at in c["attempts"]]


def hold(c):
    m = last_exec(c)["measured"]
    return m["hold_min"], m["hold_max"]


agree = {"same_candidate_sequence": 0, "same_force_verdicts": 0, "same_final_joint": 0, "same_hold_values": 0, "same_end_reasons": 0,
         "trajectory_identical_all_arrays": 0, "trajectory_compared": 0}
traj_diffs, rows = [], []
for k in KEYS:
    a, b = C[("retrieval+recover",) + k], C[("retrieval+rerun",) + k]
    same_seq = cand_seq(a) == cand_seq(b)
    agree["same_candidate_sequence"] += same_seq
    agree["same_force_verdicts"] += [x["force_band_passed"] for x in a["attempts"]] == [x["force_band_passed"] for x in b["attempts"]]
    agree["same_end_reasons"] += [x["end_reason"] for x in a["attempts"]] == [x["end_reason"] for x in b["attempts"]]
    agree["same_final_joint"] += a["joint_passed"] == b["joint_passed"]
    ha, hb = hold(a), hold(b)
    agree["same_hold_values"] += all(x is not None and y is not None and abs(x - y) < 1e-9 for x, y in zip(ha, hb))
    row = {"key": f"T{k[0]}-i{k[1]}-{k[2]}", "same_candidate_sequence": same_seq, "hold_recover": ha, "hold_rerun": hb,
           "joint_recover": a["joint_passed"], "joint_rerun": b["joint_passed"], "cost_recover": a["substeps_method"], "cost_rerun": b["substeps_method"]}
    if same_seq:
        za = np.load(OUT / last_exec(a)["trajectory_file"], allow_pickle=True)
        zb = np.load(OUT / last_exec(b)["trajectory_file"], allow_pickle=True)
        agree["trajectory_compared"] += 1
        diffs = {}
        for key in sorted(set(za.files) | set(zb.files)):
            if key not in za.files or key not in zb.files:
                diffs[key] = "missing in one"
            elif za[key].shape != zb[key].shape:
                diffs[key] = f"shape {za[key].shape} vs {zb[key].shape}"
            elif za[key].dtype.kind in "fc":
                if not np.array_equal(za[key], zb[key], equal_nan=True):   # nan == nan (hold bands are nan outside the hold)
                    diffs[key] = f"max_abs_diff={float(np.nanmax(np.abs(za[key] - zb[key]))):.3e}"
            elif not np.array_equal(za[key], zb[key]):
                diffs[key] = f"label differs in {int((za[key] != zb[key]).sum())}/{za[key].shape[0]} rows"
        row["n_differing_arrays"] = len(diffs)
        phys = {k: v for k, v in diffs.items() if k != "arr_force_target"}
        agree["trajectory_identical_all_arrays"] += not diffs
        agree["trajectory_identical_excluding_force_target_label"] = agree.get("trajectory_identical_excluding_force_target_label", 0) + (not phys)
        if diffs:
            traj_diffs.append({"key": row["key"], "n_differing_arrays": len(diffs), "diffs": dict(list(diffs.items())[:12])})
    rows.append(row)
res["retrieval_recover_vs_rerun_agreement"] = {"n": len(KEYS), **agree, "non_identical_trajectory_details": traj_diffs,
                                               "rows_not_agreeing": [r for r in rows if not r["same_candidate_sequence"] or r["joint_recover"] != r["joint_rerun"]]}

tot_rerun = cost["retrieval+rerun"]["method_substeps_total"]
budget = {"reference": "retrieval+rerun total METHOD substeps over the same 32 requests", "total_reference_substeps": tot_rerun, "curves": {}}
for pct in (25, 50, 75, 100):
    B = tot_rerun * pct / 100.0
    row = {"budget_substeps": round(B)}
    for a in ARMS:
        cum, q, reached = 0.0, 0, 0
        for k in KEYS:
            c = C[(a,) + k]
            if cum >= B:
                break
            reached += 1
            pass_cost, run = None, 0.0
            for at in c["attempts"]:
                run += at["substeps_method"]
                if at["joint_passed"]:
                    pass_cost = run
                    break
            if pass_cost is not None and cum + pass_cost <= B:
                q += 1
            cum += c["substeps_method"]
        row[a] = {"qualified_requests": q, "requests_started": reached,
                  "cum_substeps_all_32_requests": round(sum(C[(a,) + k]["substeps_method"] for k in KEYS))}
    budget["curves"][f"{pct}%"] = row
budget["rules"] = ("requests processed in (task, init, request id) order; qualified = joint pass reached (attempt granularity) with cumulative cost <= B; "
                   "failed requests consume full cost; post-run analysis, not a re-run")
res["relative_budget_curve"] = budget

interp = collections.defaultdict(lambda: {"n": 0, "pred_inside_request_band": 0, "force_band_passed": 0, "actual_inside_predicted_interval": 0,
                                          "abs_err_min": [], "abs_err_max": [], "joint_passed": 0})
conf = collections.Counter()
for a in ("rule+recover", "retrieval+recover"):
    for k in KEYS:
        c = C[(a,) + k]
        lo, hi = c["request"]["band_n"]
        for at in c["attempts"]:
            lp = at["library_prediction"]
            m = at["executions"][-1]["measured"]
            inside = bool(lp["pred_min"] >= lo and lp["pred_max"] <= hi)
            conf[(inside, bool(at["force_band_passed"]))] += 1
            if m["hold_min"] is None:
                continue
            g = interp[lp["support"]]
            g["n"] += 1
            g["pred_inside_request_band"] += int(inside)
            g["force_band_passed"] += int(at["force_band_passed"])
            g["joint_passed"] += int(at["joint_passed"])
            g["actual_inside_predicted_interval"] += int(m["hold_min"] >= lp["pred_min"] and m["hold_max"] <= lp["pred_max"])
            g["abs_err_min"].append(abs(m["hold_min"] - lp["pred_min"])); g["abs_err_max"].append(abs(m["hold_max"] - lp["pred_max"]))
res["interpolation_reliability"] = {s: {**{k: v for k, v in g.items() if not k.startswith("abs_err")},
                                        "mean_abs_err_hold_min_n": round(statistics.mean(g["abs_err_min"]), 3) if g["abs_err_min"] else None,
                                        "mean_abs_err_hold_max_n": round(statistics.mean(g["abs_err_max"]), 3) if g["abs_err_max"] else None}
                                    for s, g in sorted(interp.items())}
res["interpolation_reliability_retrieval_arm_only"] = {}
_g = collections.defaultdict(lambda: {"n": 0, "force_band_passed": 0, "actual_inside_predicted_interval": 0})
for k in KEYS:
    c = C[("retrieval+recover",) + k]
    for at in c["attempts"]:
        lp, m = at["library_prediction"], at["executions"][-1]["measured"]
        if m["hold_min"] is None:
            continue
        g = _g[lp["support"]]; g["n"] += 1; g["force_band_passed"] += int(at["force_band_passed"])
        g["actual_inside_predicted_interval"] += int(m["hold_min"] >= lp["pred_min"] and m["hold_max"] <= lp["pred_max"])
res["interpolation_reliability_retrieval_arm_only"] = dict(_g)
res["interpolation_reliability_note"] = "attempts of rule+recover and retrieval+recover; a failed interpolated candidate is NOT 'physically unreachable'"
res["predicted_inside_band_vs_actual_force_pass"] = {f"pred_inside={p},actual_pass={q}": n for (p, q), n in sorted(conf.items())}

sens = {"definition": "calibrated lookup, median aggregation (scripts/mt/s18_lookup_equivalence.py:order_lookup), offline only", "rows": []}
try:
    import s18_lookup_equivalence as L
    executed = collections.defaultdict(list)
    for c in cells:
        for at in c["attempts"]:
            executed[(c["task"], c["init"], tuple(at["candidate"]))].append((c["arm"], at["force_band_passed"], at["joint_passed"]))
    same_first = same_order = 0
    for k in KEYS:
        c = C[("retrieval+recover",) + k]
        med = L.order_lookup(k[0], c["object"], tuple(c["request"]["band_n"]))
        env = [tuple(r["candidate"]) for r in c["ranking"]]
        same_first += med[0] == env[0]; same_order += med == env
        ex = executed.get((k[0], k[1], med[0]))
        sens["rows"].append({"key": f"T{k[0]}-i{k[1]}-{k[2]}", "envelope_first": env[0], "median_first": med[0], "same_first": med[0] == env[0],
                             "median_first_outcome": ("not executed" if not ex else {"force_band_passed": ex[0][1], "joint_passed": ex[0][2]})})
    sens["same_first_choice"] = f"{same_first}/{len(KEYS)}"
    sens["same_full_order"] = f"{same_order}/{len(KEYS)}"
    sens["median_first_not_executed"] = sum(1 for r in sens["rows"] if r["median_first_outcome"] == "not executed")
except Exception as e:
    sens["error"] = f"{type(e).__name__}: {e}"
res["median_ranking_sensitivity"] = sens

(OUT / "analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print("OK analysis.json")
