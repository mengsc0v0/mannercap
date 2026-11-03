"""Step 27 audit of the P2 (experience library) / P3 (cause repair) evidence.  Offline, response grid only.

A1 fixed evaluation set: requests on inits i % 4 == 3 are NEVER in the training stream.  The library is trained online on
   the other requests (library+cause, write-back); snapshots after N executed training attempts are evaluated on the
   SAME evaluation set with write-back OFF.  N = 0 is the no-update control (rule prior only).
A2 every request (satisfiable or not): outcome classes per arm.
A3 cost references: oracle-pass (no stop rule, = s25 'oracle'), oracle-min (first passing cfg only, 0 on unsatisfiable).
A4 calibrated-rule baseline: one calibration sweep (30 cells on ONE training init per task, cost charged), then a
   fixed per-(task, cfg) lookup of the measured response; no further write-back.
A5 repair failure breakdown for each failed cause-repair transition.
A6 response dispersion per (task, cfg) across inits.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25_policies as P  # noqa: E402

TASKS = (0, 7, 1)
CFGS = [(f, s) for f in P.FORCES for s in P.SPEEDS]
SNAPS = (0, 10, 25, 50, 100, 200, 400, 800)
BUDGET = 3


def reqs_for(tasks, inits):
    return [(t, i, fb, sb) for t in tasks for i in inits[t] for fb in ("R1", "R2", "R3", "R4")
            for sb in ("slow", "normal", "fast")]


def sat_cfgs(grid, t, i, fb, sb):
    return [c for c in CFGS if P.outcome(grid.get((t, i, P.key(*c))), P.FORCE_BANDS[fb], P.SPEED_BANDS[sb])[0]]


def serve(grid, rank_fn, req, lib=None, repair="cause"):
    t, i, fb, sb = req
    q = {"force": fb, "speed": sb}
    fband, sband = P.FORCE_BANDS[fb], P.SPEED_BANDS[sb]
    rank = rank_fn(t, q)
    cur, tried, causes, cost, first, passed, stop = rank[0], [], [], 0, None, False, None
    while len(tried) < BUDGET and cur is not None:
        e = grid.get((t, i, P.key(*cur)))
        ok, cause, _ = P.outcome(e, fband, sband)
        tried.append(cur); causes.append(cause)
        cost += int(e.get("substeps_executed", 0)) if e and "error" not in e else 0
        if lib is not None:
            lib.add(t, cur[0], cur[1], e)
        first = ok if first is None else first
        if ok:
            passed = True
            break
        if repair == "cause":
            cur = P.cause_repair(cur, cause, q, tried, rank)
            stop = "declared" if cur is None else None
        else:
            cur = next((c for c in rank if c not in tried), None)
    return {"first": bool(first), "passed": passed, "stop": stop, "tried": tried, "causes": causes, "cost": cost}


def force_err(grid, t, i, cfg, fb):
    e = grid.get((t, i, P.key(*cfg)))
    if not e or "error" in e:
        return None
    h = e["objects"][e["requested_role"]].get("hold_mean")
    lo, hi = P.FORCE_BANDS[fb]
    return None if h is None else abs(h - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


def classify(r, sat):
    if r["passed"]:
        return "passed_first" if r["first"] else "passed_after_repair"
    if not sat:
        return "unsat_declared_stop" if r["stop"] else "unsat_budget_spent"
    return "sat_false_stop" if r["stop"] else "sat_budget_exhausted"


def main() -> int:
    grid = P.load_grid(TASKS)
    inits = {t: [i for i in sorted({i for (tt, i, _) in grid if tt == t})
                 if all((t, i, P.key(*c)) in grid and "error" not in grid[(t, i, P.key(*c))] for c in CFGS)]
             for t in TASKS}
    ev_inits = {t: [i for i in inits[t] if i % 4 == 3] for t in TASKS}
    tr_inits = {t: [i for i in inits[t] if i % 4 != 3] for t in TASKS}
    EV, TR = reqs_for(TASKS, ev_inits), reqs_for(TASKS, tr_inits)
    sat = {r: bool(sat_cfgs(grid, *r)) for r in EV + TR}
    out = {"inits": inits, "eval_inits": ev_inits, "n_eval": len(EV), "n_eval_sat": sum(sat[r] for r in EV),
           "n_train": len(TR), "budget": BUDGET}

    # ---- A1 learning curve on a fixed evaluation set
    curves = defaultdict(list)
    for sd in range(10):
        st = TR[:]; random.Random(sd).shuffle(st)
        lib, n_att, k, snaps = P.Library(), 0, 0, {}
        for n in SNAPS:
            while n_att < n and k < len(st):
                r = serve(grid, lambda t, q: P.ranking("library", lib, t, q), st[k], lib=lib)
                n_att += len(r["tried"]); k += 1
            snaps[n] = (list(lib.rows), n_att, k)
        for n, (rows, n_att_real, k_req) in snaps.items():
            fl = P.Library(); fl.rows = rows
            res = [serve(grid, lambda t, q: P.ranking("library", fl, t, q), q) for q in EV]
            s_ = [(q, x) for q, x in zip(EV, res) if sat[q]]
            fe = [force_err(grid, q[0], q[1], x["tried"][0], q[2]) for q, x in s_]
            fe = [v for v in fe if v is not None]
            curves[n].append({"train_attempts": n_att_real, "train_requests": k_req,
                              "first_pass_sat": sum(x["first"] for _, x in s_) / len(s_),
                              "pass_sat": sum(x["passed"] for _, x in s_) / len(s_),
                              "failed_attempts_per_req": sum(len(x["tried"]) - x["passed"] for _, x in s_) / len(s_),
                              "first_force_err": statistics.mean(fe),
                              "honest_stop_unsat": (sum(1 for q, x in zip(EV, res) if not sat[q] and x["stop"])
                                                    / max(1, sum(1 for q in EV if not sat[q])))})
    out["A1_fixed_eval_curve"] = {str(n): {k: round(statistics.mean(c[k] for c in v), 4) for k in v[0]}
                                  for n, v in curves.items()}
    out["A1_fixed_eval_curve_sd"] = {str(n): {k: round(statistics.pstdev(c[k] for c in v), 4) for k in v[0]}
                                     for n, v in curves.items()}

    # ---- A2/A3/A4 all requests, streams over ALL inits (same protocol as s25_stream)
    ALL = reqs_for(TASKS, inits)
    for r in ALL:
        sat.setdefault(r, bool(sat_cfgs(grid, *r)))
    cal_init = {t: tr_inits[t][0] for t in TASKS}
    cal_cost = sum(int(grid[(t, cal_init[t], P.key(*c))].get("substeps_executed", 0)) for t in TASKS for c in CFGS)

    def calibrated_rank(t, q):
        fband, sband = P.FORCE_BANDS[q["force"]], P.SPEED_BANDS[q["speed"]]
        sc = []
        for c in CFGS:
            ok, _, _ = P.outcome(grid[(t, cal_init[t], P.key(*c))], fband, sband)
            sc.append((-int(ok), -P.rule_prior(c[0], c[1], fband, sband), P.FORCES.index(c[0]), P.SPEEDS.index(c[1]), c))
        return [c for *_, c in sorted(sc)]

    arms = {}
    for name in ("rule+next", "rule+cause", "calibrated_rule+cause", "library+cause", "oracle_pass", "oracle_min"):
        agg = defaultdict(list)
        for sd in range(10):
            st = ALL[:]; random.Random(sd).shuffle(st)
            lib = P.Library(); cls = Counter(); cost = 0; cost_unsat = 0; first = 0
            for q in st:
                if name.startswith("oracle"):
                    ok = sat_cfgs(grid, *q)
                    if name == "oracle_min":
                        c = int(grid[(q[0], q[1], P.key(*ok[0]))]["substeps_executed"]) if ok else 0
                        r = {"first": bool(ok), "passed": bool(ok), "stop": None if ok else "declared", "cost": c}
                    else:
                        rank = ok + [c for c in CFGS if c not in ok]
                        r = serve(grid, lambda t, qq: rank, q, repair="next")
                elif name.startswith("library"):
                    r = serve(grid, lambda t, qq: P.ranking("library", lib, t, qq), q, lib=lib)
                elif name.startswith("calibrated"):
                    r = serve(grid, calibrated_rank, q)
                else:
                    r = serve(grid, lambda t, qq: P.ranking("rule", None, t, qq), q, repair=name.split("+")[1])
                cls[classify(r, sat[q])] += 1; cost += r["cost"]; first += r["first"]
                if not sat[q]:
                    cost_unsat += r["cost"]
            agg["first_pass"].append(first); agg["cost_M"].append(cost / 1e6); agg["cost_on_unsat_M"].append(cost_unsat / 1e6)
            for k in ("passed_first", "passed_after_repair", "unsat_declared_stop", "unsat_budget_spent",
                      "sat_false_stop", "sat_budget_exhausted"):
                agg[k].append(cls[k])
        arms[name] = {k: round(statistics.mean(v), 3) for k, v in agg.items()}
        if name.startswith("calibrated"):
            arms[name]["calibration_cost_M_excluded_above"] = round(cal_cost / 1e6, 3)
            arms[name]["calibration_inits"] = cal_init
    out["A2_all_requests"] = {"n": len(ALL), "n_sat": sum(sat[r] for r in ALL), "arms": arms}

    # ---- A5 repair failure breakdown (rule+cause and library+cause, seed 0)
    br = {}
    for name in ("rule+cause", "library+cause"):
        st = ALL[:]; random.Random(0).shuffle(st)
        lib = P.Library(); c5 = Counter()
        for q in st:
            r = (serve(grid, lambda t, qq: P.ranking("library", lib, t, qq), q, lib=lib) if name.startswith("library")
                 else serve(grid, lambda t, qq: P.ranking("rule", None, t, qq), q))
            ok_set = set(sat_cfgs(grid, *q))
            for j in range(len(r["tried"]) - 1 + (0 if r["passed"] else 1)):
                cause = r["causes"][j]
                if cause is None:
                    continue
                nxt = r["tried"][j + 1] if j + 1 < len(r["tried"]) else None
                if nxt is not None and nxt in ok_set:
                    continue  # successful repair
                f0, s0 = r["tried"][j]
                fband = P.FORCE_BANDS[q[2]]
                order = sorted(P.FORCES)
                up = {c for c in ok_set if order.index(c[0]) > order.index(f0)}
                down = {c for c in ok_set if order.index(c[0]) < order.index(f0)}
                same_f = {c for c in ok_set if c[0] == f0}
                if not ok_set:
                    k = "no_solution_in_grid"
                elif cause in ("slip_or_drop", "force_below_band"):
                    k = ("solution_exists_in_direction_not_reached" if up or (cause == "slip_or_drop" and same_f)
                         else "solution_only_opposite_direction(misattribution)")
                elif cause == "force_above_band":
                    k = "solution_exists_in_direction_not_reached" if down else "solution_only_opposite_direction(misattribution)"
                else:
                    k = "solution_exists_in_direction_not_reached" if same_f else "solution_needs_force_change(misattribution)"
                k += "|stop" if (nxt is None and r["stop"]) else ("|budget" if nxt is None else "|next_failed")
                c5[(cause, k)] += 1
        br[name] = [{"cause": a, "class": b, "n": n} for (a, b), n in c5.most_common()]
    out["A5_failed_repair_breakdown_seed0"] = br

    # ---- A6 dispersion of the measured hold per (task, cfg) across inits
    disp = {}
    for t in TASKS:
        for c in CFGS:
            hs = []
            for i in inits[t]:
                e = grid[(t, i, P.key(*c))]
                h = e["objects"][e["requested_role"]].get("hold_mean")
                if h is not None and e["objects"][e["requested_role"]].get("grasp_stable"):
                    hs.append(h)
            disp[f"T{t}/{P.key(*c)}"] = {"n_stable": len(hs), "mean": round(statistics.mean(hs), 2) if hs else None,
                                       "sd": round(statistics.pstdev(hs), 3) if len(hs) > 1 else None,
                                       "task_ok_rate": round(sum(grid[(t, i, P.key(*c))]["task_ok"] for i in inits[t]) / len(inits[t]), 3)}
    out["A6_hold_dispersion"] = disp
    dst = P.GRID / "stream" / "audit_s27.json"
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("n_eval", "n_eval_sat", "n_train", "A1_fixed_eval_curve")}, indent=0))
    print(json.dumps(out["A2_all_requests"], indent=0))
    for k, v in out["A5_failed_repair_breakdown_seed0"].items():
        print(k); [print("  ", x) for x in v]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
