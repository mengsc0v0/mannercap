"""Step 28: replay the step-25 selection/repair/library policies on the T6 manner grid (no new simulation, no retuning).

Arms (protocol reports/step28_t6_manner/protocol.json):
  fixed+next, rule+next, rule+cause, oracle         -- as s25
  library_empty+cause      -- online library that starts empty on T6
  library_xtask+cause      -- library preloaded with every T0/T1/T7 step-25 grid row, online write-back on T6
  library_xtask_nowb+cause -- same preload, NO write-back (pure cross-task transfer)
  library_xtask_xinit+cause-- preload + write-back, rows of the current T6 init hidden
Pass (dev) = s25 outcome(); qualified = the passing episode is also contact-policy-v1.1 clean.
Usage: s28_t6_replay.py [--seeds 10] [--budget 3]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25_policies as P  # noqa: E402
import s25_stream as S  # noqa: E402

T = 6
P.OBJECT[T] = "porcelain_mug_1"
OUT = P.ROOT / "reports" / "step28_t6_manner"
GRID6 = OUT / "grid"
ARMS = [("fixed+next", "fixed", "next", None), ("rule+next", "rule", "next", None), ("rule+cause", "rule", "cause", None),
        ("library_empty+cause", "library", "cause", "empty"), ("library_xtask+cause", "library", "cause", "xtask"),
        ("library_xtask_nowb+cause", "library", "cause", "xtask_nowb"),
        ("library_xtask_xinit+cause", "library", "cause", "xtask_xinit"),
        ("rule+resp", "rule", "resp", None), ("library_empty+resp", "library", "resp", "empty"),
        ("library_xtask+resp", "library", "resp", "xtask"), ("oracle", "oracle", "next", None)]
DEV_ARMS = [("rule+cause", "rule", "cause", None), ("rule+resp", "rule", "resp", None),
            ("library+cause", "library", "cause", "empty"), ("library+resp", "library", "resp", "empty"),
            ("library_xinit+cause", "library", "cause", "xinit"), ("library_xinit+resp", "library", "resp", "xinit"),
            ("oracle", "oracle", "next", None)]


FORCE_CAUSES = ("force_below_band", "force_above_band", "slip_or_drop")


def resp_repair(prev, cause, req, tried, rank, lib, task, obj):
    """Repair v2 (protocol_repair_v2.json): choose the next force by the MEASURED response stored in the library
    instead of the closure-stop order.  No free parameters."""
    if cause not in FORCE_CAUSES:
        return P.cause_repair(prev, cause, req, tried, rank)
    fband, sband = P.FORCE_BANDS[req["force"]], P.SPEED_BANDS[req["speed"]]
    f0, s0 = prev
    cands = [(f, s0) for f in P.FORCES if f != f0]
    if cause == "slip_or_drop":
        cands += [(f0, s) for s in sorted(P.SPEEDS) if s < s0 and sband[0] <= s <= sband[1]]
    cands = [c for c in cands if c not in tried and c[0][0] <= fband[1]]
    known, centre = [], 0.5 * (fband[0] + fband[1])
    unknown_ok = set()
    v1 = []
    t2 = list(tried)
    while True:   # v1 direction order, used only for candidates the library knows nothing about
        nx = P.cause_repair(prev, cause, req, t2, rank)
        if nx is None:
            break
        v1.append(nx)
        t2.append(nx)
    for c in cands:
        rows, _ = lib._sel(task, obj, c[0], c[1])
        if not rows:
            unknown_ok.add(c)
            continue
        st = [r for r in rows if r["stable"] and r["hold_min"] is not None]
        if not st:
            continue                                   # measured to slip -> dropped
        lo = sum(r["hold_min"] for r in st) / len(st)
        hi = sum(r["hold_max"] for r in st) / len(st)
        if fband[0] <= lo and hi <= fband[1]:
            known.append((abs(0.5 * (lo + hi) - centre), c))
    if known:
        return sorted(known, key=lambda x: (x[0], P.FORCES.index(x[1][0]), P.SPEEDS.index(x[1][1])))[0][1]
    return next((c for c in v1 if c in unknown_ok), None)


def load_t6():
    g = {}
    for f in sorted(GRID6.glob("t6_init*.json")):
        d = json.loads(f.read_text())
        for k, e in d["episodes"].items():
            g[(T, d["init"], k)] = e
    return g


def preload(src):
    lib, rows_init = P.Library(), []
    for (t, i, k), e in src.items():
        f, s = k.split("@")
        lo, hi = f.split("_")
        lib.add(t, (float(lo), float(hi)), float(s), e)
        if e is not None and "error" not in e:
            rows_init.append((t, i))
    return lib, rows_init


def run_arm(arm, stream, grid, budget, src):
    name, policy, repair, scope = arm
    if scope and scope.startswith("xtask"):
        lib, row_init = preload(src)
    else:
        lib, row_init = P.Library(), []
    log = []
    for (t, i, fb, sb) in stream:
        req = {"force": fb, "speed": sb}
        fband, sband = P.FORCE_BANDS[fb], P.SPEED_BANDS[sb]
        view = S.ScopedLibrary(lib, row_init, (t, i)) if scope in ("xtask_xinit", "xinit") else lib
        cfgs = [(f, s) for f in P.FORCES for s in P.SPEEDS]
        if policy == "oracle":
            ok = [c for c in cfgs if P.outcome(grid.get((t, i, P.key(*c))), fband, sband)[0]]
            rank = ok + [c for c in cfgs if c not in ok]
        else:
            rank = P.ranking(policy, view, t, req)
        tried, att, first, passed, stop, cost, causes, pass_cfg = [], 0, None, False, None, 0, [], None
        cur = rank[0]
        while att < budget and cur is not None:
            e = grid.get((t, i, P.key(*cur)))
            ok, cause, _ = P.outcome(e, fband, sband)
            att += 1
            tried.append(cur)
            cost += int(e.get("substeps_executed", 0)) if e and "error" not in e else 0
            if policy != "oracle" and e is not None and scope != "xtask_nowb":
                lib.add(t, cur[0], cur[1], e)
                row_init.append((t, i))
            if first is None:
                first = ok
            causes.append(cause)
            if ok:
                passed, pass_cfg = True, cur
                break
            if repair in ("cause", "resp"):
                nxt = (P.cause_repair(cur, cause, req, tried, rank) if repair == "cause"
                       else resp_repair(cur, cause, req, tried, rank, view, t, P.OBJECT[t]))
                if nxt is None:
                    stop = "manner_infeasible_declared"
                cur = nxt
            else:
                cur = next((c for c in rank if c not in tried), None)
        sat = [c for c in cfgs if P.outcome(grid.get((t, i, P.key(*c))), fband, sband)[0]]
        trans = [(causes[k], causes[k + 1] is None) for k in range(len(causes) - 1)]
        log.append({"init": i, "force": fb, "speed": sb, "attempts": att, "first_pass": bool(first), "passed": passed,
                    "qualified": bool(passed and grid[(t, i, P.key(*pass_cfg))].get("contact_v1_clean")),
                    "stop": stop, "cost": cost, "satisfiable": bool(sat),
                    "satisfiable_qualified": any(grid[(t, i, P.key(*c))].get("contact_v1_clean") for c in sat),
                    "causes": causes, "repair_transitions": trans, "first_cfg": P.key(*tried[0]) if tried else None})
    return log


def summarise(log, window):
    s = S.summarise(log, window)
    s["qualified"] = sum(r["qualified"] for r in log)
    s["n_satisfiable_qualified"] = sum(r["satisfiable_qualified"] for r in log)
    for c in FORCE_CAUSES:
        tr = [ok for r in log for (cc, ok) in r["repair_transitions"] if cc == c]
        s[f"repair_after_{c}_n"] = len(tr)
        s[f"repair_after_{c}_ok"] = sum(tr)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--dev", action="store_true", help="T0/T1/T7 step25 grid (development set), DEV_ARMS")
    a = ap.parse_args()
    if a.dev:
        grid, src, arms, tasks = P.load_grid((0, 7, 1)), {}, DEV_ARMS, [0, 7, 1]
        inits = {t: [i for i in sorted({i for (tt, i, _) in grid if tt == t})
                     if all((t, i, P.key(f, s)) in grid and "error" not in grid[(t, i, P.key(f, s))]
                            for f in P.FORCES for s in P.SPEEDS)] for t in tasks}
        reqs = S.requests(tasks, inits)
    else:
        grid, src, arms = load_t6(), P.load_grid((0, 7, 1)), ARMS
        inits = sorted({i for (_, i, _) in grid})
        reqs = S.requests([T], {T: inits})
    res = {"grid_cells": len(grid), "src_cells": len(src), "inits": inits, "n_requests": len(reqs), "budget": a.budget,
           "arms": {}}
    for arm in arms:
        per = []
        for sd in range(a.seeds):
            st = reqs[:]
            random.Random(sd).shuffle(st)
            per.append(summarise(run_arm(arm, st, grid, a.budget, src), a.window))
        agg = {k: statistics.mean(p[k] for p in per) for k in per[0] if k != "curve"}
        L = min(len(p["curve"]) for p in per)
        agg["curve"] = [{k: statistics.mean(p["curve"][j][k] for p in per) for k in per[0]["curve"][j]} for j in range(L)]
        res["arms"][arm[0]] = agg
        print(f"{arm[0]:26s} first {agg['first_pass']:.1f} pass {agg['passed']:.1f}/{agg['n_satisfiable']:.0f} "
              f"qual {agg['qualified']:.1f}/{agg['n_satisfiable_qualified']:.0f} att {agg['attempts']:.0f} "
              f"failed {agg['failed_attempts']:.0f} cost {agg['cost_substeps'] / 1e6:.2f}M "
              f"honest_stop {agg['honest_stop_on_unsatisfiable']:.1f} false_stop {agg['false_stop_on_satisfiable']:.1f}")
        print("   repair after cause (ok/n):", " ".join(f"{c} {agg[f'repair_after_{c}_ok']:.0f}/{agg[f'repair_after_{c}_n']:.0f}"
                                                         for c in FORCE_CAUSES))
        print("   first-pass curve:", " ".join(f"{x['first_pass_rate']:.2f}" for x in agg["curve"]))
    (OUT / ("replay_dev_v1.json" if a.dev else "replay_v1.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
