"""Step 25: online request stream replayed on the deterministic response grid (no new simulation).

Requests = (task, init, force band, speed band).  A stream is a random permutation of all requests (several seeds).
Each policy handles the stream in order with a fixed attempt budget; with an online library every executed attempt is
written back before the next request.  Reported per stream position (learning curve) and overall:
  first-attempt pass, pass within budget, attempts, failed attempts, physical substeps, honest stops.
Pass = task_ok AND stable bilateral hold AND hold window inside the force band AND realised carry speed inside the speed
band (development criterion; contact policy v1.1 reported separately, it is not selected on).

Usage: s25_stream.py [--tasks 0 7 1] [--seeds 10] [--budget 3] [--out NAME]
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

OUT = P.GRID / "stream"
ARMS = [  # (name, ranking policy, repair, library scope)
    ("fixed+next", "fixed", "next", None),
    ("rule+next", "rule", "next", None),
    ("rule+cause", "rule", "cause", None),
    ("library+next", "library", "next", "all"),
    ("library+cause", "library", "cause", "all"),
    ("library_xinit+cause", "library", "cause", "other_inits"),
    ("oracle", "oracle", "next", None),
]


def requests(tasks, inits_by_task):
    return [(t, i, fb, sb) for t in tasks for i in inits_by_task[t]
            for fb in ("R1", "R2", "R3", "R4") for sb in ("slow", "normal", "fast")]


class ScopedLibrary(P.Library):
    """Library view that hides the rows of the current init (strict cross-scene generalisation)."""

    def __init__(self, base, init_of_row, exclude_init):
        self.rows = [r for r, i in zip(base.rows, init_of_row) if i != exclude_init]


def run_arm(arm, stream, grid, budget):
    name, policy, repair, scope = arm
    lib, row_init = P.Library(), []
    log = []
    for (t, i, fb, sb) in stream:
        req = {"force": fb, "speed": sb}
        fband, sband = P.FORCE_BANDS[fb], P.SPEED_BANDS[sb]
        view = lib if scope != "other_inits" else ScopedLibrary(lib, row_init, (t, i))
        if policy == "oracle":
            cfgs = [(f, s) for f in P.FORCES for s in P.SPEEDS]
            ok = [c for c in cfgs if P.outcome(grid.get((t, i, P.key(*c))), fband, sband)[0]]
            rank = ok + [c for c in cfgs if c not in ok]
        else:
            rank = P.ranking(policy, view, t, req)
        tried, att, first, passed, stop, cost, causes = [], 0, None, False, None, 0, []
        cur = rank[0]
        while att < budget and cur is not None:
            e = grid.get((t, i, P.key(*cur)))
            ok, cause, facts = P.outcome(e, fband, sband)
            att += 1
            tried.append(cur)
            cost += int(e.get("substeps_executed", 0)) if e and "error" not in e else 0
            if policy != "oracle" and e is not None:
                lib.add(t, cur[0], cur[1], e)
                row_init.append((t, i))
            if first is None:
                first = ok
            causes.append(cause)
            if ok:
                passed = True
                break
            if repair == "cause":
                nxt = P.cause_repair(cur, cause, req, tried, rank)
                if nxt is None:
                    stop = "manner_infeasible_declared"
                cur = nxt
            else:
                cur = next((c for c in rank if c not in tried), None)
        satisfiable = any(P.outcome(grid.get((t, i, P.key(f, s))), fband, sband)[0]
                          for f in P.FORCES for s in P.SPEEDS)
        log.append({"task": t, "init": i, "force": fb, "speed": sb, "attempts": att, "first_pass": bool(first),
                    "passed": passed, "stop": stop, "cost": cost, "satisfiable": satisfiable, "causes": causes,
                    "first_cfg": P.key(*tried[0]) if tried else None})
    return log


def summarise(log, window):
    n = len(log)
    sat = [r for r in log if r["satisfiable"]]
    s = {"n": n, "n_satisfiable": len(sat),
         "first_pass": sum(r["first_pass"] for r in log), "passed": sum(r["passed"] for r in log),
         "attempts": sum(r["attempts"] for r in log),
         "failed_attempts": sum(r["attempts"] - r["passed"] for r in log),
         "cost_substeps": sum(r["cost"] for r in log),
         "honest_stop_on_unsatisfiable": sum(1 for r in log if not r["satisfiable"] and r["stop"]),
         "false_stop_on_satisfiable": sum(1 for r in log if r["satisfiable"] and r["stop"] and not r["passed"]),
         "attempts_on_unsatisfiable": sum(r["attempts"] for r in log if not r["satisfiable"])}
    curve = []
    for k in range(0, n, window):
        b = [r for r in log[k:k + window] if r["satisfiable"]]
        if b:
            curve.append({"start": k, "n_sat": len(b), "first_pass_rate": sum(r["first_pass"] for r in b) / len(b),
                          "pass_rate": sum(r["passed"] for r in b) / len(b),
                          "failed_per_req": sum(r["attempts"] - r["passed"] for r in b) / len(b)})
    s["curve"] = curve
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 7, 1])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--complete-only", action="store_true", help="use only inits whose 30 grid cells are all present")
    ap.add_argument("--out", default="stream_v1")
    a = ap.parse_args()
    grid = P.load_grid(tuple(a.tasks))
    inits = {}
    for t in a.tasks:
        have = sorted({i for (tt, i, _) in grid if tt == t})
        if a.complete_only:
            have = [i for i in have if all((t, i, P.key(f, s)) in grid and "error" not in grid[(t, i, P.key(f, s))]
                                           for f in P.FORCES for s in P.SPEEDS)]
        inits[t] = have
    reqs = requests(a.tasks, inits)
    res = {"grid_cells": len(grid), "inits": inits, "n_requests": len(reqs), "budget": a.budget, "arms": {}}
    for arm in ARMS:
        per_seed = []
        for sd in range(a.seeds):
            st = reqs[:]
            random.Random(sd).shuffle(st)
            per_seed.append(summarise(run_arm(arm, st, grid, a.budget), a.window))
        agg = {k: statistics.mean(p[k] for p in per_seed) for k in per_seed[0] if k != "curve"}
        L = min(len(p["curve"]) for p in per_seed)
        agg["curve"] = [{k: statistics.mean(p["curve"][j][k] for p in per_seed) for k in per_seed[0]["curve"][j]}
                        for j in range(L)]
        res["arms"][arm[0]] = agg
        print(f"{arm[0]:22s} first {agg['first_pass']:.1f}  pass {agg['passed']:.1f}/{agg['n_satisfiable']:.0f} sat "
              f"(n={agg['n']})  att {agg['attempts']:.0f}  failed {agg['failed_attempts']:.0f}  "
              f"cost {agg['cost_substeps'] / 1e6:.2f}M  honest_stop {agg['honest_stop_on_unsatisfiable']:.1f}  "
              f"false_stop {agg['false_stop_on_satisfiable']:.1f}", flush=True)
        c = agg["curve"]
        print("   first-pass curve:", " ".join(f"{x['first_pass_rate']:.2f}" for x in c))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.out}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
