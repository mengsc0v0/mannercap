"""Step 25 analysis on the response grid: per-task arm table, convergence of the selected manner (library), and
cause-aware repair cases (P3).  No new simulation.  Usage: s25_analyse.py [--seeds 10] [--out analysis_v1]"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25_policies as P  # noqa: E402
import s25_stream as S  # noqa: E402


def hold_of(grid, t, i, cfg):
    e = grid.get((t, i, P.key(*cfg)))
    if not e or "error" in e:
        return None, None
    o = e["objects"][e["requested_role"]]
    return o.get("hold_mean"), e.get("carry_speed_med_mps")


def run_logged(arm, stream, grid, budget):
    """Same as S.run_arm but keeps the tried configs per request."""
    name, policy, repair, scope = arm
    lib, row_init, out = P.Library(), [], []
    for (t, i, fb, sb) in stream:
        req = {"force": fb, "speed": sb}
        fband, sband = P.FORCE_BANDS[fb], P.SPEED_BANDS[sb]
        view = lib if scope != "other_inits" else S.ScopedLibrary(lib, row_init, (t, i))
        rank = P.ranking(policy, view, t, req)
        tried, causes, passed, stop = [], [], False, None
        cur = rank[0]
        while len(tried) < budget and cur is not None:
            e = grid.get((t, i, P.key(*cur)))
            ok, cause, _ = P.outcome(e, fband, sband)
            tried.append(cur); causes.append(cause)
            if e is not None:
                lib.add(t, cur[0], cur[1], e); row_init.append((t, i))
            if ok:
                passed = True
                break
            if repair == "cause":
                cur = P.cause_repair(cur, cause, req, tried, rank)
                stop = "declared" if cur is None else None
            else:
                cur = next((c for c in rank if c not in tried), None)
        sat = any(P.outcome(grid.get((t, i, P.key(f, s))), fband, sband)[0] for f in P.FORCES for s in P.SPEEDS)
        out.append({"t": t, "i": i, "fb": fb, "sb": sb, "tried": tried, "causes": causes, "passed": passed,
                    "stop": stop, "sat": sat})
    return out


def first_choice_error(grid, r):
    """Normalised manner error of the FIRST proposal: |hold - band centre| / half-width, |v - nominal| / half-width."""
    fband, sband = P.FORCE_BANDS[r["fb"]], P.SPEED_BANDS[r["sb"]]
    h, v = hold_of(grid, r["t"], r["i"], r["tried"][0])
    fe = abs(h - 0.5 * (fband[0] + fband[1])) / (0.5 * (fband[1] - fband[0])) if h is not None else None
    sc, sw = {"slow": (0.06, 0.03), "normal": (0.13, 0.04), "fast": (0.24, 0.07)}[r["sb"]]
    ve = abs(v - sc) / sw if v is not None else None
    return fe, ve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--out", default="analysis_v1")
    a = ap.parse_args()
    tasks = [0, 7, 1]
    grid = P.load_grid(tuple(tasks))
    inits = {t: sorted({i for (tt, i, _) in grid if tt == t}) for t in tasks}
    reqs = S.requests(tasks, inits)
    res = {"per_task": {}, "convergence": {}, "repair": {}}
    logs = {}
    for arm in S.ARMS:
        if arm[1] == "oracle":
            continue
        logs[arm[0]] = []
        for sd in range(a.seeds):
            s = reqs[:]; random.Random(sd).shuffle(s)
            logs[arm[0]].append(run_logged(arm, s, grid, a.budget))
    # 1. per-task table
    for name, L in logs.items():
        for t in tasks:
            rows = [r for lg in L for r in lg if r["t"] == t]
            sat = [r for r in rows if r["sat"]]
            fp = sum(1 for r in sat if r["causes"][0] is None)
            res["per_task"].setdefault(str(t), {})[name] = {
                "n_sat_per_seed": len(sat) / a.seeds, "n_unsat_per_seed": (len(rows) - len(sat)) / a.seeds,
                "first_pass_rate": round(fp / max(1, len(sat)), 3),
                "pass_rate": round(sum(r["passed"] for r in sat) / max(1, len(sat)), 3),
                "failed_attempts_per_req": round(sum(len(r["tried"]) - r["passed"] for r in rows) / len(rows), 3),
                "honest_stop_rate_unsat": round(sum(1 for r in rows if not r["sat"] and r["stop"]) /
                                                max(1, len(rows) - len(sat)), 3)}
    # 2. convergence of the first proposal (quartiles of stream position)
    for name in ("rule+cause", "library+cause", "library_xinit+cause"):
        q = defaultdict(lambda: {"fe": [], "ve": [], "fp": []})
        for lg in logs[name]:
            n = len(lg)
            for k, r in enumerate(lg):
                if not r["sat"]:
                    continue
                fe, ve = first_choice_error(grid, r)
                b = q[min(3, 4 * k // n)]
                if fe is not None: b["fe"].append(fe)
                if ve is not None: b["ve"].append(ve)
                b["fp"].append(r["causes"][0] is None)
        res["convergence"][name] = {f"Q{j + 1}": {"force_err": round(st.mean(q[j]["fe"]), 3),
                                                   "force_in_band": round(sum(x <= 1 for x in q[j]["fe"]) / len(q[j]["fe"]), 3),
                                                   "speed_err": round(st.mean(q[j]["ve"]), 3),
                                                   "first_pass": round(st.mean(q[j]["fp"]), 3)} for j in range(4)}
    # 2b. what the library converges to per (task, band): modal first config in last quartile
    modal = defaultdict(Counter)
    for lg in logs["library+cause"]:
        n = len(lg)
        for k, r in enumerate(lg[3 * n // 4:]):
            modal[(r["t"], r["fb"], r["sb"])][P.key(*r["tried"][0])] += 1
    res["library_converged_first_choice"] = {f"T{t}/{fb}/{sb}": c.most_common(2) for (t, fb, sb), c in sorted(modal.items())}
    # 3. repair: transitions under rule+cause (first attempt failed, repaired)
    trans = Counter(); examples = defaultdict(list)
    for lg in logs["rule+cause"][:1]:
        for r in lg:
            if r["causes"][0] is None:
                continue
            for j in range(len(r["tried"]) - 1):
                f0, f1 = r["tried"][j][0], r["tried"][j + 1][0]
                if tuple(f1) != tuple(f0):  # closure lo or hold cap changed
                    d = "force_up" if (f1[0], f1[1]) > (f0[0], f0[1]) else "force_down"
                else:
                    d = "slower" if r["tried"][j + 1][1] < r["tried"][j][1] else "speed_change"
                c = r["causes"][j]
                ok_next = r["causes"][j + 1] is None
                trans[(c, d, ok_next)] += 1
                if ok_next and len(examples[(c, d)]) < 3:
                    h0, _ = hold_of(grid, r["t"], r["i"], r["tried"][j]); h1, _ = hold_of(grid, r["t"], r["i"], r["tried"][j + 1])
                    examples[(c, d)].append({"task": r["t"], "init": r["i"], "band": r["fb"], "speed": r["sb"],
                                             "from": P.key(*r["tried"][j]), "hold_from": h0,
                                             "to": P.key(*r["tried"][j + 1]), "hold_to": h1})
    res["repair"]["transitions_seed0"] = [{"cause": c, "move": d, "next_passed": o, "n": n} for (c, d, o), n in trans.most_common()]
    res["repair"]["examples_seed0"] = {f"{c}->{d}": v for (c, d), v in examples.items()}
    # 3b. cause histogram of first failures (all non-oracle arms, seed 0)
    res["first_failure_causes_rule"] = Counter(r["causes"][0] for r in logs["rule+next"][0] if r["causes"][0]).most_common()
    out = P.GRID / "stream" / f"{a.out}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("per_task", "convergence")}, indent=0)[:6000])
    print("CONVERGED", json.dumps(res["library_converged_first_choice"])[:3000])
    print("REPAIR", json.dumps(res["repair"], default=str)[:4000])
    print("CAUSES", res["first_failure_causes_rule"])


if __name__ == "__main__":
    main()
