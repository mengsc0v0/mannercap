"""E4 supplement: how much of group B's advantage is the retrieval, and how much is the DIRECTION of the chosen baseline
order?  With every candidate executed on every trial init (reports/step10_experience/grid), the outcome of any candidate
order is computed exactly, without further simulation.  Usage: s10_order_robustness.py

Reported per ordering rule: first-attempt passes and passes within the frozen budget, out of all requests.
  * satisfiable            upper bound: at least one candidate in the set passes (order-independent)
  * A_ascending            the frozen group-A order (as declared in E1)
  * A_descending           the same list reversed (the opposite, equally arbitrary, fixed direction)
  * random_mean            mean over all 8! orders, computed in closed form from the rank of the first passing candidate
  * B_retrieval            the retrieval order actually used
"""
import json
import math
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT  # noqa: E402

D = ROOT / "reports" / "step10_experience"
E1 = json.loads((ROOT / "configs" / "experience_retrieval_e1.json").read_text())
CAND = [tuple(c) for c in E1["candidates"]]
BUDGET = E1["attempt_budget"]


def passes(ep, band):
    m, v = ep["measured"], ep["validity"]
    return bool(ep["task_ok"] and v["measurement_valid"] and v["grasp_stable"] and m["hold_min"] is not None
                and m["hold_min"] >= band[0] and m["hold_max"] <= band[1])


def first_pass_rank(order, ok):
    for k, c in enumerate(order):
        if ok[c]:
            return k + 1
    return None


def main() -> int:
    grid = {}
    for f in sorted((D / "grid").glob("t*_init*.json")):
        r = json.loads(f.read_text())
        grid[(r["task"], r["init"])] = {tuple(float(x) for x in k.split("_")): v for k, v in r["episodes"].items()}
    assert grid, "no grid episodes"
    missing = {k: [c for c in CAND if c not in v] for k, v in grid.items()}
    assert not any(missing.values()), f"grid incomplete: {missing}"

    b_order = {}
    for f in sorted((D / "trials" / "B").glob("t*_init*_*.json")):
        r = json.loads(f.read_text())
        b_order[(r["task"], r["init"], r["request"]["id"])] = [tuple(c) for c in r["order"]]

    rules = ["satisfiable", "A_ascending", "A_descending", "B_retrieval"]
    cnt = {r: {"first": 0, "budget": 0} for r in rules}
    rnd = {"first": 0.0, "budget": 0.0}
    rows, n = [], 0
    for (task, init), eps in sorted(grid.items()):
        for req in E1["manner_requests"]:
            band = req["band_n"]
            ok = {c: passes(eps[c], band) for c in CAND}
            n_ok = sum(ok.values())
            n += 1
            ranks = {"A_ascending": first_pass_rank(CAND, ok), "A_descending": first_pass_rank(list(reversed(CAND)), ok),
                     "B_retrieval": first_pass_rank(b_order[(task, init, req["id"])], ok)}
            cnt["satisfiable"]["first"] += bool(n_ok)       # upper bound: an oracle would pass on attempt 1
            cnt["satisfiable"]["budget"] += bool(n_ok)
            for r_ in ("A_ascending", "A_descending", "B_retrieval"):
                cnt[r_]["first"] += ranks[r_] == 1
                cnt[r_]["budget"] += ranks[r_] is not None and ranks[r_] <= BUDGET
            # a uniformly random order: P(first pass at position <= b) = 1 - C(N-n_ok, b)/C(N, b)
            N = len(CAND)
            if n_ok:
                rnd["first"] += n_ok / N
                rnd["budget"] += 1.0 - math.comb(N - n_ok, BUDGET) / math.comb(N, BUDGET)
            rows.append({"task": task, "init": init, "request": req["id"], "band_n": band,
                         "n_passing_candidates": n_ok, "passing_candidates": [list(c) for c in CAND if ok[c]],
                         "first_pass_rank": ranks})
    out = {"schema": "mannercap.e4_order_robustness.v1", "e1_hash": E1["content_hash"], "n_requests": n,
           "attempt_budget": BUDGET, "n_candidates": len(CAND),
           "counts": {**{k: v for k, v in cnt.items()},
                      "random_order_expected": {"first": round(rnd["first"], 2), "budget": round(rnd["budget"], 2)}},
           "reading": "'satisfiable' is an order-independent upper bound (an oracle that already knew the answer). The gap "
                      "between A_ascending and A_descending is the effect of the baseline's direction alone; the gap between "
                      "random_order_expected and B_retrieval is the part that the ordering rule itself has to explain.",
           "no_new_simulation": "computed from reports/step10_experience/grid; every candidate was executed once per init",
           "per_request": rows}
    (D / "e4_order_robustness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_request"}, indent=1))
    for r_ in rows:
        print(f"  T{r_['task']} i{r_['init']} {r_['request']} band={r_['band_n']} passing={r_['n_passing_candidates']}/8 ranks={r_['first_pass_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
