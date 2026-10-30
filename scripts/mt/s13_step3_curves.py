"""Step 3: qualified output at equal physical budget, rebuilt from the step12 logs (development-set reconstruction).

x axis  : cumulative physical substeps actually simulated for the request, planner look-ahead and every discarded or
          aborted branch included.  Cache hits are charged their RECORDED cost (the 'method' book): this is a
          reconstructed cost, not what this experiment spent.
counting: a result counts only after the execution that produced it has COMPLETED and been judged.  An aborted branch
          costs budget but never counts (its task/manner are undetermined).  Never-passing requests stay in the denominator.
Every number is derived from the logs; nothing is re-simulated and no rule is added to improve a curve.
"""
import collections
import json
from pathlib import Path

import numpy as np

from s13_components import components, is_aborted, key, load_cells

ARMS = [("rule", "rerun"), ("rule", "recover"), ("retrieval", "rerun"), ("retrieval", "recover")]
cells = load_cells()
INF = float("inf")


def trace(c):
    """[(cumulative_cost_after_execution, components)] with aborted branches costing but never counting."""
    band, cum, out = c["request"]["band_n"], 0, []
    for e in c["executions"]:
        cum += int(e["executed_substeps"])
        out.append((cum, components(e, band), is_aborted(e)))
    return out


def first_reach(tr, comp):
    for cost, comps, ab in tr:
        if not ab and comps[comp] == "pass":
            return cost
    return INF


per = {}
for c in cells:
    tr = trace(c)
    per[(c["selection"], c["recovery"]) + key(c)] = {
        "task": first_reach(tr, "task"), "manner": first_reach(tr, "manner"), "contact": first_reach(tr, "contact"),
        "joint": first_reach(tr, "joint"), "total": tr[-1][0], "n_exec": len(tr)}

# sanity: the joint-pass count here must equal the 2x2 table
for arm in ARMS:
    n = sum(1 for k, v in per.items() if k[:2] == arm and v["joint"] < INF)
    print(arm, "joint passes rebuilt from the log:", n)

grid = [5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 40000, 50000]
maxcost = max(v["total"] for v in per.values())
grid.append(int(np.ceil(maxcost / 1000) * 1000))


def rate(arm, comp, B):
    rows = [v for k, v in per.items() if k[:2] == arm]
    return sum(v[comp] <= B for v in rows), len(rows)


print("\nrequests (of 32) whose FIRST judged pass came within a per-request budget of B substeps")
for comp in ("joint", "task", "manner", "contact"):
    print(f"\n[{comp}]  B(substeps) | " + " | ".join(f"{a[0][:4]}/{a[1][:5]}" for a in ARMS))
    for B in grid:
        print(f"  {B:7d} | " + " | ".join(f"{rate(a, comp, B)[0]:2d}/32     " for a in ARMS))

# cost to first joint pass, failures kept in the denominator
print("\ncost to first joint pass (substeps); requests that never pass stay in the denominator")
summ = {}
for arm in ARMS:
    v = np.array(sorted(x["joint"] for k, x in per.items() if k[:2] == arm), dtype=float)
    fin = v[np.isfinite(v)]
    tot = sum(x["total"] for k, x in per.items() if k[:2] == arm)
    q = lambda p: ("not reached" if not np.isfinite(np.quantile(v, p)) else int(np.quantile(v, p)))
    summ[f"{arm[0]}/{arm[1]}"] = {
        "n_pass": int(len(fin)), "n": int(len(v)), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75),
        "mean_among_passers": float(fin.mean()) if len(fin) else None, "sum_among_passers": float(fin.sum()),
        "total_method_substeps_all_requests": int(tot),
        "qualified_per_million_substeps": float(len(fin) / tot * 1e6)}
    s = summ[f"{arm[0]}/{arm[1]}"]
    print(f"  {arm[0]:9s}/{arm[1]:7s} pass {s['n_pass']}/32  p25={s['p25']} p50={s['p50']} p75={s['p75']} "
          f"mean(passers)={s['mean_among_passers']:.0f}  total={s['total_method_substeps_all_requests']}  "
          f"qualified/Msub={s['qualified_per_million_substeps']:.1f}")

# paired: recovery effect at identical outcome; selection effect at identical recovery
print("\nrecovery effect (rerun vs recover), requests that pass in both:")
for sel in ("rule", "retrieval"):
    ks = [k[2:] for k in per if k[:2] == (sel, "rerun")]
    a = [per[(sel, "rerun") + k]["joint"] for k in ks]
    b = [per[(sel, "recover") + k]["joint"] for k in ks]
    both = [(x, y) for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    print(f"  {sel:9s} n={len(both)}  rerun mean {np.mean([x for x,_ in both]):.0f}  recover mean {np.mean([y for _,y in both]):.0f}  "
          f"saved {1-np.mean([y for _,y in both])/np.mean([x for x,_ in both]):.1%}  recover cheaper in {sum(y<x for x,y in both)}/{len(both)}  "
          f"pass-set equal: {sum(np.isfinite(x)!=np.isfinite(y) for x,y in zip(a,b))==0}")
print("\nselection effect (rule vs retrieval), per recovery:")
disc = {}
for rec in ("rerun", "recover"):
    ks = [k[2:] for k in per if k[:2] == ("rule", rec)]
    r = [per[("rule", rec) + k]["joint"] for k in ks]
    t = [per[("retrieval", rec) + k]["joint"] for k in ks]
    both = [(x, y) for x, y in zip(r, t) if np.isfinite(x) and np.isfinite(y)]
    only_t = sum(np.isfinite(y) and not np.isfinite(x) for x, y in zip(r, t))
    only_r = sum(np.isfinite(x) and not np.isfinite(y) for x, y in zip(r, t))
    disc[rec] = {"only_retrieval": int(only_t), "only_rule": int(only_r), "both": len(both),
                 "mean_cost_rule": float(np.mean([x for x, _ in both])), "mean_cost_retrieval": float(np.mean([y for _, y in both]))}
    print(f"  {rec:7s} both pass {len(both)}  only retrieval {only_t}  only rule {only_r}  "
          f"mean cost on both: rule {disc[rec]['mean_cost_rule']:.0f} vs retrieval {disc[rec]['mean_cost_retrieval']:.0f}")
print("\nper-(task,init) joint passes, retrieval/recover vs rule/recover (init is the unit of geometry; requests inside an init are not independent):")
byinit = collections.defaultdict(lambda: collections.Counter())
for k, v in per.items():
    if k[1] == "recover":
        byinit[(k[2], k[3])][k[0]] += v["joint"] < INF
for k in sorted(byinit):
    print("  T%d init%d  rule %d/4  retrieval %d/4" % (k[0], k[1], byinit[k]["rule"], byinit[k]["retrieval"]))

# ---- figure ------------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs = np.linspace(0, maxcost, 4000)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
style = {("rule", "rerun"): ("#8c8c8c", "-", "rule + rerun"), ("rule", "recover"): ("#8c8c8c", "--", "rule + recover"),
         ("retrieval", "rerun"): ("#1f77b4", "-", "retrieval + rerun"), ("retrieval", "recover"): ("#1f77b4", "--", "retrieval + recover")}
for arm in ARMS:
    v = np.array([x["joint"] for k, x in per.items() if k[:2] == arm])
    col, ls, lab = style[arm]
    ax.step(xs, [np.sum(v <= x) / len(v) for x in xs], where="post", color=col, ls=ls, lw=2, label=f"{lab}  ({int(np.sum(np.isfinite(v)))}/32)")
ax.set_xlabel("cumulative physical substeps per request (planner look-ahead + failed/aborted branches included; cache = recorded cost)")
ax.set_ylabel("joint-pass rate (task + manner + contact v1.1)")
ax.set_ylim(0, 1); ax.set_xlim(0, maxcost)
ax.set_title("Development-set log reconstruction: T0/T7 x init26-29 x R1-R4, n=32 per arm\nnot a held-out result")
ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = Path("reports/step13_offline"); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "joint_pass_vs_budget.png", dpi=140)

(out / "step3_budget_curves.json").write_text(json.dumps(
    {"note": "development-set log reconstruction; cost = recorded executed_substeps (method book)", "grid": grid,
     "rates": {f"{a[0]}/{a[1]}": {c: {str(B): rate(a, c, B)[0] for B in grid} for c in ("joint", "task", "manner", "contact")} for a in ARMS},
     "cost_to_first_joint_pass": summ, "selection_effect": disc,
     "per_request": {"|".join(map(str, k)): {kk: (None if vv == INF else vv) for kk, vv in v.items()} for k, v in per.items()}},
    indent=1, default=float))
print("\nwrote figure and json")
