"""Step 30 formal metrics (protocol step30_manner_formal.v1).  Usage: s30_formal_metrics.py FORMAL_DIR"""
import collections
import json
import math
import sys
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar(b, c):
    """Exact two-sided binomial McNemar on discordant pairs b (A pass, B fail) and c (A fail, B pass)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(d):
    return {f.stem: json.loads(f.read_text()) for f in sorted((d / "runs").glob("*.json"))}


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def condition(R):
    out, per_task, clause = {}, collections.defaultdict(lambda: [0, 0, 0, 0]), collections.defaultdict(lambda: [0, 0, 0])
    stops = collections.Counter()
    n = len(R)
    first = sum(bool(r["attempts"] and r["attempts"][0].get("audit", {}).get("joint_pass")) for r in R.values())
    final = sum(bool(r["final"]["joint_pass"]) for r in R.values())
    execs = sum(len(r["attempts"]) for r in R.values())
    for r in R.values():
        pt = per_task[r["task"]]
        pt[0] += 1
        pt[1] += bool(r["attempts"] and r["attempts"][0].get("audit", {}).get("joint_pass"))
        pt[2] += bool(r["final"]["joint_pass"])
        pt[3] += len(r["attempts"])
        if not r["final"]["joint_pass"]:
            stops[r["final"].get("stop")] += 1
        a = [x for x in r["attempts"] if "audit" in x]
        if not a:
            continue
        for c in a[0]["audit"]["clauses"]:
            k = f'{c["dimension"]}:{c["level"]}'
            clause[k][0] += 1
            clause[k][1] += bool(c["pass"])
        for c in a[-1]["audit"]["clauses"]:
            clause[f'{c["dimension"]}:{c["level"]}'][2] += bool(c["pass"])
    out["first_joint_pass"] = rate(first, n)
    out["final_joint_pass"] = rate(final, n)
    out["executions"] = execs
    out["executions_per_success"] = round(execs / final, 3) if final else None
    out["per_task"] = {f"T{t}": {"n": v[0], "first": v[1], "final": v[2], "execs": v[3]} for t, v in sorted(per_task.items())}
    out["clauses"] = {k: {"n": v[0], "first": v[1], "final": v[2]} for k, v in sorted(clause.items())}
    out["fail_stops"] = dict(stops.most_common())
    return out


def first_obj(R, key):
    r = R.get(key)
    if not r or not r["attempts"] or "measured" not in r["attempts"][0]:
        return None
    return r["attempts"][0]["measured"]["objects"]


def grip_roles(r):
    return {role: lvl for role, lvl in r["request"]["grip"].items()}


def paired(R):
    """Same (task, init) orderings on the first execution (no repair), so the instruction alone explains them."""
    res = collections.defaultdict(lambda: [0, 0])
    for key, r in R.items():
        if r["instruction_id"] != "gentle_X":
            continue
        t, i = r["task"], r["init"]
        x = next(iter(grip_roles(r)))
        g, f = first_obj(R, key), first_obj(R, f"t{t}_i{i}_firm_X")
        if g and f and g[x]["hold_mean"] is not None and f[x]["hold_mean"] is not None:
            res["hold gentle_X < firm_X (X)"][0] += g[x]["hold_mean"] < f[x]["hold_mean"]
            res["hold gentle_X < firm_X (X)"][1] += 1
        s, nn, fa = (first_obj(R, f"t{t}_i{i}_{k}") for k in ("slowcarry_X", "neutral", "fastcarry_X"))
        if s and nn and fa and None not in (s[x]["carry_v"], nn[x]["carry_v"], fa[x]["carry_v"]):
            res["carry v slowcarry_X < neutral < fastcarry_X (X)"][0] += s[x]["carry_v"] < nn[x]["carry_v"] < fa[x]["carry_v"]
            res["carry v slowcarry_X < neutral < fastcarry_X (X)"][1] += 1
    for key, r in R.items():
        if r["instruction_id"] not in ("Xgentle_Yfirm", "Xfirm_Ygentle"):
            continue
        o = first_obj(R, key)
        gr = grip_roles(r)
        if not o or len(gr) != 2:
            continue
        g = next(k for k, v in gr.items() if v == "gentle")
        f = next(k for k, v in gr.items() if v == "firm")
        if o[g]["hold_mean"] is None or o[f]["hold_mean"] is None:
            continue
        res["binding: hold(gentle obj) < hold(firm obj)"][0] += o[g]["hold_mean"] < o[f]["hold_mean"]
        res["binding: hold(gentle obj) < hold(firm obj)"][1] += 1
    return {k: rate(*v) for k, v in res.items()}


def main(d):
    d = Path(d)
    C = {c: load(d / c) for c in ("full", "nolib") if (d / c / "runs").exists()}
    rep = {"protocol": json.loads((d / "protocol.json").read_text())["protocol"], "conditions": {}}
    for c, R in C.items():
        rep["conditions"][c] = condition(R)
        rep["conditions"][c]["paired_first_execution"] = paired(R)
    rep["conditions"]["no_repair (=full first execution)"] = {"joint_pass": rep["conditions"]["full"]["first_joint_pass"]}
    if len(C) == 2:
        keys = sorted(set(C["full"]) & set(C["nolib"]))
        b = sum(C["full"][k]["final"]["joint_pass"] and not C["nolib"][k]["final"]["joint_pass"] for k in keys)
        c = sum(not C["full"][k]["final"]["joint_pass"] and C["nolib"][k]["final"]["joint_pass"] for k in keys)
        rep["full_vs_nolib_final"] = {"n_pairs": len(keys), "full_only": b, "nolib_only": c, "mcnemar_exact_p": mcnemar(b, c)}
        fr = C["full"]
        b2 = sum(fr[k]["final"]["joint_pass"] and not fr[k]["attempts"][0].get("audit", {}).get("joint_pass") for k in fr)
        c2 = sum(not fr[k]["final"]["joint_pass"] and fr[k]["attempts"][0].get("audit", {}).get("joint_pass", False) for k in fr)
        rep["full_repair_vs_first"] = {"recovered": b2, "lost": c2, "mcnemar_exact_p": mcnemar(b2, c2)}
    (d / "metrics.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
