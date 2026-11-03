"""Step 31 formal metrics: carry-clearance repair + clearance experience on T0/T7 vs the step-30 formal full condition
(same scenes 42..49, same intake, same frozen force library).  Paired by (task, init, instruction)."""
import json
import math
from pathlib import Path

R = Path("reports")
S30 = R / "step30_manner_gen/formal/full/runs"
S31 = R / "step31_t0_contact/formal"
OUT = S31 / "metrics.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(d, tasks):
    out = {}
    for f in sorted(Path(d).glob("t*_i*_*.json")):
        r = json.loads(f.read_text())
        if r["task"] in tasks and 42 <= r["init"] <= 49:
            out[(r["task"], r["init"], r["instruction_id"])] = r
    return out


def first_ok(r):
    a = r["attempts"]
    return bool(a) and bool(a[0].get("audit", {}).get("joint_pass"))


def summary(runs):
    s = {}
    for t in sorted({k[0] for k in runs}):
        rs = [r for k, r in runs.items() if k[0] == t]
        fin = sum(bool(r["final"]["joint_pass"]) for r in rs)
        fst = sum(first_ok(r) for r in rs)
        ex = sum(len([a for a in r["attempts"] if "audit" in a]) for r in rs)
        unclean = sum(1 for r in rs for a in r["attempts"][:1] if not a.get("measured", {}).get("contact_clean", True))
        taskfail = sum(1 for r in rs for a in r["attempts"][:1] if not a.get("measured", {}).get("task_ok", True))
        stops = {}
        for r in rs:
            if not r["final"]["joint_pass"]:
                k = r["final"].get("stop", "?")
                stops[k] = stops.get(k, 0) + 1
        clr = sum(1 for r in rs for a in r["attempts"] for x in a.get("repair", []) if x.get("action") == "clearance")
        prior = {}
        for r in rs:
            for role, v in (r.get("clearance_prior") or {}).items():
                prior.setdefault(role, {}).setdefault(str(v), 0)
                prior[role][str(v)] += 1
        s[f"T{t}"] = {"n": len(rs), "first": fst, "final": fin, "first_rate": round(fst / len(rs), 3),
                      "final_rate": round(fin / len(rs), 3), "final_wilson95": wilson(fin, len(rs)),
                      "executions": ex, "first_exec_unclean": unclean, "first_exec_task_failed": taskfail,
                      "clearance_actions": clr, "clearance_prior": prior, "fail_stops": stops}
    return s


def paired(a, b):
    ks = sorted(set(a) & set(b))
    fb = lambda f: (sum(1 for k in ks if f(b[k]) and not f(a[k])), sum(1 for k in ks if f(a[k]) and not f(b[k])))  # noqa: E731
    out = {"n_pairs": len(ks)}
    for name, f in (("final", lambda r: bool(r["final"]["joint_pass"])), ("first", first_ok)):
        nb, na = fb(f)
        out[name] = {"new_only": nb, "baseline_only": na, "mcnemar_exact_p": round(mcnemar(na, nb), 6)}
    return out


def per_iid(runs):
    out = {}
    for k, r in runs.items():
        x = out.setdefault(f"T{k[0]}:{k[2]}", [0, 0])
        x[0] += bool(r["final"]["joint_pass"])
        x[1] += 1
    return out


def main():
    base = load(S30, (0, 7))
    full = load(S31 / "full/runs", (0, 7))
    ro = load(S31 / "repair_only/runs", (0,))
    ex = load(S31 / "experience_loso/runs", (0,))
    b0, f0 = {k: v for k, v in base.items() if k[0] == 0}, {k: v for k, v in full.items() if k[0] == 0}
    m = {"protocol": "step31_carry_clearance.v1",
         "conditions": {"step30_full (baseline)": summary(base), "step31_full": summary(full), "step31_repair_only": summary(ro),
                        "step31_experience_loso": summary(ex)},
         "paired": {"step31_full vs step30_full": paired(base, full),
                    "step31_full vs step30_full (T0)": paired(b0, f0),
                    "step31_repair_only vs step30_full (T0)": paired(b0, ro),
                    "step31_full vs step31_repair_only (T0)": paired(ro, f0),
                    "step31_experience_loso vs step31_full (T0)": paired(f0, ex),
                    "step31_experience_loso vs step30_full (T0)": paired(b0, ex)},
         "per_instruction_final": {"step30_full": per_iid(base), "step31_full": per_iid(full), "step31_repair_only": per_iid(ro),
                                   "step31_experience_loso": per_iid(ex)}}
    OUT.write_text(json.dumps(m, indent=1))
    for c, s in m["conditions"].items():
        for t, v in s.items():
            print(f"{c:24s} {t} first {v['first']:2d}/{v['n']} final {v['final']:2d}/{v['n']} execs {v['executions']:3d} "
                  f"unclean1 {v['first_exec_unclean']:2d} taskfail1 {v['first_exec_task_failed']:2d} clr {v['clearance_actions']:2d} "
                  f"prior {v['clearance_prior']} stops {v['fail_stops']}")
    for k, v in m["paired"].items():
        print(k, v)


if __name__ == "__main__":
    main()
