"""E4: summary of the paired retrieval comparison.  Usage: s10_retrieval_summary.py
Reads reports/step10_experience/trials/{A,B}/*.json and writes reports/step10_experience/e4_summary.json.
All counts carry their denominator; requests that exhausted the budget are kept and counted as failures.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT  # noqa: E402

D = ROOT / "reports" / "step10_experience"
E1 = json.loads((ROOT / "configs" / "experience_retrieval_e1.json").read_text())
LIB = json.loads((D / "library.json").read_text())


def load(group):
    out = {}
    for f in sorted(glob.glob(str(D / "trials" / group / "t*_init*_*.json"))):
        r = json.loads(Path(f).read_text())
        out[(r["task"], r["init"], r["request"]["id"])] = r
    return out


def agg(rs):
    n = len(rs)
    return {"n_requests": n,
            "first_attempt_pass": sum(r["first_attempt_pass"] for r in rs),
            "pass_within_budget": sum(r["pass_within_budget"] for r in rs),
            "attempts_total": sum(r["n_attempts"] for r in rs),
            "attempts_mean": round(sum(r["n_attempts"] for r in rs) / n, 3) if n else None,
            "control_steps_total": sum(r["control_steps_used"] for r in rs),
            "episode_wall_s_total": round(sum(r["episode_wall_s"] for r in rs), 1),
            "retrieval_s_total": round(sum(r["retrieval_seconds"] for r in rs), 4),
            "contact_policy_passed_any_attempt": sum(r["contact_policy_passed_any_attempt"] for r in rs)}


def main() -> int:
    A, B = load("A"), load("B")
    keys = sorted(set(A) & set(B))
    missing = sorted(set(A) ^ set(B))
    per = []
    for k in keys:
        a, b = A[k], B[k]
        per.append({"task": k[0], "init": k[1], "request": k[2], "band_n": a["request"]["band_n"],
                    "A": {"attempts": a["n_attempts"], "passed_at": a["passed_at_attempt"], "order_tried": [x["candidate"] for x in a["attempts"]]},
                    "B": {"attempts": b["n_attempts"], "passed_at": b["passed_at_attempt"], "order_tried": [x["candidate"] for x in b["attempts"]]},
                    "change": ("B better" if (b["passed_at_attempt"] or 99) < (a["passed_at_attempt"] or 99) else
                               "A better" if (a["passed_at_attempt"] or 99) < (b["passed_at_attempt"] or 99) else "same")})
    ch = {x: sum(p["change"] == x for p in per) for x in ("B better", "A better", "same")}
    det = sorted(glob.glob(str(D / "trials" / "determinism_*.json")))
    s = {"schema": "mannercap.e4_summary.v1", "e1_hash": E1["content_hash"], "library_hash": LIB["content_hash"],
         "question": E1["question"], "pass_criterion": E1["pass_criterion"],
         "n_paired_requests": len(keys), "unpaired_requests": [list(k) for k in missing],
         "A_fixed_order": agg([A[k] for k in keys]), "B_retrieval": agg([B[k] for k in keys]),
         "per_request_change": ch,
         "library_build_seconds": LIB["build_seconds"], "library_n_records": LIB["n_records"],
         "episode_sharing_note": "A and B execute the same (task, init, candidate) episodes; each episode is simulated once "
                                 "and its cost is charged to every attempt that uses it, so the control-step and wall-second "
                                 "totals are the cost of the attempts each group WOULD have paid, not of distinct simulations.",
         "determinism_checks": [json.loads(Path(f).read_text()) for f in det],
         "contact_policy_note": "contact_policy_passed is reported separately and is NOT part of the pass criterion; "
                                "a passing request here is NOT a qualified manner demonstration.",
         "per_request": per}
    (D / "e4_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_request", "determinism_checks")}, indent=1))
    for p in per:
        print(f"  T{p['task']} i{p['init']} {p['request']} band={p['band_n']}  A pass@{p['A']['passed_at']} ({p['A']['attempts']} att)  "
              f"B pass@{p['B']['passed_at']} ({p['B']['attempts']} att)  -> {p['change']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
