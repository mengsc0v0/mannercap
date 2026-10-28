"""E3: paired comparison of a fixed-order search (group A) and a retrieval order (group B) over the SAME candidate set,
on the frozen request list configs/experience_retrieval_e1.json and the frozen library reports/step10_experience/library.json.

Usage: s10_retrieval_trial.py TASK INIT...        (runs both groups for those requests)
Episodes are cached per (task, init, candidate): the simulator is deterministic for a fixed init and plan, so the same
candidate is executed once and reused by both groups; the cost of an attempt is taken from the executed episode either way.
A determinism check re-executes one cached candidate and compares the measurements bitwise.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS  # noqa: E402
from s8_task_force import grip_for, per_object, run_episode  # noqa: E402

E1 = json.loads((ROOT / "configs" / "experience_retrieval_e1.json").read_text())
LIB = json.loads((ROOT / "reports" / "step10_experience" / "library.json").read_text())
OUT = ROOT / "reports" / "step10_experience" / "trials"
CAND = [tuple(c) for c in E1["candidates"]]
OTHER = E1["scope"]["other_object_fixed_plan"]
SETTLE = E1["executor"]["settle_after_open"]


# ---------------------------------------------------------------- retrieval (group B); rule text frozen in E1.group_B
def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for k in range(1, len(xs)):
        if x <= xs[k]:
            f = (x - xs[k - 1]) / (xs[k] - xs[k - 1]) if xs[k] != xs[k - 1] else 0.0
            return ys[k - 1] + f * (ys[k] - ys[k - 1])
    return ys[-1]


def predict(obj_name, task, lo, hi):
    """Predicted measured hold interval for a candidate, from the frozen library only."""
    usable = [r for r in LIB["records"] if r["validity"]["measurement_valid"] and r["validity"]["grasp_stable"]
              and r["measured"]["hold_min"] is not None]
    sel, basis = [r for r in usable if r["object"] == obj_name], "same_object"
    if len(sel) < 2:
        sel, basis = [r for r in usable if r["task"] == task], "same_task"
    if len(sel) < 2:
        sel, basis = usable, "all_records"
    same_hi = [r for r in sel if r["control_params"]["target_hi_n"] == hi]
    if same_hi:
        sel, basis = same_hi, basis + "+same_hi"
    else:
        basis += "+any_hi"
    agg = {}
    for r in sel:
        k = r["control_params"]["target_lo_n"]
        agg.setdefault(k, []).append((r["measured"]["hold_min"], r["measured"]["hold_max"]))
    xs = sorted(agg)
    mn = [min(v[0] for v in agg[x]) for x in xs]
    mx = [max(v[1] for v in agg[x]) for x in xs]
    return {"pred_min": _interp(xs, mn, lo), "pred_max": _interp(xs, mx, lo), "basis": basis,
            "n_records_used": len(sel), "lo_grid": xs}


def order_B(obj_name, task, band):
    lo_b, hi_b = band
    rows = []
    for idx, (lo, hi) in enumerate(CAND):
        p = predict(obj_name, task, lo, hi)
        inside = p["pred_min"] >= lo_b and p["pred_max"] <= hi_b
        margin = min(p["pred_min"] - lo_b, hi_b - p["pred_max"])
        centre_gap = abs(0.5 * (p["pred_min"] + p["pred_max"]) - 0.5 * (lo_b + hi_b))
        rows.append({"cand": (lo, hi), "fixed_rank": idx, "inside": inside, "margin": margin,
                     "centre_gap": centre_gap, **p})
    rows.sort(key=lambda r: (0, -r["margin"], r["fixed_rank"]) if r["inside"] else (1, r["centre_gap"], r["fixed_rank"]))
    return rows


# ---------------------------------------------------------------- episodes
CACHE = {}


def episode(task, init, cand):
    key = (task, init, cand)
    if key in CACHE:
        c = dict(CACHE[key]); c["reused_cached_episode"] = True
        return c
    roles = [r for r, _ in SPECS[task].objects]
    req_role = [r for r, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    grips = {r: grip_for((cand[0], cand[1]) if r == req_role else (OTHER["target_lo_n"], OTHER["target_hi_n"]), SETTLE)
             for r in roles}
    prog, ep, rec, dt = run_episode(task, init, grips, None)
    o = per_object(ep, prog, task, rec)[req_role]
    ca = rec["contact_audit"]
    out = {"candidate": list(cand), "role": req_role, "object": o.get("target"), "status": ep.status,
           "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "wall_s": round(dt, 2),
           "measured": {k: o.get(k) for k in ("hold_n", "hold_min", "hold_mean", "hold_max", "lr_imbalance", "rel_z_spread_m", "peak_simultaneous_n")},
           "validity": {"status": o.get("status"), "measurement_valid": o.get("measurement_valid"), "grasp_stable": o.get("grasp_stable")},
           "contact_policy_passed": bool(ca["clean"]), "n_forbidden_contacts_0p1N": len(ca["forbidden_contacts"]),
           "n_forbidden_ge5N": sum(1 for x in ca["forbidden_contacts"] if x["max_n"] >= 5),
           "reused_cached_episode": False}
    CACHE[key] = out
    return out


def passed(ep, band):
    m, v = ep["measured"], ep["validity"]
    return bool(ep["task_ok"] and v["measurement_valid"] and v["grasp_stable"]
                and m["hold_min"] is not None and m["hold_min"] >= band[0] and m["hold_max"] <= band[1])


def run_request(task, init, req, group):
    band = req["band_n"]
    obj = [n for _, n in SPECS[task].objects if n.startswith(E1["scope"]["requested_object"][str(task)])][0]
    t0 = time.time()
    if group == "A":
        order, ranking = list(CAND), None
    else:
        ranking = order_B(obj, task, band)
        order = [tuple(r["cand"]) for r in ranking]
    retrieval_s = round(time.time() - t0, 4)
    attempts, ok_at = [], None
    for k, cand in enumerate(order[:E1["attempt_budget"]]):
        e = episode(task, init, cand)
        e["attempt"] = k + 1
        e["task_and_force_band_checks_passed"] = passed(e, band)
        attempts.append(e)
        if e["task_and_force_band_checks_passed"]:
            ok_at = k + 1
            break
    return {"schema": "mannercap.e3_trial.v1", "e1_hash": E1["content_hash"], "library_hash": LIB["content_hash"],
            "group": group, "task": task, "init": init, "request": req, "object": obj,
            "order": [list(c) for c in order], "retrieval_ranking": ranking, "retrieval_seconds": retrieval_s,
            "attempts": attempts, "n_attempts": len(attempts), "passed_at_attempt": ok_at,
            "first_attempt_pass": bool(ok_at == 1), "pass_within_budget": bool(ok_at is not None),
            "control_steps_used": sum(a["n_steps"] for a in attempts), "episode_wall_s": round(sum(a["wall_s"] for a in attempts), 2),
            "contact_policy_passed_any_attempt": any(a["contact_policy_passed"] for a in attempts)}


def main() -> int:
    task = int(sys.argv[1])
    inits = [int(x) for x in sys.argv[2:]]
    assert all(i in E1["scope"]["inits"] for i in inits), "trial inits are frozen in E1"
    for i in inits:
        for group in ("A", "B"):
            d = OUT / group
            d.mkdir(parents=True, exist_ok=True)
            for req in E1["manner_requests"]:
                f = d / f"t{task}_init{i}_{req['id']}.json"
                if f.exists() and json.loads(f.read_text()).get("library_hash") == LIB["content_hash"]:
                    continue
                r = run_request(task, i, req, group)
                f.write_text(json.dumps(r, indent=1))
                print(f"T{task} i{i} {group} {req['id']} band={req['band_n']} attempts={r['n_attempts']} "
                      f"pass@={r['passed_at_attempt']} tried={[a['candidate'] for a in r['attempts']]} "
                      f"hold=[{','.join(str(None if a['measured']['hold_mean'] is None else round(a['measured']['hold_mean'],1)) for a in r['attempts'])}]", flush=True)
    # determinism check: re-execute one cached candidate and compare the measurement bitwise
    if CACHE:
        (t, i, c), ref = next(iter(CACHE.items()))
        CACHE.pop((t, i, c))
        again = episode(t, i, c)
        same = json.dumps(again["measured"], sort_keys=True) == json.dumps(ref["measured"], sort_keys=True)
        (OUT / f"determinism_t{task}_{'_'.join(map(str, inits))}.json").write_text(json.dumps(
            {"task": t, "init": i, "candidate": list(c), "identical_measurements": same,
             "first": ref["measured"], "second": again["measured"]}, indent=1))
        print("determinism check: identical =", same, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
