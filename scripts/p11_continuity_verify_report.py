"""Summarise p11_continuity_verify.py output: realised vs requested (held-out, non-grid) targets.

usage: p11_continuity_verify_report.py VERIFY.jsonl OUT_PREFIX
Failed episodes are counted (n_failed), not silently dropped.
"""
import json, sys
from collections import defaultdict

import numpy as np

from synthesis_v2.response_curve import spearman


def pairs(rec):
    m, t = rec["mode"], rec["targets"]
    if rec.get("status") != "completed" or not rec.get("task_ok"):
        return []
    if m == "wipe_speed":
        return [("wipe/sweep speed", list(t.values())[0], rec.get("speed_med"))]
    if m == "wipe_press":
        return [("wipe/press force", list(t.values())[0], rec.get("press_med"))]
    if m == "lift_grip":
        return [("lift/grip hold force", list(t.values())[0], rec.get("hold_med") if rec.get("hold_valid") else None)]
    return [(k, v, (rec.get("stage_speed_med") or {}).get(k.split("/")[1].split()[0])) for k, v in t.items()]


def main():
    recs = [json.loads(l) for l in open(sys.argv[1])]
    n_ep, n_fail = defaultdict(int), defaultdict(int)
    xs, ys = defaultdict(list), defaultdict(list)
    for r in recs:
        for k in r["targets"]:
            n_ep[k] += 1
            if r.get("status") != "completed" or not r.get("task_ok"):
                n_fail[k] += 1
        for k, tg, re in pairs(r):
            if re is not None and np.isfinite(re):
                xs[k].append(tg)
                ys[k].append(re)
    rows, out = [], {}
    for k in sorted(xs):
        x, y = np.asarray(xs[k]), np.asarray(ys[k])
        slope, icpt = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        err = y - x
        rel = np.abs(err) / np.abs(x)
        span = float(x.max() - x.min())
        d = {"n_targets": n_ep[k], "n_failed": n_fail[k], "n_valid": len(x), "spearman": spearman(x, y), "r2": r2,
             "slope": float(slope), "intercept": float(icpt), "mae": float(np.abs(err).mean()),
             "mae_over_span": float(np.abs(err).mean() / span), "rel_err_median": float(np.median(rel)),
             "rel_err_p90": float(np.percentile(rel, 90)), "within_15pct": float(np.mean(rel <= 0.15)),
             "within_10pct_span": float(np.mean(np.abs(err) <= 0.10 * span)), "target_lo": float(x.min()),
             "target_hi": float(x.max())}
        out[k] = d
        rows.append(f"| {k} | {d['n_valid']}/{d['n_targets']} | {d['spearman']:.3f} | {d['r2']:.3f} | {d['slope']:.2f} | "
                    f"{d['mae']:.3g} | {d['mae_over_span']:.3f} | {d['rel_err_median']:.2f} | {d['rel_err_p90']:.2f} | "
                    f"{d['within_15pct']:.2f} | {d['within_10pct_span']:.2f} |")
    head = ("| quantity | valid/total | Spearman | R2 | slope | MAE | MAE/span | rel.err med | rel.err p90 | within 15% | "
            "within 10% span |\n|" + "---|" * 11)
    open(sys.argv[2] + ".json", "w").write(json.dumps(out, indent=1))
    open(sys.argv[2] + ".md", "w").write("# Held-out closed-loop verification (realised vs requested target)\n\n" + head + "\n"
                                         + "\n".join(rows) + "\n")
    print(head + "\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
