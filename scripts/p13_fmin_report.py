import json, glob, sys, collections
import numpy as np
R = [json.loads(l) for p in glob.glob("reports/context_20261016/fmin_scan.*.jsonl") for l in open(p)]
main = [r for r in R if r["lift_v"] == 0.12]
diag = [r for r in R if r["lift_v"] != 0.12]
print("records", len(R), "main", len(main), "diag", len(diag), "exceptions", sum(r["status"] == "exception" for r in R))
cells = sorted({(r["mass"], r["mu"]) for r in main}, key=lambda c: (c[0] or 0, c[1] or 9))
grips = sorted({r["grip_lo"] for r in main})
print("\ncell pass counts (of 8 seeds) by grip command; F_min = lowest grip with >=6/8 pass; mono = pass-rate non-decreasing")
print("mass   mu    F_req | " + " ".join(f"{g:>2d}" for g in grips) + " | F_min  mono")
rows = []
for c in cells:
    k = [sum(r["task_ok"] for r in main if (r["mass"], r["mu"]) == c and r["grip_lo"] == g) for g in grips]
    n = [sum(1 for r in main if (r["mass"], r["mu"]) == c and r["grip_lo"] == g) for g in grips]
    fm = next((g for g, a, b in zip(grips, k, n) if b and a / b >= 0.75), None)
    mono = all(k[i] <= k[i + 1] + 1 for i in range(len(k) - 1))
    fr = [r["f_req"] for r in main if (r["mass"], r["mu"]) == c][0]
    rows.append((c, fr, k, fm, mono))
    print(f"{str(c[0]):>5} {str(c[1]):>5} {fr:6.1f} | " + " ".join(f"{a:>2d}" for a in k) + f" | {str(fm):>5} {mono}")
print("\nF_min vs F_req (cells with F_min):")
xs = [(fr, fm) for c, fr, k, fm, m in rows if fm is not None]
a, b = np.array(xs).T
from scipy.stats import spearmanr
print("n", len(xs), "spearman", round(float(spearmanr(a, b)[0]), 3), "ratio F_min/F_req median", round(float(np.median(b / a)), 2), "p10-p90", np.round(np.percentile(b / a, [10, 90]), 2))
print("cells without F_min (never >=6/8):", [c for c, fr, k, fm, m in rows if fm is None])
print("\nper-seed minimal passing grip spread inside cells (std/mean), cells with F_min:")
sp = []
for c, fr, k, fm, m in rows:
    if fm is None: continue
    ms = []
    for sd in sorted({r["seed"] for r in main}):
        ok = [r["grip_lo"] for r in main if (r["mass"], r["mu"]) == c and r["seed"] == sd and r["task_ok"]]
        ms.append(min(ok) if ok else np.nan)
    sp.append((c, np.round(ms, 0).tolist()))
for c, ms in sp[:30]: print(c, ms)
print("\nDIAGNOSIS lift speed at grip 36 (pass/8):")
for c in sorted({(r["mass"], r["mu"]) for r in diag}, key=lambda c: (c[0], c[1] or 9)):
    base = sum(r["task_ok"] for r in main if (r["mass"], r["mu"]) == c and r["grip_lo"] == 36)
    s = {v: sum(r["task_ok"] for r in diag if (r["mass"], r["mu"]) == c and r["lift_v"] == v) for v in (0.04, 0.3)}
    print(f"{str(c[0]):>5} {str(c[1]):>5} v=0.04:{s[0.04]} v=0.12:{base} v=0.3:{s[0.3]}")
st = collections.Counter((r["mass"] == 1.0, r["status"]) for r in main if r["grip_lo"] >= 28)
print("\nstatus (mass==1.0?, status) for grip>=28:", dict(st))
