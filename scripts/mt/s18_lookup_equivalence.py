"""Step 2 (2026-11-01): is the frozen experience library already a 'calibrated lookup' baseline?  Offline, no simulation,
and it looks at rankings only (never at trial outcomes).

Calibrated lookup (definition frozen in configs/calibrated_lookup_c1.json):
  records  : same task, same requested object, same stage window, measurement_valid and grasp_stable
  per candidate (lo, hi): records whose target_hi_n equals the candidate's hi (all same-object records if there are none);
             group by target_lo_n; summarise hold_min / hold_max per lo by the MEDIAN; piecewise-linear in lo
             (clamped outside the measured range)
  ranking  : identical to order_B (inside the requested band first by descending margin, then by distance of the
             predicted interval centre to the band centre, ties by candidate list order)
  no current-state information (no init index, no object pose) is used -- same as the library retrieval.
The library retrieval (order_B) differs only by the aggregation: envelope (min of hold_min, max of hold_max) instead of median,
and by its task/all-record fallbacks (never triggered here).
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10_retrieval_trial as T  # noqa: E402

E1, LIB, CAND = T.E1, T.LIB, T.CAND
OUT = T.ROOT / "reports" / "step18_lookup_equivalence"


def usable_same_object(task, obj):
    return [r for r in LIB["records"] if r["task"] == task and r["object"] == obj
            and r["validity"]["measurement_valid"] and r["validity"]["grasp_stable"]
            and r["measured"]["hold_min"] is not None]


def predict_lookup(task, obj, lo, hi):
    usable = usable_same_object(task, obj)
    sel = [r for r in usable if r["control_params"]["target_hi_n"] == hi] or usable
    g = {}
    for r in sel:
        g.setdefault(r["control_params"]["target_lo_n"], []).append(r["measured"])
    xs = sorted(g)
    mn = [statistics.median(m["hold_min"] for m in g[x]) for x in xs]
    mx = [statistics.median(m["hold_max"] for m in g[x]) for x in xs]
    return T._interp(xs, mn, lo), T._interp(xs, mx, lo)


def order_lookup(task, obj, band):
    lb, hb = band
    rows = []
    for idx, (lo, hi) in enumerate(CAND):
        pmin, pmax = predict_lookup(task, obj, lo, hi)
        inside = pmin >= lb and pmax <= hb
        margin = min(pmin - lb, hb - pmax)
        gap = abs(0.5 * (pmin + pmax) - 0.5 * (lb + hb))
        rows.append(((0, -margin, idx) if inside else (1, gap, idx), (lo, hi), pmin, pmax))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows]


def main():
    req = E1["scope"]["requested_object"]
    rows, coverage = [], {}
    for task in (0, 7):
        obj = [n for _, n in T.SPECS[task].objects if n.startswith(req[str(task)])][0]
        u = usable_same_object(task, obj)
        measured = sorted({(r["control_params"]["target_lo_n"], r["control_params"]["target_hi_n"]) for r in u})
        coverage[f"T{task}:{obj}"] = {"n_usable_records": len(u), "measured_candidates": [list(m) for m in measured],
                                      "pool_candidates_with_direct_measurement": [list(c) for c in CAND if tuple(c) in measured],
                                      "pool_size": len(CAND), "stage_windows_in_library": sorted({r["stage_window"] for r in u})}
        for R in E1["manner_requests"]:
            a = [tuple(r["cand"]) for r in T.order_B(obj, task, R["band_n"])]
            b = order_lookup(task, obj, R["band_n"])
            rows.append({"task": task, "object": obj, "request": R["id"], "band_n": R["band_n"],
                         "library_order": [list(x) for x in a], "lookup_order": [list(x) for x in b],
                         "same_full_order": a == b, "same_first_choice": a[0] == b[0]})
    out = {"schema": "mannercap.lookup_equivalence.v1", "library_content_hash": LIB["content_hash"],
           "e1_content_hash": E1["content_hash"], "n_requests": len(rows),
           "same_full_order": sum(r["same_full_order"] for r in rows),
           "same_first_choice": sum(r["same_first_choice"] for r in rows),
           "coverage": coverage, "rows": rows,
           "conclusion": "information-equivalent: the library retrieval uses task/object/stage/candidate-hi and the measured "
                         "response only (no current-state matching); it differs from the median lookup only by the envelope "
                         "aggregation. Position the method as measured-response parameter retrieval."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "equivalence.json").write_text(json.dumps(out, indent=1))
    print("full-order identical", out["same_full_order"], "/", len(rows), "| first choice identical", out["same_first_choice"], "/", len(rows))
    print(json.dumps(coverage)[:700])


if __name__ == "__main__":
    main()
