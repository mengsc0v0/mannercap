"""Offline audit of the arr_force_target label difference between retrieval+recover and retrieval+rerun
(step-20 frozen results; reads only reports/step20_selection_eval, nothing is modified).
For every request whose two arms executed the same candidate sequence: where (rows, stages) the per-step
force_target label differs, what the two labels are, whether the difference touches a hold window
(hold_instance != '' / finite hold band), and which object the instance binding (force_target_by_instance) names.
Run: PYTHONPATH=.:scripts/mt python scripts/mt/s21_force_target_audit.py"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s20_analyze as A  # noqa: E402  (re-uses the loaders; analysis is pure/offline)

rows, agg = [], collections.Counter()
by_pair = collections.Counter()
for k in A.KEYS:
    a, b = A.C[("retrieval+recover",) + k], A.C[("retrieval+rerun",) + k]
    ea, eb = A.last_exec(a), A.last_exec(b)
    za = np.load(A.OUT / ea["trajectory_file"], allow_pickle=True)
    zb = np.load(A.OUT / eb["trajectory_file"], allow_pickle=True)
    fa, fb = za["arr_force_target"], zb["arr_force_target"]
    assert fa.shape == fb.shape
    d = np.flatnonzero(fa != fb)
    hold_a = np.isfinite(za["arr_hold_lo_n"]) & (za["arr_hold_instance"] != "")
    hold_b = np.isfinite(zb["arr_hold_lo_n"]) & (zb["arr_hold_instance"] != "")
    in_hold = int((hold_a[d] | hold_b[d]).sum()) if len(d) else 0
    # fine-grained: do the two arms agree on the hold window itself and on the force telemetry in it?
    hold_same = bool(np.array_equal(hold_a, hold_b) and np.array_equal(za["arr_hold_instance"], zb["arr_hold_instance"]))
    # labels at hold rows (the rows the force verdict uses)
    lab_hold_a = sorted(set(fa[hold_a].tolist())) if hold_a.any() else []
    lab_hold_b = sorted(set(fb[hold_b].tolist())) if hold_b.any() else []
    hi = sorted(set(za["arr_hold_instance"][hold_a].tolist())) if hold_a.any() else []
    stages_d = collections.Counter(za["arr_stage"][d].tolist()) if len(d) else {}
    pairs_d = collections.Counter(f"{fa[i]!r}->{fb[i]!r}" for i in d)
    # instance bindings recorded by the executor (what the measurement was bound to at each 'grasp')
    ma, mb = (ea.get("meta") or {}), (eb.get("meta") or {})
    row = {"key": f"T{k[0]}-i{k[1]}-{k[2]}", "n_rows": int(len(fa)), "n_diff_rows": int(len(d)),
           "diff_row_range": [int(d.min()), int(d.max())] if len(d) else None,
           "diff_stages": dict(stages_d), "label_pairs_recover_vs_rerun": dict(pairs_d),
           "diff_rows_inside_hold_window": in_hold, "hold_window_and_instance_identical": hold_same,
           "hold_labels_recover": lab_hold_a, "hold_labels_rerun": lab_hold_b, "hold_instances": hi,
           "hold_label_one_object_each_instance": None}
    # for every hold instance: label inside the hold must be a single, same name in both arms
    ok = True
    for inst in hi:
        ma_ = hold_a & (za["arr_hold_instance"] == inst)
        mb_ = hold_b & (zb["arr_hold_instance"] == inst)
        la, lb = set(fa[ma_].tolist()), set(fb[mb_].tolist())
        ok &= (la == lb and len(la) == 1)
    row["hold_label_one_object_each_instance"] = bool(ok) if hi else None
    # force telemetry identical?
    row["force_arrays_identical"] = all(np.array_equal(za[x], zb[x], equal_nan=True) for x in za.files
                                        if x.startswith(("arr_left", "arr_right", "arr_other", "force_L_", "force_R_")))
    # (i) first-row label per arm; (ii) finger<->object contact force on the pre-grasp rows that differ
    row["row0_label_recover"], row["row0_label_rerun"] = str(fa[0]), str(fb[0])
    cfmax = 0.0
    for z in (za, zb):
        for nm in [x[len("arr_cf_L_"):] for x in z.files if x.startswith("arr_cf_L_")]:
            for side in ("L", "R"):
                v = z[f"arr_cf_{side}_{nm}"][d] if len(d) else np.array([])
                if v.size and np.isfinite(v).any():
                    cfmax = max(cfmax, float(np.nanmax(v)))
    row["max_finger_object_force_n_on_differing_rows"] = cfmax
    # (iii) does every hold instance's label equal the object the executor bound for that instance?
    bind = {}
    for m in (ma, mb):
        bind.update(m.get("force_target_by_instance") or {})
    row["executor_instance_binding"] = bind
    rows.append(row)
    agg["requests"] += 1
    agg["requests_with_label_diff"] += bool(len(d))
    agg["diff_rows_total"] += int(len(d))
    agg["diff_rows_inside_hold_total"] += in_hold
    agg["hold_window_identical"] += hold_same
    agg["hold_label_ok"] += bool(ok) if hi else 0
    agg["with_hold"] += bool(hi)
    agg["force_arrays_identical"] += row["force_arrays_identical"]
    agg["stale_label_rows_with_finger_object_force_gt0"] += row["max_finger_object_force_n_on_differing_rows"] > 0
    agg["row0_label_recover_nonempty"] += row["row0_label_recover"] != ""
    agg["row0_label_rerun_nonempty"] += row["row0_label_rerun"] != ""
res = {"schema": "mannercap.force_target_audit.v1", "source": "reports/step20_selection_eval (frozen, unmodified)",
       "summary": dict(agg), "per_request": rows}
out = Path(A.OUT) / "force_target_audit.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res["summary"]))
for r in rows:
    if r["n_diff_rows"]:
        print(r["key"], r["n_diff_rows"], r["diff_row_range"], r["diff_stages"], r["label_pairs_recover_vs_rerun"],
              "inHold", r["diff_rows_inside_hold_window"])
