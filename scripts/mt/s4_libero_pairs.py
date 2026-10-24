"""Step 4c: object-force-swapped trajectories on the frozen test initial states (0..7, disjoint from calibration 8..13)
under the frozen calibration configs/manner_calibration_libero_t1.v1.json.  Combos (cream_cheese, butter) in {G,F}^2;
the swapped pair is GF vs FG.  Each trajectory goes through the shared audit of s3 (measurement, event interface,
exported-action replay, mid-hold resume) plus a per-object force-band audit against the frozen bands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import OBJECTS, ROOT  # noqa: E402
from s3_libero_validate import hold_idx, one_init  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402

CFG = ROOT / "configs" / os.environ.get("MC_CFG", "manner_calibration_libero_t1.v1.json")
OUT = ROOT / "reports" / os.environ.get("MC_PAIRS_DIR", "step4_libero_pairs_20261025")
TEST_INIT = tuple(range(8))
LV = {"G": "gentle", "F": "firm"}
COMBOS = ("GF", "FG", "GG", "FF")


def load_cal(inits=TEST_INIT) -> dict:
    cal = json.loads(CFG.read_text())
    h = cal.pop("content_hash")
    assert h == hashlib.sha256(json.dumps(cal, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "calibration modified"
    cal["content_hash"] = h
    assert not set(inits) & set(cal["dev_inits"])
    return cal


def band_audit(a, cal, combo) -> dict:
    out = {}
    st = np.asarray(a["stage"])
    for (role, _), c in zip(OBJECTS, combo):
        o, lvl = cal["objects"][role], LV[c]
        lo, hi = o[f"{lvl}_lower"], o[f"{lvl}_upper"]
        h = hold_idx(a, role)
        tot = np.asarray(a["left_mean_n"])[h] + np.asarray(a["right_mean_n"])[h]
        pk = (np.asarray(a["left_max_n"]) + np.asarray(a["right_max_n"]))[np.isin(st, [f"{role}.pick_1.{x}" for x in ("grasp", "lift", "transport")])]
        out[role] = {"level": lvl, "band": [lo, hi], "hold_min": float(tot.min()), "hold_max": float(tot.max()), "hold_mean": float(tot.mean()),
                     "all_hold_steps_in_band": bool(tot.min() >= lo and tot.max() <= hi),
                     "peak_total_n": float(np.nanmax(pk)), "peak_le_cap": bool(np.nanmax(pk) <= o["peak_cap_n"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(TEST_INIT))
    ap.add_argument("--combos", nargs="+", default=list(COMBOS))
    args = ap.parse_args()
    cal = load_cal(args.inits)
    OUT.mkdir(parents=True, exist_ok=True)
    for i in args.inits:
        for combo in args.combos:
            f = OUT / f"init{i}_{combo}.json"
            if f.exists():
                continue
            grips = {role: GripPlan(cal["plans"][LV[c]]["target_lo_n"], cal["plans"][LV[c]]["target_hi_n"]) for (role, _), c in zip(OBJECTS, combo)}
            rec = one_init(i, grip=grips, out=OUT, tag=f"_{combo}")
            a = np.load(OUT / f"gen_init{i}_{combo}.npz")
            rec["combo"], rec["calibration_hash"] = combo, cal["content_hash"]
            rec["band_audit"] = band_audit(a, cal, combo)
            f.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
            b = rec["band_audit"]
            print(i, combo, "completion", rec["completion"], "measure", all(rec["measure_pass"].values()), "replay", rec["replay"]["bitwise_all_columns"],
                  "resume", sum(r["bitwise"] for r in rec["resume"]), "/", len(rec["resume"]), "events", rec["events"]["event_sequence_exact"],
                  rec["events"]["phase_mismatch_steps_outside_6step_boundary"], "band", {r: (v["all_hold_steps_in_band"], v["peak_le_cap"], round(v["hold_mean"], 1)) for r, v in b.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
