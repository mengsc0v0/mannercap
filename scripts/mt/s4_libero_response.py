"""Step 4a: command -> held-force response and stability of the LIBERO task, per object, on calibration
initial states (never the frozen test states 0..7).  One full two-object episode per (init, band-lower-edge);
both objects use the same band, so each run yields one response point per object."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import OBJECTS, ROOT, run  # noqa: E402
from s3_libero_validate import TRIM, hold_idx  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402

CAL_INIT = tuple(range(8, 14))
LADDER = (1.0, 4.0, 8.0, 15.0, 25.0, 35.0)
OUT = ROOT / "reports" / os.environ.get("MC_CALIB_DIR", "step4_libero_calib_20261025")


def one(init: int, lo: float, hi: float) -> dict:
    prog, s, ep = run(init, grip=GripPlan(lo, hi))
    a, st = ep.arrays, np.asarray(ep.arrays["stage"])
    rec = {"init": init, "plan": [lo, hi], "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "objects": {}}
    for role, name in OBJECTS:
        h = hold_idx(a, role)
        L, R = np.asarray(a["left_mean_n"])[h], np.asarray(a["right_mean_n"])[h]
        tot = L + R
        g = np.flatnonzero(st == f"{role}.pick_1.grasp")
        z_rel = np.asarray(a["pos_" + name])[h, 2] - np.asarray(a["state"])[h, 2]
        pk = (np.asarray(a["left_max_n"]) + np.asarray(a["right_max_n"]))[np.isin(st, [f"{role}.pick_1.{x}" for x in ("grasp", "lift", "transport")])]
        rec["objects"][role] = {
            "hold_n": int(len(h)), "hold_total": [float(x) for x in tot],
            "hold_mean": float(tot.mean()), "hold_min": float(tot.min()), "hold_max": float(tot.max()),
            "lr_imbalance": float(np.mean(np.abs(L - R) / np.maximum(L + R, 1e-9))),
            "bilateral": bool(min(L.min(), R.min()) > 0.1),
            "rel_z_spread_m": float(np.ptp(z_rel)),
            "peak_total_n": float(np.nanmax(pk)),
            "close_cmds": int((np.asarray(a["actions"])[g, 6] > 0.5).sum()),
            "grasp_trace_total": [float(x) for x in (np.asarray(a["left_mean_n"]) + np.asarray(a["right_mean_n"]))[g][:14]],
        }
    s.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(CAL_INIT))
    ap.add_argument("--los", type=float, nargs="+", default=list(LADDER))
    ap.add_argument("--plans", nargs="+", default=[], help="extra lo:hi bands (hi<100 enables reverse pulses)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plans = [(lo, 100.0) for lo in a.los] + [tuple(float(x) for x in t.split(":")) for t in a.plans]
    if a.plans:
        plans = [tuple(float(x) for x in t.split(":")) for t in a.plans]
    for i in a.inits:
        for lo, hi in plans:
            f = OUT / (f"resp_init{i}_lo{lo:g}.json" if hi == 100.0 else f"resp_init{i}_lo{lo:g}_hi{hi:g}.json")
            if f.exists():
                continue
            r = one(i, lo, hi)
            f.write_text(json.dumps(r))
            o = r["objects"]
            print(i, lo, hi, r["status"], "task_ok", r["task_ok"], {k: (round(v["hold_mean"], 1), round(v["rel_z_spread_m"], 3), v["close_cmds"]) for k, v in o.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
