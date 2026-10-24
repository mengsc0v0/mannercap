"""Step 4b: derive the per-object LIBERO force calibration from the dev response runs (same percentile
arithmetic as synthesis_v2/manner_calibration.derive_calibration; that function is task-keyed, hence re-stated here).
Levels are the command plans: gentle=(1,22) normal=(1,100) firm=(35,100); both objects use the same plan per level."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import OBJECTS, ROOT  # noqa: E402
from s4_libero_response import CAL_INIT, OUT  # noqa: E402
from synthesis_v2.manner_calibration import DEFAULT_POLICY  # noqa: E402

PLANS = {"gentle": (1.0, 22.0), "normal": (1.0, 100.0), "firm": (35.0, 100.0)}
CFG = ROOT / "configs" / os.environ.get("MC_CFG", "manner_calibration_libero_t1.v1.json")


def fname(i, lo, hi):
    return OUT / (f"resp_init{i}_lo{lo:g}.json" if hi == 100.0 else f"resp_init{i}_lo{lo:g}_hi{hi:g}.json")


def pct(v, q):
    return float(np.percentile(v, q))


def main() -> int:
    pol = dict(DEFAULT_POLICY)
    recs = {(lvl, i): json.loads(fname(i, *PLANS[lvl]).read_text()) for lvl in PLANS for i in CAL_INIT}
    cal = {"schema": "mannercap.libero_calibration.v1", "task": "libero_10/1", "dev_inits": list(CAL_INIT),
           "policy": {k: pol[k] for k in ("hold_trim_head_steps", "core_trim_fraction", "percentile_lo", "percentile_hi",
                                         "gentle_floor_fraction", "firm_cap_factor", "safety_margin_factor")},
           "plans": {k: {"target_lo_n": v[0], "target_hi_n": v[1]} for k, v in PLANS.items()},
           "contact_threshold_n": 0.1, "objects": {}}
    if os.environ.get("MC_CHANGE_NOTE"):
        cal["generation_change_note"] = os.environ["MC_CHANGE_NOTE"]  # recorded in (and hashed with) the config
    for role, _ in OBJECTS:
        hold, peaks, stable, ok = {}, [], {}, {}
        for lvl in PLANS:
            rs = [recs[(lvl, i)] for i in CAL_INIT]
            ok[lvl] = sum(bool(r["task_ok"]) for r in rs)
            stable[lvl] = sum(bool(r["task_ok"]) and r["objects"][role]["rel_z_spread_m"] <= 0.01 and r["objects"][role]["bilateral"]
                              and r["objects"][role]["lr_imbalance"] <= 0.10 for r in rs)
            good = [r for r in rs if r["task_ok"]]
            hold[lvl] = [x for r in good for x in r["objects"][role]["hold_total"]]
            peaks += [r["objects"][role]["peak_total_n"] for r in good]
        q = {lvl: {"n": len(h), "p5": pct(h, 5), "p95": pct(h, 95), "p99": pct(h, 99), "min": min(h), "max": max(h)} for lvl, h in hold.items()}
        sep_gn = q["gentle"]["p95"] < q["normal"]["p5"]
        sep_nf = q["normal"]["p95"] < q["firm"]["p5"]
        o = {"hold_stats": q, "task_ok_per_level": ok, "stable_per_level": stable, "n_dev": len(CAL_INIT),
             "gentle_lower": pol["gentle_floor_fraction"] * q["gentle"]["p5"],
             "gentle_upper": 0.5 * (q["gentle"]["p95"] + q["normal"]["p5"]) if sep_gn else None,
             "normal_lower": None, "normal_upper": None,
             "firm_lower": 0.5 * (q["normal"]["p95"] + q["firm"]["p5"]) if sep_nf else None,
             "firm_upper": pol["firm_cap_factor"] * q["firm"]["p99"] if sep_nf else None,
             "peak_cap_n": pol["safety_margin_factor"] * max(peaks), "dev_peak_max_n": max(peaks),
             "status": "ok" if (sep_gn and sep_nf and min(stable.values()) == len(CAL_INIT)) else "partially_inseparable"}
        o["normal_lower"], o["normal_upper"] = o["gentle_upper"], o["firm_lower"]
        cal["objects"][role] = o
    blob = json.dumps(cal, sort_keys=True, separators=(",", ":"))
    cal["content_hash"] = hashlib.sha256(blob.encode()).hexdigest()
    CFG.write_text(json.dumps(cal, indent=1, sort_keys=True))
    for r, o in cal["objects"].items():
        print(r, o["status"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items() if k not in ("hold_stats",)})
        print("  ", {l: (round(s["p5"], 1), round(s["p95"], 1)) for l, s in o["hold_stats"].items()})
    print("hash", cal["content_hash"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
