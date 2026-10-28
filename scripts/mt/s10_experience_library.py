"""E2: build the frozen experience library for the retrieval comparison (configs/experience_retrieval_e1.json).

One record per (episode, object) already produced during development (reports/step8_t*_response, reports/step8_t*_pairs).
Nothing is re-simulated and nothing is filtered out: records whose measurement was invalid or whose grasp was unstable are
kept with those flags, because a retrieval rule must be able to see what did NOT work.  Usage: s10_experience_library.py
"""
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT  # noqa: E402

OUT = ROOT / "reports" / "step10_experience" / "library.json"
MEAS = ("hold_n", "hold_min", "hold_mean", "hold_max", "lr_imbalance", "rel_z_spread_m", "bilateral_strict",
        "bilateral_step_fraction", "substep_bilateral_fraction", "peak_simultaneous_n", "peak_upper_bound_n")
LV = {"G": "gentle", "F": "firm"}


def plan_of(rec, role, roles):
    """Control parameters actually commanded for this object in this episode."""
    if "plan" in rec:                                     # response record: same plan for every object
        return {"target_lo_n": rec["plan"][0], "target_hi_n": rec["plan"][1], "level": rec.get("level")}
    if "combo" in rec:                                    # pair record: one letter per object, in program order
        cal = json.loads((ROOT / "configs" / f"manner_calibration_libero_t{rec['task']}.v3.json").read_text())
        lvl = LV[rec["combo"][roles.index(role)]]
        return {"target_lo_n": cal["plans"][lvl]["target_lo_n"], "target_hi_n": cal["plans"][lvl]["target_hi_n"], "level": lvl}
    return None


def main() -> int:
    t0 = time.time()
    recs, skipped = [], []
    for f in sorted(glob.glob(str(ROOT / "reports" / "step8_t*_response" / "*.json"))) + \
             sorted(glob.glob(str(ROOT / "reports" / "step8_t*_pairs" / "*.json"))):
        r = json.loads(Path(f).read_text())
        if r.get("status") == "error" or not r.get("objects"):
            skipped.append({"file": Path(f).name, "why": r.get("status", "no objects")})
            continue
        roles = list(r["objects"])
        for role, o in r["objects"].items():
            p = plan_of(r, role, roles)
            if p is None:
                skipped.append({"file": Path(f).name, "why": "no control parameters"})
                continue
            recs.append({
                "source": str(Path(f).relative_to(ROOT)),
                "task": r["task"], "init": r["init"], "role": role, "object": o.get("target"),
                "stage_window": "hold window of {role}.pick_1.grasp/lift/transport",
                "controller": "OSC_POSE, 20 Hz, LIBERO OffScreenRenderEnv",
                "executor": r.get("executor"),
                "requested": {"level": p.get("level"), "band_n": None},
                "control_params": {"target_lo_n": p["target_lo_n"], "target_hi_n": p["target_hi_n"],
                                   "settle_after_open": (r.get("executor") or {}).get("settle_after_open")},
                "measured": {k: o.get(k) for k in MEAS},
                "validity": {"status": o.get("status"), "measurement_valid": o.get("measurement_valid"),
                             "grasp_stable": o.get("grasp_stable"), "task_ok": r.get("task_ok")},
                "outcome": {"episode_status": r.get("status"), "n_steps": r.get("n_steps"), "wall_s": r.get("wall_s"),
                            "contact_policy_clean": (r.get("contact_policy") or {}).get("clean")}})
    lib = {"schema": "mannercap.experience_library.v1",
           "built_from": ["reports/step8_t0_response", "reports/step8_t7_response", "reports/step8_t0_pairs", "reports/step8_t7_pairs"],
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "development records only; the inits here (calibration 8-13 and dev 0-7/14-21) are disjoint from the "
                   "trial inits 26-29 of configs/experience_retrieval_e1.json. Invalid/unstable records are kept.",
           "n_records": len(recs), "skipped": skipped, "records": recs}
    lib["build_seconds"] = round(time.time() - t0, 2)
    lib["content_hash"] = hashlib.sha256(json.dumps(lib, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lib, indent=1))
    by = {}
    for r in recs:
        k = (r["task"], r["object"], r["control_params"]["target_lo_n"], r["control_params"]["target_hi_n"])
        by.setdefault(k, []).append(r)
    print("records", len(recs), "skipped", len(skipped), "hash", lib["content_hash"][:16], "build_s", lib["build_seconds"])
    for k in sorted(by):
        g = by[k]
        ok = [x for x in g if x["validity"]["measurement_valid"] and x["validity"]["grasp_stable"]]
        hm = [x["measured"]["hold_mean"] for x in ok if x["measured"]["hold_mean"] is not None]
        print(f"  T{k[0]} {k[1]:14s} lo={k[2]:5.1f} hi={k[3]:5.1f}  n={len(g):3d} usable={len(ok):3d} "
              f"hold_mean {min(hm):.1f}-{max(hm):.1f}" if hm else
              f"  T{k[0]} {k[1]:14s} lo={k[2]:5.1f} hi={k[3]:5.1f}  n={len(g):3d} usable=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
