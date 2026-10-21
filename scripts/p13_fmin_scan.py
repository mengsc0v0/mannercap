"""Dense minimum-feasible-grip scan (oracle) per physical context, plus a lift-speed diagnosis.

usage: p13_fmin_scan.py OUT.jsonl SEEDS(comma) SHARD/NSHARDS
Main grid : 24 (mass, mu) cells x seeds x grip commands 4..40 step 4, all stages at 0.12 m/s.
Diagnosis : mass>=0.6 cells x seeds x grip 36 x lift-stage speed {0.04, 0.3} (other stages 0.12).
Every record keeps the full context so the analysis can be re-derived.
"""
import json, os, sys, time

import numpy as np

from scripts.p12_context_probe import FRICTIONS, MASSES, f_req
from synthesis_v2.candidate_plan import MAX_STAGE_STEPS
from synthesis_v2.manner_calibration import DEFAULT_POLICY, hold_force_total, hold_indices
from synthesis_v2.manner_request import GRIP_BAND_WIDTH_N
from synthesis_v2.native_producer import GripPlan, ProductionPlan, StagePlan, produce_episode
from synthesis_v2.physical_context import ContextSession, PhysicalContext
from synthesis_v2.stage_table import stage_ids

GRIPS = tuple(range(4, 41, 4))


def run(seed, m, mu, grip_lo, lift_v):
    t0 = time.time()
    stages = tuple(StagePlan(sid, lift_v if sid == "lift" else 0.12, w_max_rps=1.0, max_steps=MAX_STAGE_STEPS)
                   for sid in stage_ids("lift"))
    plan = ProductionPlan("lift", stages, GripPlan(grip_lo, grip_lo + GRIP_BAND_WIDTH_N))
    s = ContextSession("lift", seed, PhysicalContext(mass_kg=m, friction=mu), render=False)
    try:
        ep = produce_episode("lift", seed, plan, keep_images=False, session=s)
    finally:
        pass
    rec = {"seed": seed, "mass": m, "mu": mu, "grip_lo": grip_lo, "lift_v": lift_v, "f_req": f_req(m, mu),
           "status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "sec": round(time.time() - t0, 1)}
    if ep.n_steps:
        a = ep.arrays
        idx = hold_indices(a, "lift", DEFAULT_POLICY)
        if len(idx) >= DEFAULT_POLICY["min_core_samples"]:
            rec["hold_med"] = float(np.nanmedian(hold_force_total(a, idx)))
    return rec


def main():
    out, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    shard, nshard = (int(x) for x in sys.argv[3].split("/"))
    cells = [(m, mu) for m in MASSES for mu in FRICTIONS]
    jobs = [(sd, m, mu, g, 0.12) for (m, mu) in cells for sd in seeds for g in GRIPS]
    jobs += [(sd, m, mu, 36, v) for (m, mu) in cells if m is not None and m >= 0.6 for sd in seeds for v in (0.04, 0.3)]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["seed"], r["mass"], r["mu"], r["grip_lo"], r["lift_v"]))
    with open(out, "a") as f:
        for j in jobs[shard::nshard]:
            if j in done:
                continue
            try:
                rec = run(*j)
            except Exception as exc:  # noqa: BLE001
                rec = {"seed": j[0], "mass": j[1], "mu": j[2], "grip_lo": j[3], "lift_v": j[4], "status": "exception",
                       "task_ok": False, "error": f"{type(exc).__name__}: {exc}"}
            f.write(json.dumps(rec) + "\n"); f.flush()


if __name__ == "__main__":
    main()
