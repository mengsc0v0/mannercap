"""Difficulty probe: open-loop Lift with a FIXED nominal grip command across a (mass, friction) grid.

usage: p12_context_probe.py OUT.jsonl SEEDS(comma) GRIP_LO_N [SHARD/NSHARDS]
Required total (left+right) normal force to hold the object against gravity with two frictional contacts is
F_req = m g / mu  (each finger carries half of the weight: 2 mu N >= m g, total force = 2 N).
"""
import json, os, sys, time

import numpy as np

from synthesis_v2.manner_calibration import DEFAULT_POLICY, hold_force_total, hold_indices
from synthesis_v2.manner_request import GRIP_BAND_WIDTH_N
from synthesis_v2.native_producer import GripPlan, ProductionPlan, StagePlan, produce_episode
from synthesis_v2.candidate_plan import MAX_STAGE_STEPS
from synthesis_v2.physical_context import ContextSession, PhysicalContext
from synthesis_v2.stage_table import stage_ids

G = 9.81
MASSES = (None, 0.15, 0.3, 0.6, 1.0)
FRICTIONS = (None, 1.0, 0.5, 0.25, 0.12)


def f_req(mass, mu, default_mass=0.0747):
    return (mass if mass is not None else default_mass) * G / (mu if mu is not None else 2.0)


def run(task, seed, ctx, grip_lo):
    t0 = time.time()
    stages = tuple(StagePlan(sid, 0.12, w_max_rps=1.0, max_steps=MAX_STAGE_STEPS) for sid in stage_ids(task))
    plan = ProductionPlan(task, stages, GripPlan(grip_lo, grip_lo + GRIP_BAND_WIDTH_N))
    s = ContextSession(task, seed, ctx, render=False)
    ep = produce_episode(task, seed, plan, keep_images=False, session=s)
    rec = {"task": task, "seed": seed, "mass": ctx.mass_kg, "mu": ctx.friction, "grip_lo": grip_lo,
           "f_req": f_req(ctx.mass_kg, ctx.friction), "status": ep.status, "task_ok": bool(ep.task_ok),
           "sec": round(time.time() - t0, 1), "context_record": s.context_record}
    if ep.n_steps:
        a = ep.arrays
        idx = hold_indices(a, task, DEFAULT_POLICY)
        if len(idx) >= DEFAULT_POLICY["min_core_samples"]:
            tot = hold_force_total(a, idx)
            rec.update(hold_med=float(np.nanmedian(tot)), n_hold=int(len(idx)))
        pk = np.nan_to_num(np.asarray(a["left_max_n"], dtype=float) + np.asarray(a["right_max_n"], dtype=float))
        rec["peak_total_n"] = float(pk.max())
        rec["grasp_policy"] = ep.meta.get("grasp_policy")
        rec["final_obj_z"] = ep.meta.get("final_obj_pos", [None] * 3)[2]
    return rec


def main():
    out = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(",")]
    grip_lo = float(sys.argv[3])
    shard, nshard = (int(x) for x in sys.argv[4].split("/")) if len(sys.argv) > 4 else (0, 1)
    jobs = [(m, mu, sd) for m in MASSES for mu in FRICTIONS for sd in seeds if not (m is None and mu is None)]
    jobs = [(None, None, sd) for sd in seeds] + jobs
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r["mass"], r["mu"], r["seed"]))
    with open(out, "a") as f:
        for m, mu, sd in jobs[shard::nshard]:
            if (m, mu, sd) in done:
                continue
            try:
                rec = run("lift", sd, PhysicalContext(mass_kg=m, friction=mu), grip_lo)
            except Exception as exc:  # noqa: BLE001
                rec = {"task": "lift", "seed": sd, "mass": m, "mu": mu, "status": "exception", "task_ok": False,
                       "error": f"{type(exc).__name__}: {exc}", "f_req": f_req(m, mu)}
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
