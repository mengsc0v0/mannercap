"""P1-continuity: sweep a manner command on a fine grid and record the REALISED physical value.

usage: p11_continuity_sweep.py MODE OUT.jsonl SEEDS(comma) LEVELS(comma) [SHARD/NSHARDS]
MODE: wipe_speed | wipe_press | lift_grip | stack_grip | lift_speed | stack_speed
  wipe_speed : commanded sweep speed (m/s), press fixed at 30 N        -> sweep core-median eef speed
  wipe_press : target mean tool force (N), sweep speed fixed 0.10 m/s -> sweep core-median tool force
  *_grip     : target hold force lower bound (N; band = [lo, lo+GRIP_BAND_WIDTH_N]) -> hold-window total force
  *_speed    : commanded v_max (m/s) applied to every stage            -> per-stage core-median eef speed
Appends one JSON line per (level, seed); existing (level, seed) pairs are skipped (resumable).
Use development seeds disjoint from calibration (9001-9024, 9101-9108) and production seeds.
"""
import json, os, sys, time

import numpy as np

from synthesis_v2.manner_calibration import DEFAULT_POLICY, core_indices, eef_speed_mps, hold_force_total, hold_indices
from synthesis_v2.manner_request import GRIP_BAND_WIDTH_N
from synthesis_v2.native_producer import GripPlan, ProductionPlan, StagePlan, produce_episode
from synthesis_v2.candidate_plan import MAX_ROT_RPS, MAX_STAGE_STEPS, ROT_PER_LIN, WIPE_FIXED_SPEED_MPS, WIPE_STAGE_STEPS
from synthesis_v2.stage_table import stage_ids

WIPE_FIXED_PRESS_N = 30.0
WIPE_FIXED_SPEED = 0.10
GRIP_FIXED_LO_N = 14.0


def _finite0(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def run_one(mode: str, level: float, seed: int) -> dict:
    task, kind = mode.split("_")
    rec = {"mode": mode, "level": level, "seed": seed}
    t0 = time.time()
    if task == "wipe":
        from synthesis_v2.wipe_producer import PressPlan, WipePlan, produce_wipe_episode
        v, f = (level, WIPE_FIXED_PRESS_N) if kind == "speed" else (WIPE_FIXED_SPEED, level)
        stages = tuple(StagePlan(sid, v if sid == "sweep" else WIPE_FIXED_SPEED_MPS, 1.0, WIPE_STAGE_STEPS[sid])
                       for sid in stage_ids("wipe"))
        ep = produce_wipe_episode("wipe", seed, WipePlan("wipe", stages, PressPlan(f)), keep_images=False)
    else:
        if kind == "speed":
            # a float applies to every stage; a dict {stage: v_max} sets stages independently (verification)
            cmd = (lambda sid: level[sid]) if isinstance(level, dict) else (lambda sid: level)
            stages = tuple(StagePlan(sid, cmd(sid), w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * cmd(sid)), max_steps=MAX_STAGE_STEPS)
                           for sid in stage_ids(task))
            grip = GripPlan(GRIP_FIXED_LO_N, GRIP_FIXED_LO_N + GRIP_BAND_WIDTH_N)
        else:
            stages = tuple(StagePlan(sid, 0.12, w_max_rps=1.0, max_steps=MAX_STAGE_STEPS) for sid in stage_ids(task))
            grip = GripPlan(level, level + GRIP_BAND_WIDTH_N)
        ep = produce_episode(task, seed, ProductionPlan(task, stages, grip), keep_images=False)
    rec.update(status=ep.status, task_ok=bool(ep.task_ok), n_steps=int(ep.n_steps), sec=round(time.time() - t0, 1))
    if not ep.n_steps:
        return rec
    a = ep.arrays
    st = np.asarray(a["stage"])
    spd = eef_speed_mps(a)
    if task == "wipe":
        idx = core_indices(st, "sweep", DEFAULT_POLICY)
        v = spd[idx]
        f = _finite0(a["tool_mean_n"])[idx]
        pk = _finite0(a["tool_max_n"])
        rec.update(speed_med=float(np.median(v[np.isfinite(v)])) if len(idx) else None,
                   press_med=float(np.median(f)) if len(idx) else None,
                   press_p10=float(np.percentile(f, 10)) if len(idx) else None,
                   press_p90=float(np.percentile(f, 90)) if len(idx) else None,
                   peak_total_n=float(pk.max()), n_core=int(len(idx)))
    elif kind == "speed":
        rec["stage_speed_med"] = {}
        for sid in stage_ids(task):
            idx = core_indices(st, sid, DEFAULT_POLICY)
            vv = spd[idx]
            vv = vv[np.isfinite(vv)]
            rec["stage_speed_med"][sid] = float(np.median(vv)) if len(vv) >= DEFAULT_POLICY["min_core_samples"] else None
    else:
        idx = hold_indices(a, task, DEFAULT_POLICY)
        if len(idx) >= DEFAULT_POLICY["min_core_samples"]:
            tot = hold_force_total(a, idx)
            valid = np.asarray(a["valid_fraction"], dtype=float)[idx] >= 1.0 - 1e-9
            rec.update(hold_med=float(np.nanmedian(tot)), hold_min=float(np.nanmin(tot)), hold_max=float(np.nanmax(tot)),
                       hold_valid=bool(valid.all()), n_hold=int(len(idx)))
        pk = np.nan_to_num(np.asarray(a["left_max_n"], dtype=float) + np.asarray(a["right_max_n"], dtype=float))
        rec["peak_total_n"] = float(pk.max())
        rec["grasp_policy"] = ep.meta.get("grasp_policy")
    return rec


def main() -> None:
    mode, out = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")]
    levels = [float(x) for x in sys.argv[4].split(",")]
    shard, nshard = (int(x) for x in sys.argv[5].split("/")) if len(sys.argv) > 5 else (0, 1)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["level"], r["seed"]))
    jobs = [(lv, sd) for lv in levels for sd in seeds][shard::nshard]
    with open(out, "a") as f:
        for lv, sd in jobs:
            if (lv, sd) in done:
                continue
            try:
                rec = run_one(mode, lv, sd)
            except Exception as exc:  # noqa: BLE001 - a failed episode is a data point, never silently dropped
                rec = {"mode": mode, "level": lv, "seed": sd, "status": "exception", "task_ok": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(mode, lv, sd, rec.get("status"), rec.get("task_ok"), rec.get("sec"), flush=True)


if __name__ == "__main__":
    main()
