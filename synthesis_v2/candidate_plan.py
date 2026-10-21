"""Candidate JSON (as produced by a proposer) -> executable ``ProductionPlan``."""
from __future__ import annotations

from typing import Any, Mapping

from synthesis_v2.manner_request import GRIP_BAND_WIDTH_N
from synthesis_v2.native_producer import GripPlan, ProductionPlan, StagePlan
from synthesis_v2.stage_table import stage_ids

NEUTRAL_SPEED_MPS = 0.12
MAX_STAGE_STEPS = 800
# The speed level scales the whole stage motion, rotation included: otherwise a
# stage whose duration is set by the (fixed-rate) rotation alignment barely moves
# the end effector at the "fast" level.  1 rad/s at the neutral 0.12 m/s, capped.
ROT_PER_LIN = 1.0 / NEUTRAL_SPEED_MPS
MAX_ROT_RPS = 3.0


# wipe: only the sweep stroke carries a manner; repositioning stages run at a fixed speed
WIPE_FIXED_SPEED_MPS = 0.3
WIPE_STAGE_STEPS = {"approach": 250, "descend": 250, "sweep": 1000, "retreat": 250}  # fixed budget, never raised


def _wipe_plan(candidate: Mapping[str, Any]):
    from synthesis_v2.wipe_producer import PressPlan, WipePlan

    sweep_v = next(float(e["v_max_mps"]) for e in candidate["stages"] if e["stage_id"] == "sweep")
    stages = tuple(StagePlan(sid, sweep_v if sid == "sweep" else WIPE_FIXED_SPEED_MPS, 1.0, WIPE_STAGE_STEPS[sid])
                   for sid in stage_ids("wipe"))
    return WipePlan("wipe", stages, PressPlan(float(candidate["press"]["target_n"])))


def plan_from_candidate(task: str, candidate: Mapping[str, Any]) -> ProductionPlan:
    if task == "wipe":
        return _wipe_plan(candidate)
    speeds = {e["stage_id"]: float(e["v_max_mps"]) for e in candidate["stages"]}
    stages = tuple(StagePlan(sid, speeds.get(sid, NEUTRAL_SPEED_MPS),
                       w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * speeds.get(sid, NEUTRAL_SPEED_MPS)),
                       max_steps=MAX_STAGE_STEPS) for sid in stage_ids(task))
    lo = float(candidate["grip"]["target_lo_n"])
    return ProductionPlan(task, stages, GripPlan(lo, lo + GRIP_BAND_WIDTH_N))
