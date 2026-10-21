"""Independent per-episode audit (P0-5 / P0-7).

Inputs are only the raw trajectory arrays, the *original* request language (it
is re-parsed here, the generator's own claim is not trusted) and the frozen
calibration artifact.  Output is three-state (pass / fail / unknown) for task,
manner (plus ``not_applicable`` for neutral requests) and safety.

Speed intents are checked on the stage core window through
``generation_validator``; the grip intent is checked on the hold window
(bilateral target contact and per-step total force inside the calibrated band).
Any missing / invalid / uncalibrated evidence yields ``unknown`` - never pass.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from synthesis_v2.generation_validator import (
    ConstraintSpec, Decision, GenerationCase, GenerationSample, GenerationTrace, validate_generation,
)
from synthesis_v2.grasp_force_contract import GraspState, summarize_grasp_force
from synthesis_v2.manner_calibration import (
    _finite0, core_indices, grip_bound, hold_force_total, hold_indices, press_bound, slow_transient_cap, speed_bound,
)
from synthesis_v2.manner_calibration import SPEED_METRIC, eef_speed_mps
from synthesis_v2.manner_request import ParseResult, parse_manner_request
from synthesis_v2.stage_table import stage_ids

AUDIT_VERSION = "episode_audit.v1"
REQUIRED_KEYS = (
    "state", "actions", "next_state", "timestamp", "dt", "frame_index", "sim_step_index",
    "terminated", "truncated", "success", "stage", "left_mean_n", "right_mean_n",
    "left_max_n", "right_max_n", "valid_fraction",
)
WIPE_REQUIRED_KEYS = tuple(k for k in REQUIRED_KEYS if not k.startswith(("left_", "right_", "valid_"))) + (
    "tool_mean_n", "tool_max_n")
UNITS = {"time": "s", "velocity": "rad/s"}
CONTROL_DT_S = 0.05
PASS, FAIL, UNKNOWN, NA = "pass", "fail", "unknown", "not_applicable"


def _join(*ds: str) -> str:
    if FAIL in ds:
        return FAIL
    if UNKNOWN in ds:
        return UNKNOWN
    return PASS


def _structure(arrays: Mapping[str, np.ndarray], task: str) -> list[str]:
    bad: list[str] = []
    required = WIPE_REQUIRED_KEYS if task == "wipe" else REQUIRED_KEYS
    missing = [k for k in required if k not in arrays]
    if missing:
        return [f"missing_arrays:{','.join(missing)}"]
    n = len(arrays["actions"])
    if n == 0:
        return ["empty_episode"]
    for k in required:
        if len(arrays[k]) != n:
            bad.append(f"length_mismatch:{k}")
    if bad:
        return bad
    a = np.asarray(arrays["actions"], dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() or np.abs(a).max() > 1.0 + 1e-6:
        bad.append("bad_actions")
    if task == "wipe" and np.abs(a[:, 6]).max() > 0.0:
        bad.append("wipe_gripper_action_not_zero")
    if not np.isfinite(np.asarray(arrays["state"], dtype=float)).all():
        bad.append("non_finite_state")
    t = np.asarray(arrays["timestamp"], dtype=float)
    if n > 1 and (not np.isfinite(t).all() or np.abs(np.diff(t) - CONTROL_DT_S).max() > 1e-6):
        bad.append("timestamp_not_uniform_20hz")
    valid = set(stage_ids(task))
    if not set(np.asarray(arrays["stage"]).tolist()) <= valid:
        bad.append("unknown_stage_label")
    return bad


def _speed_check(arrays, task, stage, level, cal, pol, episode_id) -> dict[str, Any]:
    lo, hi, why = speed_bound(cal, task, stage, level)
    rec: dict[str, Any] = {"stage": stage, "dimension": "speed", "level": level, "metric": SPEED_METRIC,
                           "lower": lo, "upper": hi}
    if why:
        return {**rec, "decision": UNKNOWN, "reason": why}
    idx = core_indices(np.asarray(arrays["stage"]), stage, pol)
    if len(idx) < pol["min_core_samples"]:
        return {**rec, "decision": UNKNOWN, "reason": "core_window_too_short"}
    vals = eef_speed_mps(arrays)[idx]
    if not np.isfinite(vals).all():
        return {**rec, "decision": UNKNOWN, "reason": "invalid_speed_samples"}
    # slow: the core-window median must stay below the bound and no core sample may exceed
    # the transient cap (calibrations without a cap keep "every sample below the bound").
    # fast: the core-window median must exceed the bound
    # (a stage still decelerates at its end; a single spike cannot make it "fast").
    cap = slow_transient_cap(cal, task, stage) if level == "slow" else None
    if cal.get("speed", {}).get(task, {}).get(stage, {}).get("median_only"):
        cap = None  # wipe: both slow and fast are judged on the core-window median only
    if cap is not None:
        rec["transient_cap"] = cap
    if hi is not None and float(np.median(vals)) > hi:
        decision, reason = FAIL, "speed_above_upper"
    elif cap is not None and float(vals.max()) > cap:
        decision, reason = FAIL, "speed_transient_above_cap"
    elif lo is not None and float(np.median(vals)) < lo:
        decision, reason = FAIL, "speed_below_lower"
    else:
        decision, reason = PASS, ""
    return {**rec, "decision": decision, "reason": reason, "n": int(len(idx)), "min": float(vals.min()),
            "median": float(np.median(vals)), "max": float(vals.max())}


def _grip_check(arrays, task, level, cal, pol, grip_scale: float = 1.0) -> dict[str, Any]:
    lo, hi, why = grip_bound(cal, task, level, grip_scale)
    rec: dict[str, Any] = {"stage": "grasp", "dimension": "grip", "level": level, "lower": lo, "upper": hi}
    if why:
        return {**rec, "decision": UNKNOWN, "reason": why}
    idx = hold_indices(arrays, task, pol)
    if len(idx) < pol["min_core_samples"]:
        return {**rec, "decision": UNKNOWN, "reason": "hold_window_too_short"}
    mask = np.zeros(len(arrays["actions"]), dtype=bool)
    mask[idx] = True
    valid = np.asarray(arrays["valid_fraction"], dtype=float) >= 1.0 - 1e-9
    thr = cal["safety"][task]["contact_threshold_n"]
    s = summarize_grasp_force(arrays["left_mean_n"], arrays["right_mean_n"], valid, mask=mask,
                              dt_s=CONTROL_DT_S, contact_threshold_n=thr, min_bilateral_steps=len(idx))
    rec["grasp_state"] = s.state.value
    if s.state is GraspState.UNKNOWN:
        return {**rec, "decision": UNKNOWN, "reason": f"grasp_force_{s.reason}"}
    if s.state is not GraspState.BILATERAL:
        return {**rec, "decision": FAIL, "reason": f"hold_not_bilateral:{s.state.value}"}
    tot = hold_force_total(arrays, idx)
    rec.update(n=int(len(idx)), min=float(tot.min()), median=float(np.median(tot)), max=float(tot.max()))
    if tot.min() < lo:
        return {**rec, "decision": FAIL, "reason": "hold_force_below_band"}
    if tot.max() > hi:
        return {**rec, "decision": FAIL, "reason": "hold_force_above_band"}
    return {**rec, "decision": PASS, "reason": ""}


def _press_check(arrays, task, level, cal, pol) -> dict[str, Any]:
    lo, hi, why = press_bound(cal, task, level)
    rec: dict[str, Any] = {"stage": "sweep", "dimension": "press", "level": level, "metric": "tool_mean_n_core_median",
                           "lower": lo, "upper": hi}
    if why:
        return {**rec, "decision": UNKNOWN, "reason": why}
    idx = core_indices(np.asarray(arrays["stage"]), "sweep", pol)
    if len(idx) < pol["min_core_samples"]:
        return {**rec, "decision": UNKNOWN, "reason": "core_window_too_short"}
    raw = np.asarray(arrays["tool_mean_n"], dtype=float)[idx]
    f = _finite0(raw)  # NaN = no tool contact that step = 0 N
    med = float(np.median(f))
    rec.update(n=int(len(idx)), median=med, min=float(f.min()), max=float(f.max()), n_no_contact=int((~np.isfinite(raw)).sum()))
    if lo is not None and med < lo:
        return {**rec, "decision": FAIL, "reason": "press_below_lower"}
    if hi is not None and med > hi:
        return {**rec, "decision": FAIL, "reason": "press_above_upper"}
    return {**rec, "decision": PASS, "reason": ""}


def _safety_check(arrays, task, cal, episode_id) -> dict[str, Any]:
    cap = cal.get("safety", {}).get(task, {}).get("peak_cap_n")
    if cap is None:
        return {"decision": UNKNOWN, "reason": "no_safety_cap"}
    st = np.asarray(arrays["stage"])
    t = np.asarray(arrays["timestamp"], dtype=float)
    if task == "wipe":  # tool-vs-world contact force; NaN = no contact = 0 N
        peak = _finite0(arrays["tool_max_n"])
        ok = np.ones(len(peak), dtype=bool)
    else:
        peak = np.asarray(arrays["left_max_n"], dtype=float) + np.asarray(arrays["right_max_n"], dtype=float)
        ok = (np.asarray(arrays["valid_fraction"], dtype=float) >= 1.0 - 1e-9) & np.isfinite(peak)
    samples = [GenerationSample(t=float(t[i]), phase=str(st[i]), contact_force_n=float(peak[i]) if ok[i] else None,
                                sensor_valid=bool(ok[i]), reset_id=episode_id, dt_s=CONTROL_DT_S) for i in range(len(t))]
    phases = tuple(dict.fromkeys(str(s) for s in st))
    spec = ConstraintSpec(stage_id="episode", target_phases=phases, safety_max_force_n=float(cap),
                          min_samples=1, max_gap_s=CONTROL_DT_S * 1.5)
    units = {**UNITS, "contact_force": "N"}
    res = validate_generation(GenerationCase(GenerationTrace.from_samples(samples, units=units), spec))
    fin = peak[np.isfinite(peak)]
    return {"decision": res.safety.value, "reason": ";".join(res.reasons), "cap_n": float(cap),
            "peak_total_n": float(fin.max()) if fin.size else None, "n_invalid": int((~ok).sum())}


def audit_episode(arrays: Mapping[str, np.ndarray], *, task: str, seed: int, raw_language: str,
                  cal: Mapping[str, Any], request: Optional[ParseResult] = None,
                  grip_scale: float = 1.0) -> dict[str, Any]:
    """``request``: the registered label of the request (used when the language is a paraphrase that the
    lexicon parser cannot read); without it the language is re-parsed here."""
    out: dict[str, Any] = {"audit_version": AUDIT_VERSION, "calibration_sha256": cal.get("content_sha256"),
                           "task_name": task, "seed": int(seed), "checks": [], "reasons": []}
    bad = _structure(arrays, task)
    req = request if request is not None else parse_manner_request(task, raw_language)
    out["request"] = req.to_dict()
    if bad:
        out.update(task=UNKNOWN, manner=UNKNOWN, safety=UNKNOWN, overall=UNKNOWN, reasons=bad)
        return out
    pol = cal["policy"]
    episode_id = f"{task}:{seed}"
    succ = np.asarray(arrays["success"]).astype(bool)
    task_d = PASS if succ[-1] else FAIL
    out["task"] = task_d
    if req.status == "unsupported":
        manner = UNKNOWN
        out["reasons"].extend(f"request:{r}" for r in req.reasons)
    elif req.status == "neutral":
        manner = NA
    else:
        decisions = []
        for it in req.intents:
            c = (_speed_check(arrays, task, it.stage, it.level, cal, pol, episode_id) if it.dimension == "speed"
                 else _press_check(arrays, task, it.level, cal, pol) if it.dimension == "press"
                 else _grip_check(arrays, task, it.level, cal, pol, grip_scale))
            out["checks"].append(c)
            decisions.append(c["decision"])
            if c["decision"] != PASS:
                out["reasons"].append(f"{it.dimension}:{it.stage}:{it.level}:{c['decision']}:{c['reason']}")
        manner = _join(*decisions)
    out["manner"] = manner
    saf = _safety_check(arrays, task, cal, episode_id)
    out["safety"], out["safety_detail"] = saf["decision"], saf
    if task_d != PASS:
        out["reasons"].append("task_failed")
    if saf["decision"] != PASS:
        out["reasons"].append(f"safety:{saf['decision']}:{saf['reason']}")
    out["overall"] = _join(task_d, saf["decision"], PASS if manner == NA else manner)
    return out
