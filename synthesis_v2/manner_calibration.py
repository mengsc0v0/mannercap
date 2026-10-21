"""Development-set calibration of speed / grip-force bands (P0-2).

Everything numeric that the audit compares against lives in one JSON artifact
(``configs/manner_calibration.v1.json``) with a content hash.  It is derived
only from *development* seeds; ``assert_seed_isolated`` refuses to produce or
audit a request whose seed was used for calibration.

Band rules (all protocol constants are stored in ``policy`` inside the file):

* speed: for each (task, stage) the per-sample end-effector linear speed (m/s) of the *core*
  window (head/tail trimmed) is collected for the slow/normal/fast command
  levels.  ``slow_upper`` / ``fast_lower`` are the midpoints between adjacent
  levels' p95 and p5 (slow: every core sample below; fast: core median above).  If the levels overlap, the stage is ``inseparable`` and the
  threshold is ``null`` => audit returns ``unknown`` (never a guessed pass).
* grip: hold-window total target-object force for gentle/normal/firm.  gentle
  = [floor, midpoint(gentle,normal)], firm = [midpoint(normal,firm), cap];
  the non-zero floor prevents "gentle" from being satisfied by a failed grasp.
* safety: peak total finger force on any step must stay below
  ``safety_margin * max(dev peaks)``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

CALIBRATION_SCHEMA = "synthesis_v2.manner_calibration.v2"
SPEED_METRIC = "eef_linear_speed_mps"

DEFAULT_POLICY: dict[str, Any] = {
    "core_trim_fraction": 0.2,
    "min_core_samples": 3,
    "hold_trim_head_steps": 2,
    "percentile_lo": 5.0,
    "percentile_hi": 95.0,
    "gentle_floor_fraction": 0.5,
    "firm_cap_factor": 1.25,
    "safety_margin_factor": 1.25,
    "min_contact_threshold_n": 0.1,
    "slow_criterion": "all_core_samples_below_upper",
    "fast_criterion": "core_median_above_lower",
}

HOLD_STAGES = {"lift": ("lift",), "stack": ("lift", "transport"), "nut_assembly": ("lift", "transport")}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(cal: Mapping[str, Any]) -> str:
    body = {k: v for k, v in cal.items() if k != "content_sha256"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def eef_speed_mps(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-step end-effector linear speed (m/s) recomputed from the delivered
    state/next_state (xyz = first 3 dims) and dt.  Independent of arm
    configuration, unlike joint speed.  Invalid steps are NaN."""
    s = np.asarray(arrays["state"], dtype=float)[:, :3]
    n = np.asarray(arrays["next_state"], dtype=float)[:, :3]
    dt = np.asarray(arrays["dt"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.linalg.norm(n - s, axis=1) / dt
    v[~np.isfinite(v) | (dt <= 0)] = np.nan
    return v


def core_indices(stage_arr: np.ndarray, stage: str, policy: Mapping[str, Any]) -> np.ndarray:
    idx = np.flatnonzero(stage_arr == stage)
    n = len(idx)
    if n == 0:
        return idx
    trim = int(math.ceil(float(policy["core_trim_fraction"]) * n))
    core = idx[trim: n - trim] if n - 2 * trim > 0 else idx[:0]
    return core


def hold_indices(arrays: Mapping[str, np.ndarray], task: str, policy: Mapping[str, Any]) -> np.ndarray:
    st = arrays["stage"]
    out: list[int] = []
    for s in HOLD_STAGES[task]:
        idx = np.flatnonzero(st == s)
        out.extend(idx[int(policy["hold_trim_head_steps"]):].tolist())
    return np.asarray(sorted(out), dtype=int)


def hold_force_total(arrays: Mapping[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    return (np.asarray(arrays["left_mean_n"])[idx] + np.asarray(arrays["right_mean_n"])[idx])


def episode_stats(arrays: Mapping[str, np.ndarray], task: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one raw episode to the numbers calibration needs."""
    st = arrays["stage"]
    speed: dict[str, list[float]] = {}
    spd = eef_speed_mps(arrays)
    for stage in sorted(set(st.tolist())):
        idx = core_indices(st, stage, policy)
        v = spd[idx]
        speed[stage] = [float(x) for x in v[np.isfinite(v)]]
    h = hold_indices(arrays, task, policy)
    hold = hold_force_total(arrays, h) if len(h) else np.zeros(0)
    peak = np.asarray(arrays["left_max_n"]) + np.asarray(arrays["right_max_n"])
    finite_peak = peak[np.isfinite(peak)]
    nc = np.isin(st, ["approach", "descend"])
    noise = np.asarray(arrays["left_max_n"])[nc]
    return {
        "speed": speed,
        "hold": [float(x) for x in hold[np.isfinite(hold)]],
        "peak_total_n": float(finite_peak.max()) if len(finite_peak) else None,
        "idle_force_max_n": float(np.nanmax(noise)) if noise.size and np.isfinite(noise).any() else None,
        "task_ok": bool(np.asarray(arrays["success"])[-1]) if len(arrays["success"]) else False,
    }


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if len(values) else None


def derive_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    dev_seeds: Sequence[int],
    meta: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """``records``: dicts with task, speed_level, grip_level, stats (episode_stats)."""
    pol = dict(DEFAULT_POLICY)
    pol.update(policy or {})
    lo_q, hi_q = pol["percentile_lo"], pol["percentile_hi"]
    out: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "status": "calibrated_dev",
        "dev_seeds": sorted(int(s) for s in dev_seeds),
        "policy": pol,
        "meta": dict(meta),
        "speed": {},
        "grip": {},
        "safety": {},
    }
    for task in sorted({r["task"] for r in records}):
        recs = [r for r in records if r["task"] == task and r["stats"]["task_ok"]]
        n_all = len([r for r in records if r["task"] == task])
        out["speed"][task] = {}
        stages = sorted({s for r in recs for s in r["stats"]["speed"]})
        for stage in stages:
            pooled = {lvl: [v for r in recs if r["speed_level"] == lvl and r["grip_level"] == "normal"
                            for v in r["stats"]["speed"].get(stage, [])] for lvl in ("slow", "normal", "fast")}
            entry: dict[str, Any] = {
                "n_samples": {k: len(v) for k, v in pooled.items()},
                "slow_p95": _pct(pooled["slow"], hi_q), "normal_p5": _pct(pooled["normal"], lo_q),
                "normal_p95": _pct(pooled["normal"], hi_q), "fast_p5": _pct(pooled["fast"], lo_q),
                "slow_upper": None, "fast_lower": None, "status": "ok", "reason": None,
                "fast_episode_median_min": None,
            }
            fast_meds = [float(np.median(r["stats"]["speed"][stage])) for r in recs
                         if r["speed_level"] == "fast" and r["grip_level"] == "normal"
                         and len(r["stats"]["speed"].get(stage, [])) >= pol["min_core_samples"]]
            # A "fast" episode whose realised median never reaches the normal band is a
            # stalled (infeasible) episode, not a sample of the fast band: count it apart.
            if fast_meds and entry["normal_p95"] is not None:
                realised = [m for m in fast_meds if m > entry["normal_p95"]]
                entry["fast_unrealised_episodes"] = len(fast_meds) - len(realised)
                fast_meds = realised
            if fast_meds:
                entry["fast_episode_median_min"] = min(fast_meds)
            if min(len(v) for v in pooled.values()) < pol["min_core_samples"]:
                entry["status"], entry["reason"] = "insufficient_samples", "core window too short in dev"
            else:
                if entry["slow_p95"] < entry["normal_p5"]:
                    entry["slow_upper"] = 0.5 * (entry["slow_p95"] + entry["normal_p5"])
                    # slow: core MEDIAN must stay below slow_upper; no core sample may reach
                    # the typical normal speed (normal p5), so a transient cannot hide there.
                    entry["slow_transient_cap"] = entry["normal_p5"]
                # fast is judged on the core-window MEDIAN (a fast stage still decelerates
                # at its end); its threshold must clear every normal sample and every dev
                # fast episode's median.
                fmin = entry["fast_episode_median_min"]
                if fmin is not None and entry["normal_p95"] < fmin:
                    entry["fast_lower"] = 0.5 * (entry["normal_p95"] + fmin)
                if entry["slow_upper"] is None or entry["fast_lower"] is None:
                    entry["status"] = "partially_inseparable"
                    entry["reason"] = "adjacent speed levels overlap at p95/p5; unseparated side has null threshold"
            out["speed"][task][stage] = entry
        hold = {lvl: [v for r in recs if r["grip_level"] == lvl and r["speed_level"] == "normal"
                      for v in r["stats"]["hold"]] for lvl in ("gentle", "normal", "firm")}
        g: dict[str, Any] = {
            "n_samples": {k: len(v) for k, v in hold.items()},
            "gentle_p5": _pct(hold["gentle"], lo_q), "gentle_p95": _pct(hold["gentle"], hi_q),
            "normal_p5": _pct(hold["normal"], lo_q), "normal_p95": _pct(hold["normal"], hi_q),
            "firm_p5": _pct(hold["firm"], lo_q), "firm_p95": _pct(hold["firm"], hi_q),
            "firm_p99": _pct(hold["firm"], 99.0),
            "gentle_lower": None, "gentle_upper": None, "firm_lower": None, "firm_upper": None,
            "status": "ok", "reason": None, "success_rate_dev": len(recs) / n_all if n_all else 0.0,
        }
        if min(len(v) for v in hold.values()) < pol["min_core_samples"]:
            g["status"], g["reason"] = "insufficient_samples", "hold window too short in dev"
        else:
            g["gentle_lower"] = pol["gentle_floor_fraction"] * g["gentle_p5"]
            if g["gentle_p95"] < g["normal_p5"]:
                g["gentle_upper"] = 0.5 * (g["gentle_p95"] + g["normal_p5"])
            if g["normal_p95"] < g["firm_p5"]:
                g["firm_lower"] = 0.5 * (g["normal_p95"] + g["firm_p5"])
                g["firm_upper"] = pol["firm_cap_factor"] * g["firm_p99"]
            if g["gentle_upper"] is None or g["firm_lower"] is None:
                g["status"], g["reason"] = "partially_inseparable", "grip levels overlap at p95/p5"
        out["grip"][task] = g
        peaks = [r["stats"]["peak_total_n"] for r in recs if r["stats"]["peak_total_n"] is not None]
        idle = [r["stats"]["idle_force_max_n"] for r in recs if r["stats"]["idle_force_max_n"] is not None]
        out["safety"][task] = {
            "peak_dev_max_n": max(peaks) if peaks else None,
            "peak_cap_n": pol["safety_margin_factor"] * max(peaks) if peaks else None,
            "idle_force_max_n": max(idle) if idle else None,
            "contact_threshold_n": max(pol["min_contact_threshold_n"], 3.0 * (max(idle) if idle else 0.0)),
        }
    out["content_sha256"] = content_hash(out)
    return out


def save_calibration(cal: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, Any]:
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    if cal.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("calibration schema mismatch")
    if cal.get("content_sha256") != content_hash(cal):
        raise ValueError("calibration content hash mismatch (file modified after calibration)")
    return cal


def assert_seed_isolated(cal: Mapping[str, Any], seed: int) -> None:
    if int(seed) in set(cal.get("dev_seeds", [])):
        raise ValueError(f"seed {seed} was used for calibration and cannot be a production/test seed")


def speed_bound(cal: Mapping[str, Any], task: str, stage: str, level: str) -> tuple[Optional[float], Optional[float], str]:
    """Return (lower, upper, reason).  ``None`` bound with a reason => unknown."""
    e = cal.get("speed", {}).get(task, {}).get(stage)
    if e is None:
        return None, None, "no_speed_calibration_for_stage"
    if level == "slow":
        return (None, e["slow_upper"], "") if e.get("slow_upper") is not None else (None, None, f"speed_band_uncalibrated:{e.get('status')}")
    if level == "fast":
        return (e["fast_lower"], None, "") if e.get("fast_lower") is not None else (None, None, f"speed_band_uncalibrated:{e.get('status')}")
    return None, None, "no_band_for_level"


def slow_transient_cap(cal: Mapping[str, Any], task: str, stage: str) -> Optional[float]:
    """Per-sample ceiling for a slow stage.  Calibrations without the field keep the old
    rule (every sample below ``slow_upper``)."""
    e = cal.get("speed", {}).get(task, {}).get(stage) or {}
    cap = e.get("slow_transient_cap")
    return float(cap) if cap is not None else e.get("slow_upper")


def grip_bound(cal: Mapping[str, Any], task: str, level: str,
               scale: float = 1.0) -> tuple[Optional[float], Optional[float], str]:
    """Calibrated hold-force band of a level; ``scale`` = the scene's required hold force relative to nominal."""
    lo, hi, why = _grip_bound_nominal(cal, task, level)
    if why or scale == 1.0:
        return lo, hi, why
    return lo * scale, hi * scale, why


def _grip_bound_nominal(cal: Mapping[str, Any], task: str, level: str) -> tuple[Optional[float], Optional[float], str]:
    g = cal.get("grip", {}).get(task)
    if g is None:
        return None, None, "no_grip_calibration_for_task"
    if level == "gentle":
        if g.get("gentle_lower") is None or g.get("gentle_upper") is None:
            return None, None, f"grip_band_uncalibrated:{g.get('status')}"
        return g["gentle_lower"], g["gentle_upper"], ""
    if level == "firm":
        if g.get("firm_lower") is None or g.get("firm_upper") is None:
            return None, None, f"grip_band_uncalibrated:{g.get('status')}"
        return g["firm_lower"], g["firm_upper"], ""
    return None, None, "no_band_for_level"


# ------------------------------- wipe ---------------------------------------
def _finite0(x: Any) -> np.ndarray:
    """Tool-vs-table force series; NaN means the telemetry saw no tool contact in that step (= 0 N)."""
    a = np.asarray(x, dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def wipe_episode_stats(arrays: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    st = arrays["stage"]
    spd = eef_speed_mps(arrays)
    idx = core_indices(st, "sweep", policy)
    v = spd[idx]
    f = _finite0(arrays["tool_mean_n"])[idx]
    peak = _finite0(arrays["tool_max_n"])
    appr = peak[np.isin(st, ["approach"])]
    return {
        "speed": {"sweep": [float(x) for x in v[np.isfinite(v)]]},
        "press_median": float(np.median(f)) if len(f) else None,
        "peak_total_n": float(peak.max()) if len(peak) else None,
        "idle_force_max_n": float(appr.max()) if appr.size else None,
        "task_ok": bool(np.asarray(arrays["success"])[-1]) if len(arrays["success"]) else False,
    }


def derive_wipe_calibration(records: Sequence[Mapping[str, Any]], *, dev_seeds: Sequence[int],
                            meta: Mapping[str, Any], policy: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """``records``: dicts with speed_level, press_level, stats (``wipe_episode_stats``).

    Both manner dimensions are judged on the per-episode MEDIAN over the sweep core window (tool force and
    speed fluctuate strongly inside a stroke).  A threshold is the midpoint between the extreme episode medians of
    two adjacent levels, pooled over the other dimension's levels; it is left null if the levels overlap.
    """
    pol = dict(DEFAULT_POLICY)
    pol.update(policy or {})
    ok = [r for r in records if r["stats"]["task_ok"]]
    med = lambda r: float(np.median(r["stats"]["speed"]["sweep"])) if len(r["stats"]["speed"].get("sweep", [])) >= pol["min_core_samples"] else None  # noqa: E731
    by = {lvl: [m for r in ok if r["speed_level"] == lvl and (m := med(r)) is not None] for lvl in ("slow", "normal", "fast")}
    entry: dict[str, Any] = {"n_episodes": {k: len(v) for k, v in by.items()}, "median_only": True,
                             "slow_upper": None, "fast_lower": None, "status": "ok", "reason": None,
                             "episode_median_range": {k: [min(v), max(v)] if v else None for k, v in by.items()}}
    if min(len(v) for v in by.values()) < pol["min_core_samples"]:
        entry["status"], entry["reason"] = "insufficient_samples", "too few dev episodes"
    else:
        if max(by["slow"]) < min(by["normal"]):
            entry["slow_upper"] = 0.5 * (max(by["slow"]) + min(by["normal"]))
        if max(by["normal"]) < min(by["fast"]):
            entry["fast_lower"] = 0.5 * (max(by["normal"]) + min(by["fast"]))
        if entry["slow_upper"] is None or entry["fast_lower"] is None:
            entry["status"], entry["reason"] = "partially_inseparable", "adjacent speed levels overlap in episode medians"
    pr = {lvl: [float(r["stats"]["press_median"]) for r in ok if r["press_level"] == lvl
                and r["stats"]["press_median"] is not None] for lvl in ("normal", "firm")}
    g: dict[str, Any] = {"n_episodes": {k: len(v) for k, v in pr.items()}, "firm_lower": None, "firm_upper": None,
                         "status": "ok", "reason": None, "episode_median_range": {k: [min(v), max(v)] if v else None for k, v in pr.items()}}
    if min(len(v) for v in pr.values()) < pol["min_core_samples"]:
        g["status"], g["reason"] = "insufficient_samples", "too few dev episodes"
    elif max(pr["normal"]) < min(pr["firm"]):
        g["firm_lower"] = 0.5 * (max(pr["normal"]) + min(pr["firm"]))
        g["firm_upper"] = pol["firm_cap_factor"] * max(pr["firm"])
    else:
        g["status"], g["reason"] = "partially_inseparable", "press levels overlap in episode medians"
    peaks = [r["stats"]["peak_total_n"] for r in ok if r["stats"]["peak_total_n"] is not None]
    return {"speed": {"wipe": {"sweep": entry}}, "press": {"wipe": g},
            "safety": {"wipe": {"peak_dev_max_n": max(peaks) if peaks else None,
                                "peak_cap_n": pol["safety_margin_factor"] * max(peaks) if peaks else None}},
            "dev_seeds": sorted(int(x) for x in dev_seeds), "meta": dict(meta),
            "n_dev_episodes": len(records), "n_dev_failed": len(records) - len(ok)}


def merge_wipe_calibration(base: Mapping[str, Any], wipe: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(dict(base)))
    out["speed"]["wipe"] = wipe["speed"]["wipe"]
    out.setdefault("press", {})["wipe"] = wipe["press"]["wipe"]
    out["safety"]["wipe"] = wipe["safety"]["wipe"]
    out["dev_seeds"] = sorted(set(out["dev_seeds"]) | set(wipe["dev_seeds"]))
    out["meta"] = {**out["meta"], **dict(meta), "wipe_dev_seeds": wipe["dev_seeds"],
                   "wipe_n_dev_episodes": wipe["n_dev_episodes"], "wipe_n_dev_failed": wipe["n_dev_failed"]}
    out.pop("content_sha256", None)
    out["content_sha256"] = content_hash(out)
    return out


def press_bound(cal: Mapping[str, Any], task: str, level: str) -> tuple[Optional[float], Optional[float], str]:
    g = cal.get("press", {}).get(task)
    if g is None:
        return None, None, "no_press_calibration_for_task"
    if level == "firm":
        if g.get("firm_lower") is None:
            return None, None, f"press_band_uncalibrated:{g.get('status')}"
        return g["firm_lower"], g.get("firm_upper"), ""
    return None, None, "no_band_for_level"
