"""Step 25: selection + repair policies replayed on the deterministic response grid (s25_grid.py).

A REQUEST = (task, init, force band, speed band).  A POLICY proposes a (force candidate, speed command); the grid returns
the episode that the shared executor produced for exactly that plan.  Policies differ only in how they choose:

  fixed        : force candidates in list order, nominal speed command (no memory)            -- naive baseline
  rule         : physics prior only (closure stop lo = band lower edge + margin), no memory   -- hand-designed prior
  library      : online experience library; every executed episode is written back and used by the next request
  library_frozen: same model, but write-back stops after `freeze_after` episodes              -- ablation of write-back
  oracle       : first passing config if any (upper bound)

Repair (after a failed attempt), selected by `repair`:
  next     : next candidate of the policy's own ranking (blind)
  cause    : failure-cause-aware, manner-band-bounded value adjustment (s25 stage repair):
               slip / drop / not lifted   -> raise force inside the band; if band top reached -> lower speed inside its band
               hold above band            -> lower force
               hold below band (task ok)  -> raise force
               speed outside band         -> move speed command toward the band
             a value outside the requested manner is NEVER proposed; if none remains -> manner-infeasible (stop, honest)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "reports" / "step25_response_grid"
FORCES = [(1.0, 22.0), (1.0, 100.0), (5.0, 100.0), (10.0, 22.0), (10.0, 100.0), (15.0, 100.0), (20.0, 100.0),
          (25.0, 100.0), (30.0, 100.0), (40.0, 100.0)]
SPEEDS = [0.06, 0.12, 0.24]
REQUESTED = {0: "tomato_sauce", 7: "cream_cheese", 1: "cream_cheese"}
OBJECT = {0: "tomato_sauce_1", 7: "cream_cheese_1", 1: "cream_cheese_1"}
FORCE_BANDS = {"R1": (8.0, 16.0), "R2": (16.0, 24.0), "R3": (24.0, 34.0), "R4": (34.0, 46.0), "any": (0.0, 1e9)}
SPEED_BANDS = {"slow": (0.0, 0.09), "normal": (0.09, 0.17), "fast": (0.17, 1.0), "any": (0.0, 1e9)}
NOMINAL_SPEED = {"slow": 0.06, "normal": 0.12, "fast": 0.24, "any": 0.12}


def key(force, speed):
    return f"{force[0]:g}_{force[1]:g}@{speed:g}"


def load_grid(tasks=(0, 7, 1)):
    g = {}
    for t in tasks:
        for f in sorted(GRID.glob(f"t{t}_init*.json")):
            d = json.loads(f.read_text())
            for k, e in d["episodes"].items():
                g[(t, d["init"], k)] = e
    return g


def outcome(e, fband, sband):
    """Per-attempt judgement; returns (passed, cause, facts)."""
    if e is None or "error" in e:
        return False, "error", {}
    o = e["objects"][e["requested_role"]]
    hold = o.get("hold_mean")
    facts = {"task_ok": e["task_ok"], "hold_min": o.get("hold_min"), "hold_max": o.get("hold_max"), "hold": hold,
             "stable": o.get("grasp_stable"), "valid": o.get("measurement_valid"), "v": e.get("carry_speed_med_mps"),
             "contact_v1": e.get("contact_v1_clean"), "status": e.get("status")}
    if not o.get("measurement_valid") or hold is None or not o.get("grasp_stable"):
        cause = "slip_or_drop"
    elif o["hold_max"] > fband[1]:
        cause = "force_above_band"
    elif o["hold_min"] < fband[0]:
        cause = "force_below_band"
    elif e.get("carry_speed_med_mps") is None or not (sband[0] <= e["carry_speed_med_mps"] <= sband[1]):
        cause = "speed_outside_band"
    elif not e["task_ok"]:
        cause = "task_failed_other"
    else:
        return True, None, facts
    if cause in ("force_above_band", "force_below_band", "speed_outside_band") and not e["task_ok"]:
        cause = "task_failed_other" if o.get("grasp_stable") else "slip_or_drop"
    return False, cause, facts


# ------------------------------------------------------------------ experience library (online)
class Library:
    """Measured-response memory.  Record = (task, object, force cand, speed cmd) -> measured hold interval, stable?,
    task_ok, realised speed.  Prediction for a query backs off: same object & config -> same object & force -> same task
    -> all.  Nothing about the init is used (inits are what varies between requests)."""

    def __init__(self):
        self.rows = []

    def add(self, task, force, speed, e):
        if e is None or "error" in e:
            return
        o = e["objects"][e["requested_role"]]
        self.rows.append({"task": task, "object": o.get("target"), "force": tuple(force), "speed": speed,
                          "hold_min": o.get("hold_min"), "hold_max": o.get("hold_max"), "hold": o.get("hold_mean"),
                          "stable": bool(o.get("grasp_stable")) and o.get("hold_mean") is not None,
                          "task_ok": bool(e["task_ok"]), "v": e.get("carry_speed_med_mps")})

    def _sel(self, task, obj, force, speed):
        for f, basis in ((lambda r: r["object"] == obj and r["force"] == force and r["speed"] == speed, "obj+cfg"),
                         (lambda r: r["object"] == obj and r["force"] == force, "obj+force"),
                         (lambda r: r["force"] == force and r["speed"] == speed, "any_obj+cfg")):
            s = [r for r in self.rows if f(r)]
            if s:
                return s, basis
        return [], "none"

    def p_pass(self, task, obj, force, speed, fband, sband, prior):
        """Posterior mean P(pass) with a Beta prior centred on the rule prior (strength 2)."""
        s, basis = self._sel(task, obj, force, speed)
        a, b = 2 * prior, 2 * (1 - prior)
        for r in s:
            ok = r["task_ok"] and r["stable"] and r["hold_min"] is not None and fband[0] <= r["hold_min"] \
                and r["hold_max"] <= fband[1] and r["v"] is not None and sband[0] <= r["v"] <= sband[1]
            a, b = a + ok, b + (not ok)
        return a / (a + b), len(s), basis


def rule_prior(force, speed, fband, sband):
    """Hand-designed physics prior (no data): closure stops at lo -> expect hold ~ lo + 8 N (closing overshoot) with a
    hi cap; speed command realised ~1:1.  Returns a pseudo-probability in [0.05, 0.95]."""
    lo, hi = force
    pred = min(lo + 8.0, hi)
    c, w = 0.5 * (fband[0] + fband[1]), max(0.5 * (fband[1] - fband[0]), 1.0)
    pf = math.exp(-((pred - c) / w) ** 2) if fband[1] < 1e8 else 0.8
    ps = 0.9 if sband[0] <= speed <= sband[1] else 0.05
    return max(0.05, min(0.95, pf * ps))


def ranking(policy, lib, task, req, freeze=False):
    fband, sband = FORCE_BANDS[req["force"]], SPEED_BANDS[req["speed"]]
    obj = OBJECT[task]
    cfgs = [(f, s) for f in FORCES for s in SPEEDS]
    if policy == "fixed":
        sp = NOMINAL_SPEED[req["speed"]]
        return [(f, sp) for f in FORCES] + [(f, s) for f in FORCES for s in SPEEDS if s != sp]
    if policy == "rule":
        return sorted(cfgs, key=lambda c: (-rule_prior(c[0], c[1], fband, sband), FORCES.index(c[0]), SPEEDS.index(c[1])))
    if policy.startswith("library"):
        sc = []
        for c in cfgs:
            pr = rule_prior(c[0], c[1], fband, sband)
            p, n, _ = lib.p_pass(task, obj, c[0], c[1], fband, sband, pr)
            sc.append((-p, FORCES.index(c[0]), SPEEDS.index(c[1]), c))
        return [c for *_, c in sorted(sc)]
    raise ValueError(policy)


def cause_repair(prev, cause, req, tried, rank):
    """Manner-bounded, cause-aware next proposal.  Moves along the force axis (sorted by closure stop) or the speed
    axis, never outside the requested bands; ties broken by the policy's own ranking."""
    fband, sband = FORCE_BANDS[req["force"]], SPEED_BANDS[req["speed"]]
    f0, s0 = prev
    order = sorted(FORCES, key=lambda f: (f[0], f[1]))
    i = order.index(f0)
    cands = []
    if cause in ("slip_or_drop", "force_below_band"):
        cands = [(f, s0) for f in order[i + 1:]]
        if cause == "slip_or_drop":  # if stronger closure is not allowed by the band, slow down inside the speed band
            cands += [(f0, s) for s in sorted(SPEEDS) if s < s0 and sband[0] <= s <= sband[1]]
    elif cause == "force_above_band":
        cands = [(f, s0) for f in reversed(order[:i])]
    elif cause == "speed_outside_band":
        cands = [(f0, s) for s in SPEEDS if sband[0] <= s <= sband[1]]
    elif cause in ("task_failed_other", "error"):
        cands = [(f0, s) for s in sorted(SPEEDS) if s < s0 and sband[0] <= s <= sband[1]]
    # manner bound: the closure stop must not already be above the force band (would only push the hold higher)
    cands = [c for c in cands if c not in tried and c[0][0] <= fband[1]]
    if not cands:
        return None
    pos = {c: k for k, c in enumerate(rank)}
    # keep the cause direction (nearest step first), use the ranking only as a tie-break inside one step
    return cands[0] if cands[0] in pos else None
