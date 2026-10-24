"""Post-hoc grasp audit on a finished episode (same rule for any policy version).

Force basis (stated once): per-finger contact normal force on the target object, mean over the physics substeps of one
control step; the policy's "total" is left+right.  ``target_lo_n/target_hi_n`` are *control thresholds*, not a calibrated
manner band; task success and manner acceptance are separate questions.

Flags per grasp instance
  ever_reached_band   total >= target_lo_n at some control step (history; NOT "currently held")
  contact_at_end      both fingers >= CONTACT_MIN_N on the target at the last control step of the grasp stage
  prelift_stable      contact_at_end and both fingers >= CONTACT_MIN_N on every one of the last STABLE_WINDOW readings
  contact_in_lift     (added after the first T0 comparison showed the last grasp-stage reading is transient: right-finger force
                      recovers during the lift) both fingers >= CONTACT_MIN_N on every control step of the second half of the lift stage
  lift_supported      target rose >= LIFT_SUPPORT_FRAC * LIFT_DZ during the lift stage
"""
from __future__ import annotations

from typing import Any

import numpy as np

CONTACT_MIN_N = 0.5
STABLE_WINDOW = 3
LIFT_DZ = 0.15
LIFT_SUPPORT_FRAC = 0.5


def _both(r: dict[str, Any]) -> bool:
    return bool(r["left_n"] >= CONTACT_MIN_N and r["right_n"] >= CONTACT_MIN_N)


def grasp_flags(info: dict[str, Any]) -> dict[str, Any]:
    trace, fin = info.get("trace") or [], info.get("final")
    readings = [{"left_n": r["left_n"], "right_n": r["right_n"]} for r in trace]
    if fin:
        readings.append(fin)
    contact_end = bool(fin) and _both(fin)
    window = readings[-STABLE_WINDOW:]
    return {"ever_reached_band": bool(info.get("reached_band")), "contact_at_end": contact_end,
            "prelift_stable": bool(contact_end and window and all(_both(r) for r in window)),
            "reverse_pulses": int(info.get("reverse_pulses", 0)), "reclose_pulses": int(info.get("reclose_pulses", 0)),
            "final_total_n": (fin or {}).get("total_n")}


def audit_episode(ep: Any, program_objects: dict[str, str], instance_ids: list[str] | None = None) -> dict[str, dict]:
    """``program_objects``: role -> backend object name.  Returns per grasp instance flags incl. lift_supported."""
    a = ep.arrays
    st = np.array([str(x) for x in a["stage"]])
    out: dict[str, dict] = {}
    for inst, info in ep.meta.get("grasp_policy_by_instance", {}).items():
        if instance_ids is not None and inst not in instance_ids:
            continue
        role = inst.split(".")[0]
        key = "pos_" + program_objects[role]
        lift = np.where(st == inst.rsplit(".", 1)[0] + ".lift")[0]
        rise = float(a[key][lift[-1], 2] - a[key][lift[0], 2]) if len(lift) else float("nan")
        f = grasp_flags(info)
        f["lift_rise_m"] = rise
        f["lift_supported"] = bool(rise == rise and rise >= LIFT_SUPPORT_FRAC * LIFT_DZ)
        half = lift[len(lift) // 2:]
        okcf = lambda s_: np.asarray(a["cf_" + s_ + "_" + program_objects[role]])[half]  # noqa: E731
        f["contact_in_lift"] = bool(len(half) and np.all(okcf("L") >= CONTACT_MIN_N) and np.all(okcf("R") >= CONTACT_MIN_N))
        f["lost_after_reach"] = bool(f["ever_reached_band"] and not f["contact_at_end"])
        out[inst] = f
    return out
