"""Target-object, per-finger grasp force contract (P0-1).

Replaces the legacy single-scalar ``cfrc_norm`` / "sum of everything touching
the pads" grasp force with an auditable per-step record:

* left / right finger normal force on the *target object only* (N),
* number of active contacts and the target geom id that carries the force,
* force on non-target geoms is kept apart (``other_force_n``), never merged,
* a ``valid`` bit: a failed or ambiguous read yields NaN + ``valid=False`` and
  is never silently converted to 0 N.

Episode-level summaries are pure functions of the raw arrays, so any reported
value can be recomputed from stored telemetry.  No calibrated force threshold
is defined here: ``contact_threshold_n`` is an explicit argument (P0-2 owns
the calibrated values).  Pure numpy; MuJoCo is only touched through the
duck-typed ``read_grasp_contact_sample``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

SCHEMA_VERSION = "grasp_force_v1"

REASON_OK = "ok"
REASON_READ_ERROR = "contact_read_error"
REASON_NON_FINITE = "non_finite_force"
REASON_BINDING_EMPTY = "binding_empty"
REASON_MISSING = "telemetry_missing"


class GraspState(str, Enum):
    UNKNOWN = "unknown"          # measurement invalid / absent / incomplete
    NO_CONTACT = "no_contact"    # measurement valid, no finger touches target
    SINGLE_SIDE = "single_side"  # only one finger ever loaded the target
    BILATERAL = "bilateral"      # both fingers loaded the target together


@dataclass(frozen=True)
class GraspBinding:
    """Stable geom identities for one task episode."""

    left_finger_geom_ids: tuple[int, ...]
    right_finger_geom_ids: tuple[int, ...]
    target_geom_ids: tuple[int, ...]
    target_name: str = ""

    def is_complete(self) -> bool:
        return bool(
            self.left_finger_geom_ids
            and self.right_finger_geom_ids
            and self.target_geom_ids
        )


@dataclass(frozen=True)
class GraspContactSample:
    """One physics-step reading.  NaN forces <=> ``valid`` is False."""

    left_force_n: float
    right_force_n: float
    left_contacts: int = 0
    right_contacts: int = 0
    left_geom_id: int = -1      # target geom carrying the largest left force
    right_geom_id: int = -1
    other_force_n: float = 0.0  # finger force on non-target geoms (reported only)
    other_contacts: int = 0
    valid: bool = True
    reason: str = REASON_OK


def invalid_sample(reason: str) -> GraspContactSample:
    return GraspContactSample(
        left_force_n=math.nan, right_force_n=math.nan, valid=False, reason=reason
    )


def resolve_grasp_binding(
    model: Any,
    mj: Any,
    target_geom_ids: Iterable[int],
    target_name: str = "",
) -> GraspBinding:
    """Resolve left/right finger collision geoms from a MuJoCo model.

    Robosuite Panda: ``finger1`` lives on ``leftfinger``/``finger_joint1_tip``,
    ``finger2`` on ``rightfinger``/``finger_joint2_tip``.  Only ``collision``
    geoms participate (visual geoms carry no contacts).
    """
    left: list[int] = []
    right: list[int] = []
    for gi in range(int(model.ngeom)):
        gname = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gi) or "").lower()
        if "gripper0" not in gname or "collision" not in gname:
            continue
        bname = (
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gi])) or ""
        ).lower()
        if "finger1" in gname or "leftfinger" in bname or "finger_joint1" in bname:
            left.append(gi)
        elif "finger2" in gname or "rightfinger" in bname or "finger_joint2" in bname:
            right.append(gi)
    return GraspBinding(
        tuple(left), tuple(right), tuple(int(g) for g in target_geom_ids), target_name
    )


def read_grasp_contact_sample(
    model: Any,
    data: Any,
    binding: GraspBinding,
    *,
    contact_force_fn: Optional[Callable[[Any, Any, int, np.ndarray], Any]] = None,
) -> GraspContactSample:
    """Read per-finger target-object normal force from the current contacts.

    Any exception, empty binding or non-finite force returns an invalid sample
    (NaN force) rather than 0 N.
    """
    if not binding.is_complete():
        return invalid_sample(REASON_BINDING_EMPTY)
    try:
        if contact_force_fn is None:
            import mujoco  # local import keeps the module numpy-only

            contact_force_fn = mujoco.mj_contactForce
        left_ids = set(binding.left_finger_geom_ids)
        right_ids = set(binding.right_finger_geom_ids)
        target_ids = set(binding.target_geom_ids)
        side_force = {"L": 0.0, "R": 0.0}
        side_count = {"L": 0, "R": 0}
        side_best = {"L": (0.0, -1), "R": (0.0, -1)}
        other_force, other_count = 0.0, 0
        buf = np.zeros(6)
        for ci in range(int(data.ncon)):
            c = data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            side = None
            other = None
            for g, o in ((g1, g2), (g2, g1)):
                if g in left_ids:
                    side, other = "L", o
                elif g in right_ids:
                    side, other = "R", o
                if side:
                    break
            if side is None or other in left_ids or other in right_ids:
                continue  # not a finger contact, or finger-finger
            buf[:] = 0.0
            contact_force_fn(model, data, ci, buf)
            normal = abs(float(buf[0]))
            if not math.isfinite(normal):
                return invalid_sample(REASON_NON_FINITE)
            if normal <= 0.0:
                continue
            if other in target_ids:
                side_force[side] += normal
                side_count[side] += 1
                if normal > side_best[side][0]:
                    side_best[side] = (normal, int(other))
            else:
                other_force += normal
                other_count += 1
        return GraspContactSample(
            left_force_n=side_force["L"],
            right_force_n=side_force["R"],
            left_contacts=side_count["L"],
            right_contacts=side_count["R"],
            left_geom_id=side_best["L"][1],
            right_geom_id=side_best["R"][1],
            other_force_n=other_force,
            other_contacts=other_count,
        )
    except Exception:
        return invalid_sample(REASON_READ_ERROR)


# --------------------------------------------------------------------------
# Episode-level, recomputable summary
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GraspForceSummary:
    schema_version: str
    state: GraspState
    n_steps: int
    n_valid: int
    peak_left_n: Optional[float]
    peak_right_n: Optional[float]
    peak_total_n: Optional[float]       # max_t (left + right)
    peak_bilateral_n: Optional[float]   # max_t min(left, right)
    bilateral_steps: int
    bilateral_duration_s: Optional[float]
    other_peak_n: Optional[float]
    reason: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["state"] = self.state.value
        return d


def _unknown(n_steps: int, n_valid: int, reason: str) -> GraspForceSummary:
    return GraspForceSummary(
        SCHEMA_VERSION, GraspState.UNKNOWN, n_steps, n_valid,
        None, None, None, None, 0, None, None, reason,
    )


def summarize_grasp_force(
    left_force_n: Optional[Sequence[float]],
    right_force_n: Optional[Sequence[float]],
    valid: Optional[Sequence[bool]],
    *,
    other_force_n: Optional[Sequence[float]] = None,
    mask: Optional[Sequence[bool]] = None,
    dt_s: Optional[float] = None,
    contact_threshold_n: float = 0.0,
    max_invalid_fraction: float = 0.0,
    min_bilateral_steps: int = 1,
) -> GraspForceSummary:
    """Classify an episode (or a ``mask``-selected window) from raw arrays.

    ``unknown`` whenever the arrays are missing, lengths disagree, the window
    is empty, or the invalid fraction exceeds ``max_invalid_fraction``.
    Otherwise ``no_contact`` / ``single_side`` / ``bilateral`` by strict
    ``> contact_threshold_n`` per finger.
    """
    if left_force_n is None or right_force_n is None or valid is None:
        return _unknown(0, 0, REASON_MISSING)
    left = np.asarray(left_force_n, dtype=np.float64)
    right = np.asarray(right_force_n, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    if not (left.ndim == right.ndim == ok.ndim == 1 and len(left) == len(right) == len(ok)):
        return _unknown(len(left), 0, "length_mismatch")
    sel = np.ones(len(left), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(sel) != len(left):
        return _unknown(len(left), 0, "mask_length_mismatch")
    n_steps = int(sel.sum())
    if n_steps == 0:
        return _unknown(0, 0, "empty_window")
    sel_valid = sel & ok
    n_valid = int(sel_valid.sum())
    finite = np.isfinite(left) & np.isfinite(right)
    # a step flagged valid but carrying NaN is a contract violation => invalid
    sel_valid &= finite
    n_valid = int(sel_valid.sum())
    if n_valid == 0 or (n_steps - n_valid) / n_steps > max_invalid_fraction:
        return _unknown(n_steps, n_valid, "invalid_measurement")

    l, r = left[sel_valid], right[sel_valid]
    left_on = l > contact_threshold_n
    right_on = r > contact_threshold_n
    bilateral = left_on & right_on
    n_bi = int(bilateral.sum())
    if not (left_on.any() or right_on.any()):
        state = GraspState.NO_CONTACT
    elif n_bi >= max(1, min_bilateral_steps):
        state = GraspState.BILATERAL
    else:
        state = GraspState.SINGLE_SIDE
    other_peak = None
    if other_force_n is not None:
        o = np.asarray(other_force_n, dtype=np.float64)
        if len(o) == len(left):
            o = o[sel_valid]
            o = o[np.isfinite(o)]
            other_peak = float(o.max()) if o.size else None
    return GraspForceSummary(
        schema_version=SCHEMA_VERSION,
        state=state,
        n_steps=n_steps,
        n_valid=n_valid,
        peak_left_n=float(l.max()),
        peak_right_n=float(r.max()),
        peak_total_n=float((l + r).max()),
        peak_bilateral_n=float(np.minimum(l, r).max()),
        bilateral_steps=n_bi,
        bilateral_duration_s=None if dt_s is None else float(n_bi * dt_s),
        other_peak_n=other_peak,
        reason=REASON_OK,
    )


def telemetry_grasp_arrays(telemetry: Sequence[Any]) -> Optional[dict]:
    """Pull the raw grasp channel out of ``MannerStepSnapshot`` objects.

    Returns ``None`` when snapshots lack the channel (legacy telemetry) so the
    caller reports ``unknown`` instead of defaulting to 0 N.
    """
    if not telemetry or not all(hasattr(s, "grasp_force_valid") for s in telemetry):
        return None
    return {
        "left": np.array([s.grasp_left_force_n for s in telemetry], dtype=np.float64),
        "right": np.array([s.grasp_right_force_n for s in telemetry], dtype=np.float64),
        "valid": np.array([bool(s.grasp_force_valid) for s in telemetry]),
        "other": np.array([s.grasp_other_force_n for s in telemetry], dtype=np.float64),
    }


def summarize_telemetry_grasp(
    telemetry: Sequence[Any],
    *,
    phases: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> GraspForceSummary:
    arrays = telemetry_grasp_arrays(telemetry)
    if arrays is None:
        return _unknown(len(telemetry or ()), 0, REASON_MISSING)
    mask = None
    if phases is not None:
        wanted = set(phases)
        mask = [s.phase in wanted for s in telemetry]
    return summarize_grasp_force(
        arrays["left"], arrays["right"], arrays["valid"],
        other_force_n=arrays["other"], mask=mask, **kwargs,
    )


__all__ = [
    "GraspBinding", "GraspContactSample", "GraspForceSummary", "GraspState",
    "SCHEMA_VERSION", "invalid_sample", "read_grasp_contact_sample",
    "resolve_grasp_binding", "summarize_grasp_force",
    "summarize_telemetry_grasp", "telemetry_grasp_arrays",
]
