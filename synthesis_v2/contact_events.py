"""Backend-agnostic grasp event interface (design 8.4 step 3).

Events and phases are derived ONLY from physical records (per-object finger contact force, object
position), never from generator stage labels.  The same causal ``ContactEventTracker`` runs online
(while a policy rolls out) and offline (over a recorded episode), so both give identical output.

Per-object events (grasp = both fingers load the object):
  contact_established  both fingers > thr for ``debounce`` consecutive steps (``onset`` = first step)
  lifted               object rose >= lift_dz above its height at contact onset while still grasped
  released             grasp lost (``debounce`` consecutive steps) after the object was lifted
  contact_lost         grasp lost before the object was lifted (failed / aborted grasp)
Phases (causal): free (no object grasped: approach / descend / retreat) -> grasp -> carry (grasped and lifted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

CONTACT_THR_N = 0.1  # same contact threshold as the safety calibration (configs/manner_calibration.v3.json)
PHASES = ("free", "grasp", "carry")


@dataclass
class _Obj:
    run_contact: int = 0
    run_lost: int = 0
    grasped: bool = False
    lifted: bool = False
    onset: int = -1
    z_onset: float = math.nan
    prev_z: float = math.nan
    done: bool = False  # released once -> this object is finished for phase purposes
    z_hist: list = field(default_factory=list)


class ContactEventTracker:
    def __init__(self, names: Sequence[str], *, thr: float = CONTACT_THR_N, debounce: int = 2, lift_dz: float = 0.02):
        self.names, self.thr, self.debounce, self.lift_dz = tuple(names), float(thr), int(debounce), float(lift_dz)
        self.o = {n: _Obj() for n in self.names}
        self.events: list[dict[str, Any]] = []
        self.t = -1
        self.active: Optional[str] = None  # object currently grasped / most recently released

    def update(self, forces: Mapping[str, tuple], pos: Mapping[str, np.ndarray]) -> tuple[str, str, list[dict]]:
        """One control step.  ``forces[name] = (left_n, right_n)`` (NaN = unmeasured), ``pos[name]`` = xyz.
        Returns (phase, active_object, new_events)."""
        self.t += 1
        t, new = self.t, []
        for n in self.names:
            st = self.o[n]
            l, r = forces[n]
            z = float(np.asarray(pos[n], dtype=np.float64)[2])
            both = bool(l > self.thr and r > self.thr)  # NaN compares False
            if not st.grasped:
                st.run_contact = st.run_contact + 1 if both else 0
                if st.run_contact == 1:
                    st.onset, st.z_onset = t, (st.prev_z if not math.isnan(st.prev_z) else z)
                if st.run_contact >= self.debounce:
                    st.grasped, st.lifted, st.run_lost, st.done = True, False, 0, False
                    new.append(dict(step=t, onset=st.onset, object=n, event="contact_established"))
                    self.active = n
            else:
                if not st.lifted and z - st.z_onset >= self.lift_dz:
                    st.lifted = True
                    new.append(dict(step=t, onset=t, object=n, event="lifted"))
                st.run_lost = st.run_lost + 1 if not both else 0
                if st.run_lost >= self.debounce:
                    first = t - self.debounce + 1
                    new.append(dict(step=t, onset=first, object=n, event="released" if st.lifted else "contact_lost"))
                    st.grasped, st.done = False, st.lifted
                    st.run_contact = 0
            st.prev_z = z
        self.events.extend(new)
        return self.phase(), self.active or "", new

    def phase(self) -> str:
        a = self.active
        if a is not None and self.o[a].grasped:
            return "carry" if self.o[a].lifted else "grasp"
        return "free"


def run_tracker(arrays: Mapping[str, np.ndarray], names: Sequence[str], **kw) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Offline pass over a recorded episode.  Row t uses the post-step contact ``cf_L/R_<name>[t]`` and the
    post-step object position ``pa_<name>[t]``."""
    n = len(arrays["actions"])
    tr = ContactEventTracker(names, **kw)
    ph, ob = [], []
    for t in range(n):
        f = {nm: (float(arrays[f"cf_L_{nm}"][t]), float(arrays[f"cf_R_{nm}"][t])) for nm in names}
        p = {nm: arrays[f"pa_{nm}"][t] for nm in names}
        phase, active, _ = tr.update(f, p)
        ph.append(phase)
        ob.append(active)
    return np.asarray(ph, dtype="<U12"), np.asarray(ob, dtype="<U32"), tr.events
