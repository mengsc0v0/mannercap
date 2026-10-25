"""Post-execution contact audit: which robot bodies touched what, classified against a registered policy.

A passing hold-force check says nothing about the rest of the hand/arm touching other objects, so every episode can be
audited separately.  Policy (registered, deliberately small):
  allowed      robot fingertip pads <-> the object of the current grasp (not recorded; that is what the force check measures)
  forbidden    any other robot body <-> any movable object (the grasp target's own non-tip bodies included), above CONTACT_THR_N
  observed     robot <-> support surface (table / floor): reported separately, not counted as a violation here
The audit records; it does not decide task success and is not a safety score.  ``clean`` means "no forbidden-class contact";
table/floor contact is observation-only, so a hard press on the table can still be ``clean``.  Contacts inside a
save_state/restore_state branch (settled look-ahead, recovery branch) are not part of the trajectory and are reported apart.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

CONTACT_THR_N = 0.1  # same contact threshold as the force calibration
SUPPORT_PREFIXES = ("living_room_table", "floor")
FINGER_LINK_ON_TARGET = "[finger_link_on_target]"


def _is_robot(body: str) -> bool:
    return body.startswith(("robot0", "gripper0"))


class ContactRecorder:
    """Wraps ``session.env.sim.step`` (every physics substep), ``session.step`` (control-step index) and
    ``session.save_state/restore_state``.

    Rollback isolation: whatever happens between a ``save_state`` and its ``restore_state`` (the planner's settled
    look-ahead, a recovery branch) is not part of the final trajectory.  On restore the events are reset to their value
    at save time and the discarded branch's contacts are kept apart in ``rolled_back`` (reported, not counted)."""

    def __init__(self, session) -> None:
        self.s = session
        self.m, self.d = session._mj()
        self.events: dict[tuple[str, str], list] = {}   # (robot body, other body) -> [max N, {control steps}]
        self.rolled_back: dict[tuple[str, str], float] = {}
        self.k = -1
        self._roles: Optional[dict] = None
        self._saved: dict[int, tuple[dict, dict]] = {}  # id(state) -> (events copy, state): holding state keeps the id valid

    def _scan(self):
        """Yield (robot body, other body, normal force N) for every robot<->non-robot contact of the current physics state."""
        import mujoco
        buf = np.zeros(6)
        nm = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[g])) or "?"  # noqa: E731
        for ci in range(int(self.d.ncon)):
            c = self.d.contact[ci]
            b1, b2 = nm(int(c.geom1)), nm(int(c.geom2))
            if _is_robot(b1) == _is_robot(b2):
                continue
            rb, ob = (b1, b2) if _is_robot(b1) else (b2, b1)
            mujoco.mj_contactForce(self.m, self.d, ci, buf)
            rg = int(c.geom1) if _is_robot(b1) else int(c.geom2)
            yield rb, ob, abs(float(buf[0])), rg

    def _on_target(self, ob: str) -> bool:
        return ob == f"{self.s.force_target_name}_main".replace("_1_1", "_1")

    def _role(self, geom_id) -> Optional[tuple[str, str]]:
        """(side, role) of a robot geom from the SAME identity rule the force measurement uses (grasp_force_contract)."""
        if geom_id is None:
            return None
        if self._roles is None:
            import mujoco
            from synthesis_v2.grasp_force_contract import finger_geom_roles
            self._roles = finger_geom_roles(self.m, mujoco)
        return self._roles.get(int(geom_id))

    def _observe(self) -> None:
        for item in self._scan():
            rb, ob, f, *rest = item
            if f <= 0:
                continue
            role = self._role(rest[0]) if rest else None
            if role is not None and role[1] == "pad" and self._on_target(ob):
                continue                       # allowed: finger pad <-> current grasp target (measured by the force check)
            label = rb.replace("gripper0_", "")
            if role is not None and role[1] == "finger_link" and self._on_target(ob):
                label += FINGER_LINK_ON_TARGET  # forbidden under the frozen policy, but identified: grasp finger link on the grasp target
            v = self.events.setdefault((label, ob), [0.0, set()])
            v[0] = max(v[0], f); v[1].add(self.k)

    def install(self) -> "ContactRecorder":
        s = self.s
        orig_sub, orig_step, orig_save, orig_restore = s.env.sim.step, s.step, s.save_state, s.restore_state

        def sub(*a, **kw):
            out = orig_sub(*a, **kw)
            self._observe()
            return out

        def step(*a, **kw):
            self.k = s.control_step_index
            return orig_step(*a, **kw)

        def save(*a, **kw):
            st = orig_save(*a, **kw)
            self._saved[id(st)] = ({k: [v[0], set(v[1])] for k, v in self.events.items()}, st)
            return st

        def restore(st, *a, **kw):
            out = orig_restore(st, *a, **kw)
            snap = self._saved.get(id(st))
            if snap is not None:
                for k, (f, _) in self.events.items():
                    if k not in snap[0] or f > snap[0][k][0]:
                        self.rolled_back[k] = max(self.rolled_back.get(k, 0.0), f)
                self.events = {k: [v[0], set(v[1])] for k, v in snap[0].items()}
            return out

        s.env.sim.step, s.step, s.save_state, s.restore_state = sub, step, save, restore
        return self

    def events_at(self, state) -> dict:
        """Events as they were when ``state`` was saved (the prefix a stage resume inherits)."""
        return {k: [v[0], set(v[1])] for k, v in self._saved[id(state)][0].items()}

    def seed(self, events: dict) -> None:
        """Start a resumed session with the prefix events of the run that took the snapshot."""
        for k, v in events.items():      # merge by max: the new session's own reset may already have observed contacts
            cur = self.events.setdefault(k, [0.0, set()])
            cur[0], cur[1] = max(cur[0], v[0]), cur[1] | set(v[1])

    def report(self, stages: Optional[Sequence[str]] = None) -> dict[str, Any]:
        r = classify(self.events, stages)
        r["rolled_back_not_counted"] = [{"robot_body": a, "other": b, "max_n": round(f, 1)} for (a, b), f in sorted(self.rolled_back.items()) if f >= CONTACT_THR_N]
        return r


def classify(events: dict, stages: Optional[Sequence[str]] = None, thr: float = CONTACT_THR_N) -> dict[str, Any]:
    stages = list(stages or ())
    forbidden, support = [], []
    for (rb, ob), (f, ks) in sorted(events.items()):
        if f < thr:
            continue
        row = {"robot_body": rb, "other": ob, "max_n": round(f, 1), "control_steps": len(ks)}
        row["class"] = "finger_link_on_target" if rb.endswith(FINGER_LINK_ON_TARGET) else "other"
        if ob.startswith(SUPPORT_PREFIXES):
            support.append(row)
        else:
            row["stages"] = sorted({stages[k].split(".")[-1] + "@" + stages[k].split(".")[0] for k in ks if 0 <= k < len(stages)})
            forbidden.append(row)
    other = [c for c in forbidden if c["class"] == "other"]
    return {"clean": not forbidden, "max_forbidden_n": max([c["max_n"] for c in forbidden], default=0.0), "forbidden_contacts": forbidden, "support_contacts_observed": support,
            "forbidden_by_class": {"finger_link_on_target": len(forbidden) - len(other), "other": len(other)},
            "clean_if_finger_link_on_target_allowed__preview_not_verdict": not other,
            "policy": {"threshold_n": thr, "allowed": "fingertip pads <-> current grasp target",
                       "forbidden": "any other robot body <-> any movable object", "observed_only": "robot <-> table/floor"}}
