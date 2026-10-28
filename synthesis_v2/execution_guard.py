"""Execution guard: check the ACTUAL robot state just before the gripper closes, and again right after it has closed.

Why: the approach planner evaluates clearance on the COMMANDED grasp pose.  The executed pose differs (servo tracking
error), so a candidate that the planner calls feasible can still have the palm resting on an object by the time the hand
is down.  This guard adds no new safety margin: it reuses the existing requirement of the approach planner (clearance
must be strictly positive, ``margin_m`` = 0) and the existing contact criterion (``CONTACT_THR_N``), only evaluated on
the state the robot is actually in.

Deliberately NOT done here: deriving a margin from the penetration seen in episodes that already made contact.  If a
margin is ever needed it has to come from the tracking error observed BEFORE contact, which this guard records
(``clearance_m`` per obstacle at the pre-close check) so the evidence accumulates without being used as a threshold yet.

Checks
  pre_close    non-finger gripper bodies (the palm) vs every movable object and vs the support surface, on the live
               state, immediately before the close command.  Fails if any distance is <= 0.
  post_close   instantaneous forbidden-class contact (non-finger robot geom vs a movable object) above CONTACT_THR_N,
               read right after the close policy finished.
On failure the executor restores the snapshot taken before this instance's ``approach`` and retries the next feasible
grasp candidate; the attempt is charged to the same budget.  Pure observation: the guard never edits the plan.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from synthesis_v2.approach_planner import DIST_LADDER
from synthesis_v2.contact_audit import CONTACT_THR_N, SUPPORT_PREFIXES, _is_robot
from synthesis_v2.grasp_force_contract import classify_finger_geom

GUARD_VERSION = "execution_guard_v0"
MIN_CLEARANCE_M = 0.0     # same requirement as the approach planner: strictly positive, no extra margin


def non_finger_gripper_geoms(m, mujoco) -> tuple[int, ...]:
    """Collision geoms of the gripper that are NOT a finger pad or finger link (i.e. the palm/hand mesh)."""
    out = []
    for g in range(int(m.ngeom)):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if not (gn.startswith("gripper0") or bn.startswith("gripper0")):
            continue
        if classify_finger_geom(gn, bn) is None:
            out.append(g)
    return tuple(out)


def _dist(mujoco, m, d, g, og) -> float:
    """Same distance ladder as the approach planner (mj_geomDistance returns distmax when nothing is closer)."""
    ft = np.zeros(6)
    for dm in DIST_LADDER:
        r = float(mujoco.mj_geomDistance(m, d, g, og, dm, ft))
        if r != 0.0:
            return r
    return 0.0


def pre_close_check(session, target_name: str, *, min_clearance_m: float = MIN_CLEARANCE_M) -> dict[str, Any]:
    """Palm clearance against every movable object and the support surface, on the live state."""
    import mujoco
    m, d = session._mj()
    palm = non_finger_gripper_geoms(m, mujoco)
    obstacles = {n: tuple(session.geom_ids_of(n)) for n in session.env.objects_dict}
    support = tuple(g for g in range(int(m.ngeom))
                    if ((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
                        .startswith(SUPPORT_PREFIXES)) and (m.geom_contype[g] or m.geom_conaffinity[g]))
    # the support surface is checked against EVERY gripper collision geom, not only the palm: the approach planner
    # deliberately excludes the table from the approach corridor, so a fingertip pressing the table at the end of the
    # descend is invisible to it.  Movable objects are checked against the palm only (fingers are meant to touch them).
    all_gripper = tuple(g for g in range(int(m.ngeom)) if (m.geom_contype[g] or m.geom_conaffinity[g])
                        and ((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
                             .startswith("gripper0")))
    per = {}
    for name, gids, probes in ([(n, g, palm) for n, g in obstacles.items()] + [("__support__", support, all_gripper)]):
        best = np.inf
        for og in gids:
            for g in probes:
                if np.linalg.norm(d.geom_xpos[og] - d.geom_xpos[g]) > m.geom_rbound[og] + m.geom_rbound[g] + DIST_LADDER[0]:
                    continue
                best = min(best, _dist(mujoco, m, d, g, og))
        per[name] = None if not np.isfinite(best) else round(float(best), 5)
    bad = {k: v for k, v in per.items() if v is not None and v <= min_clearance_m}
    return {"check": "pre_close", "guard_version": GUARD_VERSION, "ok": not bad,
            "min_clearance_m": min_clearance_m, "clearance_m": per, "violations": bad,
            "palm_geoms": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in palm],
            "support_probe_geoms": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in all_gripper],
            "target": target_name,
            "note": "distances from the live state; a value equal to a ladder rung is a lower bound, which is enough "
                    "for a >0 decision but is not a measured distance."}


def forbidden_contacts_now(session, *, thr: float = CONTACT_THR_N) -> dict[str, Any]:
    """Instantaneous forbidden-class contacts: a non-finger robot geom touching a movable object."""
    import mujoco
    m, d = session._mj()
    buf = np.zeros(6)
    nmb = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
    hits = []
    for ci in range(int(d.ncon)):
        c = d.contact[ci]
        b1, b2 = nmb(int(c.geom1)), nmb(int(c.geom2))
        if _is_robot(b1) == _is_robot(b2):
            continue
        rg, ob = (int(c.geom1), b2) if _is_robot(b1) else (int(c.geom2), b1)
        if ob.startswith(SUPPORT_PREFIXES):
            continue
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, rg) or ""
        if classify_finger_geom(gn, nmb(rg)) is not None:
            continue                                   # a finger geom: judged by the contact policy, not by this guard
        mujoco.mj_contactForce(m, d, ci, buf)
        f = abs(float(buf[0]))
        if f >= thr:
            hits.append({"robot_geom": gn, "robot_body": nmb(rg), "other": ob, "force_n": round(f, 2)})
    return {"check": "post_close", "guard_version": GUARD_VERSION, "ok": not hits, "threshold_n": thr,
            "forbidden_contacts_now": hits}
