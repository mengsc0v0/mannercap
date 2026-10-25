"""Step 7.0: approach-candidate generation with a whole-hand-envelope collision check.

The default approach (straight servo to a hover point, then a vertical descent) only
looks at the gripper site.  Here the hand is the full set of gripper collision geoms,
read from the model (not hard-coded), re-posed rigidly at each sample of the planned
path, and measured against every other object with ``mj_geomDistance``.

Candidates differ in the yaw of the grasp frame about the world z axis (the finger
opening axis).  The default candidate (yaw offset 0) is always evaluated first and is
returned unchanged when it is clear, so tasks that already work are not altered.
When no candidate clears the margin the plan is infeasible and carries a report; the
caller decides what to do (the executor stops and records the status).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

DIST_MAX = 0.05
DEFAULT_YAWS_DEG = (0, 90, -90, 45, -45, 135, -135, 180, 30, -30, 60, -60, 120, -120, 150, -150)


def _qmul(a, b):  # xyzw
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def yaw_quat(quat_xyzw, yaw_rad: float) -> np.ndarray:
    """``quat`` rotated by ``yaw_rad`` about the world z axis."""
    qz = np.array([0.0, 0.0, np.sin(yaw_rad / 2), np.cos(yaw_rad / 2)])
    q = _qmul(qz, np.asarray(quat_xyzw, dtype=np.float64))
    return q / np.linalg.norm(q)


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class HandEnvelope:
    """Gripper collision geoms expressed in the eef-site frame."""
    gids: tuple[int, ...]
    rel_pos: np.ndarray
    rel_mat: np.ndarray
    obs_to_site: np.ndarray = field(default_factory=lambda: np.eye(3))  # Rsite = R(obs quat) @ obs_to_site

    @classmethod
    def from_sim(cls, m, d, site_id: int, gids: Sequence[int], obs_quat_xyzw=None) -> "HandEnvelope":
        sp, sR = d.site_xpos[site_id].copy(), d.site_xmat[site_id].reshape(3, 3).copy()
        rp = np.array([sR.T @ (d.geom_xpos[g] - sp) for g in gids])
        rm = np.array([sR.T @ d.geom_xmat[g].reshape(3, 3) for g in gids])
        C = np.eye(3) if obs_quat_xyzw is None else quat_to_mat(obs_quat_xyzw).T @ sR
        return cls(tuple(int(g) for g in gids), rp, rm, C)


@dataclass
class Obstacle:
    name: str
    gids: tuple[int, ...]
    kind: str = "object"  # object | support


@dataclass
class Candidate:
    """One grasp placement: yaw about world z (0 or 180 = the two mirror grasps), a lateral shift along the finger-row
    (hand y) axis, and a height shift.  ``offset`` = hand target minus object centre (world frame)."""
    yaw_deg: float
    lateral_m: float
    dz_m: float
    quat_xyzw: list
    offset: list
    min_clearance_m: float
    blocking: Optional[str]
    phase: Optional[str]
    at_grasp_clearance_m: float
    feasible: bool
    longitudinal_m: float = 0.0  # shift along the hand x axis (perpendicular to the finger row); 0 for the default grid


@dataclass
class ApproachPlan:
    feasible: bool
    chosen: Optional[Candidate]
    candidates: list = field(default_factory=list)
    default_feasible: bool = False
    margin_m: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        def c(x):
            return None if x is None else {
                "yaw_deg": x.yaw_deg, "lateral_m": x.lateral_m, "dz_m": x.dz_m, "offset": [round(v, 5) for v in x.offset],
                "min_clearance_m": round(x.min_clearance_m, 5), "blocking": x.blocking, "phase": x.phase,
                "at_grasp_clearance_m": round(x.at_grasp_clearance_m, 5), "feasible": x.feasible,
                **({"longitudinal_m": x.longitudinal_m} if x.longitudinal_m else {})}
        return {"feasible": self.feasible, "default_feasible": self.default_feasible, "margin_m": self.margin_m,
                "n_samples": self.n_samples, "chosen": c(self.chosen), "candidates": [c(x) for x in self.candidates]}


DIST_LADDER = (DIST_MAX, 0.02, 0.01, 0.005, 0.002)


def _dist(mujoco, m, data, g, og, fromto) -> float:
    """mj_geomDistance with a guard for its spurious exact 0.0.

    Observed (reports/step7_onboard_diag, T0 init1, mesh hand geoms vs box objects): an exact 0.0 with zeroed nearest points
    comes back while an independent reference (min mesh-vertex-to-box distance, an UPPER bound that misses edge/face
    contact; scripts/mt/s7_zero_dist_validate.py) is 5-77 mm, for distmax 0.035..0.2 alike; shrinking distmax gives non-zero
    answers.  MuJoCo returns ``distmax`` itself when nothing is closer, so a rung-valued answer is only a lower bound.

    Scope (s7_zero_dist_validate.py, T0 init1, 1577 queries): 932 answers equal a rung (lower bound only), 645 are returned
    distances (all <= the vertex upper bound + 2 mm), none exact 0.0.  Valid use: "is clearance >= margin" decisions with
    margin <= the first rung that answered.  Not valid: treating rung-valued answers as measured distances, comparing
    clearances above DIST_MAX, or reading the vertex bound as proof of accuracy from below.  Geometry version is frozen."""
    for dm in DIST_LADDER:
        r = float(mujoco.mj_geomDistance(m, data, g, og, dm, fromto))
        if r != 0.0:
            return r
    return 0.0


def _clearance(m, mujoco, scratch, env: HandEnvelope, pos, R, obstacles: Sequence[Obstacle], reach: float):
    """(min signed distance, obstacle name) of the re-posed hand against ``obstacles``."""
    for k, g in enumerate(env.gids):
        scratch.geom_xpos[g] = pos + R @ env.rel_pos[k]
        scratch.geom_xmat[g] = (R @ env.rel_mat[k]).reshape(9)
    best, who = np.inf, None
    fromto = np.zeros(6)
    for ob in obstacles:
        for og in ob.gids:
            for g in env.gids:
                # mj_geomDistance is unreliable for far mesh/box pairs with a large distmax (returns 0), so only
                # pairs whose bounding spheres are within DIST_MAX are queried; the rest are >= DIST_MAX apart.
                if np.linalg.norm(scratch.geom_xpos[og] - scratch.geom_xpos[g]) > m.geom_rbound[og] + m.geom_rbound[g] + DIST_MAX:
                    continue
                dist = _dist(mujoco, m, scratch, g, og, fromto)
                if dist < best:
                    best, who = dist, ob.name
    return best, who


def candidate_grid(yaws_deg=(0,), laterals_m=(0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03), dzs_m=(0.0, 0.01, 0.02, 0.03, 0.04)):
    """Preference order: default first, then smallest departure from it (lateral, then height, then mirror yaw)."""
    g = [(y, l, z) for y in yaws_deg for l in laterals_m for z in dzs_m]
    return sorted(g, key=lambda t: (abs(t[1]) + 0.5 * abs(t[2]), t[0] != yaws_deg[0], t[1] < 0))


def extended_grid(yaws_deg=DEFAULT_YAWS_DEG, laterals_m=(0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03),
                  longitudinals_m=(0.0, 0.01, -0.01, 0.02, -0.02), dzs_m=(0.0, 0.01, 0.02, 0.03, 0.04)):
    """Opt-in widened candidate set: yaw x lateral x longitudinal x height as (yaw, lat, dz, lon) tuples.  Only a
    larger set of placements is offered; the feasibility criterion in ``plan_approach`` is not changed.  Preference
    order = smallest departure from the centred grasp (|lat| + |lon| + 0.5|dz|), then yaw order, then mirror sign."""
    g = [(y, l, z, o) for y in yaws_deg for l in laterals_m for o in longitudinals_m for z in dzs_m]
    order = {y: i for i, y in enumerate(yaws_deg)}
    return sorted(g, key=lambda t: (abs(t[1]) + abs(t[3]) + 0.5 * abs(t[2]), order[t[0]], t[1] < 0, t[3] < 0))


def plan_approach(m, d, site_id: int, env: HandEnvelope, obstacles: Sequence[Obstacle], *,
                  start_pos, start_quat_xyzw, grasp_pos, grasp_quat_xyzw, hover_dz: float,
                  margin_m: float = 0.0, grid=None, target: Optional[Obstacle] = None, evaluate_all: bool = False,
                  n_traverse: int = 10, n_descent: int = 10, above_grasp_m: float = 0.02) -> ApproachPlan:
    """Evaluate grasp placements in preference order; the first whose whole path clears ``margin_m`` is chosen.

    Path = traverse (start -> hover) + vertical descent (hover -> grasp), each sample checked with the whole hand
    envelope against every obstacle.  The target is an obstacle except within ``above_grasp_m`` of the grasp height
    (finger contact with it is the purpose of the motion); at the grasp pose the open hand must not penetrate it.
    The first grid entry is the unmodified default; the plan reports whether the default was already clear.
    """
    import mujoco
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    mujoco.mj_forward(m, scratch)
    start_pos, grasp_pos = np.asarray(start_pos, float), np.asarray(grasp_pos, float)
    R0 = quat_to_mat(start_quat_xyzw) @ env.obs_to_site
    cands: list[Candidate] = []
    chosen: Optional[Candidate] = None
    for entry in (grid or candidate_grid()):
        yaw, lat, dz = entry[:3]
        lon = float(entry[3]) if len(entry) > 3 else 0.0
        q = yaw_quat(grasp_quat_xyzw, np.deg2rad(yaw))
        Rq = quat_to_mat(q) @ env.obs_to_site
        off = Rq[:, 1] * lat + Rq[:, 0] * lon + np.array([0.0, 0.0, dz])
        gp = grasp_pos + off
        hover = gp + [0, 0, hover_dz]
        trav = [start_pos + (hover - start_pos) * t for t in np.linspace(0, 1, n_traverse)]
        desc = [hover + (gp - hover) * t for t in np.linspace(0, 1, n_descent)]
        worst, who, phase = np.inf, None, None
        for ph, pts, Rs in (("traverse", trav, (R0, Rq)), ("descent", desc, (Rq,))):
            for p in pts:
                for R in Rs:
                    obs = list(obstacles)
                    if target is not None and p[2] - gp[2] > above_grasp_m:
                        obs.append(target)
                    c, w = _clearance(m, mujoco, scratch, env, p, R, obs, 0.0)
                    if c < worst:
                        worst, who, phase = c, w, ph
        at_grasp = np.inf
        if target is not None:
            at_grasp, _ = _clearance(m, mujoco, scratch, env, gp, Rq, [target], 0.0)
        # strict: an exact 0.0 is "touching / inconclusive" (mj_geomDistance cannot tell), never a verified clearance.  T0 showed the
        # cost of accepting it: the palm rested on a tall can (clearance 0.0 reported, 13-26 N palm contact, lift lost).
        ok = worst > margin_m and at_grasp > 0.0
        cd = Candidate(float(yaw), float(lat), float(dz), [float(x) for x in q], [float(x) for x in off],
                       float(worst), who, phase, float(at_grasp), bool(ok), lon)
        cands.append(cd)
        if ok and chosen is None:
            chosen = cd
        if ok and not evaluate_all:
            break
    return ApproachPlan(chosen is not None, chosen, cands, bool(cands and cands[0].feasible), margin_m,
                        n_traverse + n_descent)


def settled_scene(session, max_steps: int = 60, vel_tol: float = 1e-3):
    """Objects can still be falling when a program starts (LIBERO init states are saved before they settle), so the
    plan must be made against the scene the hand will actually meet.  Look ahead with no-op steps, copy the settled
    MjData, and restore the session exactly (control-step counters included).  Returns (MjData, steps_used)."""
    import mujoco
    m, d = session._mj()
    snap = session.save_state()
    noop = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    used = 0
    try:
        for used in range(1, max_steps + 1):
            session.step(noop)
            v = max(float(np.abs(d.qvel[:]).max()), 0.0)
            if v < vel_tol:
                break
        d2 = mujoco.MjData(m)
        d2.qpos[:] = d.qpos
        d2.qvel[:] = 0
        mujoco.mj_forward(m, d2)
    finally:
        session.restore_state(snap)
    return d2, used


def scene_from_libero(session, target_env_name: str, *, settled: bool = True, return_steps: bool = False):
    """(m, d, site_id, HandEnvelope, obstacles, target) read from a LiberoSession (settled look-ahead by default).

    Obstacles are all other movable objects (containers included); the support table is not an obstacle of the
    approach corridor.  Hand = every collision geom whose body belongs to the gripper (envelope read from the model).
    """
    import mujoco
    m, d = session._mj()
    site = int(session.env.robots[0].eef_site_id)
    obs_quat = np.asarray(session.obs["robot0_eef_quat"], dtype=np.float64)
    used = 0
    if settled:
        d, used = settled_scene(session)
    gids = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith("gripper0")]
    env = HandEnvelope.from_sim(m, d, site, gids, obs_quat)
    obstacles = [Obstacle(n, tuple(session.geom_ids_of(n))) for n in session.env.objects_dict if n != target_env_name]
    target = Obstacle(target_env_name, tuple(session.geom_ids_of(target_env_name)))
    return (m, d, site, env, obstacles, target, used) if return_steps else (m, d, site, env, obstacles, target)
