"""Generic TaskProgram executor over shared primitives (design v3, step 1).

One executor runs any ``TaskProgram`` whose instances are drawn from the shared primitive set
(approach / descend / grasp / lift / transport / place / insert / retreat).  Geometry is resolved per
*object role* at the start of each subtask, plans are per *instance label* (one ``GripPlan`` per grasp
instance), so repeated stages such as ``cube.pick_2.grasp`` need no new task branch.

Legacy programs (``source == "legacy"``) reproduce the pre-v3 ``produce_episode`` action stream
bit-for-bit (same meta keys, same ``<U16`` stage column); new programs additionally record
``stage_instance_index``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .native_osc_env import GRIP_HOLD, GRIP_OPEN, NativeSession
from .native_producer import (
    GripPlan, ProductionPlan, RawEpisode, StagePlan, StageSnapshot, _nut_handle_grasp_quat, _Runner, _take,
    _yaw_aligned_quat,
)
from .task_program import TaskProgram, legacy_program

LIFT_DZ = 0.15
APPROACH_DZ = 0.12
STACK_PLACE_DZ = 0.041      # legacy calibrated box-on-box release height (eef above target centre)
SPOT_PLACE_DZ = 0.005       # put an object back down where it was picked


@dataclass(frozen=True)
class ProgramPlan:
    """Per-instance execution plan: ``StagePlan`` per label, ``GripPlan`` per grasp label."""

    program: TaskProgram
    stages: tuple[StagePlan, ...]
    grips: Mapping[str, GripPlan] = field(default_factory=dict)
    open_steps: int = 10
    align_yaw: bool = True
    approach_check: bool = False  # Step 7.0: whole-hand collision check + alternative grasp placements
    # Step 14 (A): end the branch at a grasp that did not reach its force band instead of lifting/transporting/placing
    # an object that is not held.  Off by default so every earlier result stays reproducible bit for bit.
    stop_on_grasp_failure: bool = False
    stop_on_place_failure: bool = False   # a place that did not reach its pose must not release (see the place branch)

    def stage(self, label: str) -> StagePlan:
        for sp in self.stages:
            if sp.stage_id == label:
                return sp
        raise KeyError(label)

    def grip_for(self, label: str) -> GripPlan:
        return self.grips[label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_hash": self.program.program_hash(),
            "stages": [{"stage_id": s.stage_id, "v_max_mps": s.v_max_mps, "w_max_rps": s.w_max_rps,
                        "max_steps": s.max_steps} for s in self.stages],
            "grips": {k: {"target_lo_n": g.target_lo_n, "target_hi_n": g.target_hi_n,
                          "max_close_steps": g.max_close_steps, "settle_steps": g.settle_steps,
                          **({"settle_after_open": g.settle_after_open} if g.settle_after_open else {}),
                          **({"reclose_max": g.reclose_max} if g.reclose_max else {})}
                      for k, g in sorted(self.grips.items())},
            "open_steps": self.open_steps, "align_yaw": self.align_yaw,
            **({"approach_check": True} if self.approach_check else {}),
            **({"stop_on_place_failure": True} if self.stop_on_place_failure else {}),
            **({"stop_on_grasp_failure": True} if self.stop_on_grasp_failure else {}),
        }

    def plan_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def program_plan_from_production(plan: ProductionPlan) -> ProgramPlan:
    prog = legacy_program(plan.task)
    return ProgramPlan(prog, tuple(plan.stages), {it.label: plan.grip for it in prog.of_primitive("grasp")},
                       plan.open_steps, plan.align_yaw)


def neutral_program_plan(program: TaskProgram, *, v_max_mps: float = 0.12,
                         grip_lo_n: float = 10.0, grip_hi_n: float = 16.0) -> ProgramPlan:
    """Default plan for any program: neutral speed everywhere, neutral band at every grasp instance."""
    return ProgramPlan(program, tuple(StagePlan(it.label, v_max_mps) for it in program.instances),
                       {it.label: GripPlan(grip_lo_n, grip_hi_n) for it in program.of_primitive("grasp")})


class _ProgramRunner(_Runner):
    def grasp_with(self, label: str, grip: GripPlan):
        saved = self.plan
        self.plan = _GripView(saved, grip)
        try:
            return self.grasp(label)
        finally:
            self.plan = saved


class _GripView:
    def __init__(self, base: ProgramPlan, grip: GripPlan) -> None:
        self._base, self.grip = base, grip

    def stage(self, label: str) -> StagePlan:
        return self._base.stage(label)


def _rotate_grasp_about_z(off: np.ndarray, quat_xyzw: np.ndarray, delta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R
    rz = R.from_euler("z", delta_rad)
    return rz.apply(off), (rz * R.from_quat(quat_xyzw)).as_quat()


def _origin_for_center(site_xyz: np.ndarray, center_body: np.ndarray, quat_xyzw: np.ndarray, delta_rad: float) -> np.ndarray:
    """Object-origin target such that the object's collision-box centre sits over ``site_xyz`` in x/y after the object has been
    turned about the vertical axis by ``delta_rad`` (z stays as given: the height is the task's release height)."""
    from scipy.spatial.transform import Rotation as R
    c = (R.from_euler("z", delta_rad) * R.from_quat(quat_xyzw)).apply(center_body)
    out = np.asarray(site_xyz, dtype=np.float64).copy()
    out[:2] -= c[:2]
    return out


def _kind(env_name: str) -> str:
    if env_name.startswith("@spot:"):
        return "spot"
    if env_name.startswith("@site:"):
        return "site"
    if env_name.endswith("Nut"):
        return "nut"
    if env_name.startswith("peg"):
        return "peg"
    return "box"


class _Geometry:
    """Resolves object roles to poses on a ``NativeSession``; records per-subtask captures in ``meta``."""

    def __init__(self, s: NativeSession, program: TaskProgram, align_yaw: bool, meta: dict) -> None:
        self.s, self.p, self.align_yaw, self.meta = s, program, align_yaw, meta
        self.legacy = program.source == "legacy"

    def name(self, role: str) -> str:
        return self.p.objects[role]

    def capture(self, sub: str, obj_role: Optional[str], tgt_role: Optional[str]) -> dict:
        s, obs = self.s, self.s.obs
        down = np.asarray(self.meta["down_quat_xyzw"], dtype=np.float64)
        rec: dict[str, Any] = {}
        if obj_role is not None:
            nm = self.name(obj_role)
            if _kind(nm) == "nut":
                nut = s.env.nuts[s.env.nut_id]
                handle = np.asarray(s.env.sim.data.get_site_xpos(nut.important_sites["handle"]), dtype=np.float64).copy()
                centre = np.asarray(obs[f"{nm}_pos"], dtype=np.float64).copy()
                quat = _nut_handle_grasp_quat(down, obs[f"{nm}_quat"]) if self.align_yaw else down
                rec.update(obj_pos0=handle.tolist(), grasp_quat_xyzw=quat.tolist(),
                           center_offset=(centre - handle).tolist())
            else:
                pos = np.asarray(obs[f"{nm}_pos"], dtype=np.float64).copy()
                quat = _yaw_aligned_quat(down, obs[f"{nm}_quat"]) if self.align_yaw else down
                rec.update(obj_pos0=pos.tolist(), grasp_quat_xyzw=quat.tolist())
        if tgt_role is not None:
            nm = self.name(tgt_role)
            k = _kind(nm)
            if k == "peg":
                if nm != "peg1":
                    raise ValueError(f"unsupported peg {nm!r}")
                rec["target_pos0"] = np.asarray(s.env.sim.data.body_xpos[s.env.peg1_body_id], dtype=np.float64).tolist()
            elif k == "site":
                self._capture_site_target(rec, sub, nm, obj_role)
            elif k == "spot":
                ref = nm.split(":", 1)[1]
                if ref == obj_role:  # put the object back where this subtask just picked it up
                    src = rec
                else:
                    src = next((r for r in self.meta["subtasks"].values() if r.get("obj_role") == ref), None)
                    if src is None:
                        raise ValueError(f"spot target {nm!r} references role {ref!r} with no earlier capture")
                rec["target_pos0"] = list(src["obj_pos0"])
            else:
                rec["target_pos0"] = np.asarray(obs[f"{nm}_pos"]).tolist()
        if self.legacy:
            self.meta["obj_pos0"] = rec["obj_pos0"]
            self.meta["down_quat_xyzw"] = self.meta.pop("down_quat_xyzw")
            self.meta["grasp_quat_xyzw"] = rec["grasp_quat_xyzw"]
            if "center_offset" in rec:
                self.meta["center_offset"] = rec["center_offset"]
                self.meta["peg_pos0"] = rec["target_pos0"]
            elif "target_pos0" in rec:
                self.meta["cubeB_pos0"] = rec["target_pos0"]
        else:
            rec.update(obj_role=obj_role, target_role=tgt_role)
            self.meta["subtasks"][sub] = rec
        return rec

    def _capture_site_target(self, rec: dict, sub: str, nm: str, obj_role: Optional[str]) -> None:
        """Target = a named region site of the scene (``@site:<site name>``), optionally displaced / yawed in the SITE frame.
        The task config supplies, as instance params of the subtask: ``target_offset_site`` (x, y, z in the site frame),
        ``target_yaw_deg`` (object yaw relative to the site) and ``target_yaw_symmetry_deg`` (the object looks the same after
        this rotation; the smallest rotation is used).  ``target_pos0`` is where the OBJECT should be; the hand target is
        derived from it through the grasp pose (executor: ``tgt0 + grasp offset``, rotated by ``target_yaw_delta_rad``)."""
        if not hasattr(self.s, "site_pose"):
            raise ValueError(f"backend {type(self.s).__name__} cannot resolve site targets")
        pos, mat = self.s.site_pose(nm.split(":", 1)[1])
        prm: dict[str, Any] = {}
        for i in self.p.instances:
            if i.subtask == sub and i.target_role:
                prm.update(i.params)
        off = np.asarray(prm.get("target_offset_site", (0.0, 0.0, 0.0)), dtype=np.float64)
        rec["target_pos0"] = (pos + mat @ off).tolist()
        rec["target_site"] = nm.split(":", 1)[1]
        if prm.get("target_align") == "collision_center":
            # the region means "the object fits in here": put the object's COLLISION centre (not its body origin) over the
            # site; the success predicate still looks at the origin, which therefore ends up offset by the same amount
            if not hasattr(self.s, "collision_center_in_body") or obj_role is None:
                raise ValueError("target_align needs a backend with collision_center_in_body and an object role")
            rec["target_site_xyz"] = list(rec["target_pos0"])
            rec["target_align_center_body"] = self.s.collision_center_in_body(self.name(obj_role)).tolist()
            rec["target_pos0"] = _origin_for_center(np.asarray(rec["target_site_xyz"]), np.asarray(rec["target_align_center_body"]),
                                                    np.asarray(self.s.obs[f"{self.name(obj_role)}_quat"], dtype=np.float64), 0.0).tolist()
        if "target_yaw_deg" in prm:
            if obj_role is None:
                raise ValueError("a target yaw needs the object role of the subtask")
            from scipy.spatial.transform import Rotation as R
            obj_yaw = R.from_quat(np.asarray(self.s.obs[f"{self.name(obj_role)}_quat"], dtype=np.float64)).as_euler("xyz")[2]
            site_yaw = math.atan2(mat[1, 0], mat[0, 0])
            sym = math.radians(float(prm.get("target_yaw_symmetry_deg", 360.0)))
            want = site_yaw + math.radians(float(prm["target_yaw_deg"]))
            rec["target_yaw_delta_rad"] = float((want - obj_yaw + sym / 2) % sym - sym / 2)
            rec["target_yaw_want_rad"], rec["target_yaw_symmetry_rad"] = float(want), float(sym)
            if "target_align_center_body" in rec:
                rec["target_pos0"] = _origin_for_center(np.asarray(rec["target_site_xyz"]), np.asarray(rec["target_align_center_body"]),
                                                        np.asarray(self.s.obs[f"{self.name(obj_role)}_quat"], dtype=np.float64),
                                                        rec["target_yaw_delta_rad"]).tolist()

    def record(self, sub: str) -> Optional[dict]:
        if self.legacy:
            if "obj_pos0" not in self.meta:
                return None
            r = {"obj_pos0": self.meta["obj_pos0"], "grasp_quat_xyzw": self.meta["grasp_quat_xyzw"]}
            if "center_offset" in self.meta:
                r.update(center_offset=self.meta["center_offset"], target_pos0=self.meta["peg_pos0"])
            elif "cubeB_pos0" in self.meta:
                r["target_pos0"] = self.meta["cubeB_pos0"]
            return r
        return self.meta["subtasks"].get(sub)


def _update_hold(rec, program: TaskProgram, plan: "ProgramPlan", it) -> None:
    """Track which grasp instance's grip window is open (grasp .. release) and its band."""
    if it.primitive == "grasp":
        rec.hold_instance = it.instance_id
    elif rec.hold_instance and it.instance_id not in program.grip_scope(rec.hold_instance):
        rec.hold_instance = ""
    if rec.hold_instance:
        g = plan.grip_for(program.instance(rec.hold_instance).label)
        rec.hold_band = (g.target_lo_n, g.target_hi_n)
    else:
        rec.hold_band = (float("nan"), float("nan"))


def _subtask_roles(program: TaskProgram, sub: str) -> tuple[Optional[str], Optional[str]]:
    obj = next((i.object_role for i in program.instances if i.subtask == sub and i.object_role), None)
    tgt = next((i.target_role for i in program.instances if i.subtask == sub and i.target_role), None)
    return obj, tgt


def reference_to_candidate(ref: dict, obj_quat_xyzw, grasp_quat_xyzw, obs_to_site) -> tuple:
    """Express a task-adapter grasp reference as one planner candidate (yaw_deg, lateral_m, dz_m, longitudinal_m).

    ``ref`` = {"rel_obj": hand position minus object position in the object's yaw frame (m),
               "hand_yaw_rel_deg": hand yaw relative to the executor's yaw-aligned grasp quaternion}.
    The planner places the hand at obj + lat*Rq[:,1] + lon*Rq[:,0] + [0,0,dz]; the three offsets are solved exactly."""
    from scipy.spatial.transform import Rotation as R
    from .approach_planner import quat_to_mat, yaw_quat
    oyaw = R.from_quat(obj_quat_xyzw).as_euler("xyz")[2]
    off_w = R.from_euler("z", oyaw).apply(np.asarray(ref["rel_obj"], float))
    yaw = float(ref["hand_yaw_rel_deg"])
    Rq = quat_to_mat(yaw_quat(np.asarray(grasp_quat_xyzw, float), np.deg2rad(yaw))) @ np.asarray(obs_to_site, float)
    lon, lat, dz = np.linalg.solve(np.stack([Rq[:, 0], Rq[:, 1], [0.0, 0.0, 1.0]], axis=1), off_w)
    return (yaw, float(lat), float(dz), float(lon))


def grasp_grid_from_params(params: dict, reference_candidate: Optional[tuple] = None):
    """Candidate list for plan_approach from instance params.  Priority: explicit ``grasp_candidate`` (one placement, e.g.
    chosen by a repair/experience step) > ``grasp_reference`` (task-adapter reference, passed in resolved as
    ``reference_candidate``) > configured ``grasp_grid`` (task ranges) > None (planner default grid).  Returns (grid, source)."""
    gc = params.get("grasp_candidate")
    if gc is not None:
        return [(float(gc["yaw_deg"]), float(gc.get("lateral_m", 0.0)), float(gc.get("dz_m", 0.0)),
                 float(gc.get("longitudinal_m", 0.0)))], "grasp_candidate"
    if reference_candidate is not None:
        return [tuple(reference_candidate)], "grasp_reference"
    gg = params.get("grasp_grid")
    if gg is not None:
        from .approach_planner import candidate_grid
        return candidate_grid(tuple(gg["yaws_deg"]), tuple(gg["laterals_m"]), tuple(gg["dzs_m"])), "grasp_grid"
    return None, "default_grid"


def _candidate_key(c) -> tuple:
    """Identity of a grasp candidate, used only to refuse one the guard has already rejected."""
    return (round(float(c.yaw_deg), 6), round(float(c.lateral_m), 6), round(float(c.dz_m), 6),
            round(float(getattr(c, "longitudinal_m", 0.0)), 6))


def _seed_rejected(pre) -> dict:
    """Normalise caller-supplied refusals to the tuples :func:`_candidate_key` produces.  A caller that round-trips
    through JSON hands us lists, and ``tuple not in [list, ...]`` silently never matches, so a whole-episode re-run
    would keep re-picking the very candidate the guard already refused."""
    return {k: [_candidate_key_of(x) for x in v] for k, v in (pre or {}).items()}


def _candidate_key_of(x) -> tuple:
    return tuple(round(float(v), 6) for v in x)


def _restore_snapshot(rn, s, snap, meta: dict) -> None:
    """Rewind the session and the recorder to ``snap`` (same mechanism as ``resume``, in-episode).

    ``meta`` is mutated in place because the geometry recorder and ``run`` close over this dict; rebinding it would
    leave them pointing at the stale copy.  Dropping the post-snapshot ``subtasks`` entries is what makes the approach
    replan instead of reusing the refused grasp offset."""
    import json as _json
    s.restore_state(snap.session)
    rn.rec.cols = {k: list(v) for k, v in snap.rec_cols.items()}
    rn.rec.images = list(snap.rec_images)
    rn.last_tel, rn.stage_steps, rn.timeouts = snap.last_tel, dict(snap.stage_steps), list(snap.timeouts)
    keep = {k: meta[k] for k in ("execution_guard",) if k in meta}
    meta.clear()
    meta.update(_json.loads(_json.dumps(snap.meta)))
    meta.update(keep)


def produce_program_episode(
    program: TaskProgram,
    seed: int,
    plan: ProgramPlan,
    *,
    keep_images: bool = True,
    session: Optional[NativeSession] = None,
    on_session: Optional[Callable[[NativeSession], None]] = None,
    snapshots: Optional[dict[str, StageSnapshot]] = None,
    resume: Optional[StageSnapshot] = None,
    plan_record: Optional[tuple[dict, str]] = None,
    execution_guard: Optional[dict] = None,
) -> RawEpisode:
    """Execute ``program`` once; same contract as ``produce_episode`` (snapshots keyed by instance label)."""
    errs = program.validate()
    if errs:
        raise ValueError(f"invalid program: {errs}")
    task = program.env_task
    plan_dict, plan_hash = plan_record or (plan.to_dict(), plan.plan_hash())
    legacy = program.source == "legacy"
    owns = session is None
    s = session or NativeSession(task, seed, render=keep_images)
    status = "completed"
    rn = None
    try:
        obs = s.reset()
        if on_session:
            on_session(s)
        if task == "nut_assembly" and resume is None:
            neutral = np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32)
            for _ in range(15):
                obs, _tel = s.step(neutral)
        rn = _ProgramRunner(s, plan, keep_images)
        if not legacy:
            rn.rec.stage_dtype = "<U48"
            rn.rec.track_binding = True
            rn.rec.track_objects = [nm for nm in program.objects.values() if not nm.startswith("@")]
        if resume is not None:
            if resume.session["control_step_index"] < 0:
                raise ValueError("invalid snapshot")
            s.restore_state(resume.session)
            rn.rec.cols = {k: list(v) for k, v in resume.rec_cols.items()}
            rn.rec.images = list(resume.rec_images)
            rn.last_tel, rn.stage_steps, rn.timeouts = resume.last_tel, dict(resume.stage_steps), list(resume.timeouts)
            status, meta = resume.status, json.loads(json.dumps(resume.meta))
        else:
            meta = {"down_quat_xyzw": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy().tolist()}
            if not legacy:
                meta["subtasks"] = {}
        geo = _Geometry(s, program, plan.align_yaw, meta)
        if resume is None:  # first subtask is captured before any block (legacy order of operations)
            first = program.instances[0].subtask
            geo.capture(first, *_subtask_roles(program, first))
        status_box = [status]
        guard = dict(execution_guard or {})          # opt-in; absent => behaviour unchanged
        guard_on = bool(guard.get("enabled"))
        rejected: dict[str, list] = {}               # instance_id -> grasp candidate keys already refused by the guard
        # (seeded from guard["pre_rejected"] below so a whole-episode re-run can carry the refusals forward)
        retry = [None]                               # set by run() when the guard refuses the current candidate
        # geometry-retry budget: without it an outer "at most N force candidates" budget silently contains an
        # unbounded number of physical executions.  max_retries counts guard refusals over the WHOLE episode.
        max_retries = guard.get("max_retries")
        max_retries = None if max_retries is None else int(max_retries)
        n_retries = [0]
        # "restore": roll back to the approach snapshot and retry in-episode (stage recovery).
        # "abort":   stop the episode and report the refused candidate, so the CALLER must re-run the whole episode.
        #            The whole-rerun arm of the recovery ablation must use this, otherwise it would silently be
        #            using stage recovery inside the episode and the comparison would be meaningless.
        on_refusal = str(guard.get("on_refusal", "restore"))
        rejected.update(_seed_rejected(guard.get("pre_rejected")))
        if guard_on:
            meta.setdefault("execution_guard", {"config": {k: v for k, v in guard.items()},
                                                "max_retries": max_retries, "checks": [], "retries": []})

        def rec_for(sub: str) -> dict:
            r = geo.record(sub)
            return r if r is not None else geo.capture(sub, *_subtask_roles(program, sub))

        stop = [False]

        def run(it) -> None:
            r = rec_for(it.subtask)
            if it.primitive == "approach" and plan.approach_check and "grasp_offset" not in r:
                from .approach_planner import plan_approach, scene_from_libero
                nm0 = program.objects[it.object_role]
                m_, d_, site_, env_, obst_, tgt_, look_ = scene_from_libero(s, nm0, return_steps=True)
                ref_c = None
                if "grasp_reference" in it.params and "grasp_candidate" not in it.params:
                    ref_c = reference_to_candidate(it.params["grasp_reference"], s.obs[f"{nm0}_quat"],
                                                   r["grasp_quat_xyzw"], env_.obs_to_site)
                grid, grid_source = grasp_grid_from_params(it.params, ref_c)
                # plan_approach stops at the first feasible candidate; a guard retry needs the alternatives too
                all_c = bool(rejected.get(it.instance_id))
                pl = plan_approach(m_, d_, site_, env_, obst_, start_pos=np.asarray(s.obs["robot0_eef_pos"], float),
                                   start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"], float),
                                   grasp_pos=np.asarray(r["obj_pos0"], float),
                                   grasp_quat_xyzw=np.asarray(r["grasp_quat_xyzw"], float),
                                   hover_dz=APPROACH_DZ, target=tgt_, grid=grid, evaluate_all=all_c)
                meta.setdefault("approach_plan", {})[it.subtask] = {**pl.to_dict(), "lookahead_control_steps": int(look_),
                                                                    "grid_source": grid_source, "grid_size": 0 if grid is None else len(grid)}   # generation cost, not part of the episode
                chosen = pl.chosen
                rej = rejected.get(it.instance_id, [])
                if rej:   # guard retry: keep the planner's preference order, skip what the guard already refused
                    chosen = next((c for c in pl.candidates
                                   if c.feasible and _candidate_key(c) not in rej), None)
                    if meta["execution_guard"]["retries"]:   # empty when the refusals were pre-seeded by the caller
                        meta["execution_guard"]["retries"][-1]["next_candidate"] = None if chosen is None else _candidate_key(chosen)
                if chosen is None:
                    status_box[0] = (f"approach_infeasible:{it.instance_id}" if not rej
                                     else f"guard_candidates_exhausted:{it.instance_id}")
                    stop[0] = True
                    return
                if guard_on:
                    meta["execution_guard"].setdefault("chosen_candidate", {})[it.instance_id] = _candidate_key(chosen)
                r["grasp_offset"], r["grasp_quat_xyzw"] = list(chosen.offset), list(chosen.quat_xyzw)
            off = np.asarray(r.get("grasp_offset", (0.0, 0.0, 0.0)), dtype=np.float64)
            obj0 = np.asarray(r["obj_pos0"], dtype=np.float64) if "obj_pos0" in r else None
            quat = np.asarray(r.get("grasp_quat_xyzw", meta["down_quat_xyzw"]), dtype=np.float64)
            tgt0 = np.asarray(r["target_pos0"], dtype=np.float64) if "target_pos0" in r else None
            lab, prim = it.label, it.primitive
            nut = "center_offset" in r
            if "target_yaw_want_rad" in r and prim == "transport" and not r.get("target_yaw_refreshed"):
                # the object may have turned in the hand while it was grasped and lifted: from here on use the object's
                # ACTUAL yaw and the ACTUAL hand pose relative to it, not the planned grasp pose
                from scipy.spatial.transform import Rotation as R
                nm_o = program.objects[it.object_role]
                o_pos = np.asarray(s.obs[f"{nm_o}_pos"], dtype=np.float64)
                o_yaw = R.from_quat(np.asarray(s.obs[f"{nm_o}_quat"], dtype=np.float64)).as_euler("xyz")[2]
                sym_r = float(r["target_yaw_symmetry_rad"])
                r["target_yaw_delta_rad"] = float((r["target_yaw_want_rad"] - o_yaw + sym_r / 2) % sym_r - sym_r / 2)
                r["grasp_offset_actual"] = (np.asarray(s.obs["robot0_eef_pos"], dtype=np.float64) - o_pos).tolist()
                r["grasp_quat_actual_xyzw"] = np.asarray(s.obs["robot0_eef_quat"], dtype=np.float64).tolist()
                if "target_align_center_body" in r:   # re-aim with the object's ACTUAL orientation in the hand
                    r["target_pos0"] = _origin_for_center(np.asarray(r["target_site_xyz"]), np.asarray(r["target_align_center_body"]),
                                                          np.asarray(s.obs[f"{nm_o}_quat"], dtype=np.float64),
                                                          r["target_yaw_delta_rad"]).tolist()
                    tgt0 = np.asarray(r["target_pos0"], dtype=np.float64)
                r["target_yaw_refreshed"] = True
            if r.get("target_yaw_refreshed") and prim in ("transport", "place", "retreat"):
                off = np.asarray(r["grasp_offset_actual"], dtype=np.float64)
                quat = np.asarray(r["grasp_quat_actual_xyzw"], dtype=np.float64)
            if "target_yaw_delta_rad" in r and prim in ("transport", "place", "retreat"):
                # the object must end up yawed by delta; the hand keeps its pose relative to the object, so the hand
                # offset and the hand orientation are rotated by the same delta about the vertical axis
                off, quat = _rotate_grasp_about_z(off, quat, float(r["target_yaw_delta_rad"]))
            if prim == "approach":
                rn.servo(lab, obj0 + off + [0, 0, APPROACH_DZ], quat, GRIP_OPEN)
            elif prim == "descend":
                if nut:
                    rn.servo(lab, obj0, quat, GRIP_OPEN, tol_pos=0.003)
                else:
                    rn.servo(lab, obj0 + off, quat, GRIP_OPEN)
            elif prim == "grasp":
                if not legacy:
                    # measure this hold against the object this instance grasps, not a task-wide target
                    s.set_force_target(program.objects[it.object_role])
                    meta.setdefault("force_target_by_instance", {})[it.instance_id] = program.objects[it.object_role]
                tgt_name = program.objects[it.object_role]
                if guard_on:
                    from .execution_guard import pre_close_check
                    chk = pre_close_check(s, tgt_name, min_clearance_m=float(guard.get("min_clearance_m", 0.0)))
                    chk.update(instance_id=it.instance_id, control_step=int(s.control_step_index))
                    meta["execution_guard"]["checks"].append(chk)
                    if not chk["ok"]:
                        retry[0] = {"instance_id": it.instance_id, "check": "pre_close",
                                    "violations": chk["violations"], "control_step": chk["control_step"]}
                        return
                meta.setdefault("close_start_step_by_instance", {})[it.instance_id] = int(s.control_step_index)
                pol = rn.grasp_with(lab, plan.grip_for(lab))
                if guard_on:
                    from .execution_guard import forbidden_contacts_now
                    chk2 = forbidden_contacts_now(s, thr=float(guard.get("contact_threshold_n", 0.1)))
                    chk2.update(instance_id=it.instance_id, control_step=int(s.control_step_index))
                    meta["execution_guard"]["checks"].append(chk2)
                    if not chk2["ok"]:
                        retry[0] = {"instance_id": it.instance_id, "check": "post_close",
                                    "violations": chk2["forbidden_contacts_now"], "control_step": chk2["control_step"]}
                        return
                info = {"closing_steps": pol.closing_steps, "reached_band": pol.reached,
                        "reverse_pulses": pol.reversed_pulses, "in_band_at": pol.in_band_at, "reclose_pulses": pol.reclose_pulses,
                        "final": pol.final, "trace": pol.trace}
                if legacy:
                    meta["grasp_policy"] = info
                else:
                    meta.setdefault("grasp_policy_by_instance", {})[it.instance_id] = info
                if not pol.reached and status_box[0] == "completed":
                    status_box[0] = "grasp_band_not_reached" if legacy else f"grasp_band_not_reached:{it.instance_id}"
                if not pol.reached and plan.stop_on_grasp_failure:
                    # nothing after the grasp depends on a held object any more: record why and end the branch; the caller
                    # (bounded repair / the request) decides what happens next
                    meta.setdefault("grasp_failure", {})[it.instance_id] = {
                        "control_step": int(s.control_step_index), "object": tgt_name, "reached_band": False,
                        "reverse_pulses": pol.reversed_pulses, "closing_steps": pol.closing_steps, "final": pol.final}
                    stop[0] = True
            elif prim == "lift":
                # per-instance lift height (e.g. above a fixture wall the carry path crosses); default unchanged
                rn.servo(lab, obj0 + off + [0, 0, float(it.params.get("lift_dz", LIFT_DZ))], quat, GRIP_HOLD)
            elif prim == "transport":
                if nut:
                    hxy = tgt0[:2] - np.asarray(r["center_offset"])[:2]
                    rn.servo(lab, np.array([hxy[0], hxy[1], obj0[2] + LIFT_DZ]), quat, GRIP_HOLD)
                else:
                    # per-instance carry height (e.g. above a container rim + finger extent); default unchanged
                    rn.servo(lab, tgt0 + off + [0, 0, float(it.params.get("transport_dz", LIFT_DZ))], quat, GRIP_HOLD)
            elif prim == "place":
                dz = SPOT_PLACE_DZ if _kind(program.objects[it.target_role]) == "spot" else STACK_PLACE_DZ
                dz = float(it.params.get("place_dz", dz))  # per-instance release height (e.g. above a container rim)
                place_tol = float(it.params.get("place_tol_m", 0.008))
                place_goal = tgt0 + off + [0, 0, dz]
                placed = rn.servo(lab, place_goal, quat, GRIP_HOLD, tol_pos=place_tol)
                if not placed and plan.stop_on_place_failure:
                    # the release is only issued once the place pose is reached: a timed-out place is a place FAILURE (the
                    # object is still held), recorded for the bounded repair / the request -- it is not turned into a
                    # release by luck
                    _eef = np.asarray(s.obs["robot0_eef_pos"], dtype=np.float64)
                    _obj = np.asarray(s.obs[f"{program.objects[it.object_role]}_pos"], dtype=np.float64)
                    meta.setdefault("place_failure", {})[it.instance_id] = {
                        "control_step": int(s.control_step_index), "tol_pos_m": place_tol,
                        "eef_pos": _eef.tolist(), "eef_goal": np.asarray(place_goal, dtype=np.float64).tolist(),
                        "eef_error_m": float(np.linalg.norm(place_goal - _eef)),
                        "object_pos": _obj.tolist(), "target_pos0": np.asarray(tgt0, dtype=np.float64).tolist()}
                    if status_box[0] == "completed":
                        status_box[0] = f"place_not_reached:{it.instance_id}"
                    stop[0] = True
                    return
                # the hold window ends here, inside place: the release is issued before the stage does.  Both
                # bounds are recorded because the opening motion itself is still authorised contact with the held
                # object -- the window closes only once the fingers have finished opening.
                _hold_id = rn.rec.hold_instance or it.instance_id
                meta.setdefault("release_start_step_by_instance", {})[_hold_id] = int(s.control_step_index)
                rn.grip_only(lab, GRIP_OPEN, plan.open_steps)
                meta.setdefault("release_end_step_by_instance", {})[_hold_id] = int(s.control_step_index)
            elif prim == "insert":
                _insert_nut(rn, s, lab, program.objects[it.object_role], obj0, quat, tgt0,
                            np.asarray(r["center_offset"], dtype=np.float64), plan.open_steps)
            elif prim == "retreat":
                rn.servo(lab, np.asarray(s.obs["robot0_eef_pos"]) + [0, 0, LIFT_DZ], quat, GRIP_OPEN)
            else:
                raise ValueError(f"primitive {prim!r} not executable on robosuite_native")

        order = [it.label for it in program.instances]
        start = order.index(resume.stage) if resume is not None else 0
        if not legacy:  # a resumed run re-enters the middle of a hold: rebuild the active window from the program
            for it in program.instances[:start]:
                _update_hold(rn.rec, program, plan, it)
        approach_snap: dict[str, tuple] = {}   # instance_id -> (loop index, snapshot, hold state) before approach
        i = start
        while i < len(program.instances):
            it = program.instances[i]
            if not legacy:
                _update_hold(rn.rec, program, plan, it)
            if snapshots is not None:
                snapshots[it.label] = _take(rn, it.label, status_box[0], meta)
            if guard_on and it.primitive == "approach" and it.instance_id not in approach_snap:
                approach_snap[it.instance_id] = (i, _take(rn, it.label, status_box[0], meta),
                                                 (rn.rec.hold_instance, rn.rec.hold_band))
            run(it)
            if retry[0] is not None:
                ev = retry[0]
                retry[0] = None
                # the grasp instance shares its subtask with the approach that chose the candidate
                owner = next((x.instance_id for x in program.instances
                              if x.primitive == "approach" and x.subtask == it.subtask), None)
                snap = approach_snap.get(owner)
                key = meta.get("execution_guard", {}).get("chosen_candidate", {}).get(owner)
                if on_refusal == "abort":
                    meta["execution_guard"]["retries"].append({**ev, "approach_instance": owner,
                                                               "refused_candidate": key, "aborted_episode": True})
                    status_box[0] = f"guard_refused_abort:{it.instance_id}"
                    break
                if snap is None or key is None:
                    status_box[0] = f"guard_refused_no_snapshot:{it.instance_id}"
                    break
                if max_retries is not None and n_retries[0] >= max_retries:
                    meta["execution_guard"]["retries"].append({**ev, "approach_instance": owner,
                                                               "refused_candidate": key, "budget_exhausted": True})
                    status_box[0] = f"guard_retry_budget_exhausted:{it.instance_id}"
                    break
                n_retries[0] += 1
                rejected.setdefault(owner, []).append(key)
                meta["execution_guard"]["retries"].append({**ev, "approach_instance": owner,
                                                           "refused_candidate": key,
                                                           "restored_to_stage": snap[1].stage,
                                                           "steps_discarded": int(len(rn.rec.cols["actions"]) - snap[1].prefix_steps),
                                                           "resume_prefix_steps": int(snap[1].prefix_steps),
                                                           "retry_index": n_retries[0]})
                _restore_snapshot(rn, s, snap[1], meta)
                rn.rec.hold_instance, rn.rec.hold_band = snap[2]
                i = snap[0]
                continue
            if stop[0]:
                break
            i += 1
        status = status_box[0]
        task_ok = s.success()
        if rn.timeouts and status == "completed":
            status = "stage_timeout:" + ",".join(rn.timeouts)
        roles = [r for r in program.objects if _kind(program.objects[r]) in ("box", "nut")]
        meta["final_obj_pos"] = np.asarray(s.obs[f"{program.objects[roles[0]]}_pos"]).tolist()
        if not legacy:
            meta["final_obj_pos_by_role"] = {r: np.asarray(s.obs[f"{program.objects[r]}_pos"]).tolist() for r in roles}
        meta["initial_state"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.initial.items()}
        meta["sim_time_end"] = s.current_sim_time()
        if resume is not None:
            meta["resumed_from"] = {"stage": resume.stage, "prefix_steps": resume.prefix_steps}
        arrays = rn.rec.arrays()
        if not legacy:
            by_label = {it.label: i for i, it in enumerate(program.instances)}
            arrays["stage_instance_index"] = np.asarray([by_label[str(x)] for x in arrays["stage"]], dtype=np.int64)
            meta["program"] = program.to_dict()
        if keep_images and "images" not in arrays:
            arrays["images"] = np.zeros((0, 2, 224, 224, 3), dtype=np.uint8)
        return RawEpisode(task, seed, plan_dict, plan_hash, status, bool(task_ok), arrays, dict(rn.stage_steps), meta)
    except Exception as exc:  # preserve whatever was recorded
        arrays = rn.rec.arrays() if rn is not None else {"actions": np.zeros((0, 7), np.float32)}
        return RawEpisode(task, seed, plan_dict, plan_hash, f"exception:{type(exc).__name__}:{exc}",
                          False, arrays, dict(rn.stage_steps) if rn else {}, {})
    finally:
        if owns:
            s.close()


def _insert_nut(rn: _Runner, s: NativeSession, lab: str, nut_name: str, obj0, quat, peg0, center_offset,
                open_steps: int) -> None:
    """Closed-loop square-nut-on-peg insertion (moved verbatim from the legacy ``run_place_nut``)."""
    handle_target_xy = peg0[:2] - center_offset[:2]
    align_z = obj0[2] + 0.15
    for _ in range(3):
        cur_nut_xy = np.asarray(s.obs[f"{nut_name}_pos"], dtype=np.float64)[:2]
        residual = peg0[:2] - cur_nut_xy
        if np.linalg.norm(residual) < 0.0015:
            break
        handle_target_xy = handle_target_xy + residual
        rn.servo(lab, np.array([handle_target_xy[0], handle_target_xy[1], align_z]), quat, GRIP_HOLD, tol_pos=0.003)
    rn.servo(lab, np.array([handle_target_xy[0], handle_target_xy[1], obj0[2] + 0.01]), quat, GRIP_HOLD, tol_pos=0.006)
    cur_nut_xy = np.asarray(s.obs[f"{nut_name}_pos"], dtype=np.float64)[:2]
    residual = peg0[:2] - cur_nut_xy
    if np.linalg.norm(residual) > 0.015:
        corr_target = np.asarray(s.obs["robot0_eef_pos"], dtype=np.float64) + np.array([residual[0], residual[1], 0.0])
        rn.servo(lab, corr_target, quat, GRIP_HOLD, tol_pos=0.004)
    rn.grip_only(lab, GRIP_OPEN, open_steps)
