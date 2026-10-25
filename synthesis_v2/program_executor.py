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


def _kind(env_name: str) -> str:
    if env_name.startswith("@spot:"):
        return "spot"
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
                gc = it.params.get("grasp_candidate")  # opt-in: evaluate exactly this placement (same clearance criterion)
                grid = None if gc is None else [(float(gc["yaw_deg"]), float(gc.get("lateral_m", 0.0)),
                                                 float(gc.get("dz_m", 0.0)), float(gc.get("longitudinal_m", 0.0)))]
                pl = plan_approach(m_, d_, site_, env_, obst_, start_pos=np.asarray(s.obs["robot0_eef_pos"], float),
                                   start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"], float),
                                   grasp_pos=np.asarray(r["obj_pos0"], float),
                                   grasp_quat_xyzw=np.asarray(r["grasp_quat_xyzw"], float),
                                   hover_dz=APPROACH_DZ, target=tgt_, grid=grid)
                meta.setdefault("approach_plan", {})[it.subtask] = {**pl.to_dict(), "lookahead_control_steps": int(look_)}   # generation cost, not part of the episode
                if not pl.feasible:
                    status_box[0] = f"approach_infeasible:{it.instance_id}"
                    stop[0] = True
                    return
                r["grasp_offset"], r["grasp_quat_xyzw"] = list(pl.chosen.offset), list(pl.chosen.quat_xyzw)
            off = np.asarray(r.get("grasp_offset", (0.0, 0.0, 0.0)), dtype=np.float64)
            obj0 = np.asarray(r["obj_pos0"], dtype=np.float64) if "obj_pos0" in r else None
            quat = np.asarray(r.get("grasp_quat_xyzw", meta["down_quat_xyzw"]), dtype=np.float64)
            tgt0 = np.asarray(r["target_pos0"], dtype=np.float64) if "target_pos0" in r else None
            lab, prim = it.label, it.primitive
            nut = "center_offset" in r
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
                pol = rn.grasp_with(lab, plan.grip_for(lab))
                info = {"closing_steps": pol.closing_steps, "reached_band": pol.reached,
                        "reverse_pulses": pol.reversed_pulses, "in_band_at": pol.in_band_at, "reclose_pulses": pol.reclose_pulses,
                        "final": pol.final, "trace": pol.trace}
                if legacy:
                    meta["grasp_policy"] = info
                else:
                    meta.setdefault("grasp_policy_by_instance", {})[it.instance_id] = info
                if not pol.reached and status_box[0] == "completed":
                    status_box[0] = "grasp_band_not_reached" if legacy else f"grasp_band_not_reached:{it.instance_id}"
            elif prim == "lift":
                rn.servo(lab, obj0 + off + [0, 0, LIFT_DZ], quat, GRIP_HOLD)
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
                rn.servo(lab, tgt0 + off + [0, 0, dz], quat, GRIP_HOLD, tol_pos=0.008)
                rn.grip_only(lab, GRIP_OPEN, plan.open_steps)
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
        for it in program.instances[start:]:
            if not legacy:
                _update_hold(rn.rec, program, plan, it)
            if snapshots is not None:
                snapshots[it.label] = _take(rn, it.label, status_box[0], meta)
            run(it)
            if stop[0]:
                break
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
