"""Stage-programmed teacher that emits native 7D OSC actions (MannerCAP producer).

Every stage id comes from :mod:`synthesis_v2.stage_table`.  The teacher executes
a ``ProductionPlan`` in the *deployment* controller (see
:mod:`synthesis_v2.native_osc_env`); the recorded ``actions`` are exactly what
was sent to ``env.step``.

Grip strength is produced by a **contact-feedback gripper policy** that only
emits the native ``+1 / 0 / -1`` GRIP command: close (+1) until the measured
target-object force enters the requested band, then hold (0), with a single
reverse pulse (-1) if the force overshoots.  The privileged force reading lives
in the teacher only; the student-visible action interface is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from synthesis_v2.native_osc_env import (
    DT,
    GRIP_CLOSE,
    GRIP_HOLD,
    GRIP_OPEN,
    POS_SCALE,
    ROT_SCALE,
    CAM_MAIN,
    CAM_WRIST,
    NativeSession,
    StepTelemetry,
    rotvec_delta,
    state8,
)
from synthesis_v2.stage_table import stage_ids

PRODUCER_VERSION = "native_producer.v1"


@dataclass(frozen=True)
class StagePlan:
    stage_id: str
    v_max_mps: float
    w_max_rps: float = 1.0
    max_steps: int = 250


@dataclass(frozen=True)
class GripPlan:
    """Total (left + right) target-object force band in N for the grasp stage."""

    target_lo_n: float
    target_hi_n: float
    max_close_steps: int = 30
    settle_steps: int = 3
    # Release/recovery controls (default 0 = historical behaviour).  The gripper action is sign-only with a fixed
    # target increment (+-0.25 of range per control step, ~10 N summed force on a blocked pair), so a pulse is coarse
    # and the control-step mean right after a pulse mixes pre- and post-pulse force.
    settle_after_open: int = 0   # hold this many steps after a pulse before judging the force again
    reclose_max: int = 0         # bounded close pulses allowed when the re-checked force is below target_lo_n


@dataclass(frozen=True)
class ProductionPlan:
    task: str
    stages: tuple[StagePlan, ...]
    grip: GripPlan
    open_steps: int = 10
    align_yaw: bool = True

    def stage(self, stage_id: str) -> StagePlan:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(stage_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stages": [
                {"stage_id": s.stage_id, "v_max_mps": s.v_max_mps, "w_max_rps": s.w_max_rps, "max_steps": s.max_steps}
                for s in self.stages
            ],
            "grip": {
                "target_lo_n": self.grip.target_lo_n,
                "target_hi_n": self.grip.target_hi_n,
                "max_close_steps": self.grip.max_close_steps,
                "settle_steps": self.grip.settle_steps,
                **({"settle_after_open": self.grip.settle_after_open} if self.grip.settle_after_open else {}),
                **({"reclose_max": self.grip.reclose_max} if self.grip.reclose_max else {}),
            },
            "open_steps": self.open_steps,
            "align_yaw": self.align_yaw,
            "producer_version": PRODUCER_VERSION,
        }

    def plan_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def default_stage_plans(task: str, v_max_mps: float = 0.12) -> tuple[StagePlan, ...]:
    return tuple(StagePlan(sid, v_max_mps) for sid in stage_ids(task))


class ContactFeedbackGrip:
    """Close until the total target force enters ``[lo, hi]``, then hold at 0."""

    def __init__(self, plan: GripPlan) -> None:
        self.plan = plan
        self.closing_steps = 0
        self.settled = 0
        self.in_band_at: Optional[int] = None
        self.reached = False
        self.reversed_pulses = 0
        self._last_total = 0.0
        # Observation only (does not change commands): what the policy saw and sent at every call.
        self.trace: list[dict[str, Any]] = []
        self._ignore = 0
        self.reclose_pulses = 0
        self.final: Optional[dict[str, float]] = None

    def _log(self, cmd: float, tel: Optional[StepTelemetry], total: float, note: str = "") -> float:
        self.trace.append({"cmd": float(cmd), "total_n": float(total), "note": note,
                           "left_n": float(tel.left_mean_n) if tel is not None else math.nan,
                           "right_n": float(tel.right_mean_n) if tel is not None else math.nan})
        return cmd

    def command(self, last_tel: Optional[StepTelemetry]) -> float:
        total = math.nan
        if last_tel is not None and last_tel.valid_fraction > 0:
            total = last_tel.left_mean_n + last_tel.right_mean_n
        if not self.reached:
            if total == total and total >= self.plan.target_lo_n:
                self.reached = True
                self.in_band_at = self.closing_steps
            else:
                self.closing_steps += 1
                return self._log(GRIP_CLOSE, last_tel, total)
        if self._ignore > 0:   # reading right after a pulse mixes old/new force: hold, judge on the next reading
            self._ignore -= 1
            return self._log(GRIP_HOLD, last_tel, total, "recheck_wait")
        if total == total and total > self.plan.target_hi_n and self.reversed_pulses < 3:
            self.reversed_pulses += 1
            self._ignore = self.plan.settle_after_open
            return self._log(GRIP_OPEN, last_tel, total, "release")
        if total == total and total < self.plan.target_lo_n and self.reclose_pulses < self.plan.reclose_max:
            self.reclose_pulses += 1
            self._ignore = max(1, self.plan.settle_after_open)
            return self._log(GRIP_CLOSE, last_tel, total, "reclose")
        self.settled += 1
        return self._log(GRIP_HOLD, last_tel, total)

    def finalize(self, last_tel: Optional[StepTelemetry]) -> None:
        """Record the reading after the last command (the policy itself never sees it)."""
        if last_tel is not None and last_tel.valid_fraction > 0:
            self.final = {"left_n": float(last_tel.left_mean_n), "right_n": float(last_tel.right_mean_n),
                          "total_n": float(last_tel.left_mean_n + last_tel.right_mean_n)}

    @property
    def done(self) -> bool:
        return (self.reached and self._ignore == 0 and self.settled >= self.plan.settle_steps) or (
            self.closing_steps >= self.plan.max_close_steps and not self.reached
        )


@dataclass
class RawEpisode:
    task: str
    seed: int
    plan: dict[str, Any]
    plan_hash: str
    status: str
    task_ok: bool
    arrays: dict[str, np.ndarray]
    stage_steps: dict[str, int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.arrays["actions"].shape[0])


class _Recorder:
    def __init__(self, session: NativeSession, keep_images: bool) -> None:
        self.s = session
        self.keep_images = keep_images
        self.stage_dtype = "<U16"
        self.track_binding = False  # program episodes also log which object is measured and the active band
        self.track_objects: list[str] = []  # scene objects whose position is logged every step
        self.hold_instance = ""
        self.hold_band = (math.nan, math.nan)
        self.cols: dict[str, list] = {k: [] for k in (
            "state", "actions", "next_state", "timestamp", "dt", "frame_index", "sim_step_index",
            "terminated", "truncated", "success", "stage",
            "left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "other_max_n", "valid_fraction",
            "joint_speed_mean", "joint_speed_max",
        )}
        self.images: list[np.ndarray] = []

    def add(self, obs_before: dict, action: np.ndarray, obs_after: dict, tel: StepTelemetry, stage: str, sim_idx: int, t0: float) -> None:
        c = self.cols
        if self.keep_images:
            self.images.append(
                np.stack([obs_before[f"{CAM_MAIN}_image"][::-1], obs_before[f"{CAM_WRIST}_image"][::-1]], axis=0).astype(np.uint8)
            )
        c["state"].append(state8(obs_before))
        c["actions"].append(np.asarray(action, dtype=np.float32))
        c["next_state"].append(state8(obs_after))
        c["timestamp"].append(t0)
        c["dt"].append(self.s.current_sim_time() - t0)
        c["frame_index"].append(len(c["state"]) - 1)
        c["sim_step_index"].append(sim_idx)
        c["terminated"].append(False)
        c["truncated"].append(False)
        c["success"].append(self.s.success())
        c["stage"].append(stage)
        if self.track_binding:
            c.setdefault("force_target", []).append(getattr(self.s, "force_target_name", None) or "")
            c.setdefault("hold_instance", []).append(self.hold_instance)
            c.setdefault("hold_lo_n", []).append(self.hold_band[0])
            c.setdefault("hold_hi_n", []).append(self.hold_band[1])
            for nm in self.track_objects:
                c.setdefault("pos_" + nm, []).append(np.asarray(obs_before[f"{nm}_pos"], dtype=np.float64))
                c.setdefault("pa_" + nm, []).append(np.asarray(obs_after[f"{nm}_pos"], dtype=np.float64))  # post-step pose
            if self.track_objects and hasattr(self.s, "object_contacts"):
                for nm, (lf, rf) in self.s.object_contacts(self.track_objects).items():
                    c.setdefault("cf_L_" + nm, []).append(lf)
                    c.setdefault("cf_R_" + nm, []).append(rf)
        for k in ("left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "other_max_n",
                  "valid_fraction", "joint_speed_mean", "joint_speed_max"):
            c[k].append(getattr(tel, k))

    def arrays(self) -> dict[str, np.ndarray]:
        c = self.cols
        out = {
            "state": np.asarray(c["state"], dtype=np.float32).reshape(-1, 8),
            "actions": np.asarray(c["actions"], dtype=np.float32).reshape(-1, 7),
            "next_state": np.asarray(c["next_state"], dtype=np.float32).reshape(-1, 8),
            "timestamp": np.asarray(c["timestamp"], dtype=np.float64),
            "dt": np.asarray(c["dt"], dtype=np.float64),
            "frame_index": np.asarray(c["frame_index"], dtype=np.int64),
            "sim_step_index": np.asarray(c["sim_step_index"], dtype=np.int64),
            "terminated": np.asarray(c["terminated"], dtype=bool),
            "truncated": np.asarray(c["truncated"], dtype=bool),
            "success": np.asarray(c["success"], dtype=bool),
            "stage": np.asarray(c["stage"], dtype=self.stage_dtype),
        }
        for k in ("left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "other_max_n",
                  "valid_fraction", "joint_speed_mean", "joint_speed_max"):
            out[k] = np.asarray(c[k], dtype=np.float64)
        if "force_target" in c:
            out["force_target"] = np.asarray(c["force_target"], dtype="<U32")
            out["hold_instance"] = np.asarray(c["hold_instance"], dtype="<U48")
            out["hold_lo_n"] = np.asarray(c["hold_lo_n"], dtype=np.float64)
            out["hold_hi_n"] = np.asarray(c["hold_hi_n"], dtype=np.float64)
            for k in c:
                if k.startswith(("pos_", "pa_")):
                    out[k] = np.asarray(c[k], dtype=np.float64).reshape(-1, 3)
                elif k.startswith("cf_"):
                    out[k] = np.asarray(c[k], dtype=np.float64)
        if self.keep_images:
            out["images"] = np.asarray(self.images, dtype=np.uint8)
        if len(out["terminated"]):
            out["terminated"][-1] = True
        return out


class _Runner:
    def __init__(self, session: NativeSession, plan: ProductionPlan, keep_images: bool) -> None:
        self.s = session
        self.plan = plan
        self.rec = _Recorder(session, keep_images)
        self.last_tel: Optional[StepTelemetry] = None
        self.stage_steps: dict[str, int] = {}
        self.timeouts: list[str] = []

    def _step(self, action: np.ndarray, stage: str) -> None:
        obs_before = self.s.obs
        sim_idx = self.s.sim_step_index
        t0 = self.s.current_sim_time()
        obs_after, tel = self.s.step(action)
        self.rec.add(obs_before, action, obs_after, tel, stage, sim_idx, t0)
        self.last_tel = tel
        self.stage_steps[stage] = self.stage_steps.get(stage, 0) + 1

    def servo(self, stage: str, target_pos, target_quat_xyzw, grip_cmd: float, *, tol_pos=0.004, tol_rot=0.03) -> bool:
        sp = self.plan.stage(stage)
        for _ in range(sp.max_steps):
            obs = self.s.obs
            pos_err = np.asarray(target_pos) - obs["robot0_eef_pos"]
            rot_err = rotvec_delta(obs["robot0_eef_quat"], target_quat_xyzw)
            pn, rn = float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))
            if pn < tol_pos and rn < tol_rot:
                return True
            mp, mr = sp.v_max_mps * DT, sp.w_max_rps * DT
            pos_step = pos_err / pn * mp if pn > mp else pos_err
            rot_step = rot_err / rn * mr if rn > mr else rot_err
            ps, rs = getattr(self.s, "pos_scale", POS_SCALE), getattr(self.s, "rot_scale", ROT_SCALE)  # metres / rad per unit action actually realised
            a = np.concatenate([np.clip(pos_step / ps, -1, 1), np.clip(rot_step / rs, -1, 1), [grip_cmd]])
            self._step(a.astype(np.float32), stage)
        self.timeouts.append(stage)
        return False

    def grip_only(self, stage: str, cmd: float, n: int) -> None:
        for _ in range(n):
            self._step(np.array([0, 0, 0, 0, 0, 0, cmd], dtype=np.float32), stage)

    def grasp(self, stage: str) -> ContactFeedbackGrip:
        pol = ContactFeedbackGrip(self.plan.grip)
        guard = self.plan.grip.max_close_steps + self.plan.grip.settle_steps + 6
        for _ in range(guard):
            cmd = pol.command(self.last_tel)
            self._step(np.array([0, 0, 0, 0, 0, 0, cmd], dtype=np.float32), stage)
            if pol.done:
                break
        pol.finalize(self.last_tel)
        return pol


def _yaw_aligned_quat(down_quat_xyzw: np.ndarray, cube_quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    yaw = R.from_quat(cube_quat_xyzw).as_euler("xyz")[2]
    yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4  # cube is 4-fold symmetric
    return (R.from_euler("z", yaw) * R.from_quat(down_quat_xyzw)).as_quat()


def _nut_handle_grasp_quat(down_quat_xyzw: np.ndarray, nut_quat_xyzw: np.ndarray) -> np.ndarray:
    """Grasp orientation for the square-nut handle tab.

    The handle geom's long axis is along the nut body's local +x (same yaw as
    the nut body quat); the gripper must close along the SHORT axis, i.e. the
    nut yaw rotated by +90 deg.  Only 2-fold symmetric (fingers are
    interchangeable under a 180 deg rotation), unlike the 4-fold cube case.
    """
    from scipy.spatial.transform import Rotation as R

    yaw = R.from_quat(nut_quat_xyzw).as_euler("xyz")[2]
    grip_yaw = yaw + math.pi / 2
    grip_yaw = (grip_yaw + math.pi / 2) % math.pi - math.pi / 2
    return (R.from_euler("z", grip_yaw) * R.from_quat(down_quat_xyzw)).as_quat()


@dataclass
class StageSnapshot:
    """State at the *start* of a stage block: lets a failed stage be re-run without redoing the prefix."""

    stage: str
    session: dict[str, Any]
    rec_cols: dict[str, list]
    rec_images: list
    last_tel: Optional[StepTelemetry]
    stage_steps: dict[str, int]
    timeouts: list[str]
    status: str
    meta: dict[str, Any]

    @property
    def prefix_steps(self) -> int:
        return len(self.rec_cols["actions"])


def block_names(task: str) -> tuple[str, ...]:
    """Re-runnable blocks in execution order (``place`` = servo + release)."""
    return tuple(sid for sid in stage_ids(task))


def first_diff_block(prev: ProductionPlan, new: ProductionPlan) -> Optional[str]:
    """Earliest block whose parameters differ; everything before it can be reused from a snapshot.

    ``None`` means the plans are identical.  ``"approach"`` means nothing is reusable.
    """
    if prev.task != new.task or prev.align_yaw != new.align_yaw:
        return "approach"
    for name in block_names(new.task):
        if name == "grasp":
            if prev.grip != new.grip:
                return name
            continue
        if prev.stage(name) != new.stage(name) or (name == "place" and prev.open_steps != new.open_steps):
            return name
    return None


def _take(rn: "_Runner", stage: str, status: str, meta: dict[str, Any]) -> StageSnapshot:
    return StageSnapshot(
        stage=stage, session=rn.s.save_state(),
        rec_cols={k: list(v) for k, v in rn.rec.cols.items()}, rec_images=list(rn.rec.images),
        last_tel=rn.last_tel, stage_steps=dict(rn.stage_steps), timeouts=list(rn.timeouts),
        status=status, meta=json.loads(json.dumps(meta)))


def produce_episode(
    task: str,
    seed: int,
    plan: ProductionPlan,
    *,
    keep_images: bool = True,
    session: Optional[NativeSession] = None,
    on_session: Optional[Callable[[NativeSession], None]] = None,
    snapshots: Optional[dict[str, StageSnapshot]] = None,
    resume: Optional[StageSnapshot] = None,
) -> RawEpisode:
    """Execute one full episode.  Always returns a ``RawEpisode``, even on failure.

    ``snapshots`` (out) receives a snapshot at the start of every stage block.  ``resume`` restarts from such
    a snapshot (taken on the same task/seed): the prefix is reused bit-for-bit and only the remaining blocks
    are executed with ``plan``.
    """

    from .program_executor import produce_program_episode, program_plan_from_production
    from .task_program import legacy_program

    return produce_program_episode(
        legacy_program(task), seed, program_plan_from_production(plan), keep_images=keep_images, session=session,
        on_session=on_session, snapshots=snapshots, resume=resume, plan_record=(plan.to_dict(), plan.plan_hash()))
