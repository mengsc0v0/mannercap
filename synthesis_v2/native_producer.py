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
                return GRIP_CLOSE
        if total == total and total > self.plan.target_hi_n and self.reversed_pulses < 3:
            self.reversed_pulses += 1
            return GRIP_OPEN
        self.settled += 1
        return GRIP_HOLD

    @property
    def done(self) -> bool:
        return (self.reached and self.settled >= self.plan.settle_steps) or (
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
            "stage": np.asarray(c["stage"], dtype="<U16"),
        }
        for k in ("left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "other_max_n",
                  "valid_fraction", "joint_speed_mean", "joint_speed_max"):
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
            a = np.concatenate([np.clip(pos_step / POS_SCALE, -1, 1), np.clip(rot_step / ROT_SCALE, -1, 1), [grip_cmd]])
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

    owns = session is None
    s = session or NativeSession(task, seed, render=keep_images)
    status = "completed"
    rn = None
    try:
        obs = s.reset()
        if on_session:
            on_session(s)
        if task == "nut_assembly" and resume is None:
            # The nut is spawned slightly above its resting contact height; let it
            # settle onto the table (un-recorded) before any pose is captured as
            # "meta", otherwise obj_pos0 / handle_pos0 / center_offset are stale.
            neutral = np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], dtype=np.float32)
            for _ in range(15):
                obs, _tel = s.step(neutral)
        rn = _Runner(s, plan, keep_images)
        key = "cubeA" if task == "stack" else ("SquareNut" if task == "nut_assembly" else "cube")
        if resume is not None:
            if resume.session["control_step_index"] < 0:
                raise ValueError("invalid snapshot")
            s.restore_state(resume.session)
            rn.rec.cols = {k: list(v) for k, v in resume.rec_cols.items()}
            rn.rec.images = list(resume.rec_images)
            rn.last_tel, rn.stage_steps, rn.timeouts = resume.last_tel, dict(resume.stage_steps), list(resume.timeouts)
            status, meta = resume.status, json.loads(json.dumps(resume.meta))
        else:
            down = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
            if task == "nut_assembly":
                nut = s.env.nuts[s.env.nut_id]
                handle_pos0 = np.asarray(s.env.sim.data.get_site_xpos(nut.important_sites["handle"]), dtype=np.float64).copy()
                nut_pos0 = np.asarray(obs["SquareNut_pos"], dtype=np.float64).copy()
                quat = _nut_handle_grasp_quat(down, obs["SquareNut_quat"]) if plan.align_yaw else down
                meta = {
                    "obj_pos0": handle_pos0.tolist(),
                    "down_quat_xyzw": down.tolist(),
                    "grasp_quat_xyzw": quat.tolist(),
                    "center_offset": (nut_pos0 - handle_pos0).tolist(),
                    "peg_pos0": np.asarray(s.env.sim.data.body_xpos[s.env.peg1_body_id], dtype=np.float64).tolist(),
                }
            else:
                obj0 = np.asarray(obs[f"{key}_pos"], dtype=np.float64).copy()
                quat = _yaw_aligned_quat(down, obs[f"{key}_quat"]) if plan.align_yaw else down
                meta = {"obj_pos0": obj0.tolist(), "down_quat_xyzw": down.tolist(), "grasp_quat_xyzw": quat.tolist()}
                if task == "stack":
                    meta["cubeB_pos0"] = np.asarray(obs["cubeB_pos"]).tolist()
        obj0 = np.asarray(meta["obj_pos0"], dtype=np.float64)
        quat = np.asarray(meta["grasp_quat_xyzw"], dtype=np.float64)
        b0 = np.asarray(meta["cubeB_pos0"], dtype=np.float64) if task == "stack" else None
        center_offset = np.asarray(meta["center_offset"], dtype=np.float64) if task == "nut_assembly" else None
        peg0 = np.asarray(meta["peg_pos0"], dtype=np.float64) if task == "nut_assembly" else None
        status_box = [status]

        def run_grasp():
            pol = rn.grasp("grasp")
            meta["grasp_policy"] = {"closing_steps": pol.closing_steps, "reached_band": pol.reached,
                                    "reverse_pulses": pol.reversed_pulses, "in_band_at": pol.in_band_at}
            if not pol.reached:
                status_box[0] = "grasp_band_not_reached"

        def run_place():
            rn.servo("place", b0 + [0, 0, 0.041], quat, GRIP_HOLD, tol_pos=0.008)
            rn.grip_only("place", GRIP_OPEN, plan.open_steps)

        def run_place_nut():
            handle_target_xy = peg0[:2] - center_offset[:2]
            align_z = obj0[2] + 0.15
            # The square peg's hole clearance is only a few mm; compliant-grasp
            # drift during transport means the frozen center_offset is no longer
            # exact, so re-correct xy using the *live* nut position (closed loop)
            # before descending, otherwise the nut's square loop catches on the
            # peg shaft and the servo stalls indefinitely (observed as
            # stage_timeout:place with the eef stuck near the peg's top).
            for _ in range(3):
                cur_nut_xy = np.asarray(s.obs["SquareNut_pos"], dtype=np.float64)[:2]
                residual = peg0[:2] - cur_nut_xy
                if np.linalg.norm(residual) < 0.0015:
                    break
                handle_target_xy = handle_target_xy + residual
                rn.servo("place", np.array([handle_target_xy[0], handle_target_xy[1], align_z]),
                          quat, GRIP_HOLD, tol_pos=0.003)
            # obj0[2] is the handle's table-contact height measured during descend;
            # going further below it drives the servo into the table (infeasible,
            # causes a stage timeout).  Target a touch just above that height,
            # mirroring the old capx teacher's grasp_pos[2] + 0.01.
            rn.servo("place", np.array([handle_target_xy[0], handle_target_xy[1], obj0[2] + 0.01]),
                      quat, GRIP_HOLD, tol_pos=0.006)
            cur_nut_xy = np.asarray(s.obs["SquareNut_pos"], dtype=np.float64)[:2]
            residual = peg0[:2] - cur_nut_xy
            if np.linalg.norm(residual) > 0.015:
                corr_target = np.asarray(s.obs["robot0_eef_pos"], dtype=np.float64) + np.array([residual[0], residual[1], 0.0])
                rn.servo("place", corr_target, quat, GRIP_HOLD, tol_pos=0.004)
            rn.grip_only("place", GRIP_OPEN, plan.open_steps)

        if task == "nut_assembly":
            handle_target_xy0 = peg0[:2] - center_offset[:2]
            blocks = {
                "approach": lambda: rn.servo("approach", obj0 + [0, 0, 0.12], quat, GRIP_OPEN),
                "descend": lambda: rn.servo("descend", obj0, quat, GRIP_OPEN, tol_pos=0.003),
                "grasp": run_grasp,
                "lift": lambda: rn.servo("lift", obj0 + [0, 0, 0.15], quat, GRIP_HOLD),
                "transport": lambda: rn.servo(
                    "transport", np.array([handle_target_xy0[0], handle_target_xy0[1], obj0[2] + 0.15]), quat, GRIP_HOLD),
                "place": run_place_nut,
                "retreat": lambda: rn.servo("retreat", np.asarray(s.obs["robot0_eef_pos"]) + [0, 0, 0.15], quat, GRIP_OPEN),
            }
        else:
            blocks = {
                "approach": lambda: rn.servo("approach", obj0 + [0, 0, 0.12], quat, GRIP_OPEN),
                "descend": lambda: rn.servo("descend", obj0, quat, GRIP_OPEN),
                "grasp": run_grasp,
                "lift": lambda: rn.servo("lift", obj0 + [0, 0, 0.15], quat, GRIP_HOLD),
                "transport": lambda: rn.servo("transport", b0 + [0, 0, 0.15], quat, GRIP_HOLD),
                "place": run_place,
                "retreat": lambda: rn.servo("retreat", np.asarray(s.obs["robot0_eef_pos"]) + [0, 0, 0.15], quat, GRIP_OPEN),
            }
        order = block_names(task)
        start = order.index(resume.stage) if resume is not None else 0
        for name in order[start:]:
            if snapshots is not None:
                snapshots[name] = _take(rn, name, status_box[0], meta)
            blocks[name]()
        status = status_box[0]
        task_ok = s.success()
        if rn.timeouts and status == "completed":
            status = "stage_timeout:" + ",".join(rn.timeouts)
        meta["final_obj_pos"] = np.asarray(s.obs[f"{key}_pos"]).tolist()
        meta["initial_state"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.initial.items()}
        meta["sim_time_end"] = s.current_sim_time()
        if resume is not None:
            meta["resumed_from"] = {"stage": resume.stage, "prefix_steps": resume.prefix_steps}
        arrays = rn.rec.arrays()
        if keep_images and "images" not in arrays:
            arrays["images"] = np.zeros((0, 2, 224, 224, 3), dtype=np.uint8)
        return RawEpisode(task, seed, plan.to_dict(), plan.plan_hash(), status, bool(task_ok), arrays,
                          dict(rn.stage_steps), meta)
    except Exception as exc:  # preserve whatever was recorded
        arrays = rn.rec.arrays() if rn is not None else {"actions": np.zeros((0, 7), np.float32)}
        return RawEpisode(task, seed, plan.to_dict(), plan.plan_hash(), f"exception:{type(exc).__name__}:{exc}",
                          False, arrays, dict(rn.stage_steps) if rn else {}, {})
    finally:
        if owns:
            s.close()
