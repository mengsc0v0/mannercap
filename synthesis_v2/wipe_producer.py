"""Stage-programmed teacher for the RoboSuite ``Wipe`` task (native 7D OSC actions).

Same contract as :mod:`synthesis_v2.native_producer`: every recorded action is exactly the
vector passed to ``env.step``; the wiping tool has no fingers, so the 7th (GRIP) component is a
constant ``0`` and the two gripper-state dimensions are constant ``0`` padding.

Stages (``stage_table``): ``approach`` (hover over the first dirt particle), ``descend``
(fast to just above the table, then a slow contact-seek), ``sweep`` (pure-pursuit along the dirt
path with closed-loop tool-force control through the z action), ``retreat``.

The privileged tool-force reading (``mj_contactForce`` summed over tool-vs-world contacts) is used
by the teacher only; the student sees the unchanged 7D delta-OSC interface.  The sweep has a fixed
step budget (``StagePlan.max_steps``): a plan that cannot finish inside it fails, the budget is
never raised to make a plan pass.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from synthesis_v2.native_osc_env import DT, POS_SCALE, ROT_SCALE, NativeSession, StepTelemetry, rotvec_delta
from synthesis_v2.native_producer import (
    RawEpisode, StagePlan, StageSnapshot, _Recorder, _Runner, _take, block_names,
)

WIPE_PRODUCER_VERSION = "wipe_producer.v1"

HOVER_M = 0.06            # tool-plane clearance above the dirt for approach
SEEK_CLEARANCE_M = 0.015  # fast descent stops here, then the slow contact-seek starts
SEEK_SPEED_MPS = 0.03     # commanded speed of the contact-seek (fixed, not a manner dimension)
CONTACT_N = 1.0
RETREAT_M = 0.10
WAYPOINT_STRIDE = 3       # every 3rd dirt particle is a path waypoint (~1.5 cm apart)
PASS_TOL_M = 0.004
END_TOL_M = 0.006
MAX_EXTRA_PASSES = 60  # effectively bounded by the fixed step budget
EXTRA_OVERSHOOT_M = 0.02

# Open-loop feed-forward: |a_z| that holds a given total tool force on the table (measured with the
# deployment controller; the PI term removes the residual).
_FF_FORCE = np.array([0.0, 1.4, 2.5, 4.3, 7.8, 13.0, 21.9])
_FF_AZ = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30])
KP_AZ_PER_N = 0.004
KI_AZ_PER_NS = 0.02
I_MAX_N_S = 20.0
AZ_UP_MAX = 0.3
KV_SPEED = 0.5            # integral gain of the realised-speed loop (friction compensation), per control step
AZ_DOWN_MAX = 0.40  # beyond ~0.5 the tool leaves the contact regime (force reads 0, tool sinks); never command it


@dataclass(frozen=True)
class PressPlan:
    """Target mean tool-vs-table normal force (N) held during the sweep."""

    target_n: float


@dataclass(frozen=True)
class WipePlan:
    task: str
    stages: tuple[StagePlan, ...]
    press: PressPlan
    align_yaw: bool = False

    def stage(self, stage_id: str) -> StagePlan:
        for item in self.stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(stage_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "stages": [{"stage_id": s.stage_id, "v_max_mps": s.v_max_mps, "w_max_rps": s.w_max_rps,
                        "max_steps": s.max_steps} for s in self.stages],
            "press": {"target_n": self.press.target_n},
            "align_yaw": self.align_yaw,
            "producer_version": WIPE_PRODUCER_VERSION,
        }

    def plan_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


class _WipeRecorder(_Recorder):
    def __init__(self, session: NativeSession, keep_images: bool) -> None:
        super().__init__(session, keep_images)
        self.cols["tool_mean_n"] = []
        self.cols["tool_max_n"] = []

    def add(self, obs_before, action, obs_after, tel: StepTelemetry, stage, sim_idx, t0) -> None:
        super().add(obs_before, action, obs_after, tel, stage, sim_idx, t0)
        self.cols["tool_mean_n"].append(tel.tool_mean_n)
        self.cols["tool_max_n"].append(tel.tool_max_n)

    def arrays(self) -> dict[str, np.ndarray]:
        out = super().arrays()
        for k in ("tool_mean_n", "tool_max_n"):
            out[k] = np.asarray(self.cols[k], dtype=np.float64)
        return out


class _WipeRunner(_Runner):
    def __init__(self, session: NativeSession, plan: WipePlan, keep_images: bool) -> None:
        super().__init__(session, plan, keep_images)
        self.rec = _WipeRecorder(session, keep_images)
        sim = session.env.sim
        corners = session.env.robots[0].gripper["right"].important_geoms["corners"]
        self._corner_ids = [sim.model.geom_name2id(n) for n in corners]
        self.integ = 0.0

    def tool_center(self) -> np.ndarray:
        return np.asarray(self.s.env.sim.data.geom_xpos[self._corner_ids].mean(axis=0), dtype=np.float64)

    def tool_offset(self) -> np.ndarray:
        """Constant (orientation is held) offset tool centre - end-effector site."""
        return self.tool_center() - np.asarray(self.s.obs["robot0_eef_pos"], dtype=np.float64)

    def _force(self) -> float:
        t = self.last_tel
        return 0.0 if t is None or not math.isfinite(t.tool_mean_n) else float(t.tool_mean_n)

    def _rot(self, quat_xyzw: np.ndarray) -> np.ndarray:
        rot = rotvec_delta(self.s.obs["robot0_eef_quat"], quat_xyzw)
        return np.clip(rot / ROT_SCALE, -1, 1)

    def seek_contact(self, stage: str, quat: np.ndarray, max_steps: int) -> bool:
        for _ in range(max_steps):
            if self._force() >= CONTACT_N:
                return True
            a = np.zeros(7, dtype=np.float32)
            a[2] = -SEEK_SPEED_MPS * DT / POS_SCALE
            a[3:6] = self._rot(quat)
            self._step(a, stage)
        self.timeouts.append(stage)
        return False

    @staticmethod
    def _az(target_n: float, force_n: float, integ: float) -> float:
        if target_n > _FF_FORCE[-1]:
            slope = (_FF_AZ[-1] - _FF_AZ[-2]) / (_FF_FORCE[-1] - _FF_FORCE[-2])
            ff = float(_FF_AZ[-1] + (target_n - _FF_FORCE[-1]) * slope)
        else:
            ff = float(np.interp(target_n, _FF_FORCE, _FF_AZ))
        e = target_n - force_n
        return float(np.clip(-(ff + KP_AZ_PER_N * e + KI_AZ_PER_NS * integ), -AZ_DOWN_MAX, AZ_UP_MAX))

    def follow(self, stage: str, pts_xy: np.ndarray, quat: np.ndarray, target_n: float, budget: int) -> int:
        """Pure-pursuit along ``pts_xy`` at the stage speed with force-controlled pressing."""
        sp = self.plan.stage(stage)
        used, j = 0, 1
        seg_dir = np.zeros(2)
        xy_int, prev_tc = 0.0, self.tool_center()[:2].copy()
        while used < budget:
            tc = self.tool_center()[:2]
            while j < len(pts_xy) - 1:
                u = pts_xy[j] - pts_xy[j - 1]
                un = np.linalg.norm(u)
                u = u / un if un > 1e-9 else u
                if float(np.dot(pts_xy[j] - tc, u)) < PASS_TOL_M:
                    j += 1
                else:
                    break
            d = pts_xy[j] - tc
            dn = float(np.linalg.norm(d))
            last = j == len(pts_xy) - 1
            if last and (dn < END_TOL_M or float(np.dot(pts_xy[-1] - tc, pts_xy[-1] - pts_xy[-2])) < 0.0):
                break  # reached, or already passed, the end of the path
            if dn > 1e-9:
                seg_dir = d / dn
            v_real = float(np.dot(tc - prev_tc, seg_dir)) / DT if used else 0.0
            prev_tc = tc.copy()
            if used:
                xy_int = float(np.clip(xy_int + KV_SPEED * (sp.v_max_mps - v_real) * DT, 0.0, POS_SCALE))
            mag = sp.v_max_mps * DT + xy_int
            vxy = seg_dir * mag  # never capped by the remaining distance: a capped command stalls under stiction
            force = self._force()
            self.integ = float(np.clip(self.integ + (target_n - force) * DT, -I_MAX_N_S, I_MAX_N_S))
            a = np.zeros(7, dtype=np.float32)
            a[0:2] = np.clip(vxy / POS_SCALE, -1, 1)
            a[2] = self._az(target_n, force, self.integ)
            a[3:6] = self._rot(quat)
            self._step(a, stage)
            used += 1
        return used


def _flat_quat(session: NativeSession, corner_ids: list[int], quat_xyzw: np.ndarray) -> np.ndarray:
    """Orientation that makes the wiping plane (the plane the env uses for its wiped test) horizontal."""
    from scipy.spatial.transform import Rotation as R

    c = session.env.sim.data.geom_xpos[corner_ids]
    v1, v2 = c[0] - c[1], c[3] - c[1]
    n = np.cross(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
    n /= np.linalg.norm(n)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    sin = float(np.linalg.norm(axis))
    if sin < 1e-9:
        return np.asarray(quat_xyzw, dtype=np.float64)
    corr = R.from_rotvec(axis / sin * math.atan2(sin, float(n[2])))
    return (corr * R.from_quat(quat_xyzw)).as_quat()


def _marker_world(session: NativeSession) -> np.ndarray:
    sim = session.env.sim
    return np.stack([np.asarray(sim.data.body_xpos[sim.model.body_name2id(m.root_body)], dtype=np.float64)
                     for m in session.env.model.mujoco_arena.markers])


def _remaining_idx(session: NativeSession) -> list[int]:
    wiped = {id(m) for m in session.env.wiped_markers}
    return [i for i, m in enumerate(session.env.model.mujoco_arena.markers) if id(m) not in wiped]


def _path_from(session: NativeSession, idx: list[int], start_xy: np.ndarray) -> np.ndarray:
    mk = _marker_world(session)[:, :2]
    pts = [mk[i] for i in idx]
    order, cur, rem = [], np.asarray(start_xy), list(range(len(pts)))
    while rem:
        k = min(rem, key=lambda r: float(np.linalg.norm(pts[r] - cur)))
        order.append(k)
        cur = pts[k]
        rem.remove(k)
    return np.asarray([pts[k] for k in order]).reshape(-1, 2)


def produce_wipe_episode(
    task: str,
    seed: int,
    plan: WipePlan,
    *,
    keep_images: bool = True,
    session: Optional[NativeSession] = None,
    on_session: Optional[Callable[[NativeSession], None]] = None,
    snapshots: Optional[dict[str, StageSnapshot]] = None,
    resume: Optional[StageSnapshot] = None,
) -> RawEpisode:
    if task != "wipe":
        raise ValueError(task)
    owns = session is None
    s = session or NativeSession("wipe", seed, render=keep_images)
    status = "completed"
    rn: Optional[_WipeRunner] = None
    try:
        obs = s.reset()
        if on_session:
            on_session(s)
        rn = _WipeRunner(s, plan, keep_images)
        if resume is not None:
            s.restore_state(resume.session)
            rn.rec.cols = {k: list(v) for k, v in resume.rec_cols.items()}
            rn.rec.images = list(resume.rec_images)
            rn.last_tel, rn.stage_steps, rn.timeouts = resume.last_tel, dict(resume.stage_steps), list(resume.timeouts)
            status, meta = resume.status, json.loads(json.dumps(resume.meta))
        else:
            mk0 = _marker_world(s)
            q0 = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
            meta = {"down_quat_xyzw": _flat_quat(s, rn._corner_ids, q0).tolist(),
                    "initial_quat_xyzw": q0.tolist(),
                    "marker_pos0": mk0.tolist(), "table_z": float(mk0[:, 2].mean()),
                    "tool_offset": rn.tool_offset().tolist()}
        quat = np.asarray(meta["down_quat_xyzw"], dtype=np.float64)
        mk = np.asarray(meta["marker_pos0"], dtype=np.float64)
        off = np.asarray(meta["tool_offset"], dtype=np.float64)
        table_z = float(meta["table_z"])
        main_path = np.vstack([mk[::WAYPOINT_STRIDE, :2], mk[-1:, :2]])
        status_box = [status]

        def to_eef(tool_pos):
            return np.asarray(tool_pos, dtype=np.float64) - off

        def run_approach():
            rn.servo("approach", to_eef([main_path[0][0], main_path[0][1], table_z + HOVER_M]), quat, 0.0)

        def run_descend():
            rn.servo("descend", to_eef([main_path[0][0], main_path[0][1], table_z + SEEK_CLEARANCE_M + 0.005]), quat, 0.0)
            if not rn.seek_contact("descend", quat, plan.stage("descend").max_steps):
                status_box[0] = "no_contact"

        def run_sweep():
            rn.integ = 0.0
            budget = plan.stage("sweep").max_steps
            used = rn.follow("sweep", main_path, quat, plan.press.target_n, budget)
            passes = 0
            while passes < MAX_EXTRA_PASSES and used < budget and not s.success():
                idx = _remaining_idx(s)
                if not idx:
                    break
                path = _path_from(s, idx, rn.tool_center()[:2])
                tc0 = rn.tool_center()[:2]
                # overshoot the last particle so the tool keeps moving over it (a particle already under the tool
                # centre is otherwise skipped by the end-of-path test while the plane sits above its centre)
                lead = path[-1] - (path[-2] if len(path) > 1 else tc0)
                ln = float(np.linalg.norm(lead))
                lead = lead / ln if ln > 1e-6 else np.array([1.0 if passes % 2 == 0 else -1.0, 0.0])
                path = np.vstack([tc0[None, :], path, (path[-1] + EXTRA_OVERSHOOT_M * lead)[None, :]])
                used += rn.follow("sweep", path, quat, plan.press.target_n, budget - used)
                passes += 1
            meta["sweep"] = {"steps": used, "extra_passes": passes, "wiped": len(s.env.wiped_markers),
                             "budget": budget}
            if used >= budget and not s.success():
                rn.timeouts.append("sweep")

        def run_retreat():
            rn.servo("retreat", np.asarray(s.obs["robot0_eef_pos"]) + [0, 0, RETREAT_M], quat, 0.0)

        blocks = {"approach": run_approach, "descend": run_descend, "sweep": run_sweep, "retreat": run_retreat}
        order = block_names("wipe")
        start = order.index(resume.stage) if resume is not None else 0
        for name in order[start:]:
            if snapshots is not None:
                snapshots[name] = _take(rn, name, status_box[0], meta)
            blocks[name]()
        status = status_box[0]
        task_ok = s.success()
        if rn.timeouts and status == "completed":
            status = "stage_timeout:" + ",".join(rn.timeouts)
        meta["final_wiped"] = len(s.env.wiped_markers)
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
