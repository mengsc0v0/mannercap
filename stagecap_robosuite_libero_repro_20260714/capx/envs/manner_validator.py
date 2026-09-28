"""Manner Validator — KITE extension for MannerCap.

Verifies that an executed trajectory satisfies the physical constraints
encoded in a :class:`MannerSpec`.  Integrates with the existing
``VerificationEvidence`` / ``StageStatus`` pipeline from
``vlm_planner_cap.schema``.

Typical usage
-------------
::

    from capx.envs.manner_validator import (
        MannerSpec, MannerValidator, MANNER_PRESETS, compose_manner_spec,
    )
    from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv

    spec = compose_manner_spec("slowly", "gently")

    env.begin_manner_recording()
    api.goto_pose_with_profile(pos, quat, v_max=0.04, a_max=0.02)
    api.set_impedance_stiffness(kp_pos=450)
    api.close_gripper_partial(force=0.75)
    api.goto_pose_with_profile(lift_pos, quat, v_max=0.04, jerk_limit=0.8)
    env.end_manner_recording()

    evidence = MannerValidator().validate(
        env.get_manner_telemetry(), spec, stage_id="lift_cube"
    )
    print(evidence.status, evidence.rationale)

Architecture
------------
* **MannerSpec** — declarative constraint set for one execution stage.
* **MannerTelemetryBuffer** — list[MannerStepSnapshot] captured by
  ``RobosuiteBaseEnv`` during motion.
* **MannerValidator.validate()** — evaluates four dimensions (A-D) and
  returns a ``VerificationEvidence`` compatible with the KITE loop.
* **MANNER_PRESETS** — maps adverbial keywords → ``MannerSpec``.
* **compose_manner_spec()** — merges multiple adverbial specs by taking
  the tightest bound in each dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Import from vlm_planner_cap for KITE compatibility.
# Graceful fallback so this module can be unit-tested stand-alone.
try:
    from vlm_planner_cap.schema import (
        StageStatus,
        VerificationEvidence,
    )
except ImportError:  # stand-alone test mode
    from enum import Enum  # type: ignore[assignment]

    class StageStatus(str, Enum):  # type: ignore[no-redef]
        SATISFIED = "satisfied"
        NOT_SATISFIED = "not_satisfied"
        NEEDS_REPAIR = "needs_repair"
        UNCERTAIN = "uncertain"

    @dataclass(frozen=True)
    class VerificationEvidence:  # type: ignore[no-redef]
        stage_id: str
        status: "StageStatus"
        rationale: str
        visual_evidence: list = field(default_factory=list)
        geometry_evidence: dict = field(default_factory=dict)
        log_evidence: list = field(default_factory=list)
        repair_hint: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
#  Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseVelocity:
    """Velocity override for one named sub-phase of a trajectory.

    Used by MannerSpec.phase_velocity to express intra-trajectory manner
    variation, e.g. "fast approach, then slow final grasp" instead of one
    constant speed for the whole episode.
    """

    max_joint_vel: float | None = None
    min_joint_vel: float | None = None


# Phase labels that correspond to profiled Cartesian/joint *moves* (as tagged
# by run_manner_ablation runners via goto_pose_with_profile(phase=...)).
# Telemetry stores these as BARE names (e.g. 'approach'), while older code /
# specs may still use the literal 'move' or a 'move:<phase>' prefix. _is_move
# accepts all three so velocity/jerk checks slice the right steps.
MOVE_PHASES = frozenset({
    "move", "approach", "descend", "lift", "transport", "final", "retreat",
})


def _is_move(phase: str) -> bool:
    """True if a telemetry phase label denotes a profiled move segment."""
    return phase in MOVE_PHASES or phase.startswith("move:")


def spec_has_phase_constraint(phase_velocity) -> bool:
    """True if any phase entry in a phase_velocity dict sets a real bound."""
    if not phase_velocity:
        return False
    return any(
        ov.max_joint_vel is not None or ov.min_joint_vel is not None
        for ov in phase_velocity.values()
    )


@dataclass
class MannerSpec:
    """Physical constraints for one manner-conditioned execution stage.

    All fields are optional; ``None`` means "no constraint on this dimension".

    Dimension mapping:
        A  velocity profiling   → max_joint_vel / min_joint_vel
        B  impedance stiffness  → kp_max / kp_min
        C  gripper partial ctrl → grip_fraction_max / grip_fraction_min
        D  jerk limiting        → max_jerk (rad/s³)
    """

    # ── Manner A: speed bounds (joint-space L2-norm, rad/s) ──
    max_joint_vel: float | None = None    # "slowly" / "缓缓" upper bound
    min_joint_vel: float | None = None    # "fast"   / "迅速" lower bound

    # ── Manner B: stiffness bounds (kp scalar or mean) ──
    kp_max: float | None = None           # "gently" / "轻轻" upper kp
    kp_min: float | None = None           # "firmly" / "用力" lower kp

    # ── Manner C: gripper fraction bounds (0=closed, 1=open) ──
    grip_fraction_max: float | None = None  # "lightly" max fraction (= 1-force)
    grip_fraction_min: float | None = None  # "firmly"  min closing

    # ── Manner D: jerk bound (rad/s³, computed from finite-diff of joint_acc) ──
    max_jerk: float | None = None

    # ── Tolerance: allowed fractional overshoot before flagging violation ──
    tolerance: float = 0.15

    # ── Human-readable label (for diagnostics) ──
    label: str = ""

    # Phase-specific velocity overrides (for intra-trajectory manner, e.g.
    # "fast approach, slow final grasp"). Keys: "approach"/"descend"/"lift"/
    # "final". None (default) = no phase overrides, entire episode uses the
    # flat max_joint_vel/min_joint_vel bounds above (old behavior).
    phase_velocity: dict[str, "PhaseVelocity"] | None = None

    def vel_bounds_for(self, phase: str) -> tuple[float | None, float | None]:
        """Return (max_joint_vel, min_joint_vel) bounds for a named phase.

        Falls back to the flat (episode-wide) bounds when no phase-specific
        override is present for `phase`.
        """
        if self.phase_velocity and phase in self.phase_velocity:
            ov = self.phase_velocity[phase]
            if ov.max_joint_vel is not None or ov.min_joint_vel is not None:
                return ov.max_joint_vel, ov.min_joint_vel
        return self.max_joint_vel, self.min_joint_vel

    def has_any_constraint(self) -> bool:
        """Return True if at least one constraint is defined."""
        flat_any = any(
            v is not None
            for v in (
                self.max_joint_vel, self.min_joint_vel,
                self.kp_max, self.kp_min,
                self.grip_fraction_max, self.grip_fraction_min,
                self.max_jerk,
            )
        )
        # A phase_velocity-only spec (no flat fallback field set) previously
        # slipped through here as "no constraint", silently skipping both
        # MannerValidator.validate() and compute_alignment_score() gating in
        # run_manner_ablation.py. All production mock/LLM specs so far always
        # pair phase_velocity with >=1 flat field, so this had no observed
        # regression, but is fixed here for correctness (2026-09-11).
        phase_any = bool(spec_has_phase_constraint(self.phase_velocity))
        return flat_any or phase_any


@dataclass
class MannerStepSnapshot:
    """One simulation step worth of physical measurements."""

    t: float                        # simulation time offset (s)
    phase: str                      # "move" | "grip" | "stiff" | "other"
    joint_vel: np.ndarray           # (7,) rad/s
    joint_acc: np.ndarray           # (7,) rad/s²
    kp: float                       # current joint position gain
    gripper_fraction: float         # 0=closed … 1=open
    gripper_qpos: np.ndarray        # (2,) m  (finger positions)
    cfrc_norm: float                # max external contact force norm (N)
    grasp_contact_force: float = 0.0  # true finger↔object normal contact force (N), via mj_contactForce
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(7))   # (7,) rad
    eef_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))     # (3,) m, world frame
    eef_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))    # (4,) wxyz, world frame
    object_names: tuple = field(default_factory=tuple)   # e.g. ("cube",) / ("cubeA","cubeB") / ("cup",)
    object_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))    # (N,3) m, world frame
    object_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # (N,4) wxyz, world frame


@dataclass
class DimensionResult:
    """Result for a single manner dimension check."""

    dim: str                        # "A", "B", "C", "D"
    name: str                       # human readable name
    passed: bool
    measured: float                 # key measured statistic
    threshold: float | None         # the applied limit
    detail: str                     # one-line explanation


@dataclass
class DimensionAlignment:
    """Continuous [0,1] language-alignment score for one manner dimension.

    Unlike ``DimensionResult`` (hard pass/fail gate used for L2 status /
    repair-retry decisions), this expresses *how well* the executed physical
    statistic matches what the instruction implied, as a smooth margin score
    suitable for averaging/reporting (e.g. "manner sensitivity" ablations,
    counterfactual-swap tests) instead of a single binary bit.
    """

    dim: str                 # "A"/"B"/"C"/"D" (same taxonomy as DimensionResult)
    name: str                # e.g. "joint_vel_upper", "joint_vel_upper:approach"
    phase: str                # "__all__" (episode-wide) or a named sub-phase
    score: float              # 1.0 = fully aligned, decays toward 0.0
    measured: float
    threshold: float
    direction: str            # "upper" (measured should stay <= threshold) or
                               # "lower" (measured should stay >= threshold)


@dataclass
class AlignmentReport:
    """Aggregate language-alignment score for one executed trajectory."""

    stage_id: str
    dimensions: list = field(default_factory=list)    # list[DimensionAlignment]
    phase_scores: dict = field(default_factory=dict)  # phase -> mean score
    overall_score: float | None = None                # mean over all dimensions
    label: str = ""


def _bound_score(
    measured: float, threshold: float, tolerance: float, direction: str,
) -> float:
    """Smooth [0,1] alignment score for a one-sided physical bound.

    ``direction="upper"``: instruction implies measured should stay <=
    threshold (e.g. "slowly" caps joint velocity). Score is 1.0 while
    compliant and decays exponentially as ``measured`` exceeds ``threshold``.

    ``direction="lower"``: instruction implies measured should stay >=
    threshold (e.g. "firmly" needs a minimum grip force/stiffness). Score
    decays as ``measured`` falls short of ``threshold``.

    ``tolerance`` reuses ``MannerSpec.tolerance`` semantics: a larger
    tolerance means a more forgiving falloff, keeping this continuous score
    directionally consistent with the existing binary pass/fail checks.
    """
    if threshold is None or threshold == 0:
        return 1.0
    ratio = measured / threshold
    if direction == "upper":
        shortfall = max(0.0, ratio - 1.0)
    else:
        shortfall = max(0.0, 1.0 - ratio)
    tol = max(float(tolerance), 1e-6)
    return float(np.exp(-shortfall / tol))


def compute_alignment_score(
    telemetry: "list[MannerStepSnapshot] | list[dict]",
    spec: MannerSpec,
    stage_id: str = "unknown_stage",
) -> AlignmentReport:
    """Score how closely an executed trajectory matches a manner instruction.

    Complements :meth:`MannerValidator.validate` (binary SATISFIED/NOT_
    SATISFIED gate used for repair/retry control flow) with a continuous,
    per-phase "language alignment score" in [0, 1] intended for reporting
    and ablation studies (e.g. manner sensitivity, counterfactual instruction
    swaps, paraphrase robustness) -- it does not gate any control flow.

    Reuses the same measured-statistic definitions as the corresponding
    ``MannerValidator._check_*`` methods (peak/mean joint velocity, min/max
    kp, min gripper fraction, peak jerk) so the two views stay comparable.
    """
    if not telemetry or not spec.has_any_constraint():
        return AlignmentReport(stage_id=stage_id, label=spec.label)

    snaps = [_coerce_snapshot(s) for s in telemetry]
    dims: list[DimensionAlignment] = []
    tol = spec.tolerance

    # -- A: velocity (episode-wide) --
    move_snaps = [
        s for s in snaps if _is_move(s.phase)
    ] or snaps
    vel_norms = np.array([float(np.linalg.norm(s.joint_vel)) for s in move_snaps])
    if len(vel_norms):
        peak_vel, mean_vel = float(vel_norms.max()), float(vel_norms.mean())
        if spec.max_joint_vel is not None:
            dims.append(DimensionAlignment(
                dim="A", name="joint_vel_upper", phase="__all__",
                score=_bound_score(peak_vel, spec.max_joint_vel, tol, "upper"),
                measured=peak_vel, threshold=spec.max_joint_vel, direction="upper",
            ))
        if spec.min_joint_vel is not None:
            dims.append(DimensionAlignment(
                dim="A", name="joint_vel_lower", phase="__all__",
                score=_bound_score(mean_vel, spec.min_joint_vel, tol, "lower"),
                measured=mean_vel, threshold=spec.min_joint_vel, direction="lower",
            ))

    # -- A (phase-specific): intra-trajectory manner --
    for phase_name, ov in (spec.phase_velocity or {}).items():
        sub_snaps = [s for s in snaps if s.phase == phase_name]
        if not sub_snaps:
            continue
        sub_vel = np.array([float(np.linalg.norm(s.joint_vel)) for s in sub_snaps])
        sub_peak, sub_mean = float(sub_vel.max()), float(sub_vel.mean())
        if ov.max_joint_vel is not None:
            dims.append(DimensionAlignment(
                dim="A", name="joint_vel_upper:" + phase_name, phase=phase_name,
                score=_bound_score(sub_peak, ov.max_joint_vel, tol, "upper"),
                measured=sub_peak, threshold=ov.max_joint_vel, direction="upper",
            ))
        if ov.min_joint_vel is not None:
            dims.append(DimensionAlignment(
                dim="A", name="joint_vel_lower:" + phase_name, phase=phase_name,
                score=_bound_score(sub_mean, ov.min_joint_vel, tol, "lower"),
                measured=sub_mean, threshold=ov.min_joint_vel, direction="lower",
            ))

    # -- B: stiffness (episode-wide only; no phase-specific kp today) --
    kp_vals = np.array([s.kp for s in snaps])
    if len(kp_vals):
        min_kp, max_kp = float(kp_vals.min()), float(kp_vals.max())
        if spec.kp_max is not None:
            dims.append(DimensionAlignment(
                dim="B", name="kp_upper", phase="__all__",
                score=_bound_score(min_kp, spec.kp_max, tol, "upper"),
                measured=min_kp, threshold=spec.kp_max, direction="upper",
            ))
        if spec.kp_min is not None:
            dims.append(DimensionAlignment(
                dim="B", name="kp_lower", phase="__all__",
                score=_bound_score(max_kp, spec.kp_min, tol, "lower"),
                measured=max_kp, threshold=spec.kp_min, direction="lower",
            ))

    # -- C: gripper (episode-wide only) --
    grip_snaps = [s for s in snaps if s.phase == "grip"] or snaps
    if grip_snaps:
        min_frac = min(s.gripper_fraction for s in grip_snaps)
        if spec.grip_fraction_max is not None:
            dims.append(DimensionAlignment(
                dim="C", name="grip_fraction_upper", phase="__all__",
                score=_bound_score(min_frac, spec.grip_fraction_max, tol, "upper"),
                measured=min_frac, threshold=spec.grip_fraction_max, direction="upper",
            ))
        if spec.grip_fraction_min is not None:
            dims.append(DimensionAlignment(
                dim="C", name="grip_fraction_min_closing", phase="__all__",
                score=_bound_score(min_frac, spec.grip_fraction_min, tol, "upper"),
                measured=min_frac, threshold=spec.grip_fraction_min, direction="upper",
            ))

    # -- D: jerk (episode-wide only) --
    if spec.max_jerk is not None:
        move_only = [s for s in snaps if _is_move(s.phase)] or snaps
        if len(move_only) >= 3:
            accs = np.stack([s.joint_acc for s in move_only])
            jerks = np.diff(accs, axis=0) / 0.05
            peak_jerk = float(np.linalg.norm(jerks, axis=1).max())
            dims.append(DimensionAlignment(
                dim="D", name="jerk", phase="__all__",
                score=_bound_score(peak_jerk, spec.max_jerk, tol, "upper"),
                measured=peak_jerk, threshold=spec.max_jerk, direction="upper",
            ))

    if not dims:
        return AlignmentReport(stage_id=stage_id, label=spec.label)

    phase_groups: dict = {}
    for d in dims:
        phase_groups.setdefault(d.phase, []).append(d.score)
    phase_scores = {p: float(np.mean(v)) for p, v in phase_groups.items()}
    overall_score = float(np.mean([d.score for d in dims]))

    return AlignmentReport(
        stage_id=stage_id, dimensions=dims,
        phase_scores=phase_scores, overall_score=overall_score,
        label=spec.label,
    )



# ──────────────────────────────────────────────────────────────────────────────
#  Preset Library  (adverbial → MannerSpec)
# ──────────────────────────────────────────────────────────────────────────────

MANNER_PRESETS: dict[str, MannerSpec] = {
    # ── speed adverbials ──
    "slowly":      MannerSpec(max_joint_vel=0.08, max_jerk=2.0,  label="缓缓"),
    "gently":      MannerSpec(kp_max=600.0, grip_fraction_max=0.30, label="轻轻"),
    "fast":        MannerSpec(min_joint_vel=0.15,                label="迅速"),
    "firmly":      MannerSpec(kp_min=1200.0, grip_fraction_min=0.85, label="用力"),
    "carefully":   MannerSpec(max_joint_vel=0.10, kp_max=800.0, max_jerk=3.0, label="小心"),
    "quickly":     MannerSpec(min_joint_vel=0.18,                label="快速"),
    "softly":      MannerSpec(kp_max=500.0, grip_fraction_max=0.25, label="柔和"),
    "forcefully":  MannerSpec(kp_min=1300.0, grip_fraction_min=0.90, label="有力"),

    # ── compound presets（原对齐 run_manner_cube_lift.py，该脚本已归档到 new/_archive/）──
    "slowly_gently": MannerSpec(
        max_joint_vel=0.08, max_jerk=2.0,
        kp_max=600.0, grip_fraction_max=0.30,
        label="缓缓靠近+轻轻夹持",
    ),
    "normal": MannerSpec(
        kp_max=1200.0,          # 常规刚度约束，速度不限
        label="正常操作",
    ),
}


def compose_manner_spec(*adverbials: str, tolerance: float = 0.15) -> MannerSpec:
    """Merge multiple adverbial specs by taking the tightest bound per dimension.

    Example::

        spec = compose_manner_spec("slowly", "gently")
        # → max_joint_vel=0.08, kp_max=600, grip_fraction_max=0.30, max_jerk=2.0

    Unknown adverbials are silently skipped.
    """
    specs = [MANNER_PRESETS[a] for a in adverbials if a in MANNER_PRESETS]
    if not specs:
        return MannerSpec(tolerance=tolerance, label=" ".join(adverbials))

    def _min_not_none(*vals: float | None) -> float | None:
        candidates = [v for v in vals if v is not None]
        return min(candidates) if candidates else None

    def _max_not_none(*vals: float | None) -> float | None:
        candidates = [v for v in vals if v is not None]
        return max(candidates) if candidates else None

    return MannerSpec(
        # upper bounds → take the tightest (minimum)
        max_joint_vel=_min_not_none(*[s.max_joint_vel for s in specs]),
        kp_max=_min_not_none(*[s.kp_max for s in specs]),
        grip_fraction_max=_min_not_none(*[s.grip_fraction_max for s in specs]),
        max_jerk=_min_not_none(*[s.max_jerk for s in specs]),
        # lower bounds → take the tightest (maximum)
        min_joint_vel=_max_not_none(*[s.min_joint_vel for s in specs]),
        kp_min=_max_not_none(*[s.kp_min for s in specs]),
        grip_fraction_min=_max_not_none(*[s.grip_fraction_min for s in specs]),
        tolerance=tolerance,
        label=" + ".join(s.label for s in specs if s.label),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Validator
# ──────────────────────────────────────────────────────────────────────────────

class MannerValidator:
    """Validates a telemetry buffer against a MannerSpec.

    Returns ``VerificationEvidence`` compatible with the StageCap KITE loop.
    """

    # fraction of steps allowed to violate a constraint before flagging
    VIOLATION_RATE_THRESHOLD: float = 0.10   # 10 % tolerance on step count

    def validate(
        self,
        telemetry: list[MannerStepSnapshot] | list[dict],
        spec: MannerSpec,
        stage_id: str = "unknown_stage",
    ) -> VerificationEvidence:
        """Validate telemetry buffer against spec.

        Parameters
        ----------
        telemetry:
            List of :class:`MannerStepSnapshot` (or plain dicts with same keys).
        spec:
            The :class:`MannerSpec` encoding the manner constraints.
        stage_id:
            Stage identifier for the returned evidence.

        Returns
        -------
        VerificationEvidence
            KITE-compatible evidence with detailed geometry_evidence dict.
        """
        if not telemetry:
            return VerificationEvidence(
                stage_id=stage_id,
                status=StageStatus.UNCERTAIN,
                rationale="No telemetry recorded — recorder may not have been started.",
                geometry_evidence={"spec": _spec_to_dict(spec), "n_steps": 0},
            )

        snaps = [_coerce_snapshot(s) for s in telemetry]
        results: list[DimensionResult] = []

        # ── A: velocity ──
        results.extend(self._check_velocity(snaps, spec))

        # ── B: stiffness ──
        results.extend(self._check_stiffness(snaps, spec))

        # ── C: gripper ──
        results.extend(self._check_gripper(snaps, spec))

        # ── D: jerk ──
        results.extend(self._check_jerk(snaps, spec))

        passed_dims = [r for r in results if r.passed]
        failed_dims = [r for r in results if not r.passed]

        geo = _build_geometry_evidence(snaps, spec, results)

        if not results:
            status = StageStatus.UNCERTAIN
            rationale = "MannerSpec has no constraints defined — nothing to verify."
            repair = None
        elif not failed_dims:
            status = StageStatus.SATISFIED
            rationale = _build_rationale_pass(passed_dims, spec, snaps)
            repair = None
        else:
            status = StageStatus.NOT_SATISFIED
            rationale = _build_rationale_fail(failed_dims, spec, snaps)
            repair = _build_repair_hint(failed_dims, spec)

        return VerificationEvidence(
            stage_id=stage_id,
            status=status,
            rationale=rationale,
            geometry_evidence=geo,
            log_evidence=_build_log_evidence(results),
            repair_hint=repair,
        )

    # ── Internal dimension checkers ─────────────────────────────────────────

    def _check_velocity(
        self, snaps: list[MannerStepSnapshot], spec: MannerSpec
    ) -> list[DimensionResult]:
        results = []
        move_snaps = [
            s for s in snaps if _is_move(s.phase)
        ] or snaps
        vel_norms = np.array([
            float(np.linalg.norm(s.joint_vel)) for s in move_snaps
        ])
        if len(vel_norms) == 0:
            return results

        peak_vel = float(vel_norms.max())
        mean_vel = float(vel_norms.mean())

        if spec.max_joint_vel is not None:
            limit = spec.max_joint_vel * (1.0 + spec.tolerance)
            viol_frac = float(np.mean(vel_norms > limit))
            passed = viol_frac <= self.VIOLATION_RATE_THRESHOLD
            results.append(DimensionResult(
                dim="A", name="joint_vel_upper",
                passed=passed,
                measured=peak_vel,
                threshold=spec.max_joint_vel,
                detail=(
                    f"peak={peak_vel:.4f} rad/s  limit={spec.max_joint_vel:.4f}  "
                    f"violation_rate={viol_frac:.1%}"
                    + ("  ✓" if passed else "  ✗ TOO FAST")
                ),
            ))

        if spec.min_joint_vel is not None:
            # At least some steps must reach min speed (motor activation check)
            frac_above = float(np.mean(vel_norms >= spec.min_joint_vel * 0.8))
            passed = frac_above >= 0.3  # 30% of steps must show meaningful speed
            results.append(DimensionResult(
                dim="A", name="joint_vel_lower",
                passed=passed,
                measured=mean_vel,
                threshold=spec.min_joint_vel,
                detail=(
                    f"mean={mean_vel:.4f} rad/s  min_required={spec.min_joint_vel:.4f}  "
                    f"active_frac={frac_above:.1%}"
                    + ("  ✓" if passed else "  ✗ TOO SLOW")
                ),
            ))

        # -- Phase-specific velocity checks (intra-trajectory manner) --
        # Only runs when spec.phase_velocity has overrides AND telemetry has
        # matching "move:<phase>" labels. Independent of the aggregate check
        # above so old specs/telemetry (no phase labels) are unaffected.
        for phase_name, ov in (spec.phase_velocity or {}).items():
            if ov.max_joint_vel is None and ov.min_joint_vel is None:
                continue
            sub_snaps = [s for s in snaps if s.phase == phase_name]
            if not sub_snaps:
                continue
            sub_vel = np.array([
                float(np.linalg.norm(s.joint_vel)) for s in sub_snaps
            ])
            sub_peak = float(sub_vel.max())
            sub_mean = float(sub_vel.mean())

            if ov.max_joint_vel is not None:
                limit = ov.max_joint_vel * (1.0 + spec.tolerance)
                viol_frac = float(np.mean(sub_vel > limit))
                passed = viol_frac <= self.VIOLATION_RATE_THRESHOLD
                results.append(DimensionResult(
                    dim="A", name="joint_vel_upper:" + phase_name,
                    passed=passed,
                    measured=sub_peak,
                    threshold=ov.max_joint_vel,
                    detail=(
                        "[" + phase_name + "] peak=%.4f rad/s  limit=%.4f  violation_rate=%.1f%%"
                        % (sub_peak, ov.max_joint_vel, viol_frac * 100)
                        + ("  OK" if passed else "  TOO FAST")
                    ),
                ))

            if ov.min_joint_vel is not None:
                frac_above = float(np.mean(sub_vel >= ov.min_joint_vel * 0.8))
                passed = frac_above >= 0.3
                results.append(DimensionResult(
                    dim="A", name="joint_vel_lower:" + phase_name,
                    passed=passed,
                    measured=sub_mean,
                    threshold=ov.min_joint_vel,
                    detail=(
                        "[" + phase_name + "] mean=%.4f rad/s  min_required=%.4f  active_frac=%.1f%%"
                        % (sub_mean, ov.min_joint_vel, frac_above * 100)
                        + ("  OK" if passed else "  TOO SLOW")
                    ),
                ))

        return results

    def _check_stiffness(
        self, snaps: list[MannerStepSnapshot], spec: MannerSpec
    ) -> list[DimensionResult]:
        results = []
        kp_vals = np.array([s.kp for s in snaps])
        if len(kp_vals) == 0:
            return results

        # Use the minimum kp actually set (most compliant moment)
        min_kp = float(kp_vals.min())
        max_kp = float(kp_vals.max())
        mean_kp = float(kp_vals.mean())

        if spec.kp_max is not None:
            # After the set_stiffness call, the kp should drop below kp_max
            passed = min_kp <= spec.kp_max * (1.0 + spec.tolerance)
            results.append(DimensionResult(
                dim="B", name="kp_upper",
                passed=passed,
                measured=min_kp,
                threshold=spec.kp_max,
                detail=(
                    f"min_kp={min_kp:.0f}  limit={spec.kp_max:.0f}"
                    + ("  ✓" if passed else f"  ✗ STIFFNESS TOO HIGH (min={min_kp:.0f})")
                ),
            ))

        if spec.kp_min is not None:
            passed = max_kp >= spec.kp_min * (1.0 - spec.tolerance)
            results.append(DimensionResult(
                dim="B", name="kp_lower",
                passed=passed,
                measured=max_kp,
                threshold=spec.kp_min,
                detail=(
                    f"max_kp={max_kp:.0f}  required≥{spec.kp_min:.0f}"
                    + ("  ✓" if passed else "  ✗ STIFFNESS TOO LOW")
                ),
            ))

        return results

    def _check_gripper(
        self, snaps: list[MannerStepSnapshot], spec: MannerSpec
    ) -> list[DimensionResult]:
        results = []
        grip_snaps = [s for s in snaps if s.phase == "grip"] or snaps
        if not grip_snaps:
            return results

        # Final gripper fraction at end of grip phase
        final_frac = grip_snaps[-1].gripper_fraction
        min_frac = min(s.gripper_fraction for s in grip_snaps)

        if spec.grip_fraction_max is not None:
            # grip_fraction_max = 1 - force_min
            # e.g. gently → force ≤ 0.30 → fraction ≥ 0.70
            # We check that the gripper DID close (fraction < 1.0) but NOT too far
            passed = min_frac <= spec.grip_fraction_max * (1.0 + spec.tolerance)
            results.append(DimensionResult(
                dim="C", name="grip_fraction_upper",
                passed=passed,
                measured=min_frac,
                threshold=spec.grip_fraction_max,
                detail=(
                    f"min_fraction={min_frac:.3f}  limit={spec.grip_fraction_max:.3f}"
                    + ("  ✓" if passed else "  ✗ GRIP FORCE TOO HIGH (closed too far)")
                ),
            ))

        if spec.grip_fraction_min is not None:
            # Must close at least this much (firmly)
            passed = min_frac <= spec.grip_fraction_min * (1.0 + spec.tolerance)
            results.append(DimensionResult(
                dim="C", name="grip_fraction_min_closing",
                passed=passed,
                measured=min_frac,
                threshold=spec.grip_fraction_min,
                detail=(
                    f"min_fraction={min_frac:.3f}  must_close_to≤{spec.grip_fraction_min:.3f}"
                    + ("  ✓" if passed else "  ✗ GRIP NOT FIRM ENOUGH")
                ),
            ))

        return results

    def _check_jerk(
        self, snaps: list[MannerStepSnapshot], spec: MannerSpec
    ) -> list[DimensionResult]:
        if spec.max_jerk is None:
            return []
        move_snaps = [s for s in snaps if _is_move(s.phase)] or snaps
        if len(move_snaps) < 3:
            return []

        accs = np.stack([s.joint_acc for s in move_snaps])  # (N, 7)
        dt = 0.05  # 20 Hz
        jerks = np.diff(accs, axis=0) / dt           # (N-1, 7)
        jerk_norms = np.linalg.norm(jerks, axis=1)   # (N-1,)

        peak_jerk = float(jerk_norms.max())
        p95_jerk = float(np.percentile(jerk_norms, 95))
        limit = spec.max_jerk * (1.0 + spec.tolerance)

        viol_frac = float(np.mean(jerk_norms > limit))
        passed = viol_frac <= self.VIOLATION_RATE_THRESHOLD

        return [DimensionResult(
            dim="D", name="jerk",
            passed=passed,
            measured=peak_jerk,
            threshold=spec.max_jerk,
            detail=(
                f"peak_jerk={peak_jerk:.3f} rad/s³  p95={p95_jerk:.3f}  limit={spec.max_jerk:.3f}  "
                f"violation_rate={viol_frac:.1%}"
                + ("  ✓" if passed else "  ✗ JERK TOO HIGH")
            ),
        )]


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience function (wraps validator + recording lifecycle)
# ──────────────────────────────────────────────────────────────────────────────

def validate_manner_stage(
    env: Any,
    spec: MannerSpec,
    stage_id: str,
    callable_fn,
    *args,
    **kwargs,
) -> tuple[Any, VerificationEvidence]:
    """Execute callable_fn and validate manner constraints.

    Parameters
    ----------
    env:
        A ``RobosuiteBaseEnv`` instance that supports
        ``begin_manner_recording`` / ``end_manner_recording`` /
        ``get_manner_telemetry``.
    spec:
        The manner constraints to enforce.
    stage_id:
        Stage name for evidence attribution.
    callable_fn:
        The function to execute (e.g. a lambda wrapping several API calls).
    *args, **kwargs:
        Forwarded to callable_fn.

    Returns
    -------
    (result, evidence)
        The return value of callable_fn and the VerificationEvidence.

    Example::

        result, ev = validate_manner_stage(
            env, spec, "descend_and_grasp",
            lambda: (
                api.goto_pose_with_profile(grasp_pos, quat, v_max=0.03),
                api.close_gripper_partial(force=0.75),
            )
        )
    """
    env.begin_manner_recording(clear=True)
    try:
        result = callable_fn(*args, **kwargs)
    finally:
        env.end_manner_recording()

    telemetry = env.get_manner_telemetry()
    evidence = MannerValidator().validate(telemetry, spec, stage_id=stage_id)
    return result, evidence


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _coerce_snapshot(s: MannerStepSnapshot | dict) -> MannerStepSnapshot:
    if isinstance(s, MannerStepSnapshot):
        return s
    return MannerStepSnapshot(
        t=float(s.get("t", 0.0)),
        phase=str(s.get("phase", "move")),
        joint_vel=np.asarray(s.get("joint_vel", np.zeros(7))),
        joint_acc=np.asarray(s.get("joint_acc", np.zeros(7))),
        kp=float(s.get("kp", 1500.0)),
        gripper_fraction=float(s.get("gripper_fraction", 1.0)),
        gripper_qpos=np.asarray(s.get("gripper_qpos", np.zeros(2))),
        cfrc_norm=float(s.get("cfrc_norm", 0.0)),
        joint_pos=np.asarray(s.get("joint_pos", np.zeros(7))),
        eef_pos=np.asarray(s.get("eef_pos", np.zeros(3))),
        eef_quat=np.asarray(s.get("eef_quat", np.zeros(4))),
    )


def _spec_to_dict(spec: MannerSpec) -> dict:
    return {
        "max_joint_vel": spec.max_joint_vel,
        "min_joint_vel": spec.min_joint_vel,
        "kp_max": spec.kp_max,
        "kp_min": spec.kp_min,
        "grip_fraction_max": spec.grip_fraction_max,
        "grip_fraction_min": spec.grip_fraction_min,
        "max_jerk": spec.max_jerk,
        "tolerance": spec.tolerance,
        "label": spec.label,
    }


def _build_geometry_evidence(
    snaps: list[MannerStepSnapshot],
    spec: MannerSpec,
    results: list[DimensionResult],
) -> dict[str, Any]:
    vel_norms = np.array([np.linalg.norm(s.joint_vel) for s in snaps])
    accs = np.stack([s.joint_acc for s in snaps])
    kp_vals = np.array([s.kp for s in snaps])
    frac_vals = np.array([s.gripper_fraction for s in snaps])

    geo: dict[str, Any] = {
        "n_steps": len(snaps),
        "duration_s": round(len(snaps) * 0.05, 3),
        "spec": _spec_to_dict(spec),
    }

    geo["velocity"] = {
        "peak_rad_s": round(float(vel_norms.max()), 5) if len(vel_norms) else 0.0,
        "mean_rad_s": round(float(vel_norms.mean()), 5) if len(vel_norms) else 0.0,
        "p95_rad_s": round(float(np.percentile(vel_norms, 95)), 5) if len(vel_norms) else 0.0,
    }
    geo["stiffness"] = {
        "min_kp": round(float(kp_vals.min()), 1) if len(kp_vals) else 1500.0,
        "max_kp": round(float(kp_vals.max()), 1) if len(kp_vals) else 1500.0,
        "mean_kp": round(float(kp_vals.mean()), 1) if len(kp_vals) else 1500.0,
    }
    geo["gripper"] = {
        "min_fraction": round(float(frac_vals.min()), 4) if len(frac_vals) else 1.0,
        "final_fraction": round(float(frac_vals[-1]), 4) if len(frac_vals) else 1.0,
    }

    if len(snaps) >= 3:
        jerks = np.diff(accs, axis=0) / 0.05
        jerk_norms = np.linalg.norm(jerks, axis=1)
        geo["jerk"] = {
            "peak_rad_s3": round(float(jerk_norms.max()), 4),
            "p95_rad_s3": round(float(np.percentile(jerk_norms, 95)), 4),
        }

    geo["dimension_results"] = [
        {
            "dim": r.dim, "name": r.name, "passed": r.passed,
            "measured": round(r.measured, 5),
            "threshold": r.threshold,
            "detail": r.detail,
        }
        for r in results
    ]
    return geo


def _build_log_evidence(results: list[DimensionResult]) -> list[str]:
    lines = []
    for r in results:
        prefix = "✓" if r.passed else "✗"
        lines.append(f"[Dim {r.dim}:{r.name}] {prefix} {r.detail}")
    return lines


def _build_rationale_pass(
    passed: list[DimensionResult], spec: MannerSpec, snaps: list[MannerStepSnapshot]
) -> str:
    parts = [f"Manner '{spec.label}' SATISFIED ({len(snaps)} steps):"]
    for r in passed:
        parts.append(f"  • {r.detail}")
    return "\n".join(parts)


def _build_rationale_fail(
    failed: list[DimensionResult], spec: MannerSpec, snaps: list[MannerStepSnapshot]
) -> str:
    parts = [
        f"Manner '{spec.label}' VIOLATED — "
        f"{len(failed)} dimension(s) out of spec ({len(snaps)} steps recorded):"
    ]
    for r in failed:
        parts.append(f"  ✗ [Dim {r.dim}] {r.detail}")
    return "\n".join(parts)


def _build_repair_hint(failed: list[DimensionResult], spec: MannerSpec) -> str:
    hints = []
    for r in failed:
        if r.dim == "A" and r.name == "joint_vel_upper":
            hints.append(
                f"Reduce v_max in goto_pose_with_profile to ≤{spec.max_joint_vel:.3f} rad/s."
            )
        elif r.dim == "A" and r.name == "joint_vel_lower":
            hints.append(
                f"Increase v_max in goto_pose_with_profile to ≥{spec.min_joint_vel:.3f} rad/s."
            )
        elif r.dim == "B" and r.name == "kp_upper":
            hints.append(
                f"Call set_impedance_stiffness(kp_pos≤{spec.kp_max:.0f}) before motion."
            )
        elif r.dim == "B" and r.name == "kp_lower":
            hints.append(
                f"Call set_impedance_stiffness(kp_pos≥{spec.kp_min:.0f}) before motion."
            )
        elif r.dim == "C":
            hints.append(
                f"Adjust close_gripper_partial(force=...) to achieve target fraction."
            )
        elif r.dim == "D":
            hints.append(
                f"Add jerk_limit≤{spec.max_jerk:.2f} to goto_pose_with_profile calls."
            )
    return " | ".join(hints) if hints else "Review manner parameters."
