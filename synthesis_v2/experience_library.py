"""Pure-numpy experience library: retrieval-augmented physical calibration.

No capx / robosuite / simulator dependency. Stores two kinds of records as
flat JSONL:

  * "experience" records: a (task, phase, dimension, level) -> value sample
    drawn from a trial that reached ``outcome == "both_pass"``. ``level`` is
    the manner tag string (fast/gentle/firm/...), used as a proxy for a
    qualitative "level" until the LLM-side IR emits levels directly (see
    repo memory: Phase 6, deferred).
  * "repair" records: a (task, phase, cause) -> (dimension, before, after)
    sample drawn from a MANNER_REPAIR_LIBRARY-triggered retry that flipped
    the trial to ``outcome_after == "both_pass"``.

``ExperienceLibrary.calibrate()`` / ``calibrate_repair_ratio()`` are the two
retrieval entry points: given a lookup key, they pull matching historical
values and return a quantile (default: median) once at least ``min_samples``
matching records exist; otherwise they return the caller-supplied fallback
unchanged (cold-start = today's hardcoded preset behaviour, no regression).

This module never runs a simulator and never calls an LLM; it is a pure
storage + statistics layer that other code (run_manner_ablation.py,
population scripts) reads from / writes to.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA = "mannercap-experience-library-v1"

GLOBAL_PHASE = "_global"  # phase bucket for flat (non-per-phase) dimensions


@dataclass
class ExperienceRecord:
    task: str
    phase: str
    dimension: str
    level: str
    value: float
    outcome: str
    source_run_stamp: str = ""
    seed: int = -1
    ts: str = ""
    kind: str = field(default="experience", init=False)


@dataclass
class RepairRecord:
    task: str
    phase: str
    cause: str
    dimension: str
    before_value: float
    after_value: float
    outcome_after: str
    source_run_stamp: str = ""
    ts: str = ""
    kind: str = field(default="repair", init=False)


def _to_dict(rec: ExperienceRecord | RepairRecord) -> dict[str, Any]:
    # rec.__dict__ misses `kind` because it's an init=False field with only a
    # class-level default (dataclasses never assigns it into the instance
    # __dict__); dataclasses.asdict() resolves fields via getattr() instead,
    # so it correctly picks up the class-level default too.
    return dataclasses.asdict(rec)


class ExperienceLibrary:
    """In-memory experience library with JSONL load/save/append."""

    def __init__(self) -> None:
        self.experiences: list[ExperienceRecord] = []
        self.repairs: list[RepairRecord] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_experience(
        self,
        task: str,
        phase: str,
        dimension: str,
        level: str,
        value: float,
        outcome: str,
        source_run_stamp: str = "",
        seed: int = -1,
        ts: str = "",
    ) -> None:
        self.experiences.append(
            ExperienceRecord(
                task=task, phase=phase or GLOBAL_PHASE, dimension=dimension,
                level=level, value=float(value), outcome=outcome,
                source_run_stamp=source_run_stamp, seed=seed, ts=ts,
            )
        )

    def add_repair(
        self,
        task: str,
        phase: str,
        cause: str,
        dimension: str,
        before_value: float,
        after_value: float,
        outcome_after: str,
        source_run_stamp: str = "",
        ts: str = "",
    ) -> None:
        self.repairs.append(
            RepairRecord(
                task=task, phase=phase or GLOBAL_PHASE, cause=cause,
                dimension=dimension, before_value=float(before_value),
                after_value=float(after_value), outcome_after=outcome_after,
                source_run_stamp=source_run_stamp, ts=ts,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query_experience(
        self, task: str, phase: str, dimension: str, level: str,
    ) -> list[float]:
        phase = phase or GLOBAL_PHASE
        return [
            r.value for r in self.experiences
            if r.task == task and r.phase == phase
            and r.dimension == dimension and r.level == level
            and r.outcome == "both_pass"
        ]

    def query_repair_ratio(
        self, task: str, phase: str, cause: str, dimension: str,
    ) -> list[float]:
        """Return historical after/before ratios for a (task, phase, cause,
        dimension) repair, restricted to retries that reached both_pass."""
        phase = phase or GLOBAL_PHASE
        out = []
        for r in self.repairs:
            if r.task == task and r.phase == phase and r.cause == cause \
                    and r.dimension == dimension and r.outcome_after == "both_pass" \
                    and r.before_value:
                out.append(r.after_value / r.before_value)
        return out

    # ------------------------------------------------------------------
    # Calibration (the actual "校准层")
    # ------------------------------------------------------------------
    def calibrate(
        self,
        task: str,
        phase: str,
        dimension: str,
        level: str,
        fallback: float,
        min_samples: int = 3,
        quantile: float = 0.5,
    ) -> tuple[float, bool]:
        """Return (value, used_library). used_library=False means the
        library had fewer than min_samples matching records and the
        caller-supplied fallback (today's hardcoded preset value) was
        returned unchanged -- this is the cold-start / no-regression path."""
        samples = self.query_experience(task, phase, dimension, level)
        if len(samples) < min_samples:
            return fallback, False
        return float(np.percentile(samples, quantile * 100.0)), True

    def calibrate_repair_ratio(
        self,
        task: str,
        phase: str,
        cause: str,
        dimension: str,
        fallback_ratio: float,
        min_samples: int = 3,
        quantile: float = 0.5,
    ) -> tuple[float, bool]:
        samples = self.query_repair_ratio(task, phase, cause, dimension)
        if len(samples) < min_samples:
            return fallback_ratio, False
        return float(np.percentile(samples, quantile * 100.0)), True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.experiences:
                f.write(json.dumps(_to_dict(rec), ensure_ascii=False) + "\n")
            for rec in self.repairs:
                f.write(json.dumps(_to_dict(rec), ensure_ascii=False) + "\n")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                kind = d.pop("kind", "experience")
                if kind == "repair":
                    self.repairs.append(RepairRecord(**d))
                else:
                    d.pop("kind", None)
                    self.experiences.append(ExperienceRecord(**d))

    def append_record(self, path: str, rec: ExperienceRecord | RepairRecord) -> None:
        """Append a single record to the JSONL file without rewriting it
        (used for live write-back during a running batch, see Phase 5)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_dict(rec), ensure_ascii=False) + "\n")


def flatten_trial_params(params: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Flatten a TrialResult.params dict into {dimension: {phase: value}}
    entries suitable for feeding into ExperienceLibrary.add_experience() in
    a loop. Handles both the per-phase dicts (phase_v_max/phase_a_max) and
    the flat episode-level scalars (v_max/a_max/grip/kp_pos/jerk_limit)."""
    out: dict[str, dict[str, float]] = {}
    phase_map = {"phase_v_max": "v_max", "phase_a_max": "a_max"}
    for src_key, dimension in phase_map.items():
        phases = params.get(src_key) or {}
        for phase, value in phases.items():
            if value is None:
                continue
            out.setdefault(dimension, {})[phase] = float(value)
    for dimension in ("grip", "kp_pos", "jerk_limit"):
        value = params.get(dimension)
        if value is not None:
            out.setdefault(dimension, {})[GLOBAL_PHASE] = float(value)
    return out
