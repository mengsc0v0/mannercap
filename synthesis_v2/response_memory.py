"""P1-8 measured-response experience memory.

Every audited attempt contributes, per (stage, dimension), the command that was issued, the
measured response (min / median / max of the audited statistic), the audit band and the audit
decision.  Retrieval predicts the command from the measured response (response ~ k * command for
speed, response ~ a + b * command for grip) instead of replaying the median of past commands.

Legacy ``ExperienceLibrary`` records carry no measured response and no audit identity, so they are
never read here (they cannot be used as positives).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from synthesis_v2.manner_request import LEVEL_RANGES, default_candidate
from synthesis_v2.production_pipeline import RuleProposer

LIBRARY_VERSION = "response_memory.v1"
SCHEMA = "mannercap-response-memory-v1"
SLOW_MARGIN = 0.85          # aim at 85% of the slow upper bound
FAST_MARGIN = 1.20          # aim at 120% of the fast lower bound
MIN_LIBRARY_SAMPLES = 3
SLOPE_RANGE = (0.3, 3.0)
SLOW_QUANTILE, FAST_QUANTILE = 75.0, 25.0   # conservative response ratio from the library


@dataclass(frozen=True)
class Observation:
    task: str
    stage: str
    dimension: str            # "speed" | "grip"
    level: str
    command: float
    stat_min: float
    stat_median: float
    stat_max: float
    lower: Optional[float]
    upper: Optional[float]
    decision: str             # "pass" | "fail" | "unknown"
    request_id: str
    attempt_id: str
    seed: int
    calibration_sha256: str
    audit_version: str
    library_version: str = LIBRARY_VERSION

    @property
    def valid(self) -> bool:
        vals = (self.command, self.stat_min, self.stat_median, self.stat_max)
        return (self.decision in ("pass", "fail") and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
                and self.command > 0)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


def observations_from_audit(audit: Mapping[str, Any], candidate: Mapping[str, Any], *, task: str,
                            request_id: str, attempt_id: str, seed: int) -> list[Observation]:
    out: list[Observation] = []
    for c in audit.get("checks", []):
        dim = c.get("dimension")
        if dim not in ("speed", "grip") or c.get("median") is None:
            continue
        if dim == "speed":
            cmd = next((e.get("v_max_mps") for e in candidate.get("stages", []) if e.get("stage_id") == c["stage"]), None)
        else:
            cmd = (candidate.get("grip") or {}).get("target_lo_n")
        if cmd is None:
            continue
        out.append(Observation(
            task=task, stage=c["stage"], dimension=dim, level=c["level"], command=float(cmd),
            stat_min=float(c["min"]), stat_median=float(c["median"]), stat_max=float(c["max"]),
            lower=c.get("lower"), upper=c.get("upper"), decision=str(c["decision"]),
            request_id=request_id, attempt_id=attempt_id, seed=int(seed),
            calibration_sha256=str(audit.get("calibration_sha256")), audit_version=str(audit.get("audit_version"))))
    return out


class ResponseMemory:
    def __init__(self, calibration_sha256: Optional[str] = None) -> None:
        self.calibration_sha256 = calibration_sha256
        self.obs: list[Observation] = []
        self.frozen = False

    def add(self, o: Observation) -> None:
        if self.frozen:
            raise RuntimeError("response memory is frozen")
        self.obs.append(o)

    def freeze(self) -> "ResponseMemory":
        self.frozen = True
        return self

    def query(self, task: str, dimension: str, level: Optional[str] = None, stage: Optional[str] = None,
              exclude_request: Optional[str] = None) -> list[Observation]:
        return [o for o in self.obs
                if o.valid and o.task == task and o.dimension == dimension
                and (level is None or o.level == level) and (stage is None or o.stage == stage)
                and o.request_id != exclude_request
                and (self.calibration_sha256 is None or o.calibration_sha256 == self.calibration_sha256)]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        head = {"schema": SCHEMA, "library_version": LIBRARY_VERSION, "calibration_sha256": self.calibration_sha256}
        p.write_text("\n".join([json.dumps(head)] + [json.dumps(o.to_dict(), sort_keys=True) for o in self.obs]) + "\n",
                     encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ResponseMemory":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        if head.get("schema") != SCHEMA:
            raise ValueError(f"not a response memory: {head.get('schema')!r}")
        mem = cls(head.get("calibration_sha256"))
        fields = Observation.__dataclass_fields__
        for ln in lines[1:]:
            d = json.loads(ln)
            mem.obs.append(Observation(**{k: v for k, v in d.items() if k in fields}))
        return mem

    @classmethod
    def from_run(cls, run_dir: str | Path, calibration_sha256: Optional[str] = None) -> "ResponseMemory":
        """Rebuild observations from a production run (attempts.jsonl + raw/<dir>/audit.json)."""
        run = Path(run_dir)
        mem = cls(calibration_sha256)
        for ln in (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            a = json.loads(ln)
            raw = (a.get("metadata") or {}).get("raw_dir")
            if not raw or not (run / raw / "audit.json").exists():
                continue
            audit = json.loads((run / raw / "audit.json").read_text(encoding="utf-8"))
            meta = json.loads((run / raw / "episode_meta.json").read_text(encoding="utf-8"))
            for o in observations_from_audit(audit, a["candidate_plan"], task=meta["task"], request_id=a["request_id"],
                                             attempt_id=a["attempt_id"], seed=meta["seed"]):
                if calibration_sha256 is None or o.calibration_sha256 == calibration_sha256:
                    mem.obs.append(o)
        return mem


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _stat(o: Observation) -> float:
    return o.stat_max if o.level == "slow" else o.stat_median


def speed_command(level: str, lib: Sequence[Observation], local: Sequence[Observation]) -> tuple[float, dict[str, Any]]:
    lo_r, hi_r, default = LEVEL_RANGES["speed"][level]
    if level not in ("slow", "fast"):
        return default, {"source": "default"}
    if local:
        k, src, ref = _stat(local[-1]) / local[-1].command, "request", local[-1]
    elif len(lib) >= MIN_LIBRARY_SAMPLES:
        ratios = [_stat(o) / o.command for o in lib]
        k = float(np.percentile(ratios, SLOW_QUANTILE if level == "slow" else FAST_QUANTILE))
        src, ref = "library", lib[-1]
    else:
        return default, {"source": "default"}
    bound = ref.upper if level == "slow" else ref.lower
    if bound is None or not k > 0:
        return default, {"source": "default"}
    raw = (bound * SLOW_MARGIN if level == "slow" else bound * FAST_MARGIN) / k
    feasible = raw >= lo_r if level == "slow" else raw <= hi_r
    return _clip(raw, lo_r, hi_r), {"source": src, "k": k, "raw": raw, "feasible": bool(feasible)}


def _fit(lib_all: Sequence[Observation]) -> tuple[float, float]:
    cmd = np.array([o.command for o in lib_all])
    mid = np.array([(o.stat_min + o.stat_max) / 2 for o in lib_all])
    if len(set(np.round(cmd, 6))) >= 3:
        b, a = np.polyfit(cmd, mid, 1)
        return float(a), _clip(float(b), *SLOPE_RANGE)
    return float(np.median(mid - cmd)), 1.0


def grip_command(level: str, lib_level: Sequence[Observation], lib_all: Sequence[Observation],
                 local: Sequence[Observation]) -> tuple[float, dict[str, Any]]:
    lo_r, hi_r, default = LEVEL_RANGES["grip"][level]
    ref = local[-1] if local else (lib_level[-1] if len(lib_level) >= MIN_LIBRARY_SAMPLES else None)
    if ref is None or ref.lower is None or ref.upper is None:
        return default, {"source": "default"}
    centre = (ref.lower + ref.upper) / 2
    if local:
        o = local[-1]
        slope = 1.0
        if len(local) >= 2 and abs(local[-1].command - local[-2].command) > 1e-6:
            slope = _clip((local[-1].stat_median - local[-2].stat_median) / (local[-1].command - local[-2].command),
                          *SLOPE_RANGE)
        raw = o.command + (centre - (o.stat_min + o.stat_max) / 2) / slope
        return _clip(raw, lo_r, hi_r), {"source": "request", "slope": slope, "raw": raw}
    a, b = _fit(lib_all)
    raw = (centre - a) / b
    return _clip(raw, lo_r, hi_r), {"source": "library", "offset": a, "slope": b, "raw": raw}


class MeasuredResponseProposer:
    """First attempt and repair both come from the measured-response model."""

    name = "measured_response.v1"

    def __init__(self, memory: ResponseMemory, online: bool = False) -> None:
        self.memory, self.online = memory, online
        self.last_info: dict[str, Any] = {}

    def observe(self, rid: str, attempt_id: str, spec, candidate, audit) -> None:
        if self.online and not self.memory.frozen:
            for o in observations_from_audit(audit, candidate, task=spec.task, request_id=rid,
                                             attempt_id=attempt_id, seed=spec.seed):
                self.memory.add(o)

    def propose(self, spec, parse, history):
        rid = spec.draft().request_id
        cand = default_candidate(spec.task, parse.intents)
        local: list[Observation] = []
        last = history[-1] if history and history[-1].get("candidate") and history[-1].get("audit") else None
        for i, h in enumerate(history):
            if h.get("candidate") and h.get("audit"):
                local += observations_from_audit(h["audit"], h["candidate"], task=spec.task, request_id=rid,
                                                 attempt_id=f"{rid}:h{i}", seed=spec.seed)
        passed = {(c["stage"], c["dimension"]) for c in (last["audit"]["checks"] if last else []) if c.get("decision") == "pass"}
        info: dict[str, Any] = {}
        for e in cand["stages"]:
            sid, level = e["stage_id"], e["level"]
            if last and (sid, "speed") in passed:
                e["v_max_mps"] = next(x["v_max_mps"] for x in last["candidate"]["stages"] if x["stage_id"] == sid)
                continue
            lib = self.memory.query(spec.task, "speed", level, sid, exclude_request=rid)
            loc = [o for o in local if o.stage == sid and o.level == level and o.dimension == "speed" and o.valid]
            e["v_max_mps"], info[sid] = speed_command(level, lib, loc)
        glevel = cand["grip"]["level"]
        if last and ("grasp", "grip") in passed:
            cand["grip"]["target_lo_n"] = last["candidate"]["grip"]["target_lo_n"]
        else:
            lib_l = self.memory.query(spec.task, "grip", glevel, exclude_request=rid)
            lib_a = self.memory.query(spec.task, "grip", None, exclude_request=rid)
            loc = [o for o in local if o.dimension == "grip" and o.level == glevel and o.valid]
            cand["grip"]["target_lo_n"], info["grip"] = grip_command(glevel, lib_l, lib_a, loc)
        self.last_info = info
        return cand, {}


class CommandMedianProposer(RuleProposer):
    """Baseline: median of previously passing commands (legacy ExperienceLibrary semantics), blind repair."""

    name = "command_median.v1"

    def __init__(self, memory: ResponseMemory) -> None:
        from synthesis_v2.experience_library import ExperienceLibrary
        self.lib = ExperienceLibrary()
        for o in memory.obs:
            if o.valid:
                self.lib.add_experience(o.task, o.stage, o.dimension, o.level, o.command,
                                        "both_pass" if o.decision == "pass" else "fail", seed=o.seed)

    def propose(self, spec, parse, history):
        if history:
            return super().propose(spec, parse, history)
        cand = default_candidate(spec.task, parse.intents)
        for e in cand["stages"]:
            e["v_max_mps"], _ = self.lib.calibrate(spec.task, e["stage_id"], "speed", e["level"], e["v_max_mps"])
        g = cand["grip"]
        g["target_lo_n"], _ = self.lib.calibrate(spec.task, "grasp", "grip", g["level"], g["target_lo_n"])
        return cand, {}
