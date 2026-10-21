"""Request -> proposer -> semantic gate -> execution -> independent audit -> ledger.

For every request the original language is registered in the attempt ledger
*before* any candidate is tried.  Each candidate (parse failure, gate
rejection, duplicate, execution failure, audit failure or acceptance) consumes
one attempt and is recorded with its resource vector.  The raw trajectory of
every executed attempt is written to disk *before* it is audited and is never
modified afterwards (images are stored only for the accepted attempt; the
other attempts are re-renderable because actions + seed + plan are stored and
the producer is deterministic).

Repair is a bounded retry of a *new candidate* that changes only the failing
dimension inside the pre-registered level range.  Rollback is a full re-run
from the same reset seed; stage-level simulator snapshots are not implemented,
so rollback cost = the whole episode and is charged to the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from synthesis_v2.attempt_ledger import AttemptLedger, AttemptOutcome, ResourceUsage, attempt_id_for, write_attempt_ledger
from synthesis_v2.candidate_plan import plan_from_candidate
from synthesis_v2.episode_audit import AUDIT_VERSION, audit_episode
from synthesis_v2.manner_calibration import assert_seed_isolated
from synthesis_v2.manner_request import (
    LEVEL_RANGES, grip_range, MannerIntent, ParseResult, parse_manner_request, default_candidate, semantic_gate, semantics_hash,
    speed_range,
)
from synthesis_v2.native_osc_env import CONTROLLER_CONFIG_SHA256
from synthesis_v2.native_producer import PRODUCER_VERSION, RawEpisode, first_diff_block, produce_episode
from synthesis_v2.request_manifest import RequestDraft
from synthesis_v2.stage_table import dimension_stages, stage_table_hash

SUBSTEPS_PER_STEP = 25
TARGET_OBJECT = {"lift": "cube", "stack": "cubeA", "wipe": "dirt", "nut_assembly": "square_nut"}


@dataclass(frozen=True)
class RequestSpec:
    task: str
    seed: int
    language: str
    split: str = "development"
    base_instruction: str = ""
    pair_key: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Registered label.  When given, it IS the request (the language is a paraphrase of it that the lexicon parser
    # need not understand); when None the language is parsed by the lexicon.
    intents: Optional[tuple[MannerIntent, ...]] = None
    # Required hold force of this scene relative to the nominal object (registered with the request; 1.0 = nominal).
    grip_scale: float = 1.0

    def parse(self) -> ParseResult:
        if self.intents is None:
            return parse_manner_request(self.task, self.language)
        its = tuple(sorted(self.intents))
        return ParseResult("ok" if its else "neutral", its, ())

    def draft(self) -> RequestDraft:
        parse = self.parse()
        return RequestDraft(
            task=self.task,
            scene_id=f"{self.task}:seed{self.seed}",
            initial_state={"seed": int(self.seed)},
            target_object=TARGET_OBJECT[self.task],
            physical_config={"controller_sha256": CONTROLLER_CONFIG_SHA256, "producer": PRODUCER_VERSION},
            raw_language=self.language,
            requested_constraints={"intents": [i.to_dict() for i in parse.intents], "parse_status": parse.status},
            stage_scope=sorted({i.stage for i in parse.intents}),
            allowed_control_channels=("native_7d_osc",),
            budget={},
            split=self.split,
            geometry_key=f"{self.task}:seed{self.seed}",
            manner_key=",".join(f"{i.stage}.{i.dimension}={i.level}" for i in parse.intents) or "neutral",
            metadata={"base_instruction": self.base_instruction, "pair_key": self.pair_key,
                      "label_source": "registered_label" if self.intents is not None else "lexicon_parse",
                      **({"grip_scale": float(self.grip_scale)} if self.grip_scale != 1.0 else {}),
                      **dict(self.extra)},
        )


class Proposer(Protocol):
    name: str

    def propose(self, spec: RequestSpec, parse: ParseResult, history: Sequence[Mapping[str, Any]]) -> tuple[Any, dict]:
        """Return (candidate dict | raw text | None, call_record)."""


GRIP_TASK_FAIL_FACTOR = 2.0


def _check_command(cand: Mapping[str, Any], stage: str, dimension: str) -> Optional[float]:
    """The scalar command this audit check's dimension currently maps to in ``cand``."""
    if dimension == "press":
        press = cand.get("press")
        v = press.get("target_n") if isinstance(press, Mapping) else None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if dimension == "grip":
        grip = cand.get("grip")
        v = grip.get("target_lo_n") if isinstance(grip, Mapping) else None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if dimension == "speed":
        stages = cand.get("stages")
        if isinstance(stages, list):
            for e in stages:
                if isinstance(e, Mapping) and e.get("stage_id") == stage:
                    v = e.get("v_max_mps")
                    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return None


def _stalled_checks(history: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    """(stage, dimension, level) keys whose last two consecutive failing attempts moved the
    command but the *realised* value (the check's ``median``) did not move toward satisfying
    the bound -- i.e. the measured response is non-monotone (or unresponsive) in that command
    region.  Repair must not keep pushing a dimension the data already shows cannot be fixed
    this way: that would silently burn attempt budget without any real chance of success."""
    if len(history) < 2:
        return set()
    prev_cand, cur_cand = history[-2].get("candidate"), history[-1].get("candidate")
    if not isinstance(prev_cand, Mapping) or not isinstance(cur_cand, Mapping):
        return set()
    prev_checks = {(c.get("stage"), c.get("dimension"), c.get("level")): c
                   for c in (history[-2].get("audit") or {}).get("checks", [])}
    frozen: set[tuple[str, str, str]] = set()
    for c in (history[-1].get("audit") or {}).get("checks", []):
        if c.get("decision") != "fail":
            continue
        key = (c.get("stage"), c.get("dimension"), c.get("level"))
        pc = prev_checks.get(key)
        if pc is None or pc.get("decision") != "fail" or "median" not in pc or "median" not in c:
            continue
        cmd_prev = _check_command(prev_cand, key[0], key[1])
        cmd_cur = _check_command(cur_cand, key[0], key[1])
        if cmd_prev is None or cmd_cur is None or cmd_cur == cmd_prev:
            continue
        reason = c.get("reason", "")
        want_dir = 1 if "below" in reason else -1 if "above" in reason else 0
        if want_dir == 0:
            continue
        cmd_dir = 1 if cmd_cur > cmd_prev else -1
        val_dir = 1 if c["median"] > pc["median"] else -1 if c["median"] < pc["median"] else 0
        if cmd_dir == want_dir and val_dir != want_dir:
            frozen.add(key)
    return frozen


class RuleProposer:
    """Deterministic proposer: level -> default command; repair on audit feedback."""

    name = "rule_proposer.v1"

    def propose(self, spec, parse, history):
        cand = default_candidate(spec.task, parse.intents)
        if not history:
            return cand, {}
        prev = json.loads(json.dumps(history[-1]["candidate"]))
        checks = (history[-1].get("audit") or {}).get("checks", [])
        frozen = _stalled_checks(history)
        for c in checks:
            if c.get("decision") != "fail":
                continue
            key = (c.get("stage"), c.get("dimension"), c.get("level"))
            if key in frozen:
                # the last two attempts show a non-monotone / unresponsive realised value for
                # this exact command direction: leave it unchanged rather than keep pushing.
                continue
            reason = c.get("reason", "")
            if c["dimension"] == "press":
                lo, hi, _ = LEVEL_RANGES["press"][c["level"]]
                f = 0.9 if "above" in reason else 1.1
                prev["press"]["target_n"] = float(min(hi, max(lo, prev["press"]["target_n"] * f)))
            elif c["dimension"] == "speed":
                lo, hi, _ = speed_range(spec.task, c["level"])
                for e in prev["stages"]:
                    if e["stage_id"] == c["stage"]:
                        f = 0.7 if c["level"] == "slow" else 1.25
                        e["v_max_mps"] = float(min(hi, max(lo, e["v_max_mps"] * f)))
            else:
                lo, hi, _ = LEVEL_RANGES["grip"][c["level"]]
                f = 0.75 if "above" in reason else 1.25
                prev["grip"]["target_lo_n"] = float(min(hi, max(lo, prev["grip"]["target_lo_n"] * f)))
        audit = history[-1].get("audit") or {}
        if (spec.task != "wipe" and audit.get("task") == "fail" and isinstance(prev.get("grip"), dict)
                and not any(c.get("decision") == "fail" for c in checks)
                and ("grasp", "grip", prev["grip"]["level"]) not in frozen):
            # task failure with no failed manner check (e.g. the object slipped): geometric search upward in grip
            lo, hi, _ = LEVEL_RANGES["grip"][prev["grip"]["level"]]
            prev["grip"]["target_lo_n"] = float(min(hi, max(lo, prev["grip"]["target_lo_n"] * GRIP_TASK_FAIL_FACTOR)))
        return prev, {}


@dataclass
class RequestResult:
    request_id: str
    spec: RequestSpec
    state: str
    accepted_attempt_id: Optional[str]
    n_attempts: int


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save_raw(raw_dir: Path, ep: RawEpisode, keep_images: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in ep.arrays.items() if k != "images"}
    np.savez_compressed(raw_dir / "trajectory.npz", **arrays)
    size = (raw_dir / "trajectory.npz").stat().st_size
    if keep_images and "images" in ep.arrays:
        np.save(raw_dir / "images.npy", ep.arrays["images"])
        size += (raw_dir / "images.npy").stat().st_size
    return size


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_requests(
    specs: Sequence[RequestSpec],
    cal: Mapping[str, Any],
    out_dir: str | Path,
    *,
    max_attempts: int = 3,
    proposer: Optional[Proposer] = None,
    produce: Optional[Callable[..., RawEpisode]] = None,
    log: Callable[[str], None] = print,
) -> list[RequestResult]:
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    proposer = proposer or RuleProposer()
    drafts = [s.draft() for s in specs]
    manifests = [d.to_manifest_dict() for d in drafts]
    ledger = AttemptLedger(max_attempts=max_attempts)
    for s in specs:
        assert_seed_isolated(cal, s.seed)
    for m in manifests:                      # register BEFORE any attempt
        ledger.register_request(m)
    (out / "requests.jsonl").write_text(
        "\n".join(json.dumps(m, sort_keys=True, ensure_ascii=False, default=_json_default) for m in manifests) + "\n",
        encoding="utf-8")
    results: list[RequestResult] = []
    for spec, draft in zip(specs, drafts):
        rid = draft.request_id
        parse = spec.parse()
        history: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        prev_plan = prev_snaps = None          # last executed plan + its stage-start snapshots (this request only)
        accepted: Optional[str] = None
        for k in range(max_attempts):
            wall0, cpu0 = time.time(), time.process_time()
            aid_hint = f"{rid}:a{k + 1}"
            raw, call = proposer.propose(spec, parse, history)
            res = dict(physics_substeps=0, api_calls=int(call.get("api_calls", 0)), api_tokens=int(call.get("api_tokens", 0)),
                       cpu_time_s=0.0, gpu_time_s=0.0, wall_time_s=0.0, storage_bytes=0)

            def record(outcome, reason, cand, meta, duplicate_of=None):
                res["wall_time_s"] = time.time() - wall0
                res["cpu_time_s"] = time.process_time() - cpu0
                return ledger.record_attempt(rid, outcome=outcome, proposal_source=proposer.name, candidate_plan=cand,
                                             call_record=call, resources=ResourceUsage(**res), reason=reason, metadata=meta,
                                             duplicate_of_attempt_id=duplicate_of)

            cand = raw
            if isinstance(raw, str):
                try:
                    cand = json.loads(raw)
                except json.JSONDecodeError as exc:
                    record(AttemptOutcome.PARSE_FAILURE, f"json_decode:{exc.msg}", {"raw_text": raw[:2000]}, {})
                    history.append({"candidate": None, "audit": None, "failure": "parse_failure"})
                    continue
            if cand is None or not isinstance(cand, Mapping):
                record(AttemptOutcome.PARSE_FAILURE, "candidate_not_object", None, {})
                history.append({"candidate": None, "audit": None, "failure": "parse_failure"})
                continue
            gate = semantic_gate(spec.task, parse, cand, spec.grip_scale)
            if not gate.accepted:
                record(AttemptOutcome.INVALID, ";".join(gate.reasons), cand, {"gate": gate.to_dict()})
                history.append({"candidate": cand, "audit": None, "failure": "gate"})
                continue
            chash = _sha(cand)
            if chash in seen:
                record(AttemptOutcome.DUPLICATE, "same_candidate_as_previous_attempt", cand, {"candidate_hash": chash},
                       duplicate_of=seen[chash])
                history.append({"candidate": cand, "audit": None, "failure": "duplicate"})
                continue
            plan = plan_from_candidate(spec.task, cand)
            resume = None
            if prev_plan is not None and prev_snaps:
                blk = first_diff_block(prev_plan, plan)
                resume = prev_snaps.get(blk) if blk and blk != "approach" else None
            snaps: dict[str, Any] = {}
            if produce is not None:
                run = produce
            elif spec.task == "wipe":
                from synthesis_v2.wipe_producer import produce_wipe_episode as run
            else:
                run = produce_episode
            if spec.task == "wipe":  # no stage-level resume for wipe: a repair re-runs the whole episode
                resume, snaps = None, None
            ep = run(spec.task, spec.seed, plan, keep_images=True, snapshots=snaps, resume=resume)
            prefix = resume.prefix_steps if resume is not None else 0
            res["physics_substeps"] = max(int(ep.n_steps) - prefix, 0) * SUBSTEPS_PER_STEP
            prev_plan, prev_snaps = plan, snaps or {}
            raw_dir = out / "raw" / f"{rid}_a{k + 1}"
            res["storage_bytes"] = _save_raw(raw_dir, ep, keep_images=False) if ep.n_steps else 0  # before audit
            audit = audit_episode(ep.arrays, task=spec.task, seed=spec.seed, raw_language=spec.language, cal=cal,
                                  request=parse, grip_scale=spec.grip_scale) \
                if ep.n_steps else {"overall": "unknown", "task": "unknown", "manner": "unknown", "safety": "unknown",
                                    "reasons": ["empty_episode"], "checks": []}
            ok = audit["overall"] == "pass"
            if ep.n_steps and hasattr(proposer, "observe"):
                proposer.observe(rid, attempt_id_for(rid, k + 1), spec, cand, audit)
            if ok and "images" in ep.arrays:
                np.save(raw_dir / "images.npy", ep.arrays["images"])
                res["storage_bytes"] += (raw_dir / "images.npy").stat().st_size
            meta = {"candidate_hash": chash, "plan_hash": ep.plan_hash, "producer_status": ep.status,
                    "audit_overall": audit["overall"], "audit_task": audit["task"], "audit_manner": audit["manner"],
                    "audit_safety": audit["safety"], "raw_dir": str(raw_dir.relative_to(out)), "n_steps": ep.n_steps,
                    "stage_steps": ep.stage_steps, "gate": gate.to_dict(),
                    "resumed_from_stage": resume.stage if resume is not None else None, "prefix_steps_reused": prefix}
            seen[chash] = attempt_id_for(rid, k + 1)
            if ep.n_steps:
                (raw_dir / "plan.json").write_text(json.dumps(ep.plan, sort_keys=True), encoding="utf-8")
                (raw_dir / "audit.json").write_text(json.dumps(audit, sort_keys=True, default=_json_default), encoding="utf-8")
                (raw_dir / "episode_meta.json").write_text(
                    json.dumps({"status": ep.status, "meta": ep.meta, "seed": spec.seed, "task": spec.task,
                                "candidate": cand, "audit_version": AUDIT_VERSION}, sort_keys=True, default=_json_default),
                    encoding="utf-8")
            if ep.status != "completed":
                record(AttemptOutcome.EXECUTION_FAILURE, ep.status, cand, meta)
                history.append({"candidate": cand, "audit": audit, "failure": "execution"})
            elif ok:
                rec = record(AttemptOutcome.ACCEPTED, "audit_pass", cand, meta)
                accepted = rec.attempt_id if hasattr(rec, "attempt_id") else aid_hint
                history.append({"candidate": cand, "audit": audit, "failure": None})
                break
            else:
                record(AttemptOutcome.VALIDATION_FAILURE, ";".join(audit["reasons"])[:500], cand, meta)
                history.append({"candidate": cand, "audit": audit, "failure": "audit"})
        state = ledger.state(rid).value
        results.append(RequestResult(rid, spec, state, accepted, ledger.attempt_count(rid)))
        log(f"{spec.task} seed={spec.seed} '{spec.language}' -> {state} ({ledger.attempt_count(rid)} attempts)")
        write_attempt_ledger(out / "attempts.jsonl", ledger)
    return results


def summarize(out_dir: str | Path) -> dict[str, Any]:
    """Recompute headline numbers from the on-disk requests + ledger only."""
    out = Path(out_dir)
    reqs = [json.loads(x) for x in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    atts = [json.loads(x) for x in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by: dict[str, list[dict]] = {r["request_id"]: [] for r in reqs}
    for a in atts:
        by[a["request_id"]].append(a)
    outcomes: dict[str, int] = {}
    for a in atts:
        outcomes[a["outcome"]] = outcomes.get(a["outcome"], 0) + 1
    n_acc = sum(1 for v in by.values() if any(a["outcome"] == "accepted" for a in v))
    first = sum(1 for v in by.values() if v and v[0]["outcome"] == "accepted")
    res = {k: sum(a["resources"].get(k, 0) for a in atts) for k in ("physics_substeps", "wall_time_s", "api_calls", "storage_bytes")}
    return {"n_requests": len(reqs), "n_attempts": len(atts), "attempt_outcomes": outcomes,
            "n_accepted_requests": n_acc, "accepted_rate": n_acc / len(reqs) if reqs else None,
            "first_attempt_accept_rate": first / len(reqs) if reqs else None, "resources_total": res,
            "attempts_per_request": len(atts) / len(reqs) if reqs else None}
