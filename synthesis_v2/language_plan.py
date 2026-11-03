"""Sentence -> admitted bindings -> production plan, for a registered task program (step 17).

compile_request():  instruction --(LLMClient + intake prompt)--> Intake --(admit: structure, task check, basis, scope gate)-->
decision --(plan_from_bindings)--> ProgramPlan.   Nothing after ``admit`` re-reads the sentence: the plan is a pure function of
the admitted bindings, so a local request cannot be widened by a later lexicon or default.
Anything not ``accept`` returns no plan (``ambiguous`` -> ``needs_clarification``, ``conflict`` -> ask the user).  Only the speed
dimension is turned into plan values here; a binding on another dimension is refused, never dropped.

Gate-feedback retry (``feedback_retry=True``, at most ONE extra model call): only when the first answer was blocked because its
bindings disagree with the sentence's speed scope (``language_scope_mismatch``).  The model is told which instances disagree, not
the plan.  An answer that needed the retry is recorded as such; it is not a first-pass success.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Optional

from synthesis_v2.candidate_plan import MAX_ROT_RPS, ROT_PER_LIN
from synthesis_v2.language_scope import SCOPE_VERSION, scope_rules_hash
from synthesis_v2.manner_request import speed_range
from synthesis_v2.request_intake import INTAKE_PROMPT_VERSION, admit, build_intake_prompt, parse_intake

LANGUAGE_PLAN_VERSION = "language_plan.v2"
FEEDBACK_VERSION = "gate_feedback.v1"
FEEDBACK_SYSTEM_SUFFIX = (
    "\nGATE FEEDBACK. If the user message contains gate_feedback, your previous_answer was rejected by a code check that compares "
    "your speed bindings with the speed words of the instruction. Return a corrected answer in the same JSON format; keep what the "
    "feedback does not criticise. If the scope of a speed word cannot be determined from the instruction, return that clause in "
    "ambiguous instead of guessing.")


class PlanRefused(ValueError):
    def __init__(self, code: str, detail: Any = None):
        super().__init__(f"{code}: {detail}"); self.code, self.detail = code, detail


def plan_from_bindings(program, base, semantics, *, family: str = "stack"):
    """Apply speed bindings to ``base`` (a ProgramPlan).  ``normal`` leaves the stage object untouched (bitwise the base).
    Bindings must be disjoint (check_semantics); two different levels on one instance are refused here as well."""
    level_of: dict[str, tuple[str, str]] = {}
    for b in semantics.bindings:
        if b.dimension != "speed":
            raise PlanRefused("unsupported_dimension", {"clause": b.clause, "dimension": b.dimension})
        for iid in b.instance_ids:
            if iid in level_of and level_of[iid][0] != b.level:
                raise PlanRefused("conflicting_levels", {"instance": iid, "levels": sorted({level_of[iid][0], b.level})})
            level_of[iid] = (b.level, b.clause)
    unknown = sorted(set(level_of) - set(program.ids()))
    if unknown:
        raise PlanRefused("unknown_instance", unknown)
    stages, trace = [], []
    for sp in base.stages:
        iid = next(it.instance_id for it in program.instances if it.label == sp.stage_id)
        lvl = level_of.get(iid, ("normal", ""))[0]
        if iid in level_of and lvl != "normal":
            v = speed_range(family, lvl)[2]
            sp = dataclasses.replace(sp, v_max_mps=v, w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * v))
        stages.append(sp)
        trace.append({"instance_id": iid, "level": lvl if iid in level_of else "neutral (not bound)",
                      "v_max_mps": sp.v_max_mps, "w_max_rps": sp.w_max_rps, "clause": level_of.get(iid, ("", ""))[1]})
    return dataclasses.replace(base, stages=tuple(stages)), trace


def _attempt(text, program, client, feedback: Optional[dict] = None, previous: Any = None):
    system, user = build_intake_prompt(text, program); pv = INTAKE_PROMPT_VERSION
    if feedback is not None:
        u = json.loads(user); u["gate_feedback"] = feedback; u["previous_answer"] = previous
        user = json.dumps(u, sort_keys=True, ensure_ascii=False, indent=2)
        system += FEEDBACK_SYSTEM_SUFFIX; pv = f"{INTAKE_PROMPT_VERSION}+{FEEDBACK_VERSION}"
    obj, rec = client.complete_json(system, user, prompt_version=pv)
    intake = parse_intake(obj, text)
    return obj, rec, intake, admit(intake, program), pv


def _feedback_from(dec: dict) -> Optional[dict]:
    for g in dec.get("guards", []):
        if g.get("gap") == "language_scope_mismatch":
            return {"problem": "your speed bindings disagree with the speed scope of the instruction",
                    "instances_bound_but_not_supported_by_the_instruction": g.get("extra", []),
                    "instances_supported_but_not_bound": g.get("missing", []),
                    "instances_bound_with_wrong_level": g.get("wrong_level", []),
                    **({"note": g["note"]} if g.get("note") else {})}
    return None


def compile_request(text: str, program, base, client, *, family: str = "stack", feedback_retry: bool = False) -> dict[str, Any]:
    n0 = len(client.records)
    obj, rec, intake, dec, pv = _attempt(text, program, client)
    attempts = [{"attempt": 1, "prompt_version": pv, "status": dec["status"], "llm": {"status": rec.status, "request_hash": rec.request_hash, "usage": rec.usage},
                 "bindings": [{"instance_ids": list(b.instance_ids), "level": b.level, "clause": b.clause} for b in intake.semantics.bindings]}]
    fb = _feedback_from(dec) if (feedback_retry and dec["status"] != "accept") else None
    if fb is not None:
        obj2, rec, intake, dec, pv = _attempt(text, program, client, feedback=fb, previous=obj)
        attempts.append({"attempt": 2, "prompt_version": pv, "feedback": fb, "status": dec["status"],
                         "llm": {"status": rec.status, "request_hash": rec.request_hash, "usage": rec.usage},
                         "bindings": [{"instance_ids": list(b.instance_ids), "level": b.level, "clause": b.clause} for b in intake.semantics.bindings]})
    out: dict[str, Any] = {"instruction": text, "status": dec["status"], "decision": dec, "intake": intake.to_dict(), "attempts": attempts,
                           "needed_feedback_retry": fb is not None and dec["status"] == "accept",
                           "llm": {"status": rec.status, "request_hash": rec.request_hash, "usage": rec.usage, "n_records": len(client.records) - n0},
                           "versions": {"plan": LANGUAGE_PLAN_VERSION, "intake_prompt": INTAKE_PROMPT_VERSION, "feedback": FEEDBACK_VERSION,
                                        "scope": SCOPE_VERSION, "scope_rules_hash": scope_rules_hash()},
                           "plan": None, "plan_hash": None, "trace": None, "needs_clarification": dec["status"] == "ambiguous"}
    if dec["status"] == "accept":
        plan, trace = plan_from_bindings(program, base, intake.semantics, family=family)
        out.update(plan=plan, plan_hash=plan.plan_hash(), trace=trace)
    return out
