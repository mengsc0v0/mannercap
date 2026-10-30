"""Sentence -> admitted bindings -> production plan, for a registered task program (step 17).

compile_request():  instruction --(LLMClient + intake prompt)--> Intake --(admit: structure, task check, basis, scope guards)-->
decision --(plan_from_bindings)--> ProgramPlan.   Nothing after ``admit`` re-reads the sentence: the plan is a pure function of
the admitted bindings, so a local request cannot be widened by a later lexicon or default.  Anything not ``accept`` returns no plan
(``ambiguous`` -> ``needs_clarification``).  Only the speed dimension is turned into plan values here; a binding on another
dimension is refused (``unsupported_dimension``), never dropped.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from synthesis_v2.candidate_plan import MAX_ROT_RPS, ROT_PER_LIN
from synthesis_v2.language_scope import SCOPE_VERSION, scope_rules_hash
from synthesis_v2.manner_request import speed_range
from synthesis_v2.request_intake import INTAKE_PROMPT_VERSION, admit, build_intake_prompt, parse_intake

LANGUAGE_PLAN_VERSION = "language_plan.v1"


class PlanRefused(ValueError):
    def __init__(self, code: str, detail: Any = None):
        super().__init__(f"{code}: {detail}"); self.code, self.detail = code, detail


def plan_from_bindings(program, base, semantics, *, family: str = "stack"):
    """Apply speed bindings to ``base`` (a ProgramPlan).  ``normal`` leaves the stage object untouched (bitwise the base)."""
    level_of: dict[str, tuple[str, str]] = {}
    for b in semantics.bindings:
        if b.dimension != "speed":
            raise PlanRefused("unsupported_dimension", {"clause": b.clause, "dimension": b.dimension})
        for iid in b.instance_ids:
            if iid in level_of and level_of[iid][0] != b.level:
                raise PlanRefused("conflicting_levels", {"instance": iid, "levels": [level_of[iid][0], b.level]})
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


def compile_request(text: str, program, base, client, *, family: str = "stack") -> dict[str, Any]:
    system, user = build_intake_prompt(text, program)
    n0 = len(client.records)
    obj, rec = client.complete_json(system, user, prompt_version=INTAKE_PROMPT_VERSION)
    intake = parse_intake(obj, text)
    dec = admit(intake, program)
    out: dict[str, Any] = {"instruction": text, "status": dec["status"], "decision": dec, "intake": intake.to_dict(),
                           "llm": {"status": rec.status, "request_hash": rec.request_hash, "usage": rec.usage,
                                   "n_records": len(client.records) - n0},
                           "versions": {"plan": LANGUAGE_PLAN_VERSION, "intake_prompt": INTAKE_PROMPT_VERSION,
                                        "scope": SCOPE_VERSION, "scope_rules_hash": scope_rules_hash()},
                           "plan": None, "plan_hash": None, "trace": None, "needs_clarification": dec["status"] == "ambiguous"}
    if dec["status"] == "accept":
        plan, trace = plan_from_bindings(program, base, intake.semantics, family=family)
        out.update(plan=plan, plan_hash=plan.plan_hash(), trace=trace)
    return out
