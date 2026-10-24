"""Instruction -> object/stage-scoped manner bindings and typed task programs (design v3 §2.8).

Three checks are kept apart on purpose:

* **structural** -- do the instance ids, dimensions and levels exist in the frozen registries?
  (``check_semantics`` / ``check_program``; no model involved, so an answer can never define its own
  vocabulary or invent a newton threshold);
* **semantic** -- does every clause of the original sentence appear, and is each one bound to the
  object the sentence names? (``clause_coverage``, scored against a human-verified answer that is
  never shown to the model);
* **physical** -- did the executed episode meet the bands? (the existing ``episode_audit``).

``lexicon_semantics`` is the deterministic baseline: same output type, no model call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from synthesis_v2.manner_request import GRIP_LEVELS, LEVEL_RANGES, PRESS_LEVELS, SPEED_LEVELS
from synthesis_v2.task_program import (
    PRIMITIVES, STAGE_WORDS, TaskProgram, resolve_stage_ref,
)

SEMANTICS_PROMPT_VERSION = "mannercap.semantics.v1"
PROGRAM_PROMPT_VERSION = "mannercap.program.v1"

DIMENSION_LEVELS: dict[str, tuple[str, ...]] = {
    "speed": SPEED_LEVELS, "grip": GRIP_LEVELS, "press": PRESS_LEVELS,
}
# Level words the deterministic baseline recognises.  The model may normalise synonyms to these;
# it may never introduce a level that is not here.
LEVEL_WORDS: dict[str, tuple[str, str]] = {
    "slowly": ("speed", "slow"), "slow": ("speed", "slow"), "carefully": ("speed", "slow"),
    "quickly": ("speed", "fast"), "fast": ("speed", "fast"), "rapidly": ("speed", "fast"),
    "gently": ("grip", "gentle"), "gentle": ("grip", "gentle"), "lightly": ("grip", "gentle"),
    "firmly": ("grip", "firm"), "firm": ("grip", "firm"), "tightly": ("grip", "firm"),
    "hard": ("press", "firm"),
}


@dataclass(frozen=True)
class MannerBinding:
    """One manner requirement, already scoped to concrete stage instances."""

    instance_ids: tuple[str, ...]
    dimension: str
    level: str
    object_role: Optional[str] = None
    clause: str = ""          # the span of the original sentence this came from
    source: str = "llm"       # llm | lexicon | registered

    def to_dict(self) -> dict[str, Any]:
        return {"instance_ids": list(self.instance_ids), "dimension": self.dimension,
                "level": self.level, "object_role": self.object_role, "clause": self.clause,
                "source": self.source}


@dataclass(frozen=True)
class Semantics:
    """The full parse of one instruction: what was bound, and what was explicitly *not*."""

    instruction: str
    bindings: tuple[MannerBinding, ...] = ()
    ambiguous: tuple[dict[str, Any], ...] = ()      # clause -> candidate instance ids
    unsupported: tuple[dict[str, Any], ...] = ()    # clause -> reason
    source: str = "llm"
    prompt_version: str = ""

    def complete(self) -> bool:
        """True only if nothing was left ambiguous or unsupported (never a silent downgrade)."""
        return not self.ambiguous and not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "bindings": [b.to_dict() for b in self.bindings],
                "ambiguous": list(self.ambiguous), "unsupported": list(self.unsupported),
                "source": self.source, "prompt_version": self.prompt_version}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Semantics":
        return Semantics(
            instruction=str(d.get("instruction", "")),
            bindings=tuple(MannerBinding(
                tuple(b.get("instance_ids", ())), str(b["dimension"]), str(b["level"]),
                b.get("object_role"), str(b.get("clause", "")), str(b.get("source", "llm")))
                for b in d.get("bindings", ())),
            ambiguous=tuple(dict(x) for x in d.get("ambiguous", ())),
            unsupported=tuple(dict(x) for x in d.get("unsupported", ())),
            source=str(d.get("source", "llm")), prompt_version=str(d.get("prompt_version", "")))


# ---- structural validation -------------------------------------------------
def check_semantics(sem: Semantics, program: TaskProgram) -> list[str]:
    """Structural errors only: unknown instance / dimension / level, or a dimension the stage lacks."""
    errs: list[str] = []
    ids = set(program.ids())
    for i, b in enumerate(sem.bindings):
        if b.dimension not in DIMENSION_LEVELS:
            errs.append(f"unknown_dimension:{i}:{b.dimension}"); continue
        if b.level not in DIMENSION_LEVELS[b.dimension]:
            errs.append(f"unknown_level:{i}:{b.dimension}:{b.level}")
        if not b.instance_ids:
            errs.append(f"unbound_clause:{i}")
        for iid in b.instance_ids:
            if iid not in ids:
                errs.append(f"unknown_instance:{i}:{iid}"); continue
            it = program.instance(iid)
            if b.dimension not in PRIMITIVES[it.primitive]["dims"]:
                errs.append(f"dimension_not_on_stage:{i}:{iid}:{b.dimension}")
            if b.object_role is not None and b.object_role not in (it.object_role, it.target_role):
                errs.append(f"object_mismatch:{i}:{iid}:{b.object_role}")
    seen: dict[tuple[str, str], int] = {}
    for i, b in enumerate(sem.bindings):
        for iid in b.instance_ids:
            k = (iid, b.dimension)
            if k in seen:
                errs.append(f"conflicting_binding:{seen[k]}:{i}:{iid}:{b.dimension}")
            seen[k] = i
    return errs


def check_program(prog: TaskProgram, *, allowed_backends: Iterable[str] = ("robosuite_native",)) -> list[str]:
    errs = list(prog.validate())
    if prog.backend not in tuple(allowed_backends):
        errs.append(f"unsupported_backend:{prog.backend}")
    for it in prog.instances:
        if it.primitive not in PRIMITIVES:
            errs.append(f"unsupported_primitive:{it.instance_id}:{it.primitive}")
    return errs


# ---- deterministic baseline ------------------------------------------------
_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bthen\b)\s*")
_WORD = re.compile(r"[a-z_][a-z0-9_]*")


def lexicon_semantics(instruction: str, program: TaskProgram) -> Semantics:
    """Controlled-vocabulary parse: no model, no guessing; unresolved clauses are reported."""
    bindings: list[MannerBinding] = []
    ambiguous: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    roles = {r.lower(): r for r in program.objects}
    for clause in (c for c in _CLAUSE_SPLIT.split(instruction.lower()) if c.strip()):
        words = _WORD.findall(clause)
        lv = next((LEVEL_WORDS[w] for w in words if w in LEVEL_WORDS), None)
        if lv is None:
            continue
        dim, level = lv
        role = next((roles[w] for w in words if w in roles), None)
        stage_word = next((w for w in words if w in STAGE_WORDS), None)
        if stage_word is None:
            unsupported.append({"clause": clause, "reason": "no_stage_word"}); continue
        ids, status = resolve_stage_ref(program, stage_word, object_role=role)
        if status == "unknown" or not ids:
            unsupported.append({"clause": clause, "reason": f"unresolved_stage:{stage_word}"}); continue
        if status == "ambiguous":
            ambiguous.append({"clause": clause, "candidates": list(ids)}); continue
        keep = tuple(i for i in ids if dim in PRIMITIVES[program.instance(i).primitive]["dims"])
        if not keep:
            unsupported.append({"clause": clause, "reason": f"dimension_not_on_stage:{dim}"}); continue
        bindings.append(MannerBinding(keep, dim, level, role, clause, "lexicon"))
    return Semantics(instruction, tuple(bindings), tuple(ambiguous), tuple(unsupported),
                     source="lexicon", prompt_version="lexicon.v1")


# ---- prompts ---------------------------------------------------------------
def scene_card(program: TaskProgram) -> dict[str, Any]:
    """What the model is allowed to see: roles, instances and the frozen level vocabulary.

    Simulator-privileged information (object poses, masses) is deliberately *not* included.
    """
    return {
        "task": program.name,
        "object_roles": dict(program.objects),
        "stage_instances": [
            {"instance_id": it.instance_id, "primitive": it.primitive, "object_role": it.object_role,
             "target_role": it.target_role, "subtask": it.subtask, "occurrence": it.occurrence,
             "dimensions": list(PRIMITIVES[it.primitive]["dims"])}
            for it in program.instances],
        "levels": {d: list(v) for d, v in DIMENSION_LEVELS.items()},
    }


SEMANTICS_SYSTEM = (
    "You bind manner requirements in a robot instruction to concrete stage instances.\n"
    "Rules:\n"
    "1. Use ONLY the instance_id values and level words given in the scene card. Never invent a stage, "
    "an object, a level, or any numeric force/speed threshold.\n"
    "2. Every manner clause of the instruction must appear exactly once, in bindings, ambiguous or "
    "unsupported. Never drop a clause.\n"
    "3. A dimension may only be bound to an instance that lists it in its dimensions.\n"
    "4. If a clause names no object and several instances match, put it in ambiguous with the candidates.\n"
    "5. If the instruction asks for something the stage set cannot express, put it in unsupported.\n"
    'Answer with JSON only: {"bindings":[{"instance_ids":[...],"dimension":"speed|grip|press",'
    '"level":"...","object_role":"... or null","clause":"verbatim span"}],'
    '"ambiguous":[{"clause":"...","candidates":[...]}],"unsupported":[{"clause":"...","reason":"..."}]}'
)


def build_semantics_prompt(instruction: str, program: TaskProgram) -> tuple[str, str]:
    user = json.dumps({"instruction": instruction, "scene": scene_card(program)},
                      sort_keys=True, ensure_ascii=False, indent=2)
    return SEMANTICS_SYSTEM, user


def parse_semantics(obj: Any, instruction: str, program: TaskProgram) -> Semantics:
    """Shape the raw answer into ``Semantics``; shape errors raise, content errors are for ``check_semantics``."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"semantics answer is not an object: {type(obj).__name__}")
    out: list[MannerBinding] = []
    for b in obj.get("bindings", ()) or ():
        if not isinstance(b, Mapping) or "dimension" not in b or "level" not in b:
            raise ValueError(f"malformed binding: {b!r}")
        ids = b.get("instance_ids") or ([b["instance_id"]] if b.get("instance_id") else [])
        out.append(MannerBinding(tuple(str(i) for i in ids), str(b["dimension"]), str(b["level"]),
                                 (str(b["object_role"]) if b.get("object_role") else None),
                                 str(b.get("clause", "")), "llm"))
    norm = lambda xs: tuple(dict(x) for x in (xs or ()) if isinstance(x, Mapping))
    return Semantics(instruction, tuple(out), norm(obj.get("ambiguous")), norm(obj.get("unsupported")),
                     source="llm", prompt_version=SEMANTICS_PROMPT_VERSION)


PROGRAM_SYSTEM = (
    "You compose a robot task program out of a fixed primitive library.\n"
    "Rules:\n"
    "1. Use ONLY the primitives listed; each instance must satisfy the 'needs' of its primitive.\n"
    "2. object_role / target_role must be keys of objects; objects maps a role to a scene object name. "
    "Use '@spot:<role>' as a target to put an object back where it started.\n"
    "3. instance_id must be unique and of the form <object>.<subtask>_<k>.<primitive>; preds must list "
    "the instance that immediately precedes it.\n"
    "4. A pick is approach, descend, grasp, lift; a put-down adds transport, place and retreat.\n"
    'Answer with JSON only: {"name":"...","backend":"robosuite_native","env_task":"...",'
    '"objects":{"role":"scene_object"},"instances":[{"instance_id":"...","primitive":"...",'
    '"label":"...","object_role":...,"target_role":...,"subtask":"...","occurrence":1,"preds":[...]}],'
    '"goal":["..."]}'
)


def build_program_prompt(goal: str, env_task: str, scene_objects: Mapping[str, str],
                         *, example: Optional[TaskProgram] = None) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "goal": goal, "env_task": env_task, "scene_objects": dict(scene_objects),
        "primitives": {k: {"dims": list(v["dims"]), "needs": list(v["needs"])} for k, v in PRIMITIVES.items()},
    }
    if example is not None:
        payload["example_program"] = example.to_dict()
    return PROGRAM_SYSTEM, json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)


def parse_program(obj: Any, *, language: str = "") -> TaskProgram:
    if not isinstance(obj, Mapping):
        raise ValueError(f"program answer is not an object: {type(obj).__name__}")
    d = dict(obj)
    d.setdefault("source", "llm")
    d.setdefault("backend", "robosuite_native")
    if language:
        d["language"] = language
    return TaskProgram.from_dict(d)


# ---- semantic scoring against a verified answer ----------------------------
def binding_key(b: MannerBinding) -> tuple[Any, ...]:
    return (tuple(sorted(b.instance_ids)), b.dimension, b.level)


def clause_coverage(got: Semantics, gold: Semantics) -> dict[str, Any]:
    """Compare a parse against a human-verified answer (the gold answer is never shown to the model)."""
    g, h = {binding_key(b) for b in got.bindings}, {binding_key(b) for b in gold.bindings}
    wrong_object = sum(1 for b in got.bindings if binding_key(b) not in h
                       and any(binding_key(x)[1:] == binding_key(b)[1:] for x in gold.bindings))
    return {"n_gold": len(h), "n_got": len(g), "matched": len(g & h), "missing": sorted(h - g),
            "extra": sorted(g - h), "wrong_scope": wrong_object,
            "exact": g == h and got.complete() == gold.complete(),
            "ambiguous_got": len(got.ambiguous), "unsupported_got": len(got.unsupported)}
