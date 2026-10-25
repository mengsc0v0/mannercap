"""Request intake: one instruction -> (task part, manner part) -> admission decision against a registered task program.

Split on purpose:
* task part   : what the sentence ITSELF states about the base task (objects, target relations, explicit order);
* manner part : manner requirements bound to stage instances, each with the evidence it was bound on.
``admit`` then checks  sentence -> registered task program  (code, no model) and applies guards that do not depend on the
model: a numeric manner value is never silently dropped or replaced, a binding must carry an admissible basis (never
"task default order"), a cited antecedent must occur in the instruction, a partial answer is never accepted.

Decisions: accept | ambiguous | unsupported | conflict | incomplete | invalid.  ``incomplete`` = the answer is structurally valid
but leaves out something the sentence states (checked against the sentence, not by the model's own report).  Final task success is still judged by the environment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from synthesis_v2.language_compiler import (DIMENSION_LEVELS, LEVEL_WORDS, MannerBinding, Semantics, _CLAUSE_SPLIT, check_semantics,
                                            lexicon_semantics)
from synthesis_v2.task_program import PRIMITIVES, TaskProgram

INTAKE_PROMPT_VERSION = "mannercap.intake.v3_1"
ACTIONS = ("put", "pick", "close", "turn_on", "open")
_ACTION_SYN = {"pick_up": "pick", "grasp": "pick", "lift": "pick", "pickup": "pick", "place": "put", "move": "put", "insert": "put", "pick_place": "put", "shut": "close", "switch_on": "turn_on", "turn on": "turn_on"}
BASES = ("explicit_object", "antecedent", "single_candidate", "lexicon", "request_scope")
# language protocol v2 (R1-R4): a SPEED request that names a stage (or no stage) and no object applies to every matching
# instance of the task; grip/press requests keep requiring an object.
STAGE_WORDS = {"approach": "approach", "approaching": "approach", "descend": "descend", "descending": "descend",
               "lift": "lift", "lifting": "lift", "carry": "transport", "carrying": "transport", "transport": "transport",
               "transporting": "transport", "place": "place", "placing": "place", "retreat": "retreat", "retreating": "retreat"}
UNSUPPORTED_CODES = ("numeric_level", "dimension_not_on_stage", "quantity_not_measured", "object_not_in_task", "stage_not_in_program", "other")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:n|newtons?|kg|g|m/s|cm/s|mm/s|%|percent|cm|mm)(?![a-z])", re.I)
PRONOUNS = ("it", "them", "its", "one", "that", "this")


@dataclass(frozen=True)
class TaskStep:
    action: str
    object_role: Optional[str]
    object_phrase: str = ""
    target_role: Optional[str] = None
    target_phrase: str = ""
    clause: str = ""


@dataclass(frozen=True)
class Intake:
    instruction: str
    restated: bool
    steps: tuple[TaskStep, ...]
    order: tuple[tuple[int, int], ...]
    semantics: Semantics
    bases: tuple[dict, ...]            # per binding: {"basis":..., "antecedent":...}
    unsupported_codes: tuple[str, ...] # per semantics.unsupported entry
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "restated": self.restated, "steps": [s.__dict__ for s in self.steps],
                "order": [list(o) for o in self.order], "semantics": self.semantics.to_dict(), "bases": list(self.bases),
                "unsupported_codes": list(self.unsupported_codes), "source": self.source}


# ---- program side -----------------------------------------------------------
def program_steps(program: TaskProgram) -> list[tuple[str, str, Optional[str]]]:
    out = []
    for it in program.instances:
        if it.primitive in ("place", "insert"):
            out.append(("put", it.object_role, it.target_role))
        elif it.primitive == "articulate":
            out.append((it.subtask.split(".")[-1].rsplit("_", 1)[0], it.object_role or it.target_role, None))
    return out


def registered_task_card(program: TaskProgram) -> dict[str, Any]:
    return {"language": program.language, "steps": [{"action": a, "object_role": o, "target_role": t} for a, o, t in program_steps(program)]}


# ---- prompt -----------------------------------------------------------------
INTAKE_SYSTEM = (
    "You analyse ONE instruction for a robot task that is already registered. Answer with JSON only.\n"
    "TASK PART. List only what the sentence itself states about the base task. Never copy the registered task into the answer. "
    "If the sentence states no task content (only a manner), set restated=false and steps=[]. "
    "Each step: action (put|pick|close|turn_on|open; 'pick' = pick up / grasp without a stated destination), object_phrase as written, object_role (a key of object_roles, or null if no role "
    "matches the phrase), target_phrase and target_role likewise (null for close/turn_on). A plural phrase naming several "
    "registered objects (both moka pots) is one step per object. Never map a phrase to a role that does not "
    "match it (a red mug is not the white mug). order: pairs [i,j] only if the sentence explicitly puts step i before step j "
    "(first, then, before, after); 'and' alone is not an order.\n"
    "MANNER PART. Use ONLY the instance_id values and level words in the scene card; never invent a stage, object, level or number.\n"
    "1. Every manner clause appears exactly once, in bindings, ambiguous or unsupported. Never drop a clause, never keep only the "
    "executable part of a clause.\n"
    "2. A binding needs a basis: 'request_scope' (rule 6), 'explicit_object' (the clause names the object), 'antecedent' (a pronoun whose referent is fixed by the "
    "instruction text; copy that phrase into 'antecedent'), or 'single_candidate' (no object named, exactly one stage instance of the task "
    "can take that dimension; this includes a pronoun-only request such as 'hold it gently' when the sentence states no task content and mentions "
    "no other object or target that 'it' could mean). Never bind by task order, nearest mention, first or last.\n"
    "3. A pronoun (it, them, one) is resolved from the instruction text alone. If exactly one mentioned object can be meant, bind it with "
    "basis antecedent. If two or more mentioned objects are plausible referents, return it in ambiguous with candidate instance ids.\n"
    "4. A numeric value (5 N, 2 cm/s, 30%) is never a level: put the clause in unsupported with code numeric_level.\n"
    "5. Unsupported codes: numeric_level | dimension_not_on_stage (the stage has no such dimension) | quantity_not_measured (e.g. gentle "
    "contact, impact, touching) | object_not_in_task (manner on an object the task program does not contain) | stage_not_in_program | other.\n"
    "6. A SPEED clause that names no object applies to every matching stage instance of the task and is NOT ambiguous: with no stage word "
    "('move slowly', 'go slowly') bind ALL instances whose dimension list contains speed; with a stage word ('when approaching', 'while "
    "carrying') bind ALL instances of that stage for every object of the task. Use basis 'request_scope' for such a binding, "
    "leave object_role null, and list every instance_id the rule covers. A global speed request never changes grip or press (separate "
    "dimensions). A grip or press clause that names no object and fits several grasp or press instances stays ambiguous with the candidates. "
    "Use ambiguous only when two clauses on the same stage and dimension contradict each other or a pronoun/quantifier has two or more plausible referents in the sentence (rule 3). A pronoun with exactly one registered candidate and no competing mention is NOT ambiguous (rule 2).\n"
    'JSON: {"task":{"restated":bool,"steps":[{"action":"...","object_phrase":"...","object_role":"..|null","target_phrase":"...",'
    '"target_role":"..|null","clause":"..."}],"order":[[0,1]]},'
    '"bindings":[{"instance_ids":[...],"dimension":"speed|grip|press","level":"...","object_role":"..|null","clause":"...",'
    '"basis":"explicit_object|antecedent|single_candidate|request_scope","antecedent":"..|null"}],'
    '"ambiguous":[{"clause":"...","candidates":[...]}],"unsupported":[{"clause":"...","code":"...","reason":"..."}]}'
)


def build_intake_prompt(instruction: str, program: TaskProgram) -> tuple[str, str]:
    from synthesis_v2.language_compiler import scene_card
    card = scene_card(program)
    card["registered_task"] = registered_task_card(program)
    return INTAKE_SYSTEM, json.dumps({"instruction": instruction, "scene": card}, sort_keys=True, ensure_ascii=False, indent=2)


def _txt(v: Any) -> str:
    """JSON null / 'None' -> ''."""
    return "" if v is None or str(v).strip().lower() in ("none", "null") else str(v)


def parse_intake(obj: Any, instruction: str) -> Intake:
    if not isinstance(obj, Mapping):
        raise ValueError(f"intake answer is not an object: {type(obj).__name__}")
    for k in ("task", "bindings", "ambiguous", "unsupported"):
        if k not in obj:
            raise ValueError(f"missing_field:{k}")
    t = obj["task"]
    if not isinstance(t, Mapping) or not isinstance(t.get("steps"), list) or not isinstance(t.get("order"), list) \
            or not isinstance(t.get("restated"), bool):
        raise ValueError("malformed_task")
    for k in ("bindings", "ambiguous", "unsupported"):
        if not isinstance(obj[k], list) or any(not isinstance(x, Mapping) or not _txt(x.get("clause")) for x in obj[k]):
            raise ValueError(f"malformed_field:{k}")
    if any(not (isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(i, int) for i in p)) for p in t["order"]):
        raise ValueError("malformed_order")
    steps = []
    for s in t["steps"]:
        if not isinstance(s, Mapping) or "action" not in s:
            raise ValueError(f"malformed step: {s!r}")
        a = str(s["action"]).strip().lower()
        steps.append(TaskStep(_ACTION_SYN.get(a, a), (str(s["object_role"]) if _txt(s.get("object_role")) else None), _txt(s.get("object_phrase")),
                              (str(s["target_role"]) if _txt(s.get("target_role")) else None), _txt(s.get("target_phrase")), _txt(s.get("clause"))))
    bind, bases = [], []
    for b in obj["bindings"]:
        if not isinstance(b, Mapping) or "dimension" not in b or "level" not in b:
            raise ValueError(f"malformed binding: {b!r}")
        ids = b.get("instance_ids") or ([b["instance_id"]] if b.get("instance_id") else [])
        bind.append(MannerBinding(tuple(str(i) for i in ids), str(b["dimension"]), str(b["level"]), (str(b["object_role"]) if b.get("object_role") else None),
                                  str(b.get("clause", "")), "llm"))
        bases.append({"basis": str(b.get("basis", "")), "antecedent": (str(b["antecedent"]) if b.get("antecedent") else "")})
    norm = lambda xs: tuple(dict(x) for x in (xs or ()) if isinstance(x, Mapping))
    unsup = norm(obj.get("unsupported"))
    order = tuple((int(p[0]), int(p[1])) for p in t["order"])
    sem = Semantics(instruction, tuple(bind), norm(obj.get("ambiguous")), unsup, source="llm", prompt_version=INTAKE_PROMPT_VERSION)
    return Intake(instruction, bool(t["restated"]), tuple(steps), order, sem, tuple(bases), tuple(str(u.get("code", "other")) for u in unsup))


def lexicon_intake(instruction: str, program: TaskProgram) -> Intake:
    sem = lexicon_semantics(instruction, program)
    codes = tuple("dimension_not_on_stage" if "dimension_not_on_stage" in str(u.get("reason")) else "other" for u in sem.unsupported)
    return Intake(instruction, False, (), (), sem, tuple({"basis": "lexicon", "antecedent": ""} for _ in sem.bindings), codes, source="lexicon")


# ---- admission --------------------------------------------------------------
def _step_index(program: TaskProgram) -> dict[tuple, int]:
    """step key -> instance index; 'pick' steps map to the grasp instance of that object."""
    out: dict[tuple, int] = {}
    for k, it in enumerate(program.instances):
        if it.primitive in ("place", "insert"):
            out[("put", it.object_role, it.target_role)] = k
        elif it.primitive == "articulate":
            out[(it.subtask.split(".")[-1].rsplit("_", 1)[0], it.object_role or it.target_role, None)] = k
        elif it.primitive == "grasp":
            out[("pick", it.object_role, None)] = k
    return out


def check_task(intake: Intake, program: TaskProgram) -> tuple[list[dict], list[tuple]]:
    """sentence -> registered program.  Returns (conflicts, program steps the sentence did not state)."""
    idx = _step_index(program)
    errs: list[dict] = []
    matched: dict[int, int] = {}
    for i, s in enumerate(intake.steps):
        if s.object_role is None or s.object_role not in program.objects:
            errs.append({"code": "object_not_in_program", "step": i, "phrase": s.object_phrase or s.object_role}); continue
        if s.target_phrase and (s.target_role is None or s.target_role not in program.objects):
            errs.append({"code": "target_not_in_program", "step": i, "phrase": s.target_phrase}); continue
        key = (s.action, s.object_role, s.target_role)
        if key in idx:
            matched[i] = idx[key]; continue
        same_obj = [k for k in idx if k[1] == s.object_role and k[0] == s.action]
        errs.append({"code": "target_mismatch" if same_obj else "step_not_in_program", "step": i, "stated": list(key),
                     "program": [list(k) for k in same_obj]})
    for i, j in intake.order:
        if i in matched and j in matched and matched[i] > matched[j]:
            errs.append({"code": "order_mismatch", "pair": [i, j]})
    stated = {k for k, v in idx.items() if v in matched.values()}
    return errs, ([k for k in idx if k[0] != "pick" and k not in stated] if intake.restated else [])


_ORDER_CUE = re.compile(r"\b(then|before|after|afterwards|first|finally|next)\b", re.I)
_TASK_VERB = re.compile(r"\b(put|place|close|open|turn on|move|insert)\b", re.I)
_PLURAL = re.compile(r"\b(both|all|each|every|them)\b", re.I)
_CUE_ONLY = {"轻": "gentle", "慢": "slow", "快": "fast", "用力": "firm", "紧": "firm"}   # coverage cues, never bindings


def _mentioned_roles(text: str, program: TaskProgram) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {r for r in program.objects if all(w in words for w in re.findall(r"[a-z]+", r.lower()) if not w.isdigit())}


def coverage_gaps(intake: Intake, program: TaskProgram) -> list[dict]:
    """What the sentence states but the answer leaves out.  Uses the sentence + the registered vocabulary only (not the
    model's own coverage claim).  Limits: only vocabulary level words (LEVEL_WORDS + a few cues), plural cues, registered
    role names, order words and action verbs are detected; an unusual manner word outside the vocabulary is not."""
    text, low = intake.instruction, intake.instruction.lower()
    sem, gaps = intake.semantics, []
    entries = [("binding", b.clause.lower(), b) for b in sem.bindings] + \
              [("ambiguous", str(a.get("clause", "")).lower(), a) for a in sem.ambiguous] + \
              [("unsupported", str(u.get("clause", "")).lower(), u) for u in sem.unsupported]
    sub = lambda iid: program.instance(iid).subtask.split(".")[0]  # noqa: E731
    # (a) every OCCURRENCE of a manner word must lie inside the verbatim span of an answered clause (a word that appears twice
    # needs two answers, or one answer whose span covers both); (c) a plural scope covers >= 2 objects.
    cues = {w: v for w, v in LEVEL_WORDS.items()}
    spans = [(k, cl, e, [(m.start(), m.end()) for m in re.finditer(re.escape(cl), low)] if cl.strip() else []) for k, cl, e in entries]
    seen_gap = set()
    for m in re.finditer(r"[a-z]+", low):
        w, s0, e0 = m.group(0), m.start(), m.end()
        if w not in cues:
            continue
        cov = [x for x in spans if any(a <= s0 and e0 <= b for a, b in x[3])]
        if not cov:   # a clause the model did not return verbatim can only be matched by the word itself (weaker)
            cov = [x for x in spans if not x[3] and re.search(rf"\b{w}\b", x[1])]
        if not cov:
            gaps.append({"gap": "manner_word_not_answered", "word": w, "at": s0}); continue
        for k, cl, e, pos in cov:
            n_same = sum(1 for k2, cl2, _, _ in spans if cl2 == cl)
            roles = {sub(i) for i in getattr(e, "instance_ids", ())} if k == "binding" else set()
            if len(pos) > 1 and n_same < len(pos) and len(roles) < 2 and (cl, w) not in seen_gap:
                seen_gap.add((cl, w)); gaps.append({"gap": "repeated_requirement_not_covered", "word": w, "clause": cl, "occurrences": len(pos), "answers": n_same})
        seg = next((c for c in _CLAUSE_SPLIT.split(low) if c.strip() and re.search(rf"\b{w}\b", c)), "")
        if _PLURAL.search(seg):
            objs = {sub(i) for k, _, e, _ in cov if k == "binding" for i in e.instance_ids}
            if any(k == "binding" for k, _, _, _ in cov) and len(objs) < 2:
                gaps.append({"gap": "plural_scope_not_covered", "word": w, "objects_bound": sorted(objs)})
    for cue in _CUE_ONLY:
        if cue in text and not any(cue in e[1] for e in entries):
            gaps.append({"gap": "manner_word_not_answered", "word": cue})
    # (b) a binding must agree with the vocabulary word in its own clause
    for k, cl, b in entries:
        if k != "binding":
            continue
        have = [cues[w] for w in re.findall(r"[a-z]+", cl) if w in cues]
        if have and (b.dimension, b.level) not in have:
            gaps.append({"gap": "level_disagrees_with_clause", "clause": b.clause, "bound": [b.dimension, b.level]})
    if intake.source == "llm":
        if intake.steps:
            covered = {x for st in intake.steps for x in (st.object_role, st.target_role) if x}   # steps only: a manner binding does not restate the task
            miss = sorted(_mentioned_roles(text, program) - covered)
            if miss:
                gaps.append({"gap": "named_object_not_in_answer", "roles": miss})
            if len(intake.steps) >= 2 and _ORDER_CUE.search(text) and not intake.order:
                gaps.append({"gap": "order_word_without_order", "word": _ORDER_CUE.search(text).group(0)})
        elif _TASK_VERB.search(text) and _mentioned_roles(text, program):
            gaps.append({"gap": "task_verb_without_steps", "word": _TASK_VERB.search(text).group(0)})
    return gaps


def _has_pronoun(clause: str) -> bool:
    return any(re.search(rf"\b{p}\b", clause.lower()) for p in PRONOUNS)


def request_scope_expected(clause: str, program: TaskProgram) -> set[str]:
    """Instances a speed ``request_scope`` clause must cover (protocol R1/R2): the instances of the stage types it names, or every
    instance that supports speed when it names none.  Code, not the model's claim."""
    words = re.findall(r"[a-z]+", clause.lower())
    prims = {STAGE_WORDS[w] for w in words if w in STAGE_WORDS}
    return {it.instance_id for it in program.instances if "speed" in it.dims and (not prims or it.primitive in prims)}


def admit(intake: Intake, program: TaskProgram) -> dict[str, Any]:
    """Decision for one request.  ``guards`` lists rules that overrode / flagged the model's answer."""
    sem, guards, codes = intake.semantics, [], []
    errs = check_semantics(sem, program)
    if errs:
        return {"status": "invalid", "codes": ["structural"], "detail": errs, "guards": guards, "unstated_program_steps": []}
    conflicts, unstated = check_task(intake, program)
    # numeric guard: a numeric manner value must come back as unsupported:numeric_level, never as a binding / nothing
    text = intake.instruction
    nums = [m.group(0) for m in NUMERIC_RE.finditer(text)]
    numeric_unsup = any(c == "numeric_level" for c in intake.unsupported_codes)
    if nums and not numeric_unsup:
        guards.append({"guard": "numeric_value_not_returned_as_unsupported", "values": nums})
    # basis guard
    ambiguous_extra = []
    for b, bs in zip(sem.bindings, intake.bases):
        if bs["basis"] not in BASES:
            guards.append({"guard": "binding_without_admissible_basis", "clause": b.clause, "basis": bs["basis"]}); ambiguous_extra.append(b.clause)
        elif bs["basis"] == "request_scope" and (b.dimension != "speed" or _mentioned_roles(b.clause, program)):
            guards.append({"guard": "request_scope_only_for_object_free_speed", "clause": b.clause, "dimension": b.dimension}); ambiguous_extra.append(b.clause)
        elif bs["basis"] == "antecedent" and (not bs["antecedent"] or bs["antecedent"].lower() not in text.lower()):
            guards.append({"guard": "antecedent_not_in_instruction", "clause": b.clause, "antecedent": bs["antecedent"]}); ambiguous_extra.append(b.clause)
    gaps = coverage_gaps(intake, program)
    for b, bs in zip(sem.bindings, intake.bases):
        if bs["basis"] == "request_scope" and b.dimension == "speed" and not _mentioned_roles(b.clause, program):
            exp, got = request_scope_expected(b.clause, program), set(b.instance_ids)
            if got != exp:
                gaps.append({"gap": "request_scope_not_covered", "clause": b.clause, "missing": sorted(exp - got), "extra": sorted(got - exp)})
    guards.extend({"guard": "incomplete_answer", **g} for g in gaps)
    if conflicts:
        status, codes = "conflict", sorted({c["code"] for c in conflicts})
    elif nums and (not numeric_unsup or any(NUMERIC_RE.search(b.clause or "") for b in sem.bindings)):
        status, codes = "unsupported", ["numeric_level"]
    elif sem.unsupported:
        status, codes = "unsupported", sorted(set(intake.unsupported_codes))
    elif sem.ambiguous or ambiguous_extra:
        status, codes = "ambiguous", ["ambiguous_reference"]
    elif gaps:
        status, codes = "incomplete", sorted({g["gap"] for g in gaps})
    else:
        status = "accept"
    return {"status": status, "codes": codes, "detail": conflicts, "guards": guards, "unstated_program_steps": [list(u) for u in unstated]}
