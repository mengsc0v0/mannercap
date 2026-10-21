"""Manner request -> stage intents, and the request-to-candidate semantic gate.

* ``parse_manner_request`` turns the *original* language into stage-scoped
  intents (``MannerIntent``).  It never invents a physical threshold: an intent
  is only (stage, dimension, level).  Words outside the lexicon make the whole
  request ``unsupported`` (explicit refusal, never a silent downgrade).
* ``candidate_intents`` reads the levels a candidate plan *declares* and checks
  that the numeric commands lie inside the pre-registered range of that level.
* ``semantic_gate`` compares request and candidate field by field: any missing,
  extra, wrong-stage or wrong-level entry is a rejection with a reason code.

The lexicon and level ranges are versioned; they carry no calibrated force or
speed thresholds (those live in ``manner_calibration``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from synthesis_v2.stage_table import ALIASES, dimension_stages, stage_ids

SEMANTICS_VERSION = "manner_semantics.v2"

SPEED_LEVELS = ("slow", "normal", "fast")
GRIP_LEVELS = ("gentle", "normal", "firm")

# Pre-registered command ranges per level.  speed: v_max_mps; grip: target_lo_n.
# Note: the OSC delta controller realises only ~25% of the commanded speed and
# commands below ~0.05 m/s stall (stage timeout), hence the slow range.
LEVEL_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    "speed": {"slow": (0.05, 0.09, 0.06), "normal": (0.10, 0.20, 0.12), "fast": (0.30, 0.80, 0.40)},
    "grip": {"gentle": (2.0, 6.0, 3.0), "normal": (10.0, 16.0, 14.0), "firm": (24.0, 34.0, 28.0)},
}
GRIP_BAND_WIDTH_N = 6.0
# The language gate reasons only about the nominal grip semantics.  Scene-
# specific required hold force is an audit/oracle concept, not a lexical one.
# We keep a finite widened upper bound so heavy/slippery objects can still
# declare a valid "firm" grasp without changing the level meaning itself.
GRIP_GATE_MAX_SCALE = 2.5

# Wipe (RoboSuite Wipe, tool with no fingers).  speed: v_max_mps of the sweep stroke; press: target tool-vs-table
# normal force (N) of the sweep.  The OSC realises ~25% of a commanded speed above ~0.3 m/s (realised speed
# saturates near 0.2 m/s), so "fast" is a wide command range; slow/normal are realised nearly 1:1.
# No "gentle" press: below ~20 N the tool does not reliably press the dirt (env wiped-test needs ~0.5 mm depth).
PRESS_LEVELS = ("normal", "firm")
LEVEL_RANGES["speed_wipe"] = {"slow": (0.04, 0.06, 0.05), "normal": (0.09, 0.12, 0.10), "fast": (0.30, 0.90, 0.60)}
LEVEL_RANGES["press"] = {"normal": (28.0, 34.0, 30.0), "firm": (42.0, 48.0, 45.0)}


def grip_range(level: str, scale: float = 1.0) -> tuple[float, float, float]:
    """Relative grip range used by audit / oracle logic.

    The scene-specific required hold force is expressed as a multiplier over the
    nominal object.  This is a calibration-side notion; the semantic gate should
    not reinterpret language labels as absolute force bands.
    """
    lo, hi, default = LEVEL_RANGES["grip"][level]
    return lo * scale, hi * scale, default * scale


def grip_gate_range(level: str) -> tuple[float, float, float]:
    """Nominal grip band with a finite widened gate ceiling."""
    lo, hi, default = LEVEL_RANGES["grip"][level]
    return lo, hi * GRIP_GATE_MAX_SCALE, default


def speed_range(task: str, level: str) -> tuple[float, float, float]:
    return LEVEL_RANGES["speed_wipe" if task == "wipe" else "speed"][level]

_STAGE_WORDS: dict[str, tuple[str, ...]] = {
    "approach": ("靠近", "接近", "approach", "reach"),
    "descend": ("下降", "下探", "descend", "lower"),
    "grasp": ("抓", "握", "夹", "grasp", "grip", "grab", "pick"),
    "lift": ("抬", "捧起", "提起", "举", "lift", "raise"),
    "transport": ("端过去", "运送", "搬运", "移动", "transport", "carry", "move"),
    "place": ("放下", "放置", "放", "place", "put"),
    "retreat": ("离开", "撤", "退", "retreat", "leave"),
    "sweep": ("擦", "wipe", "sweep", "scrub"),
}
_LEVEL_WORDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("speed", "slow"): ("慢", "slowly", "slow", "稳稳"),
    ("speed", "fast"): ("快", "quickly", "quick", "fast", "rapid"),
    ("grip", "gentle"): ("轻轻", "轻柔", "轻", "gently", "gentle", "lightly", "softly", "soft"),
    ("grip", "firm"): ("用力", "紧紧", "牢牢", "firmly", "firm", "tightly", "tight"),
}
# The verb that names the task itself (first clause) is not a stage scope:
# "Lift the cube slowly" asks for a slow *motion*, "..., then lift quickly" asks for the lift stage.
TASK_VERB_STAGES = {"lift": ("lift",), "stack": ("place",), "wipe": ("sweep",)}
_NEUTRAL_WORDS = ("中性", "普通", "正常", "默认", "neutral", "normally", "as usual", "default")
_SPLIT = re.compile(r"[,，;；。、]|\bthen\b|\band then\b|\band\b|然后|接着|再|最后|并且|并")


@dataclass(frozen=True, order=True)
class MannerIntent:
    stage: str
    dimension: str  # "speed" | "grip"
    level: str

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "dimension": self.dimension, "level": self.level}


@dataclass(frozen=True)
class ParseResult:
    status: str  # "ok" | "neutral" | "unsupported"
    intents: tuple[MannerIntent, ...]
    reasons: tuple[str, ...] = ()
    semantics_version: str = SEMANTICS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "intents": [i.to_dict() for i in self.intents],
            "reasons": list(self.reasons),
            "semantics_version": self.semantics_version,
        }


def semantics_hash() -> str:
    blob = json.dumps(
        {"v": SEMANTICS_VERSION, "ranges": LEVEL_RANGES, "task_verbs": TASK_VERB_STAGES, "stage": _STAGE_WORDS,
         "level": {f"{k[0]}:{k[1]}": v for k, v in _LEVEL_WORDS.items()}, "neutral": _NEUTRAL_WORDS},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _find_levels(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    rest = text
    # longer words first so that "轻轻" is not also read as "轻"
    items = sorted(((w, key) for key, ws in _LEVEL_WORDS.items() for w in ws), key=lambda x: -len(x[0]))
    for word, key in items:
        if word in rest:
            if key not in found:
                found.append(key)
            rest = rest.replace(word, " ")
    return found


def _find_stages(task: str, text: str) -> list[str]:
    valid = set(stage_ids(task))
    found: list[str] = []
    for stage, words in _STAGE_WORDS.items():
        if stage in valid and any(w in text for w in words):
            found.append(stage)
    return found


def parse_manner_request(task: str, text: str) -> ParseResult:
    raw = (text or "").strip().lower()
    if not raw:
        return ParseResult("unsupported", (), ("empty_request",))
    clauses = [c.strip() for c in _SPLIT.split(raw) if c and c.strip()]
    intents: set[MannerIntent] = set()
    reasons: list[str] = []
    saw_level = False
    for ci, clause in enumerate(clauses):
        levels = _find_levels(clause)
        stages = _find_stages(task, clause)
        if ci == 0:
            stages = [s for s in stages if s not in TASK_VERB_STAGES.get(task, ())]
        if not levels:
            continue
        saw_level = True
        for dim, level in levels:
            if task == "wipe" and dim == "grip":  # the force words name the pressing force of a wipe
                dim = "press"
                if level not in PRESS_LEVELS:
                    reasons.append(f"level_not_supported_for_wipe:press:{level}")
                    continue
            eligible = dimension_stages(task, dim)
            if dim == "grip":
                targets = ["grasp"]
            elif dim == "press":
                targets = list(eligible)
            elif stages:
                targets = [s for s in stages if s in eligible]
                if not targets:
                    reasons.append(f"stage_not_supported_for_{dim}:{'/'.join(stages)}")
                    continue
            else:
                targets = list(eligible)
            for stage in targets:
                intents.add(MannerIntent(stage, dim, level))
    for tok in _unknown_manner_tokens(raw):
        reasons.append(f"unrecognised_manner_word:{tok}")
    # contradictions: same (stage, dimension) with two levels
    seen: dict[tuple[str, str], str] = {}
    for it in sorted(intents):
        key = (it.stage, it.dimension)
        if key in seen and seen[key] != it.level:
            reasons.append(f"contradiction:{it.stage}:{it.dimension}:{seen[key]}/{it.level}")
        seen[key] = it.level
    if reasons:
        return ParseResult("unsupported", tuple(sorted(intents)), tuple(reasons))
    if not saw_level:
        if any(w in raw for w in _NEUTRAL_WORDS) or not _has_manner_hint(raw):
            return ParseResult("neutral", (), ())
        return ParseResult("unsupported", (), ("unrecognised_manner_words",))
    return ParseResult("ok", tuple(sorted(intents)), ())


_HINT = re.compile(r"(gentl|careful|smooth|steady|stable|硬|柔|稳|小心|平滑|急|缓)")
_ADVERB = re.compile(r"\b[a-z]{3,}ly\b")
_KNOWN_LY = {w for ws in _LEVEL_WORDS.values() for w in ws if w.endswith("ly")} | {"normally", "only", "family"}


def _unknown_manner_tokens(raw: str) -> list[str]:
    return sorted({t for t in _ADVERB.findall(raw) if t not in _KNOWN_LY})


def _has_manner_hint(raw: str) -> bool:
    return bool(_HINT.search(raw))


# ----------------------------- candidate side --------------------------------
def default_candidate(task: str, intents: Iterable[MannerIntent]) -> dict[str, Any]:
    """Deterministic rule proposer: level -> pre-registered default command."""
    if task == "wipe":
        lv = next((i.level for i in intents if i.dimension == "speed"), "normal")
        pr = next((i.level for i in intents if i.dimension == "press"), "normal")
        return {"stages": [{"stage_id": "sweep", "level": lv, "v_max_mps": LEVEL_RANGES["speed_wipe"][lv][2]}],
                "press": {"level": pr, "target_n": LEVEL_RANGES["press"][pr][2]}}
    speed = {i.stage: i.level for i in intents if i.dimension == "speed"}
    grip = next((i.level for i in intents if i.dimension == "grip"), "normal")
    return {
        "stages": [
            {"stage_id": sid, "level": speed.get(sid, "normal"),
             "v_max_mps": LEVEL_RANGES["speed"][speed.get(sid, "normal")][2]}
            for sid in stage_ids(task) if sid != "grasp"
        ],
        "grip": {"level": grip, "target_lo_n": LEVEL_RANGES["grip"][grip][2]},
    }


def candidate_intents(task: str, candidate: Mapping[str, Any],
                      grip_scale: float = 1.0) -> tuple[tuple[MannerIntent, ...], list[str]]:
    """Declared non-normal levels of a candidate, plus structural problems."""
    problems: list[str] = []
    out: list[MannerIntent] = []
    if not isinstance(candidate, Mapping):
        return (), ["candidate_not_mapping"]
    if task == "wipe":
        return _wipe_candidate_intents(candidate)
    stages = candidate.get("stages")
    if not isinstance(stages, list):
        problems.append("candidate_missing_stages")
        stages = []
    valid = stage_ids(task)
    seen: set[str] = set()
    for entry in stages:
        sid = entry.get("stage_id") if isinstance(entry, Mapping) else None
        if sid not in valid or sid == "grasp":
            problems.append(f"candidate_unknown_stage:{sid}")
            continue
        if sid in seen:
            problems.append(f"candidate_duplicate_stage:{sid}")
        seen.add(sid)
        level, v = entry.get("level"), entry.get("v_max_mps")
        if level not in SPEED_LEVELS:
            problems.append(f"candidate_bad_level:{sid}:{level}")
            continue
        lo, hi, _ = LEVEL_RANGES["speed"][level]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= float(v) <= hi):
            problems.append(f"command_outside_level_range:{sid}:{level}:{v}")
        if level != "normal":
            out.append(MannerIntent(sid, "speed", level))
    for sid in valid:
        if sid != "grasp" and sid not in seen:
            problems.append(f"candidate_missing_stage:{sid}")
    grip = candidate.get("grip")
    if not isinstance(grip, Mapping):
        problems.append("candidate_missing_grip")
    else:
        level, t = grip.get("level"), grip.get("target_lo_n")
        if level not in GRIP_LEVELS:
            problems.append(f"candidate_bad_level:grasp:{level}")
        else:
            # grip_scale is a scene-level audit/oracle parameter; the lexical gate
            # remains a check against the nominal level band, with a widened upper
            # envelope to avoid rejecting valid heavy-scene grasps.
            lo, hi, _ = grip_gate_range(level)
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not (lo <= float(t) <= hi):
                problems.append(f"command_outside_level_range:grasp:{level}:{t}")
            if level != "normal":
                out.append(MannerIntent("grasp", "grip", level))
    return tuple(sorted(out)), problems


def _in_range(v: Any, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi


def _wipe_candidate_intents(candidate: Mapping[str, Any]) -> tuple[tuple[MannerIntent, ...], list[str]]:
    problems: list[str] = []
    out: list[MannerIntent] = []
    stages = candidate.get("stages")
    entries = [e for e in stages if isinstance(e, Mapping)] if isinstance(stages, list) else []
    if not isinstance(stages, list):
        problems.append("candidate_missing_stages")
    ids = [e.get("stage_id") for e in entries]
    for sid in ids:
        if sid != "sweep":
            problems.append(f"candidate_unknown_stage:{sid}")
    if ids.count("sweep") > 1:
        problems.append("candidate_duplicate_stage:sweep")
    sweep = next((e for e in entries if e.get("stage_id") == "sweep"), None)
    if sweep is None:
        problems.append("candidate_missing_stage:sweep")
    else:
        level, v = sweep.get("level"), sweep.get("v_max_mps")
        if level not in SPEED_LEVELS:
            problems.append(f"candidate_bad_level:sweep:{level}")
        else:
            lo, hi, _ = LEVEL_RANGES["speed_wipe"][level]
            if not _in_range(v, lo, hi):
                problems.append(f"command_outside_level_range:sweep:{level}:{v}")
            if level != "normal":
                out.append(MannerIntent("sweep", "speed", level))
    press = candidate.get("press")
    if not isinstance(press, Mapping):
        problems.append("candidate_missing_press")
    else:
        level, t = press.get("level"), press.get("target_n")
        if level not in PRESS_LEVELS:
            problems.append(f"candidate_bad_level:sweep:{level}")
        else:
            lo, hi, _ = LEVEL_RANGES["press"][level]
            if not _in_range(t, lo, hi):
                problems.append(f"command_outside_level_range:press:{level}:{t}")
            if level != "normal":
                out.append(MannerIntent("sweep", "press", level))
    return tuple(sorted(out)), problems


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


def semantic_gate(task: str, request: ParseResult, candidate: Mapping[str, Any],
                  grip_scale: float = 1.0) -> GateResult:
    """Compare the immutable request with the candidate; reject any deviation."""
    if request.status == "unsupported":
        return GateResult(False, ("request_unsupported",) + tuple(request.reasons))
    cand, problems = candidate_intents(task, candidate, grip_scale)
    reasons = list(problems)
    want = {(i.stage, i.dimension): i.level for i in request.intents}
    got = {(i.stage, i.dimension): i.level for i in cand}
    for key, level in sorted(want.items()):
        if key not in got:
            other = [k for k in got if k[1] == key[1] and got[k] == level]
            reasons.append(f"missing:{key[0]}:{key[1]}:{level}" + (":moved_to_" + ",".join(k[0] for k in other) if other else ""))
        elif got[key] != level:
            reasons.append(f"wrong_level:{key[0]}:{key[1]}:{level}!={got[key]}")
    for key, level in sorted(got.items()):
        if key not in want:
            reasons.append(f"extra:{key[0]}:{key[1]}:{level}")
    return GateResult(not reasons, tuple(reasons))
