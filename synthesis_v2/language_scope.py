"""Language scope rules for Chinese speed requests (step 17, language entry of a registered task).

One question only: *which stage instances does a speed word in this clause apply to?*  Answered by code, never by the model,
so the semantic gate can check the model's binding against it and nothing downstream can widen a local request.

Rules (``SCOPE_VERSION``), applied per clause part (split on ，,；;。、) and per speed word:
  R1 local, adverb first : the speed word is directly followed by a stage word (optional 地 / 一点 / 一些 / 点 between)
                           -> only that stage type.            "缓慢放入后部隔间"  "慢一点放入"  "缓慢地搬运"
  R2 local, verb first   : a stage word, then 得 / 时 / 的时候 (optionally 要), then the speed word -> that stage type.
                           "放入的时候慢一点"  "放得慢一点"
  R3 global              : a global cue (整个过程 / 全程 / ...) is in the part, or the part names no stage word at all
                           -> every stage instance that supports speed.             "整个过程都慢一些"  "慢一点"
  R4 undetermined        : stage words are present but neither R1 nor R2 relates them to the speed word -> the request is
                           ambiguous and must be clarified; nothing is guessed.
A "normal" word (正常 / 常速) is a speed level like any other (R1-R4 apply); the plan keeps the neutral value there.
Only speed words are handled; grip / press words are not scoped here.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from synthesis_v2.manner_request import _STAGE_WORDS

SCOPE_VERSION = "language_scope.zh.v1"
SPEED_WORDS_ZH = (("缓慢", "slow"), ("慢慢", "slow"), ("慢", "slow"), ("快速", "fast"), ("迅速", "fast"), ("快", "fast"),
                  ("正常", "normal"), ("常速", "normal"))
GLOBAL_CUES = ("整个过程", "整个动作", "整个流程", "全过程", "全程", "整体", "一路", "始终", "每个阶段", "所有阶段")
_GAP_AFTER_LEVEL = ("", "地", "一点", "一些", "点")
_GAP_BEFORE_LEVEL = re.compile(r"^[入到进下上]?(得|的?时候要?|时要?)?$")
_PART_SPLIT = re.compile(r"[，,；;。、]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
# stage words that exist in the shared table and are written in Chinese
_STAGE_ZH = sorted(((w, st) for st, ws in _STAGE_WORDS.items() for w in ws if _CJK.search(w)), key=lambda x: -len(x[0]))


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def scope_rules_hash() -> str:
    blob = {"v": SCOPE_VERSION, "speed": SPEED_WORDS_ZH, "global": GLOBAL_CUES, "stage": _STAGE_ZH,
            "gap_after": _GAP_AFTER_LEVEL, "gap_before": _GAP_BEFORE_LEVEL.pattern}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _hits(part: str, table) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []
    for w, tag in table:                      # table is ordered longest word first
        for m in re.finditer(re.escape(w), part):
            if any(m.start() < b and a < m.end() for a, b in taken):
                continue
            taken.append((m.start(), m.end())); out.append((m.start(), m.end(), tag))
    return sorted(out)


def clause_speed_scope(clause: str) -> list[dict[str, Any]]:
    """One entry per speed word found in ``clause``: ``{"word","level","kind": global|stage|undetermined,"stage",...}``."""
    table = sorted(SPEED_WORDS_ZH, key=lambda x: -len(x[0]))
    res: list[dict[str, Any]] = []
    for part in _PART_SPLIT.split(clause or ""):
        levels, stages = _hits(part, table), _hits(part, _STAGE_ZH)
        for ls, le, lvl in levels:
            e: dict[str, Any] = {"word": part[ls:le], "level": lvl, "part": part}
            if any(c in part for c in GLOBAL_CUES):
                e.update(kind="global", rule="R3", stage=None)
            elif not stages:
                e.update(kind="global", rule="R3", stage=None)
            else:
                after = [s for s in stages if s[0] >= le and part[le:s[0]] in _GAP_AFTER_LEVEL]
                before = [s for s in stages if s[1] <= ls and _GAP_BEFORE_LEVEL.match(part[s[1]:ls])]
                if after:
                    e.update(kind="stage", rule="R1", stage=after[0][2])
                elif before:
                    e.update(kind="stage", rule="R2", stage=before[-1][2])
                else:
                    e.update(kind="undetermined", rule="R4", stage=None,
                             stage_words=[part[a:b] for a, b, _ in stages])
            res.append(e)
    return res


def scope_context(text: str, clause: str) -> str:
    """The part of the ORIGINAL instruction that contains the model's ``clause`` (the model may return only the adverb, e.g.
    "缓慢地"; the scope must still be read from the words around it).  Falls back to the whole instruction when the clause is
    not a verbatim span of it."""
    i = text.find(clause) if clause else -1
    if i < 0:
        return text
    pos = 0
    for part in _PART_SPLIT.split(text):
        j = text.find(part, pos)
        if j <= i < j + len(part) or (not part and j == i):
            return part
        pos = j + len(part)
    return text


def expected_speed_instances(binding_clause: str, level: str, program, object_role: Optional[str] = None,
                             text: Optional[str] = None) -> Optional[set[str]]:
    """Instances a speed binding of ``level`` must cover for this clause, or ``None`` if the scope is undetermined (R4) /
    the clause carries no matching speed word.  Pass the original ``text`` so the scope is read from the sentence itself."""
    ctx = scope_context(text, binding_clause) if text else binding_clause
    hits = [h for h in clause_speed_scope(ctx) if h["level"] == level]
    if not hits or any(h["kind"] == "undetermined" for h in hits):
        return None
    speed = [it for it in program.instances if "speed" in it.dims
             and (object_role is None or object_role in (it.object_role, it.target_role))]
    out: set[str] = set()
    for h in hits:
        out |= {it.instance_id for it in speed if h["kind"] == "global" or it.primitive == h["stage"]}
    return out
