"""Language scope rules for Chinese speed requests (language_scope.zh.v2; v1 is superseded, its spec/results are kept).

One question only: *which stage instances does a speed word in this request apply to?*  Answered by code from the ORIGINAL
sentence, never from the model's clause span or claim; the semantic gate compares the model's bindings with this answer, and
anything the rules cannot determine stops with a clarification request.  A word the vocabulary does not cover is a parsing
coverage gap, never evidence for a global request.

Rules, applied per clause part (split on ，,；;。、) and per speed word:
  R1 local, speed word first : the speed word is directly followed by a stage word (optional 地 / 一点 / 一些 / 点 between)
                               -> that stage only.          "缓慢放入后部隔间"  "慢一点放入"  "缓慢地搬运"
  R2 local, stage word first : a stage word, then optionally one of 入/到/进/下/上, then optionally 得 or 时 / 的时候
                               (optionally followed by 要/用/采用/使用/保持/改用/改为/改成), then the speed word -> that stage only.
                               The connective may be empty: "放入慢一点" is local to place.  If R1 and R2 name different stages
                               for the same speed word, the scope is undetermined (R4).
  R3 global                  : ONLY an explicit global cue (整个过程 / 全程 / 整体 / ...) in the same clause part.  A clause
                               without any stage word and without a global cue is NOT global: it is undetermined (R4).
  R4 undetermined            : the rules cannot reliably fix the scope (no stage word and no global cue; stage words not related to
                               the speed word; several stages possible; negation in front of the speed word; stage without a
                               speed dimension).  It is a limit of this system, not a claim that the sentence is ambiguous
                               in Chinese.  Result: clarification, no plan.
Request level (resolve_speed_scope):
  * a negated speed word (不要/别/不用/…) is never turned into a level (R4);
  * a global level can be overridden for a stage by an explicit local clause only if that clause carries an exception cue
    (但 / 但是 / 不过 / 除了 / 唯独 / 唯有 / 其中): "全程慢一些，但放下时用正常速度" -> every speed stage slow, place normal;
  * two different levels for the same scope (global/global, stage/stage) or a local-vs-global difference without an exception
    cue is a conflict; clause order never decides ("放下时慢一点，放下时快一点" is a conflict);
  * "normal" (正常 / 常速) is a speed level like any other; the plan keeps the neutral value there.
Only speed words are handled; grip / press words are not scoped here.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence

from synthesis_v2.manner_request import _STAGE_WORDS

SCOPE_VERSION = "language_scope.zh.v2"
SPEED_WORDS_ZH = (("缓慢", "slow"), ("慢慢", "slow"), ("慢", "slow"), ("快速", "fast"), ("迅速", "fast"), ("快", "fast"),
                  ("正常", "normal"), ("常速", "normal"))
GLOBAL_CUES = ("整个过程", "整个动作", "整个流程", "全过程", "全程", "整体", "一路", "始终", "每个阶段", "所有阶段")
EXCEPTION_CUES = ("但是", "但", "不过", "除了", "唯独", "唯有", "其中")
_GAP_AFTER_LEVEL = ("", "地", "一点", "一些", "点")
_GAP_BEFORE_LEVEL = re.compile(r"^[入到进下上]?(?:得|(?:的?时候|时)(?:要|用|采用|使用|保持|改用|改为|改成)?)?$")
_NEGATION = re.compile(r"(?:不要|不用|不必|不需要|无需|无须|不能|不可以|不可|切勿|勿|(?<![特分区类个告])别|不(?!过)|没有)[^，,；;。、]{0,4}$")
_PART_SPLIT = re.compile(r"[，,；;。、]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
# stage words that exist in the shared table and are written in Chinese
_STAGE_ZH = sorted(((w, st) for st, ws in _STAGE_WORDS.items() for w in ws if _CJK.search(w)), key=lambda x: -len(x[0]))


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def scope_rules_hash() -> str:
    """Hash of the ACTUAL tables and patterns (including the stage words taken from manner_request); the spec cites this value
    instead of keeping its own word list."""
    blob = {"v": SCOPE_VERSION, "speed": SPEED_WORDS_ZH, "global": GLOBAL_CUES, "exception": EXCEPTION_CUES, "stage": _STAGE_ZH,
            "gap_after": _GAP_AFTER_LEVEL, "gap_before": _GAP_BEFORE_LEVEL.pattern, "negation": _NEGATION.pattern}
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


def clause_speed_scope(text: str) -> list[dict[str, Any]]:
    """One entry per speed word found in ``text``: kind global | stage | undetermined (+ rule, stage, reason, exception flag)."""
    table = sorted(SPEED_WORDS_ZH, key=lambda x: -len(x[0]))
    res: list[dict[str, Any]] = []
    for part in _PART_SPLIT.split(text or ""):
        levels, stages = _hits(part, table), _hits(part, _STAGE_ZH)
        exc = any(c in part for c in EXCEPTION_CUES)
        for ls, le, lvl in levels:
            e: dict[str, Any] = {"word": part[ls:le], "level": lvl, "part": part, "exception": exc}
            if _NEGATION.search(part[:ls]):
                e.update(kind="undetermined", rule="R4", stage=None, reason="negation")
            elif any(c in part for c in GLOBAL_CUES):
                e.update(kind="global", rule="R3", stage=None)
            elif not stages:
                e.update(kind="undetermined", rule="R4", stage=None, reason="no_stage_word_and_no_global_cue")
            else:
                cand: dict[str, str] = {}
                for a, b, st in stages:
                    if a >= le and part[le:a] in _GAP_AFTER_LEVEL:
                        cand.setdefault(st, "R1")
                for a, b, st in stages:
                    if b <= ls and _GAP_BEFORE_LEVEL.match(part[b:ls]):
                        cand.setdefault(st, "R2")
                if len(cand) == 1:
                    (st, rule), = cand.items()
                    e.update(kind="stage", rule=rule, stage=st)
                elif cand:
                    e.update(kind="undetermined", rule="R4", stage=None, reason="several_stages_possible", stages=sorted(cand))
                else:
                    e.update(kind="undetermined", rule="R4", stage=None, reason="stage_word_not_linked_to_speed_word",
                             stage_words=[part[a:b] for a, b, _ in stages])
            res.append(e)
    return res


def resolve_speed_scope(text: str, program) -> dict[str, Any]:
    """Request-level result: status none | ok | needs_clarification | conflict, the per-instance ``effective`` levels (ok only)
    and the instances where an explicit local exception overrides a global level."""
    hits = clause_speed_scope(text)
    out: dict[str, Any] = {"scope_version": SCOPE_VERSION, "rules_hash": scope_rules_hash(), "status": "none", "reasons": [],
                           "effective": {}, "exceptions": [], "hits": hits}
    if not hits:
        return out
    bad = [h for h in hits if h["kind"] == "undetermined"]
    if bad:
        out.update(status="needs_clarification",
                   reasons=[{k: h[k] for k in ("word", "part", "reason") + (("stages",) if "stages" in h else ())} for h in bad])
        return out
    speed = [it for it in program.instances if "speed" in it.dims]
    glev = sorted({h["level"] for h in hits if h["kind"] == "global"})
    loc: dict[str, list[dict]] = {}
    for h in hits:
        if h["kind"] == "stage":
            loc.setdefault(h["stage"], []).append(h)
    conflicts: list[dict[str, Any]] = []
    if len(glev) > 1:
        conflicts.append({"scope": "global", "levels": glev})
    for st, hs in loc.items():
        lv = sorted({h["level"] for h in hs})
        if len(lv) > 1:
            conflicts.append({"scope": st, "levels": lv})
    if conflicts:
        out.update(status="conflict", reasons=conflicts)
        return out
    eff: dict[str, str] = {it.instance_id: glev[0] for it in speed} if glev else {}
    exc: list[str] = []
    for st, hs in loc.items():
        lvl = hs[0]["level"]
        ids = [it.instance_id for it in speed if it.primitive == st]
        if not ids:
            out.update(status="needs_clarification", reasons=[{"stage": st, "reason": "dimension_not_on_stage"}])
            return out
        for iid in ids:
            if iid in eff and eff[iid] != lvl:
                if any(h["exception"] for h in hs):
                    exc.append(iid)
                else:
                    conflicts.append({"scope": st, "levels": sorted({eff[iid], lvl}), "reason": "local_differs_from_global_without_exception_cue"})
            eff[iid] = lvl
    if conflicts:
        out.update(status="conflict", reasons=conflicts)
        return out
    out.update(status="ok", effective=eff, exceptions=sorted(set(exc)))
    return out


def model_speed_map(bindings: Sequence[tuple[Sequence[str], str]]) -> tuple[dict[str, str], list[str]]:
    """Per-instance level from the model's speed bindings; the narrower binding wins over a broader one (an exception), equally
    narrow bindings with different levels are reported as conflicts."""
    per: dict[str, list[tuple[int, str]]] = {}
    for ids, lvl in bindings:
        for i in ids:
            per.setdefault(i, []).append((len(ids), lvl))
    out: dict[str, str] = {}; conflicts: list[str] = []
    for i, xs in per.items():
        mn = min(n for n, _ in xs); lv = {l for n, l in xs if n == mn}
        if len(lv) > 1:
            conflicts.append(i)
        else:
            out[i] = lv.pop()
    return out, sorted(conflicts)


def compare_speed_bindings(effective: Mapping[str, str], bindings: Sequence[tuple[Sequence[str], str]]) -> Optional[dict[str, Any]]:
    """None if the model's bindings reproduce the code's effective levels (normal == unbound), else the differences."""
    got, conflicts = model_speed_map(bindings)
    want = {i: l for i, l in effective.items() if l != "normal"}
    got = {i: l for i, l in got.items() if l != "normal"}
    missing = sorted(set(want) - set(got)); extra = sorted(set(got) - set(want))
    wrong = sorted(i for i in want if i in got and want[i] != got[i])
    if missing or extra or wrong or conflicts:
        return {"missing": missing, "extra": extra, "wrong_level": wrong, "model_conflict": conflicts}
    return None
