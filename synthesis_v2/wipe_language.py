"""Outcome-first instructions for the wipe task, with explicit and implicit manner cues.

Every prompt starts with the desired END STATE (a neutral outcome sentence).  The manner (sweep speed, pressing force)
is then stated either with lexicon words (``explicit``), by a plain paraphrase without lexicon words
(``outcome_hint``: "I'm in a hurry", "the dirt is stubborn") or only by situation (``outcome_implicit``).

The label of a prompt is the registered ``MannerIntent`` set it was generated from; it is not parsed back from the
text.  For the ``explicit`` tier the lexicon parser must agree with the label (checked in ``build_prompt``); for the
other tiers the text is checked to contain NO lexicon level word (so a hint can never be an explicit request by
accident).  Whether a human reads an implicit cue as the intended manner is NOT validated here.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from synthesis_v2.manner_request import MannerIntent, _find_levels, parse_manner_request

TIERS = ("explicit", "outcome_hint", "outcome_implicit")
TIER_WEIGHTS = (0.30, 0.30, 0.40)

# neutral outcome sentences (no manner word, no "-ly" adverb so that they stay parseable)
BASES = (
    "I want the table free of all dirt.",
    "The table should end up clean and bare.",
    "Make sure not a speck of dirt is left on the table.",
    "Get the table to a clean state.",
    "Clear the dirt off the table surface.",
    "The dirty patch on the table needs to be gone.",
    "Keep the table spotless.",
    "Remove all the grime from the table.",
)

SPEED_CUES = {
    "fast": {
        "explicit": ("Do it quickly.", "Be fast about it.", "Wipe quickly.", "Use quick strokes.", "Make it fast."),
        "outcome_hint": ("I'm in a hurry.", "Time is short, so be brisk about it.", "We are running late.",
                         "I need it done right now.", "Hurry up with it."),
        "outcome_implicit": ("The guests ring the bell in a minute.", "My meeting starts in two minutes.",
                             "Someone is already waiting for this table.", "The next shift needs the table the moment you are done.",
                             "The timer on this job is almost out."),
    },
    "slow": {
        "explicit": ("Do it slowly.", "Take it slow.", "Wipe slowly.", "Use slow strokes.", "Go slow."),
        "outcome_hint": ("There is no rush.", "Take your time.", "I'm in no hurry at all.", "No need to hurry.",
                         "Don't rush it."),
        "outcome_implicit": ("The whole afternoon is free.", "Nobody is waiting on this table.", "Dinner is hours away.",
                             "There is plenty of time before anyone arrives.", "Nothing else is scheduled today."),
    },
}
PRESS_CUES = {
    "firm": {
        "explicit": ("Press firmly.", "Wipe firmly.", "Use firm pressure.", "Keep the pressure firm.", "Keep a firm press."),
        "outcome_hint": ("The dirt is dried on and stubborn.", "It is caked on, so put some weight into it.",
                         "A light touch won't shift this dirt.", "Lean into it; it is stuck on hard.",
                         "Don't just brush over it, bear down on it."),
        "outcome_implicit": ("That stuff has been baking onto the table since last week.",
                             "The last person who tried only smeared it around.", "Some of it has set like glue.",
                             "It has been there so long it feels like part of the table.",
                             "A damp cloth did nothing to it yesterday."),
    },
}
SPEED_LEVELS = ("slow", "normal", "fast")
PRESS_LEVELS = ("normal", "firm")
COMBOS = tuple((s, p) for s in SPEED_LEVELS for p in PRESS_LEVELS)


def combo_intents(speed: str, press: str) -> tuple[MannerIntent, ...]:
    out = []
    if speed != "normal":
        out.append(MannerIntent("sweep", "speed", speed))
    if press != "normal":
        out.append(MannerIntent("sweep", "press", press))
    return tuple(sorted(out))


def _h(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def base_for_seed(seed: int) -> str:
    return BASES[_h("base", seed) % len(BASES)]


def build_prompt(seed: int, speed: str, press: str) -> tuple[str, Optional[str]]:
    """(prompt, language_style).  Neutral combos return the seed's base sentence and ``None``."""
    base = base_for_seed(seed)
    intents = combo_intents(speed, press)
    if not intents:
        return base, None
    r = _h("tier", seed, speed, press) % 1000 / 1000.0
    tier = TIERS[0] if r < TIER_WEIGHTS[0] else TIERS[1] if r < TIER_WEIGHTS[0] + TIER_WEIGHTS[1] else TIERS[2]
    cues = []
    if speed != "normal":
        c = SPEED_CUES[speed][tier]
        cues.append(c[_h("s", seed, speed, press) % len(c)])
    if press != "normal":
        c = PRESS_CUES[press][tier]
        cues.append(c[_h("p", seed, speed, press) % len(c)])
    if len(cues) == 2 and _h("order", seed, speed, press) % 2:
        cues.reverse()
    text = " ".join([base] + cues)
    if tier == "explicit":
        got = parse_manner_request("wipe", text)
        if got.status != "ok" or tuple(sorted(got.intents)) != intents:
            raise AssertionError(f"explicit prompt does not parse to its label: {text!r} -> {got.to_dict()}")
    elif _find_levels(text.lower()):
        raise AssertionError(f"{tier} prompt contains a lexicon level word: {text!r}")
    return text, tier


def check_bank() -> None:
    """Every sentence of every bank must be consistent with its tier (raises otherwise)."""
    for b in BASES:
        assert not _find_levels(b.lower()), b
        assert parse_manner_request("wipe", b).status == "neutral", b
    for bank, key in ((SPEED_CUES, "speed"), (PRESS_CUES, "press")):
        for lvl, tiers in bank.items():
            for tier, sents in tiers.items():
                for s in sents:
                    if tier == "explicit":
                        p = parse_manner_request("wipe", BASES[0] + " " + s)
                        assert p.status == "ok" and [i.level for i in p.intents] == [lvl] and p.intents[0].dimension == key, (s, p)
                    else:
                        assert not _find_levels(s.lower()), s
