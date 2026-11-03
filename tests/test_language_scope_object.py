"""Step 31: object layer of the Chinese speed scope + English negation in the level-agreement check.
Model answers are hand-written fixtures (NOT LLM output)."""
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mt"))
from synthesis_v2.language_scope import resolve_speed_scope
from synthesis_v2.request_intake import admit, parse_intake

s7 = pytest.importorskip("s7_task_config")
PROG = s7.make_program(0)
BASE = "把字母汤和番茄酱都放进篮子里"
PH = {"字母汤": ["alphabet_soup"], "番茄酱": ["tomato_sauce"]}
TR = {r: f"{r}.pick_1.transport" for r in ("alphabet_soup", "tomato_sauce")}


def eff(text, phrases=PH):
    r = resolve_speed_scope(text, PROG, phrases)
    return r["status"], {k: v for k, v in r["effective"].items()}, r


@pytest.mark.parametrize("text,want", [
    (BASE + "，慢慢地搬运番茄酱", {TR["tomato_sauce"]: "slow"}),                      # O1 same part
    (BASE + "，番茄酱容易碎，要轻拿并且慢慢搬运", {TR["tomato_sauce"]: "slow"}),        # O1 topic of preceding part
    (BASE + "，慢慢搬运", {TR["alphabet_soup"]: "slow", TR["tomato_sauce"]: "slow"}),   # O2 base names both
    (BASE + "，两个都慢慢搬运", {TR["alphabet_soup"]: "slow", TR["tomato_sauce"]: "slow"}),
    (BASE + "，慢慢搬运番茄酱，快速搬运字母汤", {TR["tomato_sauce"]: "slow", TR["alphabet_soup"]: "fast"}),
])
def test_object_layer(text, want):
    st, e, _ = eff(text)
    assert st == "ok" and e == want


def test_without_phrases_v2_behaviour_is_unchanged():
    r = resolve_speed_scope(BASE + "，慢慢地搬运番茄酱", PROG)
    assert r["status"] == "ok" and set(r["effective"]) == set(TR.values()) and "object_scope_version" not in r
    assert resolve_speed_scope(BASE + "，慢慢搬运番茄酱，快速搬运字母汤", PROG)["status"] == "conflict"


def test_same_object_two_levels_is_conflict_and_undetermined_stays():
    assert eff(BASE + "，慢慢搬运番茄酱，快速搬运番茄酱")[0] == "conflict"
    assert eff(BASE + "，慢一点")[0] == "needs_clarification"


def test_o3_unknown_phrase_is_one_object_grounded_by_model():
    st, e, r = eff(BASE + "，慢慢搬运那个红色罐子", {})
    assert st == "ok" and e == {} and r["one_object"][0]["stage"] == "transport"


def _ans(steps, bindings):
    return {"task": {"restated": True, "steps": steps, "order": []}, "bindings": bindings, "ambiguous": [], "unsupported": []}


STEPS = [{"action": "put", "object_phrase": "字母汤", "object_role": "alphabet_soup", "target_phrase": "篮子", "target_role": "basket"},
         {"action": "put", "object_phrase": "番茄酱", "object_role": "tomato_sauce", "target_phrase": "篮子", "target_role": "basket"}]


def _b(ids, lvl, role, cl, dim="speed"):
    return {"instance_ids": ids, "dimension": dim, "level": lvl, "object_role": role, "clause": cl, "basis": "explicit_object", "antecedent": None}


def test_admit_object_scope_accepts_one_object_and_rejects_both():
    text = BASE + "，慢慢地搬运番茄酱"
    one = admit(parse_intake(_ans(STEPS, [_b([TR["tomato_sauce"]], "slow", "tomato_sauce", "慢慢地搬运番茄酱")]), text), PROG)
    assert one["status"] == "accept", one
    both = admit(parse_intake(_ans(STEPS, [_b(list(TR.values()), "slow", None, "慢慢地搬运番茄酱")]), text), PROG)
    assert both["status"] != "accept"
    assert any(g.get("gap") == "language_scope_mismatch" and g["extra"] == [TR["alphabet_soup"]] for g in both["guards"])


def test_english_negated_level_word_is_not_evidence():
    text = "Put both the alphabet soup and the tomato sauce in the basket, taking care not to squeeze the tomato sauce hard"
    cl = "taking care not to squeeze the tomato sauce hard"
    d = admit(parse_intake(_ans([], [_b(["tomato_sauce.pick_1.grasp"], "gentle", "tomato_sauce", cl, "grip")]), text), PROG)
    assert not any(g.get("gap") == "level_disagrees_with_clause" for g in d["guards"])
    # without negation the disagreement is still caught
    text2 = "Put both the alphabet soup and the tomato sauce in the basket, squeezing the tomato sauce hard"
    d2 = admit(parse_intake(_ans([], [_b(["tomato_sauce.pick_1.grasp"], "gentle", "tomato_sauce", "squeezing the tomato sauce hard", "grip")]), text2), PROG)
    assert any(g.get("gap") == "level_disagrees_with_clause" for g in d2["guards"])


def test_shared_head_noun_without_plural_cue_is_one_object():
    prog8 = PROG   # role names do not matter for the scope layer: two roles share the phrase "摩卡壶"
    ph = {"摩卡壶": ["alphabet_soup", "tomato_sauce"]}
    r = resolve_speed_scope("把摩卡壶放到炉子上，右边的摩卡壶容易碎，要慢慢搬运", prog8, ph)
    assert r["status"] == "ok" and r["effective"] == {} and r["one_object"][0]["stage"] == "transport"
    r = resolve_speed_scope("把摩卡壶都放到炉子上，慢慢搬运", prog8, ph)
    assert r["status"] == "ok" and set(r["effective"]) == set(TR.values())
