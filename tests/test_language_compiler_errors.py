"""Error handling of the language -> program/manner path (offline, no model calls).

Each case is a way an answer can be wrong; the contract is that it is reported, never silently repaired.
"""
from __future__ import annotations

import pytest

from synthesis_v2.language_compiler import (
    MannerBinding, Semantics, check_program, check_semantics, clause_coverage, lexicon_semantics,
    parse_program, parse_semantics,
)
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances

TWO = TaskProgram(
    "two_cube_putback", "robosuite_native", "stack",
    {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
    chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
    source="composed")
GA, GB = "cubeA.pick_1.grasp", "cubeB.pick_1.grasp"


def test_ambiguous_stage_without_object_is_reported_not_guessed():
    sem = lexicon_semantics("grasp gently", TWO)
    assert not sem.bindings and not sem.complete()
    assert sem.ambiguous and set(sem.ambiguous[0]["candidates"]) == {GA, GB}


def test_unknown_object_is_not_bound_to_another_object():
    sem = lexicon_semantics("grasp the cubec gently", TWO)  # cubeC does not exist
    assert not sem.bindings and not sem.complete()
    errs = check_semantics(parse_semantics(
        {"bindings": [{"instance_ids": [GA], "dimension": "grip", "level": "gentle", "object_role": "cubeC"}]},
        "x", TWO), TWO)
    assert any(e.startswith("object_mismatch") for e in errs)


def test_object_scope_binds_only_the_named_object_and_repeats_stay_separate():
    sem = lexicon_semantics("grasp cubeA gently, grasp cubeB firmly", TWO)
    got = {b.instance_ids: b.level for b in sem.bindings}
    assert got == {(GA,): "gentle", (GB,): "firm"} and sem.complete()
    assert check_semantics(sem, TWO) == []


def test_two_levels_on_the_same_stage_conflict():
    sem = Semantics("x", (MannerBinding((GA,), "grip", "gentle"), MannerBinding((GA,), "grip", "firm")))
    assert any(e.startswith("conflicting_binding") for e in check_semantics(sem, TWO))


def test_missed_manner_clause_is_visible_in_coverage():
    gold = lexicon_semantics("grasp cubeA gently, grasp cubeB firmly", TWO)
    got = Semantics("x", gold.bindings[:1])  # the model answered only the first clause
    c = clause_coverage(got, gold)
    assert c["matched"] == 1 and len(c["missing"]) == 1 and not c["exact"]


def test_wrong_scope_is_counted_separately_from_missing():
    gold = Semantics("x", (MannerBinding((GA,), "grip", "gentle"),))
    got = Semantics("x", (MannerBinding((GB,), "grip", "gentle"),))
    c = clause_coverage(got, gold)
    assert c["wrong_scope"] == 1 and c["matched"] == 0 and not c["exact"]


def test_unsupported_dimension_on_stage_is_reported():
    sem = Semantics("x", (MannerBinding((GA,), "press", "firm"),))
    assert any(e.startswith("dimension_not_on_stage") for e in check_semantics(sem, TWO))
    lex = lexicon_semantics("grasp cubeA hard", TWO)  # 'hard' is a press-force word; grasp has no such dimension
    assert not lex.bindings and not lex.complete()


@pytest.mark.parametrize("dim,level,prefix", [
    ("teleport", "gentle", "unknown_dimension"),
    ("grip", "tender_loving", "unknown_level"),
])
def test_unknown_dimension_or_level_rejected(dim, level, prefix):
    errs = check_semantics(Semantics("x", (MannerBinding((GA,), dim, level),)), TWO)
    assert any(e.startswith(prefix) for e in errs)


def test_unknown_instance_and_unbound_clause_rejected():
    errs = check_semantics(Semantics("x", (MannerBinding(("cubeZ.pick_1.grasp",), "grip", "gentle"),
                                            MannerBinding((), "grip", "gentle"))), TWO)
    assert any(e.startswith("unknown_instance") for e in errs) and any(e.startswith("unbound_clause") for e in errs)


@pytest.mark.parametrize("bad", [None, [], "grip gentle", 3])
def test_non_object_answers_raise(bad):
    with pytest.raises(ValueError):
        parse_semantics(bad, "x", TWO)
    with pytest.raises(ValueError):
        parse_program(bad)


def test_malformed_binding_raises():
    with pytest.raises(ValueError):
        parse_semantics({"bindings": [{"dimension": "grip"}]}, "x", TWO)


def test_valid_program_roundtrips_through_parse_program():
    assert check_program(parse_program(TWO.to_dict())) == []


def test_program_with_unsupported_backend_is_rejected():
    d = TWO.to_dict()
    d["backend"] = "isaac"
    assert any(e.startswith("unsupported_backend") for e in check_program(parse_program(d)))


def test_program_with_unsupported_primitive_is_rejected():
    d = TWO.to_dict()
    d["instances"][0]["primitive"] = "teleport"
    try:
        errs = check_program(parse_program(d))
    except (ValueError, KeyError):
        return  # rejected at parse time: nothing unsupported gets through
    assert any(e.startswith("unsupported_primitive") for e in errs)


def test_program_with_duplicate_ids_or_forward_preds_is_rejected():
    d = TWO.to_dict()
    d["instances"][1]["instance_id"] = d["instances"][0]["instance_id"]
    assert check_program(parse_program(d))
    d = TWO.to_dict()
    d["instances"][0]["preds"] = [d["instances"][-1]["instance_id"]]
    assert check_program(parse_program(d))


def test_program_with_unbound_role_is_rejected():
    d = TWO.to_dict()
    d["instances"][0]["object_role"] = "cubeX"
    assert any(e.startswith("unbound_object") for e in check_program(parse_program(d)))
