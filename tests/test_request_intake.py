"""Offline tests for request intake: task-text <-> program check and model-independent guards (no LLM calls)."""
from synthesis_v2.language_compiler import MannerBinding, Semantics
from synthesis_v2.request_intake import Intake, TaskStep, admit, check_task
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances

PROG = TaskProgram("t", "libero", "libero:t:0", {"a": "a_1", "b": "b_1", "basket": "basket_1"},
                   chain(pick_place_instances("a", "basket"), pick_place_instances("b", "basket")))
PUT_A, PUT_B = TaskStep("put", "a", "the a", "basket", "the basket"), TaskStep("put", "b", "the b", "basket", "the basket")
GRASP_B = MannerBinding(("b.pick_1.grasp",), "grip", "gentle", "b", "hold b gently")


def intake(text, steps=(PUT_A, PUT_B), order=(), sem=None, bases=(), codes=()):
    sem = sem or Semantics(text, (GRASP_B,))
    bases = bases or tuple({"basis": "explicit_object", "antecedent": ""} for _ in sem.bindings)
    return Intake(text, True, tuple(steps), tuple(order), sem, tuple(bases), tuple(codes))


def test_clean_accept():
    assert admit(intake("put a and b in the basket; hold b gently"), PROG)["status"] == "accept"


def test_absent_object_is_conflict_not_dropped():
    bad = TaskStep("put", None, "the red cup", "basket", "the basket")
    d = admit(intake("put the red cup and b in the basket", steps=(bad, PUT_B)), PROG)
    assert d["status"] == "conflict" and "object_not_in_program" in d["codes"]


def test_target_mismatch_and_order_mismatch():
    t = TaskStep("put", "a", "the a", "b", "the b")
    assert admit(intake("x", steps=(t, PUT_B)), PROG)["status"] == "conflict"
    d = admit(intake("b then a", steps=(PUT_B, PUT_A), order=((0, 1),)), PROG)
    assert d["status"] == "conflict" and "order_mismatch" in d["codes"]


def test_numeric_value_is_never_silently_dropped():
    d = admit(intake("put a and b in the basket; hold b with 5 N"), PROG)
    assert d["status"] == "unsupported" and d["codes"] == ["numeric_level"]
    assert any(g["guard"] == "numeric_value_not_returned_as_unsupported" for g in d["guards"])


def test_binding_basis_must_be_admissible():
    d = admit(intake("hold it gently", bases=({"basis": "task_default_order", "antecedent": ""},)), PROG)
    assert d["status"] == "ambiguous"


def test_cited_antecedent_must_be_in_the_sentence():
    d = admit(intake("hold it gently", bases=({"basis": "antecedent", "antecedent": "the plate"},)), PROG)
    assert d["status"] == "ambiguous" and d["guards"][0]["guard"] == "antecedent_not_in_instruction"
    ok = admit(intake("pick up b. hold it gently", steps=(), bases=({"basis": "antecedent", "antecedent": "b"},)), PROG)
    assert ok["status"] == "accept"


def test_partial_answer_is_not_accepted():
    sem = Semantics("x", (GRASP_B,), unsupported=({"clause": "hold b at 3 cm", "reason": "x"},))
    assert admit(intake("put a and b in the basket", sem=sem, codes=("quantity_not_measured",)), PROG)["status"] == "unsupported"
    sem = Semantics("x", (GRASP_B,), ambiguous=({"clause": "go slowly", "candidates": ["a.pick_1.approach", "b.pick_1.approach"]},))
    assert admit(intake("put a and b in the basket", sem=sem), PROG)["status"] == "ambiguous"


# ---- completeness: a structurally valid answer that leaves out something the sentence states must not be accepted ----
import copy
import pytest
from synthesis_v2.request_intake import parse_intake

P2 = TaskProgram("t0", "libero", "libero:t:0", {"alphabet_soup": "s_1", "tomato_sauce": "t_1", "basket": "b_1"},
                 chain(pick_place_instances("alphabet_soup", "basket"), pick_place_instances("tomato_sauce", "basket")))
TXT = "put the alphabet soup and the tomato sauce in the basket and hold both of them gently"
ST = lambda o, ph: {"action": "put", "object_phrase": ph, "object_role": o, "target_phrase": "the basket", "target_role": "basket", "clause": TXT}  # noqa: E731
GOOD = {"task": {"restated": True, "steps": [ST("alphabet_soup", "the alphabet soup"), ST("tomato_sauce", "the tomato sauce")], "order": []},
        "bindings": [{"instance_ids": ["alphabet_soup.pick_1.grasp", "tomato_sauce.pick_1.grasp"], "dimension": "grip", "level": "gentle",
                      "object_role": None, "clause": "hold both of them gently", "basis": "antecedent", "antecedent": "the alphabet soup and the tomato sauce"}],
        "ambiguous": [], "unsupported": []}


def run(obj, text=TXT):
    return admit(parse_intake(obj, text), P2)


def mutate(fn):
    o = copy.deepcopy(GOOD); fn(o); return o


def test_complete_answer_is_accepted():
    assert run(GOOD)["status"] == "accept"


def test_omissions_are_not_accepted():
    cases = {
        "all_bindings_dropped": (mutate(lambda o: o.update(bindings=[])), "manner_word_not_answered"),
        "second_object_missing": (mutate(lambda o: o["bindings"][0].update(instance_ids=["alphabet_soup.pick_1.grasp"])), "plural_scope_not_covered"),
        "task_part_dropped": (mutate(lambda o: o["task"].update(restated=False, steps=[])), "task_verb_without_steps"),
        "one_step_dropped": (mutate(lambda o: o["task"].update(steps=o["task"]["steps"][:1])), "named_object_not_in_answer"),
    }
    for name, (obj, gap) in cases.items():
        d = run(obj)
        assert d["status"] == "incomplete" and gap in d["codes"], (name, d)


def test_order_word_without_order_is_not_accepted():
    t = "put the alphabet soup in the basket and then the tomato sauce in the basket"
    obj = mutate(lambda o: o.update(bindings=[]))
    assert run(obj, t)["status"] == "incomplete"
    obj["task"]["order"] = [[0, 1]]
    assert run(obj, t)["status"] == "accept"


def test_level_word_must_agree_with_bound_level():
    obj = mutate(lambda o: o["bindings"][0].update(level="firm"))
    d = run(obj)
    assert d["status"] == "incomplete" and "level_disagrees_with_clause" in d["codes"]


def test_malformed_model_output_is_an_error_not_an_empty_answer():
    for bad in ({}, {"task": {"restated": True, "steps": [], "order": []}},
                mutate(lambda o: o.update(unsupported=["hold gently"])),
                mutate(lambda o: o["task"].update(steps="put")),
                mutate(lambda o: o["task"].update(order=[["a", "b"]])),
                mutate(lambda o: o["task"].pop("restated"))):
        with pytest.raises(ValueError):
            parse_intake(bad, TXT)


# ---- coverage is by occurrence in the sentence, not by "this word appears in some returned entry" ----
TXT2 = "put the alphabet soup and the tomato sauce in the basket, hold the soup gently, hold the sauce gently"
BIND = lambda role, clause: {"instance_ids": [f"{role}.pick_1.grasp"], "dimension": "grip", "level": "gentle", "object_role": role,  # noqa: E731
                             "clause": clause, "basis": "explicit_object" if "it " not in clause else "antecedent",
                             "antecedent": None if "it " not in clause else "the alphabet soup"}
GOOD2 = {"task": GOOD["task"], "ambiguous": [], "unsupported": [],
         "bindings": [BIND("alphabet_soup", "hold the soup gently"), BIND("tomato_sauce", "hold the sauce gently")]}


def test_same_manner_word_twice_needs_two_answers():
    assert run(GOOD2, TXT2)["status"] == "accept"
    one = copy.deepcopy(GOOD2); one["bindings"] = one["bindings"][:1]             # second "gently" is left out
    d = run(one, TXT2)
    assert d["status"] == "incomplete" and "manner_word_not_answered" in d["codes"], d
    dup = copy.deepcopy(GOOD2); dup["bindings"] = [BIND("alphabet_soup", "hold the soup gently")] * 2   # same answer twice != second answer
    assert run(dup, TXT2)["status"] != "accept"        # duplicated answer is rejected (conflicting binding), never counted as two


def test_identical_clause_repeated_in_the_sentence_needs_one_answer_each():
    t = "put the alphabet soup and the tomato sauce in the basket, hold it gently, hold it gently"
    one = copy.deepcopy(GOOD2); one["bindings"] = [BIND("alphabet_soup", "hold it gently")]
    d = run(one, t)
    assert d["status"] == "incomplete" and "repeated_requirement_not_covered" in d["codes"], d


# --- language protocol v2 (prompt v3): object-free speed requests cover every matching instance ---
def _scope(clause, ids, dim="speed", level="slow"):
    b = MannerBinding(tuple(ids), dim, level, None, clause)
    return intake(f"put a and b in the basket; {clause}", sem=Semantics(f"put a and b in the basket; {clause}", (b,)),
                  bases=({"basis": "request_scope", "antecedent": ""},))


def _speed_ids(prim=None):
    return [i.instance_id for i in PROG.instances if "speed" in i.dims and (prim is None or i.primitive == prim)]


def test_global_speed_request_covers_all_speed_instances():
    assert admit(_scope("move slowly", _speed_ids()), PROG)["status"] == "accept"


def test_named_stage_covers_that_stage_of_every_object():
    assert admit(_scope("go slowly when approaching", _speed_ids("approach")), PROG)["status"] == "accept"


def test_partial_scope_is_incomplete_not_accepted():
    d = admit(_scope("go slowly when approaching", _speed_ids("approach")[:1]), PROG)
    assert d["status"] == "incomplete" and "request_scope_not_covered" in d["codes"]
    d = admit(_scope("move slowly", _speed_ids("approach")), PROG)
    assert d["status"] == "incomplete"


def test_request_scope_never_applies_to_grip_or_names_an_object():
    d = admit(_scope("hold gently", ["a.pick_1.grasp", "b.pick_1.grasp"], dim="grip", level="gentle"), PROG)
    assert d["status"] == "ambiguous"
    d = admit(_scope("move a slowly", _speed_ids()), PROG)
    assert d["status"] == "ambiguous"
