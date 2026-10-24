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
