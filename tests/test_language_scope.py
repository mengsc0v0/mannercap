"""Step 17: Chinese speed-scope rules, the scope gate in admit(), and plan_from_bindings (T5 program).
The 'model answers' below are hand-written fixtures (NOT LLM output): they test the code after the model, not the model."""
import dataclasses, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "mt"))
from synthesis_v2.language_scope import clause_speed_scope, expected_speed_instances
from synthesis_v2.request_intake import admit, parse_intake
from synthesis_v2.language_plan import PlanRefused, plan_from_bindings
from synthesis_v2.manner_request import speed_range
from synthesis_v2.candidate_plan import MAX_ROT_RPS, ROT_PER_LIN
from synthesis_v2.native_producer import GripPlan

s7 = pytest.importorskip("s7_task_config")
PROG = s7.make_program(5)
IDS = {it.primitive: it.instance_id for it in PROG.instances}
SPEED = [it.instance_id for it in PROG.instances if "speed" in it.dims]
BASE = s7.make_plan(5, PROG, grip=GripPlan(1.0, 100.0, settle_after_open=1), approach_check=True,
                    stop_on_grasp_failure=True, stop_on_place_failure=True)


def ans(clauses, ambiguous=()):
    b = [{"instance_ids": ids, "dimension": "speed", "level": lvl, "object_role": role, "clause": cl, "basis": basis, "antecedent": None}
         for ids, lvl, role, cl, basis in clauses]
    return {"task": {"restated": False, "steps": [], "order": []}, "bindings": b, "ambiguous": list(ambiguous), "unsupported": []}


def status(text, clauses, **kw):
    return admit(parse_intake(ans(clauses, **kw), text), PROG)


def ref(levels):
    st = []
    for sp in BASE.stages:
        lv = levels.get(sp.stage_id.split(".")[-1])
        if lv == "slow":
            v = speed_range("stack", "slow")[2]
            sp = dataclasses.replace(sp, v_max_mps=v, w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * v))
        st.append(sp)
    return dataclasses.replace(BASE, stages=tuple(st))


@pytest.mark.parametrize("clause,level,expect", [
    ("缓慢放入后部隔间", "slow", {"place"}),
    ("缓慢搬运书", "slow", {"transport"}),
    ("到隔间后正常放下", "normal", {"place"}),
    ("放入的时候慢一点", "slow", {"place"}),
    ("整个过程都慢一些", "slow", set(IDS) - {"grasp"}),
    ("慢一点", "slow", set(IDS) - {"grasp"}),
])
def test_scope_rules(clause, level, expect):
    got = expected_speed_instances(clause, level, PROG)
    assert {i.split(".")[-1] for i in got} == expect


def test_undetermined_scope_is_not_guessed():
    assert expected_speed_instances("缓慢地把书搬到隔间里放好", "slow", PROG) is None
    assert clause_speed_scope("缓慢地把书搬到隔间里放好")[0]["kind"] == "undetermined"


def test_local_request_accepted_and_plan_equals_structured_place_only():
    t = "把书搬过去，缓慢放入后部隔间。"
    c = [([IDS["place"]], "slow", "book", "缓慢放入后部隔间", "explicit_object")]
    assert status(t, c)["status"] == "accept"
    plan, trace = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({"place": "slow"}).to_dict()
    assert [x["level"] for x in trace if x["instance_id"] == IDS["place"]] == ["slow"]


def test_carry_slow_then_normal_place_equals_structured_transport_only():
    t = "缓慢搬运书，到隔间后正常放下。"
    c = [([IDS["transport"]], "slow", "book", "缓慢搬运书", "explicit_object"),
         ([IDS["place"]], "normal", "book", "到隔间后正常放下", "explicit_object")]
    assert status(t, c)["status"] == "accept"
    plan, _ = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({"transport": "slow"}).to_dict()


def test_global_request_covers_all_speed_stages_not_grasp():
    t = "整个过程都慢一些。"
    c = [(SPEED, "slow", None, "整个过程都慢一些", "request_scope")]
    assert status(t, c)["status"] == "accept"
    plan, _ = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({p: "slow" for p in IDS if p != "grasp"}).to_dict()


def test_widened_local_request_is_blocked_by_the_gate():
    t = "把书缓慢放入后部隔间，松开并退出。"
    c = [(SPEED, "slow", None, "把书缓慢放入后部隔间", "request_scope")]       # a model answer that widens "place" to everything
    d = status(t, c)
    assert d["status"] != "accept" and any(g.get("gap") == "language_scope_mismatch" for g in d["guards"])


def test_narrowed_global_request_is_blocked_by_the_gate():
    t = "整个过程都慢一些。"
    c = [([IDS["place"]], "slow", None, "整个过程都慢一些", "request_scope")]
    assert status(t, c)["status"] != "accept"


def test_ambiguous_scope_needs_clarification_even_if_model_binds():
    t = "缓慢地把书搬到隔间里放好。"
    c = [([IDS["transport"], IDS["place"]], "slow", "book", "缓慢地把书搬到隔间里放好", "explicit_object")]
    d = status(t, c)
    assert d["status"] == "ambiguous" and any(g.get("guard") == "speed_scope_undetermined" for g in d["guards"])


def test_plan_refuses_non_speed_and_conflicting_bindings():
    t = "x"
    sem = parse_intake(ans([([IDS["place"]], "slow", "book", "a", "explicit_object"), ([IDS["place"]], "fast", "book", "b", "explicit_object")]), t).semantics
    with pytest.raises(PlanRefused):
        plan_from_bindings(PROG, BASE, sem)


def test_english_clause_path_unchanged():
    t = "move slowly"
    c = [(SPEED, "slow", None, "move slowly", "request_scope")]
    assert status(t, c)["status"] == "accept"      # cjk gate does not touch English clauses


def test_adverb_only_clause_is_read_in_its_sentence_context():
    t = "缓慢地把书搬到隔间里放好。"          # the model returned only the adverb as the clause (observed with the real LLM)
    c = [(SPEED, "slow", None, "缓慢地", "request_scope")]
    d = status(t, c)
    assert d["status"] == "ambiguous" and any(g.get("guard") == "speed_scope_undetermined" for g in d["guards"])


def test_scope_context_picks_the_clause_part():
    from synthesis_v2.language_scope import scope_context
    assert scope_context("缓慢搬运书，到隔间后正常放下。", "缓慢搬运书") == "缓慢搬运书"
    assert scope_context("缓慢搬运书，到隔间后正常放下。", "正常放下") == "到隔间后正常放下"
    assert scope_context("整个过程都慢一些。", "慢一些") == "整个过程都慢一些"
