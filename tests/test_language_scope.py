"""Step 17 (scope v2): Chinese speed-scope rules, the gate in admit(), plan_from_bindings and the feedback retry.
The 'model answers' below are hand-written fixtures (NOT LLM output): they test the code after the model, not the model."""
import dataclasses, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mt"))
from synthesis_v2.language_scope import resolve_speed_scope, scope_rules_hash, SCOPE_VERSION
from synthesis_v2.request_intake import admit, parse_intake
from synthesis_v2.language_plan import PlanRefused, compile_request, plan_from_bindings
from synthesis_v2.manner_request import SEMANTICS_VERSION, semantics_hash, speed_range
from synthesis_v2.candidate_plan import MAX_ROT_RPS, ROT_PER_LIN
from synthesis_v2.native_producer import GripPlan

s7 = pytest.importorskip("s7_task_config")
PROG = s7.make_program(5)
IDS = {it.primitive: it.instance_id for it in PROG.instances}
SPEED = [it.instance_id for it in PROG.instances if "speed" in it.dims]
BASE = s7.make_plan(5, PROG, grip=GripPlan(1.0, 100.0, settle_after_open=1), approach_check=True,
                    stop_on_grasp_failure=True, stop_on_place_failure=True)
SPEC = json.loads((ROOT / "reports/step17_t5_language/scope_spec_v2.json").read_text(encoding="utf-8"))


def ans(clauses, ambiguous=()):
    b = [{"instance_ids": ids, "dimension": "speed", "level": lvl, "object_role": role, "clause": cl, "basis": basis, "antecedent": None}
         for ids, lvl, role, cl, basis in clauses]
    return {"task": {"restated": False, "steps": [], "order": []}, "bindings": b, "ambiguous": list(ambiguous), "unsupported": []}


def decide(text, clauses):
    return admit(parse_intake(ans(clauses), text), PROG)


def ref(levels):
    st = []
    for sp in BASE.stages:
        lv = levels.get(sp.stage_id.split(".")[-1])
        if lv == "slow":
            v = speed_range("stack", "slow")[2]
            sp = dataclasses.replace(sp, v_max_mps=v, w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * v))
        st.append(sp)
    return dataclasses.replace(BASE, stages=tuple(st))


def short(eff):
    return {k.split(".")[-1]: v for k, v in eff.items()}


# ---- spec consistency -------------------------------------------------------------------------------------------------
def test_spec_cites_the_actual_vocabulary_version():
    v = SPEC["vocabulary"]
    assert v["scope_version"] == SCOPE_VERSION and v["scope_rules_hash"] == scope_rules_hash()
    assert v["semantics_version"] == SEMANTICS_VERSION and v["semantics_hash"] == semantics_hash()


@pytest.mark.parametrize("case", SPEC["boundary_sentences"], ids=lambda c: c["id"])
def test_boundary_sentences_follow_the_spec(case):
    r = resolve_speed_scope(case["text"], PROG)
    assert r["status"] == case["expect_status"]
    if "expect_levels" in case:
        assert short(r["effective"]) == case["expect_levels"]
    if "expect_reason" in case:
        assert case["expect_reason"] in {x["reason"] for x in r["reasons"]}
    if "expect_exceptions" in case:
        assert [e.split(".")[-1] for e in r["exceptions"]] == case["expect_exceptions"]


# ---- the two decisions that changed -----------------------------------------------------------------------------------
def test_no_stage_word_is_no_longer_global():
    for t in ("慢一点。", "缓慢地把书搬到隔间里。"):
        assert resolve_speed_scope(t, PROG)["status"] == "needs_clarification"
        d = decide(t, [(SPEED, "slow", None, t, "request_scope")])        # a model that "helpfully" binds everything
        assert d["status"] == "ambiguous"


def test_negation_conflict_and_exception_are_not_overridden_by_local_rules():
    assert resolve_speed_scope("不要慢慢放下。", PROG)["status"] == "needs_clarification"
    assert resolve_speed_scope("放下时慢一点，放下时快一点。", PROG)["status"] == "conflict"
    assert resolve_speed_scope("整个过程缓慢放入。", PROG)["status"] == "ok"                    # explicit global cue wins over R1
    assert len(resolve_speed_scope("整个过程缓慢放入。", PROG)["effective"]) == 6


# ---- gate ---------------------------------------------------------------------------------------------------------------
def test_local_request_accepted_and_plan_equals_structured_place_only():
    t = "把书搬过去，缓慢放入后部隔间。"
    c = [([IDS["place"]], "slow", "book", "缓慢放入后部隔间", "explicit_object")]
    assert decide(t, c)["status"] == "accept"
    plan, trace = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({"place": "slow"}).to_dict()


def test_carry_slow_then_normal_place_equals_structured_transport_only():
    t = "缓慢搬运书，到隔间后正常放下。"
    c = [([IDS["transport"]], "slow", "book", "缓慢搬运书", "explicit_object"), ([IDS["place"]], "normal", "book", "到隔间后正常放下", "explicit_object")]
    assert decide(t, c)["status"] == "accept"
    plan, _ = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({"transport": "slow"}).to_dict()


def test_global_request_covers_all_speed_stages_not_grasp():
    t = "整个过程都慢一些。"
    c = [(SPEED, "slow", None, "整个过程都慢一些", "request_scope")]
    assert decide(t, c)["status"] == "accept"
    plan, _ = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({p: "slow" for p in IDS if p != "grasp"}).to_dict()


def test_widened_and_narrowed_answers_are_blocked_and_report_the_difference():
    t = "把书缓慢放入后部隔间，松开并退出。"
    d = decide(t, [(SPEED, "slow", None, "把书缓慢放入后部隔间", "request_scope")])
    g = [x for x in d["guards"] if x.get("gap") == "language_scope_mismatch"][0]
    assert d["status"] == "incomplete" and len(g["extra"]) == 5 and not g["missing"]
    assert decide("整个过程都慢一些。", [([IDS["place"]], "slow", None, "整个过程都慢一些", "request_scope")])["status"] != "accept"


def test_adverb_only_clause_is_judged_from_the_sentence():
    t = "缓慢地把书搬到隔间里放好。"           # observed: the real model returned only "缓慢地" and bound every stage
    d = decide(t, [(SPEED, "slow", None, "缓慢地", "request_scope")])
    assert d["status"] == "ambiguous"


def test_global_with_local_exception_is_expressed_as_disjoint_bindings():
    t = "全程慢一些，但放下时用正常速度。"
    others = [IDS[p] for p in IDS if p not in ("grasp", "place")]
    c = [(others, "slow", None, "全程慢一些", "request_scope"), ([IDS["place"]], "normal", "book", "但放下时用正常速度", "explicit_object")]
    d = decide(t, c)
    assert d["status"] == "accept", d["guards"]
    assert d["speed_scope"]["exceptions"] == [IDS["place"]]
    plan, _ = plan_from_bindings(PROG, BASE, parse_intake(ans(c), t).semantics)
    assert plan.to_dict() == ref({p: "slow" for p in IDS if p not in ("grasp", "place")}).to_dict()
    assert decide(t, [(SPEED, "slow", None, "全程慢一些", "request_scope")])["status"] != "accept"          # ignores the exception
    overlap = c[:1] and [(SPEED, "slow", None, "全程慢一些", "request_scope"), c[1]]
    assert decide(t, overlap)["status"] == "invalid"                       # existing rule: one instance, one binding


def test_conflicting_requests_are_never_resolved_by_clause_order():
    for t in ("放下时慢一点，放下时快一点。", "全程慢一些，放下时快一点。"):
        c = [([IDS["place"]], "slow", "book", "放下时慢一点", "explicit_object"), ([IDS["place"]], "fast", "book", "放下时快一点", "explicit_object")]
        d = decide(t, c)
        assert d["status"] == "conflict" and d["codes"] == ["speed_level_conflict"]


def test_negated_request_gets_no_slow_plan():
    d = decide("不要慢慢放下。", [([IDS["place"]], "slow", "book", "慢慢放下", "explicit_object")])
    assert d["status"] == "ambiguous"


def test_plan_builder_refuses_two_levels_on_one_instance():
    bad = parse_intake(ans([([IDS["place"]], "slow", "book", "a", "explicit_object"), ([IDS["place"]], "fast", "book", "b", "explicit_object")]), "x").semantics
    with pytest.raises(PlanRefused):
        plan_from_bindings(PROG, BASE, bad)


def test_english_path_unchanged():
    t = "move slowly"
    assert admit(parse_intake(ans([(SPEED, "slow", None, "move slowly", "request_scope")]), t), PROG)["status"] == "accept"


# ---- feedback retry ----------------------------------------------------------------------------------------------------
class _Rec:
    def __init__(self): self.status, self.request_hash, self.usage = "ok", "h", {}


class _FakeClient:
    def __init__(self, answers): self.answers, self.records, self.calls = list(answers), [], []

    def complete_json(self, system, user, *, prompt_version):
        self.calls.append((system, user, prompt_version)); self.records.append(_Rec())
        return self.answers.pop(0), self.records[-1]


def test_feedback_retry_happens_once_only_for_scope_mismatch_and_is_flagged():
    t = "把书缓慢放入后部隔间，松开并退出。"
    wide = ans([(SPEED, "slow", None, "把书缓慢放入后部隔间", "request_scope")])
    right = ans([([IDS["place"]], "slow", "book", "缓慢放入后部隔间", "explicit_object")])
    c = _FakeClient([wide, right])
    r = compile_request(t, PROG, BASE, c, feedback_retry=True)
    assert r["status"] == "accept" and r["needed_feedback_retry"] and len(c.calls) == 2 and len(r["attempts"]) == 2
    assert r["plan"].to_dict() == ref({"place": "slow"}).to_dict()
    assert "gate_feedback" in c.calls[1][1] and "gate_feedback" not in c.calls[0][1] and c.calls[1][2].endswith("gate_feedback.v1")
    c2 = _FakeClient([wide, wide])                                         # still wrong after the single retry -> no plan, no third call
    r2 = compile_request(t, PROG, BASE, c2, feedback_retry=True)
    assert r2["status"] != "accept" and r2["plan"] is None and len(c2.calls) == 2
    c3 = _FakeClient([ans([(SPEED, "slow", None, "缓慢地", "request_scope")])])      # clarification is a valid end: no retry
    r3 = compile_request("缓慢地把书搬到隔间里放好。", PROG, BASE, c3, feedback_retry=True)
    assert r3["status"] == "ambiguous" and len(c3.calls) == 1 and r3["needs_clarification"]
