"""Shared repair rule v0: trigger needs BOTH carried unexpected contact and grasp instability; bounded candidate; one repair; no task id."""
import numpy as np

from synthesis_v2 import transport_repair_rule as R
from synthesis_v2.task_program import StageInstance, TaskProgram


def _prog(dz=0.175):
    inst = []
    for r in ("a", "b"):
        for prim in ("grasp", "lift", "transport", "place"):
            inst.append(StageInstance(f"{r}.pick_1.{prim}", prim, f"{r}.pick_1.{prim}", r, "basket", f"{r}.pick_1",
                                      params={"transport_dz": dz} if prim == "transport" else {}))
    return TaskProgram("p", "libero", "x", {"a": "a_1", "b": "b_1", "basket": "basket_1"}, tuple(inst), source="composed")


def _run(prog, one_sided_at=(), n_per=10):
    stages = [i.label for i in prog.instances for _ in range(n_per)]
    L = np.full(len(stages), 5.0); Rr = np.full(len(stages), 5.0)
    for k in one_sided_at:
        Rr[k] = 0.0
    return stages, L, Rr


def _first(prog, label):
    return [i.label for i in prog.instances for _ in range(10)].index(label)


def test_contact_without_instability_does_not_fire():
    p = _prog(); stages, L, Rr = _run(p)
    k = _first(p, "a.pick_1.transport") + 5
    assert R.diagnose(p, stages, L, Rr, {("a_1", "basket_1_main"): [31.0, {k}]}) == []


def test_instability_without_contact_does_not_fire():
    p = _prog(); k = _first(p, "a.pick_1.transport") + 6
    stages, L, Rr = _run(p, [k])
    assert R.diagnose(p, stages, L, Rr, {}) == []


def test_both_fire_and_candidate_is_bounded_plus_3cm():
    p = _prog(); k = _first(p, "a.pick_1.transport") + 5
    stages, L, Rr = _run(p, [k + 1])
    tg = R.diagnose(p, stages, L, Rr, {("a_1", "basket_1_main"): [45.0, {k, k + 1}]})
    prop, rec = R.propose(p, tg)
    assert prop and prop.instance_label == "a.pick_1.transport" and abs(prop.new_dz - 0.205) < 1e-9
    p2 = R.apply(p, prop)
    assert [i.params.get("transport_dz") for i in p2.instances if i.label == "a.pick_1.transport"] == [0.205]
    assert [i.params for i in p2.instances if i.label == "b.pick_1.transport"][0]["transport_dz"] == 0.175   # other instance untouched


def test_other_object_contact_does_not_attribute_to_wrong_role():
    p = _prog(); k = _first(p, "b.pick_1.transport") + 5
    stages, L, Rr = _run(p, [_first(p, "a.pick_1.transport") + 5])        # instability belongs to a
    assert R.diagnose(p, stages, L, Rr, {("b_1", "basket_1_main"): [40.0, {k}]}) == []


def test_support_and_instability_far_after_contact_do_not_fire():
    p = _prog(); k = _first(p, "a.pick_1.transport") + 1
    stages, L, Rr = _run(p, [k + R.LAG_STEPS + 2])
    assert R.diagnose(p, stages, L, Rr, {("a_1", "basket_1_main"): [40.0, {k}]}) == []


def test_contact_in_lift_only_is_not_repairable_and_infeasible_dz_is_refused():
    p = _prog(); k = _first(p, "a.pick_1.lift") + 5
    stages, L, Rr = _run(p, [k + 1])
    prop, rec = R.propose(p, R.diagnose(p, stages, L, Rr, {("a_1", "x"): [9.0, {k}]}))
    assert prop is None and rec["per_trigger"][0]["not_proposed_because"] == "contact_not_in_a_transport_stage"
    p = _prog(0.205); k = _first(p, "a.pick_1.transport") + 5
    stages, L, Rr = _run(p, [k + 1])
    prop, rec = R.propose(p, R.diagnose(p, stages, L, Rr, {("a_1", "x"): [9.0, {k}]}))
    assert prop is None and rec["per_trigger"][0]["not_proposed_because"].startswith("infeasible_new_dz")


def test_one_repair_per_request():
    p = _prog()
    ka, kb = _first(p, "a.pick_1.transport") + 5, _first(p, "b.pick_1.transport") + 5
    stages, L, Rr = _run(p, [ka + 1, kb + 1])
    prop, rec = R.propose(p, R.diagnose(p, stages, L, Rr, {("a_1", "x"): [9.0, {ka}], ("b_1", "x"): [9.0, {kb}]}))
    assert prop.trigger.object_role == "a" and rec["per_trigger"][1]["not_proposed_because"] == "repair_budget_one_per_request"


def test_rule_has_no_task_or_seed_input():
    import inspect
    for f in (R.diagnose, R.propose, R.apply, R.request_checks):
        assert not {"task", "t", "init", "seed"} & set(inspect.signature(f).parameters)


# ---- hold_checks_v1: four grasp states, invalid => unknown ------------------------------------------------------
def _checks(p, stages, L, Rr, valid=None, ok=True):
    return R.request_checks(p, stages, L, Rr, {}, "completed", ok, valid=valid)


def test_four_grasp_states_are_distinguished():
    st = R.grasp_state([5, 5, 0, 0, np.nan, 5], [5, 0, 5, 0, 5, np.nan])
    assert list(st) == ["bilateral", "one_sided", "one_sided", "no_contact", "invalid", "invalid"]
    assert list(R.grasp_state([5, 5], [5, 5], valid=[1.0, 0.5])) == ["bilateral", "invalid"]


def test_all_nan_forces_do_not_pass_the_checks():
    p = _prog(); stages, L, Rr = _run(p)
    ok = _checks(p, stages, L, Rr)
    assert ok["task_and_hold_checks_passed"] and ok["unknown_objects"] == []
    nan = np.full(len(stages), np.nan)
    bad = _checks(p, stages, nan, nan)
    assert not bad["task_and_hold_checks_passed"] and sorted(bad["unknown_objects"]) == ["a", "b"]
    assert bad["hold_state_counts_by_object"]["a"]["invalid"] == bad["hold_state_counts_by_object"]["a"]["n_hold"] > 0


def test_invalid_valid_fraction_does_not_pass():
    p = _prog(); stages, L, Rr = _run(p)
    v = np.ones(len(stages)); v[_first(p, "a.pick_1.transport") + 3] = 0.5
    r = _checks(p, stages, L, Rr, valid=v)
    assert not r["task_and_hold_checks_passed"] and r["unknown_objects"] == ["a"]


def test_no_contact_is_not_called_one_sided_and_still_fails_the_local_check():
    p = _prog(); stages, L, Rr = _run(p)
    k = _first(p, "a.pick_1.transport") + 3
    L = L.copy(); Rr = Rr.copy(); L[k] = 0.0; Rr[k] = 0.0
    c = _checks(p, stages, L, Rr)
    assert c["hold_state_counts_by_object"]["a"]["no_contact"] == 1 and c["hold_state_counts_by_object"]["a"]["one_sided"] == 0
    assert not c["task_and_hold_checks_passed"]


def test_empty_hold_window_of_a_lifted_object_is_unknown():
    p = _prog(); stages, L, Rr = _run(p)
    keep = [i for i, s in enumerate(stages) if not s.startswith("a.pick_1.lift") and not s.startswith("a.pick_1.transport")]
    c = _checks(p, [stages[i] for i in keep], L[keep], Rr[keep])
    assert "a" in c["unknown_objects"] and not c["task_and_hold_checks_passed"]
