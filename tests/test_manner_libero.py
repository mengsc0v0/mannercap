from types import SimpleNamespace as NS

import pytest

from synthesis_v2 import manner_libero as M


def _prog():
    its = []
    for role in ("a", "b"):
        for prim in ("approach", "descend", "grasp", "lift", "transport", "place", "retreat"):
            its.append(NS(instance_id=f"{role}.pick_1.{prim}", primitive=prim, object_role=role,
                          label=f"{role}.pick_1.{prim}"))
    d = {i.instance_id: i for i in its}
    return NS(instances=tuple(its), instance=lambda k: d[k])


def _sem(*bs):
    return NS(bindings=[NS(instance_ids=list(i), dimension=d, level=l, clause="c") for i, d, l in bs])


def test_request_speed_and_grip():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle"),
                                         (["a.pick_1.transport"], "speed", "slow")))
    assert r.grip == {"a": "gentle"} and r.speed == {"a.pick_1.transport": "slow"}
    assert M.carry_level(p, r, "a") == "slow" and M.carry_level(p, r, "b") is None


@pytest.mark.parametrize("bs,code", [
    ([(["a.pick_1.grasp"], "press", "firm")], "unsupported_dimension"),
    ([(["a.pick_1.lift"], "grip", "firm")], "grip_on_non_grasp"),
    ([(["a.pick_1.grasp"], "speed", "slow")], "speed_on_grasp"),
    ([(["a.pick_1.grasp"], "grip", "firm"), (["a.pick_1.grasp"], "grip", "gentle")], "conflict"),
])
def test_request_refusals(bs, code):
    with pytest.raises(M.MannerRefused, match=code):
        M.request_from_semantics(_prog(), _sem(*bs))


def test_initial_commands_defaults_and_choice():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "firm"),
                                         (["a.pick_1.transport"], "speed", "fast")))
    seen = []
    c = M.initial_commands(p, r, lambda role, lvl, v: seen.append((role, lvl, v)) or ((20.0, 100.0), None))
    assert seen == [("a", "firm", 0.24)]
    assert c["grip"] == {"a": (20.0, 100.0), "b": M.DEFAULT_GRIP}
    assert c["speed"]["a.pick_1.transport"] == 0.24 and c["speed"]["b.pick_1.lift"] == M.DEFAULT_V
    assert "a.pick_1.grasp" not in c["speed"]


def _meas(hmin, hmax, v, ok=True, clean=True, stable=True):
    return {"task_ok": ok, "contact_clean": clean, "stage_speed": {"a.pick_1.transport": v},
            "objects": {"a": {"hold_min": hmin, "hold_max": hmax, "hold_mean": (hmin + hmax) / 2,
                              "stable": stable, "valid": True}}}


def test_audit_and_repair_directions():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle"),
                                         (["a.pick_1.transport"], "speed", "fast")))
    a = M.audit(r, _meas(9, 15, 0.20))
    assert a["joint_pass"]
    a = M.audit(r, _meas(9, 20, 0.12))
    causes = {x["target"]: x["cause"] for x in a["clauses"]}
    assert causes == {"a": "force_above_band", "a.pick_1.transport": "speed_below_band"}
    cmd = M.initial_commands(p, r, lambda *_: ((5.0, 100.0), None))
    got = []
    new, acts = M.next_commands(p, r, cmd, a, [cmd], lambda *x: got.append(x) or ((1.0, 22.0), None))
    assert new["speed"]["a.pick_1.transport"] == 0.32 and new["grip"]["a"] == (1.0, 22.0)
    assert got[0][:5] == ("a", "gentle", (5.0, 100.0), "force_above_band", 0.24)
    # exhausted ladder -> honest stop, never a level change
    cmd["speed"]["a.pick_1.transport"] = 0.40
    new, acts = M.next_commands(p, r, cmd, a, [cmd], lambda *x: ((1.0, 22.0), None))
    assert new is None and acts[-1]["why"] == "speed_ladder_exhausted"


def test_non_manner_failure_stops():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle")))
    a = M.audit(r, _meas(9, 15, 0.1, ok=False))
    new, acts = M.next_commands(p, r, M.initial_commands(p, r, lambda *_: ((5.0, 100.0), None)), a, [], lambda *x: None)
    assert new is None and acts[0]["why"] == "task_failed_not_a_manner_failure"
    a = M.audit(r, _meas(9, 15, 0.1, stable=False))
    new, acts = M.next_commands(p, r, M.initial_commands(p, r, lambda *_: ((5.0, 100.0), None)), a, [], lambda *x: None)
    assert new is None and acts[0]["why"] == "force_candidates_exhausted"


def test_slip_repair_slows_unbound_carry_only():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle")))
    cmd = M.initial_commands(p, r, lambda *_: ((1.0, 22.0), None))
    a = M.audit(r, _meas(9, 15, 0.1, stable=False))
    new, acts = M.next_commands(p, r, cmd, a, [cmd], lambda *x: ((5.0, 100.0), 0.06))
    assert new["speed"]["a.pick_1.lift"] == 0.06 and new["speed"]["a.pick_1.transport"] == 0.06
    assert new["speed"]["b.pick_1.transport"] == M.DEFAULT_V
    r2 = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle"),
                                          (["a.pick_1.transport"], "speed", "normal")))
    cmd2 = M.initial_commands(p, r2, lambda *_: ((1.0, 22.0), None))
    a2 = M.audit(r2, _meas(9, 15, 0.12, stable=False))
    new2, _ = M.next_commands(p, r2, cmd2, a2, [cmd2], lambda *x: ((5.0, 100.0), 0.06))
    assert new2["speed"]["a.pick_1.transport"] == 0.12 and new2["speed"]["a.pick_1.lift"] == 0.06


def test_joint_choice_respects_bound_carry_band():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "firm")))
    c = M.initial_commands(p, r, lambda *_: ((25.0, 100.0), 0.24))
    assert c["speed"]["a.pick_1.lift"] == 0.24 and c["speed"]["a.pick_1.transport"] == 0.24
    assert M.allowed_carry(p, r, "a", (0.06, 0.12, 0.24)) == [0.06, 0.12, 0.24]
    r2 = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "firm"),
                                          (["a.pick_1.transport"], "speed", "slow")))
    assert M.allowed_carry(p, r2, "a", (0.06, 0.12, 0.24)) == [0.04, 0.05, 0.06]
    c2 = M.initial_commands(p, r2, lambda *_: ((25.0, 100.0), 0.24))   # out of band: transport untouched
    assert c2["speed"]["a.pick_1.transport"] == 0.06 and c2["speed"]["a.pick_1.lift"] == 0.24


def test_carry_clearance_repair_keeps_manner_and_is_bounded():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle")))
    cmd = M.initial_commands(p, r, lambda *_: ((5.0, 100.0), None))
    assert cmd["clear"] == {}
    a = M.audit(r, _meas(9, 15, 0.1, clean=False))
    # no carry evidence -> unchanged honest stop
    new, acts = M.next_commands(p, r, cmd, a, [cmd], lambda *x: None, {})
    assert new is None and acts[0]["why"] == "contact_unclean_not_a_manner_failure"
    # evidence on b's carry -> only b is raised, speed/grip untouched
    new, acts = M.next_commands(p, r, cmd, a, [cmd], lambda *x: None, {"b": ["pushed:basket_1:0.05"]})
    assert new["clear"] == {"b": M.CLEAR_STEP} and new["speed"] == cmd["speed"] and new["grip"] == cmd["grip"]
    assert acts == [{"action": "clearance", "target": "b", "from": 0.0, "to": M.CLEAR_STEP,
                     "cause": ["pushed:basket_1:0.05"]}]
    new2, _ = M.next_commands(p, r, new, a, [cmd, new], lambda *x: None, {"b": ["x"]})
    assert new2["clear"] == {"b": M.CLEAR_MAX}
    new3, acts3 = M.next_commands(p, r, new2, a, [cmd, new, new2], lambda *x: None, {"b": ["x"]})
    assert new3 is None and acts3[0]["why"] == "carry_clearance_exhausted"
    # a passing execution is never "repaired"
    ok = M.audit(r, _meas(9, 15, 0.1))
    assert M.next_commands(p, r, cmd, ok, [cmd], lambda *x: None, {"b": ["x"]})[0] is None
    # clearance + a failed manner clause in one step
    bad = M.audit(r, _meas(9, 20, 0.1, ok=False))
    new4, acts4 = M.next_commands(p, r, cmd, bad, [cmd], lambda *x: ((1.0, 14.0), None), {"a": ["x"]})
    assert new4["clear"] == {"a": M.CLEAR_STEP} and new4["grip"]["a"] == (1.0, 14.0)
    assert [x["action"] for x in acts4] == ["clearance", "grip"]


def test_initial_commands_carry_clearance_prior():
    p = _prog()
    r = M.request_from_semantics(p, _sem((["a.pick_1.grasp"], "grip", "gentle")))
    cmd = M.initial_commands(p, r, lambda *_: ((5.0, 100.0), None), {"a": 0.0, "b": 0.03})
    assert cmd["clear"] == {"b": 0.03}
