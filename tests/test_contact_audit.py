from synthesis_v2.contact_audit import classify

EV = {("right_gripper", "basket_1_main"): [192.5, {5, 6, 7}], ("leftfinger", "living_room_table_col"): [208.0, {9}],
      ("leftfinger", "butter_1_main"): [0.05, {9}], ("robot0_link5", "milk_1_main"): [3.0, {2}]}
STAGES = ["a.pick_1.approach"] * 4 + ["butter.pick_1.descend"] * 6


def test_forbidden_support_and_below_threshold_are_separated():
    r = classify(EV, STAGES)
    assert not r["clean"]
    assert {(c["robot_body"], c["other"]) for c in r["forbidden_contacts"]} == {("right_gripper", "basket_1_main"), ("robot0_link5", "milk_1_main")}
    assert [c["other"] for c in r["support_contacts_observed"]] == ["living_room_table_col"]   # observed, not a violation
    assert all(c["other"] != "butter_1_main" for c in r["forbidden_contacts"])                   # 0.05 N < threshold
    assert r["forbidden_contacts"][0]["stages"] == ["descend@butter"]


def test_clean_when_only_support_contact():
    assert classify({("rightfinger", "floor_x"): [300.0, {1}]}, STAGES)["clean"]


# ---- rollback isolation: a contact that happens only inside a save_state/restore_state branch is not a trajectory contact ----
from types import SimpleNamespace
from synthesis_v2.contact_audit import ContactRecorder


class _FakeSession:
    force_target_name = "butter_1"

    def __init__(self):
        self.control_step_index, self.touching = 0, None
        self.env = SimpleNamespace(sim=SimpleNamespace(step=lambda: None))

    def _mj(self):
        return None, None

    def step(self, a):
        self.control_step_index += 1; self.env.sim.step()

    def save_state(self):
        return {"control_step_index": self.control_step_index}

    def restore_state(self, st):
        self.control_step_index = st["control_step_index"]


class _Rec(ContactRecorder):
    def _scan(self):
        if self.s.touching:
            yield (*self.s.touching, 12.0)


def _stages():
    return ["x.pick_1.approach"] * 100


def test_contact_only_in_lookahead_branch_is_not_reported():
    s = _FakeSession(); r = _Rec(s).install()
    st = s.save_state()
    s.touching = ("right_gripper", "milk_1_main")
    for _ in range(5):
        s.step(None)                        # look-ahead: the hand touches milk here
    s.touching = None; s.restore_state(st)
    for _ in range(5):
        s.step(None)                        # the real execution never touches anything
    rep = r.report(_stages())
    assert rep["clean"] and rep["forbidden_contacts"] == []
    assert [c["other"] for c in rep["rolled_back_not_counted"]] == ["milk_1_main"]       # kept visible, apart


def test_contact_in_the_real_execution_is_reported_with_its_own_step():
    s = _FakeSession(); r = _Rec(s).install()
    st = s.save_state(); s.step(None); s.restore_state(st)
    s.touching = ("right_gripper", "basket_1_main")
    for _ in range(3):
        s.step(None)
    rep = r.report(_stages())
    assert not rep["clean"] and rep["forbidden_contacts"][0]["control_steps"] == 3
    assert rep["rolled_back_not_counted"] == []


def test_contact_before_the_branch_survives_the_rollback():
    s = _FakeSession(); r = _Rec(s).install()
    s.touching = ("right_gripper", "milk_1_main"); s.step(None); s.touching = None
    st = s.save_state(); s.touching = ("right_gripper", "milk_1_main"); s.step(None); s.touching = None; s.restore_state(st)
    assert len(r.report(_stages())["forbidden_contacts"]) == 1 and r.events[("right_gripper", "milk_1_main")][1] == {0}
