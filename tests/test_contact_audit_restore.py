"""ContactRecorder: restore of an older / repeatedly used snapshot (stage retry), seeding of a resumed session."""
from types import SimpleNamespace

from synthesis_v2.contact_audit import ContactRecorder


class _Sess:
    def __init__(self):
        self.control_step_index, self.pairs = 0, []
        self.force_target_name = "x_1"
        self.env = SimpleNamespace(sim=SimpleNamespace(step=lambda *a, **k: None))

    def _mj(self):
        return None, None

    def step(self, a):
        self.env.sim.step()
        self.control_step_index += 1

    def save_state(self):
        return {"c": self.control_step_index}

    def restore_state(self, st):
        self.control_step_index = st["c"]


class _Rec(ContactRecorder):
    def _scan(self):
        yield from self.s.pairs


def _mk():
    s = _Sess()
    return s, _Rec(s).install()


def _run(s, pairs, n=1):
    s.pairs = pairs
    for _ in range(n):
        s.step(None)


def test_restore_older_snapshot_skipping_a_newer_one():
    s, r = _mk()
    _run(s, [("gripper0_hand", "obj_a", 5.0)])
    A = s.save_state()
    _run(s, [("gripper0_hand", "obj_b", 9.0)])
    s.save_state()
    _run(s, [("gripper0_hand", "obj_c", 9.0)])
    s.restore_state(A)
    assert set(k[1] for k in r.events) == {"obj_a"}
    assert {k[1] for k in r.rolled_back} == {"obj_b", "obj_c"}


def test_same_snapshot_restored_twice():
    s, r = _mk()
    A = s.save_state()
    _run(s, [("gripper0_hand", "obj_b", 9.0)]); s.restore_state(A)
    _run(s, [("gripper0_hand", "obj_c", 9.0)]); s.restore_state(A)
    assert r.events == {} and {k[1] for k in r.rolled_back} == {"obj_b", "obj_c"}
    _run(s, [("gripper0_hand", "obj_d", 9.0)])
    assert {k[1] for k in r.events} == {"obj_d"}


def test_resumed_recorder_inherits_prefix_events():
    s1, r1 = _mk()
    _run(s1, [("gripper0_hand", "obj_a", 5.0)])
    A = s1.save_state()
    _run(s1, [("gripper0_hand", "obj_b", 9.0)])
    s2, r2 = _mk()
    r2.seed(r1.events_at(A))
    s2.restore_state(A)
    _run(s2, [("gripper0_hand", "obj_c", 3.0)])
    assert {k[1] for k in r2.events} == {"obj_a", "obj_c"}


# --- physical identity: force measurement and contact policy share one geom-role rule ---
from synthesis_v2.grasp_force_contract import classify_finger_geom  # noqa: E402


def test_finger_geom_roles_single_source():
    assert classify_finger_geom("gripper0_finger1_pad_collision", "gripper0_finger_joint1_tip") == ("L", "pad")
    assert classify_finger_geom("gripper0_finger2_pad_collision", "gripper0_finger_joint2_tip") == ("R", "pad")
    assert classify_finger_geom("gripper0_finger1_collision", "gripper0_leftfinger") == ("L", "finger_link")
    assert classify_finger_geom("gripper0_finger2_collision", "gripper0_rightfinger") == ("R", "finger_link")
    assert classify_finger_geom("gripper0_hand_collision", "gripper0_right_gripper") is None   # palm is not a finger
    assert classify_finger_geom("alphabet_soup_1_g1", "alphabet_soup_1_main") is None


class _IdRec(_Rec):
    ROLES = {1: ("L", "pad"), 2: ("L", "finger_link"), 3: None}

    def _role(self, gid):
        return self.ROLES.get(gid)


def test_audit_roles_pad_allowed_link_identified_palm_and_other_object_kept():
    s = _Sess()
    r = _IdRec(s).install()
    s.pairs = [("gripper0_finger_joint1_tip", "x_1_main", 8.0, 1),    # pad on target: allowed, not recorded
               ("gripper0_leftfinger", "x_1_main", 1.0, 2),           # finger link on target: forbidden, identified
               ("gripper0_leftfinger", "y_1_main", 1.0, 2),           # finger link on ANOTHER object: plain forbidden
               ("gripper0_right_gripper", "x_1_main", 0.7, 3)]        # palm on target: plain forbidden
    s.step(None)
    rep = r.report(["x.pick_1.grasp"])
    got = {(c["robot_body"], c["other"]): c["class"] for c in rep["forbidden_contacts"]}
    assert got == {("leftfinger[finger_link_on_target]", "x_1_main"): "finger_link_on_target",
                   ("leftfinger", "y_1_main"): "other", ("right_gripper", "x_1_main"): "other"}
    assert rep["clean"] is False and rep["clean_if_finger_link_on_target_allowed__preview_not_verdict"] is False
    assert rep["forbidden_by_class"] == {"finger_link_on_target": 1, "other": 2}
