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
