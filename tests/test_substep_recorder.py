"""Rollback isolation of the physics-substep recorder (mirrors tests/test_contact_audit.py)."""
import numpy as np

from synthesis_v2.substep_recorder import SubstepRecorder

SUBSTEPS = 3


class _FakeSession:
    def __init__(self):
        self.control_step_index, self.force = 0, 0.0
        self.recorder = None

    def step(self, a):
        for _ in range(SUBSTEPS):
            self.recorder.sample()
        self.control_step_index += 1

    def save_state(self):
        return {"c": self.control_step_index}

    def restore_state(self, st):
        self.control_step_index = st["c"]


def _mk():
    s = _FakeSession()
    s.recorder = SubstepRecorder(s, lambda sess: (sess.force,)).install_state_hooks()
    return s


def test_force_only_in_lookahead_branch_is_not_in_formal_rows():
    s = _mk()
    st = s.save_state()
    s.force = 40.0
    for _ in range(5):
        s.step(None)                       # look-ahead: the hand squeezes here
    s.force = 0.0
    s.restore_state(st)
    for _ in range(5):
        s.step(None)                       # executed trajectory: no force
    ctrl, rows = s.recorder.formal_arrays()
    assert len(ctrl) == 5 * SUBSTEPS and max(r[0] for r in rows) == 0.0
    assert np.array_equal(np.bincount(ctrl), [SUBSTEPS] * 5)          # one substep group per real control step
    assert s.recorder.rolled_back_cost() == {"branches": 1, "substeps": 5 * SUBSTEPS}   # branch cost stays visible
    assert max(x for b in s.recorder.rolled_back for x in b["ctrl"]) == 4


def test_force_in_the_executed_trajectory_is_kept():
    s = _mk()
    st = s.save_state(); s.step(None); s.restore_state(st)
    s.force = 12.0
    for _ in range(3):
        s.step(None)
    ctrl, rows = s.recorder.formal_arrays()
    assert len(ctrl) == 3 * SUBSTEPS and all(r[0] == 12.0 for r in rows)


def test_rows_before_the_branch_survive_the_rollback():
    s = _mk()
    s.force = 7.0; s.step(None); s.force = 0.0
    st = s.save_state(); s.force = 99.0; s.step(None); s.restore_state(st)
    ctrl, rows = s.recorder.formal_arrays()
    assert len(ctrl) == SUBSTEPS and all(r[0] == 7.0 for r in rows)


def test_index_only_selection_would_be_contaminated_without_rollback():
    """Discriminating control: a recorder that never rolls back selects look-ahead rows by control index."""
    s = _FakeSession()
    s.recorder = SubstepRecorder(s, lambda sess: (sess.force,))      # hooks not installed
    st = s.save_state(); s.force = 40.0
    for _ in range(5):
        s.step(None)
    s.force = 0.0; s.restore_state(st)
    for _ in range(5):
        s.step(None)
    ctrl, rows = s.recorder.formal_arrays()
    sel = np.isin(ctrl, [1, 2])                                        # a "hold window" selected by index
    assert len(ctrl) == 10 * SUBSTEPS and max(rows[i][0] for i in np.flatnonzero(sel)) == 40.0


# ---- snapshot-identity restore (stage retry restores an OLDER snapshot, and the same one repeatedly)
def _run(s, force, n):
    s.force = force
    for _ in range(n):
        s.step(None)


def test_restore_older_snapshot_drops_everything_after_it_even_with_newer_snapshots():
    s = _mk()
    _run(s, 1.0, 1)
    A = s.save_state()
    _run(s, 2.0, 2)
    B = s.save_state()
    _run(s, 3.0, 2)
    s.restore_state(A)                      # skip over B: must drop rows of both branches, not only B's
    ctrl, rows = s.recorder.formal_arrays()
    assert len(ctrl) == SUBSTEPS and all(r[0] == 1.0 for r in rows)
    assert s.recorder.rolled_back_cost()["substeps"] == 4 * SUBSTEPS
    assert s.control_step_index == 1


def test_same_snapshot_restored_twice_clears_the_second_branch_too():
    s = _mk()
    _run(s, 1.0, 1)
    A = s.save_state()
    _run(s, 2.0, 3)
    s.restore_state(A)
    _run(s, 9.0, 2)                         # retry branch
    s.restore_state(A)                      # second restore of the same snapshot
    _run(s, 5.0, 2)                         # final executed continuation
    ctrl, rows = s.recorder.formal_arrays()
    assert [r[0] for r in rows[::SUBSTEPS]] == [1.0, 5.0, 5.0]
    assert 9.0 not in {r[0] for r in rows} and 2.0 not in {r[0] for r in rows}
    assert list(np.unique(ctrl)) == [0, 1, 2]
    c = s.recorder.cost()
    assert c["executed_substeps"] == (1 + 3 + 2 + 2) * SUBSTEPS and c["formal_rows"] == 3 * SUBSTEPS
    assert c["executed_substeps"] - c["formal_rows"] == c["discarded_substeps_listed"]


def test_resumed_recorder_is_seeded_from_the_snapshot_prefix():
    s1 = _mk()
    _run(s1, 1.0, 2)
    A = s1.save_state()
    _run(s1, 7.0, 2)
    s2 = _mk()                              # new session, continues from A
    s2.recorder.seed_prefix(*s1.recorder.rows_at(A))
    s2.restore_state(A)
    _run(s2, 4.0, 1)
    ctrl, rows = s2.recorder.formal_arrays()
    assert [r[0] for r in rows[::SUBSTEPS]] == [1.0, 1.0, 4.0] and list(np.unique(ctrl)) == [0, 1, 2]
    assert s2.recorder.executed_substeps == SUBSTEPS      # only the resumed part was simulated here
