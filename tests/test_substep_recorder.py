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
