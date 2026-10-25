"""Physics-substep recorder with rollback isolation keyed by snapshot identity.

Whatever happens between a ``session.save_state`` and the ``restore_state`` of *that* snapshot (the planner's settled
look-ahead, a recovery branch, a stage retry) is not part of the executed trajectory.  Each snapshot keeps its own copy of the
formal rows at save time (registry keyed by ``id(state)``, the state object is held so the id stays valid); a restore replaces the
formal rows by that copy, so

  * restoring an older snapshot after newer ones were taken drops everything after it,
  * the same snapshot can be restored repeatedly (every retry starts from the same prefix),
  * the dropped rows are kept apart in ``rolled_back`` (the cost stays visible) and never enter ``formal_arrays()``.

Control-step indices are re-used after a restore, so selecting rows by index alone is not safe; use ``formal_arrays()``.
``executed_substeps`` counts every physics substep actually simulated (formal + discarded), so cost accounting does not depend on
how often a branch was revived.  A recorder in a *new* session (stage resume) is seeded with the prefix of the run that took the
snapshot: ``seed_prefix(*other.rows_at(snapshot_state))``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class SubstepRecorder:
    def __init__(self, session: Any, sample_fn: Callable[[Any], tuple]) -> None:
        self.s = session
        self.sample_fn = sample_fn
        self.ctrl: list[int] = []
        self.data: list[tuple] = []
        self.rolled_back: list[dict[str, Any]] = []   # {"ctrl": [...], "n_substeps": int} per discarded branch
        self.executed_substeps = 0
        self._saved: dict[int, tuple[list[int], list[tuple], Any]] = {}   # id(state) -> (ctrl copy, data copy, state)

    def sample(self) -> None:
        """Call once per physics substep (after the step)."""
        self.ctrl.append(int(self.s.control_step_index))
        self.data.append(self.sample_fn(self.s))
        self.executed_substeps += 1

    def seed_prefix(self, ctrl: list[int], data: list[tuple]) -> None:
        """Formal rows inherited from the run that produced the snapshot a resumed session starts from (not re-simulated here)."""
        assert not self.ctrl, "seed before any sample"
        self.ctrl, self.data = list(ctrl), list(data)

    def rows_at(self, state: Any) -> tuple[list[int], list[tuple]]:
        c, d, _ = self._saved[id(state)]
        return list(c), list(d)

    def install_state_hooks(self) -> "SubstepRecorder":
        s = self.s
        orig_save, orig_restore = s.save_state, s.restore_state

        def save(*a, **k):
            st = orig_save(*a, **k)
            self._saved[id(st)] = (list(self.ctrl), list(self.data), st)
            return st

        def restore(st, *a, **k):
            out = orig_restore(st, *a, **k)
            snap = self._saved.get(id(st))
            if snap is not None:
                c, d, _ = snap
                common = 0                                   # rows are unique tuples: identity marks the shared prefix
                for x, y in zip(self.data, d):
                    if x is not y:
                        break
                    common += 1
                if len(self.ctrl) > common:
                    self.rolled_back.append({"ctrl": self.ctrl[common:], "n_substeps": len(self.ctrl) - common})
                self.ctrl, self.data = list(c), list(d)
            return out

        s.save_state, s.restore_state = save, restore
        return self

    def formal_arrays(self) -> tuple[np.ndarray, list[tuple]]:
        return np.asarray(self.ctrl, dtype=np.int64), list(self.data)

    def cost(self) -> dict[str, int]:
        formal_new = len(self.ctrl)
        return {"executed_substeps": self.executed_substeps, "formal_rows": formal_new,
                "discarded_branches": len(self.rolled_back), "discarded_substeps_listed": int(sum(b["n_substeps"] for b in self.rolled_back))}

    def rolled_back_cost(self) -> dict[str, int]:
        return {"branches": len(self.rolled_back), "substeps": int(sum(b["n_substeps"] for b in self.rolled_back))}
