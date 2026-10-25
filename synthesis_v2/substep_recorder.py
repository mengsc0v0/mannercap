"""Physics-substep recorder with rollback isolation.

Same isolation rule as ``contact_audit.ContactRecorder``: whatever happens between a ``session.save_state`` and its
``restore_state`` (the planner's settled look-ahead, a recovery branch) is not part of the executed trajectory.  On restore
the rows recorded since the save are moved to ``rolled_back`` (kept, so the generation cost of the branch stays visible)
and the formal rows are exactly the ones the final trajectory produced.  Control-step indices are re-used after a restore, so
selection by index alone is not safe; consumers must use ``formal_arrays()``.
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
        self.rolled_back: list[dict[str, Any]] = []   # {"ctrl": [...], "n_substeps": int}
        self._marks: list[int] = []

    def sample(self) -> None:
        """Call once per physics substep (after the step)."""
        self.ctrl.append(int(self.s.control_step_index))
        self.data.append(self.sample_fn(self.s))

    def install_state_hooks(self) -> "SubstepRecorder":
        s = self.s
        orig_save, orig_restore = s.save_state, s.restore_state

        def save(*a, **k):
            self._marks.append(len(self.ctrl))
            return orig_save(*a, **k)

        def restore(st, *a, **k):
            out = orig_restore(st, *a, **k)
            if self._marks:
                m = self._marks.pop()
                if len(self.ctrl) > m:
                    self.rolled_back.append({"ctrl": self.ctrl[m:], "n_substeps": len(self.ctrl) - m})
                    del self.ctrl[m:], self.data[m:]
            return out

        s.save_state, s.restore_state = save, restore
        return self

    def formal_arrays(self) -> tuple[np.ndarray, list[tuple]]:
        return np.asarray(self.ctrl, dtype=np.int64), list(self.data)

    def rolled_back_cost(self) -> dict[str, int]:
        return {"branches": len(self.rolled_back), "substeps": int(sum(b["n_substeps"] for b in self.rolled_back))}
