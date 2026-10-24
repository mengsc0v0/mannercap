"""Step 7: LIBERO basket pick-place tasks as DATA.  One spec per task; program/plan/session are built by the same shared
templates and executor as task 1 (s3_libero_task).  Anything a task needs beyond this table (a new field, a changed
constant, a code branch) is a shared-code or task-specific change and must be logged as such."""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synthesis_v2.libero_backend import LiberoSession  # noqa: E402
from synthesis_v2.native_producer import GripPlan, StagePlan  # noqa: E402
from synthesis_v2.program_executor import ProgramPlan  # noqa: E402
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances  # noqa: E402

BASKET_RELEASE_DZ, BASKET_TRANSPORT_DZ = 0.14, 0.175   # measured on task 1's basket (s3_libero_task); reused as-is, same basket asset


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    objects: tuple            # ((role, backend object), ...) in program order
    target: tuple             # (role, backend object)
    suite: str = "libero_10"


SPECS = {
    1: TaskSpec(1, (("cream_cheese", "cream_cheese_1"), ("butter", "butter_1")), ("basket", "basket_1")),
    0: TaskSpec(0, (("alphabet_soup", "alphabet_soup_1"), ("tomato_sauce", "tomato_sauce_1")), ("basket", "basket_1")),
    7: TaskSpec(7, (("alphabet_soup", "alphabet_soup_1"), ("cream_cheese", "cream_cheese_1")), ("basket", "basket_1")),
}


# Named grasp-closure configurations (algorithm settings, saved here rather than patched at call sites).
GRIP_CONFIGS = {
    "baseline": GripPlan(3.0, 9.0),                                          # historical policy
    "settle": GripPlan(3.0, 9.0, settle_after_open=1),                       # wait one step after a release pulse, then re-judge
    "settle_reclose": GripPlan(3.0, 9.0, settle_after_open=1, reclose_max=2),  # + bounded re-close when re-checked force < lo
    "libero_t1": GripPlan(8.0, 40.0),                                        # force band already validated on task 1 (s3_libero_validate.GRIP)
}


def make_program(t: int) -> TaskProgram:
    sp = SPECS[t]
    roles = {r: n for r, n in sp.objects}; roles[sp.target[0]] = sp.target[1]
    inst = chain(*[pick_place_instances(r, sp.target[0]) for r, _ in sp.objects])
    inst = tuple(replace(i, params={**i.params, "place_dz": BASKET_RELEASE_DZ}) if i.primitive == "place" else i for i in inst)
    inst = tuple(replace(i, params={**i.params, "transport_dz": BASKET_TRANSPORT_DZ}) if i.primitive == "transport" else i for i in inst)
    return TaskProgram(f"libero_t{t}_basket_{len(sp.objects)}", "libero", f"libero:{sp.suite}:{t}", roles, inst, source="composed")


def make_plan(t: int, prog: TaskProgram, grip=None, v: float = 0.12, **kw) -> ProgramPlan:
    sp = SPECS[t]; grip = grip or GripPlan(3.0, 9.0)
    return ProgramPlan(prog, tuple(StagePlan(it.label, v) for it in prog.instances),
                       {f"{r}.pick_1.grasp": (grip[r] if isinstance(grip, dict) else grip) for r, _ in sp.objects}, **kw)


def make_session(t: int, init_index: int, prog: TaskProgram, render: bool = False) -> LiberoSession:
    sp = SPECS[t]
    return LiberoSession(t, init_index, suite=sp.suite, render=render, default_target=prog.objects[sp.objects[0][0]])
