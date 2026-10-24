"""Step 3: one LIBERO task through the shared primitives (task spec + helpers; no task-specific control code)."""
from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthesis_v2.libero_backend import LiberoSession  # noqa: E402
from synthesis_v2.native_producer import GripPlan, StagePlan  # noqa: E402
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode  # noqa: E402
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances  # noqa: E402

TASK_ID, SUITE = 1, "libero_10"
OBJECTS = (("cream_cheese", "cream_cheese_1"), ("butter", "butter_1"))
TARGET = ("basket", "basket_1")
BASKET_RELEASE_DZ = 0.14  # eef height above the basket body origin at release
# Carry height (eef above basket body origin).  Measured geometry (scripts/mt/s5_dip_diag.py, Step 5): basket rim top
# 0.141 m above the body origin; lowest gripper collision point 0.012 m below the eef site; margin 0.02 m for
# tracking error  ->  0.141 + 0.012 + 0.02 = 0.173, rounded up.  v1 carried at the generic 0.15 and the left finger
# (lowest point 0.139) clipped the rim in 4/8 init states.
BASKET_TRANSPORT_DZ = 0.175


def make_program(n_obj: int = 2) -> TaskProgram:
    objs = OBJECTS[:n_obj]
    roles = {r: n for r, n in objs}
    roles[TARGET[0]] = TARGET[1]
    inst = chain(*[pick_place_instances(r, TARGET[0]) for r, _ in objs])
    # task-specific data (not code): release above the basket rim instead of the cube-on-cube height
    inst = tuple(replace(i, params={**i.params, "place_dz": BASKET_RELEASE_DZ}) if i.primitive == "place" else i for i in inst)
    inst = tuple(replace(i, params={**i.params, "transport_dz": BASKET_TRANSPORT_DZ}) if i.primitive == "transport" else i for i in inst)
    return TaskProgram(f"libero_t{TASK_ID}_basket_{n_obj}", "libero", f"libero:{SUITE}:{TASK_ID}", roles, inst,
                       source="composed")


def make_plan(prog: TaskProgram, grip=GripPlan(3.0, 9.0), v: float = 0.12) -> ProgramPlan:
    """``grip``: one GripPlan for every object, or a {role: GripPlan} mapping (per-object manner)."""
    stages = tuple(StagePlan(it.label, v) for it in prog.instances)
    return ProgramPlan(prog, stages, {f"{r}.pick_1.grasp": (grip[r] if isinstance(grip, dict) else grip) for r, _ in OBJECTS})


def make_session(init_index: int, prog: TaskProgram, render: bool = False) -> LiberoSession:
    return LiberoSession(TASK_ID, init_index, suite=SUITE, render=render, default_target=prog.objects[OBJECTS[0][0]])


def run(init_index: int, n_obj: int = 2, keep_images: bool = False, grip=GripPlan(3.0, 9.0), **kw):
    prog = make_program(n_obj)
    s = make_session(init_index, prog, keep_images)
    ep = produce_program_episode(prog, init_index, make_plan(prog, grip), keep_images=keep_images, session=s, **kw)
    return prog, s, ep
