"""The force-measurement binding is rollout state: it must be logged per step and survive a resume.

A program with two grasps measures each hold against its own object.  Resuming from a snapshot taken
inside the second object's subtask must measure the second object (not the construction-time default
nor the first grasp's stale target) and must reproduce the uninterrupted run exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from synthesis_v2.native_producer import GripPlan, StagePlan
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances

PROG = TaskProgram(
    "two_cube_putback_test", "robosuite_native", "stack",
    {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
    chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
)
PLAN = ProgramPlan(
    PROG, tuple(StagePlan(it.label, 0.12) for it in PROG.instances),
    {"cubeA.pick_1.grasp": GripPlan(3.0, 9.0), "cubeB.pick_1.grasp": GripPlan(28.0, 34.0)},
)
KEYS = ("actions", "left_mean_n", "right_mean_n", "valid_fraction", "stage", "force_target", "hold_instance")


@pytest.fixture(scope="module")
def runs():
    snaps: dict = {}
    full = produce_program_episode(PROG, 0, PLAN, keep_images=False, snapshots=snaps)
    resumed = {lab: produce_program_episode(PROG, 0, PLAN, keep_images=False, resume=snaps[lab])
               for lab in ("cubeB.pick_1.approach", "cubeB.pick_1.lift")}
    return full, snaps, resumed


def test_binding_logged_per_step_follows_each_grasp(runs):
    full, _, _ = runs
    a = full.arrays
    for role in ("cubeA", "cubeB"):
        gid = f"{role}.pick_1.grasp"
        m = a["hold_instance"] == gid
        assert m.sum() > 50
        assert set(a["force_target"][m]) == {role}
        lo, hi = PLAN.grip_for(gid).target_lo_n, PLAN.grip_for(gid).target_hi_n
        assert set(a["hold_lo_n"][m]) == {lo} and set(a["hold_hi_n"][m]) == {hi}
    assert np.isnan(a["hold_lo_n"][a["hold_instance"] == ""]).all()  # no requirement outside a grip window
    assert a["pos_cubeA"].shape == (len(a["actions"]), 3)


def test_snapshot_carries_the_binding(runs):
    _, snaps, _ = runs
    assert snaps["cubeA.pick_1.approach"].session["force_target"] is None
    assert snaps["cubeB.pick_1.approach"].session["force_target"] == "cubeA"  # stale until B's grasp rebinds
    assert snaps["cubeB.pick_1.lift"].session["force_target"] == "cubeB"


@pytest.mark.parametrize("label", ["cubeB.pick_1.approach", "cubeB.pick_1.lift"])
def test_resume_across_instance_boundary_is_exact(runs, label):
    full, _, resumed = runs
    r = resumed[label]
    assert r.status == full.status
    for k in KEYS:
        assert np.array_equal(np.asarray(full.arrays[k]), np.asarray(r.arrays[k])), k
    gid = "cubeB.pick_1.grasp"
    m = r.arrays["hold_instance"] == gid
    assert set(r.arrays["force_target"][m]) == {"cubeB"}
