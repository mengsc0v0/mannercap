"""Per-instance grasp-force binding and composed-program plumbing (design v3 step 1)."""
from __future__ import annotations

import pytest

from synthesis_v2.native_osc_env import NativeSession
from synthesis_v2.native_producer import GripPlan, StagePlan
from synthesis_v2.program_executor import ProgramPlan, program_plan_from_production
from synthesis_v2.task_program import TaskProgram, chain, legacy_program, pick_place_instances


def two_cube_program() -> TaskProgram:
    return TaskProgram(
        "two_cube_putback", "robosuite_native", "stack",
        {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
        source="composed")


@pytest.fixture(scope="module")
def stack_session():
    s = NativeSession("stack", 0, render=False)
    s.reset()
    yield s
    s.close()


def test_geom_ids_resolve_per_object(stack_session):
    a, b = stack_session.geom_ids_of("cubeA"), stack_session.geom_ids_of("cubeB")
    assert a and b
    assert set(a).isdisjoint(set(b))
    assert stack_session.geom_ids_of("no_such_object") == ()


def test_set_force_target_rebinds_measurement(stack_session):
    stack_session.set_force_target("cubeA")
    first = stack_session._binding
    stack_session.set_force_target("cubeB")
    assert stack_session._binding is not first
    assert stack_session._binding != first
    with pytest.raises(ValueError):
        stack_session.set_force_target("no_such_object")


def test_program_plan_carries_one_grip_per_grasp_instance():
    prog = two_cube_program()
    grips = {"cubeA.pick_1.grasp": GripPlan(2.0, 6.0), "cubeB.pick_1.grasp": GripPlan(24.0, 34.0)}
    plan = ProgramPlan(prog, tuple(StagePlan(it.label, 0.12) for it in prog.instances), grips)
    assert plan.grip_for("cubeA.pick_1.grasp").target_hi_n == 6.0
    assert plan.grip_for("cubeB.pick_1.grasp").target_hi_n == 34.0
    assert plan.plan_hash() != ProgramPlan(
        prog, plan.stages, {k: grips[v] for k, v in
                            zip(grips, reversed(list(grips)))}).plan_hash()


def test_legacy_plan_maps_single_grip_onto_every_grasp_instance():
    from synthesis_v2.native_producer import ProductionPlan, default_stage_plans

    pp = ProductionPlan("stack", default_stage_plans("stack"), GripPlan(10.0, 16.0))
    plan = program_plan_from_production(pp)
    assert plan.program.program_hash() == legacy_program("stack").program_hash()
    assert set(plan.grips) == {it.label for it in plan.program.of_primitive("grasp")}
