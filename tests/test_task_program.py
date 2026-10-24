"""TaskProgram contract: legacy equivalence, instance scoping, reference resolution."""
from __future__ import annotations

import pytest

from synthesis_v2.stage_table import stage_ids
from synthesis_v2.task_program import (
    GRIP_HOLD_TYPES, PRIMITIVES, StageInstance, TaskProgram, chain, dimension_instances, labels,
    legacy_program, pick_place_instances, resolve_stage_ref,
)

TASKS = ("lift", "stack", "nut_assembly", "wipe")


@pytest.mark.parametrize("task", TASKS)
def test_legacy_program_is_valid_and_label_compatible(task):
    p = legacy_program(task)
    assert p.validate() == []
    assert labels(p) == tuple(stage_ids(task))


@pytest.mark.parametrize("task", TASKS)
def test_legacy_program_hash_is_stable(task):
    p = legacy_program(task)
    assert p.program_hash() == legacy_program(task).program_hash()
    assert TaskProgram.from_dict(p.to_dict()).program_hash() == p.program_hash()


@pytest.mark.parametrize("task", TASKS)
def test_instance_ids_unique_and_preds_point_backwards(task):
    p = legacy_program(task)
    ids = list(p.ids())
    assert len(ids) == len(set(ids))
    for i, it in enumerate(p.instances):
        for pred in it.preds:
            assert ids.index(pred) < i


def test_primitive_needs_are_declared_for_every_instance():
    for task in TASKS:
        for it in legacy_program(task).instances:
            assert it.primitive in PRIMITIVES


def test_repeated_subtasks_get_distinct_instance_ids():
    inst = chain(pick_place_instances("cubeA", "spotA", k=1), pick_place_instances("cubeA", "spotA", k=2))
    ids = [i.instance_id for i in inst]
    assert len(ids) == len(set(ids))
    assert "cubeA.pick_1.grasp" in ids and "cubeA.pick_2.grasp" in ids


def test_chain_links_groups_without_duplicating_preds():
    g1 = pick_place_instances("cubeA", "spotA", k=1)
    g2 = pick_place_instances("cubeB", "spotB", k=1)
    inst = chain(g1, g2)
    assert inst[len(g1)].preds == (g1[-1].instance_id,)
    assert inst[0].preds == ()


def test_two_object_program_validates_and_has_two_grasps():
    p = TaskProgram(
        "two_cube_putback", "robosuite_native", "stack",
        {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
        source="composed")
    assert p.validate() == []
    grasps = p.of_primitive("grasp")
    assert len(grasps) == 2
    assert {g.object_role for g in grasps} == {"cubeA", "cubeB"}


def test_unbound_object_role_is_rejected():
    p = TaskProgram("bad", "robosuite_native", "stack", {"cubeA": "cubeA"},
                    tuple(pick_place_instances("cubeX", "cubeA")), source="composed")
    assert any(e.startswith("unbound_object:") for e in p.validate())


def test_grip_scope_runs_from_grasp_to_release_of_the_same_object():
    p = legacy_program("stack")
    assert p.grip_scope("cubeA.pick_1.grasp") == (
        "cubeA.pick_1.grasp", "cubeA.pick_1.lift", "cubeA.pick_1.transport", "cubeA.pick_1.place")
    two = TaskProgram(
        "two", "robosuite_native", "stack",
        {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
        source="composed")
    # a second grasp of another object never falls inside the first grasp's scope
    scope_a = set(two.grip_scope("cubeA.pick_1.grasp"))
    assert all(i.instance_id not in scope_a for i in two.instances if i.object_role == "cubeB")
    assert all(p.instance(i).primitive in ("grasp",) + GRIP_HOLD_TYPES for i in p.grip_scope("cubeA.pick_1.grasp"))


def test_resolve_stage_ref_ok_unknown_ambiguous():
    p = legacy_program("stack")
    ids, st = resolve_stage_ref(p, "pick")
    assert st == "ok" and ids == ("cubeA.pick_1.grasp",)
    assert resolve_stage_ref(p, "banana")[1] == "unknown"
    two = TaskProgram(
        "two", "robosuite_native", "stack",
        {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
        source="composed")
    ids, st = resolve_stage_ref(two, "grasp")
    assert st == "ambiguous" and len(ids) == 2
    ids, st = resolve_stage_ref(two, "grasp", object_role="cubeB")
    assert st == "ok" and ids == ("cubeB.pick_1.grasp",)


def test_dimension_instances_match_primitive_table():
    p = legacy_program("wipe")
    assert "table.wipe_1.sweep" in dimension_instances(p, "press")
    assert dimension_instances(legacy_program("lift"), "press") == ()


def test_stage_instance_is_frozen():
    it = legacy_program("lift").instances[0]
    with pytest.raises(Exception):
        it.label = "x"  # type: ignore[misc]
