"""Step 24: undeclared / out-of-domain stage parameters are reported, never silently ignored."""
from dataclasses import replace

from synthesis_v2.stage_capabilities import describe, param_issues, primitive_issues
from synthesis_v2.task_program import chain, pick_place_instances, TaskProgram


def _prog(**by_prim):
    inst = tuple(replace(i, params=by_prim.get(i.primitive, {})) for i in chain(pick_place_instances("a", "t")))
    return TaskProgram("p", "libero", "libero:x:0", {"a": "a_obj", "t": "t_obj"}, inst, source="composed")


def test_declared_params_pass_and_unknown_key_is_refused():
    assert param_issues(_prog(lift={"lift_dz": 0.12}))[0] == []
    errs = param_issues(_prog(lift={"lift_dzz": 0.12}))[0]
    assert errs == ["capability_unsupported:a.pick_1.lift:lift_dzz"]
    assert any(e.startswith("capability_unsupported:") for e in _prog(lift={"speed_boost": 2}).validate())


def test_key_declared_for_another_primitive_is_refused_but_historical_co_application_is_a_note():
    errs, notes = param_issues(_prog(grasp={"lift_dz": 0.1}))
    assert errs and errs[0].endswith("not_read_by_grasp")
    errs, notes = param_issues(_prog(place={"transport_dz": 0.09}))          # T5/T8 configs co-apply it
    assert errs == [] and notes == ["param_inert:a.pick_1.place:transport_dz"]


def test_domain_and_type_are_checked():
    for bad in (-0.2, 5.0, float("nan"), "0.1", True):
        assert param_issues(_prog(lift={"lift_dz": bad}))[0], bad
    assert param_issues(_prog(transport={"target_offset_site": [0, 0]}))[0]
    assert param_issues(_prog(transport={"target_align": "bottom"}))[0]


def test_grasp_reference_fields_and_quaternion_are_checked():
    ok = {"rel_obj": [0.0, 0.0, 0.05], "hand_yaw_rel_deg": 10.0, "hand_quat_obj_xyzw": [0, 0, 0, 1]}
    assert param_issues(_prog(approach={"grasp_reference": ok}))[0] == []
    for bad in ({**ok, "hand_quat_obj_xyzw": [0, 0, 0, 2]}, {**ok, "extra": 1}, {"rel_obj": [0, 0, 0]},
                {**ok, "rel_obj": [0, 0, 5.0]}):
        assert param_issues(_prog(approach={"grasp_reference": bad}))[0], bad


def test_plan_flag_is_required_for_grasp_placement_keys():
    p = _prog(approach={"grasp_candidate": {"yaw_deg": 0.0}})
    assert param_issues(p)[0] == []                                           # structure alone is fine
    errs = param_issues(p, type("P", (), {"approach_check": False})())[0]
    assert errs == ["capability_requires:a.pick_1.approach:grasp_candidate:approach_check"]
    assert param_issues(p, type("P", (), {"approach_check": True})())[0] == []


def test_primitives_the_native_executor_cannot_run_are_reported():
    p = _prog()
    sweep = replace(p.instances[0], instance_id="a.sweep", label="a.sweep", primitive="sweep")
    assert primitive_issues(replace(p, instances=(sweep,))) == ["capability_unsupported:a.sweep:primitive:sweep"]
    assert primitive_issues(p) == []


def test_describe_lists_every_declared_key():
    d = describe()
    assert {"lift_dz", "transport_dz", "place_dz", "grasp_reference", "grasp_candidate", "grasp_grid"} <= set(d["params"])
    assert d["params"]["grasp_reference"]["required"] == ["rel_obj", "hand_yaw_rel_deg"]


def test_existing_task_programs_are_unchanged_and_valid():
    import sys
    sys.path.insert(0, "scripts/mt")
    from s7_task_config import SPECS, make_program
    for t in SPECS:
        p = make_program(t)
        assert p.validate() == [] and primitive_issues(p) == []
