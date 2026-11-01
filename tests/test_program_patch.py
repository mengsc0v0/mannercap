"""Step 24: a bounded, params-only patch; frozen anchors; structured refusals."""
from dataclasses import replace

from synthesis_v2.program_patch import anchor_diff, apply_patch, make_anchor, skeleton_hash
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances

A = make_anchor(request="put a on t", manner={"speed": "slow"}, thresholds={"lift": 0.05}, policy="v1.1", guard={"on": False})


def _prog():
    inst = tuple(replace(i, params={"lift_dz": 0.12} if i.primitive == "lift" else {}) for i in chain(pick_place_instances("a", "t")))
    return TaskProgram("p", "libero", "libero:x:0", {"a": "a_obj", "t": "t_obj"}, inst, source="composed")


def _patch(*ops, hyp="the entry is too low"):
    return {"hypothesis": hyp, "ops": list(ops)}


def _set(iid, key, value):
    return {"op": "set_param", "instance_id": iid, "key": key, "value": value}


def test_valid_patch_changes_only_params_and_reports_the_diff():
    p = _prog()
    r = apply_patch(p, _patch(_set("a.pick_1.lift", "lift_dz", 0.2)), frozen_anchor=A, current_anchor=A)
    assert r.ok and r.refusal is None
    assert r.diff == ({"instance_id": "a.pick_1.lift", "key": "lift_dz", "old": 0.12, "new": 0.2},)
    assert r.program.instance("a.pick_1.lift").params["lift_dz"] == 0.2
    assert p.instance("a.pick_1.lift").params["lift_dz"] == 0.12                  # original untouched
    assert r.parent_hash == p.program_hash() and r.child_hash != r.parent_hash
    assert skeleton_hash(r.program) == skeleton_hash(p) == r.skeleton_hash
    assert r.hypothesis == "the entry is too low"


def test_changed_anchor_blocks_the_patch():
    for name, val in (("request", "put a on b"), ("manner", {"speed": "fast"}), ("thresholds", {"lift": 0.01}),
                      ("policy", "v2"), ("guard", {"on": True})):
        cur = make_anchor(**{**dict(request="put a on t", manner={"speed": "slow"}, thresholds={"lift": 0.05}, policy="v1.1",
                                    guard={"on": False}), name: val})
        r = apply_patch(_prog(), _patch(_set("a.pick_1.lift", "lift_dz", 0.2)), frozen_anchor=A, current_anchor=cur)
        assert not r.ok and r.refusal == "anchor_changed" and r.errors == (f"anchor_changed:{name}",)
    assert anchor_diff(A, {**A, "extra": "x"}) == ["extra"]


def test_patch_cannot_touch_anything_but_params():
    p = _prog()
    r = apply_patch(p, {**_patch(_set("a.pick_1.lift", "lift_dz", 0.2)), "threshold": 0.0}, frozen_anchor=A, current_anchor=A)
    assert not r.ok and r.refusal == "invalid_patch" and "patch_forbidden_field:threshold" in r.errors
    r = apply_patch(p, _patch({"op": "replace_primitive", "instance_id": "a.pick_1.lift", "key": "lift_dz", "value": 1}),
                    frozen_anchor=A, current_anchor=A)
    assert not r.ok and r.refusal == "capability_unsupported" and r.errors[0] == "capability_unsupported:op:replace_primitive"
    r = apply_patch(p, _patch({**_set("a.pick_1.lift", "lift_dz", 0.2), "primitive": "grasp"}), frozen_anchor=A, current_anchor=A)
    assert not r.ok and "patch_forbidden_field:ops[0].primitive" in r.errors


def test_unknown_key_out_of_domain_and_unknown_instance_are_refused_with_reasons():
    p = _prog()
    r = apply_patch(p, _patch(_set("a.pick_1.lift", "approach_speed", 1.0)), frozen_anchor=A, current_anchor=A)
    assert r.refusal == "capability_unsupported" and r.errors == ("capability_unsupported:a.pick_1.lift:approach_speed",)
    r = apply_patch(p, _patch(_set("a.pick_1.lift", "lift_dz", 9.0)), frozen_anchor=A, current_anchor=A)
    assert r.refusal == "invalid_patch" and r.errors[0].startswith("param_invalid:a.pick_1.lift:lift_dz:out_of_range")
    r = apply_patch(p, _patch(_set("zzz", "lift_dz", 0.1)), frozen_anchor=A, current_anchor=A)
    assert r.refusal == "invalid_patch" and r.errors == ("unknown_instance:zzz",)
    assert r.child_hash == "" and r.program is None


def test_key_for_the_wrong_primitive_is_refused_after_the_edit():
    r = apply_patch(_prog(), _patch(_set("a.pick_1.grasp", "lift_dz", 0.1)), frozen_anchor=A, current_anchor=A)
    assert not r.ok and r.refusal == "invalid_program" and r.errors[0].endswith("not_read_by_grasp")


def test_missing_hypothesis_empty_and_oversized_patches_are_refused():
    p = _prog()
    assert "patch_missing_hypothesis" in apply_patch(p, {"ops": [_set("a.pick_1.lift", "lift_dz", 0.1)]}, frozen_anchor=A,
                                                    current_anchor=A).errors
    assert "patch_has_no_ops" in apply_patch(p, {"hypothesis": "x", "ops": []}, frozen_anchor=A, current_anchor=A).errors
    big = [_set("a.pick_1.lift", "lift_dz", 0.1)] * 9
    assert any(e.startswith("patch_too_large") for e in apply_patch(p, _patch(*big), frozen_anchor=A, current_anchor=A).errors)
    assert apply_patch(p, "not a mapping", frozen_anchor=A, current_anchor=A).refusal == "invalid_patch"


def test_unset_param_and_json_roundtrip():
    p = _prog()
    r = apply_patch(p, _patch({"op": "unset_param", "instance_id": "a.pick_1.lift", "key": "lift_dz"}), frozen_anchor=A, current_anchor=A)
    assert r.ok and "lift_dz" not in r.program.instance("a.pick_1.lift").params
    assert TaskProgram.from_dict(r.program.to_dict()).program_hash() == r.child_hash
    bad = apply_patch(p, _patch({"op": "unset_param", "instance_id": "a.pick_1.approach", "key": "grasp_reference"}), frozen_anchor=A, current_anchor=A)
    assert bad.refusal == "invalid_patch" and bad.errors == ("unset_missing_param:a.pick_1.approach:grasp_reference",)
    import json
    json.dumps(r.to_dict())
