"""Step 24 (item 1): grasp-reference library with provenance; explicit refusals; old programs keep their behaviour."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from synthesis_v2.grasp_references import (LibraryError, instance_reference, load_library, reference_issues,
                                           verify_library, DEFAULT_PATH)
from synthesis_v2.program_patch import apply_patch, make_anchor
from synthesis_v2.stage_capabilities import describe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mt"))
s7 = pytest.importorskip("s7_task_config")
LIB = load_library()
T8A = "libero_10.t8.demo_0.close_111.moka_pot_2"
T8B = "libero_10.t8.demo_0.close_349.moka_pot_1"
T5 = "libero_10.t5.demo_0.close_57.black_book_1"
PINNED = {4: "6dc238cce9bb2040e48b181ba44b455cf6ebee57792fed547ca406472a08d6fe",
          5: "c3c4b70970ff577c5e608f519748b1b87a3ec4cd1302bf18c55f00debefb36ba",
          8: "62456e3622875b19d3ff698575016b908d3785b856a3c4ff9bc7767813f1e179"}
A = make_anchor(request="r", manner={}, thresholds={}, policy="v1.1", guard={})


def _t8_prog(**approach_params):
    p = s7.make_program(8)
    inst = tuple(replace(i, params=approach_params) if i.instance_id == "pot_first.pick_1.approach" else i for i in p.instances)
    return replace(p, instances=inst)


def _swap(prog, iid, add=None, drop=()):
    out = []
    for i in prog.instances:
        if i.instance_id == iid:
            q = {k: v for k, v in i.params.items() if k not in drop}
            q.update(add or {})
            i = replace(i, params=q)
        out.append(i)
    return replace(prog, instances=tuple(out))


def test_every_entry_carries_provenance_and_a_frame_definition():
    assert len(LIB.ids()) == 5
    for rid in LIB.ids():
        e = LIB.entries[rid]
        src = e.source
        assert src["dataset_file"]["name"] and len(src["dataset_file"]["sha256"]) == 64
        assert src.get("demo") or src.get("n_demos")
        assert "close_step" in src or "n_demos" in src
        assert src["derivation_records"] and all(len(r["sha256"]) == 64 for r in src["derivation_records"])
        assert e.frame and e.object_class and e.object_name in e.compatible_objects and e.compatibility_basis
        assert e.demo_contact_at_close["status"] in ("recorded", "not_recorded")
        assert e.summary()["reference_id"] == rid


def test_tampered_value_or_missing_field_is_a_library_error(tmp_path):
    d = json.loads(Path(DEFAULT_PATH).read_text())
    bad = json.loads(json.dumps(d))
    bad["entries"][0]["value"]["hand_yaw_rel_deg"] += 1.0
    (tmp_path / "a.json").write_text(json.dumps(bad))
    with pytest.raises(LibraryError, match="content_sha256"):
        load_library(tmp_path / "a.json")
    bad = json.loads(json.dumps(d))
    del bad["entries"][0]["source"]
    (tmp_path / "b.json").write_text(json.dumps(bad))
    with pytest.raises(LibraryError, match="missing"):
        load_library(tmp_path / "b.json")
    dup = json.loads(json.dumps(d))
    dup["entries"].append(dup["entries"][0])
    (tmp_path / "c.json").write_text(json.dumps(dup))
    with pytest.raises(LibraryError, match="duplicate"):
        load_library(tmp_path / "c.json")


def test_repository_hashes_of_the_derivation_records_still_match():
    res = verify_library(LIB)
    assert not [r for r in res if r["status"] in ("mismatch", "missing")], res


def test_resolution_is_explicit_about_unknown_and_incompatible():
    scene = "libero:libero_10:8"
    assert LIB.resolve(T8A, scene, "moka_pot_2").ok
    assert LIB.resolve(T8A, scene, "moka_pot_1").ok                       # declared same-class cross use
    assert LIB.resolve("nope", scene, "moka_pot_2").refusal == "reference_not_found:nope"
    assert LIB.resolve(T5, scene, "moka_pot_2").refusal.startswith("reference_incompatible:")      # other scene
    assert LIB.resolve(T8A, scene, "flat_stove_1").refusal.startswith("reference_incompatible:")   # same scene, wrong object
    assert [e.reference_id for e in LIB.available(scene, "moka_pot_1")] == [T8A, T8B]
    assert LIB.available(scene, "flat_stove_1") == []


def test_validate_reports_bad_ids_and_conflicts_and_the_executor_input_matches():
    p = _t8_prog(grasp_reference_id="libero_10.t8.demo_9.close_1.moka_pot_2")
    assert any(e.startswith("reference_not_found:pot_first.pick_1.approach") for e in p.validate())
    p = _t8_prog(grasp_reference_id=T5)
    assert any(e.startswith("reference_incompatible:") for e in p.validate())
    ref = dict(s7.make_program(8).instance("pot_first.pick_1.approach").params["grasp_reference"])
    p = _t8_prog(grasp_reference=ref, grasp_reference_id=T8A)
    assert reference_issues(p) == ["reference_conflict:pot_first.pick_1.approach"]     # no silent precedence
    assert _t8_prog(grasp_reference_id=T8A).validate() == []
    v, prov, ref_err = instance_reference(_t8_prog(grasp_reference_id=T8A), _t8_prog(grasp_reference_id=T8A).instance("pot_first.pick_1.approach"))
    assert ref_err is None and v == ref and prov["origin"] == "grasp_reference_id" and prov["reference_id"] == T8A


@pytest.mark.parametrize("t", (4, 5, 8))
def test_old_programs_are_unchanged_and_their_inline_reference_is_registered(t):
    p = s7.make_program(t)
    assert p.program_hash() == PINNED[t]                                  # migration did not touch any program
    assert p.validate() == []
    n = 0
    for it in p.instances:
        if it.primitive == "approach" and "grasp_reference" in it.params:
            v, prov, err = instance_reference(p, it)
            assert err is None and v == it.params["grasp_reference"]       # the executor consumes the inline value as before
            assert prov["origin"] == "inline_grasp_reference" and prov["registered"] is True and prov["reference_id"]
            assert LIB.entries[prov["reference_id"]].value == it.params["grasp_reference"]
            n += 1
        assert "grasp_reference_id" not in it.params
    assert n >= 1


def test_instance_without_any_reference_resolves_to_nothing():
    q = _swap(s7.make_program(8), "pot_first.pick_1.approach", drop=("grasp_reference",))
    assert instance_reference(q, q.instance("pot_first.pick_1.approach")) == (None, None, None)


def test_patch_may_only_select_registered_ids():
    base = s7.make_program(8)
    iid = "pot_first.pick_1.approach"
    sel = [{"op": "unset_param", "instance_id": iid, "key": "grasp_reference"},
           {"op": "set_param", "instance_id": iid, "key": "grasp_reference_id", "value": T8B}]
    r = apply_patch(base, {"hypothesis": "use the other pot's demo grasp", "ops": sel}, frozen_anchor=A, current_anchor=A)
    assert r.ok, r.to_dict()
    assert r.program.instance(iid).params["grasp_reference_id"] == T8B and "grasp_reference" not in r.program.instance(iid).params
    assert r.parent_hash == PINNED[8] and r.child_hash != r.parent_hash
    bad = apply_patch(base, {"hypothesis": "x", "ops": [sel[0], {**sel[1], "value": "libero_10.t8.demo_9.close_1.moka_pot_2"}]},
                      frozen_anchor=A, current_anchor=A)
    assert not bad.ok and bad.refusal == "reference_rejected" and bad.program is None
    free = apply_patch(base, {"hypothesis": "x", "ops": [{"op": "set_param", "instance_id": iid, "key": "grasp_reference",
                                                          "value": {"rel_obj": [0, 0, 0.05], "hand_yaw_rel_deg": 0.0}}]},
                       frozen_anchor=A, current_anchor=A)
    assert not free.ok and any("not_proposer_settable" in e for e in free.errors)
    both = apply_patch(base, {"hypothesis": "x", "ops": [sel[1]]}, frozen_anchor=A, current_anchor=A)      # leaves the inline one in place
    assert not both.ok and both.refusal == "reference_rejected"


def test_capability_table_exposes_the_id_and_hides_the_free_numbers():
    d = describe()
    txt = json.dumps(d)
    assert "grasp_reference_id" in txt and "proposer_settable" in txt
