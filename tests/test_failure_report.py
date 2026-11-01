"""Step 24 (item 6): FailureReport -- classes kept apart, first failure + evidence source, unknown instead of zero/guess,
snapshots listed but never declared valid."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthesis_v2.failure_report import (CLASSES, build_failure_report, from_episode, from_script_record, is_unknown)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mt"))
s7 = pytest.importorskip("s7_task_config")
P8 = s7.make_program(8)
R1 = ROOT / "reports/step22_t8_onboard/round1/init0.json"
PLAN = ROOT / "reports/step23_shared_fixes/t8_first_grasp_init0_fullpose.json"
needs_records = pytest.mark.skipif(not (R1.exists() and PLAN.exists()), reason="recorded runs not in this checkout")


@needs_records
def test_planning_rejection_is_its_own_class_with_planner_evidence():
    r = from_script_record(P8, json.loads(PLAN.read_text()))
    ff = r["first_failure"]
    assert r["classes_present"] == ["planning_rejected"] and ff["class"] == "planning_rejected"
    assert ff["requirement_id"] == "exec.approach.feasible" and ff["instance_id"] == "pot_first.pick_1.approach"
    c = ff["detail"]["candidates"][0]
    assert c["min_clearance_m"] == -0.00025 and c["blocking"] == "moka_pot_2" and c["phase"] == "descent" and c["feasible"] is False
    assert ff["detail"]["grid_source"] == "grasp_reference"
    assert ff["control_step"] == 0 and is_unknown(ff["pose_difference"])           # no motion -> no reached pose
    assert is_unknown(ff["detail"]["reference"])                                    # record predates the provenance field
    assert "first_failure.pose_difference" in r["unknown_fields"]


@needs_records
def test_execution_failure_and_audit_failure_are_separate_and_ordered_by_stage_and_step():
    r = from_script_record(P8, json.loads(R1.read_text()))
    assert set(CLASSES) >= set(r["classes_present"]) and {"execution_failed", "audit_failed"} <= set(r["classes_present"])
    ex = [f for f in r["failures"] if f["class"] == "execution_failed"]
    au = [f for f in r["failures"] if f["class"] == "audit_failed"]
    assert len(ex) == 1 and ex[0]["requirement_id"] == "exec.stage_reached.descend" and ex[0]["instance_id"] == "pot_first.pick_1.descend"
    assert len(au) == 3 and all(f["requirement_id"] == "contact_policy_v1.1.unexcused_contact" for f in au)
    assert [f["control_step"] for f in au] == [60, 293, 513]
    assert [f["instance_id"] for f in au][0] == "pot_first.pick_1.descend"
    # sorted by (stage index, control step); the unknown-step timeout comes after the contact that began earlier in the same stage
    key = [(f["stage_index"] if f["stage_index"] is not None else 10 ** 9) for f in r["failures"]]
    assert key == sorted(key)
    assert r["first_failure"]["class"] == "audit_failed" and r["first_failure"]["control_step"] == 60
    assert "temporal" in r["ordering_note"] and r["status_code"] == "stage_timeout:pot_first.pick_1.descend"


@needs_records
def test_what_the_records_cannot_supply_is_unknown_not_zero():
    r = from_script_record(P8, json.loads(R1.read_text()))
    ex = next(f for f in r["failures"] if f["class"] == "execution_failed")
    assert is_unknown(ex["control_step"]) and is_unknown(ex["pose_difference"]) and is_unknown(ex["detail"]["eef_pos_at_failure"])
    au = next(f for f in r["failures"] if f["class"] == "audit_failed")
    assert is_unknown(au["detail"]["duration_s"]) and is_unknown(au["detail"]["n_substeps"])
    assert au["detail"]["span_control_steps"] == 233 and au["detail"]["max_n"] == 57.0           # recorded facts stay numbers
    o = r["objects"]["pot_first"]
    assert o["max_rise_lift_transport_m"] == 0.0004 and o["lift"] is False                       # a real small number, kept as is
    assert is_unknown(o["slip"]) and "slip" in o["slip"]["unknown"]
    assert o["hold_rel_z_spread_m"] is not None and not isinstance(o["hold_rel_z_spread_m"], bool)
    assert all(p.startswith(("first_failure", "failures", "objects")) for p in r["unknown_fields"]) and r["unknown_fields"]
    assert any("stage_failure" in g for g in r["record_gaps"])


def test_place_failure_with_a_recorded_goal_gives_a_known_pose_difference():
    meta = {"place_failure": {"pot_first.pick_1.place": {"control_step": 700, "tol_pos_m": 0.008, "eef_pos": [0.1, 0.2, 0.9],
                                                         "eef_goal": [0.1, 0.2, 0.95], "eef_error_m": 0.05,
                                                         "object_pos": [0, 0, 0], "target_pos0": [0, 0, 0]}}}
    r = build_failure_report(P8, status="place_not_reached:pot_first.pick_1.place", task_ok=False, meta=meta)
    ff = r["first_failure"]
    assert ff["class"] == "execution_failed" and ff["requirement_id"] == "exec.place.pose_reached" and ff["control_step"] == 700
    assert ff["pose_difference"]["eef_error_m"] == 0.05 and ff["pose_difference"]["eef_goal"] == [0.1, 0.2, 0.95]
    assert "meta.place_failure" in ff["evidence_source"]


def test_stage_timeout_with_stage_failure_meta_has_step_and_reached_pose_but_still_no_goal():
    meta = {"stage_failure": {"pot_first.pick_1.descend": {"stage": "pot_first.pick_1.descend", "primitive": "descend",
                                                          "control_step": 310, "max_steps": 250, "eef_pos": [0.1, 0.2, 1.0]}}}
    r = build_failure_report(P8, status="stage_timeout:pot_first.pick_1.descend", task_ok=False, meta=meta)
    ff = r["first_failure"]
    assert ff["control_step"] == 310 and ff["detail"]["eef_pos_at_failure"] == [0.1, 0.2, 1.0]
    assert is_unknown(ff["pose_difference"]) and "goal pose" in ff["pose_difference"]["unknown"]


def test_grasp_and_guard_statuses_are_classified():
    r = build_failure_report(P8, status="grasp_not_bilateral:pot_first.pick_1.grasp", meta={"grasp_failure": {
        "pot_first.pick_1.grasp": {"control_step": 120, "reached_band": True, "bilateral": {"ok": False}}}})
    assert r["first_failure"]["requirement_id"] == "exec.grasp.bilateral_contact" and r["first_failure"]["control_step"] == 120
    r = build_failure_report(P8, status="guard_retry_budget_exhausted:pot_first.pick_1.grasp")
    assert r["first_failure"]["class"] == "execution_failed" and r["first_failure"]["requirement_id"] == "exec.guard.retry_budget_exhausted"
    r = build_failure_report(P8, status="reference_unresolved:pot_first.pick_1.approach:reference_not_found:x")
    assert r["first_failure"]["class"] == "planning_rejected" and r["first_failure"]["requirement_id"] == "exec.reference.resolved"


def test_a_clean_completed_run_reports_no_failure():
    r = build_failure_report(P8, status="completed", task_ok=True)
    assert r["failures"] == [] and r["first_failure"] is None and r["classes_present"] == []


def test_completed_but_task_flag_false_is_an_outcome_not_an_execution_failure():
    r = build_failure_report(P8, status="completed", task_ok=False, outcome={"pot_first": {"grasp": True, "lift": True, "transport": True, "place": False,
                                                                                          "final_xy_err_m": 0.08}})
    assert r["classes_present"] == ["outcome_not_met"]
    ids = [f["requirement_id"] for f in r["failures"]]
    assert ids == ["outcome.place", "task.official_success"] and r["first_failure"]["pose_difference"] == {"final_xy_err_m": 0.08}
    assert is_unknown(r["objects"]["pot_second"]["grasp"])                      # no record for the second pot -> unknown, not False


def test_resume_points_list_existing_snapshots_only_and_never_claim_validity():
    snaps = {"pot_first.pick_1.approach": SimpleNamespace(prefix_steps=0, status="completed"),
             "pot_first.pick_1.descend": SimpleNamespace(prefix_steps=42, status="completed")}
    r = build_failure_report(P8, status="stage_timeout:pot_first.pick_1.lift", snapshots=snaps)
    rp = r["resume_points"]
    assert [x["stage_label"] for x in rp] == ["pot_first.pick_1.approach", "pot_first.pick_1.descend"]
    assert all(x["validity"] == "unchecked" and "item-7" in x["note"] for x in rp)
    assert [x["at_or_before_first_failure"] for x in rp] == [True, True]
    assert rp[1]["prefix_steps"] == 42
    assert build_failure_report(P8, status="completed")["resume_points"] == []


def test_episode_adapter_uses_the_per_step_stage_column_to_place_a_contact():
    labels = ["pot_first.pick_1.approach"] * 5 + ["pot_first.pick_1.descend"] * 10
    ep = SimpleNamespace(status="completed", task_ok=True, meta={}, arrays={"stage": labels})
    audit = {"policy_version": "v1.1", "unexcused": [{"robot_body": "hand", "other": "pot", "max_n": 3.0, "control_step_range": [7, 9]}]}
    r = from_episode(P8, ep, audit=audit)
    f = r["first_failure"]
    assert f["class"] == "audit_failed" and f["instance_id"] == "pot_first.pick_1.descend" and f["control_step"] == 7
    assert f["detail"]["span_control_steps"] == 3 and is_unknown(f["detail"]["duration_s"])


def test_report_is_json_serialisable_and_requirements_are_labels_only():
    r = build_failure_report(P8, status="approach_infeasible:pot_first.pick_1.approach", meta={"approach_plan": {"pot_first.pick_1": {
        "feasible": False, "candidates": [{"min_clearance_m": -0.001, "blocking": "x", "phase": "descent", "feasible": False}],
        "grid_source": "grasp_reference"}}})
    json.dumps(r)
    assert "not thresholds" in r["requirements_note"] and all(isinstance(v, str) for v in r["requirements"].values())
