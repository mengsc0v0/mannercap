"""Focused contract tests for the proposed v2 trajectory boundary.

These tests exercise only the pure validator.  They do not start a simulator,
render images, train a policy, or claim replay success.
"""

from __future__ import annotations

from copy import deepcopy

from synthesis_v2.trajectory_contract import validate_trajectory, validate_window


def _step(index: int, *, reset_id: str = "reset-1", lineage_id: str = "lineage-1") -> dict:
    state = [float(index), 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    next_state = [value + (0.01 if component < 6 else 0.0) for component, value in enumerate(state)]
    raw_action = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, -1.0]
    candidate = [0.01] * 6 + [-1.0]
    image = {"bytes": None, "path": f"trajectory_images/agentview/frame_{index:06d}.png"}
    wrist = {"bytes": None, "path": f"trajectory_images/wrist/frame_{index:06d}.png"}
    return {
        "observation.state": state,
        "next_observation.state": next_state,
        "observation.state.raw_joint_gripper": state,
        "next_observation.state.raw_joint_gripper": next_state,
        "observation.images.image": image,
        "observation.images.image2": wrist,
        "action": raw_action,
        "action.raw_joint_gripper": raw_action,
        "lerobot_action_candidate": candidate,
        "timestamp": index * 0.05,
        "dt": 0.05,
        "frame_index": index,
        "sim_step_index": index + 1,
        "episode_index": 0,
        "task_index": 7,
        "trajectory_id": "trajectory-1",
        "source_attempt_id": "attempt-1",
        "lineage_id": lineage_id,
        "reset_id": reset_id,
    }


def _valid_rows(count: int = 3) -> list[dict]:
    return [_step(index) for index in range(count)]


def test_valid_v2_window_passes() -> None:
    report = validate_window(_valid_rows())

    assert report.ok
    assert report.step_count == 3
    assert report.issues == ()


def test_missing_fields_are_reported_per_step() -> None:
    rows = _valid_rows(2)
    rows[1].pop("next_observation.state")
    rows[1].pop("lerobot_action_candidate")
    rows[1].pop("dt")
    rows[1].pop("lineage_id")

    report = validate_trajectory(rows)

    assert not report.valid
    missing = {(issue.index, issue.field) for issue in report.issues if issue.code == "missing_field"}
    assert (1, "next_observation.state") in missing
    assert (1, "lerobot_action_candidate") in missing
    assert (1, "dt") in missing
    assert (1, "lineage_id") in missing


def test_non_monotonic_timestamps_and_dt_mismatch_are_rejected() -> None:
    rows = _valid_rows(3)
    rows[2]["timestamp"] = 0.04

    report = validate_trajectory(rows)

    assert report.has_code("non_monotonic_time")
    assert report.has_code("dt_mismatch")


def test_reset_boundary_cannot_be_inside_one_window() -> None:
    rows = _valid_rows(3)
    rows[2]["reset_id"] = "reset-2"

    report = validate_window(rows)

    assert report.has_code("cross_reset_window")


def test_explicit_reset_marker_is_rejected() -> None:
    rows = _valid_rows(2)
    rows[1]["reset_event"] = True

    report = validate_window(rows)

    assert report.has_code("reset_marker")


def test_lineage_change_is_rejected_even_when_reset_is_constant() -> None:
    rows = _valid_rows(2)
    rows[1]["lineage_id"] = "lineage-2"

    report = validate_trajectory(rows)

    assert report.has_code("lineage_mismatch")


def test_existing_libero_like_row_is_unready_for_v2() -> None:
    row = _step(0)
    for field in ("dt", "trajectory_id", "source_attempt_id", "lineage_id", "reset_id"):
        row.pop(field)

    report = validate_trajectory([row])

    assert not report.valid
    assert report.has_code("missing_field")
    assert {issue.field for issue in report.issues if issue.code == "missing_field"} >= {
        "dt",
        "trajectory_id",
        "source_attempt_id",
        "lineage_id",
        "reset_id",
    }


def test_expected_lineage_is_checked_without_mutating_rows() -> None:
    rows = _valid_rows(2)
    before = deepcopy(rows)

    report = validate_trajectory(
        rows,
        expected_lineage={"trajectory_id": "different-trajectory", "reset_id": "reset-1"},
    )

    assert report.has_code("lineage_metadata_mismatch")
    assert rows == before
