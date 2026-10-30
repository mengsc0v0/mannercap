"""Step 14 (A): a grasp that never reaches its force band must end the branch when the plan asks for it, and must not
change anything when it does not (default), so every earlier result stays reproducible."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from synthesis_v2.native_osc_env import NativeSession
from synthesis_v2.native_producer import GripPlan, ProductionPlan, default_stage_plans
from synthesis_v2.program_executor import (ProgramPlan, _kind, _rotate_grasp_about_z, produce_program_episode,
                                           program_plan_from_production)


def _plan(stop: bool) -> ProgramPlan:
    # a band no two-finger pinch can reach: the grasp stage always ends "not reached"
    pp = ProductionPlan("stack", default_stage_plans("stack"), GripPlan(500.0, 600.0, max_close_steps=12))
    return dataclasses.replace(program_plan_from_production(pp), stop_on_grasp_failure=stop)


@pytest.fixture(scope="module")
def session():
    s = NativeSession("stack", 0, render=False)
    s.reset()
    yield s
    s.close()


def _stages(ep):
    return [k for k, v in ep.stage_steps.items() if v]


def test_default_plan_hash_unchanged_and_flag_is_hashed_only_when_set():
    assert "stop_on_grasp_failure" not in _plan(False).to_dict()
    assert _plan(True).to_dict()["stop_on_grasp_failure"] is True
    assert _plan(True).plan_hash() != _plan(False).plan_hash()


def test_failed_grasp_ends_the_branch_and_keeps_the_reason(session):
    ep = produce_program_episode(_plan(True).program, 0, _plan(True), keep_images=False, session=session)
    assert ep.status.startswith("grasp_band_not_reached")
    assert not ep.task_ok
    assert "grasp" in _stages(ep)
    assert not {"lift", "transport", "place", "retreat"} & set(_stages(ep))
    fail = ep.meta["grasp_failure"]
    (rec,) = fail.values()
    assert rec["reached_band"] is False and rec["control_step"] > 0


def test_without_the_flag_the_dependent_stages_still_run(session):
    ep = produce_program_episode(_plan(False).program, 0, _plan(False), keep_images=False, session=session)
    assert ep.status.startswith("grasp_band_not_reached")
    assert "grasp_failure" not in ep.meta
    assert "lift" in _stages(ep)


def test_site_kind_and_grasp_rotation():
    assert _kind("@site:desk_caddy_1_back_contain_region") == "site"
    off, quat = np.array([0.0, 0.02, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])
    o2, q2 = _rotate_grasp_about_z(off, quat, np.pi / 2)
    assert np.allclose(o2, [-0.02, 0.0, 0.05], atol=1e-9)  # hand offset turns with the object, height unchanged
    assert np.allclose(np.linalg.norm(q2), 1.0)
    o3, q3 = _rotate_grasp_about_z(off, quat, 0.0)
    assert np.allclose(o3, off) and np.allclose(np.abs(q3), np.abs(quat))
