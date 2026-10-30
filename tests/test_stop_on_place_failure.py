"""Step 15 (1): a place that does not reach its pose must not release when the plan asks for it (default unchanged)."""
from __future__ import annotations

import dataclasses

import pytest

from synthesis_v2.native_osc_env import NativeSession
from synthesis_v2.native_producer import GripPlan, ProductionPlan, StagePlan, default_stage_plans
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode, program_plan_from_production


def _plan(stop: bool) -> ProgramPlan:
    # the place stage gets 3 control steps: it can never reach its pose
    stages = tuple(dataclasses.replace(sp, max_steps=3) if sp.stage_id == "place" else sp
                   for sp in default_stage_plans("stack"))
    pp = ProductionPlan("stack", stages, GripPlan(10.0, 16.0))
    return dataclasses.replace(program_plan_from_production(pp), stop_on_place_failure=stop)


@pytest.fixture(scope="module")
def session():
    s = NativeSession("stack", 0, render=False)
    s.reset()
    yield s
    s.close()


def _stages(ep):
    return [k for k, v in ep.stage_steps.items() if v]


def test_flag_is_hashed_only_when_set():
    assert "stop_on_place_failure" not in _plan(False).to_dict()
    assert _plan(True).to_dict()["stop_on_place_failure"] is True


def test_timed_out_place_does_not_release_and_is_recorded(session):
    ep = produce_program_episode(_plan(True).program, 0, _plan(True), keep_images=False, session=session)
    # (the grasp band of this cheap fixture is not reached either: the first failure keeps the status, the place failure
    # is recorded in the meta and the branch ends before the release)
    assert ep.status.startswith(("place_not_reached", "grasp_band_not_reached")), ep.status
    assert "place" in _stages(ep) and "retreat" not in _stages(ep)
    (rec,) = ep.meta["place_failure"].values()
    assert rec["eef_error_m"] > rec["tol_pos_m"]
    assert "release_start_step_by_instance" not in ep.meta


def test_default_still_releases_after_a_timeout(session):
    ep = produce_program_episode(_plan(False).program, 0, _plan(False), keep_images=False, session=session)
    assert "place_failure" not in ep.meta and "release_start_step_by_instance" in ep.meta
