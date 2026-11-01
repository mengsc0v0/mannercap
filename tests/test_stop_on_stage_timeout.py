"""Step 23: a timed-out required motion stage ends the branch when the plan asks for it (default unchanged); a one-finger
reading is not a grasp when the plan requires a bilateral one."""
from __future__ import annotations

import dataclasses

import pytest

from synthesis_v2.native_osc_env import NativeSession
from synthesis_v2.native_producer import GripPlan, ProductionPlan, default_stage_plans
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode, program_plan_from_production


def _plan(**kw) -> ProgramPlan:
    # the descend stage gets 3 control steps: it can never reach its pose
    stages = tuple(dataclasses.replace(sp, max_steps=3) if sp.stage_id == "descend" else sp
                   for sp in default_stage_plans("stack"))
    pp = ProductionPlan("stack", stages, GripPlan(10.0, 16.0))
    return dataclasses.replace(program_plan_from_production(pp), **kw)


@pytest.fixture(scope="module")
def session():
    s = NativeSession("stack", 0, render=False)
    s.reset()
    yield s
    s.close()


def _stages(ep):
    return [k for k, v in ep.stage_steps.items() if v]


def test_flags_are_hashed_only_when_set():
    d = _plan().to_dict()
    assert "stop_on_stage_timeout" not in d and "require_bilateral_grasp" not in d
    assert _plan(stop_on_stage_timeout=True).to_dict()["stop_on_stage_timeout"] is True
    assert _plan(require_bilateral_grasp=True).to_dict()["require_bilateral_grasp"] is True
    assert _plan().plan_hash() != _plan(stop_on_stage_timeout=True).plan_hash()


def test_timed_out_descend_ends_the_branch_and_is_recorded(session):
    pl = _plan(stop_on_stage_timeout=True)
    ep = produce_program_episode(pl.program, 0, pl, keep_images=False, session=session)
    assert ep.status.startswith("stage_timeout:"), ep.status
    assert "descend" in _stages(ep) and "grasp" not in _stages(ep) and "lift" not in _stages(ep)
    (rec,) = ep.meta["stage_failure"].values()
    assert rec["primitive"] == "descend" and rec["max_steps"] == 3


def test_default_still_continues_after_a_timeout(session):
    pl = _plan()
    ep = produce_program_episode(pl.program, 0, pl, keep_images=False, session=session)
    assert "stage_failure" not in ep.meta and "grasp" in _stages(ep) and "lift" in _stages(ep)
    assert ep.status.startswith("stage_timeout:") or ep.status.startswith("grasp_band_not_reached")
