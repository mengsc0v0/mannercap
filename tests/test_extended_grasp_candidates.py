import sys
import types

import numpy as np
import pytest

from synthesis_v2 import approach_planner as ap


def test_default_grid_unchanged():
    g = ap.candidate_grid()
    assert len(g) == 35 and g[0] == (0, 0.0, 0.0) and {t[0] for t in g} == {0} and all(len(t) == 3 for t in g)


def test_extended_grid_shape_and_order():
    g = ap.extended_grid()
    assert len(g) == 16 * 7 * 5 * 5 and g[0] == (0, 0.0, 0.0, 0.0) and all(len(t) == 4 for t in g)
    dep = [abs(t[1]) + abs(t[3]) + 0.5 * abs(t[2]) for t in g]
    assert dep == sorted(dep)                      # preference = smallest departure from the centred grasp first
    assert len(set(g)) == len(g)


@pytest.fixture
def fake_scene(monkeypatch):
    mj = types.SimpleNamespace(MjData=lambda m: types.SimpleNamespace(qpos=np.zeros(3)), mj_forward=lambda m, d: None)
    monkeypatch.setitem(sys.modules, "mujoco", mj)
    env = types.SimpleNamespace(obs_to_site=np.eye(3))
    d = types.SimpleNamespace(qpos=np.zeros(3))
    return env, d


def _plan(fake_scene, monkeypatch, clearance, grid, **kw):
    env, d = fake_scene
    monkeypatch.setattr(ap, "_clearance", clearance)
    return ap.plan_approach(None, d, 0, env, [], start_pos=[0, 0, 1], start_quat_xyzw=[0, 0, 0, 1], grasp_pos=[0, 0, 0],
                            grasp_quat_xyzw=[0, 0, 0, 1], hover_dz=0.1, grid=grid, target=ap.Obstacle("t", ()), **kw)


def test_longitudinal_and_lateral_offsets_follow_hand_axes(fake_scene, monkeypatch):
    pl = _plan(fake_scene, monkeypatch, lambda *a, **k: (0.01, "x"), [(0, 0.02, 0.01, -0.01)])
    assert pl.chosen.offset == pytest.approx([-0.01, 0.02, 0.01]) and pl.chosen.longitudinal_m == -0.01
    pl = _plan(fake_scene, monkeypatch, lambda *a, **k: (0.01, "x"), [(0, 0.02, 0.01)])   # 3-tuple: no longitudinal shift
    assert pl.chosen.offset == pytest.approx([0.0, 0.02, 0.01]) and pl.chosen.longitudinal_m == 0.0


def test_feasibility_criterion_is_unchanged(fake_scene, monkeypatch):
    # exactly touching (0.0) on the path, or no open-hand clearance at the grasp pose, is never feasible
    assert not _plan(fake_scene, monkeypatch, lambda *a, **k: (0.0, "x"), [(0, 0, 0, 0)]).feasible
    pl = _plan(fake_scene, monkeypatch, lambda m, mu, sc, env, pos, R, obs, reach: (0.0 if [o.name for o in obs] == ["t"] and len(obs) == 1 else 0.01, "o"),
               [(0, 0, 0, 0)])
    assert not pl.feasible and pl.candidates[0].at_grasp_clearance_m == 0.0


def test_evaluate_all_keeps_going_and_first_feasible_stays_chosen(fake_scene, monkeypatch):
    grid = [(0, 0, 0, 0), (90, 0, 0.01, 0), (45, 0.01, 0, 0)]
    first = _plan(fake_scene, monkeypatch, lambda *a, **k: (0.01, "x"), grid)
    assert len(first.candidates) == 1                       # default behaviour: stop at the first feasible one
    allp = _plan(fake_scene, monkeypatch, lambda *a, **k: (0.01, "x"), grid, evaluate_all=True)
    assert len(allp.candidates) == 3 and allp.chosen is allp.candidates[0] and all(c.feasible for c in allp.candidates)
