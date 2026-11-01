"""Budget b2 of the three-arm evaluation: identical stop rule for recover and rerun (fake executor, no simulation)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "mt"))
S20 = pytest.importorskip("s20_selection_eval")


def _ex(cand, status, n_events, pre, passed=False):
    ev = [{"approach_instance": "obj.pick_1.approach", "refused_candidate": [0.0, 0.0, 0.01 * (len(pre.get("obj.pick_1.approach", [])) + 1), 0.0]}
          for _ in range(n_events)]
    return {"physical_run_id": "x", "trajectory_file": "t", "candidate": list(cand), "status": status, "task_ok": passed,
            "n_steps": 1, "wall_s": 1.0, "measured": {"hold_min": 10.0, "hold_max": 12.0},
            "validity": {"measurement_valid": passed, "grasp_stable": passed},
            "contact_v1_clean": passed, "guard_events": ev, "substeps": {"executed_total": 100}, "reused_cached_episode": False}


def _run(monkeypatch, recovery, refusals_per_candidate):
    calls = []

    def fake(task, init, cand, rec, pre):
        calls.append((tuple(cand), rec, sum(len(v) for v in pre.values())))
        k = sum(len(v) for v in pre.values())
        need = refusals_per_candidate
        if rec == "rerun":
            return _ex(cand, f"guard_refused_abort:o", 1, pre) if k < need else _ex(cand, "completed", 0, pre)
        if need > 3:
            return _ex(cand, "guard_retry_budget_exhausted:o", 4, pre)
        return _ex(cand, "completed", need, pre)

    monkeypatch.setattr(S20, "execute", fake)
    req = {"id": "R1", "band_n": [8.0, 16.0]}
    return S20.run_request(0, 30, req, "retrieval", recovery), calls


@pytest.mark.parametrize("recovery", ["recover", "rerun"])
def test_fourth_refusal_ends_the_force_candidate_in_both_arms(monkeypatch, recovery):
    r, calls = _run(monkeypatch, recovery, refusals_per_candidate=9)
    assert len(r["attempts"]) == 3 and all(a["end_reason"] == "guard_retry_budget_exhausted" for a in r["attempts"])
    assert r["budget_audit"]["within_bounds"] and r["budget_audit"]["force_candidates"] == 3
    if recovery == "rerun":
        assert r["n_physical_executions"] == 3 * 4                 # first try + 3 retries per candidate, count restarts per candidate
        assert [c[2] for c in calls[:4]] == [0, 1, 2, 3] and calls[4][2] == 0
    else:
        assert r["n_physical_executions"] == 3


def test_three_refusals_are_still_allowed(monkeypatch):
    r, _ = _run(monkeypatch, "rerun", refusals_per_candidate=3)
    assert r["attempts"][0]["end_reason"] == "completed" and r["attempts"][0]["refusals_in_attempt"] == 3
    assert len(r["attempts"][0]["executions"]) == 4
