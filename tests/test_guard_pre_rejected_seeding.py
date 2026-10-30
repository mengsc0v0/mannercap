"""A whole-episode re-run carries the guard's refusals forward through the caller, which serialises them.
If the seeded form does not compare equal to what the planner produces, the re-run silently re-picks the
refused candidate and the "re-run" arm degenerates into repeating the same failure until its budget is gone.
"""
from synthesis_v2.program_executor import _candidate_key, _seed_rejected


class _Cand:
    def __init__(self, yaw, lat, dz, lon):
        self.yaw_deg, self.lateral_m, self.dz_m, self.longitudinal_m = yaw, lat, dz, lon


def test_json_round_tripped_refusals_still_match_the_planner_key():
    import json

    c = _Cand(0.0, 0.0, 0.0, 0.0)
    key = _candidate_key(c)
    seeded = _seed_rejected(json.loads(json.dumps({"pick.approach": [list(key)]})))
    assert key in seeded["pick.approach"]


def test_a_different_candidate_is_not_treated_as_refused():
    seeded = _seed_rejected({"pick.approach": [[0.0, 0.0, 0.0, 0.0]]})
    assert _candidate_key(_Cand(0.0, 0.0, 0.01, 0.0)) not in seeded["pick.approach"]
