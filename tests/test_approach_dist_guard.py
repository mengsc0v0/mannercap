import numpy as np
from synthesis_v2 import approach_planner as ap


class _Mj:
    """mj_geomDistance stub: spurious 0.0 for distmax >= 0.01 (what the real library did for a 6 cm gap), exact below."""
    def __init__(self, true, bad_from=0.01):
        self.true, self.bad_from, self.calls = true, bad_from, []

    def mj_geomDistance(self, m, d, g, og, dm, ft):
        self.calls.append(dm)
        if dm >= self.bad_from:
            return 0.0
        return min(self.true, dm)


def test_spurious_zero_is_resolved_to_a_lower_bound():
    mj = _Mj(true=0.06)
    r = ap._dist(mj, None, None, 0, 1, np.zeros(6))
    assert r == 0.005 and mj.calls == [0.05, 0.02, 0.01, 0.005]   # certified >= 5 mm, never reported as touching


def test_true_touch_stays_zero():
    mj = _Mj(true=0.0, bad_from=0.0)                                # every rung says 0.0
    assert ap._dist(mj, None, None, 0, 1, np.zeros(6)) == 0.0 and len(mj.calls) == len(ap.DIST_LADDER)


def test_penetration_and_exact_values_pass_through():
    class M2:
        def mj_geomDistance(self, *a): return -0.004
    assert ap._dist(M2(), None, None, 0, 1, np.zeros(6)) == -0.004
    assert ap._dist(_Mj(true=0.03, bad_from=1.0), None, None, 0, 1, np.zeros(6)) == 0.03
