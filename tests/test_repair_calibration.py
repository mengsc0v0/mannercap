"""Unit tests for Phase 4: MANNER_REPAIR_LIBRARY retrieval-augmented repair
ratios (run_manner_ablation.py's _adjust_* functions + _calibrate_repair_ratio).

Verifies:
  * Cold-start (no experience_lib / task, or too few repair samples) exactly
    reproduces the pre-Phase-4 hardcoded ratio behaviour -- zero regression.
  * When the experience library has >= min_samples matching RepairRecord
    samples for a (task, phase, cause, dimension) bucket, the calibrated
    median ratio is used instead of the hardcoded constant, and the
    "_repair_calibrated" tracking list records which phase/dimension keys
    were overridden.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_manner_ablation as rma  # noqa: E402
from synthesis_v2.experience_library import ExperienceLibrary  # noqa: E402


def _base_params() -> dict:
    return {
        "v_max": 0.12, "a_max": 0.08, "jerk_limit": 1.0,
        "kp_pos": 1000.0, "grip": 0.88,
        "phase_v_max": {
            "approach": 0.12, "descend": 0.06, "lift": 0.12,
            "transport": 0.06, "final": 0.06, "retreat": 0.12,
        },
        "phase_a_max": {
            "approach": 0.08, "descend": 0.04, "lift": 0.08,
            "transport": 0.04, "final": 0.04, "retreat": 0.08,
        },
        "grasp_tries": 3,
    }


class TestAdjustGripSlipUnaffected:
    def test_accepts_new_kwargs_and_behaves_unchanged(self):
        p = rma._adjust_grip_slip(_base_params(), experience_lib=None, task="lift", min_samples=3)
        assert p["grasp_tries"] == 5
        # not a calibratable dimension -- no _repair_calibrated key expected
        assert "_repair_calibrated" not in p or p["_repair_calibrated"] == []


class TestAdjustPourSpillColdStart:
    def test_no_lib_uses_hardcoded_075_ratio(self):
        p = rma._adjust_pour_spill(_base_params())
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.75)
        assert p["phase_v_max"]["final"] == pytest.approx(0.06 * 0.75)
        assert p["phase_a_max"]["transport"] == pytest.approx(0.04 * 0.75)
        assert p["_repair_calibrated"] == []

    def test_lib_present_but_insufficient_samples_falls_back(self):
        lib = ExperienceLibrary()
        lib.add_repair("pour", "transport", "spill", "v_max",
                        before_value=0.06, after_value=0.03, outcome_after="both_pass")
        p = rma._adjust_pour_spill(_base_params(), experience_lib=lib, task="pour", min_samples=3)
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.75)
        assert p["_repair_calibrated"] == []


class TestAdjustPourSpillCalibrated:
    def test_calibrated_ratio_overrides_hardcoded(self):
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("pour", "transport", "spill", "v_max",
                            before_value=0.06, after_value=0.03, outcome_after="both_pass")
        p = rma._adjust_pour_spill(_base_params(), experience_lib=lib, task="pour", min_samples=3)
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.5)
        assert "transport.v_max" in p["_repair_calibrated"]
        # a_max dimension had no samples -> still falls back to hardcoded 0.75
        assert p["phase_a_max"]["transport"] == pytest.approx(0.04 * 0.75)
        assert "transport.a_max" not in p["_repair_calibrated"]

    def test_failed_repairs_excluded_from_calibration(self):
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("pour", "transport", "spill", "v_max",
                            before_value=0.06, after_value=0.01, outcome_after="both_fail")
        p = rma._adjust_pour_spill(_base_params(), experience_lib=lib, task="pour", min_samples=3)
        # only both_pass repairs count -- with 0 valid samples, falls back
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.75)
        assert "transport.v_max" not in p["_repair_calibrated"]


class TestAdjustTimeoutPhased:
    def test_cold_start_speeds_up_slowest_phase_by_125(self):
        p = rma._adjust_timeout(_base_params())
        # "descend"/"final" tie at 0.06 as slowest -- min() picks first match
        slowest = min(_base_params()["phase_v_max"], key=lambda k: _base_params()["phase_v_max"][k])
        assert p["phase_v_max"][slowest] == pytest.approx(0.06 * 1.25)
        assert p["_repair_calibrated"] == []

    def test_calibrated_ratio_used_for_slowest_phase(self):
        base = _base_params()
        slowest = min(base["phase_v_max"], key=lambda k: base["phase_v_max"][k])
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("lift", slowest, "timeout", "v_max",
                            before_value=0.06, after_value=0.09, outcome_after="both_pass")
        p = rma._adjust_timeout(base, experience_lib=lib, task="lift", min_samples=3)
        assert p["phase_v_max"][slowest] == pytest.approx(0.06 * 1.5)
        assert f"{slowest}.v_max" in p["_repair_calibrated"]


class TestAdjustTimeoutFlat:
    def test_flat_params_no_phase_dict_uses_global_phase(self):
        flat = {"v_max": 0.12, "a_max": 0.08}
        p = rma._adjust_timeout(flat)
        assert p["v_max"] == pytest.approx(0.12 * 1.25)
        assert p["_repair_calibrated"] == []

    def test_flat_params_calibrated_via_global_phase(self):
        from synthesis_v2.experience_library import GLOBAL_PHASE
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("stack", GLOBAL_PHASE, "timeout", "v_max",
                            before_value=0.12, after_value=0.18, outcome_after="both_pass")
        flat = {"v_max": 0.12, "a_max": 0.08}
        p = rma._adjust_timeout(flat, experience_lib=lib, task="stack", min_samples=3)
        assert p["v_max"] == pytest.approx(0.12 * 1.5)
        assert "v_max" in p["_repair_calibrated"]


class TestAdjustStackMisalign:
    def test_cold_start_07_ratio(self):
        p = rma._adjust_stack_misalign(_base_params())
        assert p["phase_v_max"]["final"] == pytest.approx(0.06 * 0.7)
        assert p["_repair_calibrated"] == []

    def test_calibrated_ratio(self):
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("stack", "final", "stack_misalign", "v_max",
                            before_value=0.06, after_value=0.024, outcome_after="both_pass")
        p = rma._adjust_stack_misalign(_base_params(), experience_lib=lib, task="stack", min_samples=3)
        assert p["phase_v_max"]["final"] == pytest.approx(0.06 * 0.4)
        assert "final.v_max" in p["_repair_calibrated"]


class TestAdjustNutMisalign:
    def test_cold_start_07_ratio(self):
        p = rma._adjust_nut_misalign(_base_params())
        assert p["phase_v_max"]["final"] == pytest.approx(0.06 * 0.7)
        assert p["_repair_calibrated"] == []


class TestAdjustWipeIncomplete:
    def test_cold_start_transport_075_ratio_and_iters_increment(self):
        p = rma._adjust_wipe_incomplete(_base_params())
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.75)
        assert p["wipe_iters"] == 12
        assert p["_repair_calibrated"] == []

    def test_calibrated_ratio(self):
        lib = ExperienceLibrary()
        for _ in range(3):
            lib.add_repair("wipe", "transport", "wipe_incomplete", "v_max",
                            before_value=0.06, after_value=0.042, outcome_after="both_pass")
        p = rma._adjust_wipe_incomplete(_base_params(), experience_lib=lib, task="wipe", min_samples=3)
        assert p["phase_v_max"]["transport"] == pytest.approx(0.06 * 0.7)
        assert "transport.v_max" in p["_repair_calibrated"]


class TestCalibrateRepairRatioHelper:
    def test_none_lib_returns_fallback(self):
        ratio, used = rma._calibrate_repair_ratio(None, "pour", "transport", "spill", "v_max", 0.75, 3)
        assert ratio == 0.75 and used is False

    def test_none_task_returns_fallback(self):
        lib = ExperienceLibrary()
        ratio, used = rma._calibrate_repair_ratio(lib, None, "transport", "spill", "v_max", 0.75, 3)
        assert ratio == 0.75 and used is False
