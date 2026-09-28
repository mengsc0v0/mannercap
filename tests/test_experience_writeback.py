"""Unit tests for Phase 5: self-evolving write-back
(run_manner_ablation.py's _write_back_experience / _write_back_repair).

Verifies:
  * None-safe no-ops when the experience library is disabled.
  * A both_pass trial's final params are flattened and appended both to the
    in-memory ExperienceLibrary and to the on-disk JSONL file.
  * A successful repair retry records a genuine before/after RepairRecord
    only for (phase, dimension) pairs whose value actually changed.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_manner_ablation as rma  # noqa: E402
from synthesis_v2.experience_library import ExperienceLibrary  # noqa: E402


def _params():
    return {
        "phase_v_max": {"approach": 0.12, "transport": 0.06, "final": 0.06},
        "phase_a_max": {"approach": 0.08, "transport": 0.04, "final": 0.04},
        "grip": 0.88, "kp_pos": 1000.0, "jerk_limit": 1.0,
    }


class TestWriteBackExperienceNoLib:
    def test_none_lib_is_noop(self):
        # must not raise even with a nonexistent path
        rma._write_back_experience(None, "/tmp/does_not_matter.jsonl",
                                    "pour", "gentle", 1, "stamp", _params(), "both_pass")


class TestWriteBackExperience:
    def test_appends_to_memory_and_disk(self):
        lib = ExperienceLibrary()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            rma._write_back_experience(lib, path, "pour", "gentle", 7, "stamp1",
                                        _params(), "both_pass")
            # 3 phase_v_max + 3 phase_a_max + grip + kp_pos + jerk_limit = 9
            assert len(lib.experiences) == 9
            assert any(r.dimension == "v_max" and r.phase == "transport"
                       and r.value == pytest.approx(0.06) and r.level == "gentle"
                       and r.task == "pour" and r.seed == 7
                       for r in lib.experiences)
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 9

            # a second write-back on the same lib/path appends further (does
            # not overwrite/dedupe -- each trial is an independent sample)
            rma._write_back_experience(lib, path, "pour", "gentle", 8, "stamp2",
                                        _params(), "both_pass")
            assert len(lib.experiences) == 18
            with open(path) as f:
                assert len(f.readlines()) == 18

    def test_calibrate_sees_freshly_written_samples_immediately(self):
        lib = ExperienceLibrary()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            for seed in (1, 2, 3):
                p = _params()
                p["phase_v_max"]["transport"] = 0.05 + seed * 0.001
                rma._write_back_experience(lib, path, "pour", "gentle", seed,
                                            "stamp", p, "both_pass")
            value, used = lib.calibrate("pour", "transport", "v_max", "gentle",
                                         fallback=0.99, min_samples=3)
            assert used is True
            assert value != pytest.approx(0.99)


class TestWriteBackRepairNoLib:
    def test_none_lib_is_noop(self):
        rma._write_back_repair(None, "/tmp/does_not_matter.jsonl", "pour",
                                "stamp", "spill", _params(), _params(), "both_pass")


class TestWriteBackRepair:
    def test_records_only_changed_dims(self):
        lib = ExperienceLibrary()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            before = _params()
            after = _params()
            after["phase_v_max"]["transport"] = 0.03  # changed: 0.06 -> 0.03
            after["phase_a_max"]["transport"] = 0.02  # changed: 0.04 -> 0.02
            # "final" phase left identical to `before` on purpose
            rma._write_back_repair(lib, path, "pour", "stamp", "spill",
                                    before, after, "both_pass")
            assert len(lib.repairs) == 2
            keys = {(r.phase, r.dimension) for r in lib.repairs}
            assert keys == {("transport", "v_max"), ("transport", "a_max")}
            rec = next(r for r in lib.repairs if r.dimension == "v_max")
            assert rec.before_value == pytest.approx(0.06)
            assert rec.after_value == pytest.approx(0.03)
            assert rec.cause == "spill"
            assert rec.outcome_after == "both_pass"
            with open(path) as f:
                assert len(f.readlines()) == 2

    def test_calibrate_repair_ratio_sees_freshly_written_samples(self):
        lib = ExperienceLibrary()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lib.jsonl")
            for i in range(3):
                before = _params()
                after = _params()
                after["phase_v_max"]["transport"] = 0.06 * 0.5
                rma._write_back_repair(lib, path, "pour", "stamp", "spill",
                                        before, after, "both_pass")
            ratio, used = lib.calibrate_repair_ratio(
                "pour", "transport", "spill", "v_max", fallback_ratio=0.75, min_samples=3)
            assert used is True
            assert ratio == pytest.approx(0.5)
