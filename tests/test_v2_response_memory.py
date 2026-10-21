import json

import pytest

from synthesis_v2.manner_request import LEVEL_RANGES, parse_manner_request
from synthesis_v2.production_pipeline import RequestSpec, run_requests
from synthesis_v2.response_memory import (CommandMedianProposer, MeasuredResponseProposer, Observation, ResponseMemory,
                                          grip_command, speed_command)
from test_v2_manner_pipeline import fake_produce_factory, make_cal

SHA = "cal0"


def obs(stage="approach", dim="speed", level="slow", cmd=0.06, mn=0.01, med=0.012, mx=0.02, lo=None, up=0.0234,
        dec="fail", rid="r0", seed=1, sha=SHA, task="lift"):
    return Observation(task, stage, dim, level, cmd, mn, med, mx, lo, up, dec, rid, f"{rid}:a1", seed, sha, "episode_audit.v1")


def test_slow_repair_uses_measured_ratio_and_flags_infeasible():
    # measured max 0.0256 at command 0.05 -> the bound 0.0234 needs a command below the registered range
    cmd, info = speed_command("slow", [], [obs(cmd=0.05, mx=0.0256)])
    assert cmd == LEVEL_RANGES["speed"]["slow"][0] and info["source"] == "request" and info["feasible"] is False


def test_slow_repair_is_exact_when_feasible():
    cmd, info = speed_command("slow", [], [obs(cmd=0.09, mx=0.0225)])
    k = 0.0225 / 0.09
    assert info["feasible"] and cmd == pytest.approx(0.0234 * 0.85 / k)
    assert cmd * k < 0.0234


def test_fast_uses_median_and_stays_in_range():
    o = obs(level="fast", cmd=0.3, med=0.05, mx=0.06, lo=0.067, up=None)
    cmd, info = speed_command("fast", [], [o])
    assert cmd == pytest.approx(0.067 * 1.2 / (0.05 / 0.3)) and cmd <= LEVEL_RANGES["speed"]["fast"][1]


def test_cold_start_falls_back_to_default():
    assert speed_command("slow", [obs(), obs()], [])[0] == LEVEL_RANGES["speed"]["slow"][2]
    assert grip_command("firm", [], [], [])[0] == LEVEL_RANGES["grip"]["firm"][2]
    assert speed_command("normal", [], [obs(level="normal")])[0] == LEVEL_RANGES["speed"]["normal"][2]


def test_library_ratio_is_conservative_quantile():
    lib = [obs(cmd=0.06, mx=m, rid=f"r{i}") for i, m in enumerate((0.012, 0.015, 0.03))]
    cmd, info = speed_command("slow", lib, [])
    assert info["source"] == "library" and info["k"] == pytest.approx(0.375)  # p75 of 0.2, 0.25, 0.5


def test_grip_repair_shifts_towards_band_centre():
    o = obs(stage="grasp", dim="grip", level="firm", cmd=28.0, mn=20.0, med=20.5, mx=21.0, lo=22.8, up=41.3)
    cmd, info = grip_command("firm", [], [], [o])
    assert cmd == min(34.0, 28.0 + (32.05 - 20.5)) and info["source"] == "request"


def test_validity_and_persistence(tmp_path):
    bad = obs(med=float("nan"))
    unk = obs(dec="unknown")
    assert not bad.valid and not unk.valid and obs().valid
    mem = ResponseMemory(SHA)
    for o in (obs(), bad, unk, obs(sha="other")):
        mem.add(o)
    assert len(mem.query("lift", "speed", "slow", "approach")) == 1      # invalid and other-calibration excluded
    mem.save(tmp_path / "m.jsonl")
    again = ResponseMemory.load(tmp_path / "m.jsonl")
    assert len(again.obs) == 4 and again.calibration_sha256 == SHA
    (tmp_path / "legacy.jsonl").write_text(json.dumps({"schema": "mannercap-experience-library-v1"}) + "\n")
    with pytest.raises(ValueError):
        ResponseMemory.load(tmp_path / "legacy.jsonl")
    mem.freeze()
    with pytest.raises(RuntimeError):
        mem.add(obs())


def test_proposer_first_attempt_from_library_and_in_request_exclusion():
    mem = ResponseMemory(SHA)
    for i in range(3):
        for st in ("approach", "descend", "lift"):
            mem.add(obs(stage=st, cmd=0.09, mx=0.0225, rid=f"w{i}"))
    spec = RequestSpec("lift", 5, "Lift the cube slowly.", base_instruction="Lift the cube.")
    parse = parse_manner_request("lift", spec.language)
    cand, _ = MeasuredResponseProposer(mem).propose(spec, parse, [])
    assert all(e["v_max_mps"] == pytest.approx(0.0234 * 0.85 / 0.25) for e in cand["stages"] if e["level"] == "slow")
    cmed, _ = CommandMedianProposer(mem).propose(spec, parse, [])
    assert cmed["stages"][0]["v_max_mps"] == LEVEL_RANGES["speed"]["slow"][2]   # no passing command recorded -> default


def test_pipeline_online_observation_ids_match_ledger(tmp_path):
    mem = ResponseMemory("calx")
    prop = MeasuredResponseProposer(mem, online=True)
    cal = make_cal(slow_upper=0.16)
    cal["content_sha256"] = "calx"
    run_requests([RequestSpec("lift", 5, "Lift the cube slowly.", base_instruction="Lift the cube.")], cal, tmp_path,
                 max_attempts=3, proposer=prop, produce=fake_produce_factory([]), log=lambda s: None)
    atts = [json.loads(x) for x in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert {o.attempt_id for o in mem.obs} == {a["attempt_id"] for a in atts}
    assert any(o.decision == "fail" for o in mem.obs) and any(o.decision == "pass" for o in mem.obs)
