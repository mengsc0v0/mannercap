"""Parser / semantic gate / audit / pipeline / export tests (no simulator)."""
import json
from pathlib import Path

import numpy as np
import pytest

from synthesis_v2.episode_audit import audit_episode
from synthesis_v2.manner_calibration import CALIBRATION_SCHEMA, DEFAULT_POLICY, assert_seed_isolated, content_hash, load_calibration, save_calibration
from synthesis_v2.manner_request import (
    LEVEL_RANGES, default_candidate, parse_manner_request, semantic_gate,
)
from synthesis_v2.native_producer import RawEpisode

REPO = Path(__file__).resolve().parents[1]


def make_cal(slow_upper=0.2, fast_lower=1.0, gentle=(1.0, 6.0), firm=(20.0, 45.0), cap=60.0):
    cal = {"schema": CALIBRATION_SCHEMA, "status": "calibrated_dev", "dev_seeds": [9001],
           "policy": dict(DEFAULT_POLICY), "meta": {},
           "speed": {"lift": {s: {"slow_upper": slow_upper, "fast_lower": fast_lower, "status": "ok"}
                              for s in ("approach", "descend", "lift")}},
           "grip": {"lift": {"gentle_lower": gentle[0], "gentle_upper": gentle[1], "firm_lower": firm[0],
                             "firm_upper": firm[1], "status": "ok"}},
           "safety": {"lift": {"peak_cap_n": cap, "contact_threshold_n": 0.1}}}
    cal["content_sha256"] = content_hash(cal)
    return cal


def _eef_states(stages, sp):
    """Chained state/next_state whose per-step eef displacement is sp[stage] * 0.05 (m/s at 20 Hz)."""
    x = np.concatenate([[0.0], np.cumsum([sp[s] * 0.05 for s in stages])]).astype(np.float32)
    st = np.zeros((len(stages), 8), np.float32)
    ns = np.zeros((len(stages), 8), np.float32)
    st[:, 0], ns[:, 0] = x[:-1], x[1:]
    return st, ns


def make_arrays(speed=0.5, hold=10.0, peak=None, valid=1.0, success=True, n=(30, 20, 10, 30)):
    stages = sum(([s] * k for s, k in zip(("approach", "descend", "grasp", "lift"), n)), [])
    T = len(stages)
    st = np.array(stages)
    force = np.where((st == "lift") | (st == "grasp"), hold / 2.0, 0.0)
    pk = force * 1.0 if peak is None else np.where(st == "grasp", peak / 2.0, force)
    sp = speed if isinstance(speed, dict) else {k: speed for k in ("approach", "descend", "lift", "grasp")}
    return {
        "state": _eef_states(stages, sp)[0], "actions": np.zeros((T, 7), np.float32),
        "next_state": _eef_states(stages, sp)[1], "timestamp": (np.arange(T) * 0.05 + 0.05).astype(np.float64),
        "dt": np.full(T, 0.05), "frame_index": np.arange(T), "sim_step_index": np.arange(T) * 25 + 25,
        "terminated": np.zeros(T, bool), "truncated": np.zeros(T, bool),
        "success": np.array([False] * (T - 1) + [success]), "stage": st,
        "left_mean_n": force, "right_mean_n": force, "left_max_n": pk, "right_max_n": pk,
        "valid_fraction": np.full(T, valid), "joint_speed_mean": np.array([sp[s] for s in stages]),
    }


# ------------------------------- parser ------------------------------------
@pytest.mark.parametrize("task,text,status,intents", [
    ("lift", "Lift the cube.", "neutral", []),
    ("lift", "Lift the cube gently.", "ok", [("grasp", "grip", "gentle")]),
    ("lift", "Lift the cube slowly and grip firmly", "ok", [("grasp", "grip", "firm"), ("approach", "speed", "slow"), ("descend", "speed", "slow"),
                                         ("lift", "speed", "slow")]),
    ("lift", "Grasp gently, then lift quickly.", "ok", [("grasp", "grip", "gentle"), ("lift", "speed", "fast")]),
    ("stack", "Stack the cubes, approach quickly and place slowly.", "ok",
     [("approach", "speed", "fast"), ("place", "speed", "slow")]),
    ("lift", "轻轻抓取", "ok", [("grasp", "grip", "gentle")]),
    ("lift", "Lift the cube elegantly", "unsupported", []),
    ("lift", "Lift slowly and quickly", "unsupported", None),
    ("lift", "", "unsupported", []),
])
def test_parser(task, text, status, intents):
    r = parse_manner_request(task, text)
    assert r.status == status
    if intents is not None:
        assert [(i.stage, i.dimension, i.level) for i in r.intents] == intents or \
            sorted((i.stage, i.dimension, i.level) for i in r.intents) == sorted(intents)


def test_gentle_is_grip_only_and_slow_not_grip():
    r = parse_manner_request("lift", "Lift the cube slowly.")
    assert {i.dimension for i in r.intents} == {"speed"}
    r = parse_manner_request("lift", "Lift the cube gently.")
    assert {i.dimension for i in r.intents} == {"grip"}


# ------------------------------- gate --------------------------------------
def test_gate_accepts_rule_candidate_and_rejects_deviations():
    req = parse_manner_request("stack", "Stack the cubes, approach quickly and place slowly.")
    good = default_candidate("stack", req.intents)
    assert semantic_gate("stack", req, good).accepted
    wrong = json.loads(json.dumps(good))
    for e in wrong["stages"]:
        if e["stage_id"] == "approach":
            e["level"], e["v_max_mps"] = "slow", 0.05
    reasons = semantic_gate("stack", req, wrong).reasons
    assert any(r.startswith("wrong_level:approach") for r in reasons)
    extra = json.loads(json.dumps(good))
    extra["grip"] = {"level": "gentle", "target_lo_n": 3.0}
    assert any(r.startswith("extra:grasp") for r in semantic_gate("stack", req, extra).reasons)
    moved = json.loads(json.dumps(good))
    for e in moved["stages"]:
        if e["stage_id"] == "place":
            e["level"], e["v_max_mps"] = "normal", 0.12
        if e["stage_id"] == "retreat":
            e["level"], e["v_max_mps"] = "slow", 0.05
    rs = semantic_gate("stack", req, moved).reasons
    assert any("missing:place" in r for r in rs) and any(r.startswith("extra:retreat") for r in rs)
    out_of_range = json.loads(json.dumps(good))
    out_of_range["stages"][0]["v_max_mps"] = 0.12  # approach declared fast but numerically normal
    assert any(r.startswith("command_outside_level_range") for r in semantic_gate("stack", req, out_of_range).reasons)


def test_gate_refuses_unsupported_request():
    req = parse_manner_request("lift", "Lift the cube elegantly")
    assert not semantic_gate("lift", req, default_candidate("lift", [])).accepted


def test_gate_uses_nominal_grip_band_and_wide_upper_bound():
    req = parse_manner_request("lift", "Lift the cube firmly.")
    candidate = {
        "stages": [
            {"stage_id": "approach", "level": "normal", "v_max_mps": 0.12},
            {"stage_id": "descend", "level": "normal", "v_max_mps": 0.12},
            {"stage_id": "lift", "level": "normal", "v_max_mps": 0.12},
        ],
        "grip": {"level": "firm", "target_lo_n": 80.0},
    }
    assert semantic_gate("lift", req, candidate, grip_scale=2.0).accepted


# ------------------------------- audit -------------------------------------
def audit(arrays, text, cal=None):
    return audit_episode(arrays, task="lift", seed=1, raw_language=text, cal=cal or make_cal())


def test_audit_neutral_is_not_applicable_and_passes():
    a = audit(make_arrays(), "Lift the cube.")
    assert a["manner"] == "not_applicable" and a["overall"] == "pass"


def test_audit_speed_levels():
    assert audit(make_arrays(speed=0.1), "Lift the cube slowly.")["overall"] == "pass"
    a = audit(make_arrays(speed=0.5), "Lift the cube slowly.")
    assert a["manner"] == "fail" and a["overall"] == "fail"
    assert audit(make_arrays(speed=1.5), "Lift the cube quickly.")["overall"] == "pass"
    assert audit(make_arrays(speed=0.5), "Lift the cube quickly.")["manner"] == "fail"


def test_audit_speed_uncalibrated_is_unknown():
    cal = make_cal(slow_upper=None)
    assert audit(make_arrays(speed=0.1), "Lift the cube slowly.", cal)["manner"] == "unknown"


def test_audit_grip_bands():
    assert audit(make_arrays(hold=4.0), "Lift the cube gently.")["overall"] == "pass"
    a = audit(make_arrays(hold=12.0), "Lift the cube gently.")
    assert a["manner"] == "fail" and any("above_band" in r for r in a["reasons"])
    assert audit(make_arrays(hold=30.0), "Lift the cube firmly.")["overall"] == "pass"


def test_audit_zero_force_gentle_is_not_a_pass():
    a = audit(make_arrays(hold=0.0), "Lift the cube gently.")
    assert a["manner"] == "fail" and a["overall"] == "fail"


def test_audit_invalid_sensor_is_unknown():
    a = audit(make_arrays(hold=4.0, valid=0.0), "Lift the cube gently.")
    assert a["manner"] == "unknown" and a["overall"] == "unknown"


def test_audit_safety_and_task():
    a = audit(make_arrays(hold=4.0, peak=200.0), "Lift the cube gently.")
    assert a["safety"] == "fail" and a["overall"] == "fail"
    a = audit(make_arrays(success=False), "Lift the cube.")
    assert a["task"] == "fail" and a["overall"] == "fail"


def test_audit_structure_errors_are_unknown():
    arr = make_arrays()
    arr["state"][5, 0] = np.nan
    assert audit(arr, "Lift the cube slowly.")["overall"] == "unknown"
    arr = make_arrays()
    arr["timestamp"] = arr["timestamp"] * 2
    assert audit(arr, "Lift the cube.")["overall"] == "unknown"


def test_audit_unsupported_request_never_passes():
    assert audit(make_arrays(), "Lift the cube elegantly")["overall"] != "pass"


def test_calibration_hash_and_seed_isolation(tmp_path):
    cal = make_cal()
    p = tmp_path / "c.json"
    save_calibration(cal, p)
    assert load_calibration(p)["content_sha256"] == cal["content_sha256"]
    d = json.loads(p.read_text())
    d["safety"]["lift"]["peak_cap_n"] = 999.0
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_calibration(p)
    with pytest.raises(ValueError):
        assert_seed_isolated(cal, 9001)
    assert_seed_isolated(cal, 1)


# ------------------------------ pipeline -----------------------------------
def fake_produce_factory(calls):
    def produce(task, seed, plan, keep_images=True, **kw):
        calls.append(plan.to_dict())
        v = {s.stage_id: s.v_max_mps for s in plan.stages}
        sp = {k: 3.0 * v[k] for k in ("approach", "descend", "lift")}  # rad/s proportional to command
        sp["grasp"] = 0.1
        arr = make_arrays(speed=sp, hold=plan.grip.target_lo_n + 1.0)
        if keep_images:
            T = len(arr["actions"])
            arr["images"] = np.zeros((T, 2, 224, 224, 3), np.uint8) + (np.arange(T) * 7 % 256).astype(np.uint8)[:, None, None, None, None]
        return RawEpisode(task, seed, plan.to_dict(), plan.plan_hash(), "completed", True, arr, {}, {})
    return produce


def test_pipeline_repair_and_ledger(tmp_path):
    from synthesis_v2.production_pipeline import RequestSpec, run_requests, summarize
    cal = make_cal(slow_upper=0.16)           # slow default 0.06*3=0.18 > 0.16 -> first attempt fails; repaired 0.05*3=0.15 passes
    calls = []
    specs = [RequestSpec("lift", 5, "Lift the cube slowly.", base_instruction="Lift the cube.")]
    res = run_requests(specs, cal, tmp_path, max_attempts=3, produce=fake_produce_factory(calls), log=lambda s: None)
    assert res[0].state == "accepted" and res[0].n_attempts == 2
    assert calls[1]["stages"][0]["v_max_mps"] < calls[0]["stages"][0]["v_max_mps"]  # repaired speed dimension only
    assert calls[1]["grip"] == calls[0]["grip"]
    s = summarize(tmp_path)
    assert s["n_requests"] == 1 and s["attempt_outcomes"] == {"validation_failure": 1, "accepted": 1}
    reqs = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert reqs[0]["raw_language"] == "Lift the cube slowly."
    failed_dir = next(p for p in (tmp_path / "raw").iterdir() if not (p / "images.npy").exists())
    assert (failed_dir / "trajectory.npz").exists() and (failed_dir / "audit.json").exists()


def test_pipeline_unsupported_request_consumes_attempts_without_execution(tmp_path):
    from synthesis_v2.production_pipeline import RequestSpec, run_requests
    calls = []
    res = run_requests([RequestSpec("lift", 6, "Lift the cube elegantly.")], make_cal(), tmp_path, max_attempts=2,
                       produce=fake_produce_factory(calls), log=lambda s: None)
    assert res[0].state == "budget_exhausted" and calls == []


def _speed_candidate(stage, v_max):
    stages = [{"stage_id": s, "level": "fast" if s == stage else "normal",
               "v_max_mps": v_max if s == stage else LEVEL_RANGES["speed"]["normal"][2]}
              for s in ("approach", "descend", "lift", "transport", "place", "retreat")]
    return {"stages": stages, "grip": {"level": "normal", "target_lo_n": LEVEL_RANGES["grip"]["normal"][2]}}


def _speed_check(stage, v_max, median):
    return {"stage": stage, "dimension": "speed", "level": "fast", "decision": "fail",
            "reason": "speed_below_lower", "median": median}


def test_stalled_checks_detects_non_monotone_response():
    from synthesis_v2.production_pipeline import _stalled_checks
    # command increased 0.4 -> 0.5 (the direction a "below_lower" repair should push), but the
    # realised median went the WRONG way (0.049 -> 0.030): the response is non-monotone here.
    history = [
        {"candidate": _speed_candidate("approach", 0.4), "audit": {"checks": [_speed_check("approach", 0.4, 0.049)]}},
        {"candidate": _speed_candidate("approach", 0.5), "audit": {"checks": [_speed_check("approach", 0.5, 0.030)]}},
    ]
    assert _stalled_checks(history) == {("approach", "speed", "fast")}


def test_stalled_checks_ignores_improving_response():
    from synthesis_v2.production_pipeline import _stalled_checks
    # command increased and the realised median also increased (toward the bound): not stalled.
    history = [
        {"candidate": _speed_candidate("approach", 0.4), "audit": {"checks": [_speed_check("approach", 0.4, 0.030)]}},
        {"candidate": _speed_candidate("approach", 0.5), "audit": {"checks": [_speed_check("approach", 0.5, 0.049)]}},
    ]
    assert _stalled_checks(history) == set()


def test_rule_proposer_freezes_non_monotone_dimension(tmp_path):
    from synthesis_v2.production_pipeline import RequestSpec, RuleProposer
    history = [
        {"candidate": _speed_candidate("approach", 0.4), "audit": {"checks": [_speed_check("approach", 0.4, 0.049)]}},
        {"candidate": _speed_candidate("approach", 0.5), "audit": {"checks": [_speed_check("approach", 0.5, 0.030)]}},
    ]
    spec = RequestSpec("stack", 1, "Approach fast.")
    parse = parse_manner_request("stack", "Approach the cubes quickly.")
    cand, _ = RuleProposer().propose(spec, parse, history)
    # frozen: the proposer must not push the non-monotone dimension any further.
    assert cand == history[-1]["candidate"]


def test_pipeline_bad_proposers(tmp_path):
    from synthesis_v2.production_pipeline import RequestSpec, run_requests

    class Bad:
        name = "bad"
        n = 0
        def propose(self, spec, parse, history):
            Bad.n += 1
            return ("not json", {"api_calls": 1}) if Bad.n == 1 else ({"stages": [], "grip": {}}, {"api_calls": 1})
    calls = []
    res = run_requests([RequestSpec("lift", 7, "Lift the cube slowly.")], make_cal(), tmp_path, max_attempts=2,
                       proposer=Bad(), produce=fake_produce_factory(calls), log=lambda s: None)
    atts = [json.loads(x) for x in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [a["outcome"] for a in atts] == ["parse_failure", "invalid"] and calls == []
    assert res[0].state == "budget_exhausted"


def test_pipeline_duplicate_candidate_links_to_previous_attempt(tmp_path):
    from synthesis_v2.attempt_ledger import attempt_id_for
    from synthesis_v2.manner_request import default_candidate, parse_manner_request
    from synthesis_v2.production_pipeline import RequestSpec, run_requests

    spec = RequestSpec("lift", 8, "Lift the cube slowly.", base_instruction="Lift the cube.")
    cand = default_candidate("lift", parse_manner_request("lift", spec.language).intents)

    class Same:
        name = "same"
        def propose(self, spec, parse, history):
            return cand, {"api_calls": 0}
    calls = []
    cal = make_cal(slow_upper=0.01)           # attempt 1 fails the audit, attempt 2 repeats it
    res = run_requests([spec], cal, tmp_path, max_attempts=2, proposer=Same(),
                       produce=fake_produce_factory(calls), log=lambda s: None)
    atts = [json.loads(x) for x in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [a["outcome"] for a in atts] == ["validation_failure", "duplicate"] and len(calls) == 1
    assert atts[1]["duplicate_of_attempt_id"] == atts[0]["attempt_id"] == attempt_id_for(atts[0]["request_id"], 1)
    assert res[0].state == "budget_exhausted"


def test_pipeline_rejects_calibration_seed(tmp_path):
    from synthesis_v2.production_pipeline import RequestSpec, run_requests
    with pytest.raises(ValueError):
        run_requests([RequestSpec("lift", 9001, "Lift the cube.")], make_cal(), tmp_path, produce=fake_produce_factory([]))


# ------------------------------- export ------------------------------------
def test_export_package_roundtrip(tmp_path):
    from synthesis_v2.pi05_export import export_package, make_tarball
    from synthesis_v2.production_pipeline import RequestSpec, run_requests
    cal = make_cal()
    calls = []
    specs = [RequestSpec("lift", s, t, base_instruction="Lift the cube.", pair_key=f"lift:{s}")
             for s in range(10, 16) for t in ("Lift the cube.", "Lift the cube slowly.")]
    run_requests(specs, cal, tmp_path / "pipe", produce=fake_produce_factory(calls), log=lambda s: None)

    def fake_replay(task, seed, actions, next_state, success_final, want_assets=False):
        return {"max_pos_err_mm": 0.1, "max_rot_err_rad": 0.0, "max_grip_err_m": 0.0, "replayed_success": True,
                "stored_success": True, "success_match": True, "n_steps": len(actions), "pass": True,
                "_states": next_state.copy(), "model_xml": "<mujoco/>", "camera_parameters": {},
                "initial": {"qpos": np.zeros(3), "qvel": np.zeros(3), "time": 0.0}}
    man = export_package(tmp_path / "pipe", tmp_path / "pkg", cal=cal, replay=fake_replay, repo_root=REPO, log=lambda s: None)
    assert man["n_episodes"] == 12 and man["pairs"]["n_groups_with_neutral_and_manner"] == 6
    recs = [json.loads(x) for x in (tmp_path / "pkg/dataset/episodes.jsonl").read_text().splitlines()]
    by_seed = {}
    for r in recs:
        by_seed.setdefault(r["seed"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_seed.values())           # split by seed group
    assert {r["manner_tag"] for r in recs} == {"neutral", "approach.speed=slow,descend.speed=slow,lift.speed=slow"}
    import subprocess, sys
    r = subprocess.run([sys.executable, str(tmp_path / "pkg/tools/verify_package.py"), str(tmp_path / "pkg")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # tampering is detected
    (tmp_path / "pkg/dataset/episode_000000/metadata.json").write_text("{}")
    r = subprocess.run([sys.executable, str(tmp_path / "pkg/tools/verify_package.py"), str(tmp_path / "pkg")],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_first_diff_block_identifies_earliest_changed_stage():
    from dataclasses import replace
    from synthesis_v2.native_producer import GripPlan, ProductionPlan, default_stage_plans, first_diff_block
    base = ProductionPlan("stack", default_stage_plans("stack"), GripPlan(10, 16))

    def chg(plan, sid, v):
        return replace(plan, stages=tuple(replace(s, v_max_mps=v) if s.stage_id == sid else s for s in plan.stages))
    assert first_diff_block(base, base) is None
    assert first_diff_block(base, chg(base, "transport", 0.2)) == "transport"
    assert first_diff_block(base, chg(chg(base, "retreat", 0.2), "descend", 0.2)) == "descend"
    assert first_diff_block(base, replace(base, grip=GripPlan(20, 26))) == "grasp"
    assert first_diff_block(base, replace(base, open_steps=5)) == "place"


def _descend_profile(vals):
    """Arrays whose descend steps follow ``vals`` (m/s); other stages stay slow."""
    a = make_arrays(speed=0.05, n=(30, len(vals), 10, 30))
    sp = np.where(a["stage"] == "descend", 0.0, 0.05)
    sp[a["stage"] == "descend"] = vals
    x = np.concatenate([[0.0], np.cumsum(sp * 0.05)]).astype(np.float32)
    a["state"][:, 0], a["next_state"][:, 0] = x[:-1], x[1:]
    return a


def test_slow_judged_on_median_with_transient_cap():
    from synthesis_v2.episode_audit import _speed_check
    cal = make_cal(slow_upper=0.10)
    for st in cal["speed"]["lift"].values():
        st["slow_transient_cap"] = 0.30
    cal["content_sha256"] = content_hash(cal)

    def run(vals):
        r = _speed_check(_descend_profile(vals), "lift", "descend", "slow", cal, cal["policy"], "e")
        return r["decision"], r["reason"]

    n = 40
    base = np.full(n, 0.05)
    assert run(base) == ("pass", "")
    spike = base.copy(); spike[15:18] = 0.20          # above slow_upper, below the cap
    assert run(spike) == ("pass", "")
    spike[16] = 0.40                                   # a sample above the cap
    assert run(spike) == ("fail", "speed_transient_above_cap")
    assert run(np.full(n, 0.15)) == ("fail", "speed_above_upper")   # median too high


def test_slow_without_cap_keeps_every_sample_rule():
    from synthesis_v2.episode_audit import _speed_check
    cal = make_cal(slow_upper=0.10)                    # no slow_transient_cap field
    vals = np.full(40, 0.05); vals[16] = 0.20
    r = _speed_check(_descend_profile(vals), "lift", "descend", "slow", cal, cal["policy"], "e")
    assert r["decision"] == "fail" and r["reason"] == "speed_transient_above_cap"
