
from synthesis_v2.manner_calibration import derive_wipe_calibration, merge_wipe_calibration
from synthesis_v2.manner_request import parse_manner_request
from synthesis_v2.wipe_language import BASES, COMBOS, TIERS, base_for_seed, build_prompt, check_bank, combo_intents


def test_bank_is_self_consistent():
    check_bank()


def test_prompts_are_outcome_first_and_cover_all_tiers():
    seen = set()
    for seed in range(7001, 7061):
        for speed, press in COMBOS:
            text, tier = build_prompt(seed, speed, press)
            assert text.startswith(base_for_seed(seed))
            if combo_intents(speed, press):
                assert tier in TIERS
                seen.add(tier)
            else:
                assert tier is None and text == base_for_seed(seed)
    assert seen == set(TIERS)
    assert build_prompt(7003, "fast", "firm") == build_prompt(7003, "fast", "firm")


def test_wipe_parse():
    p = parse_manner_request("wipe", "Wipe the table quickly and press firmly")
    assert p.status == "ok" and {(i.dimension, i.level) for i in p.intents} == {("speed", "fast"), ("press", "firm")}
    assert parse_manner_request("wipe", "Wipe the table gently").status == "unsupported"
    assert all(parse_manner_request("wipe", b).status == "neutral" for b in BASES)


def _rec(speed, press, v, f, seed):
    return {"speed_level": speed, "press_level": press, "seed": seed,
            "stats": {"speed": {"sweep": [v] * 40}, "press_median": f, "peak_total_n": 80.0, "task_ok": True}}


def _records(fast_v):
    out = []
    for s, v in (("slow", 0.05), ("normal", 0.10), ("fast", fast_v)):
        for p, f in (("normal", 30.0), ("firm", 45.0)):
            for seed in (1, 2):
                out.append(_rec(s, p, v + 0.002 * seed, f + seed, seed))
    return out


def test_derive_and_merge_separable():
    w = derive_wipe_calibration(_records(0.2), dev_seeds=[1, 2], meta={})
    sp = w["speed"]["wipe"]["sweep"]
    assert sp["status"] == "ok" and 0.05 < sp["slow_upper"] < 0.10 < sp["fast_lower"] < 0.2
    pr = w["press"]["wipe"]
    assert pr["status"] == "ok" and 32 < pr["firm_lower"] < 45
    base = {"speed": {"lift": {}}, "safety": {}, "dev_seeds": [9], "meta": {"stage_table_hash": "old"}}
    m = merge_wipe_calibration(base, w, {"stage_table_hash": "new"})
    assert m["speed"]["lift"] == {} and m["speed"]["wipe"] == w["speed"]["wipe"] and m["dev_seeds"] == [1, 2, 9]


def test_derive_leaves_threshold_null_when_levels_overlap():
    w = derive_wipe_calibration(_records(0.1), dev_seeds=[1, 2], meta={})
    sp = w["speed"]["wipe"]["sweep"]
    assert sp["fast_lower"] is None and sp["status"] == "partially_inseparable"
