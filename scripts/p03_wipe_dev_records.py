"""Run the 6 wipe manner combinations (speed x press) on development seeds and store per-episode stats.

usage: p03_wipe_dev_records.py OUT.jsonl SEEDS(comma)
"""
import json, sys, time

from synthesis_v2.candidate_plan import plan_from_candidate
from synthesis_v2.manner_calibration import DEFAULT_POLICY, wipe_episode_stats
from synthesis_v2.wipe_language import COMBOS, combo_intents
from synthesis_v2.manner_request import default_candidate
from synthesis_v2.wipe_producer import produce_wipe_episode

out, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
t0 = time.time()
with open(out, "a") as f:
    for seed in seeds:
        for speed, press in COMBOS:
            plan = plan_from_candidate("wipe", default_candidate("wipe", combo_intents(speed, press)))
            ep = produce_wipe_episode("wipe", seed, plan, keep_images=False)
            st = wipe_episode_stats(ep.arrays, DEFAULT_POLICY) if ep.n_steps else None
            if st is not None:
                st["task_ok"] = bool(ep.task_ok)
            rec = {"task": "wipe", "seed": seed, "speed_level": speed, "press_level": press, "status": ep.status,
                   "n_steps": ep.n_steps, "final_wiped": ep.meta.get("final_wiped"), "stats": st or
                   {"speed": {}, "press_median": None, "peak_total_n": None, "idle_force_max_n": None, "task_ok": False}}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(seed, speed, press, ep.status, ep.task_ok, ep.n_steps, round(time.time() - t0), flush=True)
