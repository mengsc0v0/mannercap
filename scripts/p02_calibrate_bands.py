"""Calibrate speed / grip bands on development seeds and write the artifact.

usage: p02_calibrate_bands.py OUT.json DEV_SEEDS(comma) [TASKS(comma)]
Plans per seed: speed slow/normal/fast with normal grip, plus grip gentle/firm
with normal speed (5 episodes per seed and task).
"""
import json, sys, time
from synthesis_v2.candidate_plan import plan_from_candidate
from synthesis_v2.manner_calibration import DEFAULT_POLICY, derive_calibration, episode_stats, save_calibration
from synthesis_v2.manner_request import MannerIntent, default_candidate, semantics_hash
from synthesis_v2.native_osc_env import CONTROLLER_CONFIG_SHA256
from synthesis_v2.native_producer import PRODUCER_VERSION, produce_episode
from synthesis_v2.stage_table import dimension_stages, stage_table_hash

out = sys.argv[1]
seeds = [int(x) for x in sys.argv[2].split(",")]
tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lift", "stack"]
records = []
t0 = time.time()
for task in tasks:
    for seed in seeds:
        for speed, grip in [("slow", "normal"), ("normal", "normal"), ("fast", "normal"), ("normal", "gentle"), ("normal", "firm")]:
            intents = [MannerIntent(s, "speed", speed) for s in dimension_stages(task, "speed")] if speed != "normal" else []
            if grip != "normal":
                intents.append(MannerIntent("grasp", "grip", grip))
            plan = plan_from_candidate(task, default_candidate(task, intents))
            ep = produce_episode(task, seed, plan, keep_images=False)
            st = episode_stats(ep.arrays, task, DEFAULT_POLICY) if ep.n_steps else None
            if st is not None:
                st["task_ok"] = bool(ep.task_ok)
            records.append({"task": task, "seed": seed, "speed_level": speed, "grip_level": grip,
                            "status": ep.status, "stats": st or {"speed": {}, "hold": [], "peak_total_n": None,
                                                                 "idle_force_max_n": None, "task_ok": False}})
            print(task, seed, speed, grip, ep.status, ep.task_ok, round(time.time() - t0), flush=True)
cal = derive_calibration(records, dev_seeds=seeds, meta={
    "controller_sha256": CONTROLLER_CONFIG_SHA256, "producer_version": PRODUCER_VERSION,
    "stage_table_hash": stage_table_hash(), "semantics_hash": semantics_hash(),
    "n_dev_episodes": len(records), "n_dev_failed": sum(1 for r in records if not r["stats"]["task_ok"])})
save_calibration(cal, out)
json.dump(records, open(out + ".records.json", "w"))
print("written", out, cal["content_sha256"][:12])
