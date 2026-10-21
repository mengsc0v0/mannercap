"""Held-out check of the continuous interface: realised value vs a REQUESTED realised target.

usage: p11_continuity_verify.py REPORT.json OUT.jsonl SEEDS(comma) N_PER_QUANTITY [SHARD/NSHARDS]
For each quantity whose sweep produced a monotone range, draw N random targets (fixed rng) strictly inside the
realised range of that range, convert them with the sweep's inverse map into commands (targets are NOT grid points),
run on seeds disjoint from the sweep, and record target / command / realised.  Speed targets are drawn
independently per stage inside one episode, so this also checks per-stage independence.
"""
import importlib.util, json, math, os, sys

import numpy as np

from synthesis_v2.response_curve import command_for

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p11_continuity_sweep", os.path.join(HERE, "p11_continuity_sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

SPEED_STAGES = {"lift": ("approach", "descend", "lift"), "stack": ("approach", "descend", "lift", "transport", "place", "retreat")}


def draw(rng, summary):
    r = summary["usable_range"]
    return float(rng.uniform(r["realised_lo"], r["realised_hi"]))


def main():
    rep = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")]
    n = int(sys.argv[4])
    shard, nshard = (int(x) for x in sys.argv[5].split("/")) if len(sys.argv) > 5 else (0, 1)
    rng = np.random.default_rng(20261016)
    jobs = []
    for i in range(n):
        seed = seeds[i % len(seeds)]
        for mode, key in (("wipe_speed", "wipe/sweep speed"), ("wipe_press", "wipe/press force"),
                          ("lift_grip", "lift/grip hold force")):
            if key in rep and rep[key]["usable_range"].get("found"):
                t = draw(rng, rep[key])
                jobs.append({"mode": mode, "seed": seed, "targets": {key: t},
                             "cmd": command_for(rep[key]["inverse"], t)["command"]})
        for task in ("lift", "stack"):
            keys = {s: f"{task}/{s} speed" for s in SPEED_STAGES[task]}
            ok = {s: k for s, k in keys.items() if k in rep and rep[k]["usable_range"].get("found")}
            if len(ok) == len(keys):
                tg = {k: draw(rng, rep[k]) for k in keys.values()}
                cmds = {s: command_for(rep[k]["inverse"], tg[k])["command"] for s, k in keys.items()}
                cmds["grasp"] = 0.12  # grasp is a closing stage, not part of the speed claim
                jobs.append({"mode": f"{task}_speed", "seed": seed, "targets": tg, "cmd": cmds})
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["job"]) for l in open(out)}
    with open(out, "a") as f:
        for j, job in list(enumerate(jobs))[shard::nshard]:
            if j in done:
                continue
            try:
                rec = sweep.run_one(job["mode"], job["cmd"], job["seed"])
            except Exception as exc:  # noqa: BLE001
                rec = {"status": "exception", "task_ok": False, "error": f"{type(exc).__name__}: {exc}"}
            rec.update(job=j, mode=job["mode"], seed=job["seed"], targets=job["targets"], command=job["cmd"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(job["mode"], job["seed"], rec.get("status"), rec.get("task_ok"), flush=True)


if __name__ == "__main__":
    main()
