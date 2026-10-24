"""Step 0: freeze a reproducible baseline of the existing native producers.

Records code/dependency/config identity plus per-episode action / physics digests
for fixed seeds, so that later refactors (stage instances, TaskProgram,
shared primitives) can be checked for bit-identical behaviour on old tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthesis_v2.candidate_plan import plan_from_candidate  # noqa: E402
from synthesis_v2.manner_request import MannerIntent, default_candidate  # noqa: E402
from synthesis_v2.native_producer import produce_episode  # noqa: E402
from synthesis_v2.stage_table import stage_table_hash  # noqa: E402
from synthesis_v2.wipe_producer import produce_wipe_episode  # noqa: E402

DIGEST_KEYS = ("actions", "state", "next_state", "stage", "left_mean_n", "right_mean_n",
               "left_max_n", "right_max_n", "other_max_n", "valid_fraction", "success")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_digest(arrays: dict) -> dict:
    out = {}
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        out[k] = sha_bytes(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    out["_combined"] = sha_bytes(json.dumps({k: out[k] for k in DIGEST_KEYS if k in out}, sort_keys=True).encode())
    return out


def run(task: str, seed: int, cand: dict, *, keep_images: bool, resume_stage: str | None = None) -> dict:
    plan = plan_from_candidate(task, cand)
    prod = produce_wipe_episode if task == "wipe" else produce_episode
    t0 = time.time()
    snaps: dict = {}
    ep = prod(task, seed, plan, keep_images=keep_images, snapshots=snaps)
    rec = {
        "task": task, "seed": seed, "candidate": cand, "plan_hash": ep.plan_hash, "status": ep.status,
        "task_ok": ep.task_ok, "n_steps": ep.n_steps, "stage_steps": ep.stage_steps,
        "digest": array_digest(ep.arrays), "wall_s": round(time.time() - t0, 2),
        "peak_total_grip_n": float(np.nanmax(ep.arrays["left_max_n"] + ep.arrays["right_max_n"]))
        if ep.n_steps and "left_max_n" in ep.arrays else None,
    }
    if resume_stage and resume_stage in snaps:
        t1 = time.time()
        ep2 = prod(task, seed, plan, keep_images=keep_images, resume=snaps[resume_stage])
        d2 = array_digest(ep2.arrays)
        rec["resume_check"] = {
            "stage": resume_stage, "status": ep2.status, "n_steps": ep2.n_steps,
            "combined_equal": d2["_combined"] == rec["digest"]["_combined"],
            "actions_equal": d2["actions"] == rec["digest"]["actions"],
            "wall_s": round(time.time() - t1, 2),
        }
    return rec


def env_identity() -> dict:
    def sh(cmd):
        return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout
    cfg = {}
    for p in sorted((ROOT / "configs").rglob("*")):
        if p.is_file():
            cfg[str(p.relative_to(ROOT))] = sha_bytes(p.read_bytes())
    src = {str(p.relative_to(ROOT)): sha_bytes(p.read_bytes()) for p in sorted((ROOT / "synthesis_v2").glob("*.py"))}
    import mujoco, robosuite
    return {
        "git_head": sh("git rev-parse HEAD").strip(), "git_branch": sh("git branch --show-current").strip(),
        "git_status_porcelain": sh("git status --porcelain"),
        "python": sys.version, "robosuite": robosuite.__version__, "mujoco": mujoco.__version__,
        "stage_table_hash": stage_table_hash(), "config_sha256": cfg, "synthesis_v2_sha256": src,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/baseline_20261023"))
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    (out / "env_identity.json").write_text(json.dumps(env_identity(), indent=1, ensure_ascii=False))
    variants = {
        "neutral": lambda t: default_candidate(t, ()),
        "fast_firm": lambda t: default_candidate(t, (
            MannerIntent("transport" if t != "lift" else "lift", "speed", "fast"), MannerIntent("grasp", "grip", "firm"))),
        "slow_gentle": lambda t: default_candidate(t, (
            MannerIntent("approach", "speed", "slow"), MannerIntent("grasp", "grip", "gentle"))),
    }
    wipe_variants = {
        "neutral": lambda t: default_candidate(t, ()),
        "fast_hard": lambda t: default_candidate(t, (MannerIntent("sweep", "speed", "fast"), MannerIntent("sweep", "press", "firm"))),
    }
    records = []
    for task in ("lift", "stack", "nut_assembly", "wipe"):
        vs = wipe_variants if task == "wipe" else variants
        for vname, fn in vs.items():
            for seed in seeds:
                resume = {"lift": "lift", "stack": "transport", "nut_assembly": "transport", "wipe": "sweep"}[task]
                try:
                    cand = fn(task)
                    r = run(task, seed, cand, keep_images=False, resume_stage=resume if vname == "neutral" else None)
                except Exception as exc:  # record, do not hide
                    r = {"task": task, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                r["variant"] = vname
                records.append(r)
                print(json.dumps({k: r.get(k) for k in ("task", "variant", "seed", "status", "task_ok", "n_steps", "wall_s", "error")}
                                 | ({"resume": r["resume_check"]} if "resume_check" in r else {})), flush=True)
    # one rendered episode to pin image digest
    r = run("lift", seeds[0], default_candidate("lift", ()), keep_images=True)
    r["variant"] = "neutral_with_images"
    records.append(r)
    print(json.dumps({k: r.get(k) for k in ("task", "variant", "status", "n_steps", "wall_s")}), flush=True)
    (out / "episode_digests.json").write_text(json.dumps(records, indent=1, ensure_ascii=False, default=str))
    print("WROTE", out)


if __name__ == "__main__":
    main()
