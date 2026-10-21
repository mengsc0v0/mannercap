"""Replay-verify recorded actions against a fresh re-execution from the same
seed, and produce dataset/replay_audit.json per agent_handoff.md sec 7/9.3.

For each sampled episode: rebuild the env from seed (fresh reset, same
placement_initializer.rng logic), replay the EXACT recorded actions[] via
env.step(), and compare the resulting state trajectory against the recorded
state/next_state arrays. This is a genuine control replay (re-executes real
actions in a fresh simulation), not a saved-state playback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_stack_common import make_stack_env, eef_state8  # noqa: E402

POS_ERR_THRESHOLD_M = 0.005  # 5mm reference threshold from handoff sec 9.3
ROT_ERR_THRESHOLD_RAD = 0.05
GRIPPER_ERR_THRESHOLD = 0.01


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_episode(dataset_root: str, episode_id: str):
    ep_dir = os.path.join(dataset_root, episode_id)
    meta = json.load(open(os.path.join(ep_dir, "metadata.json")))
    tr = np.load(os.path.join(ep_dir, "transitions.npz"), allow_pickle=False)
    rec_state = tr["state"]
    rec_next_state = tr["next_state"]
    rec_actions = tr["actions"]
    seed = int(meta["seed"])

    env = make_stack_env()
    env.rng = np.random.default_rng(seed)
    env.placement_initializer.rng = env.rng
    obs = env.reset()

    T = rec_actions.shape[0]
    replay_state = np.zeros((T, 8), dtype=np.float32)
    replay_next_state = np.zeros((T, 8), dtype=np.float32)
    for t in range(T):
        replay_state[t] = eef_state8(obs)
        action = rec_actions[t]
        obs, rew, done, info = env.step(action)
        replay_next_state[t] = eef_state8(obs)

    task_ok_replay = bool(env._check_success())

    pos_err = np.linalg.norm(replay_state[:, 0:3] - rec_state[:, 0:3], axis=1)
    rot_err = np.linalg.norm(replay_state[:, 3:6] - rec_state[:, 3:6], axis=1)
    grip_err = np.linalg.norm(replay_state[:, 6:8] - rec_state[:, 6:8], axis=1)

    result = dict(
        episode=episode_id,
        seed=seed,
        T=int(T),
        task_ok_recorded=bool(meta["task_ok"]),
        task_ok_replay=task_ok_replay,
        pos_err_mean=float(pos_err.mean()), pos_err_max=float(pos_err.max()),
        rot_err_mean=float(rot_err.mean()), rot_err_max=float(rot_err.max()),
        gripper_err_mean=float(grip_err.mean()), gripper_err_max=float(grip_err.max()),
        pos_err_threshold_m=POS_ERR_THRESHOLD_M,
        rot_err_threshold_rad=ROT_ERR_THRESHOLD_RAD,
        gripper_err_threshold=GRIPPER_ERR_THRESHOLD,
        passed=bool(
            pos_err.max() < POS_ERR_THRESHOLD_M
            and rot_err.max() < ROT_ERR_THRESHOLD_RAD
            and grip_err.max() < GRIPPER_ERR_THRESHOLD
            and task_ok_replay == meta["task_ok"]
        ),
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episodes", nargs="*", default=None,
                     help="episode ids to replay; default = all in episodes.jsonl")
    ap.add_argument("--sample-n", type=int, default=None,
                     help="if set, randomly sample this many episodes instead of all")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes_jsonl = os.path.join(args.dataset_root, "episodes.jsonl")
    all_records = [json.loads(l) for l in open(episodes_jsonl) if l.strip()]
    all_ids = [r["episode"] for r in all_records]

    if args.episodes:
        target_ids = args.episodes
    elif args.sample_n:
        rng = np.random.default_rng(args.sample_seed)
        target_ids = list(rng.choice(all_ids, size=min(args.sample_n, len(all_ids)), replace=False))
    else:
        target_ids = all_ids

    source_manifest_path = os.path.join(args.dataset_root, "source_manifest.json")
    source_sha256 = sha256_file(source_manifest_path) if os.path.exists(source_manifest_path) else None

    results = []
    for i, ep_id in enumerate(sorted(target_ids)):
        r = replay_episode(args.dataset_root, ep_id)
        results.append(r)
        print(f"[{i+1}/{len(target_ids)}] {ep_id}: passed={r['passed']} "
              f"pos_err_max={r['pos_err_max']*1000:.2f}mm task_ok_replay={r['task_ok_replay']}",
              file=sys.stderr)

    n_pass = sum(1 for r in results if r["passed"])
    n_task_ok_replay = sum(1 for r in results if r["task_ok_replay"])
    audit = dict(
        source_sha256=source_sha256,
        n_episodes_audited=len(results),
        n_episodes_total=len(all_ids),
        n_passed=n_pass,
        n_task_ok_replay=n_task_ok_replay,
        pos_err_threshold_m=POS_ERR_THRESHOLD_M,
        rot_err_threshold_rad=ROT_ERR_THRESHOLD_RAD,
        gripper_err_threshold=GRIPPER_ERR_THRESHOLD,
        passed=bool(n_pass == len(results)),
        results=results,
    )
    with open(args.out, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "n_episodes_audited": len(results), "n_passed": n_pass,
        "n_task_ok_replay": n_task_ok_replay,
    }))


if __name__ == "__main__":
    main()
