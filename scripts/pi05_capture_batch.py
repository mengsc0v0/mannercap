"""Batch-capture pi0.5 stack episodes in a single long-lived process (avoids
per-process robosuite/mujoco import overhead). Writes dataset/episode_XXXXXX/
directories plus appends json records to a jsonl log; failures are logged
separately and NOT included as successful episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_capture_episode import capture_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--target-success", type=int, required=True,
                     help="number of task_ok=True episodes to collect")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-max", type=int, default=10000)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--records-out", required=True)
    ap.add_argument("--failed-out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    n_train_target = int(round(args.target_success * args.train_frac))

    records = []
    failed = []
    seed = args.seed_start
    n_success = 0
    t_start = time.time()
    with open(args.records_out, "w") as rec_f, open(args.failed_out, "w") as fail_f:
        while n_success < args.target_success and seed <= args.seed_max:
            episode_id = f"episode_{n_success + 1:06d}"
            split = "train" if n_success < n_train_target else "valid"
            try:
                record, elapsed = capture_episode(episode_id, seed, split, args.out_root)
            except Exception as exc:  # noqa: BLE001
                failed.append({"seed": seed, "error": str(exc), "traceback": traceback.format_exc()})
                fail_f.write(json.dumps(failed[-1], ensure_ascii=False) + "\n")
                fail_f.flush()
                print(f"[seed={seed}] EXCEPTION: {exc}", file=sys.stderr)
                seed += 1
                continue

            if not record["task_ok"]:
                failed.append({"seed": seed, "episode_attempted": episode_id,
                                "reason": "task_ok=False (stack not achieved)"})
                fail_f.write(json.dumps(failed[-1], ensure_ascii=False) + "\n")
                fail_f.flush()
                # remove the failed episode directory so it doesn't pollute dataset/
                import shutil
                shutil.rmtree(os.path.join(args.out_root, episode_id), ignore_errors=True)
                print(f"[seed={seed}] task_ok=False, discarded, retrying next seed", file=sys.stderr)
                seed += 1
                continue

            records.append(record)
            rec_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rec_f.flush()
            n_success += 1
            elapsed_total = time.time() - t_start
            print(f"[{n_success}/{args.target_success}] seed={seed} episode={episode_id} "
                  f"split={split} frames={record['frames']} elapsed={elapsed:.1f}s "
                  f"total={elapsed_total:.0f}s", file=sys.stderr)
            seed += 1

    print(json.dumps({
        "n_success": n_success, "n_failed": len(failed),
        "seed_start": args.seed_start, "seed_end_exclusive": seed,
        "total_wall_sec": time.time() - t_start,
    }))


if __name__ == "__main__":
    main()
