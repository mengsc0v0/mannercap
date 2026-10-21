"""Format/schema/sync validation for the pi0.5 stack dataset, per
agent_handoff.md sec 9.1-9.2. Checks every episode's images.npy / transitions.npz
for correct shapes/dtypes, no object dtype, allow_pickle=False loadability,
control-frequency/timestamp sync, next_state[t]==state[t+1] consistency, image
variation over time (not a repeated first frame), and finite/bounded actions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def check_episode(ep_dir: str, episode_id: str):
    errors = []
    images_path = os.path.join(ep_dir, "images.npy")
    transitions_path = os.path.join(ep_dir, "transitions.npz")

    try:
        imgs = np.load(images_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        return [f"{episode_id}: images.npy failed to load with allow_pickle=False: {exc}"]

    if imgs.ndim != 5 or imgs.shape[1:] != (2, 224, 224, 3):
        errors.append(f"{episode_id}: images.npy shape {imgs.shape} != (T,2,224,224,3)")
    if imgs.dtype != np.uint8:
        errors.append(f"{episode_id}: images.npy dtype {imgs.dtype} != uint8")

    try:
        tr = np.load(transitions_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        return errors + [f"{episode_id}: transitions.npz failed to load with allow_pickle=False: {exc}"]

    required = {
        "state": (np.float32, 8), "actions": (np.float32, 7), "next_state": (np.float32, 8),
        "timestamp": (np.float64, None), "dt": (np.float64, None),
        "frame_index": (np.int64, None), "sim_step_index": (np.int64, None),
        "terminated": (np.bool_, None), "truncated": (np.bool_, None), "success": (np.bool_, None),
    }
    T = imgs.shape[0]
    for key, (dtype, ncols) in required.items():
        if key not in tr.files:
            errors.append(f"{episode_id}: transitions.npz missing key '{key}'")
            continue
        arr = tr[key]
        if arr.shape[0] != T:
            errors.append(f"{episode_id}: {key}.shape[0]={arr.shape[0]} != T(images)={T}")
        if ncols is not None and (arr.ndim != 2 or arr.shape[1] != ncols):
            errors.append(f"{episode_id}: {key}.shape={arr.shape} != (T,{ncols})")
        if arr.dtype != dtype:
            errors.append(f"{episode_id}: {key}.dtype={arr.dtype} != {dtype}")
        if arr.dtype == object:
            errors.append(f"{episode_id}: {key} has object dtype (forbidden)")

    if "state" in tr.files and "actions" in tr.files:
        if not np.isfinite(tr["state"]).all():
            errors.append(f"{episode_id}: state contains non-finite values")
        if not np.isfinite(tr["actions"]).all():
            errors.append(f"{episode_id}: actions contains non-finite values")
        acts = tr["actions"]
        if acts.size and (acts.min() < -1.0001 or acts.max() > 1.0001):
            errors.append(f"{episode_id}: actions out of [-1,1] range: min={acts.min()} max={acts.max()}")

    if "dt" in tr.files:
        dt = tr["dt"]
        if dt.size and not np.allclose(dt, 0.05, atol=1e-6):
            errors.append(f"{episode_id}: dt not uniformly 0.05s (found {np.unique(dt)})")

    if "frame_index" in tr.files:
        fi = tr["frame_index"]
        if not np.array_equal(fi, np.arange(len(fi))):
            errors.append(f"{episode_id}: frame_index not 0..T-1 contiguous")

    if "sim_step_index" in tr.files:
        ssi = tr["sim_step_index"]
        if len(ssi) > 1 and not np.all(np.diff(ssi) > 0):
            errors.append(f"{episode_id}: sim_step_index not strictly increasing")

    if "state" in tr.files and "next_state" in tr.files:
        state = tr["state"]; next_state = tr["next_state"]
        if len(state) > 1:
            sync_err = np.abs(next_state[:-1] - state[1:]).max()
            if sync_err > 1e-4:
                errors.append(f"{episode_id}: next_state[t] vs state[t+1] max err {sync_err} > 1e-4")

    # image variation check: not a repeated first frame
    if imgs.shape[0] > 2:
        diff = np.abs(imgs[0].astype(int) - imgs[len(imgs) // 2].astype(int)).mean()
        if diff < 1.0:
            errors.append(f"{episode_id}: images barely change over time (mean diff={diff:.3f}), suspect repeated frame")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes_jsonl = os.path.join(args.dataset_root, "episodes.jsonl")
    records = [json.loads(l) for l in open(episodes_jsonl) if l.strip()]

    all_errors = []
    n_checked = 0
    for rec in records:
        ep_id = rec["episode"]
        ep_dir = os.path.join(args.dataset_root, ep_id)
        errs = check_episode(ep_dir, ep_id)
        all_errors.extend(errs)
        n_checked += 1

    report = dict(
        n_episodes_checked=n_checked,
        n_errors=len(all_errors),
        passed=len(all_errors) == 0,
        errors=all_errors,
    )
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"n_episodes_checked": n_checked, "n_errors": len(all_errors), "passed": report["passed"]}))
    if all_errors:
        for e in all_errors[:50]:
            print(e, file=sys.stderr)


if __name__ == "__main__":
    main()
