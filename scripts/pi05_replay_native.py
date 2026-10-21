"""Replay stored 7D actions in the deployment controller and compare to the
stored states.  Standalone: needs only synthesis_v2.native_osc_env (bundled in
the delivery package under tools/).

usage: pi05_replay_native.py DATASET_ROOT OUT.json [--episodes id,id] [--sample-n N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

POS_TOL_MM = 5.0
ROT_TOL_RAD = 0.05
GRIP_TOL_M = 0.002


def _rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation as R
    return float((R.from_rotvec(a) * R.from_rotvec(b).inv()).magnitude())


def replay_episode(task: str, seed: int, actions: np.ndarray, next_state: np.ndarray, success_final: bool,
                   *, want_assets: bool = False) -> dict:
    from synthesis_v2.native_osc_env import NativeSession, state8

    s = NativeSession(task, seed, render=False)
    try:
        s.reset()
        init = s.snapshot_initial()
        extra = {"model_xml": s.model_xml(), "camera_parameters": s.camera_parameters(), "initial": init} if want_assets else {}
        pos = rot = grip = 0.0
        states = np.zeros_like(next_state, dtype=np.float32)
        for t in range(len(actions)):
            obs, _ = s.step(actions[t])
            states[t] = state8(obs)
            d = states[t].astype(np.float64) - next_state[t].astype(np.float64)
            pos = max(pos, float(np.linalg.norm(d[:3])) * 1000.0)
            rot = max(rot, _rot_angle(states[t, 3:6].astype(np.float64), next_state[t, 3:6].astype(np.float64)))
            grip = max(grip, float(np.abs(d[6:8]).max()))
        ok_final = bool(s.success())
        res = {"max_pos_err_mm": pos, "max_rot_err_rad": rot, "max_grip_err_m": grip,
               "replayed_success": ok_final, "stored_success": bool(success_final),
               "success_match": ok_final == bool(success_final), "n_steps": int(len(actions))}
        res["pass"] = bool(pos <= POS_TOL_MM and rot <= ROT_TOL_RAD and grip <= GRIP_TOL_M and res["success_match"])
        res["_states"] = states
        res.update(extra)
        return res
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("out")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--sample-n", type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(x) for x in open(os.path.join(a.dataset_root, "episodes.jsonl")) if x.strip()]
    if a.episodes:
        keep = set(a.episodes.split(","))
        recs = [r for r in recs if r["episode"] in keep]
    elif a.sample_n:
        recs = recs[:: max(1, len(recs) // a.sample_n)][: a.sample_n]
    rows = []
    for r in recs:
        tr = np.load(os.path.join(a.dataset_root, r["episode"], "transitions.npz"), allow_pickle=False)
        res = replay_episode(r["task_name"], r["seed"], tr["actions"], tr["next_state"], bool(tr["success"][-1]))
        res.pop("_states")
        res["episode"] = r["episode"]
        rows.append(res)
        print(json.dumps({k: res[k] for k in ("episode", "max_pos_err_mm", "pass")}), flush=True)
    rep = {"n": len(rows), "n_pass": sum(1 for x in rows if x["pass"]), "pos_tol_mm": POS_TOL_MM,
           "rot_tol_rad": ROT_TOL_RAD, "grip_tol_m": GRIP_TOL_M, "episodes": rows}
    json.dump(rep, open(a.out, "w"), indent=1)
    sys.exit(0 if rep["n"] == rep["n_pass"] else 1)


if __name__ == "__main__":
    main()
