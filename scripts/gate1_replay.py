"""Replay recorded control actions after restoring a Gate 1 initial snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    directory = args.episode_dir.resolve()
    result = json.loads((directory / "result.json").read_text())
    initial = json.loads((directory / "initial_state.json").read_text())

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(root / "stagecap_robosuite_libero_repro_20260714")

    import numpy as np
    import run_manner_ablation as old
    from synthesis_v2.simulator_state import restore_simulator_snapshot

    entry = old.TASK_REGISTRY[result["task"]]
    env = old.load_sim_cls(entry["sim_cls_path"])(privileged=True, enable_render=False,
          max_steps=entry["max_steps"], seed=result["seed"])
    maximum = {"time_s": 0.0, "qpos": 0.0, "qvel": 0.0, "actuator_ctrl": 0.0}
    count = 0
    first_bad = None
    try:
        model_xml = (directory / "model.xml").read_text()
        replay_model_sha256 = hashlib.sha256(model_xml.encode()).hexdigest()
        model_matches = replay_model_sha256 == initial["model_sha256"]
        env.robosuite_env.reset_from_xml_string(model_xml)
        restore_simulator_snapshot(env, initial["snapshot"])
        restored = {
            "time_s": float(env.robosuite_env.sim.data.time),
            "qpos": np.asarray(env.robosuite_env.sim.data.qpos),
            "qvel": np.asarray(env.robosuite_env.sim.data.qvel),
            "actuator_ctrl": np.asarray(env.robosuite_env.sim.data.ctrl),
        }
        initial_expected = initial["state"]
        initial_errors = {
            "time_s": abs(restored["time_s"] - initial_expected["time_s"]),
            "qpos": float(np.max(np.abs(restored["qpos"] - initial_expected["qpos"]))),
            "qvel": float(np.max(np.abs(restored["qvel"] - initial_expected["qvel"]))),
            "actuator_ctrl": float(np.max(np.abs(restored["actuator_ctrl"] - initial_expected["actuator_ctrl"]))),
        }
        with (directory / "transitions.jsonl").open() as stream:
            for count, line in enumerate(stream, start=1):
                row = json.loads(line)
                env._do_robosuite_step(np.asarray(row["action_t"], dtype=np.float64))
                actual = {
                    "time_s": float(env.robosuite_env.sim.data.time),
                    "qpos": np.asarray(env.robosuite_env.sim.data.qpos),
                    "qvel": np.asarray(env.robosuite_env.sim.data.qvel),
                    "actuator_ctrl": np.asarray(env.robosuite_env.sim.data.ctrl),
                }
                expected = row["state_t_plus_1"]
                errors = {
                    "time_s": abs(actual["time_s"] - expected["time_s"]),
                    "qpos": float(np.max(np.abs(actual["qpos"] - expected["qpos"]))),
                    "qvel": float(np.max(np.abs(actual["qvel"] - expected["qvel"]))),
                    "actuator_ctrl": float(np.max(np.abs(actual["actuator_ctrl"] - expected["actuator_ctrl"]))),
                }
                for key, value in errors.items():
                    maximum[key] = max(maximum[key], value)
                if first_bad is None and any(value > args.atol for value in errors.values()):
                    first_bad = {"transition": count, "errors": errors}
        task_success = bool(env.task_completed())
        functional_passed = task_success == bool(result["execution"]["task_completed"])
        passed = model_matches and first_bad is None and functional_passed
        report = {
            "schema": "mannercap-action-replay-v1",
            "task": result["task"],
            "source_episode": str(directory),
            "transitions_replayed": count,
            "absolute_tolerance": args.atol,
            "model_source_sha256_matches": model_matches,
            "model_source_sha256": replay_model_sha256,
            "initial_state_restore_maximum_absolute_error": initial_errors,
            "maximum_absolute_error": maximum,
            "first_mismatch": first_bad,
            "matching_prefix_transitions": count if first_bad is None else first_bad["transition"] - 1,
            "recorded_task_success": bool(result["execution"]["task_completed"]),
            "replayed_task_success": task_success,
            "functional_outcome_replay_verified": functional_passed,
            "exact_state_replay_verified": passed,
            "action_replay_verified": passed,
            "controller_object_state_captured": initial["snapshot"]["controller_object_state_captured"],
            "manner": "unknown",
            "safety": "unknown",
        }
        (directory / "action_replay.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 0 if passed else 1
    finally:
        env.robosuite_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
