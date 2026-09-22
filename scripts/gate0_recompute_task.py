"""Recompute the simulator task predicate from saved final state in a fresh env.

This is not action replay and does not evaluate manner or safety.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    directory = args.episode_dir.resolve()
    result = json.loads((directory / "result.json").read_text())
    last = None
    with (directory / "transitions.jsonl").open() as stream:
        for line in stream:
            last = json.loads(line)
    if last is None:
        raise ValueError("No recorded final state")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "stagecap_robosuite_libero_repro_20260714"))
    os.chdir(root / "stagecap_robosuite_libero_repro_20260714")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import run_manner_ablation as old
    entry = old.TASK_REGISTRY[result["task"]]
    env = old.load_sim_cls(entry["sim_cls_path"])(privileged=True, enable_render=False,
          max_steps=entry["max_steps"], seed=result["seed"])
    try:
        env.robosuite_env.reset_from_xml_string((directory / "model.xml").read_text())
        sim = env.robosuite_env.sim
        state = last["state_t_plus_1"]
        sim.data.qpos[:] = state["qpos"]
        sim.data.qvel[:] = state["qvel"]
        sim.data.ctrl[:] = state["actuator_ctrl"]
        sim.data.time = state["time_s"]
        sim.forward()
        actual = bool(env.task_completed())
        report = {"task": result["task"], "recomputed_task_success": actual,
                  "recorded_task_success": last["task_success_after"],
                  "matches_recorded": actual == last["task_success_after"],
                  "action_replay_verified": False, "manner": "unknown", "safety": "unknown"}
        (directory / "task_recomputation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 0 if report["matches_recorded"] else 1
    finally:
        env.robosuite_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
