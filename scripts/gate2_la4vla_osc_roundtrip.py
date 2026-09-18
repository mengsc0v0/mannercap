"""Run independent physical OSC_POSE steps for recorded teacher transitions.

This is a controller-interface diagnostic, not a student-policy evaluation.
It converts measured end-effector pose transitions to LA4VLA's normalized
7-D OSC action, serializes the action through JSON, restores the recorded
MuJoCo state, and executes one fresh-controller step.  No LLM or model API is
called, and no pass threshold is invented by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_CONFIG = ROOT / "configs" / "controllers" / "panda_la4vla_osc_pose.json"


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Choose unique deterministic indices including both ends when possible."""

    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(length, count)
    if count == 1:
        return [0]
    return [(index * (length - 1)) // (count - 1) for index in range(count)]


def allocate_samples(lengths: Sequence[int], total: int) -> list[int]:
    """Allocate a requested total fairly while respecting episode lengths."""

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("all episode lengths must be positive")
    if total <= 0:
        raise ValueError("total must be positive")
    target = min(total, sum(lengths))
    allocations = [0] * len(lengths)
    while sum(allocations) < target:
        progressed = False
        for index, length in enumerate(lengths):
            if allocations[index] < length and sum(allocations) < target:
                allocations[index] += 1
                progressed = True
        if not progressed:
            break
    return allocations


def _finite_metrics(values: Sequence[float]) -> dict[str, float | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))

    def percentile(fraction: float) -> float | None:
        if not finite:
            return None
        position = fraction * (len(finite) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(0.95),
        "maximum": max(finite) if finite else None,
    }


def _executed_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    executed = [row for row in rows if row.get("status") == "executed"]
    return {
        "count": len(executed),
        "position_error_norm_m": _finite_metrics(
            [row["position_error_norm_m"] for row in executed]
        ),
        "orientation_error_rad": _finite_metrics(
            [row["orientation_error_rad"] for row in executed]
        ),
    }


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    executed = [row for row in samples if row.get("status") == "executed"]
    position_vectors = [row["position_error_vector_m"] for row in executed]
    tasks = sorted({str(row["task"]) for row in samples})
    phases = sorted({str(row.get("phase_program_marker")) for row in samples})
    return {
        "samples_requested": len(samples),
        "physical_controller_steps": len(executed),
        "adapter_errors": sum(row.get("status") == "adapter_error" for row in samples),
        "execution_errors": sum(row.get("status") == "execution_error" for row in samples),
        "clipped_samples": sum(bool(row.get("adapter", {}).get("clipped")) for row in samples),
        "serialization_round_trip_exact": all(
            row.get("serialization_round_trip_exact") is True for row in executed
        ) and bool(executed),
        **_executed_metrics(executed),
        "position_error_bias_m": (
            [statistics.fmean(vector[axis] for vector in position_vectors) for axis in range(3)]
            if position_vectors else None
        ),
        "restored_before_maximum_error": {
            "position_m": max(
                (row["restored_before_error"]["position_m"] for row in executed),
                default=None,
            ),
            "orientation_rad": max(
                (row["restored_before_error"]["orientation_rad"] for row in executed),
                default=None,
            ),
        },
        "non_initial_transition_metrics": _executed_metrics([
            row for row in executed if row["transition_index_zero_based"] != 0
        ]),
        "by_task": {
            task: _executed_metrics([row for row in executed if row["task"] == task])
            for task in tasks
        },
        "by_phase_program_marker": {
            phase: _executed_metrics([
                row for row in executed if str(row.get("phase_program_marker")) == phase
            ])
            for phase in phases
        },
        "phase_program_markers_covered": phases,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_transitions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _restore_transition_state(env: Any, state: dict[str, Any]) -> None:
    import numpy as np

    sim = env.robosuite_env.sim
    data = sim.data
    data.qpos[:] = np.asarray(state["qpos"], dtype=np.float64)
    data.qvel[:] = np.asarray(state["qvel"], dtype=np.float64)
    if data.act.size:
        data.act[:] = np.asarray(state["act"], dtype=np.float64)
    data.time = float(state["time_s"])
    sim.forward()
    data.qacc_warmstart[:] = np.asarray(state["qacc_warmstart"], dtype=np.float64)


def _reinitialize_controllers_at_restored_state(env: Any) -> list[str]:
    """Bind controller references and OSC null posture to the restored state."""

    initialized: list[str] = []
    for robot_index, robot in enumerate(env.robosuite_env.robots):
        composite = robot.composite_controller
        for part_name, controller in composite.part_controllers.items():
            controller.update(force=True)
            controller.update_initial_joints(controller.joint_pos.copy())
            initialized.append(f"robot{robot_index}.{part_name}")
    return initialized


def _pose(env: Any) -> tuple[list[float], list[float]]:
    import numpy as np

    data = env.robosuite_env.sim.data
    return (
        np.asarray(data.xpos[env.gripper_link_idx], dtype=np.float64).tolist(),
        np.asarray(data.xquat[env.gripper_link_idx], dtype=np.float64).tolist(),
    )


def run_episode(
    directory: Path,
    indices: Sequence[int],
    *,
    allow_clipping: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import run_manner_ablation as old
    from synthesis_v2.la4vla_osc_adapter import (
        OscAdapterError,
        adapt_pose_transition,
        orientation_error_rad,
    )
    from synthesis_v2.simulator_state import reset_robosuite_from_model_binary

    result = _read_json(directory / "result.json")
    initial = _read_json(directory / "initial_state.json")
    transitions = _read_transitions(directory / "transitions.jsonl")
    compiled = directory / "model.mjb"
    expected_sha = initial.get("compiled_model", {}).get("sha256")
    actual_sha = hashlib.sha256(compiled.read_bytes()).hexdigest()
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(f"compiled model SHA-256 mismatch: {directory}")

    entry = old.TASK_REGISTRY[result["task"]]
    env = old.load_sim_cls(entry["sim_cls_path"])(
        controller_cfg=str(CONTROLLER_CONFIG),
        privileged=True,
        enable_render=False,
        max_steps=entry["max_steps"],
        seed=result["seed"],
    )
    samples: list[dict[str, Any]] = []
    try:
        for transition_index in indices:
            row = transitions[transition_index]
            sample: dict[str, Any] = {
                "task": result["task"],
                "source_episode": str(directory),
                "transition_index_zero_based": transition_index,
                "phase_program_marker": row.get("phase_program_marker"),
                "status": "started",
            }
            try:
                before = row["state_t"]
                target = row["state_t_plus_1"]
                source_action = row["action_t"]
                if len(source_action) < 1:
                    raise ValueError("source action has no gripper channel")
                adapted = adapt_pose_transition(
                    position_before_m=before["eef_position_m"],
                    quaternion_before_wxyz=before["eef_quaternion_wxyz"],
                    position_after_m=target["eef_position_m"],
                    quaternion_after_wxyz=target["eef_quaternion_wxyz"],
                    gripper_command=source_action[-1],
                    allow_clipping=allow_clipping,
                )
                sample["adapter"] = adapted.as_dict()
                serialized = json.dumps(list(adapted.normalized_action), allow_nan=False)
                action = json.loads(serialized)
                sample["serialization_round_trip_exact"] = tuple(action) == adapted.normalized_action

                # Rebuild from the compiled source model for every sample so
                # controller caches cannot leak from one diagnostic step to another.
                reset_robosuite_from_model_binary(env, compiled)
                _restore_transition_state(env, before)
                sample["controllers_reinitialized_at_restored_state"] = (
                    _reinitialize_controllers_at_restored_state(env)
                )
                restored_position, restored_quaternion = _pose(env)
                sample["restored_before_error"] = {
                    "position_m": float(np.linalg.norm(
                        np.asarray(restored_position) - np.asarray(before["eef_position_m"])
                    )),
                    "orientation_rad": orientation_error_rad(
                        restored_quaternion, before["eef_quaternion_wxyz"]
                    ),
                }
                base_index = env.base_link_idx
                sim_data = env.robosuite_env.sim.data
                sample["controller_base_pose_world"] = {
                    "position_m": np.asarray(sim_data.xpos[base_index]).tolist(),
                    "quaternion_wxyz": np.asarray(sim_data.xquat[base_index]).tolist(),
                }
                env.robosuite_env.step(np.asarray(action, dtype=np.float64))
                actual_position, actual_quaternion = _pose(env)
                position_error = np.asarray(actual_position) - np.asarray(target["eef_position_m"])
                sample.update({
                    "status": "executed",
                    "target_position_m": target["eef_position_m"],
                    "actual_position_m": actual_position,
                    "target_quaternion_wxyz": target["eef_quaternion_wxyz"],
                    "actual_quaternion_wxyz": actual_quaternion,
                    "position_error_vector_m": position_error.tolist(),
                    "position_error_norm_m": float(np.linalg.norm(position_error)),
                    "orientation_error_rad": orientation_error_rad(
                        actual_quaternion, target["eef_quaternion_wxyz"]
                    ),
                })
            except OscAdapterError as error:
                sample.update({
                    "status": "adapter_error",
                    "error_code": error.code,
                    "error_field": error.field,
                    "error": str(error),
                })
            except Exception:
                sample.update({"status": "execution_error", "error": traceback.format_exc()})
            samples.append(sample)
    finally:
        env.robosuite_env.close()

    metadata = {
        "task": result["task"],
        "seed": result["seed"],
        "source_episode": str(directory),
        "source_transition_count": len(transitions),
        "sample_indices_zero_based": list(indices),
        "compiled_model_sha256": actual_sha,
    }
    return metadata, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--allow-clipping", action="store_true")
    args = parser.parse_args()

    directories = [path.resolve() for path in args.episode_dirs]
    transitions = [_read_transitions(path / "transitions.jsonl") for path in directories]
    allocations = allocate_samples([len(rows) for rows in transitions], args.sample_count)

    sys.path[:0] = [str(ROOT), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")

    episodes: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for directory, rows, allocation in zip(directories, transitions, allocations):
        metadata, episode_samples = run_episode(
            directory,
            evenly_spaced_indices(len(rows), allocation),
            allow_clipping=args.allow_clipping,
        )
        episodes.append(metadata)
        samples.extend(episode_samples)

    summary = summarize_samples(samples)
    report = {
        "schema": "mannercap-la4vla-osc-roundtrip-v1",
        "purpose": "controller_interface_diagnostic",
        "controller_config": str(CONTROLLER_CONFIG),
        "controller_config_sha256": hashlib.sha256(CONTROLLER_CONFIG.read_bytes()).hexdigest(),
        "allow_clipping": args.allow_clipping,
        "episodes": episodes,
        "samples": samples,
        "summary": summary,
        "api_calls": 0,
        "student_model_inference_calls": 0,
        "gripper_physical_behavior_qualified": False,
        "systematic_bias_acceptance_threshold": None,
        "qualification": "unknown_pending_independent_threshold_and_review",
        "promotion_allowed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False))
    return 0 if summary["physical_controller_steps"] == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
