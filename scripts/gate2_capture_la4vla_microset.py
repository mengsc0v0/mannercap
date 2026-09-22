"""Capture a deterministic 64-window MannerCAP microset for LA4VLA preflight.

The output is an intermediate JSON/image bundle.  It does not modify the
external LA4VLA tree and makes no learning or generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "lift": "Lift the cube.",
    "stack": "Stack the red cube on the green cube.",
}


def evenly_spaced_window_starts(length: int, horizon: int, count: int) -> list[int]:
    if horizon <= 0 or count <= 0:
        raise ValueError("horizon and count must be positive")
    available = length - horizon + 1
    if available <= 0:
        raise ValueError("episode is shorter than the requested horizon")
    count = min(count, available)
    if count == 1:
        return [0]
    return [(index * (available - 1)) // (count - 1) for index in range(count)]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_transitions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def episode_metadata(
    directory: Path, *, include_task_failures: bool = False
) -> dict[str, Any]:
    """Validate one source episode before creating any output files."""
    result_path = directory / "result.json"
    result = _read_json(result_path)
    task = result.get("task")
    if task not in PROMPTS:
        raise ValueError(f"unsupported task in {result_path}: {task!r}")
    if result.get("status") != "executed":
        raise ValueError(f"source episode was not executed successfully: {directory}")
    task_completed = result.get("task_completed")
    if task_completed is None:
        task_completed = result.get("execution", {}).get("task_completed")
    if not bool(task_completed) and not include_task_failures:
        raise ValueError(
            f"source episode did not complete the task: {directory}; "
            "use --include-task-failures only for an explicitly diagnostic dataset"
        )
    prompt = result.get("raw_language") or PROMPTS[task]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"source episode has no usable language prompt: {directory}")
    transitions_path = directory / "transitions.jsonl"
    if not transitions_path.is_file():
        raise ValueError(f"source transitions are missing: {transitions_path}")
    episode_id = str(result.get("episode_id") or directory.name)
    file_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._")
    if not file_stem:
        raise ValueError(f"source episode has no usable identifier: {directory}")
    return {
        "result": result,
        "task": task,
        "prompt": prompt.strip(),
        "episode_id": episode_id,
        "file_stem": file_stem,
        "task_completed": bool(task_completed),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }


def _restore_state(env: Any, state: dict[str, Any]) -> None:
    import numpy as np

    data = env.robosuite_env.sim.data
    data.qpos[:] = np.asarray(state["qpos"], dtype=np.float64)
    data.qvel[:] = np.asarray(state["qvel"], dtype=np.float64)
    if data.act.size:
        data.act[:] = np.asarray(state["act"], dtype=np.float64)
    data.time = float(state["time_s"])
    env.robosuite_env.sim.forward()
    data.qacc_warmstart[:] = np.asarray(state["qacc_warmstart"], dtype=np.float64)


def _png_bytes(sim: Any, camera_name: str, size: int) -> bytes:
    import numpy as np
    from PIL import Image

    pixels = sim.render(camera_name=camera_name, width=size, height=size)
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.shape != (size, size, 3):
        raise ValueError(f"unexpected image shape from {camera_name}: {pixels.shape}")
    # Raw MuJoCo offscreen pixels have a bottom-left origin.
    pixels = np.flipud(pixels)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def capture_episode(
    directory: Path,
    output: Path,
    *,
    count: int,
    horizon: int,
    image_size: int,
    task_index: int,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import numpy as np
    import run_manner_ablation as old
    from synthesis_v2.la4vla_osc_adapter import (
        adapt_pose_transition,
        quaternion_to_rotvec_wxyz,
    )
    from synthesis_v2.simulator_state import reset_robosuite_from_model_binary

    metadata = metadata or episode_metadata(directory)
    result = metadata["result"]
    initial = _read_json(directory / "initial_state.json")
    task = result["task"]
    transitions = _read_transitions(directory / "transitions.jsonl")
    starts = evenly_spaced_window_starts(len(transitions), horizon, count)
    compiled = directory / "model.mjb"
    compiled_sha = hashlib.sha256(compiled.read_bytes()).hexdigest()
    if compiled_sha != initial["compiled_model"]["sha256"]:
        raise ValueError(f"compiled model SHA-256 mismatch: {directory}")

    entry = old.TASK_REGISTRY[task]
    env = old.load_sim_cls(entry["sim_cls_path"])(
        privileged=True,
        enable_render=True,
        max_steps=entry["max_steps"],
        seed=result["seed"],
    )
    windows: list[dict[str, Any]] = []
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    try:
        reset_robosuite_from_model_binary(env, compiled)
        gripper_reference = env.robosuite_env.robots[0]._ref_gripper_joint_pos_indexes
        if isinstance(gripper_reference, dict):
            if "right" not in gripper_reference:
                raise ValueError(
                    f"right gripper qpos indices unavailable: {sorted(gripper_reference)}"
                )
            gripper_reference = gripper_reference["right"]
        gripper_indices = list(gripper_reference)
        if len(gripper_indices) != 2:
            raise ValueError(f"expected two gripper qpos indices, got {gripper_indices}")
        for ordinal, start in enumerate(starts):
            source_state = transitions[start]["state_t"]
            source_action_sequence = [
                transitions[start + offset]["action_t"]
                for offset in range(horizon)
            ]
            _restore_state(env, source_state)
            image_paths: dict[str, str] = {}
            image_hashes: dict[str, str] = {}
            for key, camera in {
                "image": "robot0_robotview",
                "wrist_image": "robot0_eye_in_hand",
            }.items():
                payload = _png_bytes(env.robosuite_env.sim, camera, image_size)
                relative = Path("images") / (
                    f"{metadata['file_stem']}_{ordinal:03d}_{key}.png"
                )
                (output / relative).write_bytes(payload)
                image_paths[key] = relative.as_posix()
                image_hashes[key] = hashlib.sha256(payload).hexdigest()

            rows: list[dict[str, Any]] = []
            for offset in range(horizon):
                transition = transitions[start + offset]
                before = transition["state_t"]
                after = transition["state_t_plus_1"]
                adapted = adapt_pose_transition(
                    position_before_m=before["eef_position_m"],
                    quaternion_before_wxyz=before["eef_quaternion_wxyz"],
                    position_after_m=after["eef_position_m"],
                    quaternion_after_wxyz=after["eef_quaternion_wxyz"],
                    gripper_command=transition["action_t"][-1],
                )
                rotvec = quaternion_to_rotvec_wxyz(before["eef_quaternion_wxyz"])
                qpos = before["qpos"]
                state = (
                    list(before["eef_position_m"])
                    + list(rotvec)
                    + [float(qpos[index]) for index in gripper_indices]
                )
                rows.append({
                    "timestamp": float(before["time_s"]),
                    "observation_state": state,
                    "action": list(adapted.normalized_action),
                    "phase_program_marker": transition.get("phase_program_marker"),
                })
            windows.append({
                "task": task,
                "task_index": task_index,
                "prompt": metadata["prompt"],
                "source_episode_id": metadata["episode_id"],
                "source_episode": str(directory),
                "source_result_sha256": metadata["result_sha256"],
                "source_initial_state_sha256": result.get("initial_state_sha256"),
                "source_compiled_model_sha256": result.get("compiled_model_sha256"),
                "source_task_completed": metadata["task_completed"],
                "manner_tag": result.get("manner_tag", "unspecified"),
                "requested_manner": result.get("requested_manner"),
                "source_start_transition_zero_based": start,
                "observation_render_source_transition_zero_based": start,
                "observation_render_source_state_time_s": float(source_state["time_s"]),
                "observation_render_source_state_sha256": _canonical_sha256(source_state),
                "observation_render_before_action": True,
                "source_action_t_sequence_sha256": _canonical_sha256(source_action_sequence),
                "horizon": horizon,
                "images": image_paths,
                "image_sha256": image_hashes,
                "rows": rows,
            })
    finally:
        env.robosuite_env.close()
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--include-task-failures",
        action="store_true",
        help="allow failed task outcomes for an explicitly diagnostic dataset",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.samples % len(args.episode_dirs):
        parser.error("--samples must be positive and divisible by the episode count")

    directories = [directory.resolve() for directory in args.episode_dirs]
    metadata = [
        episode_metadata(
            directory, include_task_failures=args.include_task_failures
        )
        for directory in directories
    ]
    prompts = sorted({item["prompt"] for item in metadata})
    prompt_indices = {prompt: index for index, prompt in enumerate(prompts)}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")

    per_episode = args.samples // len(args.episode_dirs)
    windows: list[dict[str, Any]] = []
    for directory, item in zip(directories, metadata):
        windows.extend(capture_episode(
            directory.resolve(), output, count=per_episode,
            horizon=args.horizon, image_size=args.image_size,
            task_index=prompt_indices[item["prompt"]], metadata=item,
        ))
    manifest = {
        "schema": "mannercap-la4vla-microset-staging-v1",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sample_count": len(windows),
        "horizon": args.horizon,
        "image_size": args.image_size,
        "windows": windows,
        "api_calls": 0,
        "includes_task_failures": args.include_task_failures,
        "independent_source_episode_count": len(directories),
        "source_manner_tags": sorted({
            str(item["result"].get("manner_tag", "unspecified"))
            for item in metadata
        }),
        "purpose": "small_sample_training_pipeline_preflight",
        "generalization_claim_allowed": False,
    }
    (output / "staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sample_count": len(windows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
