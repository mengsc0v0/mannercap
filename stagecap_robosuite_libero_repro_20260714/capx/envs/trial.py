"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import ast
import base64
import builtins
import copy
import gc
import io
import json
import os
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
    is_valid_executable_python_block,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10
ROBOT_ACTION_RE = re.compile(
    r"\b("
    r"goto_pose|goto_pose_interactive_cartesian|open_gripper|close_gripper|"
    r"goto_pose_checked|goto_finger_midpoint_checked|"
    r"move_linear_delta|move_along_direction|move_finger_midpoint_delta|move_finger_midpoint_delta_strict|"
    r"verified_lift_pick_object|approach_and_grasp_object|"
        r"guarded_transport_held_object_to_target|low_release_object_to_target|pick_transport_release_object_to_target|local_placement_closure|"
    r"pull_sleeved_drawer|"
    r"move_to_joints|execute_joint_trajectory|goto_home_joint_position|"
    r"plan_grasp_trajectory|plan_with_grasped_object"
    r")\s*\("
)
COMMON_GLOBAL_NAMES = {
    "np",
    "time",
    "obs",
    "INPUTS",
    "RESULT",
    "APIS",
    "env",
}

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _filter_executable_code_blocks(
    code_blocks: list[str],
    metadata: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], int]:
    filtered_blocks: list[str] = []
    filtered_metadata: list[dict[str, Any]] = []
    skipped = 0
    for block, meta in zip(code_blocks, metadata, strict=False):
        if is_valid_executable_python_block(block):
            filtered_blocks.append(block)
            filtered_metadata.append(meta)
        else:
            skipped += 1
    return filtered_blocks, filtered_metadata, skipped


def _invalid_code_feedback(attempt: int, skipped_blocks: int) -> dict[str, Any]:
    return {
        "sandbox_rc": -1,
        "stdout": "",
        "stderr": (
            "The previous model response did not contain executable top-level Python "
            f"CAP code after sanitation (attempt {attempt}; skipped blocks: {skipped_blocks}). "
            "This usually happens when the response is prose, a tool-call fragment, "
            "or code with illegal top-level statements such as return. "
            "Regenerate exactly one fenced Python code block that uses only the allowed "
            "CAP robot/perception APIs and actually issues robot actions toward the task."
        ),
        "task_completed": False,
    }


def _non_action_code_feedback(attempt: int) -> dict[str, Any]:
    return {
        "sandbox_rc": -1,
        "stdout": "",
        "stderr": (
            "The previous CAP program was executable but contained no robot action APIs "
            f"(attempt {attempt}). It only observed, printed, or visually checked state. "
            "Regenerate exactly one fenced Python code block that physically acts on the "
            "exact LIBERO goal using robot-control APIs such as goto_pose, move_to_joints, "
            "open_gripper, or close_gripper. Perception, IK solving, and grasp planning "
            "alone do not count as robot actions. Do not claim the task is complete "
            "without contact/action evidence."
        ),
        "task_completed": False,
    }


def _contains_robot_action(code_blocks: list[str]) -> bool:
    return any(ROBOT_ACTION_RE.search(block) for block in code_blocks)


def _undefined_name_feedback(
    attempt: int,
    missing_names: list[str],
) -> dict[str, Any]:
    missing = ", ".join(f"`{name}`" for name in missing_names)
    return {
        "sandbox_rc": -1,
        "stdout": "",
        "stderr": (
            "The previous CAP program referenced variable(s) before defining them "
            f"(attempt {attempt}): {missing}. This often means the model emitted a "
            "truncated tail fragment instead of a complete program. Regenerate exactly "
            "one complete executable Python block: initialize every position, quaternion, "
            "pixel, mask, depth, and helper variable from get_observation(), perception, "
            "or explicit constants before using it, then execute the physical robot "
            "actions for the exact LIBERO goal."
        ),
        "task_completed": False,
    }


def _depth_dimension_feedback(attempt: int) -> dict[str, Any]:
    return {
        "sandbox_rc": -1,
        "stdout": "",
        "stderr": (
            "The previous CAP program indexed a LIBERO depth image as if it were "
            f"3-dimensional (attempt {attempt}), for example `depth[:, :, 0]`. "
            "In this environment `obs[...]['images']['depth']` is usually already "
            "2-dimensional with shape (H, W). Regenerate exactly one complete "
            "Python block that first assigns `depth = camera['images']['depth']`, "
            "then normalizes it with `if depth.ndim == 3: depth = depth[:, :, 0]`, "
            "and only indexes depth as `depth[v, u]` after clamping pixel bounds."
        ),
        "task_completed": False,
    }


def _has_static_depth_dimension_error(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        slice_node = node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 3:
            continue
        if _subscript_base_mentions_depth(node.value) and not _inside_depth_ndim_guard(node, parent):
            return True
    return False


def _subscript_base_mentions_depth(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "depth" in node.id.lower()
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and "depth" in node.value.lower()
    if isinstance(node, ast.Attribute):
        return "depth" in node.attr.lower() or _subscript_base_mentions_depth(node.value)
    if isinstance(node, ast.Subscript):
        return _subscript_base_mentions_depth(node.value) or _subscript_base_mentions_depth(node.slice)
    if isinstance(node, ast.Call):
        return _subscript_base_mentions_depth(node.func)
    if isinstance(node, ast.Tuple):
        return any(_subscript_base_mentions_depth(item) for item in node.elts)
    return False


def _inside_depth_ndim_guard(node: ast.AST, parent: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parent:
        current = parent[current]
        if isinstance(current, ast.If) and _test_mentions_depth_ndim(current.test):
            return True
    return False


def _test_mentions_depth_ndim(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and item.attr == "ndim":
            if _subscript_base_mentions_depth(item.value):
                return True
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            if item.func.id == "len" and item.args and _subscript_base_mentions_depth(item.args[0]):
                return True
    return False


def _preflight_missing_names(
    code: str,
    defined_names: set[str],
) -> list[str]:
    """Return names read by a code block before assignment/global availability."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    assigned: set[str] = set()
    loaded: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Param)):
                assigned.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(node.name)

    allowed = set(dir(builtins)) | COMMON_GLOBAL_NAMES | defined_names | assigned | imported
    return sorted(name for name in loaded if name not in allowed)


def _names_defined_by_code(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


def _save_trial_trajectory(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str | None:
    """Persist per-step observation/action data next to trial videos."""
    if not config.get("output_dir") or not hasattr(env, "get_trajectory_steps"):
        return None

    steps = env.get_trajectory_steps(clear=False)
    if not steps:
        return None

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    os.makedirs(base_dir, exist_ok=True)
    steps_path = os.path.join(base_dir, "steps.jsonl")
    with open(steps_path, "w") as f:
        for step in steps:
            f.write(json.dumps(_json_safe(step), allow_nan=False) + "\n")
    _save_trajectory_images(env, base_dir, steps)

    metadata_path = os.path.join(base_dir, "steps_metadata.json")
    metadata = {
        "schema_version": "capx_libero_steps_v0.2",
        "trajectory_format": "jsonl",
        "step_count": len(steps),
        "required_fields": [
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
        ],
        "state_schema": {
            "observation.state": "LeRobot-style eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)",
            "observation.state.raw_joint_gripper": "Cap-X raw joint_pos(7) + gripper_fraction(1)",
        },
        "action_schema": {
            "action": "Cap-X raw joint_delta(7) + gripper_command(1)",
            "action.raw_joint_gripper": "Alias of raw Cap-X action for explicit auditability",
            "lerobot_action_candidate": (
                "Derived 7-D candidate: delta eef_pos/eef_axis_angle(6) + raw gripper command(1)"
            ),
        },
        "candidate_action_warning": (
            "lerobot_action_candidate is derived from state deltas and raw gripper action; "
            "validate controller semantics before production training."
        ),
        "source": "capx.envs.simulators.libero.FrankaLiberoEnv",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved trajectory steps to {steps_path} ({len(steps)} steps)")
    return steps_path


def _save_trajectory_images(
    env: CodeExecutionEnvBase,
    base_dir: str,
    steps: list[dict[str, Any]],
) -> None:
    if not steps:
        return
    wants_agent = any((step.get("observation.images.image") or {}).get("path") for step in steps)
    wants_wrist = any((step.get("observation.images.image2") or {}).get("path") for step in steps)
    if not (wants_agent or wants_wrist):
        return

    agent_frames = _trajectory_frame_buffer(env, "agent")
    wrist_frames = _trajectory_frame_buffer(env, "wrist")
    if wants_agent and agent_frames:
        _write_trajectory_image_sequence(base_dir, steps, "observation.images.image", agent_frames)
    if wants_wrist and wrist_frames:
        _write_trajectory_image_sequence(base_dir, steps, "observation.images.image2", wrist_frames)


def _trajectory_frame_buffer(env: CodeExecutionEnvBase, camera: str) -> list[np.ndarray]:
    low_level_env = getattr(env, "low_level_env", env)
    attr = "_frame_buffer" if camera == "agent" else "_wrist_frame_buffer"
    frames = getattr(low_level_env, attr, None)
    if isinstance(frames, list) and frames:
        return [frame.copy() for frame in frames]
    render_name = "render" if camera == "agent" else "render_wrist"
    if hasattr(env, render_name):
        frame = getattr(env, render_name)()
        if frame is not None:
            return [frame]
    return []


def _write_trajectory_image_sequence(
    base_dir: str,
    steps: list[dict[str, Any]],
    key: str,
    frames: list[np.ndarray],
) -> None:
    written_paths = set()
    for step in steps:
        image_ref = step.get(key) or {}
        rel_path = image_ref.get("path")
        if not rel_path or rel_path in written_paths:
            continue
        frame_index = int(step.get("image_frame_index", step.get("frame_index", 0)))
        frame = frames[min(frame_index, len(frames) - 1)]
        path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(frame).save(path)
        written_paths.add(rel_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = _replace_named_prompt_slots(
        multi_turn_prompt,
        {
            "executed_code": executed_code,
            "console_stdout": info_step["stdout"],
            "console_stderr": info_step["stderr"],
        },
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        timeout_seconds = int(config.get("trial_timeout_seconds", 1000))
        signal.alarm(max(1, timeout_seconds))  # restart fresh timeout from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "env": env,
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks. Fixed/oracle CAP files are trusted source
    # code, not natural-language model replies, so avoid response sanitation when
    # the file already compiles as a complete Python module.
    if config["use_oracle_code"] and is_valid_executable_python_block(raw_code):
        initial_blocks_raw = [raw_code]
    else:
        initial_blocks_raw = _extract_code(raw_code)
    initial_metadata_raw = [{"generation": 0, "regenerated": False}] * len(initial_blocks_raw)
    initial_blocks, initial_metadata, skipped_initial_blocks = _filter_executable_code_blocks(
        initial_blocks_raw,
        initial_metadata_raw,
    )
    if skipped_initial_blocks:
        print(f"Skipped {skipped_initial_blocks} non-executable initial code block(s)")
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend(initial_metadata)
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "raw_code_blocks": initial_blocks_raw,
        "skipped_non_executable_blocks": skipped_initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    max_code_blocks = int(config.get("max_multiturn_blocks", MULTITURN_LIMIT + 1))
    max_code_blocks = max(1, max_code_blocks)
    if config["use_oracle_code"]:
        multi_turn_prompt = None

    if not code_blocks and multi_turn_prompt:
        for invalid_attempt in range(1, max(2, max_code_blocks) + 1):
            info_step = _invalid_code_feedback(invalid_attempt, skipped_initial_blocks)
            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, 0, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                wrist_base64_history=wrist_base64_history,
            )
            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision != "regenerate" or not new_code:
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "decision": f"invalid_initial_{decision}",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                num_finishes += int(decision == "finish")
                continue

            new_blocks_raw = _extract_code(new_code)
            new_metadata_raw = [
                {"generation": num_regenerations + 1, "regenerated": True,
                 "regenerated_at_idx": 0, "reason": "invalid_initial_code"}
            ] * len(new_blocks_raw)
            new_blocks, new_metadata, skipped_new_blocks = _filter_executable_code_blocks(
                new_blocks_raw,
                new_metadata_raw,
            )
            all_responses.append({
                "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                "block_idx": [0],
                "code_blocks": new_blocks,
                "raw_code_blocks": new_blocks_raw,
                "skipped_non_executable_blocks": skipped_new_blocks,
                "decision": "regenerate_invalid_initial",
                "reasoning": mt_reasoning if mt_reasoning is not None else "",
            })
            num_regenerations += 1
            if partial_artifacts is not None:
                partial_artifacts["num_regenerations"] = num_regenerations
            if new_blocks:
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(new_metadata)
                break
            skipped_initial_blocks = skipped_new_blocks

        with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
            json.dump(all_responses, f)

    if code_blocks and multi_turn_prompt and not _contains_robot_action(code_blocks):
        for invalid_attempt in range(1, max(2, max_code_blocks) + 1):
            info_step = _non_action_code_feedback(invalid_attempt)
            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, 0, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                wrist_base64_history=wrist_base64_history,
            )
            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision != "regenerate" or not new_code:
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "decision": f"non_action_initial_{decision}",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                num_finishes += int(decision == "finish")
                continue

            new_blocks_raw = _extract_code(new_code)
            new_metadata_raw = [
                {"generation": num_regenerations + 1, "regenerated": True,
                 "regenerated_at_idx": 0, "reason": "non_action_initial_code"}
            ] * len(new_blocks_raw)
            new_blocks, new_metadata, skipped_new_blocks = _filter_executable_code_blocks(
                new_blocks_raw,
                new_metadata_raw,
            )
            all_responses.append({
                "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                "block_idx": [0],
                "code_blocks": new_blocks,
                "raw_code_blocks": new_blocks_raw,
                "skipped_non_executable_blocks": skipped_new_blocks,
                "decision": "regenerate_non_action_initial",
                "reasoning": mt_reasoning if mt_reasoning is not None else "",
            })
            num_regenerations += 1
            if partial_artifacts is not None:
                partial_artifacts["num_regenerations"] = num_regenerations
            if new_blocks and _contains_robot_action(new_blocks):
                code_blocks = new_blocks
                code_block_metadata = new_metadata
                break
            skipped_initial_blocks = skipped_new_blocks

        with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
            json.dump(all_responses, f)

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx < max_code_blocks:
        code = code_blocks[code_block_idx]
        missing_names = _preflight_missing_names(
            code,
            set(getattr(env, "_exec_globals", {}).keys()),
        )
        if missing_names and multi_turn_prompt:
            info_step = _undefined_name_feedback(num_regenerations + 1, missing_names)
            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                wrist_base64_history=wrist_base64_history,
            )
            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)
            if decision == "regenerate" and new_code:
                print(
                    "Preflight rejected code block with undefined names: "
                    + ", ".join(missing_names)
                )
                new_blocks_raw = _extract_code(new_code)
                new_metadata_raw = [
                    {"generation": num_regenerations + 1, "regenerated": True,
                     "regenerated_at_idx": code_block_idx, "reason": "undefined_name_preflight"}
                ] * len(new_blocks_raw)
                new_blocks, new_metadata, skipped_new_blocks = _filter_executable_code_blocks(
                    new_blocks_raw,
                    new_metadata_raw,
                )
                if skipped_new_blocks:
                    print(f"Skipped {skipped_new_blocks} non-executable regenerated code block(s)")
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "raw_code_blocks": new_blocks_raw,
                    "skipped_non_executable_blocks": skipped_new_blocks,
                    "decision": "regenerate_undefined_name_preflight",
                    "preflight_missing_names": missing_names,
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(new_metadata)
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations
                with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
                    json.dump(all_responses, f)
                continue
            all_responses.append({
                "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                "decision": f"undefined_name_preflight_{decision}",
                "preflight_missing_names": missing_names,
                "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
            })
            num_finishes += int(decision == "finish")
            break

        if _has_static_depth_dimension_error(code) and multi_turn_prompt:
            info_step = _depth_dimension_feedback(num_regenerations + 1)
            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                wrist_base64_history=wrist_base64_history,
            )
            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)
            if decision == "regenerate" and new_code:
                print("Preflight rejected code block with static depth dimension error")
                new_blocks_raw = _extract_code(new_code)
                new_metadata_raw = [
                    {"generation": num_regenerations + 1, "regenerated": True,
                     "regenerated_at_idx": code_block_idx, "reason": "depth_dimension_preflight"}
                ] * len(new_blocks_raw)
                new_blocks, new_metadata, skipped_new_blocks = _filter_executable_code_blocks(
                    new_blocks_raw,
                    new_metadata_raw,
                )
                if skipped_new_blocks:
                    print(f"Skipped {skipped_new_blocks} non-executable regenerated code block(s)")
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "raw_code_blocks": new_blocks_raw,
                    "skipped_non_executable_blocks": skipped_new_blocks,
                    "decision": "regenerate_depth_dimension_preflight",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(new_metadata)
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations
                with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
                    json.dump(all_responses, f)
                continue
            all_responses.append({
                "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                "decision": f"depth_dimension_preflight_{decision}",
                "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
            })
            num_finishes += int(decision == "finish")
            break

        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next
        if info_step.get("sandbox_rc", 1) == 0:
            defined_names = _names_defined_by_code(code)
            if defined_names and hasattr(env, "_exec_globals"):
                # The executor keeps globals persistently; this set is only used
                # by preflight if a custom env exposes no _exec_globals initially.
                getattr(env, "_exec_globals").update({
                    name: getattr(env, "_exec_globals").get(name)
                    for name in defined_names
                    if name in getattr(env, "_exec_globals")
                })

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks_raw = _extract_code(new_code)
                new_metadata_raw = [
                    {"generation": num_regenerations + 1, "regenerated": True,
                     "regenerated_at_idx": code_block_idx}
                ] * len(new_blocks_raw)
                new_blocks, new_metadata, skipped_new_blocks = _filter_executable_code_blocks(
                    new_blocks_raw,
                    new_metadata_raw,
                )
                if skipped_new_blocks:
                    print(f"Skipped {skipped_new_blocks} non-executable regenerated code block(s)")
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "raw_code_blocks": new_blocks_raw,
                    "skipped_non_executable_blocks": skipped_new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(new_metadata)
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    if code_block_idx >= max_code_blocks and code_block_idx < len(code_blocks):
        print(
            "Reached max_multiturn_blocks="
            f"{max_code_blocks}; stopping this trial for bounded throughput."
        )

    print("Code blocks done")

    # --- 5. Build final summary ---
    executed_code_blocks = code_blocks[:code_block_idx]
    executed_code_block_metadata = code_block_metadata[:code_block_idx]
    unexecuted_code_block_count = max(0, len(code_blocks) - code_block_idx)
    final_code = _annotate_code_blocks(
        executed_code_blocks,
        executed_code_block_metadata,
    )
    num_code_blocks = len(executed_code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks
        partial_artifacts["unexecuted_code_block_count"] = unexecuted_code_block_count

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )
    _save_trial_trajectory(env, config, trial, info_step, reward)

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = (
        info_step["sandbox_rc"] == 0
        and bool(info_step.get("task_completed", False))
        and float(reward) >= 1.0
    )

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = _replace_named_prompt_slots(
            obs["full_prompt"][-1]["content"][0]["text"],
            {"libero_environment_goal": str(goal)},
        )


def _replace_named_prompt_slots(text: str, values: dict[str, Any]) -> str:
    """Replace only known prompt slots without formatting Python/JSON braces."""
    out = str(text)
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out
