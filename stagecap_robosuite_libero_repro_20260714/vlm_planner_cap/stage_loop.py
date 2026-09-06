"""Stage-level control utilities for Planner-Feedback-Replanner CaP.

This module is deliberately independent from a concrete LIBERO worker. It
defines the protocol that a stage executor must follow: generate code for one
stage, print structured evidence, verify that evidence, and decide whether to
proceed, repair locally, or replan.
"""

from __future__ import annotations

import json
import re
import ast
from dataclasses import replace
from typing import Any

from .schema import (
    CAPX_AGENT0_API_POLICY,
    DEFAULT_CLEAN_API_POLICY,
    SAM3_CLEAN_API_POLICY,
    CleanApiPolicy,
    StageDecision,
    StageExecutionRecord,
    StageLoopState,
    StagePlan,
    StageStatus,
    TaskPlan,
    VerificationEvidence,
)


STAGE_STATUS_RE = re.compile(r"STAGE_STATUS\s+(?P<payload>\{.*\})")


def build_stage_code_prompt(
    task: str,
    plan: TaskPlan,
    stage: StagePlan,
    *,
    policy: CleanApiPolicy = DEFAULT_CLEAN_API_POLICY,
    previous_records: list[StageExecutionRecord] | None = None,
    replan_context: dict[str, Any] | None = None,
    scene_snapshot_text: str | None = None,
    required_action_contracts: str | None = None,
) -> str:
    """Build the prompt for generating code for exactly one stage."""
    previous_records = previous_records or []
    return (
        "# Stage-Verifiable Zero-Shot CaP\n"
        f"Exact task: {task}\n"
        "Generate one executable Python code block for the CURRENT STAGE only.\n"
        "Do not solve by directly editing simulator state, object pose, joint state, reward, or task success.\n"
        "Readonly environment APIs may be used for grounding and verification; they are not actions.\n"
        "Planner-provided positions and diagnostics may be passed into motion code as target coordinates.\n"
        "\nAllowed robot action primitives:\n"
        + ", ".join(policy.allowed_primitives)
        + "\nAllowed readonly APIs:\n"
        + ", ".join(policy.allowed_observations)
        + "\nAllowed audited motion skills:\n"
        + ", ".join(policy.allowed_skills)
        + "\nForbidden shortcuts:\n"
        + "\n".join(f"- {symbol}" for symbol in policy.forbidden_symbols)
        + "\n\nFull stage machine:\n"
        + format_plan_for_prompt(plan)
        + "\nCURRENT STAGE:\n"
        + format_stage_for_prompt(stage, policy=policy)
        + format_required_action_contracts(required_action_contracts)
        + "\nExecution contract:\n"
        "- Execute only the current stage or a minimal local repair for it.\n"
        "- Code must be self-contained: define every helper you call, and do not rely on variables from examples.\n"
        "- Do not use placeholder names from examples such as `stage_id`, `evidence`, `repair`, `repair_msg`, `object_name`, `candidate_idx`, `target_pos`, `final_obj_pos`, or `relation_satisfied` unless you define them in this code block first. Prefer `STAGE_ID` for the current stage id and literal strings for evidence/repair messages.\n"
        "- Before using any variable in a final print, evidence dict, or status condition, assign it in the same code block. Do not reference ad-hoc names like `final_bowl_pose` after success unless you explicitly computed them; reuse already-defined variables such as `post_bowl_pose` or query the API again.\n"
        "- Once reward/task_completed or relation diagnostics already prove the current stage is satisfied, print the final `STAGE_STATUS` immediately; do not add extra summary prints that depend on new temporary variables.\n"
        "- Do not import project-specific helper modules such as robot_utils; use only bound APIs and standard Python/numpy.\n"
        "- Bound robot APIs are already available as global functions; call them directly instead of importing them.\n"
        "- `json`, `math`, `np`, and `time` are already available; do not import `sys` and do not call `sys.exit()`.\n"
        "- When printing structured evidence with `json.dumps`, always pass `default=str` so numpy arrays and diagnostics are serializable.\n"
        "- Use `get_task_status()` for reward/task_completed/current_done. Do not read `get_observation()[\"task_status\"]`; observations may not contain that key.\n"
        "- Never assign a local variable named `task_completed`; that name is reserved by the safety boundary. Use `completed = bool(get_task_status().get(\"task_completed\", False))` instead.\n"
        "- Use exact API function names from the allowed lists; do not invent visible_objects, robot, controller, or env helpers.\n"
        "- Clean runs forbid reflection/introspection helpers such as `globals()`, `locals()`, `eval`, `exec`, or direct environment access; use the provided variables and APIs directly.\n"
        "- Avoid wrapping the whole stage in `try/except`; incomplete exception blocks waste attempts. Prefer straight-line code with explicit checks and a final `STAGE_STATUS` line.\n"
        "- Do not write `try:` unless the complete matching `except` or `finally` block is included in the same code block. Prefer no `try/except` for stage code.\n"
        "- If you must stop immediately after printing a failure `STAGE_STATUS`, use bare `raise SystemExit` only after the print.\n"
        "- The runner provides `STAGE_ID` and `STAGE_ATTEMPT` globals. Use them read-only when printing status and when selecting bounded retry candidates; never write `STAGE_ID = ...` or `STAGE_ATTEMPT = ...` in generated code.\n"
        "- Print observations and geometry evidence before and after the action.\n"
        "- Never advance to a later high-risk stage if the current stage is not satisfied.\n"
        "- Never release a held object if transport/arrival is not verified.\n"
        "- For drawer stages, verify handle/contact/progress; gripper motion alone is not success.\n"
        "- End by printing exactly one JSON line of the form:\n"
        '  STAGE_STATUS {"stage_id": "...", "status": "satisfied|not_satisfied|needs_repair|uncertain", "evidence": "...", "repair": "..."}\n'
        "- Use this exact final print template, not bare `print(json.dumps(...))` and not `STAGE_STATUS:` commentary:\n"
        '  `print("STAGE_STATUS " + json.dumps({"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair}, default=str))`\n'
        + format_api_usage_rules(policy=policy)
        + format_scene_snapshot(scene_snapshot_text)
        + format_previous_records(previous_records)
        + format_replan_context(replan_context)
    )


def format_plan_for_prompt(plan: TaskPlan) -> str:
    lines: list[str] = []
    if plan.families:
        lines.append("Task families: " + ", ".join(family.value for family in plan.families))
    for idx, stage in enumerate(plan.stages, start=1):
        lines.append(f"{idx}. [{stage.stage_id}] {stage.name}")
        lines.append(f"   intent: {stage.intent}")
        if stage.family is not None:
            lines.append(f"   family: {stage.family.value}")
        lines.append(f"   max_attempts: {stage.max_attempts}")
        lines.append("   checks: " + ", ".join(check.name for check in stage.checks))
    if plan.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in plan.notes)
    return "\n".join(lines) + "\n"


def format_stage_for_prompt(stage: StagePlan, *, policy: CleanApiPolicy = DEFAULT_CLEAN_API_POLICY) -> str:
    allowed = set(policy.allowed_primitives) | set(policy.allowed_observations) | set(policy.allowed_skills)
    stage_primitives = [name for name in stage.allowed_primitives if name in allowed]
    stage_observations = [name for name in stage.allowed_observations if name in allowed]
    lines = [
        f"stage_id: {stage.stage_id}",
        f"name: {stage.name}",
        f"intent: {stage.intent}",
        f"family: {stage.family.value if stage.family is not None else 'unknown'}",
        f"max_attempts: {stage.max_attempts}",
        "target_references: " + ", ".join(stage.target_references),
        "allowed_primitives: " + ", ".join(stage_primitives),
        "observations_or_perception: " + ", ".join(stage_observations),
    ]
    if policy.profile == "robosuite_privileged_readonly":
        if stage_primitives:
            lines.append(
                "required_action_rule: call the current-stage audited action "
                f"({', '.join(stage_primitives)}); all other robot actions are forbidden"
            )
        else:
            lines.append(
                "required_action_rule: readonly grounding only; do not call robot actions"
            )
    lines.append("checks:")
    for check in stage.checks:
        lines.append(f"- {check.name} ({check.check_type}): {check.description}")
        for criterion in check.success_criteria:
            lines.append(f"  * {criterion}")
    if stage.repair_hints:
        lines.append("repair_hints:")
        lines.extend(f"- {hint}" for hint in stage.repair_hints)
    return "\n".join(lines) + "\n"


def format_required_action_contracts(contracts: str | None) -> str:
    if not contracts:
        return ""
    return (
        "\nBound signatures for the CURRENT STAGE action contract:\n"
        f"{contracts}\n"
        "Use these exact function names and signatures. For an action stage, call at least one "
        "listed audited action and do not replace it with lower-level motion primitives. "
        "Readonly observations may be used before and after the action.\n"
    )


def format_previous_records(records: list[StageExecutionRecord]) -> str:
    if not records:
        return ""
    lines = ["\nPrevious stage attempts:"]
    for record in records[-8:]:
        status = record.evidence.status.value if record.evidence else "unknown"
        rationale = record.evidence.rationale if record.evidence else ""
        repair_hint = (record.evidence.repair_hint if record.evidence else "") or ""
        logs = " | ".join(record.evidence.log_evidence or []) if record.evidence else ""
        stderr_tail = (record.stderr or "")[-300:]
        lines.append(
            f"- {record.stage_id} attempt={record.attempt} status={status} "
            f"forbidden={record.forbidden_symbols} rationale={rationale[:400]} "
            f"repair={repair_hint[:220]} stderr_tail={stderr_tail[:300]} "
            f"stage_status_log={logs[:300]}"
        )
    return "\n".join(lines) + "\n"


def format_replan_context(replan_context: dict[str, Any] | None) -> str:
    if not replan_context:
        return ""
    safe_text = json.dumps(replan_context, indent=2, ensure_ascii=False, default=str)
    return (
        "\nReplan/repair context from previous failure:\n"
        f"{safe_text}\n"
        "Use this evidence to change the local strategy; do not retrieve expert code.\n"
    )


def format_scene_snapshot(scene_snapshot_text: str | None) -> str:
    if not scene_snapshot_text:
        return ""
    return (
        "\nReadonly scene snapshot before this stage attempt:\n"
        f"{scene_snapshot_text}\n"
        "Treat this as grounding, not a shortcut. You may refresh any listed readonly API "
        "inside the code before moving, then act only through allowed robot primitives.\n"
    )


def format_api_usage_rules(*, policy: CleanApiPolicy = DEFAULT_CLEAN_API_POLICY) -> str:
    profile = getattr(policy, "profile", "default")
    if profile == "robosuite_privileged_readonly":
        return (
            "\nRoboSuite privileged-readonly Stage-CAP rules:\n"
            "- Exact simulator object poses, contacts, gripper state, reward, and task completion may be read for grounding and verification.\n"
            "- Privileged information is read-only: never access `env`, `APIS`, `sim`, qpos/qvel, or mutate object state, reward, or success.\n"
            "- SAM3 and point-prompt perception are not part of this profile. Use the bound task diagnostics and exact object/handle poses.\n"
            "- Prefer the audited skill named by the current stage. Each skill returns a dict with `success`, `recoverable`, `reason`, `metrics`, `evidence`, and `task_status`.\n"
            "- Treat `success=True` as stage satisfaction. On failure, report the returned reason and metrics as repair evidence; do not invent a replacement API.\n"
            "- Motion must still occur through the exposed Franka primitives or audited skills. Benchmark success is evidence, not an action.\n"
        )
    if profile == "capx_agent0":
        return (
            "\nCAP-X Agent0 matched API rules:\n"
            "- This run is non-privileged: do not call object-pose, BDDL, reward, task status, drawer-progress, or oracle-readonly helper APIs.\n"
            "- Use `get_observation()` to read RGB-D, camera intrinsics/extrinsics, wrist camera data, robot Cartesian pose, and joint state.\n"
            "- Observation images are nested: use `obs[\"agentview\"][\"images\"][\"rgb\"]`, `obs[\"agentview\"][\"images\"][\"depth\"]`, `obs[\"agentview\"][\"intrinsics\"]`, and `obs[\"agentview\"][\"pose_mat\"]`. For wrist view, use `obs[\"robot0_eye_in_hand\"][\"images\"][\"rgb\"]` and corresponding depth/intrinsics/pose. Never use `obs[\"rgb\"]` or `obs[\"depth\"]`.\n"
            "- Localize objects from RGB-D using `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, and/or `point_prompt_molmo`, then validate with depth.\n"
            "- `segment_sam3_text_prompt(rgb, text)` and `segment_sam3_point_prompt(rgb, (x, y))` return a list of mask candidates. Check `if not masks:` before use, then use `mask = masks[0][\"mask\"]`; do not write `masks[\"mask\"]`.\n"
            "- `point_prompt_molmo(image, text)` returns a dict from object query to `(x, y)` pixel coordinates. Validate `(x, y)` is not `(None, None)` before passing it to SAM3 point prompting.\n"
            "- Convert visual evidence into 3D using `mask_to_world_points`, `pixel_to_world_point`, `depth_to_point_cloud`, `transform_points`, and geometry helpers.\n"
            "- For grasping, use `plan_grasp` or `plan_grasp_from_point_clouds`; execute through `solve_ik`, `move_to_joints`, `goto_pose`, `open_gripper`, and `close_gripper`.\n"
            "- For motion planning with grasps, use `parse_grasp_poses_for_curobo`, `plan_grasp_trajectory`, `plan_with_grasped_object`, and `execute_joint_trajectory` when applicable.\n"
            "- Verify stage progress from new RGB-D observations, object masks/point clouds, robot pose, gripper state, and visible scene changes; print `STAGE_STATUS` with this visual/geometric evidence.\n"
            "- If a target or handle cannot be localized from vision/depth, print `needs_repair` with the missing evidence instead of inventing a world coordinate.\n"
        )
    if profile == "libero_privileged_sam3":
        return (
            "\nAPI usage rules that must be followed exactly:\n"
            "- SAM3 perception is enabled for this run.\n"
            "- You may localize objects from RGB-D using `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, and/or `point_prompt_molmo`, then validate with depth or readonly geometry.\n"
            "- `segment_sam3_text_prompt(rgb, text)` and `segment_sam3_point_prompt(rgb, (x, y))` return a list of mask candidates. Check `if not masks:` before use, then use `mask = masks[0][\"mask\"]`; do not write `masks[\"mask\"]`.\n"
            "- If point-prompt is enabled, `point_prompt_molmo(image, text)` returns a dict from object query to `(x, y)` pixel coordinates. Validate `(x, y)` is not `(None, None)` before passing it to SAM3 point prompting.\n"
            "- Convert visual evidence into 3D using `mask_to_world_points`, `pixel_to_world_point`, `depth_to_point_cloud`, `transform_points`, and geometry helpers.\n"
            "- For grasping, use audited skills or `plan_grasp` / `plan_grasp_from_point_clouds`; execute through robot motion primitives.\n"
            "- Verify stage progress from fresh observations, masks/point clouds, robot pose, gripper state, readonly diagnostics, and visible scene changes; print `STAGE_STATUS` with this visual/geometric evidence.\n"
            "- If a target or handle cannot be localized from vision/depth, print `needs_repair` with the missing evidence instead of inventing a world coordinate.\n"
        )
    return (
        "\nAPI usage rules that must be followed exactly:\n"
        "- SAM3 perception APIs are not available in this run. Do not call `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, or `point_prompt_molmo`; use readonly geometry diagnostics, task status, object poses, drawer APIs, and direct observations instead.\n"
        "- `get_object_pose(name)` and `get_ee_pose()` return `(position_xyz, quaternion_wxyz)` tuples; use `pos, quat = ...`.\n"
        "\nDrawer API contracts:\n"
        "- Drawer `level` arguments must be exactly one of the strings `\"top\"`, `\"middle\"`, or `\"bottom\"`. Do not pass object names, region names, drawer names, body names, or site names to drawer APIs.\n"
        "- `get_cabinet_drawer_handle_pose(level)` returns a tuple `(handle_position_xyz, hook_quaternion_wxyz, pull_direction_xyz)`. Correct: `handle_pos, hook_quat, pull_dir = get_cabinet_drawer_handle_pose(\"middle\")`. Never call `.get(...)` on this return value.\n"
        "- `get_cabinet_drawer_progress(level)` returns a dict. Use `progress[\"open_fraction\"]`, `progress[\"bddl_open\"]`, `progress[\"bddl_open_margin\"]`, `progress[\"bddl_close\"]`, and `progress[\"bddl_close_margin\"]`. There is no `progress[\"progress\"]` key.\n"
        "- `get_drawer_sleeve_hook_poses(level)` returns a dict with `hook_quaternion_wxyz`, `prehook_finger_position`, `mouth_finger_position`, and `sleeve_candidates`. Each sleeve candidate is a dict with `finger_midpoint_position` and `quaternion_wxyz`.\n"
        "- `measure_drawer_hook_alignment(level)` returns a dict with `ready_to_close`, `closed_on_handle`, `likely_hooked`, `finger_midpoint`, `finger_gap`, `depth_along_pull`, and `vertical_error`.\n"
        "- `measure_drawer_gripper_contacts(level)` returns a dict with `has_handle_contact` and contact counts.\n"
        "- Minimal drawer grounding example:\n"
        "  ```python\n"
        "  level = \"middle\"  # choose from current stage id/target_references: top, middle, or bottom\n"
        "  handle_pos, hook_quat, pull_dir = get_cabinet_drawer_handle_pose(level)\n"
        "  progress = get_cabinet_drawer_progress(level)\n"
        "  hooks = get_drawer_sleeve_hook_poses(level)\n"
        "  open_fraction = float(progress.get(\"open_fraction\", 0.0))\n"
        "  sleeve_candidates = hooks.get(\"sleeve_candidates\", [])\n"
        "  ```\n"
        "- Minimal drawer sleeve example:\n"
        "  ```python\n"
        "  level = \"middle\"\n"
        "  hooks = get_drawer_sleeve_hook_poses(level)\n"
        "  open_gripper()\n"
        "  goto_finger_midpoint_checked(hooks[\"prehook_finger_position\"], hooks[\"hook_quaternion_wxyz\"])\n"
        "  goto_finger_midpoint_checked(hooks[\"mouth_finger_position\"], hooks[\"hook_quaternion_wxyz\"])\n"
        "  cand = hooks[\"sleeve_candidates\"][0]\n"
        "  goto_finger_midpoint_checked(cand[\"finger_midpoint_position\"], cand[\"quaternion_wxyz\"])\n"
        "  close_gripper()\n"
        "  alignment = measure_drawer_hook_alignment(level)\n"
        "  ```\n"
        "- Minimal drawer progress test example:\n"
        "  ```python\n"
        "  alignment = measure_drawer_hook_alignment(level)\n"
        "  contacts = measure_drawer_gripper_contacts(level)\n"
        "  engaged = bool(alignment.get(\"closed_on_handle\") or alignment.get(\"likely_hooked\") or contacts.get(\"has_handle_contact\"))\n"
        "  if not engaged:\n"
        "      print('STAGE_STATUS {\"stage_id\": \"...\", \"status\": \"needs_repair\", \"evidence\": \"handle_engaged=False\", \"repair\": \"Return to sleeve/hook stage\"}')\n"
        "      raise SystemExit\n"
        "  before = get_cabinet_drawer_progress(level)\n"
        "  move_along_direction(pull_dir, 0.025, hook_quat)\n"
        "  after = get_cabinet_drawer_progress(level)\n"
        "  delta = float(after.get(\"open_fraction\", 0.0)) - float(before.get(\"open_fraction\", 0.0))\n"
        "  ```\n"
        "- Drawer `test_progress` stages must not perform a full handle recapture sequence. Treat the handle as engaged when any of `closed_on_handle`, `likely_hooked`, or `has_handle_contact` is true. If all three are false before the test pull, print `needs_repair` with evidence `handle_engaged=False`; the runner will return to the sleeve/hook stage.\n"
        "- Drawer `test_progress` stages should use only one short verified pull of about 0.02-0.03m, then compare `open_fraction` before/after. Do not keep moving the arm around the scene after the handle is lost.\n"
        "- `get_object_grasp_affordance(name)` returns a dict. Candidate grasps are in `affordance[\"candidates\"]`, not in `affordance[0]`; object dimensions are in `affordance[\"geometry\"][\"extent\"]` as `[x, y, z]`.\n"
        "- `get_goal_target_region_pose(object_name, predicate=\"in\")` takes the movable object name, not the receptacle/target name. Correct: `target = get_goal_target_region_pose(\"alphabet_soup\", predicate=\"in\")`; incorrect: `get_goal_target_region_pose(\"basket\")`.\n"
        "- The returned target dict contains `target_name` and `target_position`. Use `target[\"target_name\"]` for later transport/release skills when you need the exact BDDL region/site.\n"
        "- Target dicts from `get_goal_target_region_pose(...)` do not contain a `position` key. Use `target[\"target_position\"]`, never `target[\"position\"]`.\n"
        "- `get_site_in_diagnostic(object_name, site_name)` takes exactly two positional arguments: the movable object and a concrete site/region name such as `target[\"target_name\"]`. Do not pass `predicate=` and do not omit `site_name`.\n"
        "- Each grasp candidate is a dict with keys such as `pre_finger_position`, `outside_finger_position`, `mouth_finger_position`, `cross_finger_position`, `deep_finger_position`, and `quaternion_wxyz`.\n"
        "- In `pick_lift_verify` stages, the only pick action call should be `verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1)`. Do not call lower-level pick skills such as `verified_lift_pick_object`, `verified_low_side_pick_object`, `verified_low_profile_pick_object`, `verified_geometry_pick_object`, `grasp_tall_object_side`, or `approach_and_grasp_object` yourself.\n"
        "- The unified `verified_stage_pick_object` contract owns geometry inspection, cup/rim/tall-object strategy switching, candidate scheduling, lift-test auditing, and the final transport-stable hold check. Use its returned `held`, `transport_stable_held`, `final_grasp_state`, `final_contact_supported`, `object_name`, `lift_delta`, `skill`, and `stage_pick_contract` fields as the evidence.\n"
        "- If `grasp[\"held\"]` or `grasp[\"transport_stable_held\"]` is true and `grasp[\"lift_delta\"]` is at least about 0.05m, the pick-lift stage is satisfied; immediately print `STAGE_STATUS satisfied` and do not call extra lift, transport, release, `move_linear_delta`, `move_finger_midpoint_delta`, or `goto_*` afterward in the same pick-lift attempt.\n"
        "- In pick-lift stages, do not recompute success from `get_object_pose` differences such as `final_obj_pos[2] - initial_obj_pos[2]`; that can treat a pushed or fallen object as a lift. Trust the audited `grasp[\"held\"]`, `grasp[\"transport_stable_held\"]`, and `grasp[\"lift_delta\"]` fields.\n"
        "- If the unified pick contract returns `held=False` or insufficient lift, print `needs_repair`; the runner will retry the same stage with the next `STAGE_ATTEMPT`. Do not manually choose another lower-level pick skill in the generated code.\n"
        "- After a failed unified pick call, `measure_object_grasp_state(grasp[\"object_name\"])` may be printed as extra evidence. After a successful unified pick call, avoid extra probing motion because it can drop the object; trust the audited skill result as the primary success evidence.\n"
        "- For transport prechecks, `measure_object_grasp_state` may report `held=False` while `held_by_contact=True` or `elevated_contact=True` for contact-supported grasps. Treat the object as transportable if any of `held`, `strict_held`, `held_by_contact`, or `elevated_contact` is true, then call the audited transport skill and verify its `held_before`, `held_after`, and `arrived` fields.\n"
        "- For transport/release placement stages, choose `predicate` from the readonly scene snapshot BDDL goal when available, not only from natural language wording. For example, if the task says \"in the bowl\" but `bddl_goal_diagnostics.goal_state` shows raw predicate `on`, use `predicate=\"on\"` for `get_goal_target_region_pose`, transport, release, closure, and verification.\n"
        "- For transport stages, prefer `target = get_goal_target_region_pose(object_name, predicate=predicate)` followed by `guarded_transport_held_object_to_target(object_name, predicate=predicate, target_name=target[\"target_name\"])`; `predicate` and `target_name` must be keyword arguments.\n"
        "- `guarded_transport_held_object_to_target` accepts only `object_name`, `predicate`, `target_name`, `pre_release_z`, `xy_tolerance`, and `settle_steps`; never pass `target_position`, `target_pos`, or raw coordinates to it.\n"
        "- Transport stage canonical skeleton: resolve `object_name`, set `predicate`, call `target = get_goal_target_region_pose(object_name, predicate=predicate)`, then call exactly one `guarded_transport_held_object_to_target(object_name, predicate=predicate, target_name=target[\"target_name\"])`; verify `held_before`, `held_after`, `arrived`, `reward`, and task completion from `get_task_status()`, then print `STAGE_STATUS` using `json.dumps(..., default=str)`.\n"
        "- Transport-stage success is pre-release success, not final placement success. If the transport result has `held_after=True`, `arrived=True`, and `xy_distance_to_target` within `xy_tolerance` or a small target footprint tolerance, print `STAGE_STATUS satisfied` even when `reward=0`, `task_completed=False`, there is no contact, or the object is still high above the support. Large positive z distance above the target is expected before the release stage.\n"
        "- Do not use final `on`/`in` BDDL diagnostics, contact, low z height, or `task_completed=True` as required transport checks. The following `low_release_verify` stage owns descent, opening the gripper, retreat, and final relation verification.\n"
        "- Do not assign a local variable named `task_completed`; that name is reserved by the clean safety policy. Use `completed = bool(get_task_status().get(\"task_completed\", False))` when checking completion.\n"
        "- For release stages, prefer the semantic target from the CURRENT STAGE `target_references` when one is present (for example `cabinet_top`, `basket`, `plate`, `rack`, `front of stove`, or `microwave`). Still call `target = get_goal_target_region_pose(object_name, predicate=predicate)` for grounding, but pass the semantic target name to release/closure and use `target[\"target_name\"]` only as a fallback if no semantic target is listed.\n"
        "- For release stages, call `low_release_object_to_target(object_name, predicate=predicate, target_name=target_name)` where `target_name` is the semantic stage target or the grounded target site fallback; `predicate` and `target_name` must be keyword arguments. Valid optional release arguments are `release_z`, `approach`, `retreat`, `settle_steps`, and `require_drawer_closing_clearance`; `approach` and `retreat` are numeric distances, not strings such as \"linear\"; there is no `release_height_offset` argument.\n"
        "- For narrow support targets such as rack or caddy with predicate `on` or `in_or_on`, use `rack_supported_release_object_to_target(object_name, target_name=target_name)` instead of ordinary low release. Do not run local placement closure after a failed rack-supported release; report `needs_repair` so the loop can return to pick_lift if the object has fallen.\n"
        "- In a `low_release_verify` stage, do not import helper modules, do not call pick skills, and do not re-grasp the object. Release-stage repair should be bounded local placement cleanup, not a new pick-place episode.\n"
        "- Release stage canonical skeleton: resolve `object_name`, set `predicate`, set `semantic_target_name` from CURRENT STAGE `target_references` when available, call `target = get_goal_target_region_pose(object_name, predicate=predicate)`, set `target_name = semantic_target_name or target[\"target_name\"]`, then choose the audited release contract: use `rack_supported_release_object_to_target(object_name, target_name=target_name)` for rack/caddy `on` targets, otherwise call exactly one `low_release_object_to_target(object_name, predicate=predicate, target_name=target_name)`. If ordinary low release has `relation_satisfied=False`, call bounded `local_placement_closure(object_name, predicate=predicate, target_name=target_name, max_iters=2)` before reporting failure. Always print `STAGE_STATUS` using `json.dumps(..., default=str)`.\n"
        "- In multi-object tasks, an intermediate release stage is satisfied when the CURRENT object's `on`/`in` relation diagnostic is true after release and the object is no longer held, even if overall `task_completed` is still False because later objects remain. Do not require full-task reward/task_completed for an intermediate object's release stage.\n"
        "- Numpy arrays must not be used directly in boolean checks. Do not write `if target_position:` or `x = array_value or fallback`; use `is not None`, `len(...)`, `np.linalg.norm(...)`, `np.any(...)`, or explicit scalar conversions.\n"
        "- If the object slipped but is already near the target, call `local_placement_closure(object_name, predicate=predicate, target_name=target[\"target_name\"], max_iters=2)` before any long repair loop.\n"
        "- If release reports `relation_satisfied=False`, prefer bounded `local_placement_closure` with the same `predicate` and `target_name`; call `repair_placement_relation` only when the object is not already near the target or local closure fails.\n"
        "- `goto_pose_checked(position_xyz, quaternion_wxyz, ...)` requires both position and quaternion arguments.\n"
        "- `goto_finger_midpoint_checked(finger_midpoint_position, quaternion_wxyz, ...)` also requires both arguments.\n"
        "- `move_linear_delta(delta_xyz, quaternion_wxyz=None, ...)` and `move_along_direction(direction_xyz, distance, quaternion_wxyz=None, ...)` may keep the current orientation when quaternion is omitted.\n"
        "- Prefer exact instance names from the snapshot, for example `alphabet_soup_1`; if exact lookup fails, retry with the base object name such as `alphabet_soup`.\n"
    )


def parse_stage_status_events(text: str) -> list[VerificationEvidence]:
    """Parse `STAGE_STATUS {...}` lines into verification evidence."""
    evidences: list[VerificationEvidence] = []
    for line in text.splitlines():
        match = STAGE_STATUS_RE.search(line)
        if not match:
            continue
        payload_text = match.group("payload")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            try:
                payload = ast.literal_eval(payload_text)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        status = _coerce_stage_status(payload.get("status"))
        stage_id = str(payload.get("stage_id", "unknown"))
        evidences.append(
            VerificationEvidence(
                stage_id=stage_id,
                status=status,
                rationale=str(payload.get("evidence", "")),
                log_evidence=[line],
                repair_hint=str(payload.get("repair", "")) or None,
            )
        )
    return evidences


def choose_stage_decision(
    stage: StagePlan,
    evidence: VerificationEvidence | None,
    *,
    attempt_count: int,
    stderr: str = "",
    forbidden_symbols: list[str] | None = None,
) -> StageDecision:
    """Map verifier evidence to a control decision."""
    forbidden_symbols = forbidden_symbols or []
    if forbidden_symbols:
        return StageDecision.FAIL_TASK
    if stderr.strip() and not (
        evidence is not None and _stderr_is_only_system_exit(stderr, has_stage_status=True)
    ):
        return StageDecision.RETRY_STAGE if attempt_count < stage.max_attempts else StageDecision.REPLAN_STAGE
    if evidence is None:
        return StageDecision.RETRY_STAGE if attempt_count < stage.max_attempts else StageDecision.REPLAN_STAGE
    if evidence.status == StageStatus.SATISFIED:
        return StageDecision.PROCEED
    if evidence.status == StageStatus.UNCERTAIN:
        return StageDecision.RETRY_STAGE if attempt_count < stage.max_attempts else StageDecision.REPLAN_STAGE
    if evidence.status in {StageStatus.NOT_SATISFIED, StageStatus.NEEDS_REPAIR}:
        if _should_replan_suffix_immediately(stage, evidence):
            return StageDecision.REPLAN_SUFFIX
        if attempt_count < stage.max_attempts:
            return StageDecision.RETRY_STAGE
        stage_text = " ".join([stage.stage_id, stage.name, stage.intent]).lower()
        evidence_text = " ".join(
            [
                evidence.rationale or "",
                evidence.repair_hint or "",
                " ".join(evidence.log_evidence or []),
            ]
        ).lower()
        if (
            ("low_release_verify" in stage_text or "transport_guarded" in stage_text)
            and _release_failure_is_locally_repairable(evidence_text)
        ):
            return StageDecision.REPLAN_STAGE
        return StageDecision.REPLAN_SUFFIX if _failure_may_invalidate_suffix(stage) else StageDecision.REPLAN_STAGE
    return StageDecision.RETRY_STAGE


def append_execution_record(
    state: StageLoopState,
    record: StageExecutionRecord,
    decision: StageDecision,
) -> StageLoopState:
    """Return an updated loop state after applying a stage decision."""
    records = [*state.execution_records, record]
    repair_counts = dict(state.repair_counts)
    repair_counts[record.stage_id] = repair_counts.get(record.stage_id, 0) + (
        1 if decision in {StageDecision.RETRY_STAGE, StageDecision.REPLAN_STAGE, StageDecision.REPLAN_SUFFIX} else 0
    )
    next_index = state.current_stage_index
    if decision == StageDecision.PROCEED:
        next_index += 1
    elif decision == StageDecision.REPLAN_SUFFIX:
        fallback_index = _fallback_stage_index_for_suffix_replan(state, record)
        if fallback_index is not None:
            next_index = fallback_index
    return replace(
        state,
        current_stage_index=next_index,
        execution_records=records,
        repair_counts=repair_counts,
    )


def latest_stage_evidence(stdout: str, stderr: str = "") -> VerificationEvidence | None:
    """Return the last structured stage status, or a runtime-error evidence."""
    events = parse_stage_status_events(stdout)
    if stderr.strip() and not _stderr_is_only_system_exit(stderr, has_stage_status=bool(events)):
        return VerificationEvidence(
            stage_id="unknown",
            status=StageStatus.NEEDS_REPAIR,
            rationale="Execution produced stderr.",
            log_evidence=[stderr[-2000:]],
            repair_hint="Repair runtime/API/code error before retrying this stage.",
        )
    return events[-1] if events else None


def _stderr_is_only_system_exit(stderr: str, *, has_stage_status: bool) -> bool:
    if not has_stage_status:
        return False
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False
    allowed_prefixes = (
        "Traceback (most recent call last):",
        "File ",
    )
    for line in lines:
        if line == "SystemExit":
            continue
        if line.startswith(allowed_prefixes):
            continue
        if "_exec_user_code" in line or "exec(code" in line:
            continue
        return False
    return any(line == "SystemExit" for line in lines)


def _coerce_stage_status(value: Any) -> StageStatus:
    text = str(value or "").strip().lower()
    for status in StageStatus:
        if text == status.value:
            return status
    if text in {"success", "done", "passed", "pass"}:
        return StageStatus.SATISFIED
    if text in {"failed", "failure", "not satisfied"}:
        return StageStatus.NOT_SATISFIED
    return StageStatus.UNCERTAIN


def _failure_may_invalidate_suffix(stage: StagePlan) -> bool:
    text = " ".join([stage.stage_id, stage.name, stage.intent]).lower()
    return any(token in text for token in ("ground", "target", "drawer", "stove", "transport"))


def _should_replan_suffix_immediately(
    stage: StagePlan,
    evidence: VerificationEvidence,
) -> bool:
    stage_text = " ".join([stage.stage_id, stage.name, stage.intent]).lower()
    evidence_text = " ".join(
        [
            evidence.rationale or "",
            evidence.repair_hint or "",
            " ".join(evidence.log_evidence or []),
        ]
    ).lower()
    if "transport_guarded" in stage_text:
        if _drawer_transport_release_needs_repick(evidence_text):
            return True
        if "near_target_closure=true" in evidence_text or _release_failure_is_locally_repairable(evidence_text):
            return False
        if _transport_lost_physical_precondition(evidence_text):
            return True
        return any(
            token in evidence_text
            for token in (
                "held_before=false",
                "held_before': false",
                '"held_before": false',
                "held_after=false",
                "held_after': false",
                '"held_after": false',
                "not verified-held",
                "not preserve grasp",
            )
        )
    if "align_guarded" in stage_text:
        return bool(
            _transport_lost_physical_precondition(evidence_text)
            or "not verified-held" in evidence_text
        )
    if "low_release_verify" in stage_text:
        if _drawer_release_failure_needs_replan(evidence_text):
            return True
        if _placement_release_lost_physical_precondition(evidence_text):
            return True
        if _release_closure_exhausted(evidence_text):
            return True
        if _release_failure_is_locally_repairable(evidence_text):
            return False
        return any(
            token in evidence_text
            for token in (
                "not verified-held",
                "not close enough for local placement closure",
                "replan pick",
            )
        )
    if "pick_lift_verify" in stage_text and _drawer_pick_precondition_needs_reopen(evidence_text):
        return True
    return False


def _fallback_stage_index_for_suffix_replan(
    state: StageLoopState,
    record: StageExecutionRecord,
) -> int | None:
    stage_id = str(record.stage_id)
    evidence_text = ""
    if record.evidence is not None:
        evidence_text = " ".join(
            [
                record.evidence.rationale or "",
                record.evidence.repair_hint or "",
                " ".join(record.evidence.log_evidence or []),
            ]
        ).lower()
    if "transport_guarded" in stage_id and (
        "held_after=false" in evidence_text
        or "held_after': false" in evidence_text
        or '"held_after": false' in evidence_text
        or "held_before=false" in evidence_text
        or "held_before': false" in evidence_text
        or '"held_before": false' in evidence_text
        or "not preserve grasp" in evidence_text
        or "not verified-held" in evidence_text
        or _transport_lost_physical_precondition(evidence_text)
        or _drawer_transport_release_needs_repick(evidence_text)
    ):
        if (
            not _drawer_transport_release_needs_repick(evidence_text)
            and ("near_target_closure=true" in evidence_text or _release_failure_is_locally_repairable(evidence_text))
        ):
            return None
        prefix = stage_id.split("_transport_guarded", 1)[0]
        wanted = f"{prefix}_pick_lift_verify"
        for idx, stage in enumerate(state.plan.stages):
            if stage.stage_id == wanted:
                return idx
    if "align_guarded" in stage_id and (
        "not verified-held" in evidence_text
        or _transport_lost_physical_precondition(evidence_text)
    ):
        prefix = stage_id.split("_align_guarded", 1)[0]
        wanted = f"{prefix}_pick_lift_verify"
        for idx, stage in enumerate(state.plan.stages):
            if stage.stage_id == wanted:
                return idx
    if "low_release_verify" in stage_id and (
        "not verified-held" in evidence_text
        or "replan pick" in evidence_text
        or "not close enough for local placement closure" in evidence_text
        or _placement_release_lost_physical_precondition(evidence_text)
        or _release_closure_exhausted(evidence_text)
        or _drawer_release_failure_needs_replan(evidence_text)
    ):
        if _drawer_release_precondition_needs_reopen(evidence_text):
            drawer_index = _fallback_drawer_open_stage_index(state, evidence_text)
            if drawer_index is not None:
                return drawer_index
        if (
            _release_failure_is_locally_repairable(evidence_text)
            and not _placement_release_lost_physical_precondition(evidence_text)
            and not _release_closure_exhausted(evidence_text)
            and not _drawer_release_failure_needs_replan(evidence_text)
        ):
            return None
        prefix = stage_id.split("_low_release_verify", 1)[0]
        wanted = f"{prefix}_pick_lift_verify"
        for idx, stage in enumerate(state.plan.stages):
            if stage.stage_id == wanted:
                return idx
    if "pick_lift_verify" in stage_id and _drawer_pick_precondition_needs_reopen(evidence_text):
        drawer_index = _fallback_drawer_open_stage_index(state, evidence_text)
        if drawer_index is not None:
            return drawer_index
    if (
        "drawer_test_progress" in stage_id
        or "test_progress" in stage_id and "drawer" in stage_id
    ) and _drawer_test_progress_needs_recapture(evidence_text):
        drawer_prefix = stage_id.split("_test_progress", 1)[0]
        wanted = f"{drawer_prefix}_sleeve_or_hook_handle"
        for idx, stage in enumerate(state.plan.stages):
            if stage.stage_id == wanted:
                return idx
    return None


def _drawer_release_precondition_needs_reopen(evidence_text: str) -> bool:
    return bool(
        "drawer insertion precondition failed" in evidence_text
        or "drawer is closed or too shallow" in evidence_text
        or "drawer is not physically open enough" in evidence_text
        or "reopen/pull the drawer" in evidence_text
    )


def _drawer_pick_precondition_needs_reopen(evidence_text: str) -> bool:
    return bool(
        "drawer pre-pick open guard failed" in evidence_text
        or (
            "pre-pick open guard" in evidence_text
            and "open_fraction" in evidence_text
            and "required" in evidence_text
        )
    )


def _drawer_test_progress_needs_recapture(evidence_text: str) -> bool:
    return any(
        token in evidence_text
        for token in (
            "handle_engaged=false",
            "handle_engaged_before=false",
            "handle not engaged",
            "handle engagement failed",
            "handle engagement could not be established",
            "handle may not be properly engaged",
            "handle disengaged",
            "recapture handle",
            "recapture the handle",
            "closed_on_handle=false",
            "likely_hooked=false",
        )
    )


def _drawer_transport_release_needs_repick(evidence_text: str) -> bool:
    return bool(
        "drawer_immediate_release_failed=true" in evidence_text
        or "deeper held insertion" in evidence_text
        or "deeper held release" in evidence_text
        or "deeper held drawer release" in evidence_text
        or "re-grasp the bowl" in evidence_text
        or "regrasp the bowl" in evidence_text
        or "replan-deeper-held-release" in evidence_text
        or "front_clearance_not_improving_replan_deeper_held_release" in evidence_text
        or "drawer held insertion/release did not satisfy" in evidence_text
        or "object is not verified-held before drawer transport" in evidence_text
        or (
            "held_before=false" in evidence_text
            and ("drawer" in evidence_text or "cabinet" in evidence_text)
        )
    )


def _transport_lost_physical_precondition(evidence_text: str) -> bool:
    """Transport cannot continue locally once the object is no longer held."""
    return any(
        token in evidence_text
        for token in (
            "object not held before transport",
            "object is not held before transport",
            "not held before transport",
            "transport_held_before': false",
            '"transport_held_before": false',
            "transport_held_before=false",
            "transport_held_after': false",
            '"transport_held_after": false',
            "transport_held_after=false",
            "transport_stable_held_after': false",
            '"transport_stable_held_after": false',
            "transport_stable_held_after=false",
            "held_before': false",
            '"held_before": false',
            "held_before=false",
            "held_after': false",
            '"held_after": false',
            "held_after=false",
            "object not transportable",
            "not verified-held before transport",
        )
    )


def _placement_release_lost_physical_precondition(evidence_text: str) -> bool:
    """Release cannot continue locally once the object is unheld and not placed."""
    lost_hold = any(
        token in evidence_text
        for token in (
            "held=false",
            "held': false",
            '"held": false',
            "held_after=false",
            "held_after': false",
            '"held_after": false',
            "object_held': false",
            '"object_held": false',
            "object_held=false",
            "placement_relation_satisfied': false",
            '"placement_relation_satisfied": false',
            "gripper_released': true",
            '"gripper_released": true',
            "gripper_holding=false",
            "object not held",
            "object not transportable",
            "not verified-held",
            "object is not verified-held",
            "needs_repick': true",
            '"needs_repick": true',
            "needs_repick=true",
        )
    )
    placement_failed = any(
        token in evidence_text
        for token in (
            "relation_satisfied=false",
            "relation_satisfied': false",
            '"relation_satisfied": false',
            "placement_relation_satisfied': false",
            '"placement_relation_satisfied": false',
            "final_in_relation': false",
            '"final_in_relation": false',
            "post_in_relation': false",
            '"post_in_relation": false',
            "placement_not_satisfied",
            "relation not satisfied",
            "not satisfied",
        )
    )
    closure_failed = any(
        token in evidence_text
        for token in (
            "local_closure_result",
            "local placement closure",
            "containment_correction",
        )
    ) and placement_failed
    locally_near = _release_failure_is_locally_repairable(evidence_text)
    return bool(lost_hold and placement_failed and (not locally_near or closure_failed))


def _release_closure_exhausted(evidence_text: str) -> bool:
    """Release stage already tried bounded local closure but relation remains false."""
    placement_failed = any(
        token in evidence_text
        for token in (
            "relation_satisfied=false",
            "relation_satisfied': false",
            '"relation_satisfied": false',
            "placement_relation_satisfied': false",
            '"placement_relation_satisfied": false',
            "final_in_relation': false",
            '"final_in_relation": false',
            "post_in_relation': false",
            '"post_in_relation": false',
            "relation not satisfied",
        )
    )
    closure_attempted = any(
        token in evidence_text
        for token in (
            "after_closure=true",
            "after_closure': true",
            '"after_closure": true',
            "local_closure_result",
            "local placement closure",
            "containment_correction",
            "needs_repick': true",
            '"needs_repick": true',
            "needs_repick=true",
        )
    )
    return bool(placement_failed and closure_attempted)


def _drawer_release_failure_needs_replan(evidence_text: str) -> bool:
    return bool(
        _drawer_release_precondition_needs_reopen(evidence_text)
        or _drawer_transport_release_needs_repick(evidence_text)
        or "drawer insertion failed" in evidence_text
        or "held bowl insertion" in evidence_text
        or "deeper held release" in evidence_text
        or "deeper held drawer release" in evidence_text
        or "re-grasp the bowl" in evidence_text
        or "regrasp the bowl" in evidence_text
        or "replan-deeper-held-release" in evidence_text
        or "front_clearance_not_improving_replan_deeper_held_release" in evidence_text
        or "drawer_low_release_failed_preserve_state" in evidence_text
    )


def _fallback_drawer_open_stage_index(state: StageLoopState, evidence_text: str) -> int | None:
    level = None
    for candidate in ("bottom", "middle", "top"):
        if candidate in evidence_text:
            level = candidate
            break
    preferred_fragments = []
    if level is not None:
        preferred_fragments.extend(
            [
                f"{level}_drawer_pull_open",
                f"{level}_drawer_test_progress",
                f"{level}_drawer_sleeve_or_hook_handle",
            ]
        )
    preferred_fragments.extend(["_drawer_pull_open", "_drawer_test_progress", "_drawer_sleeve_or_hook_handle"])
    for fragment in preferred_fragments:
        for idx, stage in enumerate(state.plan.stages):
            if fragment in str(stage.stage_id):
                return idx
    return None


def _release_failure_is_locally_repairable(evidence_text: str) -> bool:
    """Detect release failures that should keep closing locally, not re-pick."""
    if (
        "insufficient_front_clearance_for_later_close" in evidence_text
        or "lacks close-time containment clearance" in evidence_text
        or (
            "require_drawer_closing_clearance=true" in evidence_text
            and (
                "has_clearance': false" in evidence_text
                or '"has_clearance": false' in evidence_text
            )
            and (
                "contact': true" in evidence_text
                or '"contact": true' in evidence_text
                or "local_delta" in evidence_text
                or "closure_relation" in evidence_text
            )
        )
    ):
        return True
    if _drawer_release_failure_needs_replan(evidence_text):
        return False
    if "microwave insertion failed" in evidence_text:
        return bool(
            "keep repair local to microwave insertion" in evidence_text
            or "microwave_relation" in evidence_text
            or "local_delta" in evidence_text
            or "contact': true" in evidence_text
            or '"contact": true' in evidence_text
        )
    if any(
        token in evidence_text
        for token in (
            "closure_near=true",
            "contact': true",
            '"contact": true',
            "local_delta",
            "containment_correction",
            "local placement closure",
        )
    ):
        if "not verified-held" not in evidence_text and "object is not verified-held" not in evidence_text:
            return True
    return False
