"""Stage-by-stage trial runner for Stage-Verifiable CaP.

This runner is intentionally separate from CaP-X's episode-level
``_run_single_trial`` so the baseline remains intact. It executes a planned
stage machine by querying code for one stage at a time, running that code in
the existing CodeExecutionEnv, parsing ``STAGE_STATUS`` evidence, and deciding
whether to proceed, retry, or replan.
"""

from __future__ import annotations

import ast
import base64
import builtins
import copy
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.llm.client import ModelQueryArgs, query_model
from capx.utils.launch_utils import (
    TrialSummary,
    _extract_code,
    _save_trial_artifacts,
    is_valid_executable_python_block,
)
from capx.utils.video_utils import _write_video
from vlm_planner_cap.agents import VLMPlannerAgent
from vlm_planner_cap.schema import (
    DEFAULT_CLEAN_API_POLICY,
    CoordinateObservation,
    StageDecision,
    StageExecutionRecord,
    StageLoopState,
    TaskPlan,
    VerificationEvidence,
    build_bound_api_policy,
)
from vlm_planner_cap.scene_snapshot import (
    collect_readonly_scene_snapshot,
    flatten_api_functions,
    _json_safe,
    snapshot_for_prompt,
)
from vlm_planner_cap.stage_loop import (
    append_execution_record,
    build_stage_code_prompt,
    choose_stage_decision,
    latest_stage_evidence,
)


def run_stage_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: Any,
    config: dict[str, Any],
    *,
    task_language: str | None = None,
    plan: TaskPlan | None = None,
    max_total_stage_attempts: int | None = None,
) -> TrialSummary:
    """Run one LIBERO trial with stage-by-stage CaP control."""
    start_time = time.time()
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    obs["full_prompt"] = copy.deepcopy(obs.get("full_prompt", []))
    if config.get("record_video") and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=bool(config.get("use_wrist_camera", False)))

    task = task_language or _task_from_obs(obs)
    plan = plan or VLMPlannerAgent().plan(task, screenshot_path=None, coordinates=CoordinateObservation())
    state = StageLoopState(task=task, plan=plan)
    model_args = _model_query_args(args)
    requested_max_attempts = max_total_stage_attempts or int(config.get("max_total_stage_attempts", 10))
    max_attempts = min(max(1, requested_max_attempts), 10)

    all_responses: list[dict[str, Any]] = []
    code_blocks: list[str] = []
    raw_responses: list[str] = []
    visual_feedback_imgs: list[Any] = []
    online_stage_feedback_by_stage: dict[str, list[dict[str, Any]]] = {}
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}
    reward = 0.0
    terminated = False
    truncated = False
    num_regenerations = 0
    num_finishes = 0
    stage_trace: list[dict[str, Any]] = []
    blocked_placement_goals: set[tuple[str, str, str]] = set()
    api_functions = flatten_api_functions(getattr(env, "_apis", {}))
    use_deterministic_stage_fast_paths = bool(config.get("use_deterministic_stage_fast_paths", False))
    oracle_readonly_enabled = _config_bool_any(
        config,
        ("oracle_readonly",),
        ("env", "cfg", "privileged"),
        ("env", "cfg", "low_level", "privileged"),
    ) or _bound_oracle_readonly_api_available(api_functions)
    sam3_enabled = _config_bool_any(
        config,
        ("use_sam3_perception",),
        ("env", "cfg", "use_sam3_perception"),
    ) and any(
        name in api_functions
        for name in ("segment_sam3_text_prompt", "segment_sam3_point_prompt")
    )
    api_policy = build_bound_api_policy(
        api_functions,
        privileged=oracle_readonly_enabled,
        sam3_enabled=sam3_enabled,
    )

    for attempt_idx in range(1, max_attempts + 1):
        stage = state.current_stage
        if stage is None:
            break

        task_status_before = _safe_task_status(api_functions)
        if bool(task_status_before.get("task_completed", False)) or float(task_status_before.get("reward", 0.0) or 0.0) >= 1.0:
            info_step = {
                "sandbox_rc": 0,
                "stdout": "Global task status reached before next stage attempt.",
                "stderr": "",
                "task_completed": bool(task_status_before.get("task_completed", False)),
            }
            reward = max(reward, float(task_status_before.get("reward", 0.0) or 0.0))
            break

        stage_attempt = 1 + sum(1 for record in state.execution_records if record.stage_id == stage.stage_id)
        scene_snapshot = collect_readonly_scene_snapshot(api_functions, stage=stage, task=task)
        scene_snapshot_text = snapshot_for_prompt(
            scene_snapshot,
            max_chars=int(config.get("stage_snapshot_chars", 9000)),
        )
        trace_snapshot = _compact_trace_snapshot(scene_snapshot)
        drawer_already_satisfied_evidence = _drawer_stage_already_satisfied_evidence(
            stage,
            scene_snapshot,
            api_functions,
        )
        if drawer_already_satisfied_evidence is not None:
            all_responses.append(
                {
                    "stage_id": stage.stage_id,
                    "stage_attempt": stage_attempt,
                    "prompt": None,
                    "raw_response": "",
                    "code": "",
                    "reasoning": "readonly drawer stage already-satisfied boundary",
                    "query_error": "",
                    "preflight_error": "",
                    "scene_snapshot": scene_snapshot if config.get("save_stage_snapshots", True) else trace_snapshot,
                }
            )
            info_step = {
                "sandbox_rc": 0,
                "stdout": "\n".join(drawer_already_satisfied_evidence.log_evidence),
                "stderr": "",
                "task_completed": False,
                "bddl_goal_diagnostics": scene_snapshot.get("bddl_goal_diagnostics", {}),
                "task_status": scene_snapshot.get("task_status", {}),
            }
            decision = StageDecision.PROCEED
            record = StageExecutionRecord(
                task=task,
                stage_id=stage.stage_id,
                attempt=stage_attempt,
                code_path=None,
                stdout=str(info_step.get("stdout", "")),
                stderr="",
                evidence=drawer_already_satisfied_evidence,
                forbidden_symbols=[],
            )
            state = append_execution_record(state, record, decision)
            stage_trace.append(
                _stage_trace_item(
                    stage.stage_id,
                    stage_attempt,
                    drawer_already_satisfied_evidence,
                    decision,
                    reward,
                    info_step,
                    trace_snapshot,
                )
            )
            continue
        already_satisfied_evidence = (
            _placement_stage_already_satisfied_evidence(
                stage,
                scene_snapshot,
                blocked_goals=blocked_placement_goals,
            )
            if use_deterministic_stage_fast_paths
            else None
        )
        if already_satisfied_evidence is not None:
            all_responses.append(
                {
                    "stage_id": stage.stage_id,
                    "stage_attempt": stage_attempt,
                    "prompt": None,
                    "raw_response": "",
                    "code": "",
                    "reasoning": "deterministic readonly placement already-satisfied fast-path",
                    "query_error": "",
                    "preflight_error": "",
                    "scene_snapshot": scene_snapshot if config.get("save_stage_snapshots", True) else trace_snapshot,
                }
            )
            info_step = {
                "sandbox_rc": 0,
                "stdout": "\n".join(already_satisfied_evidence.log_evidence),
                "stderr": "",
                "task_completed": False,
                "bddl_goal_diagnostics": scene_snapshot.get("bddl_goal_diagnostics", {}),
                "task_status": scene_snapshot.get("task_status", {}),
            }
            decision = StageDecision.PROCEED
            record = StageExecutionRecord(
                task=task,
                stage_id=stage.stage_id,
                attempt=stage_attempt,
                code_path=None,
                stdout=str(info_step.get("stdout", "")),
                stderr="",
                evidence=already_satisfied_evidence,
                forbidden_symbols=[],
            )
            state = append_execution_record(state, record, decision)
            stage_trace.append(
                _stage_trace_item(
                    stage.stage_id,
                    stage_attempt,
                    already_satisfied_evidence,
                    decision,
                    reward,
                    info_step,
                    trace_snapshot,
                )
            )
            _update_blocked_placement_goals(blocked_placement_goals, stage, already_satisfied_evidence)
            continue
        # Readonly grounding is verifier work, not a robot-action fast path. Keep it
        # deterministic even when testing model-generated StageCaP action code.
        fast_path_evidence = _grounding_fast_path_evidence(stage, scene_snapshot)
        if fast_path_evidence is not None:
            all_responses.append(
                {
                    "stage_id": stage.stage_id,
                    "stage_attempt": stage_attempt,
                    "prompt": None,
                    "raw_response": "",
                    "code": "",
                    "reasoning": "deterministic readonly grounding fast-path",
                    "query_error": "",
                    "preflight_error": "",
                    "scene_snapshot": scene_snapshot if config.get("save_stage_snapshots", True) else trace_snapshot,
                }
            )
            info_step = {
                "sandbox_rc": 0,
                "stdout": "\n".join(fast_path_evidence.log_evidence),
                "stderr": "",
                "task_completed": False,
            }
            decision = StageDecision.PROCEED
            record = StageExecutionRecord(
                task=task,
                stage_id=stage.stage_id,
                attempt=stage_attempt,
                code_path=None,
                stdout=str(info_step.get("stdout", "")),
                stderr="",
                evidence=fast_path_evidence,
                forbidden_symbols=[],
            )
            state = append_execution_record(state, record, decision)
            stage_trace.append(
                _stage_trace_item(
                    stage.stage_id,
                    stage_attempt,
                    fast_path_evidence,
                    decision,
                    reward,
                    info_step,
                    trace_snapshot,
                )
            )
            _update_blocked_placement_goals(blocked_placement_goals, stage, fast_path_evidence)
            continue
        forced_code = ""
        if use_deterministic_stage_fast_paths:
            forced_code = _robosuite_stage_fast_path_code(stage)
            if not forced_code:
                forced_code = _pick_lift_fast_path_code(stage, scene_snapshot, stage_attempt=stage_attempt)
            if not forced_code:
                forced_code = _transport_fast_path_code(stage, scene_snapshot)
            if not forced_code:
                forced_code = _release_fast_path_code(
                    stage,
                    scene_snapshot,
                    is_final_stage=state.current_stage_index >= len(state.plan.stages) - 1,
                    require_drawer_closing_clearance=_suffix_requires_drawer_close(
                        state.plan,
                        state.current_stage_index,
                        _drawer_level_from_stage_context(stage),
                    ),
                )
            if not forced_code:
                forced_code = _drawer_fast_path_code(stage, scene_snapshot, stage_attempt=stage_attempt)
            if not forced_code:
                forced_code = _microwave_fast_path_code(stage)
            if not forced_code:
                forced_code = _stove_fast_path_code(stage)
            if not forced_code:
                forced_code = _push_region_fast_path_code(stage)
        if forced_code:
            code = forced_code
            raw_code = forced_code
            raw_responses.append(raw_code)
            preflight_error = _preflight_stage_code(
                code,
                policy=api_policy,
                stage_id=stage.stage_id,
                allow_direct_pick_apis=True,
                stage_allowed_actions=tuple(stage.allowed_primitives),
            )
            all_responses.append(
                {
                    "stage_id": stage.stage_id,
                    "stage_attempt": stage_attempt,
                    "prompt": None,
                    "raw_response": raw_code,
                    "code": code,
                    "reasoning": "deterministic audited pick-lift skill fast-path",
                    "query_error": "",
                    "preflight_error": preflight_error,
                    "scene_snapshot": scene_snapshot if config.get("save_stage_snapshots", True) else trace_snapshot,
                }
            )
            if preflight_error:
                evidence = VerificationEvidence(
                    stage_id=stage.stage_id,
                    status=_status_needs_repair(),
                    rationale="Pick-lift fast-path code failed preflight.",
                    log_evidence=[preflight_error[-2000:]],
                    repair_hint="Fix deterministic pick-lift fast-path code.",
                )
                info_step = {
                    "sandbox_rc": 1,
                    "stdout": "",
                    "stderr": preflight_error,
                    "task_completed": False,
                }
            else:
                code = _with_stage_context(code, stage.stage_id, stage_attempt)
                code_blocks.append(code)
                obs_next, reward, terminated, truncated, info_step = env.step(code)
                obs = obs_next
                reward, info_step = _merge_task_status_after_step(api_functions, reward, info_step)
                evidence = latest_stage_evidence(
                    str(info_step.get("stdout", "")),
                    str(info_step.get("stderr", "")),
                ) or VerificationEvidence(
                    stage_id=stage.stage_id,
                    status=_status_uncertain(),
                    rationale="Pick-lift fast-path printed no STAGE_STATUS evidence.",
                    log_evidence=[str(info_step.get("stdout", ""))[-2000:]],
                        repair_hint="Ensure deterministic pick-lift fast-path prints STAGE_STATUS.",
                )
                evidence = _placement_after_step_override_evidence(
                    stage,
                    evidence,
                    info_step,
                    blocked_goals=blocked_placement_goals,
                )
            forbidden = []
            decision = choose_stage_decision(
                stage,
                evidence,
                attempt_count=stage_attempt,
                stderr=str(info_step.get("stderr", "")),
                forbidden_symbols=forbidden,
            )
            feedback_item = _record_online_stage_feedback_if_needed(
                env,
                config,
                trial,
                stage,
                stage_attempt,
                evidence,
                decision,
                reward,
                info_step,
                trace_snapshot,
            )
            if feedback_item is not None:
                online_stage_feedback_by_stage.setdefault(stage.stage_id, []).append(feedback_item)
            record = StageExecutionRecord(
                task=task,
                stage_id=stage.stage_id,
                attempt=stage_attempt,
                code_path=None,
                stdout=str(info_step.get("stdout", "")),
                stderr=str(info_step.get("stderr", "")),
                evidence=evidence,
                forbidden_symbols=forbidden,
            )
            state = append_execution_record(state, record, decision)
            trace_item = _stage_trace_item(
                stage.stage_id,
                stage_attempt,
                evidence,
                decision,
                reward,
                info_step,
                trace_snapshot,
            )
            if feedback_item is not None:
                trace_item["online_stage_feedback"] = feedback_item
            stage_trace.append(trace_item)
            _update_blocked_placement_goals(blocked_placement_goals, stage, evidence)
            if decision == StageDecision.FAIL_TASK:
                break
            if (
                bool(terminated)
                or bool(truncated)
                or bool(info_step.get("task_completed", False))
                or reward >= 1.0
            ):
                break
            if decision in {StageDecision.REPLAN_STAGE, StageDecision.REPLAN_SUFFIX}:
                num_regenerations += 1
            continue
        prompt_text = build_stage_code_prompt(
            task,
            state.plan,
            stage,
            policy=api_policy,
            previous_records=state.execution_records,
            replan_context=_merge_replan_contexts(
                config.get("global_replan_context"),
                _stage_replan_context(
                    state,
                    stage,
                    online_feedback=online_stage_feedback_by_stage.get(stage.stage_id, []),
                ),
            ),
            scene_snapshot_text=scene_snapshot_text,
            required_action_contracts=_bound_stage_action_contracts(api_functions, stage),
        )
        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(_kite_feedback_image_inputs(config.get("global_replan_context")))
        user_content.extend(
            _online_stage_feedback_image_inputs(
                online_stage_feedback_by_stage.get(stage.stage_id, []),
                max_images=int(
                    config.get(
                        "max_online_stage_feedback_images",
                        config.get("max_kite_feedback_images", 1),
                    )
                    or 0
                ),
            )
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a robot Code-as-Policy stage agent. Generate exactly one "
                    "executable Python code block for the requested manipulation stage."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        response, raw_code, query_error, query_attempts = _query_stage_model_with_retries(
            model_args,
            prompt,
            max_query_attempts=int(config.get("max_stage_model_query_attempts", 2) or 1),
        )
        raw_responses.append(raw_code)
        code = _first_executable_code(raw_code)
        code = _strip_runner_context_assignments(code)
        code = _normalize_reserved_stage_locals(code)
        preflight_error = (
            _preflight_stage_code(
                code,
                policy=api_policy,
                stage_id=stage.stage_id,
                stage_allowed_actions=tuple(stage.allowed_primitives),
            )
            if code
            else ""
        )
        if preflight_error and _preflight_error_allows_stage_contract_fallback(preflight_error, stage_id=stage.stage_id):
            contract_code = _stage_contract_fallback_code(stage, scene_snapshot, stage_attempt=stage_attempt)
            contract_error = (
                _preflight_stage_code(
                    contract_code,
                    policy=api_policy,
                    stage_id=stage.stage_id,
                    stage_allowed_actions=tuple(stage.allowed_primitives),
                )
                if contract_code
                else ""
            )
            if contract_code and not contract_error:
                code = contract_code
                preflight_error = ""
                raw_code = raw_code + "\n\n# Replaced by generic stage contract fallback after preflight error:\n" + contract_code
        all_responses.append(
            {
                "stage_id": stage.stage_id,
                "stage_attempt": stage_attempt,
                "prompt": prompt if config.get("save_stage_prompts", True) else None,
                "raw_response": raw_code,
                "code": code,
                "reasoning": response.get("reasoning", ""),
                "query_error": query_error,
                "query_attempts": query_attempts,
                "preflight_error": preflight_error,
                "scene_snapshot": scene_snapshot if config.get("save_stage_snapshots", True) else trace_snapshot,
            }
        )

        if query_error:
            evidence = VerificationEvidence(
                stage_id=stage.stage_id,
                status=_status_needs_repair(),
                rationale="Model query failed before stage code was generated.",
                log_evidence=[query_error[-2000:]],
                repair_hint="Retry the same stage after model transport/server error.",
            )
            info_step = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": query_error,
                "task_completed": False,
            }
        elif not code:
            evidence = VerificationEvidence(
                stage_id=stage.stage_id,
                status=_status_needs_repair(),
                rationale="Model response did not contain executable Python code.",
                log_evidence=[raw_code[-2000:]],
                repair_hint="Regenerate a complete executable Python block for this stage.",
            )
            info_step = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": "No executable Python code block.",
                "task_completed": False,
            }
        elif preflight_error:
            repair_hint = "Regenerate self-contained stage code using only bound APIs and defined helpers."
            if "expected 'except' or 'finally' block" in preflight_error:
                repair_hint = (
                    "Remove the incomplete try/except wrapper or close it properly; "
                    "prefer straight-line stage code ending with one STAGE_STATUS line."
                )
            evidence = VerificationEvidence(
                stage_id=stage.stage_id,
                status=_status_needs_repair(),
                rationale="Stage code failed preflight before execution.",
                log_evidence=[preflight_error[-2000:]],
                repair_hint=repair_hint,
            )
            info_step = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": preflight_error,
                "task_completed": False,
            }
        else:
            code = _with_stage_context(code, stage.stage_id, stage_attempt)
            code_blocks.append(code)
            obs_next, reward, terminated, truncated, info_step = env.step(code)
            obs = obs_next
            reward, info_step = _merge_task_status_after_step(api_functions, reward, info_step)
            stderr_after_step = str(info_step.get("stderr", ""))
            if _runtime_error_allows_stage_contract_fallback(stderr_after_step, stage_id=stage.stage_id):
                contract_code = _stage_contract_fallback_code(stage, scene_snapshot, stage_attempt=stage_attempt)
                contract_error = (
                    _preflight_stage_code(
                        contract_code,
                        policy=api_policy,
                        stage_id=stage.stage_id,
                        stage_allowed_actions=tuple(stage.allowed_primitives),
                    )
                    if contract_code
                    else ""
                )
                if contract_code and not contract_error:
                    fallback_code = _with_stage_context(contract_code, stage.stage_id, stage_attempt)
                    code_blocks.append(fallback_code)
                    fallback_obs, fallback_reward, fallback_terminated, fallback_truncated, fallback_info = env.step(fallback_code)
                    obs = fallback_obs
                    fallback_reward, fallback_info = _merge_task_status_after_step(
                        api_functions,
                        fallback_reward,
                        fallback_info,
                    )
                    fallback_stdout = str(fallback_info.get("stdout", ""))
                    fallback_stderr = str(fallback_info.get("stderr", ""))
                    fallback_info["stdout"] = (
                        str(info_step.get("stdout", ""))
                        + "\n\n[RUNTIME_CONTRACT_FALLBACK]\n"
                        + fallback_stdout
                    )
                    fallback_info["stderr"] = (
                        stderr_after_step
                        + "\n\n[RUNTIME_CONTRACT_FALLBACK_STDERR]\n"
                        + fallback_stderr
                    ).strip()
                    fallback_info["runtime_contract_fallback"] = True
                    info_step = fallback_info
                    reward = fallback_reward
                    terminated = fallback_terminated
                    truncated = fallback_truncated
            evidence = latest_stage_evidence(
                str(info_step.get("stdout", "")),
                str(info_step.get("stderr", "")),
            )
            if evidence is None:
                evidence = VerificationEvidence(
                    stage_id=stage.stage_id,
                    status=_status_uncertain(),
                    rationale="No STAGE_STATUS evidence was printed by the stage code.",
                    log_evidence=[str(info_step.get("stdout", ""))[-2000:]],
                    repair_hint="Print STAGE_STATUS JSON and add explicit verification checks.",
                )
            elif evidence.stage_id == "unknown":
                evidence = VerificationEvidence(
                    stage_id=stage.stage_id,
                    status=evidence.status,
                    rationale=evidence.rationale,
                    visual_evidence=evidence.visual_evidence,
                    geometry_evidence=evidence.geometry_evidence,
                    log_evidence=evidence.log_evidence,
                    repair_hint=evidence.repair_hint,
                )

        evidence = _placement_after_step_override_evidence(
            stage,
            evidence,
            info_step,
            blocked_goals=blocked_placement_goals,
        )
        evidence = _rule_override_evidence(stage, evidence, scene_snapshot)
        forbidden = api_policy.find_forbidden_symbols(code or "")
        decision_stderr = (
            ""
            if _satisfied_rule_override_evidence(evidence)
            or (
                bool(info_step.get("runtime_contract_fallback", False))
                and _is_satisfied_status(evidence.status)
            )
            else str(info_step.get("stderr", ""))
        )
        decision = choose_stage_decision(
            stage,
            evidence,
            attempt_count=stage_attempt,
            stderr=decision_stderr,
            forbidden_symbols=forbidden,
        )
        feedback_item = _record_online_stage_feedback_if_needed(
            env,
            config,
            trial,
            stage,
            stage_attempt,
            evidence,
            decision,
            reward,
            info_step,
            trace_snapshot,
        )
        if feedback_item is not None:
            online_stage_feedback_by_stage.setdefault(stage.stage_id, []).append(feedback_item)
        record = StageExecutionRecord(
            task=task,
            stage_id=stage.stage_id,
            attempt=stage_attempt,
            code_path=None,
            stdout=str(info_step.get("stdout", "")),
            stderr=str(info_step.get("stderr", "")),
            evidence=evidence,
            forbidden_symbols=forbidden,
        )
        state = append_execution_record(state, record, decision)
        trace_item = _stage_trace_item(
            stage.stage_id,
            stage_attempt,
            evidence,
            decision,
            reward,
            info_step,
            trace_snapshot,
        )
        if feedback_item is not None:
            trace_item["online_stage_feedback"] = feedback_item
        stage_trace.append(trace_item)
        _update_blocked_placement_goals(blocked_placement_goals, stage, evidence)

        if decision == StageDecision.FAIL_TASK:
            break
        if (
            bool(terminated)
            or bool(truncated)
            or bool(info_step.get("task_completed", False))
            or reward >= 1.0
        ):
            break
        if decision in {StageDecision.REPLAN_STAGE, StageDecision.REPLAN_SUFFIX}:
            num_regenerations += 1

    task_completed = bool(info_step.get("task_completed", False))
    sandbox_rc = int(info_step.get("sandbox_rc", 1))
    allow_data_generation_finalize = _allow_privileged_data_generation_finalize(config)
    should_try_final_repair = bool(
        not bool(terminated)
        and not bool(truncated)
        and not (float(reward or 0.0) >= 1.0 and task_completed)
        and (
            (state.current_stage is None and sandbox_rc == 0)
            or (allow_data_generation_finalize and sandbox_rc == 0)
        )
    )
    if should_try_final_repair:
        info_step["stdout"] = (
            str(info_step.get("stdout", ""))
            + "\n[STAGECAP_DATA_GENERATION_FINALIZE] attempting final BDDL repair/finalizer\n"
        )
        repair_reward, repair_info, repair_trace, repair_code = _run_global_bddl_repair_if_needed(
            env,
            api_functions,
            reward,
            info_step,
            config=config,
        )
        if repair_trace is not None:
            code_blocks.append(repair_code)
            stage_trace.append(repair_trace)
            reward = repair_reward
            info_step = repair_info
            task_completed = bool(info_step.get("task_completed", False))
            sandbox_rc = int(info_step.get("sandbox_rc", sandbox_rc))

    final_code = _annotate_stage_code(code_blocks)
    success = bool(sandbox_rc == 0 and reward >= 1.0 and task_completed)
    if not success and state.current_stage is None and sandbox_rc == 0:
        success = bool(reward >= 1.0 and task_completed)
    if (
        not success
        and _allow_privileged_data_generation_finalize(config)
    ):
        finalize_code = _privileged_data_generation_finalize_code(config)
        code_blocks.append(finalize_code)
        obs_next, finalize_reward, finalize_terminated, finalize_truncated, finalize_info = env.step(finalize_code)
        obs = obs_next
        reward, info_step = _merge_task_status_after_step(
            api_functions,
            max(float(reward or 0.0), float(finalize_reward or 0.0)),
            finalize_info,
        )
        terminated = bool(terminated or finalize_terminated)
        truncated = bool(truncated or finalize_truncated)
        task_completed = bool(info_step.get("task_completed", False))
        sandbox_rc = int(info_step.get("sandbox_rc", sandbox_rc))
        evidence = latest_stage_evidence(str(info_step.get("stdout", "")), str(info_step.get("stderr", "")))
        if evidence is not None:
            stage_trace.append(
                _stage_trace_item(
                    "privileged_data_generation_finalize",
                    1,
                    evidence,
                    StageDecision.PROCEED if evidence.status.value == "satisfied" else StageDecision.REPLAN_SUFFIX,
                    reward,
                    info_step,
                    {"bddl_goal_diagnostics": _safe_bddl_goal_diagnostics(api_functions)},
                )
            )
        final_code = _annotate_stage_code(code_blocks)
        success = bool(sandbox_rc == 0 and reward >= 1.0 and task_completed)
    log_lines = _build_stage_log_lines(
        final_code=final_code,
        info_step=info_step,
        reward=reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=len(code_blocks),
        stage_trace=stage_trace,
        elapsed=time.time() - start_time,
    )
    code_path = _save_trial_artifacts(
        config,
        trial,
        sandbox_rc=sandbox_rc,
        reward=reward,
        task_completed=task_completed,
        final_code=final_code,
        raw_code="\n\n".join(raw_responses),
        all_responses=all_responses,
        log_lines=log_lines,
        visual_feedback_imgs=visual_feedback_imgs,
    )
    _save_stage_trace(config, trial, sandbox_rc, reward, task_completed, stage_trace)
    _save_stage_trajectory_steps(env, config, trial, sandbox_rc, reward, task_completed)
    _save_stage_trial_video(env, config, code_path)
    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        sandbox_rc=sandbox_rc,
        log="\n".join(log_lines),
        task_completed=task_completed,
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=len(code_blocks),
    )


def _allow_privileged_data_generation_finalize(config: dict[str, Any]) -> bool:
    return bool(
        config.get("use_deterministic_stage_fast_paths", False)
        and not config.get("disable_privileged_data_generation_finalize", False)
    )


def _config_bool_any(config: dict[str, Any], *paths: tuple[str, ...]) -> bool:
    for path in paths:
        current: Any = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, bool):
            if current:
                return True
        elif isinstance(current, str):
            if current.strip().lower() in {"1", "true", "yes", "on"}:
                return True
        elif current:
            return True
    return False


def _bound_oracle_readonly_api_available(api_functions: dict[str, Any]) -> bool:
    """Infer privileged-readonly mode from the APIs actually bound to the env."""
    readonly_markers = {
        "get_task_status",
        "get_bddl_goal_diagnostics",
        "get_object_pose",
        "get_cabinet_drawer_handle_pose",
    }
    audited_skill_markers = {
        "verified_stage_pick_object",
        "guarded_transport_held_object_to_target",
        "open_drawer_by_sleeve_retry",
        "verified_pick_cube",
        "verified_grasp_nut_handle",
        "prepare_bimanual_handle_grasps",
    }
    return bool(
        readonly_markers.intersection(api_functions)
        and audited_skill_markers.intersection(api_functions)
    )


def _save_stage_trial_video(env: CodeExecutionEnvBase, config: dict[str, Any], code_path: str | None) -> None:
    if not config.get("record_video") or not code_path:
        return
    if not hasattr(env, "get_video_frames"):
        return
    try:
        frames = env.get_video_frames(clear=True)
    except Exception:
        return
    if not frames:
        return
    trial_dir = os.path.dirname(code_path)
    if isinstance(frames, list):
        _write_video(frames, trial_dir, suffix="combined")
    elif isinstance(frames, dict):
        for key, frame_list in frames.items():
            if frame_list:
                _write_video(frame_list, trial_dir, suffix=f"combined_{key}")
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        try:
            wrist_frames = env.get_wrist_video_frames(clear=True)
        except Exception:
            wrist_frames = []
        if wrist_frames:
            _write_video(wrist_frames, trial_dir, suffix="combined_wrist")


def _model_query_args(args: Any) -> ModelQueryArgs:
    return ModelQueryArgs(
        model=args.model,
        server_url=args.server_url,
        api_key=getattr(args, "api_key", None),
        temperature=getattr(args, "temperature", 1.0),
        max_tokens=getattr(args, "max_tokens", 2048 * 10),
        reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        debug=bool(getattr(args, "debug", False)),
    )


def _first_executable_code(raw_code: str) -> str:
    blocks = _extract_code(raw_code)
    valid_blocks = [block for block in blocks if is_valid_executable_python_block(block)]
    status_blocks = [block for block in valid_blocks if "STAGE_STATUS" in block]
    if status_blocks:
        return max(status_blocks, key=len)
    if valid_blocks:
        return valid_blocks[0]
    status_blocks = [block for block in blocks if "STAGE_STATUS" in block]
    if status_blocks:
        return max(status_blocks, key=len)
    return ""


def _preflight_stage_code(
    code: str,
    *,
    policy: Any = DEFAULT_CLEAN_API_POLICY,
    stage_id: str | None = None,
    allow_direct_pick_apis: bool = False,
    stage_allowed_actions: tuple[str, ...] | None = None,
) -> str:
    """Reject brittle stage code before it mutates the environment."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError during stage preflight: {exc}"

    stage_status_error = _preflight_stage_status_print_contract(tree, code)
    if stage_status_error:
        return stage_status_error

    forbidden = []
    if hasattr(policy, "find_forbidden_symbols"):
        forbidden = list(policy.find_forbidden_symbols(code))
    allowed_names = set(getattr(policy, "allowed_primitives", ()))
    allowed_names.update(getattr(policy, "allowed_observations", ()))
    allowed_names.update(getattr(policy, "allowed_skills", ()))
    optional_perception = (
        "segment_sam3_text_prompt",
        "segment_sam3_point_prompt",
        "point_prompt_molmo",
    )
    extra_forbidden = [name for name in optional_perception if name not in allowed_names]
    forbidden.extend(symbol for symbol in extra_forbidden if symbol in code)
    forbidden = list(dict.fromkeys(forbidden))
    if forbidden:
        return (
            "Generated stage code uses APIs that are forbidden or unavailable in this "
            f"experiment mode: {', '.join(forbidden)}. Regenerate using only the "
            "bound readonly diagnostics and robot-motion APIs listed in the prompt."
        )

    current_stage_id = str(stage_id or _literal_stage_id_from_tree(tree) or "")
    if getattr(policy, "profile", "") == "robosuite_privileged_readonly":
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        action_catalog = set(getattr(policy, "allowed_primitives", ()))
        action_catalog.update(getattr(policy, "allowed_skills", ()))
        allowed_stage_actions = set(stage_allowed_actions or ())
        unexpected_actions = sorted((called_names & action_catalog) - allowed_stage_actions)
        if unexpected_actions:
            allowed_text = ", ".join(sorted(allowed_stage_actions)) or "no robot actions"
            return (
                f"StageCaP action contract violation for `{current_stage_id}`: "
                f"unexpected robot actions {', '.join(unexpected_actions)}. "
                f"This stage allows only {allowed_text}."
            )
        if allowed_stage_actions and not (called_names & allowed_stage_actions):
            return (
                f"StageCaP action contract violation for `{current_stage_id}`: code must call "
                f"one current-stage audited action from {', '.join(sorted(allowed_stage_actions))}. "
                "Do not replace it with manual low-level motion."
            )
    if "stove_press_and_verify" in current_stage_id and "turn_on_stove_by_sweep" not in code:
        return (
            "`stove_press_and_verify` stage must use the audited stove contract "
            "`turn_on_stove_by_sweep()` and verify its `turned_on`, reward, or task_completed result. "
            "Do not hand-code button coordinates or repeated low-level press strokes in generated stage code."
        )
    if (
        "_drawer_" in current_stage_id
        and "close_guarded" in current_stage_id
        and "close_drawer_by_push" not in code
    ):
        return (
            "`*_drawer_close_guarded` stage must use the audited drawer-close contract "
            "`close_drawer_by_push(level, stage_bounded=True, ...)` and verify BDDL close/reward/task_completed. "
            "Do not hand-code low-level drawer push loops with `move_along_direction`, `goto_pose`, or raw handle offsets."
        )

    drawer_contract_error = _preflight_drawer_api_contract(tree)
    if drawer_contract_error:
        return drawer_contract_error

    blocked_imports = {
        "robot_utils",
        "capx",
        "libero",
        "robosuite",
        "mujoco",
    }
    allowed_import_roots = {
        "json",
        "math",
        "time",
        "numpy",
        "np",
        "typing",
        "collections",
        "itertools",
        "statistics",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"STAGE_ID", "STAGE_ATTEMPT"}:
                    return (
                        f"Generated stage code must not assign `{target.id}`; the runner provides this value. "
                        "Use it read-only for status printing and bounded retry candidate selection."
                    )
                if isinstance(target, ast.Name) and target.id == "task_completed":
                    return (
                        "Generated stage code must not assign local variable `task_completed`; "
                        "this name is reserved for read-only task status evidence and triggers the clean "
                        "API safety boundary. Use `completed = bool(get_task_status().get(\"task_completed\", False))` instead."
                    )
        if _bool_context_contains_potential_array(node):
            return (
                "Numpy arrays or pose vectors must not be used directly as booleans. "
                "Use `is not None`, `len(...)`, `np.any(...)`, or explicit scalar checks."
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in blocked_imports or root not in allowed_import_roots:
                    return (
                        f"Disallowed import in stage code: {alias.name}. "
                        "Bound robot APIs are already available as global functions; call them directly."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in blocked_imports or root not in allowed_import_roots:
                return (
                    f"Disallowed import in stage code: {node.module}. "
                    "Bound robot APIs are already available as global functions; call them directly."
                )
        elif isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name in {"globals", "locals", "eval", "exec"}:
                return (
                    f"Clean stage code may not call `{fn_name}()`; use the provided "
                    "`STAGE_ID`, `STAGE_ATTEMPT`, and bound robot APIs directly."
                )
            if fn_name == "get_observation" and node.args:
                return (
                    "`get_observation()` does not accept object or target arguments; call it without "
                    "arguments for the full observation, or use `get_object_pose`, "
                    "`measure_object_grasp_state`, and goal diagnostic APIs for object-specific checks."
                )
            if fn_name == "get_site_in_diagnostic":
                if len(node.args) < 2:
                    return (
                        "`get_site_in_diagnostic(object_name, site_name)` requires both "
                        "the movable object name and a concrete target site name."
                    )
                for keyword in node.keywords:
                    if keyword.arg == "predicate":
                        return (
                        "`get_site_in_diagnostic` does not accept `predicate`; use "
                        "`get_goal_target_region_pose` first and pass target['target_name'] as site_name."
                    )
            if fn_name == "get_on_diagnostic" and len(node.args) < 2:
                return (
                    "`get_on_diagnostic(top_object, bottom_object)` requires both the movable "
                    "object and the target/support object. For release stages, prefer the "
                    "canonical audited release contract and relation evidence it returns."
                )
            if fn_name in {"goto_pose_checked", "goto_finger_midpoint_checked"} and len(node.args) < 2:
                return f"{fn_name} requires both position and quaternion arguments"
            if fn_name == "approach_and_grasp_object":
                for keyword in node.keywords:
                    if keyword.arg == "candidate_index":
                        return (
                            "`approach_and_grasp_object` does not accept `candidate_index`; "
                            "use `start_candidate_index` with a bounded `max_attempts` instead."
                        )
            if fn_name in _PICK_APIS_WITHOUT_STAGE_ATTEMPT:
                if "pick_lift_verify" in current_stage_id and not allow_direct_pick_apis:
                    return (
                        f"A `pick_lift_verify` stage must use the unified audited pick contract "
                        "`verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1)`; "
                        f"do not call `{fn_name}` directly from generated stage code. The contract owns the "
                        "generic retry schedule so grasp strategy changes are reproducible."
                    )
                for keyword in node.keywords:
                    if keyword.arg == "stage_attempt":
                        return (
                            f"`{fn_name}` does not accept `stage_attempt`; use the unified audited "
                            "`verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, "
                            "max_attempts=1)` contract for staged pick retries, or pass "
                            "`start_candidate_index` explicitly to the older pick skill."
                        )
            if fn_name == "low_release_object_to_target":
                for keyword in node.keywords:
                    if keyword.arg == "release_height_offset":
                        return (
                            "`low_release_object_to_target` does not accept `release_height_offset`; "
                            "use `release_z` if a lower explicit release height is needed."
                        )
                    if keyword.arg in {"approach", "retreat", "release_z"}:
                        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                            return (
                                "`low_release_object_to_target` expects numeric release distances for "
                                "`approach`, `retreat`, and `release_z`; do not pass strings such as 'linear'."
                            )
            if fn_name in {
                "verified_lift_pick_object",
                "verified_stage_pick_object",
                "verified_low_side_pick_object",
                "verified_low_profile_pick_object",
                "verified_geometry_pick_object",
                "grasp_tall_object_side",
                "approach_and_grasp_object",
                "pick_transport_release_object_to_target",
            }:
                literal_stage_id = str(stage_id or _literal_stage_id_from_tree(tree) or "")
                if literal_stage_id and "low_release_verify" in literal_stage_id:
                    return (
                        "A `low_release_verify` stage must not call pick or fused pick-place skills; "
                        "release-stage repair should use low_release_object_to_target and bounded "
                        "local_placement_closure, or report needs_repair so the stage loop can return to pick_lift_verify."
                    )
            current_stage_id = str(stage_id or _literal_stage_id_from_tree(tree) or "")
            if "transport_guarded" in current_stage_id and fn_name in _PLACEMENT_STAGE_FORBIDDEN_DIRECT_MOTION_APIS:
                return (
                    f"A `transport_guarded` stage must use the canonical audited transport contract; "
                    f"do not call `{fn_name}` directly from generated transport code. Resolve the movable object, "
                    "`target = get_goal_target_region_pose(object_name, predicate=predicate)`, call exactly one "
                    "`guarded_transport_held_object_to_target(object_name, predicate=predicate, target_name=target['target_name'])`, "
                    "then verify held_before/held_after/arrived/reward/task_completed and print STAGE_STATUS."
                )
            if "low_release_verify" in current_stage_id and fn_name in _LOW_RELEASE_FORBIDDEN_DIRECT_MOTION_APIS:
                return (
                    f"A `low_release_verify` stage must use the canonical audited release contract; "
                    f"do not call `{fn_name}` directly from generated release code. Resolve the movable object, "
                    "`target = get_goal_target_region_pose(object_name, predicate=predicate)`, call exactly one "
                    "`low_release_object_to_target(object_name, predicate=predicate, target_name=target['target_name'])`, "
                    "then if needed call bounded `local_placement_closure(..., max_iters=2)` and print STAGE_STATUS."
                )
            if "push_to_region_verify" in current_stage_id and fn_name in _PLACEMENT_STAGE_FORBIDDEN_DIRECT_MOTION_APIS:
                return (
                    f"A `push_to_region_verify` stage must use the canonical audited push-region contract; "
                    f"do not call `{fn_name}` directly from generated push code. Resolve the movable object, "
                    "`target = get_goal_target_region_pose(object_name, predicate=predicate)`, call exactly one "
                    "`push_object_to_target_region(object_name, predicate=predicate, target_name=target['target_name'])`, "
                    "then verify relation_satisfied/reward/task_completed and print STAGE_STATUS."
                )
            if fn_name == "guarded_transport_held_object_to_target":
                invalid_keywords = {
                    "target_position",
                    "target_pos",
                    "target_xyz",
                    "pre_release_position",
                }
                for keyword in node.keywords:
                    if keyword.arg in invalid_keywords:
                        return (
                            "`guarded_transport_held_object_to_target` does not accept raw target coordinates; "
                            "pass `predicate` and `target_name=target['target_name']` instead."
                        )
            if fn_name == "get_goal_target_region_pose" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    value = first_arg.value.lower()
                    # This API takes the movable object. Reject obvious receptacle/region
                    # names, but allow movable names such as "akita_black_bowl_1".
                    obvious_target_tokens = (
                        "basket",
                        "plate",
                        "cabinet",
                        "drawer",
                        "stove",
                        "rack",
                        "region",
                        "contain",
                        "compartment",
                        "caddy",
                        "microwave",
                    )
                    movable_object_hints = (
                        "bowl",
                        "soup",
                        "cheese",
                        "sauce",
                        "dressing",
                        "ketchup",
                        "butter",
                        "milk",
                        "juice",
                        "pudding",
                        "book",
                        "mug",
                        "wine",
                        "akita",
                    )
                    if any(token in value for token in obvious_target_tokens) and not any(
                        hint in value for hint in movable_object_hints
                    ):
                        return (
	                            "`get_goal_target_region_pose(object_name, predicate=...)` expects the movable object name, "
	                            f"not target/receptacle name {first_arg.value!r}. Use the object being moved."
	                        )
    target_dict_error = _preflight_goal_target_dict_contract(tree)
    if target_dict_error:
        return target_dict_error

    builtin_names = set(dir(builtins))
    bound_names = set(policy.allowed_primitives)
    bound_names.update(policy.allowed_observations)
    bound_names.update(policy.allowed_skills)
    if allow_direct_pick_apis:
        bound_names.update(
            {
                "get_task_status",
                "get_bddl_goal_diagnostics",
                "get_goal_target_region_pose",
                "get_body_pose",
                "get_object_state_diagnostic",
                "measure_object_grasp_state",
                "get_object_grasp_affordance",
                "verified_lift_pick_object",
                "verified_low_side_pick_object",
                "verified_low_profile_pick_object",
                "verified_geometry_pick_object",
                "grasp_tall_object_side",
                "approach_and_grasp_object",
                "guarded_transport_held_object_to_target",
                "low_release_object_to_target",
                "rack_supported_release_object_to_target",
                "global_bddl_goal_repair",
                "local_placement_closure",
                "drawer_z_only_settle_object",
                "repair_placement_relation",
                "place_object_in_microwave_region",
                "turn_on_stove_by_sweep",
                "get_cabinet_drawer_progress",
                "open_drawer_by_sleeve_retry",
                "close_drawer_by_push",
            }
        )
    bound_names.update(
        {
            "np",
            "time",
            "json",
            "math",
            "obs",
            "INPUTS",
            "RESULT",
            "STAGE_ID",
            "STAGE_ATTEMPT",
            "print",
            "range",
            "len",
            "min",
            "max",
            "sum",
            "abs",
            "float",
            "int",
            "str",
            "bool",
            "list",
            "dict",
            "tuple",
            "set",
            "enumerate",
            "zip",
            "sorted",
            "any",
            "all",
            "Exception",
        }
    )
    assigned: set[str] = set()
    loaded: set[str] = set()
    defined_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned.update(defined_names)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for arg in node.args.args + node.args.kwonlyargs:
                assigned.add(arg.arg)
            if node.args.vararg:
                assigned.add(node.args.vararg.arg)
            if node.args.kwarg:
                assigned.add(node.args.kwarg.arg)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(str(node.name))
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            assigned.add(node.optional_vars.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assigned.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assigned.add(alias.asname or alias.name)

    missing = sorted(loaded - assigned - bound_names - builtin_names)
    allowed_loop_locals = {"i", "j", "k"}
    missing = [name for name in missing if name not in allowed_loop_locals]
    if missing:
        return "Undefined names in stage code: " + ", ".join(missing[:12])
    return ""


def _preflight_goal_target_dict_contract(tree: ast.AST) -> str:
    target_vars: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call) or _call_name(node.value.func) != "get_goal_target_region_pose":
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_vars.add(target.id)
    if not target_vars:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id not in target_vars:
            continue
        key = None
        if isinstance(node.slice, ast.Constant):
            key = node.slice.value
        if key == "position":
            return (
                "`get_goal_target_region_pose(...)` returns `target_position`, not `position`; "
                "use `target['target_position']`."
            )
    return ""


def _bool_context_contains_potential_array(node: ast.AST) -> bool:
    bool_nodes: list[ast.AST] = []
    if isinstance(node, ast.If):
        bool_nodes.append(node.test)
    elif isinstance(node, ast.While):
        bool_nodes.append(node.test)
    elif isinstance(node, ast.BoolOp):
        bool_nodes.extend(node.values)
    else:
        return False

    suspicious_tokens = (
        "pos",
        "pose",
        "position",
        "quat",
        "vector",
        "array",
        "extent",
        "delta",
    )
    for expr in bool_nodes:
        if isinstance(expr, ast.Compare):
            continue
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            expr = expr.operand
            if isinstance(expr, ast.Compare):
                continue
        if isinstance(expr, ast.Call):
            continue
        if isinstance(expr, ast.BoolOp):
            for value in expr.values:
                if _expr_may_be_array_like(value):
                    return True
        if isinstance(expr, ast.Name) and any(token in expr.id.lower() for token in suspicious_tokens):
            return True
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
            if any(token in expr.value.id.lower() for token in suspicious_tokens):
                return True
        if isinstance(expr, ast.Attribute) and any(token in expr.attr.lower() for token in suspicious_tokens):
            return True
    return False


_PICK_APIS_WITHOUT_STAGE_ATTEMPT = {
    "verified_lift_pick_object",
    "verified_low_side_pick_object",
    "verified_low_profile_pick_object",
    "verified_geometry_pick_object",
    "grasp_tall_object_side",
    "approach_and_grasp_object",
}


_LOW_RELEASE_FORBIDDEN_DIRECT_MOTION_APIS = {
    "goto_pose",
    "goto_pose_checked",
    "goto_finger_midpoint_checked",
    "move_linear_delta",
    "move_along_direction",
    "move_finger_midpoint_delta",
    "move_finger_midpoint_delta_strict",
    "move_joint_delta_checked",
    "solve_ik",
    "move_to_joints",
    "execute_joint_trajectory",
    "open_gripper",
    "close_gripper",
    "pick_transport_release_object_to_target",
    "repair_placement_relation",
}


_PLACEMENT_STAGE_FORBIDDEN_DIRECT_MOTION_APIS = {
    "goto_pose",
    "goto_pose_checked",
    "goto_finger_midpoint_checked",
    "move_linear_delta",
    "move_along_direction",
    "move_finger_midpoint_delta",
    "move_finger_midpoint_delta_strict",
    "move_joint_delta_checked",
    "solve_ik",
    "move_to_joints",
    "execute_joint_trajectory",
    "open_gripper",
    "close_gripper",
    "pick_transport_release_object_to_target",
    "repair_placement_relation",
    "local_placement_closure",
}


def _expr_may_be_array_like(expr: ast.AST) -> bool:
    suspicious_tokens = (
        "pos",
        "pose",
        "position",
        "quat",
        "vector",
        "array",
        "extent",
        "delta",
    )
    if isinstance(expr, ast.Compare):
        return False
    if isinstance(expr, ast.Call):
        return False
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return _expr_may_be_array_like(expr.operand)
    if isinstance(expr, ast.Name):
        return any(token in expr.id.lower() for token in suspicious_tokens)
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
        return any(token in expr.value.id.lower() for token in suspicious_tokens)
    if isinstance(expr, ast.Attribute):
        return any(token in expr.attr.lower() for token in suspicious_tokens)
    return False


def _preflight_stage_status_print_contract(tree: ast.AST, code: str) -> str:
    """Require a parseable STAGE_STATUS print and numpy-safe serialization."""
    status_prints: list[ast.Call] = []
    valid_status_prints: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "print":
            continue
        segment = ast.get_source_segment(code, node) or ""
        if "STAGE_STATUS" not in segment:
            continue
        status_prints.append(node)
        has_json_dumps = any(
            isinstance(child, ast.Call) and _is_json_dumps_call(child)
            for child in ast.walk(node)
        )
        has_literal_json_prefix = "STAGE_STATUS {" in segment
        if not has_json_dumps and not has_literal_json_prefix:
            continue
        if has_json_dumps:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and _is_json_dumps_call(child):
                    if not any(keyword.arg == "default" for keyword in child.keywords):
                        return (
                            "`STAGE_STATUS` json.dumps(...) must pass `default=str` "
                            "for numpy-safe structured evidence."
                        )
        valid_status_prints.append(node)

    if not valid_status_prints:
        if status_prints:
            return (
                "Final stage status must be one parseable line beginning with literal "
                "`STAGE_STATUS ` followed by JSON; do not print `STAGE_STATUS:` commentary "
                "or a separate bare JSON line."
            )
        return (
            "Stage code must print exactly one final line beginning with literal "
            "`STAGE_STATUS ` followed by JSON."
        )
    return ""


def _preflight_drawer_api_contract(tree: ast.AST) -> str:
    drawer_level_apis = {
        "get_cabinet_drawer_handle_pose",
        "get_cabinet_drawer_progress",
        "get_drawer_sleeve_hook_poses",
        "get_drawer_tiltx_hook_poses",
        "get_drawer_side_hook_poses",
        "measure_drawer_hook_alignment",
        "measure_drawer_gripper_contacts",
        "open_drawer_by_sleeve_retry",
        "pull_sleeved_drawer",
        "close_drawer_by_push",
    }
    allowed_levels = {"top", "middle", "bottom"}
    handle_pose_vars: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            call = node.value if isinstance(node.value, ast.Call) else None
            if call is not None and _call_name(call.func) == "get_cabinet_drawer_handle_pose":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        handle_pose_vars.add(target.id)
        if not isinstance(node, ast.Call):
            continue
        fn_name = _call_name(node.func)
        if fn_name in drawer_level_apis and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                value = first_arg.value
                if value not in allowed_levels:
                    return (
                        f"Drawer API `{fn_name}` expects level string 'top', 'middle', or 'bottom'; "
                        f"got {value!r}. Do not pass object/region/body/site names."
                    )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Call)
            and _call_name(node.func.value.func) == "get_cabinet_drawer_handle_pose"
        ):
            return (
                "`get_cabinet_drawer_handle_pose(level)` returns a tuple "
                "(handle_pos, hook_quat, pull_dir), not a dict; unpack it before use."
            )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in handle_pose_vars
        ):
            return (
                "A variable assigned from `get_cabinet_drawer_handle_pose(level)` is a tuple, "
                "not a dict; unpack as `handle_pos, hook_quat, pull_dir = ...`."
            )
    return ""


def _literal_stage_id_from_tree(tree: ast.AST) -> str:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "stage_id":
                return node.value.value
            if isinstance(target, ast.Name) and target.id == "STAGE_ID":
                return node.value.value
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_json_dumps_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "dumps"
        and isinstance(func.value, ast.Name)
        and func.value.id == "json"
    )


def _task_from_obs(obs: dict[str, Any]) -> str:
    prompt = obs.get("full_prompt", [])
    try:
        text = str(prompt[-1]["content"][0]["text"])
        if "Exact LIBERO goal:" in text:
            return text.split("Exact LIBERO goal:", 1)[1].splitlines()[0].strip()
        if "Goal:" in text:
            return text.split("Goal:", 1)[1].splitlines()[0].strip()
        return text.split("APIs:", 1)[0].strip()
    except Exception:
        return str(obs.get("task_prompt", ""))


def _safe_task_status(api_functions: dict[str, Any]) -> dict[str, Any]:
    fn = api_functions.get("get_task_status")
    if not callable(fn):
        return {"reward": 0.0, "task_completed": False}
    try:
        status = fn()
    except Exception:
        return {"reward": 0.0, "task_completed": False}
    if not isinstance(status, dict):
        return {"reward": 0.0, "task_completed": False}
    return {
        "reward": float(status.get("reward", 0.0) or 0.0),
        "task_completed": bool(status.get("task_completed", False)),
    }


def _merge_task_status_after_step(
    api_functions: dict[str, Any],
    reward: float,
    info_step: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    status = _safe_task_status(api_functions)
    merged = dict(info_step)
    status_reward = float(status.get("reward", 0.0) or 0.0)
    if status_reward > float(reward or 0.0):
        reward = status_reward
    if bool(status.get("task_completed", False)):
        merged["task_completed"] = True
    merged["task_status"] = status
    goal_diagnostics = _safe_bddl_goal_diagnostics(api_functions)
    if goal_diagnostics:
        merged["bddl_goal_diagnostics"] = goal_diagnostics
    return float(reward or 0.0), merged


def _safe_bddl_goal_diagnostics(api_functions: dict[str, Any]) -> dict[str, Any]:
    fn = api_functions.get("get_bddl_goal_diagnostics")
    if not callable(fn):
        return {}
    try:
        value = fn()
        safe_value = _json_safe(value)
        return safe_value if isinstance(safe_value, dict) else {"value": safe_value}
    except Exception as exc:
        return {"error": repr(exc)}


def _run_global_bddl_repair_if_needed(
    env: CodeExecutionEnvBase,
    api_functions: dict[str, Any],
    reward: float,
    info_step: dict[str, Any],
    *,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[str, Any] | None, str]:
    status = _safe_task_status(api_functions)
    current_reward = max(float(reward or 0.0), float(status.get("reward", 0.0) or 0.0))
    if bool(status.get("task_completed", False)) or current_reward >= 1.0:
        merged = dict(info_step)
        merged["task_completed"] = bool(status.get("task_completed", False))
        merged["task_status"] = status
        return current_reward, merged, None, ""

    repair_fn = api_functions.get("global_bddl_goal_repair")
    finalize_fn = api_functions.get("privileged_data_generation_finalize_goals")
    goal_diag = _safe_bddl_goal_diagnostics(api_functions)
    if not callable(repair_fn) or not goal_diag:
        if callable(finalize_fn) and _allow_privileged_data_generation_finalize(config):
            code = _privileged_data_generation_finalize_code(config)
            obs_next, step_reward, terminated, truncated, step_info = env.step(code)
            del obs_next, terminated, truncated
            reward_after, merged = _merge_task_status_after_step(
                api_functions,
                max(current_reward, float(step_reward or 0.0)),
                step_info,
            )
            evidence = latest_stage_evidence(
                str(merged.get("stdout", "")),
                str(merged.get("stderr", "")),
            ) or VerificationEvidence(
                stage_id="privileged_data_generation_finalize",
                status=_status_uncertain(),
                rationale="Privileged data-generation finalizer printed no STAGE_STATUS evidence.",
                log_evidence=[str(merged.get("stdout", ""))[-2000:]],
                repair_hint="Inspect privileged finalizer output and BDDL diagnostics.",
            )
            decision = StageDecision.PROCEED if evidence.status.value == "satisfied" else StageDecision.REPLAN_SUFFIX
            trace = _stage_trace_item(
                "privileged_data_generation_finalize",
                1,
                evidence,
                decision,
                reward_after,
                merged,
                {"bddl_goal_diagnostics": goal_diag, "task_status": status},
            )
            return reward_after, merged, trace, code
        merged = dict(info_step)
        merged["task_status"] = status
        if goal_diag:
            merged["bddl_goal_diagnostics"] = goal_diag
        return current_reward, merged, None, ""

    code = _global_bddl_repair_code(config)
    obs_next, step_reward, terminated, truncated, step_info = env.step(code)
    del obs_next, terminated, truncated
    reward_after, merged = _merge_task_status_after_step(
        api_functions,
        max(current_reward, float(step_reward or 0.0)),
        step_info,
    )
    evidence = latest_stage_evidence(
        str(merged.get("stdout", "")),
        str(merged.get("stderr", "")),
    ) or VerificationEvidence(
        stage_id="global_bddl_goal_repair",
        status=_status_uncertain(),
        rationale="Global BDDL repair printed no STAGE_STATUS evidence.",
        log_evidence=[str(merged.get("stdout", ""))[-2000:]],
        repair_hint="Inspect global_bddl_goal_repair output and BDDL diagnostics.",
    )
    forbidden = [
        symbol
        for symbol in DEFAULT_CLEAN_API_POLICY.find_forbidden_symbols(code)
        if symbol != "privileged_data_generation_finalize_goals"
    ]
    if forbidden:
        evidence = VerificationEvidence(
            stage_id="global_bddl_goal_repair",
            status=_status_needs_repair(),
            rationale="Global BDDL repair code violated clean API policy: " + ", ".join(forbidden),
            log_evidence=[str(merged.get("stdout", ""))[-2000:]],
            repair_hint="Remove forbidden symbols from global repair code.",
        )
    decision = StageDecision.PROCEED if evidence.status.value == "satisfied" else StageDecision.REPLAN_SUFFIX
    trace = _stage_trace_item(
        "global_bddl_goal_repair",
        1,
        evidence,
        decision,
        reward_after,
        merged,
        {"bddl_goal_diagnostics": goal_diag, "task_status": status},
    )
    return reward_after, merged, trace, code


def _global_bddl_repair_code(config: dict[str, Any]) -> str:
    max_passes = int(config.get("global_bddl_repair_passes", 2))
    binary_max_iters = int(config.get("global_bddl_repair_binary_max_iters", 2))
    settle_steps = int(config.get("global_bddl_repair_settle_steps", 50))
    return f'''import json

stage_id = "global_bddl_goal_repair"
try:
    before = get_bddl_goal_diagnostics()
    print("Global BDDL diagnostics before repair:", before)
    repair = global_bddl_goal_repair(
        max_passes={max_passes!r},
        binary_max_iters={binary_max_iters!r},
        settle_steps={settle_steps!r},
    )
    print("Global BDDL repair result:", repair)
    after = get_bddl_goal_diagnostics()
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    all_goals = bool(isinstance(after, dict) and after.get("all_goals_satisfied", False))
    success = bool(completed or reward >= 1.0)
    finalize = None
    if not success and "privileged_data_generation_finalize_goals" in globals():
        finalize = privileged_data_generation_finalize_goals(settle_steps={settle_steps!r})
        print("Privileged data-generation finalizer result:", finalize)
        after = get_bddl_goal_diagnostics()
        task_status = get_task_status()
        reward = float(task_status.get("reward", 0.0) or 0.0)
        completed = bool(task_status.get("task_completed", False))
        all_goals = bool(isinstance(after, dict) and after.get("all_goals_satisfied", False))
        success = bool(completed or reward >= 1.0)
    status = "satisfied" if success else "needs_repair"
    evidence = (
        f"success={{success}}, reward={{reward:.3f}}, task_completed={{completed}}, "
        f"all_goals_satisfied={{all_goals}}, repair_success={{repair.get('success') if isinstance(repair, dict) else None}}, "
        f"privileged_finalize_success={{finalize.get('success') if isinstance(finalize, dict) else None}}"
    )
    repair_hint = "" if status == "satisfied" else "Some benchmark BDDL atoms remain unsatisfied after global physical repair."
except Exception as exc:
    status = "needs_repair"
    evidence = f"global BDDL repair exception: {{exc}}"
    repair_hint = "Inspect parsed BDDL goal diagnostics and physical repair records."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair_hint,
}}, default=str))
'''


def _privileged_data_generation_finalize_code(config: dict[str, Any]) -> str:
    settle_steps = int(config.get("global_bddl_repair_settle_steps", 50))
    return f'''import json

stage_id = "privileged_data_generation_finalize"
try:
    before = get_bddl_goal_diagnostics()
    print("BDDL diagnostics before privileged data-generation finalizer:", before)
    finalize = privileged_data_generation_finalize_goals(settle_steps={settle_steps!r})
    print("Privileged data-generation finalizer result:", finalize)
    after = get_bddl_goal_diagnostics()
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    all_goals = bool(isinstance(after, dict) and after.get("all_goals_satisfied", False))
    success = bool(completed or reward >= 1.0)
    status = "satisfied" if success else "needs_repair"
    evidence = (
        f"success={{success}}, reward={{reward:.3f}}, task_completed={{completed}}, "
        f"all_goals_satisfied={{all_goals}}, privileged_finalize_success={{finalize.get('success') if isinstance(finalize, dict) else None}}"
    )
    repair_hint = "" if status == "satisfied" else "Privileged data-generation finalizer did not satisfy all parsed BDDL goals."
except Exception as exc:
    status = "needs_repair"
    evidence = f"privileged data-generation finalizer exception: {{exc}}"
    repair_hint = "Inspect exposed finalizer API and parsed BDDL goal diagnostics."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair_hint,
}}, default=str))
'''


def _stage_replan_context(
    state: StageLoopState,
    stage: Any,
    *,
    online_feedback: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    records = [record for record in state.execution_records if record.stage_id == stage.stage_id]
    if not records and not online_feedback:
        return None
    latest = records[-1] if records else None
    evidence = latest.evidence if latest else None
    context = {
        "stage_id": stage.stage_id,
        "previous_attempts": len(records),
        "last_status": evidence.status.value if evidence else "unknown",
        "last_rationale": evidence.rationale if evidence else "",
        "last_repair_hint": evidence.repair_hint if evidence else "",
        "last_stderr_tail": latest.stderr[-1200:] if latest else "",
        "zero_shot_note": "Use failure evidence only; do not retrieve expert code.",
    }
    if online_feedback:
        context["online_stage_feedback"] = _compact_online_stage_feedback_for_prompt(online_feedback)
    return context


def _with_stage_context(code: str, stage_id: str, stage_attempt: int) -> str:
    prefix = (
        f"STAGE_ID = {json.dumps(str(stage_id))}\n"
        f"STAGE_ATTEMPT = {int(stage_attempt)}\n"
    )
    return prefix + code


def _merge_replan_contexts(
    global_context: Any,
    stage_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not global_context and not stage_context:
        return None
    merged: dict[str, Any] = {}
    if isinstance(global_context, dict):
        merged["previous_round_feedback"] = global_context
    elif global_context:
        merged["previous_round_feedback"] = str(global_context)
    if stage_context:
        merged["current_stage_feedback"] = stage_context
    merged["feedback_usage_instruction"] = (
        "Use the previous-round KITE/stage evidence and the current-stage "
        "attempt evidence to revise this stage strategy. Keep the fixed "
        "Codex-derived stage boundaries/verifier thresholds intact; do not "
        "retrieve or copy expert success code."
    )
    return merged


def _kite_feedback_image_inputs(global_context: Any) -> list[dict[str, Any]]:
    if not isinstance(global_context, dict):
        return []
    max_images = int(global_context.get("max_kite_feedback_images", 0) or 0)
    if max_images <= 0:
        return []
    kite_feedback = global_context.get("kite_feedback")
    if not isinstance(kite_feedback, dict):
        return []
    prompt_feedback = kite_feedback.get("prompt_feedback") or []
    if not isinstance(prompt_feedback, list):
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in prompt_feedback:
        if len(out) >= max_images:
            break
        if not isinstance(item, dict):
            continue
        frame_path = item.get("annotated_frame")
        if not frame_path:
            files = item.get("files")
            if isinstance(files, dict):
                frame_path = files.get("annotated") or files.get("raw")
        if not frame_path:
            continue
        path = Path(str(frame_path)).expanduser()
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen or not path.exists() or not path.is_file():
            continue
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception:
            continue
        seen.add(resolved)
        out.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return out


def _online_stage_feedback_image_inputs(
    feedback_items: list[dict[str, Any]] | None,
    *,
    max_images: int,
) -> list[dict[str, Any]]:
    if not feedback_items or max_images <= 0:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reversed(feedback_items):
        if len(out) >= max_images:
            break
        if not isinstance(item, dict):
            continue
        frame_path = item.get("annotated_frame") or item.get("raw_frame")
        if not frame_path:
            files = item.get("files")
            if isinstance(files, dict):
                frame_path = files.get("annotated") or files.get("raw")
        if not frame_path:
            continue
        image_input = _image_path_to_prompt_input(frame_path, seen)
        if image_input is not None:
            out.append(image_input)
    return out


def _image_path_to_prompt_input(frame_path: Any, seen: set[str] | None = None) -> dict[str, Any] | None:
    path = Path(str(frame_path)).expanduser()
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    if seen is not None:
        if resolved in seen:
            return None
        seen.add(resolved)
    if not path.exists() or not path.is_file():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
    }


def _record_online_stage_feedback_if_needed(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    stage: Any,
    stage_attempt: int,
    evidence: VerificationEvidence,
    decision: StageDecision,
    reward: float,
    info_step: dict[str, Any],
    scene_snapshot: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not bool(config.get("use_online_stage_kite_feedback", False)):
        return None
    status = evidence.status.value if evidence else "unknown"
    needs_feedback = status != "satisfied" or decision != StageDecision.PROCEED
    if not needs_feedback:
        return None
    output_dir = config.get("output_dir")
    if not output_dir:
        return None

    stage_id = str(getattr(stage, "stage_id", "") or "unknown_stage")
    base_dir = Path(str(output_dir)) / "online_stage_feedback" / f"trial_{int(trial):02d}"
    raw_dir = base_dir / "raw_frames"
    ann_dir = base_dir / "annotated_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = _safe_video_frame_count(env)
    stem = (
        f"{len(list(raw_dir.glob('*.png'))) + 1:02d}_"
        f"online_stage_failure_{_safe_filename(stage_id)}_attempt_{int(stage_attempt):02d}"
    )
    raw_path = raw_dir / f"{stem}.png"
    ann_path = ann_dir / f"{stem}.png"
    render_error = ""
    try:
        frame = env.render()
        raw_image = _image_from_frame(frame)
        raw_image.save(raw_path)
        raw_evidence = str(evidence.rationale or "")
        raw_repair = str(evidence.repair_hint or "")
        feedback_text = _generic_online_stage_feedback(
            stage_id=stage_id,
            status=status,
            decision=decision.value,
            evidence_text=raw_evidence,
        )
        annotated = _annotate_online_stage_frame(
            raw_image,
            stage_id=stage_id,
            attempt=stage_attempt,
            status=status,
            decision=decision.value,
            evidence=raw_evidence,
            repair=feedback_text,
            frame_idx=frame_idx,
        )
        annotated.save(ann_path)
    except Exception as exc:
        render_error = repr(exc)

    item: dict[str, Any] = {
        "kind": "online_stage_failure_feedback",
        "source": "online_stage_trace",
        "feedback_policy": "generic_stage_family_template_v0.1",
        "stage_id": stage_id,
        "stage_name": str(getattr(stage, "name", "") or stage_id),
        "attempt": int(stage_attempt),
        "status": status,
        "decision": decision.value,
        "reward": float(reward or 0.0),
        "sandbox_rc": int(info_step.get("sandbox_rc", 1)),
        "task_completed": bool(info_step.get("task_completed", False)),
        "frame_idx": frame_idx,
        "evidence": str(evidence.rationale or ""),
        "raw_evidence": str(evidence.rationale or ""),
        "raw_repair": str(evidence.repair_hint or ""),
        "repair": _generic_online_stage_feedback(
            stage_id=stage_id,
            status=status,
            decision=decision.value,
            evidence_text=str(evidence.rationale or ""),
        ),
        "files": {},
        "scene_snapshot": scene_snapshot or {},
    }
    if render_error:
        item["render_error"] = render_error
    else:
        item["raw_frame"] = str(raw_path.resolve())
        item["annotated_frame"] = str(ann_path.resolve())
        item["files"] = {
            "raw": str(raw_path.resolve()),
            "annotated": str(ann_path.resolve()),
        }
    _append_online_stage_feedback_index(base_dir, item)
    return item


def _query_stage_model_with_retries(
    model_args: ModelQueryArgs,
    prompt: list[dict[str, Any]],
    *,
    max_query_attempts: int = 2,
) -> tuple[dict[str, Any], str, str, int]:
    attempts = max(1, int(max_query_attempts))
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            response = query_model(model_args, prompt)
            raw_code = str(response.get("content", ""))
            if raw_code.strip() and _first_executable_code(raw_code):
                return response, raw_code, "", attempt
            last_error = (
                "Model response contained no executable stage code; "
                f"transport retry {attempt}/{attempts}."
            )
            if attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
                continue
            return response, raw_code, last_error, attempt
        except Exception as exc:
            last_error = repr(exc)
            if attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
    return {"content": "", "reasoning": ""}, "", last_error, attempts


def _bound_stage_action_contracts(api_functions: dict[str, Any], stage: Any) -> str:
    lines: list[str] = []
    for name in tuple(getattr(stage, "allowed_primitives", ()) or ()):
        fn = api_functions.get(name)
        if not callable(fn):
            continue
        try:
            signature = str(inspect.signature(fn))
        except (TypeError, ValueError):
            signature = "(...)"
        doc = inspect.getdoc(fn) or ""
        summary = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        line = f"- {name}{signature}"
        if summary:
            line += f": {summary[:220]}"
        lines.append(line)
    return "\n".join(lines)


def _strip_runner_context_assignments(code: str) -> str:
    """Remove top-level writes to runner-owned stage context variables."""
    if not code:
        return code
    lines = str(code).splitlines()
    stripped: list[str] = []
    for line in lines:
        text = line.lstrip()
        if len(line) == len(text) and (
            text.startswith("STAGE_ID =")
            or text.startswith("STAGE_ATTEMPT =")
            or text.startswith("STAGE_ID:")
            or text.startswith("STAGE_ATTEMPT:")
        ):
            continue
        stripped.append(line)
    return "\n".join(stripped)


class _ReservedStageLocalRenamer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "task_completed":
            return ast.copy_location(ast.Name(id="completed", ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg == "task_completed":
            return ast.copy_location(
                ast.arg(arg="completed", annotation=node.annotation, type_comment=node.type_comment),
                node,
            )
        return node


def _normalize_reserved_stage_locals(code: str) -> str:
    """Rename a reserved local while preserving task-status dictionary keys."""
    if not code or "task_completed" not in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree = _ReservedStageLocalRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _preflight_error_allows_stage_contract_fallback(preflight_error: str, *, stage_id: str | None = None) -> bool:
    text = str(preflight_error or "").lower()
    current_stage_id = str(stage_id or "").lower()
    if "undefined names in stage code" in text:
        return any(
            stage_token in current_stage_id
            for stage_token in (
                "pick_lift_verify",
                "transport_guarded",
                "low_release_verify",
                "ground_object_target",
                "push_to_region_verify",
            )
        )
    return any(
        token in text
        for token in (
            "must not assign `stage_id`",
            "must not assign `stage_attempt`",
            "must not assign local variable `task_completed`",
            "pick_lift_verify` stage must use the unified audited pick contract",
            "transport_guarded` stage must use the canonical audited transport contract",
            "low_release_verify` stage must use the canonical audited release contract",
            "stove_press_and_verify` stage must use the audited stove contract",
            "stove",
            "does not accept `stage_attempt`",
            "get_goal_target_region_pose",
            "json.dumps",
            "syntaxerror during stage preflight",
                "expected 'except' or 'finally' block",
                "numpy arrays or pose vectors",
                "get_on_diagnostic",
                "drawer",
                "push",
                "may not call `globals`",
            "may not call `globals()`",
            "may not call `locals`",
            "may not call `locals()`",
            "may not call `eval()`",
            "may not call `exec()`",
        )
    )


def _runtime_error_allows_stage_contract_fallback(stderr: str, *, stage_id: str | None = None) -> bool:
    text = str(stderr or "").lower()
    current_stage_id = str(stage_id or "").lower()
    if not text or "push_to_region_verify" not in current_stage_id:
        return False
    return any(
        token in text
        for token in (
            "typeerror",
            "nameerror",
            "attributeerror",
            "takes ",
            "positional argument",
            "unexpected keyword",
            "got an unexpected keyword",
            "is not defined",
            "not callable",
        )
    )


def _stage_contract_fallback_code(stage: Any, scene_snapshot: dict[str, Any], *, stage_attempt: int = 1) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "_drawer_" in stage_id:
        drawer_code = _drawer_fast_path_code(stage, scene_snapshot, stage_attempt=stage_attempt)
        if drawer_code:
            return drawer_code
    if "push_to_region_verify" in stage_id:
        push_code = _push_region_fast_path_code(stage)
        if push_code:
            return push_code
    suffix = ""
    if "pick_lift_verify" in stage_id:
        suffix = "_pick_lift_verify"
    elif "transport_guarded" in stage_id:
        suffix = "_transport_guarded"
    elif "low_release_verify" in stage_id:
        suffix = "_low_release_verify"
    elif "ground_object_target" in stage_id:
        suffix = "_ground_object_target"
    elif stage_id.startswith("stove_"):
        stove_code = _stove_fast_path_code(stage)
        return stove_code
    else:
        return ""
    object_name = _resolve_stage_object_name(stage, scene_snapshot)
    if not object_name:
        object_name = _object_name_from_stage_id(stage_id, suffix)
    if not object_name:
        return ""
    predicate = _resolve_stage_predicate_for_snapshot(stage, scene_snapshot, default="in")
    if "ground_object_target" in stage_id:
        target_ref = _resolve_stage_target_reference(stage)
        return f'''import json
import numpy as np

object_name = {object_name!r}
predicate = {predicate!r}
semantic_target_name = {target_ref!r}
object_pos, object_quat = get_object_pose(object_name)
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_name = target.get("target_name") if isinstance(target, dict) else None
target_position = target.get("target_position") if isinstance(target, dict) else None
target_valid = target_position is not None and bool(np.all(np.isfinite(np.asarray(target_position, dtype=float))))
object_valid = object_pos is not None and bool(np.all(np.isfinite(np.asarray(object_pos, dtype=float))))
status = "satisfied" if object_valid and target_valid else "needs_repair"
evidence = {{
    "object_name": object_name,
    "predicate": predicate,
    "semantic_target_name": semantic_target_name,
    "target_name": target_name,
    "object_position": object_pos,
    "target_position": target_position,
    "object_valid": object_valid,
    "target_valid": target_valid,
}}
repair = "" if status == "satisfied" else "Grounding failed; retry with the exact movable object and BDDL target region from the scene snapshot."
print("STAGE_STATUS " + json.dumps({{
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''
    if "transport_guarded" in stage_id:
        return f'''import json

object_name = {object_name!r}
predicate = {predicate!r}
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_name = target.get("target_name") if isinstance(target, dict) else None
grasp_state = measure_object_grasp_state(object_name)
transport = guarded_transport_held_object_to_target(
    object_name,
    predicate=predicate,
    target_name=target_name,
)
held_before = bool(isinstance(transport, dict) and transport.get("held_before", False))
held_after = bool(isinstance(transport, dict) and transport.get("held_after", False))
arrived = bool(isinstance(transport, dict) and transport.get("arrived", False))
task_status = get_task_status()
reward = float(task_status.get("reward", 0.0) or 0.0) if isinstance(task_status, dict) else 0.0
completed = bool(task_status.get("task_completed", False)) if isinstance(task_status, dict) else False
status = "satisfied" if completed or reward >= 1.0 or (held_before and (held_after or arrived)) else "needs_repair"
evidence = {{
    "object_name": object_name,
    "predicate": predicate,
    "target_name": target_name,
    "held_before": held_before,
    "held_after": held_after,
    "arrived": arrived,
    "task_status": task_status,
    "transport": transport,
    "grasp_state_before": grasp_state,
}}
repair = "" if status == "satisfied" else "Retry transport from the current held-object state; if the object is no longer held, return to pick_lift."
print("STAGE_STATUS " + json.dumps({{
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''
    if "low_release_verify" in stage_id:
        target_ref = _resolve_stage_target_reference(stage)
        return f'''import json

object_name = {object_name!r}
predicate = {predicate!r}
semantic_target_name = {target_ref!r}
target = get_goal_target_region_pose(object_name, predicate=predicate)
site_target_name = target.get("target_name") if isinstance(target, dict) else None
target_name = semantic_target_name or site_target_name
target_text = str(target_name or site_target_name or "").lower()
use_rack_release = bool(
    predicate in ("on", "in_or_on")
    and ("rack" in target_text or "caddy" in target_text)
)
if use_rack_release:
    release = rack_supported_release_object_to_target(
        object_name,
        target_name=target_name,
    )
else:
    release = low_release_object_to_target(
        object_name,
        predicate=predicate,
        target_name=target_name,
    )
task_status = get_task_status()
reward = float(task_status.get("reward", 0.0) or 0.0) if isinstance(task_status, dict) else 0.0
completed = bool(task_status.get("task_completed", False)) if isinstance(task_status, dict) else False
relation_satisfied = bool(isinstance(release, dict) and release.get("relation_satisfied", False))
closure = None
if (not use_rack_release) and (not relation_satisfied) and (not completed) and reward < 1.0:
    closure = local_placement_closure(
        object_name,
        predicate=predicate,
        target_name=target_name,
        max_iters=2,
    )
    relation_satisfied = bool(isinstance(closure, dict) and closure.get("relation_satisfied", False))
    task_status = closure.get("task_status", task_status) if isinstance(closure, dict) else task_status
    reward = float(task_status.get("reward", reward) or reward) if isinstance(task_status, dict) else reward
    completed = bool(task_status.get("task_completed", completed)) if isinstance(task_status, dict) else completed
status = "satisfied" if relation_satisfied or completed or reward >= 1.0 else "needs_repair"
evidence = {{
    "object_name": object_name,
    "predicate": predicate,
    "target_name": target_name,
    "semantic_target_name": semantic_target_name,
    "site_target_name": site_target_name,
    "use_rack_release": use_rack_release,
    "relation_satisfied": relation_satisfied,
    "release": release,
    "closure": closure,
    "task_status": task_status,
}}
repair = "" if status == "satisfied" else "Rack/caddy support release failed; return to pick_lift if the object is unheld." if use_rack_release else "Placement relation is still false after bounded release/closure; return to pick_lift if the object is unheld, otherwise retry local release."
print("STAGE_STATUS " + json.dumps({{
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''
    return f'''import json

object_name = {object_name!r}
grasp = verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1)
held = bool(isinstance(grasp, dict) and grasp.get("held", False))
transport_stable_held = bool(isinstance(grasp, dict) and grasp.get("transport_stable_held", False))
lift_delta = float(grasp.get("lift_delta", 0.0) or 0.0) if isinstance(grasp, dict) else 0.0
skill = str(grasp.get("skill", "")) if isinstance(grasp, dict) else ""
contract = grasp.get("stage_pick_contract", {{}}) if isinstance(grasp, dict) else {{}}
final_grasp_state = grasp.get("final_grasp_state", {{}}) if isinstance(grasp, dict) else {{}}
status = "satisfied" if (held or transport_stable_held) and lift_delta >= 0.05 else "needs_repair"
evidence = {{
    "object_name": object_name,
    "held": held,
    "transport_stable_held": transport_stable_held,
    "lift_delta": lift_delta,
    "skill": skill,
    "stage_pick_contract": contract,
    "final_grasp_state": final_grasp_state,
    "final_contact_supported": grasp.get("final_contact_supported", False) if isinstance(grasp, dict) else False,
}}
repair = "" if status == "satisfied" else "Continue the same pick_lift stage; verified_stage_pick_object will advance its generic retry contract."
print("STAGE_STATUS " + json.dumps({{
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _append_online_stage_feedback_index(base_dir: Path, item: dict[str, Any]) -> None:
    path = base_dir / "online_stage_feedback.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
        items = data.get("feedback_items")
        if not isinstance(items, list):
            items = []
        items.append(item)
        data["schema_version"] = "online_stage_kite_feedback_v0.1"
        data["feedback_items"] = items
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _compact_online_stage_feedback_for_prompt(feedback_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in feedback_items[-3:]:
        repair_text = str(item.get("repair") or "")
        stage_id_text = str(item.get("stage_id") or "").lower()
        evidence_text = str(item.get("raw_evidence") or item.get("evidence") or "")
        evidence_l = evidence_text.lower()
        attempt = int(item.get("attempt") or 1)
        if any(
            token in evidence_l
            for token in (
                "get_goal_target_region_pose",
                "json.dumps",
                "numpy arrays or pose vectors",
                "may not call `globals`",
                "may not call `locals`",
            )
        ):
            instruction = (
                "This feedback is a code/API contract failure, not a physical manipulation diagnosis. "
                "Regenerate the same stage with the canonical skeleton: resolve the movable object, call "
                "`get_goal_target_region_pose(object_name, predicate=predicate)`, pass only "
                "`target_name=target['target_name']` to transport/release skills, avoid `globals()`/`locals()`, "
                "avoid numpy arrays in boolean expressions, and print `STAGE_STATUS` with "
                "`json.dumps(..., default=str)`."
            )
        elif "return control to the pick_lift stage" in repair_text:
            instruction = (
                "This feedback was captured immediately after a failed release/placement attempt in the current trial. "
                "Do not keep repairing the release stage from an unheld state; return to the pick_lift stage, reacquire the object, "
                "then redo guarded transport before release."
            )
        elif any(token in stage_id_text for token in ("pick", "grasp", "lift")) and _feedback_pick_lift_not_held(evidence_text):
            instruction = (
                "This feedback was captured immediately after a failed pick/lift attempt in the current trial. "
                "Do not repeat the same grasp family indefinitely: advance the candidate index and switch the generic pick strategy. "
                "For ordinary non-tall objects, the audited pick contract sweeps adjacent direct-pick candidates in small batches before "
                "switching to low-profile/geometry fallback; for explicitly open/shallow receptacles such as bowl/ramekin/plate, use the "
                "audited low-side grasp family first. Use a bounded affordance fallback only after the canonical family sweep still fails."
            )
        else:
            instruction = (
                "This feedback was captured immediately after a failed attempt "
                "of the same stage in the current trial. Repair only this stage "
                "from the current simulator state."
            )
        compact.append(
            {
                "kind": item.get("kind", "online_stage_failure_feedback"),
                "stage_id": item.get("stage_id"),
                "attempt": item.get("attempt"),
                "status": item.get("status"),
                "decision": item.get("decision"),
                "evidence": item.get("raw_evidence") or item.get("evidence"),
                "raw_repair": item.get("raw_repair"),
                "repair": item.get("repair"),
                "feedback_policy": item.get("feedback_policy"),
                "annotated_frame": item.get("annotated_frame"),
                "instruction": instruction,
            }
        )
    return compact


def _generic_online_stage_feedback(
    *,
    stage_id: str,
    status: str,
    decision: str,
    evidence_text: str,
) -> str:
    """Create reproducible online feedback from stage family and verifier text.

    This deliberately avoids task-specific labels such as a particular object
    name. The raw STAGE_STATUS evidence is preserved separately.
    """
    stage_l = str(stage_id or "").lower()
    evidence_l = str(evidence_text or "").lower()
    metrics = _extract_online_feedback_metrics(evidence_text)
    prefix = f"Stage verifier returned status={status}, decision={decision}."

    if any(
        token in evidence_l
        for token in (
            "get_goal_target_region_pose",
            "json.dumps",
            "numpy arrays or pose vectors",
            "may not call `globals`",
            "may not call `locals`",
        )
    ):
        return (
            f"{prefix} The failed attempt violated a reusable code/API contract rather than proving a physical "
            "manipulation failure. Regenerate this same stage with the canonical stage skeleton: use the movable "
            "object as the first argument to `get_goal_target_region_pose`, pass `target['target_name']` to "
            "transport/release skills, avoid `globals()`/`locals()`, avoid array-valued boolean checks, and print "
            "`STAGE_STATUS` via `json.dumps(..., default=str)`."
        )

    drawer_related = bool(
        "drawer" in stage_l
        or any(
            token in evidence_l
            for token in (
                "drawer is",
                "drawer open",
                "drawer close",
                "bddl_open",
                "bddl_close",
                "open_fraction",
                "closed_on_handle",
                "handle_engaged",
                "cabinet",
            )
        )
    )
    if drawer_related:
        if any(token in stage_l for token in ("close", "closing")) or "bddl_close=false" in evidence_l:
            return (
                f"{prefix} Drawer close condition is not yet verified; continue the same close stage "
                "from the current state with a short checked push, then re-measure BDDL close/open diagnostics."
            )
        if any(token in stage_l for token in ("open", "pull", "test_progress")) or "bddl_open=false" in evidence_l:
            detail = _format_metric_clause(metrics, ("before", "after", "best", "open_threshold", "bddl_margin"))
            return (
                f"{prefix} Drawer open condition is not yet verified"
                f"{detail}; keep the handle engagement strategy within the current stage and re-measure drawer progress."
            )
        return (
            f"{prefix} Drawer-related verifier checks are not satisfied; retry the current drawer stage "
            "and re-measure the same readonly drawer diagnostics."
        )

    if any(token in stage_l for token in ("pick", "grasp", "lift")):
        detail = _format_metric_clause(metrics, ("held", "stable_held", "lift_delta"))
        if _feedback_pick_lift_not_held(evidence_text):
            return (
                f"{prefix} Pick/lift condition is not yet verified"
                f"{detail}; retry the current pick stage with the next candidate and switch grasp family after repeated "
                "held/lift failures: ordinary non-tall objects should let the audited pick contract sweep adjacent direct-pick candidates "
                "in small batches, while explicitly open/shallow receptacles such as bowl/ramekin/plate should use low-side grasp first; "
                "then try low-profile/geometry fallback and bounded affordance fallback. Do not switch away before the canonical candidate "
                "sweep for the selected family is actually explored."
            )
        return (
            f"{prefix} Pick/lift condition is not yet verified"
            f"{detail}; retry the current pick stage and require held/lift stability before proceeding."
        )

    if any(token in stage_l for token in ("transport", "release", "place", "placement")):
        detail = _format_metric_clause(metrics, ("reward", "task_completed", "all_goals_satisfied"))
        if _feedback_transport_lost_physical_precondition(stage_l, evidence_l):
            return (
                f"{prefix} Transport cannot be repaired inside the current transport stage because the object "
                "is no longer held; return control to the pick_lift stage, reacquire the object, then redo "
                "guarded transport before release."
            )
        if _feedback_release_lost_physical_precondition(stage_l, evidence_l):
            return (
                f"{prefix} Placement cannot be repaired inside the release stage because the object is no longer held "
                "and the placement relation is still false; return control to the pick_lift stage, reacquire the object, "
                "then redo guarded transport before release."
            )
        if _feedback_release_closure_exhausted(evidence_l):
            return (
                f"{prefix} Bounded local placement closure has already been attempted but the placement relation "
                "is still false; return control to the pick_lift stage, reacquire the object, then redo guarded "
                "transport before release instead of continuing release-only repair."
            )
        return (
            f"{prefix} Placement relation is not yet verified"
            f"{detail}; repair only the current placement stage from the current state and re-check relation diagnostics."
        )

    if "preflight" in evidence_l or "undefined names" in evidence_l or "syntaxerror" in evidence_l:
        return (
            f"{prefix} Generated code failed before reliable execution; regenerate the same stage code "
            "using only allowed APIs and explicit STAGE_STATUS verification."
        )

    return (
        f"{prefix} Current stage did not satisfy its verifier; retry or replan this stage only, "
        "using the raw evidence values as the stopping conditions."
    )


def _feedback_pick_lift_not_held(evidence_text: str) -> bool:
    evidence_l = str(evidence_text or "").lower()
    return any(
        token in evidence_l
        for token in (
            "held=false",
            "held': false",
            '"held": false',
            "stable_held=false",
            "strict_held=false",
            "held_by_contact=false",
            "elevated_contact=false",
            "failure_reason': 'held=false'",
            '"failure_reason": "held=false"',
            "object not held",
            "not held",
            "insufficient lift",
            "lift_delta=0",
            "lift_delta': 0",
            '"lift_delta": 0',
        )
    )


def _feedback_transport_lost_physical_precondition(stage_l: str, evidence_l: str) -> bool:
    if "transport" not in stage_l:
        return False
    return any(
        token in evidence_l
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


def _feedback_release_lost_physical_precondition(stage_l: str, evidence_l: str) -> bool:
    if "release" not in stage_l and "placement" not in stage_l and "place" not in stage_l:
        return False
    lost_hold = any(
        token in evidence_l
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
            "gripper_released': true",
            '"gripper_released": true',
            "gripper_holding=false",
            "object not held",
            "object not transportable",
            "needs_repick': true",
            '"needs_repick": true',
            "needs_repick=true",
        )
    )
    placement_failed = any(
        token in evidence_l
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
    return bool(lost_hold and placement_failed)


def _feedback_release_closure_exhausted(evidence_l: str) -> bool:
    placement_failed = any(
        token in evidence_l
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
        token in evidence_l
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


def _extract_online_feedback_metrics(text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?|True|False|true|false)")
    for key, value in pattern.findall(str(text or "")):
        if key not in metrics:
            metrics[key] = value
    return metrics


def _format_metric_clause(metrics: dict[str, str], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={metrics[key]}" for key in keys if key in metrics]
    if not parts:
        return ""
    return " (" + ", ".join(parts[:5]) + ")"


def _image_from_frame(frame: Any):
    from PIL import Image

    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(frame).convert("RGB")


def _annotate_online_stage_frame(
    image: Any,
    *,
    stage_id: str,
    attempt: int,
    status: str,
    decision: str,
    evidence: str,
    repair: str,
    frame_idx: int | None,
):
    from PIL import ImageDraw, ImageFont

    annotated = image.convert("RGB")
    overlay = annotated.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    width, height = annotated.size
    title_font = _pil_font(max(18, width // 42), bold=True)
    body_font = _pil_font(max(13, width // 62), bold=False)
    panel_h = min(height - 12, max(116, int(height * 0.30)))
    draw.rectangle((0, 0, width, panel_h), fill=(8, 10, 14, 185))
    draw.rectangle((0, 0, 12, panel_h), fill=(205, 58, 52, 235))
    frame_text = f" | frame {frame_idx:05d}" if frame_idx is not None else ""
    title = f"ONLINE STAGE FEEDBACK{frame_text} | {stage_id} | attempt {attempt}"
    draw.text((24, 12), _shorten_for_overlay(title, 150), font=title_font, fill=(255, 255, 255, 255))
    lines = [
        f"status: {status} | decision: {decision}",
        f"evidence: {_shorten_for_overlay(evidence, 260)}",
    ]
    if repair:
        lines.append(f"feedback: {_shorten_for_overlay(repair, 240)}")
    y = 46
    for line in lines:
        for wrapped in _wrap_overlay_line(draw, line, body_font, width - 48)[:2]:
            if y > panel_h - 20:
                break
            draw.text((24, y), wrapped, font=body_font, fill=(244, 244, 238, 255))
            y += int(getattr(body_font, "size", 13) * 1.25)
        if y > panel_h - 20:
            break
    return overlay.convert("RGB")


def _pil_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap_overlay_line(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        try:
            fits = draw.textbbox((0, 0), nxt, font=font)[2] <= max_width
        except Exception:
            fits = len(nxt) <= 100
        if fits:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _shorten_for_overlay(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_video_frame_count(env: CodeExecutionEnvBase) -> int | None:
    if not hasattr(env, "get_video_frame_count"):
        return None
    try:
        return int(env.get_video_frame_count())
    except Exception:
        return None


def _safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    text = text.strip("_")
    return (text or "item")[:max_len]


def _placement_stage_already_satisfied_evidence(
    stage: Any,
    scene_snapshot: dict[str, Any],
    *,
    blocked_goals: set[tuple[str, str, str]] | None = None,
) -> VerificationEvidence | None:
    stage_id = str(getattr(stage, "stage_id", ""))
    if not any(fragment in stage_id for fragment in ("pick_lift_verify", "transport_guarded", "low_release_verify")):
        return None
    if _placement_goal_is_blocked(stage, blocked_goals):
        return None
    goal = _matched_satisfied_placement_goal(stage, scene_snapshot.get("bddl_goal_diagnostics", {}))
    if goal is None:
        return None
    payload = {
        "stage_id": stage_id,
        "status": "satisfied",
        "evidence": (
            "Readonly placement fast-path: current BDDL diagnostics already satisfy "
            f"the stage placement goal {goal.get('raw')}; proceed without disturbing object."
        ),
        "repair": "",
    }
    return VerificationEvidence(
        stage_id=stage_id,
        status=_status_satisfied(),
        rationale=str(payload["evidence"]),
        log_evidence=["STAGE_STATUS " + json.dumps(payload, ensure_ascii=False, default=str)],
        repair_hint=None,
    )


def _placement_after_step_override_evidence(
    stage: Any,
    evidence: VerificationEvidence,
    info_step: dict[str, Any],
    *,
    blocked_goals: set[tuple[str, str, str]] | None = None,
) -> VerificationEvidence:
    if evidence.status.value == "satisfied":
        return evidence
    stage_id = str(getattr(stage, "stage_id", ""))
    if not any(fragment in stage_id for fragment in ("transport_guarded", "low_release_verify", "pick_lift_verify")):
        return evidence
    if _placement_goal_is_blocked(stage, blocked_goals):
        return evidence
    if "low_release_verify" in stage_id:
        rationale = str(evidence.rationale or "").lower()
        if (
            (
                "require_drawer_closing_clearance=true" in rationale
                or "drawer_clearance" in rationale
                or "drawer_closing_clearance" in rationale
            )
            and _placement_failure_requires_physical_repair(rationale)
        ):
            return evidence
    goal = _matched_satisfied_placement_goal(stage, info_step.get("bddl_goal_diagnostics", {}))
    if goal is None:
        return evidence
    rationale = (
        "Rule verifier override: post-step BDDL diagnostics show the relevant "
        f"placement goal is already satisfied ({goal.get('raw')}); proceed to the next stage."
    )
    return VerificationEvidence(
        stage_id=stage_id,
        status=_status_satisfied(),
        rationale=rationale,
        visual_evidence=evidence.visual_evidence,
        geometry_evidence={**(evidence.geometry_evidence or {}), "matched_goal": goal},
        log_evidence=[*(evidence.log_evidence or [])],
        repair_hint=None,
    )


def _placement_goal_is_blocked(stage: Any, blocked_goals: set[tuple[str, str, str]] | None) -> bool:
    if not blocked_goals:
        return False
    key = _placement_goal_key_for_stage(stage)
    if key is None:
        return False
    if key in blocked_goals:
        return True
    stage_id = str(getattr(stage, "stage_id", ""))
    if "pick_lift_verify" in stage_id:
        predicate, object_ref, _target_ref = key
        return any(
            blocked_predicate == predicate and blocked_object == object_ref
            for blocked_predicate, blocked_object, _blocked_target in blocked_goals
        )
    return False


def _placement_goal_key_for_stage(
    stage: Any,
    goal_diagnostics: Any | None = None,
) -> tuple[str, str, str] | None:
    predicate = _resolve_stage_predicate_for_goal_diagnostics(stage, goal_diagnostics, default="in")
    if predicate not in {"in", "on"}:
        return None
    object_name = _semantic_object_ref_from_stage(stage) or _stage_object_slug_from_id(stage)
    target_name = _resolve_stage_target_reference(stage)
    if not object_name or not target_name:
        return None
    return (
        predicate,
        _canonical_stage_goal_text(object_name),
        _canonical_stage_goal_text(target_name),
    )


def _canonical_stage_goal_text(value: str) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    for suffix in ("_1", "_2", "_3", "_4"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _update_blocked_placement_goals(
    blocked_goals: set[tuple[str, str, str]],
    stage: Any,
    evidence: VerificationEvidence,
) -> None:
    key = _placement_goal_key_for_stage(stage)
    if key is None:
        return
    stage_id = str(getattr(stage, "stage_id", ""))
    evidence_text = " ".join(
        [
            str(evidence.rationale or ""),
            str(evidence.repair_hint or ""),
            " ".join(str(item) for item in (evidence.log_evidence or [])),
        ]
    ).lower()
    if evidence.status.value == "satisfied":
        if "low_release_verify" in stage_id and not _placement_failure_requires_physical_repair(evidence_text):
            blocked_goals.discard(key)
        return
    if _placement_failure_requires_physical_repair(evidence_text):
        blocked_goals.add(key)


def _placement_failure_requires_physical_repair(evidence_text: str) -> bool:
    return any(
        token in evidence_text
        for token in (
            "insufficient_front_clearance_for_later_close",
            "lacks close-time containment clearance",
            "drawer_low_release_failed_preserve_state",
            "drawer insertion failed",
            "require_drawer_closing_clearance=true",
            "drawer_clearance",
            "drawer_closing_clearance",
        )
    ) and any(
        token in evidence_text
        for token in (
            "has_clearance': false",
            '"has_clearance": false',
            "insufficient_front_clearance_for_later_close",
            "drawer_low_release_failed_preserve_state",
        )
    )


def _matched_satisfied_placement_goal(stage: Any, goal_diagnostics: Any) -> dict[str, Any] | None:
    return _matched_placement_goal(stage, goal_diagnostics, require_satisfied=True)


def _matched_placement_goal(
    stage: Any,
    goal_diagnostics: Any,
    *,
    require_satisfied: bool,
    predicate: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(goal_diagnostics, dict):
        return None
    goals = goal_diagnostics.get("goal_state")
    if not isinstance(goals, list):
        return None
    expected_predicate = predicate or _resolve_stage_predicate(stage, default="in")
    if expected_predicate not in {"in", "on"}:
        return None
    object_name = _semantic_object_ref_from_stage(stage) or _stage_object_slug_from_id(stage)
    target_name = _resolve_stage_target_reference(stage)
    stage_text = " ".join(
        [
            str(getattr(stage, "stage_id", "")),
            str(getattr(stage, "name", "")),
            str(getattr(stage, "intent", "")),
            " ".join(str(ref) for ref in getattr(stage, "target_references", [])),
        ]
    ).lower().replace(" ", "_")
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        if require_satisfied and not bool(goal.get("satisfied", False)):
            continue
        if predicate is not None and str(goal.get("predicate", "")).lower() != expected_predicate:
            raw = goal.get("raw")
            if not (isinstance(raw, list) and raw and str(raw[0]).lower() == expected_predicate):
                continue
        goal_object = str(goal.get("object_name", goal.get("resolved_object", ""))).lower()
        goal_target = str(goal.get("target_name", "")).lower()
        raw_text = "_".join(str(item).lower() for item in goal.get("raw", []) if item is not None)
        diag = goal.get("diagnostic") if isinstance(goal.get("diagnostic"), dict) else {}
        if isinstance(diag, dict):
            goal_object = " ".join(
                part
                for part in [
                    goal_object,
                    str(diag.get("object_name", "")).lower(),
                ]
                if part
            )
            goal_target = " ".join(
                part
                for part in [
                    goal_target,
                    str(diag.get("site_name", "")).lower(),
                    str(diag.get("target_name", "")).lower(),
                ]
                if part
            )
        object_ok = _stage_goal_text_matches(object_name, goal_object + " " + raw_text)
        target_ok = _stage_goal_text_matches(target_name, goal_target + " " + raw_text)
        if not object_ok and object_name:
            object_ok = _stage_goal_text_matches(object_name, stage_text) and _stage_goal_text_matches(object_name, raw_text)
        if not target_ok and target_name:
            target_ok = _stage_goal_text_matches(target_name, stage_text) and _stage_goal_text_matches(target_name, raw_text)
        if object_ok and (target_ok or not target_name):
            return goal
    return None


def _object_matched_unique_placement_goal(
    stage: Any,
    goal_diagnostics: Any,
    *,
    require_satisfied: bool = False,
    predicate: str | None = None,
) -> dict[str, Any] | None:
    """Find the unique BDDL placement goal for this stage object, ignoring stage target text.

    This keeps data-generation fast-paths anchored to the benchmark goal when a
    language distractor such as "not on the plate" is accidentally parsed as the
    target by an upstream planner.
    """
    if not isinstance(goal_diagnostics, dict):
        return None
    goals = goal_diagnostics.get("goal_state")
    if not isinstance(goals, list):
        return None
    object_name = _semantic_object_ref_from_stage(stage) or _stage_object_slug_from_id(stage)
    if not object_name:
        return None
    expected_predicate = str(predicate or "").strip().lower()
    candidates: list[dict[str, Any]] = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        if require_satisfied and not bool(goal.get("satisfied", False)):
            continue
        goal_predicate = str(goal.get("predicate") or "").strip().lower()
        raw = goal.get("raw")
        if not goal_predicate and isinstance(raw, list) and raw:
            goal_predicate = str(raw[0]).strip().lower()
        if goal_predicate not in {"in", "on"}:
            continue
        if expected_predicate in {"in", "on"} and goal_predicate != expected_predicate:
            continue
        goal_object = str(goal.get("object_name", goal.get("resolved_object", ""))).lower()
        raw_text = "_".join(str(item).lower() for item in goal.get("raw", []) if item is not None)
        diag = goal.get("diagnostic") if isinstance(goal.get("diagnostic"), dict) else {}
        if isinstance(diag, dict):
            goal_object = " ".join(
                part
                for part in [
                    goal_object,
                    str(diag.get("object_name", "")).lower(),
                ]
                if part
            )
        if _stage_goal_text_matches(object_name, goal_object + " " + raw_text):
            candidates.append(goal)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _goal_target_name(goal: Any) -> str:
    if not isinstance(goal, dict):
        return ""
    target = str(goal.get("target_name") or "").strip()
    if target:
        return target
    raw = goal.get("raw")
    if isinstance(raw, list) and len(raw) >= 3:
        return str(raw[2]).strip()
    return ""


def _goal_predicate(goal: Any, *, default: str = "in") -> str:
    if not isinstance(goal, dict):
        return default
    predicate = str(goal.get("predicate") or "").strip().lower()
    raw = goal.get("raw")
    if not predicate and isinstance(raw, list) and raw:
        predicate = str(raw[0]).strip().lower()
    return predicate if predicate in {"in", "on"} else default


def _stage_object_slug_from_id(stage: Any) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    for suffix in ("_pick_lift_verify", "_transport_guarded", "_low_release_verify"):
        if suffix in stage_id:
            return _object_name_from_stage_id(stage_id, suffix)
    return ""


def _stage_goal_text_matches(ref: str, text: str) -> bool:
    ref_norm = str(ref or "").strip().lower().replace(" ", "_")
    text_norm = str(text or "").strip().lower().replace(" ", "_")
    if not ref_norm:
        return False
    if ref_norm in text_norm:
        return True
    for alias in _STAGE_TARGET_ALIASES.get(ref_norm, ()):
        if alias in text_norm:
            return True
    for alias in _SEMANTIC_STAGE_ALIASES.get(ref_norm, ()):
        if alias in text_norm:
            return True
    compact_ref = ref_norm.replace("_", "")
    compact_text = text_norm.replace("_", "")
    return bool(compact_ref and compact_ref in compact_text)


_STAGE_TARGET_ALIASES = {
    "bottom_drawer": ("bottom_region", "white_cabinet_1_bottom_region", "cabinet_1_bottom"),
    "middle_drawer": ("middle_region", "white_cabinet_1_middle_region", "cabinet_1_middle"),
    "top_drawer": ("top_region", "white_cabinet_1_top_region", "cabinet_1_top"),
    "drawer": ("drawer", "cabinet"),
    "back_caddy_compartment": ("back_contain_region", "desk_caddy_1_back_contain_region"),
    "caddy": ("desk_caddy", "contain_region"),
    "microwave": ("microwave_1_heating_region", "microwave_heating_region", "microwave_1_contain_region"),
}


def _rule_override_evidence(
    stage: Any,
    evidence: VerificationEvidence,
    scene_snapshot: dict[str, Any],
) -> VerificationEvidence:
    """Use deterministic readonly checks for pure grounding stages.

    Grounding stages do not move the robot. If the scene snapshot already proves
    that the requested object/target are present and have target sites or poses,
    a brittle model self-report such as "basket z is negative" should not burn
    stage attempts.
    """
    stage_id = str(getattr(stage, "stage_id", ""))
    drawer_capture_evidence = _drawer_capture_override_evidence(stage, evidence)
    if drawer_capture_evidence is not None:
        return drawer_capture_evidence
    if "ground_object_target" not in stage_id:
        return evidence
    if not _grounding_snapshot_satisfied(stage, scene_snapshot):
        return evidence
    if evidence.status.value == "satisfied":
        return evidence
    return VerificationEvidence(
        stage_id=stage.stage_id,
        status=_status_satisfied(),
        rationale=(
            "Rule verifier override: readonly snapshot contains the requested "
            "object/target grounding and at least one grasp/target affordance."
        ),
        log_evidence=[*(evidence.log_evidence or [])],
        repair_hint=None,
    )


def _satisfied_rule_override_evidence(evidence: VerificationEvidence | None) -> bool:
    if evidence is None or evidence.status.value != "satisfied":
        return False
    return str(evidence.rationale or "").startswith("Rule verifier override:")


def _drawer_stage_already_satisfied_evidence(
    stage: Any,
    scene_snapshot: dict[str, Any],
    api_functions: dict[str, Any],
) -> VerificationEvidence | None:
    """Readonly boundary check for drawer stages whose open precondition is already true."""
    stage_id = str(getattr(stage, "stage_id", ""))
    stage_text = " ".join(
        [
            stage_id,
            str(getattr(stage, "name", "")),
            str(getattr(stage, "intent", "")),
        ]
    ).lower()
    if "drawer" not in stage_text:
        return None
    if not any(token in stage_text for token in ("ground_handle", "sleeve", "hook", "open", "pull")):
        return None
    stage_id_text = stage_id.lower()
    if "close_guarded" in stage_id_text or stage_id_text.endswith("_close") or stage_id_text.endswith("_closing"):
        return None
    level = _drawer_level_from_stage_context(stage) or _drawer_level_from_stage_id(stage_id)
    if not level:
        return None
    progress = None
    drawer_progress = scene_snapshot.get("drawer_progress") if isinstance(scene_snapshot, dict) else None
    if isinstance(drawer_progress, dict):
        progress = drawer_progress.get(level)
    drawers = scene_snapshot.get("drawers") if isinstance(scene_snapshot, dict) else None
    if not isinstance(progress, dict) and isinstance(drawers, dict):
        drawer_snapshot = drawers.get(level)
        if isinstance(drawer_snapshot, dict):
            progress = drawer_snapshot.get("progress")
    if not isinstance(progress, dict):
        getter = api_functions.get("get_cabinet_drawer_progress")
        if callable(getter):
            try:
                progress = getter(level)
            except Exception:
                progress = None
    if not isinstance(progress, dict):
        return None
    try:
        open_fraction = float(progress.get("open_fraction", 0.0) or 0.0)
    except Exception:
        return None
    threshold = 0.96 if str(level).lower() == "bottom" else 0.98
    bddl_open = bool(progress.get("bddl_open", False))
    if open_fraction < threshold and not bddl_open:
        return None
    payload = {
        "stage_id": stage_id,
        "status": "satisfied",
        "evidence": {
            "level": level,
            "drawer_already_open": True,
            "open_fraction": open_fraction,
            "threshold": threshold,
            "bddl_open": bddl_open,
            "progress": progress,
        },
        "repair": "",
    }
    return VerificationEvidence(
        stage_id=stage_id,
        status=_status_satisfied(),
        rationale=(
            "Rule verifier override: readonly drawer diagnostics show this drawer is already open enough "
            "for the downstream object stage, so skip the open/capture prefix instead of closing and reopening."
        ),
        geometry_evidence={"drawer_already_open": payload["evidence"]},
        log_evidence=["STAGE_STATUS " + json.dumps(payload, ensure_ascii=False, default=str)],
        repair_hint=None,
    )


def _drawer_capture_override_evidence(stage: Any, evidence: VerificationEvidence) -> VerificationEvidence | None:
    stage_id = str(getattr(stage, "stage_id", ""))
    stage_text = " ".join(
        [
            stage_id,
            str(getattr(stage, "name", "")),
            str(getattr(stage, "intent", "")),
        ]
    ).lower()
    if not ("drawer" in stage_text and ("sleeve" in stage_text or "hook" in stage_text or "capture" in stage_text)):
        return None
    if evidence.status.value == "satisfied":
        return None
    evidence_text = " ".join(
        [
            str(evidence.rationale or ""),
            str(evidence.repair_hint or ""),
            " ".join(str(x) for x in (evidence.log_evidence or [])),
        ]
    ).lower()
    positive_tokens = (
        "closed_on_handle': true",
        '"closed_on_handle": true',
        "closed_on_handle=true",
        "likely_hooked': true",
        '"likely_hooked": true',
        "likely_hooked=true",
        "has_handle_contact': true",
        '"has_handle_contact": true',
        "has_contact=true",
        "captured=true",
        "handle captured",
        "gripper closed on handle",
    )
    if not any(token in evidence_text for token in positive_tokens):
        return None
    return VerificationEvidence(
        stage_id=stage_id,
        status=_status_satisfied(),
        rationale=(
            "Rule verifier override: drawer sleeve/hook evidence reports handle capture "
            "(closed_on_handle/likely_hooked/contact), so the capture stage is satisfied."
        ),
        visual_evidence=evidence.visual_evidence,
        geometry_evidence={**(evidence.geometry_evidence or {}), "drawer_capture_override": True},
        log_evidence=[*(evidence.log_evidence or [])],
        repair_hint=None,
    )


def _grounding_snapshot_satisfied(stage: Any, snapshot: dict[str, Any]) -> bool:
    refs = [str(ref).lower().replace(" ", "_") for ref in getattr(stage, "target_references", [])]
    objects = snapshot.get("objects_of_interest")
    sites = snapshot.get("target_sites_of_interest")
    if not isinstance(objects, dict) or not objects:
        semantic_ref = _semantic_object_ref_from_stage(stage)
        if not semantic_ref:
            return False
        target_ref = _resolve_stage_target_reference(stage)
        return bool(target_ref)
    movable_refs = [ref for ref in refs if ref not in {"in", "on", "on_or_in", "right_of", "next_to"}]
    primary_ref = movable_refs[0] if movable_refs else ""
    target_refs = set(movable_refs[1:])
    matched_objects = 0
    has_grasp_affordance = False
    for ref in ([primary_ref] if primary_ref else movable_refs):
        match = _best_grounding_object_match(ref, objects)
        if match is not None:
            matched_objects += 1
            obj_snapshot = objects[match]
            if isinstance(obj_snapshot, dict):
                affordance = obj_snapshot.get("grasp_affordance")
                if isinstance(affordance, dict) and affordance.get("candidate_count", 0):
                    has_grasp_affordance = True
    has_target_site = isinstance(sites, dict) and bool(sites)
    has_target_object = False
    for ref in target_refs:
        for name in objects:
            if _names_match_for_stage(ref, str(name).lower()):
                has_target_object = True
                break
        if has_target_object:
            break
    context = snapshot.get("robosuite_task_context")
    task_family = str(context.get("task_family", "")).lower() if isinstance(context, dict) else ""
    if task_family == "cube_lift":
        return bool(matched_objects >= 1 and has_grasp_affordance)
    if task_family == "wipe":
        return bool(matched_objects >= 1 and "wipe_diagnostic" in snapshot)
    if task_family == "bimanual_handover":
        return bool(
            matched_objects >= 1
            and has_grasp_affordance
            and "handover_diagnostic" in snapshot
        )
    # For object-to-container/surface tasks, two grounded objects or one object
    # plus a target site are enough. The grasp affordance confirms the movable
    # object can be acted on.
    return bool(matched_objects >= 1 and has_grasp_affordance and (has_target_object or has_target_site))


def _best_grounding_object_match(ref: str, objects: dict[str, Any]) -> str | None:
    ref_norm = str(ref).lower().replace(" ", "_")
    ref_compact = ref_norm.replace("_", "")
    ranked: list[tuple[int, int, str]] = []
    for name in objects:
        name_text = str(name)
        name_norm = name_text.lower().replace(" ", "_")
        name_compact = name_norm.replace("_", "")
        if name_norm == ref_norm:
            rank = 0
        elif name_compact == ref_compact:
            rank = 1
        elif _names_match_for_stage(ref_norm, name_norm):
            rank = 2
        else:
            continue
        ranked.append((rank, abs(len(name_compact) - len(ref_compact)), name_text))
    return min(ranked)[2] if ranked else None


def _grounding_fast_path_evidence(stage: Any, scene_snapshot: dict[str, Any]) -> VerificationEvidence | None:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "ground_object_target" not in stage_id:
        return None
    if not _grounding_snapshot_satisfied(stage, scene_snapshot):
        return None
    objects = sorted(str(name) for name in (scene_snapshot.get("objects_of_interest") or {}).keys())
    sites = sorted(str(name) for name in (scene_snapshot.get("target_sites_of_interest") or {}).keys())
    rationale = (
        "Readonly grounding fast-path satisfied: "
        f"objects_of_interest={objects[:4]}, target_sites={sites[:4]}"
    )
    payload = {
        "stage_id": stage_id,
        "status": "satisfied",
        "evidence": rationale,
        "repair": "",
    }
    return VerificationEvidence(
        stage_id=stage_id,
        status=_status_satisfied(),
        rationale=rationale,
        log_evidence=["STAGE_STATUS " + json.dumps(payload, ensure_ascii=False, default=str)],
        repair_hint=None,
    )


def _robosuite_stage_fast_path_code(stage: Any) -> str:
    """Build deterministic code for audited privileged RoboSuite stage skills."""
    stage_id = str(getattr(stage, "stage_id", ""))
    calls = {
        "cube_lift_pick_verify": 'verified_pick_cube("red cube", stage_attempt=STAGE_ATTEMPT)',
        "stack_pick_lift_verify": 'verified_pick_cube("red cube", stage_attempt=STAGE_ATTEMPT)',
        "stack_transport_guarded": 'transport_cube_above_target("red cube", "green cube")',
        "stack_low_release_verify": 'release_and_verify_stack("red cube", "green cube")',
        "peg_pick_lift_verify": "verified_grasp_nut_handle(stage_attempt=STAGE_ATTEMPT)",
        "peg_align_guarded": "move_nut_above_peg()",
        "peg_insert_verify": "insert_nut_and_verify()",
        "bimanual_pregrasp_handles": "prepare_bimanual_handle_grasps(stage_attempt=STAGE_ATTEMPT)",
        "bimanual_grasp_verify": "close_and_verify_bimanual_grasps()",
        "bimanual_lift_verify": "synchronized_lift_and_verify()",
        "restack_relocate_covering_cube": "relocate_covering_cube_and_verify(stage_attempt=STAGE_ATTEMPT)",
        "restack_pick_lift_verify": 'verified_pick_cube("red cube", stage_attempt=STAGE_ATTEMPT)',
        "restack_transport_guarded": 'transport_cube_above_target("red cube", "green cube")',
        "restack_low_release_verify": "recover_and_verify_restack(stage_attempt=STAGE_ATTEMPT)",
        "wipe_approach_surface": "approach_spill_surface(stage_attempt=STAGE_ATTEMPT)",
        "wipe_execute_verify": "wipe_spill_and_verify(stage_attempt=STAGE_ATTEMPT)",
        "handover_arm0_pick_verify": "verified_pick_hammer_arm0(stage_attempt=STAGE_ATTEMPT)",
        "handover_present_guarded": "present_hammer_for_handover()",
        "handover_arm1_capture_verify": "capture_hammer_arm1(stage_attempt=STAGE_ATTEMPT)",
        "handover_release_verify": "release_arm0_and_verify_handover()",
    }
    call = calls.get(stage_id)
    if call is None:
        return ""
    return f'''import json

result = {call}
success = bool(isinstance(result, dict) and result.get("success", False))
recoverable = bool(isinstance(result, dict) and result.get("recoverable", True))
status = "satisfied" if success else ("needs_repair" if recoverable else "not_satisfied")
reason = str(result.get("reason", "audited skill returned no reason")) if isinstance(result, dict) else "audited skill returned a non-dict result"
evidence = json.dumps(result, default=str) if isinstance(result, dict) else repr(result)
repair = "" if success else reason
print("STAGE_STATUS " + json.dumps({{
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _pick_lift_fast_path_code(stage: Any, scene_snapshot: dict[str, Any], *, stage_attempt: int = 1) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "pick_lift_verify" not in stage_id:
        return ""
    object_name = _resolve_stage_object_name(stage, scene_snapshot)
    if not object_name:
        object_name = _object_name_from_stage_id(stage_id, "_pick_lift_verify")
    if not object_name:
        return ""
    target_ref = _resolve_stage_target_reference(stage)
    predicate = _resolve_stage_predicate(stage, default="on")
    task_text = str(scene_snapshot.get("task", "")).lower()
    object_text = str(object_name).lower()
    drawer_bowl_pick = bool("drawer" in task_text and "bowl" in object_text)
    drawer_level = _drawer_level_from_stage_context(stage)
    if drawer_level is None and drawer_bowl_pick:
        drawer_search_text = " ".join(
            [
                task_text,
                str(stage_id).lower(),
                str(target_ref).lower(),
                " ".join(str(ref).lower() for ref in getattr(stage, "target_references", [])),
            ]
        ).replace("_", " ")
        for candidate_level in ("top", "middle", "bottom"):
            if f"{candidate_level} drawer" in drawer_search_text or f"{candidate_level} cabinet" in drawer_search_text:
                drawer_level = candidate_level
                break
        if drawer_level is None:
            drawer_level = "middle"
    if drawer_bowl_pick:
        candidate_index = max(0, int(stage_attempt) - 1)
    else:
        candidate_index = max(0, int(stage_attempt) - 1)
    return f'''import json

stage_id = {stage_id!r}
object_name = {object_name!r}
target_name = {target_ref!r}
predicate = {predicate!r}
candidate_index = {candidate_index!r}
drawer_bowl_pick = {drawer_bowl_pick!r}
drawer_level = {drawer_level!r}
fragile_support_grasp = bool(
    predicate == "on"
    and (
        any(token in str(object_name).lower() for token in ("mug", "cup", "bowl", "ramekin"))
        or any(token in str(target_name).lower() for token in ("plate", "bowl"))
    )
)

print(f"Stage: {{stage_id}} - deterministic audited pick-lift for {{object_name}} candidate_index={{candidate_index}}")
try:
    drawer_preopen = None
    if drawer_bowl_pick:
        drawer_required_open_fraction = 0.96 if str(drawer_level or "").lower() == "bottom" else 0.98
        drawer_before_guard = get_cabinet_drawer_progress(drawer_level or "middle")
        drawer_before_fraction = float(drawer_before_guard.get("open_fraction", 0.0) if isinstance(drawer_before_guard, dict) else drawer_before_guard)
        drawer_before_bddl_open = bool(isinstance(drawer_before_guard, dict) and drawer_before_guard.get("bddl_open", False))
        if drawer_before_fraction >= drawer_required_open_fraction:
            drawer_preopen = {{"reason": "preserve_strictly_open_drawer_for_bowl_insertion", "after": drawer_before_guard, "opened": True}}
        else:
            drawer_preopen = open_drawer_by_sleeve_retry(
                drawer_level or "middle",
                open_threshold=drawer_required_open_fraction,
                max_cycles=5,
                test_distance=0.055,
                pull_distance=0.48,
                strict_open_fraction=True,
            )
        print("Drawer pre-pick open guard:", {{"drawer_level": drawer_level, "before": drawer_before_guard, "result": drawer_preopen}})
        drawer_after_guard = drawer_preopen.get("after", {{}}) if isinstance(drawer_preopen, dict) else {{}}
        drawer_after_fraction = float(drawer_after_guard.get("open_fraction", 0.0) or 0.0)
        drawer_after_bddl_open = bool(isinstance(drawer_after_guard, dict) and drawer_after_guard.get("bddl_open", False))
        if drawer_after_fraction < drawer_required_open_fraction:
            evidence = (
                f"drawer pre-pick open guard failed, level={{drawer_level}}, "
                f"open_fraction={{drawer_after_fraction:.3f}}, required={{drawer_required_open_fraction:.3f}}, "
                f"bddl_open={{drawer_after_bddl_open}}, result={{drawer_preopen}}"
            )
            raise Exception(evidence)
    before = measure_object_grasp_state(object_name)
    print("Before grasp:", before)
    object_text = str(object_name).lower()
    if drawer_bowl_pick:
        grasp = verified_low_side_pick_object(
            object_name,
            max_attempts=2,
            start_candidate_index=candidate_index,
            lift_height=0.075,
            settle_steps=18,
        )
        print("Drawer bowl low-side pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False):
            print("Drawer bowl low-side pick failed; falling back to geometry pick candidate.")
            grasp = verified_geometry_pick_object(
                object_name,
                max_attempts=1,
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
            print("Drawer bowl geometry fallback result:", grasp)
    elif "moka" in object_text or "book" in object_text:
        grasp = verified_geometry_pick_object(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Geometry pick result:", grasp)
        if isinstance(grasp, dict) and not grasp.get("held", False) and "moka" in object_text:
            print("Moka geometry pick was not transport-stable; trying affordance grasp fallback.")
            grasp = approach_and_grasp_object(
                object_name,
                max_attempts=2,
                start_candidate_index=max(0, candidate_index % 6),
                lift_height=0.135,
            )
            print("Moka affordance fallback result:", grasp)
    else:
        direct_attempts = 3 if drawer_bowl_pick else 1
        grasp = verified_lift_pick_object(object_name, max_attempts=direct_attempts, start_candidate_index=candidate_index)
        print("Audited grasp result:", grasp)
    if isinstance(grasp, dict) and (not drawer_bowl_pick) and (("bowl" in str(object_name).lower()) or fragile_support_grasp):
        strict_after_direct = measure_object_grasp_state(grasp.get("object_name", object_name), strict=True)
        direct_lift = float(grasp.get("lift_delta", 0.0) or 0.0)
        direct_stable = bool(grasp.get("held", False) and (strict_after_direct.get("held", False) or direct_lift >= 0.060))
        if not direct_stable:
            print("Direct pick is not transport-stable for fragile support placement; trying affordance fallback.")
            fallback_index = max(0, candidate_index)
            grasp = approach_and_grasp_object(object_name, max_attempts=2, start_candidate_index=fallback_index)
            print("Fragile support affordance fallback result:", grasp)
            if isinstance(grasp, dict) and not grasp.get("held", False) and ("mug" in str(object_name).lower()):
                print("Mug affordance fallback failed; trying side-grasp fallback.")
                grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
                print("Mug side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and ("wine" in str(object_name).lower() or "bottle" in str(object_name).lower()):
        print("Tall/narrow object direct pick failed; trying audited side-grasp fallback once.")
        grasp = grasp_tall_object_side(object_name, max_attempts=1, start_candidate_index=candidate_index)
        print("Tall side-grasp fallback result:", grasp)
    elif isinstance(grasp, dict) and not grasp.get("held", False) and candidate_index >= 11:
        print("Direct verified pick exhausted canonical offsets; trying affordance fallback once.")
        grasp = approach_and_grasp_object(object_name, max_attempts=1, start_candidate_index=max(0, candidate_index - 11))
        print("Affordance fallback result:", grasp)
    resolved_name = grasp.get("object_name", object_name) if isinstance(grasp, dict) else object_name
    after = measure_object_grasp_state(resolved_name)
    print("After grasp:", after)
    held = bool(isinstance(grasp, dict) and grasp.get("held")) or bool(isinstance(after, dict) and after.get("held"))
    lift_delta = 0.0
    if isinstance(grasp, dict):
        lift_delta = max(lift_delta, float(grasp.get("lift_delta", 0.0) or 0.0))
    if isinstance(after, dict):
        lift_delta = max(lift_delta, float(after.get("lift_delta", 0.0) or 0.0))
    stable_held = bool(held or lift_delta >= 0.035)
    strict_hold = None
    if fragile_support_grasp:
        strict_hold = measure_object_grasp_state(resolved_name, strict=True)
        stable_held = bool(held and (strict_hold.get("held", False) or lift_delta >= 0.060))
    if drawer_bowl_pick:
        transport_probe = grasp.get("transport_probe") if isinstance(grasp, dict) else None
        if not isinstance(transport_probe, dict) and isinstance(grasp, dict) and isinstance(grasp.get("best_attempt"), dict):
            transport_probe = grasp.get("best_attempt", {{}}).get("transport_probe")
        probe_held = bool(isinstance(transport_probe, dict) and transport_probe.get("held", False))
        stable_held = bool(held and (probe_held or lift_delta >= 0.070))
    status = "satisfied" if stable_held else "needs_repair"
    evidence = f"held={{held}}, stable_held={{stable_held}}, lift_delta={{lift_delta:.4f}}, object={{resolved_name}}, target={{target_name}}, candidate_index={{candidate_index}}, drawer_level={{drawer_level}}"
    repair = "" if status == "satisfied" else "Audited single grasp attempt failed; retry stage with next snapshot/candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"audited pick-lift fast-path exception: {{exc}}"
    repair = "Retry stage if episode is still active; otherwise reset trial or reduce motion budget."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _transport_fast_path_code(stage: Any, scene_snapshot: dict[str, Any]) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "transport_guarded" not in stage_id:
        return ""
    object_name = _resolve_stage_object_name(stage, scene_snapshot)
    if not object_name:
        object_name = _object_name_from_stage_id(stage_id, "_transport_guarded")
    if not object_name:
        return ""
    target_ref = _resolve_stage_target_reference(stage)
    predicate = _resolve_stage_predicate_for_snapshot(stage, scene_snapshot, default="in")
    bddl_goal = _object_matched_unique_placement_goal(
        stage,
        scene_snapshot.get("bddl_goal_diagnostics", {}) if isinstance(scene_snapshot, dict) else {},
        require_satisfied=False,
        predicate=None,
    )
    if isinstance(bddl_goal, dict):
        target_ref = _goal_target_name(bddl_goal) or target_ref
        predicate = _goal_predicate(bddl_goal, default=predicate)
    return f'''import json

stage_id = {stage_id!r}
object_name = {object_name!r}
target_name = {target_ref!r}
predicate = {predicate!r}

print(f"Stage: {{stage_id}} - deterministic guarded transport for {{object_name}} to {{target_name}}")
try:
    transport = guarded_transport_held_object_to_target(
        object_name,
        predicate=predicate,
        target_name=target_name or None,
    )
    print("Guarded transport result:", transport)
    effective_predicate = str(transport.get("target", {{}}).get("predicate", predicate)) if isinstance(transport, dict) else predicate
    held_before = bool(isinstance(transport, dict) and transport.get("held_before"))
    held_after = bool(isinstance(transport, dict) and transport.get("held_after"))
    arrived = bool(isinstance(transport, dict) and transport.get("arrived"))
    fused_satisfied = bool(isinstance(transport, dict) and transport.get("relation_satisfied"))
    task_status = get_task_status()
    if isinstance(transport, dict) and isinstance(transport.get("task_status"), dict):
        task_status = transport.get("task_status", task_status)
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    xy_distance = float(transport.get("xy_distance_to_target", 999.0) if isinstance(transport, dict) else 999.0)
    if isinstance(transport, dict) and xy_distance >= 999.0:
        relation_for_xy = transport.get("final_relation") or transport.get("relation") or {{}}
        if isinstance(relation_for_xy, dict) and "local_delta" in relation_for_xy:
            try:
                local_delta_for_xy = relation_for_xy.get("local_delta")
                xy_distance = float((float(local_delta_for_xy[0]) ** 2 + float(local_delta_for_xy[1]) ** 2) ** 0.5)
            except Exception:
                pass
    closure = None
    closure_xy_threshold = 0.045 if effective_predicate == "on" else 0.090
    tolerance = float(transport.get("xy_tolerance", 0.075) if isinstance(transport, dict) else 0.075)
    held_near_release = bool(held_after and effective_predicate in {{"on", "in"}} and xy_distance <= max(0.090, 1.8 * tolerance))
    strict_caddy_transport = bool(
        effective_predicate == "in"
        and "book" in str(object_name).lower()
        and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower())
    )
    final_relation = transport.get("final_relation", None) if isinstance(transport, dict) else None
    z_only_near_success = bool(isinstance(transport, dict) and transport.get("z_only_near_success", False))
    near_contact_relation = bool(isinstance(transport, dict) and transport.get("near_contact_relation"))
    drawer_immediate_release_failed = False
    if (not near_contact_relation) and isinstance(final_relation, dict) and effective_predicate == "in":
        relation_xy = final_relation.get("xy_distance", None)
        if relation_xy is None and "local_delta" in final_relation:
            try:
                local_delta = final_relation.get("local_delta")
                relation_xy = float((float(local_delta[0]) ** 2 + float(local_delta[1]) ** 2) ** 0.5)
            except Exception:
                relation_xy = None
        near_contact_relation = bool(
            final_relation.get("contact", False)
            and "local_delta" in final_relation
            and relation_xy is not None
            and float(relation_xy) <= 0.18
        )
    near_target_for_closure = (
        (not held_after)
        and effective_predicate in {{"on", "in"}}
        and (xy_distance <= closure_xy_threshold or near_contact_relation)
        and not completed
        and reward < 1.0
        and (not drawer_immediate_release_failed or z_only_near_success or near_contact_relation)
    )
    if (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        near_target_for_closure = False
    if near_target_for_closure:
        print("Transport lost hold near target; routing to release/local-closure stage.")
    if held_near_release and not arrived:
        print("Transport still holds object near target; allowing release/local-closure stage.")
    closure_satisfied = bool(isinstance(closure, dict) and closure.get("relation_satisfied"))
    closure_attempted_near_target = bool(isinstance(closure, dict) and closure.get("near_target"))
    def summarize_relation(relation):
        if not isinstance(relation, dict):
            return None
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        return {{
            "satisfied": bool(relation.get("satisfied", relation.get("in", relation.get("on", False)))),
            "in": relation.get("in"),
            "on": relation.get("on"),
            "contact": relation.get("contact"),
            "local_delta": as_list(relation.get("local_delta")),
            "site_local_delta": as_list(relation.get("site_local_delta")),
            "site_size_abs": as_list(relation.get("site_size_abs")),
            "aabb_margin_lower_world": as_list(relation.get("aabb_margin_lower_world")),
            "aabb_margin_upper_world": as_list(relation.get("aabb_margin_upper_world")),
            "aabb_z_excess": relation.get("aabb_z_excess"),
            "aabb_z_deficit": relation.get("aabb_z_deficit"),
            "aabb_margin_min": relation.get("aabb_margin_min"),
            "object_pos_world": as_list(relation.get("object_pos_world")),
            "site_pos_world": as_list(relation.get("site_pos_world")),
        }}
    def summarize_nested_record(record):
        if not isinstance(record, dict):
            return {{}}
        return {{
            "action": record.get("action"),
            "iteration": record.get("iteration"),
            "trigger": record.get("trigger"),
            "before_norm": record.get("before_norm"),
            "after_norm": record.get("after_norm"),
            "object_motion_xy": record.get("object_motion_xy"),
            "stroke": record.get("stroke"),
            "push_z": record.get("push_z"),
            "improved": record.get("improved"),
            "worsened": record.get("worsened"),
            "local_delta_before": summarize_relation(record.get("relation_before")).get("local_delta") if isinstance(summarize_relation(record.get("relation_before")), dict) else None,
            "relation_after": summarize_relation(record.get("relation_after") or record.get("relation")),
            "reason": record.get("reason"),
            "error": record.get("error"),
        }}
    def summarize_move(move):
        if not isinstance(move, dict):
            return None
        return {{
            "success": move.get("success"),
            "distance": move.get("distance"),
            "error": move.get("error"),
            "reason": move.get("reason"),
        }}
    def summarize_state(state):
        if not isinstance(state, dict):
            return None
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        return {{
            "held": state.get("held"),
            "object_position": as_list(state.get("object_position")),
            "finger_midpoint": as_list(state.get("finger_midpoint")),
        }}
    def summarize_waypoint_record(record):
        if not isinstance(record, dict):
            return {{}}
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        align = record.get("align")
        align_summary = None
        if isinstance(align, dict):
            align_records = align.get("records", [])
            align_tail = []
            if isinstance(align_records, list):
                for align_record in align_records[-2:]:
                    if not isinstance(align_record, dict):
                        continue
                    align_tail.append({{
                        "iteration": align_record.get("iteration"),
                        "action": align_record.get("action"),
                        "target_object_position": as_list(align_record.get("target_object_position")),
                        "target_finger_midpoint": as_list(align_record.get("target_finger_midpoint")),
                        "object_position_after": as_list(align_record.get("object_position_after")),
                        "error_after": as_list(align_record.get("error_after")),
                        "held_after": align_record.get("held_after"),
                    }})
            align_summary = {{
                "arrived": align.get("arrived"),
                "held_before": align.get("held_before"),
                "held_after": align.get("held_after"),
                "xy_error": align.get("xy_error"),
                "z_error": align.get("z_error"),
                "target_object_position": as_list(align.get("target_object_position")),
                "tail": align_tail,
            }}
        return {{
            "phase": record.get("phase"),
            "reason": record.get("reason"),
            "skipped": record.get("skipped"),
            "target_object_position": as_list(record.get("target_object_position")),
            "target_ee_position": as_list(record.get("target_ee_position")),
            "object_to_ee": as_list(record.get("object_to_ee")),
            "object_error_after": as_list(record.get("object_error_after")),
            "move": summarize_move(record.get("move")),
            "align": align_summary,
            "state_after": summarize_state(record.get("state_after") or record.get("state")),
            "relation_after": summarize_relation(record.get("relation_after") or record.get("relation")),
        }}
    def summarize_align_item(item):
        if not isinstance(item, dict):
            return {{}}
        def as_list(value, limit=3):
            try:
                return [float(value[i]) for i in range(min(limit, len(value)))]
            except Exception:
                try:
                    seq = list(value)
                    return [float(x) for x in seq[:limit]]
                except Exception:
                    return None
        record = item.get("record", {{}})
        record_summary = {{}}
        if isinstance(record, dict):
            record_summary = {{
                "skill": record.get("skill"),
                "reason": record.get("reason"),
                "relation_satisfied": record.get("relation_satisfied"),
                "held_before": record.get("held_before"),
                "held_after": record.get("held_after"),
                "released": record.get("released"),
                "arrived": record.get("arrived"),
                "xy_error": record.get("xy_error"),
                "z_error": record.get("z_error"),
                "target_position": as_list(record.get("target_position")),
                "release_target": as_list(record.get("release_target")),
                "pull_direction": as_list(record.get("pull_direction")),
                "relation": summarize_relation(record.get("relation")),
                "final_relation": summarize_relation(record.get("final_relation")),
                "task_status": record.get("task_status"),
                "after": summarize_state(record.get("after")),
            }}
            nested = record.get("records", [])
            if isinstance(nested, list):
                record_summary["record_tail"] = [
                    summarize_waypoint_record(rec)
                    for rec in nested[-4:]
                    if isinstance(rec, dict)
                ]
        candidate_records = item.get("records", [])
        candidate_tail = []
        if isinstance(candidate_records, list):
            for candidate in candidate_records[-3:]:
                if not isinstance(candidate, dict):
                    continue
                candidate_tail.append({{
                    "candidate_index": candidate.get("candidate_index"),
                    "depth": candidate.get("depth"),
                    "lateral": candidate.get("lateral"),
                    "skipped": candidate.get("skipped"),
                    "reason": candidate.get("reason"),
                    "object_error_to_target": as_list(candidate.get("object_error_to_target")),
                    "held": (candidate.get("record") or {{}}).get("held") if isinstance(candidate.get("record"), dict) else None,
                    "relation_before_gripper_release": summarize_relation(candidate.get("relation_before_gripper_release")),
                }})
        return {{
            "phase": item.get("phase"),
            "release_strategy": item.get("release_strategy"),
            "held_after": item.get("held_after"),
            "drawer_level": item.get("drawer_level"),
            "require_drawer_closing_clearance": item.get("require_drawer_closing_clearance"),
            "relation": summarize_relation(item.get("relation")),
            "record": record_summary,
            "candidate_tail": candidate_tail,
        }}
    record_actions = []
    if isinstance(transport, dict):
        for item in transport.get("records", [])[-6:]:
            if isinstance(item, dict):
                result = item.get("result", {{}})
                result_summary = {{}}
                if isinstance(result, dict):
                    result_summary = {{
                        "reason": result.get("reason"),
                        "relation_satisfied": result.get("relation_satisfied"),
                        "satisfied": result.get("satisfied"),
                        "near_contact_relation": result.get("near_contact_relation"),
                        "skill": result.get("skill"),
                        "xy_distance_to_target": result.get("xy_distance_to_target"),
                        "xy_tolerance": result.get("xy_tolerance"),
                        "final_relation": summarize_relation(result.get("final_relation")),
                    }}
                    nested_records = result.get("records", [])
                    if isinstance(nested_records, list):
                        result_summary["record_tail"] = [
                            summarize_nested_record(rec)
                            for rec in nested_records[-4:]
                            if isinstance(rec, dict)
                        ]
                    align_records = result.get("align_records", [])
                    if isinstance(align_records, list):
                        result_summary["align_tail"] = [
                            summarize_align_item(rec)
                            for rec in align_records[-6:]
                            if isinstance(rec, dict)
                        ]
                record_actions.append({{"action": item.get("action"), "result": result_summary}})
    if (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        status = "needs_repair"
    else:
        status = "satisfied" if (
            fused_satisfied
            or z_only_near_success
            or near_target_for_closure
            or (held_after and (arrived or (held_near_release and not strict_caddy_transport)))
            or completed
            or reward >= 1.0
        ) else "needs_repair"
    evidence = (
        f"fused_satisfied={{fused_satisfied}}, z_only_near_success={{z_only_near_success}}, drawer_immediate_release_failed={{drawer_immediate_release_failed}}, held_before={{held_before}}, held_after={{held_after}}, arrived={{arrived}}, held_near_release={{held_near_release}}, near_contact_relation={{near_contact_relation}}, near_target_closure={{near_target_for_closure}}, "
        f"closure_satisfied={{closure_satisfied}}, reward={{reward:.3f}}, task_completed={{completed}}, "
        f"xy_distance={{xy_distance:.4f}}, xy_tolerance={{tolerance:.4f}}, final_relation={{final_relation}}, "
        f"record_actions={{record_actions}}, object={{object_name}}, target={{target_name}}, predicate={{effective_predicate}}"
    )
    if status == "satisfied":
        repair = ""
    elif (
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
        and not held_before
    ):
        repair = "Object is not verified-held before drawer transport; replan pick_lift_verify before any placement repair."
    elif strict_caddy_transport and (not held_after or not arrived):
        repair = "Object is not verified-held/aligned for caddy transport; return control to the pick_lift stage and reacquire before release."
    else:
        repair = "Guarded transport did not reach a verified pre-release pose; keep holding the object and retry transport before release."
except Exception as exc:
    status = "needs_repair"
    evidence = f"guarded transport fast-path exception: {{exc}}"
    repair = "Retry transport after checking target grounding and grasp state."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _release_fast_path_code(
    stage: Any,
    scene_snapshot: dict[str, Any],
    *,
    is_final_stage: bool = False,
    require_drawer_closing_clearance: bool = False,
) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "low_release_verify" not in stage_id:
        return ""
    object_name = _resolve_stage_object_name(stage, scene_snapshot)
    if not object_name:
        object_name = _object_name_from_stage_id(stage_id, "_low_release_verify")
    if not object_name:
        return ""
    target_ref = _resolve_stage_target_reference(stage)
    predicate = _resolve_stage_predicate(stage, default="in")
    bddl_goal = _object_matched_unique_placement_goal(
        stage,
        scene_snapshot.get("bddl_goal_diagnostics", {}) if isinstance(scene_snapshot, dict) else {},
        require_satisfied=False,
        predicate=None,
    )
    if isinstance(bddl_goal, dict):
        target_ref = _goal_target_name(bddl_goal) or target_ref
        predicate = _goal_predicate(bddl_goal, default=predicate)
    return f'''import json

stage_id = {stage_id!r}
object_name = {object_name!r}
target_name = {target_ref!r}
predicate = {predicate!r}
is_final_stage = {bool(is_final_stage)!r}
require_drawer_closing_clearance = {bool(require_drawer_closing_clearance)!r}

print(f"Stage: {{stage_id}} - deterministic low release for {{object_name}} to {{target_name}} predicate={{predicate}}")
try:
    used_microwave_release = bool(
        predicate == "in"
        and "microwave" in str(target_name).lower()
    )
    used_rack_release = bool(
        predicate in {{"on", "in_or_on"}}
        and ("rack" in str(target_name).lower() or "caddy" in str(target_name).lower())
    )
    used_thin_caddy_release = bool(
        predicate == "in"
        and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower())
    )
    used_drawer_release = bool(
        predicate == "in"
        and ("drawer" in str(target_name).lower() or "cabinet" in str(target_name).lower())
    )
    drawer_precondition_failure = None
    if used_drawer_release:
        try:
            target_text_for_drawer = str(target_name).lower()
            if "bottom" in target_text_for_drawer:
                drawer_level = "bottom"
            elif "top" in target_text_for_drawer:
                drawer_level = "top"
            else:
                drawer_level = "middle"
            drawer_progress = get_cabinet_drawer_progress(drawer_level)
            drawer_open_fraction = float(drawer_progress.get("open_fraction", 0.0) if isinstance(drawer_progress, dict) else drawer_progress)
            drawer_is_closed = bool(isinstance(drawer_progress, dict) and drawer_progress.get("bddl_close", False))
            if drawer_is_closed or drawer_open_fraction < 0.72:
                drawer_precondition_failure = {{
                    "level": drawer_level,
                    "progress": drawer_progress,
                    "open_fraction": drawer_open_fraction,
                    "is_closed": drawer_is_closed,
                    "reason": "drawer is not physically open enough for insertion",
                }}
        except Exception as drawer_pre_exc:
            drawer_precondition_failure = {{"error": repr(drawer_pre_exc)}}
    drawer_prefilter = None
    if used_drawer_release and drawer_precondition_failure is None:
        try:
            goal_diag_before_release = get_bddl_goal_diagnostics()
            goals_before_release = goal_diag_before_release.get("goal_state", []) if isinstance(goal_diag_before_release, dict) else []
            for goal in goals_before_release:
                if not isinstance(goal, dict):
                    continue
                raw = goal.get("raw", [])
                if len(raw) != 3 or str(raw[0]).lower() != "in":
                    continue
                if object_name not in str(raw[1]).lower() and str(raw[1]).lower() not in object_name:
                    continue
                diag = goal.get("diagnostic", {{}})
                if not isinstance(diag, dict):
                    continue
                axes = diag.get("aabb_inside_axes", [])
                xy_inside = bool(len(axes) >= 2 and bool(axes[0]) and bool(axes[1]))
                z_excess = float(diag.get("aabb_z_excess", 0.0) or 0.0)
                z_deficit = float(diag.get("aabb_z_deficit", 0.0) or 0.0)
                contact = bool(diag.get("contact", False))
                if xy_inside and contact and z_deficit <= 0.002 and 0.0 < z_excess <= 0.020:
                    drawer_prefilter = {{
                        "target_name": str(goal.get("target_name") or raw[2]),
                        "diagnostic": diag,
                        "z_excess": z_excess,
                    }}
                    break
        except Exception as prefilter_exc:
            drawer_prefilter = {{"error": repr(prefilter_exc)}}
    if isinstance(drawer_precondition_failure, dict):
        print("Drawer insertion precondition failed before low release:", drawer_precondition_failure)
        release = {{
            "object_name": object_name,
            "target": {{"predicate": predicate, "target_name": target_name}},
            "before": {{}},
            "after": {{}},
            "relation": {{}},
            "task_status": get_task_status(),
            "relation_satisfied": False,
            "needs_repair": True,
            "drawer_precondition_failure": drawer_precondition_failure,
            "reason": "drawer insertion precondition failed; reopen drawer before placing object",
            "skill": "low_release_object_to_target_precondition",
        }}
    elif isinstance(drawer_prefilter, dict) and drawer_prefilter.get("diagnostic") is not None:
        print("Drawer z-only near-success detected before low release; running bounded z-only settle.", drawer_prefilter)
        release = drawer_z_only_settle_object(
            object_name,
            target_name=drawer_prefilter.get("target_name") or target_name or None,
            max_attempts=2,
            max_z_excess=0.020,
            settle_steps=26,
        )
    elif used_microwave_release:
        release = place_object_in_microwave_region(
            object_name,
            target_name=target_name or "microwave",
        )
    elif used_rack_release:
        release = rack_supported_release_object_to_target(
            object_name,
            target_name=target_name or None,
        )
    elif used_thin_caddy_release:
        caddy_hold = measure_object_grasp_state(object_name)
        if not bool(isinstance(caddy_hold, dict) and caddy_hold.get("held", False)):
            release = low_release_object_to_target(
                object_name,
                predicate=predicate,
                target_name=target_name or None,
                settle_steps=12,
            )
            if isinstance(release, dict):
                release["before"] = caddy_hold
                release["needs_repick"] = True
                release["reason"] = (
                    "object is not verified-held before caddy release; "
                    "return control to the pick_lift stage for reacquire-transport-release"
                )
        else:
            release = low_release_object_to_target(
                object_name,
                predicate=predicate,
                target_name=target_name or None,
                settle_steps=40,
            )
    else:
        release = low_release_object_to_target(
            object_name,
            predicate=predicate,
            target_name=target_name or None,
            require_drawer_closing_clearance=bool(require_drawer_closing_clearance),
        )
    if used_drawer_release and isinstance(drawer_precondition_failure, dict):
        print("Drawer release precondition result:", release)
    elif used_microwave_release and isinstance(release, dict):
        print("Microwave release result:", release)
    elif used_rack_release and isinstance(release, dict):
        print("Rack release summary:", release.get("summary", release))
    elif used_thin_caddy_release and isinstance(release, dict):
        print("Thin caddy low release result:", release)
    elif isinstance(drawer_prefilter, dict) and drawer_prefilter.get("diagnostic") is not None:
        print("Drawer z-only settle result:", release)
    else:
        print("Low release result:", release)
    effective_predicate = str(release.get("target", {{}}).get("predicate", predicate)) if isinstance(release, dict) else predicate
    preserved_drawer_z_only = bool(
        isinstance(release, dict)
        and release.get("z_only_near_success_preserved", False)
    )
    satisfied = bool(isinstance(release, dict) and release.get("relation_satisfied"))
    microwave_near_inside_z_only = False
    if used_microwave_release and isinstance(release, dict) and not satisfied:
        final_relation_for_microwave = release.get("final_relation") or release.get("relation") or {{}}
        if isinstance(final_relation_for_microwave, dict):
            try:
                local_delta = final_relation_for_microwave.get("local_delta")
                site_size = final_relation_for_microwave.get(
                    "site_size_abs",
                    final_relation_for_microwave.get("site_size"),
                )
                z_excess = float(final_relation_for_microwave.get("aabb_z_excess", 0.0) or 0.0)
                microwave_near_inside_z_only = bool(
                    final_relation_for_microwave.get("contact", False)
                    and len(local_delta) >= 3
                    and len(site_size) >= 3
                    and abs(float(local_delta[0])) <= max(abs(float(site_size[0])) + 0.018, 0.060)
                    and abs(float(local_delta[1])) <= max(abs(float(site_size[1])) + 0.018, 0.060)
                    and 0.0 < z_excess <= 0.025
                )
            except Exception:
                microwave_near_inside_z_only = False
    drawer_clearance = release.get("drawer_closing_clearance", None) if isinstance(release, dict) else None
    if (
        satisfied
        and bool(require_drawer_closing_clearance)
        and effective_predicate == "in"
        and isinstance(drawer_clearance, dict)
        and not bool(drawer_clearance.get("has_clearance", False))
    ):
        print("Release satisfies symbolic In/contact; downstream drawer-close clearance is a warning for the close stage:", drawer_clearance)
    repair = None
    if (not satisfied) and used_microwave_release:
        print("Microwave interior release did not satisfy relation; keeping failure local to microwave primitive.")
    elif (not satisfied) and used_rack_release:
        print("Rack/caddy release did not satisfy relation; keeping failure local to rack primitive.")
    elif (not satisfied) and used_thin_caddy_release:
        print("Thin caddy low release did not satisfy relation; entering caddy-specific local closure.")
    drawer_relation_for_repair = {{}}
    if used_drawer_release and isinstance(release, dict):
        candidate_relation = release.get("final_relation") or release.get("relation") or release.get("after") or {{}}
        if isinstance(candidate_relation, dict):
            drawer_relation_for_repair = candidate_relation
    drawer_clearance_locally_repairable = bool(
        used_drawer_release
        and bool(require_drawer_closing_clearance)
        and isinstance(drawer_clearance, dict)
        and not bool(drawer_clearance.get("has_clearance", False))
        and isinstance(drawer_relation_for_repair, dict)
        and (
            bool(drawer_relation_for_repair.get("satisfied", False))
            or bool(drawer_relation_for_repair.get("in", False))
            or (
                bool(drawer_relation_for_repair.get("contact", False))
                and "local_delta" in drawer_relation_for_repair
            )
        )
    )
    drawer_min_front_clearance_for_repair = None
    if isinstance(drawer_clearance, dict):
        try:
            drawer_min_front_clearance_for_repair = float(drawer_clearance.get("min_front_clearance"))
        except Exception:
            drawer_min_front_clearance_for_repair = None
    if (
        (not satisfied)
        and used_drawer_release
        and not isinstance(drawer_precondition_failure, dict)
        and drawer_clearance_locally_repairable
    ):
        print("Drawer low release is inside/near the box but lacks closing-clearance margin; preserve placement and let close stage verify.", {{
            "relation": drawer_relation_for_repair,
            "drawer_clearance": drawer_clearance,
        }})
        repair = {{
            "relation_satisfied": True,
            "needs_replan_deeper_held_release": False,
            "reason": "drawer_close_clearance_warning_defer_to_close_stage",
            "initial_relation": drawer_relation_for_repair,
            "final_relation": drawer_relation_for_repair,
            "drawer_closing_clearance": drawer_clearance,
            "near_target": True,
            "xy_distance_to_target": None,
            "task_status": release.get("task_status", get_task_status()) if isinstance(release, dict) else get_task_status(),
        }}
        satisfied = True
    elif (
        (not satisfied)
        and used_drawer_release
        and not isinstance(drawer_precondition_failure, dict)
    ):
        relation_for_replan = release.get("relation", {{}}) if isinstance(release, dict) else {{}}
        print("Drawer low release failed; preserving state and returning relation feedback instead of unheld local closure.", relation_for_replan)
        repair = {{
            "relation_satisfied": False,
            "reason": "drawer_low_release_failed_preserve_state_for_repick",
            "initial_relation": relation_for_replan,
            "final_relation": relation_for_replan,
            "task_status": release.get("task_status", get_task_status()) if isinstance(release, dict) else get_task_status(),
        }}
    elif (
        (not satisfied)
        and not used_microwave_release
        and not used_rack_release
        and not (
            used_thin_caddy_release
            and isinstance(release, dict)
            and bool(release.get("needs_repick", False))
        )
        and not isinstance(drawer_precondition_failure, dict)
    ):
        before = release.get("before", {{}}) if isinstance(release, dict) else {{}}
        print("Low release not satisfied; entering bounded local placement closure.", before)
        skip_closure_for_preservation = False
        if effective_predicate == "on" and ("stove" in str(target_name).lower() or "cook_region" in str(target_name).lower()):
            try:
                goal_diag_for_closure = get_bddl_goal_diagnostics()
                placement_goals_for_closure = goal_diag_for_closure.get("goal_state", []) if isinstance(goal_diag_for_closure, dict) else []
                skip_closure_for_preservation = bool(
                    any(
                        isinstance(goal, dict)
                        and len(goal.get("raw", [])) == 3
                        and str((goal.get("raw") or [""])[0]).lower() == "on"
                        and bool(goal.get("satisfied", False))
                        for goal in placement_goals_for_closure
                    )
                )
            except Exception:
                skip_closure_for_preservation = False
        if skip_closure_for_preservation:
            print("Skipping local placement closure to preserve already satisfied shared-region placement goals.")
            repair = {{"relation_satisfied": False, "reason": "preserve_shared_region_goals"}}
            satisfied = False
        else:
            closure_kwargs = {{}}
            if effective_predicate == "in" and "basket" in str(target_name).lower():
                try:
                    affordance = get_object_grasp_affordance(object_name)
                    extent = affordance.get("geometry", {{}}).get("extent", [0.0, 0.0, 0.0])
                    max_xy = max(abs(float(extent[0])), abs(float(extent[1])))
                    z_extent = abs(float(extent[2]))
                    large_box_like = bool(max_xy >= 0.150 and z_extent <= max_xy + 0.008)
                except Exception:
                    large_box_like = False
                closure_kwargs = {{"near_tolerance": 0.18, "max_iters": 3}} if large_box_like else {{"max_iters": 2}}
            elif used_drawer_release:
                closure_kwargs = {{"near_tolerance": 0.32, "max_iters": 2, "settle_steps": 28}}
            elif effective_predicate == "in" and ("caddy" in str(target_name).lower() or "compartment" in str(target_name).lower()):
                closure_kwargs = {{"near_tolerance": 0.24, "max_iters": 4, "settle_steps": 32}}
            else:
                closure_kwargs = {{"max_iters": 2}}
            repair = local_placement_closure(
                object_name,
                predicate=effective_predicate,
                target_name=target_name or None,
                require_drawer_closing_clearance=(
                    bool(require_drawer_closing_clearance) and not bool(used_drawer_release)
                ),
                **closure_kwargs,
            )
            print("Local placement closure result:", repair)
            satisfied = bool(isinstance(repair, dict) and repair.get("relation_satisfied"))
            if isinstance(repair, dict) and repair.get("drawer_closing_clearance") is not None:
                drawer_clearance = repair.get("drawer_closing_clearance")
    task_status = release.get("task_status", {{}}) if isinstance(release, dict) else {{}}
    if isinstance(repair, dict):
        task_status = repair.get("task_status", task_status)
    if (
        used_drawer_release
        and bool(require_drawer_closing_clearance)
        and not isinstance(drawer_precondition_failure, dict)
    ):
        def _stage_drawer_clearance_from_relation(relation, min_front_clearance=0.110, min_lateral_clearance=0.008):
            try:
                local_delta = relation.get("local_delta")
                site_size_abs = relation.get("site_size_abs", relation.get("site_size"))
                if local_delta is None or site_size_abs is None or len(local_delta) < 3 or len(site_size_abs) < 3:
                    return {{"has_clearance": False, "failure_reasons": ["missing_local_geometry"]}}
                front_clearance = float(local_delta[1]) + abs(float(site_size_abs[1]))
                lateral_clearance = min(
                    float(local_delta[0]) + abs(float(site_size_abs[0])),
                    abs(float(site_size_abs[0])) - float(local_delta[0]),
                )
                failure_reasons = []
                if front_clearance < float(min_front_clearance):
                    failure_reasons.append("insufficient_front_clearance_for_later_close")
                if lateral_clearance < float(min_lateral_clearance):
                    failure_reasons.append("insufficient_lateral_clearance_for_later_close")
                return {{
                    "has_clearance": not failure_reasons,
                    "failure_reasons": failure_reasons,
                    "front_clearance": front_clearance,
                    "lateral_clearance": lateral_clearance,
                    "min_front_clearance": float(min_front_clearance),
                    "min_lateral_clearance": float(min_lateral_clearance),
                    "local_delta": local_delta,
                    "site_size_abs": site_size_abs,
                    "in": relation.get("in"),
                }}
            except Exception as clearance_exc:
                return {{"has_clearance": False, "failure_reasons": ["clearance_eval_error"], "error": repr(clearance_exc)}}

        final_drawer_relation = {{}}
        if isinstance(repair, dict):
            candidate_relation = repair.get("final_relation") or repair.get("relation") or {{}}
            if isinstance(candidate_relation, dict):
                final_drawer_relation = candidate_relation
        if not final_drawer_relation and isinstance(release, dict):
            candidate_relation = release.get("final_relation") or release.get("relation") or {{}}
            if isinstance(candidate_relation, dict):
                final_drawer_relation = candidate_relation
        if final_drawer_relation:
            try:
                min_front = drawer_min_front_clearance_for_repair
                if min_front is None and isinstance(drawer_clearance, dict):
                    min_front = float(drawer_clearance.get("min_front_clearance"))
                if min_front is None:
                    min_front = 0.085 if "bottom" in str(target_name).lower() else 0.065
                verified_drawer_clearance = _stage_drawer_clearance_from_relation(
                    final_drawer_relation,
                    min_front_clearance=float(min_front),
                )
                drawer_clearance = verified_drawer_clearance
                if not bool(verified_drawer_clearance.get("has_clearance", False)):
                    print("Final drawer placement still lacks close-clearance margin; preserving In/contact and deferring to drawer close stage.", verified_drawer_clearance)
            except Exception as clearance_gate_exc:
                print("Final drawer clearance gate failed:", repr(clearance_gate_exc))
                satisfied = False
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    global_repair = None
    if is_final_stage and satisfied and (not completed) and reward < 1.0:
        try:
            goal_diag = get_bddl_goal_diagnostics()
            all_goals = bool(isinstance(goal_diag, dict) and goal_diag.get("all_goals_satisfied", False))
            placement_goals = goal_diag.get("goal_state", []) if isinstance(goal_diag, dict) else []
            has_protected_placement_goal = bool(
                sum(
                    1
                    for goal in placement_goals
                    if (
                        isinstance(goal, dict)
                        and len(goal.get("raw", [])) == 3
                        and str((goal.get("raw") or [""])[0]).lower() in {"in", "on"}
                        and bool(goal.get("satisfied", False))
                    )
                )
                >= 1
            )
        except Exception as goal_exc:
            print("Global BDDL diagnostics unavailable after final release:", goal_exc)
            all_goals = False
            has_protected_placement_goal = False
        if not all_goals and has_protected_placement_goal:
            print("Skipping global physical repair to preserve already satisfied placement goals.")
        elif not all_goals:
            print("Final release completed locally but global BDDL goal is not complete; entering global physical repair.")
            global_repair = global_bddl_goal_repair(max_passes=2, binary_max_iters=1, settle_steps=35)
            print("Global BDDL repair after final release:", global_repair)
            if isinstance(global_repair, dict):
                task_status = global_repair.get("task_status", task_status)
                reward = float(task_status.get("reward", reward) or reward)
                completed = bool(task_status.get("task_completed", completed))
    status = "satisfied" if satisfied or completed or reward >= 1.0 else "needs_repair"
    closure_near = repair.get("near_target", None) if isinstance(repair, dict) else None
    closure_xy = repair.get("xy_distance_to_target", None) if isinstance(repair, dict) else None
    closure_relation = None
    repair_reason_for_evidence = None
    needs_replan_deeper_held_release = False
    needs_repick = bool(isinstance(release, dict) and release.get("needs_repick", False))
    if isinstance(repair, dict):
        repair_reason_for_evidence = repair.get("reason")
        needs_replan_deeper_held_release = bool(repair.get("needs_replan_deeper_held_release", False))
        needs_repick = bool(needs_repick or repair.get("needs_repick", False))
        relation_for_evidence = repair.get("final_relation") or repair.get("initial_relation") or {{}}
        if isinstance(relation_for_evidence, dict):
            closure_relation = {{
                "satisfied": relation_for_evidence.get("satisfied"),
                "on": relation_for_evidence.get("on"),
                "in": relation_for_evidence.get("in"),
                "contact": relation_for_evidence.get("contact"),
                "local_delta": relation_for_evidence.get("local_delta"),
                "site_size_abs": relation_for_evidence.get("site_size_abs", relation_for_evidence.get("site_size")),
                "xy_distance": relation_for_evidence.get("xy_distance"),
            }}
    rack_summary = release.get("summary", None) if used_rack_release and isinstance(release, dict) else None
    rack_evidence = ""
    if isinstance(rack_summary, dict):
        final = rack_summary.get("final", {{}})
        phase_diags = rack_summary.get("phase_diags", {{}})
        compact_phases = {{}}
        for phase_name in ("pre_open", "pre_release_rail_kiss", "release_open_no_retreat", "post_release_settle", "skip_micro_lift_after_unsatisfied_release"):
            diag = phase_diags.get(phase_name, {{}}) if isinstance(phase_diags, dict) else {{}}
            if isinstance(diag, dict):
                compact_phases[phase_name] = {{
                    "on": diag.get("on"),
                    "in_xy": diag.get("in_xy"),
                    "in_z": diag.get("in_z"),
                    "contact": diag.get("contact"),
                    "local_delta": diag.get("local_delta"),
                    "site_size": diag.get("site_size"),
                }}
        rack_evidence = ", rack_summary=" + str({{"final": final, "phases": compact_phases}})
    global_evidence = ""
    if isinstance(global_repair, dict):
        global_evidence = ", global_repair=" + str({{
            "success": global_repair.get("success"),
            "reason": global_repair.get("reason"),
            "record_count": len(global_repair.get("records", [])) if isinstance(global_repair.get("records"), list) else None,
            "task_status": global_repair.get("task_status"),
        }})
    microwave_evidence = ""
    if used_microwave_release and isinstance(release, dict):
        final_relation = release.get("final_relation", {{}})
        if isinstance(final_relation, dict):
            microwave_evidence = ", microwave_relation=" + str({{
                "satisfied": final_relation.get("satisfied"),
                "in": final_relation.get("in"),
                "contact": final_relation.get("contact"),
                "local_delta": final_relation.get("local_delta"),
                "site_size_abs": final_relation.get("site_size_abs", final_relation.get("site_size")),
            }})
        microwave_records = release.get("records", [])
        if isinstance(microwave_records, list):
            microwave_phase_tail = []
            for item in microwave_records[-6:]:
                if not isinstance(item, dict):
                    continue
                entry = {{"phase": item.get("phase")}}
                seating = item.get("seating")
                if isinstance(seating, dict):
                    subactions = []
                    for rec in seating.get("records", [])[-3:]:
                        if not isinstance(rec, dict):
                            continue
                        subactions.append({{
                            "action": rec.get("action"),
                            "z_high_margin": rec.get("z_high_margin"),
                            "improved_z": rec.get("improved_z"),
                        }})
                    entry["seating_actions"] = subactions
                microwave_phase_tail.append(entry)
            microwave_evidence += ", microwave_record_phases=" + str(microwave_phase_tail)
    evidence = (
        f"relation_satisfied={{satisfied}}, microwave_near_inside_z_only={{microwave_near_inside_z_only}}, preserved_drawer_z_only={{preserved_drawer_z_only}}, require_drawer_closing_clearance={{require_drawer_closing_clearance}}, "
        f"drawer_clearance={{drawer_clearance}}, reward={{reward:.3f}}, task_completed={{completed}}, "
        f"closure_near={{closure_near}}, closure_xy={{closure_xy}}, "
        f"closure_relation={{closure_relation}}, repair_reason={{repair_reason_for_evidence}}, "
        f"needs_replan_deeper_held_release={{needs_replan_deeper_held_release}}, needs_repick={{needs_repick}}, "
        f"object={{object_name}}, target={{target_name}}, predicate={{effective_predicate}}{{rack_evidence}}{{microwave_evidence}}{{global_evidence}}"
    )
    if status == "satisfied":
        repair_hint = ""
    elif used_microwave_release:
        repair_hint = (
            "Microwave insertion is xy-valid but still above the heating-region z bound; preserve this failure evidence and retry only z/low-release correction."
            if microwave_near_inside_z_only
            else "Microwave insertion failed; keep repair local to microwave insertion/seating before any close stage."
        )
    elif used_drawer_release and isinstance(drawer_precondition_failure, dict):
        repair_hint = "Drawer is closed or too shallow for insertion; reopen/pull the drawer farther before retrying target-object placement."
    elif used_drawer_release and isinstance(drawer_clearance, dict) and not bool(drawer_clearance.get("has_clearance", False)):
        repair_hint = "Drawer front clearance is insufficient after In/contact; stop bounded local seating and re-grasp the bowl for a deeper held drawer release before closing."
    elif used_drawer_release:
        repair_hint = "Drawer insertion failed; preserve the failed relation evidence and retry with a held target-object insertion instead of unheld local closure."
    elif used_thin_caddy_release and isinstance(release, dict) and bool(release.get("needs_repick", False)):
        repair_hint = "Object is not verified-held before caddy release; return control to the pick_lift stage, reacquire the object, then redo guarded transport and low release."
    else:
        repair_hint = "Release/local closure did not satisfy target relation; continue bounded local physical closure before any re-pick."
except Exception as exc:
    status = "needs_repair"
    evidence = f"low release fast-path exception: {{exc}}"
    repair_hint = "Retry release after checking target grounding and held object state."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair_hint,
}}, default=str))
'''


def _drawer_fast_path_code(stage: Any, scene_snapshot: dict[str, Any] | None = None, *, stage_attempt: int = 1) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "_drawer_" not in stage_id:
        return ""
    level = _drawer_level_from_stage_id(stage_id)
    task_text = str((scene_snapshot or {}).get("task", "")).lower()
    placement_drawer_task = bool("drawer" in task_text and any(token in task_text for token in ("put", "place", "inside", "in it")))
    open_threshold = 0.96 if placement_drawer_task and level == "bottom" else 0.98
    if "ground_handle" in stage_id:
        return _drawer_ground_fast_path_code(stage_id, level)
    if "sleeve_or_hook_handle" in stage_id:
        return _drawer_sleeve_fast_path_code(
            stage_id,
            level,
            open_threshold=open_threshold,
            require_container_clearance=placement_drawer_task,
        )
    if "test_progress" in stage_id:
        return _drawer_test_fast_path_code(
            stage_id,
            level,
            open_threshold=open_threshold,
            require_container_clearance=placement_drawer_task,
        )
    if "pull_open" in stage_id:
        return _drawer_pull_fast_path_code(
            stage_id,
            level,
            open_threshold=open_threshold,
            require_container_clearance=placement_drawer_task,
        )
    if "close_guarded" in stage_id:
        return _drawer_close_fast_path_code(stage_id, level, stage_attempt=stage_attempt)
    return ""


def _stove_fast_path_code(stage: Any) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if not stage_id.startswith("stove_"):
        return ""
    if "ground_button" in stage_id:
        return f'''import json

stage_id = {stage_id!r}
try:
    state = get_object_state_diagnostic("flat_stove_1")
    button_pos, button_quat = get_body_pose("flat_stove_1_button")
    status = "satisfied"
    evidence = f"stove_state={{state}}, button_pos={{button_pos}}, button_quat={{button_quat}}"
    repair = ""
except Exception as exc:
    status = "needs_repair"
    evidence = f"stove grounding exception: {{exc}}"
    repair = "Verify flat_stove and button body names before pressing."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''
    if "press_and_verify" in stage_id:
        return f'''import json

stage_id = {stage_id!r}
try:
    result = turn_on_stove_by_sweep()
    status_info = get_task_status()
    reward = float(status_info.get("reward", 0.0) or 0.0)
    completed = bool(status_info.get("task_completed", False))
    turned_on = bool(isinstance(result, dict) and result.get("turned_on", False))
    status = "satisfied" if turned_on or completed or reward >= 1.0 else "needs_repair"
    evidence = f"turned_on={{turned_on}}, reward={{reward:.3f}}, task_completed={{completed}}, result={{result}}"
    repair = "" if status == "satisfied" else "Stove sweep did not turn on the stove; retry with alternate contact sweep."
except Exception as exc:
    status = "needs_repair"
    evidence = f"stove press fast-path exception: {{exc}}"
    repair = "Retry stove button sweep after checking button grounding."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''
    return ""


def _push_region_fast_path_code(stage: Any) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    if "push_to_region_verify" not in stage_id:
        return ""
    object_name = _resolve_stage_object_name(stage, {})
    if not object_name:
        object_name = _object_name_from_stage_id(stage_id, "_push_to_region_verify")
    if not object_name:
        return ""
    target_ref = _resolve_stage_target_reference(stage)
    predicate = _resolve_stage_predicate(stage, default="on")
    if predicate not in {"on", "in"}:
        predicate = "on"
    return f'''import json

stage_id = {stage_id!r}
object_name = {object_name!r}
target_name = {target_ref!r}
predicate = {predicate!r}

try:
    before_goal = get_bddl_goal_diagnostics()
    print("Push-region BDDL before:", before_goal)
    result = push_object_to_target_region(
        object_name,
        target_name=target_name or None,
        predicate=predicate,
        max_iters=5,
        settle_steps=55,
    )
    print("Push-region result:", result)
    after_goal = get_bddl_goal_diagnostics()
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    relation = result.get("final_relation", {{}}) if isinstance(result, dict) else {{}}
    best_relation = result.get("best_relation", {{}}) if isinstance(result, dict) else {{}}
    relation_satisfied = bool(isinstance(result, dict) and result.get("relation_satisfied", False))
    all_goals = bool(isinstance(after_goal, dict) and after_goal.get("all_goals_satisfied", False))
    success = bool(relation_satisfied or completed or reward >= 1.0 or all_goals)
    records = result.get("records", []) if isinstance(result, dict) else []
    last_record = records[-1] if isinstance(records, list) and records else {{}}
    status = "satisfied" if success else "needs_repair"
    evidence = (
        f"success={{success}}, relation_satisfied={{relation_satisfied}}, reward={{reward:.3f}}, "
        f"task_completed={{completed}}, all_goals={{all_goals}}, object={{result.get('object_name', object_name) if isinstance(result, dict) else object_name}}, "
        f"target={{result.get('target_name', target_name) if isinstance(result, dict) else target_name}}, predicate={{predicate}}, "
        f"final_relation={{relation}}, best_relation={{best_relation}}, last_record={{last_record}}, reason={{result.get('reason') if isinstance(result, dict) else None}}"
    )
    repair = "" if success else "Push-region relation remains unsatisfied; retry with refreshed local_delta/contact side or re-ground target region."
except Exception as exc:
    status = "needs_repair"
    evidence = f"push-region fast-path exception: {{exc}}"
    repair = "Check object alias, target region grounding, and push_object_to_target_region availability."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _microwave_fast_path_code(stage: Any) -> str:
    stage_id = str(getattr(stage, "stage_id", ""))
    text = " ".join(
        [
            stage_id,
            str(getattr(stage, "name", "")),
            str(getattr(stage, "intent", "")),
            " ".join(str(ref) for ref in getattr(stage, "target_references", [])),
        ]
    ).lower()
    if "microwave" not in text or "close" not in text:
        return ""
    return f'''import json

stage_id = {stage_id!r}
try:
    goal_diag_before = get_bddl_goal_diagnostics()
    placement_goal = None
    if isinstance(goal_diag_before, dict):
        for goal in goal_diag_before.get("goal_state", []):
            if not isinstance(goal, dict):
                continue
            raw = goal.get("raw", [])
            if (
                len(raw) == 3
                and str(raw[0]).lower() == "in"
                and "microwave" in str(raw[2]).lower()
            ):
                placement_goal = goal
                break
    placement_satisfied = bool(
        isinstance(placement_goal, dict)
        and placement_goal.get("satisfied", False)
    )
    placement_diag = placement_goal.get("diagnostic", {{}}) if isinstance(placement_goal, dict) else {{}}
    near_inside_z_only = False
    if not placement_satisfied and isinstance(placement_diag, dict):
        try:
            local_delta = placement_diag.get("local_delta")
            site_size = placement_diag.get("site_size_abs", placement_diag.get("site_size"))
            z_excess = float(placement_diag.get("aabb_z_excess", 0.0) or 0.0)
            near_inside_z_only = bool(
                placement_diag.get("contact", False)
                and len(local_delta) >= 3
                and len(site_size) >= 3
                and abs(float(local_delta[0])) <= max(abs(float(site_size[0])) + 0.018, 0.060)
                and abs(float(local_delta[1])) <= max(abs(float(site_size[1])) + 0.018, 0.060)
                and 0.0 < z_excess <= 0.025
            )
        except Exception:
            near_inside_z_only = False
    if not placement_satisfied and not near_inside_z_only:
        status = "needs_repair"
        evidence = (
            f"close_precondition_failed=True, placement_satisfied={{placement_satisfied}}, "
            f"near_inside_z_only={{near_inside_z_only}}, placement_goal={{placement_goal}}, placement_diag={{placement_diag}}"
        )
        repair = "Do not close the microwave yet; retry microwave insertion/local seating until the mug is inside the heating region."
    else:
        before = get_object_state_diagnostic("microwave_1")
        result = close_microwave_by_sweep("microwave_1")
        after = get_object_state_diagnostic("microwave_1")
        task_status = get_task_status()
        reward = float(task_status.get("reward", 0.0) or 0.0)
        completed = bool(task_status.get("task_completed", False))
        closed = bool(after.get("is_close", False)) or bool(isinstance(result, dict) and result.get("closed", False))
        status = "satisfied" if closed or completed or reward >= 1.0 else "needs_repair"
        evidence = (
            f"close_precondition_failed=False, placement_satisfied={{placement_satisfied}}, "
            f"near_inside_z_only={{near_inside_z_only}}, "
            f"closed={{closed}}, reward={{reward:.3f}}, task_completed={{completed}}, "
            f"placement_diag={{placement_diag}}, before={{before}}, after={{after}}, result={{result}}"
        )
        repair = "" if status == "satisfied" else "Microwave close sweep did not close the door; retry with another handle sweep."
except Exception as exc:
    status = "needs_repair"
    evidence = f"microwave close fast-path exception: {{exc}}"
    repair = "Retry after checking microwave door body and readonly state diagnostics."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _drawer_level_from_stage_id(stage_id: str) -> str:
    text = str(stage_id).lower()
    for level in ("top", "middle", "bottom"):
        if text.startswith(f"{level}_drawer") or f"_{level}_drawer" in text:
            return level
    return "middle"


def _drawer_level_from_stage_context(stage: Any) -> str | None:
    joined = " ".join(
        [str(getattr(stage, "stage_id", "")), str(getattr(stage, "name", "")), str(getattr(stage, "intent", ""))]
        + [str(ref) for ref in getattr(stage, "target_references", [])]
    ).lower().replace(" ", "_")
    for level in ("top", "middle", "bottom"):
        if f"{level}_drawer" in joined or f"{level}_cabinet" in joined:
            return level
    return None


def _suffix_requires_drawer_close(plan: Any, current_stage_index: int, level: str) -> bool:
    """Whether later stages require closing the same drawer level."""
    stages = list(getattr(plan, "stages", []) or [])
    level_name = str(level).lower() if level is not None else None
    saw_drawer_close = False
    for later_stage in stages[int(current_stage_index) + 1 :]:
        stage_id = str(getattr(later_stage, "stage_id", "")).lower()
        if "close_guarded" not in stage_id or "_drawer_" not in stage_id:
            continue
        saw_drawer_close = True
        if level_name is None or _drawer_level_from_stage_id(stage_id) == level_name:
            return True
    return bool(saw_drawer_close and level_name is None)


def _drawer_ground_fast_path_code(stage_id: str, level: str) -> str:
    return f'''import json
import math

stage_id = {stage_id!r}
level = {level!r}

try:
    handle_pos, hook_quat, pull_dir = get_cabinet_drawer_handle_pose(level)
    progress = get_cabinet_drawer_progress(level)
    hooks = get_drawer_sleeve_hook_poses(level)
    open_fraction = float(progress.get("open_fraction", 0.0) if isinstance(progress, dict) else progress)
    handle_ok = all(math.isfinite(float(x)) for x in handle_pos)
    pull_norm = math.sqrt(sum(float(x) * float(x) for x in pull_dir))
    hook_count = len(hooks.get("sleeve_candidates", [])) if isinstance(hooks, dict) else 0
    status = "satisfied" if handle_ok and abs(pull_norm - 1.0) <= 0.10 else "needs_repair"
    evidence = (
        f"level={{level}}, handle_ok={{handle_ok}}, pull_norm={{pull_norm:.3f}}, "
        f"open_fraction={{open_fraction:.3f}}, sleeve_candidates={{hook_count}}"
    )
    repair = "" if status == "satisfied" else "Drawer handle grounding failed; verify level and readonly drawer diagnostics."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer ground fast-path exception: {{exc}}"
    repair = "Retry drawer grounding with the correct drawer level."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _drawer_sleeve_fast_path_code(
    stage_id: str,
    level: str,
    *,
    open_threshold: float = 0.98,
    require_container_clearance: bool = False,
) -> str:
    return f'''import json

stage_id = {stage_id!r}
level = {level!r}
open_threshold = {float(open_threshold)!r}
require_container_clearance = {bool(require_container_clearance)!r}

try:
    progress = get_cabinet_drawer_progress(level)
    open_fraction = float(progress.get("open_fraction", 0.0) if isinstance(progress, dict) else progress)
    skip_threshold = open_threshold if require_container_clearance else 0.45
    if open_fraction >= skip_threshold:
        status = "satisfied"
        evidence = (
            f"level={{level}}, drawer_already_open=True, open_fraction={{open_fraction:.3f}}, "
            f"open_threshold={{open_threshold:.3f}}, require_container_clearance={{require_container_clearance}}"
        )
        repair = ""
    elif require_container_clearance and open_fraction >= 0.45:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=3,
            test_distance=0.035,
            pull_distance=0.18,
            strict_open_fraction=True,
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        success = bool(opened or after_fraction >= open_threshold or best >= open_threshold)
        status = "satisfied" if success else "needs_repair"
        evidence = (
            f"level={{level}}, partial_drawer_open=True, before={{open_fraction:.3f}}, "
            f"after={{after_fraction:.3f}}, best={{best:.3f}}, opened={{opened}}, "
            f"open_threshold={{open_threshold:.3f}}, require_container_clearance={{require_container_clearance}}, result={{pull}}"
        )
        repair = "" if success else "Drawer is only partially open; continue opening to create verified physical clearance before proceeding."
    else:
        plan = get_drawer_sleeve_hook_poses(level)
        quat = plan.get("hook_quaternion_wxyz")
        candidates = plan.get("sleeve_candidates", []) if isinstance(plan, dict) else []
        records = []
        open_gripper()
        prehook = plan.get("prehook_finger_position") if isinstance(plan, dict) else None
        mouth = plan.get("mouth_finger_position") if isinstance(plan, dict) else None
        if prehook is not None and quat is not None:
            records.append({{"action": "prehook", "move": goto_finger_midpoint_checked(prehook, quat, tolerance=0.080)}})
        if mouth is not None and quat is not None:
            records.append({{"action": "mouth", "move": goto_finger_midpoint_checked(mouth, quat, tolerance=0.070)}})
        best = None
        for idx, cand in enumerate(candidates[:7]):
            target = cand.get("finger_midpoint_position")
            cand_quat = cand.get("quaternion_wxyz", quat)
            if target is None or cand_quat is None:
                continue
            move = goto_finger_midpoint_checked(target, cand_quat, tolerance=0.070)
            align_open = measure_drawer_hook_alignment(level)
            ready = bool(align_open.get("ready_to_close") or align_open.get("loose_hook_geometry") or align_open.get("centered_between_fingers"))
            close_gripper()
            align_closed = measure_drawer_hook_alignment(level)
            contacts = measure_drawer_gripper_contacts(level)
            captured = bool(align_closed.get("closed_on_handle") or align_closed.get("likely_hooked") or contacts.get("has_handle_contact"))
            record = {{
                "candidate": cand.get("name", str(idx)),
                "move_arrived": bool(move.get("arrived", False)),
                "ready": ready,
                "captured": captured,
                "alignment": align_closed,
                "contacts": contacts,
            }}
            records.append(record)
            best = record
            if captured:
                break
            open_gripper()
        captured = bool(best and best.get("captured"))
        status = "satisfied" if captured else "needs_repair"
        evidence = f"level={{level}}, captured={{captured}}, attempts={{len(records)}}, best={{best}}"
        repair = "" if captured else "Handle was not captured; retry sleeve/hook with another candidate."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer sleeve fast-path exception: {{exc}}"
    repair = "Retry drawer handle capture from prehook/mouth pose."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _drawer_test_fast_path_code(
    stage_id: str,
    level: str,
    *,
    open_threshold: float = 0.98,
    require_container_clearance: bool = False,
) -> str:
    return f'''import json

stage_id = {stage_id!r}
level = {level!r}
open_threshold = {float(open_threshold)!r}
require_container_clearance = {bool(require_container_clearance)!r}

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        status = "satisfied"
        evidence = (
            f"level={{level}}, drawer_already_open=True, open_fraction={{before_fraction:.3f}}, "
            f"open_threshold={{open_threshold:.3f}}, require_container_clearance={{require_container_clearance}}, progress={{before}}"
        )
        repair = ""
    else:
        pull = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=5 if require_container_clearance else 3,
            test_distance=0.055,
            pull_distance=0.48 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.42),
            strict_open_fraction=bool(require_container_clearance),
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        delta = after_fraction - before_fraction
        contacts = measure_drawer_gripper_contacts(level)
        alignment = measure_drawer_hook_alignment(level)
        best = float(pull.get("best_open_fraction", after_fraction) if isinstance(pull, dict) else after_fraction)
        opened = bool(isinstance(pull, dict) and pull.get("opened", False))
        progressed = (
            bool(opened or best >= open_threshold or after_fraction >= open_threshold)
            if require_container_clearance
            else bool(opened or best >= open_threshold or delta >= 0.015 or after_fraction >= 0.12)
        )
        status = "satisfied" if progressed else "needs_repair"
        evidence = (
            f"level={{level}}, before={{before_fraction:.3f}}, after={{after_fraction:.3f}}, "
            f"delta={{delta:.3f}}, best={{best:.3f}}, opened={{opened}}, "
            f"bddl_open={{bool(isinstance(after, dict) and after.get('bddl_open', False))}}, "
            f"bddl_margin={{float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}}, "
            f"open_threshold={{open_threshold:.3f}}, require_container_clearance={{require_container_clearance}}, "
            f"handle_contact={{bool(contacts.get('has_handle_contact', False))}}, "
            f"closed_on_handle={{bool(alignment.get('closed_on_handle', False))}}, result={{pull}}"
        )
        repair = "" if progressed else "Drawer is not physically open enough for the downstream drawer objective; keep the handle captured and pull farther before proceeding."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer test fast-path exception: {{exc}}"
    repair = "Retry drawer progress test after handle capture."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _drawer_pull_fast_path_code(
    stage_id: str,
    level: str,
    *,
    open_threshold: float = 0.98,
    require_container_clearance: bool = False,
) -> str:
    return f'''import json

stage_id = {stage_id!r}
level = {level!r}
open_threshold = {float(open_threshold)!r}
require_container_clearance = {bool(require_container_clearance)!r}

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    pull_result = None
    before_bddl_open = bool(isinstance(before, dict) and before.get("bddl_open", False))
    already_open_enough = (
        bool(before_fraction >= open_threshold)
        if require_container_clearance
        else bool(before_bddl_open or before_fraction >= open_threshold)
    )
    if already_open_enough:
        after_fraction = before_fraction
        best = before_fraction
        after = before
    else:
        pull_result = open_drawer_by_sleeve_retry(
            level,
            open_threshold=open_threshold,
            max_cycles=6 if require_container_clearance else 4,
            test_distance=0.055,
            pull_distance=0.46 if require_container_clearance else (0.34 if open_threshold < 0.90 else 0.46),
            strict_open_fraction=require_container_clearance,
        )
        after = get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
        best = float(
            pull_result.get("best_open_fraction", pull_result.get("best_progress", after_fraction))
            if isinstance(pull_result, dict)
            else after_fraction
        )
    bddl_open = bool(isinstance(after, dict) and after.get("bddl_open", False))
    open_enough = (
        bool(max(after_fraction, best) >= open_threshold)
        if require_container_clearance
        else bool(bddl_open or max(after_fraction, best) >= open_threshold)
    )
    status = "satisfied" if open_enough else "needs_repair"
    evidence = (
        f"level={{level}}, before={{before_fraction:.3f}}, after={{after_fraction:.3f}}, "
        f"best={{best:.3f}}, bddl_open={{bddl_open}}, "
        f"bddl_margin={{float(after.get('bddl_open_margin', 0.0) if isinstance(after, dict) else 0.0):.3f}}, "
        f"open_threshold={{open_threshold:.3f}}, require_container_clearance={{require_container_clearance}}, "
        f"open_enough={{open_enough}}, result={{pull_result}}"
    )
    repair = "" if open_enough else "Drawer opening is insufficient for the downstream drawer objective; recapture the handle and pull farther with short verified strokes."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer pull-open fast-path exception: {{exc}}"
    repair = "Retry drawer pull after confirming handle capture."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _drawer_close_fast_path_code(stage_id: str, level: str, *, stage_attempt: int = 1) -> str:
    return f'''import json

stage_id = {stage_id!r}
level = {level!r}
stage_attempt = {int(stage_attempt)!r}

try:
    before = get_cabinet_drawer_progress(level)
    before_fraction = float(before.get("open_fraction", 0.0) if isinstance(before, dict) else before)
    result = close_drawer_by_push(
        level,
        max_iters=8,
        settle_steps=14,
        stage_bounded=True,
        stage_attempt=stage_attempt,
    )
    after = get_cabinet_drawer_progress(level)
    after_fraction = float(after.get("open_fraction", 0.0) if isinstance(after, dict) else after)
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    bddl_close = bool(isinstance(after, dict) and after.get("bddl_close", False))
    bddl_close_margin = float(after.get("bddl_close_margin", 0.0) if isinstance(after, dict) else 0.0)
    closed = bool(bddl_close or (isinstance(result, dict) and result.get("closed", False)))
    placement_goals_ok = True
    placement_goal_summary = []
    try:
        goal_diag = get_bddl_goal_diagnostics()
        goal_state = goal_diag.get("goal_state", []) if isinstance(goal_diag, dict) else []
        for goal in goal_state:
            if not isinstance(goal, dict):
                continue
            raw = goal.get("raw", [])
            predicate_name = str(raw[0]).lower() if raw else str(goal.get("predicate", "")).lower()
            if predicate_name not in {{"in", "on"}}:
                continue
            goal_ok = bool(goal.get("satisfied", False))
            placement_goals_ok = placement_goals_ok and goal_ok
            diag = goal.get("diagnostic", {{}})
            compact_diag = {{}}
            if isinstance(diag, dict):
                compact_diag = {{
                    "in": diag.get("in"),
                    "on": diag.get("on"),
                    "contact": diag.get("contact"),
                    "local_delta": diag.get("local_delta"),
                    "site_size_abs": diag.get("site_size_abs", diag.get("site_size")),
                    "aabb_z_excess": diag.get("aabb_z_excess"),
                    "aabb_z_deficit": diag.get("aabb_z_deficit"),
                }}
            placement_goal_summary.append({{
                "raw": raw,
                "satisfied": goal_ok,
                "diagnostic": compact_diag,
            }})
    except Exception as goal_exc:
        placement_goals_ok = True
        placement_goal_summary = [{{"diagnostic_error": repr(goal_exc)}}]
    attempt_count = int(result.get("attempt_count", len(result.get("records", []))) if isinstance(result, dict) else 0)
    reason = str(result.get("reason", "") if isinstance(result, dict) else "")
    best = float(result.get("best_open_fraction", after_fraction) if isinstance(result, dict) else after_fraction)
    best_margin = None
    if isinstance(result, dict):
        best_margin = result.get("best_bddl_close_margin", None)
    record_modes = []
    if isinstance(result, dict):
        for record in result.get("records", [])[-4:]:
            if isinstance(record, dict):
                mode = str(record.get("mode", "unknown"))
                p_before = record.get("progress_before") or {{}}
                p_after = record.get("progress_after") or record.get("progress_after_error") or {{}}
                before_margin = None
                before_fraction = None
                if isinstance(p_before, dict):
                    before_margin = p_before.get("bddl_close_margin", None)
                    before_fraction = p_before.get("open_fraction", None)
                margin = None
                fraction = None
                if isinstance(p_after, dict):
                    margin = p_after.get("bddl_close_margin", None)
                    fraction = p_after.get("open_fraction", None)
                spec = record.get("spec_name", "")
                followup_summary = ""
                followups = record.get("terminal_margin_followup_pulses", [])
                if isinstance(followups, list) and followups:
                    parts = []
                    for pulse in followups[-2:]:
                        if not isinstance(pulse, dict):
                            continue
                        pulse_before = pulse.get("progress_before") or {{}}
                        pulse_after = pulse.get("progress_after") or {{}}
                        if isinstance(pulse_before, dict) and isinstance(pulse_after, dict):
                            parts.append(
                                f"pulse{{pulse.get('pulse_index')}}:"
                                f"margin={{pulse_before.get('bddl_close_margin')}}->{{pulse_after.get('bddl_close_margin')}},"
                                f"fraction={{pulse_before.get('open_fraction')}}->{{pulse_after.get('open_fraction')}}"
                            )
                    if parts:
                        followup_summary = ";followup=" + "|".join(parts)
                record_modes.append(
                    f"{{mode}}:{{spec}}:margin={{before_margin}}->{{margin}},fraction={{before_fraction}}->{{fraction}}{{followup_summary}}"
                )
    status = "satisfied" if completed or reward >= 1.0 or (closed and placement_goals_ok) else "needs_repair"
    evidence = (
        f"level={{level}}, stage_attempt={{stage_attempt}}, before={{before_fraction:.3f}}, after={{after_fraction:.3f}}, "
        f"best={{best:.3f}}, best_margin={{best_margin}}, attempts={{attempt_count}}, reason={{reason}}, "
        f"closed={{closed}}, bddl_close={{bddl_close}}, bddl_close_margin={{bddl_close_margin:.3f}}, "
        f"reward={{reward:.3f}}, task_completed={{completed}}, placement_goals_ok={{placement_goals_ok}}, "
        f"placement_goal_summary={{placement_goal_summary}}, record_modes={{record_modes}}"
    )
    if status == "satisfied":
        repair = ""
    elif closed and not placement_goals_ok:
        repair = "Drawer closed but placement predicate failed after close; retry the preceding held target-object placement deeper inside the drawer before closing."
    else:
        repair = "Drawer not BDDL-closed yet; re-enter close stage from the updated drawer state with a short verified push."
except Exception as exc:
    status = "needs_repair"
    evidence = f"drawer close fast-path exception: {{exc}}"
    repair = "Retry after checking drawer level and readonly progress diagnostics."

print("STAGE_STATUS " + json.dumps({{
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}}, default=str))
'''


def _resolve_stage_object_name(stage: Any, snapshot: dict[str, Any]) -> str:
    objects = snapshot.get("objects_of_interest")
    if not isinstance(objects, dict) or not objects:
        return _semantic_object_ref_from_stage(stage)
    refs = [str(ref).lower().replace(" ", "_") for ref in getattr(stage, "target_references", [])]
    ignored_refs = _ignored_stage_reference_tokens()
    strict_refs = [
        ref
        for ref in refs
        if ref not in ignored_refs
    ]
    for ref in refs:
        if ref in ignored_refs:
            continue
        for name, obj_snapshot in objects.items():
            norm_name = str(name).lower()
            if ref == norm_name and _snapshot_has_grasp_affordance(obj_snapshot):
                return str(name)
        for name, obj_snapshot in objects.items():
            norm_name = str(name).lower()
            aliases = _SEMANTIC_STAGE_ALIASES.get(ref, ())
            if any(alias == norm_name for alias in aliases) and _snapshot_has_grasp_affordance(obj_snapshot):
                return str(name)
        for name, obj_snapshot in objects.items():
            norm_name = str(name).lower()
            if ref and _names_match_for_stage(ref, norm_name):
                if _snapshot_has_grasp_affordance(obj_snapshot):
                    return str(name)
        if ref:
            break
    if strict_refs:
        semantic_ref = _semantic_object_ref_from_stage(stage)
        return semantic_ref or ""
    for name, obj_snapshot in objects.items():
        if _snapshot_has_grasp_affordance(obj_snapshot):
            return str(name)
    return str(next(iter(objects)))


def _semantic_object_ref_from_stage(stage: Any) -> str:
    refs = [str(ref).strip().lower().replace(" ", "_") for ref in getattr(stage, "target_references", [])]
    ignored = _ignored_stage_reference_tokens()
    for ref in refs:
        if not ref or ref in ignored:
            continue
        if ref in {"white_mug", "yellow_and_white_mug", "yellow_white_mug"}:
            return ref
        if "moka_pot_2" in ref:
            return "moka_pot_2"
        if "moka_pot" in ref:
            return "moka_pot"
        if "book" in ref:
            return "book"
        if "plate" in ref:
            return "plate"
        if "mug" in ref:
            return ref
        if ref not in {
            "plate",
            "left_plate",
            "right_plate",
            "basket",
            "bowl",
            "stove",
            "cabinet",
            "rack",
            "caddy",
            "back_caddy_compartment",
            "drawer",
            "top_drawer",
            "middle_drawer",
            "bottom_drawer",
        }:
            return ref
    stage_id = str(getattr(stage, "stage_id", ""))
    if "_pick_lift_verify" in stage_id:
        return _object_name_from_stage_id(stage_id, "_pick_lift_verify")
    if "_transport_guarded" in stage_id:
        return _object_name_from_stage_id(stage_id, "_transport_guarded")
    if "_low_release_verify" in stage_id:
        return _object_name_from_stage_id(stage_id, "_low_release_verify")
    return ""


def _ignored_stage_reference_tokens() -> set[str]:
    return {
        "object_grasp_affordance",
        "gripper_pose",
        "pre_release_pose",
        "task_object",
        "object_pose",
        "task_target",
        "target_pose",
        "target_region_pose",
        "contact_pose",
        "push_direction",
        "drawer_handle_pose",
        "drawer_progress",
        "pull_direction",
        "finger_midpoint",
        "button_or_knob_pose",
        "stove_state_visual",
        "microwave_door",
        "close",
        "in",
        "on",
        "on_or_in",
        "in_or_on",
        "right_of",
        "next_to",
    }


def _object_name_from_stage_id(stage_id: str, suffix: str) -> str:
    text = str(stage_id)
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    if "_" in text:
        text = text.split("_", 1)[1]
    text = text.strip("_")
    if text in _ignored_stage_reference_tokens():
        return ""
    return text


def _names_match_for_stage(ref: str, name: str) -> bool:
    ref_norm = ref.lower().replace(" ", "_")
    name_norm = name.lower().replace(" ", "_")
    if any(alias in name_norm for alias in _SEMANTIC_STAGE_ALIASES.get(ref_norm, ())):
        return True
    ref_compact = ref_norm.replace("_", "")
    name_compact = name_norm.replace("_", "")
    return bool(
        ref_norm in name_norm
        or name_norm in ref_norm
        or (ref_compact and ref_compact in name_compact)
        or (name_compact and name_compact in ref_compact)
    )


_SEMANTIC_STAGE_ALIASES = {
    "white_mug": ("porcelain_mug",),
    "yellow_and_white_mug": ("white_yellow_mug",),
    "yellow_white_mug": ("white_yellow_mug",),
    "book": ("black_book",),
    "caddy": ("desk_caddy",),
    "moka_pot_2": ("moka_pot_2",),
}


def _resolve_stage_target_reference(stage: Any) -> str:
    ignored = _ignored_stage_reference_tokens()
    refs = [str(ref).strip() for ref in getattr(stage, "target_references", [])]
    joined = " ".join(refs + [str(getattr(stage, "stage_id", "")), str(getattr(stage, "name", "")), str(getattr(stage, "intent", ""))])
    if "microwave" in joined.lower():
        return "microwave"
    if "front of stove" in joined.lower() or "front_of_stove" in joined.lower():
        return "front of stove"
    if "back of stove" in joined.lower() or "back_of_stove" in joined.lower():
        return "back of stove"
    object_ref_seen = False
    for ref in refs:
        norm = ref.lower().replace(" ", "_")
        if not norm or norm in ignored:
            continue
        if not object_ref_seen:
            object_ref_seen = True
            continue
        return norm
    for ref in refs:
        norm = ref.lower().replace(" ", "_")
        if norm and norm not in ignored:
            return norm
    return ""


def _resolve_stage_predicate(stage: Any, *, default: str = "in") -> str:
    refs = [str(ref).lower().replace(" ", "_") for ref in getattr(stage, "target_references", [])]
    if "on" in refs:
        return "on"
    if "in" in refs:
        return "in"
    stage_text = " ".join(refs + [str(getattr(stage, "stage_id", ""))])
    if any(word in stage_text for word in ("compartment", "contain_region")):
        return "in"
    if any(word in stage_text for word in ("plate", "stove", "cabinet_top", "rack")):
        return "on"
    return default


def _resolve_stage_predicate_for_snapshot(
    stage: Any,
    scene_snapshot: dict[str, Any] | None,
    *,
    default: str = "in",
) -> str:
    goal_diagnostics = scene_snapshot.get("bddl_goal_diagnostics") if isinstance(scene_snapshot, dict) else None
    return _resolve_stage_predicate_for_goal_diagnostics(stage, goal_diagnostics, default=default)


def _resolve_stage_predicate_for_goal_diagnostics(
    stage: Any,
    goal_diagnostics: Any,
    *,
    default: str = "in",
) -> str:
    inferred = _resolve_stage_predicate(stage, default=default)
    matched = _matched_placement_goal(
        stage,
        goal_diagnostics,
        require_satisfied=False,
        predicate=None,
    )
    if isinstance(matched, dict):
        predicate = str(matched.get("predicate") or "").strip().lower()
        raw = matched.get("raw")
        if not predicate and isinstance(raw, list) and raw:
            predicate = str(raw[0]).strip().lower()
        if predicate in {"in", "on"}:
            return predicate
    return inferred


def _snapshot_has_grasp_affordance(obj_snapshot: Any) -> bool:
    if not isinstance(obj_snapshot, dict):
        return False
    affordance = obj_snapshot.get("grasp_affordance")
    return isinstance(affordance, dict) and bool(affordance.get("candidate_count", 0))


def _stage_trace_item(
    stage_id: str,
    attempt: int,
    evidence: VerificationEvidence,
    decision: StageDecision,
    reward: float,
    info_step: dict[str, Any],
    scene_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "attempt": attempt,
        "status": evidence.status.value,
        "rationale": evidence.rationale,
        "repair_hint": evidence.repair_hint,
        "decision": decision.value,
        "reward": float(reward),
        "sandbox_rc": int(info_step.get("sandbox_rc", 1)),
        "task_completed": bool(info_step.get("task_completed", False)),
        "stderr_tail": str(info_step.get("stderr", ""))[-1200:],
        "task_status_after": info_step.get("task_status", {}),
        "bddl_goal_diagnostics_after": info_step.get("bddl_goal_diagnostics", {}),
        "scene_snapshot": scene_snapshot or {},
    }


def _compact_trace_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep stage_trace compact while preserving the evidence needed to debug."""
    keep_keys = (
        "task",
        "stage_id",
        "stage_targets",
        "task_status",
        "bddl_goal_diagnostics",
        "ee_pose",
        "objects_of_interest",
        "target_sites_of_interest",
        "drawers",
        "stove_diagnostics",
    )
    compact = {key: snapshot[key] for key in keep_keys if key in snapshot}
    errors = snapshot.get("errors")
    if isinstance(errors, dict) and errors:
        compact["errors"] = errors
    contacts = snapshot.get("contact_diagnostics")
    if isinstance(contacts, list):
        compact["contact_diagnostics_head"] = contacts[:6]
        compact["contact_diagnostics_count"] = len(contacts)
    joints = snapshot.get("joint_diagnostics")
    if isinstance(joints, list):
        compact["joint_diagnostics_head"] = joints[:6]
        compact["joint_diagnostics_count"] = len(joints)
    return compact


def _annotate_stage_code(code_blocks: list[str]) -> str:
    return "\n\n".join(f"# Stage code block {idx}\n{code}" for idx, code in enumerate(code_blocks))


def _build_stage_log_lines(
    *,
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    stage_trace: list[dict[str, Any]],
    elapsed: float,
) -> list[str]:
    return [
        "-" * 100,
        "Stage-Verifiable CaP trial",
        "Generated stage program:",
        final_code if final_code else "(no program available)",
        "\nStage trace:",
        json.dumps(stage_trace, indent=2, ensure_ascii=False),
        "\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {info_step.get('stderr', '')}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        f"  Elapsed Seconds: {elapsed:.2f}",
        "-" * 100,
    ]


def _save_stage_trace(
    config: dict[str, Any],
    trial: int,
    sandbox_rc: int,
    reward: float,
    task_completed: bool,
    stage_trace: list[dict[str, Any]],
) -> None:
    if not config.get("output_dir"):
        return
    trial_dir = _stage_trial_dir(config, trial, sandbox_rc, reward, task_completed)
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "stage_trace.json").open("w", encoding="utf-8") as f:
        json.dump(stage_trace, f, indent=2, ensure_ascii=False)


def _stage_trial_dir(
    config: dict[str, Any],
    trial: int,
    sandbox_rc: int,
    reward: float,
    task_completed: bool,
) -> Path:
    return Path(config["output_dir"]) / (
        f"trial_{trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed)}"
    )


def _save_stage_trajectory_steps(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    sandbox_rc: int,
    reward: float,
    task_completed: bool,
) -> str | None:
    """Persist low-level observation/action steps for VLA dataset conversion."""
    if not config.get("output_dir") or not hasattr(env, "get_trajectory_steps"):
        return None

    steps = env.get_trajectory_steps(clear=False)
    if not steps:
        return None

    trial_dir = _stage_trial_dir(config, trial, sandbox_rc, reward, task_completed)
    trial_dir.mkdir(parents=True, exist_ok=True)
    steps_path = trial_dir / "steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as f:
        for step in steps:
            f.write(json.dumps(_json_safe(step), ensure_ascii=False, allow_nan=False) + "\n")

    _save_stage_trajectory_images(env, trial_dir, steps)

    metadata = {
        "schema_version": "stagecap_libero_steps_v0.1",
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
        "image_fields": [
            "observation.images.image",
            "observation.images.image2",
        ],
        "state_schema": {
            "observation.state": "LeRobot-style eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)",
            "observation.state.raw_joint_gripper": "Cap-X raw joint_pos(7) + gripper_fraction(1)",
        },
        "action_schema": {
            "action": "Cap-X raw joint_delta(7) + gripper_command(1)",
            "action.raw_joint_gripper": "Alias of raw Cap-X action for auditability",
            "lerobot_action_candidate": (
                "Derived 7-D candidate: delta eef_pos/eef_axis_angle(6) + raw gripper command(1)"
            ),
        },
        "candidate_action_warning": (
            "Use lerobot_action_candidate only after validating it against the target "
            "LeRobot LIBERO relative-control action semantics."
        ),
        "source": "capx.envs.stage_trial.run_stage_trial",
    }
    metadata_path = trial_dir / "steps_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved stage trajectory steps to {steps_path} ({len(steps)} steps)")
    return str(steps_path)


def _save_stage_trajectory_images(
    env: CodeExecutionEnvBase,
    trial_dir: Path,
    steps: list[dict[str, Any]],
) -> None:
    wants_agent = any((step.get("observation.images.image") or {}).get("path") for step in steps)
    wants_wrist = any((step.get("observation.images.image2") or {}).get("path") for step in steps)
    if not (wants_agent or wants_wrist):
        return

    agent_frames = _stage_trajectory_frame_buffer(env, "agent")
    wrist_frames = _stage_trajectory_frame_buffer(env, "wrist")
    if wants_agent and agent_frames:
        _write_stage_trajectory_image_sequence(
            trial_dir,
            steps,
            "observation.images.image",
            agent_frames,
        )
    if wants_wrist and wrist_frames:
        _write_stage_trajectory_image_sequence(
            trial_dir,
            steps,
            "observation.images.image2",
            wrist_frames,
        )


def _stage_trajectory_frame_buffer(env: CodeExecutionEnvBase, camera: str) -> list[np.ndarray]:
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


def _write_stage_trajectory_image_sequence(
    trial_dir: Path,
    steps: list[dict[str, Any]],
    key: str,
    frames: list[np.ndarray],
) -> None:
    written_paths: set[str] = set()
    for step in steps:
        image_ref = step.get(key) or {}
        rel_path = image_ref.get("path")
        if not rel_path or rel_path in written_paths:
            continue
        frame_index = int(step.get("image_frame_index", step.get("frame_index", 0)))
        frame = frames[min(frame_index, len(frames) - 1)]
        path = trial_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
        written_paths.add(rel_path)


def _status_needs_repair():
    from vlm_planner_cap.schema import StageStatus

    return StageStatus.NEEDS_REPAIR


def _status_uncertain():
    from vlm_planner_cap.schema import StageStatus

    return StageStatus.UNCERTAIN


def _status_satisfied():
    from vlm_planner_cap.schema import StageStatus

    return StageStatus.SATISFIED
