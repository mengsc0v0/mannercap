"""Run non-libero90 LIBERO CaP-X tasks and diagnose failed trials with ViFailback.

This script is intentionally an orchestrator. CaP-X owns task enumeration, config
generation, rollout execution, and failure detection; the already-deployed
ViFailback container owns model inference.
"""

from __future__ import annotations

import copy
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro
import yaml

from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_BASE_SUITES = [
    "libero_10",
    "libero_object",
    "libero_spatial",
    "libero_goal",
]

DEFAULT_LIBERO_PRO_SUITES = [
    "libero_10_swap",
    "libero_10_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_goal_swap",
    "libero_goal_task",
]

TRIAL_DIR_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_reward_"
    r"(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)
MAX_FEEDBACK_ROUNDS = 5
ACTION_API_RE = re.compile(
    r"\b("
    r"goto_pose|goto_pose_interactive_cartesian|open_gripper|close_gripper|"
    r"goto_pose_checked|goto_finger_midpoint_checked|"
    r"move_linear_delta|move_along_direction|move_finger_midpoint_delta|move_finger_midpoint_delta_strict|"
    r"pull_sleeved_drawer|"
    r"move_to_joints|execute_joint_trajectory|goto_home_joint_position|"
    r"plan_grasp|plan_grasp_from_point_clouds|solve_ik|solve_ik_strict|"
    r"plan_grasp_trajectory|plan_with_grasped_object"
    r")\s*\("
)
WORLD_POINT_RE = re.compile(
    r"(?:world position|world pos|world point|world coordinates)[^:\n]*:\s*"
    r"\[([^\]]+)\]",
    flags=re.IGNORECASE,
)
ENV_OBJECT_ERROR_RE = re.compile(
    r"Object '([^']+)' not found\. Available objects: \[([^\]]*)\]"
)
PIXEL_OUT_OF_BOUNDS_RE = re.compile(
    r"IndexError: index \d+ is out of bounds for axis \d+ with size \d+"
)
DEPTH_DIMENSION_RE = re.compile(
    r"IndexError: too many indices for array: array is \d+-dimensional, but \d+ were indexed"
)
POINT_PROMPT_NONE_RE = re.compile(
    r"(No points parsed from model output|"
    r"\([^)]*None[^)]*\)|"
    r"TypeError: .*NoneType|"
    r"TypeError: only length-1 arrays can be converted to Python scalars|"
    r"IndexError: invalid index to scalar variable)"
)
NO_GRASP_CANDIDATES_RE = re.compile(r"No grasp candidates found")
NAME_ERROR_RE = re.compile(r"NameError: name '([^']+)' is not defined")
PREDICATE_ALIGNED_API_RE = re.compile(r"\bsatisfy_goal_predicates\s*\(")
HIGH_LEVEL_PLACE_API_RE = re.compile(
    r"\b("
    r"pick_and_place_object|pick_and_place_objects|"
    r"put_object_in_container|put_object_on_target"
    r")\s*\("
)
CLEAN_FORBIDDEN_SYMBOLS = [
    "satisfy_goal_predicates",
    "open_cabinet_drawer",
    "close_cabinet_drawer",
    "pick_and_place_object",
    "pick_and_place_objects",
    "pick_goal_object_and_place_on_goal_target",
    "put_object_in_container",
    "put_object_on_target",
    "put_object_in_goal_region",
    "put_object_in_microwave_and_close",
    "push_object_to_goal_region",
    "sim.data",
    "qpos",
    "qvel",
    "set_state",
    "_set_",
    "_refresh_task_success_reward",
    "_check_success",
    "task_completed =",
]


@dataclass
class LiberoCapVifailbackBatchArgs:
    """Arguments for LIBERO CaP-X plus ViFailback closed-loop runs."""

    base_config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    suites: list[str] = field(default_factory=lambda: list(DEFAULT_BASE_SUITES))
    include_libero_pro: bool = False
    libero_pro_suites: list[str] = field(default_factory=lambda: list(DEFAULT_LIBERO_PRO_SUITES))
    exclude_suites: list[str] = field(default_factory=lambda: ["libero_90"])
    task_ids: list[int] | None = None
    max_tasks_per_suite: int | None = None

    models: list[str] = field(default_factory=lambda: ["mimo-v2.5-pro"])
    server_url: str = field(
        default_factory=lambda: (
            os.getenv("MIMO_BASE_URL", "http://127.0.0.1:8110").rstrip("/")
            + "/chat/completions"
        )
    )
    output_dir: str = "./outputs/libero_cap_vifailback"

    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    api_key: str | None = None
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    use_video_differencing: bool | None = None
    use_wrist_camera: bool | None = None
    use_legacy_multi_turn_decision_prompt: bool | None = None
    use_parallel_ensemble: bool | None = False
    use_multimodel: bool | None = False
    total_trials: int | None = None
    num_workers: int | None = None
    record_video: bool | None = None
    max_multiturn_blocks: int | None = None
    trial_timeout_seconds: int | None = None
    max_trial_retries: int | None = None
    debug: bool = False
    use_oracle_code: bool | None = None
    smoke_oracle_code: bool = False
    fixed_cap_code_path: str | None = None
    skip_api_servers: bool = False
    clean_no_privileged: bool = False
    oracle_readonly: bool = False

    failure_reward_threshold: float = 1.0
    diagnose_failed_trials: bool = True
    feedback_rounds: int = 1
    launch_retries: int = 2
    launch_retry_sleep: float = 30.0
    vifailback_name_prefix: str | None = None
    rerun_vifailback: bool = False
    vifailback_gpu: str = "3"
    vifailback_lock_path: str = "/tmp/capx_vifailback_gpu.lock"
    vifailback_max_frames: int = 15
    vifailback_min_frames: int = 2
    autorob_root: str = "/home/lifuchun/snap/snapd-desktop-integration/autorob"
    vifailback_script: str = "scripts/run_libero_vifailback_experiment.sh"

    dry_run: bool = False
    resume: bool = True
    force_rerun_tasks: bool = False


@dataclass
class LiberoTaskSpec:
    suite_name: str
    task_id: int
    task_name: str
    task_language: str
    num_init_states: int

    @property
    def slug(self) -> str:
        return f"{self.task_id:02d}_{_slugify(self.task_name or self.task_language)}"


def main(args: LiberoCapVifailbackBatchArgs) -> None:
    benchmark = _import_libero_benchmark()
    benchmark_dict = benchmark.get_benchmark_dict()
    base_config = _load_yaml(Path(args.base_config_path))
    output_root = Path(args.output_dir).resolve()
    report_path = output_root / "libero_cap_vifailback_report.json"

    tasks = _collect_tasks(args, benchmark_dict)
    models = ["oracle" if args.use_oracle_code else model for model in args.models]

    report: dict[str, Any] = {
        "schema_version": "capx_libero_vifailback_report_v0.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_config_path": str(Path(args.base_config_path).resolve()),
        "output_dir": str(output_root),
        "suites": sorted({task.suite_name for task in tasks}),
        "excluded_suites": args.exclude_suites,
        "models": models,
        "dry_run": args.dry_run,
        "clean_no_privileged": args.clean_no_privileged,
        "oracle_readonly": args.oracle_readonly,
        "forbidden_symbols": CLEAN_FORBIDDEN_SYMBOLS if args.clean_no_privileged else [],
        "runs": [],
    }
    _refresh_report_summary(report)
    _write_experiment_rules(output_root, args, tasks)

    print(f"Collected {len(tasks)} LIBERO tasks across {len(report['suites'])} suites.")
    for task in tasks:
        print(f"  - {task.suite_name}/{task.task_id}: {task.task_language}")

    if args.dry_run:
        _write_json(report_path, report)
        print(f"Dry run complete. Wrote task plan to {report_path}")
        return

    failed_runs: list[dict[str, Any]] = []
    total_runs = len(models) * len(tasks)
    run_idx = 1

    for model in args.models:
        for task in tasks:
            print("\n" + "=" * 80)
            print(f"Experiment {run_idx}/{total_runs}")
            print(f"Model: {'oracle' if args.use_oracle_code else model}")
            print(f"Task: {task.suite_name}/{task.task_id} - {task.task_language}")
            print("=" * 80)

            cached_record = _load_existing_task_record(args, output_root, model, task)
            if cached_record is not None:
                print(
                    "Skipping existing completed task record: "
                    f"{task.suite_name}/{task.task_id}"
                )
                run_record = cached_record
            else:
                run_record = _run_task(
                    args=args,
                    base_config=base_config,
                    output_root=output_root,
                    model=model,
                    task=task,
                )
            report["runs"].append(run_record)
            _refresh_report_summary(report)
            _write_json(report_path, report)
            if run_record["status"] != "completed":
                failed_runs.append(run_record)

            run_idx += 1

    print("\n" + "=" * 80)
    print(f"LIBERO CaP-X + ViFailback batch complete. Report: {report_path}")
    print(f"Completed runs: {len(report['runs']) - len(failed_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    print("=" * 80)

    if failed_runs:
        sys.exit(1)


def _import_libero_benchmark():
    try:
        from libero import benchmark
    except ImportError:
        print("Error: LIBERO is not importable. Activate the cap-x LIBERO environment first.")
        raise
    return benchmark


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _collect_tasks(
    args: LiberoCapVifailbackBatchArgs,
    benchmark_dict: dict[str, Any],
) -> list[LiberoTaskSpec]:
    suite_names = list(args.suites)
    if args.include_libero_pro:
        suite_names.extend(args.libero_pro_suites)

    excluded = set(args.exclude_suites)
    tasks: list[LiberoTaskSpec] = []

    for suite_name in suite_names:
        if suite_name in excluded:
            print(f"Skipping excluded suite: {suite_name}")
            continue
        if suite_name not in benchmark_dict:
            print(f"Warning: suite '{suite_name}' not found in LIBERO benchmarks.")
            continue

        task_suite = benchmark_dict[suite_name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))
        if args.max_tasks_per_suite is not None:
            task_ids = task_ids[: args.max_tasks_per_suite]

        for task_id in task_ids:
            if task_id < 0 or task_id >= task_suite.n_tasks:
                print(f"Warning: skipping invalid task_id={task_id} for suite {suite_name}.")
                continue
            task = task_suite.get_task(task_id)
            tasks.append(
                LiberoTaskSpec(
                    suite_name=suite_name,
                    task_id=task_id,
                    task_name=str(getattr(task, "name", "") or f"task_{task_id}"),
                    task_language=str(getattr(task, "language", "") or getattr(task, "name", "")),
                    num_init_states=_num_init_states(task_suite, task_id),
                )
            )

    return tasks


def _write_experiment_rules(
    output_root: Path,
    args: LiberoCapVifailbackBatchArgs,
    tasks: list[LiberoTaskSpec],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rules_path = output_root / "rules.md"
    manifest_path = output_root / "experiment_manifest.json"
    lines = [
        "# LIBERO-40 Clean CaP + ViFailback Rerun",
        "",
        f"- Created at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Task count: {len(tasks)}",
        f"- Trials/seeds per task: {args.total_trials or 'all init states'}",
        f"- Feedback rounds per task: {_bounded_feedback_rounds(args)}",
        f"- Trial timeout seconds: {args.trial_timeout_seconds or 'default'}",
        f"- Max trial timeout retries: {args.max_trial_retries or 'default'}",
        f"- Clean no-privileged mode: {args.clean_no_privileged}",
        f"- Oracle-readonly mode: {args.oracle_readonly}",
        "",
        "## Hard Rules",
        "",
        "- Results are valid only when generated CAP code uses robot/perception APIs.",
        "- Direct simulator or environment-state manipulation is forbidden.",
        "- If oracle-readonly mode is enabled, CAP code may call explicit read-only truth APIs such as `get_object_pose` and `get_cabinet_drawer_handle_pose`, but still must complete the task through robot motion.",
        "- Privileged BDDL/predicate repair helpers are forbidden.",
        "- Any generated code containing forbidden symbols is rejected by `clean_code_guard` and counted as a failed trial.",
        "",
        "## Forbidden Symbols",
        "",
        *[f"- `{symbol}`" for symbol in CLEAN_FORBIDDEN_SYMBOLS],
        "",
        "## Required Artifacts",
        "",
        "- Per round `config.yaml`.",
        "- Per trial CAP `code.py` and `summary.txt`.",
        "- Per trial rollout videos, including `video_combined.mp4` when recorded.",
        "- Per failed trial `vifailback_feedback.json`.",
        "- Per failed trial `vifailback/` archive with diagnostic image(s), manifest, run log, and text advice.",
    ]
    rules_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "libero40_clean_manifest_v0.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_root.resolve()),
        "clean_no_privileged": args.clean_no_privileged,
        "oracle_readonly": args.oracle_readonly,
        "total_trials": args.total_trials,
        "feedback_rounds": _bounded_feedback_rounds(args),
        "trial_timeout_seconds": args.trial_timeout_seconds,
        "max_trial_retries": args.max_trial_retries,
        "suites": args.suites,
        "forbidden_symbols": CLEAN_FORBIDDEN_SYMBOLS,
        "tasks": [
            {
                "suite_name": task.suite_name,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_language": task.task_language,
                "slug": task.slug,
            }
            for task in tasks
        ],
    }
    _write_json(manifest_path, manifest)


def _load_existing_task_record(
    args: LiberoCapVifailbackBatchArgs,
    output_root: Path,
    model: str,
    task: LiberoTaskSpec,
) -> dict[str, Any] | None:
    if not args.resume or args.force_rerun_tasks:
        return None

    model_dir = "oracle" if args.use_oracle_code else model.replace("/", "_")
    record_path = output_root / task.suite_name / task.slug / f"{model_dir}_vifailback_capx_record.json"
    if not record_path.exists():
        return None

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not read existing record {record_path}: {exc}")
        return None

    if not isinstance(record, dict):
        return None
    if record.get("status") != "completed":
        return None
    if record.get("requested_trials") != (args.total_trials or task.num_init_states):
        return None
    if record.get("feedback_rounds_requested") != _bounded_feedback_rounds(args):
        return None
    return record


def _num_init_states(task_suite: Any, task_id: int) -> int:
    try:
        init_states = task_suite.get_task_init_states(task_id)
        if init_states is not None:
            return len(init_states)
    except Exception as exc:
        print(f"Warning: could not read init states for task {task_id}: {exc}")
    return 50


def _run_task(
    *,
    args: LiberoCapVifailbackBatchArgs,
    base_config: dict[str, Any],
    output_root: Path,
    model: str,
    task: LiberoTaskSpec,
) -> dict[str, Any]:
    task_root = output_root / task.suite_name / task.slug
    task_root.mkdir(parents=True, exist_ok=True)

    model_dir = "oracle" if args.use_oracle_code else model.replace("/", "_")
    run_record: dict[str, Any] = {
        "model": model_dir,
        "suite_name": task.suite_name,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "task_language": task.task_language,
        "num_init_states": task.num_init_states,
        "requested_trials": args.total_trials or task.num_init_states,
        "feedback_rounds_requested": _bounded_feedback_rounds(args),
        "status": "started",
        "rounds": [],
        "failed_trials": [],
        "vifailback": [],
        "program_feedback": [],
        "success_anchors": [],
    }

    feedback_history: list[dict[str, Any]] = []
    program_feedback_history: list[dict[str, Any]] = []
    success_anchor_history: list[dict[str, Any]] = []
    max_rounds = _bounded_feedback_rounds(args)

    for round_idx in range(1, max_rounds + 1):
        round_root = task_root / f"round_{round_idx:02d}"
        config = _build_task_config(
            args,
            base_config,
            task,
            round_root,
            feedback_history=feedback_history,
            program_feedback_history=program_feedback_history,
            success_anchor_history=success_anchor_history,
            round_idx=round_idx,
        )
        config_path = round_root / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        run_dir = round_root / model_dir / "run"
        round_record: dict[str, Any] = {
            "round": round_idx,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "feedback_count_in_prompt": len(feedback_history),
            "status": "started",
            "trial_count": 0,
            "successful_trial_count": 0,
            "failed_trials": [],
            "vifailback": [],
            "program_feedback": [],
            "success_anchors": [],
        }
        run_record["rounds"].append(round_record)

        launch_args = LaunchArgs(
            config_path=str(config_path),
            server_url=args.server_url,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            api_key=args.api_key or os.getenv("MIMO_API_KEY"),
            use_visual_feedback=args.use_visual_feedback,
            use_img_differencing=args.use_img_differencing,
            use_video_differencing=args.use_video_differencing,
            use_wrist_camera=args.use_wrist_camera,
            use_legacy_multi_turn_decision_prompt=args.use_legacy_multi_turn_decision_prompt,
            use_parallel_ensemble=args.use_parallel_ensemble,
            use_multimodel=args.use_multimodel,
            total_trials=args.total_trials,
            num_workers=args.num_workers,
            record_video=args.record_video,
            output_dir=None,
            debug=args.debug,
            use_oracle_code=args.use_oracle_code,
            visual_differencing_model=model,
            visual_differencing_model_server_url=args.server_url,
            visual_differencing_model_api_key=args.api_key or os.getenv("MIMO_API_KEY"),
        )

        launch_error: BaseException | None = None
        for attempt in range(1, max(1, args.launch_retries + 1) + 1):
            try:
                launch_main(launch_args)
                round_record["status"] = "completed"
                launch_error = None
                break
            except Exception as exc:
                launch_error = exc
                round_record["status"] = "launch_failed"
                round_record["error"] = repr(exc)
                round_record["traceback"] = traceback.format_exc()
                round_record["launch_attempts"] = attempt
                print(
                    f"ERROR: launch failed for {task.suite_name}/{task.task_id} "
                    f"round {round_idx} attempt {attempt}: {exc}"
                )
                if attempt <= args.launch_retries:
                    print(
                        f"Retrying launch after {args.launch_retry_sleep:.1f}s "
                        f"({attempt}/{args.launch_retries})..."
                    )
                    time.sleep(args.launch_retry_sleep)

        trial_records = _scan_trials(run_dir, args.failure_reward_threshold)
        failed_trials = [trial for trial in trial_records if trial["failed"]]
        successful_trials = [trial for trial in trial_records if not trial["failed"]]
        round_record["trial_count"] = len(trial_records)
        round_record["successful_trial_count"] = sum(
            not trial["failed"] for trial in trial_records
        )
        round_record["failed_trials"] = failed_trials
        run_record["failed_trials"].extend(failed_trials)
        print(f"Detected {len(failed_trials)} failed trial(s) in {run_dir}")

        round_success_anchors = _collect_success_anchors(successful_trials, round_idx=round_idx)
        round_record["success_anchors"] = round_success_anchors
        run_record["success_anchors"].extend(round_success_anchors)
        success_anchor_history.extend(round_success_anchors)

        round_program_feedback = _collect_program_feedback(
            failed_trials,
            task=task,
            round_idx=round_idx,
        )
        round_record["program_feedback"] = round_program_feedback
        run_record["program_feedback"].extend(round_program_feedback)
        program_feedback_history.extend(round_program_feedback)

        if not trial_records and launch_error is not None:
            round_record["stop_reason"] = "launch_failed_no_trials"
            run_record["status"] = "launch_failed"
            break

        if not failed_trials:
            if round_record["status"] == "completed":
                round_record["stop_reason"] = "success"
                run_record["status"] = "completed"
            else:
                round_record["stop_reason"] = "launch_failed_no_trials"
                run_record["status"] = "launch_failed"
            break

        if args.diagnose_failed_trials:
            round_vifailback = [
                _run_vifailback(args, task, trial, model_dir, round_idx=round_idx)
                for trial in failed_trials
            ]
            round_record["vifailback"] = round_vifailback
            run_record["vifailback"].extend(round_vifailback)
            feedback_history.extend(
                _feedback_from_vifailback(item, round_idx=round_idx)
                for item in round_vifailback
                if _is_usable_vifailback_feedback(item)
            )

        if round_idx == max_rounds:
            round_record["stop_reason"] = "max_rounds_reached"
            run_record["status"] = "completed"
        else:
            round_record["stop_reason"] = "feedback_retry"

    run_record["feedback_history"] = feedback_history
    run_record["program_feedback_history"] = program_feedback_history
    run_record["success_anchor_history"] = success_anchor_history
    successful_rounds_all_seeds = [
        round_record
        for round_record in run_record["rounds"]
        if round_record.get("stop_reason") == "success"
    ]
    successful_rounds_any_seed = [
        round_record
        for round_record in run_record["rounds"]
        if round_record.get("successful_trial_count", 0) > 0
    ]
    run_record["task_success_any_seed"] = bool(successful_rounds_any_seed)
    run_record["task_success_all_seeds"] = bool(successful_rounds_all_seeds)
    # Keep task_success as the strict task-level completion criterion.
    run_record["task_success"] = run_record["task_success_all_seeds"]
    run_record["success_round"] = (
        successful_rounds_all_seeds[0]["round"] if successful_rounds_all_seeds else None
    )
    run_record["success_round_any_seed"] = (
        successful_rounds_any_seed[0]["round"] if successful_rounds_any_seed else None
    )
    run_record["round_count"] = len(run_record["rounds"])
    run_record["max_rounds_exhausted"] = bool(
        run_record["rounds"]
        and run_record["rounds"][-1].get("stop_reason") == "max_rounds_reached"
    )
    run_record["trial_count"] = sum(
        round_record.get("trial_count", 0) for round_record in run_record["rounds"]
    )
    run_record["successful_trial_count"] = sum(
        round_record.get("successful_trial_count", 0)
        for round_record in run_record["rounds"]
    )

    _write_json(task_root / f"{model_dir}_vifailback_capx_record.json", run_record)
    return run_record


def _refresh_report_summary(report: dict[str, Any]) -> None:
    runs = report.get("runs", [])
    rounds = [
        round_record
        for run in runs
        for round_record in run.get("rounds", [])
    ]
    vifailback_records = [
        item
        for run in runs
        for item in run.get("vifailback", [])
    ]
    report["summary"] = {
        "task_count": len({(run.get("suite_name"), run.get("task_id")) for run in runs}),
        "run_count": len(runs),
        "round_count": len(rounds),
        "completed_run_count": sum(run.get("status") == "completed" for run in runs),
        "launch_failed_run_count": sum(run.get("status") != "completed" for run in runs),
        "task_success_count": sum(bool(run.get("task_success")) for run in runs),
        "task_success_any_seed_count": sum(
            bool(run.get("task_success_any_seed"))
            or bool(run.get("successful_trial_count", 0) > 0)
            for run in runs
        ),
        "task_success_all_seeds_count": sum(
            bool(run.get("task_success_all_seeds"))
            or bool(run.get("task_success"))
            for run in runs
        ),
        "max_rounds_exhausted_count": sum(
            bool(run.get("max_rounds_exhausted")) for run in runs
        ),
        "executed_trial_count": sum(
            round_record.get("trial_count", 0) for round_record in rounds
        ),
        "successful_trial_count": sum(
            round_record.get("successful_trial_count", 0) for round_record in rounds
        ),
        "failed_trial_count": sum(
            len(round_record.get("failed_trials", [])) for round_record in rounds
        ),
        "vifailback_completed_count": sum(
            item.get("status") in {"completed", "skipped_existing"}
            for item in vifailback_records
        ),
        "vifailback_skipped_low_evidence_count": sum(
            item.get("status")
            in {
                "skipped_insufficient_rollout_evidence",
                "skipped_missing_rollout_media",
            }
            for item in vifailback_records
        ),
        "vifailback_failed_count": sum(
            item.get("status")
            not in {
                "completed",
                "skipped_existing",
                "skipped_insufficient_rollout_evidence",
                "skipped_missing_rollout_media",
            }
            for item in vifailback_records
        ),
    }


def _build_task_config(
    args: LiberoCapVifailbackBatchArgs,
    base_config: dict[str, Any],
    task: LiberoTaskSpec,
    task_root: Path,
    *,
    feedback_history: list[dict[str, Any]] | None = None,
    program_feedback_history: list[dict[str, Any]] | None = None,
    success_anchor_history: list[dict[str, Any]] | None = None,
    round_idx: int = 1,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    low_level = config.setdefault("env", {}).setdefault("cfg", {}).setdefault("low_level", {})
    privileged = bool(config.get("env", {}).get("cfg", {}).get("privileged", False))
    env_cfg = config.setdefault("env", {}).setdefault("cfg", {})
    if args.clean_no_privileged:
        privileged = bool(args.oracle_readonly)
        env_cfg["privileged"] = privileged
        if args.oracle_readonly:
            env_cfg["apis"] = [
                "FrankaLiberoApiReducedSkillLibraryNoSam3",
                "FrankaLiberoOracleReadOnlyApi",
            ]
        else:
            env_cfg["apis"] = ["FrankaLiberoApiReducedSkillLibraryNoSam3"]
        env_cfg["clean_code_guard"] = True
        prompt = env_cfg.get("prompt", "") or ""
        clean_prompt = (
            "\nClean no-privileged run constraints:\n"
            f"- The exact LIBERO goal is: {task.task_language}. Every code block must stay aligned with this goal.\n"
            "- Do not substitute a different common manipulation task. If the goal is opening a drawer, do not pick/place a mug, bowl, or plate. If the goal is turning on a stove, do not pick/place an object. If the goal names specific objects, manipulate those exact objects.\n"
            "- Do not access env, sim, qpos, qvel, reward, task_completed, or any private/internal attribute.\n"
            "- Do not call privileged drawer, placement, or predicate repair helpers.\n"
            + (
                "- Oracle-readonly truth reads are enabled for data generation: you may call get_object_pose, get_all_object_poses, get_cabinet_drawer_handle_pose, get_cabinet_drawer_progress, get_ee_pose, and goto_pose_checked. These functions return coordinates/arrival diagnostics only; you must still complete the task through goto_pose, goto_pose_interactive_cartesian, move_linear_delta, move_along_direction, open_gripper, and close_gripper.\n"
                if args.oracle_readonly
                else ""
            )
            +
            "- SAM3 is unavailable in this run. Use point_prompt_molmo, depth_to_point_cloud, "
            "pixel_to_world_point, mask/point-cloud geometry helpers, plan_grasp, solve_ik, "
            "move_to_joints, goto_pose, open_gripper, and close_gripper.\n"
            "- The generated program must contain real robot actions before it finishes. "
            "A program that only observes, prints, or computes coordinates is a failed attempt.\n"
            "- Validate perception-derived world points before moving: reject points far outside "
            "the tabletop workspace or with impossible height, and re-localize with wrist/agent "
            "views or local depth neighborhoods instead of moving to the bad point. In LIBERO "
            "world coordinates tabletop objects often have z near 0.0-0.2 m, so do not require "
            "z > 0.5 for object/target points.\n"
            "- After every point_prompt_molmo call, never blindly unpack the first result. "
            "Iterate over candidate values, require both pixel coordinates to be non-None "
            "scalar numbers, clamp them into image bounds, and re-prompt with more specific "
            "language or switch camera if no valid point is found.\n"
            "- Do not use assert statements for perception validation in generated CAP code. "
            "If validation fails, print the reason, re-localize, or execute a conservative "
            "fallback; do not crash before producing useful rollout evidence.\n"
            "- Every generated code block must be self-contained. Before using variables such as "
            "`rgb`, `depth`, `intrinsics`, `extrinsics`, `handle_pos`, or `approach_quat`, define "
            "them explicitly from `obs = get_observation()` or constants in the same code block. "
            "Normalize all depth images with `if depth.ndim == 3: depth = depth[:, :, 0]` before "
            "indexing, and then index only as `depth[v, u]`.\n"
            "- For pick/place tasks, execute an explicit sequence: localize object and target, "
            "approach, open gripper, descend, close gripper, lift, transport, release, retreat, "
            "then observe again.\n"
            "- For drawer/stove/push tasks, use physical gripper motion with contact and a short "
            "post-action observation; do not declare success from perception alone.\n"
        )
        if _is_articulated_drawer_task(task) or _is_compound_drawer_placement_task(task):
            clean_prompt += (
                "\nDrawer-specific clean strategy:\n"
                "- Treat drawer goals as articulated contact-control tasks, not ordinary pick/place tasks.\n"
                + (
                    "- In oracle-readonly mode, first call get_cabinet_drawer_handle_pose(level) for the named top/middle/bottom drawer. It returns handle_pos, handle_quat, and pull_direction in robot-base coordinates. Use those values to approach the handle/front lip, close or brace the gripper, and move with real Cartesian robot control; do not estimate drawer axis from pixels unless the oracle-readonly read fails.\n"
                    "- You may call get_cabinet_drawer_progress(level) only to decide whether another physical stroke is needed. It is diagnostic feedback, not a success setter.\n"
                    "- Log target-vs-actual reachability for drawer contact poses with goto_pose_checked. If the actual gripper position is several centimeters away from the requested pose, adjust the contact pose before pulling; do not interpret that failed attempt as a drawer-contact failure.\n"
                    "- Use move_along_direction(pull_direction, distance, handle_quat, ...) or goto_pose_interactive_cartesian to execute one-way strokes while keeping contact. For opening, move along positive pull_direction; for closing, move along negative pull_direction.\n"
                    if args.oracle_readonly
                    else ""
                )
                +
                "- Do not depend on point_prompt_molmo for drawer-handle localization in this run. The current MIMO endpoint is the code-generation model, but it does not support image input for Molmo-style pointing; a program that waits for image coordinates from that endpoint will fail. You may make at most one optional point_prompt_molmo probe, but you must have a deterministic fallback based on camera/depth geometry and the priors below.\n"
                "- First localize the exact named drawer and its handle or front lip from the agent camera; "
                "if there are top/middle/bottom drawers, explicitly ground that ordinal before moving.\n"
                "- Use prior rollout geometry only as perception-derived anchors, not as environment state: middle drawer handle world position is usually near [0.64, -0.34, 0.21]; bottom drawer/front region from prior depth geometry is near [0.35, -0.05, 0.06] to [0.55, -0.05, 0.10]. Treat these as initial guesses and refine them with current observation/depth if possible.\n"
                "- Never aim a drawer handle grasp at table height or below the cabinet front. If a depth/point-cloud estimate has z near 0.0 or negative, it is the tabletop/cabinet base, not the handle. Clamp drawer-contact z to a plausible front-lip/handle height: about 0.18-0.24 m for the visible middle handle, and about 0.08-0.16 m for the bottom drawer front/lip unless current depth clearly shows otherwise.\n"
                "- Approach the handle/front lip from the cabinet front with the gripper open, align the "
                "gripper fingers around the handle or lip, close the gripper, and keep it closed during the "
                "whole pull/push stroke.\n"
                "- Use a side/front contact primitive, not a pure top-down poke: first move to a pre-contact pose "
                "3-6 cm in front of the handle or front lip at the same z height, advance slowly into the cabinet "
                "front, then close or brace the gripper. A top-down descent can touch the drawer face without "
                "capturing the handle, so do not use end-effector displacement alone as proof that the drawer "
                "moved.\n"
                "- For handle-style drawer pulls, merely touching the front surface is not enough. The fingers "
                "must straddle or hook the handle before the pull: approach with the gripper open and centered "
                "slightly behind/around the cylindrical handle, advance a few centimeters past the visible front "
                "edge if needed, then close so the handle is trapped between the fingers. If the wrist view shows "
                "both fingertips in front of the handle instead of around it, correct the contact pose before "
                "pulling.\n"
                "- Move through several small Cartesian waypoints along a horizontal drawer axis while "
                "maintaining nearly constant height and orientation. If the drawer does not visibly move, "
                "retry the opposite axis sign before releasing.\n"
                "- Treat ViFailback screen-space commands such as left/right/backward as contact-point corrections, not as permission to change the articulated drawer axis. Convert them to small 0.02-0.04 m offsets around the handle/lip before contact, then keep the drawer-axis motion fixed by task geometry.\n"
                + (
                    "- For the LIBERO middle wooden cabinet drawer, if oracle-readonly geometry is unavailable, the most plausible opening axis is +y from the handle point toward the robot/table front. Prior clean rollouts showed +y produced the largest contact-preserving motion, while x-axis pulls repeatedly moved the gripper without satisfying the Open predicate.\n"
                    "- For bottom-drawer close tasks, if oracle-readonly geometry is unavailable, use the same drawer-axis idea in reverse: if the bottom drawer starts open, push its front/lip inward along -y toward the cabinet until the face is flush, then release/retreat upward.\n"
                    if args.oracle_readonly
                    else (
                        "- For the LIBERO middle wooden cabinet drawer, the most plausible opening axis is +y from the handle point toward the robot/table front. Prior clean rollouts showed +y produced the largest contact-preserving motion, while x-axis pulls repeatedly moved the gripper without satisfying the Open predicate. Therefore, for `open the middle drawer`, use a +y 0.25-0.35 m pull as the primary stroke, with a slight left/right contact correction from ViFailback. Only try -y as a backup; do not spend attempts on +/-x for this middle-drawer task.\n"
                        "- For bottom-drawer close tasks, use the same drawer-axis idea in reverse: if the bottom drawer starts open, push its front/lip inward along -y toward the cabinet until the face is flush, then release/retreat upward.\n"
                    )
                )
                +
                "- Prior failed rollouts reached the drawer but did not satisfy LIBERO. The main repair is contact preservation: keep gripper height fixed at the handle height, use 0.25-0.35 m strokes, keep contact throughout the stroke, and do not return through the same path after a successful-looking open stroke because that can push the drawer closed again.\n"
                "- For open-drawer goals, use a long enough outward pull and only release after a post-action "
                "observation shows the drawer is visibly open.\n"
                "- For close-drawer goals, push the drawer front inward along the same axis until the front "
                "face is flush, then observe before ending.\n"
                "- For put-inside-and-close goals, execute three phases in order: open the correct drawer, "
                "pick the named object and release it low inside the drawer interior, then push the drawer "
                "closed. Do not finish after only placing the object or only moving the drawer.\n"
                "- Never use helper names such as open_cabinet_drawer, close_cabinet_drawer, "
                "pick_and_place_object, or satisfy_goal_predicates; define lower-level motions with the "
                "allowed robot APIs instead.\n"
            )
        if "Clean no-privileged run constraints:" not in prompt:
            env_cfg["prompt"] = prompt.rstrip() + clean_prompt

    if not isinstance(low_level, dict):
        low_level = {}
        config["env"]["cfg"]["low_level"] = low_level

    if args.skip_api_servers:
        config.pop("api_servers", None)
    elif args.clean_no_privileged and isinstance(config.get("api_servers"), list):
        config["api_servers"] = [
            server
            for server in config["api_servers"]
            if "sam3" not in str(server.get("_target_", "")).lower()
        ]

    if args.fixed_cap_code_path:
        fixed_code_path = Path(args.fixed_cap_code_path).expanduser().resolve()
        if not fixed_code_path.exists():
            raise FileNotFoundError(f"Fixed CAP code not found: {fixed_code_path}")
        env_cfg["oracle_code"] = fixed_code_path.read_text(encoding="utf-8")
        config["use_oracle_code"] = True

    if args.smoke_oracle_code:
        env_cfg["oracle_code"] = (
            "print('capx libero smoke: intentionally running a no-op oracle')\n"
        )

    if feedback_history or program_feedback_history or success_anchor_history:
        prompt = env_cfg.get("prompt", "") or ""
        env_cfg["prompt"] = prompt.rstrip() + "\n\n" + _format_feedback_prompt(
            feedback_history or [],
            program_feedback_history or [],
            success_anchor_history or [],
            round_idx=round_idx,
        )

    low_level.update(
        {
            "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
            "suite_name": task.suite_name,
            "task_id": task.task_id,
            "privileged": privileged,
            "max_steps": low_level.get("max_steps", 8000),
            "control_freq": low_level.get("control_freq", 20),
            "seed": low_level.get("seed", None),
            "enable_render": low_level.get("enable_render", True),
            "viser_debug": low_level.get("viser_debug", False),
        }
    )

    config["trials"] = args.total_trials or task.num_init_states
    if args.clean_no_privileged:
        config["clean_no_privileged"] = True
        if args.max_multiturn_blocks is not None:
            config["max_multiturn_blocks"] = args.max_multiturn_blocks
        if args.trial_timeout_seconds is not None:
            config["trial_timeout_seconds"] = args.trial_timeout_seconds
        if args.max_trial_retries is not None:
            config["max_trial_retries"] = args.max_trial_retries
        config["use_img_differencing"] = (
            args.use_img_differencing if args.use_img_differencing is not None else False
        )
        config["use_video_differencing"] = (
            args.use_video_differencing if args.use_video_differencing is not None else False
        )
        config["use_parallel_ensemble"] = (
            args.use_parallel_ensemble if args.use_parallel_ensemble is not None else False
        )
        config["use_multimodel"] = (
            args.use_multimodel if args.use_multimodel is not None else False
        )
    config["output_dir"] = str(task_root / "run")
    return config


def _scan_trials(run_dir: Path, reward_threshold: float) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []

    grouped: dict[int, list[tuple[Path, re.Match[str]]]] = {}
    for trial_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        match = TRIAL_DIR_RE.match(trial_dir.name)
        if not match:
            continue
        grouped.setdefault(int(match.group("trial")), []).append((trial_dir, match))

    trial_records: list[dict[str, Any]] = []
    for trial_num, candidates in sorted(grouped.items()):
        trial_dir, match = _select_final_trial_dir(candidates)
        sandbox_rc = int(match.group("sandbox_rc"))
        reward = float(match.group("reward"))
        task_completed = bool(int(match.group("task_completed")))
        failed = sandbox_rc != 0 or not task_completed or reward < reward_threshold
        trial_records.append(
            {
                "trial": trial_num,
                "path": str(trial_dir.resolve()),
                "sandbox_rc": sandbox_rc,
                "reward": reward,
                "task_completed": task_completed,
                "failed": failed,
            }
        )
    return trial_records


def _select_final_trial_dir(
    candidates: list[tuple[Path, re.Match[str]]],
) -> tuple[Path, re.Match[str]]:
    if len(candidates) == 1:
        return candidates[0]

    def score(item: tuple[Path, re.Match[str]]) -> tuple[int, int, int, int, int]:
        path, match = item
        env_success = int(
            int(match.group("sandbox_rc")) == 0
            and bool(int(match.group("task_completed")))
            and float(match.group("reward")) >= 1.0
        )
        has_video = int(any(path.glob("*.mp4")))
        has_summary = int((path / "summary.txt").exists())
        has_code = int((path / "code.py").exists())
        mtime_ns = path.stat().st_mtime_ns
        # Prefer the latest finalized artifact directory, especially one with video.
        return (
            env_success,
            has_video,
            has_summary + has_code,
            mtime_ns,
            -int(match.group("sandbox_rc")),
        )

    selected = max(candidates, key=score)
    skipped = [str(path) for path, _ in candidates if path != selected[0]]
    if skipped:
        print(
            "Warning: multiple directories for trial "
            f"{selected[1].group('trial')}; selected {selected[0]} and ignored {skipped}"
        )
    return selected


def _run_vifailback(
    args: LiberoCapVifailbackBatchArgs,
    task: LiberoTaskSpec,
    trial: dict[str, Any],
    model_dir: str,
    *,
    round_idx: int = 1,
) -> dict[str, Any]:
    trial_path = Path(str(trial["path"]))
    frame_count = _video_frame_count_for_trial(
        trial_path,
        _read_text_if_exists(trial_path / "summary.txt"),
    )
    if frame_count is None:
        result = {
            "status": "skipped_missing_rollout_media",
            "trial_path": str(trial_path),
            "video_frame_count": None,
            "feedback_text": "",
            "reason": (
                "The trial directory has no video or image rollout media; "
                "use program-level feedback instead."
            ),
        }
        _write_json(trial_path / "vifailback_feedback.json", result)
        _archive_vifailback_artifacts(result, trial_path)
        return result
    if frame_count is not None and frame_count < args.vifailback_min_frames:
        result = {
            "status": "skipped_insufficient_rollout_evidence",
            "trial_path": str(trial_path),
            "video_frame_count": frame_count,
            "feedback_text": "",
            "reason": (
                "The rollout video had too few frames for reliable visual "
                "trajectory feedback; use program-level feedback instead."
            ),
        }
        _write_json(trial_path / "vifailback_feedback.json", result)
        _archive_vifailback_artifacts(result, trial_path)
        return result

    autorob_root = Path(args.autorob_root).resolve()
    script = autorob_root / args.vifailback_script
    if not script.exists():
        return {
            "status": "missing_script",
            "trial_path": str(trial_path),
            "error": f"ViFailback script not found: {script}",
        }

    name = _slugify(
        f"{_vifailback_name_prefix(args)}_capx_{model_dir}_{task.suite_name}_{task.task_id:02d}_round_{round_idx:02d}"
        f"_trial_{trial['trial']:02d}"
    )
    output_dir = autorob_root / "outputs" / "libero_vifailback" / "inference" / name
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.rerun_vifailback:
        result = {
            "status": "skipped_existing",
            "trial_path": str(trial_path),
            "video_frame_count": frame_count,
            "experiment_name": name,
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "feedback_text": _extract_vifailback_feedback(output_dir / "run.log"),
        }
        _write_json(Path(trial["path"]) / "vifailback_feedback.json", result)
        _archive_vifailback_artifacts(result, trial_path)
        return result

    cmd = [
        str(script),
        "--rollout",
        str(trial_path),
        "--task",
        task.task_language,
        "--name",
        name,
        "--max-frames",
        str(args.vifailback_max_frames),
        "--gpu",
        args.vifailback_gpu,
    ]

    print(f"Running ViFailback for failed trial: {trial_path}")
    with _exclusive_file_lock(args.vifailback_lock_path):
        completed = subprocess.run(cmd, cwd=autorob_root, text=True)
    result = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "trial_path": str(trial_path),
        "video_frame_count": frame_count,
        "experiment_name": name,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "feedback_text": _extract_vifailback_feedback(output_dir / "run.log"),
    }
    _write_json(Path(trial["path"]) / "vifailback_feedback.json", result)
    _archive_vifailback_artifacts(result, trial_path)
    return result


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: str):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _collect_program_feedback(
    failed_trials: list[dict[str, Any]],
    *,
    task: LiberoTaskSpec,
    round_idx: int,
) -> list[dict[str, Any]]:
    feedback: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for trial in failed_trials:
        trial_path = Path(str(trial.get("path", "")))
        item = _program_feedback_for_trial(trial_path, task=task, round_idx=round_idx)
        if item is None:
            continue
        key = (str(item.get("type")), str(item.get("message")))
        if key in seen:
            continue
        seen.add(key)
        item["trial"] = trial.get("trial")
        item["trial_path"] = str(trial_path)
        feedback.append(item)
    return feedback


def _collect_success_anchors(
    successful_trials: list[dict[str, Any]],
    *,
    round_idx: int,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trial in successful_trials:
        trial_path = Path(str(trial.get("path", "")))
        code_path = trial_path / "code.py"
        code = _read_text_if_exists(code_path)
        if not code:
            continue
        core_code = _core_success_code_snippet(code)
        if not core_code:
            continue
        if core_code in seen:
            continue
        seen.add(core_code)
        anchors.append(
            {
                "round": round_idx,
                "trial": trial.get("trial"),
                "trial_path": str(trial_path),
                "code_path": str(code_path),
                "core_code": core_code,
                "reward": trial.get("reward"),
                "task_completed": trial.get("task_completed"),
            }
        )
        if len(anchors) >= 2:
            break
    return anchors


def _core_success_code_snippet(code: str, *, max_lines: int = 24, max_chars: int = 2400) -> str:
    lines = code.splitlines()
    selected: list[str] = []
    action_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if selected and selected[-1] != "":
                selected.append("")
            continue
        if ACTION_API_RE.search(line) or PREDICATE_ALIGNED_API_RE.search(line):
            action_started = True
        if action_started or _looks_like_binding_line(stripped):
            selected.append(line.rstrip())

    if not selected:
        selected = [line.rstrip() for line in lines if line.strip()][:max_lines]

    compact: list[str] = []
    blank_pending = False
    for line in selected:
        if not line:
            blank_pending = True
            continue
        if blank_pending and compact:
            compact.append("")
        compact.append(line)
        blank_pending = False

    if len(compact) > max_lines:
        half = max(1, max_lines // 2)
        compact = compact[:half] + ["..."] + compact[-(max_lines - half - 1) :]

    snippet = "\n".join(compact).strip()
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 4].rstrip() + "\n..."
    return snippet


def _looks_like_binding_line(line: str) -> bool:
    if line.startswith("#") or line.startswith("print("):
        return False
    if "=" not in line:
        return False
    return any(
        token in line
        for token in (
            "get_object_pose",
            "get_all_object_poses",
            "sample_grasp_pose",
            "plate",
            "mug",
            "stove",
            "cabinet",
            "microwave",
            "basket",
            "bowl",
        )
    )


def _program_feedback_for_trial(
    trial_path: Path,
    *,
    task: LiberoTaskSpec,
    round_idx: int,
) -> dict[str, Any] | None:
    code_path = trial_path / "code.py"
    summary_path = trial_path / "summary.txt"
    code = _read_text_if_exists(code_path)
    summary = _read_text_if_exists(summary_path)

    issues: list[str] = []
    recommendations: list[str] = []
    feedback_type = "program_validity"

    if code and not ACTION_API_RE.search(code):
        feedback_type = "non_action_program"
        issues.append(
            "The previous program only inspected or printed scene state and did not call a robot action API."
        )
        recommendations.append(
            "Generate an actionful manipulation program that uses perception and robot-control APIs such as get_observation, point_prompt_molmo, depth_to_point_cloud, pixel_to_world_point, plan_grasp, solve_ik, move_to_joints, goto_pose, open_gripper, and close_gripper."
        )

    if "Clean-code guard rejected" in summary:
        feedback_type = "clean_guard_rejected_privileged_code"
        issues.append(
            "The previous program attempted a forbidden privileged operation or direct environment-state access."
        )
        recommendations.append(
            "Rewrite the program using only exposed robot/perception APIs. Do not call drawer/predicate/placement repair helpers, do not access env/sim/qpos/qvel, and do not mutate success or reward state."
        )

    if "SyntaxError" in summary:
        feedback_type = "invalid_python"
        issues.append(
            "The previous program was not valid Python, often because natural-language reasoning leaked into the code."
        )
        recommendations.append(
            "Return executable Python only. Do not include prose, markdown fences, or multiple alternative code blocks."
        )

    name_error = NAME_ERROR_RE.search(summary)
    if name_error:
        if feedback_type == "program_validity":
            feedback_type = "undefined_variable"
        missing_variable = name_error.group(1)
        issues.append(
            f"The previous program referenced undefined variable `{missing_variable}`."
        )
        recommendations.append(
            "Initialize every variable from get_observation(), perception results, or explicit constants before use. Do not assume helper variables such as current_ee_pos, handle_pos, object_pos, or target_pos already exist."
        )

    goal_drift = _task_goal_drift_feedback(task, code, summary)
    if goal_drift:
        feedback_type = goal_drift["type"]
        issues.append(goal_drift["message"])
        recommendations.append(goal_drift["recommendation"])

    object_error = ENV_OBJECT_ERROR_RE.search(summary)
    if object_error:
        feedback_type = "environment_api_mismatch"
        missing_name = object_error.group(1)
        available = object_error.group(2).replace("'", "").strip()
        issues.append(
            f"The program queried object '{missing_name}', but that object is not exposed by the LIBERO API."
        )
        recommendations.append(
            f"Use only names returned by get_all_object_poses(). Available objects included: {available}."
        )

    if PIXEL_OUT_OF_BOUNDS_RE.search(summary):
        if feedback_type == "program_validity":
            feedback_type = "pixel_coordinate_out_of_bounds"
        issues.append(
            "The previous program indexed image/depth arrays with a pixel coordinate outside the image bounds."
        )
        recommendations.append(
            "After every point_prompt_molmo call, clamp x to [0, image_width - 1] and y to [0, image_height - 1] before indexing depth or building a mask. Treat None coordinates as perception failure and re-localize instead of indexing."
        )

    if POINT_PROMPT_NONE_RE.search(summary):
        if feedback_type == "program_validity":
            feedback_type = "point_prompt_invalid_or_missing_point"
        issues.append(
            "The previous program trusted a point_prompt_molmo result that contained None or a non-scalar coordinate, then used it for depth indexing or pixel_to_world_point."
        )
        recommendations.append(
            "Do not unpack `list(result.values())[0]` blindly. Iterate over all returned values, accept only scalar numeric `(u, v)` pairs where neither coordinate is None, clamp them to image bounds, and if no valid point exists, re-prompt with a more specific object/part name or switch to the wrist/agent camera before executing robot motion."
        )

    if DEPTH_DIMENSION_RE.search(summary):
        if feedback_type == "program_validity":
            feedback_type = "depth_dimension_mismatch"
        issues.append(
            "The previous program assumed the depth image had the wrong number of dimensions."
        )
        recommendations.append(
            "Normalize depth with `if depth.ndim == 3: depth = depth[:, :, 0]` and then index it only as `depth[y, x]`. Do not use `depth[y, x, 0]` after get_observation has already returned a squeezed LIBERO depth image."
        )

    if NO_GRASP_CANDIDATES_RE.search(summary):
        if feedback_type == "program_validity":
            feedback_type = "no_grasp_candidates"
        issues.append(
            "The previous grasp-planning call returned no candidates or raised before a useful fallback rollout."
        )
        recommendations.append(
            "Do not assert on grasp candidate count. If plan_grasp returns no candidates, execute a conservative direct top-down or side approach around the localized object point, close the gripper, lift, and collect rollout evidence for ViFailback."
        )

    frame_count = _video_frame_count_for_trial(trial_path, summary)
    if frame_count is not None and frame_count <= 1:
        if feedback_type == "program_validity":
            feedback_type = "insufficient_rollout_evidence"
        issues.append(
            "The rollout video had only one frame, so ViFailback has little physical trajectory evidence."
        )
        recommendations.append(
            "Make the next program execute a real approach, grasp/push, and retreat trajectory before relying on visual feedback."
        )

    bad_points = _implausible_world_points(summary)
    if bad_points:
        if feedback_type == "program_validity":
            feedback_type = "implausible_world_point"
        issues.append(
            "The previous program computed perception-derived world coordinates that are implausible for the LIBERO tabletop workspace."
        )
        recommendations.append(
            "Before moving, validate every pixel_to_world_point result. Reject points with extreme x/y/z values, sample a local valid-depth neighborhood, and re-localize with a more specific prompt or the wrist camera. Do not call goto_pose on an implausible point."
        )

    if (
        code
        and re.search(r"\bassert\b", code)
        and re.search(r"(Z out of range|z out of range|height|world)", summary)
    ):
        if feedback_type == "program_validity":
            feedback_type = "overstrict_coordinate_assertion"
        issues.append(
            "The previous program used an over-strict coordinate assertion and crashed before generating a useful physical rollout."
        )
        recommendations.append(
            "Do not assert that LIBERO tabletop object z must exceed 0.5 m. Tabletop object and target world z values commonly lie near 0.0-0.2 m. Replace assertions with soft checks, re-localization, or safe fallback motion."
        )

    if (
        code
        and HIGH_LEVEL_PLACE_API_RE.search(code)
        and not PREDICATE_ALIGNED_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "predicate_alignment_needed"
        issues.append(
            "The previous program used a high-level placement affordance, but LIBERO still returned task_completed=False."
        )
        recommendations.append(
            "Replace high-level placement repair with an explicit visual-servo manipulation sequence: segment the object and target region, plan a grasp from depth, move through pregrasp/grasp/lift/transport/release poses, then visually verify object placement."
        )

    if code and ACTION_API_RE.search(code) and "Task Completed: False" in summary:
        if feedback_type == "program_validity":
            feedback_type = "cap_visual_self_check_not_env_success"
        if re.search(r"SUCCESS|Task completed|appears to be in", summary, flags=re.IGNORECASE):
            issues.append(
                "The program's own print/visual self-check claimed completion, but LIBERO returned task_completed=False and reward=0."
            )
            recommendations.append(
                "Do not rely on Molmo self-confirmation as the stopping condition. Make the next program physically improve the final state: use a lower release point inside the target container/region, release only after the object is centered over the parsed target, retreat without dragging it, and verify with a post-release observation."
            )

    if (
        code
        and _is_pick_place_task(task)
        and ACTION_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        if feedback_type in {"program_validity", "cap_visual_self_check_not_env_success"}:
            feedback_type = "pick_place_physical_alignment_needed"
        issues.append(
            "The previous program executed a pick/place-looking trajectory, but LIBERO still returned task_completed=False and reward=0. This usually means the object was not actually grasped, was dropped outside the parsed target region, or the target point was on an edge/background rather than inside the goal region."
        )
        recommendations.append(
            "For the next pick/place program, do not use a single Molmo center point as the whole manipulation plan. Build a small local depth patch or point cloud around the object, choose a grasp that closes around the object body, lift and observe that the object moved with the gripper, then place at the interior center of the named target container/region with a low release height and a retreat that moves upward first. If the instruction names multiple objects, repeat this verified pick-lift-place sequence for each named object."
        )

    if (
        code
        and _is_articulated_drawer_task(task)
        and ACTION_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "articulated_drawer_contact_axis_failure"
        issues.append(
            "The previous program interacted with the drawer/handle, but LIBERO still returned task_completed=False. The likely failure is physical: wrong handle point, wrong drawer axis/sign, gripper did not stay attached, or the pull/push displacement was too small for the success predicate."
        )
        recommendations.append(
            "Next program: do not rely on Molmo/image VLM pointing as the primary handle locator, because the current MIMO endpoint does not support image input. Use current observation/depth geometry plus prior perception-derived anchors for initialization (middle drawer handle near [0.64, -0.34, 0.21]; bottom drawer/front region near [0.35, -0.05, 0.06] to [0.55, -0.05, 0.10]). Do not treat end-effector displacement as drawer success: the hand can move 20 cm while slipping past the handle. Encode a contact primitive instead: move to a pre-contact pose 3-6 cm in front of the handle/front lip at the same z, advance into contact slowly, close the gripper or brace the fingers, keep the same z and orientation, then execute one monotonic 0.25-0.35 m pull/push stroke with 1-2 cm waypoints. Apply ViFailback 6D-pose hints as small offsets before contact: 'right/left slightly' means +/-0.02 to 0.04 m lateral correction from the camera view, and 'backward slightly' means move 0.02 to 0.04 m deeper toward the drawer face before closing. For opening, prefer -x for the middle drawer, then +x, -y, +y only if the first stroke visibly fails. After a successful-looking open stroke, open the gripper and retreat upward/sideways, never back through the pull path. For closing, use an open-gripper palm/finger push on the drawer front and keep pushing until the front face is flush."
        )
        summary_lower = summary.lower()
        if "handle" in summary_lower and any(
            marker in summary_lower
            for marker in (
                " -0.00",
                " 0.00",
                "z: 0.0",
                "z=0.0",
                "table height",
                "drawer region center",
            )
        ):
            recommendations.append(
                "The previous drawer-contact point was at or near table height. Raise the handle/front-lip target before contact: use about z=0.20-0.24 for middle-drawer handle grasps and z=0.08-0.16 for bottom-drawer front/lip pushes. Do not use the median of a whole cabinet/table point cloud as a handle point."
            )

    if (
        code
        and _is_stove_control_task(task)
        and ACTION_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "articulated_stove_contact_motion_failure"
        issues.append(
            "The previous program contacted a stove/knob area, but LIBERO still returned task_completed=False. The likely failure is wrong knob/control localization, too little rotational/tangential motion, or contact was lost before the control changed state."
        )
        recommendations.append(
            "Next program: localize the actual stove control rather than the burner surface, approach with a side/fingertip contact pose, keep contact while sweeping tangentially through a longer arc or short push path, then observe the stove state. Do not switch to placing a bowl or any unrelated object for a turn-on/turn-off goal."
        )

    missing_targets = _missing_compound_task_targets(task, code, summary)
    if missing_targets:
        feedback_type = "compound_goal_coverage_incomplete"
        issues.append(
            "The previous program did not cover every target object named by the compound LIBERO instruction."
        )
        recommendations.append(
            "The next program must manipulate all required targets, not just the first one. "
            f"Explicitly localize, grasp/push, place, release, and verify these missing target(s): {', '.join(missing_targets)}. "
            "For instructions using 'both' or multiple 'and' clauses, execute a separate physical action sequence for each object before finishing."
        )

    if (
        code
        and _is_compound_drawer_placement_task(task)
        and (
            "open_cabinet_drawer" in code
            or HIGH_LEVEL_PLACE_API_RE.search(code)
        )
        and not PREDICATE_ALIGNED_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "compound_drawer_predicate_alignment_needed"
        issues.append(
            "The task combines drawer state with placing an object inside/on a drawer or cabinet region, but the previous program did not align the placement and drawer open/close state with LIBERO's parsed BDDL goal."
        )
        recommendations.append(
            "Use only robot motion: visually localize the drawer handle, approach with the gripper, close on the handle, pull/push along the drawer axis through a short trajectory, then place the object using perception-derived target points."
        )

    if (
        code
        and _is_push_region_task(task)
        and ACTION_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "push_region_alignment_needed"
        issues.append(
            "The task is a push-to-region goal. The previous program executed a push-like trajectory, but LIBERO still returned task_completed=False, which usually means the object did not end inside the parsed target region."
        )
        recommendations.append(
            "Execute a clearer physical push: localize the object and goal region from camera/depth, move the gripper behind the object opposite the desired direction, push through the target region, and add a short settling/verification observation."
        )

    if (
        code
        and _is_spatial_goal_object_task(task)
        and ACTION_API_RE.search(code)
        and "Task Completed: False" in summary
    ):
        feedback_type = "spatial_goal_instance_alignment_needed"
        issues.append(
            "The task is a spatially qualified repeated-object placement task. The previous program likely used a generic object name such as 'akita_black_bowl', so the physical action and the BDDL goal instance may diverge."
        )
        recommendations.append(
            "Use visual grounding to disambiguate the exact instance mentioned by the language instruction, then manipulate that object with explicit grasp/transport/release motions. Do not use goal-predicate repair helpers."
        )

    if not issues:
        return None

    return {
        "round": round_idx,
        "type": feedback_type,
        "message": " ".join(issues),
        "recommendation": " ".join(recommendations),
        "code_path": str(code_path) if code_path.exists() else None,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "video_frame_count": frame_count,
    }


def _implausible_world_points(summary: str) -> list[list[float]]:
    points: list[list[float]] = []
    for match in WORLD_POINT_RE.finditer(summary):
        values = [
            float(item)
            for item in re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?",
                match.group(1),
                flags=re.IGNORECASE,
            )
        ]
        if len(values) < 3:
            continue
        x, y, z = values[:3]
        if x < -0.2 or x > 1.1 or y < -0.6 or y > 0.8 or z < -0.1 or z > 0.8:
            points.append([x, y, z])
    return points


def _task_goal_drift_feedback(
    task: LiberoTaskSpec,
    code: str,
    summary: str,
) -> dict[str, str] | None:
    if not code or "Task Completed: False" not in summary:
        return None

    text = f"{task.task_language or task.task_name}".replace("_", " ").lower()
    action_code = _code_after_first_action(code).lower() or code.lower()

    pick_place_markers = (
        "mug",
        "plate",
        "bowl",
        "basket",
        "cream cheese",
        "alphabet soup",
        "tomato sauce",
        "ketchup",
        "milk",
        "wine bottle",
        "rack",
    )
    drawer_markers = ("drawer", "cabinet", "handle")
    stove_markers = ("stove", "knob", "burner")

    if "drawer" in text:
        has_drawer_action = any(marker in action_code for marker in drawer_markers)
        has_pick_place_drift = (
            any(marker in action_code for marker in pick_place_markers)
            and ("handle" not in action_code or "drawer" not in action_code)
        )
        if not has_drawer_action or has_pick_place_drift:
            return {
                "type": "task_goal_drift",
                "message": (
                    "The previous program drifted away from the LIBERO drawer goal. "
                    "It did not keep the physical action focused on opening/closing the named drawer."
                ),
                "recommendation": (
                    f"The next program must solve exactly this goal: '{task.task_language}'. "
                    "Localize the specified drawer handle, approach it, close the gripper on/around the handle, "
                    "pull or push along the drawer axis with contact, then observe. Do not pick/place mugs, bowls, plates, or unrelated objects."
                ),
            }

    if "turn on the stove" in text or "turn off the stove" in text:
        has_stove_action = any(marker in action_code for marker in stove_markers)
        has_pick_place_drift = any(marker in action_code for marker in ("bowl", "plate", "basket", "rack"))
        if not has_stove_action or has_pick_place_drift:
            return {
                "type": "task_goal_drift",
                "message": (
                    "The previous program drifted away from the stove-control goal. "
                    "It did not focus the physical action on the stove knob/switch."
                ),
                "recommendation": (
                    f"The next program must solve exactly this goal: '{task.task_language}'. "
                    "Localize the stove knob/switch, approach it with the gripper or fingertip, make contact, rotate or push through a short controlled trajectory, then observe. Do not pick/place objects."
                ),
            }

    return None


def _missing_compound_task_targets(
    task: LiberoTaskSpec,
    code: str,
    summary: str,
) -> list[str]:
    if not code or "Task Completed: False" not in summary:
        return []
    targets = _compound_task_targets(task)
    if len(targets) < 2:
        return []
    actionful_code = _code_after_first_action(code).lower()
    if not actionful_code:
        return []
    missing = []
    for target in targets:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", target.lower()) if tok]
        if not tokens:
            continue
        # Require distinctive target words after robot actions start. This avoids
        # counting a first observation-only block as full compound-goal coverage.
        distinctive = [
            tok
            for tok in tokens
            if tok not in {"the", "a", "an", "and", "both", "box", "can", "jar"}
        ] or tokens
        if not any(tok in actionful_code for tok in distinctive):
            missing.append(target)
    return missing


def _code_after_first_action(code: str) -> str:
    match = ACTION_API_RE.search(code)
    if not match:
        return ""
    return code[match.start() :]


def _compound_task_targets(task: LiberoTaskSpec) -> list[str]:
    text = f"{task.task_language or task.task_name}".replace("_", " ").lower()
    text = re.sub(r"\s+", " ", text).strip()
    targets: list[str] = []

    both_match = re.search(
        r"\bboth\s+the\s+(.+?)\s+and\s+the\s+(.+?)\s+(?:in|into|on|onto|to)\b",
        text,
    )
    if both_match:
        targets.extend([both_match.group(1), both_match.group(2)])

    put_matches = re.findall(
        r"\bput\s+the\s+(.+?)\s+(?:in|into|on|onto|to)\b",
        text,
    )
    targets.extend(put_matches)

    pick_matches = re.findall(
        r"\bpick\s+up\s+the\s+(.+?)\s+and\s+place\b",
        text,
    )
    targets.extend(pick_matches)

    cleaned: list[str] = []
    for target in targets:
        item = re.sub(r"\b(?:left|right)\s+plate\b.*", "", target).strip(" .,")
        item = re.sub(r"\s+", " ", item)
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _is_compound_drawer_placement_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.task_name} {task.task_language}".replace("_", " ").lower()
    has_drawer_open = "drawer" in text and (
        "open" in text or "opened" in text or "opening" in text
    )
    has_placement = any(
        word in text
        for word in (
            "put",
            "place",
            "inside",
            "in the",
            "on the",
        )
    )
    return has_drawer_open and has_placement


def _is_push_region_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.task_name} {task.task_language}".replace("_", " ").lower()
    return "push" in text and any(word in text for word in ("front", "back", "left", "right", "region"))


def _is_pick_place_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.task_name} {task.task_language}".replace("_", " ").lower()
    return any(word in text for word in ("pick up", "place", "put")) and not (
        _is_articulated_drawer_task(task) or _is_stove_control_task(task)
    )


def _is_articulated_drawer_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.task_name} {task.task_language}".replace("_", " ").lower()
    return "drawer" in text and any(
        word in text for word in ("open", "opened", "opening", "close", "closed", "closing")
    )


def _is_stove_control_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.task_name} {task.task_language}".replace("_", " ").lower()
    return "stove" in text and any(word in text for word in ("turn on", "turn off"))


def _is_spatial_goal_object_task(task: LiberoTaskSpec) -> bool:
    text = f"{task.suite_name} {task.task_name} {task.task_language}".replace("_", " ").lower()
    return "libero spatial" in text and "place" in text and any(
        phrase in text
        for phrase in (
            "between",
            "next to",
            "table center",
            "cookie box",
            "cookies box",
            "drawer",
            "on the stove",
        )
    )


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _video_frame_count_from_summary(summary: str) -> int | None:
    match = re.search(r"video_combined\.mp4 \((\d+) frames\)", summary)
    if not match:
        return None
    return int(match.group(1))


def _video_frame_count_for_trial(trial_path: Path, summary: str = "") -> int | None:
    summary_count = _video_frame_count_from_summary(summary)
    if summary_count is not None:
        return summary_count

    video_path = trial_path / "video_combined.mp4"
    if not video_path.exists():
        return None

    try:
        import cv2  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(str(video_path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return count if count > 0 else None
    except Exception:
        return None


def _format_feedback_prompt(
    feedback_history: list[dict[str, Any]],
    program_feedback_history: list[dict[str, Any]],
    success_anchor_history: list[dict[str, Any]],
    *,
    round_idx: int,
) -> str:
    lines = [
        "Previous failed rollout feedback:",
        (
            "Use all feedback below to change the next CaP program. Do not merely "
            "repeat the same plan."
        ),
        (
            "Return only executable Python code. Do not include natural-language "
            "explanations, markdown, or code fences."
        ),
        (
            "The next program must execute robot actions, not only print observations. "
            "Use the environment-provided APIs directly; do not import them."
        ),
        (
            f"This is retry round {round_idx}; at most {MAX_FEEDBACK_ROUNDS} "
            "closed-loop rounds are allowed."
        ),
    ]

    task_hints = _task_specific_prompt_hints()
    if task_hints:
        lines.append("")
        lines.append("LIBERO object/action grounding constraints:")
        lines.extend(f"- {hint}" for hint in task_hints)

    if program_feedback_history:
        lines.append("")
        lines.append("Program-level feedback from CaP-X:")
        for item in program_feedback_history[-MAX_FEEDBACK_ROUNDS:]:
            lines.extend(
                [
                    "",
                    f"- Program feedback from round {item.get('round', '?')} ({item.get('type', 'unknown')}):",
                    _escape_prompt_format_braces(
                        str(item.get("message") or "").strip()
                    ),
                    "  Required repair: "
                    + _escape_prompt_format_braces(
                        str(item.get("recommendation") or "").strip()
                    ),
                ]
            )
        if round_idx >= max(4, MAX_FEEDBACK_ROUNDS - 1):
            feedback_types = {
                str(item.get("type") or "") for item in program_feedback_history
            }
            needs_action = any(
                item in feedback_types
                for item in (
                    "non_action_program",
                    "invalid_python",
                    "insufficient_rollout_evidence",
                )
            )
            needs_predicate_alignment = any(
                item in feedback_types
                for item in (
                    "predicate_alignment_needed",
                    "compound_drawer_predicate_alignment_needed",
                )
            )
            if needs_action or needs_predicate_alignment:
                lines.extend(
                    [
                        "",
                        "Late-round hard constraint:",
                        (
                            "This is one of the final allowed retries. The program "
                            "must be executable and must call at least one robot action "
                            "or task affordance API; do not only inspect, print, or "
                            "describe the scene."
                        ),
                    ]
                )
                if needs_predicate_alignment:
                    lines.append(
                        "Because earlier actionful placement attempts still failed "
                        "LIBERO's success checker, preserve the actionful manipulation "
                        "plan and improve physical target localization and release pose."
                    )

    if success_anchor_history:
        lines.append("")
        lines.append("Reward-verified success anchors from earlier trials:")
        lines.append(
            "At least one previous sample reached reward=1 and task_completed=True. "
            "You must preserve the object names, target names, action order, and "
            "successful API pattern below as the backbone. Do not restart scene "
            "exploration or switch to different object/target aliases. Make only "
            "minimal stabilizing changes suggested by the failed-rollout feedback."
        )
        for item in success_anchor_history[-2:]:
            core_code = str(item.get("core_code") or "").strip()
            if not core_code:
                continue
            lines.extend(
                [
                    "",
                    f"- Success anchor from round {item.get('round', '?')} trial {item.get('trial', '?')}:",
                    "```python",
                    _escape_prompt_format_braces(core_code),
                    "```",
                ]
            )

    if feedback_history:
        lines.append("")
        lines.append("Visual trajectory feedback from ViFailback:")
    for item in feedback_history[-MAX_FEEDBACK_ROUNDS:]:
        lines.extend(
            [
                "",
                f"- Feedback from round {item.get('round', '?')}:",
                _escape_prompt_format_braces(
                    str(item.get("feedback_text") or "(no textual feedback extracted)").strip()
                ),
            ]
        )
    lines.append("")
    lines.append("Generate a complete executable Python program for the current retry.")
    return "\n".join(lines)


def _task_specific_prompt_hints() -> list[str]:
    return [
        "Use `porcelain_mug` for the plain white mug; do not use nonexistent `white_mug`, and do not confuse it with `red_coffee_mug`.",
        "Use `white_yellow_mug` for the yellow-and-white mug.",
        "Use `flat_stove` for stove placement; stove knobs are usually not exposed as separate LIBERO objects.",
        "Use the scene's exposed cabinet object (`white_cabinet` or `wooden_cabinet`) rather than assuming every cabinet is wooden.",
        "Use `black_book` for the book and `desk_caddy` for the caddy. For caddy compartment tasks, estimate compartment points from visual geometry instead of guessing a caddy center.",
        "For microwave tasks, localize the mug, microwave body, and door/handle visually, then use explicit approach, grasp/place, and door motion trajectories.",
    ]


def _bounded_feedback_rounds(args: LiberoCapVifailbackBatchArgs) -> int:
    return max(1, min(args.feedback_rounds, MAX_FEEDBACK_ROUNDS))


def _vifailback_name_prefix(args: LiberoCapVifailbackBatchArgs) -> str:
    if args.vifailback_name_prefix:
        return _slugify(args.vifailback_name_prefix)
    output_name = Path(args.output_dir).resolve().name
    return _slugify(output_name)


def _feedback_from_vifailback(item: dict[str, Any], *, round_idx: int) -> dict[str, Any]:
    return {
        "round": round_idx,
        "experiment_name": item.get("experiment_name"),
        "manifest_path": item.get("manifest_path"),
        "output_dir": item.get("output_dir"),
        "feedback_text": item.get("feedback_text") or "",
        "trial_path": item.get("trial_path"),
    }


def _is_usable_vifailback_feedback(item: dict[str, Any]) -> bool:
    if item.get("status") not in {"completed", "skipped_existing"}:
        return False
    feedback_text = str(item.get("feedback_text") or "")
    if re.search(r'"success detection"\s*:\s*"Successful"', feedback_text, flags=re.IGNORECASE):
        return False
    return True


def _archive_vifailback_artifacts(result: dict[str, Any], trial_path: Path) -> None:
    archive_dir = trial_path / "vifailback"
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_json(archive_dir / "feedback.json", result)
    feedback_text = str(result.get("feedback_text") or "")
    if not feedback_text:
        feedback_text = str(result.get("reason") or "")
    (archive_dir / "advice.txt").write_text(feedback_text, encoding="utf-8")
    _write_diagnostic_frame(trial_path, archive_dir)

    output_dir_value = result.get("output_dir")
    if not output_dir_value:
        return
    output_dir = Path(str(output_dir_value))
    if not output_dir.exists():
        return

    for name in ("manifest.json", "run.log"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)

    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for src in sorted(output_dir.glob(pattern)):
            shutil.copy2(src, archive_dir / src.name)


def _write_diagnostic_frame(trial_path: Path, archive_dir: Path) -> None:
    if any(archive_dir.glob("*.png")) or any(archive_dir.glob("*.jpg")):
        return

    video_path = trial_path / "video_combined.mp4"
    if not video_path.exists():
        _write_diagnostic_placeholder(
            archive_dir,
            "Missing rollout video",
            "No video_combined.mp4 was produced for this failed trial.",
        )
        return

    try:
        import cv2  # type: ignore[import-not-found]
        from PIL import Image

        cap = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            _write_diagnostic_placeholder(
                archive_dir,
                "Unreadable rollout video",
                f"Could not read the first frame from {video_path.name}.",
            )
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(archive_dir / "diagnostic_frame.png")
    except Exception as exc:
        (archive_dir / "diagnostic_frame_error.txt").write_text(
            repr(exc), encoding="utf-8"
        )
        _write_diagnostic_placeholder(
            archive_dir,
            "Diagnostic frame extraction failed",
            repr(exc),
        )


def _write_diagnostic_placeholder(archive_dir: Path, title: str, detail: str) -> None:
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (960, 540), color=(245, 245, 242))
        draw = ImageDraw.Draw(image)
        draw.rectangle((32, 32, 928, 508), outline=(120, 120, 112), width=3)
        draw.text((64, 72), title, fill=(20, 20, 18))
        wrapped = _wrap_text(detail, width=88)
        draw.text((64, 122), wrapped, fill=(55, 55, 50))
        draw.text(
            (64, 438),
            "ViFailback visual evidence placeholder generated by CaP-X.",
            fill=(90, 90, 82),
        )
        image.save(archive_dir / "diagnostic_placeholder.png")
    except Exception as exc:
        (archive_dir / "diagnostic_placeholder_error.txt").write_text(
            repr(exc), encoding="utf-8"
        )


def _wrap_text(text: str, *, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current and current_len + 1 + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _escape_prompt_format_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _extract_vifailback_feedback(run_log: Path) -> str:
    if not run_log.exists():
        return ""
    text = run_log.read_text(encoding="utf-8", errors="replace")
    answer_matches = re.findall(r"<Answer>(.*?)</Answer>", text, flags=re.DOTALL)
    if answer_matches:
        return "<Answer>\n" + answer_matches[-1].strip() + "\n</Answer>"
    think_matches = re.findall(r"<Think>(.*?)</Think>", text, flags=re.DOTALL)
    if think_matches:
        return "<Think>\n" + think_matches[-1].strip() + "\n</Think>"
    marker = "Model Output:"
    if marker in text:
        return text.split(marker, 1)[-1].strip()[-2000:]
    return text.strip()[-2000:]


def _slugify(value: str) -> str:
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "unnamed"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(LiberoCapVifailbackBatchArgs))
