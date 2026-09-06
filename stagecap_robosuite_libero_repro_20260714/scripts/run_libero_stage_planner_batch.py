"""Run LIBERO tasks with a VLM-planner stage plan injected into clean CaP prompts.

This is the first executable bridge between the planner-verifier idea and
LIBERO. It intentionally keeps the runner conservative:

- CaP-X source is used as reference infrastructure.
- CaP-X is not claimed as our contribution.
- The generated policy code is constrained to clean action primitives.
- Planner output is represented as explicit stages and verifier checks.
- Reports include forbidden-symbol audits so environment-helper shortcuts are
  visible instead of silently counted as success.

The next implementation step is to execute one stage at a time with a real
verifier between stages. This runner already gives us a 40-task clean baseline
with stage-plan conditioning.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing as mp
import os
import re
import signal
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
from vlm_planner_cap.agents import VLMPlannerAgent
from vlm_planner_cap.schema import (
    CAPX_AGENT0_API_POLICY,
    DEFAULT_CLEAN_API_POLICY,
    SAM3_CLEAN_API_POLICY,
    CoordinateObservation,
    CleanApiPolicy,
    StagePlan,
    TaskPlan,
)
from vlm_planner_cap.stage_loop import (
    STAGE_STATUS_RE,
    parse_stage_status_events,
)

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


@dataclass(frozen=True)
class LiberoTaskSpec:
    suite_name: str
    task_id: int
    task_name: str
    task_language: str
    num_init_states: int
    bddl_file_path: str | None = None
    source_bddl: str | None = None
    success_predicates: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.task_id:02d}_{_slugify(self.task_name or self.task_language)}"


@dataclass
class StagePlannerBatchArgs:
    """Arguments for clean LIBERO stage-planner CaP runs."""

    base_config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    suites: list[str] = field(default_factory=lambda: list(DEFAULT_BASE_SUITES))
    include_libero_pro: bool = False
    libero_pro_suites: list[str] = field(default_factory=lambda: list(DEFAULT_LIBERO_PRO_SUITES))
    exclude_suites: list[str] = field(default_factory=lambda: ["libero_90"])
    task_ids: list[int] | None = None
    max_tasks_per_suite: int | None = None
    custom_task_manifest: str | None = None

    models: list[str] = field(default_factory=lambda: ["mimo-v2.5"])
    server_url: str = field(
        default_factory=lambda: (
            os.getenv("MIMO_BASE_URL", "http://127.0.0.1:8110").rstrip("/")
            + "/chat/completions"
        )
    )
    api_key: str | None = None
    output_dir: str = "./outputs/libero_stage_planner_clean"

    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    total_trials: int | None = 1
    num_workers: int | None = 1
    record_video: bool | None = True
    debug: bool = False

    feedback_rounds: int = 1
    launch_retries: int = 0
    launch_retry_sleep: float = 10.0
    failure_reward_threshold: float = 1.0
    round_timeout_sec: float | None = 1200.0
    max_multiturn_blocks: int = 8
    cuda_visible_devices: str | None = "0"
    replan_on_failure: bool = True
    replan_after_failed_rounds: int = 1
    allow_point_prompt_molmo: bool = False
    use_sam3_perception: bool = False
    sam3_service_url: str = field(
        default_factory=lambda: os.getenv("SAM3_SERVICE_URL", "http://127.0.0.1:8114")
    )
    molmo_base_url: str | None = None
    molmo_api_key: str | None = None
    molmo_model: str = "mimo-v2.5"
    use_stage_verifiable_runner: bool = False
    max_total_stage_attempts: int = 10
    use_deterministic_stage_fast_paths: bool = False
    save_stage_prompts: bool = True
    stage_snapshot_chars: int = 9000
    save_stage_snapshots: bool = True
    experiment_label: str = "codex_boundaries_mimo_stage_feedback"
    use_kite_feedback: bool = False
    kite_root: str | None = None
    kite_strategy: str = "mixed"
    kite_max_keyframes: int = 8
    kite_stride: int = 16
    kite_max_stage_successes: int = 4
    kite_merge_window: int = 12
    max_kite_feedback_items: int = 6
    max_kite_feedback_chars: int = 5000
    max_kite_feedback_images: int = 1
    use_online_stage_kite_feedback: bool = True
    max_online_stage_feedback_images: int = 1

    oracle_readonly: bool = True
    skip_api_servers: bool = False
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = False
    use_video_differencing: bool | None = False
    use_wrist_camera: bool | None = None
    use_parallel_ensemble: bool | None = False
    use_multimodel: bool | None = False
    use_legacy_multi_turn_decision_prompt: bool | None = None

    dry_run: bool = False
    resume: bool = True
    force_rerun_tasks: bool = False
    success_reference_roots: list[str] = field(
        default_factory=lambda: ["/home/lifuchun/data/capx/repos/cap-x/outputs"]
    )
    max_reference_examples: int = 3
    max_replan_summary_chars: int = 6000


def main(args: StagePlannerBatchArgs) -> None:
    _apply_gpu_visibility(args)
    _configure_perception_backends(args)
    benchmark = _import_libero_benchmark()
    benchmark_dict = benchmark.get_benchmark_dict()
    base_config = _load_yaml(Path(args.base_config_path))
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(args, benchmark_dict)
    report_path = output_root / "libero_stage_planner_report.json"
    report: dict[str, Any] = {
        "schema_version": "libero_stage_planner_report_v0.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_config_path": str(Path(args.base_config_path).resolve()),
        "output_dir": str(output_root),
        "suites": sorted({task.suite_name for task in tasks}),
        "models": args.models,
        "dry_run": args.dry_run,
        "oracle_readonly": args.oracle_readonly,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "replan_on_failure": args.replan_on_failure,
        "replan_after_failed_rounds": args.replan_after_failed_rounds,
        "allow_point_prompt_molmo": args.allow_point_prompt_molmo,
        "use_sam3_perception": args.use_sam3_perception,
        "sam3_service_url": os.getenv("SAM3_SERVICE_URL"),
        "molmo_backend": _molmo_backend_record(args),
        "use_stage_verifiable_runner": args.use_stage_verifiable_runner,
        "max_total_stage_attempts": _effective_max_total_stage_attempts(args),
        "use_deterministic_stage_fast_paths": args.use_deterministic_stage_fast_paths,
        "stage_snapshot_chars": args.stage_snapshot_chars,
        "save_stage_snapshots": args.save_stage_snapshots,
        "round_timeout_sec": args.round_timeout_sec,
        "method_condition": _method_condition(args),
        "clean_api_policy": {
            "allowed_primitives": list(_api_policy_for_args(args).allowed_primitives),
            "allowed_observations": list(_api_policy_for_args(args).allowed_observations),
            "allowed_skills": list(_api_policy_for_args(args).allowed_skills),
            "forbidden_symbols": list(_api_policy_for_args(args).forbidden_symbols),
            "forbidden_symbols_effective": _effective_forbidden_symbols(args),
        },
        "runs": [],
    }
    _write_experiment_manifest(output_root, args, tasks)

    print(f"Collected {len(tasks)} LIBERO tasks.")
    for task in tasks:
        print(f"  - {task.suite_name}/{task.task_id}: {task.task_language}")

    if args.dry_run:
        for model in args.models:
            for task in tasks:
                plan = _plan_for_task(task)
                task_root = output_root / task.suite_name / task.slug / "dry_run"
                config = _build_task_config(args, base_config, task, plan, task_root)
                config_path = task_root / "config.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, sort_keys=False)
                report["runs"].append(
                    {
                        "model": model.replace("/", "_"),
                        "suite_name": task.suite_name,
                        "task_id": task.task_id,
                        "task_language": task.task_language,
                        "status": "dry_run",
                        "config_path": str(config_path.resolve()),
                        "stage_plan": _task_plan_to_dict(plan),
                    }
                )
        _refresh_report_summary(report)
        _write_json(report_path, report)
        print(f"Dry run complete. Wrote {report_path}")
        return

    total_runs = len(args.models) * len(tasks)
    run_idx = 1
    for model in args.models:
        for task in tasks:
            print("\n" + "=" * 80)
            print(f"Stage-planner experiment {run_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Task: {task.suite_name}/{task.task_id} - {task.task_language}")
            print("=" * 80)

            run_record = _load_existing_record(args, output_root, model, task)
            if run_record is None:
                run_record = _run_task(args, base_config, output_root, model, task)
            else:
                print(f"Skipping completed record for {task.suite_name}/{task.task_id}")

            report["runs"].append(run_record)
            _refresh_report_summary(report)
            _write_json(report_path, report)
            run_idx += 1

    print(f"Stage-planner batch complete. Report: {report_path}")


def _import_libero_benchmark():
    try:
        from libero import benchmark
    except ImportError:
        print("Error: LIBERO is not importable. Activate the cap-x LIBERO environment first.")
        raise
    return benchmark


def _apply_gpu_visibility(args: StagePlannerBatchArgs) -> None:
    if args.cuda_visible_devices is None:
        return
    value = str(args.cuda_visible_devices).strip()
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def _configure_perception_backends(args: StagePlannerBatchArgs) -> None:
    sam3_url = str(args.sam3_service_url or "").strip().rstrip("/")
    if sam3_url:
        os.environ["SAM3_SERVICE_URL"] = sam3_url

    if args.allow_point_prompt_molmo:
        molmo_base = (args.molmo_base_url or os.getenv("MOLMO_BASE_URL") or os.getenv("MIMO_BASE_URL") or "").strip().rstrip("/")
        if molmo_base:
            os.environ["MOLMO_BASE_URL"] = molmo_base
        molmo_key = args.molmo_api_key or os.getenv("MOLMO_API_KEY") or os.getenv("MIMO_API_KEY") or os.getenv("OPENAI_API_KEY")
        if molmo_key:
            os.environ["MOLMO_API_KEY"] = molmo_key
        if args.molmo_model:
            os.environ["MOLMO_MODEL"] = args.molmo_model
        os.environ.pop("CAPX_DISABLE_MOLMO_POINT_PROMPT", None)
    else:
        os.environ["CAPX_DISABLE_MOLMO_POINT_PROMPT"] = "1"


def _molmo_backend_record(args: StagePlannerBatchArgs) -> dict[str, Any]:
    return {
        "enabled": bool(args.allow_point_prompt_molmo),
        "base_url": os.getenv("MOLMO_BASE_URL") if args.allow_point_prompt_molmo else None,
        "model": os.getenv("MOLMO_MODEL") if args.allow_point_prompt_molmo else None,
        "api_key_set": bool(os.getenv("MOLMO_API_KEY")) if args.allow_point_prompt_molmo else False,
        "implementation": "point_prompt_molmo backed by OpenAI-compatible VLM endpoint",
    }


def _rewrite_api_servers(api_servers: list[Any], args: StagePlannerBatchArgs) -> list[Any]:
    rewritten: list[Any] = []
    sam3_url = os.getenv("SAM3_SERVICE_URL", "").rstrip("/")
    sam3_match = re.match(r"^https?://([^:/]+):(\d+)$", sam3_url)
    for server in api_servers:
        if not isinstance(server, dict):
            rewritten.append(server)
            continue
        target = str(server.get("_target_", "")).lower()
        if "sam3" in target:
            if not args.use_sam3_perception:
                continue
            server = copy.deepcopy(server)
            if sam3_match:
                server["host"] = sam3_match.group(1)
                server["port"] = int(sam3_match.group(2))
            rewritten.append(server)
            continue
        rewritten.append(server)
    return rewritten


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _collect_tasks(args: StagePlannerBatchArgs, benchmark_dict: dict[str, Any]) -> list[LiberoTaskSpec]:
    if args.custom_task_manifest:
        return _collect_custom_manifest_tasks(args)

    suite_names = list(args.suites)
    if args.include_libero_pro:
        suite_names.extend(args.libero_pro_suites)

    tasks: list[LiberoTaskSpec] = []
    excluded = set(args.exclude_suites)
    for suite_name in suite_names:
        if suite_name in excluded:
            continue
        if suite_name not in benchmark_dict:
            print(f"Warning: suite '{suite_name}' not found.")
            continue
        task_suite = benchmark_dict[suite_name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))
        if args.max_tasks_per_suite is not None:
            task_ids = task_ids[: args.max_tasks_per_suite]
        for task_id in task_ids:
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


def _collect_custom_manifest_tasks(args: StagePlannerBatchArgs) -> list[LiberoTaskSpec]:
    manifest_path = Path(args.custom_task_manifest or "").expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_tasks = data.get("tasks", [])
    selected = set(args.task_ids or [])
    tasks: list[LiberoTaskSpec] = []
    for idx, item in enumerate(raw_tasks):
        task_id = int(item.get("task_index", idx))
        if selected and task_id not in selected:
            continue
        bddl_file_path = Path(str(item["generated_bddl"])).expanduser()
        if not bddl_file_path.is_absolute():
            bddl_file_path = (manifest_path.parent / bddl_file_path).resolve()
        tasks.append(
            LiberoTaskSpec(
                suite_name=str(item.get("suite_name") or data.get("suite_name") or "sfb1"),
                task_id=task_id,
                task_name=str(item.get("name") or item.get("task_id") or f"task_{task_id:02d}"),
                task_language=str(item.get("language") or item.get("name") or f"task {task_id}"),
                num_init_states=1,
                bddl_file_path=str(bddl_file_path),
                source_bddl=str(item.get("source_bddl") or ""),
                success_predicates=tuple(str(x) for x in item.get("success_predicates", [])),
                diagnostics=tuple(str(x) for x in item.get("diagnostics", [])),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "generated_bddl",
                        "source_bddl",
                        "success_predicates",
                        "diagnostics",
                    }
                },
            )
        )
    if args.max_tasks_per_suite is not None:
        tasks = tasks[: args.max_tasks_per_suite]
    return tasks


def _run_task(
    args: StagePlannerBatchArgs,
    base_config: dict[str, Any],
    output_root: Path,
    model: str,
    task: LiberoTaskSpec,
) -> dict[str, Any]:
    model_dir = model.replace("/", "_")
    task_root = output_root / task.suite_name / task.slug
    task_root.mkdir(parents=True, exist_ok=True)
    plan = _plan_for_task(task)

    run_record: dict[str, Any] = {
        "model": model_dir,
        "method_condition": _method_condition(args),
        "suite_name": task.suite_name,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "task_language": task.task_language,
        "bddl_file_path": task.bddl_file_path,
        "source_bddl": task.source_bddl,
        "success_predicates": list(task.success_predicates),
        "diagnostics": list(task.diagnostics),
        "metadata": task.metadata,
        "requested_trials": args.total_trials or task.num_init_states,
        "feedback_rounds_requested": max(1, args.feedback_rounds),
        "status": "started",
        "stage_plan": _task_plan_to_dict(plan),
        "rounds": [],
    }

    replan_context: dict[str, Any] | None = None
    for round_idx in range(1, max(1, args.feedback_rounds) + 1):
        round_root = task_root / f"round_{round_idx:02d}"
        config = _build_task_config(args, base_config, task, plan, round_root, replan_context)
        config_path = round_root / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        run_dir = round_root / model_dir / "run"
        round_record: dict[str, Any] = {
            "round": round_idx,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "status": "started",
            "trial_count": 0,
            "successful_trial_count": 0,
            "failed_trials": [],
            "forbidden_symbol_violations": [],
            "replan_context": replan_context,
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
            use_oracle_code=False,
            visual_differencing_model=model,
            visual_differencing_model_server_url=args.server_url,
            visual_differencing_model_api_key=args.api_key or os.getenv("MIMO_API_KEY"),
        )

        launch_error: BaseException | None = None
        launch_timed_out = False
        for attempt in range(1, max(1, args.launch_retries + 1) + 1):
            try:
                _launch_with_timeout(launch_args, args.round_timeout_sec)
                round_record["status"] = "completed"
                launch_error = None
                break
            except TimeoutError as exc:
                launch_timed_out = True
                launch_error = exc
                round_record["status"] = "timeout"
                round_record["error"] = repr(exc)
                round_record["launch_attempts"] = attempt
                round_record["timeout_sec"] = args.round_timeout_sec
                print(f"Launch timed out attempt {attempt}: {exc}")
                break
            except Exception as exc:
                launch_error = exc
                round_record["status"] = "launch_failed"
                round_record["error"] = repr(exc)
                round_record["traceback"] = traceback.format_exc()
                round_record["launch_attempts"] = attempt
                print(f"Launch failed attempt {attempt}: {exc}")
                if attempt <= args.launch_retries:
                    time.sleep(args.launch_retry_sleep)

        trial_records = _scan_trials(run_dir, args.failure_reward_threshold)
        forbidden_violations = _audit_forbidden_symbols(run_dir, args)
        skill_usage = _audit_skill_usage(run_dir)
        local_success_signals = _scan_local_success_signals(run_dir)
        stage_status_events = _scan_stage_status_events(run_dir)
        failed_trials = [trial for trial in trial_records if trial["failed"]]
        successful_trials = [trial for trial in trial_records if not trial["failed"]]
        round_record["trial_count"] = len(trial_records)
        round_record["successful_trial_count"] = len(successful_trials)
        round_record["failed_trials"] = failed_trials
        round_record["forbidden_symbol_violations"] = forbidden_violations
        round_record["skill_usage"] = skill_usage
        round_record["local_success_signals"] = local_success_signals
        round_record["stage_status_events"] = stage_status_events
        if args.use_kite_feedback and failed_trials:
            round_record["kite_feedback"] = _build_kite_feedback_context(
                args,
                task=task,
                trial=failed_trials[0],
            )
        else:
            round_record["kite_feedback"] = None

        if launch_timed_out:
            round_record["stop_reason"] = "round_timeout"
            run_record["status"] = "completed"
            break
        if launch_error is not None and not trial_records:
            round_record["stop_reason"] = "launch_failed_no_trials"
            run_record["status"] = "launch_failed"
            break
        if forbidden_violations:
            round_record["stop_reason"] = "forbidden_symbol_violation"
            run_record["status"] = "completed"
            break
        if not failed_trials and trial_records:
            round_record["stop_reason"] = "success"
            run_record["status"] = "completed"
            break
        if round_idx == args.feedback_rounds:
            round_record["stop_reason"] = "max_rounds_reached"
            run_record["status"] = "completed"
        else:
            round_record["stop_reason"] = "retry_without_feedback_yet"
            replan_context = _build_replan_context(args, task, plan, round_record)

    _finalize_run_record(run_record)
    if not run_record.get("task_success"):
        run_record["offline_success_reference_audit"] = _find_success_code_references(args, task)
        run_record["offline_success_reference_note"] = (
            "These references are recorded for post-hoc analysis only and are never "
            "inserted into the zero-shot planner/replanner prompt."
        )
    _write_json(task_root / f"{model_dir}_stage_planner_record.json", run_record)
    return run_record


def _build_task_config(
    args: StagePlannerBatchArgs,
    base_config: dict[str, Any],
    task: LiberoTaskSpec,
    plan: TaskPlan,
    round_root: Path,
    replan_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    env_cfg = config.setdefault("env", {}).setdefault("cfg", {})
    low_level = env_cfg.setdefault("low_level", {})
    if not isinstance(low_level, dict):
        low_level = {}
        env_cfg["low_level"] = low_level

    env_cfg["privileged"] = bool(args.oracle_readonly)
    env_cfg["clean_code_guard"] = True
    if args.oracle_readonly:
        env_cfg["apis"] = [
            "FrankaLiberoCleanSkillApi",
        ]
    else:
        env_cfg["apis"] = [
            (
                "FrankaLiberoApiReducedSkillLibrary"
                if args.use_sam3_perception
                else (
                    "FrankaLiberoApiReducedSkillLibraryNoSam3"
                    if args.allow_point_prompt_molmo
                    else "FrankaLiberoApiReducedSkillLibraryNoSam3NoMolmo"
                )
            )
        ]

    low_level.update(
        {
            "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
            "suite_name": task.suite_name,
            "task_id": task.task_id,
            "privileged": bool(args.oracle_readonly),
            "bddl_file_path": task.bddl_file_path,
            "task_language": task.task_language,
            "max_steps": max(int(low_level.get("max_steps", 8000) or 8000), 24000),
            "control_freq": low_level.get("control_freq", 20),
            "seed": low_level.get("seed", None),
            "enable_render": low_level.get("enable_render", True),
            "viser_debug": low_level.get("viser_debug", False),
        }
    )

    if args.skip_api_servers:
        config.pop("api_servers", None)
    elif isinstance(config.get("api_servers"), list):
        config["api_servers"] = _rewrite_api_servers(config["api_servers"], args)

    config["trials"] = args.total_trials or task.num_init_states
    config["record_video"] = bool(args.record_video)
    config["use_wrist_camera"] = bool(args.use_wrist_camera)
    config["max_multiturn_blocks"] = args.max_multiturn_blocks
    config["use_stage_verifiable_runner"] = bool(args.use_stage_verifiable_runner)
    config["oracle_readonly"] = bool(args.oracle_readonly)
    config["max_total_stage_attempts"] = _effective_max_total_stage_attempts(args)
    config["use_deterministic_stage_fast_paths"] = bool(args.use_deterministic_stage_fast_paths)
    config["use_sam3_perception"] = bool(args.use_sam3_perception)
    config["sam3_service_url"] = os.getenv("SAM3_SERVICE_URL")
    config["molmo_backend"] = _molmo_backend_record(args)
    config["save_stage_prompts"] = bool(args.save_stage_prompts)
    config["stage_snapshot_chars"] = int(args.stage_snapshot_chars)
    config["save_stage_snapshots"] = bool(args.save_stage_snapshots)
    config["use_img_differencing"] = bool(args.use_img_differencing)
    config["use_video_differencing"] = bool(args.use_video_differencing)
    config["use_parallel_ensemble"] = bool(args.use_parallel_ensemble)
    config["use_multimodel"] = bool(args.use_multimodel)
    config["output_dir"] = str(round_root / "run")
    config["method_condition"] = _method_condition(args)
    config["global_replan_context"] = replan_context
    config["use_online_stage_kite_feedback"] = bool(args.use_kite_feedback and args.use_online_stage_kite_feedback)
    config["max_online_stage_feedback_images"] = int(args.max_online_stage_feedback_images)
    config["max_kite_feedback_images"] = int(args.max_kite_feedback_images)

    original_prompt = env_cfg.get("prompt", "") or ""
    env_cfg["prompt"] = _build_stage_prompt(original_prompt, task, plan, args, replan_context)
    return config


def _launch_with_timeout(launch_args: LaunchArgs, timeout_sec: float | None) -> None:
    if timeout_sec is None or float(timeout_sec) <= 0:
        launch_main(launch_args)
        return

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_launch_worker, args=(launch_args, queue), daemon=False)
    process.start()
    process.join(float(timeout_sec))
    if process.is_alive():
        _terminate_process_tree(process.pid)
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(
            f"round exceeded timeout_sec={float(timeout_sec):.1f}; "
            f"terminated launch process pid={process.pid}"
        )

    if not queue.empty():
        payload = queue.get()
        if payload.get("status") == "ok":
            return
        message = payload.get("traceback") or payload.get("error") or "unknown launch worker error"
        raise RuntimeError(message)

    if process.exitcode not in (0, None):
        raise RuntimeError(f"launch worker exited with code {process.exitcode}")


def _launch_worker(launch_args: LaunchArgs, queue: Any) -> None:
    try:
        launch_main(launch_args)
    except BaseException as exc:
        queue.put(
            {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    else:
        queue.put({"status": "ok"})


def _terminate_process_tree(pid: int | None) -> None:
    if not pid:
        return
    try:
        children = _child_pids(pid)
        for child_pid in reversed(children):
            _signal_process(child_pid, signal.SIGTERM)
        _signal_process(pid, signal.SIGTERM)
        time.sleep(3)
        for child_pid in reversed(children):
            if _pid_exists(child_pid):
                _signal_process(child_pid, signal.SIGKILL)
        if _pid_exists(pid):
            _signal_process(pid, signal.SIGKILL)
    except Exception as exc:
        print(f"Warning: failed to terminate process tree for pid={pid}: {exc}")


def _child_pids(pid: int) -> list[int]:
    try:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    direct = [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]
    all_children: list[int] = []
    for child in direct:
        all_children.extend(_child_pids(child))
        all_children.append(child)
    return all_children


def _signal_process(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _method_condition(args: StagePlannerBatchArgs) -> dict[str, Any]:
    return {
        "label": args.experiment_label,
        "codex_derived_boundaries_enabled": True,
        "codex_runtime_human_assistance": False,
        "mimo_model_generates_or_repairs_cap": True,
        "deterministic_stage_fast_paths_enabled": bool(args.use_deterministic_stage_fast_paths),
        "sam3_perception_enabled": bool(args.use_sam3_perception),
        "kite_feedback_enabled": bool(args.use_kite_feedback),
        "kite_feedback_images_enabled": int(args.max_kite_feedback_images) > 0,
        "max_kite_feedback_images": int(args.max_kite_feedback_images),
        "online_stage_kite_feedback_enabled": bool(args.use_kite_feedback and args.use_online_stage_kite_feedback),
        "max_online_stage_feedback_images": int(args.max_online_stage_feedback_images),
        "max_total_stage_attempts": _effective_max_total_stage_attempts(args),
        "max_total_stage_attempts_hard_cap": 10,
        "boundary_source": (
            "Stage schemas, verifier checks, action/API contracts, and "
            "relation thresholds are the method artifacts distilled during prior "
            "Codex-assisted exploration. During this benchmark run they are fixed "
            "in code/config and are not manually edited per task."
        ),
        "representative_boundaries": [
            "Drawer placement tasks require physical drawer progress, often strict open_fraction near 0.98 before insertion.",
            "Pick stages must verify held/lift stability before transport.",
            "Transport/release stages verify BDDL relation diagnostics, AABB axes, z excess/deficit, reward, and task_completed.",
            "Failures route to bounded local repair or suffix replanning instead of blindly continuing.",
        ],
    }


def _effective_max_total_stage_attempts(args: StagePlannerBatchArgs) -> int:
    return min(max(1, int(args.max_total_stage_attempts)), 10)


def _api_policy_for_args(args: StagePlannerBatchArgs) -> CleanApiPolicy:
    if not args.oracle_readonly:
        return CAPX_AGENT0_API_POLICY
    return SAM3_CLEAN_API_POLICY if args.use_sam3_perception else DEFAULT_CLEAN_API_POLICY


def _build_stage_prompt(
    original_prompt: str,
    task: LiberoTaskSpec,
    plan: TaskPlan,
    args: StagePlannerBatchArgs,
    replan_context: dict[str, Any] | None = None,
) -> str:
    return (
        original_prompt.rstrip()
        + "\n\n"
        + "# Clean Stage-Planner CaP Constraints\n"
        + f"Exact LIBERO goal: {task.task_language}\n"
        + "You must solve the exact goal through robot actions, not by directly editing simulator state.\n"
        + "Use the stage plan below as an explicit task state machine. Write executable Python code that attempts the stages in order, observes after each stage, and prints stage status.\n"
        + "Do not write markdown or explanations outside Python comments.\n"
        + "\nAllowed action style:\n"
        + "- Use clean low-level motion/control primitives: "
        + ", ".join(_api_policy_for_args(args).allowed_primitives)
        + ".\n"
        + "- You may read environment facts through readonly observation/diagnostic APIs: "
        + ", ".join(_api_policy_for_args(args).allowed_observations)
        + ".\n"
        + "- Planner-provided coordinates, target sites, object poses, drawer progress, contacts, and relation diagnostics may be passed to code generation and verifier agents as grounding/evidence. They are not actions.\n"
        + "- You may also use audited robot-motion skills when helpful: "
        + ", ".join(_api_policy_for_args(args).allowed_skills)
        + ". These skills must still move the robot physically and must be followed by stage checks.\n"
        + "- This reproduction mode is privileged simulation data construction: use read-only simulator diagnostics for grounding and verification, but all scene changes must still come from robot motion.\n"
        + (
            "- SAM3 perception is enabled for this optional ablation: you may use `segment_sam3_text_prompt` or `segment_sam3_point_prompt` for initial visual localization, then validate any mask/point with depth or readonly geometry before acting.\n"
            if args.use_sam3_perception
            else "- SAM3 perception is disabled in the main privileged data-construction mode; use readonly geometry diagnostics and direct observations instead.\n"
        )
        + (
            "- Do not call `point_prompt_molmo`; the optional point-prompt/VLM localization path is not part of the main privileged data-construction mode.\n"
            if not args.allow_point_prompt_molmo
            else "- You may call `point_prompt_molmo` only if the local Molmo service is actually available, and failure to point must trigger a coordinate fallback.\n"
        )
        + (
            "- Oracle-readonly coordinates are enabled: you may read object poses, target poses, end-effector pose, drawer handle pose, and drawer progress if available. These are diagnostic reads only; you must still move the robot physically.\n"
            if args.oracle_readonly
            else ""
        )
        + "\nForbidden shortcuts:\n"
        + "".join(f"- Do not use `{symbol}`.\n" for symbol in _effective_forbidden_symbols(args))
        + "\nStage plan:\n"
        + _format_task_plan(plan, policy=_api_policy_for_args(args))
        + _format_replan_context(replan_context)
        + "\nVerifier behavior expected in code:\n"
        + "- After each stage, observe the environment and print whether each stage check appears satisfied.\n"
        + "- Print one machine-readable line per stage in this exact form: `STAGE_STATUS {\"stage_id\": \"...\", \"status\": \"satisfied|not_satisfied|needs_repair|uncertain\", \"evidence\": \"short visual/geometric/log evidence\", \"repair\": \"short repair hint or empty\"}`.\n"
        + "- If a required motion or skill reports failure, prints `arrived=False`, `success=False`, or returns an uncertain result, do not claim the stage is done. Retry with a small local correction, choose a fallback target, or stop with a clear failed-stage print.\n"
        + "- Never release a grasped object at a target if the transport motion to that target did not arrive; repair the transport stage first.\n"
        + "- If a stage check is not satisfied, retry with a small local correction before moving to the next stage.\n"
        + "- For drawer tasks, never treat mere gripper displacement as success; verify handle capture and drawer progress/opening.\n"
        + "- End only after the exact LIBERO goal is plausibly satisfied by observation and allowed read-only diagnostics.\n"
    )


def _format_task_plan(plan: TaskPlan, *, policy: CleanApiPolicy | None = None) -> str:
    if not plan.stages:
        return "- No task-specific template exists. Create a short visual/geometric stage plan before acting.\n"
    allowed = (
        set(policy.allowed_primitives) | set(policy.allowed_observations) | set(policy.allowed_skills)
        if policy is not None
        else None
    )
    lines: list[str] = []
    for idx, stage in enumerate(plan.stages, start=1):
        lines.append(f"{idx}. [{stage.stage_id}] {stage.name}")
        lines.append(f"   Intent: {stage.intent}")
        if stage.family is not None:
            lines.append(f"   Task family: {stage.family.value}")
        lines.append(f"   Max attempts before replanning/repair escalation: {stage.max_attempts}")
        stage_primitives = [
            name for name in stage.allowed_primitives if allowed is None or name in allowed
        ]
        stage_observations = [
            name for name in stage.allowed_observations if allowed is None or name in allowed
        ]
        lines.append(f"   Allowed primitives: {', '.join(stage_primitives)}")
        if stage_observations:
            lines.append(f"   Observations/perception: {', '.join(stage_observations)}")
        lines.append(f"   Target references: {', '.join(stage.target_references)}")
        lines.append("   Checks:")
        for check in stage.checks:
            lines.append(f"   - {check.name} ({check.check_type}): {check.description}")
            for criterion in check.success_criteria:
                lines.append(f"     * {criterion}")
        if stage.repair_hints:
            lines.append("   Repair hints:")
            for hint in stage.repair_hints:
                lines.append(f"   - {hint}")
    if plan.notes:
        lines.append("Notes:")
        for note in plan.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _format_replan_context(replan_context: dict[str, Any] | None) -> str:
    if not replan_context:
        return ""
    text = _format_replan_value(replan_context)
    return (
        "\nReplan context from previous failed round:\n"
        + text
        + "\nReplanning instruction:\n"
        + "- Identify which stage or substep most likely failed.\n"
        + "- Revise the next attempt's local strategy before writing code.\n"
        + "- If prior feedback was too coarse, add finer print/check points around the uncertain contact or visibility moment.\n"
        + "- If key failure evidence is missing, explicitly capture or print the missing observation, coordinate, contact, progress, or image-derived signal.\n"
    )


def _build_kite_feedback_context(
    args: StagePlannerBatchArgs,
    *,
    task: LiberoTaskSpec,
    trial: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract KITE keyframes for a failed trial and return prompt-sized evidence.

    This is intentionally post-hoc evidence extraction from the just-failed
    trial. It never retrieves success code and never changes the fixed
    Codex-derived verifier/skill boundaries during the benchmark.
    """
    trial_dir = Path(str(trial.get("path", ""))).expanduser()
    if not trial_dir.exists():
        return {"status": "missing_trial_dir", "trial_dir": str(trial_dir)}
    video_path = trial_dir / "video_combined.mp4"
    stage_trace_path = trial_dir / "stage_trace.json"
    if not video_path.exists():
        return {
            "status": "missing_video",
            "trial_dir": str(trial_dir.resolve()),
            "expected_video": str(video_path),
        }
    try:
        module = _load_kite_extractor_module()
        out_dir = module.process_trial(
            trial_dir,
            video_path=video_path,
            stage_trace_path=stage_trace_path if stage_trace_path.exists() else None,
            record_path=None,
            out_dir=None,
            kite_root=args.kite_root,
            strategy=args.kite_strategy,
            max_keyframes=int(args.kite_max_keyframes),
            stride=int(args.kite_stride),
            max_stage_successes=int(args.kite_max_stage_successes),
            merge_window=int(args.kite_merge_window),
        )
        payload = json.loads((Path(out_dir) / "kite_failure_keyframes.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "kite_extraction_failed",
            "trial_dir": str(trial_dir.resolve()),
            "error": repr(exc),
        }

    feedback_items = _compact_kite_keyframes_for_prompt(
        payload,
        max_items=int(args.max_kite_feedback_items),
        max_chars=int(args.max_kite_feedback_chars),
    )
    return {
        "status": "ready",
        "source": "KITE keyframe selector + stage_trace failure evidence",
        "task": {
            "suite_name": task.suite_name,
            "task_id": task.task_id,
            "language": task.task_language,
        },
        "trial_dir": str(trial_dir.resolve()),
        "artifact_dir": str(Path(out_dir).resolve()),
        "contact_sheet": payload.get("contact_sheet"),
        "keyframe_method": (payload.get("method") or {}).get("keyframe_method"),
        "prompt_feedback": feedback_items,
    }


def _load_kite_extractor_module() -> Any:
    path = Path(__file__).resolve().parent / "extract_kite_failure_keyframes.py"
    module_name = "extract_kite_failure_keyframes"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load KITE extractor module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compact_kite_keyframes_for_prompt(
    payload: dict[str, Any],
    *,
    max_items: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    keyframes = payload.get("keyframes") or []
    if not isinstance(keyframes, list):
        return []
    priority = {
        "failure_feedback": 0,
        "stage_event": 1,
        "stage_success": 2,
        "kite_motion": 3,
    }
    ordered = sorted(
        [item for item in keyframes if isinstance(item, dict)],
        key=lambda item: (
            priority.get(str(item.get("kind") or ""), 9),
            int(item.get("frame_idx") or 0),
        ),
    )
    out: list[dict[str, Any]] = []
    used_chars = 0
    for item in ordered:
        if len(out) >= max_items or used_chars >= max_chars:
            break
        compact = {
            "kind": item.get("kind"),
            "frame_idx": item.get("frame_idx"),
            "time_sec": item.get("time_sec"),
            "stage_id": item.get("stage_id"),
            "stage_name": item.get("stage_name"),
            "status": item.get("status"),
            "motion_score": item.get("motion_score"),
            "motion_level": item.get("motion_level"),
            "evidence": _truncate_for_prompt(str(item.get("evidence") or ""), 700),
            "repair": _truncate_for_prompt(str(item.get("repair") or ""), 500),
            "annotated_frame": (item.get("files") or {}).get("annotated"),
        }
        text_len = len(json.dumps(compact, ensure_ascii=False, default=str))
        if used_chars + text_len > max_chars and out:
            break
        out.append(compact)
        used_chars += text_len
    return out


def _truncate_for_prompt(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_replan_value(value: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            safe_key = str(key).replace("{", "(").replace("}", ")")
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{prefix}- {safe_key}:")
                lines.append(_format_replan_value(item, indent + 1))
            else:
                lines.append(f"{prefix}- {safe_key}: {_sanitize_prompt_braces(str(item))}")
        return "\n".join(line for line in lines if line) + "\n"
    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{prefix}-")
                lines.append(_format_replan_value(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_sanitize_prompt_braces(str(item))}")
        return "\n".join(line for line in lines if line) + "\n"
    return f"{prefix}{_sanitize_prompt_braces(str(value))}\n"


def _sanitize_prompt_braces(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def _plan_for_task(task: LiberoTaskSpec) -> TaskPlan:
    planner = VLMPlannerAgent()
    return planner.plan(
        task=task.task_language,
        screenshot_path=None,
        coordinates=CoordinateObservation(),
    )


def _task_plan_to_dict(plan: TaskPlan) -> dict[str, Any]:
    return {
        "task": plan.task,
        "notes": list(plan.notes),
        "stages": [_stage_to_dict(stage) for stage in plan.stages],
    }


def _stage_to_dict(stage: StagePlan) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "name": stage.name,
        "intent": stage.intent,
        "family": stage.family.value if stage.family is not None else None,
        "max_attempts": stage.max_attempts,
        "allowed_primitives": list(stage.allowed_primitives),
        "allowed_observations": list(stage.allowed_observations),
        "target_references": list(stage.target_references),
        "checks": [
            {
                "name": check.name,
                "description": check.description,
                "check_type": check.check_type,
                "success_criteria": list(check.success_criteria),
            }
            for check in stage.checks
        ],
        "repair_hints": list(stage.repair_hints),
    }


def _scan_trials(run_dir: Path, reward_threshold: float) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    grouped: dict[int, list[tuple[Path, re.Match[str]]]] = {}
    for trial_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        match = TRIAL_DIR_RE.match(trial_dir.name)
        if match:
            grouped.setdefault(int(match.group("trial")), []).append((trial_dir, match))
    records = []
    for trial_num, candidates in sorted(grouped.items()):
        trial_dir, match = sorted(candidates, key=lambda item: item[0].stat().st_mtime)[-1]
        sandbox_rc = int(match.group("sandbox_rc"))
        reward = float(match.group("reward"))
        task_completed = bool(int(match.group("task_completed")))
        records.append(
            {
                "trial": trial_num,
                "path": str(trial_dir.resolve()),
                "sandbox_rc": sandbox_rc,
                "reward": reward,
                "task_completed": task_completed,
                "failed": sandbox_rc != 0 or not task_completed or reward < reward_threshold,
            }
        )
    return records


def _effective_forbidden_symbols(args: StagePlannerBatchArgs) -> list[str]:
    symbols = list(DEFAULT_CLEAN_API_POLICY.forbidden_symbols)
    if not args.use_sam3_perception:
        symbols.extend(["segment_sam3_text_prompt", "segment_sam3_point_prompt"])
    if not args.allow_point_prompt_molmo:
        symbols.append("point_prompt_molmo")
    return list(dict.fromkeys(symbols))


def _audit_forbidden_symbols(run_dir: Path, args: StagePlannerBatchArgs) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if not run_dir.exists():
        return violations
    for code_path in sorted(run_dir.glob("trial_*/code.py")):
        try:
            code = code_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code = code_path.read_text(errors="replace")
        symbols = [symbol for symbol in _effective_forbidden_symbols(args) if symbol in code]
        if symbols:
            violations.append({"code_path": str(code_path.resolve()), "symbols": symbols})
    return violations


def _audit_skill_usage(run_dir: Path) -> list[dict[str, Any]]:
    usages: list[dict[str, Any]] = []
    if not run_dir.exists():
        return usages
    for code_path in sorted(run_dir.glob("trial_*/code.py")):
        try:
            code = code_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code = code_path.read_text(errors="replace")
        symbols = [
            symbol
            for symbol in DEFAULT_CLEAN_API_POLICY.allowed_skills
            if re.search(rf"\b{re.escape(symbol)}\s*\(", code)
        ]
        if symbols:
            usages.append({"code_path": str(code_path.resolve()), "symbols": symbols})
    return usages


def _scan_local_success_signals(run_dir: Path) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    if not run_dir.exists():
        return signals
    for summary_path in sorted(run_dir.glob("trial_*/summary.txt")):
        text = _read_text_if_exists(summary_path)
        stdout_text = _extract_environment_stdout(text) or text
        lowered = stdout_text.lower()
        hits: list[str] = []
        if "=== success" in lowered:
            hits.append("explicit_success_print")
        if re.search(r"goal achieved[^:\n]*:\s*true", lowered):
            hits.append("goal_achieved_true_print")
        if "opened sufficiently" in lowered or "middle drawer is open" in lowered:
            hits.append("drawer_open_success_print")
        drawer_progress = _max_drawer_progress_from_text(stdout_text)
        if drawer_progress is not None and drawer_progress >= 0.30:
            hits.append(f"drawer_progress_ge_0.30:{drawer_progress:.3f}")
        if hits:
            signals.append(
                {
                    "summary_path": str(summary_path.resolve()),
                    "trial_dir": str(summary_path.parent.resolve()),
                    "signals": hits,
                }
            )
    return signals


def _scan_stage_status_events(run_dir: Path) -> list[dict[str, Any]]:
    """Parse structured stage evidence printed by generated code."""
    events: list[dict[str, Any]] = []
    if not run_dir.exists():
        return events
    for summary_path in sorted(run_dir.glob("trial_*/summary.txt")):
        text = _read_text_if_exists(summary_path)
        stdout_text = _extract_environment_stdout(text) or text
        for line_no, evidence in _enumerate_stage_status_evidence(stdout_text):
            events.append(
                {
                    "summary_path": str(summary_path.resolve()),
                    "trial_dir": str(summary_path.parent.resolve()),
                    "line_no": line_no,
                    "payload": {
                        "stage_id": evidence.stage_id,
                        "status": evidence.status.value,
                        "evidence": evidence.rationale,
                        "repair": evidence.repair_hint or "",
                    },
                }
            )
    return events


def _enumerate_stage_status_evidence(text: str) -> list[tuple[int, Any]]:
    events = parse_stage_status_events(text)
    if not events:
        return []
    indexed: list[tuple[int, Any]] = []
    remaining = list(events)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not remaining:
            break
        if STAGE_STATUS_RE.search(line):
            indexed.append((line_no, remaining.pop(0)))
    return indexed


DRAWER_PROGRESS_RE = re.compile(
    r"(?:final\s+drawer\s+progress|drawer\s+progress\s+after\s+pull|drawer\s+progress)"
    r"[^0-9\n-]{0,120}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_environment_stdout(summary_text: str) -> str:
    marker = "Environment response:"
    if marker not in summary_text:
        return ""
    tail = summary_text.split(marker, 1)[1]
    stdout_marker = "\n  Stdout:"
    stderr_marker = "\n  Stderr:"
    if stdout_marker not in tail:
        return tail
    stdout_tail = tail.split(stdout_marker, 1)[1]
    if stderr_marker in stdout_tail:
        stdout_tail = stdout_tail.split(stderr_marker, 1)[0]
    return stdout_tail.strip()


def _max_drawer_progress_from_text(text: str) -> float | None:
    values: list[float] = []
    for match in DRAWER_PROGRESS_RE.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


STAGE_LINE_RE = re.compile(r"={2,}\s*(?:Stage\s*)?(?P<stage>[^=\n]{3,120})\s*={2,}", re.IGNORECASE)


def _build_replan_context(
    args: StagePlannerBatchArgs,
    task: LiberoTaskSpec,
    plan: TaskPlan,
    round_record: dict[str, Any],
) -> dict[str, Any] | None:
    if not args.replan_on_failure:
        return None
    failed_trials = round_record.get("failed_trials", [])
    if not failed_trials:
        return None
    if int(round_record.get("round", 1)) < max(1, args.replan_after_failed_rounds):
        return None

    trial = failed_trials[0]
    trial_dir = Path(str(trial.get("path", "")))
    summary_text = _read_text_if_exists(trial_dir / "summary.txt")
    code_text = _read_text_if_exists(trial_dir / "code.py")
    evidence = _compact_failure_evidence(summary_text, code_text, args.max_replan_summary_chars)
    stage_events = _extract_stage_status_events_from_text(summary_text)
    stage_hints = _infer_failed_stage_hints(summary_text, plan, stage_events)
    return {
        "trigger": "previous_round_failed",
        "failed_round": round_record.get("round"),
        "failed_trial": trial,
        "probable_failed_stages": stage_hints,
        "used_action_or_skill_symbols": sorted(set(ACTION_API_RE.findall(code_text))),
        "forbidden_symbol_violations": round_record.get("forbidden_symbol_violations", []),
        "skill_usage": round_record.get("skill_usage", []),
        "local_success_signals": round_record.get("local_success_signals", []),
        "stage_status_events": round_record.get("stage_status_events", []),
        "failure_evidence": evidence,
        "kite_feedback": round_record.get("kite_feedback"),
        "max_kite_feedback_images": int(args.max_kite_feedback_images),
        "historical_success_references": [],
        "zero_shot_note": "Historical successful CAP code is not inserted into replanning context.",
    }


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


def _compact_failure_evidence(summary_text: str, code_text: str, max_chars: int) -> dict[str, str]:
    merged = summary_text or ""
    if not merged and code_text:
        merged = code_text
    interesting_lines: list[str] = []
    for line in merged.splitlines():
        lowered = line.lower()
        if (
            "traceback" in lowered
            or "error" in lowered
            or "failed" in lowered
            or "not " in lowered
            or "stage" in lowered
            or "progress" in lowered
            or "alignment" in lowered
            or "reward" in lowered
            or "task completed" in lowered
        ):
            interesting_lines.append(line.rstrip())
    compact = "\n".join(interesting_lines[-120:])
    if not compact:
        compact = merged[-max_chars:]
    return {
        "summary_tail": merged[-max_chars:],
        "interesting_lines": compact[-max_chars:],
    }


def _extract_stage_status_events_from_text(summary_text: str) -> list[dict[str, Any]]:
    stdout_text = _extract_environment_stdout(summary_text) or summary_text
    return [
        {
            "stage_id": evidence.stage_id,
            "status": evidence.status.value,
            "evidence": evidence.rationale,
            "repair": evidence.repair_hint or "",
        }
        for evidence in parse_stage_status_events(stdout_text)
    ]


def _infer_failed_stage_hints(
    summary_text: str,
    plan: TaskPlan,
    stage_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    stage_events = stage_events or []
    for event in reversed(stage_events):
        status = str(event.get("status", "")).lower()
        if status in {"not_satisfied", "needs_repair", "uncertain", "failed", "failure"}:
            stage_id = str(event.get("stage_id", "unknown"))
            stage_name = next((stage.name for stage in plan.stages if stage.stage_id == stage_id), stage_id)
            hints.append(
                {
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "reason": f"structured_stage_status:{status}",
                }
            )
            break
    if hints:
        return hints
    stage_names = {stage.stage_id: stage.name for stage in plan.stages}
    seen_lines: list[str] = []
    for match in STAGE_LINE_RE.finditer(summary_text):
        seen_lines.append(match.group("stage").strip())
    tail = "\n".join(summary_text.splitlines()[-80:]).lower()
    for stage in plan.stages:
        stage_blob = " ".join([stage.stage_id, stage.name] + [check.name for check in stage.checks]).lower()
        if any(token in tail for token in stage_blob.split("_")):
            hints.append({"stage_id": stage.stage_id, "stage_name": stage.name, "reason": "mentioned_in_failure_tail"})
    if not hints and seen_lines:
        last = seen_lines[-1]
        for stage_id, name in stage_names.items():
            if name.lower() in last.lower() or stage_id.lower() in last.lower():
                hints.append({"stage_id": stage_id, "stage_name": name, "reason": "last_printed_stage"})
                break
        if not hints:
            hints.append({"stage_id": "unknown", "stage_name": last, "reason": "last_stage_like_print"})
    return hints[:3]


ACTION_API_RE = re.compile(
    r"\b("
    r"goto_pose|goto_pose_interactive_cartesian|goto_pose_checked|"
    r"goto_finger_midpoint_checked|open_gripper|close_gripper|"
    r"move_linear_delta|move_along_direction|move_finger_midpoint_delta|"
    r"move_finger_midpoint_delta_strict|move_to_joints|execute_joint_trajectory|"
    r"goto_home_joint_position|plan_grasp|plan_grasp_from_point_clouds|"
    r"solve_ik|solve_ik_strict|plan_grasp_trajectory|plan_with_grasped_object|"
    r"approach_and_grasp_object|local_placement_closure|pick_transport_release_object_to_target|pull_sleeved_drawer"
    r")\s*\("
)


def _find_success_code_references(
    args: StagePlannerBatchArgs,
    task: LiberoTaskSpec,
) -> list[dict[str, Any]]:
    """Find historical successful CAP code for offline failure analysis only."""
    examples: list[dict[str, Any]] = []
    if args.max_reference_examples <= 0:
        return examples

    task_id_prefix = f"{task.task_id:02d}_"
    task_slug = _slugify(task.task_language)
    for root_str in args.success_reference_roots:
        root = Path(root_str).expanduser()
        if not root.exists():
            continue
        for code_path in root.glob(f"**/{task.suite_name}/**/trial_*_taskcompleted_1/code.py"):
            if len(examples) >= args.max_reference_examples:
                return examples
            path_text = str(code_path)
            parent_bits = [part.lower() for part in code_path.parts]
            if not any(part.startswith(task_id_prefix) for part in parent_bits):
                continue
            if task_slug and task_slug not in _slugify(path_text):
                # Some historical dirs include canonical task names rather than
                # language strings, so task id remains the primary matcher.
                pass
            try:
                code = code_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = code_path.read_text(errors="replace")
            symbols = sorted(set(ACTION_API_RE.findall(code)))
            forbidden = DEFAULT_CLEAN_API_POLICY.find_forbidden_symbols(code)
            examples.append(
                {
                    "code_path": str(code_path.resolve()),
                    "trial_dir": str(code_path.parent.resolve()),
                    "action_symbols": symbols,
                    "forbidden_symbols": forbidden,
                    "clean_reference": not forbidden,
                    "line_count": len(code.splitlines()),
                    "snippet": _compact_code_snippet(code),
                }
            )
    return examples


def _compact_code_snippet(code: str, max_lines: int = 40) -> str:
    lines = []
    for line in code.splitlines():
        stripped = line.rstrip()
        if stripped:
            lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _finalize_run_record(record: dict[str, Any]) -> None:
    rounds = record.get("rounds", [])
    success_rounds = [
        item for item in rounds if item.get("successful_trial_count", 0) > 0
    ]
    record["round_count"] = len(rounds)
    record["trial_count"] = sum(item.get("trial_count", 0) for item in rounds)
    record["successful_trial_count"] = sum(
        item.get("successful_trial_count", 0) for item in rounds
    )
    record["failed_trial_count"] = sum(len(item.get("failed_trials", [])) for item in rounds)
    record["task_success_any_seed"] = bool(success_rounds)
    record["task_success"] = bool(success_rounds)
    record["success_round_any_seed"] = success_rounds[0]["round"] if success_rounds else None
    record["forbidden_symbol_violation_count"] = sum(
        len(item.get("forbidden_symbol_violations", [])) for item in rounds
    )
    record["local_success_signal_count"] = sum(
        len(item.get("local_success_signals", [])) for item in rounds
    )
    record["local_success_but_env_failed"] = bool(
        record["local_success_signal_count"] > 0 and not record["task_success"]
    )
    record["max_rounds_exhausted"] = bool(
        rounds and rounds[-1].get("stop_reason") == "max_rounds_reached"
    )


def _refresh_report_summary(report: dict[str, Any]) -> None:
    runs = report.get("runs", [])
    rounds = [round_record for run in runs for round_record in run.get("rounds", [])]
    report["summary"] = {
        "task_count": len({(run.get("suite_name"), run.get("task_id")) for run in runs}),
        "run_count": len(runs),
        "round_count": len(rounds),
        "completed_run_count": sum(run.get("status") == "completed" for run in runs),
        "launch_failed_run_count": sum(run.get("status") == "launch_failed" for run in runs),
        "timeout_run_count": sum(
            any(item.get("stop_reason") == "round_timeout" for item in run.get("rounds", []))
            for run in runs
        ),
        "task_success_count": sum(bool(run.get("task_success")) for run in runs),
        "task_success_any_seed_count": sum(
            bool(run.get("task_success_any_seed")) for run in runs
        ),
        "max_rounds_exhausted_count": sum(
            bool(run.get("max_rounds_exhausted")) for run in runs
        ),
        "executed_trial_count": sum(item.get("trial_count", 0) for item in rounds),
        "successful_trial_count": sum(
            item.get("successful_trial_count", 0) for item in rounds
        ),
        "failed_trial_count": sum(len(item.get("failed_trials", [])) for item in rounds),
        "forbidden_symbol_violation_count": sum(
            len(item.get("forbidden_symbol_violations", [])) for item in rounds
        ),
        "skill_usage_count": sum(len(item.get("skill_usage", [])) for item in rounds),
        "local_success_signal_count": sum(
            len(item.get("local_success_signals", [])) for item in rounds
        ),
        "local_success_but_env_failed_count": sum(
            bool(run.get("local_success_but_env_failed")) for run in runs
        ),
        "timeout_round_count": sum(item.get("stop_reason") == "round_timeout" for item in rounds),
    }


def _write_experiment_manifest(
    output_root: Path,
    args: StagePlannerBatchArgs,
    tasks: list[LiberoTaskSpec],
) -> None:
    lines = [
        "# LIBERO Stage-Planner Clean CaP Experiment",
        "",
        f"- Created at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Method condition: {args.experiment_label}",
        f"- Task count: {len(tasks)}",
        f"- Suites: {', '.join(args.suites)}",
        f"- Trials per task per round: {args.total_trials}",
        f"- Feedback rounds: {args.feedback_rounds}",
        f"- Round timeout seconds: {args.round_timeout_sec}",
        f"- KITE feedback: {args.use_kite_feedback}",
        f"- Oracle-readonly coordinates: {args.oracle_readonly}",
        f"- SAM3 perception: {args.use_sam3_perception}",
        f"- SAM3 service URL: {os.getenv('SAM3_SERVICE_URL')}",
        f"- Point-prompt service enabled: {args.allow_point_prompt_molmo}",
        f"- Point-prompt backend model: {os.getenv('MOLMO_MODEL') if args.allow_point_prompt_molmo else 'disabled'}",
        f"- Point-prompt backend URL: {os.getenv('MOLMO_BASE_URL') if args.allow_point_prompt_molmo else 'disabled'}",
        f"- CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}",
        f"- Replan on failure: {args.replan_on_failure}",
        f"- Replan after failed rounds: {args.replan_after_failed_rounds}",
        "",
        "## Scientific Boundary",
        "",
        "- CaP-X is used as a reference implementation and baseline infrastructure.",
        "- This experiment studies stage-plan conditioning for clean policy-code generation.",
        "- Codex-derived CAP skills, stage checks, verifier thresholds, and repair routing are fixed method components for this run.",
        "- Codex is not used as a runtime assistant during the benchmark; no per-task boundary edits are made after launch.",
        "- MIMO generates or repairs CAP under the fixed stage/verifier boundary conditions.",
        "- When enabled, KITE feedback is extracted only from the failed trial video/stage trace and inserted into the next MIMO round.",
        "- When enabled, point_prompt_molmo is treated as a generic VLM point-prompt API; in this run it may be backed by MIMO rather than a separate Molmo server.",
        "- Generated code must not directly modify simulator state or call task-completion helpers.",
        "- Versioned robot-motion skills may be used when they act through the robot and remain auditable.",
        "- Historical successful CAP code may be logged for offline audit only and is not inserted into prompts.",
    ]
    (output_root / "experiment_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "libero_stage_planner_manifest_v0.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "tasks": [
            {
                "suite_name": task.suite_name,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_language": task.task_language,
                "bddl_file_path": task.bddl_file_path,
                "source_bddl": task.source_bddl,
                "success_predicates": list(task.success_predicates),
                "diagnostics": list(task.diagnostics),
                "slug": task.slug,
            }
            for task in tasks
        ],
    }
    _write_json(output_root / "experiment_manifest.json", manifest)


def _load_existing_record(
    args: StagePlannerBatchArgs,
    output_root: Path,
    model: str,
    task: LiberoTaskSpec,
) -> dict[str, Any] | None:
    if not args.resume or args.force_rerun_tasks:
        return None
    record_path = (
        output_root
        / task.suite_name
        / task.slug
        / f"{model.replace('/', '_')}_stage_planner_record.json"
    )
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if record.get("status") != "completed":
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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "task"


if __name__ == "__main__":
    main(tyro.cli(StagePlannerBatchArgs))
