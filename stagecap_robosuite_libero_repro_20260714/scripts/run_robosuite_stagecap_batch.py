"""RoboSuite privileged comparison: Agent0 vs Stage-CAP vs Stage-CAP+KITE.

用法::

    # privileged Agent0 基线
    python scripts/run_robosuite_stagecap_batch.py --condition agent0 --total-trials 10

    # Stage-CAP + KITE
    python scripts/run_robosuite_stagecap_batch.py --condition stagecap_kite --total-trials 10
"""
from __future__ import annotations
import copy, json, os, re, sys, time, traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml, tyro

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from capx.envs.launch import LaunchArgs, main as launch_main

ROBOSUITE_TASKS: list[dict[str, Any]] = [
    {
        "name": "cube_lifting",
        "agent0_config": "env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml",
        "stagecap_config": "env_configs/cube_lifting/franka_robosuite_cube_lifting_stagecap_privileged.yaml",
        "task_language": "pick up the red cube and lift it",
    },
    {
        "name": "cube_restack",
        "agent0_config": "env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml",
        "stagecap_config": "env_configs/cube_restack/franka_robosuite_cube_restack_stagecap_privileged.yaml",
        "task_language": "restack the cubes by moving green off red, then stacking red on green",
    },
    {
        "name": "nut_assembly",
        "agent0_config": "env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml",
        "stagecap_config": "env_configs/nut_assembly/franka_robosuite_nut_assembly_stagecap_privileged.yaml",
        "task_language": "grasp and insert the brown square nut onto the brown square block",
    },
    {
        "name": "cube_stack",
        "agent0_config": "env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
        "stagecap_config": "env_configs/cube_stack/franka_robosuite_cube_stack_stagecap_privileged.yaml",
        "task_language": "pick up the red cube and gently stack it on top of the green cube",
    },
    {
        "name": "two_arm_lift",
        "agent0_config": "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml",
        "stagecap_config": "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_stagecap_privileged.yaml",
        "task_language": "coordinate two arms to lift a pot by grasping each handle and lifting simultaneously",
    },
    {
        "name": "spill_wipe",
        "agent0_config": "env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml",
        "stagecap_config": "env_configs/spill_wipe/franka_robosuite_spill_wipe_stagecap_privileged.yaml",
        "task_language": "wipe up the brown spill with the attached wiping tool",
    },
    {
        "name": "two_arm_handover",
        "agent0_config": "env_configs/two_arm_handover/two_arm_handover_privileged.yaml",
        "stagecap_config": "env_configs/two_arm_handover/two_arm_handover_stagecap_privileged.yaml",
        "task_language": "arm 0 should lift the hammer and hand it over to arm 1 by the handle",
    },
]


@dataclass
class RoboSuiteBatchArgs:
    condition: str = "agent0"
    """Experiment condition: 'agent0', 'stagecap', or 'stagecap_kite'."""
    tasks: list[str] = field(
        default_factory=lambda: [
            "cube_lifting",
            "cube_stack",
            "cube_restack",
            "nut_assembly",
            "spill_wipe",
            "two_arm_lift",
            "two_arm_handover",
        ]
    )
    output_dir: str = ""
    total_trials: int = 10
    num_workers: int = 2
    model: str = ""
    server_url: str = ""
    api_key: str | None = None
    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    max_total_stage_attempts: int = 10
    stage_snapshot_chars: int = 2200
    deterministic_fast_paths: bool = True
    """Use audited deterministic action code in StageCaP conditions."""
    debug: bool = False
    round_timeout_sec: float = 1200.0
    record_video: bool = True
    skip_api_servers: bool = False


def _resolve_api(args: RoboSuiteBatchArgs) -> tuple[str, str, str | None]:
    model = args.model or os.getenv("MIMO_MODEL", "mimo-v2.5")
    base_url = (args.server_url or os.getenv("MIMO_BASE_URL", "http://127.0.0.1:8110")).rstrip("/")
    server_url = base_url + "/chat/completions"
    api_key = args.api_key or os.getenv("MIMO_API_KEY")
    return model, server_url, api_key


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate_condition_config(
    base_config: dict[str, Any],
    condition: str,
    config_path: str,
) -> None:
    """Enforce the privileged-observation and API-isolation experiment protocol."""
    env_cfg = base_config.get("env", {}).get("cfg", {})
    apis = tuple(str(name) for name in env_cfg.get("apis", ()))
    stage_apis = tuple(name for name in apis if "StageApi" in name)

    if env_cfg.get("privileged") is not True:
        raise ValueError(f"{config_path}: RoboSuite comparison requires privileged: true")
    if condition == "agent0":
        if stage_apis:
            raise ValueError(
                f"{config_path}: Agent0 must not receive StageCaP APIs: {stage_apis}"
            )
        if not any(name.endswith("PrivilegedApi") for name in apis):
            raise ValueError(f"{config_path}: Agent0 requires a CapX privileged API")
    elif condition in {"stagecap", "stagecap_kite"}:
        if not stage_apis:
            raise ValueError(f"{config_path}: StageCaP requires an audited StageApi")


def _build_config(
    base_config: dict[str, Any],
    args: RoboSuiteBatchArgs,
    output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["output_dir"] = str(output_dir)
    config["trials"] = args.total_trials
    config["num_workers"] = args.num_workers
    config["record_video"] = args.record_video
    if args.skip_api_servers:
        config.pop("api_servers", None)

    if args.condition == "agent0":
        config["use_stage_verifiable_runner"] = False
        config["use_deterministic_stage_fast_paths"] = False
        config["use_online_stage_kite_feedback"] = False
        config["max_online_stage_feedback_images"] = 0
    elif args.condition in {"stagecap", "stagecap_kite"}:
        config["use_stage_verifiable_runner"] = True
        config["use_deterministic_stage_fast_paths"] = args.deterministic_fast_paths
        config["use_online_stage_kite_feedback"] = args.condition == "stagecap_kite"
        config["max_total_stage_attempts"] = args.max_total_stage_attempts
        config["stage_snapshot_chars"] = args.stage_snapshot_chars
        config["save_stage_snapshots"] = True
        config["save_stage_prompts"] = True
        config["max_online_stage_feedback_images"] = 1 if args.condition == "stagecap_kite" else 0
        config["max_kite_feedback_images"] = 1 if args.condition == "stagecap_kite" else 0
    else:
        raise ValueError(f"未知条件: {args.condition!r}")
    return config


TRIAL_RE = re.compile(
    r"^trial_\d+_sandboxrc_-?\d+_reward_(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<done>[01])$"
)


def _scan_trials(output_dir: Path) -> dict[str, int]:
    total, success = 0, 0
    if output_dir.exists():
        for entry in output_dir.iterdir():
            m = TRIAL_RE.match(entry.name)
            if m:
                total += 1
                if int(m.group("done")) == 1 or float(m.group("reward")) >= 1.0:
                    success += 1
    return {"total": total, "success": success}


def _model_output_dir(output_dir: Path, model: str) -> Path:
    """Mirror capx.envs.runner._setup_output_dir for report scanning."""
    return output_dir.parent / model.replace("/", "_") / output_dir.name


def _run_task(
    task: dict[str, Any],
    args: RoboSuiteBatchArgs,
    model: str,
    server_url: str,
    api_key: str | None,
    task_out_dir: Path,
) -> dict[str, Any]:
    config_key = "agent0_config" if args.condition == "agent0" else "stagecap_config"
    selected_config = str(task[config_key])
    base_config = _load_yaml(ROOT / selected_config)
    _validate_condition_config(base_config, args.condition, selected_config)
    run_dir = task_out_dir / "run"
    config = _build_config(base_config, args, run_dir)

    config_path = task_out_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    launch_args = LaunchArgs(
        config_path=str(config_path),
        server_url=server_url,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        api_key=api_key,
        total_trials=args.total_trials,
        num_workers=args.num_workers,
        record_video=args.record_video,
        output_dir=None,
        debug=args.debug,
        use_oracle_code=False,
    )

    start = time.time()
    status, error = "started", None
    try:
        launch_main(launch_args)
        status = "completed"
    except TimeoutError as exc:
        status, error = "timeout", repr(exc)
        print(f"  [WARN] {task['name']} 超时: {exc}")
    except Exception as exc:
        status, error = "launch_failed", repr(exc)
        print(f"  [ERROR] {task['name']}: {exc}")
        traceback.print_exc()

    elapsed = time.time() - start
    resolved_run_dir = _model_output_dir(run_dir, model)
    stats = _scan_trials(resolved_run_dir)
    return {
        "task": task["name"],
        "condition": args.condition,
        "config": selected_config,
        "status": status,
        "error": error,
        "elapsed_sec": round(elapsed, 1),
        "total_trials": stats["total"],
        "success_trials": stats["success"],
        "tsr": stats["success"] / max(stats["total"], 1),
        "output_dir": str(resolved_run_dir),
    }


def main(args: RoboSuiteBatchArgs) -> None:
    model, server_url, api_key = _resolve_api(args)

    if not args.output_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / f"outputs/robosuite_{args.condition}_{ts}"
    else:
        out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    task_filter = set(args.tasks)
    tasks = [t for t in ROBOSUITE_TASKS if t["name"] in task_filter]

    print(f"\n{'='*60}")
    print(f"  RoboSuite 实验  条件={args.condition}")
    print(f"  任务: {[t['name'] for t in tasks]}")
    print(f"  输出: {out_root}")
    print(f"  模型: {model}  workers={args.num_workers}  trials={args.total_trials}")
    print(f"{'='*60}\n")

    results: list[dict[str, Any]] = []
    for task in tasks:
        task_out = out_root / task["name"]
        task_out.mkdir(parents=True, exist_ok=True)
        print(f"[{args.condition}] >>> {task['name']} ...")
        rec = _run_task(task, args, model, server_url, api_key, task_out)
        results.append(rec)
        print(
            f"[{args.condition}] <<< {task['name']}  "
            f"TSR={rec['success_trials']}/{rec['total_trials']} ({rec['tsr']:.0%})"
            f"  {rec['elapsed_sec']:.0f}s"
        )

    report_path = out_root / "robosuite_stagecap_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "robosuite_stagecap_report_v0.1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "condition": args.condition,
                "model": model,
                "observation_mode": "privileged",
                "api_profile": (
                    "capx_privileged"
                    if args.condition == "agent0"
                    else "stagecap_audited_privileged"
                ),
                "stage_runner": args.condition != "agent0",
                "deterministic_fast_paths": bool(
                    args.condition != "agent0" and args.deterministic_fast_paths
                ),
                "total_trials": args.total_trials,
                "results": results,
            },
            f, indent=2, ensure_ascii=False,
        )

    total_s = sum(r["success_trials"] for r in results)
    total_t = sum(r["total_trials"] for r in results)
    print(f"\n{'='*60}")
    print(f"  汇总  条件={args.condition}")
    for r in results:
        print(f"  {'✓' if r['tsr']>=0.5 else '·'} {r['task']:20s}  {r['success_trials']:3d}/{r['total_trials']:3d}  {r['tsr']:5.0%}")
    print(f"  {'总计':20s}  {total_s:3d}/{total_t:3d}  {total_s/max(total_t,1):.0%}")
    print(f"\n  报告: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main(tyro.cli(RoboSuiteBatchArgs))
