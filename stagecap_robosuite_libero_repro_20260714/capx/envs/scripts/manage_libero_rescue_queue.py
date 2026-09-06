"""Keep LIBERO suite-parallel rescues within a fixed rollout concurrency budget."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


RECORD_GLOB = "**/*_vifailback_capx_record.json"
DEFAULT_VIFAILBACK_SCRIPT = (
    "/home/lifuchun/data/capx/repos/cap-x/"
    "capx/envs/scripts/run_libero_vifailback_gpu_pool.sh"
)
DEFAULT_SUITE_ORDER = ("libero_10", "libero_object", "libero_spatial", "libero_goal")


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    main_root = (repo_root / args.main_root).resolve()
    rescue_root = (repo_root / args.rescue_root).resolve()
    state_path = rescue_root / "rescue_queue_state.json"
    pid_path = rescue_root / "controller.pid"
    log_root = rescue_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    while True:
        state = _load_state(state_path)
        state.setdefault("created_at", _now())
        state["updated_at"] = _now()
        state["main_root"] = str(main_root)
        state["rescue_root"] = str(rescue_root)
        state["max_concurrent"] = args.max_concurrent
        state["suite_order"] = args.suite_order
        state.setdefault("jobs", [])

        _refresh_jobs(state)
        pending = _discover_pending_rescues(main_root, rescue_root, state, args.max_attempts)
        active_count = _active_rollout_count(main_root, state)
        active_suite_counts = _active_rollout_counts_by_suite(main_root, state)
        state["active_suite_counts"] = active_suite_counts
        slots = max(0, args.max_concurrent - active_count)
        selected = _select_suite_balanced_pending(
            pending,
            active_suite_counts=active_suite_counts,
            slots=slots,
            suite_order=args.suite_order,
        )

        launched: list[dict[str, Any]] = []
        for item in selected:
            launched.append(
                _launch_rescue(
                    item=item,
                    state=state,
                    repo_root=repo_root,
                    rescue_root=rescue_root,
                    log_root=log_root,
                    args=args,
                )
            )

        launched_keys = {job["key"] for job in launched}
        state["pending"] = [item for item in pending if item["key"] not in launched_keys]
        state["active_rollout_count"] = active_count + len(launched)
        _write_json(state_path, state)

        if args.once:
            _print_summary(state_path, state, launched)
            return
        time.sleep(args.poll_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/lifuchun/data/capx/repos/cap-x")
    parser.add_argument(
        "--main-root",
        default="outputs/libero_suite_parallel_5trial_5round_v8",
        help="Suite-parallel run root that contains workers.tsv.",
    )
    parser.add_argument(
        "--rescue-root",
        default="outputs/libero_suite_parallel_5trial_5round_rescue_v9",
    )
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--suite-order",
        nargs="+",
        default=list(DEFAULT_SUITE_ORDER),
        help=(
            "Preferred suite lanes for rescue launches. Pending jobs are selected "
            "to cover these suites before filling extra slots."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--venv-python", default=".venv-libero/bin/python")
    parser.add_argument(
        "--base-config-path",
        default="env_configs/libero/franka_libero_spatial_0_privileged.yaml",
    )
    parser.add_argument("--total-trials", type=int, default=5)
    parser.add_argument("--feedback-rounds", type=int, default=5)
    parser.add_argument("--launch-retries", type=int, default=3)
    parser.add_argument("--launch-retry-sleep", type=float, default=45.0)
    parser.add_argument("--vifailback-script", default=DEFAULT_VIFAILBACK_SCRIPT)
    parser.add_argument("--vifailback-gpu", default="3")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pid_running(pid: int | str | None) -> bool:
    if pid is None:
        return False
    try:
        return subprocess.run(
            ["kill", "-0", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except (TypeError, ValueError):
        return False


def _active_rollout_count(main_root: Path, state: dict[str, Any]) -> int:
    active = 0
    workers_path = main_root / "workers.tsv"
    if workers_path.exists():
        for line in workers_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and _pid_running(parts[1]):
                active += 1
    for job in state.get("jobs", []):
        if job.get("status") == "running" and _pid_running(job.get("pid")):
            active += 1
    return active


def _active_rollout_counts_by_suite(main_root: Path, state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    workers_path = main_root / "workers.tsv"
    if workers_path.exists():
        for line in workers_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and _pid_running(parts[1]):
                suite = parts[0]
                counts[suite] = counts.get(suite, 0) + 1
    for job in state.get("jobs", []):
        if job.get("status") != "running" or not _pid_running(job.get("pid")):
            continue
        suite = str(job.get("suite"))
        counts[suite] = counts.get(suite, 0) + 1
    return counts


def _refresh_jobs(state: dict[str, Any]) -> None:
    for job in state.get("jobs", []):
        if job.get("status") != "running":
            continue
        if _pid_running(job.get("pid")):
            continue
        record = _find_task_record(Path(job["output_dir"]), job["suite"], int(job["task_id"]))
        job["finished_at"] = _now()
        if record and record.get("task_success"):
            job["status"] = "completed_success"
            job["record_path"] = record.get("_path")
            job["success_round"] = record.get("success_round")
        elif record:
            job["status"] = "completed_failed"
            job["record_path"] = record.get("_path")
            job["max_rounds_exhausted"] = record.get("max_rounds_exhausted")
        else:
            job["status"] = "stopped_no_record"


def _select_suite_balanced_pending(
    pending: list[dict[str, Any]],
    *,
    active_suite_counts: dict[str, int],
    slots: int,
    suite_order: list[str],
) -> list[dict[str, Any]]:
    if slots <= 0 or not pending:
        return []

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    suite_rank = {suite: idx for idx, suite in enumerate(suite_order)}

    def ordered_candidates() -> list[dict[str, Any]]:
        return sorted(
            pending,
            key=lambda item: (
                suite_rank.get(str(item["suite"]), len(suite_rank)),
                int(item["task_id"]),
                item["key"],
            ),
        )

    candidates = ordered_candidates()
    while len(selected) < slots:
        best: dict[str, Any] | None = None
        best_key: tuple[int, int, int, str] | None = None
        selected_counts = _selected_counts_by_suite(selected)
        for item in candidates:
            if item["key"] in selected_keys:
                continue
            suite = str(item["suite"])
            load = active_suite_counts.get(suite, 0) + selected_counts.get(suite, 0)
            rank = suite_rank.get(suite, len(suite_rank))
            candidate_key = (load, selected_counts.get(suite, 0), rank, item["key"])
            if best_key is None or candidate_key < best_key:
                best = item
                best_key = candidate_key
        if best is None:
            break
        selected.append(best)
        selected_keys.add(best["key"])
    return selected


def _selected_counts_by_suite(selected: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in selected:
        suite = str(item["suite"])
        counts[suite] = counts.get(suite, 0) + 1
    return counts


def _discover_pending_rescues(
    main_root: Path,
    rescue_root: Path,
    state: dict[str, Any],
    max_attempts: int,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    successful = _successful_rescue_keys(rescue_root)
    running = {
        _job_key(job)
        for job in state.get("jobs", [])
        if job.get("status") == "running" and _pid_running(job.get("pid"))
    }
    attempts = _attempt_counts(state)

    for record_path in sorted(main_root.glob(RECORD_GLOB)):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("task_success"):
            continue
        if not record.get("max_rounds_exhausted"):
            continue
        suite = str(record.get("suite_name", ""))
        task_id = int(record.get("task_id"))
        key = _task_key(suite, task_id)
        if key in successful or key in running:
            continue
        if attempts.get(key, 0) >= max_attempts:
            continue
        pending.append(
            {
                "key": key,
                "suite": suite,
                "task_id": task_id,
                "task_language": record.get("task_language"),
                "source_record": str(record_path),
                "previous_round_count": record.get("round_count"),
                "previous_successful_trials": record.get("successful_trial_count"),
                "previous_trial_count": record.get("trial_count"),
            }
        )
    return pending


def _successful_rescue_keys(rescue_root: Path) -> set[str]:
    keys: set[str] = set()
    for record_path in rescue_root.glob(RECORD_GLOB):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("task_success"):
            keys.add(_task_key(str(record.get("suite_name")), int(record.get("task_id"))))
    return keys


def _attempt_counts(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in state.get("jobs", []):
        key = _job_key(job)
        counts[key] = max(counts.get(key, 0), int(job.get("attempt", 0)))
    return counts


def _task_key(suite: str, task_id: int) -> str:
    return f"{suite}:{task_id:02d}"


def _job_key(job: dict[str, Any]) -> str:
    return _task_key(str(job.get("suite")), int(job.get("task_id")))


def _launch_rescue(
    *,
    item: dict[str, Any],
    state: dict[str, Any],
    repo_root: Path,
    rescue_root: Path,
    log_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempt = _attempt_counts(state).get(item["key"], 0) + 1
    suite = item["suite"]
    task_id = int(item["task_id"])
    run_name = f"{suite}_{task_id:02d}_attempt_{attempt:02d}"
    output_dir = rescue_root / run_name
    log_path = log_root / f"{run_name}.log"
    prefix = f"rescue_v9_{suite}_{task_id:02d}_attempt_{attempt:02d}"

    command = [
        args.venv_python,
        "capx/envs/scripts/run_libero_cap_vifailback_batch.py",
        "--suites",
        suite,
        "--task-ids",
        str(task_id),
        "--total-trials",
        str(args.total_trials),
        "--num-workers",
        "1",
        "--feedback-rounds",
        str(args.feedback_rounds),
        "--launch-retries",
        str(args.launch_retries),
        "--launch-retry-sleep",
        str(args.launch_retry_sleep),
        "--base-config-path",
        args.base_config_path,
        "--skip-api-servers",
        "--record-video",
        "True",
        "--output-dir",
        str(output_dir),
        "--vifailback-name-prefix",
        prefix,
        "--vifailback-script",
        args.vifailback_script,
        "--vifailback-gpu",
        args.vifailback_gpu,
        "--use-img-differencing",
        "False",
        "--use-parallel-ensemble",
        "False",
        "--use-multimodel",
        "False",
    ]
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    job = {
        **item,
        "attempt": attempt,
        "status": "running",
        "pid": proc.pid,
        "launched_at": _now(),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "command": command,
    }
    state.setdefault("jobs", []).append(job)
    return job


def _find_task_record(root: Path, suite: str, task_id: int) -> dict[str, Any] | None:
    for record_path in root.glob(RECORD_GLOB):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("suite_name") == suite and int(record.get("task_id")) == task_id:
            record["_path"] = str(record_path)
            return record
    return None


def _print_summary(path: Path, state: dict[str, Any], launched: list[dict[str, Any]]) -> None:
    print(f"State: {path}")
    print(f"Active rollout count: {state.get('active_rollout_count')}")
    print(f"Pending rescues: {len(state.get('pending', []))}")
    for item in state.get("pending", []):
        print(f"  pending {item['key']}: {item.get('task_language')}")
    print(f"Launched: {len(launched)}")
    for job in launched:
        print(f"  launched {job['key']} pid={job['pid']} log={job['log_path']}")


if __name__ == "__main__":
    main()
