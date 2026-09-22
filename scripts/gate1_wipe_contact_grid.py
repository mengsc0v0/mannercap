"""Plan or audit a multi-seed x surface-penetration Wipe probe grid.

The default action writes a plan or audits an existing plan's outputs. Passing
``--run`` is the only way this entry point launches MuJoCo probes. The launched
child is the existing ``gate1_wipe_probe.py``; this runner adds no simulated
force and does not treat geometric surface penetration as a known load.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis_v2.wipe_contact_grid import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    DEFAULT_SEEDS,
    DEFAULT_SURFACE_PENETRATIONS_M,
    DEFAULT_WIPE_ATTEMPTS,
    PLAN_SCHEMA,
    audit_grid,
    build_existing_runs_plan,
    build_grid_plan,
    render_audit_markdown,
    validate_plan,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _new_plan(args: argparse.Namespace) -> dict[str, Any]:
    seeds = args.seed if args.seed is not None else DEFAULT_SEEDS
    penetrations = (
        args.surface_penetration_m
        if args.surface_penetration_m is not None
        else DEFAULT_SURFACE_PENETRATIONS_M
    )
    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
    wipe_attempts = (
        args.wipe_attempts if args.wipe_attempts is not None else DEFAULT_WIPE_ATTEMPTS
    )
    return build_grid_plan(
        seeds,
        penetrations,
        max_steps=max_steps,
        wipe_attempts=wipe_attempts,
    )


def _check_existing_plan_overrides(plan: Mapping[str, Any], args: argparse.Namespace) -> None:
    if args.seed is not None and list(args.seed) != plan.get("seeds"):
        raise ValueError("--seed values do not match the existing plan")
    if (
        args.surface_penetration_m is not None
        and list(args.surface_penetration_m) != plan.get("surface_penetrations_m")
    ):
        raise ValueError("--surface-penetration-m values do not match the existing plan")
    budget = plan.get("budget", {})
    if args.max_steps is not None and budget.get("environment_max_steps") != args.max_steps:
        raise ValueError("--max-steps does not match the existing plan")
    if args.wipe_attempts is not None and budget.get("wipe_attempts") != args.wipe_attempts:
        raise ValueError("--wipe-attempts does not match the existing plan")


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output path is not a directory: {output}")
    else:
        output.mkdir(parents=True)


def _load_or_create_plan(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output = args.output.expanduser().resolve()
    _prepare_output(output)
    plan_path = output / "plan.json"
    if args.audit_run:
        if args.run:
            raise ValueError("--audit-run is read-only and cannot be combined with --run")
        if any(
            value is not None
            for value in (
                args.seed,
                args.surface_penetration_m,
                args.max_steps,
                args.wipe_attempts,
            )
        ):
            raise ValueError("grid parameters cannot be combined with --audit-run")
        if plan_path.exists():
            raise ValueError(f"refusing to replace existing plan: {plan_path}")
        if any(output.iterdir()):
            raise ValueError(f"audit output directory must be empty: {output}")
        plan = build_existing_runs_plan(args.audit_run)
        _write_json(plan_path, plan)
        return output, plan

    if plan_path.exists():
        plan = _load_json(plan_path)
        validate_plan(plan)
        _check_existing_plan_overrides(plan, args)
        return output, plan

    if any(output.iterdir()):
        raise ValueError(
            f"output directory is non-empty and has no plan.json; refusing to overwrite: {output}"
        )
    plan = _new_plan(args)
    _write_json(plan_path, plan)
    return output, plan


def _safe_planned_paths(output: Path, case: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    case_directory_value = case.get("case_directory")
    run_directory_value = case.get("run_directory")
    if not isinstance(case_directory_value, str) or not isinstance(run_directory_value, str):
        raise ValueError("planned case is missing case_directory/run_directory")
    case_directory = (output / case_directory_value).resolve()
    run_directory = (output / run_directory_value).resolve()
    for path in (case_directory, run_directory):
        try:
            path.relative_to(output.resolve())
        except ValueError as error:
            raise ValueError(f"planned case path escapes output root: {path}") from error
    if not run_directory.is_relative_to(case_directory):
        raise ValueError("probe output directory must be inside its case directory")
    return case_directory, run_directory, case_directory / "process.json"


def _write_audit(output: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    audit = audit_grid(output, plan)
    _write_json(output / "audit.json", audit)
    (output / "summary.md").write_text(render_audit_markdown(audit), encoding="utf-8")
    return audit


def _text_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_planned_probes(
    output: Path,
    plan: Mapping[str, Any],
    *,
    timeout_s: float | None,
) -> dict[str, Any]:
    if plan.get("mode") != "full_factorial":
        raise ValueError("--run is available only for a full-factorial probe plan")
    probe_script = ROOT / str(plan.get("probe", {}).get("script", "scripts/gate1_wipe_probe.py"))
    if not probe_script.is_file():
        raise FileNotFoundError(f"real Wipe probe script not found: {probe_script}")

    for case in plan["cases"]:
        case_directory, run_directory, process_path = _safe_planned_paths(output, case)
        if (run_directory / "probe.json").exists():
            # Existing raw evidence is immutable. Audit it and move to the next cell.
            _write_audit(output, plan)
            continue

        previous_process = None
        if process_path.is_file():
            try:
                previous_process = _load_json(process_path)
            except (OSError, json.JSONDecodeError, ValueError):
                previous_process = {"status": "unreadable"}
            # A previous attempt, including an interrupted one, is preserved.
            # A separate plan/output directory is required for a fresh execution.
            _write_audit(output, plan)
            continue

        if case_directory.exists() and any(case_directory.iterdir()):
            process_record = {
                "status": "execution_error",
                "started_at_utc": None,
                "finished_at_utc": _utc_now(),
                "exit_code": None,
                "error": (
                    "case directory already contains files but has no process record; "
                    "preserved without rerunning to avoid overwriting evidence"
                ),
                "previous_process_record": previous_process,
            }
            _write_json(process_path, process_record)
            _write_audit(output, plan)
            continue

        case_directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(probe_script), "--output", str(run_directory)]
        command.extend(str(value) for value in case.get("probe_arguments", []))
        started_at = _utc_now()
        started_monotonic = time.perf_counter()
        running_record = {
            "status": "running",
            "command": command,
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "exit_code": None,
            "timeout_s": timeout_s,
            "requested_budget": case.get("budget"),
        }
        _write_json(process_path, running_record)
        _write_audit(output, plan)

        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            process_record = {
                **running_record,
                "status": "completed",
                "finished_at_utc": _utc_now(),
                "elapsed_s": time.perf_counter() - started_monotonic,
                "exit_code": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            if completed.returncode != 0:
                process_record["error"] = f"probe exited with code {completed.returncode}"
        except subprocess.TimeoutExpired as error:
            process_record = {
                **running_record,
                "status": "execution_error",
                "finished_at_utc": _utc_now(),
                "elapsed_s": time.perf_counter() - started_monotonic,
                "exit_code": None,
                "timed_out": True,
                "error": f"probe timed out after {timeout_s} seconds",
                "stdout": _text_output(error.stdout),
                "stderr": _text_output(error.stderr),
                "traceback": traceback.format_exc(),
            }
        except Exception as error:
            process_record = {
                **running_record,
                "status": "execution_error",
                "finished_at_utc": _utc_now(),
                "elapsed_s": time.perf_counter() - started_monotonic,
                "exit_code": None,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        _write_json(process_path, process_record)
        audit = _write_audit(output, plan)
        print(
            json.dumps(
                {
                    "case_id": case.get("case_id"),
                    "process_status": process_record.get("status"),
                    "exit_code": process_record.get("exit_code"),
                    "run_state": audit["cases"][
                        next(
                            index
                            for index, audited_case in enumerate(audit["cases"])
                            if audited_case.get("case_id") == case.get("case_id")
                        )
                    ]["run_state"],
                    "audit": str(output / "audit.json"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return _write_audit(output, plan)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="grid plan/audit directory")
    parser.add_argument("--seed", type=int, action="append", help="seed; repeat for a grid")
    parser.add_argument(
        "--surface-penetration-m",
        type=float,
        action="append",
        help="geometric/controller target offset in metres; repeat for a grid",
    )
    parser.add_argument("--max-steps", type=int, help="probe environment step cap")
    parser.add_argument("--wipe-attempts", type=int, help="wipe_spill_and_verify call limit")
    parser.add_argument(
        "--audit-run",
        type=Path,
        action="append",
        help="existing gate1_wipe_probe output directory to audit; repeat as needed",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly run planned real probes; without this, the command only plans/audits",
    )
    parser.add_argument("--timeout-s", type=float, help="optional per-probe process timeout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout_s is not None and (
        not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0
    ):
        parser.error("--timeout-s must be positive and finite")
    try:
        output, plan = _load_or_create_plan(args)
        if args.run:
            audit = _run_planned_probes(output, plan, timeout_s=args.timeout_s)
        else:
            audit = _write_audit(output, plan)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "plan": str(output / "plan.json"),
                    "audit": str(output / "audit.json"),
                    "summary": str(output / "summary.md"),
                    "matrix_sha256": plan.get("matrix_sha256"),
                    "counts": audit.get("counts"),
                    "mode": plan.get("mode"),
                    "execution_requested": bool(args.run),
                    "promotion_allowed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
