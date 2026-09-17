"""Pure-Python planning and audit helpers for the Wipe contact-response grid.

``surface_penetration_m`` is a geometric/controller target offset. It is not a
known applied wrench, an independent force calibration, or a safety threshold.
The helpers in this module only plan and summarize observations from the real
``gate1_wipe_probe.py`` output; they never run a simulator or apply a load.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLAN_SCHEMA = "mannercap-gate1-wipe-contact-grid-plan-v1"
AUDIT_SCHEMA = "mannercap-gate1-wipe-contact-grid-audit-v1"
PROBE_SCHEMA = "mannercap-gate1-wipe-probe-v1"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SURFACE_PENETRATIONS_M = (0.0, 0.002, 0.01)
DEFAULT_MAX_STEPS = 4000
DEFAULT_WIPE_ATTEMPTS = 3
# This is inherited from the probe's marker-recognition geometry. It is not a
# physical, task, or safety limit.
PROBE_PENETRATION_UPPER_BOUND_M = 0.02

PARAMETER_SEMANTICS = (
    "geometric/controller target offset used by gate1_wipe_probe.py; not a known "
    "applied force or torque and not an approved safety threshold"
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "matrix_sha256"}


def _float_label(value: float) -> str:
    text = format(value, ".9f").rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if not values:
        raise ValueError("at least one seed is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("seeds must be integers")
    if any(value < 0 or value >= 2**32 for value in values):
        raise ValueError("seeds must fit NumPy's [0, 2**32) seed range")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique in a factorial grid")
    return values


def _validate_penetrations(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("at least one surface penetration is required")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("surface penetrations must be finite")
    if any(value < 0.0 or value >= PROBE_PENETRATION_UPPER_BOUND_M for value in result):
        raise ValueError(
            "surface penetrations must be in [0, 0.02) to match the probe's "
            "marker-recognition geometry; this is not a safety limit"
        )
    if len(set(result)) != len(result):
        raise ValueError("surface penetrations must be unique in a factorial grid")
    return result


def _validate_budget(max_steps: int, wipe_attempts: int) -> dict[str, Any]:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if (
        isinstance(wipe_attempts, bool)
        or not isinstance(wipe_attempts, int)
        or wipe_attempts < 0
    ):
        raise ValueError("wipe_attempts must be a non-negative integer")
    return {
        "environment_max_steps": max_steps,
        "wipe_attempts": wipe_attempts,
        "wipe_attempt_budget_scope": "calls to wipe_spill_and_verify; approach_spill_surface is additional",
    }


def build_grid_plan(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    surface_penetrations_m: Iterable[float] = DEFAULT_SURFACE_PENETRATIONS_M,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    wipe_attempts: int = DEFAULT_WIPE_ATTEMPTS,
    probe_script: str = "scripts/gate1_wipe_probe.py",
) -> dict[str, Any]:
    """Build a full-factorial plan without importing or running the simulator."""

    seed_values = _validate_seeds(seeds)
    penetration_values = _validate_penetrations(surface_penetrations_m)
    budget = _validate_budget(max_steps, wipe_attempts)
    cases: list[dict[str, Any]] = []
    index = 0
    for seed in seed_values:
        for penetration in penetration_values:
            case_id = (
                f"case-{index:03d}-seed-{seed}-penetration-"
                f"{_float_label(penetration)}m"
            )
            cases.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "surface_penetration_m": penetration,
                    "surface_penetration_semantics": PARAMETER_SEMANTICS,
                    "budget": budget,
                    "case_directory": f"cases/{case_id}",
                    "run_directory": f"cases/{case_id}/probe_output",
                    "probe_arguments": [
                        "--seed",
                        str(seed),
                        "--execute-wipe",
                        "--wipe-attempts",
                        str(wipe_attempts),
                        "--surface-penetration-m",
                        repr(penetration),
                        "--max-steps",
                        str(max_steps),
                    ],
                }
            )
            index += 1

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "mode": "full_factorial",
        "task": "wipe",
        "purpose": "development_contact_response_observation",
        "qualification": "candidate_development_observation_only",
        "physical_qualification": "unknown",
        "contact_model_calibration_verified": False,
        "thresholds_frozen": False,
        "promotion_allowed": False,
        "parameter": {
            "name": "surface_penetration_m",
            "unit": "m",
            "semantics": PARAMETER_SEMANTICS,
            "known_applied_load": False,
            "approved_safety_threshold": False,
            "accepted_range_source": "existing probe marker-recognition geometry [0, 0.02); not a safety limit",
        },
        "seeds": list(seed_values),
        "surface_penetrations_m": list(penetration_values),
        "budget": budget,
        "probe": {
            "script": probe_script,
            "schema": PROBE_SCHEMA,
            "execution_default": "plan_and_audit_only",
        },
        "cases": cases,
        "interpretation_limits": [
            "surface_penetration_m is a geometric/controller target offset, not a known force/torque load",
            "contact-pair forces are observed MuJoCo contact-local values, not calibrated safety measurements",
            "recent EEF wrench remains qualified as an uncalibrated RoboSuite sensor channel",
            "no contact or safety threshold is frozen by this plan or audit",
        ],
    }
    plan["matrix_sha256"] = _canonical_sha256(_plan_payload(plan))
    return plan


def build_existing_runs_plan(run_directories: Sequence[Path | str]) -> dict[str, Any]:
    """Build an audit-only manifest for existing probe directories.

    The source directories are read-only inputs. Each can be an earlier single
    probe directory, including results created before the grid runner existed.
    """

    if not run_directories:
        raise ValueError("at least one existing probe directory is required")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    observed_seeds: list[int] = []
    observed_penetrations: list[float] = []
    for index, raw_directory in enumerate(run_directories):
        directory = Path(raw_directory).expanduser().resolve()
        key = str(directory)
        if key in seen:
            raise ValueError(f"duplicate existing run directory: {directory}")
        seen.add(key)
        metadata: dict[str, Any] = {}
        probe_path = directory / "probe.json"
        if probe_path.is_file():
            try:
                loaded = json.loads(probe_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                # Keep the input in the manifest so the audit records its error.
                metadata = {}
        seed = metadata.get("seed")
        penetration = metadata.get("surface_penetration_m")
        if isinstance(seed, int) and not isinstance(seed, bool):
            observed_seeds.append(seed)
        if isinstance(penetration, (int, float)) and math.isfinite(float(penetration)):
            observed_penetrations.append(float(penetration))
        cases.append(
            {
                "case_id": f"existing-{index:03d}-{directory.name or 'run'}",
                "seed": seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
                "surface_penetration_m": (
                    float(penetration)
                    if isinstance(penetration, (int, float))
                    and math.isfinite(float(penetration))
                    else None
                ),
                "surface_penetration_semantics": PARAMETER_SEMANTICS,
                "case_directory": key,
                "run_directory": key,
                "source": "existing_probe_run",
                "probe_status_at_manifest_creation": metadata.get("status"),
            }
        )

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "mode": "existing_runs_audit",
        "task": "wipe",
        "purpose": "development_contact_response_observation",
        "qualification": "candidate_development_observation_only",
        "physical_qualification": "unknown",
        "contact_model_calibration_verified": False,
        "thresholds_frozen": False,
        "promotion_allowed": False,
        "parameter": {
            "name": "surface_penetration_m",
            "unit": "m",
            "semantics": PARAMETER_SEMANTICS,
            "known_applied_load": False,
            "approved_safety_threshold": False,
            "accepted_range_source": "values are read from existing probes; no threshold is inferred",
        },
        "seeds": sorted(set(observed_seeds)),
        "surface_penetrations_m": sorted(set(observed_penetrations)),
        "budget": {"source": "read from each probe result when available"},
        "probe": {"schema": PROBE_SCHEMA, "execution_default": "audit_only"},
        "cases": cases,
        "interpretation_limits": [
            "existing runs are summarized as observed; absent factorial cells are not inferred",
            "surface_penetration_m is not a known applied force/torque load",
            "contact-pair forces are not calibrated safety measurements",
            "no contact or safety threshold is frozen by this audit",
        ],
    }
    plan["matrix_sha256"] = _canonical_sha256(_plan_payload(plan))
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unexpected contact-grid plan schema: {plan.get('schema')!r}")
    expected = _canonical_sha256(_plan_payload(plan))
    if plan.get("matrix_sha256") != expected:
        raise ValueError("contact-grid plan SHA-256 does not match its contents")
    if plan.get("mode") not in {"full_factorial", "existing_runs_audit"}:
        raise ValueError(f"unsupported contact-grid plan mode: {plan.get('mode')!r}")
    if not isinstance(plan.get("cases"), list):
        raise ValueError("contact-grid plan cases must be a list")


def _resolve_case_directory(root: Path, case: Mapping[str, Any]) -> Path:
    raw = case.get("run_directory")
    if not isinstance(raw, str) or not raw:
        raise ValueError("case run_directory must be a non-empty string")
    candidate = Path(raw)
    if candidate.is_absolute():
        if case.get("source") != "existing_probe_run":
            raise ValueError("absolute run_directory is allowed only for imported existing runs")
        return candidate
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"case run_directory escapes grid root: {raw!r}") from error
    return resolved


def _resolve_process_directory(root: Path, case: Mapping[str, Any]) -> Path | None:
    if case.get("source") == "existing_probe_run":
        return None
    raw = case.get("case_directory")
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("absolute case_directory is not allowed for planned grid cases")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"case_directory escapes grid root: {raw!r}") from error
    return resolved


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{path.name}: {type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, f"{path.name}: expected a JSON object"
    return value, None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    converted = [_float_value(item) for item in value]
    if any(item is None for item in converted):
        return None
    return [float(item) for item in converted]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _observe_transitions(
    path: Path,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    arm_stats: dict[str, dict[str, Any]] = {}
    parsed_lines = 0
    contacts_seen = 0
    eef_samples_seen = 0
    transitions_missing_contacts = 0
    transitions_missing_eef = 0
    maximum_contact_force: float | None = None
    maximum_eef_force: float | None = None
    maximum_eef_torque: float | None = None

    if not path.is_file():
        return (
            {
                "source": "transitions.jsonl",
                "file_present": False,
                "parsed_transition_count": 0,
                "contacts_seen": 0,
                "eef_samples_seen": 0,
                "maximum_contact_pair_force_n": None,
                "contact_pairs": [],
                "recent_eef": {
                    "qualification": "uncalibrated_robosuite_sensor",
                    "maximum_force_n": None,
                    "maximum_torque_nm": None,
                    "arms": [],
                },
                "transitions_missing_contacts": 0,
                "transitions_missing_eef": 0,
            },
            ["transitions.jsonl is missing"],
        )

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                transition = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(f"transitions.jsonl line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(transition, dict):
                issues.append(f"transitions.jsonl line {line_number}: expected a JSON object")
                continue
            parsed_lines += 1

            contacts = transition.get("contacts")
            if not isinstance(contacts, list):
                transitions_missing_contacts += 1
                issues.append(f"transition line {line_number}: contacts list is missing")
                contacts = []
            for contact_index, contact in enumerate(contacts):
                if not isinstance(contact, dict):
                    issues.append(
                        f"transition line {line_number} contact {contact_index}: expected an object"
                    )
                    continue
                geom1 = str(contact.get("geom1") or "")
                geom2 = str(contact.get("geom2") or "")
                pair = tuple(sorted((geom1, geom2)))
                force_vector = _vector(contact.get("force_contact_local_n"), 3)
                force_norm = _float_value(contact.get("force_norm_n"))
                if force_norm is None and force_vector is not None:
                    force_norm = _norm(force_vector)
                if force_norm is None or force_norm < 0.0:
                    issues.append(
                        f"transition line {line_number} contact {contact_index}: "
                        "force_norm_n or force_contact_local_n is invalid"
                    )
                    continue
                contacts_seen += 1
                maximum_contact_force = (
                    force_norm
                    if maximum_contact_force is None
                    else max(maximum_contact_force, force_norm)
                )
                stats = pair_stats.setdefault(
                    pair,
                    {
                        "geom_pair": list(pair),
                        "observations": 0,
                        "maximum_force_n": None,
                        "force_contact_local_n_at_maximum": None,
                        "distance_m_at_maximum": None,
                    },
                )
                stats["observations"] += 1
                if stats["maximum_force_n"] is None or force_norm > stats["maximum_force_n"]:
                    stats["maximum_force_n"] = force_norm
                    stats["force_contact_local_n_at_maximum"] = force_vector
                    stats["distance_m_at_maximum"] = _float_value(contact.get("distance_m"))

            eef_wrenches = transition.get("eef_wrenches")
            if not isinstance(eef_wrenches, list):
                transitions_missing_eef += 1
                issues.append(f"transition line {line_number}: eef_wrenches list is missing")
                eef_wrenches = []
            for wrench_index, wrench in enumerate(eef_wrenches):
                if not isinstance(wrench, dict):
                    issues.append(
                        f"transition line {line_number} EEF wrench {wrench_index}: expected an object"
                    )
                    continue
                arm = str(wrench.get("arm") or "unknown")
                force_vector = _vector(wrench.get("force_n"), 3)
                torque_vector = _vector(wrench.get("torque_nm"), 3)
                force_norm = _float_value(wrench.get("force_norm_n"))
                torque_norm = _float_value(wrench.get("torque_norm_nm"))
                if force_norm is None and force_vector is not None:
                    force_norm = _norm(force_vector)
                if torque_norm is None and torque_vector is not None:
                    torque_norm = _norm(torque_vector)
                if force_norm is None or torque_norm is None or force_norm < 0.0 or torque_norm < 0.0:
                    issues.append(
                        f"transition line {line_number} EEF wrench {wrench_index}: "
                        "force/torque values are invalid"
                    )
                    continue
                eef_samples_seen += 1
                maximum_eef_force = (
                    force_norm if maximum_eef_force is None else max(maximum_eef_force, force_norm)
                )
                maximum_eef_torque = (
                    torque_norm
                    if maximum_eef_torque is None
                    else max(maximum_eef_torque, torque_norm)
                )
                stats = arm_stats.setdefault(
                    arm,
                    {
                        "arm": arm,
                        "observations": 0,
                        "maximum_force_n": None,
                        "force_n_at_maximum": None,
                        "maximum_torque_nm": None,
                        "torque_nm_at_maximum": None,
                        "qualifications": set(),
                    },
                )
                stats["observations"] += 1
                qualification = wrench.get("qualification")
                if isinstance(qualification, str) and qualification:
                    stats["qualifications"].add(qualification)
                if stats["maximum_force_n"] is None or force_norm > stats["maximum_force_n"]:
                    stats["maximum_force_n"] = force_norm
                    stats["force_n_at_maximum"] = force_vector
                if stats["maximum_torque_nm"] is None or torque_norm > stats["maximum_torque_nm"]:
                    stats["maximum_torque_nm"] = torque_norm
                    stats["torque_nm_at_maximum"] = torque_vector

    pairs = sorted(pair_stats.values(), key=lambda item: tuple(item["geom_pair"]))
    arms: list[dict[str, Any]] = []
    for arm in sorted(arm_stats):
        stats = dict(arm_stats[arm])
        stats["qualifications"] = sorted(stats["qualifications"])
        arms.append(stats)
    return (
        {
            "source": "transitions.jsonl",
            "file_present": True,
            "parsed_transition_count": parsed_lines,
            "contacts_seen": contacts_seen,
            "eef_samples_seen": eef_samples_seen,
            "maximum_contact_pair_force_n": maximum_contact_force,
            "contact_force_frame": "contact_local",
            "contact_pairs": pairs,
            "recent_eef": {
                "qualification": "uncalibrated_robosuite_sensor",
                "maximum_force_n": maximum_eef_force,
                "maximum_torque_nm": maximum_eef_torque,
                "arms": arms,
            },
            "transitions_missing_contacts": transitions_missing_contacts,
            "transitions_missing_eef": transitions_missing_eef,
        },
        issues,
    )


def _task_outcome(task_completed: Any) -> str:
    if task_completed is True:
        return "success"
    if task_completed is False:
        return "failure"
    return "unknown"


def audit_case(root: Path | str, case: Mapping[str, Any]) -> dict[str, Any]:
    """Read one planned or imported probe run and preserve all observed failures."""

    root_path = Path(root)
    result: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "seed": case.get("seed"),
        "surface_penetration_m": case.get("surface_penetration_m"),
        "surface_penetration_semantics": case.get(
            "surface_penetration_semantics", PARAMETER_SEMANTICS
        ),
        "requested_budget": case.get("budget"),
        "run_directory": case.get("run_directory"),
        "run_state": "planned",
        "probe_status": None,
        "task_outcome": "unknown",
        "task_completed": None,
        "initial_marker_status": None,
        "final_marker_status": None,
        "final_reward": None,
        "model_sha256": None,
        "observed_budget_use": {
            "control_transitions": None,
            "attempt_result_count": None,
            "wipe_spill_and_verify_calls": None,
            "wall_s": None,
        },
        "contact_response": {
            "source": "transitions.jsonl",
            "interpretation": "observed contact-local force; not a calibrated safety measurement",
            "maximum_contact_pair_force_n": None,
            "pairs": [],
        },
        "recent_eef_response": {
            "qualification": "uncalibrated_robosuite_sensor",
            "maximum_force_n": None,
            "maximum_torque_nm": None,
            "arms": [],
        },
        "attempt_results": [],
        "failure_events": [],
        "execution_errors": [],
        "audit_issues": [],
    }

    try:
        run_directory = _resolve_case_directory(root_path, case)
    except ValueError as error:
        result["run_state"] = "execution_error"
        result["execution_errors"].append(str(error))
        return result

    try:
        process_directory = _resolve_process_directory(root_path, case)
    except ValueError as error:
        result["run_state"] = "execution_error"
        result["execution_errors"].append(str(error))
        return result
    process_path = process_directory / "process.json" if process_directory is not None else None
    process, process_error = (
        _read_json(process_path) if process_path is not None else (None, None)
    )
    if process_error:
        result["audit_issues"].append(process_error)
    if process is not None:
        result["process"] = process
        process_state = process.get("status")
        if process_state == "execution_error":
            result["execution_errors"].append(
                str(process.get("error") or "grid runner reported an execution error")
            )
        if process.get("exit_code") not in (None, 0):
            result["execution_errors"].append(
                f"probe process exited with code {process.get('exit_code')}"
            )

    probe, probe_error = _read_json(run_directory / "probe.json")
    if probe_error:
        result["audit_issues"].append(probe_error)
        result["execution_errors"].append(probe_error)
    if probe is None:
        if process is not None and process.get("status") == "running":
            result["run_state"] = "running"
        elif process is not None or probe_error:
            result["run_state"] = "execution_error"
            result["execution_errors"].append(
                "probe.json is missing after a recorded process"
            )
        else:
            result["run_state"] = "planned"
        return result

    if probe.get("schema") != PROBE_SCHEMA:
        result["audit_issues"].append(
            f"unexpected probe schema: {probe.get('schema')!r}"
        )
    probe_status = probe.get("status")
    result["probe_status"] = probe_status
    if probe_status == "executed":
        result["run_state"] = "executed"
    else:
        result["run_state"] = "execution_error"
        result["execution_errors"].append(
            str(probe.get("error") or f"probe status is {probe_status!r}")
        )

    observed_seed = probe.get("seed")
    observed_penetration = _float_value(probe.get("surface_penetration_m"))
    expected_seed = case.get("seed")
    expected_penetration = _float_value(case.get("surface_penetration_m"))
    if expected_seed is not None and observed_seed != expected_seed:
        result["audit_issues"].append(
            f"probe seed mismatch: expected {expected_seed!r}, observed {observed_seed!r}"
        )
    if expected_penetration is not None and (
        observed_penetration is None
        or not math.isclose(expected_penetration, observed_penetration, rel_tol=0.0, abs_tol=1e-12)
    ):
        result["audit_issues"].append(
            "probe surface_penetration_m mismatch: "
            f"expected {expected_penetration!r}, observed {observed_penetration!r}"
        )

    initial, initial_error = _read_json(run_directory / "initial_state.json")
    if initial_error:
        result["audit_issues"].append(initial_error)
    if initial is not None:
        result["initial_marker_status"] = initial.get("marker_status")
        result["model_sha256"] = initial.get("model_sha256")
    if result["model_sha256"] is None:
        result["model_sha256"] = probe.get("model_sha256")

    wipe_attempt = probe.get("wipe_attempt")
    if not isinstance(wipe_attempt, dict):
        wipe_attempt = {}
    final_marker_status = wipe_attempt.get("final_marker_status")
    if isinstance(final_marker_status, dict):
        result["final_marker_status"] = final_marker_status
        result["task_completed"] = final_marker_status.get("task_completed")
        result["final_reward"] = _float_value(final_marker_status.get("reward"))
    result["task_outcome"] = _task_outcome(result["task_completed"])

    attempt_results = wipe_attempt.get("attempt_results")
    if isinstance(attempt_results, list):
        result["attempt_results"] = attempt_results
        for index, attempt_result in enumerate(attempt_results):
            if not isinstance(attempt_result, dict):
                result["failure_events"].append(
                    {"attempt_index": index, "detail": attempt_result}
                )
                continue
            error = attempt_result.get("error")
            if attempt_result.get("success") is False or error:
                result["failure_events"].append(
                    {
                        "attempt_index": index,
                        "skill": attempt_result.get("skill"),
                        "success": attempt_result.get("success"),
                        "error": error,
                        "result": attempt_result,
                    }
                )
            if error:
                result["execution_errors"].append(str(error))
    elif wipe_attempt:
        result["audit_issues"].append("wipe_attempt.attempt_results is not a list")

    trace_summary = probe.get("trace_summary")
    if not isinstance(trace_summary, dict):
        trace_summary = {}
    transition_path = run_directory / "transitions.jsonl"
    observed_trace, trace_issues = _observe_transitions(transition_path)
    result["audit_issues"].extend(trace_issues)
    result["contact_response"] = {
        "source": "transitions.jsonl",
        "interpretation": "observed contact-local force; not a calibrated safety measurement",
        "maximum_contact_pair_force_n": observed_trace["maximum_contact_pair_force_n"],
        "pairs": observed_trace["contact_pairs"],
        "observations": observed_trace["contacts_seen"],
        "force_frame": observed_trace.get("contact_force_frame"),
    }
    result["recent_eef_response"] = observed_trace["recent_eef"]

    control_transitions = trace_summary.get("control_transitions")
    if isinstance(control_transitions, int) and not isinstance(control_transitions, bool):
        result["observed_budget_use"]["control_transitions"] = control_transitions
        if control_transitions != observed_trace["parsed_transition_count"]:
            result["audit_issues"].append(
                "trace transition count mismatch: "
                f"probe summary={control_transitions}, parsed log={observed_trace['parsed_transition_count']}"
            )
    else:
        result["observed_budget_use"]["control_transitions"] = observed_trace[
            "parsed_transition_count"
        ]
    result["observed_budget_use"]["attempt_result_count"] = len(result["attempt_results"])
    result["observed_budget_use"]["wipe_spill_and_verify_calls"] = sum(
        isinstance(item, dict) and item.get("skill") == "wipe_spill_and_verify"
        for item in result["attempt_results"]
    )
    result["observed_budget_use"]["wall_s"] = _float_value(probe.get("wall_s"))
    result["raw_probe_trace_summary"] = trace_summary
    result["recorded_transition_observations"] = observed_trace
    if result["run_state"] == "executed" and probe_status == "executed":
        # A non-zero wrapper exit code remains an execution error even if the
        # child managed to write a probe result before exiting.
        if process is not None and process.get("exit_code") not in (None, 0):
            result["run_state"] = "execution_error"

    return result


def audit_grid(root: Path | str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Audit all cases while preserving task failures and execution errors."""

    validate_plan(plan)
    root_path = Path(root).resolve()
    cases = [audit_case(root_path, case) for case in plan["cases"]]
    counts = {
        "planned": sum(case["run_state"] == "planned" for case in cases),
        "running": sum(case["run_state"] == "running" for case in cases),
        "executed": sum(case["run_state"] == "executed" for case in cases),
        "execution_error": sum(case["run_state"] == "execution_error" for case in cases),
        "task_success": sum(case["task_outcome"] == "success" for case in cases),
        "task_failure": sum(case["task_outcome"] == "failure" for case in cases),
        "task_unknown": sum(case["task_outcome"] == "unknown" for case in cases),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "mode": plan["mode"],
        "task": "wipe",
        "purpose": "development_contact_response_observation",
        "matrix_sha256": plan["matrix_sha256"],
        "qualification": "candidate_development_observation_only",
        "physical_qualification": "unknown",
        "contact_model_calibration_verified": False,
        "thresholds_frozen": False,
        "promotion_allowed": False,
        "counts": counts,
        "cases": cases,
        "interpretation_limits": list(plan.get("interpretation_limits", [])),
    }


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render a compact observation report; no safety/pass threshold is used."""

    lines = [
        "# Gate 1 Wipe 接触响应网格审计",
        "",
        f"- 模式：`{audit.get('mode')}`",
        f"- 矩阵 SHA-256：`{audit.get('matrix_sha256')}`",
        f"- 运行计数：`{json.dumps(audit.get('counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "- physical qualification：`unknown`；安全阈值未冻结；`promotion_allowed=false`。",
        "",
        "`surface_penetration_m` 是擦拭目标的几何/控制压入参数，不是已知外加载荷。",
        "接触对数值为 MuJoCo contact-local 力观测；recent EEF wrench 仍标记为未校准通道。",
        "下表只报告探测结果，不构成接触力校准或安全判定。",
        "",
        "| case | seed | surface_penetration_m | model SHA-256 | run state | task success | markers | reward | max contact-pair force (N) | max recent EEF force (N) | transitions | wall (s) | errors |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in audit.get("cases", []):
        marker = case.get("final_marker_status") or {}
        markers = None
        if isinstance(marker, dict) and marker.get("wiped_markers") is not None:
            markers = f"{marker.get('wiped_markers')}/{marker.get('total_markers')}"
        contact = case.get("contact_response") or {}
        eef = case.get("recent_eef_response") or {}
        use = case.get("observed_budget_use") or {}
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    case.get("case_id"),
                    case.get("seed"),
                    case.get("surface_penetration_m"),
                    case.get("model_sha256"),
                    case.get("run_state"),
                    case.get("task_outcome"),
                    markers,
                    case.get("final_reward"),
                    contact.get("maximum_contact_pair_force_n"),
                    eef.get("maximum_force_n"),
                    use.get("control_transitions"),
                    use.get("wall_s"),
                    len(case.get("execution_errors", [])),
                )
            )
            + " |"
        )
    lines.extend(["", "完整 attempt 级失败、执行错误和 transition 原始记录保留在各 probe 目录。", ""])
    return "\n".join(lines)
