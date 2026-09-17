"""Deterministic, offline Gate 2/E0 preflight for synthesis requests.

The preflight runner is deliberately a planning boundary.  It reads a draft
request manifest and the checked-in method/constraint contracts, then writes
one pending candidate slot and one uncharged budget summary for each
request/method lane.  G0--G3 receive structural candidate plans.  M is
explicitly skipped with ``api_not_configured`` because this runner never calls
a model or another service.

No simulator, controller, LLM, external API, or final-audit artifact is
loaded.  A preflight row is not an :class:`AttemptRecord`: it has a
deterministic ledger-compatible identity, but its ledger outcome is ``None``
and its three validation dimensions remain ``unknown`` until a real
candidate attempt is recorded by the attempt ledger.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Final, TypeAlias

from .attempt_ledger import AttemptLedger, RequestSnapshot, attempt_id_for
from .constraint_registry import load_registry
from .method_config import (
    METHOD_IDS,
    MethodConfig,
    MethodConfigError,
    load_method_suite,
)
from .request_manifest import (
    DEVELOPMENT_SPLIT,
    ManifestError,
    coerce_request,
    pair_id_for,
    request_id_for,
    validate_pair,
)


JSONValue: TypeAlias = Any

PREFLIGHT_SCHEMA_VERSION: Final[str] = "synthesis_v2.e0_preflight.v1"
CANDIDATE_SCHEMA_VERSION: Final[str] = "synthesis_v2.e0_candidate.v1"
LEDGER_SCHEMA_VERSION: Final[str] = "synthesis_v2.attempt.v1"
RUN_METHOD_IDS: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3")
SKIPPED_METHOD_ID: Final[str] = "M"
UNKNOWN: Final[str] = "unknown"
PENDING: Final[str] = "pending"
SKIPPED: Final[str] = "skipped"
_PENDING_STATUSES: Final[frozenset[str]] = frozenset(
    {"pending", "pending_calibration", "incomplete"}
)


class PreflightError(ValueError):
    """Raised when a manifest or offline method contract cannot be preflighted."""


def _canonical_json(value: JSONValue, field_name: str = "value") -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{field_name} must be JSON-compatible and finite") from exc


def _json_copy(value: JSONValue, field_name: str = "value") -> JSONValue:
    copied = deepcopy(value)
    _canonical_json(copied, field_name)
    return copied


def canonical_preflight_hash(value: JSONValue) -> str:
    """Hash one preflight value with the same canonical JSON policy everywhere."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def preflight_attempt_id_for(method_id: str, request_id: str, attempt_index: int = 1) -> str:
    """Return a deterministic ID unique across methods and request lanes."""

    if not isinstance(method_id, str) or not method_id.strip():
        raise PreflightError("method_id must be a non-empty string")
    if not isinstance(request_id, str) or not request_id.strip():
        raise PreflightError("request_id must be a non-empty string")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        raise PreflightError("attempt_index must be a positive integer")
    payload = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "method_id": method_id.strip(),
        "request_id": request_id.strip(),
        "attempt_index": attempt_index,
    }
    return "preflight_" + canonical_preflight_hash(payload)[:24]


def attempt_seed_for(method_id: str, request_id: str, attempt_index: int = 1) -> str:
    """Return the deterministic seed material without sampling any values."""

    payload = {
        "method_id": method_id,
        "request_id": request_id,
        "attempt_index": attempt_index,
    }
    return canonical_preflight_hash(payload)


def _assert_no_audit_artifact(value: JSONValue, path: str = "manifest") -> None:
    """Reject attempts to feed a final-audit artifact into preflight."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if (
                "sealed" in key_text
                or "final_audit" in key_text
                or key_text in {"finalaudit", "audit_answer"}
            ):
                raise PreflightError(f"{path}.{key} cannot be loaded by public preflight")
            _assert_no_audit_artifact(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_no_audit_artifact(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("sealed://") or "sealed_final_audit" in lowered or "final_audit" in lowered:
            raise PreflightError(f"{path} contains a final-audit reference")


def _read_manifest_jsonl(path: str | Path) -> list[dict[str, JSONValue]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise PreflightError(f"draft manifest does not exist: {source}")
    if not source.is_file():
        raise PreflightError(f"draft manifest is not a file: {source}")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PreflightError(f"could not read draft manifest: {source}") from exc

    records: list[dict[str, JSONValue]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PreflightError(f"manifest line {line_number} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise PreflightError(f"manifest line {line_number} must be a JSON object")
        record = _json_copy(dict(value), f"manifest line {line_number}")
        _assert_no_audit_artifact(record, f"manifest line {line_number}")
        if record.get("manifest_status") != "draft":
            raise PreflightError(
                f"manifest line {line_number} must have manifest_status='draft'"
            )
        if record.get("request_status") != "planned":
            raise PreflightError(
                f"manifest line {line_number} must have request_status='planned'"
            )
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PreflightError(f"manifest line {line_number} is missing request_id")
        if request_id in seen:
            raise PreflightError(f"manifest contains duplicate request_id {request_id!r}")
        try:
            draft = coerce_request(record)
            computed_id = request_id_for(draft)
        except ManifestError as exc:
            raise PreflightError(f"manifest line {line_number} is not a valid request: {exc}") from exc
        if request_id != computed_id:
            raise PreflightError(
                f"manifest request_id {request_id!r} does not match deterministic ID {computed_id!r}"
            )
        if draft.task not in ("lift", "stack"):
            raise PreflightError(
                f"request {request_id!r} uses unsupported Gate 2 task {draft.task!r}"
            )
        if draft.split != DEVELOPMENT_SPLIT:
            raise PreflightError(
                f"request {request_id!r} must remain in split={DEVELOPMENT_SPLIT!r}"
            )
        if "requested_constraints_hash" in record:
            expected_constraints_hash = canonical_preflight_hash(draft.requested_constraints)
            if record["requested_constraints_hash"] != expected_constraints_hash:
                raise PreflightError(
                    f"request {request_id!r} requested_constraints_hash does not match original constraints"
                )
        seen.add(request_id)
        records.append(record)

    if not records:
        raise PreflightError("draft manifest contains no request records")
    records.sort(key=lambda row: str(row["request_id"]))
    _validate_manifest_pairs(records)
    return records


def _validate_manifest_pairs(records: Sequence[Mapping[str, JSONValue]]) -> None:
    """Recheck pair IDs and the existing same-split request contract."""

    drafts: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, JSONValue]]] = defaultdict(list)
    for record in records:
        request_id = str(record["request_id"])
        try:
            drafts[request_id] = coerce_request(record)
        except ManifestError as exc:
            raise PreflightError(f"request {request_id!r} cannot be coerced: {exc}") from exc
        pair_id = record.get("pair_id")
        if pair_id is None:
            if record.get("pair_type") is not None or record.get("pair_member_ids") not in (None, [], ()):
                raise PreflightError(f"request {request_id!r} has pair metadata without pair_id")
            continue
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise PreflightError(f"request {request_id!r} has an invalid pair_id")
        groups[pair_id].append(record)

    for pair_id, members in groups.items():
        if len(members) != 2:
            raise PreflightError(f"pair {pair_id!r} must have exactly two manifest members")
        first, second = members
        pair_type = first.get("pair_type")
        if pair_type is None or pair_type != second.get("pair_type"):
            raise PreflightError(f"pair {pair_id!r} has inconsistent pair_type")
        raw_ids = first.get("pair_member_ids")
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)) or len(raw_ids) != 2:
            raise PreflightError(f"pair {pair_id!r} must list exactly two pair_member_ids")
        member_ids = tuple(str(item) for item in raw_ids)
        if len(set(member_ids)) != 2 or set(member_ids) != {str(item["request_id"]) for item in members}:
            raise PreflightError(f"pair {pair_id!r} has inconsistent member IDs")
        for member in members:
            if tuple(str(item) for item in member.get("pair_member_ids", ())) != member_ids:
                raise PreflightError(f"pair {pair_id!r} member IDs are not identical on both rows")
        try:
            expected_pair_id = pair_id_for(str(pair_type), member_ids)
            if pair_id != expected_pair_id:
                raise PreflightError(
                    f"pair {pair_id!r} does not match deterministic pair ID {expected_pair_id!r}"
                )
            validate_pair(drafts[member_ids[0]], drafts[member_ids[1]], str(pair_type))
        except ManifestError as exc:
            raise PreflightError(f"pair {pair_id!r} violates request contract: {exc}") from exc


def load_draft_manifest(path: str | Path) -> tuple[dict[str, JSONValue], ...]:
    """Read, validate and deterministically order a draft request JSONL."""

    return tuple(_read_manifest_jsonl(path))


def _request_is_pending(record: Mapping[str, JSONValue]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("incomplete") is True or metadata.get("requests_enabled") is False:
        return True
    for key in ("lifecycle_status", "applicability", "status"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip().lower() in _PENDING_STATUSES:
            return True
    return False


def _unknown_decision() -> dict[str, str]:
    return {"task": UNKNOWN, "manner": UNKNOWN, "safety": UNKNOWN, "overall": UNKNOWN}


def _resource_reservation(dimensions: Sequence[str]) -> dict[str, JSONValue]:
    return {
        "status": "not_measured",
        "charged": False,
        "values": {str(dimension): None for dimension in dimensions},
    }


def _budget_record(
    shared: Mapping[str, JSONValue],
    *,
    planned_candidate_slots: int,
) -> dict[str, JSONValue]:
    attempt_policy = dict(shared["attempt_policy"])
    budget = dict(shared["budget"])
    dimensions = tuple(str(item) for item in budget["resource_dimensions"])
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "max_attempts": int(attempt_policy["max_attempts"]),
        "budget_scope": attempt_policy["budget_scope"],
        "planned_candidate_slots": planned_candidate_slots,
        "actual_attempts_charged": 0,
        "remaining_attempts": int(attempt_policy["max_attempts"]),
        "hidden_retries": bool(attempt_policy["hidden_retries"]),
        "stop_on_first_accepted": bool(attempt_policy["stop_on_first_accepted"]),
        "acceptance_terminal": bool(attempt_policy["acceptance_terminal"]),
        "preserve_original_request": bool(attempt_policy["preserve_original_request"]),
        "resource_dimensions": list(dimensions),
        "resource_reservation": _resource_reservation(dimensions),
        "configured_caps": deepcopy(dict(budget["caps"])),
    }


def _pair_summary(record: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {
        "pair_id": record.get("pair_id"),
        "pair_type": record.get("pair_type"),
        "pair_member_role": record.get("pair_member_role"),
        "pair_member_ids": _json_copy(record.get("pair_member_ids", []), "pair_member_ids"),
    }


def _candidate_plan(
    record: Mapping[str, JSONValue],
    config: MethodConfig,
    snapshot: RequestSnapshot,
    *,
    attempt_index: int,
    status: str,
    reason_code: str,
) -> dict[str, JSONValue]:
    method = config.method
    sampling = method["sampling"]
    method_id = config.method_id
    request_id = snapshot.request_id
    ledger_id = attempt_id_for(request_id, attempt_index)
    method_attempt_id = preflight_attempt_id_for(method_id, request_id, attempt_index)
    seed = attempt_seed_for(method_id, request_id, attempt_index)
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason_code,
        "method_id": method_id,
        "request_id": request_id,
        "attempt_id": ledger_id,
        "method_attempt_id": method_attempt_id,
        "attempt_index": attempt_index,
        "proposal_source": method["proposal_source"],
        "planner_kind": method["planner_kind"],
        "sampling": {
            "kind": sampling["kind"],
            "scope": sampling["scope"],
            "seed_policy": sampling["seed_policy"],
            "seed_sha256": seed,
            "sampled": False,
            "numeric_domain": deepcopy(dict(sampling["numeric_domain"])),
        },
        "stage_scope": _json_copy(record.get("stage_scope", []), "stage_scope"),
        "allowed_control_channels": _json_copy(
            record.get("allowed_control_channels", []), "allowed_control_channels"
        ),
        "requested_constraints": _json_copy(
            record.get("requested_constraints", {}), "requested_constraints"
        ),
        "request_snapshot_hash": snapshot.request_hash,
        "control_plan": None,
        "predicted_plan": None,
        "executed_plan": None,
        "execution_status": "not_run",
        "validation": _unknown_decision(),
        "final_audit_status": UNKNOWN,
    }


def _candidate_attempt_record(
    record: Mapping[str, JSONValue],
    config: MethodConfig,
    snapshot: RequestSnapshot,
    *,
    request_pending: bool,
) -> dict[str, JSONValue]:
    reason_code = "request_pending_calibration" if request_pending else "dry_run_only"
    plan = _candidate_plan(
        record,
        config,
        snapshot,
        attempt_index=1,
        status=PENDING,
        reason_code=reason_code,
    )
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "attempt_id": plan["attempt_id"],
        "method_attempt_id": plan["method_attempt_id"],
        "parent_request_id": snapshot.request_id,
        "attempt_index": 1,
        "status": PENDING,
        "preflight_outcome": UNKNOWN,
        "ledger_outcome": None,
        "ledger_recorded": False,
        "proposal_source": config.method["proposal_source"],
        "candidate_plan": plan,
        "call_record": {
            "mode": "local_preflight",
            "called": False,
            "service": None,
            "api_not_configured": False,
        },
        "resource_reservation": _resource_reservation(
            config.shared["budget"]["resource_dimensions"]
        ),
        "reason_code": reason_code,
        "request_snapshot": deepcopy(dict(snapshot.request)),
        "request_snapshot_hash": snapshot.request_hash,
        "validation": _unknown_decision(),
        "final_audit_status": UNKNOWN,
    }


def _skipped_attempt_record(
    record: Mapping[str, JSONValue],
    config: MethodConfig,
    snapshot: RequestSnapshot,
) -> dict[str, JSONValue]:
    request_id = snapshot.request_id
    ledger_id = attempt_id_for(request_id, 1)
    method_attempt_id = preflight_attempt_id_for(config.method_id, request_id, 1)
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "attempt_id": ledger_id,
        "method_attempt_id": method_attempt_id,
        "parent_request_id": request_id,
        "attempt_index": 1,
        "status": SKIPPED,
        "preflight_outcome": UNKNOWN,
        "ledger_outcome": None,
        "ledger_recorded": False,
        "proposal_source": config.method["proposal_source"],
        "candidate_plan": None,
        "call_record": {
            "mode": "local_preflight",
            "called": False,
            "service": None,
            "api_not_configured": True,
        },
        "resource_reservation": _resource_reservation(
            config.shared["budget"]["resource_dimensions"]
        ),
        "reason_code": "api_not_configured",
        "request_snapshot": deepcopy(dict(snapshot.request)),
        "request_snapshot_hash": snapshot.request_hash,
        "validation": _unknown_decision(),
        "final_audit_status": UNKNOWN,
    }


def _method_request_record(
    record: Mapping[str, JSONValue],
    config: MethodConfig,
    snapshot: RequestSnapshot,
) -> dict[str, JSONValue]:
    request_pending = _request_is_pending(record)
    if config.method_id == SKIPPED_METHOD_ID:
        method_status = SKIPPED
        preflight_status = SKIPPED
        reason_code = "api_not_configured"
        planned_slots = 0
        attempts = [_skipped_attempt_record(record, config, snapshot)]
    else:
        method_status = PENDING
        preflight_status = PENDING
        reason_code = "request_pending_calibration" if request_pending else "dry_run_only"
        planned_slots = 1
        attempts = [_candidate_attempt_record(record, config, snapshot, request_pending=request_pending)]

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "request_id": snapshot.request_id,
        "method_id": config.method_id,
        "method_status": method_status,
        "preflight_status": preflight_status,
        "reason_code": reason_code,
        "request_lifecycle": "pending_calibration" if request_pending else "applicable",
        "request_snapshot": deepcopy(dict(snapshot.request)),
        "request_snapshot_hash": snapshot.request_hash,
        "requested_constraints": _json_copy(
            record.get("requested_constraints", {}), "requested_constraints"
        ),
        "requested_constraints_hash": canonical_preflight_hash(
            record.get("requested_constraints", {})
        ),
        "pair": _pair_summary(record),
        "candidate_attempts": attempts,
        "budget": _budget_record(config.shared, planned_candidate_slots=planned_slots),
        "validation": _unknown_decision(),
        "final_audit": {"status": UNKNOWN, "checked": False},
    }


def _registry_summary(document: Mapping[str, JSONValue], report: Any) -> dict[str, JSONValue]:
    calibration = document.get("calibration")
    calibration_status = calibration.get("status") if isinstance(calibration, Mapping) else None
    decision = getattr(getattr(report, "decision", None), "value", None)
    return {
        "schema_version": document.get("schema_version"),
        "registry_id": document.get("registry_id"),
        "registry_version": document.get("registry_version"),
        "status": document.get("status"),
        "calibration_status": calibration_status,
        "validation_valid": bool(report.valid),
        "promotion_allowed": bool(report.promotion_allowed),
        "decision": decision or UNKNOWN,
        "source_loaded": True,
        "public_view_only": True,
    }


def build_preflight_report(
    records: Sequence[Mapping[str, JSONValue]],
    *,
    method_configs: Sequence[MethodConfig],
    registry_document: Mapping[str, JSONValue],
    registry_report: Any,
    source_path: str | Path | None = None,
) -> dict[str, JSONValue]:
    """Build a deterministic report from already loaded local contracts."""

    if not records:
        raise PreflightError("preflight requires at least one manifest record")
    copied_records = [_json_copy(dict(record), "manifest record") for record in records]
    copied_records.sort(key=lambda row: str(row.get("request_id", "")))
    _validate_manifest_pairs(copied_records)

    if any(not isinstance(config, MethodConfig) for config in method_configs):
        raise PreflightError("method_configs must contain MethodConfig objects")
    by_id = {config.method_id: config for config in method_configs}
    if len(method_configs) != len(METHOD_IDS) or set(by_id) != set(METHOD_IDS):
        raise PreflightError(f"method suite must contain exactly {METHOD_IDS}")
    shared = method_configs[0].shared
    max_attempts = shared["attempt_policy"]["max_attempts"]
    # Register snapshots through the existing ledger contract.  No attempt is
    # recorded here because pending/unknown is not one of its six outcomes.
    try:
        ledger = AttemptLedger(copied_records, max_attempts=max_attempts)
        snapshots = {
            str(record["request_id"]): RequestSnapshot.from_value(record)
            for record in copied_records
        }
    except Exception as exc:
        raise PreflightError(f"manifest cannot be registered in attempt ledger: {exc}") from exc

    method_records: list[dict[str, JSONValue]] = []
    for request_id in sorted(snapshots):
        record = next(item for item in copied_records if item["request_id"] == request_id)
        snapshot = snapshots[request_id]
        for method_id in METHOD_IDS:
            method_records.append(_method_request_record(record, by_id[method_id], snapshot))

    manifest_content = [dict(record) for record in copied_records]
    manifest_hash = canonical_preflight_hash(manifest_content)
    config_hashes = {
        config.method_id: canonical_preflight_hash(config.to_dict()) for config in method_configs
    }
    report: dict[str, JSONValue] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_mode": "local_deterministic_preflight",
        "status": "dry_run",
        "manifest": {
            "source_path": str(Path(source_path).expanduser().resolve()) if source_path is not None else None,
            "manifest_status": "draft",
            "manifest_hash": manifest_hash,
            "request_count": len(copied_records),
            "request_ids": sorted(snapshots),
            "tasks": sorted({str(record["task"]) for record in copied_records}),
            "splits": sorted({str(record["split"]) for record in copied_records}),
        },
        "method_suite": {
            "configured": list(METHOD_IDS),
            "preflight_methods": list(RUN_METHOD_IDS),
            "skipped": [
                {
                    "method_id": SKIPPED_METHOD_ID,
                    "status": SKIPPED,
                    "reason_code": "api_not_configured",
                }
            ],
            "config_schema_version": method_configs[0].schema_version,
            "config_hashes": config_hashes,
        },
        "shared_contract": {
            "executor": deepcopy(dict(shared["executor"])),
            "request_contract": shared["request_contract"],
            "task_space_contract": shared["task_space_contract"],
            "task_scope": deepcopy(list(shared["task_scope"])),
            "constraint_source": deepcopy(dict(shared["constraint_source"])),
            "attempt_policy": deepcopy(dict(shared["attempt_policy"])),
            "budget": deepcopy(dict(shared["budget"])),
            "validation": deepcopy(dict(shared["validation"])),
            "reproducibility": deepcopy(dict(shared["reproducibility"])),
        },
        "constraint_registry": _registry_summary(registry_document, registry_report),
        "execution_guards": {
            "candidate_generation_called": False,
            "simulator_called": False,
            "controller_called": False,
            "model_api_called": False,
            "external_api_called": False,
            "final_audit_loaded": False,
            "final_audit_feedback_used": False,
        },
        "attempt_ledger": {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "registered_request_count": len(copied_records),
            "actual_attempt_record_count": len(ledger.records()),
            "actual_attempt_records": [],
            "planned_method_request_lanes": len(method_records),
            "charged_attempts": 0,
        },
        "records": method_records,
        "unexecuted_result_policy": {
            "candidate_status": PENDING,
            "validation": _unknown_decision(),
            "unknown_is_qualified": False,
            "accepted_count": 0,
            "final_audit_status": UNKNOWN,
        },
    }
    report["report_hash"] = canonical_preflight_hash(report)
    return report


def run_preflight(
    manifest_path: str | Path,
    *,
    method_config_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, JSONValue]:
    """Load local contracts and build the offline Gate 2/E0 report."""

    root = Path(__file__).resolve().parents[1]
    method_dir = Path(method_config_dir) if method_config_dir is not None else root / "configs" / "generation_methods"
    registry_file = Path(registry_path) if registry_path is not None else root / "configs" / "constraint_registry.yaml"
    records = load_draft_manifest(manifest_path)
    try:
        registry = load_registry(registry_file)
    except Exception as exc:
        raise PreflightError(f"could not load constraint registry: {registry_file}") from exc
    if not registry.report.valid:
        raise PreflightError("constraint registry is structurally invalid")
    try:
        configs = load_method_suite(method_dir, registry_document=registry.document)
    except (MethodConfigError, OSError) as exc:
        raise PreflightError(f"could not load method suite: {method_dir}") from exc
    return build_preflight_report(
        records,
        method_configs=configs,
        registry_document=registry.document,
        registry_report=registry.report,
        source_path=manifest_path,
    )


def write_preflight_report(path: str | Path, report: Mapping[str, JSONValue]) -> None:
    """Write one canonical UTF-8 JSON report without invoking any service."""

    if not isinstance(report, Mapping):
        raise PreflightError("report must be a mapping")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = _canonical_json(dict(report), "preflight report")
        output.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise PreflightError(f"could not write preflight report: {output}") from exc


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "RUN_METHOD_IDS",
    "PreflightError",
    "attempt_seed_for",
    "build_preflight_report",
    "canonical_preflight_hash",
    "load_draft_manifest",
    "preflight_attempt_id_for",
    "run_preflight",
    "write_preflight_report",
]
