"""Pure JSONL round-trip preflight for the Gate 1 action boundary.

This module checks the part of an action round-trip that can be established
without importing LeRobot, PyArrow, a policy, or a simulator:

``source JSON/JSONL records -> versioned JSONL bundle -> re-import -> field comparison``

The bundle carries each source record unchanged, together with a hash and an
action-field classification.  The preflight never rewrites timestamps,
recomputes an action, resamples a trajectory, or executes a controller.  Its
controller result is therefore always ``pending`` / ``unknown`` until a
separate execution audit supplies evidence.

The source classifier deliberately keeps these records distinct:

* ``action.raw_joint_gripper`` or an 8-D ``action`` is the LIBERO low-level
  joint-delta-plus-gripper command;
* ``lerobot_action_candidate`` or a 7-D exported ``action`` is the current
  EEF position/axis-angle state-difference candidate and is unverified.

Only the standard library is used.  The CLI accepts JSON and JSONL/NDJSON
records, including a Stage-CAP manifest whose rows point at ``steps_path``.
Binary LeRobot parquet is reported as an unsupported input by this boundary;
it requires a separate dependency-aware adapter and cannot be silently
interpreted as an 8-D controller command.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from .action_interface import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    InterfaceId,
    LEROBOT_EXPORT_ACTION_DIM,
    LIBERO_LOW_LEVEL_INTERFACE,
    LEROBOT_EXPORT_CANDIDATE_INTERFACE,
    STUDENT_ACTION_DIM,
)


ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "synthesis_v2.action_round_trip_preflight.v1"
)
ROUND_TRIP_MANIFEST_SCHEMA_VERSION: Final[str] = ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION
ROUND_TRIP_RECORD_SCHEMA_VERSION: Final[str] = (
    "synthesis_v2.action_round_trip_record.v1"
)
ROUND_TRIP_TARGET_COUNT: Final[int] = 30
CONTROLLER_PENDING_STATUS: Final[str] = "pending"
CONTROLLER_UNKNOWN_DECISION: Final[str] = "unknown"


class RoundTripPreflightError(ValueError):
    """Raised for malformed source/bundle data or an unsupported format."""

    def __init__(self, message: str, *, code: str = "invalid_round_trip", path: str | None = None):
        super().__init__(message)
        self.code = code
        self.path = path


@dataclass(frozen=True)
class FieldMismatch:
    """One field-level difference between source and re-imported records."""

    record_index: int | None
    field_path: str
    code: str
    expected: Any = None
    actual: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "field_path": self.field_path,
            "code": self.code,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class RoundTripConsistencyReport:
    """Result of comparing records after export and re-import."""

    source_count: int
    imported_count: int
    checked_record_count: int
    checked_field_count: int
    mismatches: tuple[FieldMismatch, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.mismatches and self.source_count == self.imported_count

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    def has_code(self, code: str) -> bool:
        return any(item.code == code for item in self.mismatches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "source_count": self.source_count,
            "imported_count": self.imported_count,
            "checked_record_count": self.checked_record_count,
            "checked_field_count": self.checked_field_count,
            "mismatch_count": self.mismatch_count,
            "mismatches": [item.as_dict() for item in self.mismatches],
        }


@dataclass(frozen=True)
class RoundTripManifest:
    """Versioned preflight manifest; controller execution stays pending."""

    source_path: str
    export_path: str
    requested_count: int
    selected_count: int
    available_count: int
    consistency: RoundTripConsistencyReport
    action_semantics: tuple[dict[str, Any], ...]
    source_format: str
    source_manifest_rows: int = 0
    timing_missing_dt_count: int = 0
    timing_missing_timestamp_count: int = 0
    controller_status: str = CONTROLLER_PENDING_STATUS
    controller_decision: str = CONTROLLER_UNKNOWN_DECISION
    controller_reason: str = (
        "preflight does not execute a student policy or controller; external replay evidence is pending"
    )
    manifest_version: str = ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.requested_count <= 0:
            raise RoundTripPreflightError("requested_count must be positive", code="invalid_count")
        if self.selected_count < 0 or self.available_count < 0:
            raise RoundTripPreflightError("record counts cannot be negative", code="invalid_count")
        if self.controller_status != CONTROLLER_PENDING_STATUS:
            raise RoundTripPreflightError(
                "controller status must remain pending in a non-executing preflight",
                code="controller_status_not_pending",
            )
        if self.controller_decision != CONTROLLER_UNKNOWN_DECISION:
            raise RoundTripPreflightError(
                "controller decision must remain unknown in a non-executing preflight",
                code="controller_decision_not_unknown",
            )

    @property
    def preflight_passed(self) -> bool:
        return self.selected_count == self.requested_count and self.consistency.valid

    @property
    def status(self) -> str:
        return "preflight_pass" if self.preflight_passed else "preflight_incomplete"

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def as_dict(self) -> dict[str, Any]:
        semantics = list(self.action_semantics)
        counts = Counter(item.get("interface_id", "unknown") for item in semantics)
        return {
            "schema_version": self.manifest_version,
            "manifest_type": "action_round_trip_preflight",
            "action_interface_schema_version": ACTION_INTERFACE_SCHEMA_VERSION,
            "status": self.status,
            "preflight_status": self.status,
            "requested_count": self.requested_count,
            "target_count": self.requested_count,
            "selected_count": self.selected_count,
            "available_count": self.available_count,
            "count_satisfied": self.selected_count == self.requested_count,
            "source": {
                "path": self.source_path,
                "format": self.source_format,
                "manifest_rows": self.source_manifest_rows,
            },
            "export": {
                "path": self.export_path,
                "format": "jsonl",
                "schema_version": ROUND_TRIP_RECORD_SCHEMA_VERSION,
                "record_count": self.selected_count,
            },
            "import": {
                "format": "jsonl",
                "record_count": self.consistency.imported_count,
            },
            "field_consistency": self.consistency.as_dict(),
            "action_semantics": {
                "interface_counts": dict(sorted(counts.items())),
                "records": semantics,
            },
            "timing": {
                "missing_dt_count": self.timing_missing_dt_count,
                "missing_timestamp_count": self.timing_missing_timestamp_count,
                "dt_policy": "missing/nominal dt is not inferred; source rows retain unknown timing",
            },
            "controller_execution": {
                "status": self.controller_status,
                "decision": self.controller_decision,
                "replay_verified": False,
                "reason": self.controller_reason,
            },
            # These aliases keep machine readers from confusing preflight with
            # physical evidence while making the boundary easy to query.
            "controller_status": self.controller_status,
            "controller_decision": self.controller_decision,
            "replay_verified": False,
            "physical_replay": {"status": "pending", "decision": "unknown", "verified_count": 0},
        }

    to_dict = as_dict


def _canonical_json(value: Any, *, field: str = "value") -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RoundTripPreflightError(
            f"{field} must be JSON-compatible and finite", code="non_json_value"
        ) from exc


def record_sha256(record: Mapping[str, Any]) -> str:
    """Hash one record using the stable JSON representation used by the bundle."""

    if not isinstance(record, Mapping):
        raise RoundTripPreflightError("record must be a mapping", code="not_mapping")
    return hashlib.sha256(_canonical_json(dict(record), field="record").encode("utf-8")).hexdigest()


def _json_record_list(value: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("records", "steps"):
            nested = value.get(key)
            if isinstance(nested, list):
                value = nested
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise RoundTripPreflightError(
            "JSON input must be an object, a list, or an object containing records/steps",
            code="invalid_record_container",
            path=str(path),
        )
    records: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise RoundTripPreflightError(
                f"record {index} is not a JSON object", code="not_mapping", path=str(path)
            )
        # Validate JSON compatibility now so export cannot fail halfway through
        # after writing a partial bundle.
        _canonical_json(dict(row), field=f"record[{index}]")
        records.append(copy.deepcopy(dict(row)))
    return records


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL/NDJSON records with no semantic rewriting."""

    source = Path(path)
    if not source.exists():
        raise RoundTripPreflightError(f"source path does not exist: {source}", code="missing_source", path=str(source))
    if not source.is_file():
        raise RoundTripPreflightError(f"source path is not a file: {source}", code="invalid_source", path=str(source))
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        raise RoundTripPreflightError(
            "parquet requires an explicit dependency-aware adapter; it is not guessed as a controller action",
            code="unsupported_format",
            path=str(source),
        )
    if suffix not in {".json", ".jsonl", ".ndjson"}:
        raise RoundTripPreflightError(
            f"unsupported source format {suffix or '<none>'}; use JSON or JSONL",
            code="unsupported_format",
            path=str(source),
        )
    try:
        if suffix == ".json":
            return _json_record_list(json.loads(source.read_text(encoding="utf-8")), path=source)
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RoundTripPreflightError(
                        f"invalid JSON at line {line_number}: {exc.msg}",
                        code="invalid_json",
                        path=str(source),
                    ) from exc
                if not isinstance(row, Mapping):
                    raise RoundTripPreflightError(
                        f"line {line_number} is not a JSON object", code="not_mapping", path=str(source)
                    )
                _canonical_json(dict(row), field=f"record line {line_number}")
                records.append(copy.deepcopy(dict(row)))
        return records
    except UnicodeDecodeError as exc:
        raise RoundTripPreflightError(
            f"source is not UTF-8 JSON/JSONL: {source}", code="invalid_encoding", path=str(source)
        ) from exc


read_records = read_json_records


def _resolve_manifest_steps(manifest_path: Path, row: Mapping[str, Any]) -> Path:
    raw = row.get("steps_path")
    if not isinstance(raw, str) or not raw.strip():
        raise RoundTripPreflightError(
            "manifest row has no steps_path", code="missing_steps_path", path=str(manifest_path)
        )
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    candidate = manifest_path.parent / candidate
    if candidate.is_file():
        return candidate
    trial = row.get("trial_dir")
    if isinstance(trial, str) and trial.strip():
        candidate = manifest_path.parent / trial / Path(raw)
        if candidate.is_file():
            return candidate
    raise RoundTripPreflightError(
        f"manifest steps_path cannot be resolved: {raw}", code="missing_steps_path", path=str(manifest_path)
    )


def load_source_records(path: str | Path) -> tuple[list[dict[str, Any]], int, str]:
    """Read direct records or expand a Stage-CAP manifest's ``steps_path`` rows.

    The returned manifest-row count is retained as provenance; no manifest
    metadata is injected into the source steps themselves.
    """

    source = Path(path)
    rows = read_json_records(source)
    manifest_like = bool(rows) and any("steps_path" in row for row in rows)
    if not manifest_like:
        return rows, 0, source.suffix.lower().lstrip(".") or "jsonl"
    expanded: list[dict[str, Any]] = []
    manifest_rows = 0
    for row in rows:
        if "steps_path" not in row:
            continue
        manifest_rows += 1
        steps_path = _resolve_manifest_steps(source, row)
        expanded.extend(read_json_records(steps_path))
    return expanded, manifest_rows, "stagecap_manifest+jsonl"


def _sequence_dimension(value: Any) -> int | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    return len(value)


def _numeric_vector_status(value: Any) -> tuple[int | None, str | None]:
    dimension = _sequence_dimension(value)
    if dimension is None:
        return None, "invalid_vector"
    for item in value:
        if isinstance(item, bool):
            return dimension, "non_numeric"
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            return dimension, "non_numeric"
        if not math.isfinite(number):
            return dimension, "non_finite"
    return dimension, None


def inspect_action_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify all action fields in one source row without changing values."""

    if not isinstance(record, Mapping):
        raise RoundTripPreflightError("record must be a mapping", code="not_mapping")
    fields: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    field_names = (
        "action.raw_joint_gripper",
        "action",
        "lerobot_action_candidate",
    )
    for name in field_names:
        if name not in record:
            continue
        dimension, numeric_issue = _numeric_vector_status(record[name])
        field_info: dict[str, Any] = {"dimension": dimension, "value_present": True}
        if name == "lerobot_action_candidate" or (name == "action" and dimension == LEROBOT_EXPORT_ACTION_DIM):
            field_info.update(
                {
                    "interface_id": InterfaceId.LEROBOT_EXPORT_CANDIDATE.value,
                    "semantic": "eef_position_axis_angle_state_difference_plus_raw_gripper",
                    "verified": False,
                    "units": ["m", "rad", "source_value_unverified"],
                }
            )
            if dimension != LEROBOT_EXPORT_ACTION_DIM:
                issues.append({"code": "dimension_mismatch", "field": name, "expected": LEROBOT_EXPORT_ACTION_DIM, "actual": dimension})
        elif dimension == STUDENT_ACTION_DIM:
            field_info.update(
                {
                    "interface_id": InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value,
                    "semantic": "joint_delta_plus_gripper",
                    "verified": False,
                    "units": ["rad", "normalized_command"],
                }
            )
        else:
            field_info.update(
                {
                    "interface_id": "unknown",
                    "semantic": "unresolved_action_semantics",
                    "verified": False,
                    "units": [],
                }
            )
            issues.append({"code": "unrecognized_action_dimension", "field": name, "expected": [7, 8], "actual": dimension})
        if numeric_issue is not None:
            field_info["numeric_status"] = numeric_issue
            issues.append({"code": numeric_issue, "field": name})
        else:
            field_info["numeric_status"] = "finite"
        fields[name] = field_info

    raw = record.get("action.raw_joint_gripper")
    action = record.get("action")
    if raw is not None and action is not None:
        if raw != action:
            issues.append({
                "code": "raw_action_alias_mismatch",
                "field": "action.raw_joint_gripper",
                "related_field": "action",
            })

    if "action.raw_joint_gripper" in fields and fields["action.raw_joint_gripper"].get("interface_id") == InterfaceId.LIBERO_LOW_LEVEL_CONTROLLER.value:
        primary = "action.raw_joint_gripper"
    elif "action" in fields:
        primary = "action"
    elif "lerobot_action_candidate" in fields:
        primary = "lerobot_action_candidate"
    else:
        primary = None
    primary_info = fields.get(primary, {}) if primary else {}
    return {
        "primary_field": primary,
        "fields": fields,
        "interface_id": primary_info.get("interface_id", "unknown"),
        "semantic": primary_info.get("semantic", "unresolved_action_semantics"),
        "verified": bool(primary_info.get("verified", False)),
        "issues": issues,
    }


def _compare_values(
    expected: Any,
    actual: Any,
    *,
    record_index: int,
    field_path: str,
    mismatches: list[FieldMismatch],
) -> int:
    """Recursively compare every JSON field and return leaf count."""

    checked = 0
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            child_path = f"{field_path}.{key}" if field_path else str(key)
            if key not in expected:
                mismatches.append(FieldMismatch(record_index, child_path, "unexpected_field", None, actual[key]))
            elif key not in actual:
                mismatches.append(FieldMismatch(record_index, child_path, "missing_field", expected[key], None))
            else:
                checked += _compare_values(
                    expected[key], actual[key], record_index=record_index, field_path=child_path, mismatches=mismatches
                )
        return checked
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(FieldMismatch(record_index, field_path, "length_mismatch", len(expected), len(actual)))
        for index in range(min(len(expected), len(actual))):
            checked += _compare_values(
                expected[index], actual[index], record_index=record_index, field_path=f"{field_path}[{index}]", mismatches=mismatches
            )
        return checked
    checked = 1
    if type(expected) is not type(actual):
        mismatches.append(FieldMismatch(record_index, field_path, "type_mismatch", type(expected).__name__, type(actual).__name__))
    elif expected != actual:
        mismatches.append(FieldMismatch(record_index, field_path, "value_mismatch", expected, actual))
    return checked


def compare_round_trip_records(
    source_records: Iterable[Mapping[str, Any]],
    imported_records: Iterable[Mapping[str, Any]],
) -> RoundTripConsistencyReport:
    """Compare source and re-imported records field-by-field."""

    source = [dict(record) for record in source_records]
    imported = [dict(record) for record in imported_records]
    mismatches: list[FieldMismatch] = []
    if len(source) != len(imported):
        mismatches.append(FieldMismatch(None, "records", "record_count_mismatch", len(source), len(imported)))
    checked_fields = 0
    for index, (expected, actual) in enumerate(zip(source, imported)):
        checked_fields += _compare_values(expected, actual, record_index=index, field_path="", mismatches=mismatches)
    return RoundTripConsistencyReport(
        source_count=len(source),
        imported_count=len(imported),
        checked_record_count=min(len(source), len(imported)),
        checked_field_count=checked_fields,
        mismatches=tuple(mismatches),
    )


check_round_trip_consistency = compare_round_trip_records


def export_round_trip_records(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
) -> Path:
    """Write a self-contained JSONL bundle preserving records exactly."""

    materialized = [copy.deepcopy(dict(record)) for record in records]
    for index, record in enumerate(materialized):
        if not isinstance(record, Mapping):
            raise RoundTripPreflightError(f"record {index} is not a mapping", code="not_mapping")
        _canonical_json(record, field=f"record[{index}]")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(materialized):
            entry = {
                "schema_version": ROUND_TRIP_RECORD_SCHEMA_VERSION,
                "record_index": index,
                "source_path": str(source_path) if source_path is not None else None,
                "source_record_sha256": record_sha256(record),
                "action_inspection": inspect_action_record(record),
                "record": record,
            }
            handle.write(_canonical_json(entry, field=f"bundle entry {index}") + "\n")
    return destination


export_records = export_round_trip_records


def import_round_trip_records(
    path: str | Path,
    *,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Read a versioned bundle and return copied records or full entries."""

    bundle = Path(path)
    if not bundle.exists() or not bundle.is_file():
        raise RoundTripPreflightError(f"bundle path is not a file: {bundle}", code="missing_bundle", path=str(bundle))
    entries: list[dict[str, Any]] = []
    with bundle.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RoundTripPreflightError(
                    f"invalid bundle JSON at line {line_number}", code="invalid_json", path=str(bundle)
                ) from exc
            if not isinstance(entry, Mapping):
                raise RoundTripPreflightError(
                    f"bundle line {line_number} is not an object", code="not_mapping", path=str(bundle)
                )
            if entry.get("schema_version") != ROUND_TRIP_RECORD_SCHEMA_VERSION:
                raise RoundTripPreflightError(
                    f"bundle line {line_number} has an unexpected schema version",
                    code="schema_version_mismatch",
                    path=str(bundle),
                )
            record = entry.get("record")
            if not isinstance(record, Mapping):
                raise RoundTripPreflightError(
                    f"bundle line {line_number} has no record object", code="missing_record", path=str(bundle)
                )
            record_copy = copy.deepcopy(dict(record))
            expected_hash = entry.get("source_record_sha256")
            if expected_hash != record_sha256(record_copy):
                raise RoundTripPreflightError(
                    f"bundle line {line_number} record hash mismatch",
                    code="record_hash_mismatch",
                    path=str(bundle),
                )
            entries.append(copy.deepcopy(dict(entry)))
    if include_metadata:
        return entries
    return [entry["record"] for entry in entries]


import_records = import_round_trip_records


def _output_paths(output: str | Path) -> tuple[Path, Path]:
    target = Path(output)
    if target.suffix.lower() in {".json", ".manifest"}:
        manifest_path = target
        bundle_path = target.with_name(target.stem + ".records.jsonl")
    else:
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "action_round_trip_manifest.json"
        bundle_path = target / "action_round_trip_records.jsonl"
    return manifest_path, bundle_path


def write_round_trip_manifest(manifest: RoundTripManifest | Mapping[str, Any], path: str | Path) -> Path:
    """Write one manifest mapping with finite, deterministic JSON."""

    payload = manifest.as_dict() if isinstance(manifest, RoundTripManifest) else dict(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_canonical_json(payload, field="manifest") + "\n", encoding="utf-8")
    return destination


def read_round_trip_manifest(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate a preflight manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoundTripPreflightError("could not read round-trip manifest", code="invalid_manifest", path=str(manifest_path)) from exc
    if not isinstance(payload, Mapping):
        raise RoundTripPreflightError("round-trip manifest must be an object", code="invalid_manifest", path=str(manifest_path))
    if payload.get("schema_version") != ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION:
        raise RoundTripPreflightError("unexpected round-trip manifest schema version", code="schema_version_mismatch", path=str(manifest_path))
    return copy.deepcopy(dict(payload))


def build_round_trip_manifest(
    source: str | Path,
    output: str | Path,
    *,
    count: int = ROUND_TRIP_TARGET_COUNT,
    start: int = 0,
) -> RoundTripManifest:
    """Run a 30-record (by default) export/import consistency preflight.

    If fewer than ``count`` source records are present, the manifest is still
    written with the available records and ``status=preflight_incomplete``;
    records are never duplicated to manufacture a 30-row result.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RoundTripPreflightError("count must be a positive integer", code="invalid_count")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise RoundTripPreflightError("start must be a non-negative integer", code="invalid_start")
    source_path = Path(source)
    records, manifest_rows, source_format = load_source_records(source_path)
    selected = records[start : start + count]
    manifest_path, bundle_path = _output_paths(output)
    export_round_trip_records(selected, bundle_path, source_path=source_path)
    imported = import_round_trip_records(bundle_path)
    consistency = compare_round_trip_records(selected, imported)
    semantics: list[dict[str, Any]] = []
    missing_dt = 0
    missing_timestamp = 0
    for index, record in enumerate(selected):
        inspection = inspect_action_record(record)
        semantics.append({"record_index": index, **inspection})
        if record.get("dt") is None and record.get("dt_s") is None:
            missing_dt += 1
        if record.get("timestamp") is None and record.get("timestamp_s") is None:
            missing_timestamp += 1
    manifest = RoundTripManifest(
        source_path=str(source_path),
        export_path=str(bundle_path),
        requested_count=count,
        selected_count=len(selected),
        available_count=len(records),
        consistency=consistency,
        action_semantics=tuple(semantics),
        source_format=source_format,
        source_manifest_rows=manifest_rows,
        timing_missing_dt_count=missing_dt,
        timing_missing_timestamp_count=missing_timestamp,
    )
    write_round_trip_manifest(manifest, manifest_path)
    return manifest


run_round_trip_preflight = build_round_trip_manifest
preflight_round_trip = build_round_trip_manifest


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="JSON/JSONL steps or Stage-CAP manifest")
    parser.add_argument("--output", type=Path, required=True, help="Manifest path or output directory")
    parser.add_argument("--count", type=int, default=ROUND_TRIP_TARGET_COUNT)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    try:
        manifest = build_round_trip_manifest(args.source, args.output, count=args.count, start=args.start)
    except RoundTripPreflightError as exc:
        parser.error(f"{exc.code}: {exc}")
    print(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return _cli()


__all__ = [
    "CONTROLLER_PENDING_STATUS",
    "CONTROLLER_UNKNOWN_DECISION",
    "FieldMismatch",
    "ROUND_TRIP_MANIFEST_SCHEMA_VERSION",
    "ROUND_TRIP_PREFLIGHT_SCHEMA_VERSION",
    "ROUND_TRIP_RECORD_SCHEMA_VERSION",
    "ROUND_TRIP_TARGET_COUNT",
    "RoundTripConsistencyReport",
    "RoundTripManifest",
    "RoundTripPreflightError",
    "build_round_trip_manifest",
    "check_round_trip_consistency",
    "compare_round_trip_records",
    "export_records",
    "export_round_trip_records",
    "import_records",
    "import_round_trip_records",
    "inspect_action_record",
    "load_source_records",
    "main",
    "preflight_round_trip",
    "read_json_records",
    "read_records",
    "read_round_trip_manifest",
    "record_sha256",
    "run_round_trip_preflight",
    "write_round_trip_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
