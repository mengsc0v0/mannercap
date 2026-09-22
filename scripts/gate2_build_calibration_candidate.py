"""Build a local, calibration-gated candidate evidence package.

This command intentionally reads only checked-in Gate 1/Gate 2 reports,
small JSON artifacts, hash manifests, and the pure measurement contracts.  It
never fetches a run directory and never treats a report aggregate as a raw
sensor trace.  When a raw artifact named by a hash manifest is absent from the
workspace, the corresponding measurement remains ``unknown``.

The output is a development candidate package, not a calibrated registry.  It
contains three independent artifacts:

* ``calibration_input_manifest_<stamp>.json``: source hashes and raw-input
  availability;
* ``calibration_candidate_report_<stamp>.md``: evidence-linked findings and
  the missing next-experiment inputs;
* ``constraint_registry_candidate_<stamp>.json``: a pending, null-threshold
  registry that is structurally compatible with the v2 registry contract.

The package can be generated without MuJoCo, a server, an API, or a model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_STAMP = "20260921"
MANIFEST_SCHEMA = "mannercap-calibration-input-manifest-v1"
REPORT_SCHEMA = "mannercap-calibration-candidate-report-v1"
REGISTRY_SCHEMA = "synthesis_v2.constraint_registry.v1"
HASH_RE = re.compile(r"^(?P<digest>[0-9a-fA-F]{64})\s+(?P<path>.+?)\s*$")


@dataclass(frozen=True)
class SourceSpec:
    artifact_id: str
    relative_path: str
    kind: str
    role: str
    raw_measurement: bool = False


# These are source documents, not a claim that the remote run payloads are
# present locally.  The hash manifests are separate source entries below.
SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "gate1_force_calibration_protocol",
        "reports/gate1/force_calibration_protocol.md",
        "report",
        "measurement_contract",
    ),
    SourceSpec(
        "gate1_force_calibration_results",
        "reports/gate1/force_calibration_results.md",
        "report",
        "reported_summary",
    ),
    SourceSpec(
        "gate1_force_measurement_spec",
        "reports/gate1/force_measurement_spec.md",
        "report",
        "measurement_contract",
    ),
    SourceSpec(
        "gate1_wipe_controlled_load_spec",
        "reports/gate1/wipe_controlled_load_spec.md",
        "report",
        "measurement_contract",
    ),
    SourceSpec(
        "gate1_wipe_controlled_load_results",
        "reports/gate1/wipe_controlled_load_results.md",
        "report",
        "reported_summary",
    ),
    SourceSpec(
        "gate1_wipe_calibration_results",
        "reports/gate1/wipe_calibration_results.md",
        "report",
        "geometry_task_summary",
    ),
    SourceSpec(
        "gate1_wipe_contact_dense_results",
        "reports/gate1/wipe_contact_dense_results.md",
        "report",
        "geometry_task_summary",
    ),
    SourceSpec(
        "gate1_wipe_contact_grid_results",
        "reports/gate1/wipe_contact_grid_results.md",
        "report",
        "geometry_task_summary",
    ),
    SourceSpec(
        "gate1_constraint_registry_spec",
        "reports/gate1/constraint_registry_spec.md",
        "report",
        "registry_contract",
    ),
    SourceSpec(
        "checked_in_constraint_registry",
        "configs/constraint_registry.yaml",
        "config",
        "pending_registry_reference",
    ),
    SourceSpec(
        "force_calibration_contract",
        "synthesis_v2/force_calibration.py",
        "code_contract",
        "measurement_contract",
    ),
    SourceSpec(
        "contact_force_contract",
        "synthesis_v2/contact_force_contract.py",
        "code_contract",
        "measurement_contract",
    ),
    SourceSpec(
        "trajectory_contract",
        "synthesis_v2/trajectory_contract.py",
        "code_contract",
        "timing_lineage_contract",
    ),
    SourceSpec(
        "d0_development_summary",
        "reports/gate2/d0_development_summary_64ep_20260919.json",
        "json_summary",
        "development_summary_excluded_from_calibration",
    ),
    SourceSpec(
        "five_method_audit",
        "reports/gate2/five_method_audit_20260921_v2.json",
        "json_summary",
        "method_output_excluded_from_calibration",
    ),
)


SHA_MANIFEST_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "gate1_force_evidence_sha",
        "reports/gate1/evidence_manifest_attempt03.sha256",
        "hash_manifest",
        "remote_raw_evidence_index",
    ),
    SourceSpec(
        "gate1_wipe_controlled_load_sha",
        "reports/gate1/wipe_controlled_load_20260917.sha256",
        "hash_manifest",
        "remote_raw_evidence_index",
    ),
    SourceSpec(
        "gate1_wipe_calibration_sha",
        "reports/gate1/wipe_calibration_20260916.sha256",
        "hash_manifest",
        "remote_raw_evidence_index",
    ),
    SourceSpec(
        "gate1_wipe_contact_grid_sha",
        "reports/gate1/wipe_contact_grid_20260917.sha256",
        "hash_manifest",
        "remote_raw_evidence_index",
    ),
    SourceSpec(
        "gate1_wipe_contact_dense_sha",
        "reports/gate1/wipe_contact_dense_20260918.sha256",
        "hash_manifest",
        "remote_raw_evidence_index",
    ),
)


RAW_NAME_HINTS = (
    "transition",
    "controlled_load_calibration",
    "controlled_wrench",
    "force_diagnostic",
    "contact_force",
    "probe.json",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a local file without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: Mapping[str, Any], *, excluded: Iterable[str] = ()) -> str:
    """Hash JSON-compatible content deterministically, excluding named fields."""

    copied = dict(document)
    for key in excluded:
        copied.pop(key, None)
    payload = json.dumps(
        copied,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_record(spec: SourceSpec, root: Path) -> dict[str, Any]:
    path = root / spec.relative_path
    exists = path.is_file()
    return {
        "artifact_id": spec.artifact_id,
        "path": spec.relative_path,
        "kind": spec.kind,
        "role": spec.role,
        "exists_local": exists,
        "sha256": sha256_file(path) if exists else None,
        "raw_measurement": spec.raw_measurement,
        "eligible_for_measurement": bool(spec.raw_measurement and exists),
        "scope_note": (
            "local bytes are available"
            if exists
            else "source is absent from the local workspace; no measurement is inferred"
        ),
    }


def _manifest_lines(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    parsed: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HASH_RE.match(line)
        if match:
            parsed.append((match.group("digest").lower(), match.group("path")))
    return parsed


def _safe_local_candidate(root: Path, declared_path: str) -> Path | None:
    """Resolve only a path already inside ``root``; never search or fetch."""

    candidate = Path(declared_path)
    if candidate.is_absolute():
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return None
        return candidate
    candidate = root / candidate
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _measurement_like(declared_path: str) -> bool:
    lowered = declared_path.replace("\\", "/").lower()
    return any(hint in lowered for hint in RAW_NAME_HINTS)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3:
        return None
    numbers = tuple(_finite_number(item) for item in values)
    if any(item is None for item in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _first(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _looks_like_measurement(record: Mapping[str, Any]) -> bool:
    keys = set(record)
    return bool(
        keys.intersection({"force_n", "force", "force_vector_n", "force_vector"})
        and keys.intersection({"torque_nm", "torque", "torque_vector_nm", "torque_vector"})
    )


def _known_load_norm(known_load: Mapping[str, Any]) -> float | None:
    force = _vector3(_first(known_load, "force_world_n", "force_n", "force"))
    if force is None:
        return None
    return math.sqrt(sum(component * component for component in force))


def _measurement_record(value: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize an explicit raw row while retaining only declared metadata.

    Gate 1 payloads use two related forms: controlled-load samples place their
    vectors under ``channels.raw_sensordata`` and body diagnostics expose
    ``measured_body_*`` vectors.  Both are explicit rows.  Aggregate fields
    such as ``response_norm_n`` and audit counters are deliberately ignored.
    """

    force = _first(value, "force_n", "force", "force_vector_n", "force_vector")
    torque = _first(value, "torque_nm", "torque", "torque_vector_nm", "torque_vector")
    channels = value.get("channels")
    if (force is None or torque is None) and isinstance(channels, Mapping):
        raw = channels.get("raw_sensordata")
        if not isinstance(raw, Mapping):
            raw = channels.get("recent_ee_forcetorques")
        if isinstance(raw, Mapping):
            force = _first(raw, "force_n", "force", "force_vector_n", "force_vector")
            torque = _first(raw, "torque_nm", "torque", "torque_vector_nm", "torque_vector")
    if force is None or torque is None:
        force = value.get("measured_body_force_n")
        torque = value.get("measured_body_torque_nm")
    if _vector3(force) is None or _vector3(torque) is None:
        return None

    row: dict[str, Any] = dict(value)
    row["force_n"] = force
    row["torque_nm"] = torque

    known_load = value.get("known_load")
    if not isinstance(known_load, Mapping):
        known_load = context.get("known_load")
    if isinstance(known_load, Mapping):
        known_norm = _known_load_norm(known_load)
        if known_norm is not None:
            row["known_load_n"] = known_norm
        row.setdefault("frame", known_load.get("sensor_frame"))
        row.setdefault("units_force", known_load.get("force_unit"))
        row.setdefault("units_torque", known_load.get("torque_unit"))

    # Body diagnostics carry their application frame at the top level; a
    # controlled-load sample inherits the sensor frame from its case.
    protocol = context.get("load_protocol")
    if isinstance(protocol, Mapping):
        row.setdefault("frame", protocol.get("sensor_frame"))
        row.setdefault("units_force", protocol.get("known_load_force_unit"))
        row.setdefault("units_torque", protocol.get("known_load_torque_unit"))
    if not row.get("frame"):
        row["frame"] = value.get("wrench_frame")
    if not isinstance(row.get("sign_convention"), str) or not row["sign_convention"].strip():
        for sign_source in (value.get("sign_convention"), value.get("sensor_transform"), context.get("sensor_transform")):
            if isinstance(sign_source, str) and sign_source.strip():
                row["sign_convention"] = sign_source
                break

    binding = value.get("binding")
    if not isinstance(binding, Mapping):
        binding = context.get("binding")
    body = binding.get("body") if isinstance(binding, Mapping) else None
    if isinstance(body, Mapping):
        row.setdefault("body_id", body.get("id"))
        row.setdefault("body_name", body.get("name"))
        row.setdefault("geom_ids", body.get("geom_ids"))
        row.setdefault("geom_names", body.get("geom_names"))

    # These fields are carried by case/sample containers rather than by every
    # individual row in the Gate 1 payload.  Keeping them on normalized rows
    # makes stream scope explicit: timing checks must never pair timestamps
    # across cases, reset IDs, or load conditions.
    for context_key in ("case_id", "sample_role", "sampling", "source_reset_id"):
        if context_key in context and context_key not in row:
            row[context_key] = context[context_key]
    if "known_load_force_n" not in row and isinstance(known_load, Mapping):
        known_force = _first(known_load, "force_world_n", "force_n", "force")
        if _vector3(known_force) is not None:
            row["known_load_force_n"] = list(known_force)
    if "known_load_torque_nm" not in row and isinstance(known_load, Mapping):
        known_torque = _first(known_load, "torque_world_nm", "torque_nm", "torque")
        if _vector3(known_torque) is not None:
            row["known_load_torque_nm"] = list(known_torque)

    return row


def _collect_measurement_records(
    value: Any,
    inherited_context: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Collect explicit rows, propagating case metadata but never summaries."""

    if isinstance(value, Mapping):
        context = dict(inherited_context or {})
        case_id = value.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            context["case_id"] = case_id
        reset_id = value.get("reset_id")
        if isinstance(reset_id, str) and reset_id.strip():
            context["source_reset_id"] = reset_id
        for key in ("known_load", "binding", "sampling", "load_protocol"):
            child = value.get(key)
            if isinstance(child, Mapping):
                context[key] = child
        if isinstance(value.get("sensor_transform"), str):
            context["sensor_transform"] = value["sensor_transform"]
        record = _measurement_record(value, context)
        if record is not None:
            return [record]
        records: list[Mapping[str, Any]] = []
        for key in (
            "samples",
            "records",
            "rows",
            "transitions",
            "contact_samples",
            "contact_normal_samples",
            "cases",
            "baseline_samples",
            "loaded_samples",
            "representative_samples",
            "body_aggregates",
            "object_contacts",
        ):
            child = value.get(key)
            if isinstance(child, (Mapping, list, tuple)):
                child_context = dict(context)
                if key in {"baseline_samples", "loaded_samples"}:
                    child_context["sample_role"] = key
                records.extend(_collect_measurement_records(child, child_context))
        return records
    if isinstance(value, (list, tuple)):
        records = []
        for item in value:
            records.extend(_collect_measurement_records(item, inherited_context))
        return records
    return []


def _load_measurement_payload(path: Path) -> tuple[list[Mapping[str, Any]], list[str], dict[str, Any]]:
    """Parse a local payload conservatively and return records plus issues."""

    issues: list[str] = []
    metadata: dict[str, Any] = {}
    try:
        if path.suffix.lower() == ".jsonl":
            payload: list[Any] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    payload.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    issues.append(f"jsonl_line_{line_number}_invalid:{exc.msg}")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [f"payload_unreadable:{type(exc).__name__}"], metadata

    if isinstance(payload, Mapping):
        for key in ("independent_reference", "calibration_split", "measurement_resolution_n", "noise_std_n"):
            if key in payload:
                metadata[key] = payload[key]
    records = _collect_measurement_records(payload)
    if not records:
        issues.append("no_explicit_measurement_rows")
    return records, issues, metadata


def _vector_norm(value: Any) -> float | None:
    vector = _vector3(value)
    if vector is None:
        return None
    return math.sqrt(sum(component * component for component in vector))


def _audit_timestamp_streams(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check timestamps within explicit case/reset/sample streams.

    Gate 1 controlled-load rows contain real ``sim.data.time`` samples but do
    not store a per-row ``dt`` field.  We therefore derive only adjacent
    differences within one case, reset, and baseline/loaded stream.  The first
    row of each stream has no preceding sample and is never assigned a dt.
    """

    grouped: dict[tuple[str, str, str], list[tuple[int, Mapping[str, Any], float]]] = {}
    issues: list[str] = []
    explicit_dt_count = 0
    missing_timestamp_count = 0
    missing_reset_count = 0
    for index, record in enumerate(records):
        timestamp = _finite_number(_first(record, "timestamp_s", "timestamp"))
        reset_id = record.get("reset_id")
        if timestamp is None:
            missing_timestamp_count += 1
            issues.append("timestamp_missing")
            continue
        if not isinstance(reset_id, str) or not reset_id.strip():
            missing_reset_count += 1
            issues.append("reset_id_missing")
            continue
        case_id = record.get("case_id")
        sample_role = record.get("sample_role")
        case_key = case_id if isinstance(case_id, str) and case_id.strip() else "__unscoped_case__"
        role_key = sample_role if isinstance(sample_role, str) and sample_role.strip() else "__unscoped_stream__"
        grouped.setdefault((case_key, reset_id, role_key), []).append((index, record, timestamp))
        explicit_dt = _finite_number(record.get("dt"))
        if explicit_dt is not None and explicit_dt > 0.0:
            explicit_dt_count += 1

    derived_dts: list[float] = []
    explicit_dts: list[float] = []
    valid_groups = 0
    groups_with_first_dt_unknown = 0
    gap_groups = 0
    for group_key, rows in grouped.items():
        timestamps = [row[2] for row in rows]
        if len(timestamps) < 2:
            issues.append(f"stream_too_short:{group_key[0]}:{group_key[2]}")
            continue
        valid_groups += 1
        if _finite_number(rows[0][1].get("dt")) is None:
            groups_with_first_dt_unknown += 1
        diffs = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if any(not math.isfinite(delta) or delta <= 0.0 for delta in diffs):
            issues.append(f"timestamps_not_strictly_increasing:{group_key[0]}:{group_key[2]}")
            continue
        derived_dts.extend(diffs)
        group_explicit = [_finite_number(row[1].get("dt")) for row in rows[1:]]
        if any(value is not None and value <= 0.0 for value in group_explicit):
            issues.append(f"explicit_dt_not_positive:{group_key[0]}:{group_key[2]}")
        for delta, explicit in zip(diffs, group_explicit):
            if explicit is not None:
                explicit_dts.append(explicit)
                if not math.isclose(delta, explicit, rel_tol=1e-6, abs_tol=1e-9):
                    issues.append(f"explicit_dt_mismatch:{group_key[0]}:{group_key[2]}")
        # Without a pre-registered cadence, consistency inside the stream is
        # the only defensible no-gap check.  Do not compare across reset/case.
        if max(diffs) - min(diffs) > max(1e-9, max(diffs) * 1e-6):
            gap_groups += 1
            issues.append(f"sampling_gap_or_variable_dt:{group_key[0]}:{group_key[2]}")

    status = (
        "pass"
        if grouped
        and valid_groups == len(grouped)
        and not issues
        else "unknown"
    )
    return {
        "status": status,
        "stream_count": len(grouped),
        "valid_stream_count": valid_groups,
        "first_row_dt_unknown_stream_count": groups_with_first_dt_unknown,
        "explicit_dt_row_count": explicit_dt_count,
        "derived_dt_count": len(derived_dts),
        "min_dt_s": min(derived_dts) if derived_dts else None,
        "max_dt_s": max(derived_dts) if derived_dts else None,
        "explicit_dt_min_s": min(explicit_dts) if explicit_dts else None,
        "explicit_dt_max_s": max(explicit_dts) if explicit_dts else None,
        "gap_stream_count": gap_groups,
        "missing_timestamp_count": missing_timestamp_count,
        "missing_reset_count": missing_reset_count,
        "method": "adjacent_timestamp_difference_within_case_reset_sample_role",
        "issues": sorted(set(issues)),
    }


def _recompute_body_response(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute paired baseline/loaded norm response from explicit rows."""

    streams: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        case_id = record.get("case_id")
        reset_id = record.get("reset_id")
        role = record.get("sample_role")
        if not all(isinstance(item, str) and item.strip() for item in (case_id, reset_id, role)):
            continue
        if role not in {"baseline_samples", "loaded_samples"}:
            continue
        streams.setdefault((case_id, reset_id), {}).setdefault(role, []).append(record)

    force_errors: list[float] = []
    torque_errors: list[float] = []
    paired_rows = 0
    paired_cases = 0
    timestamp_mismatch_rows = 0
    for stream in streams.values():
        baseline = stream.get("baseline_samples", [])
        loaded = stream.get("loaded_samples", [])
        if not baseline or not loaded:
            continue
        paired_cases += 1
        for base_row, load_row in zip(baseline, loaded):
            base_time = _finite_number(_first(base_row, "timestamp_s", "timestamp"))
            load_time = _finite_number(_first(load_row, "timestamp_s", "timestamp"))
            if base_time is None or load_time is None or not math.isclose(base_time, load_time, rel_tol=0.0, abs_tol=1e-9):
                timestamp_mismatch_rows += 1
                continue
            base_force = _vector3(_first(base_row, "force_n", "force", "force_vector_n", "force_vector"))
            load_force = _vector3(_first(load_row, "force_n", "force", "force_vector_n", "force_vector"))
            base_torque = _vector3(_first(base_row, "torque_nm", "torque", "torque_vector_nm", "torque_vector"))
            load_torque = _vector3(_first(load_row, "torque_nm", "torque", "torque_vector_nm", "torque_vector"))
            expected_force_norm = _finite_number(load_row.get("known_load_n"))
            expected_torque_norm = _vector_norm(load_row.get("known_load_torque_nm"))
            if None in (base_force, load_force, base_torque, load_torque, expected_force_norm):
                continue
            force_delta_norm = _vector_norm([load_force[i] - base_force[i] for i in range(3)])
            torque_delta_norm = _vector_norm([load_torque[i] - base_torque[i] for i in range(3)])
            if force_delta_norm is None or torque_delta_norm is None or expected_force_norm is None:
                continue
            force_errors.append(abs(force_delta_norm - expected_force_norm))
            if expected_torque_norm is not None:
                torque_errors.append(abs(torque_delta_norm - expected_torque_norm))
            paired_rows += 1

    return {
        "status": "pass" if paired_rows and not timestamp_mismatch_rows else "partial" if paired_rows else "unknown",
        "paired_case_count": paired_cases,
        "paired_row_count": paired_rows,
        "timestamp_mismatch_row_count": timestamp_mismatch_rows,
        "force_norm_abs_error_mean_n": sum(force_errors) / len(force_errors) if force_errors else None,
        "force_norm_abs_error_max_n": max(force_errors) if force_errors else None,
        "torque_norm_abs_error_mean_nm": sum(torque_errors) / len(torque_errors) if torque_errors else None,
        "torque_norm_abs_error_max_nm": max(torque_errors) if torque_errors else None,
        "comparison": "loaded_minus_baseline_norm_against_explicit_case_known_load_norm",
        "threshold_inference": False,
    }


def _audit_contact_diagnostics(root: Path, raw_inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit representative contact rows without calling them calibrated."""

    candidates = [
        item
        for item in raw_inventory
        if item.get("eligible_for_measurement")
        and "force_diagnostic" in str(item.get("declared_path", "")).lower()
    ]
    if not candidates:
        return {
            "status": "unknown",
            "files_audited": 0,
            "episode_count": 0,
            "representative_sample_count": 0,
            "contact_row_count": 0,
            "valid_contact_row_count": 0,
            "sampling_scope": "no_local_force_diagnostic_payload",
            "full_transition_rows_available": False,
            "frame_status": "unknown",
            "sign_status": "unknown",
            "object_binding_status": "unknown",
            "timestamp_status": "unknown",
            "known_load_mapping_status": "unknown",
            "calibration_verified": False,
            "issues": ["no_local_hash_verified_force_diagnostic_payload"],
        }

    episodes: list[Mapping[str, Any]] = []
    issues: list[str] = []
    for item in candidates:
        local_path = item.get("local_path")
        if not isinstance(local_path, str):
            issues.append("force_diagnostic_missing_local_path")
            continue
        try:
            payload = json.loads((root / local_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(f"force_diagnostic_unreadable:{type(exc).__name__}")
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("episodes"), list):
            episodes.extend(item for item in payload["episodes"] if isinstance(item, Mapping))

    sample_count = 0
    contact_count = 0
    valid_count = 0
    frame_ok = True
    sign_ok = True
    binding_ok = True
    time_ok = True
    known_load_mapping = False
    for episode in episodes:
        samples = episode.get("representative_samples")
        if not isinstance(samples, list) or not samples:
            issues.append("representative_samples_missing")
            time_ok = False
            continue
        sample_count += len(samples)
        sample_times: list[float] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                time_ok = False
                continue
            timestamp = _finite_number(sample.get("sim_time_s"))
            if timestamp is None:
                time_ok = False
            else:
                sample_times.append(timestamp)
            contacts = sample.get("object_contacts")
            if not isinstance(contacts, list):
                issues.append("object_contacts_missing")
                continue
            for contact in contacts:
                if not isinstance(contact, Mapping):
                    continue
                contact_count += 1
                vectors_ok = _vector3(contact.get("force_n")) is not None and _vector3(contact.get("torque_nm")) is not None
                ids_ok = all(
                    isinstance(contact.get(field), int) and not isinstance(contact.get(field), bool)
                    for field in ("geom1_id", "geom2_id", "body1_id", "body2_id")
                )
                frame_rows = contact.get("contact_frame_rows_world")
                frame_rows_ok = isinstance(frame_rows, list) and len(frame_rows) == 3 and all(_vector3(row) is not None for row in frame_rows)
                frame_ok = frame_ok and contact.get("wrench_frame") == "contact_local" and frame_rows_ok
                binding_ok = binding_ok and ids_ok and isinstance(contact.get("object_ids"), list) and bool(contact.get("object_ids"))
                attribution = contact.get("force_direction_attribution")
                sign_ok = sign_ok and attribution == "qualified"
                if vectors_ok and ids_ok and frame_rows_ok:
                    valid_count += 1
                else:
                    issues.append("contact_row_missing_vector_frame_or_binding")
        if len(sample_times) < len(samples) or any(right <= left for left, right in zip(sample_times, sample_times[1:])):
            time_ok = False

    if not sign_ok:
        issues.append("contact_force_direction_attribution_not_qualified")
    issues.append("representative_samples_only_full_transition_rows_unavailable")
    return {
        "status": "partial" if episodes and sample_count and valid_count else "unknown",
        "files_audited": len(candidates),
        "episode_count": len(episodes),
        "representative_sample_count": sample_count,
        "contact_row_count": contact_count,
        "valid_contact_row_count": valid_count,
        "sampling_scope": "representative_samples_only",
        "full_transition_rows_available": False,
        "frame_status": "pass" if frame_ok and contact_count else "unknown",
        "sign_status": "pass" if sign_ok and contact_count else "unknown",
        "object_binding_status": "pass" if binding_ok and contact_count else "unknown",
        "timestamp_status": "pass" if time_ok and sample_count else "unknown",
        "known_load_mapping_status": "pass" if known_load_mapping else "unknown",
        "calibration_verified": False,
        "issues": sorted(set(issues)),
    }


def audit_local_raw_measurements(root: Path, raw_inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit locally available rows against the measurement evidence contract.

    A summary JSON with no explicit force/torque rows is not a measurement
    payload.  Timing is derived only inside a case/reset/load stream from
    adjacent explicit timestamps; no fixed-step assumption or report
    aggregate is used.  This function never infers a threshold or calibration
    constant.
    """

    candidates = [item for item in raw_inventory if item.get("eligible_for_measurement")]
    contact_audit = _audit_contact_diagnostics(root, raw_inventory)
    if not candidates:
        return {
            "status": "unknown",
            "measurement_contract_executed": False,
            "raw_files_seen": 0,
            "record_count": 0,
            "valid_record_count": 0,
            "issues": ["no_local_hash_verified_measurement_payload"],
            "field_status": {
                "known_load_n": "unknown",
                "force_torque_vectors": "unknown",
                "sensor_valid": "unknown",
                "frame_sign_units": "unknown",
                "timestamp_dt": "unknown",
                "object_body_geom": "unknown",
                "object_attribution": "unknown",
                "resolution_noise": "unknown",
            },
            "distinct_known_load_n": [],
            "independent_reference": False,
            "timestamp_dt_audit": {"status": "unknown", "issues": ["no_records"]},
            "body_binding_audit": {"status": "unknown", "row_count": 0},
            "response_recompute": {"status": "unknown", "paired_row_count": 0},
            "contact_pair_observation_audit": contact_audit,
        }

    # Contact diagnostics are retained in the inventory but are not allowed
    # to satisfy body-channel calibration fields.  Prefer controlled-load
    # payloads, which have case-level known-load and sensor metadata; fall back
    # to any explicitly named calibration payload for small fixtures.
    body_candidates = [
        item
        for item in candidates
        if any(marker in str(item.get("declared_path", "")).lower() for marker in ("controlled_load_calibration", "controlled_wrench_calibration"))
    ]
    audit_candidates = [
        item for item in body_candidates if "controlled_load_calibration" in str(item.get("declared_path", "")).lower()
    ] or body_candidates or candidates

    all_records: list[Mapping[str, Any]] = []
    issues: list[str] = []
    metadata: list[Mapping[str, Any]] = []
    for item in audit_candidates:
        local_path = item.get("local_path")
        if not isinstance(local_path, str):
            issues.append("eligible_item_missing_local_path")
            continue
        records, payload_issues, payload_metadata = _load_measurement_payload(root / local_path)
        all_records.extend(records)
        issues.extend(f"{local_path}:{issue}" for issue in payload_issues)
        metadata.append(payload_metadata)

    field_status = {
        "known_load_n": "pass",
        "force_torque_vectors": "pass",
        "sensor_valid": "pass",
        "frame_sign_units": "pass",
        "timestamp_dt": "pass",
        "object_body_geom": "pass",
        "object_attribution": "pass",
        "resolution_noise": "unknown",
    }
    valid_count = 0
    known_loads: set[float] = set()
    body_binding_rows = 0
    object_attribution_rows = 0
    for index, record in enumerate(all_records):
        force = _first(record, "force_n", "force", "force_vector_n", "force_vector")
        torque = _first(record, "torque_nm", "torque", "torque_vector_nm", "torque_vector")
        force_ok = _vector3(force) is not None
        torque_ok = _vector3(torque) is not None
        if not force_ok or not torque_ok:
            field_status["force_torque_vectors"] = "unknown"
            issues.append(f"record_{index}:force_or_torque_not_finite_3_vector")
        sensor_valid = record.get("sensor_valid")
        if sensor_valid is not True:
            field_status["sensor_valid"] = "unknown"
            issues.append(f"record_{index}:sensor_valid_not_true")
        frame = record.get("frame")
        units_force = _first(record, "units_force", "force_unit")
        units_torque = _first(record, "units_torque", "torque_unit")
        sign = _first(record, "sign_convention", "force_sign", "sign")
        if frame not in {"mujoco_subtree_com", "contact_local", "world", "site"} or units_force != "N" or units_torque not in {"N*m", "N·m"} or not isinstance(sign, str) or not sign.strip():
            field_status["frame_sign_units"] = "unknown"
            issues.append(f"record_{index}:frame_sign_or_units_missing")
        reset_id = record.get("reset_id")
        if not isinstance(reset_id, str) or not reset_id.strip():
            issues.append(f"record_{index}:reset_id_missing")
        body_id = record.get("body_id")
        body_name = record.get("body_name")
        geom_ids = record.get("geom_ids")
        geom_names = record.get("geom_names")
        body_ok = (
            isinstance(body_id, int)
            and not isinstance(body_id, bool)
            and isinstance(body_name, str)
            and bool(body_name)
            and isinstance(geom_ids, (list, tuple))
            and isinstance(geom_names, (list, tuple))
            and len(geom_ids) == len(geom_names)
            and bool(geom_ids)
        )
        if body_ok:
            body_binding_rows += 1
        else:
            field_status["object_body_geom"] = "unknown"
            issues.append(f"record_{index}:body_geom_binding_missing")
        object_id = record.get("object_id")
        if isinstance(object_id, str) and object_id.strip():
            object_attribution_rows += 1
        known_load = _finite_number(record.get("known_load_n"))
        if known_load is None or known_load < 0.0:
            field_status["known_load_n"] = "unknown"
        else:
            known_loads.add(known_load)
        if force_ok and torque_ok:
            valid_count += 1

    if object_attribution_rows != len(all_records):
        # Body calibration can pass from an explicit body binding.  Contact
        # object attribution remains a separate field and is not promoted by
        # inheriting the body name.
        field_status["object_attribution"] = "unknown"
        issues.append("object_id_missing_on_some_body_rows")
    if len(known_loads) < 2:
        field_status["known_load_n"] = "unknown"
        issues.append("at_least_two_distinct_known_loads_required")

    timestamp_audit = _audit_timestamp_streams(all_records)
    if timestamp_audit.get("status") != "pass":
        field_status["timestamp_dt"] = "unknown"
        issues.extend(f"timestamp_audit:{issue}" for issue in timestamp_audit.get("issues", ()))

    explicit_resolution = all(
        _finite_number(meta.get("measurement_resolution_n")) is not None
        and _finite_number(meta.get("noise_std_n")) is not None
        for meta in metadata
    ) and bool(metadata)
    if explicit_resolution:
        field_status["resolution_noise"] = "pass"
    independent_reference = any(meta.get("independent_reference") is True for meta in metadata)
    if not independent_reference:
        issues.append("independent_reference_not_declared")

    response_recompute = _recompute_body_response(all_records)
    body_binding_audit = {
        "status": "pass" if body_binding_rows == len(all_records) and all_records else "unknown",
        "row_count": len(all_records),
        "complete_row_count": body_binding_rows,
        "object_id_row_count": object_attribution_rows,
        "scope": "body_binding_is_explicit; object_contact_attribution_is_separate",
    }
    return {
        "status": "pass" if all(value == "pass" for value in field_status.values()) and independent_reference and not issues else "unknown",
        "measurement_contract_executed": True,
        "raw_files_seen": len(candidates),
        "calibration_files_audited": len(audit_candidates),
        "record_count": len(all_records),
        "valid_record_count": valid_count,
        "issues": sorted({re.sub(r"record_\d+:", "record_*:", issue) for issue in issues}),
        "field_status": field_status,
        "distinct_known_load_n": sorted(known_loads),
        "independent_reference": independent_reference,
        "timestamp_dt_audit": timestamp_audit,
        "body_binding_audit": body_binding_audit,
        "response_recompute": response_recompute,
        "contact_pair_observation_audit": contact_audit,
    }


def _raw_inventory_entry(
    *,
    root: Path,
    raw_root: Path | None,
    manifest_spec: SourceSpec,
    declared_sha256: str,
    declared_path: str,
) -> dict[str, Any]:
    candidate = None
    for base in (raw_root, root):
        if base is None:
            continue
        candidate = _safe_local_candidate(base, declared_path)
        if candidate is not None and candidate.is_file():
            break
    if candidate is None or not candidate.is_file():
        candidate = None
    exists = candidate is not None and candidate.is_file()
    local_hash = sha256_file(candidate) if exists and candidate is not None else None
    eligible = _measurement_like(declared_path)
    hash_matches = exists and local_hash == declared_sha256
    return {
        "manifest_artifact_id": manifest_spec.artifact_id,
        "manifest_path": manifest_spec.relative_path,
        "declared_path": declared_path,
        "expected_sha256": declared_sha256,
        "local_path": _relative(candidate, root) if exists and candidate is not None else None,
        "exists_local": exists,
        "local_sha256": local_hash,
        "hash_matches": hash_matches,
        "measurement_like_name": eligible,
        "eligible_for_measurement": bool(eligible and hash_matches),
        "availability_status": "available_and_hash_verified" if eligible and hash_matches else "absent_or_not_eligible",
    }


def _evidence_refs(records: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = {record.get("artifact_id"): record for record in records}
    refs: list[dict[str, Any]] = []
    for artifact_id in ids:
        record = by_id.get(artifact_id)
        if record is None:
            refs.append({"artifact_id": artifact_id, "sha256": None, "missing_record": True})
        else:
            refs.append({
                "artifact_id": artifact_id,
                "path": record.get("path"),
                "sha256": record.get("sha256"),
            })
    return refs


def _raw_evidence_refs(
    manifest: Mapping[str, Any],
    needles: Sequence[str],
) -> list[dict[str, Any]]:
    """Return hash-verifiable raw entries without treating expected bytes as local."""

    inventory = manifest.get("raw_artifact_inventory", ())
    if not isinstance(inventory, Sequence):
        return []
    refs: list[dict[str, Any]] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            continue
        declared = str(item.get("declared_path", ""))
        if needles and not any(needle.lower() in declared.lower() for needle in needles):
            continue
        refs.append(
            {
                "artifact_id": f"raw:{item.get('manifest_artifact_id')}:{declared}",
                "path": item.get("local_path") or declared,
                "sha256": item.get("local_sha256") or item.get("expected_sha256"),
                "expected_sha256": item.get("expected_sha256"),
                "hash_matches": bool(item.get("hash_matches")),
                "local": bool(item.get("exists_local")),
            }
        )
    return refs


def _conditions(
    *,
    source_records: Sequence[Mapping[str, Any]],
    raw_inventory: Sequence[Mapping[str, Any]],
    raw_audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_local = [item for item in raw_inventory if item.get("eligible_for_measurement")]
    field_status = raw_audit.get("field_status")
    if not isinstance(field_status, Mapping):
        field_status = {}
    required_source_ids = {
        "gate1_force_calibration_protocol",
        "gate1_force_measurement_spec",
        "gate1_wipe_controlled_load_spec",
        "trajectory_contract",
        "contact_force_contract",
    }
    source_ok = all(
        any(
            record.get("artifact_id") == artifact_id
            and record.get("exists_local")
            and isinstance(record.get("sha256"), str)
            for record in source_records
        )
        for artifact_id in required_source_ids
    )
    # This candidate package deliberately does not infer raw contract validity
    # from report prose or summary JSON.  It becomes true only when local raw
    # measurement records are supplied and checked by a future runner.
    return {
        "contract_sources_present": {
            "satisfied": source_ok,
            "status": "pass" if source_ok else "unknown",
            "reason": "required pure contracts and specifications are locally hashed",
        },
        "local_raw_measurements_present": {
            "satisfied": bool(raw_local) and raw_audit.get("record_count", 0) > 0,
            "status": "pass" if bool(raw_local) and raw_audit.get("record_count", 0) > 0 else "unknown",
            "reason": (
                "at least one raw measurement artifact is local, hash verified, and has explicit rows"
                if raw_local
                else "hash manifests name remote raw payloads, but no eligible raw payload is local"
            ),
        },
        "known_loads_and_sensor_vectors": {
            "satisfied": field_status.get("known_load_n") == "pass" and field_status.get("force_torque_vectors") == "pass" and field_status.get("sensor_valid") == "pass",
            "status": "pass" if field_status.get("known_load_n") == "pass" and field_status.get("force_torque_vectors") == "pass" and field_status.get("sensor_valid") == "pass" else "unknown",
            "reason": "known_load_n, force/torque vectors, and sensor_valid require explicit local rows; summaries cannot satisfy this condition",
        },
        "frame_sign_and_units": {
            "satisfied": field_status.get("frame_sign_units") == "pass",
            "status": "pass" if field_status.get("frame_sign_units") == "pass" else "unknown",
            "reason": "frame/unit/sign declarations require explicit local vectors and transforms",
        },
        "timestamp_dt_resolution": {
            "satisfied": field_status.get("timestamp_dt") == "pass" and field_status.get("resolution_noise") == "pass",
            "status": "pass" if field_status.get("timestamp_dt") == "pass" and field_status.get("resolution_noise") == "pass" else "unknown",
            "reason": "raw timestamps and within-stream adjacent dt checks are available, but explicit resolution/noise repeats are still required",
        },
        "object_body_geom_attribution": {
            "satisfied": field_status.get("object_body_geom") == "pass",
            "status": "pass" if field_status.get("object_body_geom") == "pass" else "unknown",
            "reason": "body/body-geom binding is checked on local body-channel rows; object/contact attribution remains a separate audit",
        },
        "independent_calibration_input": {
            "satisfied": raw_audit.get("independent_reference") is True,
            "status": "pass" if raw_audit.get("independent_reference") is True else "unknown",
            "reason": "an explicit independent_reference declaration and separate raw known-load source are required",
        },
        "contact_pair_calibration_verified": {
            "satisfied": False,
            "status": "unknown",
            "reason": "scope boundary: this candidate package intentionally leaves contact_pair calibration unverified",
        },
        "contact_pair_rows_observed": {
            "satisfied": raw_audit.get("contact_pair_observation_audit", {}).get("status") in {"partial", "pass"},
            "status": raw_audit.get("contact_pair_observation_audit", {}).get("status", "unknown"),
            "reason": "representative contact rows may be inspected for frame, IDs, and time, but they do not establish contact-pair calibration or force sign",
        },
        "method_output_excluded": {
            "satisfied": True,
            "status": "pass",
            "reason": "method summaries are explicitly tagged excluded and never used as measurements",
        },
    }


def build_manifest(
    root: str | Path,
    *,
    stamp: str = DEFAULT_STAMP,
    raw_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest from local bytes only.

    ``raw_root`` is an optional package directory containing files laid out by
    their original SHA-manifest relative paths.  It permits a reviewed,
    hash-verified retrieval to remain separate from the repository's source
    tree while keeping the audit fully offline and reproducible.
    """

    repo_root = Path(root).resolve()
    raw_input_root: Path | None = None
    if raw_root is not None:
        raw_input_root = Path(raw_root)
        if not raw_input_root.is_absolute():
            raw_input_root = repo_root / raw_input_root
        raw_input_root = raw_input_root.resolve()
        try:
            raw_input_root.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError("raw_root must be inside repo root") from exc
    source_specs = SOURCE_SPECS + SHA_MANIFEST_SPECS
    source_records = [_source_record(spec, repo_root) for spec in source_specs]
    raw_inventory: list[dict[str, Any]] = []
    for spec in SHA_MANIFEST_SPECS:
        manifest_path = repo_root / spec.relative_path
        for digest, declared_path in _manifest_lines(manifest_path):
            raw_inventory.append(
                _raw_inventory_entry(
                    root=repo_root,
                    raw_root=raw_input_root,
                    manifest_spec=spec,
                    declared_sha256=digest,
                    declared_path=declared_path,
                )
            )

    raw_audit = audit_local_raw_measurements(repo_root, raw_inventory)
    conditions = _conditions(
        source_records=source_records,
        raw_inventory=raw_inventory,
        raw_audit=raw_audit,
    )
    candidate_ready = all(bool(condition["satisfied"]) for condition in conditions.values())
    field_status = raw_audit.get("field_status", {})
    field_status = field_status if isinstance(field_status, Mapping) else {}
    frame_status = field_status.get("frame_sign_units", "unknown")
    output_manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": f"gate2.calibration_candidate.{stamp}",
        "generated_by": "scripts/gate2_build_calibration_candidate.py",
        "generation_policy": {
            "local_only": True,
            "remote_access": False,
            "raw_input_root": _relative(raw_input_root, repo_root) if raw_input_root is not None else None,
            "summary_json_is_measurement": False,
            "hash_manifest_is_raw_payload": False,
            "raw_measurement_required_for_candidate_ready": True,
        },
        "source_artifacts": source_records,
        "raw_artifact_inventory": raw_inventory,
        "raw_measurement_audit": raw_audit,
        "required_measurement_fields": {
            "known_load_n": field_status.get("known_load_n", "unknown"),
            "force_vector_n": field_status.get("force_torque_vectors", "unknown"),
            "torque_vector_nm": field_status.get("force_torque_vectors", "unknown"),
            "sensor_valid": field_status.get("sensor_valid", "unknown"),
            "frame": frame_status,
            "sign_convention": frame_status,
            "timestamp_s": "pass" if field_status.get("timestamp_dt") == "pass" else "unknown",
            "measured_dt_s": field_status.get("timestamp_dt", "unknown"),
            "control_step_and_physics_substep": "unknown",
            "resolution_and_noise": field_status.get("resolution_noise", "unknown"),
            "object_body_geom_binding": field_status.get("object_body_geom", "unknown"),
            "object_contact_attribution": field_status.get("object_attribution", "unknown"),
        },
        "conditions": conditions,
        "candidate_ready": candidate_ready,
        "physical_qualification": "pass" if candidate_ready else "unknown",
        "thresholds_frozen": False,
        "promotion_allowed": False,
    }
    output_manifest["manifest_content_sha256"] = canonical_hash(
        output_manifest,
        excluded=("manifest_content_sha256",),
    )
    return output_manifest


def _constraint_calibration_block(manifest_ref: str, evidence_refs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pending_calibration",
        "source_ref": manifest_ref,
        "version": "calibration-candidate.v1.pending",
        "evidence_sha256": None,
        "evidence_refs": evidence_refs,
        "measurement_status": "unknown",
    }


def build_registry(manifest: Mapping[str, Any], *, stamp: str = DEFAULT_STAMP) -> dict[str, Any]:
    """Build a pending/null-threshold registry from the manifest."""

    manifest_ref = f"reports/gate2/calibration_input_manifest_{stamp}.json"
    source_records = manifest.get("source_artifacts", ())
    evidence_refs = _evidence_refs(
        source_records if isinstance(source_records, Sequence) else (),
        (
            "gate1_force_calibration_protocol",
            "gate1_force_measurement_spec",
            "gate1_wipe_controlled_load_spec",
            "force_calibration_contract",
            "contact_force_contract",
            "trajectory_contract",
        ),
    )
    task_phases = {
        "lift": ("approach", "descend", "grasp", "lift"),
        "stack": ("approach", "descend", "grasp", "lift", "transport", "place", "retreat"),
    }
    constraints: dict[str, list[dict[str, Any]]] = {"lift": [], "stack": []}
    for task_id, phases in task_phases.items():
        for phase in phases:
            metrics = ("gripper_fraction", "force_norm") if phase == "grasp" else ("force_norm",) if phase == "place" else ("joint_velocity",)
            for metric in metrics:
                unit = {"joint_velocity": "rad/s", "gripper_fraction": "fraction", "force_norm": "N"}[metric]
                constraint_id = f"{task_id}.{phase}.{metric}"
                constraints[task_id].append(
                    {
                        "constraint_id": constraint_id,
                        "task_id": task_id,
                        "phase": phase,
                        "metric": metric,
                        "unit": unit,
                        "scope": {"kind": "stage", "stage_instance_id": phase},
                        "decision_dimensions": ["task", "manner", "safety"],
                        "unknown_policy": "not_qualified",
                        "status": "pending_calibration",
                        "thresholds": {"lower": None, "upper": None, "tolerance": None},
                        "calibration": _constraint_calibration_block(manifest_ref, evidence_refs),
                    }
                )

    document: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": "mannercap_gate2_constraint_registry_candidate",
        "registry_version": f"{stamp}.candidate.1",
        "hash_algorithm": "sha256",
        "registry_version_hash": None,
        "status": "development_pending",
        "calibration": {
            "status": "pending_calibration",
            "source_ref": manifest_ref,
            "version": "calibration-candidate.v1.pending",
            "evidence_sha256": None,
            "evidence_refs": evidence_refs,
            "candidate_ready": bool(manifest.get("candidate_ready", False)),
        },
        "decision_contract": {
            "dimensions": {
                dimension: {"states": ["pass", "fail", "unknown"], "unknown_is_qualified": False}
                for dimension in ("task", "manner", "safety")
            },
            "unknown_policy": "not_qualified",
            "joint_rule": {"any_fail": "fail", "all_pass": "pass", "otherwise": "unknown"},
        },
        "public_generation": {
            "config_id": "public_generation.candidate.v1",
            "visibility": "public",
            "constraint_refs": [
                f"{task_id}.{phase}.{metric}"
                for task_id, phase_list in task_phases.items()
                for phase in phase_list
                for metric in (("gripper_fraction", "force_norm") if phase == "grasp" else ("force_norm",) if phase == "place" else ("joint_velocity",))
            ],
            "allowed_fields": [
                "task_id",
                "phase",
                "metric",
                "unit",
                "constraint_id",
                "calibration.source_ref",
                "calibration.version",
                "calibration.evidence_refs",
                "status",
            ],
            "excluded_fields": ["numeric_thresholds", "sealed_final_audit_contents", "final_audit_results"],
            "final_audit_ref": {"config_id": "sealed_final_audit.candidate.v1", "visibility": "sealed_reference"},
        },
        "sealed_final_audit": {
            "config_id": "sealed_final_audit.candidate.v1",
            "visibility": "sealed",
            "status": "sealed_pending",
            "config_ref": "sealed://mannercap/gate2/final_audit/candidate.v1.json",
            "config_sha256": None,
            "source_of_truth": "independent final-audit artifact; not visible to the generator",
        },
        "tasks": [
            {
                "task_id": task_id,
                "status": "supported",
                "calibration_status": "pending_calibration",
                "phases": list(task_phases[task_id]),
                "constraints": constraints[task_id],
            }
            for task_id in ("lift", "stack")
        ],
        "candidate_evidence": {
            "manifest_content_sha256": manifest.get("manifest_content_sha256"),
            "candidate_ready": bool(manifest.get("candidate_ready", False)),
            "physical_qualification": "pass" if manifest.get("candidate_ready", False) else "unknown",
            "thresholds_frozen": False,
            "promotion_allowed": False,
        },
    }
    document["registry_version_hash"] = canonical_hash(document, excluded=("registry_version_hash",))
    return document


def _finding(
    finding_id: str,
    *,
    status: str,
    decision: str,
    claim: str,
    reason: str,
    evidence_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "status": status,
        "decision": decision,
        "claim": claim,
        "reason": reason,
        "evidence_refs": evidence_refs,
    }


def build_report(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    stamp: str = DEFAULT_STAMP,
) -> dict[str, Any]:
    """Build structured report data; rendering is kept separate for tests."""

    records = manifest.get("source_artifacts", ())
    records = records if isinstance(records, Sequence) else ()
    raw_audit = manifest.get("raw_measurement_audit", {})
    raw_audit = raw_audit if isinstance(raw_audit, Mapping) else {}
    field_status = raw_audit.get("field_status", {})
    field_status = field_status if isinstance(field_status, Mapping) else {}
    raw_rows_available = int(raw_audit.get("record_count", 0) or 0) > 0
    body_status = "partially_recomputed" if raw_rows_available else "contract_declared_only"
    raw_issue_text = ", ".join(str(item) for item in raw_audit.get("issues", ()))
    timestamp_audit = raw_audit.get("timestamp_dt_audit", {})
    timestamp_audit = timestamp_audit if isinstance(timestamp_audit, Mapping) else {}
    response_recompute = raw_audit.get("response_recompute", {})
    response_recompute = response_recompute if isinstance(response_recompute, Mapping) else {}
    contact_audit = raw_audit.get("contact_pair_observation_audit", {})
    contact_audit = contact_audit if isinstance(contact_audit, Mapping) else {}
    if raw_rows_available:
        known_load_status = "raw_rows_recomputed_partial"
        known_load_claim = (
            "SHA-verified controlled-load rows were parsed locally and paired within each case; "
            f"{response_recompute.get('paired_row_count', 0)} baseline/loaded row pairs were recomputed. "
            "The body response remains a candidate input and does not promote any threshold."
        )
        known_load_reason = (
            "Remaining contract gaps are recorded from the parsed rows: " + raw_issue_text
            if raw_issue_text
            else "The parsed rows do not by themselves establish an independent reference or contact-pair qualification."
        )
        time_status = "partially_recomputed"
        time_claim = (
            "Raw sample timestamps were checked within case/reset/load streams; adjacent dt was derived only "
            "between explicit timestamps, with each stream's first dt left unknown."
        )
        time_reason = (
            "timestamp_dt=%s; streams=%s; derived_dt_rows=%s; gaps=%s; resolution_noise=%s."
            % (
                field_status.get("timestamp_dt", "unknown"),
                timestamp_audit.get("valid_stream_count", 0),
                timestamp_audit.get("derived_dt_count", 0),
                timestamp_audit.get("gap_stream_count", 0),
                field_status.get("resolution_noise", "unknown"),
            )
        )
    else:
        known_load_status = "reported_remote_not_recomputed"
        known_load_claim = "Gate 1 reports controlled known-load response cases, but this package has no local explicit sensor rows to recompute their vectors."
        known_load_reason = "The raw controlled-load JSON named by the local SHA manifest is not present in the selected local input root; report aggregates are not measurements."
        time_status = "unknown"
        time_claim = "Measured timestamp, dt, sampling gap, sensor validity, resolution, and repeatability cannot be recomputed."
        time_reason = "Only reports and hash indexes are local; no raw transition stream is available for the trajectory or force contracts."
    findings = [
        _finding(
            "body_wrench_layout_and_units",
            status=body_status,
            decision="unknown",
            claim="The contracts distinguish MuJoCo body wrench torque (N*m) and force (N), and reject mixed six-dimensional norms.",
            reason=(
                "The retrieved controlled-load rows verify vector shape, frame, sign transform, units, and explicit "
                f"body/body-geom binding for {raw_audit.get('record_count', 0)} rows; object/contact attribution, "
                "resolution/noise, and physical qualification remain open."
                if raw_rows_available
                else "No local body-channel rows are available to recompute the declared layout."
            ),
            evidence_refs=(
                _evidence_refs(records, ["gate1_force_measurement_spec", "force_calibration_contract", "contact_force_contract"])
                + _raw_evidence_refs(manifest, ["controlled_load_calibration", "controlled_wrench_calibration"])
            ),
        ),
        _finding(
            "known_load_response",
            status=known_load_status,
            decision="unknown",
            claim=known_load_claim,
            reason=known_load_reason,
            evidence_refs=(
                _evidence_refs(records, ["gate1_wipe_controlled_load_results", "gate1_wipe_controlled_load_sha", "gate1_force_calibration_results", "gate1_force_evidence_sha"])
                + _raw_evidence_refs(manifest, ["controlled_load_calibration", "controlled_wrench_calibration"])
            ),
        ),
        _finding(
            "contact_pair_and_object_attribution",
            status="partially_observed" if contact_audit.get("status") == "partial" else "incomplete",
            decision="unknown",
            claim=(
                "Hash-verified representative contact rows expose contact frame, geom/body IDs, vectors, and sample time, "
                "but contact force direction attribution and calibration remain unknown."
                if contact_audit.get("status") == "partial"
                else "Contact-pair frame, sign, object binding, and geom attribution remain unverified in this candidate package."
            ),
            reason=(
                f"{contact_audit.get('valid_contact_row_count', 0)} of {contact_audit.get('contact_row_count', 0)} "
                "representative contact rows passed structural checks; the payload does not expose full transition rows "
                "or an independent known-load mapping, and all contact calibration remains unverified."
                if contact_audit.get("status") == "partial"
                else "The retrieved body-channel payload does not calibrate contact_pair; this package keeps that boundary explicitly uncalibrated."
            ),
            evidence_refs=(
                _evidence_refs(records, ["gate1_force_calibration_protocol", "gate1_force_calibration_results", "gate1_wipe_contact_dense_results", "gate1_wipe_contact_dense_sha"])
                + _raw_evidence_refs(manifest, ["force_diagnostic", "controlled_wrench_calibration", "transitions"])
            ),
        ),
        _finding(
            "wipe_geometry_vs_force",
            status="geometry_only",
            decision="unknown",
            claim="The Wipe 0--4 mm grid can be referenced as a development geometry/task response, not as a known-force calibration or safety threshold.",
            reason="Surface penetration is a control/geometry input; it is not a known Newton load, and local raw transitions are absent.",
            evidence_refs=_evidence_refs(records, ["gate1_wipe_contact_dense_results", "gate1_wipe_contact_grid_results", "gate1_wipe_contact_dense_sha"]),
        ),
        _finding(
            "time_sampling_and_resolution",
            status=time_status,
            decision="unknown",
            claim=time_claim,
            reason=time_reason,
            evidence_refs=(
                _evidence_refs(records, ["trajectory_contract", "gate1_force_calibration_protocol", "gate1_force_evidence_sha"])
                + _raw_evidence_refs(manifest, ["controlled_load_calibration", "transitions"])
            ),
        ),
        _finding(
            "independent_calibration_boundary",
            status="not_ready",
            decision="unknown",
            claim="No independent known-load reference and no verified contact_pair calibration permit this candidate to be promoted to a physical qualification.",
            reason="The method output is explicitly excluded; independent_reference remains %s and contact_pair calibration is intentionally unverified. All thresholds remain null/pending." % str(raw_audit.get("independent_reference", False)).lower(),
            evidence_refs=(
                _evidence_refs(records, ["gate1_constraint_registry_spec", "checked_in_constraint_registry", "five_method_audit"])
                + _raw_evidence_refs(manifest, ["controlled_load_calibration", "controlled_wrench_calibration", "force_diagnostic"])
            ),
        ),
    ]
    return {
        "schema": REPORT_SCHEMA,
        "report_id": f"gate2.calibration_candidate_report.{stamp}",
        "manifest_ref": f"reports/gate2/calibration_input_manifest_{stamp}.json",
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "registry_ref": f"reports/gate2/constraint_registry_candidate_{stamp}.json",
        "registry_version_hash": registry.get("registry_version_hash"),
        "findings": findings,
        "candidate_ready": bool(manifest.get("candidate_ready", False)),
        "physical_qualification": "pass" if manifest.get("candidate_ready", False) else "unknown",
        "thresholds_frozen": False,
        "promotion_allowed": False,
        "next_experiment_inputs": [
            {
                "input_id": "raw_controlled_load_payloads",
                "required": True,
                "description": "Retain the SHA-verified controlled-load JSON used here and retrieve any additional calibration cases or raw streams named by the manifests before parsing.",
            },
            {
                "input_id": "known_load_and_application_metadata",
                "required": True,
                "description": "For each sample record known force/torque in N/N*m, world direction, application point, site/body frame, sign convention, and baseline pairing.",
            },
            {
                "input_id": "raw_contact_rows",
                "required": True,
                "description": "For a future contact_pair audit, record ncon/contact geom IDs and mj_contactForce force/torque vectors per pair; keep cfrc_ext body aggregate separate. This package leaves contact_pair uncalibrated.",
            },
            {
                "input_id": "timing_and_validity",
                "required": True,
                "description": "Retain the within-stream timestamp audit and add control step, physics substep, reset/stage IDs, invalid spans, and explicit sampling resolution/noise; derived dt here never assumes a fixed step.",
            },
            {
                "input_id": "repeatability_and_noise",
                "required": True,
                "description": "Repeat no-contact and at least two known-load conditions from matched snapshots and record resolution/noise metadata; no formal threshold is set by this candidate package.",
            },
            {
                "input_id": "independent_calibration_split",
                "required": True,
                "description": "Freeze task/seed source initial-state hashes and lineage for a calibration split independent of method outputs; report object/config coverage without claiming object generalization.",
            },
        ],
    }


def render_markdown(report: Mapping[str, Any], manifest: Mapping[str, Any], registry: Mapping[str, Any]) -> str:
    """Render a concise evidence-linked human report."""

    generation_policy = manifest.get("generation_policy", {})
    generation_policy = generation_policy if isinstance(generation_policy, Mapping) else {}
    raw_root = generation_policy.get("raw_input_root")
    if raw_root:
        input_scope = (
            f"本报告只读本地报告、哈希清单、纯测量契约以及 `{raw_root}` 下已取回的原始 payload；"
            "每个 payload 先按原 SHA-256 清单核验。"
        )
    else:
        input_scope = "本报告只读本地报告、哈希清单、小型 JSON 和纯测量契约；未提供额外 raw input root。"
    lines = [
        "# Gate 2 独立物理校准候选包",
        "",
        f"报告 ID：`{report.get('report_id')}`",
        f"候选包状态：`candidate_ready={str(report.get('candidate_ready')).lower()}`",
        f"物理资格：`{report.get('physical_qualification')}`；阈值冻结：`{str(report.get('thresholds_frozen')).lower()}`；晋升：`{str(report.get('promotion_allowed')).lower()}`",
        "",
        f"{input_scope} 汇总数字仍不被当作原始传感器、时间轴或噪声；contact_pair 保持未校准。",
        "",
        "## 输入证据",
        "",
        f"- manifest：`{report.get('manifest_ref')}`，canonical SHA-256 `{manifest.get('manifest_content_sha256')}`",
        f"- candidate registry：`{report.get('registry_ref')}`，registry hash `{registry.get('registry_version_hash')}`",
        f"- 本地 raw measurement 条目：`{sum(1 for item in manifest.get('raw_artifact_inventory', ()) if item.get('eligible_for_measurement'))}`",
        f"- raw contract audit：`{manifest.get('raw_measurement_audit', {}).get('status', 'unknown')}`，records=`{manifest.get('raw_measurement_audit', {}).get('record_count', 0)}`，valid=`{manifest.get('raw_measurement_audit', {}).get('valid_record_count', 0)}`",
        "",
        "## 结论",
        "",
    ]
    for finding in report.get("findings", ()):
        lines.extend(
            [
                f"### `{finding['finding_id']}` — {finding['status']}",
                "",
                f"- 判定：`{finding['decision']}`",
                f"- 结论：{finding['claim']}",
                f"- 限制：{finding['reason']}",
                "- 证据：",
            ]
        )
        for ref in finding.get("evidence_refs", ()):
            lines.append(
                f"  - `{ref.get('artifact_id')}` — `{ref.get('path')}` — SHA-256 `{ref.get('sha256')}`"
            )
        lines.append("")
    lines.extend(
        [
            "## 下一实验必须补齐的输入",
            "",
        ]
    )
    for item in report.get("next_experiment_inputs", ()):
        lines.append(f"- `{item['input_id']}`：{item['description']}")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "candidate registry 保持 `development_pending`，所有 lower/upper/tolerance 为 null；本包不设正式阈值、不把 geometry penetration 改名为力，且 contact_pair 不因 body-channel 或诊断数据而自动校准。",
            "",
        ]
    )
    return "\n".join(lines)


def write_package(
    root: str | Path,
    output_dir: str | Path,
    *,
    stamp: str = DEFAULT_STAMP,
    raw_root: str | Path | None = None,
) -> dict[str, Path]:
    """Generate all three outputs and return their paths."""

    repo_root = Path(root).resolve()
    target = Path(output_dir)
    if not target.is_absolute():
        target = repo_root / target
    target.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(repo_root, stamp=stamp, raw_root=raw_root)
    registry = build_registry(manifest, stamp=stamp)
    report = build_report(manifest, registry, stamp=stamp)

    manifest_path = target / f"calibration_input_manifest_{stamp}.json"
    report_path = target / f"calibration_candidate_report_{stamp}.md"
    registry_path = target / f"constraint_registry_candidate_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(report, manifest, registry), encoding="utf-8")
    return {"manifest": manifest_path, "report": report_path, "registry": registry_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate2"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="optional local package root containing SHA-manifest relative raw payload paths",
    )
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    args = parser.parse_args(argv)
    paths = write_package(args.repo_root, args.output_dir, stamp=args.stamp, raw_root=args.raw_root)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
