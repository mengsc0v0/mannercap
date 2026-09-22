"""Leakage-safe split manifest contract for source episodes and derived samples."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from .request_manifest import SUPPORTED_SPLITS


SPLIT_MANIFEST_SCHEMA: Final[str] = "synthesis_v2.split_manifest.v1"
GROUP_FIELDS: Final[tuple[str, ...]] = (
    "lineage_root_id",
    "initial_state_id",
    "scene_configuration_id",
    "template_family_id",
    "pair_id",
)
RECORD_KINDS: Final[frozenset[str]] = frozenset(
    {"source_episode", "derived_window", "request"}
)


class SplitManifestError(ValueError):
    """Raised when a split assignment is incomplete or leaks related data."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SplitManifestError("split manifest must be finite JSON") from exc


def _text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SplitManifestError(f"{field} must be a non-empty string")
    return value


def manifest_hash(document: Mapping[str, Any]) -> str:
    """Hash the manifest while excluding an optional embedded hash field."""

    payload = deepcopy(dict(document))
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_split_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate split exclusivity and return an audit summary.

    All relationship identifiers are treated as indivisible groups.  A pair,
    initial state, lineage root, scene/configuration, or template family may
    occur many times, but every occurrence must stay in one split.  Derived
    windows must reference a source episode in the same manifest and inherit
    its split and all grouping fields.
    """

    if not isinstance(document, Mapping):
        raise SplitManifestError("split manifest must be a mapping")
    if document.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise SplitManifestError(f"schema must be {SPLIT_MANIFEST_SCHEMA!r}")
    records = document.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise SplitManifestError("records must be a sequence")
    if not records:
        raise SplitManifestError("records must not be empty")

    normalized: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_counts: dict[str, int] = {split: 0 for split in SUPPORTED_SPLITS}
    kind_counts: dict[str, int] = {kind: 0 for kind in sorted(RECORD_KINDS)}

    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise SplitManifestError(f"records[{index}] must be a mapping")
        record = deepcopy(dict(raw))
        record_id = _text(record.get("record_id"), f"records[{index}].record_id")
        if record_id in by_id:
            raise SplitManifestError(f"duplicate record_id: {record_id}")
        kind = _text(record.get("kind"), f"records[{index}].kind")
        if kind not in RECORD_KINDS:
            raise SplitManifestError(f"record {record_id!r} has unsupported kind {kind!r}")
        split = _text(record.get("split"), f"records[{index}].split")
        if split not in SUPPORTED_SPLITS:
            raise SplitManifestError(f"record {record_id!r} has unsupported split {split!r}")

        for field in GROUP_FIELDS:
            optional = field == "pair_id"
            value = _text(record.get(field), f"record {record_id}.{field}", optional=optional)
            if value is not None:
                group_splits[(field, value)].add(split)

        source_episode_id = _text(
            record.get("source_episode_id"),
            f"record {record_id}.source_episode_id",
            optional=kind != "derived_window",
        )
        if kind == "source_episode" and source_episode_id not in (None, record_id):
            raise SplitManifestError(
                f"source episode {record_id!r} cannot name a different source_episode_id"
            )
        by_id[record_id] = record
        normalized.append(record)
        split_counts[split] += 1
        kind_counts[kind] += 1

    leaks = [
        {"field": field, "group_id": group_id, "splits": sorted(splits)}
        for (field, group_id), splits in sorted(group_splits.items())
        if len(splits) > 1
    ]
    if leaks:
        first = leaks[0]
        raise SplitManifestError(
            f"split leakage for {first['field']}={first['group_id']!r}: {first['splits']}"
        )

    for record in normalized:
        if record["kind"] != "derived_window":
            continue
        record_id = str(record["record_id"])
        source_id = str(record["source_episode_id"])
        source = by_id.get(source_id)
        if source is None or source.get("kind") != "source_episode":
            raise SplitManifestError(
                f"derived window {record_id!r} references missing source episode {source_id!r}"
            )
        if record["split"] != source["split"]:
            raise SplitManifestError(
                f"derived window {record_id!r} must inherit source split {source['split']!r}"
            )
        for field in GROUP_FIELDS:
            if record.get(field) != source.get(field):
                raise SplitManifestError(
                    f"derived window {record_id!r} must inherit source field {field}"
                )

    embedded = document.get("manifest_sha256")
    computed = manifest_hash(document)
    if embedded is not None and embedded != computed:
        raise SplitManifestError("manifest_sha256 does not match manifest contents")

    return {
        "schema": "synthesis_v2.split_manifest.audit.v1",
        "valid": True,
        "record_count": len(normalized),
        "split_counts": split_counts,
        "kind_counts": kind_counts,
        "group_count": len(group_splits),
        "leak_count": 0,
        "manifest_sha256": computed,
    }


__all__ = [
    "GROUP_FIELDS",
    "RECORD_KINDS",
    "SPLIT_MANIFEST_SCHEMA",
    "SplitManifestError",
    "manifest_hash",
    "validate_split_manifest",
]
