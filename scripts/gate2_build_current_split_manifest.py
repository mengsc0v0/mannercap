"""Build and audit the development-only split manifest for current D0 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis_v2.split_manifest import (  # noqa: E402
    SPLIT_MANIFEST_SCHEMA,
    manifest_hash,
    validate_split_manifest,
)


def _source_record(record: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(record["episode_id"])
    pair_id = str(record["pair_id"])
    task = str(record["task"])
    manner = str(record["target_manner_tag"])
    return {
        "record_id": episode_id,
        "kind": "source_episode",
        "split": "development",
        "source_episode_id": episode_id,
        "lineage_root_id": pair_id,
        "initial_state_id": pair_id,
        "scene_configuration_id": pair_id,
        "template_family_id": f"d0:{task}:{manner}",
        "pair_id": pair_id,
        "task": task,
        "manner_tag": manner,
        "source_transitions_sha256": record["source"]["transitions_sha256"],
    }


def build_current_manifest(
    gold_manifest: dict[str, Any],
    *,
    staging_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources = [_source_record(dict(record)) for record in gold_manifest["records"]]
    source_by_id = {record["record_id"]: record for record in sources}
    records = list(sources)

    if staging_manifest is not None:
        for index, window in enumerate(staging_manifest.get("windows", [])):
            source_id = str(window["source_episode_id"])
            if source_id not in source_by_id:
                raise ValueError(f"window {index} references unknown source episode {source_id!r}")
            source = source_by_id[source_id]
            records.append(
                {
                    "record_id": f"la4vla-window-{index:06d}",
                    "kind": "derived_window",
                    "split": source["split"],
                    "source_episode_id": source_id,
                    "lineage_root_id": source["lineage_root_id"],
                    "initial_state_id": source["initial_state_id"],
                    "scene_configuration_id": source["scene_configuration_id"],
                    "template_family_id": source["template_family_id"],
                    "pair_id": source["pair_id"],
                    "window_index": index,
                    "source_start_transition_zero_based": window[
                        "source_start_transition_zero_based"
                    ],
                    "horizon": window["horizon"],
                }
            )

    document = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "status": "development_only_not_frozen",
        "formal_splits_frozen": False,
        "data_use_mapping": {
            "development_calibration": ["development"],
            "generator_sealed_test": ["generator_test"],
            "student_train_validation": ["student_train", "student_validation"],
            "student_sealed_test": ["student_test"],
        },
        "records": records,
    }
    document["manifest_sha256"] = manifest_hash(document)
    validate_split_manifest(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--la4vla-staging-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold = json.loads(args.gold_manifest.read_text(encoding="utf-8"))
    staging = (
        json.loads(args.la4vla_staging_manifest.read_text(encoding="utf-8"))
        if args.la4vla_staging_manifest
        else None
    )
    document = build_current_manifest(gold, staging_manifest=staging)
    audit = validate_split_manifest(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **audit}, sort_keys=True))


if __name__ == "__main__":
    main()
