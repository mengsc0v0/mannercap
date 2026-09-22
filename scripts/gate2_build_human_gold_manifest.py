"""Build a blinded, deterministic human-gold candidate manifest from D0 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HIDDEN_EXECUTOR_FIELDS = (
    "task_completed",
    "formal_manner_status",
    "formal_safety_status",
    "promotion_allowed",
    "execution",
    "descriptive_metrics",
    "descriptive_phase_metrics",
    "offline_recomputation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_results(run_dirs: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    observed: list[tuple[Path, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for run_dir in run_dirs:
        for result_path in sorted(run_dir.glob("episodes/*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            episode_id = str(result["episode_id"])
            if episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            seen_ids.add(episode_id)
            observed.append((result_path, result))
    if not observed:
        raise ValueError("no episode result.json files found")
    return observed


def build_manifest(
    run_dirs: Iterable[Path],
    *,
    double_per_cell: int = 3,
    selection_salt: str = "mannercap-d0-gold-v1",
) -> dict[str, Any]:
    if double_per_cell < 0:
        raise ValueError("double_per_cell must be non-negative")

    observed = _load_results(run_dirs)
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_episode: dict[str, tuple[Path, dict[str, Any]]] = {}
    for result_path, result in observed:
        episode_id = str(result["episode_id"])
        task = str(result["task"])
        manner_tag = str(result["manner_tag"])
        cells[(task, manner_tag)].append(episode_id)
        by_episode[episode_id] = (result_path, result)

    double_ids: set[str] = set()
    cell_selection: list[dict[str, Any]] = []
    for (task, manner_tag), episode_ids in sorted(cells.items()):
        ranked = sorted(
            episode_ids,
            key=lambda item: hashlib.sha256(
                f"{selection_salt}\0{item}".encode("utf-8")
            ).hexdigest(),
        )
        selected = ranked[: min(double_per_cell, len(ranked))]
        double_ids.update(selected)
        cell_selection.append(
            {
                "task": task,
                "manner_tag": manner_tag,
                "candidate_count": len(episode_ids),
                "double_annotation_episode_ids": selected,
            }
        )

    records: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode):
        result_path, result = by_episode[episode_id]
        episode_dir = result_path.parent
        transitions_path = episode_dir / "transitions.jsonl"
        if not transitions_path.is_file():
            raise ValueError(f"missing transitions for {episode_id}: {transitions_path}")
        double_annotation = episode_id in double_ids
        annotation_id = "gold-" + hashlib.sha256(
            f"{selection_salt}\0annotation\0{episode_id}".encode("utf-8")
        ).hexdigest()[:16]
        records.append(
            {
                "annotation_id": annotation_id,
                "episode_id": episode_id,
                "task": result["task"],
                "seed": result["seed"],
                "target_instruction": result["raw_language"],
                "target_manner_tag": result["manner_tag"],
                "requested_manner": result["requested_manner"],
                "pair_id": result["pair_id"],
                "data_use": "development_calibration",
                "annotation_mode": "double_independent_then_adjudicate"
                if double_annotation
                else "single_independent",
                "source": {
                    "transitions_ref": str(transitions_path),
                    "transitions_sha256": _sha256(transitions_path),
                    "result_sha256_for_provenance_only": _sha256(result_path),
                },
                "assignment": {
                    "annotator_a": None,
                    "annotator_b": None if double_annotation else "not_required",
                    "adjudicator": None if double_annotation else "not_required",
                },
                "annotation_status": "pending",
            }
        )

    return {
        "schema": "mannercap-human-gold-candidate-manifest-v1",
        "status": "pending_human_annotation",
        "data_use": "development_calibration",
        "selection_salt": selection_salt,
        "candidate_count": len(records),
        "double_annotation_count": len(double_ids),
        "cell_count": len(cells),
        "blinding_contract": {
            "annotators_may_see": [
                "task",
                "target_instruction",
                "target_manner_tag",
                "requested_manner",
                "raw trajectory transitions and images",
            ],
            "executor_derived_fields_hidden": list(HIDDEN_EXECUTOR_FIELDS),
            "result_file_is_for_provenance_only": True,
            "labels_must_come_from_independent_human_review": True,
        },
        "required_labels": [
            "task_outcome",
            "manner_following",
            "phase_binding",
            "trajectory_data_quality",
        ],
        "cell_selection": cell_selection,
        "records": records,
        "formal_thresholds_frozen": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--double-per-cell", type=int, default=3)
    parser.add_argument("--selection-salt", default="mannercap-d0-gold-v1")
    args = parser.parse_args()

    manifest = build_manifest(
        args.run_dirs,
        double_per_cell=args.double_per_cell,
        selection_salt=args.selection_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": manifest["candidate_count"],
                "double_annotation_count": manifest["double_annotation_count"],
                "status": manifest["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
