"""Validate the bounded seed-42/seed-43 student development split.

This module intentionally performs provenance and data-contract checks only.  It
does not assign a formal split, choose a checkpoint, or infer task success.
The seed-43 staging manifest is expected to contain the pre-action rendering
provenance fields emitted by ``gate2_capture_la4vla_microset.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "mannercap-la4vla-microset-staging-v1"
REQUIRED_HORIZON = 50
STATE_DIM = 8
ACTION_DIM = 7


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_transitions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"transition {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def array_statistics(rows: Iterable[Iterable[float]], dim: int) -> dict[str, list[float]]:
    columns = [[] for _ in range(dim)]
    count = 0
    for row in rows:
        values = [float(value) for value in row]
        if len(values) != dim or not all(math.isfinite(value) for value in values):
            raise ValueError("statistics input contains a non-finite or wrong-width row")
        for index, value in enumerate(values):
            columns[index].append(value)
        count += 1
    if count == 0:
        raise ValueError("statistics input is empty")

    result: dict[str, list[float]] = {}
    result["mean"] = [sum(column) / count for column in columns]
    result["std"] = [
        math.sqrt(sum((value - mean) ** 2 for value in column) / count)
        for column, mean in zip(columns, result["mean"])
    ]
    result["min"] = [min(column) for column in columns]
    result["max"] = [max(column) for column in columns]
    result["q01"] = [_quantile(column, 0.01) for column in columns]
    result["q99"] = [_quantile(column, 0.99) for column in columns]
    return result


def _finite_vector(value: Any, width: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must be a {width}-element list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _source_metadata(
    window: dict[str, Any],
    *,
    expected_seed: int,
    require_render_provenance: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_dir = Path(str(window["source_episode"]))
    if not source_dir.is_dir():
        raise ValueError(f"source episode is missing: {source_dir}")
    result_path = source_dir / "result.json"
    initial_path = source_dir / "initial_state.json"
    model_path = source_dir / "model.mjb"
    transitions_path = source_dir / "transitions.jsonl"
    for path in (result_path, initial_path, model_path, transitions_path):
        if not path.is_file():
            raise ValueError(f"source artifact is missing: {path}")

    result = read_json(result_path)
    initial = read_json(initial_path)
    transitions = read_transitions(transitions_path)
    if int(result.get("seed")) != expected_seed:
        raise ValueError(f"seed mismatch for {source_dir}: {result.get('seed')}")
    if result.get("episode_id") != window.get("source_episode_id"):
        raise ValueError(f"episode id mismatch for {source_dir}")
    if result.get("task") != window.get("task"):
        raise ValueError(f"task mismatch for {source_dir}")
    if result.get("manner_tag") != window.get("manner_tag"):
        raise ValueError(f"manner mismatch for {source_dir}")
    if not bool(result.get("task_completed")):
        raise ValueError(f"source task is not completed: {source_dir}")

    expected_result_sha = str(window.get("source_result_sha256"))
    actual_result_sha = file_sha256(result_path)
    if actual_result_sha != expected_result_sha:
        raise ValueError(f"result hash mismatch for {source_dir}")
    initial_sha = str(initial.get("initial_state_sha256"))
    if initial_sha != str(window.get("source_initial_state_sha256")):
        raise ValueError(f"initial-state hash mismatch for {source_dir}")
    model_sha = file_sha256(model_path)
    if model_sha != str(window.get("source_compiled_model_sha256")):
        raise ValueError(f"compiled model hash mismatch for {source_dir}")
    if result.get("initial_state_sha256") != initial_sha:
        raise ValueError(f"result/initial initial-state hash mismatch for {source_dir}")
    if result.get("compiled_model_sha256") != model_sha:
        raise ValueError(f"result/model hash mismatch for {source_dir}")
    initial_compiled = initial.get("compiled_model", {}).get("sha256")
    if initial_compiled is not None and initial_compiled != model_sha:
        raise ValueError(f"initial/model hash mismatch for {source_dir}")

    start = int(window.get("source_start_transition_zero_based"))
    horizon = int(window.get("horizon"))
    if horizon != REQUIRED_HORIZON:
        raise ValueError(f"unexpected horizon={horizon} for {source_dir}")
    if start < 0 or start + horizon > len(transitions):
        raise ValueError(f"window bounds exceed transitions for {source_dir}: {start}")

    if require_render_provenance:
        if window.get("observation_render_before_action") is not True:
            raise ValueError(f"missing before-action render marker for {source_dir}")
        if int(window.get("observation_render_source_transition_zero_based")) != start:
            raise ValueError(f"render transition mismatch for {source_dir}")
        source_state = transitions[start]["state_t"]
        if window.get("observation_render_source_state_sha256") != canonical_sha256(source_state):
            raise ValueError(f"render state hash mismatch for {source_dir}")
        if float(window.get("observation_render_source_state_time_s")) != float(source_state["time_s"]):
            raise ValueError(f"render state time mismatch for {source_dir}")
        source_actions = [
            transitions[start + offset]["action_t"]
            for offset in range(horizon)
        ]
        if window.get("source_action_t_sequence_sha256") != canonical_sha256(source_actions):
            raise ValueError(f"source action sequence hash mismatch for {source_dir}")

    return source_dir, result, initial, transitions


def validate_manifest(
    path: Path,
    *,
    expected_seed: int,
    expected_count: int,
    require_render_provenance: bool,
    expected_episode_ids: set[str] | None = None,
) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {path}")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != expected_count:
        raise ValueError(f"window count mismatch in {path}: {len(windows or [])}")
    if int(manifest.get("sample_count")) != expected_count:
        raise ValueError(f"sample_count mismatch in {path}")
    if int(manifest.get("horizon")) != REQUIRED_HORIZON:
        raise ValueError(f"manifest horizon mismatch in {path}")

    source_ids: set[str] = set()
    lineage_keys: set[tuple[str, int]] = set()
    initial_groups: dict[tuple[str, str], set[str]] = {}
    result_hashes: set[str] = set()
    source_hashes: set[str] = set()
    states: list[list[float]] = []
    actions: list[list[float]] = []
    source_cache: dict[str, tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    image_count = 0
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"window {index} is not an object")
        source_id = str(window.get("source_episode_id"))
        task = str(window.get("task"))
        manner = str(window.get("manner_tag"))
        if not source_id or task not in {"lift", "stack"} or not manner:
            raise ValueError(f"window {index} has incomplete source identity")
        source_ids.add(source_id)
        start = int(window.get("source_start_transition_zero_based"))
        lineage = (source_id, start)
        if lineage in lineage_keys:
            raise ValueError(f"duplicate source window lineage: {lineage}")
        lineage_keys.add(lineage)

        source_dir = Path(str(window["source_episode"]))
        if source_id not in source_cache:
            source_cache[source_id] = _source_metadata(
                window,
                expected_seed=expected_seed,
                require_render_provenance=require_render_provenance,
            )
        else:
            cached_dir, cached_result, cached_initial, _ = source_cache[source_id]
            if cached_dir != source_dir or cached_result.get("initial_state_sha256") != window.get("source_initial_state_sha256"):
                raise ValueError(f"source identity changed within windows: {source_id}")

        initial_hash = str(window.get("source_initial_state_sha256"))
        initial_groups.setdefault((task, initial_hash), set()).add(source_id)
        result_hashes.add(str(window.get("source_result_sha256")))
        source_hashes.add(initial_hash)

        rows = window.get("rows")
        if not isinstance(rows, list) or len(rows) != REQUIRED_HORIZON:
            raise ValueError(f"window {index} row count mismatch")
        for row in rows:
            states.append(_finite_vector(row.get("observation_state"), STATE_DIM, "observation_state"))
            actions.append(_finite_vector(row.get("action"), ACTION_DIM, "action"))

        images = window.get("images", {})
        image_hashes = window.get("image_sha256", {})
        if require_render_provenance:
            for key in ("image", "wrist_image"):
                image_path = path.parent / str(images[key])
                if not image_path.is_file():
                    raise ValueError(f"missing captured image: {image_path}")
                if file_sha256(image_path) != str(image_hashes[key]):
                    raise ValueError(f"captured image hash mismatch: {image_path}")
                image_count += 1

    if expected_episode_ids is not None and source_ids != expected_episode_ids:
        raise ValueError(
            f"source episode set mismatch in {path}: "
            f"missing={sorted(expected_episode_ids - source_ids)}, "
            f"extra={sorted(source_ids - expected_episode_ids)}"
        )
    if len(source_ids) != int(manifest.get("independent_source_episode_count")):
        raise ValueError(f"independent source count mismatch in {path}")

    return {
        "manifest": str(path),
        "seed": expected_seed,
        "windows": len(windows),
        "frames": len(states),
        "source_episode_count": len(source_ids),
        "source_episode_ids": sorted(source_ids),
        "initial_state_group_count": len(initial_groups),
        "initial_state_groups": {
            f"{task}|{initial_hash}": sorted(ids)
            for (task, initial_hash), ids in sorted(initial_groups.items())
        },
        "source_initial_state_hashes": sorted(source_hashes),
        "source_result_hash_count": len(result_hashes),
        "lineage_count": len(lineage_keys),
        "captured_image_count": image_count,
        "effective_supervised_frames": len(actions),
        "effective_supervised_scalar_targets": len(actions) * ACTION_DIM,
        "descriptive_stats": {
            "action": array_statistics(actions, ACTION_DIM),
            "observation.state": array_statistics(states, STATE_DIM),
        },
    }


def build_report(
    *,
    plan_sha256: str,
    train_summary: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    train_keys = {
        key.split("|", 1)[0] + "|" + key.split("|", 1)[1]
        for key in train_summary["initial_state_groups"]
    }
    validation_keys = set(validation_summary["initial_state_groups"])
    train_sources = set(train_summary["source_episode_ids"])
    validation_sources = set(validation_summary["source_episode_ids"])
    return {
        "schema": "mannercap-la4vla-student-dev-validation-qc.v1",
        "plan_sha256": plan_sha256,
        "status": "pass",
        "claim_boundary": "development_only_same_task_seed_holdout",
        "train": train_summary,
        "validation": validation_summary,
        "cross_split": {
            "source_episode_intersection": sorted(train_sources & validation_sources),
            "task_initial_state_group_intersection": sorted(train_keys & validation_keys),
            "passed": not (train_sources & validation_sources or train_keys & validation_keys),
        },
        "same_seed_manner_note": "seed42 has eight source episodes but four manners share one initial-state group per task; independent group count is two",
        "formal_split_assignment": "development",
        "formal_thresholds_frozen": False,
        "promotion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-stats-output", type=Path, required=True)
    args = parser.parse_args()

    expected_val_ids = {
        "d0-lift-seed-43-neutral",
        "d0-lift-seed-43-slow",
        "d0-lift-seed-43-fast",
        "d0-lift-seed-43-fast_approach_slow_contact",
        "d0-stack-seed-43-neutral",
        "d0-stack-seed-43-slow",
        "d0-stack-seed-43-fast",
        "d0-stack-seed-43-fast_approach_slow_contact",
    }
    expected_train_ids = {
        "d0-lift-seed-42-neutral",
        "d0-lift-seed-42-slow",
        "d0-lift-seed-42-fast",
        "d0-lift-seed-42-fast_approach_slow_contact",
        "d0-stack-seed-42-neutral",
        "d0-stack-seed-42-slow",
        "d0-stack-seed-42-fast",
        "d0-stack-seed-42-fast_approach_slow_contact",
    }
    train_summary = validate_manifest(
        args.train_manifest.resolve(),
        expected_seed=42,
        expected_count=64,
        require_render_provenance=False,
        expected_episode_ids=expected_train_ids,
    )
    validation_summary = validate_manifest(
        args.validation_manifest.resolve(),
        expected_seed=43,
        expected_count=64,
        require_render_provenance=True,
        expected_episode_ids=expected_val_ids,
    )
    report = build_report(
        plan_sha256=args.plan_sha256,
        train_summary=train_summary,
        validation_summary=validation_summary,
    )
    if not report["cross_split"]["passed"]:
        report["status"] = "fail"
    args.train_stats_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.train_stats_output.resolve().write_text(
        json.dumps(train_summary["descriptive_stats"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
