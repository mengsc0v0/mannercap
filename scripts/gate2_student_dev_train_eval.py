"""Run one bounded action-head student development train/eval.

The runner consumes two already-QC'd 64-window datasets.  It uses the native
LA4VLA loader with the explicit ``max_samples_per_file=50`` guard, caches the
frozen VLM embeddings, trains only the action head for a fixed number of
optimizer steps, writes one final checkpoint, and evaluates real flow-matching
ODE actions in the exported seven action dimensions.  It never runs a
simulator or an external API.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Any, Callable


ACTION_DIM = 7
HORIZON = 50
MODEL_ACTION_DIM = 24
LOADER_MAX_SAMPLES_PER_FILE = 50
DISK_CAP_GIB = 15
DISK_CAP_BYTES = DISK_CAP_GIB * 1024**3
VALIDATION_PASSES = 2
GPU_MEMORY_CAP_GIB = 6


def file_sha256(path: Path, *, progress_callback: Callable[[], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for index, chunk in enumerate(iter(lambda: stream.read(1024 * 1024), b"")):
            digest.update(chunk)
            if progress_callback is not None and index % 64 == 0:
                progress_callback()
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    """Return the size of one tracked output path without following symlinks."""
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _tracked_output_paths(args: argparse.Namespace) -> list[Path]:
    # Only paths owned by this attempt are tracked.  Source datasets, model
    # weights, reports and the external LA4VLA tree are read-only inputs.
    return [
        args.train_cache.resolve(),
        args.validation_cache.resolve(),
        args.checkpoint_output.resolve(),
        args.output.resolve(),
    ]


def _new_progress(args: argparse.Namespace) -> dict[str, Any]:
    paths = _tracked_output_paths(args)
    before = sum(_path_size(path) for path in paths)
    return {
        "phase": "startup",
        "phase_history": [{"phase": "startup", "elapsed_seconds": 0.0}],
        "completed_steps": 0,
        "steps_requested": int(args.steps),
        "elapsed_seconds": 0.0,
        "resource_paths": [str(path) for path in paths],
        "resource_before_bytes": before,
        "resource_after_bytes": before,
        "new_disk_bytes": 0,
        "disk_cap_bytes": DISK_CAP_BYTES,
        "checkpoint_written": False,
        "checkpoint_sha256": None,
        "_started_monotonic": time.perf_counter(),
        "_tracked_paths": paths,
    }


def _public_progress(progress: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in progress.items() if not key.startswith("_")}


def _refresh_progress(progress: dict[str, Any]) -> None:
    elapsed = time.perf_counter() - float(progress["_started_monotonic"])
    after = sum(_path_size(path) for path in progress["_tracked_paths"])
    before = int(progress["resource_before_bytes"])
    progress["elapsed_seconds"] = elapsed
    progress["resource_after_bytes"] = after
    progress["new_disk_bytes"] = max(0, after - before)


def _set_phase(progress: dict[str, Any], phase: str) -> None:
    if progress.get("phase") == phase:
        return
    progress["phase"] = phase
    _refresh_progress(progress)
    progress.setdefault("phase_history", []).append(
        {"phase": phase, "elapsed_seconds": progress["elapsed_seconds"]}
    )


def _check_budget(
    progress: dict[str, Any], args: argparse.Namespace, *, phase: str | None = None
) -> None:
    if phase is not None:
        _set_phase(progress, phase)
    _refresh_progress(progress)
    if progress["elapsed_seconds"] > float(args.max_wall_seconds):
        raise TimeoutError(
            f"wall-clock budget exceeded during {progress['phase']} "
            f"({progress['elapsed_seconds']:.3f}s > {args.max_wall_seconds:.3f}s)"
        )
    if progress["new_disk_bytes"] > DISK_CAP_BYTES:
        raise RuntimeError(
            f"new disk budget exceeded during {progress['phase']} "
            f"({progress['new_disk_bytes']} > {DISK_CAP_BYTES} bytes)"
        )


def apply_train_only_stats(dataset: Any, train_stats: dict[str, Any]) -> None:
    """Force every loaded data group to use the supplied train-only stats."""
    groups = getattr(dataset, "arm2stats_dict", None)
    if not isinstance(groups, dict) or not groups:
        raise RuntimeError("dataset does not expose arm2stats_dict for stats override")
    for datasets in groups.values():
        if not isinstance(datasets, dict) or not datasets:
            raise RuntimeError("dataset has no normalization data groups")
        for dataset_name in list(datasets):
            datasets[dataset_name] = copy.deepcopy(train_stats)


def validate_action_mask(cache: dict[str, Any]) -> None:
    """Require the exported seven dimensions to be the only active mask entries."""
    mask = cache["action_mask"]
    expected = (cache["sample_count"], HORIZON, MODEL_ACTION_DIM)
    if tuple(mask.shape) != expected:
        raise RuntimeError(f"unexpected action mask shape: {tuple(mask.shape)} != {expected}")
    if not bool(mask[..., :ACTION_DIM].all().item()):
        raise RuntimeError("action mask has an inactive exported action dimension")
    if bool(mask[..., ACTION_DIM:].any().item()):
        raise RuntimeError("action mask activates padded model action dimensions")


def configure_gpu_memory_cap(torch_module: Any, args: argparse.Namespace, progress: dict[str, Any]) -> None:
    """Apply the attempt's per-process CUDA allocator cap before model loading."""
    if not hasattr(torch_module.cuda, "set_per_process_memory_fraction"):
        raise RuntimeError("CUDA per-process memory cap is unavailable")
    properties = torch_module.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    requested_bytes = int(float(args.max_gpu_memory_gib) * 1024**3)
    if requested_bytes <= 0 or requested_bytes >= total_bytes:
        raise ValueError(
            f"invalid GPU memory cap {requested_bytes} for device capacity {total_bytes}"
        )
    fraction = requested_bytes / total_bytes
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=0)
    progress["gpu_memory_cap_bytes"] = requested_bytes
    progress["gpu_total_memory_bytes"] = total_bytes
    progress["gpu_memory_cap_fraction"] = fraction


def masked_flow_loss(prediction, target, action_mask):
    mask = action_mask.reshape(action_mask.shape[0], -1).to(dtype=prediction.dtype)
    error = (prediction.float() - target.float()).pow(2) * mask.float()
    return error.sum() / mask.sum().clamp_min(1.0)


def load_stats(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        stats = json.load(stream)
    if set(stats) != {"action", "observation.state"}:
        raise ValueError(f"unexpected train-only stats shape: {path}")
    for section, dimension in (("action", ACTION_DIM), ("observation.state", 8)):
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            values = stats[section].get(key)
            if not isinstance(values, list) or len(values) != dimension:
                raise ValueError(f"stats {section}.{key} has wrong width")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"stats {section}.{key} contains non-finite values")
    return stats


class DatasetBindingError(RuntimeError):
    """Raised when a configured dataset does not match the QC evidence."""

    def __init__(self, details: dict[str, Any]):
        self.details = details
        mismatch_text = "; ".join(details.get("mismatches", []))
        super().__init__(
            f"dataset binding failed for {details.get('split')}: {mismatch_text}"
        )


def _config_dataset_root(config: dict[str, Any], config_path: Path) -> Path:
    """Resolve the one dataset root referenced by a native LA4VLA config."""
    roots: set[Path] = set()
    for group in (config.get("data_groups") or {}).values():
        if not isinstance(group, dict):
            continue
        for dataset_config in group.values():
            if isinstance(dataset_config, dict) and dataset_config.get("path"):
                roots.add(Path(str(dataset_config["path"])).expanduser().resolve())
    if len(roots) != 1:
        raise RuntimeError(
            f"dataset config must reference exactly one dataset root: {config_path}"
        )
    root = next(iter(roots))
    config_root = config_path.resolve().parent
    if root != config_root:
        raise RuntimeError(
            f"dataset config path/root mismatch: config={config_root} data_group={root}"
        )
    return root


def _source_episode_id(window: dict[str, Any]) -> str:
    explicit = window.get("source_episode_id")
    if explicit:
        return str(explicit)
    source_episode = str(window.get("source_episode", "")).rstrip("/")
    if not source_episode:
        return ""
    return Path(source_episode).name


def _episode_task_manner(source_episode_id: str) -> tuple[str, str | None]:
    """Parse the frozen d0 source id without inferring missing manners."""
    parts = source_episode_id.split("-")
    try:
        seed_index = parts.index("seed")
    except ValueError:
        return source_episode_id, None
    if len(parts) <= seed_index + 2:
        return "-".join(parts[1:seed_index]) or source_episode_id, None
    task = "-".join(parts[1:seed_index]) or source_episode_id
    return task, "-".join(parts[seed_index + 2 :])


def _json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_qc_binding(
    *, qc_report_path: Path, expected_qc_sha256: str, plan_sha256: str
) -> str:
    """Require the exact frozen QC report and its frozen-plan reference."""
    actual_sha256 = file_sha256(qc_report_path)
    if actual_sha256 != expected_qc_sha256:
        raise RuntimeError(
            "QC report SHA-256 mismatch: "
            f"actual={actual_sha256} expected={expected_qc_sha256}"
        )
    qc = _json_file(qc_report_path)
    if qc.get("plan_sha256") != plan_sha256:
        raise RuntimeError(
            "QC report plan SHA-256 mismatch: "
            f"actual={qc.get('plan_sha256')} expected={plan_sha256}"
        )
    return actual_sha256


def validate_dataset_binding(
    *,
    config_path: Path,
    qc_report_path: Path,
    split: str,
    yaml_module: Any,
) -> dict[str, Any]:
    """Bind the loaded config to the exact source set certified by QC.

    Window counts alone are insufficient: a synthetic two-source bundle can
    also contain 64 windows.  The binding therefore compares the resolved
    dataset root, manifest hash, source ids, task/manner pairs, initial-state
    hashes, and source-result hash count against the split's QC record.
    """
    with config_path.open(encoding="utf-8") as stream:
        config = yaml_module.safe_load(stream)
    if not isinstance(config, dict):
        raise RuntimeError(f"dataset config is not a mapping: {config_path}")
    root = _config_dataset_root(config, config_path)
    manifest_path = root / "source_staging_manifest.json"
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    for required in (manifest_path, info_path, tasks_path):
        if not required.is_file():
            raise RuntimeError(f"dataset binding input is missing: {required}")
    manifest = _json_file(manifest_path)
    info = _json_file(info_path)
    with tasks_path.open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    windows = manifest.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError(f"dataset manifest windows must be a list: {manifest_path}")

    qc = _json_file(qc_report_path)
    expected_qc = qc.get(split)
    if not isinstance(expected_qc, dict):
        raise RuntimeError(f"QC report has no {split} record: {qc_report_path}")
    expected_manifest_path = Path(str(expected_qc.get("manifest", "")))
    if not expected_manifest_path.is_file():
        raise RuntimeError(
            f"QC-referenced {split} manifest is missing: {expected_manifest_path}"
        )
    expected_manifest = _json_file(expected_manifest_path)

    actual_ids = sorted({_source_episode_id(window) for window in windows})
    actual_source_window_counts: dict[str, int] = {}
    for window in windows:
        source_id = _source_episode_id(window)
        actual_source_window_counts[source_id] = actual_source_window_counts.get(source_id, 0) + 1
    expected_source_window_counts: dict[str, int] = {}
    for window in expected_manifest.get("windows", []):
        source_id = _source_episode_id(window)
        expected_source_window_counts[source_id] = expected_source_window_counts.get(source_id, 0) + 1
    actual_task_manner = sorted(
        {
            (
                str(window.get("task", "")),
                str(window.get("manner_tag"))
                if window.get("manner_tag") is not None
                else _episode_task_manner(_source_episode_id(window))[1],
            )
            for window in windows
        }
    )
    expected_ids = sorted(str(value) for value in expected_qc.get("source_episode_ids", []))
    expected_task_manner = sorted(_episode_task_manner(value) for value in expected_ids)
    actual_initial_states = sorted(
        {
            str(window["source_initial_state_sha256"])
            for window in windows
            if window.get("source_initial_state_sha256")
        }
    )
    expected_initial_states = sorted(
        str(value) for value in expected_qc.get("source_initial_state_hashes", [])
    )
    actual_result_hash_count = len(
        {
            str(window["source_result_sha256"])
            for window in windows
            if window.get("source_result_sha256")
        }
    )
    expected_result_hash_count = int(expected_qc.get("source_result_hash_count", 0))
    actual_manifest_sha256 = file_sha256(manifest_path)
    expected_manifest_sha256 = file_sha256(expected_manifest_path)
    expected_windows = int(expected_qc.get("windows", -1))
    expected_source_count = int(expected_qc.get("source_episode_count", -1))
    actual_source_count = len(actual_ids)
    mismatches: list[str] = []
    if actual_manifest_sha256 != expected_manifest_sha256:
        mismatches.append(
            f"manifest_sha256 actual={actual_manifest_sha256} expected={expected_manifest_sha256}"
        )
    if actual_ids != expected_ids:
        mismatches.append(f"source_episode_ids actual={actual_ids} expected={expected_ids}")
    if actual_source_window_counts != expected_source_window_counts:
        mismatches.append(
            "source_window_counts "
            f"actual={actual_source_window_counts} expected={expected_source_window_counts}"
        )
    if actual_source_count != expected_source_count:
        mismatches.append(
            f"source_episode_count actual={actual_source_count} expected={expected_source_count}"
        )
    if len(windows) != expected_windows:
        mismatches.append(f"windows actual={len(windows)} expected={expected_windows}")
    if int(manifest.get("sample_count", -1)) != expected_windows:
        mismatches.append(
            f"manifest.sample_count actual={manifest.get('sample_count')} expected={expected_windows}"
        )
    if int(manifest.get("horizon", -1)) != HORIZON:
        mismatches.append(f"manifest.horizon actual={manifest.get('horizon')} expected={HORIZON}")
    if actual_task_manner != expected_task_manner:
        mismatches.append(
            f"task_manner_pairs actual={actual_task_manner} expected={expected_task_manner}"
        )
    if actual_initial_states != expected_initial_states:
        mismatches.append(
            f"source_initial_state_hashes actual={actual_initial_states} expected={expected_initial_states}"
        )
    if actual_result_hash_count != expected_result_hash_count:
        mismatches.append(
            f"source_result_hash_count actual={actual_result_hash_count} expected={expected_result_hash_count}"
        )
    if int(info.get("independent_source_episode_count", -1)) != expected_source_count:
        mismatches.append(
            "info.independent_source_episode_count "
            f"actual={info.get('independent_source_episode_count')} expected={expected_source_count}"
        )
    binding = {
        "schema": "mannercap-la4vla-dataset-binding.v1",
        "status": "pass" if not mismatches else "fail",
        "split": split,
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "dataset_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "qc_manifest": str(expected_manifest_path),
        "qc_manifest_sha256": expected_manifest_sha256,
        "info_sha256": file_sha256(info_path),
        "tasks_sha256": file_sha256(tasks_path),
        "normalization_type": config.get("normalization_type"),
        "sample_count": len(windows),
        "horizon": manifest.get("horizon"),
        "source_episode_ids": actual_ids,
        "source_episode_count": actual_source_count,
        "source_window_counts": actual_source_window_counts,
        "task_manner_pairs": [list(pair) for pair in actual_task_manner],
        "source_initial_state_hashes": actual_initial_states,
        "source_result_hash_count": actual_result_hash_count,
        "info": {
            "independent_source_episode_count": info.get("independent_source_episode_count"),
            "total_episodes": info.get("total_episodes"),
            "total_tasks": info.get("total_tasks"),
            "source_manner_tags": info.get("source_manner_tags", []),
        },
        "tasks": tasks,
        "expected_qc": {
            "manifest": str(expected_manifest_path),
            "seed": expected_qc.get("seed"),
            "source_episode_ids": expected_ids,
            "source_episode_count": expected_source_count,
            "source_window_counts": expected_source_window_counts,
            "windows": expected_windows,
            "source_initial_state_hashes": expected_initial_states,
            "source_result_hash_count": expected_result_hash_count,
            "task_manner_pairs": [list(pair) for pair in expected_task_manner],
        },
        "mismatches": mismatches,
    }
    if mismatches:
        raise DatasetBindingError(binding)
    return binding


def import_laur(root: Path):
    import sys

    sys.path.insert(0, str(root / "LA4VLA_1B"))
    import torch
    import yaml
    from config import LA4VLAConfig
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    from scripts.LA4VLA_1B import LA4VLA_1B

    return torch, yaml, LA4VLAConfig, LeRobotDataset, LA4VLA_1B


def load_dataset(
    *,
    config_path: Path,
    cache_dir: Path,
    yaml_module: Any,
    dataset_class: Any,
):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml_module.safe_load(stream)
    if config.get("normalization_type") != "bounds":
        raise RuntimeError(
            f"student development runner requires bounds normalization: {config_path}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return dataset_class(
        config=config,
        image_size=448,
        max_samples_per_file=LOADER_MAX_SAMPLES_PER_FILE,
        action_horizon=HORIZON,
        video_backend="av",
        cache_dir=cache_dir,
        use_augmentation=False,
    )


def cache_embeddings(
    model,
    dataset,
    *,
    batch_size: int,
    device,
    torch_module,
    budget_callback: Callable[[], None] | None = None,
):
    fused_chunks = []
    state_chunks = []
    action_chunks = []
    mask_chunks = []
    prompt_count = 0
    for start in range(0, len(dataset), batch_size):
        if budget_callback is not None:
            budget_callback()
        items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = [item["images"].to(device) for item in items]
        image_mask = torch_module.stack([item["image_mask"] for item in items]).to(device)
        prompts = [item["prompt"] for item in items]
        states = torch_module.stack([item["state"] for item in items]).float()
        actions = torch_module.stack([item["action"] for item in items]).float()
        action_mask = torch_module.stack([item["action_mask"] for item in items])
        with torch_module.inference_mode():
            fused = model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=prompts,
                return_cls_only=False,
            ).detach().cpu()
        fused_chunks.append(fused)
        state_chunks.append(states)
        action_chunks.append(actions)
        mask_chunks.append(action_mask)
        prompt_count += len(items)
        del items, images, image_mask, prompts, states, actions, action_mask, fused
        if budget_callback is not None:
            budget_callback()
    return {
        "fused": torch_module.cat(fused_chunks, dim=0),
        "state": torch_module.cat(state_chunks, dim=0),
        "action": torch_module.cat(action_chunks, dim=0),
        "action_mask": torch_module.cat(mask_chunks, dim=0),
        "sample_count": prompt_count,
    }


def _batch(cache: dict[str, Any], start: int, end: int, *, device, torch_module):
    return (
        cache["fused"][start:end].to(device),
        cache["state"][start:end].to(device),
        cache["action"][start:end].to(device),
        cache["action_mask"][start:end].to(device),
    )


def fixed_flow_loss(
    model,
    cache: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
    device,
    torch_module,
    budget_callback: Callable[[], None] | None = None,
) -> float:
    model.eval()
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    total = 0.0
    count = 0
    with torch_module.inference_mode():
        for start in range(0, cache["sample_count"], batch_size):
            if budget_callback is not None:
                budget_callback()
            end = min(start + batch_size, cache["sample_count"])
            fused, state, actions, action_mask = _batch(
                cache, start, end, device=device, torch_module=torch_module
            )
            prediction, noise = model(
                fused, state=state, actions_gt=actions, action_mask=action_mask
            )
            target = (actions - noise).reshape(actions.shape[0], -1)
            loss = masked_flow_loss(prediction, target, action_mask)
            active = int(action_mask.sum().item())
            total += float(loss.item()) * active
            count += active
    return total / max(count, 1)


def _denormalize(values, stats: dict[str, Any], *, torch_module, device):
    low = torch_module.tensor(stats["action"]["min"], dtype=torch_module.float32, device=device)
    high = torch_module.tensor(stats["action"]["max"], dtype=torch_module.float32, device=device)
    return (values + 1.0) / 2.0 * (high - low + 1e-8) + low


def ode_metrics(
    model,
    cache: dict[str, Any],
    stats: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
    device,
    torch_module,
    budget_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the real model.get_action ODE in exported seven-dimensional units."""
    model.eval()
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    abs_sum = torch_module.zeros(ACTION_DIM, dtype=torch_module.float64)
    sq_sum = torch_module.zeros(ACTION_DIM, dtype=torch_module.float64)
    count = 0
    with torch_module.inference_mode():
        for start in range(0, cache["sample_count"], batch_size):
            if budget_callback is not None:
                budget_callback()
            end = min(start + batch_size, cache["sample_count"])
            fused, state, actions, action_mask = _batch(
                cache, start, end, device=device, torch_module=torch_module
            )
            prediction = model.predict_action(
                fused, state=state, action_mask=action_mask
            )
            prediction = prediction.reshape(prediction.shape[0], HORIZON, MODEL_ACTION_DIM)
            target = actions.reshape(actions.shape[0], HORIZON, MODEL_ACTION_DIM)
            prediction_7 = _denormalize(prediction[..., :ACTION_DIM], stats, torch_module=torch_module, device=device)
            target_7 = _denormalize(target[..., :ACTION_DIM], stats, torch_module=torch_module, device=device)
            error = (prediction_7 - target_7).double().reshape(-1, ACTION_DIM).cpu()
            abs_sum += error.abs().sum(dim=0)
            sq_sum += error.pow(2).sum(dim=0)
            count += error.shape[0]
    mae = abs_sum / max(count, 1)
    mse = sq_sum / max(count, 1)
    return {
        "seed": seed,
        "samples": cache["sample_count"],
        "frames": count,
        "active_action_dimensions": ACTION_DIM,
        "units": "exported_7d_normalized_action_units_after_train_stats_denormalization",
        "inference_method": "model.predict_action(actions_gt=None)->action_head.get_action",
        "per_dimension_mae": mae.tolist(),
        "per_dimension_mse": mse.tolist(),
        "overall_mae": float(mae.mean().item()),
        "overall_mse": float(mse.mean().item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model,
    config,
    source_checkpoint_sha256: str,
    train_stats: dict[str, Any],
    train_stats_sha256: str,
    plan_sha256: str,
    qc_sha256: str,
    loader_smoke_sha256: str,
    steps: int,
    seed: int,
    lr: float,
    torch_module,
    progress_callback: Callable[[], None] | None = None,
    attempt_plan_sha256: str | None = None,
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mannercap-la4vla-student-dev-checkpoint.v1",
        "step": steps,
        "seed": seed,
        "learning_rate": lr,
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_only_stats": train_stats,
        "train_only_stats_sha256": train_stats_sha256,
        "plan_sha256": plan_sha256,
        "attempt_plan_sha256": attempt_plan_sha256,
        "qc_sha256": qc_sha256,
        "loader_smoke_sha256": loader_smoke_sha256,
        "closed_loop_claim_allowed": False,
        "generalization_claim_allowed": False,
    }
    torch_module.save(payload, path)
    return file_sha256(path, progress_callback=progress_callback)


def run(
    args: argparse.Namespace, progress: dict[str, Any] | None = None
) -> dict[str, Any]:
    progress = progress or _new_progress(args)

    def enter(phase: str) -> None:
        _check_budget(progress, args, phase=phase)

    def check() -> None:
        _check_budget(progress, args)

    enter("imports")
    root = args.la4vla_root.resolve()
    torch, yaml, config_class, dataset_class, model_class = import_laur(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    configure_gpu_memory_cap(torch, args, progress)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    enter("input_hashes")
    source_checkpoint_sha256 = file_sha256(args.source_checkpoint.resolve(), progress_callback=check)
    train_stats_sha256 = file_sha256(args.train_stats.resolve())
    qc_sha256 = file_sha256(args.qc_report.resolve())
    loader_smoke_sha256 = file_sha256(args.loader_smoke_report.resolve())
    validate_qc_binding(
        qc_report_path=args.qc_report.resolve(),
        expected_qc_sha256=args.expected_qc_sha256,
        plan_sha256=args.plan_sha256,
    )
    train_stats = load_stats(args.train_stats.resolve())
    validation_dataset_root = args.validation_dataset_config.resolve().parent
    validation_stats_path = validation_dataset_root / "meta" / "stats.json"
    if file_sha256(validation_stats_path) != train_stats_sha256:
        raise RuntimeError("validation normalization stats do not match train-only stats")
    check()

    enter("dataset_loader")
    train_binding = validate_dataset_binding(
        config_path=args.train_dataset_config.resolve(),
        qc_report_path=args.qc_report.resolve(),
        split="train",
        yaml_module=yaml,
    )
    validation_binding = validate_dataset_binding(
        config_path=args.validation_dataset_config.resolve(),
        qc_report_path=args.qc_report.resolve(),
        split="validation",
        yaml_module=yaml,
    )
    check()
    train_dataset = load_dataset(
        config_path=args.train_dataset_config.resolve(),
        cache_dir=args.train_cache.resolve(),
        yaml_module=yaml,
        dataset_class=dataset_class,
    )
    apply_train_only_stats(train_dataset, train_stats)
    validation_dataset = load_dataset(
        config_path=args.validation_dataset_config.resolve(),
        cache_dir=args.validation_cache.resolve(),
        yaml_module=yaml,
        dataset_class=dataset_class,
    )
    apply_train_only_stats(validation_dataset, train_stats)
    if len(train_dataset) != 64 or len(validation_dataset) != 64:
        raise RuntimeError(f"unexpected loader length train={len(train_dataset)} val={len(validation_dataset)}")
    check()

    enter("model_load")
    payload = torch.load(args.source_checkpoint.resolve(), map_location="cpu")
    config_dict = dict(payload["config"])
    config_dict.update(
        {
            "device": "cuda",
            "vlm_name": str(root / "InternVL3-1B"),
            "finetune_vlm": False,
            "finetune_language_model": False,
            "finetune_text_embedder": False,
            "finetune_vision_model": False,
            "finetune_action_head": True,
            "use_augmentation": False,
            "sample_weights_path": None,
            "loss_weights_path": None,
            "num_workers": 0,
        }
    )
    config = config_class.from_dict(config_dict)
    model = model_class(config)
    model.set_finetune_flags()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.to(device)
    model.embedder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.embedder.parameters()):
        raise RuntimeError("action-head-only freeze contract failed")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    check()

    enter("embedding_cache")
    train_cache = cache_embeddings(
        model,
        train_dataset,
        batch_size=args.batch_size,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    validation_cache = cache_embeddings(
        model,
        validation_dataset,
        batch_size=args.batch_size,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    if train_cache["sample_count"] != 64 or validation_cache["sample_count"] != 64:
        raise RuntimeError("embedding cache sample count mismatch")
    if train_cache["action_mask"].sum().item() != 64 * HORIZON * ACTION_DIM:
        raise RuntimeError("train action mask active target count mismatch")
    if validation_cache["action_mask"].sum().item() != 64 * HORIZON * ACTION_DIM:
        raise RuntimeError("validation action mask active target count mismatch")
    validate_action_mask(train_cache)
    validate_action_mask(validation_cache)
    check()

    enter("initial_comparison")
    fixed_seed = args.seed + 100
    initial_fixed_train = fixed_flow_loss(
        model,
        train_cache,
        batch_size=args.batch_size,
        seed=fixed_seed,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    initial_fixed_validation = fixed_flow_loss(
        model,
        validation_cache,
        batch_size=args.batch_size,
        seed=fixed_seed,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    initial_train_ode = ode_metrics(
        model,
        train_cache,
        train_stats,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    initial_validation_ode = ode_metrics(
        model,
        validation_cache,
        train_stats,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    check()

    enter("fixed_training")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.eval()
    model.action_head.train()
    optimizer.zero_grad(set_to_none=True)
    train_samples_seen = 0
    train_frames_seen = 0
    train_scalar_targets_seen = 0
    losses: list[float] = []
    peak_allocated = 0
    for step in range(args.steps):
        check()
        start = (step * args.batch_size) % train_cache["sample_count"]
        indices = [(start + offset) % train_cache["sample_count"] for offset in range(args.batch_size)]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        fused = train_cache["fused"][index_tensor].to(device)
        state = train_cache["state"][index_tensor].to(device)
        actions = train_cache["action"][index_tensor].to(device)
        action_mask = train_cache["action_mask"][index_tensor].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction, noise = model(
            fused, state=state, actions_gt=actions, action_mask=action_mask
        )
        target = (actions - noise).reshape(actions.shape[0], -1)
        loss = masked_flow_loss(prediction, target, action_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        progress["completed_steps"] = step + 1
        losses.append(float(loss.item()))
        train_samples_seen += len(indices)
        train_frames_seen += int(action_mask.shape[0] * action_mask.shape[1])
        train_scalar_targets_seen += int(action_mask.sum().item())
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
        del fused, state, actions, action_mask, prediction, noise, target, loss
    check()

    enter("checkpoint_save")
    checkpoint_path = args.checkpoint_output.resolve()
    checkpoint_hash = save_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        source_checkpoint_sha256=source_checkpoint_sha256,
        train_stats=train_stats,
        train_stats_sha256=train_stats_sha256,
        plan_sha256=args.plan_sha256,
        qc_sha256=qc_sha256,
        loader_smoke_sha256=loader_smoke_sha256,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        torch_module=torch,
        progress_callback=check,
        attempt_plan_sha256=getattr(args, "attempt_plan_sha256", None),
    )
    progress["checkpoint_written"] = True
    progress["checkpoint_sha256"] = checkpoint_hash
    check()

    enter("final_comparison")
    final_fixed_train = fixed_flow_loss(
        model,
        train_cache,
        batch_size=args.batch_size,
        seed=fixed_seed,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    final_fixed_validation = fixed_flow_loss(
        model,
        validation_cache,
        batch_size=args.batch_size,
        seed=fixed_seed,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    final_train_ode = ode_metrics(
        model,
        train_cache,
        train_stats,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    final_validation_ode = ode_metrics(
        model,
        validation_cache,
        train_stats,
        batch_size=args.batch_size,
        seed=args.seed + 200,
        device=device,
        torch_module=torch,
        budget_callback=check,
    )
    check()

    _refresh_progress(progress)

    return {
        "schema": "mannercap-la4vla-student-dev-training.v1",
        "status": "pass",
        "claim_boundary": "development_only_same_task_seed_holdout",
        "formal_split_assignment": "development",
        "promotion_allowed": False,
        "closed_loop_claim_allowed": False,
        "generalization_claim_allowed": False,
        "plan_sha256": args.plan_sha256,
        "attempt_plan_sha256": getattr(args, "attempt_plan_sha256", None),
        "source_checkpoint": {
            "path": str(args.source_checkpoint.resolve()),
            "sha256": source_checkpoint_sha256,
        },
        "train_dataset": {
            "config": str(args.train_dataset_config.resolve()),
            "cache": str(args.train_cache.resolve()),
            "windows": train_cache["sample_count"],
            "source_seed": train_binding["expected_qc"].get("seed"),
            "binding": train_binding,
        },
        "validation_dataset": {
            "config": str(args.validation_dataset_config.resolve()),
            "cache": str(args.validation_cache.resolve()),
            "windows": validation_cache["sample_count"],
            "source_seed": validation_binding["expected_qc"].get("seed"),
            "normalization_source": "train_only",
            "normalization_sha256": train_stats_sha256,
            "binding": validation_binding,
        },
        "loader_contract": {
            "max_samples_per_file": LOADER_MAX_SAMPLES_PER_FILE,
            "effective_train_windows": train_cache["sample_count"],
            "effective_validation_windows": validation_cache["sample_count"],
            "action_horizon": HORIZON,
            "active_action_dimensions": ACTION_DIM,
            "model_action_dimensions": MODEL_ACTION_DIM,
            "normalization_type": "bounds",
        },
        "training": {
            "steps": args.steps,
            "seed": args.seed,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "samples_seen": train_samples_seen,
            "frames_seen": train_frames_seen,
            "scalar_action_targets_seen": train_scalar_targets_seen,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_minimum": min(losses),
            "all_losses_finite": True,
            "peak_gpu_allocated_bytes": peak_allocated,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "count": 1,
            "selection": "final_step_only",
        },
        "fixed_noise_flow_loss": {
            "seed": fixed_seed,
            "initial_train": initial_fixed_train,
            "final_train": final_fixed_train,
            "initial_validation": initial_fixed_validation,
            "final_validation": final_fixed_validation,
        },
        "initial_ode": {
            "train": initial_train_ode,
            "validation": initial_validation_ode,
        },
        "final_ode": {
            "train": final_train_ode,
            "validation": final_validation_ode,
            "validation_passes": VALIDATION_PASSES,
            "validation_noise_seed": args.seed + 200,
        },
        "validation_sampling": {
            "passes": VALIDATION_PASSES,
            "initial_noise_seed": args.seed + 200,
            "final_noise_seed": args.seed + 200,
            "same_noise_seed": True,
        },
        "artifacts": {
            "train_stats_sha256": train_stats_sha256,
            "qc_report_sha256": qc_sha256,
            "loader_smoke_report_sha256": loader_smoke_sha256,
        },
        "resources": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": progress["elapsed_seconds"],
            "max_wall_seconds": args.max_wall_seconds,
            "peak_gpu_allocated_bytes": peak_allocated,
            "gpu_memory_cap_bytes": progress.get("gpu_memory_cap_bytes"),
            "gpu_memory_cap_fraction": progress.get("gpu_memory_cap_fraction"),
            "new_disk_bytes": progress["new_disk_bytes"],
            "disk_cap_bytes": DISK_CAP_BYTES,
        },
        "api_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la4vla-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset-config", type=Path, required=True)
    parser.add_argument("--validation-dataset-config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--qc-report", type=Path, required=True)
    parser.add_argument("--loader-smoke-report", type=Path, required=True)
    parser.add_argument("--expected-qc-sha256", required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--attempt-plan-sha256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    parser.add_argument("--max-gpu-memory-gib", type=float, default=GPU_MEMORY_CAP_GIB)
    args = parser.parse_args()
    if args.steps != 100:
        parser.error("this bounded runner requires exactly --steps 100")
    if args.batch_size <= 0 or args.lr <= 0:
        parser.error("batch size and lr must be positive")
    if args.max_wall_seconds <= 0:
        parser.error("max wall seconds must be positive")
    progress = _new_progress(args)
    previous_alarm = None
    alarm_installed = False
    try:
        # On the Linux server this turns a hang in imports/model loading into a
        # normal failure artifact with the phase recorded.  The in-loop guards
        # below provide the same bound on platforms without SIGALRM.
        if hasattr(signal, "SIGALRM"):
            previous_alarm = signal.getsignal(signal.SIGALRM)

            def _alarm_handler(signum, frame):
                raise TimeoutError(
                    f"wall-clock budget exceeded during {progress['phase']}"
                )

            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
            alarm_installed = True
        for output_path in (
            args.output.resolve(),
            args.checkpoint_output.resolve(),
        ):
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite attempt artifact: {output_path}")
        report = run(args, progress)
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
        return 0
    except Exception as exc:
        _refresh_progress(progress)
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "mannercap-la4vla-student-dev-training.v1",
            "status": "fail",
            "phase": progress["phase"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "plan_sha256": args.plan_sha256,
            "attempt_plan_sha256": getattr(args, "attempt_plan_sha256", None),
            "completed_steps": progress["completed_steps"],
            "elapsed_seconds": progress["elapsed_seconds"],
            "progress": _public_progress(progress),
            "claim_boundary": "development_only",
            "promotion_allowed": False,
            "closed_loop_claim_allowed": False,
        }
        if isinstance(exc, DatasetBindingError):
            failure["dataset_binding"] = exc.details
        # A pre-existing report is evidence from an earlier attempt; preserve it.
        if not args.output.resolve().exists():
            args.output.resolve().write_text(
                json.dumps(failure, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(failure))
        return 1
    finally:
        if alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == "__main__":
    raise SystemExit(main())
