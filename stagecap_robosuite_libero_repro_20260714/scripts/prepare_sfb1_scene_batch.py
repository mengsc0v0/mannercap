from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOROB_ROOT = REPO_ROOT.parents[1]
DEFAULT_MANIFEST = AUTOROB_ROOT / "docs" / "smolvla_failure_driven_libero_scene_batch_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "smolvla_failure_driven_libero_scene_batch" / "v1"


def _load_libero() -> Any:
    vendor = Path("/home/lifuchun/data/capx/repos/cap-x/capx/third_party/LIBERO-PRO")
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.append(str(vendor))
    from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]

    return OffScreenRenderEnv


def _resolve_source_bddl(path_text: str) -> Path:
    candidates = [
        AUTOROB_ROOT / path_text,
        Path("/home/lifuchun/data/capx/repos/cap-x/capx") / path_text,
        Path("/home/lifuchun/data/capx/repos/cap-x/capx/third_party/LIBERO-PRO/libero/libero/bddl_files")
        / "/".join(Path(path_text).parts[-2:]),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve source BDDL: {path_text}")


def _replace_language(content: str, language: str) -> str:
    pattern = re.compile(r"\(:language\s+.*?\)", flags=re.IGNORECASE | re.DOTALL)
    replacement = f"(:language {language})"
    new_content, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise ValueError("Expected exactly one (:language ...) field")
    return new_content


def _task_filename(task: dict[str, Any]) -> str:
    task_id = str(task["task_id"]).replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(task["name"]).strip()).strip("_")
    return f"{task_id}_{name}.bddl"


def _write_scene_files(tasks: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    bddl_dir = output_root / "generated_bddl"
    metadata_dir = output_root / "metadata"
    bddl_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        source = _resolve_source_bddl(str(task["source_bddl"]))
        content = source.read_text(encoding="utf-8")
        content = _replace_language(content, str(task["language"]))
        filename = _task_filename(task)
        generated = bddl_dir / filename
        generated.write_text(content, encoding="utf-8")

        record = {
            "task_index": index,
            "task_id": task["task_id"],
            "suite_name": "sfb1",
            "name": task["name"],
            "family_id": task["family_id"],
            "source_bddl": str(source),
            "generated_bddl": str(generated),
            "language": task["language"],
            "success_predicates": task.get("success_predicates", []),
            "diagnostics": task.get("diagnostics", []),
            "edits": task.get("edits", {}),
        }
        (metadata_dir / f"{task['task_id'].replace('-', '_')}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    return records


def _image_from_obs(obs: dict[str, Any], camera: str) -> np.ndarray:
    key = f"{camera}_image"
    if key in obs:
        return np.asarray(obs[key])
    value = obs[camera]
    if isinstance(value, dict):
        if "images" in value and "rgb" in value["images"]:
            return np.asarray(value["images"]["rgb"])
        if "rgb" in value:
            return np.asarray(value["rgb"])
    raise KeyError(f"Cannot find RGB image for camera {camera}")


def _validate_task(record: dict[str, Any], output_root: Path, render_gpu_device_id: int) -> dict[str, Any]:
    OffScreenRenderEnv = _load_libero()
    images_dir = output_root / "images" / str(record["task_id"])
    images_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": record["task_id"],
        "generated_bddl": record["generated_bddl"],
        "reset_ok": False,
        "render_ok": False,
        "initial_success": None,
        "image_paths": {},
        "error": None,
    }
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=record["generated_bddl"],
            camera_heights=256,
            camera_widths=256,
            camera_depths=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            controller="JOINT_POSITION",
            horizon=200,
            control_freq=20,
            render_gpu_device_id=render_gpu_device_id,
        )
        obs = env.reset()
        result["reset_ok"] = True
        result["initial_success"] = bool(env.check_success())
        for camera in ["agentview", "robot0_eye_in_hand"]:
            rgb = _image_from_obs(obs, camera)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            image_path = images_dir / f"{camera}.png"
            Image.fromarray(rgb).save(image_path)
            result["image_paths"][camera] = str(image_path)
        result["render_ok"] = all(Path(path).exists() for path in result["image_paths"].values())
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def _write_reports(
    *,
    output_root: Path,
    source_manifest: Path,
    scene_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> None:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cap_manifest = {
        "schema_version": "stagecap.sfb1.cap_manifest.v1",
        "suite_name": "sfb1",
        "source_manifest": str(source_manifest),
        "tasks": scene_records,
    }
    (output_root / "sfb1_cap_manifest.json").write_text(
        json.dumps(cap_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "task_count": len(scene_records),
        "reset_ok_count": sum(bool(item["reset_ok"]) for item in validation_records),
        "render_ok_count": sum(bool(item["render_ok"]) for item in validation_records),
        "initial_success_false_count": sum(item["initial_success"] is False for item in validation_records),
        "valid_for_cap_count": sum(
            bool(item["reset_ok"]) and bool(item["render_ok"]) and item["initial_success"] is False
            for item in validation_records
        ),
        "validation": validation_records,
        "cap_manifest": str(output_root / "sfb1_cap_manifest.json"),
    }
    (reports_dir / "scene_batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SFB1 Scene Batch Report",
        "",
        f"- Source manifest: `{source_manifest}`",
        f"- CAP manifest: `{output_root / 'sfb1_cap_manifest.json'}`",
        f"- Tasks: {summary['task_count']}",
        f"- Reset OK: {summary['reset_ok_count']}/{summary['task_count']}",
        f"- Render OK: {summary['render_ok_count']}/{summary['task_count']}",
        f"- Initial success false: {summary['initial_success_false_count']}/{summary['task_count']}",
        "",
        "| Task | Reset | Render | Initial Success | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_task = {item["task_id"]: item for item in validation_records}
    for record in scene_records:
        item = by_task.get(record["task_id"], {})
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                record["task_id"],
                "ok" if item.get("reset_ok") else "fail",
                "ok" if item.get("render_ok") else "fail",
                item.get("initial_success"),
                "" if not item.get("error") else str(item.get("error")).replace("|", "/"),
            )
        )
    (reports_dir / "scene_batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_records = _write_scene_files(list(data["tasks"]), output_root)
    validation_records = []
    if not args.skip_validation:
        for record in scene_records:
            print(f"Validating {record['task_id']} {record['name']}")
            validation_records.append(_validate_task(record, output_root, args.render_gpu_device_id))
    else:
        validation_records = [
            {
                "task_id": record["task_id"],
                "generated_bddl": record["generated_bddl"],
                "reset_ok": None,
                "render_ok": None,
                "initial_success": None,
                "image_paths": {},
                "error": "validation skipped",
            }
            for record in scene_records
        ]
    _write_reports(
        output_root=output_root,
        source_manifest=manifest_path,
        scene_records=scene_records,
        validation_records=validation_records,
    )
    print(output_root / "sfb1_cap_manifest.json")


if __name__ == "__main__":
    main()
