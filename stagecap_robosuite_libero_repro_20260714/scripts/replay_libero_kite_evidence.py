#!/usr/bin/env python3
"""Replay a LIBERO stage-code trajectory and export KITE-style evidence panels.

This script is intentionally evidence-only: it reuses an existing generated
``code.py`` and config, executes the same stage blocks, and saves observations
after each block. It does not call a planner or synthesize new task code.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from capx.envs.configs.instantiate import instantiate as cfg_instantiate
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_rgb


STAGE_RE = re.compile(r"(?m)^# Stage code block\s+(\d+)\s*$")
STATUS_RE = re.compile(r"^STAGE_STATUS\s+(\{.*\})\s*$")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def split_stage_blocks(code: str) -> list[dict[str, Any]]:
    matches = list(STAGE_RE.finditer(code))
    if not matches:
        return [{"stage_block": 0, "code": code.strip()}]
    blocks: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(code)
        block_no = int(match.group(1))
        block_code = code[start:end].strip()
        if block_code:
            blocks.append({"stage_block": block_no, "code": block_code})
    return blocks


def parse_stage_status(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        match = STATUS_RE.match(line.strip())
        if not match:
            continue
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"raw": match.group(1)}
    return None


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "stage"


def save_png(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def segmentation_to_rgb(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = np.squeeze(seg, axis=-1) if seg.shape[-1] == 1 else seg[..., 0]
    seg = seg.astype(np.int64)
    unique = np.unique(seg)
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for value in unique:
        if value < 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        elif value == 0:
            color = np.array([24, 28, 34], dtype=np.uint8)
        else:
            rng = np.random.default_rng(int(value) * 1009 + 17)
            color = rng.integers(48, 245, size=3, dtype=np.uint8)
        out[seg == value] = color
    return out


def make_segmentation_overlay(rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    seg_rgb = segmentation_to_rgb(seg).astype(np.float32)
    base = rgb.astype(np.float32)
    mask = np.asarray(seg)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1) if mask.shape[-1] == 1 else mask[..., 0]
    valid = mask > 0
    overlay = base.copy()
    overlay[valid] = base[valid] * (1.0 - alpha) + seg_rgb[valid] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_segmentation_boxes(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    mask = np.asarray(seg)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1) if mask.shape[-1] == 1 else mask[..., 0]
    h, w = mask.shape[:2]
    for value in np.unique(mask):
        if value <= 0:
            continue
        ys, xs = np.where(mask == value)
        if xs.size < max(20, h * w * 0.0001):
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if (x1 - x0) < 3 or (y1 - y0) < 3:
            continue
        rng = np.random.default_rng(int(value) * 1009 + 17)
        color = tuple(int(x) for x in rng.integers(48, 245, size=3))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((min(x0 + 3, w - 30), max(2, y0 - 14)), f"id {int(value)}", fill=color)
    return np.asarray(image)


def render_segmentation_snapshot(
    env: Any,
    camera_name: str,
    height: int,
    width: int,
    level: str,
) -> np.ndarray | None:
    low_env = getattr(env, "low_level_env", None)
    handle = getattr(low_env, "handle", None)
    sim_env = getattr(handle, "env", None)
    sim = getattr(sim_env, "sim", None)
    if sim is None:
        return None
    try:
        raw = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False,
            segmentation=True,
        )
    except Exception:
        return None
    raw_arr = np.asarray(raw)
    if raw_arr.ndim != 3 or raw_arr.shape[-1] < 2:
        return None
    seg = raw_arr[::-1, :, 1].astype(np.int32)
    task_env = getattr(sim_env, "env", None)
    model = getattr(task_env, "model", None)
    mapping = None
    if model is not None and level == "instance":
        instances = list(getattr(model, "instances_to_ids", {}).keys())
        name2id = {inst: i for i, inst in enumerate(instances)}
        mapping = {
            int(geom_id): int(name2id[inst])
            for geom_id, inst in getattr(model, "geom_ids_to_instances", {}).items()
            if inst in name2id
        }
    elif model is not None and level == "class":
        classes = list(getattr(model, "classes_to_ids", {}).keys())
        name2id = {cls: i for i, cls in enumerate(classes)}
        mapping = {
            int(geom_id): int(name2id[cls])
            for geom_id, cls in getattr(model, "geom_ids_to_classes", {}).items()
            if cls in name2id
        }
    if mapping is not None:
        flat = np.fromiter((mapping.get(int(x), -1) for x in seg.reshape(-1)), dtype=np.int32)
        seg = flat.reshape(seg.shape) + 1
    return seg


def make_optical_flow(prev_rgb: np.ndarray | None, rgb: np.ndarray) -> np.ndarray:
    if prev_rgb is None:
        return np.zeros_like(rgb)
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(rgb)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 220
    hsv[..., 2] = np.clip(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, limit: int = 90000) -> None:
    if points.shape[0] == 0:
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        return
    if points.shape[0] > limit:
        idx = np.linspace(0, points.shape[0] - 1, limit).astype(np.int64)
        points = points[idx]
        colors = colors[idx]
    cols = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, cols, strict=False):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def transform_points(points: np.ndarray, pose_mat: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (pose_mat @ pts_h.T).T[:, :3]


def pointcloud_render(points: np.ndarray, colors: np.ndarray, out_path: Path, title: str) -> None:
    if points.shape[0] == 0:
        fig = plt.figure(figsize=(5, 4), dpi=140)
        plt.text(0.5, 0.5, "No valid points", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return
    limit = min(18000, points.shape[0])
    idx = np.linspace(0, points.shape[0] - 1, limit).astype(np.int64)
    pts = points[idx]
    cols = colors[idx]
    fig = plt.figure(figsize=(5.4, 4.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.25, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mins = np.percentile(pts, 2, axis=0)
    maxs = np.percentile(pts, 98, axis=0)
    center = (mins + maxs) / 2
    span = float(np.max(maxs - mins))
    span = max(span, 1e-3)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.view_init(elev=23, azim=-55)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def pseudo_bev(points_robot: np.ndarray, colors: np.ndarray, out_path: Path, facts: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    if points_robot.shape[0] > 0:
        mask = np.isfinite(points_robot).all(axis=1)
        pts = points_robot[mask]
        cols = colors[mask]
        if pts.shape[0] > 25000:
            idx = np.linspace(0, pts.shape[0] - 1, 25000).astype(np.int64)
            pts = pts[idx]
            cols = cols[idx]
        z = pts[:, 2]
        keep = (z > np.percentile(z, 1)) & (z < np.percentile(z, 99))
        pts = pts[keep]
        cols = cols[keep]
        ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=0.3, linewidths=0, alpha=0.85)
    for name, color, marker in [
        ("akita_black_bowl", "#f05a28", "o"),
        ("bottom_drawer_target", "#1f77b4", "s"),
        ("bottom_drawer_handle", "#2ca02c", "^"),
    ]:
        pos = facts.get(name)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            ax.scatter([float(pos[0])], [float(pos[1])], c=color, marker=marker, s=48, edgecolors="white", linewidths=0.8)
            ax.text(float(pos[0]), float(pos[1]), name.replace("akita_black_", "").replace("bottom_drawer_", ""), fontsize=7, color=color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("robot x")
    ax.set_ylabel("robot y")
    ax.set_title("Pseudo-BEV from depth")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def collect_readonly_facts(env: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    namespace = getattr(env, "_exec_globals", {})

    def call(name: str, *args: Any, **kwargs: Any) -> Any:
        fn = namespace.get(name)
        if not callable(fn):
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return {"error": repr(exc)}

    facts["task_status"] = call("get_task_status")
    facts["bddl_goal_diagnostics"] = call("get_bddl_goal_diagnostics")
    facts["bottom_drawer_progress"] = call("get_cabinet_drawer_progress", "bottom")
    facts["bottom_drawer_handle_raw"] = call("get_cabinet_drawer_handle_pose", "bottom")
    facts["bowl_pose_raw"] = call("get_object_pose", "akita_black_bowl")
    facts["bottom_drawer_target_raw"] = call("get_goal_target_region_pose", "bottom_drawer")

    handle = facts.get("bottom_drawer_handle_raw")
    if isinstance(handle, (list, tuple)) and len(handle) >= 1:
        try:
            facts["bottom_drawer_handle"] = np.asarray(handle[0], dtype=float).tolist()
        except Exception:
            pass
    bowl_pose = facts.get("bowl_pose_raw")
    if isinstance(bowl_pose, (list, tuple)) and len(bowl_pose) >= 1:
        try:
            facts["akita_black_bowl"] = np.asarray(bowl_pose[0], dtype=float).tolist()
        except Exception:
            pass
    target_pose = facts.get("bottom_drawer_target_raw")
    if isinstance(target_pose, (list, tuple)) and len(target_pose) >= 1:
        try:
            facts["bottom_drawer_target"] = np.asarray(target_pose[0], dtype=float).tolist()
        except Exception:
            pass
    return facts


def save_observation_bundle(
    env: Any,
    stage_dir: Path,
    stage_label: str,
    prev_rgbs: dict[str, np.ndarray | None],
    stage_status: dict[str, Any] | None,
    exec_info: dict[str, Any],
    facts: dict[str, Any],
    snapshot_segmentation: str | None = None,
) -> dict[str, Any]:
    obs = env.low_level_env.get_observation()
    frame_count = int(env.get_video_frame_count()) if hasattr(env, "get_video_frame_count") else 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "stage_label": stage_label,
        "frame_count": frame_count,
        "stage_status": stage_status,
        "exec_info": exec_info,
        "readonly_facts": facts,
        "cameras": {},
    }

    for camera_name in ("agentview", "robot0_eye_in_hand"):
        if camera_name not in obs:
            continue
        cam_obs = obs[camera_name]
        images = cam_obs.get("images", {})
        if "rgb" not in images:
            continue
        cam_dir = stage_dir / camera_name
        cam_dir.mkdir(exist_ok=True)
        rgb = np.asarray(images["rgb"], dtype=np.uint8)
        save_png(cam_dir / "rgb.png", rgb)
        cam_manifest: dict[str, Any] = {
            "rgb": str(cam_dir / "rgb.png"),
            "intrinsics": str(cam_dir / "intrinsics.npy"),
            "pose_mat": str(cam_dir / "pose_mat.npy"),
        }
        np.save(cam_dir / "intrinsics.npy", np.asarray(cam_obs.get("intrinsics")))
        np.save(cam_dir / "pose_mat.npy", np.asarray(cam_obs.get("pose_mat")))

        flow_rgb = make_optical_flow(prev_rgbs.get(camera_name), rgb)
        save_png(cam_dir / "optical_flow_from_prev_stage.png", flow_rgb)
        cam_manifest["optical_flow"] = str(cam_dir / "optical_flow_from_prev_stage.png")
        prev_rgbs[camera_name] = rgb.copy()

        depth = images.get("depth")
        if depth is not None:
            depth_arr = np.asarray(depth, dtype=np.float32)
            if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
                depth_arr = depth_arr[..., 0]
            np.save(cam_dir / "depth.npy", depth_arr)
            depth_rgb = depth_to_rgb(depth_arr, use_percentiles=(2, 98), cmap_name="turbo")
            save_png(cam_dir / "depth_colormap.png", depth_rgb)
            cam_manifest["depth"] = str(cam_dir / "depth.npy")
            cam_manifest["depth_colormap"] = str(cam_dir / "depth_colormap.png")

            intrinsics = np.asarray(cam_obs.get("intrinsics"), dtype=np.float64)
            pose_mat = np.asarray(cam_obs.get("pose_mat"), dtype=np.float64)
            subsample_factor = 3
            h_crop = (depth_arr.shape[0] // subsample_factor) * subsample_factor
            w_crop = (depth_arr.shape[1] // subsample_factor) * subsample_factor
            depth_for_pc = depth_arr[:h_crop, :w_crop]
            rgb_for_pc = rgb[:h_crop, :w_crop]
            points, colors = depth_color_to_pointcloud(
                depth_for_pc,
                rgb_for_pc,
                intrinsics,
                subsample_factor=subsample_factor,
                depth_clip_range=(0.05, 3.5),
            )
            np.save(cam_dir / "pointcloud_camera.npy", points)
            np.save(cam_dir / "pointcloud_colors.npy", colors)
            write_ply(cam_dir / "pointcloud_camera.ply", points, colors)
            pointcloud_render(points, colors, cam_dir / "pointcloud_render.png", f"{stage_label} {camera_name}")
            cam_manifest["pointcloud_camera"] = str(cam_dir / "pointcloud_camera.npy")
            cam_manifest["pointcloud_ply"] = str(cam_dir / "pointcloud_camera.ply")
            cam_manifest["pointcloud_render"] = str(cam_dir / "pointcloud_render.png")
            if pose_mat.shape == (4, 4):
                points_robot = transform_points(points, pose_mat)
                np.save(cam_dir / "pointcloud_robot.npy", points_robot)
                pseudo_bev(points_robot, colors, cam_dir / "pseudo_bev.png", facts)
                cam_manifest["pointcloud_robot"] = str(cam_dir / "pointcloud_robot.npy")
                cam_manifest["pseudo_bev"] = str(cam_dir / "pseudo_bev.png")

        segmentation = images.get("segmentation")
        if segmentation is None and snapshot_segmentation is not None:
            segmentation = render_segmentation_snapshot(
                env,
                camera_name,
                height=int(rgb.shape[0]),
                width=int(rgb.shape[1]),
                level=snapshot_segmentation,
            )
        if segmentation is not None:
            seg_arr = np.asarray(segmentation)
            np.save(cam_dir / "segmentation.npy", seg_arr)
            save_png(cam_dir / "segmentation_colormap.png", segmentation_to_rgb(seg_arr))
            save_png(cam_dir / "segmentation_overlay.png", make_segmentation_overlay(rgb, seg_arr))
            save_png(cam_dir / "detections_overlay.png", make_segmentation_boxes(rgb, seg_arr))
            cam_manifest["segmentation"] = str(cam_dir / "segmentation.npy")
            cam_manifest["segmentation_colormap"] = str(cam_dir / "segmentation_colormap.png")
            cam_manifest["segmentation_overlay"] = str(cam_dir / "segmentation_overlay.png")
            cam_manifest["detections_overlay"] = str(cam_dir / "detections_overlay.png")
            cam_manifest["segmentation_ids"] = [int(x) for x in np.unique(seg_arr)]

        manifest["cameras"][camera_name] = cam_manifest

    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    return manifest


def draw_captioned_tile(image_path: Path, title: str, subtitle: str, size: tuple[int, int] = (360, 260)) -> Image.Image:
    tile = Image.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(tile)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    if image_path.exists():
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((size[0] - 18, size[1] - 58), Image.Resampling.LANCZOS)
        tile.paste(img, ((size[0] - img.width) // 2, 8))
    y = size[1] - 46
    draw.text((9, y), title[:48], fill=(20, 24, 30), font=font_title)
    draw.text((9, y + 20), subtitle[:70], fill=(78, 85, 94), font=font_sub)
    return tile


def make_contact_sheet(stage_manifests: list[dict[str, Any]], out_path: Path) -> None:
    tiles: list[Image.Image] = []
    for manifest in stage_manifests:
        stage_label = str(manifest.get("stage_label", "stage"))
        status = manifest.get("stage_status") or {}
        status_text = str(status.get("status", "unknown")) if isinstance(status, dict) else "unknown"
        evidence = str(status.get("evidence", "")) if isinstance(status, dict) else ""
        camera = manifest.get("cameras", {}).get("agentview", {})
        path = Path(camera.get("segmentation_overlay") or camera.get("rgb") or "")
        tiles.append(draw_captioned_tile(path, stage_label, f"{status_text}: {evidence}"))
    if not tiles:
        return
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (238, 240, 243))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * w, (idx // cols) * h))
    sheet.save(out_path)


def write_index(out_dir: Path, manifests: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# LIBERO KITE-style multimodal evidence",
        "",
        f"- config: `{args.config}`",
        f"- code: `{args.code}`",
        f"- trial seed: `{args.trial}`",
        f"- stage limit: `{args.stage_limit}`",
        "",
        "## Stages",
        "",
    ]
    for manifest in manifests:
        stage_label = str(manifest.get("stage_label", "stage"))
        status = manifest.get("stage_status") or {}
        status_text = status.get("status", "unknown") if isinstance(status, dict) else "unknown"
        evidence = status.get("evidence", "") if isinstance(status, dict) else ""
        repair = status.get("repair", "") if isinstance(status, dict) else ""
        cam = manifest.get("cameras", {}).get("agentview", {})
        lines.append(f"### {stage_label}")
        lines.append(f"- status: `{status_text}`")
        lines.append(f"- evidence: {evidence}")
        if repair:
            lines.append(f"- repair: {repair}")
        for key in [
            "rgb",
            "depth_colormap",
            "segmentation_overlay",
            "detections_overlay",
            "optical_flow",
            "pseudo_bev",
            "pointcloud_render",
            "pointcloud_ply",
        ]:
            if key in cam:
                rel = Path(cam[key]).relative_to(out_dir)
                lines.append(f"- {key}: `{rel}`")
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines))


def instantiate_env(config_path: Path, segmentation: str | None) -> Any:
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg.env, resolve=True)
    if segmentation is not None:
        cfg_dict["cfg"]["low_level"]["camera_segmentations"] = segmentation
    return cfg_instantiate(cfg_dict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--stage-trace", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--segmentation", default="instance", choices=["instance", "class", "element", "none"])
    parser.add_argument(
        "--snapshot-segmentation",
        default="none",
        choices=["instance", "class", "element", "none"],
        help="Render segmentation only when saving snapshots. Useful for faster action replay.",
    )
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--copy-source", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_source:
        shutil.copy2(args.config, out_dir / "source_config.yaml")
        shutil.copy2(args.code, out_dir / "source_code.py")
        if args.stage_trace and args.stage_trace.exists():
            shutil.copy2(args.stage_trace, out_dir / "source_stage_trace.json")

    code = args.code.read_text()
    blocks = split_stage_blocks(code)
    if args.stage_limit is not None:
        blocks = blocks[: args.stage_limit]

    segmentation = None if args.segmentation == "none" else args.segmentation
    snapshot_segmentation = None if args.snapshot_segmentation == "none" else args.snapshot_segmentation
    env = instantiate_env(args.config, segmentation)
    env.enable_video_capture(True, clear=True, wrist_camera=True)
    env.reset(seed=args.trial, options={"trial": args.trial})

    manifests: list[dict[str, Any]] = []
    prev_rgbs: dict[str, np.ndarray | None] = {"agentview": None, "robot0_eye_in_hand": None}
    execution_log: list[dict[str, Any]] = []

    initial_facts = collect_readonly_facts(env)
    initial_manifest = save_observation_bundle(
        env,
        out_dir / "stage_00_initial",
        "stage_00_initial",
        prev_rgbs,
        {"stage_id": "initial", "status": "snapshot", "evidence": "post-reset observation", "repair": ""},
        {"sandbox_rc": None, "stdout_tail": "", "stderr_tail": ""},
        initial_facts,
        snapshot_segmentation=snapshot_segmentation,
    )
    manifests.append(initial_manifest)

    for idx, block in enumerate(blocks, start=1):
        block_no = int(block["stage_block"])
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            obs, reward, terminated, truncated, info = env.step(block["code"])
        stdout = stdout_buffer.getvalue() + str(info.get("stdout", ""))
        stderr = stderr_buffer.getvalue() + str(info.get("stderr", ""))
        stage_status = parse_stage_status(stdout)
        stage_id = "unknown"
        if isinstance(stage_status, dict):
            stage_id = str(stage_status.get("stage_id", stage_id))
        stage_label = f"stage_{idx:02d}_block_{block_no:02d}_{sanitize_name(stage_id)}"
        exec_info = {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "sandbox_rc": info.get("sandbox_rc"),
            "task_completed": info.get("task_completed"),
            "stdout_tail": "\n".join(stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-40:]),
        }
        facts = collect_readonly_facts(env)
        stage_dir = out_dir / stage_label
        manifest = save_observation_bundle(
            env,
            stage_dir,
            stage_label,
            prev_rgbs,
            stage_status,
            exec_info,
            facts,
            snapshot_segmentation=snapshot_segmentation,
        )
        manifests.append(manifest)
        execution_log.append(
            {
                "stage_label": stage_label,
                "stage_block": block_no,
                "status": stage_status,
                "exec_info": exec_info,
            }
        )
        print(f"[{stage_label}] status={stage_status} reward={reward:.3f} frames={env.get_video_frame_count()}", flush=True)
        if truncated:
            print(f"Episode truncated after {stage_label}; stopping replay.", flush=True)
            break

    (out_dir / "execution_log.json").write_text(json.dumps(execution_log, indent=2, default=_json_default))
    (out_dir / "manifest.json").write_text(json.dumps(manifests, indent=2, default=_json_default))
    make_contact_sheet(manifests, out_dir / "contact_sheet_multimodal_evidence.png")
    write_index(out_dir, manifests, args)
    print(json.dumps({"out_dir": str(out_dir), "stages": len(manifests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
