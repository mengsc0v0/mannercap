#!/usr/bin/env python3
"""Collect LIBERO/KITE paper-figure materials into one folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def tile(path: Path, label: str, size: tuple[int, int] = (260, 220)) -> Image.Image:
    canvas = Image.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    if path.exists():
        image = Image.open(path).convert("RGB")
        image.thumbnail((size[0] - 12, size[1] - 40), Image.Resampling.LANCZOS)
        canvas.paste(image, ((size[0] - image.width) // 2, 8))
    draw.text((10, size[1] - 28), label, fill=(24, 28, 34), font=font)
    return canvas


def make_modality_panel(stage_dir: Path, out_path: Path, title: str) -> None:
    cam = stage_dir / "agentview"
    entries = [
        ("rgb.png", "RGB"),
        ("depth_colormap.png", "Depth"),
        ("segmentation_overlay.png", "Seg."),
        ("pseudo_bev.png", "Pseudo-BEV"),
        ("pointcloud_render.png", "Point Cloud"),
    ]
    tiles = [tile(cam / filename, label) for filename, label in entries]
    header_h = 42
    w, h = tiles[0].size
    panel = Image.new("RGB", (w * len(tiles), h + header_h), (238, 240, 243))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((14, 10), title, fill=(18, 22, 28), font=font)
    for idx, image in enumerate(tiles):
        panel.paste(image, (idx * w, header_h))
    panel.save(out_path)


def make_feedback_panel(kite_dir: Path, out_path: Path) -> None:
    annotated = kite_dir / "annotated_frames"
    examples = [
        ("05_stage_success_bottom_drawer_sleeve_or_hook_handle_frame_00786.png", "Drawer open OK"),
        ("09_stage_success_obj1_bowl_low_release_verify_frame_03107.png", "Bowl placed OK"),
        ("14_failure_feedback_bottom_drawer_close_guarded_frame_03918.png", "Close failed"),
        ("18_failure_feedback_bottom_drawer_close_guarded_frame_05147.png", "Feedback retry"),
        ("22_failure_feedback_bottom_drawer_close_guarded_frame_05998.png", "Persistent failure"),
    ]
    tiles = [tile(annotated / filename, label, size=(280, 220)) for filename, label in examples]
    header_h = 44
    w, h = tiles[0].size
    panel = Image.new("RGB", (w * len(tiles), h + header_h), (238, 240, 243))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((14, 10), "KITE-selected success/failure frames with feedback", fill=(18, 22, 28), font=font)
    for idx, image in enumerate(tiles):
        panel.paste(image, (idx * w, header_h))
    panel.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kite-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--script-path", type=Path, required=True)
    parser.add_argument("--docs-path", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copytree(args.kite_dir, out_dir / "kite_rgb_keyframes")
    copytree(args.evidence_dir, out_dir / "real_multimodal_replay_smoke")
    copy_file(args.script_path, out_dir / "scripts" / args.script_path.name)
    if args.docs_path and args.docs_path.exists():
        copy_file(args.docs_path, out_dir / "scripts" / args.docs_path.name)

    panels_dir = out_dir / "figure_panels"
    panels_dir.mkdir()
    make_modality_panel(
        args.evidence_dir / "stage_00_initial",
        panels_dir / "modality_panel_initial_agentview.png",
        "Real LIBERO sensor evidence: initial state",
    )
    make_modality_panel(
        args.evidence_dir / "stage_01_block_00_bottom_drawer_ground_handle",
        panels_dir / "modality_panel_handle_grounding_agentview.png",
        "Real LIBERO sensor evidence: handle grounding",
    )
    make_feedback_panel(
        args.kite_dir,
        panels_dir / "kite_feedback_frame_strip.png",
    )

    manifest = {
        "kite_rgb_keyframes": str(out_dir / "kite_rgb_keyframes"),
        "real_multimodal_replay_smoke": str(out_dir / "real_multimodal_replay_smoke"),
        "figure_panels": str(panels_dir),
        "script": str(out_dir / "scripts" / args.script_path.name),
        "notes": [
            "KITE keyframes are selected from the existing RGB video and preserve success/failure feedback frames.",
            "Depth, segmentation, BEV, and point clouds in real_multimodal_replay_smoke are generated by rerunning LIBERO, not reconstructed from RGB video.",
            "For faster future replays use: --segmentation none --snapshot-segmentation instance.",
        ],
    }
    (out_dir / "materials_manifest.json").write_text(json.dumps(manifest, indent=2))

    readme = [
        "# LIBERO KITE Figure Materials",
        "",
        "This folder collects materials for the paper method figure.",
        "",
        "## What Is Inside",
        "",
        "- `kite_rgb_keyframes/`: KITE-selected RGB frames from the existing trajectory, including success and failure-feedback frames.",
        "- `real_multimodal_replay_smoke/`: real LIBERO replay snapshots with RGB, depth, instance segmentation, pseudo-BEV, point-cloud previews, raw `.npy`, and `.ply` files.",
        "- `figure_panels/`: compact PNG strips that can be directly inserted into a method figure.",
        "- `scripts/`: replay and keyframe extraction scripts for regenerating or extending the assets.",
        "",
        "## Important Note",
        "",
        "The old `video_combined.mp4` is RGB-only. True depth, segmentation, and point cloud assets here come from rerunning LIBERO with evidence recording enabled.",
        "",
        "## Fast Replay Command Pattern",
        "",
        "Use `--segmentation none --snapshot-segmentation instance` to avoid rendering segmentation at every simulation step while still saving instance segmentation at evidence snapshots.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
