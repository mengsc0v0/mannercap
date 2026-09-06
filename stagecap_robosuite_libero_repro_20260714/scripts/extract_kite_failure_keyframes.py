#!/usr/bin/env python3
"""Extract KITE-style key failure frames from CaP-X stage trials.

The tool combines two evidence streams:
1. KITE motion keyframes from video optical flow.
2. CaP-X stage_trace status/repair events.

It writes raw screenshots, annotated screenshots, a contact sheet, and JSON/MD
indexes that can be reused for paper figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STATUS_FAIL = {"needs_repair", "failed", "failure", "blocked", "uncertain"}


@dataclass
class VideoMeta:
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class FrameCandidate:
    frame_idx: int
    time_sec: float
    kind: str
    stage_id: str = ""
    attempt: int | None = None
    status: str = ""
    source: str = ""
    motion_score: float | None = None
    motion_level: str = ""
    evidence: str = ""
    repair: str = ""
    stage_name: str = ""
    checks: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _autorob_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = text.strip("_")
    return (text or "item")[:max_len]


def _shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _video_meta(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    return VideoMeta(frame_count=frame_count, fps=fps, width=width, height=height)


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def _find_trial_dirs(experiment_dir: Path) -> list[Path]:
    trials = []
    for video_path in experiment_dir.rglob("video_combined.mp4"):
        trial_dir = video_path.parent
        if (trial_dir / "stage_trace.json").exists():
            trials.append(trial_dir)
    return sorted(set(trials))


def _find_record_json(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        candidates = sorted(parent.glob("*_stage_planner_record.json"))
        if candidates:
            return candidates[0]
    return None


def _stage_plan_index(record_path: Path | None) -> dict[str, dict[str, Any]]:
    if not record_path or not record_path.exists():
        return {}
    try:
        record = _json_load(record_path)
    except Exception:
        return {}
    stages = (record.get("stage_plan") or {}).get("stages") or []
    out: dict[str, dict[str, Any]] = {}
    for stage in stages:
        stage_id = str(stage.get("stage_id") or "")
        checks: list[str] = []
        criteria: list[str] = []
        for check in stage.get("checks") or []:
            name = str(check.get("name") or "").strip()
            desc = str(check.get("description") or "").strip()
            if name or desc:
                checks.append(f"{name}: {desc}" if name and desc else name or desc)
            for item in check.get("success_criteria") or []:
                criteria.append(str(item))
        out[stage_id] = {
            "name": str(stage.get("name") or stage_id),
            "intent": str(stage.get("intent") or ""),
            "family": str(stage.get("family") or ""),
            "checks": checks,
            "success_criteria": criteria,
            "repair_hints": [str(x) for x in stage.get("repair_hints") or []],
        }
    return out


def _trace_sim_start(trace_item: dict[str, Any]) -> int | None:
    status = ((trace_item.get("scene_snapshot") or {}).get("task_status") or {})
    value = status.get("sim_step_count")
    try:
        return int(value)
    except Exception:
        return None


def _trace_max_steps(trace_item: dict[str, Any]) -> int | None:
    status = ((trace_item.get("scene_snapshot") or {}).get("task_status") or {})
    value = status.get("max_steps")
    try:
        return int(value)
    except Exception:
        return None


def _sim_to_frame(sim_step: int, max_steps: int, meta: VideoMeta) -> int:
    if max_steps <= 0:
        return max(0, min(meta.frame_count - 1, sim_step))
    ratio = sim_step / float(max_steps)
    return int(max(0, min(meta.frame_count - 1, round(ratio * (meta.frame_count - 1)))))


def _stage_end_sims(trace: list[dict[str, Any]]) -> list[int | None]:
    starts = [_trace_sim_start(item) for item in trace]
    ends: list[int | None] = []
    for idx, start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else None
        if start is None:
            ends.append(next_start)
        elif next_start is not None and next_start > start:
            ends.append(next_start)
        else:
            ends.append(start)
    return ends


def _extract_numbers(evidence: str) -> str:
    fields = [
        "open_fraction",
        "before",
        "after",
        "delta",
        "best",
        "best_margin",
        "bddl_margin",
        "bddl_close_margin",
        "front_clearance",
        "lift_delta",
        "relation_satisfied",
        "placement_goals_ok",
        "task_completed",
    ]
    pieces: list[str] = []
    for field_name in fields:
        pattern = rf"{re.escape(field_name)}=([^,\]\) ]+)"
        match = re.search(pattern, evidence)
        if match:
            pieces.append(f"{field_name}={match.group(1)}")
    return ", ".join(pieces)


def _candidate_kind(status: str, repair: str) -> str:
    status_l = status.lower()
    if status_l in STATUS_FAIL or repair.strip():
        return "failure_feedback"
    if status_l == "satisfied":
        return "stage_success"
    return "stage_event"


def _stage_candidates(
    stage_trace_path: Path | None,
    record_path: Path | None,
    meta: VideoMeta,
    *,
    max_stage_successes: int,
) -> list[FrameCandidate]:
    if not stage_trace_path or not stage_trace_path.exists():
        return []
    trace = _json_load(stage_trace_path)
    if not isinstance(trace, list):
        return []
    plan = _stage_plan_index(record_path)
    ends = _stage_end_sims(trace)
    max_steps = next((m for m in (_trace_max_steps(item) for item in trace) if m), 0)
    if not max_steps:
        max_sim = max([s for s in ends if s is not None] + [meta.frame_count - 1])
        max_steps = max(max_sim, meta.frame_count - 1)

    candidates: list[FrameCandidate] = []
    success_count = 0
    for idx, item in enumerate(trace):
        status = str(item.get("status") or "")
        repair = str(item.get("repair_hint") or "")
        kind = _candidate_kind(status, repair)
        if kind == "stage_success":
            if success_count >= max_stage_successes:
                continue
            success_count += 1
        stage_id = str(item.get("stage_id") or "")
        stage_info = plan.get(stage_id, {})
        end_sim = ends[idx]
        if end_sim is None and idx == len(trace) - 1:
            frame_idx = meta.frame_count - 1
        else:
            frame_idx = _sim_to_frame(int(end_sim or 0), max_steps, meta)
        if idx == len(trace) - 1 and kind == "failure_feedback":
            frame_idx = max(frame_idx, meta.frame_count - 1)
        evidence = str(item.get("rationale") or "")
        numeric = _extract_numbers(evidence)
        if numeric:
            evidence = f"{numeric}; {_shorten(evidence, 320)}"
        candidates.append(
            FrameCandidate(
                frame_idx=frame_idx,
                time_sec=frame_idx / max(meta.fps, 1e-6),
                kind=kind,
                stage_id=stage_id,
                attempt=_safe_int(item.get("attempt")),
                status=status,
                source="stage_trace",
                evidence=evidence,
                repair=repair,
                stage_name=str(stage_info.get("name") or stage_id),
                checks=[str(x) for x in stage_info.get("checks") or []],
                success_criteria=[str(x) for x in stage_info.get("success_criteria") or []],
            )
        )
    return candidates


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _kite_root_candidates(user_root: str | None) -> list[Path]:
    candidates: list[Path] = []
    if user_root:
        candidates.append(Path(user_root).expanduser().resolve())
    candidates.append(_autorob_root_from_script() / "third_party" / "kite")
    candidates.append(_repo_root_from_script() / "third_party" / "kite")
    return candidates


def _kite_keyframes(
    video_path: Path,
    meta: VideoMeta,
    *,
    kite_root: str | None,
    strategy: str,
    max_keyframes: int,
    stride: int,
) -> tuple[list[FrameCandidate], str, str]:
    last_error = ""
    for root in _kite_root_candidates(kite_root):
        if not root.exists():
            continue
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from kite.video.keyframes import KeyframeSelector  # type: ignore

            selector = KeyframeSelector(
                strategy=strategy, max_keyframes=max_keyframes, stride=stride
            )
            kfs = selector.select(str(video_path))
            out = [
                FrameCandidate(
                    frame_idx=max(0, min(meta.frame_count - 1, int(kf.frame_idx))),
                    time_sec=float(kf.time_sec),
                    kind="kite_motion",
                    source="kite",
                    motion_score=float(kf.motion_score),
                    motion_level=str(kf.motion_level),
                    evidence=(
                        f"KITE motion keyframe: score={float(kf.motion_score):.3f}, "
                        f"level={kf.motion_level}"
                    ),
                )
                for kf in kfs
            ]
            return out, str(root), "kite.video.keyframes.KeyframeSelector"
        except Exception as exc:
            last_error = repr(exc)
    out = _fallback_motion_keyframes(
        video_path, meta, max_keyframes=max_keyframes, stride=stride
    )
    method = f"fallback Farneback optical flow; KITE import error: {last_error}"
    return out, "", method


def _fallback_motion_keyframes(
    video_path: Path,
    meta: VideoMeta,
    *,
    max_keyframes: int,
    stride: int,
) -> list[FrameCandidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    prev = None
    scores: list[tuple[int, float]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(1, stride) != 0:
                frame_idx += 1
                continue
            small = cv2.resize(frame, (512, 512))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                scores.append((frame_idx, float(np.mean(mag))))
            prev = gray
            frame_idx += 1
    finally:
        cap.release()
    if not scores:
        return []
    values = np.array([s for _, s in scores], dtype=np.float32)
    mn, mx = float(values.min()), float(values.max())
    norm = (values - mn) / (mx - mn + 1e-6)
    order = np.argsort(-norm)
    chosen: list[int] = []
    for pos in order:
        if len(chosen) >= max_keyframes:
            break
        frame = scores[int(pos)][0]
        if all(abs(frame - scores[c][0]) > 5 * stride for c in chosen):
            chosen.append(int(pos))
    chosen.sort(key=lambda i: scores[i][0])
    selected = norm[chosen] if chosen else np.array([], dtype=np.float32)
    low_th = float(np.quantile(selected, 0.33)) if len(selected) else 0.0
    high_th = float(np.quantile(selected, 0.66)) if len(selected) else 1.0
    out: list[FrameCandidate] = []
    for pos in chosen:
        frame = scores[pos][0]
        score = float(norm[pos])
        if score <= low_th:
            level = "low"
        elif score >= high_th:
            level = "high"
        else:
            level = "medium"
        out.append(
            FrameCandidate(
                frame_idx=frame,
                time_sec=frame / max(meta.fps, 1e-6),
                kind="kite_motion",
                source="fallback_optflow",
                motion_score=score,
                motion_level=level,
                evidence=f"Fallback motion keyframe: score={score:.3f}, level={level}",
            )
        )
    return out


def _merge_candidates(
    candidates: list[FrameCandidate],
    *,
    merge_window: int,
) -> list[FrameCandidate]:
    priority = {
        "failure_feedback": 0,
        "stage_event": 1,
        "stage_success": 2,
        "kite_motion": 3,
    }
    ordered = sorted(candidates, key=lambda c: (c.frame_idx, priority.get(c.kind, 9)))
    merged: list[FrameCandidate] = []
    for cand in ordered:
        target = None
        for existing in reversed(merged):
            if abs(existing.frame_idx - cand.frame_idx) <= merge_window:
                target = existing
                break
            if existing.frame_idx < cand.frame_idx - merge_window:
                break
        if target is None:
            merged.append(cand)
            continue
        if priority.get(cand.kind, 9) < priority.get(target.kind, 9):
            cand.source = "+".join(filter(None, [target.source, cand.source]))
            cand.evidence = _join_unique(target.evidence, cand.evidence)
            cand.repair = _join_unique(target.repair, cand.repair)
            cand.motion_score = target.motion_score if cand.motion_score is None else cand.motion_score
            cand.motion_level = cand.motion_level or target.motion_level
            merged[merged.index(target)] = cand
        else:
            target.source = "+".join(filter(None, [target.source, cand.source]))
            target.evidence = _join_unique(target.evidence, cand.evidence)
            target.repair = _join_unique(target.repair, cand.repair)
            if target.motion_score is None:
                target.motion_score = cand.motion_score
            if not target.motion_level:
                target.motion_level = cand.motion_level
    return sorted(merged, key=lambda c: c.frame_idx)


def _join_unique(a: str, b: str) -> str:
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a:
        return b
    if not b or b in a:
        return a
    if a in b:
        return b
    return f"{a} | {b}"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else f"{cur} {word}"
        if draw.textbbox((0, 0), nxt, font=font)[2] <= max_width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _annotate_frame(frame_bgr: np.ndarray, cand: FrameCandidate, index: int) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    title_font = _font(max(18, width // 42), bold=True)
    body_font = _font(max(13, width // 62), bold=False)
    small_font = _font(max(11, width // 72), bold=False)
    if cand.kind == "failure_feedback":
        accent = (205, 58, 52, 235)
        label = "FAILURE + FEEDBACK"
    elif cand.kind == "stage_success":
        accent = (50, 132, 84, 235)
        label = "VERIFIED STAGE"
    elif cand.kind == "kite_motion":
        accent = (55, 94, 172, 235)
        label = "KITE MOTION KEYFRAME"
    else:
        accent = (120, 120, 120, 235)
        label = "STAGE EVENT"

    panel_h = min(height - 12, max(106, int(height * 0.28)))
    draw.rectangle((0, 0, width, panel_h), fill=(8, 10, 14, 182))
    draw.rectangle((0, 0, 12, panel_h), fill=accent)
    title = (
        f"{index:02d} {label} | frame {cand.frame_idx:05d} | "
        f"t={cand.time_sec:.2f}s"
    )
    if cand.stage_id:
        title += f" | {cand.stage_id}"
    if cand.attempt is not None:
        title += f" a{cand.attempt}"
    draw.text((24, 12), _shorten(title, 130), font=title_font, fill=(255, 255, 255, 255))

    y = 44
    lines = []
    if cand.status:
        lines.append(f"status: {cand.status}")
    if cand.stage_name and cand.stage_name != cand.stage_id:
        lines.append(f"stage: {cand.stage_name}")
    if cand.motion_score is not None:
        lines.append(f"motion: {cand.motion_level} score={cand.motion_score:.3f}")
    if cand.evidence:
        lines.append(f"evidence: {_shorten(cand.evidence, 240)}")
    if cand.repair:
        lines.append(f"feedback: {_shorten(cand.repair, 220)}")
    if cand.checks:
        lines.append(f"verify: {_shorten(cand.checks[0], 180)}")

    max_width = width - 48
    for line in lines:
        for wrapped in _wrap_lines(draw, line, body_font, max_width)[:2]:
            if y > panel_h - 22:
                break
            draw.text((24, y), wrapped, font=body_font, fill=(244, 244, 238, 255))
            y += int(body_font.size * 1.25)
        if y > panel_h - 22:
            break

    footer = f"source: {cand.source or 'unknown'}"
    draw.rounded_rectangle(
        (16, height - 34, min(width - 16, 18 + 10 * len(footer)), height - 10),
        radius=4,
        fill=(8, 10, 14, 160),
    )
    draw.text((24, height - 30), footer, font=small_font, fill=(240, 240, 232, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _save_candidate_frames(
    video_path: Path,
    candidates: list[FrameCandidate],
    out_dir: Path,
) -> list[FrameCandidate]:
    raw_dir = out_dir / "raw_frames"
    ann_dir = out_dir / "annotated_frames"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if ann_dir.exists():
        shutil.rmtree(ann_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    saved: list[FrameCandidate] = []
    for idx, cand in enumerate(candidates, start=1):
        frame = _read_frame(video_path, cand.frame_idx)
        base = "_".join(
            [
                f"{idx:02d}",
                _safe_filename(cand.kind, 32),
                _safe_filename(cand.stage_id or cand.source or "motion", 48),
                f"frame_{cand.frame_idx:05d}",
            ]
        )
        raw_path = raw_dir / f"{base}.png"
        ann_path = ann_dir / f"{base}.png"
        cv2.imwrite(str(raw_path), frame)
        annotated = _annotate_frame(frame, cand, idx)
        annotated.save(ann_path)
        cand.files = {
            "raw": str(raw_path.resolve()),
            "annotated": str(ann_path.resolve()),
        }
        saved.append(cand)
    return saved


def _make_contact_sheet(
    candidates: list[FrameCandidate],
    out_path: Path,
    *,
    tile_width: int = 360,
    columns: int = 3,
) -> None:
    if not candidates:
        return
    images: list[Image.Image] = []
    for cand in candidates:
        ann = cand.files.get("annotated")
        if not ann:
            continue
        img = Image.open(ann).convert("RGB")
        ratio = tile_width / img.width
        tile = img.resize((tile_width, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        images.append(tile)
    if not images:
        return
    tile_h = max(img.height for img in images)
    rows = int(math.ceil(len(images) / float(columns)))
    sheet = Image.new("RGB", (tile_width * columns, tile_h * rows), (245, 245, 240))
    for idx, img in enumerate(images):
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_h
        sheet.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _candidate_to_json(cand: FrameCandidate) -> dict[str, Any]:
    return {
        "frame_idx": cand.frame_idx,
        "time_sec": round(cand.time_sec, 3),
        "kind": cand.kind,
        "stage_id": cand.stage_id,
        "attempt": cand.attempt,
        "status": cand.status,
        "source": cand.source,
        "motion_score": cand.motion_score,
        "motion_level": cand.motion_level,
        "stage_name": cand.stage_name,
        "evidence": cand.evidence,
        "repair": cand.repair,
        "checks": cand.checks,
        "success_criteria": cand.success_criteria,
        "files": cand.files,
    }


def _write_markdown(
    out_path: Path,
    *,
    trial_dir: Path,
    video_path: Path,
    stage_trace_path: Path | None,
    record_path: Path | None,
    contact_sheet_path: Path,
    candidates: list[FrameCandidate],
    kite_root_used: str,
    kite_method: str,
) -> None:
    lines = [
        "# KITE Failure Keyframes",
        "",
        f"- Trial: `{trial_dir}`",
        f"- Video: `{video_path}`",
        f"- Stage trace: `{stage_trace_path or ''}`",
        f"- Stage plan record: `{record_path or ''}`",
        f"- KITE root: `{kite_root_used or 'not imported'}`",
        f"- Keyframe method: `{kite_method}`",
        "",
        f"![contact sheet]({contact_sheet_path.name})",
        "",
        "| # | kind | frame | time | stage | status | evidence / feedback | raw | annotated |",
        "|---:|---|---:|---:|---|---|---|---|---|",
    ]
    for idx, cand in enumerate(candidates, start=1):
        raw = Path(cand.files.get("raw", "")).name if cand.files.get("raw") else ""
        ann = Path(cand.files.get("annotated", "")).name if cand.files.get("annotated") else ""
        raw_link = f"[raw](raw_frames/{raw})" if raw else ""
        ann_link = f"[annotated](annotated_frames/{ann})" if ann else ""
        evidence = _shorten(_join_unique(cand.evidence, cand.repair), 190).replace("|", "\\|")
        stage = cand.stage_id
        if cand.attempt is not None:
            stage += f" a{cand.attempt}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    cand.kind,
                    str(cand.frame_idx),
                    f"{cand.time_sec:.2f}s",
                    stage.replace("|", "\\|"),
                    cand.status.replace("|", "\\|"),
                    evidence,
                    raw_link,
                    ann_link,
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_out_dir(trial_dir: Path) -> Path:
    return trial_dir / "kite_failure_keyframes"


def process_trial(
    trial_dir: Path,
    *,
    video_path: Path | None,
    stage_trace_path: Path | None,
    record_path: Path | None,
    out_dir: Path | None,
    kite_root: str | None,
    strategy: str,
    max_keyframes: int,
    stride: int,
    max_stage_successes: int,
    merge_window: int,
) -> Path:
    trial_dir = trial_dir.resolve()
    video_path = (video_path or trial_dir / "video_combined.mp4").resolve()
    stage_trace_path = (stage_trace_path or trial_dir / "stage_trace.json").resolve()
    if not stage_trace_path.exists():
        stage_trace_path = None
    record_path = (record_path.resolve() if record_path else _find_record_json(trial_dir))
    out_dir = (out_dir or _default_out_dir(trial_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _video_meta(video_path)
    kite_candidates, kite_root_used, kite_method = _kite_keyframes(
        video_path,
        meta,
        kite_root=kite_root,
        strategy=strategy,
        max_keyframes=max_keyframes,
        stride=stride,
    )
    stage_candidates = _stage_candidates(
        stage_trace_path,
        record_path,
        meta,
        max_stage_successes=max_stage_successes,
    )
    candidates = _merge_candidates(
        stage_candidates + kite_candidates,
        merge_window=merge_window,
    )
    candidates = _save_candidate_frames(video_path, candidates, out_dir)
    contact_sheet_path = out_dir / "contact_sheet_kite_failure_keyframes.png"
    _make_contact_sheet(candidates, contact_sheet_path)
    payload = {
        "method": {
            "name": "KITE-style keyframe-indexed evidence for CaP-X failure screenshots",
            "kite_repository": "https://github.com/m80hz/kite",
            "kite_root_used": kite_root_used,
            "keyframe_method": kite_method,
            "integration": (
                "KITE motion keyframes are merged with CaP-X stage status, "
                "verification evidence, and repair hints from stage_trace.json."
            ),
        },
        "trial_dir": str(trial_dir),
        "video_path": str(video_path),
        "stage_trace_path": str(stage_trace_path) if stage_trace_path else None,
        "stage_record_path": str(record_path) if record_path else None,
        "video": {
            "frame_count": meta.frame_count,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
        },
        "contact_sheet": str(contact_sheet_path.resolve()),
        "keyframes": [_candidate_to_json(c) for c in candidates],
    }
    _json_dump(out_dir / "kite_failure_keyframes.json", payload)
    _write_markdown(
        out_dir / "index.md",
        trial_dir=trial_dir,
        video_path=video_path,
        stage_trace_path=stage_trace_path,
        record_path=record_path,
        contact_sheet_path=contact_sheet_path,
        candidates=candidates,
        kite_root_used=kite_root_used,
        kite_method=kite_method,
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract KITE-style failure keyframes from CaP-X stage trials."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial-dir", type=Path, help="Trial directory containing video_combined.mp4.")
    group.add_argument("--experiment-dir", type=Path, help="Experiment directory; all trial videos are processed.")
    parser.add_argument("--video", type=Path, help="Override video path for --trial-dir.")
    parser.add_argument("--stage-trace", type=Path, help="Override stage_trace.json path for --trial-dir.")
    parser.add_argument("--record", type=Path, help="Override *_stage_planner_record.json path.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Only valid for one --trial-dir.")
    parser.add_argument("--kite-root", type=str, default=None, help="Path to a local m80hz/kite checkout.")
    parser.add_argument("--strategy", default="mixed", choices=["motion", "uniform", "mixed"])
    parser.add_argument("--max-keyframes", type=int, default=12)
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help=(
            "Frame stride for KITE optical-flow sampling. Larger values are faster "
            "for long robot rollouts; use 2-4 for dense short videos."
        ),
    )
    parser.add_argument("--max-stage-successes", type=int, default=8)
    parser.add_argument("--merge-window", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.experiment_dir and args.out_dir:
        raise SystemExit("--out-dir can only be used with --trial-dir.")
    if args.experiment_dir:
        trial_dirs = _find_trial_dirs(args.experiment_dir.resolve())
        if not trial_dirs:
            raise SystemExit(f"No trial dirs with video_combined.mp4 + stage_trace.json under {args.experiment_dir}")
    else:
        trial_dirs = [args.trial_dir.resolve()]

    outputs: list[Path] = []
    for trial_dir in trial_dirs:
        out = process_trial(
            trial_dir,
            video_path=args.video,
            stage_trace_path=args.stage_trace,
            record_path=args.record,
            out_dir=args.out_dir,
            kite_root=args.kite_root,
            strategy=args.strategy,
            max_keyframes=args.max_keyframes,
            stride=args.stride,
            max_stage_successes=args.max_stage_successes,
            merge_window=args.merge_window,
        )
        outputs.append(out)
        print(f"Wrote KITE failure keyframes: {out}")
    print("Processed trial count:", len(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
