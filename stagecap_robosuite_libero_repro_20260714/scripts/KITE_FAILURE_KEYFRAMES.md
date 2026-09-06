# KITE Failure Keyframes

This workflow integrates the KITE keyframe-indexed evidence idea into CaP-X
stage-verifiable rollouts.

It is intentionally lightweight:

- Use `m80hz/kite`'s `KeyframeSelector` for optical-flow motion keyframes.
- Merge those visual candidates with CaP-X `stage_trace.json` status, numeric
  verification evidence, and repair hints.
- Export clean screenshots, annotated screenshots, a contact sheet, and
  machine-readable JSON for paper figures.

This handoff package includes a vendored KITE checkout at:

```bash
third_party/kite
```

You can also pass a replacement checkout with `--kite-root`.

## Single Trial

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python \
  scripts/extract_kite_failure_keyframes.py \
  --trial-dir <trial_dir_with_video_combined_mp4>
```

The output is written to:

```bash
<trial_dir>/kite_failure_keyframes/
```

Important files:

- `contact_sheet_kite_failure_keyframes.png`: quick visual review sheet.
- `raw_frames/*.png`: clean screenshot frames for figure composition.
- `annotated_frames/*.png`: screenshots with stage status, evidence, and feedback.
- `kite_failure_keyframes.json`: full metadata for selecting frames programmatically.
- `index.md`: human-readable index.

## Experiment Directory

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python \
  scripts/extract_kite_failure_keyframes.py \
  --experiment-dir <experiment_output_dir>
```

This finds every trial directory containing both `video_combined.mp4` and
`stage_trace.json`.

## Notes

- The default `--stride 16` is chosen for long LIBERO videos. Use a smaller
  stride, such as `--stride 4`, for short clips or denser motion keyframes.
- Stage failure frames are always preserved, even if they are not high-motion
  KITE keyframes.
- The script keeps both failure-feedback frames and verified success frames, so
  a paper method figure can show `failure -> feedback -> corrected/success`
  instead of only a final rollout.
