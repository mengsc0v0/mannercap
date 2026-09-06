# Stage-CAP Simulation Data Generation Handoff

This handoff package contains the code we use for automatic simulation training-data construction with Stage-CAP. The main story is not to reproduce CAP-X under a perfectly fair Agent0 setting. The target use case is different:

> Use simulator-accessible privileged diagnostics, staged code generation, and failure-frame feedback to produce high-quality robot manipulation trajectories for downstream VLA training.

Privileged simulator information is allowed here because this is a data-construction pipeline. The generated program still acts through robot motion APIs; it must not directly set object poses, drawer joints, success flags, reward, or task-completion state.

## What Is Included

```text
capx/
  CAP-X-derived execution, LIBERO simulator wrappers, robot APIs, trial recorder,
  privileged readonly LIBERO diagnostics, and video / step logging.

vlm_planner_cap/
  Stage plan schema, stage loop parsing, prompt templates, task-family priors,
  and planner-facing utilities.

scripts/
  Main entry points for Stage-CAP evaluation, failure-frame extraction, SFB1
  scene generation, VLA trajectory inspection, and LeRobot conversion.

examples/success_cap/
  Recent successful generated CAP code artifacts, with source manifest and
  notes about the no-direct-simulator-mutation screening.

third_party/kite/
  Vendored m80hz/kite source used by `scripts/extract_kite_failure_keyframes.py`
  for motion keyframe selection. If this import fails, our extractor still has
  a fallback motion-keyframe implementation, but the packaged dependency should
  make the KITE path work out of the box.

env_configs/libero/
  LIBERO/CAP-X configs used by Stage-CAP.

docs/
  Method notes, privileged data-generation instructions, KITE/feedback notes,
  and VLA conversion documentation.

outputs/smolvla_failure_driven_libero_scene_batch/v1/
  Small scene-edit artifact: generated BDDL files, metadata, and scene manifest
  for the SFB1 failure-driven LIBERO task batch.

outputs/stagecap_vla_gold_final_sfb1_20260628/
  Small final manifest summary for the SFB1 VLA data collection result. This
  does not include large trajectory videos or frame folders.
```

Large generated videos, trial frame folders, logs, caches, and model assets are intentionally excluded from the release package.

Vendored dependency note:

- `third_party/kite` is copied from `https://github.com/m80hz/kite`.
- Local commit included in this package: `afead04`.
- Heavy example videos and `.git/` metadata are excluded from the tarball.

## Method Overview

The pipeline has four layers.

1. Stage planning

   The task language is converted into explicit stages such as grounding, pick/lift, drawer opening, placement, closing, and final verification. Each stage has an expected boundary condition.

2. CAP code generation and execution

   MIMO generates executable Python CAP code. The code uses robot motion APIs and, in data-generation mode, simulator readonly diagnostics such as object poses, goal regions, drawer handles, drawer progress, and BDDL-style relation checks.

3. Stage verification and Online KITE-style feedback

   After a stage attempt, the runner reads structured `STAGE_STATUS` evidence. If a stage fails, the system preserves failure-frame evidence and feeds compact diagnostics back into the next code generation round. This is useful because we identify failures inside a trajectory rather than only after the whole rollout fails.

4. VLA data recording

   Successful rollouts can be re-run with step recording enabled. The recorder saves synchronized `steps.jsonl`, agent-view images, wrist-view images, robot state, candidate LeRobot action, stage trace, code, and video.

## Recommended Data-Generation Mode

Use this mode for constructing training data:

```text
Stage-CAP + Online KITE + oracle-readonly / privileged diagnostics
```

Recommended defaults:

- `mimo-v2.5`
- `--oracle-readonly`
- `--use-stage-verifiable-runner`
- `--use-kite-feedback`
- `--use-online-stage-kite-feedback`
- `--record-video True`
- `--use-wrist-camera True` when recollecting VLA data
- deterministic fast paths disabled unless explicitly doing a debugging run

## Environment Setup

Before running experiments, prepare the runtime environment:

```bash
cd /path/to/stage_cap_data_generation_handoff_release_20260628
cp .env.example .env
# Fill MIMO_API_KEY and adjust PYTHON_BIN if needed.
source scripts/env_stagecap.sh
"${PYTHON_BIN}" scripts/check_environment.py
```

Use the stricter check before long runs:

```bash
"${PYTHON_BIN}" scripts/check_environment.py --require-mimo-key --strict-services
```

See `ENVIRONMENT.md` for the full setup notes, dependency list, service ports,
and the distinction between bundled code and external simulator/model assets.
The KITE requirements file in this package is intentionally minimal for
failure-frame extraction; the full upstream KITE requirements are kept under
`third_party/kite/requirements.txt` only for reference.

## Reproduction Entry Point

For collaborators, the recommended unified entry point is:

```bash
bash scripts/reproduce_stagecap.sh help
```

Typical first checks:

```bash
bash scripts/reproduce_stagecap.sh check
bash scripts/reproduce_stagecap.sh dry-run
bash scripts/reproduce_stagecap.sh smoke-libero
```

For the full Chinese reproduction guide, see `REPRODUCE_STAGECAP_ZH.md`.

## Quick Start: LIBERO40 Data Construction

```bash
cd /path/to/stage_cap_data_generation_handoff_release_20260628

source scripts/env_stagecap.sh

"${PYTHON_BIN}" \
  scripts/run_libero_stage_planner_batch.py \
  --experiment-label stagecap_privileged_data_generation \
  --suites libero_10 libero_object libero_spatial libero_goal \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_privileged_data_generation_$(date +%Y%m%d_%H%M%S) \
  --total-trials 1 \
  --feedback-rounds 2 \
  --num-workers 1 \
  --record-video True \
  --oracle-readonly \
  --use-stage-verifiable-runner \
  --use-kite-feedback \
  --use-online-stage-kite-feedback \
  --max-total-stage-attempts 10 \
  --round-timeout-sec 420 \
  --save-stage-snapshots \
  --skip-api-servers
```

For a smoke test:

```bash
TASK_IDS="0 1" SUITES="libero_object" bash scripts/run_libero40_codex_boundaries_mimo_kite.sh
```

## Quick Start: SFB1 Scene-Edited Tasks

SFB1 is a small failure-driven task batch created by editing LIBERO BDDL task language and metadata around tasks where SmolVLA struggled.

Generate or refresh the scene batch:

```bash
python3 scripts/prepare_sfb1_scene_batch.py \
  --manifest ../../docs/smolvla_failure_driven_libero_scene_batch_manifest.json \
  --output-root outputs/smolvla_failure_driven_libero_scene_batch/v1
```

The generated CAP manifest is:

```text
outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json
```

Run Stage-CAP on SFB1:

```bash
TASK_IDS="6 9" bash scripts/run_stagecap_vla_recollect_sfb1.sh
```

Run all SFB1 tasks:

```bash
bash scripts/run_stagecap_vla_recollect_sfb1.sh
```

## Inspect And Convert To VLA Data

Inspect completed trials:

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/stagecap_vla_recollect_sfb1_YYYYMMDD_HHMMSS \
  --output-jsonl outputs/stagecap_vla_recollect_sfb1_YYYYMMDD_HHMMSS/vla_trials_manifest.jsonl \
  --output-csv outputs/stagecap_vla_recollect_sfb1_YYYYMMDD_HHMMSS/vla_trials_manifest.csv
```

Convert gold trajectories to LeRobot:

```bash
python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl outputs/stagecap_vla_recollect_sfb1_YYYYMMDD_HHMMSS/vla_trials_manifest.jsonl \
  --repo-id local/stagecap_libero_sfb1 \
  --root /path/to/lerobot/datasets/local/stagecap_libero_sfb1 \
  --include-quality gold \
  --overwrite
```

Quality tiers:

- `gold`: successful, complete recording, no privileged finalizer.
- `silver_privileged_finalize`: successful and explicitly allowed privileged finalizer.
- `weak_privileged_finalize`: successful but should not be mixed into clean training unless explicitly intended.
- `reject_failed_task`: task failed.
- `reject_incomplete_recording`: missing steps/images/state/action.

## Key Output Files

Each Stage-CAP output root contains:

- `experiment_manifest.json`
- `experiment_rules.md`
- `libero_stage_planner_report.json`

Each successful trial usually contains:

- `code.py`
- `stage_trace.json`
- `video_combined.mp4`
- `steps.jsonl`
- `steps_metadata.json`
- `trajectory_images/agentview/frame_*.png`
- `trajectory_images/wrist/frame_*.png`

## Important Boundaries

- Privileged diagnostics are allowed for data construction.
- Direct simulator state mutation remains forbidden.
- Do not report privileged data-construction success rates as clean policy evaluation.
- Keep privileged and clean datasets separate in manifest names and dataset names.
- Old `video_combined.mp4` files are not enough for VLA training; re-run with step recording to collect state/action/image pairs.

## Useful Documents

- `docs/privileged_libero_data_generation_zh.md`
- `docs/stagecap_vla_conversion_pipeline_zh.md`
- `docs/stagecap_to_vla_data.md`
- `docs/stage_cap_online_kite_reproducible_method_zh.md`
- `scripts/KITE_FAILURE_KEYFRAMES.md`
