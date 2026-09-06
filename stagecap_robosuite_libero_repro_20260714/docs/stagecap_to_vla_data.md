# Stage-CAP LIBERO Trajectories to VLA Training Data

This note describes how to recollect Stage-CAP LIBERO trajectories as VLA training data and convert them into a LeRobot-compatible dataset.

## Why Re-run Is Required

Old `video_combined.mp4` files are useful for papers and qualitative inspection, but they are not enough for VLA training. A VLA dataset needs synchronized:

- RGB observations from `agentview`
- RGB observations from `robot0_eye_in_hand`
- proprioceptive state
- low-level action
- task language

The Stage-CAP runner now saves these fields at trial end:

```text
steps.jsonl
steps_metadata.json
trajectory_images/agentview/frame_*.png
trajectory_images/wrist/frame_*.png
stage_trace.json
video_combined.mp4
```

## Camera And Feature Schema

We align with `HuggingFaceVLA/libero`:

```text
observation.images.image   <- agentview RGB, 256x256x3
observation.images.image2  <- robot0_eye_in_hand RGB, 256x256x3
observation.state          <- 8D eef_pos + eef_axis_angle + gripper_qpos
action                     <- 7D relative end-effector delta candidate + gripper
fps                        <- 10
```

The current Stage-CAP recorder writes both the raw Cap-X action and a `lerobot_action_candidate`. For the first LeRobot conversion we use `lerobot_action_candidate`, but keep `steps_metadata.json` and `stagecap_conversion_manifest.json` for auditability.

## Smoke Recollection

Run one or two tasks first and confirm `steps.jsonl` appears.

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation
export PYTHONPATH=.
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PYTHON_BIN=/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python

${PYTHON_BIN} scripts/run_libero_stage_planner_batch.py \
  --custom-task-manifest outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json \
  --task-ids 6 9 \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_vla_recollect_smoke_$(date +%Y%m%d_%H%M%S) \
  --use-stage-verifiable-runner \
  --record-video True \
  --use-wrist-camera True \
  --feedback-rounds 1 \
  --total-trials 1 \
  --num-workers 1 \
  --round-timeout-sec 420 \
  --max-total-stage-attempts 10 \
  --use-kite-feedback \
  --use-online-stage-kite-feedback \
  --oracle-readonly
```

Notes:

- `--use-wrist-camera True` is required for `observation.images.image2`.
- `--record-video True` also enables the frame buffers used by `trajectory_images/`.
- For training-data construction, `oracle-readonly=True` is allowed if this dataset is explicitly marked as privileged data generation. For clean method comparison, use the non-privileged configuration separately.

## Inspect Readiness

After recollection:

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/stagecap_vla_recollect_smoke_YYYYMMDD_HHMMSS \
  --output-jsonl outputs/stagecap_vla_recollect_smoke_manifest.jsonl \
  --output-csv outputs/stagecap_vla_recollect_smoke_manifest.csv
```

Quality tiers:

```text
gold                       successful, complete recording, no privileged finalizer
weak_privileged_finalize   successful but used privileged finalizer; keep separate
silver_privileged_finalize successful privileged trajectory if explicitly allowed
reject_failed_task         task did not finish
reject_incomplete_recording missing steps/images/state/action
```

For the first SmolVLA training smoke test, use only `gold`.

## Convert To LeRobot

Run the converter in a Python environment that can import LeRobot and its dataset dependencies.

```bash
python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl outputs/stagecap_vla_recollect_smoke_manifest.jsonl \
  --repo-id local/stagecap_libero_sfb1_smoke \
  --root /home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_smoke \
  --include-quality gold \
  --overwrite
```

If you intentionally want to include privileged finalizer trajectories:

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/stagecap_vla_recollect_batch \
  --allow-privileged-finalize \
  --output-jsonl outputs/stagecap_vla_recollect_batch_manifest.jsonl

python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl outputs/stagecap_vla_recollect_batch_manifest.jsonl \
  --repo-id local/stagecap_libero_sfb1_privileged \
  --root /home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_privileged \
  --include-quality gold silver_privileged_finalize \
  --overwrite
```

Do not mix privileged and non-privileged data silently. The conversion manifest records `has_privileged_finalize` for every episode.

## Full SFB1 Recollection

Once the smoke run passes:

```bash
${PYTHON_BIN} scripts/run_libero_stage_planner_batch.py \
  --custom-task-manifest outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_vla_recollect_sfb1_$(date +%Y%m%d_%H%M%S) \
  --use-stage-verifiable-runner \
  --record-video True \
  --use-wrist-camera True \
  --feedback-rounds 1 \
  --total-trials 1 \
  --num-workers 1 \
  --round-timeout-sec 420 \
  --max-total-stage-attempts 10 \
  --use-kite-feedback \
  --use-online-stage-kite-feedback \
  --oracle-readonly
```

Then inspect and convert using the commands above.

## First Training Smoke Test

Use the official SmolVLA LIBERO recipe style:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=local/smolvla_stagecap_sfb1_smoke \
  --dataset.repo_id=local/stagecap_libero_sfb1_smoke \
  --policy.input_features=null \
  --policy.output_features=null \
  --env.type=libero \
  --env.task=libero_spatial \
  --steps=10 \
  --batch_size=2 \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=64
```

This command is only a dataloader/model smoke test. Real training should use a larger dataset and a fixed evaluation protocol.
