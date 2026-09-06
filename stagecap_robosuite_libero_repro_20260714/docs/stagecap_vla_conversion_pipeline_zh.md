# Stage-CAP 轨迹到 VLA 训练数据转换说明

本文档说明我们当前的数据生成 pipeline 如何从 Stage-CAP/CAP 轨迹转成 VLA 可训练数据。目标是把 LIBERO 中完成的任务轨迹整理成 LeRobot / SmolVLA 能直接读取的数据格式。

## 1. 总体思路

我们的 pipeline 分成四层：

```text
Stage-CAP 任务执行
  -> 逐步轨迹记录 steps.jsonl
  -> 轨迹质量筛选 inspect
  -> LeRobot 数据集转换
  -> SmolVLA / VLA 微调
```

这里最关键的一点是：**不能只用旧的 `video_combined.mp4` 转训练数据**。视频只有 RGB 画面，没有每一步对应的 action 和 robot state，所以只能做展示、失败帧诊断和论文素材。真正能训练 VLA 的数据必须包含：

```text
image_t
state_t
action_t
task_language
episode/frame index
```

因此我们现在做的是：重新运行 Stage-CAP 成功轨迹，在执行过程中把每一步的 observation/action/state 同步保存下来。

## 2. Stage-CAP 执行层

入口脚本：

```text
scripts/run_stagecap_vla_recollect_sfb1.sh
```

底层调用：

```text
scripts/run_libero_stage_planner_batch.py
```

关键配置：

```text
--use-stage-verifiable-runner
--record-video True
--use-wrist-camera True
--use-kite-feedback
--use-online-stage-kite-feedback
```

其中：

- `--use-stage-verifiable-runner`：使用我们的阶段规划/阶段核验执行逻辑。
- `--record-video True`：保存视频，同时开启 frame buffer，后续用于保存逐帧图像。
- `--use-wrist-camera True`：保存 wrist camera，对齐 LeRobot 的第二视角。
- `--use-kite-feedback`：保留失败帧诊断链路，用于失败分析和后续修复。

一次 smoke 运行示例：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation
TASK_IDS="6 9" bash scripts/run_stagecap_vla_recollect_sfb1.sh
```

全量 SFB1 20 个任务：

```bash
bash scripts/run_stagecap_vla_recollect_sfb1.sh
```

## 3. 逐步轨迹记录层

Stage-CAP trial 结束后，现在会额外保存：

```text
trial_xx_sandboxrc_0_reward_1.000_taskcompleted_1/
  code.py
  raw_response.sh
  all_responses.json
  summary.txt
  stage_trace.json
  video_combined.mp4
  steps.jsonl
  steps_metadata.json
  trajectory_images/
    agentview/
      frame_000000.png
      ...
    wrist/
      frame_000000.png
      ...
```

其中 `steps.jsonl` 是转换 VLA 数据的核心文件。每一行对应一个低层控制步，包含：

```json
{
  "observation.state": [...],
  "next_observation.state": [...],
  "observation.images.image": {
    "path": "trajectory_images/agentview/frame_000000.png"
  },
  "observation.images.image2": {
    "path": "trajectory_images/wrist/frame_000000.png"
  },
  "action": [...],
  "action.raw_joint_gripper": [...],
  "lerobot_action_candidate": [...],
  "timestamp": 0.0,
  "frame_index": 0,
  "task_index": 6,
  "reward": 0.0,
  "task_completed": false
}
```

注意：当前 `action` 是 Cap-X 原始控制 action；我们转 LeRobot 时默认使用：

```text
lerobot_action_candidate
```

它是从前后 `observation.state` 差分得到的 7 维候选动作：

```text
delta eef_pos 3D
delta eef_axis_angle 3D
gripper command 1D
```

这更接近 LeRobot LIBERO 的 VLA action schema。

## 4. 与 LeRobot / HuggingFaceVLA/libero 对齐

官方 LIBERO VLA 数据格式是：

```text
observation.images.image   agentview RGB, 256x256x3
observation.images.image2  robot0_eye_in_hand RGB, 256x256x3
observation.state          8D
action                     7D
fps                        10
task                       natural language instruction
```

所以我们的转换关系是：

| Stage-CAP 记录 | LeRobot 字段 | 说明 |
| --- | --- | --- |
| `trajectory_images/agentview/*.png` | `observation.images.image` | 主视角 |
| `trajectory_images/wrist/*.png` | `observation.images.image2` | 手腕视角 |
| `observation.state` | `observation.state` | 8 维 robot proprioception |
| `lerobot_action_candidate` | `action` | 7 维相对动作 |
| task language / task dir | `task` | 自然语言指令 |

depth、segmentation、point cloud 暂时不进入 VLA 主训练数据。它们保留给 KITE 失败诊断、关键失败帧分析和论文图素材。

## 5. 质量筛选层

检查脚本：

```text
scripts/inspect_stagecap_vla_trials.py
```

用法：

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/stagecap_vla_recollect_sfb1_xxxxxxxx_xxxxxx \
  --output-jsonl outputs/stagecap_vla_recollect_sfb1_xxxxxxxx_xxxxxx/vla_trials_manifest.jsonl \
  --output-csv outputs/stagecap_vla_recollect_sfb1_xxxxxxxx_xxxxxx/vla_trials_manifest.csv
```

检查内容：

- trial 是否成功：`sandbox_rc == 0`
- 任务是否完成：`reward >= 1.0` 且 `task_completed == true`
- 是否存在 `steps.jsonl`
- step 数是否足够
- `observation.state` 是否为 8 维
- `lerobot_action_candidate` 是否为 7 维
- 两路图像是否都存在
- 是否使用了 privileged finalizer

输出的质量等级：

```text
gold
  真正成功，记录完整，未使用 privileged finalizer。

weak_privileged_finalize
  成功但用了 privileged finalizer；不能混入 clean 主训练集。

silver_privileged_finalize
  显式允许 privileged finalizer 时，可作为 privileged data generation 数据。

reject_failed_task
  任务没有完成。

reject_incomplete_recording
  任务可能成功，但缺少 steps/images/state/action，不能训练。
```

第一版训练只建议使用：

```text
gold
```

如果我们要构造“特权数据生成”版本，可以单独转换：

```text
gold + silver_privileged_finalize
```

但必须在数据集名和 manifest 里明确标注。

## 6. 本次已生成的数据与路径

2026-06-28 已完成一版 SFB1 20 个任务的数据整理。最终 manifest 目录：

```text
/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation/outputs/stagecap_vla_gold_final_sfb1_20260628
```

其中有两种口径：

```text
vla_trials_manifest_strict_gold.jsonl
  18 条严格 gold 轨迹，不含 privileged finalizer。

vla_trials_manifest_data_construction_ready.jsonl
  20 条数据构造可用轨迹，其中 18 条 gold，2 条 silver_privileged_finalize。
  带特权 finalizer 的任务是：
  03_caddy_back_compartment
  10_wine_bottle_cabinet_top_easy
```

已经转换出一版严格 gold 的 HuggingFaceVLA/LeRobot v3 风格 parquet 数据集：

```text
/home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_strict_gold_10hz
```

这版数据的设置是：

```text
episodes: 18
frames: 24355
fps: 10
sample_stride: 2
quality: gold only
privileged_finalize: none
views:
  observation.images.image  = agentview
  observation.images.image2 = wrist
```

注意：Stage-CAP 原始记录是 20Hz，每步间隔约 0.05s。为了对齐已有 `HuggingFaceVLA/libero` 的 10Hz 数据，本次 parquet 转换默认使用：

```text
--sample-stride 2 --fps 10
```

下采样后，`action` 会重新按保留帧之间的 `observation.state` 差分计算，而不是直接复用被跳过前的单步 action。这样图像、state、action 的时间间隔是一致的。

直接转换命令：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation

/home/lifuchun/.local/bin/uv run \
  --with pyarrow --with pillow --with numpy \
  python scripts/convert_stagecap_manifest_to_hfvla_parquet.py \
  --manifest-jsonl outputs/stagecap_vla_gold_final_sfb1_20260628/vla_trials_manifest_strict_gold.jsonl \
  --root /home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_strict_gold_10hz \
  --repo-id local/stagecap_libero_sfb1_strict_gold_10hz \
  --sample-stride 2 \
  --fps 10 \
  --overwrite
```

如果要生成“特权数据构造可用”版本，应使用 `vla_trials_manifest_data_construction_ready.jsonl`，并在数据集名中保留 `silver` 或 `privileged` 标记，避免和严格 gold 混用。

## 6. LeRobot 转换层

转换脚本：

```text
scripts/convert_stagecap_steps_to_lerobot.py
```

输入：

```text
vla_trials_manifest.jsonl
```

输出：

```text
/home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_smoke/
  meta/
  data/
  videos/ 或 images/
  stagecap_conversion_manifest.json
```

转换命令：

```bash
python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl outputs/stagecap_vla_recollect_sfb1_xxxxxxxx_xxxxxx/vla_trials_manifest.jsonl \
  --repo-id local/stagecap_libero_sfb1_smoke \
  --root /home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_smoke \
  --include-quality gold \
  --overwrite
```

转换脚本会通过 LeRobot 的官方写入接口：

```text
LeRobotDataset.create()
dataset.add_frame()
dataset.save_episode()
```

这样生成的数据结构与 `HuggingFaceVLA/libero` 对齐，后续更容易接 SmolVLA 训练。

## 7. 训练层

转换完成后，可以用 SmolVLA 做 dataloader smoke test：

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

这个阶段只验证数据能被模型读取，不评估最终性能。

## 8. 当前 pipeline 里各类数据的用途

| 数据 | 是否用于训练 | 用途 |
| --- | --- | --- |
| `video_combined.mp4` | 否 | 展示、GIF、人工检查 |
| `stage_trace.json` | 间接使用 | 阶段规划、成功/失败原因、metadata |
| `kite_failure_keyframes/` | 否 | 失败诊断、论文图、修复反馈 |
| `depth/segmentation/pointcloud` | 暂不用于主训练 | KITE 风格诊断和筛选 |
| `steps.jsonl` | 是 | VLA 转换核心数据 |
| `trajectory_images/agentview` | 是 | VLA 主视角 |
| `trajectory_images/wrist` | 是 | VLA wrist 视角 |
| `stagecap_conversion_manifest.json` | 否 | 数据来源审计、privileged 标注 |

## 9. 推荐执行顺序

第一步，跑 smoke：

```bash
TASK_IDS="6 9" bash scripts/run_stagecap_vla_recollect_sfb1.sh
```

第二步，检查：

```bash
python3 scripts/inspect_stagecap_vla_trials.py <output_dir> \
  --output-jsonl <output_dir>/vla_trials_manifest.jsonl \
  --output-csv <output_dir>/vla_trials_manifest.csv
```

第三步，只转 `gold`：

```bash
python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl <output_dir>/vla_trials_manifest.jsonl \
  --repo-id local/stagecap_libero_sfb1_smoke \
  --root /home/lifuchun/data/lerobot/datasets/local/stagecap_libero_sfb1_smoke \
  --include-quality gold \
  --overwrite
```

第四步，SmolVLA dataloader smoke test。

第五步，确认 smoke 没问题后，跑 SFB1 20 个任务全量收集。

## 10. 需要特别说明的点

1. 旧视频不能直接训练 VLA，必须重新记录 `steps.jsonl`。
2. 训练主数据先只用两个 RGB 视角、state、action。
3. KITE 的 depth/segmentation/point cloud 先用于诊断和筛选，不进入第一版 SmolVLA 主训练。
4. 使用 privileged finalizer 的轨迹必须单独标注，不能和 clean gold 数据混在一起。
5. 当前 action 使用 `lerobot_action_candidate`，后续正式训练前需要再做一次 action 语义 sanity check。
