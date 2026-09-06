# Stage-CAP 特权仿真数据构造复现说明

这份交付包用于复现我们的 **Stage-CAP 自动仿真训练数据构造方案**。它的目标不是公平复现 CAP-X agent0，也不是证明纯视觉策略能力，而是利用仿真环境中可读取的只读真值信息，自动生成高质量机器人操作轨迹，供后续 VLA/SmolVLA 等模型训练使用。

核心思想：

```text
任务语言
  -> 阶段规划
  -> MIMO 生成每个阶段的 CAP 代码
  -> 机器人在 LIBERO 中真实执行动作
  -> 每阶段打印 STAGE_STATUS 做边界核验
  -> 失败时保存关键失败帧 / KITE 风格反馈
  -> 成功轨迹保存为 steps.jsonl + 多视角图片 + stage trace
```

## 1. 当前默认模式

本包默认运行的是：

```text
Stage-CAP + Online KITE + oracle-readonly privileged diagnostics
```

也就是 **特权仿真数据构造模式**。

允许：

- 读取物体 pose、目标区域、BDDL 关系诊断。
- 读取抽屉把手位置、抽屉开合进度、接触信息。
- 读取任务完成状态、reward、stage 诊断信息。
- 使用经过审计的机器人动作技能，例如抓取、运输、释放、开关抽屉、关微波炉、开炉灶等。

禁止：

- 直接设置物体位置。
- 直接设置关节状态。
- 直接设置 reward / task success。
- 使用 `open_cabinet_drawer`、`pick_and_place_object` 这类一键 oracle 动作。
- 在主模式中暴露 SAM3 / Molmo / point-prompt / agent0 视觉定位链路。

换句话说：**真值只读信息可以用于规划和核验，但所有场景变化必须来自机器人动作。**

## 2. 目录结构

关键目录：

```text
capx/
  LIBERO 环境、机器人 API、阶段执行器、轨迹记录器。

vlm_planner_cap/
  阶段规划 schema、任务模板、stage loop、prompt 约束。

scripts/
  统一复现入口、批量评测、KITE 失败帧抽取、VLA 数据检查与转换脚本。

third_party/kite/
  vendored m80hz/kite，用于失败帧关键帧抽取。

examples/success_cap/
  一些成功 CAP 代码样例，可用于理解生成代码形态，但不会自动注入评测 prompt。

docs/
  方法边界、KITE/反馈、VLA 转换等说明文档。

current_experiment_snapshot/
  当前这轮 LIBERO40 的轻量部分结果快照，只包含 report/log，不包含大轨迹图片和视频。
```

本压缩包不包含完整 4GB+ 实验输出。完整轨迹通常保存在运行机的：

```text
outputs/reproduce_stagecap/<run_name>/
```

## 3. 环境准备

进入目录：

```bash
cd /path/to/stage_cap_privileged_data_generation_repro_20260629
```

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，至少确认：

```text
PYTHON_BIN=/path/to/libero/python
MIMO_BASE_URL=https://...
MIMO_API_KEY=...
MIMO_MODEL=mimo-v2.5
MUJOCO_GL=egl
```

原始机器上使用过的 Python 是：

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python
```

加载环境：

```bash
source scripts/env_stagecap.sh
```

检查 Python 包和服务：

```bash
bash scripts/reproduce_stagecap.sh check
```

如果要严格检查本地服务端口：

```bash
STRICT_SERVICES=1 bash scripts/reproduce_stagecap.sh check
```

## 4. 依赖和服务

Python 依赖参考：

```text
requirements-stagecap-runtime.txt
requirements-kite-vendored.txt
third_party/kite/requirements.txt
```

建议优先复用已经装好 LIBERO/CAP-X 的 Python 环境，因为 MuJoCo、LIBERO、robosuite、jax/torch 等依赖较重。

常用本地服务：

```text
PyRoKi:   127.0.0.1:8116
GraspNet: 127.0.0.1:8115
SAM3:     127.0.0.1:8114
```

主模式不依赖 SAM3 / Molmo。SAM3 只用于其他视觉定位 ablation 或素材诊断。

启动服务：

```bash
bash scripts/launch_stagecap_services.sh minimal
```

或：

```bash
bash scripts/launch_stagecap_services.sh default
```

服务脚本通常会占用当前终端，建议单独开一个终端运行服务。

## 5. 统一复现入口

入口脚本：

```bash
bash scripts/reproduce_stagecap.sh help
```

常用模式：

| 模式 | 作用 |
| --- | --- |
| `check` | 检查环境变量、Python 包和服务端口 |
| `dry-run` | 只生成 stage plan 和 config，不执行机器人 |
| `smoke-libero` | 跑一个 LIBERO 小任务 |
| `libero40` | 跑 LIBERO 40 个基础任务 |
| `sfb1-smoke` | 跑两个 SFB1 场景编辑任务 |
| `sfb1` | 跑完整 SFB1 场景编辑任务 |
| `inspect-sfb1` | 检查 SFB1 输出并生成 VLA manifest |

## 6. 最小验证流程

先做 dry-run：

```bash
bash scripts/reproduce_stagecap.sh dry-run
```

指定任务 dry-run：

```bash
SUITES="libero_goal" TASK_IDS="3" bash scripts/reproduce_stagecap.sh dry-run
```

跑一个 smoke task：

```bash
bash scripts/reproduce_stagecap.sh smoke-libero
```

指定 smoke task：

```bash
SUITES="libero_goal" TASK_IDS="3" bash scripts/reproduce_stagecap.sh smoke-libero
```

## 7. LIBERO40 复现实验

完整 LIBERO40 包含：

```text
libero_10
libero_object
libero_spatial
libero_goal
```

启动完整复现：

```bash
bash scripts/reproduce_stagecap.sh libero40
```

建议指定输出目录和 GPU：

```bash
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/reproduce_stagecap/libero40_privileged_${RUN_STAMP}" \
CUDA_VISIBLE_DEVICES_ARG=0 \
ROUND_TIMEOUT_SEC=420 \
FEEDBACK_ROUNDS=2 \
MAX_TOTAL_STAGE_ATTEMPTS=10 \
TOTAL_TRIALS=1 \
NUM_WORKERS=1 \
bash scripts/reproduce_stagecap.sh libero40
```

只跑某一组：

```bash
SUITES="libero_object" TASK_IDS="" bash scripts/reproduce_stagecap.sh libero40
```

只跑某些任务：

```bash
SUITES="libero_spatial" TASK_IDS="0 1 2" bash scripts/reproduce_stagecap.sh libero40
```

注意：`TASK_IDS` 会应用到所有传入 suite。如果只想跑一个 suite 的某几个任务，就只传一个 suite。

## 8. 当前默认关键参数

统一脚本默认会打开：

```text
model = mimo-v2.5
oracle_readonly = true
use_stage_verifiable_runner = true
use_kite_feedback = true
use_online_stage_kite_feedback = true
record_video = true
use_wrist_camera = true
allow_point_prompt_molmo = false
use_sam3_perception = false
skip_api_servers = true
round_timeout_sec = 420
feedback_rounds = 2
max_total_stage_attempts = 10
```

其中 `use_wrist_camera=true` 是为了训练数据构造。它会尽量同时保存：

```text
trajectory_images/agentview/*.png
trajectory_images/wrist/*.png
```

## 9. 输出内容

每个 trial 目录通常包含：

```text
code.py
summary.txt
all_responses.json
stage_trace.json
steps.jsonl
steps_metadata.json
video_combined.mp4
trajectory_images/
  agentview/
  wrist/
online_stage_feedback/
kite_failure_keyframes/
```

训练数据核心文件是：

```text
steps.jsonl
trajectory_images/agentview/*.png
trajectory_images/wrist/*.png
```

`steps.jsonl` 每一行包含：

```text
observation.state
next_observation.state
observation.state.raw_joint_gripper
next_observation.state.raw_joint_gripper
observation.images.image
observation.images.image2
action
action.raw_joint_gripper
lerobot_action_candidate
timestamp
frame_index
reward
task_completed
```

说明：

- `action` 是 CAP-X 原始 joint delta + gripper command。
- `lerobot_action_candidate` 是从状态差分推出来的候选 7D 动作，正式训练前还要和目标 LeRobot 控制语义核验。
- `observation.images.image` 对应 agentview。
- `observation.images.image2` 对应 wrist camera。

## 10. 检查输出和成功率

查看 report：

```bash
python3 scripts/summarize_stage_planner_report.py \
  --report-path outputs/reproduce_stagecap/<run_name>/libero_stage_planner_report.json
```

或直接：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reproduce_stagecap/<run_name>/libero_stage_planner_report.json")
r = json.loads(p.read_text())
print(json.dumps(r["summary"], indent=2, ensure_ascii=False))
for run in r["runs"]:
    print(run["suite_name"], run["task_id"], run["task_success"], run["task_language"])
PY
```

检查训练数据文件是否完整：

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/reproduce_stagecap/<run_name> \
  --output-jsonl outputs/reproduce_stagecap/<run_name>/vla_trials_manifest.jsonl \
  --output-csv outputs/reproduce_stagecap/<run_name>/vla_trials_manifest.csv
```

这个脚本会检查：

- 是否有 `steps.jsonl`
- 是否有 agentview 图片
- 是否有 wrist 图片
- 是否成功完成任务
- 是否可作为 gold / partial / failed 轨迹

## 11. 转换为 LeRobot / VLA 数据

先生成 manifest：

```bash
python3 scripts/inspect_stagecap_vla_trials.py \
  outputs/reproduce_stagecap/<run_name> \
  --output-jsonl outputs/reproduce_stagecap/<run_name>/vla_trials_manifest.jsonl \
  --output-csv outputs/reproduce_stagecap/<run_name>/vla_trials_manifest.csv
```

再转换：

```bash
python3 scripts/convert_stagecap_steps_to_lerobot.py \
  --manifest-jsonl outputs/reproduce_stagecap/<run_name>/vla_trials_manifest.jsonl \
  --repo-id local/stagecap_libero40_privileged \
  --root /path/to/lerobot/datasets/local/stagecap_libero40_privileged \
  --include-quality gold
```

如果要转 HuggingFace VLA parquet，可参考：

```bash
python3 scripts/convert_stagecap_manifest_to_hfvla_parquet.py --help
```

## 12. KITE / 失败帧反馈

在线运行时，如果某个阶段失败，系统会保存：

```text
online_stage_feedback/
kite_failure_keyframes/
```

离线抽取失败关键帧：

```bash
python3 scripts/extract_kite_failure_keyframes.py \
  --trial-dir outputs/reproduce_stagecap/<run_name>/<suite>/<task>/round_XX/mimo-v2.5/run/trial_...
```

批量从 experiment dir 中寻找可处理 trial：

```bash
python3 scripts/extract_kite_failure_keyframes.py \
  --experiment-dir outputs/reproduce_stagecap/<run_name>
```

KITE vendored 源码位于：

```text
third_party/kite/
```

## 13. 当前实验快照

本包包含一个轻量快照：

```text
current_experiment_snapshot/
  libero40_privileged_partial_report_20260629_193812.json
  libero40_privileged_partial.log
```

该快照来自当前正在跑的 LIBERO40 复现实验，不包含大图像和视频目录。

截至快照写入时：

```text
已完成任务: 20 / 40
成功任务: 17 / 20
已完成部分成功率: 85%
libero_object: 10 / 10
forbidden_symbol_violation_count: 0
```

后续完整结果以运行结束后的 `libero_stage_planner_report.json` 为准。

## 14. 常见问题

### 14.1 report 里有 timeout_run_count，是否表示任务失败？

不一定。当前脚本有外层 `round_timeout_sec`，某些 round 会被 420s 截断；有些任务虽然 round 标记过 timeout，但 trial 已经产生成功轨迹并被计入 success。最终应以每个 run 的 `task_success`、trial 目录名中的 `reward_1.000_taskcompleted_1`、以及 `summary.txt` 为准。

### 14.2 为什么不暴露 SAM3 / Molmo？

这版主线是特权仿真数据构造。我们允许读仿真只读诊断，因此不需要 agent0 的视觉定位链路。去掉 SAM3/Molmo 可以让方法边界更清楚：阶段规划和失败反馈用真值诊断做 grounding 和 verification，动作仍由机器人执行。

### 14.3 是否用了 deterministic fast-path？

默认没有。正式复现不走 deterministic fast-path。方法依赖阶段规划、阶段 CAP 代码生成、边界核验、失败帧/状态反馈和重规划。

### 14.4 成功 CAP 示例会不会注入 prompt？

不会。`examples/success_cap/` 只是交付给合作者阅读的样例，用于理解生成代码风格和 API 使用方式。批量评测不会把这些成功代码作为答案插入 prompt。

### 14.5 这套数据是否可以直接训练？

需要先筛选 gold 轨迹，并核验 action 语义。`steps.jsonl` 已经保存状态、动作和图片路径；`lerobot_action_candidate` 是候选动作表示，正式训练前建议用目标训练框架检查控制语义、频率、图像同步和 gripper 定义。

## 15. 推荐复现顺序

建议接手同学按下面顺序走：

```text
1. source scripts/env_stagecap.sh
2. bash scripts/reproduce_stagecap.sh check
3. bash scripts/reproduce_stagecap.sh dry-run
4. SUITES="libero_object" TASK_IDS="0" bash scripts/reproduce_stagecap.sh smoke-libero
5. SUITES="libero_object" TASK_IDS="" bash scripts/reproduce_stagecap.sh libero40
6. bash scripts/reproduce_stagecap.sh libero40
7. python3 scripts/inspect_stagecap_vla_trials.py outputs/reproduce_stagecap/<run_name> ...
8. python3 scripts/convert_stagecap_steps_to_lerobot.py ...
```

先跑 `libero_object` 是因为当前实验中这组任务最稳，适合验证整个管线。

