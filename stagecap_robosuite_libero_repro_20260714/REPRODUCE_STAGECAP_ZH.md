# Stage-CAP 复现说明

这份文档给接手代码的同学使用，目标是用一个统一入口复现 Stage-CAP 的核心流程：

```text
任务语言
  -> 阶段规划
  -> CAP 代码生成
  -> 仿真执行
  -> 阶段验证
  -> Online KITE 失败帧反馈
  -> 成功轨迹 / VLA 数据记录
```

本文档里的默认复现口径是：

```text
Stage-CAP + Online KITE + oracle-readonly privileged diagnostics
```

也就是说，这套复现主要服务于“仿真训练数据自动构造”，不是 clean policy evaluation。系统允许读取仿真中的只读诊断信息，例如物体位姿、目标区域、抽屉开合程度、抽屉把手位置和关系谓词；但生成代码仍必须通过机器人运动 API 执行动作，不能直接修改物体位姿、抽屉关节、reward 或 success flag。

## 1. 入口脚本

统一复现脚本在：

```bash
scripts/reproduce_stagecap.sh
```

查看帮助：

```bash
bash scripts/reproduce_stagecap.sh help
```

常用模式：

| 模式 | 作用 | 是否执行机器人动作 |
| --- | --- | --- |
| `check` | 检查 Python 环境、变量和服务端口 | 否 |
| `dry-run` | 只生成 stage plan 和 config | 否 |
| `smoke-libero` | 跑一个 LIBERO 小任务 | 是 |
| `libero40` | 跑 LIBERO 40 个基础任务 | 是 |
| `sfb1-smoke` | 跑两个 SFB1 场景编辑任务 | 是 |
| `sfb1` | 跑完整 SFB1 任务批次 | 是 |
| `inspect-sfb1` | 检查 SFB1 输出并生成 VLA manifest | 否 |

## 2. 环境准备

进入代码目录：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation_handoff_release_20260628
```

复制环境配置：

```bash
cp .env.example .env
```

编辑 `.env`，至少确认：

```text
PYTHON_BIN=/path/to/your/libero/python
MIMO_BASE_URL=https://...
MIMO_API_KEY=...
MIMO_MODEL=mimo-v2.5
MUJOCO_GL=egl
```

原始机器上测试过的 Python 是：

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python
```

加载环境：

```bash
source scripts/env_stagecap.sh
```

检查环境：

```bash
bash scripts/reproduce_stagecap.sh check
```

如果希望把本地服务端口也作为硬性检查：

```bash
STRICT_SERVICES=1 bash scripts/reproduce_stagecap.sh check
```

## 3. 服务启动

Stage-CAP 运行时通常需要 PyRoKi 运动规划服务；部分抓取动作还会用到 GraspNet。SAM3 只在视觉诊断或 agent0/perception 模式中需要，默认 privileged 数据构造路径不强依赖 SAM3。

启动最小服务：

```bash
bash scripts/launch_stagecap_services.sh minimal
```

启动默认服务：

```bash
bash scripts/launch_stagecap_services.sh default
```

这个命令会占用当前终端并持续运行。建议单独开一个终端跑服务，另一个终端跑复现实验。

常用端口：

```text
PyRoKi:   http://127.0.0.1:8116
GraspNet: http://127.0.0.1:8115
SAM3:     http://127.0.0.1:8114
```

## 4. 最小复现流程

### 4.1 Dry-run：只检查阶段规划和配置生成

这个模式不执行机器人动作，适合确认 LIBERO、任务收集、stage plan 和配置生成是否正常。

```bash
bash scripts/reproduce_stagecap.sh dry-run
```

默认任务是 `libero_object` 的 task 0。输出目录类似：

```text
outputs/reproduce_stagecap/dry_run_YYYYMMDD_HHMMSS
```

也可以指定任务：

```bash
SUITES="libero_goal" TASK_IDS="3" bash scripts/reproduce_stagecap.sh dry-run
```

### 4.2 Smoke test：跑一个 LIBERO 任务

```bash
bash scripts/reproduce_stagecap.sh smoke-libero
```

默认任务：

```text
suite = libero_object
task_id = 0
```

指定任务示例：

```bash
SUITES="libero_goal" TASK_IDS="3" bash scripts/reproduce_stagecap.sh smoke-libero
```

输出目录类似：

```text
outputs/reproduce_stagecap/smoke_libero_YYYYMMDD_HHMMSS
```

## 5. LIBERO40 复现

完整 LIBERO40 包含四个 suite：

```text
libero_10
libero_object
libero_spatial
libero_goal
```

启动完整运行：

```bash
bash scripts/reproduce_stagecap.sh libero40
```

这个实验可能运行很久。默认参数：

```text
model = mimo-v2.5
total_trials = 1
feedback_rounds = 2
max_total_stage_attempts = 10
round_timeout_sec = 420
oracle_readonly = true
use_stage_verifiable_runner = true
use_kite_feedback = true
use_online_stage_kite_feedback = true
deterministic_fast_path = false
```

常用覆盖方式：

```bash
ROUND_TIMEOUT_SEC=600 \
FEEDBACK_ROUNDS=2 \
MAX_TOTAL_STAGE_ATTEMPTS=10 \
CUDA_VISIBLE_DEVICES_ARG=0 \
bash scripts/reproduce_stagecap.sh libero40
```

如果只想跑某个 suite 或任务：

```bash
SUITES="libero_spatial" TASK_IDS="0 1 2" bash scripts/reproduce_stagecap.sh libero40
```

注意：`TASK_IDS` 会应用到传入的每个 suite。如果只想跑某个 suite 的某个 task，只传一个 suite。

## 6. SFB1 场景编辑数据构造复现

SFB1 是我们根据下游模型失败场景整理的小批量 scene-edited LIBERO 任务集。任务 manifest 默认在：

```text
outputs/smolvla_failure_driven_libero_scene_batch/v1/sfb1_cap_manifest.json
```

如果 manifest 不存在，先重新生成：

```bash
python scripts/prepare_sfb1_scene_batch.py
```

### 6.1 SFB1 smoke test

默认跑 task 6 和 task 9：

```bash
bash scripts/reproduce_stagecap.sh sfb1-smoke
```

指定任务：

```bash
TASK_IDS="2" bash scripts/reproduce_stagecap.sh sfb1-smoke
```

### 6.2 完整 SFB1 批次

```bash
bash scripts/reproduce_stagecap.sh sfb1
```

输出目录类似：

```text
outputs/reproduce_stagecap/sfb1_full_YYYYMMDD_HHMMSS
```

### 6.3 检查 SFB1 轨迹并生成 VLA manifest

运行结束后：

```bash
bash scripts/reproduce_stagecap.sh inspect-sfb1 outputs/reproduce_stagecap/sfb1_full_YYYYMMDD_HHMMSS
```

会生成：

```text
vla_trials_manifest.jsonl
vla_trials_manifest.csv
```

质量分级含义：

| 等级 | 含义 |
| --- | --- |
| `gold` | 成功、记录完整、没有 privileged finalizer |
| `silver_privileged_finalize` | 成功但使用了明确标记的 privileged finalizer，只能放入特权数据构造集合 |
| `reject_failed_task` | 任务没有完成 |
| `reject_incomplete_recording` | 缺少可训练的 step/image/state/action 记录 |

## 7. 结果怎么看

每个输出目录下优先看：

```text
experiment_manifest.json
experiment_rules.md
libero_stage_planner_report.json
```

每个任务目录里重点看：

```text
<suite>/<task_slug>/<model>_stage_planner_record.json
<suite>/<task_slug>/round_XX/config.yaml
<suite>/<task_slug>/round_XX/.../trial_*/code.py
<suite>/<task_slug>/round_XX/.../trial_*/stage_trace.json
<suite>/<task_slug>/round_XX/.../trial_*/video_combined.mp4
<suite>/<task_slug>/round_XX/.../trial_*/steps.jsonl
<suite>/<task_slug>/round_XX/.../trial_*/trajectory_images/
```

汇总报告：

```bash
python scripts/summarize_stage_planner_report.py \
  --report-path outputs/reproduce_stagecap/.../libero_stage_planner_report.json
```

主要看：

```text
summary.task_success_count
summary.task_count
summary.forbidden_symbol_violation_count
每个 run 的 task_success
每个 round 的 stop_reason
是否存在 local_success_but_env_failed
```

## 8. 方法边界和禁止事项

正式复现和论文数据构造中允许：

- 使用 `oracle_readonly` 读取仿真只读诊断；
- 使用 Stage-CAP 阶段规划；
- 使用 Online KITE 失败帧反馈；
- 使用通用 task-family verifier 和 action contract；
- 使用 robot motion API 执行动作。

正式复现中禁止：

- 直接设置物体 pose；
- 直接设置 drawer joint / qpos / qvel；
- 直接设置 reward、success flag 或 task_completed；
- 用 deterministic fast-path 结果当作正式成功率；
- 为某个任务写死坐标或写死成功程序；
- 把 privileged data-construction success rate 当成 clean policy evaluation success rate。

## 9. 当前可核验的参考结果

本 release 包里包含一份 SFB1 最终 manifest 摘要：

```text
outputs/stagecap_vla_gold_final_sfb1_20260628/summary.json
```

当前记录：

```text
strict_gold:
  trial_count = 18
  total_valid_steps = 48701

data_construction_ready:
  trial_count = 20
  quality_counts = 18 gold + 2 silver_privileged_finalize
  total_valid_steps = 68498
```

这份 summary 用于对照复现输出是否大体合理。由于模型服务、随机种子、底层规划服务和运行时状态可能不同，重新运行不一定逐字节复现同样轨迹；正式论文实验需要固定配置后重新报告一次冻结结果。

