# StageCaP 在 RoboSuite 上的复现说明

本目录是 StageCaP 的可执行复现副本，重点覆盖 CapX 中的 7 个 RoboSuite
任务，同时保留 LIBERO 40-task 的 StageCaP 代码、配置和入口。RoboSuite
实验允许使用只读仿真特权信息，因此不依赖 SAM3；Agent0 和 StageCaP 都接收
特权观测，区别只在执行框架和 API 能力。所有状态变化仍必须由机器人动作产生，
禁止直接修改 MuJoCo 状态或成功标志。

## 1. 复现范围

RoboSuite 任务：

| 任务 | task id | 主要阶段 |
| --- | --- | --- |
| Cube Lifting | `cube_lifting` | 定位、抓取、抬升验证 |
| Cube Stack | `cube_stack` | 抓取、运输、低位释放验证 |
| Cube Restack | `cube_restack` | 移开上层方块、重抓、运输、堆叠 |
| Nut Assembly | `nut_assembly` | 抓取螺母把手、对齐 peg、插入 |
| Spill Wipe | `spill_wipe` | 接近桌面、剩余 marker 聚类、擦拭 |
| Two-Arm Lift | `two_arm_lift` | 双臂预抓、接触验证、同步抬升 |
| Two-Arm Handover | `two_arm_handover` | Arm0 抓取、递送、Arm1 接管、释放 |

归档还包含：

- LIBERO、LIBERO-PRO 和 LIBERO 专用 RoboSuite fork；
- 4 个基础 LIBERO suite 的 40-task StageCaP 批处理入口；
- 可选的 SAM3 源码。RoboSuite 特权复现实验不使用 SAM3；
- 精简后的 RoboSuite 回归报告，不包含模型权重和大体积视频。

## 2. 方法实现

StageCaP 的执行闭环为：

1. `pattern_priors.py` 根据任务语言构建显式 stage machine；
2. `scene_snapshot.py` 从绑定 API 收集只读场景快照；
3. 每个 stage 只允许调用白名单中的观测、机器人原语和审计技能；
4. `clean_code_guard` 和 AST preflight 拒绝仿真器直写、环境对象访问和成功标志注入；
5. 代码经 PyRoKi IK 转换为关节目标，再由 RoboSuite 控制器执行；
6. stage verifier 根据位姿、接触、reward 和官方 success predicate 决定
   `proceed`、`retry_stage` 或 `replan`；
7. 轨迹、stage snapshot、代码和验证证据写入 trial 目录。

关键文件：

| 路径 | 作用 |
| --- | --- |
| `capx/envs/stage_trial.py` | stage-by-stage 执行、验证、重试和 deterministic fast path |
| `capx/integrations/franka/robosuite_stage.py` | 7 个任务的只读诊断与高级动作 API |
| `vlm_planner_cap/pattern_priors.py` | 任务到 stage machine 的规则先验 |
| `vlm_planner_cap/scene_snapshot.py` | 特权只读场景快照 |
| `vlm_planner_cap/schema.py` | API 白名单、禁用符号、stage 数据结构 |
| `scripts/run_robosuite_stagecap_batch.py` | 7 任务统一入口与报告生成 |
| `tests/test_robosuite_stagecap.py` | stage、策略边界、几何和配置回归测试 |

### 特权信息边界

允许读取：

- 物体、handle、peg、末端执行器和 gripper 的精确位姿；
- MuJoCo contact 的只读统计；
- wipe marker 的只读位置与剩余状态；
- reward、官方 success predicate 和任务诊断；
- 物体几何尺寸、抓取候选和任务上下文。

禁止行为：

- 写入 `sim.data.qpos`、`qvel`、`ctrl`；
- 调用 `set_state`、`set_object_pose`、`set_success` 等状态注入接口；
- 直接调用环境内部 `_check_success` 来伪造执行结果；
- 绕过机器人控制器直接移动物体；
- 在生成代码中访问 `env`、`sim`、`model` 或 API 对象内部字段。

因此这里的“特权”只用于观测、规划和验证，不用于改变物理状态。

## 3. 第三方版本

归档内置以下源码，解压后不需要再次 clone：

| 组件 | 版本 / commit | 用途 |
| --- | --- | --- |
| RoboSuite | 1.5.1 / `97292732ed909ac3ae116579fb768607034a4dbd` | 7 个 RoboSuite 任务 |
| LIBERO-PRO | 0.1.0 / `5368540790fde9a18584d64105ec5ac16d8926bd` | LIBERO benchmark 与资产 |
| LIBERO RoboSuite fork | 1.4.0 / `a498b087d4bc5a3981e3d27030d09bc537a537f3` | LIBERO 环境 |
| SAM3 | `6fe87d64a5beb9084923d7a9e002741178635b09` | 可选 LIBERO 感知服务 |
| PyRoKi | `95afccc22658c461ab1042a048ae4e9c24bc2a47` | IK 服务，安装时按 commit 获取 |

LIBERO RoboSuite fork 额外带有
`third_party_patches/libero_robosuite_binding_utils.patch`，用于安全释放未完整初始化的
OpenGL context。该补丁不改变物理、控制器或 success predicate。

注意：RoboSuite 1.5.1 和 LIBERO 的 RoboSuite 1.4.0 不能同时出现在同一条
运行时搜索路径中。本包提供两个环境脚本进行隔离。

## 4. 解压与校验

```bash
sha256sum -c stagecap_robosuite_libero_repro_20260714_v2.tar.gz.sha256
tar -xzf stagecap_robosuite_libero_repro_20260714_v2.tar.gz
cd stagecap_robosuite_libero_repro_20260714
```

归档内部的 `PACKAGE_MANIFEST.sha256` 可逐文件校验：

```bash
sha256sum -c PACKAGE_MANIFEST.sha256
```

## 5. 环境准备

测试环境：Linux x86_64、Python 3.12.13、MuJoCo 3.5.0、NumPy 1.26.4、
PyTorch 2.8.0、EGL headless rendering。Python 3.10 到 3.12 是项目声明的
支持范围；本归档的最终验证使用 Python 3.12。

系统需要可用的 NVIDIA 驱动和 EGL。Ubuntu / Debian 常用依赖为：

```bash
sudo apt-get install -y libgl1 libegl1 libglfw3 libosmesa6 ffmpeg git
```

使用 `uv` 创建环境：

```bash
uv venv --python 3.12 .venv-stagecap
source .venv-stagecap/bin/activate
uv pip install -r requirements-stagecap-repro.txt
```

`requirements-stagecap-repro.txt` 固定了本次验证使用的关键版本，并固定 PyRoKi
commit。若机器已经有可用的 CapX / LIBERO 环境，可直接复用，在下面命令中设置：

```bash
export PYTHON_BIN=/path/to/venv/bin/python
```

首次启动 PyRoKi 时，`robot_descriptions` 会下载 Panda URDF 和 mesh 到
`~/.cache/robot_descriptions`。离线机器需要预先复制该缓存。默认目标 link 为
`panda_hand`，服务端口为 `8116`。

配置模型服务时只通过环境变量传入 key，不要写入 YAML 或命令历史：

```bash
cp .env.template .env
# 编辑 .env 中的 MIMO_BASE_URL、MIMO_API_KEY、MIMO_MODEL 和 PYTHON_BIN
```

RoboSuite deterministic 复现不需要模型 key；Agent0、StageCaP+KITE 以及模型生成
代码的实验才需要 OpenAI-compatible endpoint。

## 6. RoboSuite 复现

### 6.1 激活正确的 RoboSuite 版本

```bash
source scripts/env_robosuite_stagecap.sh
$PYTHON_BIN -c 'import robosuite; print(robosuite.__version__, robosuite.__file__)'
```

版本必须是 `1.5.1`，路径必须位于本归档的
`capx/third_party/robosuite`。如果显示 1.4.0，说明环境脚本没有生效。

### 6.2 运行回归测试

```bash
$PYTHON_BIN -m pytest -q tests/test_robosuite_stagecap.py
```

本归档预期结果：`42 passed`。Python 3.12 下依赖源码可能产生 2 条 deprecation
warning 和 1 条 vendored RoboSuite `SyntaxWarning`，不影响测试结果。

### 6.3 单任务 smoke test，无 VLM

StageCaP 配置中的 7 个任务都启用了审计过的 deterministic stage fast path。
下面的 endpoint 故意不可用；若代码意外请求模型，实验会直接报错，因而可用于确认
无 VLM 复现路径。

```bash
$PYTHON_BIN scripts/run_robosuite_stagecap_batch.py \
  --condition stagecap \
  --tasks cube_lifting \
  --model stagecap-privileged \
  --server-url http://127.0.0.1:1 \
  --total-trials 1 \
  --num-workers 1 \
  --no-record-video \
  --output-dir outputs/repro_cube_lifting_smoke
```

配置会自动启动并停止 PyRoKi 8116 服务。若已经手工运行该服务，请增加
`--skip-api-servers`。

### 6.4 七任务统一运行，无 VLM

```bash
$PYTHON_BIN scripts/run_robosuite_stagecap_batch.py \
  --condition stagecap \
  --model stagecap-privileged \
  --server-url http://127.0.0.1:1 \
  --total-trials 5 \
  --num-workers 1 \
  --no-record-video \
  --output-dir outputs/repro_robosuite_stagecap_7tasks
```

建议先使用 `num_workers=1` 检查环境，再提高到 2。每个 trial 使用 trial id 作为
reset seed；不同 MuJoCo、驱动或线程调度版本仍可能造成接触任务的微小变化。

报告生成在：

```text
outputs/repro_robosuite_stagecap_7tasks/robosuite_stagecap_report.json
```

### 6.5 Agent0 和 StageCaP+KITE

先设置模型服务：

```bash
export MIMO_BASE_URL=https://your-openai-compatible-endpoint/v1
export MIMO_API_KEY=YOUR_KEY
export MIMO_MODEL=mimo-v2.5
```

Agent0 基线：

```bash
$PYTHON_BIN scripts/run_robosuite_stagecap_batch.py \
  --condition agent0 \
  --total-trials 5 \
  --num-workers 1 \
  --max-tokens 20480 \
  --no-record-video \
  --output-dir outputs/repro_robosuite_agent0
```

StageCaP+KITE：

```bash
$PYTHON_BIN scripts/run_robosuite_stagecap_batch.py \
  --condition stagecap_kite \
  --total-trials 5 \
  --num-workers 1 \
  --output-dir outputs/repro_robosuite_stagecap_kite
```

`agent0` 使用 7 个官方 `*_privileged.yaml`，可读取物体真值位姿，但只绑定 CapX
原有的 `*PrivilegedApi`；它关闭 stage runner、verifier、replan 和 deterministic
fast path，也不能调用 `FrankaRobosuite*StageApi`。批处理入口会在启动前强制检查
这些边界。`stagecap_kite` 保留 stage runner，并在失败阶段加入在线视觉差分反馈。
这两个条件需要真实模型服务。

## 7. LIBERO 复现

### 7.1 激活 LIBERO 运行时

```bash
source scripts/env_libero_stagecap.sh
$PYTHON_BIN - <<'PY'
import robosuite
from libero import benchmark
print(robosuite.__version__, robosuite.__file__)
print(len(benchmark.get_benchmark_dict()))
PY
```

这里 RoboSuite 必须是 `1.4.0`，路径位于
`capx/third_party/libero_dependencies/robosuite`。

### 7.2 只生成 stage plan，不执行机器人

```bash
SUITES=libero_object TASK_IDS=0 \
  bash scripts/reproduce_stagecap.sh dry-run
```

### 7.3 单任务 smoke test

先在一个终端启动最小服务：

```bash
source scripts/env_libero_stagecap.sh
bash scripts/launch_stagecap_services.sh minimal
```

再在另一个终端运行：

```bash
source scripts/env_libero_stagecap.sh
export MIMO_API_KEY=YOUR_KEY
SUITES=libero_object TASK_IDS=0 TOTAL_TRIALS=1 NUM_WORKERS=1 \
  bash scripts/reproduce_stagecap.sh smoke-libero
```

### 7.4 官方 40 tasks

基础 40 tasks 来自 `libero_10`、`libero_object`、`libero_spatial`、
`libero_goal`，每个 suite 10 个任务：

```bash
source scripts/env_libero_stagecap.sh
export MIMO_API_KEY=YOUR_KEY
TOTAL_TRIALS=1 NUM_WORKERS=1 \
  bash scripts/reproduce_stagecap.sh libero40
```

该命令可能运行数小时。报告位置为：

```text
outputs/reproduce_stagecap/libero40_<timestamp>/libero_stage_planner_report.json
```

RoboSuite 特权任务不需要 SAM3；LIBERO 默认也可通过 `oracle_readonly` 和 clean
skill API 在无 SAM3 模式运行。只有显式开启 `--use-sam3-perception` 时才需要启动
SAM3 服务和准备模型权重。

## 8. 已验证结果

### 8.1 公平 privileged Agent0 smoke

2026-07-14 使用 `mimo-v2.5`、`max_tokens=20480`、单 worker、每任务 1 trial
完成了公平基线。7 个生成配置均为 `privileged: true`，且没有 StageCaP API：

| 任务 | Agent0 | 最终 reward |
| --- | ---: | ---: |
| Cube Lifting | 1/1 | 1.000 |
| Cube Stack | 1/1 | 1.000 |
| Cube Restack | 0/1 | 0.024 |
| Nut Assembly | 0/1 | 0.353 |
| Spill Wipe | 1/1 | 1.000 |
| Two-Arm Lift | 0/1 | 0.015 |
| Two-Arm Handover | 0/1 | 0.094 |

总计为 `3/7 (43%)`。同一代码版本的完整 deterministic StageCaP 统一 7-task
smoke 为 `7/7`。这只是功能 smoke，不是统计显著性结论；正式论文表格仍需执行
相同 seed、相同 trial 数的多次配对实验。完整协议、失败分析和报告路径见
`ROBOSUITE_FAIR_COMPARISON_20260714.md`。

归档内的紧凑原始报告位于：

```text
repro_results/robosuite/mimo_agent0_privileged_all7_1trial_20260714.json
repro_results/robosuite/stagecap_all7_1trial_post_contract_20260714.json
```

旧目录 `outputs/mimo_agent0_all7_1trial_20260714` 的 `5/7` 运行绑定了
`FrankaRobosuite*StageApi`，只能标为“StageCaP API without stage loop”消融，不能
作为 Agent0 基线。

### 8.2 StageCaP 任务级回归

修复后的 5-seed 回归结果：

| 任务 | 成功数 | TSR |
| --- | ---: | ---: |
| Cube Lifting | 5/5 | 100% |
| Cube Stack | 5/5 | 100% |
| Cube Restack | 5/5 | 100% |
| Nut Assembly | 5/5 | 100% |
| Spill Wipe | 5/5 | 100% |
| Two-Arm Lift | 5/5 | 100% |
| Two-Arm Handover | 5/5 | 100% |

这些数字来自最后一次任务级回归的组合，不是一条同时间启动的统一 35-trial run。
其中 Nut Assembly 的额外 10-seed 压力测试为 `9/10`，说明接触和插入仍存在随机
尾部失败。归档保留的原始小型报告位于 `repro_results/robosuite/`。

关键原始报告：

- `stagecap_remaining_5tasks_final_5trial_v1.json`：24/25，handover 修复前；
- `stagecap_handover_5trial_dynamic_handle_v2.json`：handover 5/5；
- `stagecap_nut_10trial_v7.json`：nut 9/10；
- `robosuite_stagecap_privileged_5seed_v2_20260713.json`：two-arm lift 5/5；
- `robosuite_stagecap_privileged_5seed_20260713.json`：cube stack 5/5。

## 9. 可复现性注意事项

- 固定代码和 seed 不代表跨 MuJoCo / GPU driver 的 bitwise deterministic；接触、
  抓取和插入可能出现小幅差异；
- `record_video=True` 会显著增加渲染开销，先用 `--no-record-video` 做功能验证；
- PyRoKi 端口默认 8116。端口已占用时使用 `--skip-api-servers` 复用现有服务，或先
  关闭旧进程；
- deterministic StageCaP 不消耗 VLM 额度；Agent0、KITE 和缺少 fast path 的
  fallback 会消耗模型额度；
- API key、模型权重、虚拟环境、Git 历史、缓存、论文 PDF 和实验视频未写入归档；
- 归档中的所有第三方源码都保留各自 license，研究发布前仍需逐项核对许可证要求。
