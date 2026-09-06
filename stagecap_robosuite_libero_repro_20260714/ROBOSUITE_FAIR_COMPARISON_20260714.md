# RoboSuite StageCaP 公平对比记录（2026-07-14）

## 实验协议

三种条件必须分开报告：

| 条件 | 特权观测 | API | Stage loop / verifier | 确定性 audited skill |
| --- | --- | --- | --- | --- |
| Privileged Agent0 | 是 | CapX 原有 `*PrivilegedApi` | 否 | 否 |
| StageCaP API without stage loop | 是 | `FrankaRobosuite*StageApi` | 否 | 模型可调用 |
| Full StageCaP | 是 | `FrankaRobosuite*StageApi` | 是 | 是 |

Agent0 可以读取精确物体、handle、peg 和末端位姿，但不能接收 StageCaP 的 stage
machine、stage snapshot、verifier、retry/replan 逻辑或高级 audited skill。Agent0
核心执行器和 LLM client 未做任务特化修改。

批处理入口现在会拒绝以下无效配置：

- `privileged` 不为 `true`；
- Agent0 未绑定 CapX 原有 privileged API；
- Agent0 绑定任何 `*StageApi`；
- StageCaP 未绑定 audited `*StageApi`。

## 单 trial smoke 结果

公共设置：RoboSuite 1.5.1、`mimo-v2.5`、`max_tokens=20480`、temperature 1.0、
单 worker、关闭视频、每任务 1 trial。Agent0 每个任务都进行了真实模型调用。

| 任务 | Privileged Agent0 | Reward | 模型查询 | Full StageCaP |
| --- | ---: | ---: | ---: | ---: |
| Cube Lifting | 1/1 | 1.000 | 19.30 s | 1/1 |
| Cube Restack | 0/1 | 0.024 | 42.92 s | 1/1 |
| Nut Assembly | 0/1 | 0.353 | 43.18 s | 1/1 |
| Cube Stack | 1/1 | 1.000 | 30.85 s | 1/1 |
| Two-Arm Lift | 0/1 | 0.015 | 59.23 s | 1/1 |
| Spill Wipe | 1/1 | 1.000 | 41.98 s | 1/1 |
| Two-Arm Handover | 0/1 | 0.094 | 76.28 s | 1/1 |
| **总计** | **3/7 (43%)** |  |  | **7/7 (100%)** |

Full StageCaP 使用同一代码版本的 deterministic audited action path，故没有请求
模型。这个对比证明配置和执行链可运行，但每任务只有 1 个 trial，不能据此声称
统计显著性。

## Agent0 失败分析

- Restack 直接尝试抓取被绿色方块覆盖的红色方块，没有先移开绿色方块；
- Nut Assembly 采用自行推导的位姿变换，执行异常且没有插入后验证；
- Two-Arm Lift 自行构造双臂姿态，未形成稳定双侧抓取；
- Handover 生成了不存在的 `from your_api import *`，sandbox 返回非零状态。

这些失败对应 StageCaP 要解决的问题：显式阶段依赖、受限动作合同、执行后验证和
局部重试，而不是给 Agent0 暗中增加任务技能。

## 结果标签修正

`outputs/mimo_agent0_all7_1trial_20260714` 的历史 `5/7` 运行虽然关闭了 stage
runner，却仍加载 StageCaP privileged YAML 和 `FrankaRobosuite*StageApi`。正确
标签是 **StageCaP API without stage loop ablation**，不是 Agent0。

## 原始产物

- 公平 Agent0：
  `outputs/mimo_agent0_privileged_all7_1trial_20260714_v3/robosuite_stagecap_report.json`
- 完整 StageCaP：
  `outputs/stagecap_all7_1trial_post_contract_20260714/robosuite_stagecap_report.json`
- 历史 API/no-loop 消融：
  `outputs/mimo_agent0_all7_1trial_20260714/robosuite_stagecap_report.json`
- 最终归档回归测试：`42 passed, 3 warnings`。

复现归档另保留前两项的紧凑 JSON 副本于 `repro_results/robosuite/`。
公平 Agent0 的逐任务生成配置、程序和 summary 位于
`repro_results/robosuite/mimo_agent0_privileged_all7_1trial_20260714_artifacts/`。

下一步应按相同 seed 列表对 Privileged Agent0、StageCaP model path 和 Full
StageCaP 各运行至少 10 trials/task，并报告 bootstrap 置信区间及配对成功差异。
