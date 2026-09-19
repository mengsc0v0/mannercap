# D0 人工金标协议（待执行）

版本：2026-09-19 draft  
候选清单：`reports/gate2/d0_human_gold_candidate_manifest_20260919.json`

## 范围

64 条 D0 轨迹全部进入候选池。每个 `task × manner` 单元有 8 条；清单通过固定
salt 确定性选择每单元 3 条，共 24 条双人独立标注，其余 40 条单人独立标注。
双标分歧必须由第三人仲裁。候选选择不使用任务成功、方式判定、速度统计或安全结果。

## 盲化

标注员可以查看任务、原始指令、目标方式、请求阶段范围、原始双相机图像和轨迹转移。
标注时不得查看 `result.json` 中的以下执行器派生字段：

- `task_completed`；
- `formal_manner_status`、`formal_safety_status`、`promotion_allowed`；
- `execution`、`descriptive_metrics`、`descriptive_phase_metrics`；
- `offline_recomputation`。

候选清单只保留 `result.json` 的 SHA-256 作为 provenance，不把上述字段暴露给标注表。

## 标签

每名标注员独立填写：

| 字段 | 允许值 | 判定问题 |
| --- | --- | --- |
| `task_outcome` | pass / fail / unknown | 从可见证据能否确认完成原任务 |
| `manner_following` | pass / fail / unknown | 整体方式是否与指令一致 |
| `phase_binding` | pass / fail / unknown / not_applicable | 方式是否只作用于指定阶段，且顺序正确 |
| `trajectory_data_quality` | pass / fail / unknown | 图像、状态、动作和时间轴是否足以评审 |
| `confidence` | 1 / 2 / 3 | 低 / 中 / 高置信度 |
| `evidence` | 时间段与简短理由 | 指向原始观测，不引用执行器标签 |

unknown 不计作通过。标注员不得自行创造数值物理阈值；无法从可见数据判断时标 unknown。
安全标签不在这份视觉方式金标里推断，仍由独立物理校准与 sealed audit 负责。

## 双标与仲裁

双标记录先由 A、B 独立提交，双方在提交前不能查看对方答案。四个主标签完全一致时
直接形成一致标签；任一主标签不同则进入仲裁。仲裁员查看原始轨迹、A/B 的理由和
冲突字段，给出最终标签与理由，但不得查看执行器预测结果。

报告至少包含逐字段原始一致率、Cohen's kappa（在类别分布允许时）、unknown 比例、
仲裁比例，以及按 task 和 manner 的分层结果。24 条双标是质量检查下限，不用于证明
统计功效。

## 发布条件

完成后生成单独的 `human_gold_labels.jsonl`、仲裁记录、标注员匿名 ID、协议版本和
SHA-256。B 角色复核清单完整性与盲化记录后，才可把状态从
`pending_human_annotation` 改为已标注。该签收不自动冻结物理方式或安全阈值。
