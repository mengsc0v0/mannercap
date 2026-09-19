# D0 质量启动集：64 条开发轨迹结果

日期：2026-09-19  
执行代码：`1b07fed6bd776dd7ef0e6601df4ff342eeda36e9`  
补采预注册：`reports/gate2/d0_extension_seeds45_49_plan.json`

结论：**D0 已达到方案建议的约 60 条质量启动轨迹规模，执行、配对和方向检查通过；全部数据仍属于 development，正式方式阈值、安全阈值和晋级状态均未冻结。**

## 覆盖与执行审计

最终矩阵为 `2 个任务 × 4 个方式标签 × 8 个独立初态 = 64 条轨迹`：

- 任务：Lift、Stack；
- 方式：neutral、slow、fast、fast approach / slow contact；
- seed：42--49；
- 每个 task/seed 的四种方式都从同一个未推进的仿真状态通过 POSIX fork 写时复制执行。

seed 45--49 补采目录：
`/data/gaochuanhao/mannercap_runs/gate2_d0_manner_extension_seeds45_49_20260919`。

| 项目 | seed 45--49 补采 | D0 合计 |
| --- | ---: | ---: |
| 计划 / 观测 / 执行 | 40 / 40 / 40 | 64 / 64 / 64 |
| 执行错误 | 0 | 0 |
| 教师任务终态成功 | 40 / 40 | 64 / 64 |
| 精确同初态方式组 | 10 / 10 | 16 / 16 |
| 外部 API 调用 | 0 | 0 |

每个补采 seed 的 `audit.json` 均记录 8/8 执行、0 error、8/8 教师任务终态、
2/2 精确同初态方式组。运行清单记录干净的代码提交 `1b07fed`。

`task_completed=true` 只表示教师执行器达到任务终态，不代表方式或安全正式通过。

## 配对速度比

下表按每个 task/seed 将指定方式与同初态 neutral 配对。共有 16 个独立配对单元。
数值是开发期描述统计，不是接受阈值。

| 配对量 | n | 均值 | 最小 | 最大 | 预期方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| slow 全程 / neutral | 16 | 0.4073 | 0.3955 | 0.4358 | 低于 1，16/16 |
| fast 全程 / neutral | 16 | 1.3745 | 1.3276 | 1.3983 | 高于 1，16/16 |
| 组合 approach / neutral | 16 | 1.0263 | 1.0050 | 1.0369 | 高于 1，16/16 |
| 组合 descend / neutral | 16 | 0.4515 | 0.4315 | 0.4665 | 低于 1，16/16 |
| 组合 lift / neutral | 16 | 0.4078 | 0.3910 | 0.4323 | 低于 1，16/16 |

五项预注册方向检查均通过。组合方式的 approach 差异较小，最小配对比只有
1.0050；在独立校准给出最小可分辨变化前，只能说方向一致，不能说阶段方式已合格。

## 证据边界

64 条轨迹可用于字段质量、回放检查、方差估计和人工金标候选池。它们不能用于：

- 从执行器自身输出反推正式速度、力或安全阈值；
- 声称 64/64 方式通过或零安全违规；
- 充当生成器封存测试或学生封存测试；
- 证明 MannerCAP 优于 G0--G3/M；
- 证明 LA4VLA 泛化或闭环任务性能。

当前状态保持：`formal_manner_status=unknown`、`formal_safety_status=unknown`、
`formal_thresholds_frozen=false`、`promotion_allowed=false`。

## 固化工件

- 64 条汇总：`reports/gate2/d0_development_summary_64ep_20260919.json`，文件 SHA-256
  `f4787915011e937a139a2adf70f27de258ba0d9c4c6c1271e77cfab04d2e835f`；
- 补采证据清单：`reports/gate2/d0_extension_seeds45_49_20260919.sha256`，清单自身
  SHA-256 `b110d1695e5fcce090a3d3dd90423d773e3a81bcb800a9fcd4a8b720e97ff73a`；
- 原 24 条汇总：`reports/gate2/d0_development_summary_20260918.json`，文件 SHA-256
  `5fb45fc735dfdddd55b2fb1a8907c5fb13a9762f6cca5b31f1b79f450ea35a68`；
- 原始运行保存在 4090 的 `/data/gaochuanhao/mannercap_runs`，不提交大文件到 Git。

下一步是生成不含执行器预测标签的人工金标候选清单，并由独立评测角色完成至少
20 条双人标注与仲裁。独立物理校准、正式 split、资源预算和 sealed 审计完成前，
不运行 E1--E4 或 D1/D2 长任务。

候选清单现已生成：64 条全部进入候选池，其中每个 `task × manner` 单元确定性抽取
3 条，共 24 条双人独立标注；状态为 `pending_human_annotation`。清单 SHA-256 为
`40a4b944d604e25487c71a3ba53d594198412157d1caac9036fba7ce35963b02`。
