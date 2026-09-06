# Stage-CAP / CAP-X 公平复现实验清单

本文档定义 Stage-CAP、Online KITE 失败帧反馈与 CAP-X Agent0 在 LIBERO-40 上的公平比较方案。核心原则是：**方法设计可以不同，但原始可获取输入必须对齐**。阶段规划、提示词包装、API contract、失败帧提取与反馈都可以作为 Stage-CAP 的方法组件；但真值位姿、抽屉关节、BDDL predicate、reward/task_completed 等信息不能在 clean 对比中额外提供给我们的方法。

## 1. 总体目标

我们要回答两个问题：

1. 在和 CAP-X Agent0 相同的非特权视觉输入下，Stage-CAP 是否能提高 LIBERO-40 的数据生成成功率？
2. Online KITE 失败帧反馈是否能在单轮任务执行过程中定位子阶段失败，并帮助下一段 CAP 代码修复，而不是只在整条轨迹结束后做失败归因？

因此主实验必须以 **clean / non-privileged / vision-matched** 为主；oracle-readonly 只能作为辅助上限实验，不能和 CAP-X 原始 clean 配置直接比较。

## 2. 统一评测范围

- Benchmark：LIBERO-40，即 `libero_10`、`libero_object`、`libero_spatial`、`libero_goal` 各 10 个任务。
- 每个任务默认 `total_trials=1`，用于 Pass@1 数据生成成功率。
- 模型：优先固定为 `mimo-v2.5`，因为需要支持图像输入。
- 视频：所有组都开启 `record_video=True`，保留成功/失败 gif 或 mp4。
- 超时：每个 task 统一设置相同 `round_timeout_sec` 和 `trial_timeout_seconds`，超时计失败并进入下一个任务。
- 成功判定：以 LIBERO 环境最终 `task_completed/reward` 为统计结果；但 clean 方法内部不能把这些真值作为实时规划输入。

## 3. 禁止混入 clean 对比的输入

clean 对比中，CAP-X 与 Stage-CAP 都不能读取或使用：

- `FrankaLiberoOracleReadOnlyApi`
- `FrankaLiberoCleanSkillApi`
- `FrankaLiberoPrivilegedApi`
- `get_object_pose` 真值物体位姿
- `get_all_object_poses` 真值场景位姿
- `get_goal_target_region_pose` 真值目标区域
- `get_cabinet_drawer_handle_pose` 真值把手位置
- `get_cabinet_drawer_progress` 真值抽屉进度
- BDDL relation diagnostics / predicate diagnostics
- `reward` / `task_completed` 作为中间反馈输入
- `sim.data.qpos`、`.data.qpos`、`set_qpos`、`set_object_pose`、`set_state`
- `satisfy_goal_predicates`
- `open_cabinet_drawer` / `close_cabinet_drawer` 这类直接设置关节状态的特权 helper

clean 对比中允许使用：

- RGB-D observation
- wrist camera RGB-D，如果 CAP-X 同组也启用
- SAM3 segmentation
- Molmo point prompt
- Contact-GraspNet
- PyRoKi / IK / joint trajectory / gripper control
- 由 RGB-D、mask、depth、point cloud 推导出来的几何量
- 执行日志、stdout/stderr、是否到达目标位姿等非真值执行反馈

## 4. 实验组清单

### E0. CAP-X Agent0 Clean Baseline

目的：得到原始 CAP-X 在 LIBERO-40 上的 clean baseline。

配置：

- 基础配置：`/home/lifuchun/data/capx/repos/cap-x/env_configs/libero/franka_libero_cap_agent0_repro.yaml`
- `privileged: false`
- API：`FrankaLiberoApiReducedSkillLibrary`
- 感知：RGB-D + SAM3 + Molmo + GraspNet
- 交互：multi-turn + image differencing
- ensemble/multimodel：按 CAP-X Agent0 原始配置开启

记录：

- 40 task 成功率
- 每个任务最终代码、日志、视频
- 失败类型：感知失败、抓取失败、放置失败、开关抽屉失败、超时、代码错误

### E1. Stage-CAP Clean, No KITE

目的：只评估阶段规划和边界条件本身的收益。

配置：

- `--no-oracle-readonly`
- `--use-sam3-perception`
- `--use-stage-verifiable-runner`
- `--no-use-kite-feedback`
- 不启用 deterministic fast-path
- API 应只包含 `FrankaLiberoApiReducedSkillLibrary`

方法组件：

- 固定的任务阶段规划模板
- 固定的阶段边界检查模板
- 阶段状态打印与局部重规划
- 不使用真值位姿/BDDL/抽屉进度

记录：

- 40 task 成功率
- 每个任务的 stage plan
- 每个阶段的生成代码
- 每个阶段的 `STAGE_STATUS`
- 失败时最后一个未满足阶段

### E2. Stage-CAP + Online KITE Clean

目的：评估单轮内失败帧监测与反馈是否进一步提升。

配置：

- 在 E1 基础上开启 `--use-kite-feedback`
- `--use-online-stage-kite-feedback`
- KITE 输入只能来自同样的 RGB-D/视频/点云/分割结果，不能读取真值状态
- KITE 输出反馈作为下一段 CAP 生成的上下文

方法组件：

- E1 的阶段规划与边界条件
- Online KITE 关键失败帧选择
- 失败帧诊断图，例如 RGB、depth、segmentation、point cloud、局部几何关系
- 通用反馈模板，不针对单个任务手写反馈文字

记录：

- 40 task 成功率
- KITE 触发次数
- 每次触发对应的阶段、关键帧、诊断图、反馈文本
- 修复后是否成功
- 相比 E1 新增成功的任务列表

### E3. CAP-X + KITE Clean Control

目的：验证收益是否只来自 KITE，而不是 Stage-CAP。

配置：

- CAP-X Agent0 clean 输入不变
- 额外接入同样的 Online KITE 失败帧反馈
- 不加入 Stage-CAP 的阶段规划与阶段边界状态机

解释价值：

- 如果 E3 提升有限，而 E2 明显提升，说明 KITE 需要和阶段状态机结合才有效。
- 如果 E3 也明显提升，说明失败帧反馈本身就是主要贡献之一。

记录：

- 40 task 成功率
- 与 E0 的增量成功任务
- KITE 反馈是否被 CAP-X 原始 multi-turn prompt 有效利用

### E4. Stage-CAP Oracle-Readonly Upper Bound

目的：作为方法上限和调试参考，不作为与 CAP-X clean 的主比较。

配置：

- `--oracle-readonly`
- `--use-stage-verifiable-runner`
- `--use-kite-feedback`
- `--use-deterministic-stage-fast-paths=False`
- 允许只读真值坐标、目标区域、抽屉把手和进度
- 禁止直接改状态、禁止 predicate repair

解释边界：

- 这组能说明如果给可靠状态诊断，Stage-CAP + KITE 的阶段闭环可以达到什么水平。
- 不能把这组成功率直接写成“超过 CAP-X Agent0 clean”的公平结果。

## 5. 推荐主表格

最终论文/PPT 主表建议只放以下列：

| 组别 | 真值输入 | 视觉输入 | 阶段规划 | Online KITE | 成功率 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 CAP-X Agent0 Clean | 否 | RGB-D + SAM3 + Molmo | 否 | 否 | 待测 | 原始 baseline |
| E1 Stage-CAP Clean | 否 | 同 E0 | 是 | 否 | 待测 | 只看阶段规划收益 |
| E2 Stage-CAP + KITE Clean | 否 | 同 E0 | 是 | 是 | 待测 | 主方法 |
| E3 CAP-X + KITE Clean | 否 | 同 E0 | 否 | 是 | 待测 | KITE-only control |
| E4 Stage-CAP Oracle-Readonly | 是，只读 | 可选 | 是 | 是 | 已有参考/待复测 | 上限，不作主公平对比 |

## 6. 关键消融

如果时间允许，建议补充以下消融：

- No stage replan：有阶段计划，但失败后不重规划。
- No online KITE images：只给文本执行日志，不给失败帧图像。
- No SAM3：仅 RGB-D + depth 几何，不用 SAM3。
- No Molmo：不用点提示模型，只用 SAM3 text prompt / depth。
- Same prompt budget：确保 CAP-X 与 Stage-CAP 的最大轮数、token budget、超时相同。

## 7. 报告必须包含的材料

每个实验组都要保存：

- 完整配置文件
- 每个任务的最终 CAP 代码
- 每轮模型原始输入与输出
- 每个任务的视频/gif
- 每个失败任务的失败阶段
- 如果使用 KITE：失败关键帧原图、depth、segmentation、point cloud、诊断反馈 JSON
- 汇总表：成功/失败/超时/未测试

## 8. 当前结论边界

已有的 37/40 结果应标注为：

> Stage-CAP + Online KITE under oracle-readonly diagnostics, without deterministic fast-paths.

它说明阶段化和反馈闭环在只读诊断条件下很强，但不是 clean vision-only 公平对比结果。对外主结论必须等待 E0/E1/E2 的 clean matched 输入复测。

