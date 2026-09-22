# Gate 0 / WP2 评测审计

日期：2026-09-16
负责人：B（独立评测）
审计范围：`stagecap_robosuite_libero_repro_20260714/capx/envs/manner_validator.py`、`run_manner_ablation.py`，以及 v2 实施方案第 6、9、10、14、15、19 节。
结论状态：**评测接口可开始 Gate 1 修复，但当前仓库的既有验证器不能作为最终审计器，暂不放行正式扩量。**

本文是 Gate 0 的工程审计和反例合同，不是物理校准报告，也不声称 Lift、Stack 或 Wipe 已完成测力、阶段质量或学生闭环验证。方案明确要求这些证据在后续工作包中独立产生。

## 1. 审计证据和当前边界

已检查：

* 既有生成期验证器 `capx/envs/manner_validator.py`（四个 manner 维度 A-D）。
* 既有驱动 `run_manner_ablation.py` 的任务/方式联合分类、裁剪和重试路径。
* v2 方案 WP2 的要求：生成期筛选和最终审计分离；任务、方式、安全分别使用 `pass/fail/unknown`；unknown 不计为合格；至少覆盖 12 类自动反例。

本次新增的独立内核为 `capx/evaluation_contract_v2.py`。它只实现 Gate 1 评测语义和手工轨迹合同，不替换或修改现有模块，也不内置任何“已校准”的阈值。其接口为：

* `Decision`: `pass`、`fail`、`unknown` 三态。
* `Trace` / `TraceSample`: 保留缺失字段、实际时间、阶段、传感器有效位和 reset 标识。
* `ConstraintSpec`: 阶段作用域、单位合同、方式约束和独立安全上限。
* `EvaluationCase` / `evaluate`: 分别给出 `task`、`manner`、`safety` 和联合 `overall`。

联合规则固定为：任一维明确 `fail` 则联合 `fail`；全部为 `pass` 才联合 `pass`；其余为 `unknown`。任务失败不会被重命名为“方式不可行”，缺证据不会被补零后通过。

## 2. 既有实现的可定位缺陷

| 编号 | 定位 | 发现 | 对验收的影响 |
|---|---|---|---|
| F1 | `manner_validator.py:291-299`；`run_manner_ablation.py:861-871` | 验证器本身对空日志返回 `UNCERTAIN`，但驱动在裁剪后无约束时将 `manner_ok=True`；空日志在驱动分类中又落为 `error`。 | 无日志、全裁剪、执行异常没有统一三态语义，可能进入成功/错误混合口径。 |
| F2 | `run_manner_ablation.py:837` | `bool(exec_result.get("task_completed", False))` 把缺少任务结果的记录直接变成任务失败。 | 缺字段和真实任务失败不可区分；不能按 unknown 计算缺失边界。 |
| F3 | `run_manner_ablation.py:851-857` | 对被裁剪的维度把原约束字段整体置为 `None`，之后只验证 `clipped_spec`。 | 原始请求被静默放宽；最终审计看不到被删除的约束。 |
| F4 | `manner_validator.py:347-349,395-400` | 速度默认只找 `move`，找不到时退回全部帧；阶段覆盖缺失时，阶段覆盖约束会被跳过。 | 错阶段数据可能支撑通过，缺目标阶段不一定产生 unknown。 |
| F5 | `manner_validator.py:359-389` | 上界允许 10% 违规；下界只要求 30% 帧达到 `0.8×` 下界。 | 瞬间达到阈值或大部分时间低于下界仍可能通过，不满足 v2 的持续作用语义。 |
| F6 | `manner_validator.py:444-480` | `kp_max` 只看全程最小 kp，`kp_min` 只看全程最大 kp；没有阶段作用域和持续时间。 | 一个合规离群值即可掩盖持续过硬/过软；非目标阶段会改变结果。 |
| F7 | `manner_validator.py:486-522` | 夹爪检查找不到 `grip` 时退回全部帧；上下界都以 `min_frac` 的同一方向比较，且 `final_frac` 未参与判定。 | 夹爪分数的方向和阶段含义不清，无法稳定表达“轻柔/用力”两种相反要求。 |
| F8 | `manner_validator.py:529-557`；`_build_geometry_evidence` `:682-688` | jerk 使用固定 `dt=0.05`，不读取样本时间，不检查时间缺口、重复时间、reset 或阶段边界。 | 跨缺口/跨 reset 的导数可能被写成有效物理量；几何证据的持续时间也固定按帧数计算。 |
| F9 | `manner_validator.py:163-170,619-634` | `MannerStepSnapshot` 没有单位声明和传感器有效位；字典输入缺字段时默认 `t=0`、速度/加速度为零、kp=1500、夹爪=1、接触力=0。 | 缺采样可以被默认值伪装成合规；N、N·m、rad/s 等单位错误无法被拒绝。 |
| F10 | `manner_validator.py:260-339` | 仅生成 manner `VerificationEvidence`，没有独立的安全判定；`cfrc_norm` 只在 `run_manner_ablation.py:455-469` 被提取为指标。 | 观测到安全越界仍可能得到方式通过；任务、方式、安全无法独立签收。 |
| F11 | `run_manner_ablation.py:588-605` | `not task_ok and manner_ok` 直接分类为 `manner_infeasible`；包含 “ik/solver/solve” 的异常字符串也直接映射成方式不可行。 | 任务失败、IK/执行异常和真正方式物理不可行被混淆，违背 WP2 的失败分类边界。 |
| F12 | `manner_validator.py:319-330` | 无约束虽然返回 `UNCERTAIN`，但现有驱动把 baseline/fully-clipped 作为 manner 通过。 | 生成对照和最终质量分子可能把“没有检查”当成“满足方式”。 |
| F13 | 全局 `MannerSpec.tolerance=0.15`、`VIOLATION_RATE_THRESHOLD=0.10` | 容忍比例和阈值硬编码在生成期模块，没有独立校准版本、封存测试配置或版本追踪。 | 正式结果可能依赖方法作者修改的隐含容忍，而非独立冻结规范。 |

已有代码的可复现实例（使用字典输入，绕过仿真依赖加载）显示：

* 只给 `phase` 的缺字段样本，`MannerSpec(max_joint_vel=0.1)` 返回 `satisfied`，原因是 `_coerce_snapshot` 补了零速度。
* 三帧速度 `[0.5, 0, 0]`，要求 `min_joint_vel=0.5`，返回 `satisfied`，原因是 30% 活跃帧规则和 15% 容忍。
* 三帧接触力均为 `999`，只给速度约束，仍返回 `satisfied`，因为没有安全维度。
* 重复时间 `[0, 0, 0.1]`，只给 jerk 约束，仍返回 `satisfied`，因为导数固定除以 0.05。

这些是验证器语义的单元级复现，不是物理实验结果。

## 3. 十二类必测反例合同

下面的“既有风险”描述的是当前验证路径在没有额外外围保护时的结果；“严格合同结果”是 Gate 1 应冻结的结果。对应自动测试位于 `tests/test_v2_evaluation_contract.py`。

| # | 反例 | 最小构造 | 既有风险 | 严格合同结果 | 自动测试 |
|---:|---|---|---|---|---|
| 1 | 裁剪 | 请求有 `max_velocity`，执行侧把该字段置空 | 原驱动只审 `clipped_spec`，可能 manner 通过 | `manner=fail`，保留裁剪原因 | `test_clipped_requested_bound_cannot_disappear_into_a_pass` |
| 2 | 无日志 | `trace=None` 或空列表 | 验证器是 uncertain，驱动会转成 error，口径不一致 | `manner=unknown`、`safety=unknown`、联合 `unknown` | `test_no_log_is_unknown_and_never_an_automatic_pass` |
| 3 | 缺阶段 | 约束目标为 `contact`，日志只有 `approach` | 目标阶段可能被跳过或退回全程 | `manner=unknown` | `test_missing_target_stage_is_unknown` |
| 4 | 传感失效 | 目标帧 `sensor_valid=False` | 现有字典输入无有效位，缺值可补成默认值 | `manner/safety=unknown` | `test_sensor_failure_is_unknown_instead_of_zero_filled_pass` |
| 5 | 瞬间达标 | 只有第一帧达到速度下界，其余持续低速 | 下界只需约 30% 帧，可能通过 | 明确低于下界即 `manner=fail` | `test_one_instant_high_speed_does_not_satisfy_a_stage_lower_bound` |
| 6 | 持续低于下界 | 目标阶段所有速度低于下界 | 与 F5 相同，现有规则只看比例 | `manner=fail` | `test_sustained_below_lower_bound_is_a_known_failure` |
| 7 | 安全越界 | 任一帧 `contact_force_n > safety_max_force_n` | `cfrc_norm` 只是指标，未进入判定 | `safety=fail`，联合 `fail` | `test_safety_violation_fails_joint_result_even_when_manner_passes` |
| 8 | 重复时间 | 相邻样本 `t_i=t_{i+1}` | jerk 固定 dt，重复样本不会被拒绝 | `manner/safety=unknown` | `test_duplicate_timestamps_are_unknown` |
| 9 | 采样缺口 | 相邻 `dt` 超过冻结 `max_gap_s` | jerk 和持续时间可被错误当连续数据 | `manner/safety=unknown` | `test_sampling_gap_is_unknown_and_cannot_be_hidden_by_fixed_dt` |
| 10 | 单位错误 | 速度声明为 `m/s` 而约束是关节 `rad/s` | 现有快照无单位合同 | `manner/safety=unknown` | `test_unit_mismatch_is_unknown_instead_of_treating_meters_as_radians` |
| 11 | 错误阶段生效 | 非目标 `approach` 超速，目标 `contact` 合规 | kp/夹爪全局检查可能被非目标阶段污染；阶段缺失也可能退回全程 | 只按 `target_phases` 判；目标合规则 `manner=pass` | `test_wrong_phase_effect_does_not_contaminate_target_phase_decision` |
| 12 | 任务失败但方式通过 | 任务状态 fail，轨迹方式和安全均合规 | 驱动将其命名为 `manner_infeasible` | `task=fail`、`manner=pass`、`safety=pass`、联合 `fail` | `test_task_failure_does_not_get_relabelled_as_manner_infeasible` |

额外覆盖的硬门：全无方式约束返回 `manner=unknown`，不能自动通过；缺失速度或缺失 jerk 加速度也返回 `unknown`。对应测试为 `test_unconstrained_trace_is_unknown_even_when_telemetry_looks_good`、`test_missing_velocity_is_unknown_not_default_value` 和 `test_missing_acceleration_is_unknown_when_jerk_is_required`。

## 4. Gate 1 工作项和放行证据

### B：冻结独立规范

1. 新建版本化 `constraint_registry.yaml`，至少记录任务、阶段实例、作用范围、每个阈值的单位、来源、校准版本、允许容忍和适用接触阶段。
2. 将 `requested_constraints`、`predicted_plan`、`executed_plan`、实际测量四层分开保存；裁剪只能产生显式执行异常/失败记录，不能覆盖原请求。
3. 公开规范与生成器封存测试配置分离；生成器不能读取最终审计逐回合结果。
4. 固定三态映射和联合规则。`unknown` 不进合格分子，执行异常独立于“原要求未满足”；发布统计同时给 `V/N` 与已发布 unknown 边界。

### A：补齐可相信的记录和校准输入

1. 轨迹 schema 必须记录真实 `t`、采样频率、reset/episode 标识、阶段事件、传感器有效位、缺采原因、控制指令与实测值。
2. 力、力矩、夹持力分开；声明单位、坐标系和对象归属。当前 `cfrc_norm` 不能直接作为已校准抓力。
3. 对每个支持任务提供成功、欠作用、过作用和安全越界样例，并由 B 以独立数据冻结阈值。若 Wipe 没有可靠接触/结果定义，范围必须收缩，不能写成已验证接触任务。

### C/D：对齐阶段、数据和动作接口

1. 每个阶段实例登记进入/结束事件，严格按阶段作用域判定；目标阶段缺失、阶段标签异常或事件不完整一律 unknown。
2. 数据窗口先按回合/对象/组合切分，再切片；不得跨 reset、缺口或非法 padding。保存完整 lineage 和所有失败候选。
3. D 在扩量前完成动作导出—导入—学生控制器回放检查；教师使用的 kp/特权通道若学生不能表达，不得归因给学生学习。

### 放行门槛

在 Gate 1 之前至少提交：

* 上述 12 类反例的自动测试和手算样例，三态输出及失败原因可查。
* 真实时间导数只在同 reset、同阶段、连续有效区间内计算；不得使用固定 dt 代替时间轴。
* 现有生成期筛选与独立最终审计的调用边界、版本哈希和封存集隔离证明。
* 初步校准包和传感器分辨率说明；当前仓库已有旧轨迹及已有结果只能作审计输入，不能作为新校准完成的证明。
* 至少 30 条动作回放/记录检查的计划与结果入口；阶段、对象、方式和安全分别出表。

在这些证据齐全前，建议状态为“允许修复和开发集调试，禁止正式 E1/E2 扩量”。

## 5. 本次代码验证

运行于当前 Windows 工作区：

```text
python -m py_compile capx/evaluation_contract_v2.py tests/test_v2_evaluation_contract.py
结果：通过

16 个测试函数的无 pytest 手工调用：
called=16 failures=0
结果：通过

python -m pytest tests/test_v2_evaluation_contract.py -q
结果：未执行；当前解释器没有安装 pytest（ModuleNotFoundError: No module named 'pytest'）
```

测试只验证独立三态合同和可定位反例，不证明仿真物理真实性、测力校准、任务成功率或学生学习价值。待服务器环境安装仓库锁定依赖后，应重新运行 pytest，并将 Gate 1 校准样例接入同一审计入口。
