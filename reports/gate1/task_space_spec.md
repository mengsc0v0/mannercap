# MannerCAP v2 任务/阶段/属性请求空间规格（Gate 1 / WP3）

版本：`synthesis_v2.task_space.v1`，2026-09-16  
负责人：C（合成方法）  
状态：纯 Python 声明式合同与测试初版；`TaskSpace.status=draft`，尚未冻结正式 manifest。

## 1. 目的与边界

`synthesis_v2/task_space.py` 为请求生成前的任务空间提供一个可版本化、可审计的结构。它把任务、阶段、对象属性、控制通道、请求单元以及配对规则写成 JSON 兼容记录，并在构造时拒绝引用不存在的阶段、任务或控制通道。

本版只把 **Lift** 和 **Stack** 标为 `supported` 且列入开发任务。第三任务 **Wipe** 只保留为 `candidate` + `pending_calibration`，带有一个不可执行的 `pending_calibration` 请求单元；它没有阶段、对象或控制通道，不能被构造成 `applicable` 单元。后续 4090 探测已证明 Wipe 可清除 100/100 marker 并触发 RoboSuite 官方成功判定，但尚未完成 Wipe 专属力/接触和约束阈值校准，因此本合同仍不提升其状态。这里没有假定质量、摩擦、接触力、位姿阈值或其他物理数值。

## 2. 版本与任务状态

顶层 `TaskSpace.to_dict()` 始终写出 `schema_version`、`status`、任务注册表、请求单元、配对单元、split policy 和三类配对合同。当前默认注册表如下：

| `task_id` | `status` | `calibration_status` | 开发请求 | 说明 |
| --- | --- | --- | --- | --- |
| `lift` | `supported` | `ready_for_development` | 允许 | 四个阶段：approach、descend、grasp、lift |
| `stack` | `supported` | `ready_for_development` | 允许 | 七个阶段：approach、descend、grasp、lift、transport、place、retreat |
| `wipe` | `candidate` | `pending_calibration` | 禁止 | 接触指标和任务结果定义仍待校准 |

`supported` 只表示可以进入开发请求空间，不表示物理属性或成功阈值已经校准。默认 `TaskSpace` 不生成 Lift/Stack 的正式 request cell，因为具体 `property_config_id` 和 `initial_state_id` 尚未冻结；调用方必须显式提供它们。

## 3. 阶段与边界事件

每个 `PhaseSpec` 必须包含：

- 唯一的 `stage_instance_id` 和显示名称；
- `entry_event` 与 `exit_event`，每个事件都必须有非空 `name`、`source`，可选 `condition_ref`、对象角色和说明；
- 阶段涉及的对象角色 `object_roles`；
- 阶段允许的控制通道 `allowed_control_channels`；
- `manner_scope_allowed`，显式表示该阶段是否允许 manner/stage scope 请求。

默认阶段边界使用 `episode`、`task_adapter`、`control_trace` 等来源的符号事件。它们是适配器事件名，不是已经通过真实轨迹验证的物理阈值。TaskSpace 会检查阶段引用的对象角色和控制通道已经在同一任务声明；阶段 ID、对象 ID、控制通道名在任务内不能重复。

## 4. 对象属性来源、单位和可用性

每个 `ObjectSpec` 绑定一个稳定的 `object_id`、角色和 `ObjectPropertySpec` 列表。每个属性至少写出：

| 字段 | 约束 |
| --- | --- |
| `property_name` | 非空属性名 |
| `source` | 非空来源标识，例如 `calibration_registry:pending` |
| `unit` | 非空单位或接口单位标签；未知物理单位写 `adapter-defined` |
| `value` | 可选；本版默认是 JSON `null`，不填猜测值 |
| `availability` | `ready_for_development`、`pending_calibration` 或 `calibrated` |

Lift/Stack 默认只声明 `geometry` 和 `physical_configuration`，来源为待校准 registry，值保持 `null`。没有从任务名称虚构质量、尺寸、摩擦、接触力或任何数值阈值。

## 5. 合法控制域

`ControlChannelSpec` 对每个控制通道显式声明 `name`、接口 `unit`、`source` 和必填的 `domain_ref`。默认通道为：

| 名称 | 单位标签 | 合法域引用 | 当前边界 |
| --- | --- | --- | --- |
| `joint_velocity` | `rad/s` | `calibration_registry:joint_velocity` | `lower_bound`/`upper_bound` 均为 `null` |
| `acceleration` | `rad/s^2` | `calibration_registry:acceleration` | `lower_bound`/`upper_bound` 均为 `null` |
| `gripper` | `adapter-defined` | `calibration_registry:gripper` | `lower_bound`/`upper_bound` 均为 `null` |

`domain_ref` 是合法域的来源/注册表引用；数值边界只有在校准和执行器接口检查后才能填写。若填写边界，合同拒绝布尔值、NaN、无穷和负向区间；本版默认不把任何数值范围当作已批准域。阶段只能使用同一任务中 `enabled` 的控制通道。

## 6. Request cell 与不适用单元

一个 `RequestCell` 是任务/阶段/属性配置/初态/split 的显式组合，字段包括：

```text
task_id
stage_instance_ids
property_config_id
initial_state_id
split
applicability
reason
pair_group_key / pair_type（可选）
cell_id（由合同计算）
```

`applicable` 单元必须指向已支持任务、至少一个已声明阶段，以及非空的 `property_config_id` 和 `initial_state_id`。`inapplicable` 或 `pending_calibration` 单元必须给出非空 `reason`；不适用单元可保留在空间中作为分母和缺口证据，但不能形成可执行配对。Wipe 的默认单元就是这种 pending 单元，并明确写出 `requests_enabled=false`。

`cell_id` 为请求单元规范化 JSON（含 schema version、任务/阶段/属性/初态、split、适用性、原因和 metadata）的 SHA-256 前 24 位，格式为 `cell_<24 hex>`。对象 key 顺序变化不改变 ID；手工传入的 ID 若不匹配会被拒绝。

## 7. Split 与三类配对合同

`SplitPolicy` 显式列出允许的 split，并将 `development` 设为开发集。默认允许：`development`、`generator_test`、`student_train`、`student_validation`、`student_test`。`pair_members_same_split=true` 是硬约束；`TaskSpace` 拒绝跨 split 的 PairCell，也要求三条配对合同都保留 split。

三类 `PairConstraint` 必须恰好各有一条：

| `pair_type` | 必须相同（摘要） | 允许变化（摘要） |
| --- | --- | --- |
| `modifier_word` | task、scene、initial state、target、physical config、stage scope、控制域、split、预算 | `raw_language`、其声明的 requested constraints |
| `physical_attribute` | task、scene、raw language、requested constraints、stage scope、geometry、split | physical config、initial state、attribute intervention |
| `stage_scope` | task、scene、initial state、target、physical config、manner、控制域、split、预算 | `raw_language`、其声明的 requested constraints、stage scope |

`PairCell` 必须引用两个不同的、同一任务且均为 `applicable` 的 request cell，并带有配对类型和 split。其 ID 对成员顺序不敏感，格式为 `paircell_<24 hex>`。本模块只声明这三类结构规则；具体自然语言和物理配置比较仍由 `request_manifest.py` 的请求合同负责，不能由此处自动补全。

## 8. Gate 1 当前状态与未实现部分

已交付：

- Lift/Stack/Wipe 的版本化任务注册表和阶段入口/出口事件合同；
- 对象属性来源、单位、可用性以及不填猜测物理值的约束；
- 带合法域引用的控制通道和阶段控制引用校验；
- 显式开发 split、不可跨 split 配对、三类配对约束、确定性 cell/pair ID；
- `applicable`、`inapplicable`、`pending_calibration` 单元的校验和稳定 JSON writer；
- 12 项纯 Python 契约测试。

仍未完成：

- `TaskSpace.status` 仍为 `draft`；已用真实 Lift/Stack 初态生成一个 14 条记录的 development-only 清单，但正式 request manifest 尚未冻结，属性配置、场景分母和配额仍未批准；
- 尚无物理属性、控制数值域或任务成功阈值的校准结论；第三任务虽已通过实际功能探测，仍保持 candidate/pending calibration；
- 尚未接入解析器、生成器、执行器、模型/API 或正式实验；没有运行结果和资源实测；
- Gate 1 已取得 Lift/Stack 初始 `time/qpos/qvel/ctrl` 零误差恢复和功能终态重放证据，但 `1e-8` 全轨迹逐状态重放仍未通过；本合同的 `initial_state_id` 只是请求引用，不能自动充当恢复成功证据；
- 尚未实现最终审计、真实轨迹事件判定、去重证据和 release/freeze ledger；这些应在后续阶段追加，不能在本 schema 中伪造通过。

## 9. 验证记录

执行命令：

```text
python -m py_compile synthesis_v2/task_space.py tests/test_v2_task_space.py
python -m unittest -v tests/test_v2_task_space.py
```

结果：12 项测试全部通过，覆盖开发任务边界、Wipe candidate/pending、阶段入口/出口事件、对象属性来源/单位、无猜测值、控制域引用、显式轴、不可适用单元、确定性 ID、三类配对、同任务和不可跨 split 约束、JSON writer。验证仅使用本地 Python；本次未调用外部 API或正式实验。
