# Gate1 独立约束注册表与最终审计边界

版本：`synthesis_v2.constraint_registry.v1`，2026-09-16

状态：**开发注册表已建立；Lift/Stack 的物理阈值仍为
`pending_calibration`，最终审计配置尚未封存可用。**

配套文件：

- `configs/constraint_registry.yaml`
- `synthesis_v2/constraint_registry.py`
- `tests/test_v2_constraint_registry.py`

## 1. 注册表用途与边界

`constraint_registry.yaml` 是独立的版本化声明，描述任务、阶段、测量量、单位、
作用范围、三态决策合同和校准来源。它不是生成器配置，也不是最终审计结果。
当前 Lift 与 Stack 可以进入开发调度，但所有物理阈值均保持 `null`，来源明确
指向待校准协议，不能从当前成功率、`cfrc_norm`、kp 或夹爪开度猜出正式值。

注册表顶层包含：

| 字段 | 规则 |
| --- | --- |
| `schema_version` / `registry_version` | 必须存在并版本化 |
| `registry_version_hash` | SHA-256 canonical content hash，排除自身字段后重算并核对 |
| `status` | 当前为 `development_pending`；不等同于正式冻结 |
| `calibration` | 必须有状态、来源引用、版本；当前状态为 `pending_calibration` |
| `decision_contract` | 必须同时声明 `task`、`manner`、`safety` 三个维度 |
| `unknown_policy` | 固定为 `not_qualified` |
| `joint_rule` | 任一 fail→fail；全 pass→pass；其他→unknown |

每个约束还必须写出 task、阶段、作用域、metric、单位、三态维度、unknown 策略、
阈值字段和独立校准来源。阶段约束的 `scope.stage_instance_id` 必须和 `phase`
一致；每个已声明阶段至少有一条约束。空约束清单和不存在的阶段都会被拒绝。

## 2. Lift/Stack 开发态约束

当前登记了 Lift 的 `approach/descend/grasp/lift` 以及 Stack 的
`approach/descend/grasp/lift/transport/place/retreat`。指标包括：

- `joint_velocity`，单位 `rad/s`；
- `gripper_fraction`，接口比例单位 `fraction`；
- `force_norm`，单位 `N`。

每条约束的 `thresholds.lower/upper/tolerance` 目前都必须为 `null`。
`pending_calibration` 状态下出现任何数字阈值会使注册表无效，并产生
`pending_threshold_value` / `threshold_before_calibration`，禁止晋升。

这一区分了“字段和阶段可以被生成器引用”与“数值可以用于正式判定”。当前注册表
校验通过时返回 `valid=true, decision=unknown, promotion_allowed=false`，这是
预期结果，不是失败。

## 3. 公开生成与封存最终审计的隔离

注册表明确分成两个引用边界：

### 3.1 `public_generation`

公开段只列出 `constraint_refs` 和允许给生成器看的字段（任务、阶段、metric、
单位、来源引用和 pending 状态）。它不得包含 `thresholds`、数值阈值、最终审计
内容或封存结果。它仅带：

```text
final_audit_ref.config_id = sealed_final_audit.v1
final_audit_ref.visibility = sealed_reference
```

### 3.2 `sealed_final_audit`

封存段只保存独立 B-owned 配置的引用：`config_ref`、状态和待填的
`config_sha256`。当前为 `sealed_pending`，没有内嵌约束、阈值或审计结果；生成器
不能读取该配置内容。只有外部校准证据、审计配置哈希和正式签收完成后，才能把
状态改为 `sealed_ready`。

公开配置 ID 与封存配置 ID 必须不同，但公开引用的封存 ID 必须精确匹配。把两者
合并、将审计阈值内联到公开段、或在封存段内联约束都会使注册表无效。

## 4. 三态与校准晋升规则

`task`、`manner`、`safety` 各自只能使用 `pass`、`fail`、`unknown`。缺失数据、
错误单位、错误阶段/对象范围、采样 gap、reset 跨越和未校准阈值不能被当成 pass。
联合结果固定为：

```text
if any dimension == fail: fail
elif all dimensions == pass: pass
else: unknown
```

`validate_registry` 将结构合法性与正式晋升分开：

1. `valid=false`：schema、单位、阶段、哈希、来源或隔离边界存在错误，结果为
   `decision=fail`；
2. `valid=true` 但仍有 pending 校准或 sealed audit 未就绪：结果为
   `decision=unknown`，`promotion_allowed=false`；
3. 只有顶层和每条约束均为 `calibrated`、所有阈值已由独立证据填写、每条证据有
   SHA-256、任务已校准且封存审计为 `sealed_ready` 时，才允许晋升。

当前文档只满足第 2 类，不能声称任何正式物理约束已经冻结。

## 5. 测试覆盖

`tests/test_v2_constraint_registry.py` 包含 21 个标准库 `unittest` 测试，覆盖：

- 当前 YAML 的版本哈希、Lift/Stack 阶段和 pending 状态；
- 公开/封存引用分离和禁止内联审计数据；
- 错误单位、缺阶段、空约束、错作用域；
- 缺校准来源/版本；
- 缺 task/manner/safety、缺 unknown 或将 unknown 声明为合格；
- pending 数字阈值不得晋升；
- 哈希错配、缺任务、确定性 hash、JSON 输出和输入不变性。

测试只读取和变换内存中的合成文档，不运行仿真、生成器、模型服务或最终审计。

## 6. 尚未完成的放行工作

1. 为每个 Lift/Stack 阶段完成独立速度、夹爪、接触力/力矩和安全测量校准，记录
   已知载荷、单位、坐标系、采样时间、误差和有效率；
2. 冻结正式阈值和 tolerance，生成独立 sealed final-audit artifact，并提交其
   SHA-256、版本和非作者复核记录；
3. 以真实轨迹验证 task/manner/safety 三态、阶段作用域、缺失/失效/跨 reset
   处理和 unknown 分母；
4. 将生成器只接入公开引用，将最终审计器只接入封存引用，避免生成期读取封存
   测试样本或逐回合结果；
5. 在上述证据完成前，不发布正式 JPR、物理合格率或已校准阈值结论。
