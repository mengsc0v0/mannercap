# Gate 1 / WP2：生成期验证器合同

日期：2026-09-16  
状态：**开发期合同，未接入真实任务数据，未完成物理校准**  
范围：`synthesis_v2/generation_validator.py` 及其标准库测试

本文件定义一个可独立运行的生成期筛选器。它用于发现候选轨迹的证据和约束问题，不能把生成期结果当作封存集上的最终审计结果，也不能替代真实传感器校准。

## 1. 接口和输入边界

公开接口是：

```python
from synthesis_v2.generation_validator import (
    ConstraintSpec, Decision, GenerationCase, GenerationSample,
    GenerationTrace, validate_generation,
)

result = validate_generation(
    GenerationCase(
        trace=trace,
        requested_constraints=requested,
        effective_constraints=executed_or_none,
        task=True,
    )
)
```

核心输入分为四层：

* `GenerationSample`：实际时间 `t`、阶段、实测速度/加速度、`kp`、夹爪比例、接触力、`sensor_valid` 和 `reset_id`。缺字段使用 `None`，不会补成零、默认增益或默认夹爪值。
* `GenerationTrace`：不可变样本序列和单位声明。要求 `time=s`；使用到的字段还必须声明 `velocity=rad/s`、`acceleration=rad/s^2`、`kp=gain`、`gripper_fraction=fraction`、`contact_force=N`。
* `requested_constraints`：执行前的原始约束，必须保留。
* `effective_constraints`：执行侧实际生效的约束。省略该参数表示原约束原样生效；若要表示裁剪，必须传入一个显式的 `ConstraintSpec`，将被移除的字段置为 `None`。

安全上限 `safety_max_force_n` 是独立的硬约束，作用于整段记录；速度、刚度、夹爪和 jerk 只在 `target_phases` 指定的阶段区间内判定。接触力向量按力的范数读取，单位仍须为 N；本模块不把 MuJoCo 混合空间 wrench 或力矩当作抓力。

该接口没有 `method_name`、solver 名称或方法专用分支。相同轨迹和约束只要输入相同，结果相同。生成器可以在上游记录方法信息，但不得以方法名改变合同结论。

## 2. 三态和联合规则

每个维度都只有三种状态：`pass`、`fail`、`unknown`。

* `pass`：规则和所需证据齐全，所有被检查样本满足约束。
* `fail`：证据有效且至少一项硬规则被违反，或已知执行侧放宽/删除了原始约束。
* `unknown`：无法可靠判定，例如无日志、单位不符、传感器失效、时间轴断裂、目标阶段缺失或测量字段缺失。

联合规则固定为：任一维度为 `fail` 时 `overall=fail`；三个维度全部为 `pass` 时 `overall=pass`；否则 `overall=unknown`。只有 `overall=pass` 才有 `publishable=True`，所以 `unknown` 不能进入合格分子。

任务状态由调用方显式提供；缺失任务状态为 `task=unknown`。任务失败不会被改写成 `manner_infeasible`。例如任务失败但轨迹和安全都合规时，结果是 `task=fail, manner=pass, safety=pass, overall=fail`。

## 3. 生成期判定规则

### 时间、reset 和采样

相邻样本时间必须有限且严格递增，所有样本必须属于同一个 `reset_id`。相邻间隔使用实际 `t[i+1]-t[i]`；若超过原始 `max_gap_s`，两项物理结论均为 `unknown`。若输入了 `dt_s`，它还必须与时间差一致。不存在固定 `dt` 替代真实时间的路径。

### 阶段区间

目标阶段在日志中必须出现，并且从第一个目标样本到最后一个目标样本形成一个连续区间；中间插入非目标阶段会得到 `phase_interval_not_contiguous`。多阶段目标还要求按 `target_phases` 的顺序出现。目标样本数和实际持续时间分别满足 `min_samples`、`min_duration_s`。

阶段缺失、阶段标签缺失或区间不连续使 `manner=unknown`。安全判断按整段接触力证据独立进行；当安全力记录、单位和时间轴有效时，阶段缺失不改变安全结论。

### manner

每个目标样本都会检查上/下界。下界不是“有一帧达标”或“达到某个帧比例”：任何目标样本低于下界都会得到 `fail`。速度和加速度向量使用范数；`kp` 和夹爪比例按标量检查。需要 jerk 时，仅在同阶段、同 reset、实际时间连续的相邻样本间用加速度差除以实际 `dt`。

没有 manner 规则、目标字段缺失、字段非有限或单位错误时，`manner=unknown`。

### safety

设置 `safety_max_force_n` 后，整段每个样本都必须有有效接触力并声明 N。任一力超过上限即 `safety=fail`；力缺失、力传感器失效、错误单位、负值或非有限值为 `safety=unknown`。未设置安全上限也不会自动通过，而是 `safety=unknown`。

### 原始约束与裁剪

上限被增大或置空、下限被减小或置空、目标阶段/阶段 ID 被改动，以及 `min_samples`、`min_duration_s`、`max_gap_s` 或 tolerance 被放宽，都会产生显式 `manner_clip:*` 或 `safety_clip:*` 原因。安全上限被删除或放宽会使 `safety=fail`；manner 约束被删除或放宽会使 `manner=fail`。因此将裁剪后的空约束交给验证器不能把原始请求变成通过。

## 4. 反例合同覆盖

以下 12 类反例均有自动测试；表中结果是三态合同，不是仿真或校准结论。

| # | 反例 | 构造 | 期望结果 |
|---:|---|---|---|
| 1 | 裁剪 | 原始 `max_velocity` 被执行 spec 置空 | `manner=fail`，原始裁剪原因保留 |
| 2 | 无日志 | `trace=None` | `manner=unknown,safety=unknown`，不能发布 |
| 3 | 缺阶段 | 目标为 `contact`，日志只有 `approach` | `manner=unknown` |
| 4 | 传感失效 | 任一记录 `sensor_valid=False` | `manner=unknown,safety=unknown` |
| 5 | 瞬间达标 | 仅第一帧达到速度下界，其余低于下界 | `manner=fail` |
| 6 | 持续低于下界 | 所有目标帧低于速度下界 | `manner=fail` |
| 7 | 安全越界 | 任一接触力超过安全上限 | `safety=fail,overall=fail`，即使 manner 通过 |
| 8 | 重复时间 | 相邻 `t` 相等 | `manner=unknown,safety=unknown` |
| 9 | 采样缺口 | 实际时间差超过 `max_gap_s` | `manner=unknown,safety=unknown` |
| 10 | 单位错误 | 关节速度声明成 `m/s` | `manner=unknown`；安全维度仍按自己的力单位独立判断 |
| 11 | 错误阶段生效 | 非目标 `approach` 超速，目标 `contact` 合规 | `manner=pass`；非目标速度不污染目标阶段 |
| 12 | 任务失败但方式通过 | `task=False`，方式和安全证据合规 | `task=fail,manner=pass,safety=pass,overall=fail` |

测试还覆盖：无约束、目标阶段交错、reset 切换、实测 `dt_s` 不匹配、力单位错误、缺失速度、缺失 jerk 加速度，以及原始 spec 不被调用过程修改。

## 5. 公开生成验证与 sealed final audit 的边界

本模块是公开的生成期合同，允许在候选轨迹产生后即时反馈；它只接收当前 trace、原始请求、执行 spec 和任务状态。它不读取最终审计的封存样本、封存结果、最终统计分子/分母或校准数据库，也不生成最终审计签名。

独立 final audit 应在导出完整 lineage、真实时间、阶段事件和传感器证据后重新读取原始请求，并对 sealed 配置、校准版本和发布规则独立判定。生成期 `pass` 只是候选记录满足当前开发合同的信号，不能转写为“已通过最终审计”。

## 6. 运行证据和限制

在当前工作区执行：

```text
python -m py_compile synthesis_v2/generation_validator.py tests/test_v2_generation_validator.py
通过

python -m unittest -v tests/test_v2_generation_validator.py
Ran 21 tests ... OK
```

测试本身使用 Python 标准库 `unittest`，不依赖 pytest、NumPy、MuJoCo 或 RoboSuite。完整本地合同测试以 `unittest discover` 运行 140 项并通过；同步后的 4090 仓库以项目环境运行 pytest，164 项通过。命令和结果同时记录在 Gate 1 退出评审中。

这些 21 个测试是合同级合成样例。当前没有把它们写成 Lift、Stack 或 Wipe 的真实成功率、测力响应或学生闭环证据；阈值也没有因本模块而获得物理校准资格。接入真实数据前，仍需由独立校准流程提供来源、单位、传感器有效性和封存版本。
