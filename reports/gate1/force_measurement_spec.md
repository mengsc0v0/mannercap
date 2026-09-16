# Gate1 对象接触力测量规范（开发版）

状态：**结构合同已新增；对象接触力实际接入与物理校准尚未完成。**

本规范对应 `synthesis_v2/contact_force_contract.py` 及其单元测试。它只约束
记录边界和数值判定，不能把当前 MuJoCo 仿真量称为真实抓力、环境载荷或已
校准传感器读数。

## 1. 当前实现审查

现有 RoboSuite/MuJoCo 记录器在
`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:526-529`
和 `robosuite_base.py:453-458` 读取 `sim.data.cfrc_ext`，随后执行
`np.linalg.norm(cfrc, axis=1)` 并取所有 body 的最大值。这个实现有四个明确
问题：

1. `cfrc_ext` 每个 body 的六维外部 wrench 包含三维 force（N）和三维 torque
   （N·m）。对六维向量直接求范数会混合不同单位，结果既不是 N 也不是 N·m。
2. 沿 body 维取最大值没有任务对象、body 名称或 geom 对的映射。它可能选到
   机器人、容器或其他 body 的外力，不能命名为目标物体的接触力。
3. `cfrc_ext` 是 body 级合成外力/力矩，单行不包含实际接触对；它不能证明
   哪个 geom 与哪个对象 geom 接触，也不能把所有外力来源自动分成夹持力与
   环境受力。
4. 读取异常时旧代码写入 `0.0`。缺失、失效和无接触因此不可区分，可能把
   采集故障伪装成低力合规。

## 2. 新的纯函数边界

### 2.1 输入与对象映射

`record_object_contact_wrench(cfrc_ext, binding=..., timestamp_s=...,
sensor_valid=...)` 要求：

| 字段 | 要求 |
| --- | --- |
| `cfrc_ext` | body-row 集合，按明确 `body_id` 选择一行；不能传没有身份的裸六维向量 |
| `object_id` | 不为空，来自对象注册表 |
| `body_id/body_name` | 非负整数和稳定名称，明确对应 MuJoCo body |
| `geom_ids/geom_names` | 非空且一一对齐的对象 geom 清单 |
| `frame` | 显式为 `world`；坐标变换不能隐含 |
| `geom_attribution` | 固定为 `body_aggregate`，表示 cfrc_ext 不能定位实际接触 geom |
| `timestamp_s` | 有限的仿真时间；缺失为 unknown |
| `sensor_valid` | 必须显式为布尔值；缺失或 false 不得填零通过 |

输出的 `ObjectContactWrench` 永远保存：

```text
force_n   = cfrc_ext[body_id][0:3]   # 3-D, N
torque_nm = cfrc_ext[body_id][3:6]   # 3-D, N*m
```

记录和评测结果分别给出 `force_norm_n` 与 `torque_norm_nm`，没有混合
`wrench_norm` 字段。异常 shape、非有限值、缺 body row、缺对象/body/geom
绑定、缺时间或缺传感器有效位都保留 issue 并进入 `unknown`。

### 2.2 三态对象力判定

`evaluate_object_contact_force` 只对三维 force 求范数，并使用预注册的
`ContactForceSpec(min_force_n, max_force_n, require_contact)`：

| 情况 | 数值判定 |
| --- | --- |
| 有效 force 为零 | `contact_state=no_contact`；若要求接触或有正下界则 `fail` |
| 有效 force 在上下界内 | `pass` |
| 有效 force 超上界或低于下界 | `fail` |
| 缺数据、传感器无效、身份/单位/形状不明 | `unknown` |
| 没有任何力约束 | `unknown`，避免“无约束自动通过” |
| 将六维 force+torque 当作 force | `unknown`，issue=`mixed_wrench_not_force` |

模块的数值 `pass` 只表示测试输入满足结构化数值边界。默认
`calibration_verified=False`，因此 `physical_qualification` 仍为
`unknown`；当前工程没有任何物理校准数据可供放行。

## 3. 测试覆盖

`tests/test_v2_contact_force_contract.py` 为标准库 `unittest` 测试，覆盖：

* 无接触零向量被明确识别，并在要求接触时失败；
* 受控值保持 3-D force N 与 3-D torque N·m 分离，分别计算 5 N 和 0.2 N·m；
* 六维混合向量作为 force 被拒绝并返回 unknown；
* 缺失 `sensor_valid`、缺 body row、缺 object/geom mapping 返回 unknown；
* 空力约束不会自动通过；
* 输出 JSON 保留独立单位字段，不生成混合范数。

这些是纯函数结构测试，不启动 MuJoCo、不读取真实 `cfrc_ext`，也不证明
仿真力与真实传感器的比例关系。

## 4. Gate1 放行前工作项

1. 在统一 recorder 中建立对象注册表：`object_id -> body_id/body_name ->
   geom_ids/geom_names`，每次 reset、episode 和 stage 都写入 immutable
   provenance；body aggregate 不能冒充 contact pair。
2. 若评测需要实际接触对，读取 `ncon`、`contact.geom1/geom2`，并通过
   `mujoco.mj_contactForce` 按接触对提取，再将接触力/力矩明确变换到声明的
   frame。`cfrc_ext` 只能作为 body 外部 wrench 辅助通道。
3. 保留 simulator time、control step、physics substep、对象/body/geom
   身份和 sensor validity；缺采、reset、异常和 sensor failure 分段或进入
   unknown，禁止异常回退零值。
4. 完成力单位、力矩参考点/坐标系和符号的校准。使用预注册的已知质量、已知
   接触载荷和变化载荷检查响应；单独核对夹爪对物体的力与物体对环境的力，
   不能使用 `cfrc_norm`、kp 或夹爪开度替代。
5. 对连续阶段按实际 `dt` 汇总峰值、持续时间和冲量（N·s），不跨 reset、
   sampling gap 或无效片段求导/积分；上下界来源必须是独立校准配置。
6. 由独立最终审计复核 object mapping、传感器有效率、无接触/受控接触响应
   和校准记录后，才可以把 numeric result 转为物理 qualification。当前
   Gate1 尚未达到这一步。

## 5. 当前结论边界

当前可签收的是“拒绝混合单位、保留对象/body/geom 证据、缺失进入 unknown”
的纯函数接口及其测试。尚不可签收对象级真实接触力、夹持力、环境载荷、力矩
校准、接触对级归因或任何真实机器人物理结论。此前 Gate0 Lift/Stack 的任务
成功也不改变这一边界。
