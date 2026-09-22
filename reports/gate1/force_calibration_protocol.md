# Gate1 对象接触力校准样例协议（开发版）

状态：**样例审计接口与结构测试已新增；4090 已完成 body wrench 受控通道
样例，但接触对和真实传感器仍未校准。**

本文件配套：

- `synthesis_v2/force_calibration.py`
- `tests/test_v2_force_calibration.py`

这两个文件只做纯函数输入审计和固定夹具测试。实际 4090 结果由
`scripts/gate1_force_calibration.py` 产生，并在 `force_calibration_results.md`
单独报告。

## 1. 需要校准的对象

现有记录器在 `libero.py:526-529` 和 `robosuite_base.py:453-458` 读取
`sim.data.cfrc_ext`，再对每个 body 的六维行直接求范数并取最大值。这个
结果不能作为校准后的对象接触力，原因是：

1. `cfrc_ext` 六维行必须按 MuJoCo 空间向量拆成 `torque[0:3]`（N·m）和
   `force[3:6]`（N）；二者不能合并求一个“力范数”。`mj_contactForce` 则是
   `[force(3), torque(3)]`，两种布局不得混用。
2. `cfrc_ext` 是 body 级外部 wrench，可能汇总多个外部作用，行本身不携带
   实际接触 geom 对。它不能从全 body 最大值推导目标对象或接触对。
3. 读取异常不能回退到 `0.0`。缺失、无效传感器和真实无接触必须分别记录。
4. 物理单位、参考点、坐标系、符号和模拟器到真实传感器的比例关系尚无独立
   校准证据。

## 2. 样例记录格式

每个 `CalibrationSample` 是一个不可变快照，至少包含：

| 字段 | 约束 |
| --- | --- |
| `timestamp_s` | 有限仿真时间；窗口内严格递增，实际 `dt` 不超过预注册上限 |
| `reset_id` | 每帧明确且同一窗口不能跨 reset |
| `object_id` | 来自对象注册表，不能从 body 最大值猜测 |
| `body_id/body_name` | 明确的 MuJoCo body 身份 |
| `geom_ids/geom_names` | 非空、一一对齐；`cfrc_ext` 只能标记 `body_aggregate` |
| `force_n` | 恰好 3 个坐标，单位 N |
| `torque_nm` | 恰好 3 个坐标，单位 N·m |
| `sensor_valid` | 必须显式为 `true` 才能进入数值判断 |
| `frame` | body wrench 明确为 `mujoco_subtree_com`；接触对另用 `contact_local` |
| `known_load_n` | controlled-load response 样例的外部已知载荷参考，仅用于响应顺序检查 |

`CalibrationCase.executed` 是防止“计划样例冒充结果”的硬标记。为
`false` 的样例即使填了看似合理的数值，也只返回 `unknown` 并记录
`case_not_executed`。

## 3. 必须覆盖的校准/异常案例

当前注册 13 类（under/over action 分开计数），包括至少要求的 12 类：

| 类型 | 结构审计行为 | 期望三态 |
| --- | --- | --- |
| `no_contact` | 有效三维 force 为零，区分“无接触”与缺失 | `pass`（仅表示无接触样例成立） |
| `controlled_load_response` | 至少两个不同 `known_load_n`，实测 force 随载荷不下降且有变化 | `pass` 或异常时 `fail` |
| `missing_sensor` | `sensor_valid` 缺失/false 或向量缺失 | `unknown` |
| `wrong_force_units` | force 单位不是 N 或缺失 | `unknown` |
| `wrong_torque_units` | torque 单位不是 N·m 或缺失 | `unknown` |
| `wrong_object` | `object_id` 与注册表不一致 | `unknown` |
| `wrong_body` | body id/name 与注册表不一致 | `unknown` |
| `wrong_geom` | geom id/name 映射不一致 | `unknown` |
| `mixed_wrench` | 把六维 force+torque 塞入一个 force 字段 | `unknown`，拒绝混合范数 |
| `timestamp_gap` | 重复/逆序时间或 `dt` 超界 | `unknown` |
| `reset_gap` | 一个窗口出现多个 reset id | `unknown` |
| `under_action` | 有效 force 全部低于注册的最小作用力 | `fail` |
| `over_action` | 有效 force 全部高于注册的最大允许力 | `fail` |

如果异常类型没有实际异常，审计返回 `fail` 和
`anomaly_not_reproduced`，避免把错误标注的夹具误称为通过。任何结构缺失
优先进入 `unknown`。

## 4. 运行真实校准前的协议

1. 冻结对象注册表，记录每个对象的 body 和所有 geom；需要精确接触对时，
   另读 `ncon/contact.geom1/contact.geom2` 和 `mujoco.mj_contactForce`，不要
   用 `cfrc_ext` 冒充单个接触对。
2. 固定 MuJoCo timestep、控制频率、采样位置和 frame；每条样例记录
   `timestamp_s`、control step、physics substep、reset/episode/stage id。
3. 先采无接触基线，再施加至少两个已知、不同且可重复的受控载荷。分别记录
   force 三向量和 torque 三向量，检查响应方向、单调性、重复性和噪声范围。
4. 明确 torque 的参考点和坐标变换。不能把物体对环境的外力、夹爪对物体的
   力、控制命令、kp 或夹爪开度互换命名。
5. 预注册单位换算、允许误差、接触阈值、最大允许力和有效采样率；校准样例
   中的异常必须保留，不能因为失败而删除。
6. 由独立评测成员检查原始日志、对象映射、时间轴、传感器有效位和已知载荷
   记录，形成校准版本与签核。没有这份外部证据时，所有
   `physical_qualification` 保持 `unknown`。

## 5. 当前测试和限制

`tests/test_v2_force_calibration.py` 包含 18 个标准库 `unittest` 测试，覆盖
上述全部注册类型、受控响应平坦失败、未执行样例、套件覆盖缺失、JSON 输出
及输入不变性。测试只使用合成夹具；它们验证的是“缺失与异常不会被误判为
校准通过”的结构规则，不是实际接触力测量。

当前已完成 4090 上 12 个 table-support/悬空/受控外加载荷样例，body wrench
通道的布局、单位和输入响应已通过；详见 `force_calibration_results.md`。

当前尚未完成：

- 接触法向已知的 `mj_contactForce` 标定夹具与阈值冻结；
- 多接触求和、符号和 contact frame 变换校验；
- 真实传感器对照、重复性/噪声误差和阈值冻结；
- 任何真实机器人或真实抓力结论。

因此本轮可签收结构审计和 MuJoCo body 通道校准，不能写作“对象接触力已完整
校准”，也不能据此发布物理方式/JPR 结论。
