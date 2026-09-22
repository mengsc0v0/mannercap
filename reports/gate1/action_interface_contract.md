# Gate 1：RoboSuite 教师、LIBERO 控制器、LA4VLA 与 LeRobot 动作接口合同

日期：2026-09-16  
范围：动作向量的语义、通道、单位、坐标、时刻和可表达性  
状态：开发合同；纯静态代码/结构测试；未启动仿真、控制器、API 或数据回放

## 1. 目的和边界

Gate 0 已指出，当前代码实际存在三套语义：RoboSuite 教师发送 8 维绝对关节
目标；LIBERO 底层控制器接收 8 维关节增量；默认 LeRobot 导出则写入 7 维 EEF
状态差分候选。仅凭 shape 或“student”名称不能互换它们，也不能把旧
`lerobot_action_candidate` 当作已经验证的训练或控制真值。

本合同新增纯标准库模块
[`synthesis_v2/action_interface.py`](../../synthesis_v2/action_interface.py)。它只
描述接口和执行严格结构检查，不导入 NumPy、RoboSuite 或 LIBERO，不启动仿真，
不调用控制器/API。测试位于
[`tests/test_v2_action_interface.py`](../../tests/test_v2_action_interface.py)。

## 2. 代码依据

| 事实 | 当前代码依据 |
| --- | --- |
| RoboSuite 关节控制器是 `JOINT_POSITION` 且 `input_type="absolute"` | `stagecap_robosuite_libero_repro_20260714/capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json:4-18` |
| RoboSuite wrapper 构造 7 个当前/目标关节位置和两个 gripper 值，再按 `-1=open,+1=close` 映射 | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/robosuite_base.py:164-178,212-257` |
| RoboSuite profile waypoint 使用 `v_max`（rad/s）、`a_max`（rad/s²）和可选 `jerk_limit`，最终仍逐步发送绝对关节 waypoint | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/robosuite_base.py:269-330` |
| RoboSuite `set_joint_stiffness` 修改隐藏的 joint-controller `kp`；`close_gripper_partial` 接收归一化 force 或米制 width，width 优先 | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/robosuite_base.py:336-420` |
| LIBERO blocking/profiled 路径使用 7 维 `(target-current)*control_freq`，再拼一个 gripper 命令 | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:264-305,329-406` |
| LIBERO 当前默认 `control_freq=20`；记录器的 `timestamp` 是按 frame index 合成的，旧行缺少实测 `dt` | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:76-115,922-927`；Gate 0 审计第 4 节 |
| 旧导出中的 `lerobot_action_candidate` 是 EEF 状态差分 6 维加 raw gripper 1 维，并明确标记未验证 | `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:871-920`；Gate 0 审计第 5 节 |

## 3. 版本和向量布局

动作接口版本固定为 `synthesis_v2.action_interface.v1`。接口必须按 ID 和通道
语义解释，不能按 shape 推断。

| 接口 | 向量布局 | 第 1--7 维单位/坐标 | 控制器语义 | 第 8 维 |
| --- | --- | --- | --- | --- |
| `robosuite_teacher` | `[joint_target(7), gripper_command(1)]` | `rad`；`robot_joint_coordinates`；顺序固定为 `robot0_joint1..robot0_joint7` | 每一步是绝对 Panda 关节目标；profile 可能在外部生成多个 waypoint | `normalized_command`，`-1=open, +1=close` |
| `libero_low_level_controller` | `[joint_delta(7), gripper_command(1)]` | `rad` 的单控制命令关节位移；同一关节坐标和顺序 | 目标差由测得的当前关节状态得到；当前 wrapper 以 `control_freq` 缩放后送入 `handle.step` | 同一显式 gripper 符号 |
| `lerobot_export_candidate` | `[eef_translation_state_difference(3), eef_axis_angle_state_difference(3), raw_gripper_value(1)]` | `m`、`rad`；robot base 下记录状态的逐分量差 | 没有已验证控制器语义；源码标记 `validated_as_ground_truth=false` | 原样复制 `raw_action[-1]`，语义未提升 |
| `la4vla_libero_osc_pose` | `[normalized_translation_delta(3), normalized_rotation_delta(3), gripper_command(1)]` | 归一化 controller command；逐轴缩放至 ±0.05 m、±0.5 rad | LIBERO 默认 `OSC_POSE`、`control_delta=true`、20 Hz | `-1=open, +1=close` |

旧字符串 `libero_student` 只作为 8 维底层控制器接口的兼容查找别名；它不表示
当前 7 维 LeRobot 导出，也不证明存在与该接口一致的学生策略 head。

模块中的 `nominal_control_frequency_hz=20.0` 只记录源码默认值，不能被当成
`dt`。控制输入时刻定义为 transition 开始的 `timestamp_s`；每行必须另有
实测、正数 `dt_s`，其语义是该 transition 的实际持续时间。reset settling、
图像采样时刻和控制器 settle 时间不能由本合同猜测填补。

## 4. 通道和可表达性

教师动作的可见向量只有绝对关节目标和 gripper command。以下方式字段需要在
教师侧作为 side channel 单独保留，并且不能隐式丢给底层控制候选：

| 字段 | 单位 / 坐标 | 时刻 | 学生向量可表达？ | 合同处理 |
| --- | --- | --- | --- | --- |
| `v_max` | `rad/s`，关节空间 | 后续 waypoint profile | 否 | 默认转换报 `unexpressed_teacher_field` |
| `a_max` | `rad/s^2`，关节空间 | 后续 waypoint profile | 否 | 同上 |
| `jerk_limit` | `rad/s^3`，关节空间 | 后续 waypoint profile | 否 | 同上 |
| `kp_pos` | `controller_gain`；不猜物理阻抗单位 | controller state at command | 否 | 同上 |
| `kp_ori` | `controller_gain`；当前代码接受但未单独应用 | controller state at command | 否 | 同上 |
| `grip_force` | `fraction` `[0,1]`；归一化请求，不是 N | gripper controller state | 否 | 同上；不可改名为实测夹持力 |
| `grip_width` | `m`；夹爪指间隙 | gripper controller state | 否 | 同上；当前实现 width 覆盖 force |
| contact force / torque | 诊断通道，不是 action；不能合成混合 wrench 范数 | observation diagnostic | 否 | 不进入 action；force 和 torque 必须分列 |

`HiddenTeacherControls` 会保留这些字段及其单位；严格转换默认拒绝任何已给出
的教师专属字段。只有调用者明确设置 `allow_unexpressed=True` 时，转换结果才
会带有 `unexpressed_fields` sidecar；这代表有意的有损候选，仍不代表学生能
重现这些方式。

## 5. 转换规则和显式失败

唯一实现的方向是教师绝对目标到 LIBERO 底层关节增量候选。对测量到的当前关节向量
`q_before`，转换关系是：

```text
controller_candidate.joint_delta = teacher.joint_target - q_before
controller_candidate.gripper_command = teacher.gripper_command
```

转换必须同时收到：

1. 7 维 `current_joint_rad`（不能从缺失 state、EEF pose 或默认 home pose 猜）；
2. transition `timestamp_s`；
3. 实测正数 `dt_s`（不能把 20 Hz 自动换算为 0.05 s）；
4. 版本兼容且相同的 joint 顺序、单位、坐标 frame 和 gripper 符号。

以下情况硬失败，并由异常的 `code` 区分：

| 情况 | code |
| --- | --- |
| 6/7/8 维不匹配 | `dimension_mismatch` |
| 单位不匹配 | `unit_mismatch` |
| joint frame 不匹配 | `coordinate_mismatch` |
| gripper sign 不匹配 | `sign_mismatch` |
| 缺少实际 `dt_s` | `missing_dt` |
| 缺少测得的当前关节状态 | `missing_state` |
| profile/kp/force/width 无学生通道 | `unexpressed_teacher_field` |
| 尝试把教师动作转换为 7 维 LeRobot EEF 状态差分候选 | `unverified_export_semantics` |
| 尝试在物理回放前转换为 LA4VLA OSC_POSE | `adapter_not_verified` |
| LIBERO delta 反向恢复 RoboSuite teacher | `non_invertible_conversion` |
| source/target 接口不匹配 | `interface_mismatch` |

反向转换默认关闭：一个 delta 不能独自恢复 RoboSuite 的绝对目标、控制器状态
和 hidden profile。任何未来反向 adapter 都必须以单独版本、显式 state/profile
和回放证据注册，不能由本模块猜测。

## 6. 冻结的 round-trip 候选 schema

候选记录版本固定为 `synthesis_v2.action_round_trip_candidate.v1`，由
`ROUND_TRIP_CANDIDATE_SCHEMA` 和 `RoundTripCandidate` 表达。每条候选至少保留：

```text
timestamp_s
dt_s
state_before.joint_position_rad       # 7 rad, robot joint order
teacher.action                         # 8-D absolute target + gripper
student.action                         # 8-D low-level controller candidate; compatibility field name
source_interface_schema_version
target_interface_schema_version
unexpressed_fields
replay_status = "not_run"
```

候选对象固定带有 `candidate_only=true` 和 `replay_verified=false`。构造候选只做
代数一致性和结构检查，不启动控制器、不比较执行后的状态，也不生成回放结果。
任何没有通过上述转换前置条件的记录都不能进入候选 schema。

## 7. 结构测试和当前放行边界

`tests/test_v2_action_interface.py` 共 14 个纯 Python 测试，覆盖：

* 两个 8 维接口、LA4VLA 7 维 OSC_POSE 和 7 维状态差分候选的布局与版本化通道元数据；
* rad、坐标 frame、joint order、gripper `-1/+1` 符号；
* 缺 `dt`、缺当前 state、非法维度和非法 gripper 值；
* profile、kp、归一化 force、width 的 hidden side channel；
* unit/frame/sign/interface mismatch 和不可逆方向；
* 候选 schema 的 `not_run` / `replay_verified=false` 边界及 JSON 表示。

本地验证命令：

```text
python -m unittest -v tests/test_v2_action_interface.py
python -m py_compile synthesis_v2/action_interface.py tests/test_v2_action_interface.py
```

两项均通过。2026-09-18 的后续工作已经为独立的
`la4vla_libero_osc_pose` 接口实现 adapter，并完成 30/30 个真实 OSC 控制步；
详见 `la4vla_osc_roundtrip_results.md`。该结果没有把
`lerobot_action_candidate` 提升为 ground truth，也没有绕过独立误差阈值和学生
闭环任务评测。
