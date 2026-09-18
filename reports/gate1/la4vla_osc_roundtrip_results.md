# LA4VLA OSC_POSE adapter 与 30 步物理控制回放

日期：2026-09-18  
代码：`bca9046906a10277d9d2692bc01314a9bd131cf8`  
结论：**adapter 已实现，30/30 控制步已执行；候选仍不冻结。**

## 接口实现

`synthesis_v2/la4vla_osc_adapter.py` 将相邻、与 `qpos` 同步的末端位姿转换为
LA4VLA 评估时使用的 7 维 RoboSuite `OSC_POSE` 命令：

- 平移使用 `position_after - position_before`，逐轴按 0.05 m 归一化；
- 旋转使用 `R_after @ R_before.T` 的最短 axis-angle，逐轴按 0.5 rad 归一化；
- gripper 保留 `-1=open, +1=close`；
- 默认拒绝越界，只有显式 `allow_clipping` 才能裁剪并记录被裁剪维度；
- 动作经过 JSON 写入/读取后才送入控制器。

控制器配置为 RoboSuite 1.5 的 `OSC_POSE`、20 Hz、`kp=150`、临界阻尼、
delta 输入、base 参考系；本场景 base 朝向为单位四元数。配置 SHA-256 为
`2108e558c12c005ead58da406e1374a8dbbeb3ace30b69b115a0b6d3ae06d121`。

教师记录器使用独立 `MjData` 从保存的 `qpos` 调用 `mj_kinematics`，因此末端位姿
与状态严格同步，又不会修改在线动力学和 solver warm start。Lift 766/766、
Stack 1503/1503 的原始 JOINT_POSITION 动作严格回放均通过 `1e-8`，功能结果均
成功。

## 最终运行结果

证据目录：

```text
/data/gaochuanhao/mannercap_runs/gate2_action_adapter_source_20260918_attempt02
/data/gaochuanhao/mannercap_runs/gate2_la4vla_osc_roundtrip_20260918
```

从 Lift 和 Stack 各等距抽取 15 条，共 30 条。每条样本都从 MJB 独立重建环境、
恢复 `qpos/qvel/act/time/qacc_warmstart`、把 OSC 零空间参考绑定到恢复状态，再执行
一个真实控制步。

| 指标 | 结果 |
| --- | ---: |
| 物理 OSC 控制步 | 30/30 |
| adapter / execution error | 0 / 0 |
| 裁剪 | 0 |
| JSON 动作精确 round-trip | 30/30 |
| 恢复前最大位置/方向误差 | 0 m / 0 rad |
| 位置误差 mean / median / max | 0.921 / 0.212 / 10.381 mm |
| 方向误差 mean / median / max | 0.003708 / 0.000185 / 0.046828 rad |
| 全量位置偏差向量 | `[-0.243, +0.611, -0.128]` mm |
| 覆盖程序阶段 | approach, descend, grip, lift, transport, final, retreat, other |

两个任务的 transition 0 都包含一次较大的初始化位姿变化；OSC torque controller
在单个 50 ms 控制周期内没有完全到达目标，分别产生约 10.14 mm 和 10.38 mm 的
位置误差，以及约 0.0468 rad 的方向误差。该边界样本保留在全量统计中。

其余 28 条非初始化样本的位置误差 mean / p95 / max 为
0.254 / 0.622 / 0.769 mm，方向误差 mean / p95 / max 为
0.000639 / 0.002134 / 0.002177 rad。这是描述性结果，不是事后通过阈值。

## 诊断迭代

所有失败或偏差较大的尝试均保留：

1. attempt01 未把 OSC 零空间参考绑定到恢复状态，位置均值 11.418 mm、方向均值
   0.04869 rad；
2. attempt02 绑定控制器后，位置均值降至 0.845 mm、方向均值降至 0.003475 rad，
   同时暴露旧记录中的 EEF 派生状态滞后一帧；
3. attempt03 使用与 `qpos` 同步的 EEF 记录验证恢复误差为 0；
4. attempt04 在相同源轨迹上写出全量、非初始化、按任务和按阶段的最终统计。

## 放行边界

本运行证明 adapter 的代数、序列化和 RoboSuite OSC 控制路径均可执行，也满足
“至少 30 条真实控制回放”的数量要求。它没有调用 LA4VLA checkpoint 做推理，
没有评测 Lift/Stack 的学生闭环任务成功率，也没有独立冻结跟踪误差阈值。由于
两个初始化边界样本存在明显一步跟踪误差，`fixed_student_approved` 继续为
`false`，`action_interface.py` 继续以 `adapter_not_verified` 失败关闭。

本阶段外部 LLM/API 调用数为 0，不需要 API key。
