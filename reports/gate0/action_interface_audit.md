# Gate 0：教师动作、记录与学生导出接口审计

日期：2026-09-16
审计角色：D（数据与学习验证）
审计范围：教师动作通道、低层记录、学生导出和 v2 轨迹边界
审计性质：静态代码/文档审查；本报告没有启动仿真回放，也没有训练学生模型。

## 1. 依据与结论边界

本审计读取了用户提供的实施合同
`C:\Users\Chuanhao Gao\.codex\attachments\c853c7fb-cc90-4061-baac-ea71412251e1\pasted-text.txt`
以及当前仓库。合同明确这是“拟实施与验收合同；不是已完成方法或实验结果”（附件第 5 行），并要求动作回放、测量校准、失败保留和闭环学习证据（附件第 31、169--171、213--217 行）。因此，下面把**代码能够发出/保存的信号**和**已经被物理验证的信号**分开叙述。

**结论：现有学生导出主要走 LIBERO 路径；当前 Manner Lift/Stack 教师驱动走另一条 RoboSuite 路径，两者动作语义尚未闭合。** LIBERO 底层执行 7 维关节增量加 1 维夹爪命令，并导出一个 7 维候选动作。RoboSuite Manner 驱动把 7 维绝对关节目标和 1 维夹爪命令交给绝对位置控制器，并通过隐藏的速度 profile、kp 和夹持参数改变行为。当前不能把 RoboSuite 教师轨迹直接视为已有 LIBERO/学生导出，也不能把任一路径作为可回放数据交付。

仓库当前没有发现已提交的 `steps.jsonl` 或 parquet 产物；文件搜索命中的 parquet 只有转换脚本。因此本文的“当前行为”来自源码路径，不把源码能力写成已经生成的轨迹、回放或学习结果。

## 2. 实际链路

```text
Stage-CAP 代码块
  -> CodeExecutionEnvBase.step(code)
  -> Franka*Api 高层函数
  -> FrankaLiberoEnv.handle.step(raw 8-D action)
  -> _record_trajectory_step(state_before, raw action, state_after)
  -> steps.jsonl + trajectory_images/ + stage_trace.json
  -> LeRobot / HFVLA parquet 转换
```

* `CodeExecutionEnvBase.reset` 调低层 reset、合并观测并用 reset 观测初始化 `INPUTS`；`step` 执行字符串代码块后再计算 reward、完成标记和 sandbox 状态（`stagecap_robosuite_libero_repro_20260714/capx/envs/tasks/base.py:L391-L449`）。这里的 `action` 类型是代码字符串，不是学生动作头的数值向量。
* 每个代码块可能调用几十个低层步。`FrankaLiberoEnv._step_once` 和 `move_to_joints_blocking` 才是将 8 维数值动作送入 `handle.step` 的位置（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L291-L308`、`L552-L571`）。
* 普通 trial 和 stage trial 各自实现一次写盘逻辑，分别写 `capx_libero_steps_v0.2` 和 `stagecap_libero_steps_v0.1`（`stagecap_robosuite_libero_repro_20260714/capx/envs/trial.py:L438-L490`；`stagecap_robosuite_libero_repro_20260714/capx/envs/stage_trial.py:L5789-L5850`）。两种元数据都缺少 v2 所需的 reset、源尝试和 lineage 字段。
* `ExecutionHistory`/`ExecutionStep` 是 UI/审计日志，记录工具名、文字、墙钟 ISO 时间和可选 base64 图片（`stagecap_robosuite_libero_repro_20260714/capx/utils/execution_logger.py:L50-L103`、`L266-L343`），不能替代低层状态—动作轨迹。

## 3. 教师接口盘点

### 3.1 可被普通 API 映射的接口

`FrankaLiberoApi.functions()` 注册了观测、目标物位姿、抓取位姿采样、`goto_pose`、开/关夹爪、几何点/OBB 和回 home；关节轨迹执行函数在映射中仍是注释状态（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/libero.py:L74-L91`）。所以常规教师程序表达的是高层目标和几何调用，不是一个稳定的 Cartesian/force action head。

* `get_observation` 可返回 agentview 与 wrist 的 RGB/depth、内外参/位姿，以及 robot joint/cartesian state；文档给出的 Cartesian state 是位置、WXYZ 四元数和归一化夹爪量（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/libero.py:L93-L113`）。低层实现实际在 `get_observation` 组装两路相机和 8 维 `robot_joint_pos`/`robot_cartesian_pos`（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L720-L804`）。
* `goto_pose` 接受米制 XYZ、WXYZ 四元数和可选 z approach，先加 TCP offset，再 IK 到 7 个关节并调用 blocking 执行（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/libero.py:L215-L277`）。目标位姿和 IK 结果没有逐行保存到 `steps.jsonl`。
* `open_gripper`/`close_gripper` 只是设置开度分数并分别保持 40/60 个低层步（同文件 `L279-L297`）。

### 3.2 特权教师路径

`FrankaLiberoEnv` 构造函数的 `privileged` 默认值是 `True`（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L76-L92`）。特权 API 注册了对象位姿、全部对象/抽屉状态、直接 pick-and-place 和其他高层谓词操作（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/libero_privileged.py:L13-L56`）。这符合附件关于“教师可用当前仿真对象状态、阶段事件和属性”的设定，但它是特权辅助合成，不是视觉推理证据；若学生输入未显式隔离，存在信息条件不对称。

另有一个 `FrankaControlPrivilegedApi`，其 `functions()` 当前只注册对象/抓取位姿、`goto_pose`、开/关夹爪（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/control_privileged.py:L19-L57`）。类中虽然实现了带 profile 的位姿调用、刚度和部分夹爪接口（`L184-L256`），但这些方法不在该 `functions()` 映射中；因此“源码存在”不等于“当前教师运行时能调用”。`FrankaLiberoPrivilegedApi` 的交互式 Cartesian 调用也在高层方法中更新目标、逐周期重新 IK 后只调用关节 blocking 步（`stagecap_robosuite_libero_repro_20260714/capx/integrations/franka/libero_privileged.py:L1298-L1368`）。逐周期目标、速度请求和重规划事件没有写入低层轨迹行。

### 3.3 底层动作实际含义

`move_to_joints_blocking` 将目标关节位置（7 个，单位 rad）与当前关节差乘 `control_freq`，拼成 7 维 joint delta；第 8 维来自开度分数，并按 `1.0=open -> -1.0`、`0.0=closed -> +1.0` 重映射后送入 `handle.step`（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L264-L305`）。`control_freq` 默认 20 Hz（同文件 `L76-L115`）。因此当前物理执行通道的可确定部分是：

| 通道 | 当前教师能否发出 | 代码中实际语义 | 是否进入学生导出 |
|---|---|---|---|
| 7-D joint delta | 可以，blocking/profiled 运动都会走这里 | rad 目标差乘 20 Hz 后作为控制输入；实际控制器响应未单独保存 | 作为 raw 8-D 的前 7 维保留，但默认导出不用它 |
| 1-D gripper command | 可以 | 开度分数映射到控制器符号，非力值 | 候选动作最后 1 维沿用该 raw command |
| Cartesian pose target | 高层可以 | 米制 XYZ + WXYZ，经 TCP offset 与 IK 变成关节目标 | 目标、IK 解和执行时刻均未保存 |
| 速度/加速度/jerk profile | 类中有实现；是否注册需单独确认 | `v_max` rad/s、`a_max` rad/s²、`jerk_limit`；最终仍落到关节控制 | 未进入 `action` 或导出字段 |
| joint stiffness / impedance | 有 `set_joint_stiffness` 路径；`kp_ori` 被接受但不单独应用 | 改 joint controller kp，不是已校准的 Cartesian 阻抗 | 只可能在独立 manner telemetry 中出现，学生动作无此维度 |
| grip force / width | 高层参数可传 | force 是归一化 `[0,1]`，再映射为开度；width 是米制目标 | 没有实测夹持力字段，学生只见 gripper state/command |
| environment contact force/torque | 不能作为动作直接发出；只能读诊断量 | 记录的是 `cfrc_ext` 六维量按 body 取 norm 的最大值 | 不导出 |
| object pose/属性/阶段事件 | 特权教师可读或使用 | 当前仿真真值/高层谓词 | 不进入学生主特征；只在 stage trace 或未绑定日志中间接出现 |

### 3.4 当前 Lift/Stack 的 RoboSuite Manner 路径

`run_manner_ablation.py` 的 Lift/Stack 入口调用 `FrankaControlPrivilegedApi.goto_pose_with_profile`，经 IK 得到 7 维关节目标，再由 `RobosuiteBaseEnv.move_to_joints_profiled` 生成绝对关节 waypoint。控制器配置明确为 `JOINT_POSITION` 和 `input_type="absolute"`。`_build_action` 先构造 7 维关节目标加两个相同 gripper 值，再按 `_ACTION_SLICE=-1` 传入 8 维 RoboSuite 动作，即 7 维绝对关节目标加 1 维夹爪命令。

这与上文 LIBERO 学生路径的 7 维 joint delta 加 1 维夹爪命令不是同一个动作接口。速度、加速度和 jerk 只决定 waypoint 时间序列，`set_joint_stiffness` 直接修改控制器 kp，夹持强度又通过 `close_gripper_partial` 的内部策略生效；这些方式通道都不在学生候选动作向量内。因此首轮学习验证必须先选择并冻结统一的学生可执行接口，或证明固定控制器可由学生动作重现这些效果。

Gate 0 的 `scripts/gate0_smoke.py` 现在为 RoboSuite 路径保存真实 `state_t/action_t/state_t_plus_1`、控制步与物理子步，但未保存 RGB，且初态快照仍不完整。它不能通过现有 LIBERO 转换器直接成为训练数据。本报告后续关于 `steps.jsonl`、两路图像和 HFVLA/LeRobot 导出的结论只适用于 LIBERO 数据路径。

`goto_pose_with_profile` 的参数和 `set_impedance_stiffness` 的语义可见于
`control_privileged.py:L184-L256` 与 `libero_privileged.py:L1391-L1512`，但不能据此声称学生能够学习或回放这些维度。

## 4. 记录器实际保存什么、保存在哪个时刻

### 4.1 每行的 state/action 关系

`_record_trajectory_step` 在调用 `handle.step(action)` 前采 `state_before`，step 返回后采 `state_after`；行中保存 `observation.state=state_before`、`next_observation.state=state_after`、raw state、8-D `action`/`action.raw_joint_gripper` 及 7-D `lerobot_action_candidate`（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L882-L921`）。所以行级 transition 的方向是可读的：

```text
state_before -- handle.step(raw joint_delta + gripper) --> state_after
```

但候选动作只是 `after[:6] - before[:6]` 加 raw gripper 标量（`L871-L880`），并且记录中明确写了 `validated_as_ground_truth: false`（`L912-L920`）。它没有证明这个 7-D 差分能被目标 LeRobot/LIBERO 控制器解释为同一物理动作。

### 4.2 时间、控制步和 reset 偏移

* 行的 `timestamp` 是 `frame_index / _control_freq`，不是 `sim.data.time` 或一次控制调用的实际持续时间；`sim_step_index` 则在 `handle.step` 后递增计数后写入（`L922-L927`）。没有 `dt`、控制命令时刻、观测采样时刻或 controller settle 时长。
* reset 先把 `_sim_step_count` 置 0、清空轨迹，然后执行 10 次 settling `_step_once()`，最后清空轨迹但不把 `_sim_step_count` 置回 0（`L220-L235`）。因此标准路径的第一条保留行可以有 `frame_index=0/timestamp=0.0`，同时 `sim_step_index=11`；这是一个可定位的实际时间轴偏移。
* trial 在 reset 返回后才打开视频采集（`stagecap_robosuite_libero_repro_20260714/capx/envs/trial.py:L1008-L1023`），stage trial 同样在 reset 后打开（`stagecap_robosuite_libero_repro_20260714/capx/envs/stage_trial.py:L61-L77`）。因此标准路径不把 reset settling 行写入 `steps.jsonl`；但行中也没有 reset id 或 controller/RNG snapshot 来证明初态可恢复。
* `CodeExecutionEnvBase` 只把整个代码块作为一次外层 step，记录器的低层行没有 `code_block_index`、API 调用边界或 stage instance id（`stagecap_robosuite_libero_repro_20260714/capx/envs/tasks/base.py:L414-L449`）。Execution logger 的 ISO 墙钟时间不能和低层行一一配对（`stagecap_robosuite_libero_repro_20260714/capx/utils/execution_logger.py:L54-L81`）。

### 4.3 图像时刻和路径

`enable_video_capture` 清空 buffer 后立即存一帧；之后 `_step_once` 每 4 个仿真步记录一帧（`stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:L941-L996`）。每行用 `image_frame_index=frame_index//subsample_rate` 指向稀疏 frame（`L893-L907`）。写盘函数没有检查引用是否越界，而是使用 `frames[min(frame_index, len(frames)-1)]`；缺帧会静默重复最后一帧（`stagecap_robosuite_libero_repro_20260714/capx/envs/trial.py:L531-L548`，stage 版本同样如此：`stage_trial.py:L5895-L5912`）。没有观测帧时间、仿真帧号或图像 hash，不能证明 `image_t` 与 `state_t/action_t` 同步。

普通与 stage writer 的 schema 也不一致：普通 writer 的 required fields 只有 state、action、timestamp、frame/episode/task index（`trial.py:L461-L489`）；stage writer 另列两路 image field，但仍没有 `next_observation.state`、`dt`、reset 或 lineage（`stage_trial.py:L5814-L5846`）。

## 5. 学生导出接口的实际行为

### 5.1 LeRobot 转换

`convert_stagecap_steps_to_lerobot.py` 将 image 两路定为 256×256×3，state 定为 8 维，action 定为 7 维，并默认 `--fps 10 --action-field lerobot_action_candidate`（`stagecap_robosuite_libero_repro_20260714/scripts/convert_stagecap_steps_to_lerobot.py:L18-L57`）。转换循环只加载 JSONL、检查 state/action 的 shape 并 `dataset.add_frame`；生成的 conversion manifest 仅保存 trial 路径、stage trace、质量层级和帧数（`L75-L136`、`L167-L224`）。它不检查：

* timestamp 是否递增，或是否与仿真时间/真实 dt 相符；
* `next_observation.state` 是否存在并与动作一致；
* frame/sim step 是否连续，是否跨 reset 或采样缺口；
* trajectory/source attempt/lineage/pair 是否可追溯；
* raw joint action 与 7-D candidate 是否在同一控制器语义下等价。

### 5.2 HFVLA parquet 转换

HFVLA schema 只有两路图像、state、action、timestamp、frame/episode/index/task index（`stagecap_robosuite_libero_repro_20260714/scripts/convert_stagecap_manifest_to_hfvla_parquet.py:L36-L49`）。脚本宣称原始 20 Hz 通过 `sample_stride=2` 变成 10 Hz（`L193-L215`），但实现按 JSONL 源行号抽样（`L434-L444`），不是按源 timestamp 或 sim step 抽样。写出的 `timestamp` 是 `frame_index/fps` 新造的 export clock（`L377-L407`），源时间被丢弃。

更严重的是 `_action_for_sampled_step` 对相邻保留行重新取 `next_state[:6]-state[:6]`，再取当前 candidate 的最后一维；最后一行则直接复用 candidate（`L447-L460`）。这不是对原始 `next_observation.state` 的验证，而是导出期的二次合成。图像只检查路径存在并读取（`L473-L493`），没有同步检查。由此产生的 parquet 不能作为“动作已回放”的证据。

### 5.3 现有 inspector 的覆盖边界

`inspect_stagecap_vla_trials.py` 只计数 JSONL、检查 state=8、candidate action=7、两路图片存在和 task/privileged finalizer（`stagecap_robosuite_libero_repro_20260714/scripts/inspect_stagecap_vla_trials.py:L69-L153`）；质量层级由 sandbox/reward/task completed/记录完整度决定（`L156-L165`）。它没有检查时间单调、dt、sim step、reset、lineage、`next_observation.state`、动作语义或回放。因此 `gold` 的含义是当前 shape/任务筛选规则下的完整记录，不是物理可回放或学生可学。

## 6. 物理测量通道的证据等级

Manner telemetry 的数据类声明了 `joint_vel`、`joint_acc`、kp、gripper qpos 和 `cfrc_norm`（`stagecap_robosuite_libero_repro_20260714/capx/envs/manner_validator.py:L159-L177`）。LIBERO 记录器实际把 `joint_acc` 固定为全零（`libero.py:L502-L506`），把 `sim.data.cfrc_ext` 的六维向量按 body 求范数再取最大值（`libero.py:L525-L539`）。这不能区分 force 与 torque，也没有对象级力校准；不能把它命名为真实抓力或环境载荷。

验证器对 jerk 固定使用 `dt=0.05`，不是从每个样本的时间轴计算（`manner_validator.py:L526-L557`）；缺失 telemetry 字典会被 `_coerce_snapshot` 用零速度/零加速度/零外力等默认值补齐（`L619-L634`）。无 telemetry 虽返回 `UNCERTAIN`，但一旦部分字段缺失，默认填充会掩盖缺采问题。几何 evidence 的 duration 也按 `len(snaps)*0.05` 写入（`L651-L665`）。这正触及附件要求的“缺采、reset 和失效不填零冒充合规”，当前链路尚未满足。

## 7. 回放阻塞清单

按阻塞严重度排序如下。任何一项 P0 未解决，都不能把导出包交付为“可回放”。

| 级别 | 阻塞 | 代码证据 | 影响 |
|---|---|---|---|
| P0 | **动作语义未闭合**：raw 8-D 是 joint delta+gripper，学生默认吃由状态差分得到的 7-D candidate；candidate 明确标记未验证 | `libero.py:L871-L921`；`convert_stagecap_steps_to_lerobot.py:L192-L212` | 学生输出无法确认能被同一 controller 执行；不能开始 30 条动作回放 |
| P0 | **没有真实 dt/仿真时间**，reset settling 后 sim step 与记录 timestamp 已错位 | `libero.py:L220-L235`、`L922-L927` | 不能按实际动作时刻重放，也不能可靠求速度/jerk 或对齐视觉 |
| P0 | **没有 reset/controller/RNG 快照、trajectory/source attempt/lineage** | 两个 writer 的 metadata：`trial.py:L461-L489`、`stage_trial.py:L5814-L5846`；episode 行固定为 0：`libero.py:L922-L927` | 窗口可能跨 reset，配对/去重/同初态恢复不可审计，训练泄漏无法排除 |
| P0 | **图像对齐没有证据且缺帧会静默复制末帧** | `libero.py:L941-L996`；`trial.py:L531-L548` | `image_t/state_t/action_t` 可能错配；视觉学生数据不能视为同步 |
| P1 | **导出器按源行号抽样、重造 timestamp、重算 action** | HFVLA `convert_stagecap_manifest_to_hfvla_parquet.py:L377-L460` | 缺口/重复时间/跨 reset 不会被发现，导出动作不再是原始动作 |
| P1 | **高层方式通道未逐行记录**：profile、kp、force/width、stage scope 没有与每条动作绑定；部分接口甚至未注册 | `control_privileged.py:L46-L57`、`L184-L256`；`libero_privileged.py:L1298-L1368`、`L1391-L1512` | 即使视频成功，也无法证明“只在指定阶段改变方式”或重现控制器状态 |
| P1 | **接触/加速度测量不具备校准证据** | `libero.py:L502-L539`；`manner_validator.py:L526-L557`、`L619-L665` | force/torque、jerk、阻抗结论可能是合成默认值或混合量 |
| P1 | **教师特权与学生观测不对称**：对象真值/属性可供特权教师使用，导出只有 RGB、state、7-D action | `libero_privileged.py:L33-L56`；`convert_stagecap_steps_to_lerobot.py:L18-L38` | 不能把特权合成成功解释成学生从视觉学会方式或上下文 |
| P2 | **其他 simulator recorder 不共享 LIBERO 8/7 schema**，例如若干后端写自定义 state/action、wall time、固定 episode/task=0 且无图像 | `isaacgym_allegro_kuka.py:L315-L324`；`robel_dclaw.py:L377-L386`；`mfr_screwdriver.py:L404-L413`；`dexscrew_xhand_nutbolt.py:L625-L634` | 不能把这些记录直接送入当前 LIBERO 学生转换器，需另立 adapter/contract |
| P2 | **质量 inspector 不是 replay gate** | `inspect_stagecap_vla_trials.py:L69-L171` | `gold` 只能表示当前检查通过，不能作为回放或学习可用性的替代证据 |

## 8. 新增的 v2 轨迹合同与测试

为让缺口可以在不改旧模块的情况下被机器检查，本轮只新增：

* [`synthesis_v2/trajectory_contract.py`](../../synthesis_v2/trajectory_contract.py)：纯标准库、无副作用的 `validate_trajectory`/`validate_window`。默认要求 8-D state、8-D raw action、7-D candidate、两路图像引用、`timestamp`、实测 `dt`、连续 `frame_index`/`sim_step_index`、常量 `episode_index/task_index`，以及每行的 `trajectory_id`、`source_attempt_id`、`lineage_id`、`reset_id`。它拒绝缺字段、非法维度/非有限值、非单调时间、dt 与相邻 timestamp 不符、非连续帧、reset marker、跨 reset 和 lineage 变化；可选 `image_root` 才检查实际图片文件。
* [`tests/test_v2_trajectory_contract.py`](../../tests/test_v2_trajectory_contract.py)：8 个小测试覆盖有效窗口、缺字段、非单调时间/dt mismatch、跨 reset、reset marker、lineage 变化、现有 LIBERO 行对 v2 不足、expected lineage 及不修改输入。

合同的严格默认值有意让当前 `steps.jsonl` 记录失败：现有 writer 没有 `dt` 和 lineage/reset 字段。这个失败是接口审计结果，不是对真实轨迹质量的统计。

### 测试证据

在 `C:\Users\Chuanhao Gao\Desktop\manner_cap\upstream` 执行：

```text
python -m py_compile synthesis_v2/trajectory_contract.py tests/test_v2_trajectory_contract.py
  -> pass

python -c "import runpy; ns=runpy.run_path('tests/test_v2_trajectory_contract.py'); tests=[v for k,v in ns.items() if k.startswith('test_') and callable(v)]; [t() for t in tests]; print(f'{len(tests)} trajectory contract tests passed (pytest unavailable)')"
  -> 8 trajectory contract tests passed (pytest unavailable)

python -m pytest -q tests/test_v2_trajectory_contract.py
  -> 未运行：当前 Python 环境没有 pytest（No module named pytest）
```

这些测试是结构合同测试，不执行 simulator、真实控制器、图片渲染或训练；不能替代动作回放。

## 9. 放行前必须补齐的接口证据

1. 在统一 recorder 中为每条 transition 保存 `observation_t`、明确的控制输入、`observation_t_plus_1`、实测 `dt`/仿真时刻、control step 与 physics substep，并显式写单位、坐标、四元数顺序和 gripper 符号。控制命令与 measured state、force 与 torque、夹持力与环境受力必须分列。
2. 每次 reset 生成不可变 `reset_id`，同时保存能恢复 controller/RNG/初态的 snapshot 引用；每个候选生成写 `trajectory_id`、`source_attempt_id`、`lineage_id`、pair/split/label version。先按 episode/object/组合划分，再切不重叠窗口；窗口遇到 reset、缺口、无效 padding 立即停止。
3. 在学生动作接口上冻结一个可实际执行的 schema。用同一 controller 做原始动作 round-trip：导出动作 → 学生控制器 → 逐步比较目标/实际关节、末端、夹爪和时间误差。候选动作不能只靠 state delta 命名为 ground truth。
4. 给每一帧记录 capture id/仿真时间/来源 step，遇到越界、丢帧或重复时硬失败，不再 `min(..., len(frames)-1)` 静默补最后一帧。转换器只接受通过 v2 contract 的源窗口，并保留 lineage，禁止在导出阶段无审计地改时间和动作。
5. 独立校准 `cfrc_ext` 的 force/torque 单位、对象级接触量、夹持力/宽度和真实加速度；缺失数据进入 `unknown` 或失败，不能用默认零值变成通过。高层方式参数要绑定到生效的控制周期和阶段实例。
6. 完成合同要求的证据后再执行至少 30 条导出—导入—学生控制器回放和 32--64 条小样本学习检查。**截至本报告，30 条回放和 32--64 样本学习均未完成；没有真实 replay/training 结果可报告。** Gate 0 已验证 RoboSuite Lift/Stack 的任务执行和最终状态重算，但没有验证学生动作回放。

## 10. 当前可签收范围

* 可签收：静态接口清单、当前 raw/candidate 字段及其生成位置、学生两个转换器的实际重采样行为、明确的 replay blockers，以及独立 v2 结构合同和 8 个单元测试。
* 暂不可签收：物理动作 round-trip、初态/控制器恢复、图像同步、接触力校准、30 条回放、32--64 样本学习、学生泛化或任何训练收益。
* 结论边界：教师使用特权仿真对象状态的成功只能证明调用链/任务执行（待实际运行证据），不能证明学生已获得相同信息或学会方式。实施合同要求的这些后续证据仍需由统一执行器、独立评测和学生验证共同提供。
