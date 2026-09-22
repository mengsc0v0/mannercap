# Gate 1 / Wipe：受控载荷传感器通道校准探测

日期：2026-09-17  
状态：**已在 4090 完成可重复的传感器通道响应校准；未完成接触模型校准或安全阈值签收**

## 1. 对现有 Wipe 路径的审查

### `gate1_wipe_probe.py`

现有探测器证明了 Wipe 的 marker 任务路径可以运行，但它不是受控载荷校准：

* `named_contacts()` 调用 `mj_contactForce`，记录 contact-local 的 3D force 和 3D torque；这些是接触对诊断，不是传感器校准结果。
* `eef_wrench_state()` 读取 `robot.recent_ee_forcetorques[arm].current`，并拆成 `force_n=values[:3]`、`torque_nm=values[3:6]`，明确标注 `uncalibrated_robosuite_sensor`。
* 原探测只改变 `surface_penetration_m` 并比较 marker 结果和 EEF 最大力。表面压入深度是控制/几何参数，不是已知牛顿载荷；不同轨迹的接触峰值也不构成载荷响应曲线。
* 探测器调用 Wipe 的 approach/wipe 方法和局部 IK。新脚本不调用这些技能或 IK；它使用固定关节目标的正常 RoboSuite hold-controller step，以保持机器人稳定并刷新与 Wipe 探测相同的 recent 通道。

### `gate1_wipe_calibration_audit.py`

旧审计会检查三例是否使用同一 model、压入量是否为 `0/0.002/0.01 m`、任务结果是否为欠作用/标称/过作用，以及 `max_eef_force_n` 是否单调。它保留 `wrench_calibration_verified=false` 和 `promotion_allowed=false`，这是正确的边界；但它没有：

* 已知 N/N·m 载荷、世界坐标方向和施加点；
* 原始 `sim.data.sensordata` 与 filtered `recent_ee_forcetorques` 的分开记录；
* site 旋转变换、每个窗口的真实采样时间、reset 标识和 baseline；
* 无接触前提、传感器有效位、窗口稳态和 force/torque/cross-coupling 误差门槛。

因此旧审计的“penetration response characterized”只能表示开发探测的几何/任务响应，不能晋升为传感器校准。

### RoboSuite/MuJoCo Wipe 绑定

当前服务器探测确认 Wipe 的受测通道为：

| 项目 | 绑定 |
|---|---|
| body | `gripper0_right_wiping_gripper`，当前模型 ID `114` |
| FT site | `gripper0_right_ft_frame` |
| raw force sensor | `gripper0_right_force_ee` |
| raw torque sensor | 对应 FT site 的 torque sensor |
| filtered channel | `robot.recent_ee_forcetorques["right"].current` |

新脚本在运行时重新按名称解析 ID，检查 sensor dimension=3、force/torque sensor 都挂在声明 site，并列出 body 的全部 geom ID/name。模型 body ID 仍被默认固定为 114；若模型变化，脚本停止并写执行错误，避免把错误 body 当作 Wipe 工具。

MuJoCo 的 `xfrc_applied[body_id, :3]` 是世界系 force（N），`[3:]` 是世界系 torque（N·m），施加点为目标 body COM。`cfrc_ext` 是另一路 body aggregate spatial wrench，布局为 `[torque (N·m), force (N)]`；`mj_contactForce` 是 contact-local `[force (N), torque (N·m)]`。三种布局不能混算。新脚本只把 `cfrc_ext` 作为附带诊断，校准主通道是两条独立 FT 传感器输出。MuJoCo 文档将 `cfrc_ext` 定义为 body 的 com-based external force，并规定 spatial vector 的 rotation/translation 顺序；`xfrc_applied` 参与净外力输入。[MuJoCo API types](https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html)、[MuJoCo simulation spatial-vector convention](https://mujoco.readthedocs.io/en/3.2.2/programming/simulation.html)

## 2. 新增受控载荷探测器

新增 `scripts/gate1_wipe_controlled_load_calibration.py`。运行命令示例：

```text
python scripts/gate1_wipe_controlled_load_calibration.py \
  --output /data/gaochuanhao/mannercap_runs/gate1_wipe_controlled_load_20260917_attempt02
```

默认协议为开发探测参数：

* baseline 和 known-load 各运行 150 个控制 transition；各窗口记录最后 20 个样本；两侧使用相同时间长度以消除 controller settling 偏差；
* force：世界系 `+X/+Y/+Z` 三个方向，`2/5/10 N` 多级载荷；
* torque：世界系 `+Z` 方向，`0.05/0.1 N·m` 多级载荷；方向和级别都可由 CLI 重复参数显式覆盖；
* 每个载荷 case 都先从相同完整 simulator/controller/RNG snapshot 运行无载 baseline，再恢复同一 snapshot 运行有载窗口；最后再次恢复并检查 `qpos/qvel/act/ctrl/mocap/time` 和 `xfrc_applied`；
* 不调用 Wipe skill 或 IK。每个 transition 通过 wrapper 的 `_step_once()` 发送保存的关节保持目标，使 RoboSuite 正常刷新 `recent_ee_forcetorques`；唯一主动改变的物理输入是写入 `sim.data.xfrc_applied` 的已知 wrench；
* 每个样本记录 `sim.data.time`、reset ID、`sensor_valid`、site 旋转矩阵、两条通道的独立 force/torque 三向量、selected-body contact 数和接触对身份；
* 记录理论响应 `-R_site.T @ applied_world`，分别计算 raw 和 recent 通道的 baseline、loaded 均值、delta、force/torque 误差以及 force-only torque / torque-only force cross-coupling。

默认误差门槛为开发期数值检查：force `0.01 N`、torque `1e-4 N·m`，cross-coupling 分别 `0.01 N` / `1e-4 N·m`。这些值不是安全阈值，也不是从真实传感器校准得到的阈值；CLI 和结果 JSON 会完整记录它们。

## 3. 纯 Python 审计合同

新增 `synthesis_v2/wipe_controlled_load.py`，只使用 Python 标准库。它对运行结果 JSON 做以下硬检查：

1. body/site/sensor/geom 绑定必须完整且与 Wipe 期望名称/ID一致；
2. force/torque 单位、施加 frame、sensor frame 和变换字符串必须显式；六维 mixed wrench 被拒绝；
3. baseline 和 loaded 窗口都必须有足够样本，时间严格递增、gap 不超过 `max_gap_s`，reset ID 不变；
4. raw `sensordata` 和 recent filtered channel 都必须存在并分别通过误差检查，不能用一个通道掩盖另一个；
5. 每个样本的 `sensor_valid` 必须为 `true`；selected body 发生接触、状态未恢复或缺少 site rotation 时返回 `unknown`；
6. 多级 force/torque case 必须覆盖声明的方向和级别，按级别检查响应单调性；缺失 case 不能通过；
7. case/suite 的 `physical_qualification` 恒为 `unknown`，`promotion_allowed` 恒为 `false`。

审计返回 `pass/fail/unknown`：结构、身份、时间、传感器或接触前提缺失为 `unknown`；已执行且误差超过门槛为 `fail`；所有证据存在且误差在门槛内才为数值合同 `pass`。该 `pass` 只代表 MuJoCo channel response check，不能被改写成物理校准完成。

## 4. 自动测试

新增 `tests/test_v2_wipe_controlled_load.py`，使用合成 JSON 和标准库 `unittest`，覆盖：

* 理论 `-R.T @ wrench` 变换；
* no-contact 与已知载荷；
* raw/recent 两通道分别审计；
* 缺通道、mixed wrench、错误 body/site/sensor/geom、错误单位/frame；
* 传感器失效、接触前提失败、时间缺口、reset gap、状态未恢复；
* direction/level 与实际载荷不一致；
* 多级覆盖、响应单调性和缺少 level；
* 输入不被审计函数修改。

当前工作区已执行：

```text
python -m py_compile \
  synthesis_v2/wipe_controlled_load.py \
  scripts/gate1_wipe_controlled_load_calibration.py \
  tests/test_v2_wipe_controlled_load.py
通过

python -m unittest -q tests/test_v2_wipe_controlled_load.py
Ran 14 tests ... OK

python scripts/gate1_wipe_controlled_load_calibration.py --help
通过，CLI 可解析
```

4090 的真实运行、首次失败、重复运行、误差和哈希见 `wipe_controlled_load_results.md` 与 `wipe_controlled_load_20260917.sha256`。该证据完成了 MuJoCo raw/recent 传感器通道响应检查；接触几何、摩擦、任务安全阈值和独立 B 角色签收仍未完成。
