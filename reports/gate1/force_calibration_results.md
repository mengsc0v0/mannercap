# Gate 1 MuJoCo 力通道校准与任务轨迹诊断

日期：2026-09-16  
服务器：4090x8  
证据目录：`/data/gaochuanhao/mannercap_runs/gate1_20260916_attempt03`

## 1. 语义修正

MuJoCo `cfrc_ext` 是 body 级空间向量，布局为
`[rotation/torque(3), translation/force(3)]`，并位于以子树质心为原点、
轴与全局坐标对齐的 c-frame。此前 Gate 1 初版合同错误地按 force、torque
顺序拆分；本轮已修正代码、测试与规范。

`mj_contactForce` 的输出采用另一种布局：`[force(3), torque(3)]`，位于接触
局部坐标系。两个接口现在由不同函数拆分，并以自动测试锁定。

## 2. 十六个受控样例

`scripts/gate1_force_calibration.py` 每个样例都从同一 Lift 完整快照恢复，随后
构造 table support、悬空无接触以及已知 `xfrc_applied` 输入。悬空样例把物体
提高 0.30 m，确认对象接触数为零，再测 body wrench 通道响应。

| 样例组 | 数量 | 结果 |
| --- | ---: | --- |
| 桌面支撑、零外加载 | 1 | 4 个对象相关接触；竖直 body force 为 0.668620 N |
| 悬空、零加载 | 1 | 0 接触；force=0、torque=0 |
| 悬空 X 向力 1/2/4 N | 3 | force 响应误差均为 0 N |
| 悬空 Z 向力 -1/-2/-4 N | 3 | force 响应误差均为 0 N |
| 悬空 Z 向力矩 0.05/0.1/0.2 N·m | 3 | torque 响应误差均为 0 N·m |
| 悬空组合 F=(1,2,3) N、T=(0.1,0.2,0.3) N·m | 1 | force 误差 0；torque 误差 4.84e-16 N·m |
| 桌面法向加载 0/1/2/4 N | 4 | 4 个接触点；响应增益 0.987013，最大相对误差 1.299% |

前 12 个 body 通道样例的最大 force 响应误差为 0 N，最大 torque 响应误差为
`8.883e-16 N·m`。因此可放行的是 **MuJoCo body wrench 通道的布局、单位和
受控输入响应**。

4 个桌面接触法向样例呈严格线性响应，但每 1 N 已知向下载荷只增加
`0.987013 N` 的接触法向合力，最大绝对误差 `0.051948 N`。它超过本轮预置的
`1e-8 N` 严格容差，因此只标记为“已表征”，没有标记为接触对校准通过。

## 3. Lift/Stack 观测轨迹

新记录器在实际物理 step 后显式调用 `mj_rnePostConstraint`，逐 transition
保存目标对象 body aggregate、对象/body/geom 身份和所有涉及目标对象的接触
对；接触对通过 `mj_contactForce` 分列 force 与 torque。

| 指标 | Lift primary | Stack primary | Stack secondary |
| --- | ---: | ---: | ---: |
| transition 数 | 779 | 1,441 | 1,441 |
| body force 范围 N | 0.6714–0.6962 | 0–1.0035 | 1.1922–1.2456 |
| body torque 峰值 N·m | 0.000406 | 0.000531 | 5.04e-7 |
| 单接触对 force 峰值 N | 10.405 | 7.948 | 0.563 |

所有 transition 都至少有一个目标对象相关接触。识别到桌面—方块、左右夹指/
指垫—方块，以及 Stack 的 cubeA—cubeB 接触对。另抽取了 12 个等间隔代表样例，
保存于 `force_diagnostic.json`。

body aggregate 与单接触对的幅值明显不同，且 Lift body force 长时间接近物体
支撑/重量尺度，而夹指接触对峰值超过 10 N。这说明 `cfrc_ext` body 行不能命名
为“抓力”，也不能代替接触对级别测量。

## 4. 放行边界

- `body_channel_response_verified=true`：仅表示 MuJoCo 仿真内部 body wrench
  通道对已知施加载荷按声明单位响应。
- `contact_pair_calibration_verified=false`：桌面接触法向响应已表征，但未通过
  `1e-8 N` 严格容差；夹爪接触、摩擦和 condim 仍未校准。
- `real_sensor_calibration_verified=false`：没有真机或真实传感器证据。
- `physical_qualification=unknown`：本轮不能据此冻结正式方式/安全阈值。

下一步必须针对 `mj_contactForce` 建立已知法向载荷夹具，确认接触 frame、符号、
摩擦维度和多接触求和；随后才能为 Lift/Stack 冻结对象接触阈值。Wipe 或其他
第三任务仍需独立任务结果与接触校准，当前没有获得放行。

## 5. 可复算证据

- `controlled_wrench_calibration.json`：16 个受控样例及误差。
- `force_diagnostic.json`：Lift/Stack 全轨迹汇总和 12 个代表样例。
- `lift/stack/transitions.jsonl`：逐 transition 原始接触证据。
- `evidence_manifest_attempt03.sha256`：14 个文件；服务器 `sha256sum -c` 全部通过。
