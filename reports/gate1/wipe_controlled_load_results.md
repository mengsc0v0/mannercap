# Gate 1 Wipe 受控载荷通道结果

日期：2026-09-17  
服务器：4090x8  
模型 SHA-256：`41460239267280f1a5bdd1eecb8f4a8002907d34acd0cd43dfc31fc0a587bf58`

## 结果

通过 `sim.data.xfrc_applied` 对 `gripper0_right_wiping_gripper` 的 body COM
施加世界坐标已知 wrench，并以固定关节目标的正常 RoboSuite 控制 transition
保持机械臂稳定。raw `sensordata` 和 Wipe 实际使用的
`recent_ee_forcetorques` 分开审计，理论响应为 `-R_site.T @ applied_world`。

正式协议覆盖：

- 无载 1 例；
- 世界系 `+X/+Y/+Z` 的 `2/5/10 N`，共 9 例；
- 世界系 `+Z` 的 `0.05/0.1 N·m`，共 2 例；
- 每例 baseline 与 load 均从相同 simulator/controller/RNG snapshot 开始，
  各运行 150 个控制 transition，比较最后 20 个样本；
- 受测 body 全程无接触，force/torque 单位、site frame、实际时间和传感器有效位
  均被记录。

两次相同 seed 42 的正式运行得到字节一致的 JSON：

```text
/data/gaochuanhao/mannercap_runs/gate1_wipe_controlled_load_20260917_attempt02
/data/gaochuanhao/mannercap_runs/gate1_wipe_controlled_load_20260917_attempt03
SHA-256 e94417bd5ba302376ba38db6ae26fa105463e3c4a90b38eb747e8c0f46181f48
```

12/12 case 在 raw 和 recent 两个通道均通过：

| 指标 | 最大值 | 开发检查门槛 |
| --- | ---: | ---: |
| force response error | `0.000439997 N` | `0.01 N` |
| torque response error | `8.09496e-10 N·m` | `1e-4 N·m` |
| torque-only force cross-coupling | `4.71439e-6 N` | `0.01 N` |
| force-only torque cross-coupling | `5.61641e-10 N·m` | `1e-4 N·m` |

`audit_decision=pass` 和 `sensor_channel_response_verified=true` 只说明 MuJoCo
已知外载到两个 F/T 读数通道的数值响应满足合同。

## 首次失败保留

`gate1_wipe_controlled_load_20260917` 首次运行使用 baseline 150、load 60 个
transition。两窗口处于不同 controller settling 时刻，产生约
`0.00018–0.00022 N·m` 的系统偏差，超过 `1e-4 N·m` 门槛，审计正确返回
`fail`。没有放宽门槛；修复是让成对窗口使用相同的 150 transition。首次失败
JSON 的 SHA-256 为
`59058f26c9b54532e6798fffbdc92e8f1430deb56012cf89ba68d1cd5328c061`。

## 结论边界

以下仍为 `unknown` / 未签收：

- Wipe 接触几何、摩擦和接触对 force 的物理校准；
- 2 mm 擦拭压入量对应的安全或任务阈值；
- 真实硬件传感器校准；
- 独立 B 角色冻结的数值阈值和 sealed final audit。

因此 `physical_qualification=unknown`、`promotion_allowed=false` 保持不变，Wipe
继续为 `candidate/pending_calibration`。
