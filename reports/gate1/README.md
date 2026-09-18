# Gate 1 开发交付

日期：2026-09-16  
分支：`work/synthesis-v2-gate1-20260916`

本阶段实现并验证三个边界：

1. MuJoCo、封装器、随机数及控制器缓存的初始状态快照与动作重放；
2. 对象接触力与力矩分列、身份绑定及 unknown 语义；
3. 全候选尝试、共同预算和资源用量的追加式台账。

4090 开发运行位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_20260916_attempt02
```

`evidence_manifest.sha256` 固定该目录下 12 个原始证据文件。原始轨迹较大，
不进入 Git；代码、合同、测试、结论和哈希清单进入 Git。

当前放行边界：Lift/Stack 功能重放和当前环境下 `1e-8` 严格逐状态重放均通过；对象力已完成
结构合同、接触对记录和 body wrench 受控通道校准，但接触对阈值仍未校准；attempt ledger 只完成合同，尚未接入
真实生成器/API/执行器。因此 Gate 1 仍是开发阶段，不构成正式实验封版。

后续力证据位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_20260916_attempt03
```

详见 `force_calibration_results.md` 和 `evidence_manifest_attempt03.sha256`。

Wipe 第三任务探测位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_wipe_probe_20260916_attempt08
```

该运行完成 100/100 marker，并通过 RoboSuite 自带成功判定；Wipe 仍保持
`candidate / pending_calibration`，因为其接触力和约束阈值尚未校准。详见
`wipe_probe_results.md` 和 `wipe_probe_attempt08.sha256`。

Wipe 的欠作用/标称/过作用三例位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_wipe_calibration_20260916
```

三例证明任务结果和内部 EEF 力通道会随表面压入变化，但三例本身不构成已知
载荷校准。后续已完成 raw/recent F/T 通道的无接触已知外载响应检查；接触模型
和安全阈值仍未签收，因此不提升 Wipe 的支持状态。详见
`wipe_calibration_results.md`、`wipe_controlled_load_results.md` 和相应哈希清单。

本阶段还增加了：

- `generation_validator_spec.md`：task/manner/safety 三态和 12 类生成期反例；
- `constraint_registry_spec.md`：公开生成配置与 sealed final audit 的隔离合同；
- `dev_manifest_spec.md`：由真实 Lift/Stack 初态生成的 draft 请求清单；
- `synthesis_method_spec.md`：G0/G1/G2/G3/M 五方法的统一配置入口；
- `action_interface_contract.md`：教师 8D、LIBERO 底层控制 8D 与现有 LeRobot
  导出候选 7D 的独立语义；
- `action_round_trip_preflight.md`：真实动作记录的 30 条无损导出/导入预检入口；
- `la4vla_osc_roundtrip_results.md`：LA4VLA 7D OSC adapter 与 30 步真实控制回放；
- `../gate2/la4vla_microset_learning_preflight.md`：64 窗口数据加载、训练与固定微批过拟合检查；
- `wipe_controlled_load_results.md`：Wipe raw/recent F/T 通道的已知外载响应；
- `gate1_exit_review.md`：当前退出判定和继续条件。

退出评审决定为**暂不放行扩量**。Wipe 传感器通道的已知外载响应已经验证；
主要未完成项是接触/安全阈值、sealed audit、OSC 跟踪误差的独立阈值，以及
LA4VLA checkpoint 在 MannerCAP 目标任务上的闭环评测。adapter 与 30 步控制
round-trip 已完成，但尚未据此冻结学生候选。

新增严格回放证据位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_strict_replay_mjb_20260917_attempt02
```

XML 重编译导致的模型差异已通过保存 MJB 解决；阶段间 `kp` 等 controller
side-channel 已逐 transition 记录。Lift 762/762、Stack 1404/1404 均通过
`1e-8`。详见 `replay_report.md` 和 `strict_replay_mjb_20260917.sha256`。

Wipe 的 3 seed × 0/2/10 mm 开发网格位于：

```text
/data/gaochuanhao/mannercap_runs/gate1_wipe_contact_grid_20260917
```

2 mm 为 3/3 成功，10 mm 为 0/3 且均发生 controller target 错误；force 响应不
稳定单调，因此没有把压入量提升为 force 或安全阈值。详见
`wipe_contact_grid_results.md` 和 `wipe_contact_grid_20260917.sha256`。
