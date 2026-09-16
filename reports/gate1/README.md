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

当前放行边界：Lift/Stack 功能重放通过，严格逐状态重放未通过；对象力已完成
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

三例证明任务结果和内部 EEF 力通道会随表面压入变化，但尚未构成独立载荷
校准，因此不提升 Wipe 的支持状态。详见 `wipe_calibration_results.md` 和
`wipe_calibration_20260916.sha256`。
