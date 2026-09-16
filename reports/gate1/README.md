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

当前放行边界：Lift/Stack 功能重放通过，严格逐状态重放未通过；对象力只完成
结构合同，尚未完成接触对接入和物理校准；attempt ledger 只完成合同，尚未接入
真实生成器/API/执行器。因此 Gate 1 仍是开发阶段，不构成正式实验封版。

