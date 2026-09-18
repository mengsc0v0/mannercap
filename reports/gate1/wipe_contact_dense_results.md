# Gate 1 Wipe 0--4 mm 加密接触响应结果

日期：2026-09-18  
服务器：4090x8  
运行目录：`/data/gaochuanhao/mannercap_runs/gate1_wipe_contact_dense_20260918`  
矩阵 SHA-256：`d6aa2c733cac6c6004d65a91fde7b49ca79ffc0d649cc2a91fb358ecbcf2641b`

## 协议

在先前 0/2/10 mm 三点网格之后，将候选范围缩到 0--4 mm，运行完整的
5 seed × 5 压入量矩阵：

- seed：42、43、44、45、46；
- `surface_penetration_m`：0、0.001、0.002、0.003、0.004 m；
- 每格最多 4000 个 environment step、3 次 `wipe_spill_and_verify`；
- 每格独立创建环境，保留 plan、process、probe、initial state、attempt 和
  transition 记录；
- `surface_penetration_m` 仍是几何/控制目标偏移，不是已知外加载荷或安全阈值。

执行从提交 `9569ae97f3a340580e9df0180dc26dadcbeea834` 启动。按压入量聚合与最终
audit 由提交 `7e74a9cbdbe2ef88e43c6bf4f74fdcee635938dd` 生成；后一个提交只增加聚合
输出和测试，没有修改 `gate1_wipe_probe.py` 或探测行为。

## 结果

25/25 格完成，0 个执行错误、0 个 audit issue，共记录 36,414 个控制
transition，仿真 wall time 合计 744.12 s。任务完成 22 格，失败 3 格：

| 压入量 | task success | contact-local 每格最大力 min / median / max (N) | recent EEF 每格最大力 min / median / max (N) |
| ---: | ---: | ---: | ---: |
| 0 mm | 2/5 | 0.613 / 1.609 / 2.209 | 24.863 / 62.447 / 76.226 |
| 1 mm | 5/5 | 0.413 / 0.476 / 5.029 | 19.730 / 22.020 / 133.221 |
| 2 mm | 5/5 | 0.681 / 0.782 / 5.840 | 32.561 / 32.653 / 148.970 |
| 3 mm | 5/5 | 0.988 / 1.158 / 6.747 | 46.575 / 47.466 / 165.706 |
| 4 mm | 5/5 | 1.352 / 1.472 / 8.030 | 60.540 / 62.805 / 177.349 |

三个任务失败全部发生在 0 mm：seed 42、45、46 分别只擦除 97、90、84 个
marker；seed 43、44 在 0 mm 成功。1--4 mm 的 20 格全部达到 100/100 marker。

## 判定边界

本矩阵支持两个开发结论：

1. 在这 5 个初态中，1--4 mm 是稳定完成 Wipe 的候选几何区间；0 mm 存在明显
   seed 敏感性。
2. 1--4 mm 的 contact-local 中位数随压入量增加，但跨 seed 极差很大；例如
   1 mm 最大值 5.029 N，而同组中位数只有 0.476 N。recent EEF 通道也存在同样
   的大跨度。因此单个压入量不能直接换算成通用 force。

这仍不是安全阈值校准。contact-local 值是 MuJoCo 接触坐标系观测，recent EEF
值仅完成了已知外载通道响应验证；两者都未获得独立的接触模型/硬件安全语义。
样本也只有 5 个 seed。保持：

- `physical_qualification=unknown`；
- `contact_model_calibration_verified=false`；
- `thresholds_frozen=false`；
- `promotion_allowed=false`。

下一步应由独立 B 角色结合几何、摩擦、接触持续时间和任务结果，为开发生成范围
与 sealed safety audit 分别制定协议；不能只用本次 5/5 成功格反推安全上界。

证据目录为 567 MiB。`wipe_contact_dense_20260918.sha256` 覆盖 130 个文件并已
逐项校验，清单自身 SHA-256 为
`9a8e72c69346faeb0005763e37639f5199a27a6a9699048b09578702f670e07b`。
外部 LLM/API 调用数为 0，不需要 API key。
