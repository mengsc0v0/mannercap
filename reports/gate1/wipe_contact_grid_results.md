# Gate 1 Wipe 多 seed × 压入量接触响应结果

日期：2026-09-17  
运行目录：`/data/gaochuanhao/mannercap_runs/gate1_wipe_contact_grid_20260917`  
矩阵 SHA-256：`4d690e605ff2a831e0b229e6548177071f18d00ebc50aa7901074201d36bf6dd`

## 结果

9/9 格均真实执行并留下 transition、attempt、process 和模型证据；runner 层没有
丢失或覆盖。任务成功 5 格、失败 4 格：

| seed | 压入量 | 任务 | markers | 最大 contact-local force (N) | 最大 recent EEF force (N) | 错误 |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 42 | 0 mm | fail | 97/100 | 1.767586 | 62.447280 | 无执行错误；三次 wipe 后仍未完成 |
| 42 | 2 mm | pass | 100/100 | 5.840500 | 148.969596 | 无 |
| 42 | 10 mm | fail | 24/100 | 4.663797 | 167.988064 | tool-center target 未到达 |
| 43 | 0 mm | pass | 100/100 | 1.609134 | 67.721162 | 无 |
| 43 | 2 mm | pass | 100/100 | 1.174528 | 51.041674 | 无 |
| 43 | 10 mm | fail | 13/100 | 3.827634 | 143.406933 | tool-center target 未到达 |
| 44 | 0 mm | pass | 100/100 | 0.880041 | 35.328908 | 无 |
| 44 | 2 mm | pass | 100/100 | 0.775016 | 32.652540 | 无 |
| 44 | 10 mm | fail | 15/100 | 3.346986 | 130.430082 | tool-center target 未到达 |

2 mm 在三个 seed 都完成任务；10 mm 在三个 seed 都在 approach 阶段出现
`RuntimeError: Controller did not reach the Wipe tool-center target`，未进入 wipe
调用。0 mm 为 2/3 成功，seed 42 在三次 wipe 后停在 97/100，说明仅用任务成功
选择单一阈值会隐藏 seed 敏感性。

观测 force 对几何压入量并不稳定单调：seed 43/44 的 2 mm contact-local 最大值
反而低于各自 0 mm，而 seed 42 显著更高。recent EEF 最大力跨 seed 和压入量变化
也很大。因此 `surface_penetration_m` 不能当作已知施力或通用 force proxy。

## 判定

该网格确认了一个开发稳定区：2 mm 在本次 3 个 seed 的任务完成率为 3/3；它也
确认 10 mm 是当前 controller/几何组合的不稳定区。样本量只有 3 个 seed，且两种
force 通道仍各有其限定语义，所以不从这些结果反推出安全上下界。

保持以下状态不变：

- `physical_qualification=unknown`；
- `contact_model_calibration_verified=false`；
- `thresholds_frozen=false`；
- `promotion_allowed=false`。

后续需把候选范围缩到 0--4 mm 做更密的 seed 网格，并由独立 B 角色结合接触
几何、摩擦、持续时间和任务结果冻结开发/安全阈值。正式阈值不能只看本次成功格。

证据清单 `wipe_contact_grid_20260917.sha256` 共 48 个原始文件；清单自身 SHA-256
为 `8dafe7643cf706ef01c8c8743a4229308ea6aa96ebb670838074bbca22a808d9`。
本运行外部 API/LLM 调用为 0。
