# Gate 1 Wipe 欠作用/标称/过作用探测

日期：2026-09-16  
汇总目录：`/data/gaochuanhao/mannercap_runs/gate1_wipe_calibration_20260916`  
结论：任务结果三例已验证；Wipe 力传感仍未完成独立载荷校准。

## 固定条件

三例均使用 seed 42、相同模型 SHA-256 `41460239267280f1a5bdd1eecb8f4a8002907d34acd0cd43dfc31fc0a587bf58`、相同 marker 布局、同一局部 IK 和阶段路径。唯一主动改变的探测参数是擦拭角点平面相对 marker 中心的表面压入量。

| 案例 | 压入量 | marker | 官方成功 | 最终 reward | 最大 contact-pair force | 最大 RoboSuite EEF force |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 欠作用 | 0 mm | 97/100 | false | 0.02565 | 1.76759 N | 62.4473 N |
| 标称探测 | 2 mm | 100/100 | true | 0.95291 | 5.84050 N | 148.970 N |
| 过作用 | 10 mm | 24/100 | false | -0.21409 | 4.66380 N | 167.988 N |

10 mm 案例在接近阶段出现控制器未达到工具中心目标，保留为执行错误；不能将其重命名为纯物理不可行。三例的 RoboSuite EEF 最大力随压入量单调增加，说明该通道对这组开发探测有响应。contact-pair 最大值来自不同长度和不同终止原因的轨迹，不形成单调载荷曲线。

## 判定边界

`gate1_wipe_calibration_audit.py` 得到：

- `task_outcome_cases_verified=true`；
- `penetration_response_characterized=true`；
- `wrench_calibration_verified=false`；
- `promotion_allowed=false`。

EEF wrench 是 RoboSuite 内部传感器输出，尚未与独立受控载荷对照；2 mm 也是开发探测参数，不是已批准的安全阈值。因此 Wipe 继续保持 `candidate / pending_calibration`。正式支持仍需独立载荷校准、重复样本、阶段级安全阈值和 B 角色签收。

`wipe_calibration_20260916.sha256` 固定三组各四个原始文件以及 `summary.json`。原始轨迹只保留在 4090 运行目录，不进入 Git。
