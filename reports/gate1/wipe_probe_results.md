# Gate 1 Wipe 第三任务探测结果

日期：2026-09-16  
运行目录：`/data/gaochuanhao/mannercap_runs/gate1_wipe_probe_20260916_attempt08`  
状态：任务可运行；仍为 `candidate / pending_calibration`。

## 结论

4090 上的 RoboSuite Wipe 环境能够完成重置、合法控制、表面接触和官方任务成功判定。在 seed 42 的最终探测中，擦拭工具通过 1163 个控制转换清除 100/100 个 marker，`_check_success()` 返回 `true`。整个探测未调用 LLM 或外部 API。

该结果只证明第三任务的仿真和控制路径可运行。Wipe 专属的力/接触通道尚未做受控校准，公开约束注册表也没有已批准阈值，因此不把它提升为正式支持任务。

## 接口发现与修复

- RoboSuite 动作维度为 7；封装器内部构造 9 维动作并用 `_ACTION_SLICE=-2` 传入前 7 维。
- 任务包含 100 个 marker；成功条件是 `len(wiped_markers) == num_markers`。
- 原重置提示误写为“堆叠方块”，已改为擦除所有 brown spill marker。
- 原封装器没有把传入的 seed 交给 RoboSuite Wipe；修正后，两次独立进程的模型摘要、完整初态、marker 布局和完整 transition 日志逐字节一致。
- `spill_wipe_privileged.py` 导入了未使用的 Open3D/视觉依赖，导致服务器即使已安装 RoboSuite也无法导入 Wipe Stage API；已删除这些未使用导入。
- 旧的通用 IK 控制末端刚体，而路径目标表示擦拭工具角点中心，约有 10.7 cm 的几何偏差，导致 4000 步均无接触。探测器改为直接最小化四个 wiping corner 的几何中心误差。
- RoboSuite 只在 marker 严格位于 wiping-corner 平面上方 0–2 cm 时计为擦除；最终探测使用 2 mm 表面压入量，参数和所有接触力均写入证据。

## 最终运行

| 项目 | 结果 |
| --- | ---: |
| marker 初始/最终 | 0/100 → 100/100 |
| 官方任务成功 | `true` |
| 控制转换 | 1163 |
| 本地 IK 求解 | 19 |
| approach 接触数 | 70，0 → 25 marker |
| 第一次擦拭 | 16 waypoints，25 → 100 |
| 表面压入量 | 0.002 m |
| 观测最大接触力 | 5.840500 N |
| LLM/API 调用 | 0 |

记录的接触对全部是 wiping corner/surface 与 `table_collision`。接触力来自 `mj_contactForce` 的 contact-local force 三元组范数，只作为观测诊断；它没有经过 Wipe 工具的受控载荷校准。

## 证据边界

- 原始 `initial_state.json`、`model.xml`、`transitions.jsonl` 和 `probe.json` 保留在服务器运行目录，不进入 Git。
- `wipe_probe_attempt08.sha256` 固定四个原始文件的 SHA-256。
- 探测器保持 `promotion_allowed=false`，并把未完成的 Wipe 力校准和约束审批列为阻断项。
- 相同命令的 attempt09 复跑得到完全相同的 `initial_state.json`、`model.xml` 和 `transitions.jsonl` SHA-256。该相等性覆盖探测器记录的动作、marker 进度、接触位置和接触力；日志没有记录完整 `qpos/qvel`，因此不把它扩张为完整仿真状态的严格重放结论。
