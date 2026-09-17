# Wipe 多 seed × 压入量接触响应网格规格

日期：2026-09-17  
状态：开发校准工具；不冻结物理或安全阈值。

## 网格与入口

入口 `scripts/gate1_wipe_contact_grid.py` 默认建立完整的
`3 seed × 3 surface_penetration_m = 9` 格网格：

- seed：42、43、44；
- 几何/控制目标压入量：0、2、10 mm；
- 每格 `max_steps=4000`，最多调用 3 次 `wipe_spill_and_verify`；
- 默认只写 `plan.json`、`audit.json` 和 `summary.md`，只有显式 `--run`
  才调用真实 `gate1_wipe_probe.py`。

计划或审计：

```text
python scripts/gate1_wipe_contact_grid.py \
  --output /data/gaochuanhao/mannercap_runs/gate1_wipe_contact_grid_20260917
```

真实运行：

```text
python scripts/gate1_wipe_contact_grid.py \
  --output /data/gaochuanhao/mannercap_runs/gate1_wipe_contact_grid_20260917 \
  --run
```

## 记录与失败保留

每格使用独立目录保存 probe 原始输出和 `process.json`。runner 在每格启动前和
结束后重写顶层审计，因此中断后的已完成、运行中和缺失格均可见。已有非空格
不会被覆盖或静默重跑；需要重跑时必须使用新的顶层运行目录。

审计逐 transition 汇总：

- marker 总数、擦除数、任务成功和最终 reward；
- 命名 MuJoCo 接触对、接触局部坐标系下的最大力及观测次数；
- recent EEF force 和 torque，继续标记为未校准 RoboSuite 传感通道；
- 实际 control transition 数、wipe 调用数、attempt 失败和完整错误；
- 模型 SHA-256、请求预算、runner exit code/stdout/stderr。

probe 的顶层错误、每次 skill attempt 错误、缺失/损坏文件及 runner 错误分别
保留；失败不会因没有 `probe.json` 而从矩阵中消失。

## 结论边界

`surface_penetration_m` 是几何/控制目标偏移，不是已知施加的力或力矩。工具沿用
`[0,20 mm)` 仅为了匹配现有 marker 识别几何范围，该范围不是安全阈值。接触对
force 是 MuJoCo 接触局部值，不能直接晋升为独立标定的物理安全测量。

因此，无论任务成功率或观测力是否单调，输出固定保持：

- `physical_qualification=unknown`；
- `contact_model_calibration_verified=false`；
- `thresholds_frozen=false`；
- `promotion_allowed=false`。

该网格用于定位稳定区间、执行错误和 seed 敏感性，后续仍需 B 角色独立冻结
接触几何、摩擦、任务及安全阈值。
