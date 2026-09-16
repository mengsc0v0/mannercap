# Gate 1 初态恢复与动作重放报告

日期：2026-09-16  
服务器：4090x8  
运行目录：`/data/gaochuanhao/mannercap_runs/gate1_20260916_attempt02`  
后端：`mujoco-local`（开发用；不等价于缺失的 PyRoKi 服务）

## 实现

`synthesis_v2/simulator_state.py` 的 v2 快照保存：

- MuJoCo `time/qpos/qvel/act/qacc_warmstart/ctrl/qfrc_applied/xfrc_applied`；
- mocap、userdata、plugin state、equality activation；
- wrapper 和 RoboSuite 计数器、Python/NumPy/env RNG；
- robot、part controller 与线性 interpolator 的标量和数组缓存。

`scripts/gate1_replay.py` 从保存的 XML 建立新仿真，恢复快照，按原顺序重放
每个 `action_t`，并逐步比较 `time/qpos/qvel/actuator_ctrl`。严格容差为绝对误差
`1e-8`；任务终态另行比较，不能代替逐状态一致性。

## 4090 结果

| 项目 | Lift | Stack |
| --- | ---: | ---: |
| 原运行任务成功 | 是 | 是 |
| 重放任务成功 | 是 | 是 |
| 控制转移数 | 765 | 1,244 |
| 物理子步数 | 19,125 | 31,100 |
| 仿真时长 | 38.25 s | 62.20 s |
| 模型源文件 SHA-256 一致 | 是 | 是 |
| 初始 `time/qpos/qvel/ctrl` 最大误差 | 0 | 0 |
| 首个严格失配转移 | 1 | 1 |
| 首步 `qpos` 最大误差 | 1.775e-8 | 1.671e-8 |
| 首步 `qvel` 最大误差 | 4.117e-7 | 4.563e-7 |
| 首步 `ctrl` 最大误差 | 1.762e-6 | 1.995e-6 |
| 全程 `qpos` 最大误差 | 1.246e-2 | 1.278e-2 |
| 功能终态重放 | **通过** | **通过** |
| 严格逐状态重放 | **未通过** | **未通过** |

第一次诊断运行 `gate1_20260916_attempt01` 没有保存控制器/插值器缓存，首步
控制量误差约 `0.10`。加入这些缓存后，首步控制量误差降到约 `2e-6`，证明
该缺口已被定位并显著收敛；剩余微小数值差异在长时接触动力学中被放大。

## 判定

- 初始数值状态恢复：通过。
- 任务终态功能重放：通过。
- `1e-8` 全轨迹逐状态重放：未通过，保持 blocker 状态。
- manner/safety/JPR：unknown；本运行没有外部 LLM 调用，不能用于相应结论。

后续若要求严格逐状态放行，需要保存并加载编译后的 MuJoCo model（二进制 MJB
或等价无损模型状态），并验证 MuJoCo 派生缓存/求解器状态的可迁移边界。不得
通过放宽容差或只比较任务成功来改写本次严格判定。

## 证据与验证

- `evidence_manifest.sha256`：12 个原始文件；服务器执行 `sha256sum -c` 全部通过。
- 根测试集：4090 上 `python -m pytest -q`，39 项通过。
- 完整仓库的无范围 pytest 会收集 vendored/third-party 自带测试并因其独立
  import path 失败；根目录 `pytest.ini` 将项目默认测试范围固定为 `tests/`。

