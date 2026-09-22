# Gate 1 初态恢复与严格动作重放报告

日期：2026-09-17
服务器：4090x8
最终运行目录：`/data/gaochuanhao/mannercap_runs/gate1_strict_replay_mjb_20260917_attempt02`
后端：`mujoco-local`（开发用；不等价于缺失的 PyRoKi 服务）

## 修复内容

旧回放从 `model.xml` 重新编译 MuJoCo 模型。实测 XML round-trip 会改变已编译
runtime 数组；例如 Lift 的 `body_mass` 最大差约 38.09、`body_inertia` 最大差约
6.21，`dof_armature` 最大差约 `3.33e-6`。初始 qpos/qvel 即使精确恢复，下一
物理步仍会因此漂移。

新记录同时保存：

- 人可读 `model.xml` 及 SHA-256；
- MuJoCo 编译模型 `model.mjb` 及 SHA-256，严格回放优先加载 MJB；
- 原有 MuJoCo data、wrapper、RNG、controller 和 interpolator 快照；
- 每个 transition 的 `controller_command_t`，显式记录 action vector 之外的
  `kp/kd/impedance/input_type` 等 side channel。

中间运行 `gate1_strict_replay_mjb_20260917` 使用 MJB 后，Lift 前 334 步、Stack
前 382 步严格匹配，随后在 `set_joint_stiffness` 后分歧。原因是原教师在阶段间
修改 `kp`，但旧 `action_t` 没有这个 side channel。补齐逐 transition controller
command 后，最终 attempt02 全程通过。没有放宽 `1e-8` 容差。

## 4090 最终结果

| 项目 | Lift | Stack |
| --- | ---: | ---: |
| 原运行任务成功 | 是 | 是 |
| 重放任务成功 | 是 | 是 |
| 控制 transition | 762 | 1,404 |
| 物理子步 | 19,050 | 35,100 |
| 编译模型格式 | MJB | MJB |
| MJB SHA-256 | `cd2b9f6f...84f5e30` | `d82a1dc6...39feb14` |
| XML 与 MJB 来源 hash 校验 | 通过 | 通过 |
| 初始 `time/qpos/qvel/ctrl` 最大误差 | 0 | 0 |
| 全程 `time` 最大误差 | 0 | 0 |
| 全程 `qpos` 最大误差 | `3.357e-14` | 0 |
| 全程 `qvel` 最大误差 | `3.261e-14` | 0 |
| 全程 `actuator_ctrl` 最大误差 | `3.695e-13` | 0 |
| matching prefix | 762/762 | 1,404/1,404 |
| 功能终态重放 | **通过** | **通过** |
| `1e-8` 严格逐状态重放 | **通过** | **通过** |

## 判定和边界

- 完整初态恢复、编译模型恢复、controller side-channel 回放及逐状态动作回放：
  通过，可解除原严格回放 blocker；
- 该结果覆盖本次固定 seed、`mujoco-local` Lift/Stack 开发轨迹，不证明跨版本、
  跨硬件 bitwise 可移植；
- `manner`、`safety` 和 JPR 仍为 unknown；严格动力学回放不等于物理阈值或学生
  policy 回放；
- 旧 XML-only 运行继续保留为失败诊断，不重写为通过。

证据清单为 `strict_replay_mjb_20260917.sha256`，共 15 个原始文件；清单自身
SHA-256 为 `71b55caeb74151754d85b5e384b7ff42c0b7d51770137552e3b10c7050bbdb63`。
本运行外部 API/LLM 调用为 0。
