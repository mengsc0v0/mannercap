# Gate 1 退出评审

日期：2026-09-18
分支：`work/synthesis-v2-gate1-20260916`  
决定：**暂不放行扩量；保留为开发里程碑。**

## 判定

当前实现已经消除恒定力占位、约束裁剪后误通过、动作接口按 shape 混用等主要
风险，并形成可复现的 Lift、Stack、Wipe 开发证据。Lift/Stack 的 MJB 与
controller side-channel 严格回放已经通过；Wipe raw/recent F/T 通道的已知外载
响应和 3 seed × 3 压入量网格已经运行。但接触模型和数值安全阈值尚未校准，
sealed final audit 尚未冻结；教师到 LA4VLA OSC_POSE 的 adapter 与 30 步物理控制
回放已经完成，但独立跟踪误差阈值和学生闭环任务评测尚未冻结。

因此可以继续合同、执行器和小样本开发，不能启动正式扩量、发布质量统计或长
训练。任何 task/manner/safety 缺证据项继续记为 `unknown`，不能计作通过。

## 验收矩阵

| Gate 1 项目 | 当前证据 | 状态 | 放行含义 |
| --- | --- | --- | --- |
| Lift/Stack 真实执行 | 两任务原运行与功能终态重放成功 | 通过（功能级） | 可用于开发调试 |
| 完整初态恢复 | `time/qpos/qvel/ctrl` 初始最大误差为 0，controller/interpolator cache 已保存 | 通过（初态） | 可复用相同开发初态 |
| 严格逐状态重放 | MJB + 逐 transition controller side-channel 下，Lift 762/762、Stack 1404/1404 均低于 `1e-8` | **通过（当前环境）** | 解除当前固定版本/后端 blocker；不外推跨版本 bitwise 一致 |
| 时间轴与控制记录 | 控制步、物理子步和实测 `dt` 分列；验证器拒绝重复时间、缺口和跨 reset | 通过（合同与已运行轨迹） | 可继续接入真实候选 |
| 力/力矩结构 | force 与 torque 分列，对象身份绑定，失效为 `unknown` | 通过（结构） | 不得混合 N 与 N·m |
| MuJoCo body wrench | 无接触和受控 body wrench 载荷已区分 | 通过（该通道） | 仅覆盖已校准 body wrench 条件 |
| Wipe 第三任务 | 同 seed/model 的标称探测完成 100/100 marker；欠/标称/过三例结果已记录 | 通过（任务探测） | Wipe 保持 candidate |
| Wipe F/T 传感通道 | 12 个无接触已知 wrench case 在 raw/recent 通道均通过，两次运行字节一致 | 通过（仿真通道响应） | 不覆盖接触模型或真实硬件 |
| Wipe 接触/安全阈值 | 9 格均执行；2 mm 为 3/3 成功，10 mm 为 0/3 且均有 controller target 错误，0 mm 为 2/3；force 不稳定单调 | **未通过** | `pending_calibration`，不能冻结安全阈值 |
| 12 类严格反例 | 生成期验证器覆盖裁剪、无日志、缺阶段、失效、瞬时达标、时间错误等 12 类及附加案例 | 通过（合同测试） | 不等于真实轨迹最终审计 |
| task/manner/safety 三态 | `pass/fail/unknown` 与联合规则已实现，unknown 不可发布 | 通过（合同测试） | 正式结果仍需独立 B 角色审计 |
| 约束注册表 | 公开生成引用和 sealed audit 引用分离；pending 阈值为空且禁止晋升 | 通过（开发合同） | 数值阈值尚未冻结 |
| 请求清单 | 4090 上从真实 Lift/Stack 初态生成 14 条 draft 记录、6 个 pair、2 个 pending | 通过（开发清单） | 不是正式实验 manifest |
| 五方法入口 | G0/G1/G2/G3/M 的信息权限、采样范围和配置约束已登记 | 通过（配置合同） | 生成器执行与同预算比较尚未运行 |
| 动作接口 | LA4VLA 7D OSC adapter 已实现；旋转、缩放、裁剪、gripper 与时刻语义已固定 | 通过（实现与代数） | 正式转换入口仍等待独立阈值签收 |
| 动作 round-trip | Lift/Stack 各 15 条，30/30 真实 OSC 步执行，0 error、0 clipping；两个初始化边界约 10 mm / 0.047 rad | **完成（数量与执行）** | 结果保留，未设事后阈值，不据此放行训练 |
| 32--64 样本学习预检 | 64 个连续窗口由 LA4VLA 原生 loader 读取；20 步训练有限；固定四样本 100 步 loss 降至 25.0% | **通过（通路/过拟合）** | 仅 2 个独立 episode、无方式 prompt、无闭环，不是泛化证据 |
| 最终审计隔离 | 生成验证器不读取 sealed 结果，注册表保留独立引用 | 通过（接口边界） | sealed 数据、签名和非作者复核尚未生成 |

## 已固定证据

- Lift/Stack：`evidence_manifest.sha256`、`evidence_manifest_attempt03.sha256`；
- Wipe 确定性成功探测：`wipe_probe_attempt08.sha256`；
- Wipe 欠/标称/过三例：`wipe_calibration_20260916.sha256`；
- Wipe 已知外载通道：`wipe_controlled_load_20260917.sha256`；
- Wipe 3×3 接触响应网格：`wipe_contact_grid_20260917.sha256`；
- Lift/Stack MJB 严格回放：`strict_replay_mjb_20260917.sha256`；
- LA4VLA OSC 30 步回放：`la4vla_osc_roundtrip_20260918.sha256`；
- 开发请求清单：`dev_manifest_20260916.sha256`；
- 原始大文件保留在 `/data/gaochuanhao/mannercap_runs/`，Git 只保存代码、报告和
  SHA-256 清单。

本阶段没有调用外部 LLM/API；`gemini-3.7-flash` 未参与上述生成、校准或评测。

## 继续条件

1. 基于已验证的 Wipe F/T 通道继续校准接触几何、摩擦和阶段级安全阈值，并由
   独立 B 角色签收；
2. 由 B 角色冻结公开阈值引用和 sealed final-audit artifact，保留非作者复核与
   SHA-256；
3. 由独立评测角色冻结 OSC 跟踪误差阈值，签收已完成的 30 步结果，并处理两个
   初始化边界的大误差；
4. 接入 LA4VLA checkpoint，在 Lift/Stack 目标任务验证观测、相机、语言、动作
   horizon 和 reset/init-state 后，才运行学生闭环评测；
5. 负责人冻结正式 request manifest、对象配置、阈值、预算和 split 后，再复审
   Gate 1 并进入 E0/Gate 2。

## 验证记录

本地运行：

```text
python -m unittest discover -s tests -p "test*.py" -q
Ran 197 tests ... OK
```

4090 服务器运行：

```text
source /data/gaochuanhao/mannercap_env/bin/activate
python -m pytest -q
221 passed in 5.96s
```

该结果验证代码合同与合成反例；30 条动作回放已有独立原始证据，但它不替代
物理校准、独立误差阈值、学生闭环评测或 sealed final audit。
