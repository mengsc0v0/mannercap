# Gate 1 退出评审

日期：2026-09-16  
分支：`work/synthesis-v2-gate1-20260916`  
决定：**暂不放行扩量；保留为开发里程碑。**

## 判定

当前实现已经消除恒定力占位、约束裁剪后误通过、动作接口按 shape 混用等主要
风险，并形成可复现的 Lift、Stack、Wipe 开发证据。但 Gate 1 要求的“范围和
校准可信”尚未满足：Wipe 力通道没有独立受控载荷校准，数值阈值和 sealed
final audit 尚未冻结，教师到实际 7 维 LeRobot 导出的动作语义也没有闭合。

因此可以继续合同、执行器和小样本开发，不能启动正式扩量、发布质量统计或长
训练。任何 task/manner/safety 缺证据项继续记为 `unknown`，不能计作通过。

## 验收矩阵

| Gate 1 项目 | 当前证据 | 状态 | 放行含义 |
| --- | --- | --- | --- |
| Lift/Stack 真实执行 | 两任务原运行与功能终态重放成功 | 通过（功能级） | 可用于开发调试 |
| 完整初态恢复 | `time/qpos/qvel/ctrl` 初始最大误差为 0，controller/interpolator cache 已保存 | 通过（初态） | 可复用相同开发初态 |
| 严格逐状态重放 | 首步已出现大于 `1e-8` 的误差，长轨迹误差放大 | **未通过** | 不得宣称 bitwise/严格动力学重放 |
| 时间轴与控制记录 | 控制步、物理子步和实测 `dt` 分列；验证器拒绝重复时间、缺口和跨 reset | 通过（合同与已运行轨迹） | 可继续接入真实候选 |
| 力/力矩结构 | force 与 torque 分列，对象身份绑定，失效为 `unknown` | 通过（结构） | 不得混合 N 与 N·m |
| MuJoCo body wrench | 无接触和受控 body wrench 载荷已区分 | 通过（该通道） | 仅覆盖已校准 body wrench 条件 |
| Wipe 第三任务 | 同 seed/model 的标称探测完成 100/100 marker；欠/标称/过三例结果已记录 | 通过（任务探测） | Wipe 保持 candidate |
| Wipe 接触/EEF 力 | 三例有响应，但 EEF sensor 未与独立载荷对照；10 mm 例含控制器执行错误 | **未通过** | `pending_calibration`，不能冻结安全阈值 |
| 12 类严格反例 | 生成期验证器覆盖裁剪、无日志、缺阶段、失效、瞬时达标、时间错误等 12 类及附加案例 | 通过（合同测试） | 不等于真实轨迹最终审计 |
| task/manner/safety 三态 | `pass/fail/unknown` 与联合规则已实现，unknown 不可发布 | 通过（合同测试） | 正式结果仍需独立 B 角色审计 |
| 约束注册表 | 公开生成引用和 sealed audit 引用分离；pending 阈值为空且禁止晋升 | 通过（开发合同） | 数值阈值尚未冻结 |
| 请求清单 | 4090 上从真实 Lift/Stack 初态生成 14 条 draft 记录、6 个 pair、2 个 pending | 通过（开发清单） | 不是正式实验 manifest |
| 五方法入口 | G0/G1/G2/G3/M 的信息权限、采样范围和配置约束已登记 | 通过（配置合同） | 生成器执行与同预算比较尚未运行 |
| 动作接口 | 教师 8D、LIBERO 底层控制 8D、现有 LeRobot 导出候选 7D 分开登记 | 通过（静态合同） | 7D export 仍标记未验证 |
| 动作 round-trip | 尚未完成至少 30 条导出—导入—学生控制器回放 | **未完成** | 按方案在 Gate 2 前完成，完成前暂停扩量/长训练 |
| 最终审计隔离 | 生成验证器不读取 sealed 结果，注册表保留独立引用 | 通过（接口边界） | sealed 数据、签名和非作者复核尚未生成 |

## 已固定证据

- Lift/Stack：`evidence_manifest.sha256`、`evidence_manifest_attempt03.sha256`；
- Wipe 确定性成功探测：`wipe_probe_attempt08.sha256`；
- Wipe 欠/标称/过三例：`wipe_calibration_20260916.sha256`；
- 开发请求清单：`dev_manifest_20260916.sha256`；
- 原始大文件保留在 `/data/gaochuanhao/mannercap_runs/`，Git 只保存代码、报告和
  SHA-256 清单。

本阶段没有调用外部 LLM/API；`gemini-3.7-flash` 未参与上述生成、校准或评测。

## 继续条件

1. 用独立受控载荷校准 Wipe 的实际采用力通道，冻结单位、坐标、误差、有效率和
   阶段级安全阈值；
2. 由 B 角色冻结公开阈值引用和 sealed final-audit artifact，保留非作者复核与
   SHA-256；
3. 明确固定 VLA 实际消费的动作 head。若沿用当前 7 维 EEF 状态差分导出，先
   定义可执行变换和控制器语义；若改用 8 维关节增量，需要版本化数据 schema；
4. 完成至少 30 条导出—导入—学生控制器回放，按任务、阶段、对象、方式和安全
   分列结果；发现系统性偏差时停止扩量；
5. 负责人冻结正式 request manifest、对象配置、阈值、预算和 split 后，再复审
   Gate 1 并进入 E0/Gate 2。

## 验证记录

本地运行：

```text
python -m unittest discover -s tests -p "test*.py" -q
Ran 140 tests ... OK
```

4090 服务器运行：

```text
source /data/gaochuanhao/mannercap_env/bin/activate
python -m pytest -q
164 passed in 1.71s
```

该结果验证代码合同与合成反例；它不替代物理校准、30 条动作回放或 sealed
final audit。
