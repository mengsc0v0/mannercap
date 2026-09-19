# 启动任务台账

实施依据：`docs/plans/MannerCAP_synthesis_v2_20260915.md`。
技术执行：主 agent 负责 A/集成；三个子 agent 均按用户要求配置为 `gpt-5.6-luna`、`max`，分别执行 B/C/D。项目范围与正式预算仍由项目负责人在关口冻结。

| 项 | 负责人 | 状态 | 证据 / 下一依赖 |
|---|---|---|---|
| WP0 仓库与环境 | 主 agent/A | 已完成开发环境锁定 | baseline_inventory、environment-lock；4090 项目与运行证据分目录保存 |
| WP0 Lift/Stack | 主 agent/A | 当前固定后端真实执行通过 | 两条教师轨迹功能重算通过；MJB 与 controller side-channel 严格回放通过 |
| WP1 可信记录 | 主 agent/A | 开发记录与恢复通路通过 | 控制步、物理子步、真实时间、状态和 controller cache 已记录；正式物理阈值待校准 |
| WP2 严格判定 | B 子 agent | 内核/反例初版 | evaluation_audit；下一步接入真实轨迹、独立阈值 registry |
| WP3 请求空间 | C 子 agent | draft 编排初版 | synthesis_method_spec；下一步真实初态、物理配置、Wipe 可行性 |
| WP4 五方法及台账 | C + A | 未实现完整流水线 | 依赖 WP1—3；失败和成本覆盖所有内部尝试 |
| WP6 数据接口 | D 子 agent | LA4VLA 候选接口已实现并预检 | 7D OSC adapter、双相机、8D state、50-step action horizon 已接通；候选仍未批准 |
| WP6 30 条回放 | A + D | 已完成数量与真实控制执行 | Lift/Stack 各 15 条，30/30 执行、0 error、0 clipping；独立误差阈值未冻结 |
| WP6 32—64 样本学习 | D | 已完成通路与过拟合预检 | 64 窗口由 2 条独立教师 episode 产生；原生 loader/20-step 训练/100-step 固定微批通过；无泛化或闭环结论 |
| Gate 1 校准 | A + B | 未放行 | Wipe F/T 通道与 0--4 mm 加密网格已完成；1--4 mm 为开发候选区间，接触模型、数值安全阈值、独立签收和 sealed audit 尚缺 |
| Gate 2 D0 质量启动集 | A + B | 64 条开发轨迹已完成 | 2 任务 × 4 指令 × 8 初态；64/64 教师任务终态、16/16 精确配对，方式/安全仍为 unknown |
| Gate 2 E0/正式协议 | 项目负责人 | 预检已运行，协议草案已记录，正式协议未冻结 | 仍需资源预算、正式 split、封存集、人工金标、独立阈值和学生闭环协议 |

运行限制：当前只做开发与校准；不将 unknown 计作成功，不启动正式扩量或长训练。
