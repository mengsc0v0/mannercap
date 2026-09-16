# 启动任务台账

实施依据：`docs/plans/MannerCAP_synthesis_v2_20260915.md`。
技术执行：主 agent 负责 A/集成；三个子 agent 均按用户要求配置为 `gpt-5.6-luna`、`max`，分别执行 B/C/D。项目范围与正式预算仍由项目负责人在关口冻结。

| 项 | 负责人 | 状态 | 证据 / 下一依赖 |
|---|---|---|---|
| WP0 仓库与环境 | 主 agent/A | 已启动、独立环境可用 | baseline_inventory、environment-lock |
| WP0 Lift/Stack | 主 agent/A | 开发后端成功，原后端受阻 | 两条真实轨迹和 task_recomputation；须冻结 IK 后端 |
| WP1 可信记录 | 主 agent/A | 控制频率原型 | 原始状态转移；下一步子步接触力、传感有效性、完整恢复 |
| WP2 严格判定 | B 子 agent | 内核/反例初版 | evaluation_audit；下一步接入真实轨迹、独立阈值 registry |
| WP3 请求空间 | C 子 agent | draft 编排初版 | synthesis_method_spec；下一步真实初态、物理配置、Wipe 可行性 |
| WP4 五方法及台账 | C + A | 未实现完整流水线 | 依赖 WP1—3；失败和成本覆盖所有内部尝试 |
| WP6 数据接口 | D 子 agent | 审计/契约初版 | action_interface_audit；下一步统一学生动作表达与图像记录 |
| WP6 30 条回放 | A + D | 未开始 | 完整 snapshot、控制器状态、RGB 和导出/导入 |
| WP6 32—64 样本学习 | D | 未开始 | 回放通过、固定现有学生权重与训练协议 |
| Gate 1 校准 | A + B | 未放行 | 12 条校准异常样例、阶段事件、可信测力、第三任务 |
| Gate 2 E0/正式协议 | 项目负责人 | 未到关口 | 回放、小样本学习、资源预算、封存集与人工金标 |

运行限制：当前只做开发与校准；不将 unknown 计作成功，不启动正式扩量或长训练。
