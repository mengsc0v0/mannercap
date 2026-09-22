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
| WP4 五方法及台账 | C + A | 五方法共同初态开发执行及台账已核验 | `five_method_paired_v2_20260920` 完成 40/40（G0/G1/G2/G3/M 各 8），40/40 教师终态、0 执行错误；M 使用已缓存 8 条提案，本轮新增 API=0；初态/模型哈希与 15 项 explicit-neutral 参数检查通过，40/40 语义审计、240/240 阶段参数检查通过。请求约束匹配 13/40（G0 2/8、G1 2/8、G2 1/8、G3 8/8、M 0/8），差异记录已保留；方式/安全仍 unknown，正式筛选与发布未放行；见 `reports/gate2/five_method_checkpoint_20260921.md` 与 `five_method_audit_20260921_v2.json` |
| WP6 数据接口 | D 子 agent | LA4VLA 候选接口已实现并预检 | 7D OSC adapter、双相机、8D state、50-step action horizon 已接通；候选仍未批准 |
| WP6 30 条回放 | A + D | 已完成数量与真实控制执行 | Lift/Stack 各 15 条，30/30 执行、0 error、0 clipping；独立误差阈值未冻结 |
| WP6 32—64 样本学习 | D | 已完成通路与过拟合预检 | 64 窗口由 8 条独立教师 episode 产生；原生 loader/20-step 训练/100-step 固定微批通过；无泛化或闭环结论 |
| Gate 1 校准 | A + B | 未放行 | Wipe F/T 通道与 0--4 mm 加密网格已完成；1--4 mm 为开发候选区间，接触模型、数值安全阈值、独立签收和 sealed audit 尚缺 |
| Gate 2 D0 质量启动集 | A + B | 64 条开发轨迹已完成 | 2 任务 × 4 指令 × 8 初态；64/64 教师任务终态、16/16 精确配对，方式/安全仍为 unknown |
| Gate 2 人工金标 | B + 标注支持 | 64 条候选清单与盲化协议已生成 | 24 条双人独立标注；尚待分配标注员、实际标注与仲裁 |
| Gate 2 数据划分 | B + D | 当前 128 条 source/window 全部锁为 development | 72 个 lineage 分组零泄漏；正式请求的四类用途与配额仍待冻结 |
| Gate 2 E0/正式协议 | 项目负责人 | 预检已运行，协议草案已记录，正式协议未冻结 | 仍需资源预算、正式 split、封存集、人工金标、独立阈值和学生闭环协议 |
| Gate 1 calibration reference `ee626e9` | A + B | 仅纳入台账，未改动工件 | 记录该校准参考提交供主 agent 验收；校准候选底层文件保持原样，contact-pair calibration 仍未验证，未放行物理阈值 |
| Gate 2 学生 action-head bounded attempt02 | D 子 agent | 已完成但标记 protocol deviation | 原启动失败保留；attempt02 完成 100 steps、固定最终 checkpoint、初始/最终各一次 ODE，原始报告 status=pass 但 train 实际为 2-source `lift/stack` 包，不满足冻结 8-source seed42 QC；validation 8-source 绑定通过。完整 parquet SHA、manifest/source/task×manner 对齐审计与结果均已保存；不得按冻结 plan 通过、不得泛化/闭环/正式推广，attempt03 需新计划/新预算 |
| Gate 2 学生数据绑定守门 | D 子 agent | 修复与回归通过 | runner 从 config 解析 dataset root，fail-closed 比对 QC manifest/file SHA、source IDs/count、task×manner、initial-state/result hashes、window counts 与 metadata；8 个聚焦测试通过；不重训、不覆盖旧证据 |
| Gate 2 学生 action-head bounded attempt03 | D 子 agent | PASS（development-only） | 正确冻结 8-source seed42 train + 8-source seed43 development holdout；100 steps、7D adapter 导出动作单位的真实 ODE 初/终 MAE/MSE、固定 final checkpoint；wall 227.13s、GPU3 peak 4.34GB、new disk 1.812GB、API=0；同噪声 seed 两次验证。7D 指标是复用 train-only stats 后的 adapter 控制输入单位，不是米/弧度物理误差；单 seed 小开发观察，不作方式学习/泛化/闭环结论。attempt02+03 共 200 optimizer steps，已知训练 wall 548.8265+227.1302=775.9567s；旧 startup 耗时 unknown，capture/setup 未计入。旧 failure 与 attempt02 protocol deviation/cost 均保留；详见 `reports/gate2/la4vla_student_dev_attempt03_report_20260922.md` 与 result manifest |

运行限制：当前只做开发与校准；不将 unknown 计作成功，不启动正式扩量或长训练。
