# M 缓存提案语义偏差分析

来源：`five_method_audit_20260921_v2.json`，计划哈希 `b53161c4945f3da6ba7254f769d90c7446828eccfa6419166513dabd9cbd9d5b`。本分析只读取已经缓存的 8 条 M 提案和已有执行语义审计，不重新调用模型或仿真，也不修改请求、提案或原审计文件。

8/8 候选的执行参数审计有效；五方法全体 40 条记录的阶段参数检查为 240/240，其中 M 子集为 8 条×6 个阶段项，即 48/48。通过只说明候选已按记录执行。请求→候选的整体分类为 8 条 `substantive_conflict`；逐字段另有 13 条 `representation_difference` 和 4 条 `unknown`。这些分类描述约束表示和速度计划，不是物理方式失败判定；8 条的 `manner_qualification` 仍为 `unknown`，`promotion_allowed=false`。

| M 候选 | 实质冲突 | 表示差异 | unknown |
|---|---|---|---|
| Lift neutral | 全局加入 `max_joint_vel=0.08`；descend 加入慢速覆盖 | approach、lift 显式 neutral 与原省略等效 | — |
| Lift slow | approach、descend 的显式 neutral 覆盖全局 slow | lift 的 slow 覆盖与全局 slow 等效 | `max_jerk` 不在 v1 速度契约 |
| Lift fast | approach 显式 neutral；descend 加入 slow 覆盖 | lift 的 fast 覆盖与全局 fast 等效 | — |
| Lift fast approach / slow contact | descend 改为 neutral；transport、final 请求阶段缺失 | — | `max_jerk` 不在 v1 速度契约 |
| Stack neutral | descend、transport、final 加入速度约束 | approach、lift、retreat 显式 neutral 与原省略等效 | — |
| Stack slow | descend、transport、final 的逐阶段参数改变全局 slow 的有效轮廓 | approach、lift、retreat 的重复 slow 覆盖等效 | `max_jerk` 不在 v1 速度契约 |
| Stack fast | descend、transport、final 改变全局 fast 的有效轮廓 | approach、lift、retreat 的重复 fast 覆盖等效 | — |
| Stack fast approach / slow contact | 全局加入 fast；descend、final、transport 阶段约束改变 | approach、lift 与请求一致 | `max_jerk` 不在 v1 速度契约 |

“表示差异”按有效阶段速度轮廓比较：例如全局 slow 下重复声明 lift slow 不改变该阶段参数，因此不扩大为实质冲突；全局 slow 下把 descend 显式置 neutral 会改变实际阶段参数，所以列为实质冲突。`max_jerk` 缺失只证明当前 v1 契约无法观察该请求维度，不能从中推断仿真中的物理失败。

本轮新增的 v2 提示/解析契约要求模型：未提及阶段必须省略并继承全局，明确 neutral 阶段才输出 `neutral`，并声明 `max_jerk`、stiffness、grip/contact force 等速度契约之外的请求约束。相关反例覆盖重复 neutral、阶段缺失、未知约束声明和非法阶段；历史 v1 请求与提案保持原样。

机器可读结果：`m_semantic_analysis_20260921.json`，报告哈希 `6f89eb0e13f6d5f87a10e417fec166d16dfbcd7df05adffc29ba335135438ffa`。
