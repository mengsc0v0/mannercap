# M 开发端到端运行

模型：服务返回 `qwen3.7-flash-2026-07-15`。本机生成，服务器缓存执行。
本次新增 8 次生成调用，全部解析成功；输入/输出合计 1942 tokens。
保留的较早批次为 8 次、1935 tokens；累计 16 次、3877 tokens。
两批都是开发数据，第二批显式加入 JSON response_format，不能伪称同一调用或隐藏重采样。
未取得服务商金额账单，不报告费用。

当前 8 个缓存候选完成仿真：8/8 教师任务终态成功、0 执行异常，accepted=0。
报告中的 api_calls=0 仅指缓存仿真阶段，外部生成调用必须与之分别计入总账。
每任务内缓存候选初态一致；与旧 G0--G3 批次相比，Lift 初态一致，Stack 不一致。
因此这不是可发布的严格五方法比较，后续必须把五方法放入同一未推进父环境再执行。

提案存在可见指令偏差，例如 neutral Lift 被指定全局 slow、组合 Lift 的 descend
未被指定 slow。合法 JSON 不等于正确方式。阶段 neutral 当前编码为空约束，是否继承
全局参数仍需纳入语义合同；该问题解决前不进行正式扩量。

运行目录：

- `/data/gaochuanhao/mannercap_runs/gate2_m_api_qwen_20260920`：较早 API 批次。
- `/data/gaochuanhao/mannercap_runs/m_api_smoke_20260920`：本次 API 原始响应、用量和候选。
- `/data/gaochuanhao/mannercap_runs/m_cached_execution_20260920`：8 条仿真轨迹及 M 台账。

全部仍是 development。正式流程尚缺共同父环境五方法重跑、方式语义审计、独立物理
阈值、人工双标与仲裁、封存集、学生独立验证和闭环评测；不能宣称论文全部流程完成。
人工标注和独立签收不能由执行器自行生成标签替代。
