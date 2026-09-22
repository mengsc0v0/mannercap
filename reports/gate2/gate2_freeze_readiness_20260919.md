# Gate 2 冻结准备状态（当前更新）

更新时间：2026-09-21（Asia/Shanghai）
本文件保留原 `gate2_freeze_readiness_20260919.md` 的历史结论，但以下状态以本次
更新引用的运行和审计工件为准。它是开发/校准状态报告，不是正式 Gate 2 签收。

## 当前决定

**五方法共同初态开发批次已完成离线核验；仍不冻结 E1--E4、D1/D2 或长训练。**

当前硬状态保持：

```text
formal_manner_status = unknown
formal_safety_status = unknown
accepted = 0
promotion_allowed = false
```

原因是独立物理阈值、阶段/安全金标准、人工金标、正式 split 和学生接口放行均未
完成。40/40 只表示教师执行到任务终态，不能改写为方式或安全通过。

代码与审计引用：

| 项 | 当前值 |
|---|---|
| 工作区 HEAD | `ee5507a` |
| 五方法执行提交 | `ab6a21f` |
| 最新审计代码 | `5ffdfd2` |
| 计划哈希 | `b53161c4945f3da6ba7254f769d90c7446828eccfa6419166513dabd9cbd9d5b` |
| 最新阶段语义审计 | `reports/gate2/five_method_audit_20260921_v2.json` |
| 阶段语义审计 SHA-256 | `cca28cc7056f8e1f59a44b3242ccf04bb6c21c1300408d61f6c69f4c5421b5e9` |

## 与 2026-09-19 版本的状态差异

2026-09-19 版本是在五方法运行前写的开发检查点。其“D0 合计 64 条”的补采
结论仍有效；“五方法未执行”“M 跳过”“下一步运行五方法 E0”已经由后续证据
取代，不能继续作为当前状态。

| 旧版本表述 | 当前状态 |
|---|---|
| D0 由 24 条起步，等待 seed45--49 补采 | 已完成 64 条 development episode、16 个精确同初态方式组；64/64 教师终态，方式/安全仍 unknown |
| G0--G3 槽位已验证、五方法尚未实际运行 | 已完成 40/40：8 请求 × G0/G1/G2/G3/M，各 1 次；0 执行错误 |
| M 尚未接入、当前不需要 key | 本轮使用已缓存的 8 条 M 提案，新增 API=0；两次开发批次累计 16 次 generation calls、3877 tokens |
| 下一步先运行共同请求矩阵 | 共同请求矩阵已完成并通过最新离线审计；下一步应转向独立物理校准、M 语义缺口处理和学生放行前置条件 |

旧版本中关于独立阈值、Wipe 物理语义、金标、正式 split、学生闭环仍未完成的
限制继续有效。

## 已确认的当前证据

### 五方法开发执行与语义完整性

- 矩阵为 `Lift/Stack × neutral/slow/fast/fast_approach_slow_contact × G0/G1/G2/G3/M`，共 40 个候选，每槽位 1 次尝试。
- 40/40 教师任务终态，0 执行错误；两组精确初态组分别为 Lift、Stack，候选共享未推进父环境和编译模型哈希。
- 最新阶段语义审计完成 240/240 个阶段参数检查，15 个显式空阶段检查通过；统计为 `explicit_neutral=15`、`explicit_override=40`、`omitted_inherits_global=185`。
- 运行资源合计为 1,355,175 physics substeps、466.8504 wall-clock seconds、3,476,660,486 bytes；不含环境设置、父进程编排和日志存储，不能当作总成本。
- 五方法 `manner`、`safety`、`overall` 均为 `unknown`。阶段参数检查是执行语义和记录完整性证据，不是物理合格证据。

### 请求语义与 M

原始 `requested_spec` 与候选 `executed_parameters` 均已保留，不能因执行成功而
重写原请求。精确请求约束匹配为 13/40：G0 2/8、G1 2/8、G2 1/8、G3 8/8、
M 0/8。M 的 8 条缓存提案已执行，但与原请求的语义一致性尚未签收；可见差异
包括额外全局/阶段约束、缺失请求约束和阶段约束改变。下一步应先离线修复/审计
请求到执行的合同，再决定是否需要新的有界模型调用；不以重复调用掩盖语义差异。
这些语义差异进入差异分类和资格待审队列，不直接等同于物理失败或物理通过；
`accepted=0` 和三态 `unknown` 的当前状态来自独立物理资格审计尚未完成。

### 独立物理校准状态

- Wipe 已有 0--4 mm、5 seed × 5 压入量的 25 格开发网格；1--4 mm 在该开发矩阵中任务完成 20/20，0 mm 为 2/5，仍只是候选几何稳定区。
- contact-local 与 recent EEF force 对压入量的响应跨 seed 变化大，且语义不同；`surface_penetration_m` 不能改名为通用外加载荷或安全阈值。
- 目前没有独立冻结的速度、jerk、接触力上下界、冲量、持续时间、阶段串扰或安全峰值阈值；`thresholds_frozen=false` 继续成立。
- 所有阈值都必须来自与方法输出隔离的校准输入，并保留单位、坐标系、传感器有效位、真实时间轴和测量分辨率。现有开发结果只能生成候选范围和误差描述，不能自校准。

### 学生验证前置状态

- LA4VLA OSC adapter 的现有证据是 **30/30 个真实物理控制步**，含 JSON 动作往返，0 adapter/execution error、0 clipping；这不是 30 个完整 episode，也不是 30 次学生闭环任务评测。
- 两个初始化边界控制步出现约 10.14/10.38 mm 位置误差和约 0.0468 rad 方向误差；独立跟踪误差阈值未冻结，因此 `adapter_not_verified` 仍是关闭状态。
- 64 个训练窗口来自 2 条独立教师 episode（Lift/Stack 各一条），窗口存在重叠；固定微批 loss 下降只证明加载、梯度和过拟合通路，不证明泛化、方式绑定或闭环。
- 学生正式训练前仍需单独登记 `episode_count`、`sequence_count`、`control_step_count`，并完成完整序列的导出—导入—同一控制器回放。不能把已有 30 控制步写成 30 episode。
- 学生候选的观测键、相机同步、动作 horizon、reset/init-state、归一化和 checkpoint/选模规则尚未在 Lift/Stack 目标任务上冻结；`fixed_student_approved=false` 继续成立。

## Gate 2 状态表

| 项目 | 当前证据 | 状态 | 仍需完成 |
|---|---|---|---|
| D0 开发启动集 | 64 episodes、16 组配对、清单和 hash | development complete | 独立金标标注与仲裁 |
| 五方法共同初态 | 40/40 教师终态、2 组初态 hash、0 错误 | development audit complete | 物理/安全最终审计；不得升级为方法通过 |
| 阶段参数语义 | 240/240 检查通过，15 explicit-neutral 通过 | execution semantics complete | 阶段行为的独立物理阈值和串扰校准 |
| 请求语义 | 13/40 精确匹配；M 0/8 | unresolved | 离线合同修复、差异分类、再决定是否有界重跑 |
| Wipe/接触 | 25 格开发网格 | candidate range only | 独立测量语义、阈值 registry、sealed audit |
| 方式/安全阈值 | 未冻结 | blocked | 独立校准数据和非作者复核 |
| 金标 | 24 条双标候选清单已生成 | annotation pending | 实际盲标、仲裁、一致性统计 |
| 数据 split | 当前 128 条 source/window 全部 development | development only | generator sealed、student train/val/test 配额和 lineage 冻结 |
| 学生动作接口 | 30 control-step smoke replay 通过；adapter 未验证 | preflight only | 完整序列回放、独立阈值、任务 reset/观测合同 |
| 小样本学习 | 64 window 固定微批通过 | path check only | 独立 source-episode 验证、方式/阶段覆盖、闭环前置检查 |
| 正式预算/扩量 | 未冻结 | no-go | 先完成以上前置条件，再由负责人冻结 |

## 当前不放行条件

以下任一项存在时，继续只做开发/校准和离线证据整理：

1. `formal_manner_status`、`formal_safety_status` 或 `overall` 仍为 `unknown`；
2. 请求约束被候选静默增加、删除或改变，尤其是 M 的 0/8 精确匹配未解释；
3. 接触力、力矩、夹持力、时间轴或阶段事件没有单位/有效位/真实 `dt` 证据；
4. 30 控制步被误写为 30 episode，或学生 adapter 仍以 `adapter_not_verified` 关闭；
5. 训练窗口无法按真实 `task`、`seed`、源初态 `initial_state_hash`、`reset_id` 和
   `lineage_id` 先切分，或缺少独立验证集；`object/config` 覆盖重叠必须如实报告，
   不得据此声称对象泛化；
6. 金标、sealed split、阈值 registry 和审计哈希尚未形成可追溯工件。

## 下一工作包

具体顺序和机器验收条件见 `reports/gate2/gate2_next_action_plan_20260921.md`。该
工作包允许在预先固定预算内执行开发运行和必要代码修复，但不启动正式扩量，
不把人工未签收的候选写成正式通过。

## 状态推进规则

不得通过人工改写报告、删除差异、重命名 `unknown`、把控制步计数升格为 episode，
或其他 artificial promotion 推进状态。状态只能由相应的可追溯审计工件和独立物理
评测更新；在此之前继续保持 `accepted=0`、三态 `unknown` 和
`promotion_allowed=false`。
