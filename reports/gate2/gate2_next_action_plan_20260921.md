# Gate 2 下一轮行动计划：独立校准候选与学生放行前置

版本：2026-09-21
范围：开发/校准证据收口；不启动正式 E1--E4、D1/D2、长训练或无上限扩量。
下一轮允许在启动前固定物理子步、请求/候选、GPU/API、存储和停止规则的开发预算
内执行必要的开发运行与代码修复；新增执行、失败、重试和调用都必须进入成本台账，
并保持 `development` 标记。

## 目标和当前边界

当前最有价值的下一步是把“教师执行完成”与“可物理签收、可供学生学习”之间的
缺口变成机器可检查的前置条件。40/40 五方法候选已经完成共同初态和阶段参数审计，
但 `manner`、`safety`、`overall` 仍是 `unknown`。M 的 8 条缓存提案与原请求精确
匹配为 0/8，因此先做离线语义收口；任何后续模型调用都要在语义合同修复后、预算
重新冻结并计入台账。

学生侧必须把三种计数分开：

```text
episode_count       = 完整 reset 到终态的独立回合数
sequence_count      = 导出—导入后完整回放序列数
control_step_count  = 实际送入控制器的 transition/step 数
```

现有 LA4VLA 证据为 `control_step_count=30`、`adapter/execution error=0`、
`clipping=0`；它不是 `episode_count=30`，也不是 30 次学生闭环评测。两个初始化
边界步的误差仍必须保留，不能删除后再计算通过率。

## WP-A：离线请求语义收口（先于任何 M 重跑）

输入：

- `reports/gate2/five_method_audit_20260921_v2.json`
- 40 条候选的 `requested_spec`、`executed_parameters` 和缓存 M 提案
- `reports/gate2/five_method_checkpoint_20260921.md`

执行：

1. 对 40 个候选逐项重算请求与候选差异，按 `candidate_global_extra`、
   `candidate_phase_extra`、`neutral_extra_constraints`、`phase_constraint_changed`、
   `requested_global_missing`、`requested_phase_missing` 六类登记；同一候选的多项
   差异保留为多条差异记录，不把记录数当候选数。
2. 把原始请求、模型提案、解析后的候选、实际执行参数四层分开；不修写原请求，
   不把执行成功当作语义匹配。
3. 对 M 的 0/8 精确匹配单独生成差异清单，标出是模型提案偏差、阶段映射、
   neutral 默认值还是执行器转换造成；无法归类的项保持 `unknown`。
4. 只使用已缓存提案完成本轮审计；是否重新调用模型留到语义合同修复并由预算
   重新冻结之后。

机器验收：

- 40/40 候选均有请求快照、提案/解析状态、执行参数和差异分类；无孤立候选；
- 每个方法的匹配分母仍为 8，合计 40，当前记录应保持 G0 2/8、G1 2/8、G2 1/8、
  G3 8/8、M 0/8；
- 差异分类可由原始 JSON 确定性重算，重复运行输出字节一致；
- 语义差异只进入差异分类和资格待审队列，不直接等同于物理失败或物理通过；
  候选资格由独立物理审计根据原始请求、实际执行参数和任务/方式/安全三态判定；
- 当前 `accepted=0`、`formal_manner_status=unknown`、`formal_safety_status=unknown`
  和 `promotion_allowed=false` 的原因是独立资格审计尚未完成，不能把它们改写成
  “语义差异导致的失败”；
- 生成一份可追溯的小型审计报告即可；若开发预算批准新增运行或调用，必须保留
  新 episode、失败/重试和 API 成本记录。

## WP-B：独立物理校准候选包

这是本轮的主工作包。它的产物是“可交给独立评测角色复核的候选校准包”，不是
最终阈值签收。阈值来源必须与五方法输出隔离，不能用 40 条候选的均值、最小值
或最大值自校准。

输入和用途：

| 输入 | 可用于什么 | 不可用于什么 |
|---|---|---|
| Wipe 0--4 mm、5 seed × 5 网格 | 检查几何候选范围、任务响应、测量缺失和 seed 敏感性 | 直接命名为外加载荷或安全阈值 |
| Lift/Stack 已有连续状态和控制记录 | 检查时间轴、阶段事件、速度/jerk 字段是否可计算 | 从方法表现反推正式上下界 |
| 五方法 40 候选 | 仅作隔离性反例和审计输入 | 作为校准训练集或阈值来源 |

交付三个小工件：

1. `calibration_input_manifest`：每条输入记录 task、phase instance、object/config、
   `trajectory_id`、`reset_id`、`lineage_id`、split、真实 timestamp/dt、采样频率、
   传感器有效位、单位、坐标系和来源 hash。
2. `calibration_candidate_report`：分别报告无接触、受控接触、过作用/越界样例
   是否存在；速度、加速度、jerk、contact force、torque、gripper force/width、
   冲量和阶段串扰的可观测性、分辨率、缺失率和候选范围；无法确认的量写
   `unknown`，不补零。
3. `constraint_registry_candidate`：写入量的定义、单位、适用阶段、计算公式、
   数据版本、来源和状态 `candidate_only`；不写成 `frozen`。

机器验收：

- 所有被用来计算速度/jerk/冲量的样本通过轨迹合同：时间严格递增、`dt` 与相邻
  timestamp 一致、无缺口、无 reset 跨越、无 lineage 混合；导数只在连续有效区间
  内计算；
- force、torque、gripper force/width、控制指令和实测值字段分列，单位/坐标系/
  传感器有效位齐全；缺字段或失效区间输出 `unknown`；
- 每个声称支持的 task×phase 至少能列出无接触、受控接触和过作用/安全越界中的
  可复核样例；若现有输入没有已知载荷或独立安全语义，则该维度明确标为未校准，
  不将 Wipe 几何压入量代替力；
- 候选阈值只由预注册校准输入生成，不能读取方法名、正式测试结果或学生测试
  结果；同一输入重复运行的报告和 hash 一致；
- 包完成后状态为 `calibration_candidate_ready=true`、`thresholds_frozen=false`、
  `physical_qualification=unknown`。这使工作包可以无需人工签收地验收“工件完整”，
  同时保留正式阈值必须由独立角色复核的边界。

如果缺少独立受控载荷或安全上限证据，正确结果是“候选包未满足完整性、正式接触
范围收缩为 Lift/Stack 抓放或继续 unknown”，不得用 1--4 mm 的任务成功率替代。

## WP-C：学生动作接口前置检查

输入：

- `reports/gate1/la4vla_osc_roundtrip_results.md`
- `reports/gate1/student_baseline_audit.md`
- `reports/gate2/la4vla_microset_learning_preflight.md`
- `synthesis_v2/la4vla_osc_adapter.py` 及其固定配置 hash
- D0 source-episode lineage 和当前 split manifest

执行：

1. 建立学生回放 manifest，强制同时记录 `episode_count`、`sequence_count`、
   `control_step_count`、task、phase、reset/init-state、adapter 版本和控制器版本。
   现有 30 个样本填入 `control_step_count=30`；若没有完整序列，不填充
   `episode_count`。
2. 把初始化边界样本与普通样本分层报告，不删除边界样本，不用非初始化子集隐藏
   最大误差。记录平移、旋转、夹爪的单位、方向、频率和是否裁剪。
3. 按真实 `task`、`seed`、源初态 `initial_state_hash`、`reset_id` 和 `lineage_id`
   先切开发训练/开发验证，再生成窗口；窗口不跨 reset、缺口、无效 padding 或
   lineage。归一化只用开发训练部分，验证部分不参与调参。`object/config` 作为
   覆盖报告字段保留；若两侧共享对象配置，必须如实报告，不能据此声称对象泛化。
4. 对动作导出—JSON 序列化—导入—同一 `OSC_POSE` 控制器执行做闭环字段检查；
   这一步验证动作可表达性和接口一致性，不调用学生 checkpoint，也不声称学生已学会。

机器验收：

- 30/30 现有证据被标为 control-step smoke replay；报告中不存在“30 个 episode”
  或“30 次闭环评测”的表述；
- 任一完整序列的观测、动作、timestamp、dt、相机帧、state、reset 和 lineage
  可逐行追溯，JSON 往返无改变，维度/单位/夹爪符号一致；
- 0 clipping、0 非有限值、0 schema 错误；初始化边界误差单列并保留；
- 独立阈值未冻结前，`adapter_verified` 保持 false，`adapter_not_verified` 不得
  被静默清除；
- 形成可供独立复核的完整序列计数后，才可进入下一层学生小样本检查。若只有
  step-level smoke 证据，状态仍为 `preflight_only`。

## WP-D：学生开发集小样本可学性

该包只证明数据和训练通路能工作，不回答泛化、方式学习或闭环收益。使用现有 D0
development 数据做按 source episode 的开发划分，避免把 64 个重叠窗口当成 64 条
独立轨迹。

建议执行结构：

- 按真实 `task`、`seed`、源初态 `initial_state_hash`、`reset_id` 和 `lineage_id`
  冻结开发 train/validation lineage；`object/config` 作为覆盖字段报告，不作为未见
  对象泛化的声明依据；
- 训练窗口只来自 train source episodes，validation 窗口来自未见 source episodes；
- 固定起点 checkpoint、归一化、optimizer、步数、batch、随机种子和选模规则；
- 保存训练前后 loss、非有限计数、有效监督量、数据 hash、checkpoint hash 和
  validation 输出；不使用学生 sealed test。

机器验收：

- train/validation 不共享 source episode、`initial_state_hash`、`reset_id` 或派生
  `lineage_id`；这些键的交集必须机器检查为零；
- `object/config` 是否跨 split 共享只作为覆盖统计报告，不作为硬隔离条件，也不据此
  宣称未见对象或配置泛化；
- 所有输入字段有限、图像引用可追溯、动作维度和 mask 与 adapter 合同一致；
- 训练无非有限 loss，固定微批可复现；validation 结果单列，不能只报告训练 loss；
- 失败时保留失败样本和原因，状态回退到 `student_preflight_unknown`，不启动长训练。

## 统一放行门

在以下布尔条件全部满足前，只允许离线审计、开发校准和有限的开发集诊断：

```text
semantic_diff_audit_complete
and calibration_candidate_ready
and complete_sequence_manifest_ready
and adapter_verified
and student_dev_validation_complete
and formal_split_and_budget_frozen
```

其中 `adapter_verified` 和 `formal_split_and_budget_frozen` 仍是正式放行条件，不能
由本轮报告自行设为 true。任何条件不满足都不触发 E1--E4、D1/D2 或 18 次正式训练。

## 停止条件和结果边界

- 发现 force/torque 单位、时间轴、reset 或传感器有效位不可信：停止阈值计算，
  只交缺口清单；
- 发现请求被静默修改：停止 M 扩量，保留原请求和全部差异；
- 发现学生动作回放存在系统性偏差或 adapter 仍无法验证：停止长训练，先修接口；
- 发现开发窗口跨 lineage 或验证集泄漏：删除相应派生窗口的资格，保留源记录和
  原因，不重算成更好的产率；
- 本计划完成只允许把状态推进到 `ready_for_independent_review`，不能把
  `unknown`、候选阈值或 step-level replay 改写为正式成果。
