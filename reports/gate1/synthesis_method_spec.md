# MannerCAP v2 合成方法配置与复现规格（Gate 1 / WP4）

版本：`synthesis_v2.method_config.v1`，2026-09-16  
负责人：C（合成方法）  
状态：五方法配置合同初版；仅声明结构与校验边界，`development_pending`，尚未冻结正式方法配置或实验清单。

## 1. 目的和范围

本交付把方案第 8.1 节的五种合成方法写成可加载、可校验、可做消融的 YAML 合同。实现位于 `synthesis_v2/method_config.py`，配置位于 `configs/generation_methods/`。它只处理请求编排边界，不加载模型、不调用 LLM/API、不运行仿真、不生成候选轨迹，也不改变现有执行器和评测脚本。

五个配置必须共享同一请求合同、任务空间、公开约束视图、执行器、安全层、配对编排、尝试上限、资源台账字段、校验入口和复现字段。方法差异只放在 `method` 能力块中，并由校验器拒绝越出声明的差异。

## 2. 五方法定义（按方案第 8.1）

| 方法 | 本版结构定义 | 关键差异 |
| --- | --- | --- |
| `G0` | 固定脚本、固定执行参数的参考方法 | `fixed_script`；固定全局采样；不使用语言、阶段局部控制或对象属性 |
| `G1` | 全局控制参数随机采样，再走共同公开校验 | `global_random`；随机域为全局；不按阶段局部采样，不使用对象属性 |
| `G2` | 按阶段采样候选，再走相同物理筛选 | `stage_random`；采样 scope 为 `stage`，启用阶段局部控制；不使用对象属性 |
| `G3` | 使用同一对象属性/先验的人工上下文规则，再走共同筛选 | `context_rules`；规则只引用公开任务空间和公开约束注册表，并输出阶段局部控制计划 |
| `M` | 使用上下文生成阶段候选计划，再走共同筛选 | `context_stage_plan`；同时使用公开属性、阶段上下文和阶段候选计划 |

`G1` 明确是“全局控制参数随机采样”，不是语言直映方法。`G2/G3/M` 中的语言和阶段字段只表示方法能力合同，不代表本交付已经调用语言模型或完成校准。

每个方法的输出字段都保留 `stage_instance_id`、`object_id`、`control_plan`、`predicted_plan`、`executed_plan`，以便后续把计划、预测和实际执行分层记录；本版不声称这些字段已有运行值。

## 3. 共同执行与请求合同

五份 YAML 的 `shared` 必须相同，当前共同值如下：

- `request_contract=synthesis_v2.request.v1`，`task_space_contract=synthesis_v2.task_space.v1`；开发任务范围明确为 `lift`、`stack`。第三任务仍是 task-space 中的 `candidate/pending_calibration`，不进入本套可执行方法配置。
- 执行器为 `shared_executor.v1`，安全层为 `shared_safety.v1`。方法配置不携带私有执行器参数或封存引用。
- 约束来源只指向 `configs/constraint_registry.yaml#public_generation`，可见性为 `public`。允许字段仅是任务、阶段、指标、单位、约束 ID、校准来源/版本和状态；`lower`、`upper`、`tolerance` 全部显式为 `null`，未知策略为 `not_qualified`。
- 校验维度固定为 `task`、`manner`、`safety`，未知结果不能当作通过；生成筛选器固定使用公开约束注册表，生成阶段不接收最终审计反馈，也不加载审计工件。
- 配对编排来源固定为已有 `request_manifest`，因此三类 pair（`modifier_word`、`physical_attribute`、`stage_scope`）的同 split、同任务和原始约束保留由请求合同负责。方法配置不能重新定义 pair 或把成员移到其他 split。

方法入口必须接收原始 request 的不可变副本。候选计划可以记录派生字段，但不能改写 `raw_language`、`requested_constraints`、目标、初态引用、split、pair 类型或属性来源。原始约束若未知或未校准，保留原值并按 `not_qualified` 处理，不得补阈值。

## 4. 尝试与资源共同预算

每个 request 的方法比较使用同一共同预算：最多 `3` 次 attempt，范围为 `per_request_shared`。下列结果都必须由 attempt ledger 计数并带有唯一 `attempt_id`：

`parse_failure`、`invalid`、`execution_failure`、`validation_failure`、`duplicate`、`accepted`。

禁止隐藏重试；成功后 `accepted` 是终止状态，首个通过即停止，后续方法或候选不能继续消耗该 request 的 attempt 预算。`preserve_original_request=true` 保证失败反馈也不能变更原请求。资源维度与台账合同一致：

`physics_substeps`、`api_calls`、`api_tokens`、`cpu_time_s`、`gpu_time_s`、`wall_time_s`、`storage_bytes`。

目前所有资源 cap 都保持 `null`，因为尚未完成校准和 Gate 2 预算冻结。配置声明共同字段和停止规则，但没有执行记录。

## 5. 配置 schema 和公开边界

顶层固定包含 `schema_version`、`method_id`、`method_name`、`method_version`、`shared`、`method`、`ablations`、`metadata`。加载器会拒绝未知顶层/子字段、错误 schema、非有限数值、重复列表项以及不完整的共同合同。

`method` 块固定包含 planner/proposal 类型、语言/阶段/对象属性能力、阶段局部控制、公开 manner filter、pair 编排、失败反馈、采样、规则引用和输出字段。采样数值域显式写出 `lower:null`、`upper:null`；本版不把任意数值域、速度、接触或成功阈值当作已批准事实。

规则边界为 `visibility=public`、空 `inline_rules`、`hidden_rules=false`。`G3` 和 `M` 只允许引用 `task_space.public.v1` 与 `constraint_registry.public_generation.v1`。任何隐藏/私有规则、封存内容、最终审计结果字段或内联未版本化规则都会被拒绝。`metadata` 只记录开发状态和未校准说明，不是规则或审计通道。

## 6. 消融合同

`ablations.active` 与五个显式开关必须一致。支持的名称为：

- `no_phase_local_control`
- `no_context_attributes`
- `no_generation_manner_filter`
- `no_pair_orchestration`
- `no_failure_feedback`

`apply_ablation(config, name)` 返回深拷贝，关闭对应能力并记录 active 名称；它不能修改 shared 合同。校验器拒绝“开关已改变但没有声明消融”、把方法能力偷偷改回去、未知消融名以及通过消融引入方法配置之外的新规则或执行参数。基准配置的 `active` 为空，方法固有的能力差异仍由各自 profile 固定。

## 7. 复现规则

请求按 `request_id_lexicographic` 排序。attempt seed 固定为 `sha256(method_id,request_id,attempt_index)`，model cache key 使用同一三元组。配置 hash 使用 UTF-8、排序 key、紧凑 JSON 和 SHA-256；加载器深拷贝输入，避免校验或消融修改调用方对象。完整方法套件必须恰好包含 `G0/G1/G2/G3/M` 五个 ID，并且所有 `shared` 的 canonical hash 相同。

这些规则只保证规划边界的确定性；它们不等于模型服务版本、随机数库、执行器版本或初态恢复已经被实测锁定。当前 `temperature`、`service_version` 和所有未校准数值仍为 `null`。

## 8. 当前 Gate 0/Gate 1/M 状态与未实现部分

| 项目 | 当前状态 |
| --- | --- |
| `G0` | 结构合同和校验已完成；固定脚本/固定参数尚未由本模块执行或冻结 |
| `G1` | 结构合同和校验已完成；全局随机域数值尚待校准，未执行采样 |
| `G2` | 结构合同和校验已完成；阶段采样与共同物理筛选尚未执行 |
| `G3` | 结构合同和公开规则边界已完成；人工上下文规则仍是公开引用结构，未形成实验候选 |
| `M` | 结构合同和阶段计划字段已完成；上下文阶段候选计划尚未生成或评测 |
| `G0/G1/G2/G3/M` 共同比较 | 已锁定 schema 级 shared 合同、三类 pair 入口、最多 3 次共同预算和首个通过停止语义；没有正式实验结果 |

当前仍未实现或未冻结：

- 方法配置、开发 request manifest 和 constraint registry 都仍处于结构/开发状态，正式 manifest 尚未实际冻结；当前只允许真实 Lift/Stack 的 Gate 0/1 检查，不扩量。
- Gate 1 已取得 Lift/Stack 初始 `time/qpos/qvel/ctrl` 零误差恢复和功能终态重放证据，但 controller/cache 完整性及 `1e-8` 逐状态重放仍未通过。`initial_state_id` 只能作为内容 lineage 引用，不能替代这些证据。
- 尚无 Lift/Stack 的物理属性、合法控制数值域、接触/安全/成功阈值校准结论；未知值不得从方法配置推断。Wipe 已完成确定性成功轨迹、欠/标称/过作用开发探测及 F/T 通道已知外载响应检查，但接触模型和安全阈值仍未签收，继续保持 candidate/pending calibration。
- 尚未接入候选解析、共享执行器、真实 attempt/resource ledger、去重证据、轨迹事件判定、最终审计或学生评测；因此没有成功率、轨迹或资源实测结果。
- 尚未冻结模型/服务版本、场景分母、配额和 release ledger。后续 Gate 2 需在公开约束和预算校准后另行审核，不能在本版配置中预填数值或隐藏规则。

## 9. 验证记录

执行命令：

```text
python -m py_compile synthesis_v2/method_config.py tests/test_v2_method_config.py
python -m unittest -v tests/test_v2_method_config.py
python -m unittest -q tests/test_v2_method_config.py tests/test_v2_constraint_registry.py tests/test_v2_requests.py tests/test_v2_task_space.py tests/test_v2_attempt_ledger.py
```

结果：方法配置测试 12 项全部通过；当前仓库完整标准库合同回归共 140 项全部通过。测试只解析本地 YAML/JSON 并验证纯 Python 合同，未调用外部 API、未运行 LLM、仿真或正式实验。
