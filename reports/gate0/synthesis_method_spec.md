# MannerCAP v2 请求编排方法规格（Gate 0）

版本：`request-manifest-v1`，2026-09-16
负责人：C（合成方法）
状态：Gate 0 初版交付；这是请求空间与配对的工程接口，不是合成效果或物理实验结果。

## 1. Gate 0 结论与范围

本次交付实现了生成前的请求登记和配对校验，入口为
`synthesis_v2/request_manifest.py`。每个请求在候选生成前就进入清单，后续解析失败、执行失败、重试耗尽或没有发布轨迹时仍应保留该请求。模块不会运行仿真，不调用模型或外部 API，不计算物理阈值，也不修改已有执行器。

首轮默认开发集明确命名为 `development`，默认任务白名单只有已经存在运行入口的 `lift` 和 `stack`。`wipe` 及其他第三任务仍处于校准待定状态；它们不进入默认请求空间。当前实现没有填写任何未经 A/B 校准的摩擦、质量、力、速度、阻抗或安全阈值。`physical_config`、`attribute_intervention` 和完整 `initial_state` 是由后续校准适配器提供的透明 JSON 字段，编排器不解释其物理含义。

Gate 0 交付的放行边界是：可以把一组显式请求转成稳定的 JSONL 记录，并在生成前发现任务、split、配对和字段问题。当前 manifest 仍标记为 `draft`，尚未由负责人冻结；环境真实运行、动作/测力校准、候选执行、独立最终审计和学生学习检查尚未由本文件完成，不能据此签收 R1—R8 或任何合格率。完整初态的 reset/恢复和控制器状态回放校验也尚缺，`initial_state` 目前只是登记字段。

## 2. 请求记录合同

`RequestDraft` 表示一个尚未绑定配对元数据的请求。`build_request_manifest` 或首轮专用的 `build_development_manifest` 输出每请求一条 JSON 对象；`write_request_manifest` 以 UTF-8 JSONL 写出。

每条记录至少保留以下内容：

| 字段 | 语义 |
| --- | --- |
| `request_id` | 由全部声明内容确定的稳定请求 ID |
| `task`、`scene_id`、`target_object` | 任务、场景和目标对象标识；首轮任务限 Lift/Stack |
| `split` | 显式数据用途；首轮构造器只接受 `development` |
| `initial_state` | 完整初态快照，由适配器提供；seed 不能替代它 |
| `physical_config` | 物理配置的原样 JSON；当前编排器不解释、不补全属性 |
| `raw_language` | 原始语言要求，保留原文字串 |
| `requested_constraints` | 原始约束对象，复制后原样写出；不得裁剪、放宽或用派生阈值覆盖 |
| `stage_scope` | 阶段实例和作用范围声明；顺序由调用方给出 |
| `allowed_control_channels` | 允许候选计划使用的控制通道名 |
| `budget` | 请求级预算上限；当前只登记，不执行计费 |
| `request_status` | 初始值为 `planned`；不以成功请求列表替代完整分母 |
| `manifest_status` | 初始值为 `draft`；Gate 0 尚未冻结清单 |
| `requested_constraints_hash` | 原始约束规范化 JSON 的 SHA-256 校验摘要，便于审计 |
| `physical_state_hash` | `initial_state` 与 `physical_config` 的完整内容摘要 |

构造时会深拷贝 JSON 字段，因此调用方之后修改输入对象不会改变清单中的原始约束。哈希只用于可复查身份，不表示约束已经满足，也不表示物理状态已经校准。原始约束、预测计划、实际执行计划和实测行为仍需在后续台账中分别保存；本模块只登记第一层原始请求。

## 3. 确定性 ID 与排序

请求 ID 的内容是去掉生成 ID 和配对元数据后的请求字段。实现使用 UTF-8、`ensure_ascii=False`、按键排序、紧凑分隔符和禁止 NaN 的 JSON 表示，再取 SHA-256 前 24 个十六进制字符，格式为：

```text
req_<24 hex characters>
```

同一语义对象的映射键顺序不会改变 ID；允许控制通道会排序并去重。配对 ID 使用规范化的 `pair_type` 和两个已排序的 `request_id`，格式为：

```text
pair_<24 hex characters>
```

配对成员输入顺序不会改变配对 ID。输出记录按 `request_id` 排序；配对角色 `a`/`b` 也按请求 ID 排序。角色只表示稳定的成员位置，不应被解释为因果方向。

## 4. Split 与开发集

支持的名字为 `development`、`generator_test`、`student_train`、`student_validation` 和 `student_test`；`dev` 只作为输入别名，写出时统一为 `development`。配对校验在规范化后的 split 上进行。

`build_development_manifest` 是首轮入口：它要求每个请求的 `split == "development"`，并以 `lift`、`stack` 为唯一默认任务。`build_request_manifest` 允许后续在已冻结的其他 split 上复用相同 schema，但仍默认拒绝白名单以外的任务。跨 split 的配对直接失败，不会把关联样本静默移动到某个 split。划分应先于窗口切片和轨迹生成；本初版只登记请求，尚未实现数据窗口导出。

## 5. 三类配对

一条请求在初版 schema 中最多归属一个配对，因而 `pair_id` 始终无歧义。每个配对恰好有两个不同成员。配对不代表两条轨迹已经成功；后续只有两侧均经过最终审计才能计为完整配对，单侧成功仍须保留并计入请求分母。

### 5.1 `modifier_word`

两侧必须任务、场景、完整初态、目标对象、物理配置、阶段作用域、控制通道、split 和预算一致，只改变原始语言及其对应的原始约束声明。两侧原始语言必须不同。编排器只检查结构，不能判断两个自然语言短语是否真的只改了修饰词；该语义应在开发集抽查和独立标注中确认。

### 5.2 `physical_attribute`

两侧必须使用相同任务、场景、中性语言、原始约束、阶段作用域和控制域，并提供同一个 `geometry_key`（也可由输入别名 `common_geometry_id` 提供）。完整初态或物理配置应有实际内容变化，因而 `physical_state_hash` 必须不同。属性干预不被解释为某一种材料、质量或力学性质；调用方可通过 `attribute_intervention` 保存校准后的干预字段，配对元数据同时保存共同几何标识和两侧完整状态摘要。

几何标识与完整物理状态摘要必须同时保留：前者用于证明几何共用，后者用于防止把属性变化误写成相同初态。未提供共同几何标识或两侧状态完全相同时，配对被拒绝。

### 5.3 `stage_scope`

两侧必须任务、场景、完整初态、目标对象、物理配置、控制域、split 和预算一致；`manner_key`（若声明）必须一致。两侧的 `stage_scope` 必须不同，原始语言和原始约束可以随作用阶段声明一起变化，并会逐侧原样保留。编排器不会把阶段名解释成视觉真值，也不会生成阶段阈值。

三类关系都执行同一条 split 检查；任何跨训练/测试或开发/封存测试的配对都会抛出 `ManifestError`。输入配对 ID 若与确定性计算值不一致也会失败，避免手工覆盖 lineage。

## 6. G0/G1/G2/G3/M 方法现状

下表中的编号是上游合成方法编号，不是 Gate 关口。当前仓库已有 Lift/Stack 的 manner 执行脚本，但本次没有把它改造成请求生成器，也没有宣称已有下列方法的正式结果。

| 方法 | 方案定义 | 本次现状 | 明确未实现部分 |
| --- | --- | --- | --- |
| G0 | 固定脚本和固定执行参数，使用共同验证与预算 | 未接入请求编排；可作为后续共享 manifest 的固定执行参考 | 固定参数注册、统一执行/记录、生成期验证、失败台账、成本计费、独立审计 |
| G1 | 全局控制参数随机采样，加共同验证 | 未实现采样器；没有随机候选或实验结果 | 随机域冻结、seed/资源账本、候选执行、筛选与去重、完整分母统计 |
| G2 | 按阶段采样候选，加相同物理筛选 | manifest 可保存 `stage_scope`，但没有候选计划 | 阶段采样域、阶段事件与执行器接线、共同物理筛选、阈值校准、消融 |
| G3 | 使用相同属性/先验的人工上下文规则生成，加共同筛选 | 未实现规则生成器；不向本模块注入物理常识 | 公开先验注册、规则版本、候选计划、共同筛选、开发集调参冻结 |
| M | 本方案的上下文阶段候选计划与合成流程 | 本次只实现其请求层前置编排；没有模型调用 | 阶段候选计划、控制通道合法域接线、执行/记录、生成期验证、有限重采样、去重与最终审计 |

G2、G3 的公平比较条件和 M 的候选计划都必须在开发集调试后冻结。当前没有 API 请求、模型响应、候选轨迹、发布轨迹、成功率、质量阈值或成本结果；任何后续结果都应使用同一 manifest、执行器、安全层和计数协议。

## 7. 与后续台账的边界

本初版输出的是请求计划，不是 attempt、trajectory 或 release ledger。后续实现必须为每次提案（包括解析失败）、实际执行、安全接管、失败、重试、重复和资源消耗建立独立记录，并让 `source_attempt_id` 回到这一层的 `request_id`。一个请求最多发布一条示范，但可以有多个有计费的候选尝试；不能在内部无限重试，也不能删除失败轨迹后拼接成功轨迹。执行前还必须验证完整初态快照可恢复、控制器状态一致，并记录恢复失败；这些检查当前未实现。

生成期验证和 B 管理的最终审计必须分开。最终测试的拒绝原因不能回流生成器修到通过；变更后需另行封版。物理阈值、力单位、阶段真值和动作接口由 A/B/D 校准并冻结后，才可在执行和评测模块中引用。本文件故意不提供这些数值。

## 8. 消融与未完成工作

请求层已能为后续 `无配对编排` 消融提供相同的任务/属性/方式边际字段，但尚未生成或匹配数据。`无阶段局部控制`、`无上下文属性` 和 `无生成期方式筛选` 尚未实现；也没有失败反馈重采样、近重复去重、阶段标签导出、动作回放、学生训练或闭环评估。

下一关口前至少需要：

1. A/B 完成 Lift/Stack 的执行记录与独立约束校准，并登记真实可用控制通道。
2. C 在开发集冻结请求配额、阶段模板、属性来源、预算和 G2/G3 规则版本。
3. B 明确生成期验证与最终审计的状态、unknown、失败分类和计数分母。
4. D 核对 pair/lineage 不跨 split，完成源轨迹不可变、回放和小样本学习预检。
5. Gate 2 冻结候选上限、实验矩阵和数据划分后，才进入正式生成。

## 9. 仅请求层的复现命令

```python
from synthesis_v2.request_manifest import (
    RequestDraft,
    build_development_manifest,
    make_pair,
    write_request_manifest,
)

slow = RequestDraft(
    task="lift",
    scene_id="scene-01",
    initial_state={"snapshot_ref": "calibration-supplied"},
    target_object="red cube",
    physical_config={"config_ref": "calibration-supplied"},
    raw_language="place the cube slowly",
    requested_constraints={"raw": "preserve-me"},
    stage_scope=[{"stage_instance_id": "lift"}],
    allowed_control_channels=("joint_velocity", "gripper"),
    budget={"max_attempts": 3},
    split="development",
    geometry_key="geometry-supplied",
)
fast = RequestDraft(
    **{**slow.__dict__,
       "raw_language": "place the cube quickly",
       "requested_constraints": {"raw": "preserve-me-fast"}}
)
records = build_development_manifest(
    [slow, fast], [make_pair(slow, fast, "modifier_word")]
)
write_request_manifest("request_manifest.jsonl", records)
```

上例只展示请求登记；它不会启动仿真或模型服务。实际适配器应显式提供完整初态、已校准配置和真实约束原文，并保存本文件以外的候选与执行台账。

## 10. 验证记录

已执行：

```text
python -m py_compile synthesis_v2/request_manifest.py
python -m py_compile tests/test_v2_requests.py
python -m unittest -v tests/test_v2_requests.py
```

结果：7 项请求层契约测试通过，覆盖确定性 ID、开发集白名单、三类配对、跨 split 拒绝、原始约束复制、未配对请求保留和 JSONL 写出。本次未运行仿真、正式实验、模型服务或外部 API。
