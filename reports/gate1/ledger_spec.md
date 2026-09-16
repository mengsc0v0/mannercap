# MannerCAP v2 候选尝试与资源台账规格（Gate 1）

版本：`attempt-ledger-v1`，2026-09-16  
负责人：C（合成方法）  
状态：Gate 1 纯 Python schema/JSONL writer 初版；不是仿真运行结果，也不是正式资源测量结果。

## 1. 目的与边界

`synthesis_v2/attempt_ledger.py` 位于请求 manifest 与未来执行器之间，负责把每个候选尝试登记成一个可审计对象。它不启动仿真、不调用模型或外部 API、不执行重试、不验证物理约束，也不更改主 agent 脚本。调用方必须对每一次候选显式调用 `record_attempt`；该方法只追加一条记录。

一个 `request_id` 可以有多个候选，但所有候选共享 ledger 的 `max_attempts` 上限。解析失败、非法参数、执行失败、验证失败、重复候选和接受都各自占用一个连续的 attempt index，并且都会生成 `attempt_id`。请求本身在候选失败或预算耗尽后仍保留，不能只写成功样本。

当前默认 `max_attempts=3` 与方案首轮建议一致；Gate 2 之前仍可根据校准和预算冻结。这里的默认值不是已批准的正式实验规模。

## 2. 资源向量

每条 attempt 都必须带一个完整的 `resources` 对象。字段名和单位如下：

| 字段 | 单位 | 约束 |
| --- | --- | --- |
| `physics_substeps` | simulation substeps | 非负整数；覆盖本候选实际推进的物理子步 |
| `api_calls` | calls | 非负整数；没有服务调用时显式写 0 |
| `api_tokens` | tokens | 非负整数；没有服务调用时显式写 0 |
| `cpu_time_s` | seconds | 非负有限数 |
| `gpu_time_s` | seconds | 非负有限数；未使用 GPU 时显式写 0 |
| `wall_time_s` | seconds | 非负有限数 |
| `storage_bytes` | bytes | 非负整数；包括该候选产生的记录/数组等存储 |

缺少任一字段、出现负数、布尔值、NaN、无穷或错误的整数/数值类型都会抛出 `LedgerError`。`ResourceUsage.from_mapping` 也接受少量输入别名（例如 `sim_physics_substeps`、`cpu_seconds`），但输出统一为上述字段。当前模块只校验和保存调用方报数；它没有连接计时器、仿真器、模型客户端或文件扫描器，因而不能把这些值写成已测结果。

## 3. Attempt 记录合同

每条 JSONL 记录包含：

| 字段 | 语义 |
| --- | --- |
| `attempt_id` | 由 `parent_request_id` 与连续 `attempt_index` 生成的确定性 ID |
| `parent_request_id` / `request_id` | 两个同值别名，指向原始请求 |
| `attempt_index` / `attempt_number` | 从 1 开始、无缺口递增的尝试编号 |
| `outcome` / `status` | `parse_failure`、`invalid`、`execution_failure`、`validation_failure`、`duplicate` 或 `accepted` |
| `proposal_source` | 提案来源/方法版本，不能为空 |
| `candidate_plan` | 候选阶段计划；解析失败可以是 JSON `null`，字段仍须存在 |
| `call_record` | 提案调用记录对象；无服务调用时用 `{}`，不得省略 |
| `resources` | 上述完整资源向量 |
| `reason` | 本候选结局的显式原因，不能为空 |
| `request_snapshot` | 注册时复制的完整原始请求 |
| `request_snapshot_hash` | 原始请求快照摘要 |
| `duplicate_of_attempt_id` | 仅对 `duplicate` 必填，指向被判为重复的候选 |
| `request_state_after` | `open`、`accepted` 或 `budget_exhausted` |
| `metadata` | 其他 JSON 证据；不用于隐藏候选或修改请求 |

所有失败分类都保留为独立 outcome；`accepted` 也必须有 attempt 记录。解析失败没有可执行计划时，`candidate_plan=null` 是明确值，不是漏字段。重复候选必须给出 `duplicate_of_attempt_id`；其他 outcome 不得携带该字段。`outcome` 和 `status` 若同时提供必须一致，输入别名会序列化成统一的 canonical 值。

## 4. 原请求不变

调用 `register_request` 时，ledger 深拷贝请求并计算 `request_snapshot_hash`。每条后续 attempt 都从这个副本写出 `request_snapshot`，不会引用调用方可变对象。即使外部随后修改 `requested_constraints`、`initial_state` 或其他请求字段，台账仍保留注册时原文。

`record_attempt` 可接收可选的 `request_snapshot` 供调用方复核；其规范化 JSON 摘要若与注册摘要不一致，直接拒绝追加。这种检查用于发现原请求被静默修改，不用于裁剪、单位转换、阈值补全或重写约束。请求 manifest 中的 `requested_constraints` 应继续作为原始约束来源，预测计划、执行计划和实测行为须在其他字段/台账中单独保存。

## 5. 共同预算与终态

ledger 对每个请求维护独立计数，但所有 outcome 共用同一个 `max_attempts`：

```text
register -> open
open -- parse/invalid/execute/validate/duplicate --> open
open -- any failure at max_attempts --> budget_exhausted
open -- accepted --> accepted
accepted or budget_exhausted -- any further attempt --> reject
```

接受后立即进入 `accepted`，即使尚有剩余预算也不能继续候选或补采。失败达到上限立即进入 `budget_exhausted`，不产生额外的“预算耗尽伪 attempt”。每次 `record_attempt` 只执行一次 append，不包含内部循环、隐式候选、未计费重试或成功后修补。attempt index 必须连续；传入手工 `attempt_id` 时必须与确定性计算值一致。

一个候选通过生成期验证后即可记录 `accepted`，但这不等于最终审计合格；最终审计结果应在后续 ledger 中关联，而不能用测试结果回流继续修同一请求。

## 6. 确定性 ID

`attempt_id_for(request_id, attempt_index)` 对规范化 JSON
`{"request_id": ..., "attempt_index": ...}` 取 SHA-256 前 24 个十六进制字符，格式为：

```text
attempt_<24 hex characters>
```

同一请求的每个 attempt index 只有一个 ID，资源或 outcome 改变不会重写该 ID。记录写出按 `request_id`、注册顺序内的 attempt index 排序，便于从请求追到全候选。

## 7. JSONL writer

`write_attempt_ledger(path, ledger)` 或 `write_attempt_ledger(path, records)` 使用 UTF-8、紧凑 JSON、按键排序和真实换行写出。一条 attempt 对应一行；writer 不过滤失败、重复或空计划，也不按成功率删记录。ledger 对象写出时会包含所有已注册请求产生的 attempt，尚无候选的请求不会伪造一行 attempt，应由 request manifest 保留其完整分母。

推荐调用形态：

```python
from synthesis_v2.attempt_ledger import AttemptLedger, write_attempt_ledger

ledger = AttemptLedger(requests=[request_record], max_attempts=3)
ledger.record_attempt(
    request_record["request_id"],
    outcome="parse_failure",
    proposal_source="g0-fixed-v1",
    candidate_plan=None,
    call_record={},
    resources={
        "physics_substeps": 0,
        "api_calls": 1,
        "api_tokens": 20,
        "cpu_time_s": 0.01,
        "gpu_time_s": 0.0,
        "wall_time_s": 0.02,
        "storage_bytes": 256,
    },
    reason="candidate response could not be parsed",
)
write_attempt_ledger("attempt_ledger.jsonl", ledger)
```

示例只登记一个失败候选，不执行重试。下一候选必须由调用方再次显式调用 `record_attempt`，并使用同一请求 ID 与共享预算。

## 8. Gate 1 当前状态与剩余工作

已完成：

- 所有要求的候选结局都有统一 attempt schema 和 `attempt_id`。
- 原始请求深拷贝、hash 复核和每次 attempt 的快照保留已实现。
- 七维资源向量、缺字段/负值/非有限值拒绝和稳定 JSONL writer 已实现。
- 接受后终止、预算耗尽终止、连续 attempt index 和重复引用规则已实现。

尚未完成：

- 尚未接入现有 Lift/Stack 执行器、解析器、生成期验证器或任何 API；当前没有候选轨迹和资源实测。
- 尚未实现初态恢复/控制器状态一致性校验；`request_snapshot` 只是请求证据，不是仿真 reset 成功证明。
- 尚未实现独立最终审计、物理阈值、阶段真值、去重特征或 release ledger；`duplicate` 只能保存调用方给出的引用。
- `max_attempts=3` 仍是可复现的开发默认值，Gate 2 才冻结正式共同预算；当前 manifest 和 ledger 都不构成正式实验封版。
- 尚未生成正式 `attempt_ledger.jsonl` 数据；本次不运行仿真、不扩量、不训练。

## 9. 验证记录

已执行：

```text
python -m py_compile synthesis_v2/attempt_ledger.py tests/test_v2_attempt_ledger.py
python -m unittest -v tests/test_v2_attempt_ledger.py
```

结果：8 项纯 Python 契约测试通过，覆盖完整 outcome 集、资源字段、负值/缺字段拒绝、共同 `max_attempts`、接受后禁止继续、原请求快照、重复引用、确定性 ID 和 JSONL 写出。本次没有外部 API、模型服务、仿真或正式实验调用。
