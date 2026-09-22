# MannerCAP v2 Gate 2 / E0 本地确定性预检规格

版本：`synthesis_v2.e0_preflight.v1`，2026-09-16  
负责人：C（合成方法）  
状态：离线预检入口初版；只生成开发期计划记录，不构成 E0 实验结果，不冻结 manifest、阈值或预算。

## 1. 目的和边界

入口为 `scripts/gate2_e0_preflight.py`，核心实现为 `synthesis_v2/preflight.py`。它读取 `manifest_status=draft`、`request_status=planned` 的 JSONL 请求清单，重新检查请求 ID、原始约束摘要、`development` split、Lift/Stack 白名单和已有三类 pair 合同，然后加载五份方法 YAML 与本地 constraint registry。

本入口只做本地确定性预检：

- G0、G1、G2、G3 各建立一个 request/method 预检候选槽位；槽位有确定性的 ledger `attempt_id`、method 专属 `method_attempt_id`、seed 摘要、候选来源和原请求快照，但 `ledger_outcome=null`、`ledger_recorded=false`，不会冒充真实 `AttemptRecord`。
- M 永远不调用服务，逐 request 建立显式 `skipped` 记录，`reason_code=api_not_configured`，不计入已消耗 attempt 或资源。
- 所有未执行候选保持 `pending`；task/manner/safety 和 overall 保持 `unknown`；最终审计状态为 `unknown` 且 `checked=false`。
- 不加载仿真器、控制器、模型、外部 API 或最终审计工件；不生成动作、不执行轨迹、不判断物理成功、不改变请求约束。

命令示例：

```text
python scripts/gate2_e0_preflight.py \
  --manifest /path/to/request_manifest.jsonl \
  --output /path/to/e0_preflight.json \
  --method-config-dir configs/generation_methods \
  --constraint-registry configs/constraint_registry.yaml
```

## 2. 输入校验

读取器按 JSONL 非空行解析并按 `request_id` 排序。每条记录必须：

1. 是 JSON object，且 `manifest_status=draft`、`request_status=planned`；
2. 通过现有 `request_manifest.coerce_request`，并使重算的 `request_id` 与输入一致；
3. 使用 `task=lift|stack`、`split=development`；第三任务不会通过本入口偷偷进入清单；
4. 若有 `requested_constraints_hash`，必须与原始 `requested_constraints` 的 canonical SHA-256 一致；
5. pair 记录必须恰好两侧、同任务、同 scene、同 split，pair ID 与三类 `pair_type` 合同均由现有 request contract 重验；
6. 不含封存或最终审计 artifact/reference。原始 request 以深拷贝快照传入 attempt ledger，任何约束值都按输入保留。

pending/incomplete 请求仍保留在输出中。读取器只依据清单元数据（`incomplete=true`、`requests_enabled=false` 或 pending 状态）将其标为 `request_lifecycle=pending_calibration`；不会把它改成失败，也不会把它排除出分母。

## 3. 方法和候选槽位

方法套件必须通过现有 `method_config.load_method_suite` 与 public constraint registry 校验，并且恰好包含 `G0/G1/G2/G3/M`，共享合同 canonical hash 相同。预检方法范围固定为 `G0/G1/G2/G3`：

| 方法 | 预检候选记录 | 本版实际行为 |
| --- | --- | --- |
| G0 | `status=pending` | 仅记录固定脚本/参数 profile，不运行脚本 |
| G1 | `status=pending` | 仅记录全局随机 profile 和 seed，不采样数值 |
| G2 | `status=pending` | 仅记录阶段随机 profile 和 stage scope，不采样或物理筛选 |
| G3 | `status=pending` | 仅记录公开上下文规则 profile，不加载/执行规则生成器 |
| M | `status=skipped` | `reason_code=api_not_configured`，不建立已计费执行记录 |

每个 G0–G3 槽位的 `candidate_plan.control_plan`、`predicted_plan`、`executed_plan` 均为 `null`，`sampling.sampled=false`，并保留 `stage_scope`、允许控制通道、原始 constraints 和 request snapshot hash。`preflight_attempt_id` 由 method/request/index 的 canonical 内容确定；ledger `attempt_id` 仍遵循现有 request/index 算法，在每个方法比较 lane 内使用。

M 的 skip 行保留同一 request/index identity 以便对齐方法矩阵，但 `candidate_plan=null`、`planned_candidate_slots=0`、`actual_attempts_charged=0`。它不是失败、成功或可发布候选。

## 4. 共同 attempt 和资源预算

每个方法/request lane 复制现有 shared attempt contract：

- `max_attempts=3`，scope 为 `per_request_shared`；预检只规划一个候选槽位，实际计费次数为 0，剩余次数为 3；
- 不允许隐藏重试；真实运行由后续 attempt ledger 逐条记录 parse failure、invalid、execution failure、validation failure、duplicate 或 accepted；接受后终止；
- 本预检阶段不写任何六类 ledger outcome，因为 `pending/unknown` 不是候选执行结果；`attempt_ledger.actual_attempt_record_count=0`；
- 资源字段沿用 `physics_substeps`、`api_calls`、`api_tokens`、`cpu_time_s`、`gpu_time_s`、`wall_time_s`、`storage_bytes`。每个槽位输出完整维度，但 reservation 状态为 `not_measured`、`charged=false`、值为 `null`；不把本地预检 CPU 时间伪装成仿真/API 成本；
- 各 method lane 使用相同的每 request 预算合同。预检不会把 G0–G3 的槽位合并为一个共享计数，也不会让 M 的跳过消耗预算。

## 5. 三态和审计边界

报告顶层的 `execution_guards` 全部为 `false`：candidate generation、simulator、controller、model API、external API、final audit loading 和 final-audit feedback 均未发生。每条 request/method 记录的结果固定为：

```json
{
  "validation": {
    "task": "unknown",
    "manner": "unknown",
    "safety": "unknown",
    "overall": "unknown"
  },
  "final_audit": {"status": "unknown", "checked": false}
}
```

constraint registry 当前 `development_pending`，其结构可校验但 `promotion_allowed=false`，预检报告记录 `decision=unknown`。未知不能作为通过，不能据此发布轨迹或计算 E0 成功率。

## 6. 输出 schema

报告包含：

- `manifest`：规范化 source path、draft 状态、内容 hash、request 数、ID、任务和 split；
- `method_suite`：五方法配置 hash、G0–G3 预检范围以及 M 的 `api_not_configured` skip；
- `shared_contract`：executor、request/task-space、public constraint、attempt、预算、校验和复现字段；
- `attempt_ledger`：已注册 request 数、实际 AttemptRecord 数（本版为 0）、计划 lane 数和计费数；
- `records`：每个 request/method 的 request snapshot、pair 信息、pending/skip 候选槽位、共同预算和 unknown 结果；
- `unexecuted_result_policy`：pending/unknown、`unknown_is_qualified=false`、接受数为 0 和最终审计 unknown。

输出经 canonical JSON 写入单个 UTF-8 JSON 文件，并带有不包含自身的 `report_hash`。相同 manifest 内容即使 JSONL 行顺序改变，也得到相同 report hash 和记录顺序。

## 7. 当前 Gate2/E0 状态和未实现部分

已完成：

- draft manifest 的 request/pair/split/原始约束边界重验；
- 五方法配置加载和 public registry 边界检查；
- G0–G3 确定性 pending 候选槽位、M `api_not_configured` 显式跳过；
- request 快照、attempt ID/seed、共同三次预算、七类资源维度和不计费 reservation；
- 不伪造仿真成功、执行轨迹、最终审计或未知通过的保护；
- 本地 JSON writer、CLI 和纯 Python 单元测试。

仍未完成：

- 本入口尚未实际执行 G0–G3 候选、仿真或物理筛选；所有结果仍是 pending/unknown，不是 E0 数据；
- M 的 API/模型配置尚未提供，故本版只记录 skip，未生成上下文阶段候选计划；
- manifest、方法配置、constraint registry 和资源 cap 尚未正式冻结；当前阈值与 cap 继续为 null；
- Gate 1 已取得初始 `time/qpos/qvel/ctrl` 零误差恢复和 Lift/Stack 功能终态重放；`1e-8` 全轨迹逐状态重放仍未通过，`initial_state_id` 只能作为 lineage 引用；
- 尚未接入实际候选解析、共享执行器、attempt ledger 写入、去重、轨迹事件判定、最终审计、学生评测或正式 E0 统计；
- Wipe 未加入本入口，继续保持 candidate/pending calibration，未修改任何 Wipe 文件。

## 8. 验证记录

执行命令：

```text
python -m py_compile synthesis_v2/preflight.py scripts/gate2_e0_preflight.py tests/test_gate2_e0_preflight.py
python -m unittest -v tests/test_gate2_e0_preflight.py
```

结果：9 项测试全部通过，覆盖 draft/ID/pair 校验、G0–G3 pending lane、M skip、pending request 保留、全 unknown/零计费、稳定 hash、公共 registry pending 状态、审计边界和 CLI JSON 输出。测试使用临时本地 manifest 与仓库 YAML，仅调用标准库及已有纯 Python 合同；未调用 API、LLM、仿真或正式实验。
