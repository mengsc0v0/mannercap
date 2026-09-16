# MannerCAP v2 Gate 1 开发请求清单入口规格

版本：`synthesis_v2.dev_manifest.v1`，2026-09-16  
负责人：C（合成方法）  
状态：开发清单生成入口初版；输出沿用 `request_manifest` 的 `draft`/`planned` 规划语义，尚未冻结正式 manifest。

## 1. 目的与边界

`scripts/gate1_build_dev_manifest.py` 把实际 Lift/Stack `initial_state.json` 文件接到已有的 `synthesis_v2.request_manifest` 合同，生成一个小型、仅用于 `development` split 的请求清单。脚本只读取 JSON、计算内容摘要、构造 `RequestDraft`、调用本地 manifest builder 并写 JSONL；不导入仿真环境，不调用模型或外部 API，不执行候选，不重试，也不运行正式实验。

入口要求调用者分别提供 Lift 和 Stack 的实际初态文件路径，不在仓库中硬编码服务器运行目录：

```text
python scripts/gate1_build_dev_manifest.py \
  --lift-initial-state /path/to/lift/initial_state.json \
  --stack-initial-state /path/to/stack/initial_state.json \
  --output /path/to/request_manifest.jsonl
```

## 2. 初态文件与稳定 `initial_state_id`

每个输入文件必须是 JSON object，并包含：

| 字段 | 约束 |
| --- | --- |
| `model_sha256` | 64 位 SHA-256 十六进制摘要；输出统一为小写 |
| `state` | JSON object；作为动态状态摘要的输入 |

脚本保留整个输入文档，并计算两个内容摘要：

- `state_sha256`：对 `state` object 做 UTF-8、`ensure_ascii=False`、按 key 排序、紧凑分隔符、禁止 NaN 的 canonical JSON 后取 SHA-256；
- `initial_state_hash`：对完整的初态 JSON 文档使用同一 canonical JSON 规则取 SHA-256。这样 controller/snapshot 等初态证据变化不会被只保留 `qpos` 的短摘要掩盖。

稳定 ID 的输入为：

```json
{
  "schema_version": "synthesis_v2.initial_state_ref.v1",
  "task": "lift 或 stack",
  "model_sha256": "...",
  "initial_state_hash": "..."
}
```

取该对象 SHA-256 前 24 位，格式为 `initial_<24 hex>`。文件路径和 JSON 空白不参与 ID；不同任务、模型摘要或初态内容会得到不同 ID。写入每条 manifest request 的 `initial_state` 中会附带 `initial_state_id`、`initial_state_hash` 和 `state_sha256`，同时在 `metadata` 中保留同名引用，方便 lineage 检查。输入原文的字段值仍被复制，派生注记不会替换模型摘要或状态内容。

## 3. 清单规模与任务范围

默认调用生成：

- Lift：`modifier_word`、`physical_attribute`、`stage_scope` 各一对；
- Stack：`modifier_word`、`physical_attribute`、`stage_scope` 各一对；
- Lift 和 Stack 各保留一个 `lifecycle_status=pending_calibration`、`incomplete=true` 的未完成规划项。

因此默认输出 14 条 JSONL request（12 条配对成员 + 2 条 pending/incomplete），全部为 `split=development`、`manifest_status=draft`、`request_status=planned`。`--exclude-pending` 可只写 12 条配对成员。任务白名单只包括 `lift` 和 `stack`；Wipe 等第三任务不被这个入口偷偷加入清单，仍由 task-space 合同标为 candidate/pending calibration。

初态内容会完整传入每个任务的 6 条配对成员。脚本从现有 task-space 注册表读取目标对象、阶段 ID 和 enabled 控制通道，因而不会自行发明阶段名或动作通道。属性配置和几何标识只使用待校准的 opaque registry 引用，不填质量、摩擦、力、速度或成功阈值。

## 4. 三类配对示例

配对通过已有 `request_manifest.make_pair` 与 `build_development_manifest` 校验，具体约束仍由已有合同解释：

| 类型 | 示例变化 | 保持不变的关键字段 |
| --- | --- | --- |
| `modifier_word` | `raw_language` 和其 `requested_constraints` 的原始声明 | task、scene、初态、目标、物理配置、stage scope、控制域、split、预算 |
| `physical_attribute` | opaque `configuration_ref` A/B | task、scene、中性语言、原始约束、stage scope、geometry key、split |
| `stage_scope` | 第一个阶段和最后一个阶段的 `stage_scope` | task、scene、初态、目标、物理配置、manner key、控制域、split、预算 |

每对成员均有两个不同的 request ID，pair 成员顺序由已有 builder 规范化。所有配对都发生在同一 `development` split；脚本不跨 split 配对、不把成员移动到其他 split，也不把“示例配对”当作已执行成功。

## 5. pending/incomplete 与原始约束

基础 `request_manifest` 合同在规划边界把所有请求写成 `request_status=planned`。因此未完成项不会伪造新的执行状态，而是在 `metadata` 中明确写出：

```json
{
  "lifecycle_status": "pending_calibration",
  "applicability": "pending_calibration",
  "incomplete": true,
  "requests_enabled": false,
  "reason": "stage scope, property configuration, and legal domain remain pending calibration"
}
```

这些项仍保留在清单分母中，`pair_id` 为 `null`，不参与可执行配对。若调用方使用 `--exclude-pending`，只是显式选择不输出这两条开发占位项；默认行为保留它们。

配对示例的 `requested_constraints` 由脚本直接作为 JSON object 传入 `RequestDraft`。已有合同在构造时深拷贝并原样写出；本入口不裁剪、放宽、排序语义、补阈值或改写约束。输出同时保留 `requested_constraints_hash`，便于比较原文。示例中的 `raw_marker` 只是结构测试标记，不代表物理属性。

## 6. 当前 Gate1 状态与剩余工作

已完成：

- 实际初态文件路径、`model_sha256`、完整初态摘要和稳定 `initial_state_id` 的本地读取入口；
- Lift/Stack 的 development-only 小清单与三类 pair 结构示例；
- pending/incomplete 保留、原始约束原样输出、稳定 JSONL writer；
- 缺失文件、非法 JSON、缺少/非法模型摘要、缺少 `state` 的明确拒绝。

尚未完成：

- 该输出仍是 `draft`，正式 request manifest 尚未由负责人冻结；示例配额、场景分母、属性配置和预算不是正式实验矩阵；
- `initial_state_id` 是内容 lineage 引用，不等于真实环境 reset 成功证明；尚缺初态恢复、controller/cache 一致性和逐状态重放校验；
- 尚未接入候选解析、执行器、资源/attempt ledger、最终审计或学生评测；没有由本脚本产生的仿真结果和成功率；
- Wipe 仍为 `candidate/pending_calibration`，不在本开发 manifest 的默认任务白名单中；
- 本版不冻结物理数值域、接触阈值或安全阈值。

## 7. 验证记录

执行命令：

```text
python -m py_compile scripts/gate1_build_dev_manifest.py tests/test_gate1_build_dev_manifest.py
python -m unittest -v tests/test_gate1_build_dev_manifest.py
```

结果：6 项测试全部通过，覆盖初态文件和 SHA-256 校验、规范化内容下的稳定 ID、模型/初态/任务变化导致的 ID 区分、Lift/Stack 与三类配对、pending/incomplete 保留、原始约束不改写、JSONL CLI 输出和零外部 API 调用。

随后使用 4090 上 `gate1_20260916_attempt03` 的真实 Lift/Stack `initial_state.json` 生成了开发清单：

```text
/data/gaochuanhao/mannercap_runs/gate1_dev_manifest_20260916/request_manifest.jsonl
```

输出包含 14 条记录、6 个配对、3 类 pair type、2 条 pending/incomplete，任务为 Lift/Stack，split 全部为 `development`，外部 API 调用为 0。文件 SHA-256 为 `3277a9898e11a9021dca28c520b40e634f8b3eab77a16e4ba486281118fc752d`，同时记录在 `dev_manifest_20260916.sha256`。该文件仍是 `draft`，没有冻结为正式实验清单。
