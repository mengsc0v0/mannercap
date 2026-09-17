# Gate 1：动作导出—再导入 round-trip 预检

日期：2026-09-16  
状态：结构预检入口；控制器执行 `pending`，物理判定 `unknown`  
范围：Stage-CAP JSONL → 版本化 JSONL bundle → 再导入 → 逐字段一致性

## 1. 当前真实动作链

当前仓库的实际字段链如下。它们不能按字段名或宽度互换。

| 链路位置 | 实际字段 | 维度 | 语义 | 当前证据边界 |
| --- | --- | ---: | --- | --- |
| LIBERO recorder 原始 transition | `action`、`action.raw_joint_gripper` | 8 | 7 维 joint-delta + 1 维 gripper command；`-1=open,+1=close` | 可作为低层命令候选；仍需同一 controller 的逐步 replay |
| LIBERO recorder 的旧 LeRobot 候选 | `lerobot_action_candidate` | 7 | EEF position 差分 3 + axis-angle 状态差分 3 + raw gripper 1 | 源码标记 `validated_as_ground_truth=false`，不能当 8-D controller action |
| `convert_stagecap_steps_to_lerobot.py` | 输出 `action` | 7 | 默认把 `lerobot_action_candidate` 传入 LeRobot `dataset.add_frame` | 只做 shape/image 转换；不验证低层 controller 语义 |
| `convert_stagecap_manifest_to_hfvla_parquet.py` | 输出 `action` | 7 | 采样后对相邻保留行重新计算 EEF state 差分 | 可能重造时间和 action；不能作为 raw action 的无损 round-trip |
| LIBERO 执行入口 | `handle.step(action)` | 8 | 当前 wrapper 用关节目标差乘 `control_freq` 构造 joint-delta 输入 | 本预检不调用执行器 |

代码依据：

* `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:871-920`
  保存 8-D raw action 和 7-D EEF candidate；
* `stagecap_robosuite_libero_repro_20260714/capx/envs/simulators/libero.py:264-305,329-406`
  构造并发送 LIBERO joint-delta；
* `stagecap_robosuite_libero_repro_20260714/scripts/convert_stagecap_steps_to_lerobot.py:18-57,75-136,192-212`
  定义 8-D state / 7-D action 的 LeRobot 导出；
* `stagecap_robosuite_libero_repro_20260714/scripts/convert_stagecap_manifest_to_hfvla_parquet.py:36-57,377-460`
  定义 parquet 字段、按源行号抽样、重造 timestamp 和重算 action；
* `stagecap_robosuite_libero_repro_20260714/capx/envs/trial.py:475-487`
  写出旧 action metadata 和 candidate warning。

当前仓库没有已提交的可直接读取 `steps.jsonl` 或 parquet 轨迹产物；本入口
接受真实运行产生的 JSON/JSONL，也接受 Stage-CAP manifest 中的 `steps_path`，
但不会用测试 fixture 冒充真实 30 条轨迹。parquet 由标准库入口显式报
`unsupported_format`，需要依赖感知的单独 adapter，不能猜作 8-D 控制器输入。

## 2. 版本化预检入口

新增纯标准库模块
[`synthesis_v2/action_roundtrip.py`](../../synthesis_v2/action_roundtrip.py)，预检
版本为 `synthesis_v2.action_round_trip_preflight.v1`，bundle 行版本为
`synthesis_v2.action_round_trip_record.v1`。

典型调用：

```text
PYTHONPATH=. python -m synthesis_v2.action_roundtrip \
  --source /path/to/steps.jsonl \
  --output /path/to/action_round_trip_out
```

输出两个文件：

```text
action_round_trip_out/action_round_trip_records.jsonl
action_round_trip_out/action_round_trip_manifest.json
```

入口 `build_round_trip_manifest(source, output, count=30)` 的行为是：

1. 读取 JSON/JSONL/NDJSON；Stage-CAP manifest 行会展开其 `steps_path`，不把
   manifest metadata 注入 step；
2. 按源顺序取最多 30 条，绝不复制最后一条来凑数；少于 30 条时写出
   `status=preflight_incomplete` 和 `count_satisfied=false`；
3. 每行原样放入版本化 JSONL bundle，另写 canonical JSON SHA-256 和动作字段
   分类；不重采样、不重造 timestamp、不重算 action；
4. 重新读取 bundle，验证行 schema 和 hash；再递归比较 source/imported 的每个
   JSON leaf field，报告缺失、意外、类型、长度和值差异及字段路径；
5. 写出 action interface 分类、缺 `dt` 计数和 controller 状态。

## 3. 动作分类规则

`inspect_action_record` 对一行中的所有候选字段分别分类：

| 字段/形状 | interface id | semantic | `verified` |
| --- | --- | --- | --- |
| `action.raw_joint_gripper` 或 8-D `action` | `libero_low_level_controller` | `joint_delta_plus_gripper` | `false` |
| `lerobot_action_candidate` 或仅有的 7-D `action` | `lerobot_export_candidate` | `eef_position_axis_angle_state_difference_plus_raw_gripper` | `false` |
| 其他维度/非数值 | `unknown` | `unresolved_action_semantics` | `false` |

同时存在 raw `action` 与 `action.raw_joint_gripper` 时，入口检查二者逐值一致；
不一致会记录 `raw_action_alias_mismatch`。7-D 候选始终保留为独立接口，不能
因目标列名字叫 `action` 就提升为 8-D LIBERO 低层命令。

## 4. 控制器状态边界

该入口没有 simulator、LeRobot、PyArrow、policy 或外部 API 依赖，也没有执行
`handle.step`。所以每份 manifest 固定写出：

```json
{
  "controller_execution": {
    "status": "pending",
    "decision": "unknown",
    "replay_verified": false
  },
  "physical_replay": {
    "status": "pending",
    "decision": "unknown",
    "verified_count": 0
  }
}
```

`field_consistency.valid=true` 只表示 JSONL export/import 保留了字段值。它不表示
动作被学生策略输出、不表示控制器接受了动作，也不表示物理状态逐步一致。
后续真实 replay 必须另行记录初态、controller/RNG snapshot、实测 `dt`、目标和
实际关节/末端/夹爪误差；本入口不会填充这些缺失证据。

## 5. 测试和当前结论

新增 [`tests/test_v2_action_roundtrip.py`](../../tests/test_v2_action_roundtrip.py)，
覆盖：

* 8-D raw low-level 和 7-D EEF candidate 分类；
* JSONL export/import 的全字段递归一致性；
* 嵌套字段修改、record hash 篡改和 parquet 输入的显式失败；
* 默认 30 条目标、短输入不复制补齐、精确 30 条预检；
* Stage-CAP manifest → `steps_path` 展开且不注入 metadata；
* controller `pending` / decision `unknown` / replay 未验证。

本地纯标准库验证：

```text
python -m unittest -v tests/test_v2_action_roundtrip.py tests/test_v2_action_interface.py
python -m py_compile synthesis_v2/action_roundtrip.py tests/test_v2_action_roundtrip.py
```

21 项测试通过。当前结果是结构预检入口已经具备，真实 30 条记录的数量、
LeRobot parquet 适配和学生控制器 replay 证据仍待后续运行；本报告不声称
30 条物理回放已完成。
