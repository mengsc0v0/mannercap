# StageCap — Method Draft
> AAAI 投稿用 Method 正文草稿（英文 + 中文双版本）
> 修订日期：2026-07-06（v2，按审稿意见全面修订）

---

## English Version

### 4. StageCap

#### 4.1 Overview

StageCap is a stage-wise Code-as-Policy data engine for generating successful robot manipulation trajectories in simulation. It consists of five components: (1) pattern-prior stage decomposition, (2) per-stage executable code generation, (3) privileged readonly scene snapshot collection, (4) self-reporting stage verification with ground-truth override, and (5) failure-frame-driven local repair.

Given a language instruction $l$ and a simulator $\mathcal{E}$ initialized with seed $s$, StageCap classifies the task into a manipulation family and produces a stage plan $P = \{g_1, \ldots, g_K\}$ using rule-based pattern priors distilled from a corpus of successful Code-as-Policy programs (CAP-35; see §4.2). For each stage $g_k$ in sequence, a language model generates a self-contained inline Python script $\pi_k$ that calls audited robot motion primitives and readonly simulator APIs, then self-reports its execution outcome as a structured `STAGE_STATUS` signal. The runner parses this signal and applies one of five control decisions: advance to the next stage, retry the current stage with failure context, roll back to an earlier grasp stage within the same episode, regenerate the current stage code, or terminate the trial. When a stage fails, the system captures a diagnostic frame at the stage execution boundary via online rendering (KITE), annotates it with stage evidence, and injects it into the next repair prompt. The complete generation loop is given in Algorithm 1.

---

#### 4.2 Stage Decomposition and Stage Interface

**Family-aware pattern priors.** Stage decomposition in StageCap is *rule-based*: the planner classifies the natural language instruction $l$ into a task family and instantiates a stage sequence from pre-defined templates. The templates were distilled from CAP-35 — a corpus of 35 manually verified successful Code-as-Policy programs for LIBERO tasks. These 35 programs collectively span the 9 manipulation families that cover LIBERO's structural task space: `PICK_PLACE`, `CONTAINER_PLACE`, `SURFACE_PLACE`, `SPATIAL_RELATION`, `DRAWER`, `STOVE_BUTTON`, `RACK_OR_CADDY`, `MULTI_OBJECT`, and `PUSH_REPAIR`. The templates encode abstract execution schemas — what the success corpus taught us about reliable manipulation loops (e.g., verify a lift before transport, release low before retreat, test drawer contact before pulling far) — rather than retrieving the programs themselves. The 35-program corpus was grown incrementally until every task family in LIBERO had at least three independently verified representatives; we report an ablation on template count in the experiments section.

Rule-based decomposition offers two concrete advantages over LLM-driven decomposition. First, it produces deterministic, auditable stage plans, which means any failure is attributable to within-stage code generation rather than to decomposition errors — a critical property for controlled evaluation. Second, it decouples stage structure reliability from LLM stochasticity: the language model is responsible solely for *code generation within each stage*, not for stage sequencing. The current scope of CAP-35 is bounded to LIBERO's nine task families; extending to a new domain requires curating representative programs for the new families and rerunning the incremental verification process.

The decomposition is family-aware: `DRAWER` tasks automatically insert open/close sub-stages; `MULTI_OBJECT` tasks order objects by occlusion risk; `PUSH_REPAIR` tasks route execution through a dedicated push-region path.

**Stage interface.** Each stage $g_k$ is represented as a `StagePlan` dataclass (Table 1). There is no explicit precondition field; stage precedence is enforced implicitly through the stage ordering and the `STAGE_STATUS` decision chain. The `checks` field specifies geometric or contact success criteria that the generated code should evaluate during execution and self-report. The output of decomposition is a `TaskPlan` containing the stage list and a `notes` field injected into the LLM system prompt as global execution constraints.

**Table 1: Stage interface fields (aligned with `StagePlan` dataclass)**

| Field | Example | Purpose |
|---|---|---|
| `stage_id` | `obj1_bowl_pick_lift_verify` | Unique stage key; validated against `STAGE_STATUS` output |
| `intent` | `Pick and lift the bowl above the table` | Execution goal for code generation |
| `family` | `PICK_PLACE` | Task family; determines the skill allowlist |
| `allowed_primitives` | `goto_pose, close_gripper, ...` | Permitted low-level motion API calls |
| `checks` | `StageCheck(lift_delta ≥ 0.05)` | Geometric/contact success criteria included in prompt for self-verification |
| `repair_hints` | `"Retry with a side grasp if flat pick fails"` | Injected into repair prompts to guide LLM correction |
| `max_attempts` | `2` | Per-stage retry budget |

---

#### 4.3 Stage-wise Code-as-Policy Generation

**One stage at a time.** For each stage $g_k$, the LLM generates exactly one self-contained inline Python script. The script executes directly in the simulator sandbox and must terminate with a mandatory output line:

```python
print("STAGE_STATUS " + json.dumps({"stage_id": ..., "status": ..., "evidence": ..., "repair": ...}))
```

If this line is absent from stdout, the runner treats the outcome as `uncertain`. A shortened representative script is shown below; the full example appears in Appendix A.

```python
# (... motion and observation calls ...)
status = "satisfied" if (not held_after and relation_ok) else "needs_repair"
evidence = f"held_after={held_after}, relation_satisfied={relation_ok}"
repair = "" if status == "satisfied" else "Check approach/release_z or retry transport."
print("STAGE_STATUS " + json.dumps(
    {"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair},
    default=str,
))
```

**API policy sandboxing.** Each script executes under a `CleanApiPolicy` restricting the callable symbol set to three layers: (1) low-level motion primitives (`goto_pose`, `open_gripper`, `move_linear_delta`, ...); (2) readonly simulator observations (`get_object_pose`, `get_bddl_goal_diagnostics`, `get_contact_diagnostics`, ...); and (3) audited high-level skills (`verified_lift_pick_object`, `guarded_transport_held_object_to_target`, ...). Symbols that bypass simulation state — `set_object_pose`, `eval`, `globals()`, `.data.qpos` — are explicitly forbidden. A preflight checker intercepts forbidden symbols, undefined variables, and illegal imports before the script enters the sandbox. Any forbidden-symbol violation triggers an immediate `FAIL_TASK` decision without execution.

**API policy tiers.** The runner selects among three configurations: *restricted-API mode* (`CAPX_AGENT0`; baseline, no goal diagnostics), *privileged-API mode* (`DEFAULT_CLEAN`; adds BDDL goal predicates and contact APIs), and *visual-API mode* (`SAM3_CLEAN`; additionally exposes visual segmentation interfaces). All three tiers execute through the same code path, differing only in the LLM-accessible API whitelist. This enables controlled ablation of privileged observations without altering any other system component.

Unlike end-to-end neural policies, StageCap's per-stage code representation is programmatic and auditable: a failed placement stage corresponds to a concrete, self-contained script that can be re-prompted with failure evidence rather than requiring full policy retraining.

---

#### 4.4 Stage Verification and Evidence Override

**Two-layer verification design.** StageCap's verification is intentionally layered, with each layer serving a distinct role. The *first layer* is self-reporting: the generated script reads privileged simulator states — object poses, contact diagnostics, BDDL goal predicates, drawer progress — assesses whether the stage's success criteria are satisfied, and emits a structured `STAGE_STATUS` JSON. Self-reporting provides fine-grained diagnostic content: the `evidence` field records the exact geometric measurements that caused the failure, and the `repair` field encodes a stage-specific correction hint. This information is inexpressible in a binary ground-truth oracle and is the primary input for constructing the repair prompt.

The *second layer* is a rule-based ground-truth override (`_rule_override_evidence()`) applied after parsing the self-reported status. It handles two systematic failure modes in LLM self-reports: (1) **over-conservatism** — if the BDDL goal is actually satisfied, a conservatively reported `not_satisfied` is overridden to `satisfied`, preventing unnecessary rollbacks; (2) **over-optimism** — if the object is no longer held but the placement relation is unmet, the release stage is forced to `needs_repair` regardless of the script's report, preventing wasted retries in an invalid state. The GT override does not produce repair hints; it only corrects the binary decision. The two layers are thus complementary: self-reporting supplies the diagnostic content that drives repair, while the GT override enforces data quality at system boundaries.

**Control decisions.** The runner maps the four-valued status (`satisfied` / `not_satisfied` / `needs_repair` / `uncertain`) to five control decisions:

- **PROCEED**: stage satisfied; advance stage pointer.
- **RETRY_STAGE**: reuse current code with appended failure context ($\text{attempt}_k < g_k.\texttt{max\_attempts}$).
- **REPLAN_STAGE**: invalidate current code; regenerate for this stage.
- **REPLAN_SUFFIX**: failure invalidates downstream preconditions; roll back the stage pointer to the nearest prior pick-lift stage *within the same episode*, without resetting the simulator.
- **FAIL_TASK**: forbidden-symbol violation detected by preflight checker, or total attempt budget exhausted; terminate trial immediately.

The output trajectory contains only $(o, a)$ pairs; privileged states are a generation-time resource internal to the data engine and are stripped before the trajectory is saved.

---

#### 4.5 Failure-Frame-Driven Local Repair

**Online KITE.** When a stage fails, the runner immediately calls `env.render()` to capture the simulator frame at the stage execution boundary — before any subsequent action can alter the scene state. The frame is annotated with the stage identifier, attempt index, `STAGE_STATUS` evidence, and repair hint, then serialized as base64 and injected into the LLM multimodal prompt for the next repair attempt. We refer to this mechanism as *online KITE (Keyframe-based In-Trial Evidence)*, in contrast to offline keyframe extraction applied post-hoc to recorded video. Online KITE requires an image-capable language model; in the text-only ablation condition, KITE image injection is disabled and only the textual `STAGE_STATUS` evidence is passed to the repair prompt.

Capturing frames at stage boundaries rather than at episode termination offers three advantages: (1) stage boundaries correspond to a single, well-defined action intent (grasp, transport, release, or drawer-open), making failure attribution unambiguous; (2) the frame is temporally aligned with the script's stdout/stderr output, enabling joint visual-textual diagnosis; (3) no video recording infrastructure is required.

**Repair context construction.** The repair prompt is assembled by `build_stage_code_prompt()` and contains: the full stage plan (all stages listed for context, but the LLM generates code for the failed stage only), the most recent $N{=}8$ stage execution records, the current scene snapshot (object poses, end-effector pose, contact diagnostics), and KITE image inputs. The window of $N{=}8$ records balances repair context richness against token budget; preliminary runs showed diminishing returns beyond 8 records while context beyond 10 caused token-limit truncation on the models evaluated. Formally, the repair context for stage $k$ at attempt $t$ is:

$$c_k^{(t)} = \text{merge}\!\left(\texttt{prev\_records},\; F_k^{(t-1)},\; \texttt{snapshot}_k^{(t)},\; \texttt{KITE\_images}^{(t-1)}\right)$$

$$\pi_k^{(t)} = \text{LLM}\!\left(\text{stage\_plan},\; g_k,\; c_k^{(t)},\; \texttt{api\_policy}\right)$$

where $F_k^{(t-1)}$ is the annotated failure frame from the previous attempt and $\texttt{snapshot}_k^{(t)}$ is the current readonly scene state.

**Stateless stage scripts.** Each script is contractually required to be self-contained: it must define every helper it calls and must not rely on variables from other stages. The only shared state across stages is the physical simulator itself (joint angles, object poses, contact state). This property guarantees that local repair is non-regressive: patching the code for stage $k$ cannot affect the variables or logic of stages $1, \ldots, k{-}1$ or $k{+}1, \ldots, K$.

**Two-level repair.** StageCap implements repair at two temporal granularities. *Within-trial repair* (the primary loop) executes RETRY / REPLAN\_STAGE / REPLAN\_SUFFIX decisions within the same simulated episode under a total attempt budget (`max_total_stage_attempts`, default 10). *Cross-trial repair* packages failure context from a completed trial into `replan_context` and injects it into the global system prompt of the subsequent trial (`feedback_rounds` controls the number of re-runs). Within-trial repair operates entirely inside a single simulated episode and is the primary mechanism evaluated in this work. Cross-trial repair operates across episode boundaries by injecting prior failure context into the global system prompt of subsequent trials. The two mechanisms are independently configurable and are reported separately in ablation experiments.

---

#### 4.6 Scene Diversity

Trial seeds control initial object poses, layouts, and instance randomization via `env.reset(seed=s)`. Running StageCap across $T$ seeds exposes the verification and repair loop to the full distribution of scene-induced execution failures — grasps that succeed in one layout may fail in another due to occlusion, reach limits, or pose variation — without any task-specific manual adjustment. Diversity seeds are drawn uniformly and are held fixed across all system configurations in comparative experiments to ensure fair evaluation.

---

**Algorithm 1: StageCap Trajectory Generation**

```
Input : instruction l, simulator E, trial seed s, budget B
        stage plan P ← Decompose(l)            ▷ rule-based, scene-independent
Output: trajectory τ or failure

E.reset(seed=s)
for attempt = 1, …, B do
    k ← P.current_stage                        ▷ stage pointer; no episode reset
    if k is None then break                     ▷ all stages passed

    snapshot ← collect_readonly_snapshot(E, k)
    π_k ← LLM(l, P, k, prev_records, kite_frames, snapshot)

    forbidden ← preflight_check(π_k, api_policy)  ▷ check before sandbox entry
    if forbidden then
        decision ← FAIL_TASK; break            ▷ immediate termination

    stdout, stderr ← E.execute(π_k)
    ev ← parse_STAGE_STATUS(stdout)
    ev ← rule_override(ev, E)                  ▷ ground-truth arbiter

    if ev.status == satisfied then
        decision ← PROCEED
    else if attempt_k < k.max_attempts then
        decision ← RETRY_STAGE
    else if failure_invalidates_suffix(k, ev) then
        decision ← REPLAN_SUFFIX               ▷ roll back within episode, no env reset
    else
        decision ← REPLAN_STAGE

    if total_attempts ≥ B then
        decision ← FAIL_TASK; break            ▷ budget exhausted

    if online_kite_enabled and ev.status ≠ satisfied then
        frame ← E.render()                     ▷ capture frame at stage boundary
        kite_frames[k] ← annotate(frame, ev)

    update_stage_pointer(P, decision)           ▷ REPLAN_SUFFIX rolls back to pick-lift

if all stages passed then
    save trajectory τ
```

---
---

## 中文版本

### 4. StageCap

#### 4.1 总览

StageCap 是一种面向仿真机器人操控轨迹生成的阶段化 Code-as-Policy 数据引擎，由五个模块组成：（1）基于 pattern prior 的阶段分解；（2）逐阶段可执行代码生成；（3）特权只读场景快照采集；（4）自汇报阶段验证与仿真真值覆盖；（5）失败帧驱动的局部修复。

给定语言指令 $l$ 和以 seed $s$ 初始化的仿真器 $\mathcal{E}$，StageCap 首先将任务分类到对应的操控家族，利用从成功 Code-as-Policy 程序语料（CAP-35，见 §4.2）中蒸馏的规则化 pattern prior 生成阶段计划 $P = \{g_1, \ldots, g_K\}$。对每个阶段 $g_k$，语言模型生成一段自包含的内联 Python 脚本 $\pi_k$，调用经审计的机器人动作 primitive 和只读仿真状态 API，并以结构化 `STAGE_STATUS` 信号自我汇报执行结果。Runner 解析该信号后执行五种控制决策之一：推进到下一阶段、携带失败上下文重试当前阶段、在同一 episode 内回滚到更早的抓取阶段、重新生成当前阶段代码或终止任务。阶段失败时，系统在阶段执行边界处通过在线 KITE 渲染立即捕获诊断帧，标注阶段证据后注入下一次修复 prompt。完整生成循环见算法 1。

---

#### 4.2 阶段分解与阶段接口

**家族感知 pattern prior。** StageCap 的阶段分解是**规则化**的：规划器将自然语言指令 $l$ 分类到任务家族，并从预定义模板中实例化阶段序列。模板来自 CAP-35——35 个人工验证通过的 LIBERO 任务成功 Code-as-Policy 程序语料。这 35 个程序共覆盖 9 个操控家族，涵盖了 LIBERO 任务的结构化全域：`PICK_PLACE`、`CONTAINER_PLACE`、`SURFACE_PLACE`、`SPATIAL_RELATION`、`DRAWER`、`STOVE_BUTTON`、`RACK_OR_CADDY`、`MULTI_OBJECT`、`PUSH_REPAIR`。模板编码的是抽象执行 schema（例如"搬运前先验证拿起"、"低位释放后再退出"、"抽屉先试接触再大幅拉开"），而不是直接检索原始程序。CAP-35 语料采用增量扩充策略——每个任务家族至少有三个独立验证的代表程序后方停止扩充；我们在实验节对模板数量的影响进行了消融分析。

规则化分解具有两项相对 LLM 分解的具体优势。**第一**，产生确定性、可审查的阶段计划——任何失败都可归因于阶段内代码生成，而非阶段序列划分错误，是控制变量实验的必要前提。**第二**，将阶段结构的可靠性与 LLM 的随机性解耦——语言模型仅负责**各阶段内部的代码生成**，不参与阶段序列决定。CAP-35 的当前覆盖范围限于 LIBERO 的九个任务家族；扩展到新领域需为新任务家族补充代表程序并重新执行增量验证流程。

分解具有家族感知性：`DRAWER` 类任务自动插入 open/close 子阶段；`MULTI_OBJECT` 类任务按遮挡风险对对象排序；`PUSH_REPAIR` 类任务走独立的 push-region 路径。

**阶段接口。** 每个阶段 $g_k$ 表示为 `StagePlan` 数据类（表 1）。系统中不存在独立的前提条件字段——前提约束通过阶段顺序与 `STAGE_STATUS` 决策链隐式保证。`checks` 字段指定几何/接触成功条件，随 prompt 提供给生成代码用于自验证。分解输出为 `TaskPlan`，包含阶段列表和 `notes` 字段（全局执行约束备注，注入 LLM system prompt）。

**表 1：阶段接口字段（与 `StagePlan` 数据类对齐）**

| 字段 | 示例 | 作用 |
|---|---|---|
| `stage_id` | `obj1_bowl_pick_lift_verify` | 唯一阶段标识，用于校验 `STAGE_STATUS` 输出 |
| `intent` | `Pick and lift the bowl above the table` | 代码生成时的执行意图描述 |
| `family` | `PICK_PLACE` | 任务家族，决定 skill 白名单 |
| `allowed_primitives` | `goto_pose, close_gripper, ...` | 允许调用的底层动作 API |
| `checks` | `StageCheck(lift_delta ≥ 0.05)` | 阶段成功的几何/接触条件，注入 prompt 供生成代码自验 |
| `repair_hints` | `"Retry with a side grasp if flat pick fails"` | 注入修复 prompt，引导 LLM 修正方向 |
| `max_attempts` | `2` | 该阶段的重试次数上限 |

---

#### 4.3 逐阶段 Code-as-Policy 生成

**逐阶段生成。** 对每个阶段 $g_k$，语言模型仅生成一段自包含的内联 Python 脚本。脚本直接在仿真沙箱中执行（非函数定义），并必须以如下行结尾：

```python
print("STAGE_STATUS " + json.dumps({"stage_id": ..., "status": ..., "evidence": ..., "repair": ...}))
```

若 stdout 中缺少该行，runner 将该阶段标记为 `uncertain`。完整示例脚本见附录 A；以下仅展示核心输出部分：

```python
# (... 动作与观测调用 ...)
status = "satisfied" if (not held_after and relation_ok) else "needs_repair"
evidence = f"held_after={held_after}, relation_satisfied={relation_ok}"
repair = "" if status == "satisfied" else "Check approach/release_z or retry transport."
print("STAGE_STATUS " + json.dumps(
    {"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair},
    default=str,
))
```

**API 策略沙箱。** 每段脚本在 `CleanApiPolicy` 约束下执行，可调用符号集分为三层：（1）底层动作 primitive（`goto_pose`、`open_gripper`、`move_linear_delta` 等）；（2）只读仿真状态 API（`get_object_pose`、`get_bddl_goal_diagnostics`、`get_contact_diagnostics` 等）；（3）经审计的高层 skill。可绕过仿真状态的符号（`set_object_pose`、`eval`、`globals()` 等）被明确禁止。Preflight 检查器在脚本进入沙箱**前**拦截违禁符号、未定义变量及非法 import。一旦检测到违禁符号，立即触发 `FAIL_TASK` 决策，不执行脚本。

**API 策略层级。** Runner 从三套配置中选择：**受限 API 模式**（`CAPX_AGENT0`，baseline，不含目标诊断接口）、**特权 API 模式**（`DEFAULT_CLEAN`，开放 BDDL 目标谓词与接触 API）、**视觉 API 模式**（`SAM3_CLEAN`，额外开放视觉分割接口）。三套策略共享完全相同的执行路径，仅 LLM 可调用的 API 白名单不同，可直接用于 privileged 模式的消融分析。

与端到端神经策略不同，StageCap 的逐阶段代码表示是可执行、可审查的：某个放置阶段的失败对应一段具体的自包含脚本，可携带失败证据重新提示，而无需重新训练整个策略。

---

#### 4.4 阶段验证与真值覆盖

**两层验证设计。** StageCap 的验证机制有意分为两层，各层职责明确。**第一层**是自汇报：生成的脚本自行读取特权仿真状态（物体位姿、接触诊断、BDDL 目标谓词、抽屉进度），评估当前阶段是否满足成功条件，并输出结构化 `STAGE_STATUS` JSON。自汇报提供细粒度的诊断内容：`evidence` 字段记录导致失败的具体几何测量值，`repair` 字段编码阶段特定的修正建议。这些信息是纯二值真值 oracle 无法提供的，也是构建修复 prompt 的主要输入。

**第二层**是规则化真值覆盖层（`_rule_override_evidence()`），在解析自汇报状态后应用，处理 LLM 自汇报中的两类系统性偏差：（1）**过于保守**——若 BDDL 目标实际已满足，则将 `not_satisfied` 强制覆盖为 `satisfied`，避免不必要的回滚；（2）**过于乐观**——若物体已不在可抓持状态但位置关系未满足，则将释放阶段强制标为 `needs_repair` 以触发回滚。GT 覆盖层不产生修复建议，只纠正二值决策。两层因而互补：自汇报驱动修复内容，GT 覆盖保障系统边界处的数据质量。

输出轨迹仅包含 $(o, a)$ 序列；特权状态是数据引擎生成阶段的内部资源，在轨迹保存前被剥离。

**控制决策。** Runner 将四值状态（`satisfied` / `not_satisfied` / `needs_repair` / `uncertain`）映射到五种控制决策：

- **PROCEED**：阶段通过，推进阶段指针；
- **RETRY_STAGE**：保留当前代码，追加失败上下文重试（$\text{attempt}_k < g_k.\texttt{max\_attempts}$）；
- **REPLAN_STAGE**：放弃当前代码，为该阶段重新生成；
- **REPLAN_SUFFIX**：失败使后续阶段前提条件无效，在**同一 episode 内**将阶段指针回滚到最近的 pick-lift 阶段，不重置仿真环境；
- **FAIL_TASK**：Preflight 检测到违禁符号（最高优先级，立即触发），或总预算耗尽；终止当前 trial。

---

#### 4.5 失败帧驱动的局部修复

**在线 KITE。** 阶段失败时，runner 立即调用 `env.render()` 在阶段执行边界捕获当前仿真帧——此时后续动作尚未改变场景状态。帧图像经标注（阶段 ID、尝试次数、`STAGE_STATUS` 证据、修复提示）后序列化为 base64，注入 LLM 多模态 repair prompt。我们称此机制为**在线 KITE（Keyframe-based In-Trial Evidence）**，以区别于对已录制视频的后处理关键帧提取。在线 KITE 需要支持图像输入的语言模型；在纯文本消融条件下，KITE 图像注入被禁用，仅传递文本 `STAGE_STATUS` 证据。

在阶段边界而非 episode 终止时捕帧有三重优势：（1）阶段边界对应单一明确的动作意图（抓取、运输、释放、开抽屉），失败归因清晰；（2）帧图像与脚本 stdout/stderr 精确对齐，支持视觉—文本联合诊断；（3）无需视频录制基础设施。

**修复上下文构建。** 修复 prompt 由 `build_stage_code_prompt()` 构建，包含：完整阶段计划（LLM 仅为当前失败阶段生成代码）、最近 $N{=}8$ 次阶段执行记录、当前场景快照（物体位姿、末端执行器位姿、接触诊断）及 KITE 图像。窗口大小 $N{=}8$ 在修复上下文丰富度与 token 预算之间取得平衡：预实验表明超过 8 条记录收益递减，超过 10 条会触发所评测模型的 token 截断。阶段 $k$ 第 $t$ 次尝试的修复上下文形式化为：

$$c_k^{(t)} = \operatorname{merge}\!\left(\texttt{prev\_records},\; F_k^{(t-1)},\; \texttt{snapshot}_k^{(t)},\; \texttt{KITE\_images}^{(t-1)}\right)$$

$$\pi_k^{(t)} = \operatorname{LLM}\!\left(\text{stage\_plan},\; g_k,\; c_k^{(t)},\; \texttt{api\_policy}\right)$$

其中 $F_k^{(t-1)}$ 为上一次尝试的标注失败帧，$\texttt{snapshot}_k^{(t)}$ 为当前只读场景状态。

**Stage 脚本无状态性。** 每段脚本在合约层面要求完全自包含：必须自行定义所调用的所有辅助函数，不得依赖其他阶段的变量。各阶段脚本之间唯一的共享状态是物理仿真器本身（关节角、物体位姿、接触状态）。这一属性保证了局部修复的无回归性：对阶段 $k$ 代码的修改不会影响阶段 $1, \ldots, k{-}1$ 或 $k{+}1, \ldots, K$ 的逻辑。

**两级修复结构。** StageCap 在两个时间粒度上实施修复。**Trial 内修复**（核心循环）在同一仿真 episode 内执行 RETRY / REPLAN\_STAGE / REPLAN\_SUFFIX 决策，总尝试次数受 `max_total_stage_attempts`（默认 10）约束。**Trial 间修复**将失败 trial 的上下文通过 `_build_replan_context()` 打包为 `replan_context`，注入下一轮 trial 的全局 system prompt（`feedback_rounds` 控制重跑轮数）。Trial 内修复完全在单个仿真 episode 内运行，是本文主要评测的机制。Trial 间修复跨 episode 边界，将先前失败上下文注入后续 trial 的全局 system prompt。两种机制可独立配置，消融实验中分别报告。

---

#### 4.6 场景多样性

Trial seed 通过 `env.reset(seed=s)` 控制初始物体位姿、布局及实例随机化。在 $T$ 个 seed 上运行 StageCap，使阶段验证与修复循环暴露于场景诱发执行失败的完整分布——在某一布局下成功的抓取，可能因遮挡、可达范围限制或位姿变化在另一布局下失败——无需任何任务特定的人工干预。多样性 seed 从均匀分布中抽取，并在所有对比配置间固定，以保证评测公平性。

---

**算法 1：StageCap 轨迹生成**

```
输入：指令 l，仿真器 E，trial seed s，预算 B
     阶段计划 P ← Decompose(l)              ▷ 规则化，不依赖场景图像
输出：成功轨迹 τ 或失败

E.reset(seed=s)
for attempt = 1, …, B do
    k ← P.current_stage                     ▷ 阶段指针，非 episode 重置
    if k is None then break                  ▷ 所有阶段通过

    snapshot ← collect_readonly_snapshot(E, k)
    π_k ← LLM(l, P, k, prev_records, kite_frames, snapshot)

    forbidden ← preflight_check(π_k, api_policy)  ▷ 沙箱前拦截违禁符号
    if forbidden then
        decision ← FAIL_TASK; break         ▷ 立即终止，最高优先级

    stdout, stderr ← E.execute(π_k)
    ev ← parse_STAGE_STATUS(stdout)
    ev ← rule_override(ev, E)               ▷ 仿真真值最终裁决

    if ev.status == satisfied then
        decision ← PROCEED
    else if attempt_k < k.max_attempts then
        decision ← RETRY_STAGE
    else if failure_invalidates_suffix(k, ev) then
        decision ← REPLAN_SUFFIX             ▷ episode 内回滚，不重置环境
    else
        decision ← REPLAN_STAGE

    if total_attempts ≥ B then
        decision ← FAIL_TASK; break         ▷ 总预算耗尽

    if online_kite_enabled and ev.status ≠ satisfied then
        frame ← E.render()                   ▷ 阶段边界实时捕帧
        kite_frames[k] ← annotate(frame, ev)

    update_stage_pointer(P, decision)        ▷ REPLAN_SUFFIX 可回滚到 pick-lift

if all stages passed then
    save trajectory τ
```

---

> **LaTeX 排版说明**
> - 行内公式用 `$...$`，独立公式用 `equation` 或 `align` 环境。
> - Algorithm 1 建议使用 `algorithm2e` 或 `algorithmicx` 包排版。
> - Table 1 放 4.2 节末，Algorithm 1 作为独立浮动体置于 4.6 节后。
> - 数值占位符（$T$、消融对比数字等）在实验完成后填入实际数字。
> - 完整 low-release 脚本示例置附录 A（已从正文中移出）。
