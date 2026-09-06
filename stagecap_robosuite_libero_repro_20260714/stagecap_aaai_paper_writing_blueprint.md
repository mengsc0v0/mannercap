# StageCap AAAI 论文写作蓝图

> 用途：这是给后续正式写 AAAI 论文用的结构化写作指南，不是最终论文正文。它按照 AI/robotics conference paper 的常规结构组织，重点服务于 StageCap 的论文叙事：从现有仿真数据引擎的缺陷出发，提出面向 VLA 训练轨迹构建的阶段化 Code-as-Policy 数据生产 agent。

## 0. 推荐论文定位

### 0.1 一句话定位

StageCap is a stage-wise Code-as-Policy data engine that generates successful simulation trajectories for vision-language-action policy training by validating each task stage, localizing failure frames, and repairing only the failed stage policy code.

中文意思是：StageCap 不是单纯做机器人控制，也不是单纯做 LLM 任务生成，而是把 Code-as-Policy 变成仿真轨迹数据生产引擎。它的重点是高成功率、可诊断、可局部修复地生成 VLA 训练样本。

### 0.2 推荐标题方向

可以从三个风格里选：

1. **StageCap: Stage-wise Code-as-Policy for Simulation Trajectory Generation**
   - 最稳，强调方法和任务。

2. **Failure-Aware Code-as-Policy Data Generation for Robotic Manipulation**
   - 更突出失败帧复盘和局部修复。

3. **From Code-as-Policy to Data Engine: Stage-wise Repair for Scalable Robot Trajectory Generation**
   - 叙事性更强，但标题稍长。

如果实验足够扎实，推荐第 1 个；如果失败帧定位是最大亮点，推荐第 2 个。

### 0.3 核心贡献表述

建议贡献写 3 条，不要写太散：

1. **Problem formulation**：We formulate simulation trajectory construction as a stage-wise Code-as-Policy data generation problem, where the goal is not deploying the generated policy itself but producing successful multimodal trajectories for downstream VLA training.

2. **Method**：We introduce StageCap, a closed-loop data engine that decomposes long-horizon manipulation into verifiable stages, localizes failure frames, and repairs only the failed stage policy code.

3. **Evaluation**：We evaluate StageCap across diverse simulated manipulation scenes and show improvements in trajectory generation success rate, repair efficiency, scene robustness, and downstream imitation/VLA policy performance.

不要说 “first LLM data generator” 或 “first stage decomposition”。GenSim、RoboGen、MimicGen、RoboCasa、SayCan 等工作会让这个 claim 很危险。更稳的是说：**systematically integrates stage verification, failure-frame diagnosis, and local CaP repair for trajectory data generation**。

## 1. AAAI 常规论文结构建议

AAAI 主文通常应该走紧凑的 conference paper 结构。建议你采用下面这个结构：

```text
1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Method: StageCap
6. Experiments
7. Results and Analysis
8. Discussion and Limitations
9. Conclusion
References
```

如果页数紧张，可以合并：

- 把 **Problem Formulation** 放进 Method 开头。
- 把 **Discussion and Limitations** 合并到 Results 最后。
- 把 Related Work 写短，重点围绕 data generation, CaP, LLM simulation 三条线。

推荐主文篇幅分配（按 7 页正文估计，不含 references）：

| Section | 建议篇幅 | 核心任务 |
|---|---:|---|
| Abstract | 150-250 words | 一眼说明问题、方法、结果 |
| Introduction | 0.8-1.0 page | 建立动机、缺口、贡献 |
| Related Work | 0.7-1.0 page | 说明不是重复 MimicGen/GenSim/CaP |
| Problem Formulation | 0.3-0.5 page | 定义 trajectory data engine |
| Method | 1.8-2.2 pages | 讲清 StageCap 三个创新 |
| Experiments | 1.0-1.3 pages | 任务、基线、指标、实现 |
| Results and Analysis | 1.2-1.6 pages | 主结果、消融、案例分析 |
| Discussion/Limitations | 0.3-0.5 page | 控制 overclaim，回答边界 |
| Conclusion | 0.15-0.25 page | 收束贡献 |

## 2. Abstract 写作建议

### 2.1 Abstract 的五句话结构

AAAI 摘要要短而硬，建议用 5 句：

1. **背景需求**：VLA/robot manipulation policies need large-scale successful trajectories, but real-world collection is expensive.
2. **现有缺陷**：Existing simulation data engines scale data through demonstrations, replay, planning, or LLM-generated tasks, but often treat trajectory failure as an episode-level signal and lack fine-grained repair.
3. **核心方法**：We propose StageCap, a stage-wise Code-as-Policy data engine for simulation trajectory generation.
4. **技术机制**：StageCap decomposes long-horizon tasks into verifiable stages, localizes failure frames, and repairs only the failed stage policy code under diverse scene variations.
5. **实验结果**：Across [N] manipulation tasks and [M] scene variations, StageCap improves [success rate / repair efficiency / downstream policy performance] over [baselines].

### 2.2 可直接改写的英文模板

```text
Vision-language-action policies require large amounts of successful robot trajectories, yet collecting such data in the real world remains expensive and difficult to scale. Simulation data engines alleviate this bottleneck through demonstration replay, motion planning, and LLM-assisted task generation, but long-horizon failures are often detected only at the episode level, making failed trials hard to diagnose and repair. We propose StageCap, a stage-wise Code-as-Policy data engine for generating successful manipulation trajectories in simulation. StageCap represents a task as a sequence of verifiable policy stages, localizes diagnostic failure frames when execution breaks, and repairs only the failed stage code while preserving the rest of the trajectory program. Experiments on [tasks/scenes] show that StageCap improves trajectory generation success, reduces repair cost, and produces data that benefits downstream [BC/VLA] policy learning.
```

### 2.3 摘要里避免写什么

- 不要在摘要里先讲 “LLM is powerful” 这类泛泛背景。
- 不要说 “we are the first”，除非你能非常严格限定 first 的范围。
- 不要把 StageCap 描述成最终部署策略。它是 data engine，产物是成功轨迹数据。

## 3. Introduction 写作建议

Introduction 是最关键的一节。它要完成四个动作：需求、已有路线、缺口、你的解法。

### 3.1 推荐段落结构

#### Paragraph 1: 数据瓶颈

写作任务：

- 从 VLA/generalist robot policy 对大规模 trajectory data 的需求切入。
- 强调一条训练样本包含 language instruction、multi-view images、robot states/actions、success trajectory。
- 点出真实采集昂贵，仿真数据引擎因此重要。

建议内容：

```text
Recent vision-language-action models and generalist robot policies rely on large collections of robot trajectories that align language instructions, visual observations, proprioceptive states, and actions. However, collecting successful long-horizon manipulation trajectories in the real world is costly, slow, and difficult to scale across objects, layouts, and task variants. This motivates simulation-based data engines that can automatically produce large amounts of successful training trajectories.
```

#### Paragraph 2: 现有数据引擎路线

写作任务：

- 简短列出现有路线：human demos、motion planner/waypoints、MimicGen 示教重放、GenSim/RoboGen 任务生成、RoboCasa 平台。
- 注意这里不是 Related Work 全展开，只是建立领域版图。

建议内容：

```text
Existing approaches scale robot data in different ways: human teleoperation datasets provide high-quality demonstrations, benchmark simulators use manually designed waypoints or motion planners, MimicGen-style systems transform a small number of demonstrations into many synthetic trajectories, and recent LLM-assisted systems generate tasks, scenes, rewards, or supervision. These systems have substantially advanced scalable robot learning.
```

#### Paragraph 3: 核心缺口

写作任务：

- 引出你真正要打的痛点：长时程任务失败是阶段性的，但现有系统多在 episode 级别处理失败。
- 说明失败轨迹被筛掉、整体重试、或需要人工调试，造成生成效率低。
- 这段要写得很锋利，是审稿人判断 novelty 的地方。

建议内容：

```text
Despite this progress, current simulation data engines often handle execution failure at a coarse granularity. A long-horizon manipulation task may fail because the grasp is unstable, the object slips during transport, or the placement pose is invalid, but many pipelines only observe the final episode-level success signal. As a result, failed trials are frequently discarded or regenerated from scratch, leaving little mechanism to identify which stage failed and how the corresponding control logic should be locally repaired.
```

#### Paragraph 4: StageCap 的核心观点

写作任务：

- 讲你的 insight：长时程任务天然是阶段结构，CaP 刚好可执行、可读、可修改。
- 强调从 online policy deployment 转向 offline data generation，仿真特权状态可以合法用于生成器。

建议内容：

```text
Our key observation is that simulation trajectory construction differs from online robot deployment. A data engine may use privileged simulator states, stage-level success checkers, contact information, and diagnostic frames, as long as the final output is a valid observation-action trajectory for downstream learning. This makes Code-as-Policy a suitable intermediate representation: policy code can express staged manipulation logic, can be executed and verified in simulation, and can be locally patched when a specific stage fails.
```

#### Paragraph 5: 方法与贡献

写作任务：

- 简洁定义 StageCap。
- 列出 3 个贡献。

建议贡献段：

```text
We introduce StageCap, a stage-wise Code-as-Policy data engine for simulation-based trajectory generation. StageCap decomposes a manipulation instruction into executable stages, validates each stage boundary, localizes failure frames when execution fails, and repairs only the failed stage policy code under diverse scene variations. Our contributions are:
(1) a formulation of Code-as-Policy as a patchable trajectory data engine rather than only an online control policy;
(2) a stage-wise verification and local repair loop driven by failure-frame replay;
(3) an empirical evaluation showing improved generation success, repair efficiency, scene robustness, and downstream policy learning.
```

### 3.2 Introduction 的图

建议放 **Figure 1: StageCap overview**，最好在第一页或第二页顶部。

图中包含：

```text
Natural language task
   -> Stage decomposition
   -> Stage policy code generation
   -> Simulation execution
   -> Stage verifier
      -> success: save trajectory
      -> failure: locate failure frame -> diagnose -> patch failed stage -> retry
   -> VLA training dataset
```

图要突出一个闭环：不是线性生成，而是 stage verification and repair loop。

### 3.3 Introduction 审稿人会看什么

审稿人会问：

- 你和 MimicGen 的区别是不是足够清楚？
- 你和 GenSim/RoboGen 的区别是不是足够清楚？
- 为什么非要用 CaP，而不是 motion planner 或 behavior cloning？
- 你的贡献是一个系统工程，还是有明确技术创新？

Introduction 必须提前回答这些问题。

## 4. Related Work 写作建议

Related Work 建议分 3 个小节，不要罗列太多。

### 4.1 Simulation Data Engines for Robot Learning

写作内容：

- 包括 RoboTurk、robomimic、RLBench、ManiSkill2、MimicGen、DexMimicGen、RoboCasa/RoboCasa365。
- 重点比较数据从哪里来、如何扩增、失败如何处理。

推荐论点：

```text
Prior simulation data engines scale robot learning through human demonstrations, motion-planner-generated demonstrations, or demonstration transformation. These methods are effective for producing large datasets, but they typically depend on seed demonstrations, hand-designed task scripts, or object-centric segmentation. StageCap is complementary: instead of transforming existing demonstrations, it generates and repairs executable stage policies to produce successful trajectories.
```

要点：

- 对 MimicGen 要尊重：它是最强相关工作。
- 不要说 MimicGen 没有阶段思想；它有 subtask segments。
- 你的区别是：MimicGen 的分段主要用于示教变换和拼接，StageCap 的阶段用于验证、诊断、局部代码修复。

### 4.2 LLMs for Robotic Task and Simulation Generation

写作内容：

- 包括 GenSim、RoboGen、RoboCasa 的 LLM task generation。
- 也可以提一些 LLM 用于 reward generation、task proposal、scene generation 的工作。

推荐论点：

```text
LLM-based systems have recently been used to generate robotic tasks, scenes, rewards, and curricula. These methods expand the task and environment space, whereas StageCap focuses on the execution-level data production loop: given a task and simulator, how can the system repeatedly obtain successful trajectories when long-horizon execution fails?
```

区别：

- GenSim/RoboGen 更像 task/scene/supervision generator。
- StageCap 更像 trajectory production engine。
- 如果你实验中也生成多样场景，要说这是为了 stress-test 和 repair，不只是为了任务扩展。

### 4.3 Code-as-Policy and Programmatic Robot Control

写作内容：

- 包括 Code as Policies、VoxPoser、ProgPrompt、SayCan。
- 重点讲可执行代码、调用 primitive、表达控制逻辑、可组合性。

推荐论点：

```text
Code-as-Policy methods use language models to synthesize executable programs for embodied control. StageCap adopts this programmatic representation but changes its objective: the generated code is not evaluated as the final deployed robot policy; rather, it serves as an inspectable and patchable generator of successful training trajectories.
```

这句话很关键：它解释为什么 CaP 适合你的任务。

### 4.4 Related Work 最后一段

最后用一段总结定位：

```text
In summary, existing data engines emphasize scalable demonstrations, task diversity, or programmatic control. StageCap addresses a complementary bottleneck: when long-horizon simulated execution fails, how can a data engine diagnose the failed stage and repair the trajectory program without regenerating the entire policy?
```

## 5. Problem Formulation 写作建议

这一节的作用是把你的任务从 “robot policy learning” 分离出来，明确 StageCap 是 data generation。

### 5.1 定义输入输出

建议定义：

- Task instruction: `l`
- Simulator/environment: `E`
- Scene distribution: `S`
- Stage sequence: `G = {g_1, ..., g_K}`
- Stage policy code: `pi_k`
- Observation/action trajectory: `tau = {(o_t, a_t)}_{t=1}^T`
- Success verifier: `V_k`

写作内容：

```text
Given a language instruction l and a simulator E with a distribution of scenes S, the goal is to produce a set of successful trajectories D = {tau_i}. Each trajectory tau contains visual observations, robot states, actions, and metadata required for downstream imitation or VLA training. Unlike online policy learning, the data generator is allowed to access privileged simulator states and stage-level verifiers during generation; these signals are not included as inputs to the downstream learner.
```

### 5.2 定义成功目标

可以把目标写成：

```text
maximize successful trajectory yield under a fixed generation budget
```

指标：

- `success yield`: 成功轨迹数 / 尝试次数
- `repair efficiency`: 成功修复次数 / 修复尝试次数
- `generation cost`: LLM calls、simulation rollouts、wall-clock time
- `trajectory diversity`: scene coverage、object pose coverage、action/state diversity
- `downstream utility`: 用生成数据训练出的 policy success rate

### 5.3 定义 StageCap 的关键假设

要诚实写出假设，避免被审稿人质疑：

- 有一组低层控制 primitives 或 motion primitives 可调用。
- 仿真器能提供 success checker 或可计算的 stage postcondition。
- 任务可以合理分解成长时程阶段。
- 生成器可以访问仿真特权状态，但下游策略训练不能依赖这些状态。

这些假设不削弱论文，反而让边界清楚。

## 6. Method: StageCap 写作建议

Method 建议分成 5 个小节。

> **⚠️ 代码对齐注记（2026-07）**：本节已根据 `vlm_planner_cap/`、`capx/envs/stage_trial.py`、`vlm_planner_cap/pattern_priors.py`、`vlm_planner_cap/schema.py`、`vlm_planner_cap/stage_loop.py`、`scripts/extract_kite_failure_keyframes.py` 等核心代码做了全面对齐，删除了与实现不符的表述，并补充了关键实现细节。写正式论文时请以本节为准。

### 6.1 Overview

写作任务：

- 用一段总览 StageCap。
- 对应 Figure 1。

建议结构：

```text
StageCap consists of five components: (1) pattern-prior stage decomposition, (2) per-stage executable code generation, (3) privileged readonly scene snapshot collection, (4) self-reporting stage verification, and (5) failure-frame-driven local repair. Given a language instruction and a simulated scene, StageCap first classifies the task into a manipulation family and generates a sequence of verifiable stages using rule-based pattern priors distilled from successful Code-as-Policy programs. For each stage in turn, an LLM generates a self-contained executable Python script that calls audited robot motion primitives and readonly simulator APIs, then self-reports its outcome as a structured STAGE_STATUS signal. The runner parses this signal and decides whether to proceed to the next stage, retry the current stage with failure context, roll back to an earlier grasp stage within the same episode, or terminate the trial. When a stage fails, the system captures a diagnostic frame via online rendering, annotates it with stage evidence, and includes it in the repair prompt for the next attempt.
```

**实现事实**：代码在 `vlm_planner_cap/agents.py` → `build_pattern_plan()` 完成分解，在 `capx/envs/stage_trial.py` → `run_stage_trial()` 完成主循环，在 `vlm_planner_cap/stage_loop.py` → `choose_stage_decision()` 完成决策。Stage 分解**不依赖场景图像**，仅依赖自然语言任务字符串。

### 6.2 Stage Decomposition and Stage Interface

写作内容：

- 如何把任务拆成阶段：pick-lift, guarded-transport, low-release, drawer-open, drawer-close, stove-button, push-region 等。
- 阶段分解机制是 **rule-based pattern priors**（从 CAP-35 成功语料蒸馏），而非 LLM 实时动态分解。MIMO 仅用于每个阶段的**代码生成**，不参与 stage 序列的生成。
- **设计备注**：`VLMPlannerAgent` 的接口预留了 `model: ChatModel | None` 字段和对应的 LLM planner prompt（`vlm_planner_cap/prompts.py`），供未来切换为 LLM 分解；当前实现中该字段为 `None`，调用路径直接走 `build_pattern_plan(task)`（`pattern_priors.py`）。
- 每个阶段（`StagePlan`）包含的字段：
  - `stage_id`：唯一阶段标识符（字符串）
  - `name`：自然语言阶段名
  - `intent`：当前阶段的执行目标描述
  - `family`：任务家族枚举（`TaskFamily`）
  - `allowed_primitives`：允许调用的动作 primitive 白名单
  - `allowed_observations`：允许调用的只读 API 白名单
  - `checks`：阶段完成条件列表（`StageCheck`，包含 `name`, `check_type`, `success_criteria`）
  - `repair_hints`：修复建议，当阶段失败时注入 repair prompt
  - `max_attempts`：该阶段最大重试次数

**实现事实**：`StagePlan` 没有独立的 `precondition` 字段，前提条件通过 `checks` 的 `success_criteria` 和前序阶段的 STAGE_STATUS 决策链隐式保证。`TaskFamily` 枚举包括 `PICK_PLACE`、`CONTAINER_PLACE`、`SURFACE_PLACE`、`SPATIAL_RELATION`、`DRAWER`、`STOVE_BUTTON`、`RACK_OR_CADDY`、`MULTI_OBJECT`、`PUSH_REPAIR`。

建议表格：

**Table 1: Stage interface（与 `StagePlan` 数据类对齐）**

| Field | Example | Purpose |
|---|---|---|
| `stage_id` | `obj1_bowl_pick_lift_verify` | 唯一标识当前阶段，作为 STAGE_STATUS 校验键 |
| `intent` | `Pick and lift the bowl above the table` | LLM 生成代码时的执行意图 |
| `family` | `PICK_PLACE` | 任务家族，决定允许调用的 skill 集合 |
| `allowed_primitives` | `goto_pose, close_gripper, ...` | 允许的底层动作 API 白名单 |
| `checks` | `StageCheck(lift_delta ≥ 0.05)` | 阶段完成的几何/接触条件，供生成代码自验 |
| `repair_hints` | `"Retry with a side grasp if flat pick fails"` | 注入修复 prompt，指导 LLM 修正方向 |
| `max_attempts` | `2` | 该阶段最大执行尝试次数 |

写作重点：

- Stage 分解是任务家族感知的（family-aware）：DRAWER 类任务自动插入 open/close 子阶段，PUSH_REPAIR 类走单独 push-region 路径，MULTI_OBJECT 类按干扰风险排序对象处理顺序。
- 分解结果是 `TaskPlan`，包含 `stages` 列表和 `notes`（执行约束备注，注入 system prompt）。
- **诚实说明**：当前实现中 stage 模板来自 pattern priors，不是 LLM 零样本分解。这是系统可控性和稳定性的来源，也是 Method 的设计选择，不需要回避。

### 6.3 Stage-wise Code-as-Policy Generation

写作内容：

- LLM 每次只为**当前阶段**生成一个 executable Python 脚本，而非一次性生成所有阶段的代码。
- 代码通过 `CleanApiPolicy` 严格约束可调用的 API 集合，分为三层：
  - **动作 primitives**：`goto_pose`, `open_gripper`, `close_gripper`, `move_linear_delta`, `move_along_direction` 等底层 API
  - **只读 observations**：`get_object_pose`, `get_task_status`, `get_bddl_goal_diagnostics`, `get_contact_diagnostics`, `get_cabinet_drawer_progress` 等仿真状态 API
  - **经过审计的高层 skills**：`verified_lift_pick_object`, `guarded_transport_held_object_to_target`, `low_release_object_to_target`, `open_drawer_by_sleeve_retry` 等封装好的操控 skill
- **禁止符号**（forbidden symbols）：`satisfy_goal_predicates`, `set_state`, `set_object_pose`, `.data.qpos`, `eval`, `exec`, `globals()` 等直接修改仿真状态或绕过执行的捷径。

建议强调：

```text
Unlike end-to-end neural policies, StageCap's per-stage code representation is programmatic and inspectable: a failed placement stage corresponds to a concrete, self-contained script that can be re-prompted with failure evidence rather than requiring full policy retraining.
```

代码片段示例（与实际生成格式一致）：

```python
# Stage: obj1_bowl_low_release_verify
import json
stage_id = STAGE_ID  # injected by runner

obj_name = "bowl_1"
predicate = "in"
target_name = "basket_1"

result = low_release_object_to_target(
    obj_name, predicate=predicate, target_name=target_name,
    approach=0.08, retreat=0.05, release_z=0.02,
)
held_after = measure_object_grasp_state(obj_name).get("held", True)
goal_diag = get_bddl_goal_diagnostics()
relation_ok = goal_diag.get("satisfied", False)

status = "satisfied" if (not held_after and relation_ok) else "needs_repair"
evidence = f"held_after={held_after}, relation_satisfied={relation_ok}"
repair = "" if status == "satisfied" else "Check approach/release_z or retry transport."

print("STAGE_STATUS " + json.dumps(
    {"stage_id": STAGE_ID, "status": status, "evidence": evidence, "repair": repair},
    default=str,
))
```

**实现事实**：
- 生成的是 **inline 脚本**，不是函数定义（`def ...`）。
- 每个脚本必须以 `print("STAGE_STATUS " + json.dumps({...}))` 结尾，否则 runner 将其视为 `uncertain`。
- Runner 预注入全局变量 `STAGE_ID`（当前阶段 ID）和 `STAGE_ATTEMPT`（当前尝试次数）；生成代码只能只读访问，不能重新赋值。
- `json`, `math`, `np`, `time` 已预 import，代码不能 `import sys` 或调用 `sys.exit()`。
- Preflight 检查在代码进入仿真沙箱执行前运行，拦截 forbidden symbols、未定义变量和非法 import。
- **Oracle-readonly 与 API policy 选择**：runner 根据配置自动切换三套 `CleanApiPolicy`：
  - `oracle_readonly=False` → `CAPX_AGENT0_API_POLICY`（clean baseline，只读 API 受限，用于公平对比实验）
  - `oracle_readonly=True` → `DEFAULT_CLEAN_API_POLICY`（开放特权只读 API，如 `get_bddl_goal_diagnostics`，用于高成功率数据生成）
  - `oracle_readonly=True` + `SAM3=True` → `SAM3_CLEAN_API_POLICY`（额外开放 `segment_sam3_*` 视觉接口）
  
  三套 policy 运行的是**完全相同的代码路径**，区别仅在 LLM 可调用的 API 白名单。这意味着 privileged 模式和 clean baseline 的对比是受控变量实验，可以直接在消融表中呈现。

### 6.4 Stage Verification

写作内容：

- StageCap 的验证不依赖独立的外部 verifier 模型，而是**由生成的阶段代码自我汇报**：代码通过调用只读 privileged APIs（物体位姿、抽屉开合进度、BDDL 关系诊断、接触诊断等）检查自己的执行结果，并以结构化 `STAGE_STATUS` JSON 的形式输出。
- Runner 从 stdout 解析 `STAGE_STATUS`，提取 `status` 字段（四值离散决策：`satisfied / not_satisfied / needs_repair / uncertain`），并映射到五种控制决策：
  - `PROCEED`：当前阶段通过，进入下一阶段
  - `RETRY_STAGE`：重用同一代码，新增 failure context 再次尝试（attempt < max_attempts）
  - `REPLAN_STAGE`：当前阶段代码失效，重新为该阶段生成新代码
  - `REPLAN_SUFFIX`：当前阶段失败会影响后续阶段的前提（如物体已丢失），回滚到 episode 内更早的 pick 阶段重新执行
  - `FAIL_TASK`：检测到 forbidden symbols 或连续失败超出预算，直接中止当前任务

推荐表达：

```text
Stage verification converts sparse episode-level success into dense per-stage feedback. The generated code reads privileged simulator states—object poses, contact diagnostics, BDDL goal predicates, drawer progress—to assess whether its own execution met the stage's success criteria, and reports the outcome as a structured STAGE_STATUS signal. This signal is consumed only by the data engine; the downstream VLA policy receives only observation-action trajectories.
```

**实现事实**：
- 只读 privileged APIs 包括 `get_bddl_goal_diagnostics()`, `get_on_diagnostic()`, `get_cabinet_drawer_progress()`, `measure_object_grasp_state()`, `get_contact_diagnostics()` 等，这些 API 访问的仿真真值不出现在最终训练轨迹中。
- `REPLAN_SUFFIX` 的回滚目标由 `_fallback_stage_index_for_suffix_replan()` 自动从 stage plan 中确定，回退到同 episode 内最近的 pick-lift 阶段，不重置仿真环境。
- 执行失败后有 preflight error override 机制：如果代码因为 preflight 错误未能执行，runner 尝试使用 stage contract fallback code（基于 stage family 的保底实现）。
- **仿真真值覆盖层（Rule-based Evidence Override）**：runner 在解析 STAGE_STATUS 之后，还会用仿真器真值对自我汇报结果做最终修正（`_rule_override_evidence()` 和 `_placement_after_step_override_evidence()`）：
  - 若 BDDL goal 实际已 satisfied，即使代码汇报了 `not_satisfied`，也强制覆盖为 `satisfied`（避免 LLM 过于保守导致误回滚）
  - 若物体已不在可释放状态（`held=False` 但 relation 未满足），强制将 release 阶段标为 `needs_repair` 并触发回滚（避免 LLM 在无效状态下浪费尝试次数）
  
  这层 override 是 StageCap 验证可靠性的关键：最终裁决来自仿真器真值，而非单纯信任 LLM 生成的 STAGE_STATUS。论文中应清晰说明这一分层设计，以回应审稿人对 "self-reported verification 是否可靠" 的质疑。

### 6.5 Failure-Frame Localization and Replay

写作内容：

StageCap 使用**在线 KITE（Online KITE）**机制，在 trial 内部、每次阶段失败时实时捕获和传递失败诊断，而非等整条轨迹结束后再事后分析。

两种模式：

1. **Online KITE（主要模式，用于数据生成）**：每次 stage 执行后，若 STAGE_STATUS 不为 `satisfied`，runner 立刻调用 `env.render()` 渲染当前状态帧，与 stage evidence 合并，生成带注释的诊断图像，注入下一轮代码生成 prompt。
2. **Offline KITE（后处理分析）**：从已录制视频中提取光流运动关键帧（KITE motion keyframes）与 `stage_trace` 状态事件对齐，用于数据可视化和论文图表制作（`scripts/extract_kite_failure_keyframes.py`）。

在线 KITE 诊断包（`online_stage_failure_feedback`）包含：
- 渲染帧（带 stage_id、attempt、status、evidence、repair hint 文字标注的 RGB 图像）
- 结构化文字反馈：stage family 专属的修复指导模板
- 当前 stage_id、attempt、decision（RETRY/REPLAN/FAIL）

建议图：

**Figure 2: Failure-frame-driven repair**

图中展示：

```text
Stage k execution
   -> print STAGE_STATUS (not_satisfied / needs_repair)
   -> runner: choose_stage_decision -> RETRY_STAGE or REPLAN_STAGE or REPLAN_SUFFIX
   -> online KITE: env.render() -> annotate frame (stage_id, attempt, evidence, repair)
   -> build_stage_code_prompt(previous_records + KITE image inputs + scene_snapshot)
   -> LLM generates new stage k code
   -> execute -> new STAGE_STATUS
```

这节要回答：为什么在 stage 边界而不是 episode 结束时捕帧？

答案：

- stage 边界对应一个明确的动作意图（grasp / transport / release / open-drawer），失败帧在 stage 边界附近时，其归因是清晰的。
- 在 episode 终止时捕帧，物体状态可能已经被后续阶段的意外动作改变，帧与真正错误的动作距离过远。
- Online KITE 不依赖视频录制，frame 在执行过程中即时采集，可以与该阶段的 stdout/stderr 精确对应。

**实现事实**：
- 在线 KITE 由 `use_online_stage_kite_feedback=True` 配置项开启（`--use-kite-feedback` 参数）。
- 图像通过 `_online_stage_feedback_image_inputs()` 被序列化为 base64 注入 LLM 多模态 prompt，因此需要支持图像输入的模型（如 mimo-v2.5, gemini, gpt-4o）。
- `max_online_stage_feedback_images`（默认 1）控制每次 repair prompt 包含的最大图像数。

### 6.6 Local Policy Repair

写作内容：

- 修复的本质是**重新生成**当前失败阶段的代码，而非对原始代码做字符串 patch。
- 修复只针对失败的单个阶段；其他已经通过的阶段不受影响，也不需要重新生成代码。
- **局部修复不引发 regression 的机制基础**：每个 stage 脚本是**完全无状态的**（stateless）——执行合约明确要求 "Code must be self-contained: define every helper you call, and do not rely on variables from examples"。各 stage 脚本之间**唯一的共享状态是物理仿真器本身**（机器人关节角、物体位姿、接触状态）。因此修复某一阶段的代码，不会影响其他阶段的变量或逻辑；"局部修复不引入 regression" 这一 claim 的代码基础正在于此。
- 修复 prompt 由 `build_stage_code_prompt()` 构建，包含：
  - 完整 stage plan（展示所有阶段列表，但 LLM 只负责当前失败阶段）
  - `previous_records`：最近 8 次阶段执行记录（stage_id、attempt、status、rationale、repair_hint、stderr_tail、log_evidence）
  - `replan_context`：合并全局 KITE 反馈和当前阶段特定修复上下文
  - `scene_snapshot_text`：当前状态的只读场景快照（object poses、EE pose、contact diagnostics 等）
  - Online KITE 图像（失败帧 base64 编码，视模型支持情况注入）

建议写一个公式：

```text
c_k^(t) = merge(previous_records, F_k^(t-1), snapshot_k^(t), KITE_images^(t-1))
pi_k^(t) = LLM(stage_plan, stage_k, c_k^(t), api_policy)
```

其中 `pi_k^(t)` 是第 t 次尝试时为阶段 k 生成的代码，`F_k^(t-1)` 是上一次失败的诊断帧，`snapshot_k^(t)` 是当前场景快照，`c_k^(t)` 是合并后的修复上下文。

建议放 **Algorithm 1: StageCap trajectory generation**（与实现对齐的准确版本）：

```text
Input: instruction l, simulator E, trial seed s, budget B
     : stage plan P = Decompose(l)          -- rule-based pattern priors, scene-independent
Output: successful trajectory tau or failure

env.reset(seed=s)
for attempt in 1..B:
    stage k = P.current_stage              -- stage pointer, NOT episode reset
    if k is None: break                    -- all stages passed

    snapshot = collect_readonly_scene_snapshot(env, stage=k)
    if fast_path_satisfied(k, snapshot):   -- deterministic readonly pre-check
        advance(k); continue

    code_k = LLM(l, P, k, previous_records, kite_frames, snapshot)  -- generate ONE stage
    preflight_check(code_k, api_policy)    -- reject forbidden symbols before sandbox
    stdout, stderr = env.execute(code_k)   -- run in sandbox; code calls robot APIs
    evidence = parse_STAGE_STATUS(stdout)  -- code self-reports: satisfied/not_satisfied/needs_repair/uncertain

    if evidence.status == satisfied:
        decision = PROCEED                 -- advance to next stage
    elif attempt_k < k.max_attempts:
        decision = RETRY_STAGE             -- retry same stage with failure context
    elif failure_invalidates_suffix(k, evidence):
        decision = REPLAN_SUFFIX           -- roll back to earlier pick stage (same episode, no env reset)
    else:
        decision = REPLAN_STAGE            -- regenerate code for same stage

    if decision == FAIL_TASK: break        -- forbidden symbol detected

    if online_kite_enabled and status != satisfied:
        frame = env.render()               -- capture frame at failure boundary
        kite_frames[k].append(annotate(frame, evidence))

    update_stage_pointer(P, decision)      -- REPLAN_SUFFIX may move pointer back to pick-lift

if all stages passed:
    save trajectory tau
```

**关键实现差异（与常见误解的对比）**：
1. Stage 分解是 scene-independent 的，不接收场景图像作为输入。
2. 代码每次只生成当前阶段的，不是事先为所有阶段预生成。
3. `REPLAN_SUFFIX` 在同 episode 内回滚到更早的 pick 阶段，**不重置仿真环境**。
4. Failure-frame 在 stage 边界实时捕获（online KITE），不是从完整 rollout 事后提取。
5. 存在 `FAIL_TASK` 决策（forbidden symbol 检测），直接终止当前 trial。
6. 所有 stage 通过后，若 task reward 未达到 1.0，会尝试 `global_bddl_goal_repair` 作为最终兜底修复。

**两级修复结构（Within-trial 与 Cross-trial）**：

StageCap 实际存在两个修复层次，论文中应明确区分：

| 层次 | 触发条件 | 修复动作 | 控制参数 |
|---|---|---|---|
| **Within-trial**（trial 内） | 某 stage STAGE_STATUS 非 satisfied | RETRY / REPLAN_STAGE / REPLAN_SUFFIX，同 episode 内立即重试 | `max_total_stage_attempts`（默认 10） |
| **Cross-trial**（trial 间） | 整条 trial 失败后进入下一轮 | `_build_replan_context()` 将失败分析打包为 `replan_context`，注入下一轮 trial 的全局 prompt | `feedback_rounds`（控制对同一任务重跑的轮数） |

Within-trial 是核心 repair loop（本论文重点）；cross-trial 是额外的跨轮反馈机制，主要用于提高数据生成质量，正式评测中需明确报告是否启用（`--feedback-rounds > 1`）。

### 6.7 Diverse Scene Generation

写作内容：

- 场景多样性来自 LIBERO 仿真环境的内置 scene variation（trial seed 控制初始物体位姿、布局、实例随机化）。
- StageCap 通过多轮 trial（不同 seed）覆盖多种场景配置，每个 trial 独立执行完整的 stage-verify-repair 循环。
- 多样化场景本身不是 StageCap 的独立创新，而是数据引擎设计的 stress-test 条件：不同的初始场景会触发不同种类的阶段失败，从而验证 repair loop 的通用性。

建议表达：

```text
Scene variation exposes the data engine to execution failures that are unlikely to appear in a fixed layout. StageCap treats these failures as repair signals within the stage verification loop, allowing the stage code generator to adapt across object poses, layouts, and configurations without task-specific manual adjustments.
```

**实现事实**：
- `env.reset(options={"trial": trial}, seed=trial)` 是场景多样化的入口；trial 索引由 batch 脚本循环分配。
- 每个 trial 对应一个完整的 `run_stage_trial()` 调用，有独立的 stage pointer 和 execution records，trial 之间完全独立。
- `--total-trials` 参数控制每个任务的总试验次数；`--feedback-rounds` 控制每轮 KITE 反馈后是否对同一 trial 重跑（主要用于数据质量提升而非正式评测）。
## 7. Experiments 写作建议

Experiments 要清楚回答四个研究问题。

### 7.1 推荐 Research Questions

写成 RQ 很适合 AAAI：

```text
RQ1: Does StageCap improve successful trajectory generation compared with whole-policy regeneration or non-repair baselines?
RQ2: How much does stage-wise local repair reduce generation cost?
RQ3: Do failure-frame localization and stage verification each contribute to performance?
RQ4: Do trajectories generated by StageCap improve downstream imitation/VLA policy learning?
```

### 7.2 Tasks and Environments

写作内容：

- 仿真平台：robosuite / RoboCasa / ManiSkill / RLBench，按你实际用的写。
- 任务类型：pick-and-place、drawer、stack、place into container、multi-object manipulation。
- 每个任务的 stage sequence。

建议表格：

**Table 2: Evaluation tasks**

| Task | Horizon | Stages | Scene variations | Success condition |
|---|---:|---|---:|---|
| Pick-and-place | medium | approach-grasp-transport-place | N | object in target region |
| Open drawer and place object | long | open-grasp-transport-place-close | N | object inside drawer |
| Stack object | long | grasp-align-place-verify | N | stable stack |

如果任务少，要把 scene variations 做扎实。

### 7.3 Baselines

至少需要这些 baseline：

1. **Whole-CaP Retry**
   - 失败后重新生成整段 policy code。
   - 用来证明 local repair 有价值。

2. **StageCap w/o Failure Frame**
   - 有 stage verifier，但只给最终失败信息，不给诊断帧。
   - 用来证明 failure-frame replay 有价值。

3. **StageCap w/o Stage Verification**
   - 只看 episode-level success。
   - 用来证明阶段验证有价值。

4. **Planner/Scripted Expert**
   - 如果有现成任务脚本或 motion planner。
   - 用来说明 StageCap 相对手工工程的优势或局限。

5. **MimicGen-style baseline**
   - 如果你有 seed demos 和适配条件，最好加入。
   - 如果做不了完整 MimicGen，至少在 Related Work 和 limitation 里说明没有直接复现的原因。

### 7.4 Metrics

建议分四类：

**Generation metrics**

- Success rate: 成功轨迹 / 尝试轨迹
- Attempts per success: 平均多少次 rollout 得到一条成功数据
- LLM calls per success
- Wall-clock time per success

**Repair metrics**

- Repair success rate
- Local patch regression rate: 修复某阶段后，前/后阶段是否被破坏
- Failure localization accuracy: 诊断阶段是否等于人工标注失败阶段

**Diversity metrics**

- object pose coverage
- scene layout coverage
- action/state trajectory diversity
- task variation coverage

**Downstream learning metrics**

- BC/VLA success rate on seen scenes
- BC/VLA success rate on unseen scenes
- data efficiency: 用多少 demo 达到某个 success

### 7.5 Implementation Details

要写清：

- LLM model
- prompt structure
- max repair attempts
- primitive library
- simulator frequency / control mode
- observation modalities
- VLA/BC model architecture
- train/test split
- number of seeds

这部分不需要长，但要可复现。

## 8. Results and Analysis 写作建议

Results 不要只堆表，要按 claim 展开。

### 8.1 Main Result: Trajectory Generation Success

建议图表：

**Table 3: Generation success rate**

| Method | Success rate ↑ | Attempts/success ↓ | LLM calls/success ↓ | Time/success ↓ |
|---|---:|---:|---:|---:|
| Whole-CaP Retry | | | | |
| w/o stage verification | | | | |
| w/o failure frame | | | | |
| StageCap | | | | |

写作逻辑：

- 先说 StageCap 在大多数任务上提升成功率。
- 再说长时程任务提升更明显，因为失败更阶段化。
- 如果某些任务提升不大，要解释可能原因：任务短、planner already strong、失败不易从单帧诊断。

### 8.2 Ablation: 三个创新分别有没有用

建议至少三个消融：

- no stage verification
- no failure-frame replay
- global repair instead of local repair

写作逻辑：

```text
Removing stage verification reduces the granularity of feedback, causing the repair module to modify unrelated code blocks. Removing failure frames makes repair prompts less grounded and increases repeated failure. Replacing local repair with whole-program regeneration increases variance and often introduces regressions in previously successful stages.
```

### 8.3 Repair Behavior Analysis

这是你论文的亮点分析。建议单独做：

**Figure 3: Failure distribution by stage**

| Stage | Failure count | Repair success | Common cause |
|---|---:|---:|---|
| Grasp | | | missed contact |
| Transport | | | object slip |
| Place | | | target misalignment |
| Verify | | | unstable final pose |

这张表能让审稿人看到 StageCap 不是黑盒 LLM 试错，而是在分析数据生产过程。

### 8.4 Downstream Policy Learning

如果你的目标是 VLA 数据构建，这部分非常重要。

实验设计：

- human/demo only
- generated demos without repair
- StageCap generated demos
- mixed human + StageCap

指标：

- policy success rate
- seen/unseen scenes
- robustness to pose/layout variation

写作逻辑：

```text
The ultimate goal of StageCap is not merely to execute tasks in simulation but to produce useful training data. We therefore train downstream policies on generated trajectories and evaluate whether StageCap data improves generalization.
```

如果暂时没有 VLA，用 BC 或 diffusion policy 也可以，但论文里要说这是 proxy for downstream trajectory utility。

### 8.5 Qualitative Case Study

建议放 1 个失败修复案例：

场景：

- 任务：把碗放进抽屉。
- 初始失败：place stage 把碗放到抽屉边缘，导致碰撞或掉落。
- failure frame：wrist view 显示碗偏右，sim state 显示 target pose offset。
- repair：修改 place offset / approach height / gripper release timing。
- 成功：重新执行后完成。

图：

```text
(a) last valid frame
(b) failure frame
(c) repaired code diff
(d) successful final frame
```

这个案例可以很好地解释“失败帧定位复盘”。

## 9. Discussion and Limitations 写作建议

AAAI 审稿人会喜欢诚实的边界。建议写短，但必须写。

### 9.1 Discussion

可以讨论：

- StageCap 和 MimicGen 是互补关系，不是替代关系。
- StageCap 适合长时程、阶段结构明显的操控任务。
- StageCap 的优势来自可执行程序中间表示和仿真反馈闭环。

建议表达：

```text
StageCap is complementary to demonstration transformation methods such as MimicGen. Whereas those methods reuse human demonstrations through object-centric transformations, StageCap repairs executable stage policies using simulator feedback. This makes it particularly useful when seed demonstrations are limited or when scene variations induce stage-specific failures.
```

### 9.2 Limitations

建议写 4 点：

1. 依赖低层 primitive 和 stage verifier。
2. 对很难阶段化的任务，效果可能下降。
3. 仿真成功不等于真实机器人成功，存在 sim-to-real gap。
4. LLM 修复可能引入无效代码或 unsafe action，需要 sandbox/checker。

限制要带 future work：

```text
Future work will investigate automatic verifier synthesis, richer multi-modal failure diagnosis, and sim-to-real validation of StageCap-generated datasets.
```

### 9.3 Ethical / Safety Note

如果篇幅允许，可以简单写：

- StageCap 用于仿真数据生成，不直接控制真实机器人。
- 真实部署前需要 safety constraints 和 human validation。
- 生成的数据和代码应记录 provenance，避免不可追踪数据污染。

## 10. Conclusion 写作建议

Conclusion 三句话即可：

1. 重申问题：仿真数据引擎需要细粒度失败诊断和修复。
2. 重申方法：StageCap 用阶段化 CaP、失败帧复盘、局部修复生成成功轨迹。
3. 重申结果和意义：提高生成效率，并为 VLA/robot learning 提供更可靠的数据生产路径。

模板：

```text
We presented StageCap, a stage-wise Code-as-Policy data engine for simulation trajectory generation. By validating task stages, localizing failure frames, and repairing only failed stage policies, StageCap turns long-horizon execution failures into actionable feedback for data production. Our results suggest that programmatic, failure-aware trajectory generation is a promising direction for scaling robot learning data beyond manual demonstrations and coarse episode-level filtering.
```

## 11. 推荐图表清单

| 编号 | 类型 | 内容 | 放置位置 |
|---|---|---|---|
| Figure 1 | Overview | StageCap pipeline and repair loop | Introduction / Method |
| Figure 2 | Method detail | Failure-frame localization and local repair | Method |
| Algorithm 1 | Pseudocode | StageCap trajectory generation | Method |
| Table 1 | Method table | Stage interface fields | Method |
| Table 2 | Experiment setup | Tasks, stages, scene variations | Experiments |
| Table 3 | Main results | Success rate, attempts/success, LLM calls | Results |
| Figure 3 | Ablation plot | w/o stage verifier, w/o failure frame, global repair | Results |
| Table 4 | Failure analysis | Failure type/stage distribution | Results |
| Figure 4 | Downstream learning | Policy success vs data source | Results |
| Figure 5 | Case study | Failure frame, code patch, success frame | Results/Appendix |

如果页数紧张，主文保留 Figure 1、Algorithm 1、Table 3、Figure 4；其他放 appendix。

## 12. 论文主线的 Claim-Evidence-Reasoning

| Claim | Evidence you need | Reasoning |
|---|---|---|
| 现有数据引擎失败反馈粒度粗 | 文献对比 + baseline 表现 | episode-level 成败无法指导局部修复 |
| 阶段验证提升生成成功率 | w/o stage verifier 消融 | dense feedback 让失败定位更准确 |
| 失败帧复盘提升修复效率 | w/o failure frame 消融 | grounded diagnosis 减少盲目重写 |
| 局部修复优于整体重生成 | local vs global repair | 保留已成功阶段，降低 regression |
| 多变场景下 StageCap 更有价值 | scene split / unseen layout | scene variation 制造阶段性失败，repair loop 可适配 |
| 生成数据对下游学习有用 | BC/VLA 训练结果 | 数据引擎价值最终体现在 policy learning |

## 13. 审稿人可能攻击点与预防写法

### 攻击 1：这只是 prompt engineering

预防：

- 给出清楚系统结构：stage interface、verifier、failure frame selector、repair loop。
- 做消融证明每个模块有效。
- 记录 repair statistics，而不只是展示最终成功率。

### 攻击 2：MimicGen 已经做了数据生成和 subtask 分段

预防：

- 明确 MimicGen 分段用于 object-centric demonstration transformation。
- StageCap 分段用于 stage verification、failure diagnosis、local code repair。
- 如果可能，加入 MimicGen 或 MimicGen-like baseline。

### 攻击 3：GenSim/RoboGen 已经用 LLM 生成仿真数据

预防：

- 说明它们主要扩展 task/scene/supervision space。
- StageCap 解决 execution-level trajectory production failure。

### 攻击 4：使用 privileged simulator state 不公平

预防：

- 强调 data engine 和 downstream learner 的分离。
- privileged state 只用于生成/验证/修复，不作为 VLA 输入。
- 类比 motion planner / scripted expert 也常用仿真状态生成演示。

### 攻击 5：只能在有 verifier 的任务上用

预防：

- 承认 limitation。
- 说明许多仿真 benchmark 本来就有 success predicates。
- future work: automatic verifier synthesis。

## 14. 可直接使用的小节标题

推荐标题：

```text
1 Introduction
2 Related Work
  2.1 Simulation Data Engines for Robot Learning
  2.2 LLM-Assisted Robotic Simulation and Task Generation
  2.3 Code-as-Policy for Embodied Control
3 Problem Formulation
4 StageCap
  4.1 Overview
  4.2 Stage Interface and Policy Generation
  4.3 Stage-wise Verification
  4.4 Failure-Frame-Driven Local Repair
  4.5 Diverse Scene Trajectory Generation
5 Experiments
  5.1 Experimental Setup
  5.2 Baselines and Metrics
  5.3 Downstream Policy Training
6 Results
  6.1 Trajectory Generation Performance
  6.2 Ablation Study
  6.3 Failure and Repair Analysis
  6.4 Downstream Policy Performance
7 Discussion and Limitations
8 Conclusion
```

如果 Method 很长，可以把 `Problem Formulation` 合并进 `StageCap` 的 4.1。

## 15. 写作风格建议

### 15.1 语气

AAAI 论文要直接、克制、证据导向。建议少用：

- "revolutionary"
- "unprecedented"
- "fully autonomous"
- "general-purpose"
- "solves data scarcity"

建议多用：

- "we formulate"
- "we introduce"
- "we show empirically"
- "complementary to"
- "under the assumption that"
- "in simulation-based data generation"

### 15.2 关键词统一

全文建议统一使用：

- `trajectory data engine`
- `stage-wise verification`
- `failure-frame localization`
- `local policy repair`
- `simulation trajectory generation`
- `downstream VLA policy`

不要一会儿叫 agent，一会儿叫 framework，一会儿叫 data engine，除非先定义：

```text
We refer to StageCap as a data engine because its output is a dataset of successful trajectories rather than a deployed control policy.
```

### 15.3 结果表述

结果出来前，可以先用占位：

```text
StageCap improves generation success by X% over whole-program regeneration and reduces LLM calls per successful trajectory by Y%.
```

正式写作时，所有 X/Y 必须来自实验，不要用模糊词替代。

## 16. 下一步写作顺序

建议不要从 Abstract 开始写。按这个顺序更顺：

1. 先画 Figure 1 和 Algorithm 1。
2. 写 Method，确定系统到底怎么跑。
3. 写 Experiments，确定任务、基线、指标。
4. 写 Results 的表格模板，把实验缺口列出来。
5. 写 Introduction，把方法和实验能证明的东西收束成动机。
6. 写 Related Work，围绕 Introduction 的缺口补文献。
7. 最后写 Abstract 和 Conclusion。

这能避免最常见的问题：Introduction 讲得太大，实验支撑不住。

## 17. 参考文献分组

写 Related Work 时建议按下面分组引用：

**Simulation data and demonstration generation**

- MimicGen
- DexMimicGen
- RoboCasa / RoboCasa365
- robomimic
- RoboTurk
- RLBench
- ManiSkill2

**LLM-assisted simulation/task generation**

- GenSim
- RoboGen
- 相关 reward/task/scene generation 工作

**Code-as-Policy and LLM robot control**

- Code as Policies
- VoxPoser
- SayCan
- ProgPrompt

**Downstream robot policy learning**

- VLA / BC / diffusion policy / RT-style generalist policy，根据你实际实验补。

## 18. 最推荐的论文核心句

这几句可以贯穿全文：

1. **StageCap changes the role of Code-as-Policy from an online controller to a patchable trajectory data generator.**

2. **Long-horizon manipulation failures are often local to a stage, but most data engines observe them only as episode-level failures.**

3. **Stage-wise verification turns sparse success signals into actionable repair feedback.**

4. **Failure frames provide grounded evidence for repairing policy code, reducing blind whole-program regeneration.**

5. **The value of StageCap lies not in producing the smartest policy code, but in producing diverse, successful, and learnable trajectories.**

