# Stage-CAP: Introduction Draft

*AAAI / ICLR Style — Draft v2.0 (2026-07-09, revised)*

---

## English Version (For Submission)

### 1. Introduction

Training capable robotic manipulation policies requires large-scale, high-quality demonstration trajectory data. Manual teleoperation, while producing high-fidelity trajectories, is costly, labor-intensive, and fundamentally difficult to scale to the task breadth required by modern robot learning. Simulation-based automatic trajectory generation has therefore emerged as an essential alternative: by leveraging privileged information directly accessible within the simulator, large volumes of training demonstrations can be produced without human supervision.

Automated trajectory generation spans several paradigms, from classical to recent. *Kinematic planning with handcrafted waypoints* (e.g., RLBench~\cite{james2019rlbench}) achieves near-perfect execution success but requires per-task manual engineering of keypoints and success checkers, limiting scalability to new tasks. *Reinforcement learning*~\cite{luo2024serl} can handle contact-rich tasks but suffers from sample-intensive reward engineering that does not generalize across tasks. *Demonstration replay and spatial transformation* methods such as MimicGen~\cite{mandlekar2023mimicgen} amplify small seed sets into large datasets but fundamentally depend on pre-existing human demonstrations. *3D value-map synthesis*~\cite{huang2023voxposer} and *code-as-policy generation*~\cite{liang2023codeaspolicies,fu2026capx} represent the most autonomous directions, enabling zero-shot trajectory generation directly from language instructions.

Despite their diversity, all existing approaches share a critical and under-examined inefficiency: **trajectory generation is treated as an all-or-nothing process**. We formally define this as the **"Failure-as-Discard" bottleneck**: regardless of where a local failure occurs—a grasp misalignment at step 40 of a 200-step sequence, a kinematic singularity mid-transport, or a contact failure at final placement—the entire trajectory is discarded and generation restarts from scratch. This wastes not only the physical progress of successfully executed stages, but also the rich diagnostic signals embedded in the failure moment itself. A five-stage manipulation sequence in which Stages 1–4 succeed but Stage 5 fails yields zero training data under the current paradigm, despite the first four stages constituting directly usable high-fidelity manipulation demonstrations.

This bottleneck is particularly acute in code-as-policy approaches. While CaP-X~\cite{fu2026capx} introduces multi-turn visual differencing feedback and demonstrates impressive zero-shot task completion capability, it operates at **whole-program granularity**: when an execution fails, the agent regenerates the entire control program without identifying *which specific stage* failed or leveraging privileged simulator state for precise failure localization. When a trial fails, physical progress is irretrievably lost. In our experimental evaluation on the LIBERO benchmark~\cite{liu2023libero}, this approach achieves ~[B]% task success rate on complex multi-stage manipulation tasks—meaning [100-B]% of LLM query budget is consumed by failed attempts that yield no recoverable training data.

To break this bottleneck, we propose **Stage-CAP** (**Stage-wise Code-as-Policy**), a simulation trajectory generation framework that explicitly decomposes long-horizon manipulation tasks into a sequence of execution stages—approach, grasp, transport, placement, and verification—each equipped with formal preconditions and postconditions verified against privileged simulator state (object poses, BDDL relational predicates, contact indicators, joint progress). For each stage, a multimodal large language model independently generates and executes control code grounded in the current physical context.

Critically, when a stage fails, Stage-CAP employs a **two-level failure localization mechanism** to precisely identify the failure instant: (1) *online stage-boundary interception*, which captures the exact embodied visual evidence at the moment execution first deviates from success criteria; and (2) *cross-episode optical-flow keyframe extraction*, which identifies subtle motion anomalies across trial history. This localization feeds into a **dual-channel adaptive repair mechanism**—a text diagnostic channel (structured stage status report with spatial displacement, gripper state, and relational context) and an image feedback channel (annotated failure keyframes injected into the subsequent LLM prompt)—to regenerate code only for the failing stage, resuming execution from the current physical state without resetting the environment or re-running any previously verified stages.

We evaluate Stage-CAP on LIBERO-40~\cite{liu2023libero} and RoboSuite~\cite{zhu2020robosuite}. Under an equal LLM query budget, Stage-CAP achieves ~[X]% task success rate, improving over the whole-program regeneration baseline by ~[Δ] percentage points. Generated demonstrations are exported in LeRobot format~\cite{lerobot2024} and used to fine-tune BC and π₀~\cite{black2024pi0} policies, which outperform policies trained on equal-volume baseline data by ~[Z]% on downstream policy success rate, demonstrating architecture-agnostic quality gains.

**Our main contributions are:**

1. **Identification of the "Failure-as-Discard" Bottleneck.** We formally characterize this systemic limitation present in all major automated trajectory generation paradigms and quantify how discarding local failures drains data production throughput for long-horizon manipulation tasks.

2. **Stage-CAP: A Stage-Decomposed Code-as-Policy Data Engine.** We propose a modular trajectory generation framework that decomposes manipulation tasks into formally-specified execution stages with precondition/postcondition verification against privileged simulator state, enabling stage-level success accounting and targeted failure recovery without full-trajectory resets.

3. **Two-Level Failure Localization and Dual-Channel Adaptive Repair.** We introduce a mechanism that (a) precisely localizes failure keyframes at stage boundaries via online interception and cross-episode optical-flow analysis; (b) constructs compact structured diagnostics from privileged simulator state at the failure instant; and (c) applies these diagnostics as targeted LLM context for per-stage code repair—replacing costly whole-trajectory regeneration with surgical local fixes.

---

## 中文版本 (Chinese Translation)

### 1. 引言

训练具备强大能力的机器人操控策略，依赖于大规模、高质量的示范轨迹数据。人工遥操作虽然能产生高保真轨迹，但成本高昂、极耗人力，且在规模化扩展上存在根本困难。这一数据瓶颈已成为限制具身智能广泛应用的首要障碍。仿真平台提供了一种极具潜力的可扩展替代方案——通过利用仿真器内部的特权信息，可以在无人类监督的情况下自动生成大规模训练数据。

自动化轨迹生成涵盖多种技术路线，从经典到近期依次为：以 RLBench~\cite{james2019rlbench} 为代表的*运动学规划与脚本化航点控制*执行成功率接近完美，但需为每个新任务手工设计关键点与成功判定条件，难以规模化迁移；*基于强化学习的试错搜索*~\cite{luo2024serl}能处理接触丰富的操控任务，但依赖不可跨任务泛化的样本密集奖励设计；以 MimicGen~\cite{mandlekar2023mimicgen} 为代表的*示教重放与空间变换*方法能从少量种子示教生成大规模数据集，但根本依赖于已有的人工示教；*三维价值图合成*~\cite{huang2023voxposer}与*代码即策略生成*~\cite{liang2023codeaspolicies,fu2026capx}则代表自主化程度最高的方向，能够直接从自然语言指令零样本生成轨迹。

尽管技术路线各异，现有所有方法都共享一个关键且鲜被正视的设计缺陷：**轨迹生成被视为"全有或全无"的过程**。我们将这一系统性缺陷正式定义为**"失败即丢弃"瓶颈**：无论局部失败发生在何处——200步序列第40步的抓取偏差、运输途中的运动学奇异，还是最终放置时的接触失败——整条轨迹都会被直接丢弃并从零重试。这不仅浪费了已成功执行阶段的物理进度，更丢弃了失败时刻本身蕴含的最丰富诊断信号。一个五阶段操控序列中，第1至第4阶段均已成功执行、仅第5阶段失败，当前范式下产出的有效训练数据为零——尽管前四个阶段的高保真物理执行过程完全可以直接利用。

这一瓶颈在代码即策略方法中尤为突出。尽管 CaP-X~\cite{fu2026capx} 引入了多轮视觉差异反馈机制并展示了出色的零样本任务完成能力，这类方法仍以**整段程序为最小工作粒度**：当执行失败时，智能体重新生成整个控制程序，而无法识别*具体哪个阶段*失败，也无法利用仿真器特权状态对失败时刻进行精确定位。当整次尝试失败时，已积累的物理进度随之消失。在我们对 LIBERO 基准~\cite{liu2023libero}的实验评估中，这一方式在复杂多阶段操控任务上的轨迹生成任务成功率约为 **[B]%**，意味着 **[100-B]%** 的大模型查询预算被失败尝试所消耗，且无法产出任何可复用的训练数据。

为突破这一瓶颈，我们提出 **Stage-CAP**（**分阶段代码即策略**），一种将长时域操控任务显式分解为带形式化行为合约执行阶段序列的仿真轨迹生成框架。每个阶段（接近、抓取、运输、放置与验证）均配有前置条件与后置条件，并通过仿真器特权状态（物体位姿、BDDL 关系谓词、接触指示量、关节进度）实时验证；多模态大语言模型在当前物理上下文的支持下，为每个阶段独立生成并执行控制代码。

至关重要的是，当某阶段失败时，Stage-CAP 通过**两级关键帧定位机制**精准识别失败时刻：（1）*在线阶段边界截获*，在执行状态首次偏离成功标准时精确捕获具身视觉证据；（2）*跨回合光流关键帧抽取*，用于识别跨历史轨迹的细微异常运动。该精准时空定位结果随即馈入**双通道自适应修复机制**——文本诊断通道（包含空间位移、夹爪状态与任务相关关系上下文的结构化阶段状态报告）与图像反馈通道（将标注关键帧注入后续大模型提示）——系统仅针对失败阶段重新生成代码，并在当前物理状态下继续执行，无需重置仿真环境，也无需重跑任何已验证的前序阶段，从根本上改变了仿真数据生产的经济模型。

我们在 LIBERO-40 基准~\cite{liu2023libero}与 RoboSuite~\cite{zhu2020robosuite}上对 Stage-CAP 进行评估。在相同大模型查询预算下，Stage-CAP 达到约 **[X]%** 的任务成功率，相较整体程序重生成基线提升约 **[Δ] 个百分点**。所生成的示范轨迹以 LeRobot 格式~\cite{lerobot2024}输出，用于微调行为克隆（BC）与 π₀~\cite{black2024pi0}策略，在相同数据量下显著优于基线生成数据，下游策略成功率提升约 **[Z]%**，表明数据质量提升具有架构无关性。

**本文的主要贡献如下：**

1. **识别"失败即丢弃"瓶颈。** 我们正式定义了当前所有主流轨迹生成范式中普遍存在的这一系统性缺陷，并量化了局部失败的丢弃处理方式对长时域操控任务数据吞吐量的损耗，论证阶段级修复机制是突破这一瓶颈的根本路径。

2. **Stage-CAP：分阶段代码即策略数据引擎。** 我们提出了一种模块化轨迹生成框架，通过形式化的阶段分解与特权状态验证执行长时域操控任务，在无需人工种子示教的前提下实现阶段级成功核算与定向失败修复，消除全轨迹重置的必要性。

3. **两级关键帧定位与双通道自适应修复机制。** 我们提出了一种机制，能够（a）通过在线截获与跨回合光流分析精确定位阶段边界处的失败关键帧；（b）从失败时刻的特权仿真状态中构建紧凑的结构化诊断信息；（c）将诊断信息作为大模型的定向上下文实施阶段级策略代码修复——以局部微创修复彻底取代高耗能的全轨迹重建流程，在轨迹生成成功率与下游 VLA 策略表现上均取得显著提升。

---

## Writing Notes

### CaP-X 论文核实结果（2026-07-09 核查 arXiv:2603.22435v2）

| 用户草稿中的描述 | CaP-X 原论文实际情况 | 处理方式 |
|---|---|---|
| "在LIBERO的40个任务中平均成功率仅约20%，5个任务0%" | CaP-X 主基准为 7 个 Robosuite 任务；LIBERO-PRO（30任务，含扰动）是扩展评估；CaP-Agent0（最强配置）约 12-26%；原论文无"LIBERO-40基础版"的 20%/5任务0% 数据 | **已改为占位符 [B]%**，注明来源为"我们的实验评估" |
| "CaP-X 完全丢弃整次尝试" | 多轮模式在一次 trial 内不重置环境；但整次失败后物理进度确实丢弃 | **已修正**：改为"以整段程序为粒度"+"物理进度消失" |
| "CaP-X 2026年4月" | arXiv 提交时间：2026年3月23日（v1） | 已删除月份，仅保留 `\cite{fu2026capx}` |
| CaP-Agent0 可核实数据 | Peg Insertion 0%，Two-Arm Handover 20%，7任务平均 68%（论文 Table H.4）；LIBERO-PRO：12-26% | 可引用为脚注；当前版本不直接引用，避免混淆测量来源 |

### 关于 KITE 名称

已完全移除 KITE 及其缩写展开（Keyframe-Informed Trajectory Engineering）。  
替换为描述性术语：**"两级关键帧定位与双通道自适应修复机制"** / **"Two-Level Failure Localization and Dual-Channel Adaptive Repair"**。  
若后续需要便于记忆的名称，候选方案：**SLADE**（Stage-Level Adaptive Diagnosis and rEpair）、**FOCAL**（Failure-Oriented Critical frAme Localization）、或直接沿用描述性文字（当前方案）。

### Quantitative Placeholder Summary

| 占位符 | 含义 | 当前锚定值 |
|---|---|---|
| `[X]%` | Stage-CAP 完整系统 LIBERO TSR | 预实验约 85%（2轮反馈，1个初始状态） |
| `[B]%` | CaP-X baseline TSR（Stage-CAP 自身实验测量） | 待 E1 实验完成；预期 Stage-CAP - 15~25 pp |
| `[Δ]` | Stage-CAP vs CaP-X TSR 差值（百分点） | 预期 15—25 pp（见 stage_cap_experiments.md E1） |
| `[Z]%` | 下游 SmolVLA / BC PSR 相对提升幅度 | 待 E2 实验完成后确定 |

### Citation Key Reference

| 引用键 | 对应工作 | arXiv |
|--------|---------|-------|
| `james2019rlbench` | RLBench | 1909.12271 |
| `mandlekar2023mimicgen` | MimicGen | 2310.17596 |
| `liang2023codeaspolicies` | Code as Policies | 2209.07753 |
| `huang2023voxposer` | VoxPoser | 2307.05973 |
| `fu2026capx` | CaP-X | 2603.22435 |
| `luo2024serl` | SERL | — |
| `black2024pi0` | π₀ | 2410.24164 |
| `liu2023libero` | LIBERO | — |
| `zhu2020robosuite` | RoboSuite | 2009.12293 |
| `lerobot2024` | LeRobot | — |

### Removed / Changed Items vs Previous Draft

- **GenSim / RoboGen**: 已从引言移除（沿用上一版本作者指令）
- **OpenVLA / Octo**: 已替换为 BC / π₀（沿用上一版本修订）
- **Open X-Embodiment / RoboCasa365 数据量声明**: 已移除（沿用上一版本）
- **Contribution #4（部分轨迹利用）**: 保持移除；代码确认转换流水线只保留 `taskcompleted_1`，部分前缀利用未实现
- **KITE 缩写**: 完全移除，替换为描述性机制名称

### Word Count Reference

- 英文正文约 **780 词**（AAAI 700–900 词范围内）
- 中文版本约 **810 字**

---

*文档位置：`report/stage_cap_introduction.md`*  
*关联文件：`report/stage_cap_experiments.md`, `report/stage_cap_method.md`*
