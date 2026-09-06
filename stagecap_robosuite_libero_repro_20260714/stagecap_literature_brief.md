# StageCap 论文动机调研备忘录

## 0. 一句话结论

StageCap 最稳的论文定位不是“又一个生成机器人数据的方法”，而是：面向仿真中的 VLA 训练数据生产，现有数据引擎多数依赖人工示教、手工 waypoint/heuristic、轨迹重放变换或一次性 LLM 生成管线；它们能扩大数据规模，但对长时程操控中的阶段性失败缺少细粒度诊断与局部修复机制。StageCap 可以把 Code-as-Policy 从“直接执行机器人任务”重新定位为“可验证、可修改、可复盘的仿真成功轨迹生产程序”。

## 1. 现有方法版图

| 方法类型 | 代表工作 | 核心方法 | 主要产出 | 典型缺陷 |
|---|---|---|---|---|
| 人工示教数据集 | RoboTurk, robomimic, RT-1/Open X-Embodiment | 人远程操作机器人，收集真实或仿真轨迹 | 高质量 imitation learning 数据 | 成本高，扩展慢；长时程任务失败和恢复依赖人；数据多样性受采集场景限制 |
| 手工任务 + motion planner/waypoints | RLBench, ManiSkill/ManiSkill2 | 任务创建时人工设计 waypoint、奖励、成功条件，用规划器生成 demo | 可大规模生成标准 benchmark demos | 强依赖人工任务工程；对新任务迁移需要重写启发式；失败通常不是被局部修复，而是重新规划或丢弃 |
| 示教重放/变换型数据引擎 | MimicGen, DexMimicGen, RoboCasa/RoboCasa365 中的 MimicGen 数据扩增 | 将少量 human demos 分割为 object-centric subtasks，按新物体位姿变换、拼接并执行 | 从少量示教生成大量成功轨迹 | 依赖种子示教、已知子任务序列、对象位姿和分段检测；失败后主要按整条轨迹成功与否筛选，缺少代码级局部修复 |
| LLM 生成任务/环境/监督 | GenSim, RoboGen, RoboCasa task blueprints | LLM 生成任务、场景、课程、奖励/监督，配合 RL、规划、轨迹优化 | 更多任务和场景，以及用于训练的专家 demo 或策略 | 重点在任务/场景多样性，不一定解决单条轨迹生成失败的细粒度定位；管线工程复杂，验证多在任务级或 episode 级 |
| LLM/CaP 直接控制 | Code as Policies, VoxPoser, ProgPrompt, SayCan | LLM 生成可执行程序或计划，调用感知、几何、控制 primitive | 机器人任务执行策略或 waypoint/trajectory | 原始 CaP 往往把任务当作整体程序生成，失败后缺少阶段边界、失败帧复盘和局部代码修复 |

## 2. 核心文献摘要

### MimicGen

MimicGen 是最需要重点对比的工作。它从少量人类示教出发，把轨迹解析成 object-centric segments，然后在新场景中选择参考 segment，按物体位姿做空间变换并拼接执行；若任务最终成功，则把状态和动作序列加入生成数据集。论文报告用约 200 条源示教生成 50K+ demos，覆盖 18 个任务、多种场景和机器人。

对 StageCap 的启发与缺口：

- 它明确把问题定义为 data generation system，而不是 policy deployment。
- 它已经使用“子任务/分段”的思想，但分段主要服务于轨迹变换，不是服务于失败诊断与代码修复。
- 它假设任务由已知 object-centric subtask sequence 构成，并且在数据生成时可观测相关物体位姿。
- 执行失败时，机制主要是最终成功筛选；论文也指出轨迹偏差、控制和运动学问题会导致任务失败，成功轨迹才被保留。StageCap 可以把这里推进为“失败帧定位 -> 阶段诊断 -> 局部 policy 代码修复 -> 继续生成”。

来源：[MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations](https://arxiv.org/abs/2310.17596)

### RLBench / ManiSkill2

RLBench 提供 100 个手工设计任务，每个任务通过任务创建时给定的 waypoint 和 motion planner 获得近乎无限 demo。ManiSkill2 提供 20 个 manipulation task families、2000+ object models 和 4M+ demonstration frames，强调动态仿真、多输入和统一评测接口。

对 StageCap 的启发与缺口：

- 这类方法擅长 benchmark 标准化与可重复性。
- 缺点是任务和轨迹生成逻辑通常需要人工工程化：waypoint、reward、success checker、planner constraint。
- 它们不是以 LLM 代码修复为核心，面对新任务时很难自动把失败归因到“哪个阶段的哪段控制逻辑”。

来源：[RLBench](https://arxiv.org/abs/1909.12271), [ManiSkill2](https://arxiv.org/abs/2302.04659)

### GenSim / RoboGen

GenSim 用 LLM 的 grounding 和 coding ability 自动生成仿真环境和 expert demonstrations，强调从 scene-level diversity 推进到 task-level diversity，并有 goal-directed 与 exploratory 两种生成模式。RoboGen 提出 propose-generate-learn 循环：提出任务、生成场景、分解子任务、选择 RL/规划/轨迹优化等学习方式并生成监督。

对 StageCap 的启发与缺口：

- 它们已经说明 LLM 适合自动生成任务、场景、代码和监督。
- 但它们的主战场是“任务/场景/技能空间扩张”，不是“单条长时程成功轨迹的失败定位与局部代码修复”。
- StageCap 应避免声称“第一个用 LLM 生成仿真数据”。更稳的 novelty claim 是：把 CaP 组织成阶段验证闭环的数据生产 agent，并把失败帧诊断与局部代码修复系统化整合到轨迹生成流程中。

来源：[GenSim](https://arxiv.org/abs/2310.01361), [RoboGen](https://arxiv.org/abs/2311.01455)

### RoboCasa / RoboCasa365

RoboCasa 把高保真厨房仿真、AI 生成资产、LLM 辅助复合任务、人工示教和 MimicGen 数据扩增结合起来。RoboCasa365 在 2026 年进一步扩展到 365 个任务、2500 个厨房场景、600+ 小时人类示教和 1600+ 小时合成示教。

对 StageCap 的启发与缺口：

- 说明当前趋势是把“仿真平台 + 大规模合成演示 + VLA/通用策略训练”连成数据工厂。
- 但 RoboCasa/RoboCasa365 主要是平台和 benchmark；其合成数据仍大量依赖 MimicGen。
- StageCap 可以作为这类平台上的“trajectory production engine”，补足高成功率、低人工、可诊断修复的轨迹生产环节。

来源：[RoboCasa](https://arxiv.org/abs/2406.02523), [RoboCasa365](https://arxiv.org/abs/2603.04356)

### Code as Policies / VoxPoser

Code as Policies 表明 LLM 生成的 policy code 可以处理感知输出、调用控制 primitive、表达反馈循环，并利用 NumPy/Shapely 等库进行空间几何推理；它既能表示 reactive policies，也能表示 waypoint-based policies。VoxPoser 进一步利用 LLM 的代码能力与 VLM 组合 3D value maps，生成 6-DoF end-effector waypoint 序列。

对 StageCap 的启发与缺口：

- CaP 的优势是程序可读、可执行、可局部修改，天然适合仿真中的离线数据生产。
- 原始 CaP 更多展示的是“给自然语言指令，生成并执行机器人策略”；StageCap 应重新定义任务：不是最终部署策略，而是生产 VLA 训练样本。
- 一旦目标从 deployment 变成 data engine，使用仿真特权状态、阶段 success checker、失败帧复盘就更合理，因为生成器可以用 privileged information，但最终训练出的 VLA 不需要依赖这些信息。

来源：[Code as Policies](https://arxiv.org/abs/2209.07753), [VoxPoser](https://arxiv.org/abs/2307.05973)

## 3. 现有数据引擎的可归纳缺陷

### 缺陷 A：成功/失败反馈粒度太粗

多数数据引擎以 episode 成功作为最终筛选标准。它们能丢弃失败轨迹，却不一定知道失败发生在“接近、抓取、运输、放置、验证”中的哪一段，更不一定能把这段失败转化为可修改的控制逻辑。

StageCap 对应补足：阶段边界验证。每个阶段都有 precondition、postcondition 和 success checker，使失败从 episode-level signal 变成 stage-level signal。

### 缺陷 B：失败样本没有被充分利用

失败轨迹通常被视为无效尝试或训练负样本，而不是作为修复生成器的诊断材料。对于数据生产系统，这会造成大量仿真预算浪费。

StageCap 对应补足：失败帧定位复盘。定位 earliest failure frame 或关键诊断帧，把图像、状态、接触、物体位姿、动作历史反馈给代码生成/修复模块，形成“失败 -> 修复”的闭环。

### 缺陷 C：新任务需要较多人工工程或种子示教

MimicGen/DexMimicGen 需要少量高质量源示教和可分段结构；RLBench/ManiSkill 需要任务特定 waypoint、planner、success checker；RoboCasa 规模更大，但仍依赖人工示教与 MimicGen 扩增。

StageCap 对应补足：用 CaP 生成初始阶段 policy，再通过仿真验证局部修复。它不一定完全摆脱人工，但可以减少人工从“写整套 waypoint/示教”变成“定义任务接口、阶段验证器和安全 primitive”。

### 缺陷 D：场景多样性和轨迹可靠性常被分开处理

GenSim/RoboGen/RoboCasa 强调任务、场景、资产扩张；MimicGen 强调示教变换扩增。但当场景变化导致轨迹失败时，常见做法是重试、筛选或换规划策略，而不是自动修改失败阶段代码。

StageCap 对应补足：训练场景多变 + 阶段局部修复。多样化不是只随机化初始状态，而是把多样化带来的失败转化为阶段 policy 的适配压力。

### 缺陷 E：生成器与最终学习器之间缺少显式数据质量闭环

很多工作证明“更多 synthetic demos 有用”，但数据生产过程本身的质量指标常停留在数量、成功率或下游 policy success。缺少对“每一阶段为何成功/失败、哪些修复提高了生成率、哪些场景覆盖不足”的过程级分析。

StageCap 对应补足：记录 stage-level success rate、repair count、failure type distribution、scene coverage、trajectory diversity 和 downstream VLA performance，构成数据引擎本身的评价体系。

## 4. StageCap 是否弥补了缺陷

| 现有缺陷 | StageCap 创新 | 能否弥补 | 需要实验证明 |
|---|---|---|---|
| episode-level 成败反馈粗 | 阶段验证闭环 | 能，若每阶段 checker 定义清楚 | stage-level failure localization accuracy；相比 whole-program retry 的生成成功率 |
| 失败轨迹被丢弃 | 失败帧定位复盘 | 能，若诊断帧能稳定指向错误阶段和错误原因 | repair success rate；平均每条成功 demo 的尝试次数/LLM 调用次数 |
| 整体 policy 失败只能重写 | 阶段修改 policy 代码 | 能，若代码结构模块化并支持 patch | local patch vs global regeneration 的成功率、成本、引入回归率 |
| 新场景变化导致轨迹脆弱 | 训练场景多变 | 部分能，需证明不是简单 domain randomization | seen/unseen scene success；object pose/layout/asset split 泛化 |
| 依赖人工示教/手工 waypoint | CaP 初始化 + 修复 | 部分能，不能夸大为完全免人工 | 人工成本对比：task setup time、seed demos 数量、manual interventions |

更稳的表述：StageCap 不是完全替代 MimicGen/RLBench/RoboCasa，而是补齐它们的数据生产闭环。它可以和这些平台结合，在 robosuite/RoboCasa/ManiSkill/RLBench 中生成 VLA 训练轨迹。

## 5. 为什么 Code-as-Policy 适合“轨迹构建”而不只是“机器人控制”

这点很关键，可以作为论文单独一节。

1. 轨迹构建是离线生产任务，不是在线部署任务。生成器可以使用仿真特权信息、object pose、contact state、success checker、debug logs；最终交付给 VLA 的只是观测-语言-动作序列。

2. 长时程操作天然是程序结构。抓取、移动、放置、验证，本身就是可组合的阶段；代码比端到端神经策略更容易表达阶段边界、循环、条件分支、异常处理和几何约束。

3. CaP 的可编辑性匹配局部修复。失败发生在放置阶段时，只改 `place_policy()` 或该阶段参数，而不需要重采示教、重写整条轨迹或重训 policy。

4. 仿真环境提供可执行验证器。每次生成的代码可以立即运行、检查阶段后置条件、记录失败帧，然后把具体证据反馈给 LLM 或修复器。

5. 数据引擎看重“高产合格轨迹”，不是“策略本身最优”。即使 CaP 生成的代码不是最优控制策略，只要能稳定产出多样、成功、可学习的轨迹，它就是合格的数据引擎。

建议论文里这样说：Code-as-Policy is particularly suitable for simulation-based trajectory construction because executable policy code provides an inspectable and patchable intermediate representation between natural-language task intent and low-level robot trajectories.

## 6. 推荐论文叙事骨架

### Introduction 逻辑

1. VLA/通用机器人策略需要大量图像-语言-动作轨迹，但真实采集昂贵。
2. 仿真数据引擎成为扩大机器人数据的重要路径，包括 motion-planner demos、MimicGen 式示教扩增、GenSim/RoboGen 式 LLM 任务生成、RoboCasa 式大规模仿真平台。
3. 然而，现有数据引擎对长时程任务失败的处理仍然粗粒度：失败常在 episode 末端被发现，失败轨迹被筛掉或整体重试，缺少阶段级定位和局部修复。
4. 长时程操作任务天然具有阶段结构。若把轨迹生产程序组织为阶段化 Code-as-Policy，并在每个阶段边界验证，就可以把失败从“整条轨迹失败”转化为“某阶段代码/参数需要修复”。
5. 因此提出 StageCap，一个面向仿真 VLA 数据构建的阶段化 CaP 数据生产 agent。

### Contributions 推荐写法

1. 提出 StageCap：将 Code-as-Policy 重新定位为仿真轨迹数据引擎，而非仅作为在线机器人控制器。
2. 提出 stage-wise policy repair：把长时程任务拆成可验证阶段，并在失败时只修改相关阶段 policy code。
3. 提出 failure-frame-driven replay：从失败执行中定位诊断帧，结合仿真状态和视觉观测生成修复提示。
4. 构建多变训练场景数据生产流程，并验证其在成功率、生成效率、数据多样性和下游 VLA 学习上的收益。

## 7. 必做实验

| 实验 | 对比基线 | 指标 |
|---|---|---|
| 轨迹生成成功率 | whole CaP retry, MimicGen/hand planner where applicable | success demos / attempts, attempts per success |
| 修复效率 | global code regeneration, no failure frame, no stage checker | LLM calls per success, wall-clock time, repair success |
| 失败定位有效性 | final-only success signal | localization accuracy, earliest failure frame distance |
| 多场景泛化 | fixed scene, randomization without repair | unseen object pose/layout/category success |
| 下游学习收益 | human-only demos, generated demos without repair, MimicGen demos | VLA/BC policy success, robustness under scene shift |

## 8. 风险与写作边界

- 不要说“首次用 LLM 生成机器人仿真数据”：GenSim/RoboGen 已经做了。
- 不要说“首次用阶段分解”：MimicGen、RoboCasa、RoboGen、SayCan 等都有 subtask/skill decomposition。
- 可以说“针对仿真轨迹数据生产，把阶段验证、失败帧诊断和局部 CaP 代码修复整合为闭环”。
- 如果实验只在少数任务上做，标题和摘要要避免 “generalist data engine” 过度承诺，可写 “towards scalable simulation trajectory generation”。

## 9. 参考文献链接

- [MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations](https://arxiv.org/abs/2310.17596)
- [DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning](https://arxiv.org/abs/2410.24185)
- [RLBench: The Robot Learning Benchmark & Learning Environment](https://arxiv.org/abs/1909.12271)
- [ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills](https://arxiv.org/abs/2302.04659)
- [GenSim: Generating Robotic Simulation Tasks via Large Language Models](https://arxiv.org/abs/2310.01361)
- [RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation](https://arxiv.org/abs/2311.01455)
- [RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://arxiv.org/abs/2406.02523)
- [RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](https://arxiv.org/abs/2603.04356)
- [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753)
- [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://arxiv.org/abs/2307.05973)
- [SayCan: Do As I Can, Not As I Say](https://arxiv.org/abs/2204.01691)
- [ProgPrompt: Generating Situated Robot Task Plans using Large Language Models](https://arxiv.org/abs/2209.11302)
