# 机器人操控轨迹生成方法近三年综述（2023–2026）

> **撰写背景**：本报告面向 Stage-CAP 项目，系统梳理近三年（2023–2026）用于仿真环境中生成高质量机器人操控训练轨迹的主要方法，重点关注与 LIBERO/Robosuite 仿真平台相关的工作，以及 VLA 训练数据生产场景下的轨迹生成技术。报告覆盖五大路线：人工示教/遥操作、运动规划与 waypoint 系统、示教重放/变换型数据引擎、LLM/VLM 代码生成路线、以及强化学习路线；并专门讨论 VLA 下游消费侧对训练数据的需求，最后给出方法横向对比与 Stage-CAP 定位分析。

---

## 目录

1. [研究背景与问题定义](#1)
2. [方法全景图](#2)
3. [路线一：人工示教与遥操作数据收集](#3)
4. [路线二：运动规划与 Waypoint 系统](#4)
5. [路线三：示教重放与变换型数据引擎](#5)
6. [路线四：LLM/VLM 代码生成轨迹方法](#6)
7. [路线五：强化学习辅助轨迹生成](#7)
8. [VLA 下游训练数据需求与代表系统](#8)
9. [自进化数据引擎：新兴趋势（2025–2026）](#9)
10. [方法横向对比](#10)
11. [Stage-CAP 在版图中的定位](#11)
12. [未来研究方向](#12)
13. [参考文献](#13)

---

## 1. 研究背景与问题定义

机器人学习的核心瓶颈始终是高质量训练轨迹的获取成本。近三年，随着视觉-语言-动作（VLA）模型的兴起，对大规模、多任务、多场景仿真轨迹的需求急剧增长。与此同时，大语言模型（LLM）和视觉语言模型（VLM）的代码生成与空间推理能力也为自动化轨迹生产提供了新途径。

**核心研究问题**：如何以最低人工成本、最高成功率，在仿真环境中自动批量生成高质量的机器人操控示范轨迹，用于后续策略学习（尤其是 VLA fine-tuning）？

**本报告关注的场景**：
- 仿真平台：LIBERO、Robosuite、ManiSkill、IsaacGym 等
- 任务类型：桌面抓取、放置、装配、开关门/抽屉等操控任务
- 下游用途：VLA 模型（OpenVLA、Pi0、Octo 等）的 fine-tuning 数据
- 时间范围：2023 年至 2026 年 7 月

---

## 2. 方法全景图

| 方法类别 | 代表工作 | 核心思路 | 主要优势 | 主要局限 |
|---|---|---|---|---|
| 人工示教/遥操作 | RT-1/Open X-Emb (2023), ACT/ALOHA (2023), RoboAgent (2023) | 人操控机器人收集轨迹 | 质量高、真实可信 | 成本极高，难以大规模扩展 |
| 运动规划+Waypoint | RLBench (2019), ManiSkill2 (2023) | 人工设计任务 + 规划器生成 demo | 可无限生成标准 demo | 需人工工程化每个任务 |
| 示教重放/变换 | MimicGen (CoRL 2023), DexMimicGen (ICRA 2025), RoboCasa (RSS 2024), RoboCasa365 (ICLR 2026) | 将少量示教变换到新场景 | 从少量种子大规模生成 | 依赖种子示教和已知子任务结构 |
| LLM/VLM 代码生成 | Code as Policies (2022), VoxPoser (2023), GenSim (ICLR 2024), RoboGen (ICML 2024), CAP-X (2026) | LLM 生成可执行机器人策略代码 | 无需人工示教，任务灵活 | 单次生成成功率有限 |
| 强化学习 | RoboGen (ICML 2024), SERL (2024), CaP-RL (2026) | RL 探索生成成功轨迹 | 可应对复杂接触动力学 | 样本效率低，奖励设计困难 |
| 自进化数据引擎 | Seed2Scale (2026), SAGE (2026) | 小模型采集 + 大模型评估闭环 | 从极少种子扩展，成功率自增长 | 技术较新，验证任务有限 |

---

## 3. 路线一：人工示教与遥操作数据收集

### 3.1 Open X-Embodiment（2023）

Google 等机构联合发布的跨平台机器人数据集，聚合了来自 22 个研究机构、22 种机器人形态的真实世界操控轨迹。数据集包含超过 100 万条轨迹，覆盖 500+ 任务。其意义在于证明了大规模跨机器人预训练可以显著提升策略泛化性，奠定了 OpenVLA、Octo 等 VLA 模型的数据基础。

**局限**：数据收集成本极高；不同平台的动作空间异构性带来对齐难题；真实世界数据难以按需扩展到新任务或新场景。

### 3.2 ACT / ALOHA（RSS 2023，arXiv: 2304.13705）

Tony Z. Zhao 等人提出使用低成本遥操作硬件（ALOHA）收集双臂灵巧操控示教，并提出 Action Chunking with Transformers（ACT）算法。ACT 以 Transformer 对动作序列建模，解决了高精度任务中策略误差累积的问题。仅需 10 分钟示教（约 50 条），即可在电池装配等精细操控任务上达到 80–90% 成功率。

**对 Stage-CAP 的启示**：ACT 证明了**动作分块**（action chunking）在高精度任务中的重要性；Stage-CAP 生成的轨迹需考虑与 ACT/Diffusion Policy 等 action representation 的兼容性。

### 3.3 RoboAgent（2023，arXiv: 2309.01918）

Bharadhwaj 等人用 7500 条遥操作示教训练了能完成 12 种技能、在 38 个任务上泛化的单一策略。关键创新是语义增强（semantic augmentation）——用图像生成模型扩充背景/纹理多样性，使数据在不新增实际收集成本的情况下多样化。

**对 Stage-CAP 的启示**：说明少量高质量示教 + 虚拟增强可以超越更多低质量数据；仿真数据引擎若能生成语义上多样的场景，则质量比数量更重要。

---

## 4. 路线二：运动规划与 Waypoint 系统

### 4.1 RLBench（2019，arXiv: 1909.12271）

100 任务机器人基准，每个任务通过手工设计的 waypoint 序列和运动规划器生成近乎无限数量的成功示教。RLBench 依然是评估和生成标准操控 demo 的主流基准之一。

**核心机制**：任务设计者手动定义关键 waypoint（如抓握点、目标位置）；在线运动规划器在每次场景随机化时求解可行轨迹；成功完成任务的轨迹记录为 demo。

**局限**：每个新任务需要人工工程化（waypoint、reward、success checker）；对新物体或场景的适应依赖重新编写启发式；不提供失败帧诊断或局部修复机制。

### 4.2 ManiSkill2（2023）

提供 20 个操控任务家族、2000+ 3D 物体模型、4M+ 示教帧，强调动态仿真精度和统一评测接口。使用运动规划器（motion planner）结合物体位姿信息批量生成成功轨迹。

**对 Stage-CAP 的启示**：ManiSkill 系列平台提供了标准化的多物体多场景仿真框架；其 demo 生成机制仍依赖手工任务工程，Stage-CAP 的 CaP 路线可在同类平台上减少这种人工。

---

## 5. 路线三：示教重放与变换型数据引擎

这一路线在近三年取得了最显著的规模突破，以 MimicGen 家族为代表。

### 5.1 MimicGen（CoRL 2023，arXiv: 2310.17596）

**核心方法**：
1. 将少量（约 200 条）人工示教分割为 object-centric subtask segments；
2. 对每个 subtask segment，按新场景中相关物体的位姿做空间变换（rigid body transformation）；
3. 将变换后的 segments 拼接并执行；
4. 最终成功的轨迹加入生成数据集。

**规模**：从约 200 条种子示教生成 50,000+ 条 demo，覆盖 18 个任务（咖啡制作、多部件装配等）、多种场景配置和机器人型号。

**实验结果**：在生成数据上训练的模仿学习策略性能与直接收集人工 demo 相当，且成本显著更低。

**关键局限**：
- 依赖已知的 object-centric subtask sequence 和物体位姿（特权信息）；
- 执行失败后只能整条丢弃，不能定位失败阶段并局部修复；
- 对高精度任务（如毫米级对齐）的轨迹变换可靠性受限。

**对 Stage-CAP 的启示**：MimicGen 是最直接的数据引擎参照。Stage-CAP 应将自己定位为不依赖种子示教、而用 CaP 代码从零生成初始轨迹的替代路径，并用阶段失败诊断替代整条筛选机制。

### 5.2 DexMimicGen（ICRA 2025，arXiv: 2410.24185）

将 MimicGen 扩展到双臂灵巧手（humanoid）场景。从 60 条人工示教生成 21,000 条 demo，覆盖多种双手协作操控任务。提出 real-to-sim-to-real pipeline，在真实人形机器人上验证了可罐子分拣任务。

**关键挑战**：双臂协调约束使简单的 per-arm 变换失效，需要考虑双臂耦合；灵巧手的接触丰富性大幅增加轨迹执行失败率。

**对 Stage-CAP 的启示**：说明对复杂灵巧操控，变换型引擎的适用性受限，LLM 代码生成路线或许更灵活。

### 5.3 RoboCasa（RSS 2024，arXiv: 2406.02523）

高保真厨房仿真框架，提供：
- 150+ 物体类别、数千 3D 资产（部分由 text-to-3D 模型生成）；
- 100 个任务（含 LLM 辅助设计的复合任务）；
- 高质量人工示教 + MimicGen 自动扩增。

**核心贡献**：证明合成仿真数据在大规模模仿学习中呈现清晰的 scaling trend（数据量增加 → 策略性能稳定提升），并展示了仿真数据到真实任务的迁移潜力。

**对 Stage-CAP 的启示**：RoboCasa 是"仿真数据工厂"路线的旗帜性工作；但其合成示教仍大量依赖 MimicGen，Stage-CAP 的 LLM 代码生成路线可作为 MimicGen 的互补或替代。

### 5.4 RoboCasa365（ICLR 2026，arXiv: 2603.04356）

RoboCasa 的大规模扩展版：
- **365 个日常任务**，涵盖厨房中的各类家务操作；
- **2500 个多样化厨房场景**；
- **600+ 小时人工示教数据**；
- **1600+ 小时合成示教数据**（主要通过 MimicGen）。

重要结论：
- **任务多样性**比数据量对 generalist policy 性能的影响更显著；
- **场景变化**（布局/光线/资产）是泛化的关键挑战；
- 现有 SoTA 方法在 365 任务上的总体成功率仍然较低，存在大量改进空间。

**对 Stage-CAP 的启示**：RoboCasa365 是 VLA 训练的重要参考 benchmark；其 1600+ 小时合成数据仍依赖 MimicGen，Stage-CAP 可定位为能在此类平台上更灵活生产轨迹的引擎，尤其对非 object-centric subtask structure 的任务。

---

## 6. 路线四：LLM/VLM 代码生成轨迹方法

这一路线是 Stage-CAP 的直接技术路线，也是近三年变化最快的方向之一。

### 6.1 Code as Policies（ICRA 2023，arXiv: 2209.07753）

奠基性工作，证明经过代码补全预训练的 LLM 可以将自然语言指令转化为可执行的机器人策略代码。策略代码可以调用感知 primitive、参数化控制 primitive API、利用 NumPy/Shapely 做空间几何推理、以及表达 reactive policies 和 waypoint-based policies。

**局限**：主要展示了"给指令→生成→执行"的一次性流程，不针对轨迹生产场景，缺少阶段验证和失败局部修复。

### 6.2 VoxPoser（CoRL 2023，arXiv: 2307.05973）

LLM + VLM 联合生成 3D value maps，用于 6-DoF end-effector waypoint 序列合成。LLM 推断抓握约束和避障条件，VLM 提供视觉 grounding，model-based planning 优化 closed-loop 轨迹。

**关键优势**：无需任务特定示教即可在真实机器人上执行多种操控任务；在 zero-shot 场景下成功率优于纯代码生成。

**局限**：对接触丰富操控的轨迹质量较差；value map 的合成依赖视觉定位精度。

### 6.3 GenSim（ICLR 2024，arXiv: 2310.01361）

**核心贡献**：用 GPT-4 自动生成仿真任务（任务描述、RLBench 任务代码、waypoint 序列），将现有 RLBench 任务集从 10 个扩展到 100+。支持两种生成模式：goal-directed（给定目标任务，LLM 提出分解课程）和 exploratory（LLM 从已有任务 bootstrap，迭代提出新颖任务）。

**实验结果**：在 GPT-4 生成的仿真任务上进行多任务预训练，在未见过的长时程真实机器人任务上性能提升 25%。

**关键局限**：
- 重点在任务/场景空间的多样性扩展，不专门解决单条轨迹生成中的失败定位与局部修复；
- GPT-4 生成的任务代码仍需验证，失败时通常重新生成而非局部修复；
- 任务生成成功率约 30–50%。

**对 Stage-CAP 的启示**：Stage-CAP 应避免声称"第一个用 LLM 生成仿真数据"，而是强调"阶段验证闭环 + 失败帧诊断 + 局部代码修复"的系统化创新。

### 6.4 RoboGen（ICML 2024，arXiv: 2311.01455）

提出 **propose-generate-learn 自主学习循环**：
1. **Propose**：LLM 提出新技能和对应任务；
2. **Generate**：LLM 生成仿真场景（物体配置、空间布局）；
3. **Learn**：子任务分解后，按难度选择最优学习方式（强化学习 / 运动规划 / 轨迹优化），生成训练监督。

**核心优势**：一个统一框架，能根据任务自动选择最适合的轨迹生成策略。持续循环可生成无限多的技能示教。

**关键局限**：管线工程复杂，验证粒度在任务级，不专注于单条长时程轨迹的阶段性失败修复。

### 6.5 CAP-X（arXiv 2026，arXiv: 2603.22435）

**最直接相关工作**。CAP-X 是系统化研究 Code-as-Policy 智能体的开放框架，包含：

- **CaP-Gym**：包含 Robosuite、LIBERO-PRO、BEHAVIOR 的 39 个 Gymnasium 环境；
- **CaP-Bench**：系统化评测，涵盖 8 个层级（S1-S4 单轮、M1-M4 多轮）；
- **CaP-Agent0**：无训练 agentic framework，配备多轮视觉差分、自动合成技能库、并行集成推理；
- **CaP-RL**：通过 GRPO 对代码生成模型做强化学习后训练，利用仿真环境奖励，sim-to-real 迁移 gap 小。

**对 Stage-CAP 的启示**：
- Stage-CAP 从 CAP-X/CaP-Agent0 派生而来，在其基础上增加了阶段验证和 KITE 反馈，专注于数据生产而非 agent 评测；
- CAP-X 是 LIBERO 上最直接的基线/前驱工作，Stage-CAP 论文需清楚区分两者目标：CAP-X 面向 agent benchmark，Stage-CAP 面向 VLA 训练数据生产。

### 6.6 Reflexion（NeurIPS 2023，arXiv: 2303.11366）

Reflexion 框架：LLM agent 将任务反馈（execution outcome）转化为语言反思（verbal reflection），存入 episodic memory，引导下一次尝试。在序列决策、代码生成、语言推理任务上显著优于无 reflection 的基线，在 HumanEval 上达到 91% pass@1。

**对 Stage-CAP 的启示**：Stage-CAP 的失败帧诊断 → 代码修复反馈循环在概念上与 Reflexion 类似；Stage-CAP 的创新是将 Reflexion 机制专门化为机器人仿真轨迹生产的**阶段级**结构化失败反馈，而不是通用的语言反思。

---

## 7. 路线五：强化学习辅助轨迹生成

### 7.1 RL in RoboGen

RoboGen 的 propose-generate-learn 框架在某些子任务上选择 RL 而非运动规划作为学习方式。RL 适合**接触丰富**（contact-rich）的子任务，如插孔、螺母装配，这类任务通过运动规划难以生成高质量轨迹。

### 7.2 SERL（Sample-Efficient Robot Learning，2024）

通过结合少量人工 demo 和 off-policy RL（RLPD），在真实机器人上实现小时级训练收敛。在 PCB 插拔、USB 插拔等高精度接触任务上成功率 > 90%。

**对仿真轨迹生产的意义**：SERL 风格的方法可在仿真中快速为难以规划的任务生成成功轨迹；但奖励函数设计仍需人工工程。

### 7.3 Staged Reward（DrS 类方法，2024）

将 RL 的奖励分解为阶段性奖励（staged reward），使 RL 能更高效地解决长时程操控任务。与 Stage-CAP 的阶段验证思想在哲学上相近，但实现机制不同：DrS 是 RL 的奖励工程，Stage-CAP 是 CaP 的阶段代码验证。

### 7.4 CaP-RL（CAP-X 框架，2026）

在代码生成模型上进行 GRPO 强化学习后训练，以仿真环境的任务成功奖励作为强化信号。相当于把 LLM 代码生成 agent 当成 RL 的 policy，通过采样-执行-评分-梯度更新的循环来提升代码生成成功率。在 LIBERO、Robosuite 任务上展示了显著性能提升，sim-to-real 迁移 gap 小。

---

## 8. VLA 下游训练数据需求与代表系统

理解 VLA 的训练数据需求，对于设计轨迹生产引擎至关重要。

### 8.1 Octo（RSS 2024，arXiv: 2405.12213）

基于 800k 条真实世界机器人轨迹（Open X-Embodiment）训练的 Transformer 策略。支持语言指令或目标图像输入，可在数小时内在消费级 GPU 上通过 fine-tuning 适配新任务/新机器人平台。在 9 个机器人平台上验证了有效的 policy initialization。

### 8.2 OpenVLA（CoRL 2024，arXiv: 2406.09246）

7B 参数 VLA，在 970k 真实世界机器人示教上预训练（基于 Llama 2 + DINOv2 + SigLIP 编码器）。在 29 个任务上以 16.5% 绝对成功率优于 RT-2-X（55B），参数量降低 7 倍。支持通过 LoRA 在消费级 GPU 上进行高效 fine-tuning。

**对仿真数据生产的意义**：OpenVLA 的 fine-tuning 路线明确说明仿真 demo（如 LIBERO 场景）可以作为 fine-tuning 种子数据；Stage-CAP 生成的轨迹可直接转换为 OpenVLA 的 fine-tuning 格式。

### 8.3 π₀（Pi0，RSS 2025，arXiv: 2410.24164）

Flow matching 架构的 VLA，在 pre-trained VLM 基础上引入 flow-based action head，在多种灵巧操控任务（折叠衣物、桌面清洁、装箱）上展示了 zero-shot 预训练和 fine-tuning 能力。

**对轨迹生产的意义**：π₀ 的 flow matching action head 对高质量、多模态动作分布有更强的表达能力；这对数据生产引擎提出更高要求——需要动作轨迹在同一任务的不同初始条件下保持一致性。

### 8.4 VLA 训练数据对轨迹的核心需求总结

| 需求维度 | 具体要求 |
|---|---|
| 数量 | 通常需要数千至数万条同任务或跨任务 demo 才能有效 fine-tune |
| 多样性 | 场景（物体位置/姿态/光线）的多样性对泛化至关重要 |
| 成功率 | 训练数据中低质量/失败轨迹会显著降低 fine-tuned 策略性能 |
| 动作格式 | 通常为 end-effector 位置+姿态+夹爪状态（LeRobot、HF 格式） |
| 观测格式 | 需要同步的 agent-view（和可选的 wrist-view）图像序列 |
| 语言标注 | 任务自然语言描述（task language instruction） |
| 元数据 | 任务阶段信息、成功标志、物体标注（可选） |

---

## 9. 自进化数据引擎：新兴趋势（2025–2026）

### 9.1 Seed2Scale（2026，arXiv: 2603.08260）

**核心思路**：异构协作的"小模型采集 + 大模型评估 + 目标模型学习"循环：
- **SuperTiny VLA**（轻量 collector）：在并行仿真环境中高效探索，利用其强归纳偏置做 robust 探索；
- **预训练 VLM Verifier**：对大量生成轨迹进行自主成功/失败判断和质量打分；
- **Target VLA**：在经过筛选的高质量数据上 fine-tune，性能持续提升。

**实验结果**：从 4 条种子示教出发，经过多轮迭代，目标模型成功率提升 131.2%，显著优于现有数据增强方法。

**对 Stage-CAP 的启示**：Seed2Scale 与 Stage-CAP 的轨迹生产引擎定位高度互补——Stage-CAP 侧重 CaP 代码生成+阶段验证的轨迹生产，Seed2Scale 侧重 VLA collector+VLM verifier 的自进化循环。两者可以结合：Stage-CAP 生成初始高质量种子 → Seed2Scale 放大。

### 9.2 SAGE（Scalable Agentic 3D Scene Generation，2026，arXiv: 2602.10116）

Agentic 3D 场景生成框架（NVIDIA），给定用户指定的 embodied 任务（如"把碗放到桌子上"），自动生成仿真就绪的 3D 环境，用于策略训练。多个生成器（布局、物体组合）+ 评价器（语义合理性、视觉真实性、物理稳定性）迭代 self-refine。

**实验结果**：在 SAGE-10k 数据集上训练的策略呈现清晰的 scaling trend，能泛化到未见过的物体和布局。

**对 Stage-CAP 的启示**：SAGE 解决的是"场景生成"问题，Stage-CAP 解决的是"轨迹生成"问题；两者互补——SAGE 生成多样化场景 → Stage-CAP 在这些场景中生成成功轨迹。

---

## 10. 方法横向对比

### 10.1 核心维度对比

| 方法 | 是否需要种子示教 | 是否需要人工任务工程 | 失败处理粒度 | 最大规模 | VLA 友好性 |
|---|---|---|---|---|---|
| RLBench (规划器) | 无需 | **高**（每任务 waypoint） | Episode 级筛选 | 无限（规划器保证） | 中 |
| MimicGen | **需要** (~200条) | 中 | Episode 级筛选 | **50K+ demos** | **高** |
| DexMimicGen | **需要** (~60条) | 高 | Episode 级筛选 | **21K demos** | 中 |
| GenSim | 无需 | 低（LLM 自动） | Episode 级筛选 | 100+ 任务 | 中 |
| RoboGen | 无需 | 低（LLM 自动） | Episode 级 | 任务依赖 | 低 |
| CAP-X Agent0 | 无需 | 低 | 多轮视觉 differencing | 中 | 中 |
| Seed2Scale | **需要** (4条) | 低 | Episode 级（VLM 打分） | **自动扩展** | **高** |
| **Stage-CAP** | **无需** | 低 | **阶段级（stage-level）** | 中（LIBERO 40+） | **高** |

### 10.2 失败处理能力专项对比

| 方法 | 失败检测粒度 | 失败是否用于修复 | 修复机制 |
|---|---|---|---|
| MimicGen / DexMimicGen | Episode 级 | 否，直接丢弃 | 无 |
| GenSim | Episode 级 | 部分（重新生成代码） | 整体重新生成 |
| RoboGen | Episode 级 | 部分（RL 重新探索） | RL 重新探索 |
| Reflexion | Episode/step 级（语言反馈） | 是 | 语言反思→下一轮策略 |
| CAP-X Agent0 | 多轮视觉差分 | 是 | 局部视觉诊断→代码修改 |
| **Stage-CAP** | **阶段级**（stage boundary） | **是**（失败帧+结构化诊断） | **局部 policy 代码修复** |

---

## 11. Stage-CAP 在版图中的定位

### 11.1 技术独特性

Stage-CAP 在上述方法版图中占据了一个独特位置：

**与 MimicGen/DexMimicGen 的区别**：
- MimicGen 从人工示教变换生成数据；Stage-CAP 从 LLM 代码生成初始轨迹，无需种子示教；
- Stage-CAP 引入阶段验证闭环，将失败从 episode 级降低到 stage 级，实现局部修复而非整条丢弃。

**与 GenSim/RoboGen 的区别**：
- GenSim/RoboGen 专注于任务/场景空间的多样性扩展；Stage-CAP 专注于单条长时程轨迹的**可靠生成**；
- Stage-CAP 将特权诊断信息（仿真可读物体位姿、成功条件、接触状态）合法地用于数据生产阶段，而不涉及最终策略推理。

**与 CaP-Agent0（CAP-X）的区别**：
- CAP-X 是评测 benchmark，CaP-Agent0 是通用 agent 框架；Stage-CAP 是专门面向 VLA 训练数据生产的 pipeline；
- Stage-CAP 额外引入了 VLA 轨迹记录（steps.jsonl、wrist camera、LeRobot 格式）、阶段边界 KITE 反馈、以及 VLA 数据质量指标。

**与 Seed2Scale 的区别**：
- Seed2Scale 是 VLA-driven 的自进化采集引擎；Stage-CAP 是 LLM-code-driven 的仿真数据生产引擎；
- 两者互补：Stage-CAP 侧重高成功率的受控生成（有特权信息辅助），Seed2Scale 侧重低人工的大规模扩展。

### 11.2 Stage-CAP 的四层创新

1. **阶段规划（Stage Planning）**：将任务语言分解为明确的阶段（接触、抓握、运输、放置、验证），每个阶段有清晰的 precondition 和 postcondition；

2. **CaP 代码生成 + 执行（MIMO 多输出）**：生成的 Python CAP 代码调用机器人 motion API，在数据生产模式下可读取 simulator 只读特权信息（物体位姿、抽屉进度、BDDL 关系）；

3. **阶段验证 + Online KITE 反馈**：每个阶段执行后读取 STAGE_STATUS 结构化证据；若阶段失败，定位失败帧，将紧凑诊断信息反馈入下一轮代码生成；

4. **VLA 数据录制（Step Recording）**：成功轨迹以 steps.jsonl + agent-view + wrist-view 格式录制，支持 LeRobot 格式转换，可直接用于 OpenVLA/Octo 等 VLA 的 fine-tuning。

### 11.3 Stage-CAP 的定位陈述

> Stage-CAP 不是"又一个生成机器人数据的方法"，而是面向仿真中的 VLA 训练数据生产，把 Code-as-Policy 从"直接执行机器人任务"重新定位为"可验证、可修改、可复盘的仿真成功轨迹生产程序"。它将阶段验证闭环、失败帧诊断、局部代码修复系统化整合到轨迹生成流程中，是对 MimicGen（依赖种子示教）和 GenSim/RoboGen（缺乏细粒度失败修复）两类主流方法的互补。

---

## 12. 未来研究方向

### 12.1 阶段验证闭环的精细化

- **阶段 success checker 的自动化**：目前阶段边界条件仍需人工定义接口，如何从任务描述自动生成 BDDL-style checker 是开放问题；
- **多步失败模式分类**：建立失败模式 taxonomy（接触失败、视觉定位失败、运动学奇异、物理碰撞等），针对性生成修复代码。

### 12.2 与 VLA 的更紧密闭环

- **VLA-in-the-loop 验证**：Stage-CAP 生成的轨迹直接 fine-tune VLA，再用 VLA 的成功率反馈哪类场景需要更多轨迹；
- **Sim-to-real gap 缩减**：在仿真中使用 domain randomization 和真实感渲染，减少下游 VLA 的 sim-to-real 迁移代价。

### 12.3 与 Seed2Scale 的结合

Stage-CAP 高质量种子轨迹 → Seed2Scale 小模型 collector 放大 → VLM Verifier 筛选 → 目标 VLA fine-tune。这一联合 pipeline 有望实现更高效的数据工厂。

### 12.4 扩展到更复杂任务

- **非 object-centric 任务**：流体处理、布料操控等 MimicGen 难以处理的类别；
- **多步长时程任务**（10 个以上阶段）：Stage-CAP 的阶段局部修复机制在此场景下优势更突出；
- **多机器人协作**：双臂协调任务的阶段验证设计是新挑战。

### 12.5 代码生成质量的提升

- **CaP-RL 路线**（参考 CAP-X）：对代码生成模型做 GRPO 强化学习，以 stage-level 成功率为奖励；
- **技能库自动积累**：已成功的阶段代码片段作为可复用原语，降低后续任务的代码生成难度。

---

## 13. 参考文献

| # | 论文标题 | arXiv ID | 发表场合 | 年份 |
|---|---|---|---|---|
| 1 | RLBench: The Robot Learning Benchmark & Learning Environment | 1909.12271 | RA-L 2020 | 2019 |
| 2 | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation | 2112.03227 | RA-L 2022 | 2021 |
| 3 | Code as Policies: Language Model Programs for Embodied Control | 2209.07753 | ICRA 2023 | 2022 |
| 4 | Reflexion: Language Agents with Verbal Reinforcement Learning | 2303.11366 | NeurIPS 2023 | 2023 |
| 5 | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT/ALOHA) | 2304.13705 | RSS 2023 | 2023 |
| 6 | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | 2306.03310 | NeurIPS 2023 | 2023 |
| 7 | VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models | 2307.05973 | CoRL 2023 | 2023 |
| 8 | RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking | 2309.01918 | ICRA 2024 | 2023 |
| 9 | GenSim: Generating Robotic Simulation Tasks via Large Language Models | 2310.01361 | ICLR 2024 | 2023 |
| 10 | MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations | 2310.17596 | CoRL 2023 | 2023 |
| 11 | RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation | 2311.01455 | ICML 2024 | 2023 |
| 12 | Octo: An Open-Source Generalist Robot Policy | 2405.12213 | RSS 2024 | 2024 |
| 13 | RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots | 2406.02523 | RSS 2024 | 2024 |
| 14 | OpenVLA: An Open-Source Vision-Language-Action Model | 2406.09246 | CoRL 2024 | 2024 |
| 15 | DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning | 2410.24185 | ICRA 2025 | 2024 |
| 16 | pi0: A Vision-Language-Action Flow Model for General Robot Control | 2410.24164 | RSS 2025 | 2024 |
| 17 | SAGE: Scalable Agentic 3D Scene Generation for Embodied AI | 2602.10116 | — | 2026 |
| 18 | RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots | 2603.04356 | ICLR 2026 | 2026 |
| 19 | Seed2Scale: A Self-Evolving Data Engine for Embodied AI via Small to Large Model Synergy | 2603.08260 | — | 2026 |
| 20 | CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation | 2603.22435 | — | 2026 |

---

*报告生成时间：2026-07-07*
*撰写工具：GitHub Copilot（Claude Sonnet 4.6）*
*面向项目：Stage-CAP Privileged Data Generation（LIBERO 仿真轨迹生产与 VLA 数据引擎）*
