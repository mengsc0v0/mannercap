# MimicGen / CAP-X / VoxPoser 三论文深度精读报告

> **撰写目的**：面向 Stage-CAP 项目，精读三篇最直接相关论文的 Introduction 叙事逻辑、方法细节与实验设计，并梳理 CAP-X 参考文献中的其他轨迹生成方法。
>
> **报告结构**：每篇论文独立一章，覆盖"叙事翻译"、"方法解析"、"实验内容"三部分。最后附 CAP-X 参考文献中的轨迹生成相关工作总结。

---

## 第一篇：MimicGen — 基于人类示教的可扩展机器人学习数据生成系统

**原文**：MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations  
**来源**：CoRL 2023 | arXiv: 2310.17596 | NVIDIA + UT Austin  
**作者**：Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, Dieter Fox

---

### 1.1 Introduction 叙事逻辑（翻译+分析）

**叙事主线**：数据稀缺 → 人工示教很贵 → 已有数据扩增方法的局限 → MimicGen 的解决思路

以下为 Introduction 核心内容的中文翻译与叙事结构分析：

> **第一段（问题定位）**：从大量人类示教进行模仿学习，已被证明是构建有能力的机器人智能体的有效范式。然而，示教的收集极其昂贵且耗时。这一扩展瓶颈限制了模仿学习在更广泛场景中的应用——要同时覆盖多种任务、多样的物体实例和场景配置，需要的数据量远超人工收集的可行上限。

> **第二段（现有方案的不足）**：已有一些自动化数据扩增工作。轨迹变换方法依赖对物体位姿的已知分割（需要人工任务工程）；domain randomization 提升了策略的泛化能力但不能扩展 demo 数量；数据增强方法（如图像增强）也不能生成全新的状态-动作序列。

> **第三段（MimicGen 的核心洞察）**：MimicGen 的关键观察是：机器人操控任务可以被分解为一系列 **object-centric subtasks**（以物体为中心的子任务）。每个子任务对应一段人工示教轨迹，其中机器人主要与某一个特定物体发生交互。如果能把这段轨迹变换到新物体位姿下，就可以复用人工示教。

> **第四段（贡献概述）**：我们用 MimicGen 从约 200 条人工示教生成了超过 50,000 条 demo，覆盖 18 个任务、多种场景配置、物体实例和机器人型号。在这些数据上训练的模仿学习策略在长时程和高精度任务（如多部件装配、咖啡制作）上取得了强劲表现。我们进一步表明，MimicGen 数据的有效性可与额外收集人工示教相媲美，使其成为扩展机器人学习的有力经济方案。

**叙事结构总结**：
- **Problem**：人工示教贵且稀缺 → 无法覆盖多样场景/物体/机器人
- **Gap**：已有扩增方法不能生成全新 demo，且需要大量人工任务工程
- **Key Insight**：操控任务可分解为 object-centric subtask sequence；每段轨迹可按物体位姿变换复用
- **Solution**：MimicGen 自动分割+变换+拼接+过滤，生成大规模高质量 demo
- **Result**：50K+ demos，18 任务，媲美直接收集人工示教的策略性能

---

### 1.2 方法（Method）

MimicGen 的核心是一个三步数据生成流程：

#### 步骤一：Source Human Trajectory 标注与分割

给定少量（约 10–200 条）人工示教，将每条轨迹按 **object-centric subtask** 手工标注分割点。

每个 subtask segment 的定义：
- 机器人在该 segment 内主要与**某一特定物体**交互（如抓握咖啡杯托）
- Segment 以机器人与该物体之间的相对姿态作为参考帧（object-centric frame）

分割后的 segment 数据结构：
```
segment_i = {
    "relevant_object": Object_k,
    "trajectory": [action_t, ..., action_t+n],  # 相对于 Object_k 的动作序列
    "start_pose": robot_pose_relative_to_Object_k,
    "end_pose": robot_pose_relative_to_Object_k
}
```

#### 步骤二：新场景中的 Segment 变换与拼接

给定新场景（物体随机初始化），MimicGen 对每个 subtask segment 执行以下操作：

1. **参考帧对齐**：在新场景中，找到对应物体 Object_k 的新位姿 T_new；
2. **轨迹变换**：将 source segment 中相对于 Object_k 的动作序列，通过刚体变换（rigid body transformation）映射到新场景中的绝对动作序列：
   - T_new_action = T_new @ T_source_inv @ T_source_action
3. **关节空间插值**：若变换后的末端执行器路径需要机器人从上一 segment 的末态过渡到本 segment 的起态，使用 IK 求解器和插值生成过渡路径；
4. **拼接**：依次执行所有 segment 的变换轨迹，得到完整的新场景 demo。

#### 步骤三：执行与过滤

在仿真中执行生成的轨迹：
- 若任务最终成功（所有子任务完成），将该轨迹加入生成数据集；
- 若失败，丢弃（整条轨迹级别的过滤，不支持局部修复）。

成功率取决于：
- 物体位姿变化幅度（Reset Distribution D0/D1/D2 对应从窄到宽的随机化程度）
- 任务的精度要求
- IK 求解成功率

#### 关键设计选择

| 设计 | 说明 |
|---|---|
| Object-centric frame | 使变换对任意物体初始位姿成立（前提：物体是刚体） |
| 分 segment 而非整条变换 | 避免中间状态的物理约束冲突（如咖啡机门打开后才能放杯托） |
| 过渡路径插值 | 确保 segment 之间的 IK 可达性 |
| 整条成功才录入 | 保证数据质量；缺点是浪费了部分成功但最终失败的 demo |

#### 支持的扩展

- **新场景变体**（不同物体初始位置/朝向）
- **新物体实例**（不同形状/尺寸的同类物体，前提：有对应物体的 pose 估计）
- **新机器人**（不同机械臂，通过重新 IK 求解）
- **长时程任务**（Kitchen、Coffee Preparation 等多阶段任务）
- **Isaac Gym Factory**（高精度接触任务，如 Gear Assembly、Frame Assembly）

---

### 1.3 实验内容

#### 实验环境与任务

| 类别 | 任务 | 场景变体 |
|---|---|---|
| 单臂桌面操控 | Stack, Stack Three, Square, Coffee, Threading, Three Piece Assembly, Hammer Cleanup, Mug Cleanup, Pick Place, Nut Assembly | D0, D1, D2 |
| 长时程多阶段 | Kitchen, Coffee Preparation | D0, D1 |
| 移动操控 | Mobile Kitchen | D0 |
| 高精度接触（Isaac Gym） | Nut-and-Bolt Assembly, Gear Assembly, Frame Assembly | D0, D1, D2 |
| 真实机器人 | Stack (Real), Coffee (Real) | D0, D1 |

场景变体：D0（窄随机化）< D1（中）< D2（宽随机化）

#### 实验一：核心规模验证

- **数据规模**：从约 200 条人工 source demos，生成 **50K+ generated demos**（每任务 1000 条）
- **策略训练**：在生成数据上训练 Behavioral Cloning（BC）策略（带图像观测）
- **代表性成功率**：
  - Stack Three D1: **91%** | Coffee D1: **93%** | Threading D1: **80%**
  - Square D1: **69%** | Three Piece Assembly D1: **61%** | Kitchen D1: **78%**
  - Gear Assembly D1: **76%** | Frame Assembly D1: **71%** | Mobile Kitchen D0: **77%**

#### 实验二：MimicGen 数据 vs. 人工示教数据

核心实验：在 Square D0 任务上，用 200 条 MimicGen 生成 demo（来自 10 条 source）训练的策略 vs. 200 条真实人工 demo 训练的策略：
- MimicGen 200 demos → **79% 成功率**
- Human 200 demos → **84% 成功率**

结论：两者性能相当（5% 差距在误差范围内），表明 MimicGen 可以作为等效替代。

#### 实验三：Source Demo 质量的影响

使用 robomimic Square MH 数据集中高质量（better operator）和低质量（worse operator）各 10 条 source demos 生成数据集，分别训练策略：
- 两个策略在 Square D2 上成功率**相近**
- **结论**：在大规模数据范围内，source demo 质量对下游策略影响小于预期，数量比质量更重要

#### 实验四：真实机器人部署

- Coffee 任务：用 100 条 MimicGen 生成的 demo 训练，咖啡杯托可放置在大范围任意位置
- 策略能在真实机器人（Franka Panda）上实现鲁棒执行

#### 消融分析（来自论文描述）

- **无过渡路径插值**：IK 失败率上升，低精度任务影响小但高精度任务影响大
- **减少 source demos 数量**：10 条 source demos 的生成质量接近 200 条（说明即使极少示教也能扩展）
- **不同 Reset Distribution**：D2 比 D0 难度高，但所有场景均能生成有效策略

---

## 第二篇：CAP-X — 评测与提升机器人操控编程智能体的统一框架

**原文**：CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation  
**来源**：ICML 2026 | arXiv: 2603.22435 | NVIDIA + UCB + Stanford + CMU  
**作者**：Max Fu*, Justin Yu*, Karim El-Refai*, Ethan Kou*, Haoru Xue*, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Jim Fan

---

### 2.1 Introduction 叙事逻辑（翻译+分析）

**叙事主线**：经典控制 vs. VLA 两个对立范式 → LLM 编程 agent 是否能填补中间 → 系统化评测发现的 gap → CaP-X 的贡献

#### 段落翻译（精要版）

> **第一段（经典控制范式）**：机器人长期以程序化控制为主——明确的程序将高层目标分解为子任务，组合感知与控制模块，通过显式逻辑处理故障。这从 STRIPS（1971）、Hierarchical TAMP 一路发展而来。这类范式凭借显式结构提供了强大的可解释性和几何精度保证。然而，它长期依赖人类专家手动设计抽象层和预见边角情形，造成了扩展瓶颈——任务特定的脆弱解决方案难以泛化到开放场景。

> **第二段（VLA 范式的兴起与局限）**：数据缩放的成功驱动了另一种范式：Vision-Language-Action（VLA）模型直接从大规模视觉-动作数据学习。这些方法在接触丰富的任务（如叠衣服、工具使用）上表现令人印象深刻。然而，VLA 继承了训练数据的局限性和设计约束：缺乏可解释性，在新机器人形态和新颖长时程任务上的泛化往往需要额外数据收集和重新训练。

> **第三段（LLM 编程 agent 的潜力与现有 gap）**：LLM 的最新进展提供了一条弥合两种范式的路径：用通用编程 agent 替代程序化控制循环中的人类工程师。现代编程 agent 已能合成可执行代码、定义函数并在 SWE-Bench 等软件基准上调试故障。然而，尽管 Code-as-Policy 的先驱工作在机器人应用上验证了这一路径，它们大多依赖高层 primitive（如 `stack_objs_in_order()`）——这些人工设计的封装提供了大量任务特定的简化。由此，代理的表现有多少来自自身推理能力、有多少来自这些 primitive 的过度工程化，目前仍不清楚。

> **第四段（CaP-X 的贡献）**：为填补这一 gap，我们提出 CaP-X：一个用于系统化评测和提升编程机器人控制 agent 的统一框架。CaP-Gym 集成了 Robosuite、LIBERO-PRO、BEHAVIOR 共 187 个任务，暴露了从高层 macro 到低层原子 primitive 的完整原语栈。CaP-Bench 沿三个轴（抽象层级、时序交互、感知接地）评测 frontier 模型。CaP-Agent0 结合视觉差分、技能库和并行推理，在无需训练的情况下达到近人类水平。CaP-RL 通过 GRPO 强化学习对编程 agent 进行后训练，大幅提升成功率并实现 sim-to-real 零差距迁移。

**叙事结构总结**：
```
经典程控范式（可解释但不可扩展）
      ↓ 数据驱动
VLA 范式（强泛化但不透明且难以迁移）
      ↓ LLM 代码生成可弥合两者
CaP 先驱工作（有效但依赖高层 primitive 的过度工程化）
      ↓ 尚无系统评测：去掉人工脚手架，LLM 真实能力几何？
CaP-X（系统化评测 + agentic 提升 + RL 后训练）
```

---

### 2.2 方法（Method）

CaP-X 的方法由四个互相依赖的组件构成：

#### 组件一：CaP-Gym（交互式机器人编程环境）

**架构**：基于 Gymnasium 接口的分层控制框架。

```
CaP-Gym
├── 底层：物理仿真器（Robosuite / LIBERO-PRO / BEHAVIOR）或真实机器人
├── 中层：Code Executor（Python REPL sandbox）
└── 上层：感知与控制 Primitive API
```

**Primitive API 设计**：
- **感知 Primitive**：SAM3（语言条件分割）、Molmo2（开放词汇指点）、OpenCV、Open3D
- **控制 Primitive**：PyRoki IK 求解器、碰撞检测、可达性约束检查
- **API 层级**：High-Level（人工设计的 macro，如 `pick_and_place()`） vs. Low-Level（原子 primitive，如 `solve_ik()`, `sam3_text_prompt()`）

**关键设计**：单个"代码执行轮次"（turn）对应生成代码的完整执行，可触发多个底层仿真步；code executor 以 REPL 方式运行，支持多轮交互。

#### 组件二：CaP-Bench（系统化基准评测）

8 个评测层级，沿三个轴变化：

| 层级 | 感知 | Primitive 抽象度 | 交互模式 | 视觉接地 |
|---|---|---|---|---|
| S1 | Ground-truth 状态 | 高层（macro） | 单轮 | — |
| S2 | 真实感知（噪声） | 高层 | 单轮 | — |
| S3 | 真实感知 | 低层（+用法示例） | 单轮 | — |
| S4 | 真实感知 | 低层（无示例） | 单轮 | — |
| M1 | 真实感知 | 高层 | 多轮（stdout/stderr） | — |
| M2 | 真实感知 | 高层 | 多轮 | 原始 RGB 图像 |
| M3 | 真实感知 | 高层 | 多轮 | 视觉差分模块（VDM） |
| M4 | 真实感知 | 低层（+示例） | 多轮 | VDM |

**评测协议**：Zero-Shot Pass@1，每个 task instance 只允许一段连续交互 episode（环境不重置），共 100 trials/task/tier。

#### 组件三：CaP-Agent0（无训练 Agentic 框架）

三个核心增强模块：

**1. 多轮视觉差分模块（VDM）**：
- 将原始 RGB 图像转化为结构化自然语言描述
- 第一轮：生成场景描述 + 任务相关视觉特征
- 后续轮次：描述场景变化 + 判断任务是否完成
- 关键发现：直接将原始图像（M2）传入 LLM 反而降低性能（因跨模态对齐 gap）；VDM 文本化后（M3）显著优于 M1 和 M2

**2. 自动合成任务无关技能库**：
- 从 CaP-Bench S3 成功执行的代码中提取可复用的辅助函数
- LLM 分析常见重复出现的任务无关逻辑（如 `select_top_down_grasp()`、`rotation_matrix_to_quaternion()`）
- 自动提炼为持久化技能库，跨 trial 保留
- 当前共 9 个自动合成技能（包括旋转矩阵/四元数转换、top-down 抓握过滤等）

**3. 并行推理（Ensemble）**：
- 单模型：对 Gemini-3-Pro 进行 9 次不同温度（0.1–0.9）的并行采样
- 多模型：Gemini-3-Pro / GPT-5.2 / Claude Opus 4.5 各 3 次，共 9 个候选
- 中央 coding agent 将候选合成为最终代码
- 发现：ensemble 能**预先**构建故障处理分支，而单次采样只能**事后**修补

#### 组件四：CaP-RL（编程 Agent 的 RL 后训练）

**方法**：对 Qwen2.5-Coder-7B-Instruct 应用 GRPO（Group Relative Policy Optimization）。

```
训练流程：
1. 在 CaP-Gym 中执行当前策略（代码生成 + 执行）
2. 获取仿真环境的 task success 奖励信号
3. GRPO 更新语言模型权重（以组内相对奖励为梯度信号）
4. 在 S1（特权状态）层级训练，避免感知噪声引入的奖励归因错误
```

**RL 后训练的质量提升（定性分析）**：
- **训练前**：模型常出现"步骤跳过"（step skipping）——直接尝试将夹爪移到目标位置而不先抓取物体；位置偏移计算使用硬编码数值（0.05m）
- **训练后**：学会正确的因果顺序（识别 → 抓取 → 运输 → 放置）；使用 `return_bbox_extent=True` 动态计算叠放高度，而非依赖记忆的偏移量

---

### 2.3 实验内容

#### 实验一：CaP-Bench 主实验（12 模型 × 7 任务 × 8 层级）

**模型**：Gemini-3-Pro, GPT-o1/o4-mini/5.1/5.2, Claude Haiku4.5/Opus4.5, GPT-OSS-20B/120B, Qwen3-235B, Qwen2.5-Coder-7B, Kimi-K2, DeepSeek-V3.1

**7 核心任务**（Robosuite）：Cube Lift, Cube Stack, Spill Wipe, Peg Insertion, Cube Re-stack, Two-Arm Lift, Two-Arm Handover

**核心发现**：

| 发现 | 关键数据 |
|---|---|
| Frontier LM 与人类差距 | 最强模型 Gemini-3-Pro（S4）32.3% vs. 人类 88.5% |
| 高层抽象单调提升性能 | S4→S1 成功率单调递增 |
| 代码编译率与抽象度 | 开源模型 S4 编译率仅 58%，S1 达 93% |
| 多轮文本反馈（M1）提升 | 几乎所有模型均受益于 stdout/stderr 反馈 |
| 直接 RGB（M2）反而下降 | 跨模态对齐 gap，视觉特征难以与代码生成联合推理 |
| VDM（M3）最有效 | 一致优于 M1 和 M2 |
| 低层+多轮（M4）≥ 高层+多轮 | 测试时计算扩展可弥补抽象度不足 |

#### 实验二：CaP-Agent0 vs. 训练后 VLA（LIBERO-PRO）

**任务**：LIBERO-PRO 30 个任务（位置扰动 Pos + 指令扰动 Task 两种评测）

| 方法 | libero-object Pos | libero-goal Pos | libero-spatial Pos | Avg. Task |
|---|---|---|---|---|
| OpenVLA | 0% | 0% | 0% | 0% |
| π0 | 0% | 0% | 0% | 0% |
| π0.5 | 17% | 38% | 20% | ~0.7% |
| **CaP-Agent0** | **22%** | **26%** | **12%** | **~15%** |

关键结论：训练无关的 CaP-Agent0 在位置扰动下超越了 π0.5（训练方法），在指令扰动下大幅优于所有 VLA（VLA 对指令分布极度敏感，而代码生成自然支持指令泛化）。

#### 实验三：CaP-Agent0 在 BEHAVIOR 移动操控（AgiBot G1）

| 任务 | 导航成功率（Human/S3/CaP-Agent0） | 任务完成率（Human/S3/CaP-Agent0） |
|---|---|---|
| Pick up Radio | 88% / 72% / 80% | 36% / 24% / **56%** |
| Pick up Soda Can | 80% / 52% / 84% | 72% / 32% / **72%** |

#### 实验四：CaP-RL 后训练（Qwen 2.5 Coder 7B × GRPO）

| 任务 | 基础 7B 模型（仿真） | CaP-RL 后（仿真） | 人类专家（仿真） | CaP-RL 后（真实）| 人类（真实） |
|---|---|---|---|---|---|
| Cube Lift | 25% | **80%** | 93% | **84%** | 92% |
| Cube Stack | 4% | **44%** | 73% | **76%** | 84% |
| Spill Wipe | 30% | **93%** | 100% | — | — |

关键结论：50 次 GRPO 迭代后，7B 模型在 Cube Lift 从 25% 提升到 80%；零样本迁移到真实 Franka 机器人，成功率从 24% 提升到 84%（接近人类 92%）。

#### 实验五：真实机器人 zero-shot（Franka + AgiBot G1）

CaP-Agent0 在无任务特定训练的条件下完成了一系列复杂真实世界任务：
- **机械搜索**：系统搜索三个杯子下的隐藏物体
- **数学符号推理**：感知积木上的数字方程，选择正确积木填入答案
- **工具域知识泛化**：调用 SciPy RANSAC 算法计算墙面法线方向，准确按压电梯按钮
- **物理常识推理**：判断球形物体只能放在方形物体上方，自动选择最稳定的叠放顺序

---

### 2.4 CAP-X 参考文献中的轨迹生成相关工作

CAP-X 的 Related Work 和参考文献中出现了以下与轨迹生成相关的重要工作：

#### LLM 辅助奖励函数生成（用于 RL 轨迹生成）

| 工作 | 方法 | 关联 |
|---|---|---|
| **Eureka** (Ma et al., ICLR 2024) | LLM 生成奖励函数代码，用于 RL 策略学习（如灵巧手操控） | LLM 生成 RL 奖励 → RL 生成轨迹 |
| **Text2Reward** (Xie et al., ICLR 2024) | 从自然语言描述自动生成奖励函数，驱动 RL | 轨迹间接生成 |
| **Language to Rewards** (Yu et al., CoRL 2023) | LLM 将语言指令转为奖励函数组合，训练 RL 策略 | LLM → 奖励 → RL 轨迹 |
| **EurekaVerse** (Liang et al., CoRL 2024) | LLM 生成环境课程，加速 RL 训练 | 环境多样化 → 更多轨迹 |

#### Code-as-Policy 家族（直接生成控制代码轨迹）

| 工作 | 方法 | 关联 |
|---|---|---|
| **Code as Policies** (Liang et al., ICRA 2023) | LLM 生成 Python 策略代码，调用感知+控制 primitive | CaP 奠基；直接代码轨迹 |
| **ProgPrompt** (Singh et al., ICRA 2023) | situated task plan + LLM code generation | situated code-based trajectory |
| **ChatGPT for Robotics** (Vemprala et al., 2023) | 通过 ChatGPT 生成机器人控制代码 | 同系列 |
| **VoxPoser** (Huang et al., CoRL 2023) | LLM+VLM 生成 3D value maps → 运动规划轨迹 | 3D grounding + 轨迹合成 |
| **Instruct2Act** (Huang et al., 2023) | 多模态指令 → LLM 生成机器人行动序列 | 同系列 |
| **RoboCodeX** (Mu et al., 2024) | 多模态代码生成，行为合成 | 同系列 |
| **RoboClaw** (Li et al., 2026) | Agentic framework，可扩展长时程任务 | 最新同系列工作 |

#### 轨迹生成的数据引擎类工作（CAP-X 参考文献中的间接相关）

| 工作 | 方法 | 关联 |
|---|---|---|
| **RoboGen** (Wang et al., ICML 2024) | propose-generate-learn 循环，LLM 生成任务+场景+RL/规划轨迹 | 自动化数据生产 |
| **AutoRT** (Ahn et al., 2024) | 大规模编排机器人 agent，自动收集数据 | 真实机器人数据引擎 |
| **DrEureka** (Ma et al., RSS 2024) | LLM 指导 sim-to-real 迁移，domain randomization | 从仿真轨迹迁移真实 |

#### Agent 自我提升框架（与 CaP-Agent0 设计相关）

| 工作 | 方法 | 关联 |
|---|---|---|
| **Voyager** (Wang et al., 2023) | 开放世界 LLM agent，积累持久技能库 | 技能库设计来源 |
| **Reflexion** (Shinn et al., NeurIPS 2023) | 语言反思驱动 agent 迭代改进 | 多轮反馈设计来源 |
| **Self-Refine** (Madaan et al., NeurIPS 2023) | 迭代自我反馈精化输出 | 多轮反馈设计来源 |
| **Self-Consistency** (Wang et al., ICLR 2023) | 多次采样 → 多数投票 | Ensemble 设计来源 |

---

## 第三篇：VoxPoser — LLM 与 VLM 组合 3D Value Maps 实现机器人操控轨迹合成

**原文**：VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models  
**来源**：CoRL 2023 (Oral) | arXiv: 2307.05973 | Stanford University  
**作者**：Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei

---

### 3.1 Introduction 叙事逻辑（翻译+分析）

**叙事主线**：LLM 有丰富的行动知识 → 但大多数方法仍需预定义 motion primitive → VoxPoser 直接合成 6-DoF 末端执行器轨迹 → 零样本、closed-loop、鲁棒、在线可学习

#### 段落翻译（精要版）

> **第一段（LLM 的潜力）**：大语言模型被证明拥有丰富的可操作性知识，可以用于机器人操控的推理和规划形式。尽管取得了进展，大多数方法仍然依赖预定义的运动 primitive 来完成与环境的物理交互，这仍然是一个主要瓶颈——robot 必须从有限的原子动作库中拼装，无法自由生成连续的 6-DoF 轨迹。

> **第二段（目标定位）**：在本工作中，我们旨在为开放集指令和开放集物体的大量操控任务合成机器人轨迹（即密集的 6-DoF 末端执行器 waypoint 序列）。我们的核心观察是：LLM 擅长从自由形式语言指令中推断**可供性（affordances）**和**约束（constraints）**。更重要的是，借助其代码编写能力，LLM 可以与视觉语言模型（VLM）交互，合成 **3D value maps**，将知识接地（ground）到 agent 的观测空间中。

> **第三段（方法概述）**：合成的 value maps 随后在 model-based planning 框架中用作目标函数，零样本地合成 closed-loop 机器人轨迹，并对动态扰动保持鲁棒性。我们进一步展示了如何通过在线经验高效学习接触丰富（contact-rich）场景的动力学模型，从而让框架从中受益。

**叙事结构总结**：
```
LLM 有规划知识 → 但绑定在离散 primitive 上 → 无法直接控制
     ↓
VoxPoser：LLM 推断 affordance/constraint → 代码生成 → VLM 接地 → 3D value maps
     ↓
value maps 作为连续目标函数 → 运动规划器（MPC）合成密集 6-DoF 轨迹
     ↓
闭环执行 + 动态扰动鲁棒 + 在线动力学模型学习
```

---

### 3.2 方法（Method）

VoxPoser 的核心贡献是一个**零样本轨迹合成流水线**，无需任何额外训练。

#### 步骤一：LLM Planner — 任务分解

给定自然语言任务指令（如 "close the top drawer"），LLM Planner：
1. 将指令分解为子任务序列（如 ["grasp drawer handle", "pull drawer closed"]）
2. 为每个子任务生成 Composer 调用代码

Planner 使用 few-shot 提示，输出结构化的子任务规划代码。

#### 步骤二：LLM Composer — 生成 Value Map 代码

Composer 是 VoxPoser 的核心创新。LLM 生成 Python 代码，调用 VLM API 来合成以下类型的 3D value maps：

| Value Map 类型 | 含义 | 生成方式 |
|---|---|---|
| **Affordance Map**（可供性图）| 末端执行器"应该去哪里" | VLM 生成 2D 热图 → 深度补全到 3D |
| **Avoidance Map**（避障图）| 末端执行器"不应该去哪里" | VLM 识别障碍物 → 3D 排斥区域 |
| **Rotation Map**（旋转图）| 末端执行器的目标朝向 | VLM + 几何推断 |
| **Velocity Map**（速度图）| 末端执行器的运动速度偏好 | 任务语义推断 |
| **Gripper Map**（夹爪图）| 夹爪开/合状态 | 任务阶段推断 |

每种 value map 都是一个 3D voxel grid，定义在机器人工作空间中。最终 value map 是所有 map 的加权组合（composable）。

**代码生成示例（概念性）**：
```python
# Composer 为 "grasp drawer handle" 生成的代码
affordance_map = get_affordance_map(
    rgb_image, 
    query="drawer handle to grasp"
)
avoidance_map = get_avoidance_map(
    rgb_image, 
    query="drawer body, wall"
)
rotation_map = get_rotation_map(
    rgb_image,
    query="grasp orientation for horizontal drawer handle"
)
value_map = compose_maps(affordance_map, avoidance_map, rotation_map)
```

#### 步骤三：VLM 接地（Grounding）

LLM 生成的代码调用 VLM（论文使用 CLIP 和 ViLD 等视觉模型）：
- **2D 热图生成**：VLM 对查询文本（如 "drawer handle"）生成像素级相关性图
- **3D 提升**：结合 RGB-D 深度信息，将 2D 热图提升为 3D voxel value map
- **坐标系对齐**：所有 map 在机器人工作空间的统一 3D 坐标系下合成

#### 步骤四：Model-based Planning — 轨迹优化

以合成的 value map 为目标函数，使用 **Model Predictive Control（MPC）** 优化末端执行器轨迹：

```
轨迹优化：
maximize: Σ_t V(x_t)    # value map 在轨迹上的累积值
subject to: f(x_t, a_t) = x_{t+1}   # 动力学约束
            collision_free(x_t)      # 碰撞约束
            x_T ∈ goal_region        # 终止条件
```

- **CEM（Cross-Entropy Method）**：用于无梯度的轨迹优化
- **Closed-loop 执行**：LLM 输出在整个任务过程中缓存（不重新查询），value map 随观测更新后 MPC 重规划 → 对动态扰动鲁棒

#### 步骤五：接触丰富交互的在线动力学模型学习

对于需要接触（如开抽屉、转旋钮）的子任务：
- 通过少量真实交互（online experience）高效学习局部动力学模型
- 将学习的动力学整合进 MPC 规划 → 提升接触任务的轨迹质量

#### 零样本推理的关键特性

- **无额外训练**：整个流水线（LLM + VLM + MPC）在推理时 zero-shot 运行
- **composable**：多个 value map 可以通过代码灵活组合（加减乘除、空间约束等）
- **语言可解释**：每个 value map 都有对应的自然语言语义，可以用语言反馈（如"偏了 1cm"）精确修正
- **涌现能力**：
  - 行为常识推理（"我是左撇子" → 叉子放左边）
  - 细粒度语言修正（"偏了 1cm" → 精确调整）
  - 多步视觉规划（"精确打开一半抽屉" → 先打开再关到一半）

---

### 3.3 实验内容

#### 实验一：仿真实验（RLBench 任务集）

**平台**：RLBench（25+ 任务）

**任务覆盖**：
- 单步抓取放置（pick and place）
- 关节操控（开门、转旋钮、拉抽屉）
- 精度操控（将物体放入容器）
- 顺序多步任务（先开抽屉再放物体）

**评测维度**：
- Zero-shot 成功率（无任何任务特定训练）
- 对动态扰动的鲁棒性（执行过程中随机移动物体）
- 在线动力学学习对接触任务的提升

**对比基线**：
- 无 value map（只用 affordance map，无 avoidance）
- 无 closed-loop（开环轨迹）
- 无动力学学习（固定 planner）

**主要结论**：VoxPoser 在 25+ RLBench 任务上取得了有竞争力的 zero-shot 成功率，显著优于仅使用 LLM 高层规划（不合成 value map）的基线。

#### 实验二：真实机器人实验（Franka Panda）

**任务**：多种日常操控任务，使用 free-form 自然语言指令：
- "sort the paper trash into the blue tray"
- "close the top drawer"
- "set the table for a left-handed person"
- "cover the teapot with the lid"
- "open the drawer precisely by half"

**关键实验结果**：

| 能力 | 实验设计 | 结果 |
|---|---|---|
| Zero-shot 泛化 | 未见过的指令+物体组合 | 成功执行大量日常任务 |
| 动态扰动鲁棒性 | 执行中随机推移物体 | MPC 实现快速重规划 |
| 行为常识 | "我是左撇子，帮我摆餐具" | 叉子自动放到左侧 |
| 细粒度语言修正 | "你偏了1cm" | 精确调整位置 |
| 多步视觉规划 | "精确打开抽屉到一半" | 先全开记录位移，再关到半处 |
| 接触丰富任务 | 旋转旋钮、拉抽屉 | 在线动力学学习显著提升 |

#### 实验三：消融研究

| 消融项 | 成功率变化 |
|---|---|
| 去掉 Avoidance Map | 碰撞率上升，操控精度下降 |
| 去掉 Rotation Map | 夹爪姿态随机，抓取成功率下降 |
| 开环（不用 MPC 重规划） | 对动态扰动完全失鲁棒性 |
| 去掉在线动力学学习 | 接触任务成功率明显下降 |
| 直接用 LLM 输出坐标 | 远差于 VoxPoser（无视觉接地） |

---

## 综合对比与对 Stage-CAP 的启示

| 维度 | MimicGen | CaP-X | VoxPoser |
|---|---|---|---|
| **叙事主题** | 人工示教太贵，用变换扩增 | LLM 代码能力 vs. 人工脚手架，谁才是真实能力 | LLM 有知识但被 primitive 束缚，用 value map 解锁 |
| **核心技术** | Object-centric 轨迹变换 | CaP-Gym + 8 级 benchmark + VDM + 技能库 + RL | LLM code → VLM grounding → 3D value maps → MPC |
| **轨迹生成范式** | 变换现有轨迹 | 代码生成后执行 | value map 驱动运动规划 |
| **失败处理** | 整条丢弃 | 多轮 VDM + 代码修复 | MPC 在线重规划 |
| **种子数据依赖** | 需要少量人工示教 | 不需要（zero-shot） | 不需要（zero-shot） |
| **是否用特权信息** | 是（物体位姿） | 是（S1 层级，ground-truth 状态） | 否（纯视觉感知） |
| **VLA 数据生产适用性** | **强**（直接生成大量 demo） | 中（主要评测；CaP-RL 生成轨迹） | 弱（零样本执行，不专门针对数据生产） |

### Stage-CAP 相对于三篇论文的差异化定位

1. **vs. MimicGen**：Stage-CAP 不需要种子示教，通过 LLM 代码生成从零起步；失败处理是阶段级而非整条丢弃；同样面向 VLA 数据生产。

2. **vs. CaP-X（CaP-Agent0）**：Stage-CAP 增加了 KITE 失败帧诊断和阶段边界验证，专注于数据生产质量而非 benchmark 成绩；Stage-CAP 使用特权 readonly 信息（如物体位姿、BDDL 关系）合法地辅助数据生产，而 CaP-Agent0 更侧重 zero-shot 无特权执行。

3. **vs. VoxPoser**：Stage-CAP 使用代码生成机器人动作序列（而非 value map），更易于阶段验证和代码级局部修复；VoxPoser 适合单步或少步任务，而 Stage-CAP 的阶段机制专门面向长时程多阶段任务。

---

*报告生成时间：2026-07-07*
*内容来源：MimicGen (2310.17596)、CaP-X (2603.22435) 原始论文全文及项目主页，VoxPoser (2307.05973) 摘要及项目主页*
