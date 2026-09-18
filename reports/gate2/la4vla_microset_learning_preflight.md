# LA4VLA 64 窗口小样本学习预检

日期：2026-09-18  
代码：`ff618f2b172c745242620f945fb7c2f3f9fbf498`  
结论：**数据加载、反向传播和固定微批过拟合通路通过；不构成泛化或闭环证据。**

## 数据包

从已通过严格回放的 Lift 766 步和 Stack 1503 步教师轨迹，各等距抽取 32 个
长度为 50 的连续窗口。每个窗口包含：

- 与 `qpos` 同步的 8D state：EEF position 3D、最短 rotvec 3D、夹爪 qpos 2D；
- 50×7 的 LA4VLA OSC action：归一化平移 3D、旋转 3D、gripper 1D；
- 当前时刻的第三人称与腕部 256×256 RGB；
- 任务 prompt：`Lift the cube.` 或 `Stack the red cube on the green cube.`；
- 原 episode、起始 transition、图片 SHA-256 和完整窗口 lineage。

为适配 LA4VLA 的 LeRobot loader，每个窗口被包装成一个 synthetic episode。
元数据明确写出 `synthetic_window_episodes=true` 和
`independent_source_episode_count=2`。因此 64 是训练窗口数，不是 64 条独立采集
episode；这些窗口存在重叠，不能用于样本效率或泛化统计。

最终数据目录：

```text
/data/gaochuanhao/mannercap_runs/gate2_la4vla_microset_capture_20260918
/data/gaochuanhao/mannercap_runs/gate2_la4vla_microset_packed_20260918_attempt02
```

原生 LA4VLA loader 实测得到 64 个样本，Lift/Stack 各 32；每条 image tensor
为 `2×3×448×448`，image mask 为 `[true,true]`，state 补齐为 24D，action 与 mask
均为 `50×24`，所有值有限。capture 为 8.6 MiB，packed 数据为 9.3 MiB。

## 两级训练检查

第一层使用项目原生 `train.py` 从固定候选 checkpoint 载入，仅训练 122,029,360
个 action-head 参数，运行 20 个 optimizer step：

| 指标 | 结果 |
| --- | ---: |
| 非有限 loss | 0 |
| step 0 / step 19 loss | 0.1814 / 0.1403 |
| 前 5 / 后 5 step 均值 | 0.22086 / 0.13732 |
| 最小 loss | 0.0486（step 11） |
| 训练态峰值 GPU allocation | 3.40 GiB |

Flow Matching 每步采样不同噪声，单步 loss 不可直接作为严格前后对比。因此第二层
固定样本 `[0,1,32,33]` 和固定评估噪声，预先缓存冻结的视觉语言 embedding，
对 action head 运行 100 步微批过拟合：

| 指标 | 结果 |
| --- | ---: |
| fixed-noise 初始 loss | 0.3102648854 |
| fixed-noise 最终 loss | 0.0776840895 |
| 最终/初始 | 0.25038 |
| 训练首步 / 末步 / 最小 loss | 0.40347 / 0.08410 / 0.07876 |
| 非有限 loss | 0 |
| wall time | 11.69 s |
| 峰值 GPU allocation | 3,945,053,184 bytes |

两次运行都未保存新 checkpoint。固定微批 loss 下降证明当前图片、文本、state、
动作、归一化和 action-head 梯度链可以共同工作；它只证明过拟合能力。

## 未通过的范围

- 数据只有 2 条独立教师 episode，窗口高度相关；
- prompt 只有任务语义，没有慢、快、轻柔或阶段组合方式；
- 没有验证未见初态、未见组合、阶段绑定或中性语言属性响应；
- 没有把训练后权重保存并送入 RoboSuite 闭环；
- 没有独立验证集、训练种子比较或 checkpoint 选择协议。

因此 `mannercap_task_evaluation_run` 和 `fixed_student_approved` 继续为 `false`。
下一步应先扩展带方式标注和独立初态的 D0 数据，再冻结训练/验证划分与闭环评测，
而不是直接启动长训练。

完整证据清单包含 200 个文件，清单自身 SHA-256 为
`198c843f2a72687987e8ca7c093826f997690d7c3752b9d19b1ec7f2a51c6807`；
`sha256sum -c` 全部通过。外部 LLM/API 调用数为 0，不需要 API key。
