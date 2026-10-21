# 上游 Agent 执行交接：π0.5 训练轨迹采集与交付

> 本文件是独立交接任务书。接收此文件的 Agent 应在自己可用的服务器与代码库中实现采集、导出、验证和打包；无需读取发起方目录，也不依赖聊天历史。文中所有 dataset 路径均为交付包内部相对路径。
>
> 请实际完成工作，不要只返回方案。先检查现有采集器与本接口的差异，完成先导批次、格式检查和真实动作回放，再扩充数据。不得用占位图像、虚构动作或未执行的验证结果补齐交付。如果关键条件无法满足，提交具体缺口及已完成的可用产物。

日期：2026-09-30。依据本项目已固定的代码与训练配置核对；这是本项目接口，不是所有 π0.5 模型的通用格式。

## 1. 对接目标与兼容边界

接收方已有自定义 NPY/NPZ 读取器和训练入口；你无需访问接收方代码或服务器。openpi 固定提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。继承 `pi05_libero`：π0.5、action_horizon=10、内部 action_dim=32、extra_delta_transform=False。

上游交付单步原始序列：双相机 RGB + 8D state + 7D actions + 自然语言指令 + 可复现环境与审计记录。训练端负责 10 步动作块、统计归一化和内部补零；上游不要预先做这些变换。无需先转换成 LeRobot，当前代码直接读取下述 NPY/NPZ 文件。

现有读取器只支持 manifest 顶层统一的 `task` 指令。单一任务/单一指令包可按现接口读取；多任务、多 manner 混合训练必须让训练侧增加逐 episode 的 prompt 读取，不能只加字段便声称已支持。本文同时约定该扩展字段，但本次没有修改训练代码。

当前闭环基线是 Panda / Lift / 固定刚度 OSC_POSE。Stack 可沿用张量格式，但必须另配任务环境、逐轨迹指令和对应的回放验收，不能冒充已验证的 Lift 数据。

## 2. 文件结构

```text
dataset/
  manifest.json
  episodes.jsonl
  source_manifest.json
  replay_audit.json
  episode_000001/
    images.npy
    transitions.npz
    metadata.json
    model.xml
    replay_state.npz
  episode_000002/
    ...
```

`manifest.json`、`images.npy`、`transitions.npz` 是当前数据读取器的输入。其他文件为本次合成交付要求的来源、语义和回放证据；当前读取器不会自动使用它们。路径相对 dataset 根目录，episode ID 全包唯一，不使用 pickle/object 数组。

## 3. images.npy：真实逐时刻双视角图像

| 属性 | 要求 |
|---|---|
| shape | `(T, 2, 224, 224, 3)` |
| dtype | `uint8` |
| 色彩/布局 | RGB，HWC，值域 0–255，正向显示 |
| camera 0 | `agentview`，第三人称 |
| camera 1 | `robot0_eye_in_hand`，腕部相机 |
| 时序 | `images[t]` 与 `state[t]` 都来自发送 `actions[t]` 前的同一仿真状态 |

当前渲染代码对 `env.sim.render(...)` 做 `[::-1]` 垂直翻转。上游交付最终正向图像，并记录原始渲染与翻转方式；若原始 API 已输出正向图像，不可再翻一次。两路相机的位置、姿态、FOV、机器人安装关系写入环境配置，训练与评估保持一致。

不能重复首帧充当整段视频，不能用执行后图像冒充执行前图像。MP4 可作预览，但不能代替当前读取器需要的 images.npy。

## 4. transitions.npz：核心训练字段

必须包含以下两个键，名称精确匹配：

| 键 | shape / dtype | 含义 |
|---|---|---|
| `state` | `(T,8)` / float32 | 执行动作前的机器人状态 |
| `actions` | `(T,7)` / float32 | 该状态下实际发送到目标环境 `env.step()` 的 7D 命令 |

### state 的 8 维顺序

```text
[eef_x, eef_y, eef_z, rotvec_x, rotvec_y, rotvec_z, finger_qpos_0, finger_qpos_1]
```

- xyz：世界坐标系末端位置，米。
- rotvec：末端绝对姿态的轴角旋转向量，弧度；不是 Euler/RPY，也不是四元数。
- finger_qpos：按 `robot0_gripper_qpos` 原始顺序保存两个夹爪关节位置，米；保留符号，不替换成开合比例或单一夹爪宽度。
- 当前参考构造：`concat(obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])`。
- RoboSuite 该观测四元数使用 xyzw；旧轨迹若为 wxyz，必须先重排，不能直接传入同一转换函数。
- 不要把 7 个机械臂关节角加 1 个夹爪值当作这个 8D state。

### actions 的 7 维顺序和控制语义

```text
[u_dx, u_dy, u_dz, u_drx, u_dry, u_drz, u_gripper]
```

- `float32`，所有分量有限，归一化控制输入范围 `[-1,1]`。
- 前 3 维：world-frame OSC 位置增量命令；当前控制器物理缩放为 `0.05 * u_dxyz` 米。
- 中间 3 维：world-frame OSC 旋转增量命令；物理缩放为 `0.5 * u_drot` 弧度旋转向量，按控制器旋转组合规则施加；不是相邻绝对轴角向量直接相减。
- 第 7 维：原生 Panda GRIP 控制输入，不能替换为测到的手指 qpos、gripper_fraction 或牛顿值。上游必须在实际环境做开/合方向验证并在 metadata 写明符号，训练侧沿用相同映射，不自行翻转。
- 控制频率 20 Hz，动作间隔 0.05 s。
- 当前参考控制器：`OSC_POSE`、`input_type=delta`、`input_ref_frame=world`、`input_min=-1`、`input_max=1`、`output_max=[0.05,0.05,0.05,0.5,0.5,0.5]`、对称 output_min、`impedance_mode=fixed`、`kp=150`、`damping_ratio=1`。完整参考环境配置已内嵌于本文附录 A，不需要访问接收方目录。

动作必须是在这一控制接口下执行并验证的命令。关节控制器的目标关节角、关节速度、力矩，即使维数恰好为 7，也不兼容。由实际相邻末端位姿反推的 pose delta 只是候选命令，不等于原始控制动作；必须在目标控制器下重新执行，采集这次执行的图像/状态/动作并验收。不得静默裁剪越界命令后继续沿用原轨迹图像。

当前动作空间不包含可变 kp、阻尼或显式握力。若 gentle/firm 的区别来自上游改 kp 或另加握力控制，必须先确认能在固定 7D 接口中复现，否则需另议动作空间与训练/部署适配，不能宣称当前接口已经能学习该区别。

## 5. 同步规则与审计字段

```text
读取 obs_t 与两路 image_t
生成 action_t
记录 state[t]、images[t]、actions[t]
obs_next = env.step(action_t)
记录 next_state[t]、时间、结束/成功状态
```

同一个 episode 从一次 reset 开始，不能拼接不同 reset 或不同尝试。T 是实际发送的动作数；终止后额外的最终观测不要伪造一个动作来凑长度。

本次上游交付额外要求在 transitions.npz 保存下表字段，供验收用；当前训练器只取 state/actions：

| 字段 | shape / dtype | 说明 |
|---|---|---|
| `next_state` | `(T,8)` float32 | 每个动作执行后的状态，最后一步也保留 |
| `timestamp` | `(T,)` float64 | 动作前仿真时间，秒，从 0 起 |
| `dt` | `(T,)` float64 | 每步仿真持续时间，应为 0.05 s |
| `frame_index` | `(T,)` int64 | 连续 0…T-1 |
| `sim_step_index` | `(T,)` int64 | 动作前仿真物理步索引，严格递增 |
| `terminated` | `(T,)` bool | 环境终止标志 |
| `truncated` | `(T,)` bool | 时间上限等截断标志 |
| `success` | `(T,)` bool | 环境任务成功判定，独立于 done |

非末尾时刻应满足 `next_state[t] ≈ state[t+1]`，两者用同一姿态转换。不得把 0.1–0.4 s 的缺失区间当成 0.05 s。若原采集存在时间空洞，应修复采集或在目标环境重新执行，不能仅修改 timestamp。

## 6. manifest.json 示例

以下是两条示例记录；T 和哈希占位符必须替换为实际值。`complete=true` 只能在本批预定数据全部落盘且验收后设置，不要求合成数据固定为官方的 200 条。

```json
{
  "schema": "pi05-lift-rgb-v1",
  "complete": true,
  "source_sha256": "<source_manifest.json 的真实 SHA-256>",
  "task": "Lift the cube.",
  "fps": 20,
  "cameras": ["agentview", "robot0_eye_in_hand"],
  "image_convention": "top_down_rgb",
  "state": "eef_xyz, eef_axisangle, two_gripper_qpos",
  "action": "world_delta_osc7",
  "episodes": [
    {
      "episode": "episode_000001",
      "split": "train",
      "frames": 120,
      "images": "episode_000001/images.npy",
      "transitions": "episode_000001/transitions.npz",
      "images_sha256": "<真实 SHA-256>",
      "transitions_sha256": "<真实 SHA-256>",
      "prompt": "Lift the cube gently.",
      "task_name": "lift",
      "manner_tag": "gentle",
      "seed": 1,
      "reset_id": "reset_000001",
      "source_attempt_id": "attempt_000001",
      "lineage_id": "source_000001",
      "task_ok": true,
      "manner_ok": true,
      "metadata": "episode_000001/metadata.json"
    },
    {
      "episode": "episode_000002",
      "split": "valid",
      "frames": 100,
      "images": "episode_000002/images.npy",
      "transitions": "episode_000002/transitions.npz",
      "prompt": "Lift the cube quickly.",
      "task_name": "lift",
      "manner_tag": "fast",
      "seed": 2,
      "reset_id": "reset_000002",
      "source_attempt_id": "attempt_000002",
      "lineage_id": "source_000002",
      "task_ok": true,
      "manner_ok": true,
      "metadata": "episode_000002/metadata.json"
    }
  ]
}
```

示例中的不同 prompt 说明扩展交付方式，不代表旧读取器已读取这些字段：未适配时，两条记录都会使用顶层 `Lift the cube.`。上游务必保留逐 episode prompt；训练侧应显式接入并测试后才混合训练。若要求完全不改读取器，应每份包只包含同一指令，顶层 task 就写该完整指令。

每条 `episodes.jsonl` 与 episodes 数组中一条记录对应，作为便于检索的索引，不替代 NPZ/NPY。manifest 还须附完整环境配置及软件版本，按本文附录 A 的结构记录实际使用的完整参数，不要填写另一台服务器的绝对路径。

`source_manifest.json` 记录生成代码提交、配置、随机化范围、源文件及其哈希；不要复制官方 RoboMimic 的 source_sha256 来冒充新数据来源。避免自引用哈希：先完成稳定的 source_manifest，再计算其哈希写入主 manifest。

## 7. metadata 与回放证据

每个 episode 的 metadata 至少包含：原始语言 prompt、task、manner tag、seed、reset/source_attempt/lineage ID、环境和控制器完整配置或可解析引用、生成器版本、初始物体位姿、相机参数、夹爪正负方向实测结论、动作来源（原始执行或转换后重新执行）、task_ok/manner_ok 及判据和实测指标。

`model.xml` 和 `replay_state.npz` 保存足够重建环境的模型/资源与初始仿真状态；若控制器内部状态、随机数状态或 wrapper 状态影响回放，也须保存或明确一致的重置流程。仅提供 seed 不保证可复现。优先额外保留逐步完整 simulator state，便于定位图像和动作错位。

`replay_audit.json` 至少包含：source_sha256、被验证 episode 列表、控制器/环境版本、任务成功、manner 达标、位置误差、姿态误差、夹爪误差和通过判据。不得手填 passed=true。当前训练启动器会检查该文件的 passed 和 source_sha256，但检查通过标志本身不能代替真实回放。

当前官方样本回放脚本绑定官方 HDF5 哈希，不能直接用于新合成数据。训练侧需为新数据提供回放入口，并将真实审计结果交给当前启动器要求的位置（weights.parent/replay_audit.json）；不要覆盖正在运行的基线审计文件。新数据采用独立 run/资产目录并重新计算 train 统计。

## 8. 训练端输入样本与分块

当前读取器针对起点 t 输出：

```python
{
    'image': images[t, 0],       # uint8 [224,224,3]
    'wrist_image': images[t, 1], # uint8 [224,224,3]
    'state': state[t],          # float32 [8]
    'actions': actions[indices],# float32 [10,7]
    'prompt': prompt            # str；旧版取 manifest['task']
}
# indices = minimum(arange(t, t+10), T-1)
```

动作从当前 t 开始，不从 t+1 开始。尾部不足 10 步时当前实现重复最后一条动作；不能跨 episode 拼块。内部补到 32D 是模型变换的工作，上游只交真实 7D。不要先标准化 state/actions；训练端只用 train 数据生成新的统计量。

## 9. 交付验收与先导批次

1. 所有 episode 的图像、状态、动作长度一致；数值有限、无 object dtype；NPY/NPZ 可在 allow_pickle=False 下读取。
2. 控制频率、帧索引和时间同步验证通过，图像确实随执行变化，两个视角与状态一致；核对四元数顺序、旋转向量定义和夹爪方向。
3. 从相同初始条件执行 actions 完成真实动作回放，而非逐帧恢复状态播放。按任务和 manner 组合覆盖先导样本；报告全部结果，不只挑成功案例。当前基线的位置误差阈值为 5 mm，可作为先导参考；姿态、夹爪、manner 阈值必须另外明确。
4. 作为成功监督示范的记录必须 task_ok=true 且 manner_ok=true。失败和恢复尝试另存并标注，不作为成功演示静默混入。
5. 按源轨迹、初始场景/seed 分组划分 train/valid，来源相同的语言改写、同源变体不跨集合；不要随机拆帧。同 seed 不同 manner 可用于配对比较，但属于同一分组。
6. 使用多 manner 时，训练侧验证两条不同 prompt 确实分别到达模型；旧读取器只读统一 task 的问题修复前，不宣称已做语言条件训练。
7. 先交每个目标 task×manner 的少量完整 episode，完成读取、可视检查、控制回放和训练 smoke，再扩大 seed 和场景范围。数量只是覆盖指标，不以条数代替质量验收。

## 10. 可直接发送给上游的简版

请按当前 π0.5 的 Panda OSC 接口重新采集可训练轨迹：每条 episode 交付 images.npy（uint8，[T,2,224,224,3]，agentview+robot0_eye_in_hand）和 transitions.npz（state float32 [T,8]，actions float32 [T,7]）。state 顺序为世界系 EEF xyz、绝对轴角 rotvec、两个原始 gripper qpos；actions 必须是实际发送给固定刚度 world-frame delta OSC_POSE 的归一化 7D 命令，位置缩放 0.05 m、旋转缩放 0.5 rad、最后一维为原生夹爪控制输入，20 Hz。图像和 state 必须位于对应 action 执行前。另交逐轨迹完整语言指令、task/manner/seed、来源和 reset ID、成功与风格标签、时间及 next_state、完整环境配置和可回放初态。旧的关节/末端 telemetry 不能直接当动作标签。不同 manner 不可仅通过未进入模型动作空间的隐藏 kp/握力参数实现。训练侧负责 10 步分块、归一化及逐 episode prompt 接入；先交先导样本并通过实际动作回放，再批量扩充。

## 附录 A：可直接参考的完整环境参数

参考环境为 RoboSuite 1.5.2、Panda、Lift。以下参数来自接收方现有基线的实际环境配置；GPU 编号因机器不同已移除，由你方本机选择。使用 `robosuite.make("Lift", **kwargs)` 时以此为基准，若安装版本需要字段适配，应记录版本与映射，并证明控制语义未改变，不得静默回退到另一控制器。

相机的实际外参、内参/FOV 和模型资源由创建后的环境导出随包交付，不仅记录相机名称。若本机版本无法构建此环境，报告兼容性缺口，不要伪造等价性。

```json
{
  "has_renderer": false,
  "has_offscreen_renderer": true,
  "ignore_done": true,
  "use_object_obs": true,
  "use_camera_obs": false,
  "control_freq": 20,
  "controller_configs": {
    "type": "BASIC",
    "body_parts": {
      "right": {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [
          0.05,
          0.05,
          0.05,
          0.5,
          0.5,
          0.5
        ],
        "output_min": [
          -0.05,
          -0.05,
          -0.05,
          -0.5,
          -0.5,
          -0.5
        ],
        "kp": 150,
        "impedance_mode": "fixed",
        "kp_limits": [
          0,
          300
        ],
        "position_limits": null,
        "orientation_limits": null,
        "uncouple_pos_ori": true,
        "interpolation": null,
        "ramp_ratio": 0.2,
        "input_ref_frame": "world",
        "gripper": {
          "type": "GRIP"
        },
        "damping_ratio": 1,
        "damping_ratio_limits": [
          0,
          10
        ],
        "input_type": "delta"
      }
    }
  },
  "robots": [
    "Panda"
  ],
  "camera_depths": false,
  "camera_heights": 224,
  "camera_widths": 224,
  "lite_physics": false,
  "reward_shaping": false,
  "camera_names": [
    "agentview",
    "robot0_eye_in_hand"
  ]
}
```

## 附录 B：跨服务器打包与最终回复

上游负责：采集器和导出器、先导数据、格式验证、连续执行保存动作的回放工具及报告、可复现实验配置。训练侧负责：逐 episode prompt 接入、新数据统计归一化、训练 smoke 和实际训练。上游不必持有模型权重，也无需修改接收方正在运行的训练。

最终交付一个可复制到任意服务器的压缩包，例如 `pi05_synthetic_pilot.zip` 或 `.tar.gz`，其中包含：

- `dataset/`：本文要求的数据与审计附件。
- `tools/`：生成/导出入口、格式验证脚本、动作回放脚本及其必需的本地模块。
- `requirements.txt` 或锁文件：记录 Python、RoboSuite、MuJoCo、NumPy 等实际版本。
- `README.md`：从解压后的目录出发可执行的安装、生成、验证、回放命令；说明硬件/渲染要求。
- `SHA256SUMS`：交付文件的哈希清单，不对该清单自身计算递归哈希。

包内不得依赖仅存在于你方服务器的绝对路径、软链接目标或未交付私有模块。XML 引用的 meshes/textures 等资源必须随包提供，或明确可复现的依赖版本与获取方式；不包含凭证、token 或私钥。文件体积较大时可分卷，同时提供分卷哈希与重组命令。

交付前将压缩包解压到一个不同目录，重新运行格式检查与至少一条真实动作回放，确认相对路径和依赖有效。该检查不能替代前述按 task×manner 覆盖的先导验收。

最终回复请列出：

1. 可传输的压缩包文件或实际可访问的下载地址；不要仅给本机目录。
2. episode 数、总帧数、train/valid 划分、各 task×manner 的数量。
3. 格式/时序检查结果、真实动作回放结果、任务与 manner 达标率及分母。
4. 仍存在的不兼容项、未验证项和限制。
5. 完整可复现命令与压缩包 SHA-256。

不要在未通过验证时声明“可直接训练”。格式兼容、动作可回放、manner 可复现、模型已训练是不同阶段，分别报告。
