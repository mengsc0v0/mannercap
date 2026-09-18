# D0 方式条件轨迹与 LA4VLA 学习链路结果

日期：2026-09-18  
代码：`fa0817d7bbc14326e8cab3b9239f793939f10acf`  
最终轨迹：`/data/gaochuanhao/mannercap_runs/gate2_d0_manner_pilot_20260918_attempt04`  
结论：**零 API 的同初态 D0 采集与八提示 LA4VLA 微拟合链路通过；尚不构成方式达标、泛化或闭环学生策略证据。**

## 配对方法与缺陷修复

最初用同一环境执行多种方式，再恢复 MuJoCo 快照。反转执行顺序后，第一条轨迹
成功而恢复后的轨迹失败，证明控制器隐藏状态造成顺序混杂。之后尝试为每条轨迹
重建环境，也发现 RoboSuite 包装器没有把 seed 传给底层 Lift/Stack，且 Stack
另有初始化噪声和采样器随机源。

最终方法完成三项修复：

- Lift/Stack 包装器把 seed 传入底层 RoboSuite；
- Stack 关闭机器人初始关节噪声，并为物体采样器使用独立固定 RNG；
- 每个任务只创建一个未执行的基准环境，四种方式分别在 POSIX `fork` 的写时复制
  子进程中运行。父进程从不推进仿真，因此模型、控制器及全部内存状态一致。

早期失败目录继续保留，不进入最终训练数据。最终审计得到：

| 项目 | 结果 |
| --- | ---: |
| 计划 / 执行 | 8 / 8 |
| 执行错误 | 0 |
| 任务成功 | 8 / 8 |
| 精确同初态配对 | 2 / 2 |
| 时间积分、转移向量检查 | 全部通过 |
| 外部 API 调用 | 0 |

随后按同一冻结矩阵采集 seed 43、44：
`/data/gaochuanhao/mannercap_runs/gate2_d0_manner_extension_seeds43_44_20260918`。
新增 16/16 执行和任务成功、4/4 精确配对通过。D0 开发矩阵目前合计 24 条，
覆盖 `2 任务 × 4 指令 × 3 独立初态`；全部仍属于 development，不作为验证集或
封存测试集。

Lift 四条轨迹共享初态 SHA-256
`db368d81ec252a88553e3e5d7f365b37fa72b4e915389674fb9a77047c092e7d`；
Stack 四条共享
`2e866b28bdca1dfaeba3bcc2d30ac65acf23930987c79e6c251ac00efed3fc79`。
每组内的 MJB 哈希也完全一致。

## 描述性速度结果

下表单位为 rad/s。均值用于确认数据中存在速度差异，不是冻结后的方式合格阈值。

| 任务 | 条件 | 步数 | 全程均速 | approach | descend | lift | 任务成功 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lift | neutral | 766 | 0.1539 | 0.2736 | 0.0569 | 0.0859 | 是 |
| Lift | slow | 2016 | 0.0621 | 0.0987 | 0.0208 | 0.0346 | 是 |
| Lift | fast | 545 | 0.2126 | 0.4067 | 0.0847 | 0.1212 | 是 |
| Lift | fast approach / slow contact | 1218 | 0.1009 | 0.2808 | 0.0259 | 0.0346 | 是 |
| Stack | neutral | 1523 | 0.1040 | 0.2718 | 0.0569 | 0.0817 | 是 |
| Stack | slow | 3962 | 0.0418 | 0.0979 | 0.0208 | 0.0345 | 是 |
| Stack | fast | 1094 | 0.1437 | 0.4050 | 0.0841 | 0.1122 | 是 |
| Stack | fast approach / slow contact | 2320 | 0.0704 | 0.2784 | 0.0259 | 0.0345 | 是 |

slow 的全程和各操作阶段均低于 neutral，fast 均高于 neutral；阶段组合在接触、
抬升和后续操作阶段接近 slow。approach 包含启动瞬态，因此峰值和部分均值不能直接
解释为方式约束达标。`formal_manner_status` 与 `formal_safety_status` 仍为
`unknown`，`promotion_allowed=false`。

## 八提示 LA4VLA 数据与微拟合

从 8 条成功轨迹各等距抽取 8 个长度为 50 的窗口，共 64 个窗口、3200 帧。
数据包含第三人称与腕部 RGB、8D state、50×7 OSC action、phase marker、原始
自然语言、manner tag、请求约束、结果与初态/MJB lineage。失败轨迹默认拒绝导入，
图像文件名包含 episode id，避免同任务多方式互相覆盖。

最终目录：

```text
/data/gaochuanhao/mannercap_runs/gate2_d0_full_la4vla_microset_capture_20260918
/data/gaochuanhao/mannercap_runs/gate2_d0_full_la4vla_microset_packed_20260918
/data/gaochuanhao/mannercap_runs/gate2_d0_full_la4vla_microfit_20260918
```

LeRobot 包含 64 个 synthetic window episodes、8 条独立源轨迹、8 个不同提示词和
4 个 manner tag。打包器同时生成训练集自身的 mean/std/min/max/q01/q99，避免复用
旧数据统计。

固定微批从每条源轨迹各取一个窗口，对 LA4VLA action head 训练 100 步：

| 指标 | 结果 |
| --- | ---: |
| 固定噪声初始 loss | 0.33489233 |
| 固定噪声最终 loss | 0.03055456 |
| 最终 / 初始 | 0.09124 |
| 非有限 loss | 0 |
| 训练参数量 | 122,029,360 |
| GPU 峰值 allocation | 4,372,007,424 bytes |
| wall time | 11.30 s |
| 写入 checkpoint | 否 |

这证明八种文本、图像、状态与动作能共同进入梯度链并被固定微批过拟合。数据仍只有
两个独立初态，没有验证未见初态、未见方式组合或闭环任务成功，因此不能据此启动
正式训练结论或论文主表比较。

## 证据

五个最终目录共 349 个文件，合计约 2.05 GiB。清单：
`reports/gate2/d0_manner_final_20260918.sha256`；清单自身 SHA-256 为
`91f76037ae4eae40a554025d2791c398c8a89fc5c4c60cf9557ab499708e46ac`，
服务器执行 `sha256sum -c` 全部通过。外部 LLM/API 调用数为 0，不需要 API key。
