# 固定 VLA 学生候选审计

日期：2026-09-17  
候选：`LA4VLA_1B / phasefocus35 step_500`  
决定：**登记为固定学生候选；adapter 与 30 步控制回放已完成，暂不冻结。**

## 已确认内容

服务器 `/data/gaochuanhao/la4vla` 中当前有现成评估证据的最佳候选是
`ckpts/libero_phasefocus35_from55k_lr1e6_1k/step_500`。配置为 LA4VLA-1B、
InternVL3-1B 视觉语言骨干、Flow Matching 动作头、50 步动作 horizon、24 维
训练槽位，其中评估脚本只启用前 7 维。外部目录没有 Git 元数据，因此没有编造
commit；检查点、配置、归一化统计和评估脚本均在候选配置中以 SHA-256 固定。

现有 LA4VLA 报告记录该 checkpoint 在 LIBERO-Goal 两个种子、共 100 条闭环
评估中为 43/50 和 41/50，均值 84%。这是外部候选已有证据，不是 MannerCAP
Lift、Stack 或 Wipe 的结果，也未作为本项目论文结果复用。

版本化候选配置：
`configs/student_baselines/la4vla_1b_libero_goal_candidate.yaml`。

## 动作接口结论

`eval_libero_goal.py` 将反归一化后的前 7 维直接送给 LIBERO 默认控制器：

- 前 3 维：归一化的 OSC 平移增量，控制器逐轴缩放到 ±0.05 m；
- 中间 3 维：归一化的 OSC 旋转增量，控制器逐轴缩放到 ±0.5 rad；
- 最后 1 维：Panda gripper，`-1=open`、`+1=close`；
- 控制频率：20 Hz；控制器为 `OSC_POSE` 且 `control_delta=true`。

这与当前 MannerCAP `lerobot_export_candidate` 的 7 维含义不同。后者保存的是
相邻观测的 EEF 平移状态差、axis-angle 坐标分量差和未验证的 raw gripper 值。
维数相同不能证明两者可互换。代码合同已增加独立
`la4vla_libero_osc_pose` 接口，并实现了基于相邻同步 EEF 位姿的显式 adapter。
30/30 个真实 OSC 控制步已执行，但独立误差阈值尚未冻结，两个初始化边界样本
还有约 10 mm / 0.047 rad 的一步误差，因此正式转换入口仍以
`adapter_not_verified` 失败关闭。详见 `la4vla_osc_roundtrip_results.md`。

## 冻结前阻塞项

1. 由独立评测角色冻结 OSC 跟踪误差阈值并签收 30 步结果；
2. 在 Lift/Stack（以及 Wipe 若保留）目标任务上验证观测键、相机、语言、动作
   horizon 和 reset/init-state 接口；
3. 冻结 checkpoint、代码文件 hash、训练配置、选模规则及 MannerCAP 数据输入
   配额后，才将 `fixed_student_approved` 改为 true。

本次审计没有训练模型、没有运行新闭环评估、没有修改外部 LA4VLA 目录，也没有
调用外部 API。
