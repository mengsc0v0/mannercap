# Stage-CAP 训练数据构造代码库

这个目录是从论文实验代码 `vlm_planner_cap` 拆出来的独立工作区，专门用于 **CAP 训练数据构造**。后续关于特权数据生成、成功轨迹筛选、训练样本导出、失败帧反馈实验等改动，都优先放在这里；原来的 `vlm_planner_cap` 保持作为论文复现和图表材料工作区。

## 代码库边界

- 当前代码库：`/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation`
- 原论文工作区：`/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/vlm_planner_cap`
- 默认输出目录：当前代码库下的 `outputs/`
- 默认日志目录：当前代码库下的 `logs/`

这个代码库默认关注高成功率数据生成，因此主路线是：

> Stage-CAP + Online KITE + oracle-readonly / privileged diagnostics

也就是说，它可以读取 LIBERO 仿真中的只读真值诊断信息来提高数据构造成功率，但生成的 CAP 代码仍然需要通过机器人动作完成任务，不能直接修改仿真状态。

## 推荐入口

详细启动说明见：

```text
docs/privileged_libero_data_generation_zh.md
```

推荐主脚本：

```text
scripts/run_libero_stage_planner_batch.py
```

推荐先跑小批量 smoke test：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation

export PYTHONPATH=/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation:${PYTHONPATH:-}
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
export MIMO_API_KEY=<your_key>

/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python scripts/run_libero_stage_planner_batch.py \
  --experiment-label stagecap_data_generation_smoke \
  --suites libero_spatial \
  --task-ids 0 1 \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_data_generation_smoke_$(date +%Y%m%d_%H%M%S) \
  --total-trials 1 \
  --feedback-rounds 2 \
  --num-workers 1 \
  --record-video True \
  --oracle-readonly \
  --use-stage-verifiable-runner \
  --use-kite-feedback \
  --max-total-stage-attempts 10 \
  --round-timeout-sec 420 \
  --stage-snapshot-chars 2200 \
  --save-stage-snapshots \
  --no-allow-point-prompt-molmo \
  --skip-api-servers
```

## 数据构造注意事项

- 训练数据构造结果必须标注 `oracle_readonly=true`，不要和 clean 公平评测结果混报。
- `deterministic fast-path` 只用于调试，不作为正式数据构造路径。
- `outputs/`、`logs/`、视频和大体积数组默认不进版本库。
- 如果要收集视觉诊断素材，可以在文档里的视觉诊断版本中打开 SAM3 和 MIMO point prompt。
- 成功样本优先筛选 `sandboxrc_0_reward_1.0_taskcompleted_1` 的 trial；`taskcompleted_1` 但 `sandboxrc!=0` 的样本只能作为候选，需单独复查。

