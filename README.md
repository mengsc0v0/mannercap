# Manner 消融实验运行指南

> **2026-09-16：v2 已启动。** 当前服务器环境、真实 Lift/Stack 开发运行、失败记录、验证结果和后续任务见 [v2 启动交付](reports/gate0/README.md)。下文是旧驱动说明；其 L2/`manner_infeasible` 输出不能作为 v2 物理验收。当前代码的原 PyRoKi 路径仍依赖 8116 IK 服务；新增本地 MuJoCo IK 是显式选择的开发后端，不能据此声称原环境已复现。

`run_manner_ablation.py` 生成 manner-conditioned 轨迹，输出 L1（任务是否成功）
/ L2（是否忠实按人话执行，`MannerValidator` 判定）/ L3（抓取力、jerk 等物理量
与 baseline 对比）三层指标。L2 满足但 L1 失败 → `manner_infeasible`（诚实失败）。

## 目录结构

```
new/
├── run_manner_ablation.py          # 唯一的实验驱动脚本
├── upgrade.txt                     # MannerCap 设计笔记
├── repro_results/robosuite/        # 实验结果输出位置（用这里，不要用默认路径）
├── _archive/                       # 本次清理的备份包
├── LA4VLA/                         # VLA 训练代码
└── stagecap_robosuite_libero_repro_20260714/   # 官方 StageCap 包（cwd 必须切到这里）
```

## 运行

Python：`/workdata/cap-x/.venv/bin/python3`。**必须先 cd 到内层包目录**（`--output`
等路径按 cwd 解析），不需要启动 IK 服务（lift/stack 用关节空间控制器）。

```bash
cd /workdata/stage_cap_privileged_data_generation_repro_20260629/robosuite/stagecap_robosuite_libero_repro_20260714/new/stagecap_robosuite_libero_repro_20260714

# Mock 模式：不消耗 API，几分钟内跑完
/workdata/cap-x/.venv/bin/python3 ../run_manner_ablation.py \
  --tasks lift stack --seeds 1 --mock-llm \
  --output ../repro_results/robosuite/manner_ablation_smoke.jsonl

# 真实 LLM 模式（Gemini via aihubmix）
export AIHUBMIX_API_KEY=sk-...
/workdata/cap-x/.venv/bin/python3 ../run_manner_ablation.py \
  --tasks lift stack --seeds 5 \
  --output ../repro_results/robosuite/manner_ablation_llm.jsonl

# 后台跑批
setsid nohup /workdata/cap-x/.venv/bin/python3 -u ../run_manner_ablation.py \
  --tasks lift stack --seeds 5 --mock-llm \
  --output ../repro_results/robosuite/manner_ablation_v2.jsonl \
  > /tmp/manner_ablation_run.log 2>&1 < /dev/null & disown
```

## 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--tasks` | `lift` / `stack`，可多选 | `["lift"]` |
| `--seeds` / `--seed-start` | 每组合 seed 数 / 起始 seed | `1` / `42` |
| `--mock-llm` | 固定 `MannerSpec`，跳过真实 LLM | 关闭 |
| `--no-baseline` | 跳过无约束 baseline 对照 | 默认跑 baseline |
| `--output` | 结果 jsonl 路径（相对 cwd） | `repro_results/robosuite/manner_ablation.jsonl` |
| `--dry-run` | 只打印计划不执行 | 关闭 |

## 输出

每行一条 JSON：`task`、`manner_tag`（9 个内置 case 或 `baseline`）、`task_ok`(L1)、
`manner_ok`(L2)、`outcome`（`both_pass`/`task_only`/`manner_infeasible`/`both_fail`/
`error`）、`metrics`（`peak_grip_force`/`jerk_rms`/`finger_sep_hold_mean`等 L3）。

内置 9 个 case：`gentle`/`ultra_careful`/`ultra_gentle`/`normal`/`fast`/`firm`/
`fast_then_slow`/`gentle_en`/`fast_en`。

**已知基线**（seed=42, mock-llm，见 `manner_ablation_smoke_grip_fix_v1.jsonl`）：
lift 10/10 both_pass；stack 8/10 both_pass，2/10 manner_infeasible（`fast`掉块、
`gentle` clip 后打滑，均为合理诚实失败）。

## 常见问题

- 改了脚本但行为没变：确认用终端 bash（而非 VS Code 编辑器）改的文件，二者在本
  工作区可能访问不同文件系统视图。
- 输出路径不对：检查是否忘记 `cd` 到 `stagecap_robosuite_libero_repro_20260714/`。
