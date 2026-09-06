# LIBERO 特权数据构造方案启动说明

本文档面向接手数据生成工作的同学，说明如何启动我们当前在 LIBERO 中成功率最高的一条数据构造 pipeline。这里的目标不是做公平对比实验，而是尽量稳定地产生可用于训练的成功轨迹、视频、阶段代码和失败修复记录。

## 1. 这条 pipeline 用来做什么

这条方案用于自动构造训练数据：

- 输入：LIBERO task 的自然语言目标和初始仿真状态。
- 中间过程：MIMO 生成阶段化 CAP 代码，机器人在仿真中真实执行动作。
- 反馈：每个阶段都有边界检查；失败时保存关键失败帧，并由 Online KITE 反馈给下一轮代码生成。
- 输出：成功轨迹、执行视频、每轮生成的 CAP 代码、阶段状态、失败帧诊断和最终汇总报告。

这条方案应该被标注为：

> Stage-CAP + Online KITE under oracle-readonly / privileged diagnostics.

意思是：模型可以读取仿真里的只读真值诊断信息，例如物体位姿、目标区域、抽屉把手位置、抽屉开合进度和关系诊断；但生成的代码仍然需要通过机器人动作完成任务，不能直接修改仿真状态。

## 2. 为什么训练数据构造要用特权方案

我们的目的不是证明机器人只靠视觉就能完成任务，而是先把高质量的成功数据构造出来。对于训练数据来说，最重要的是：

1. 成功率要高，否则大量算力会浪费在失败轨迹上。
2. 轨迹要真实执行，不能直接 set state 或瞬移物体。
3. 过程要可解释，要知道每个任务被拆成了哪些阶段、每个阶段为什么成功或失败。
4. 失败要能在单轮内部修复，而不是等整条轨迹失败后才总结原因。

因此我们使用 oracle-readonly 特权诊断作为数据构造工具。它给模型可靠的状态依据，减少纯视觉定位误差，同时仍保留机器人真实运动和阶段验证闭环。后续如果要做公平评测，需要单独跑 clean / CAP-X Agent0 对齐配置，不能把这里的成功率直接拿去和 clean baseline 比。

## 3. 当前推荐方案

默认推荐：

- Planner / code generator：`mimo-v2.5`
- 主脚本：`scripts/run_libero_stage_planner_batch.py`
- 执行方式：Stage-CAP 阶段规划 + stage-verifiable runner
- 反馈方式：Online KITE failure-frame feedback
- 状态输入：`--oracle-readonly`
- deterministic fast-path：关闭。它只用于调试，不进入正式数据构造。
- SAM3 / point prompt：默认关闭。oracle-readonly 本身已经能提供可靠状态；如果需要额外保存视觉诊断素材，再打开 SAM3/MIMO point prompt 版本。

核心启动参数是：

```bash
--oracle-readonly
--use-stage-verifiable-runner
--use-kite-feedback
--max-total-stage-attempts 10
--record-video True
--no-allow-point-prompt-molmo
--skip-api-servers
```

## 4. 启动前检查

进入项目目录：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation
```

需要确认 Python 环境存在：

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python --version
```

需要确认 MIMO endpoint 可用，并设置：

```bash
export MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
export MIMO_API_KEY=<your_key>
```

如果使用本地 OpenRouter 代理，也可以把 `MIMO_BASE_URL` 设成：

```bash
export MIMO_BASE_URL=http://127.0.0.1:8110
```

机器人运动服务至少需要 PyRoKi。通常端口是：

- PyRoKi：`127.0.0.1:8116`
- GraspNet：`127.0.0.1:8115`，有些抓取 skill 会用到，建议保持启动。
- SAM3：`127.0.0.1:8214` 或 `127.0.0.1:8114`，默认特权数据构造不依赖它，视觉诊断版本才需要。

可以用下面方式粗略检查端口：

```bash
ss -tln | grep -E ':8115|:8116|:8214|:8114'
```

如果 SAM3 要用于视觉诊断，当前已验证可用的服务是：

```bash
export SAM3_SERVICE_URL=http://127.0.0.1:8214
```

## 5. 推荐启动命令：高成功率特权数据构造

这是当前建议交给数据生成同学使用的主命令：

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation

export PYTHONPATH=/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation:${PYTHONPATH:-}
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
export MIMO_API_KEY=<your_key>

/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python scripts/run_libero_stage_planner_batch.py \
  --experiment-label stagecap_privileged_data_generation \
  --suites libero_10 libero_object libero_spatial libero_goal \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_privileged_data_generation_$(date +%Y%m%d_%H%M%S) \
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

如果只是先试一个 suite 或几个 task，可以加：

```bash
--suites libero_spatial --task-ids 0 1 2
```

如果中途断掉，默认 `resume=True`，重新用同一个 `--output-dir` 启动即可跳过已有结果。若要强制重跑，需要加：

```bash
--force-rerun-tasks
```

## 6. 可选启动命令：打开 SAM3 + MIMO point prompt 诊断

如果希望同时保留更接近 KITE 图像诊断的视觉素材，可以打开 SAM3 和 point-prompt。这里的 point-prompt 服务名仍叫 Molmo，但后端可以直接用 `mimo-v2.5`。

```bash
cd /home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation

export PYTHONPATH=/home/lifuchun/snap/snapd-desktop-integration/autorob/research_code/stage_cap_data_generation:${PYTHONPATH:-}
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
export MIMO_API_KEY=<your_key>
export SAM3_SERVICE_URL=http://127.0.0.1:8214
export MOLMO_BASE_URL="$MIMO_BASE_URL"
export MOLMO_API_KEY="$MIMO_API_KEY"
export MOLMO_MODEL=mimo-v2.5

/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python scripts/run_libero_stage_planner_batch.py \
  --experiment-label stagecap_privileged_data_generation_with_vision_diagnostics \
  --suites libero_10 libero_object libero_spatial libero_goal \
  --models mimo-v2.5 \
  --output-dir outputs/stagecap_privileged_data_generation_vision_$(date +%Y%m%d_%H%M%S) \
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
  --use-sam3-perception \
  --allow-point-prompt-molmo \
  --sam3-service-url "$SAM3_SERVICE_URL" \
  --molmo-base-url "$MIMO_BASE_URL" \
  --molmo-api-key "$MIMO_API_KEY" \
  --molmo-model mimo-v2.5 \
  --skip-api-servers
```

注意：这个版本更适合收集失败诊断素材；如果目标只是尽快构造成功训练轨迹，优先使用上一节的默认版本。

## 7. 输出目录怎么看

每次实验输出根目录类似：

```text
outputs/stagecap_privileged_data_generation_YYYYMMDD_HHMMSS/
```

根目录下最重要的文件：

- `experiment_manifest.json`：本次跑了哪些 suite/task、用什么模型和参数。
- `experiment_rules.md`：本次实验的边界说明，包括是否 oracle-readonly、是否启用 KITE、是否启用 SAM3。
- `libero_stage_planner_report.json`：最终汇总报告，统计每个任务是否成功。

每个任务目录一般形如：

```text
outputs/.../<suite>/<task_id_and_name>/
```

里面重点看：

- `mimo-v2.5_stage_planner_record.json`：这个任务的完整阶段规划、每轮执行记录、是否成功、失败原因摘要。
- `round_XX/config.yaml`：该轮实际启动的 LIBERO/CAP-X 配置。
- `round_XX/run/`：CAP-X 实际执行输出，包含 trial 目录、生成代码、日志和视频。

成功 trial 目录名通常包含：

```text
taskcompleted_1
```

例如：

```text
trial_0_sandboxrc_0_reward_1.0_taskcompleted_1
```

筛选训练数据时，优先保留同时满足下面条件的样本：

1. `task_completed=True`
2. `reward >= 1.0`
3. `sandbox_rc == 0`
4. 视频存在且动作过程完整
5. 没有直接修改仿真状态的 forbidden shortcut

有些历史运行可能出现 `reward=1/task_completed=True` 但 `sandbox_rc!=0` 的情况，通常是任务完成后诊断或收尾报错。它可以作为“候选可用轨迹”单独复查，但不要直接混入严格成功集。

## 8. 如何快速统计成功率

推荐先看汇总报告：

```bash
/home/lifuchun/data/capx/repos/cap-x/.venv-libero/bin/python scripts/summarize_stage_planner_report.py \
  outputs/stagecap_privileged_data_generation_YYYYMMDD_HHMMSS/libero_stage_planner_report.json
```

也可以直接从目录粗略统计成功 trial：

```bash
find outputs/stagecap_privileged_data_generation_YYYYMMDD_HHMMSS -type d -name '*taskcompleted_1*' | wc -l
```

这个目录统计只适合快速看进度，正式结果仍以 `libero_stage_planner_report.json` 和每个任务的 `mimo-v2.5_stage_planner_record.json` 为准。

## 9. 这条方案和 clean 公平评测的区别

特权数据构造允许读取：

- 物体真值位姿
- 目标区域真值位姿
- 抽屉把手真值位置
- 抽屉开合进度
- BDDL relation diagnostics
- `reward/task_completed` 作为阶段检查或最终成功判断

但仍然禁止：

- `set_state`
- `set_qpos`
- 直接移动物体到目标位置
- 直接调用任务完成 helper
- 重放隐藏 oracle action trajectory
- deterministic fast-path 计入正式数据构造结果

clean 公平评测则不能使用上述真值诊断输入，必须和 CAP-X Agent0 对齐原始可获得输入。clean 实验请看：

```text
docs/stage_cap_fair_evaluation_protocol_zh.md
```

## 10. 给接手同学的推荐工作流

建议按下面顺序推进：

1. 先跑 1 个简单 task，确认 MIMO、PyRoKi、视频保存和 report 都正常。
2. 再跑每个 suite 的 1-2 个 task，确认不同任务族都能产出完整目录。
3. 然后跑 LIBERO-40 全量，保持 `--num-workers 1`，避免 GPU/仿真服务抢资源。
4. 跑完后用 report 统计严格成功、候选成功、失败和超时。
5. 对失败任务查看 `mimo-v2.5_stage_planner_record.json`，重点看失败 stage、KITE feedback、stage snapshot。
6. 对成功任务检查视频，确认轨迹确实完成了自然语言目标。
7. 最后把成功样本复制或索引到训练数据清单中，同时保留原始输出目录，方便追溯。

## 11. 数据标注建议

生成的数据集建议至少记录这些字段：

- `task_suite`
- `task_id`
- `task_language`
- `output_dir`
- `trial_dir`
- `video_path`
- `generated_code_path`
- `stage_record_path`
- `oracle_readonly=true`
- `use_kite_feedback=true`
- `use_stage_verifiable_runner=true`
- `use_sam3_perception`
- `allow_point_prompt_molmo`
- `strict_success`
- `candidate_success`

其中 `oracle_readonly=true` 很关键。后续训练可以使用这批数据，但论文或 PPT 中必须明确它是“特权数据构造方案”的产物。

## 12. 常见问题

### 12.1 `point_prompt_molmo` 报 127.0.0.1:8122 连接失败

说明代码启用了 point-prompt，但对应服务没有启动。当前推荐做法有两个：

- 默认特权数据构造：关闭 point-prompt，使用 `--no-allow-point-prompt-molmo`。
- 需要视觉诊断：把 Molmo 后端改成 MIMO，设置 `MOLMO_BASE_URL/MOLMO_API_KEY/MOLMO_MODEL`，并在命令中加入 `--allow-point-prompt-molmo`。

### 12.2 SAM3 端口冲突

如果 `8114` 已被占用，可以使用已验证的 `8214`：

```bash
export SAM3_SERVICE_URL=http://127.0.0.1:8214
```

并在启动命令里传：

```bash
--sam3-service-url "$SAM3_SERVICE_URL"
```

### 12.3 任务明明成功但 report 里不算严格成功

先看 trial 目录名。如果是 `taskcompleted_1` 但 `sandboxrc` 非 0，可能是任务完成后的诊断或收尾代码报错。训练数据构造时可以把这类样本放入候选集人工复查；正式严格成功集只保留 `sandboxrc_0_reward_1.0_taskcompleted_1`。

### 12.4 是否可以打开 deterministic fast-path

不建议。fast-path 适合调试服务、动作 primitive 或验证某个任务本身是否可完成，但它绕过了“让 MIMO 生成阶段代码并由反馈修复”的核心流程。正式数据构造默认关闭它。
