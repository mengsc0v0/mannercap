# Stage-CAP + Online KITE 可复现优化说明

这份文档说明我们现在优化 LIBERO40 数据生成方案时，什么算方法本身，什么不算。核心目标是让后续同学可以复现同一套流程，而不是依赖某个任务的手工坐标、隐藏轨迹或临时提示词。

## 1. 正式评测边界

正式结果必须满足这些条件：

- 关闭 deterministic fast-path。它只能用于调试 API 或定位问题，不能计入正式成功率。
- 每个任务最多 10 次 stage 尝试，超时后直接进入下一个任务。
- 使用 `mimo-v2.5`，因为后续 Online KITE 失败帧反馈需要支持图像输入。
- 每次结果必须记录是否启用 SAM3 初始感知。
- 需要分开报告两组结果：clean/no-SAM3 Stage-CAP + Online KITE，以及 SAM3 initial perception 对比组。

## 2. 我们的方法到底优化什么

我们的优化不是给某个任务写死成功程序，而是不断收紧一套通用的“生成-执行-验证-反馈”边界。

1. Stage-CAP 先把自然语言任务拆成可验证子阶段。
2. MIMO 只为当前子阶段生成 CAP 代码。
3. 执行当前子阶段，并收集结构化状态、视频帧、关键失败帧和几何诊断。
4. Online KITE 在单轮次内部定位失败发生在哪个子步骤，而不是等整条轨迹失败后再笼统分析。
5. 反馈信息回到当前子阶段或剩余后缀阶段，指导 MIMO 重新生成更合适的 stage code。
6. verifier 根据边界条件决定继续、重试当前阶段、重规划后续阶段，或者判定任务失败。

因此，真正有效的应该是：

- 阶段规划更合理；
- 每个阶段的通过/失败条件更明确；
- 生成代码更遵守 API 和动作 contract；
- 失败帧诊断能告诉模型当前阶段具体哪里没达成；
- 下一次生成代码能基于这些通用反馈改正。

## 3. 允许的通用优化

以下修改可以进入正式方法：

- 通用任务族的 stage 分解规则，例如 pick-place、drawer、stove、push-region。
- 通用阶段边界，例如 object held、lift_delta、transport arrived、drawer open fraction、handle contact、BDDL relation。
- 通用 API contract，例如函数参数、返回值结构、`STAGE_STATUS` 输出格式、numpy 序列化规则。
- 生成代码应使用 runner 提供的 `STAGE_ID` 和 `STAGE_ATTEMPT`，不要在代码里重新写死这两个值。
- 通用动作 contract，例如 pick 阶段每次只调用一个 audited pick skill，release 阶段不能重新抓取物体。
- 通用 preflight 检查，在代码真正执行前拦截错误参数、错误 import、未定义变量和危险调用。
- 通用 KITE 反馈模板，把失败帧和诊断转换成可复用的 repair context。

## 4. 禁止的捷径

以下内容不能进入正式结果：

- 为某个 task 写死成功代码。
- 写死某个物体、抽屉、把手或容器的固定坐标。
- 重放隐藏 oracle 轨迹或 demo action。
- 针对某个失败样例人工写一条特殊提示词。
- 恢复 deterministic fast-path，然后把结果当成 Stage-CAP + Online KITE 的效果。

## 5. 当前已经明确的通用 contract

### 5.1 Pick-lift 阶段

- 先读取 `get_object_grasp_affordance(object_name)`。
- 几何尺寸是主要依据，物体名字只作为弱提示。
- 如果 z extent 明显大于 xy extent，才优先 side grasp。
- flat、wide、cube-like 物体不要强行 side grasp。
- 每次 stage attempt 只调用一个 audited pick skill。
- 如果 audited pick skill 返回 `held=True` 且 `lift_delta` 足够，立刻结束该阶段。
- 成功 pick 后不要再额外 lift 或 probe，因为这可能把已经抓住的物体弄掉。

### 5.2 Transport 阶段

- 必须先验证物体仍可运输。
- 使用 `guarded_transport_held_object_to_target(object_name, predicate=..., target_name=...)`。
- 不允许把 raw target coordinate 传给 guarded transport。
- 如果 transport 过程中物体掉了，应该回到 pick-lift，而不是继续 release。

### 5.3 Release 阶段

- 使用 `low_release_object_to_target(object_name, predicate=..., target_name=...)`。
- `approach`、`retreat`、`release_z` 必须是数字或省略，不能是 `"linear"` 这类字符串。
- release 阶段不允许 import 项目 helper，也不允许重新调用 pick skill。
- 如果物体已经在目标附近但 relation 还没满足，用 bounded `local_placement_closure(..., max_iters=2)` 做局部闭合。
- 如果物体已经不在可释放状态，输出 `needs_repair`，让 stage loop 返回 pick-lift，而不是在 release stage 里重新抓。
- 如果已经 `held=False` 且 placement relation 仍然是 false，这说明 release 阶段的物理前提已经丢失，verifier 应该回退到 pick-lift/transport，而不是继续消耗 release 尝试。

### 5.4 Drawer 阶段

- 抽屉任务必须先有可验证的开抽屉子阶段。
- 不能把“手移动了”当成“抽屉打开了”。
- handle capture 需要检查 `ready_to_close`、`closed_on_handle`、`likely_hooked` 或 contact。
- 小拉测试阶段只做短距离 pull，并比较前后 `open_fraction`。
- 放物体进抽屉前，需要确认 drawer open fraction 足够。

## 6. Online KITE 在单轮次里做什么

我们想要的 KITE 不是只在整条轨迹失败后解释原因，而是在一个 task trial 内部持续提供反馈：

- 当前 stage 执行失败时，保存关键失败帧。
- 同步保存 RGB、depth、segmentation、point cloud 或相关几何诊断。
- 把失败帧和结构化状态一起输入给支持图像的模型。
- 输出当前 stage 的失败原因和下一次 stage code 应该遵守的 repair guidance。
- repair guidance 必须来自通用模板和诊断字段，不能人工为单个 task 编写。

这样做的意义是：失败不再只是“这个任务没成功”，而是变成“哪个阶段、哪个条件没满足、下一次该怎样修正”。

## 7. 当前实验记录规则

每次实验至少记录：

- output directory；
- suite 和 task ids；
- 模型名称；
- SAM3 是否开启；
- deterministic fast-path 是否关闭；
- max total stage attempts；
- timeout 设置；
- 每个 task 的 reward、task_completed、sandbox return code；
- 失败任务的最后失败 stage、原始 `STAGE_STATUS`、KITE 反馈和截图路径。

## 8. 当前推进顺序

短期先做：

1. 在 clean/no-SAM3 条件下稳定简单 object tasks。
2. 修复已经暴露的通用 code contract 问题，例如 release 参数、未定义变量、错误 import。
3. 针对同一组 task 跑 SAM3 initial perception 对比。
4. 再推进 drawer/contact-heavy 任务。

中期再做：

1. LIBERO object 0-9 小批量稳定。
2. 扩展到 LIBERO40 全量。
3. 汇总 clean/no-SAM3 与 SAM3 对比成功率。
4. 整理失败帧、诊断图、成功/失败 gif，作为论文图和 PPT 素材。

## 9. 如何判断一次优化是有效的

一次修改只有在满足下面条件时才算方法改进：

- 它不依赖单个 task 的坐标或人工特殊提示。
- 它能解释为通用 stage boundary、API contract、action contract 或 feedback template。
- 它至少在一个小批量任务上没有明显回退。
- 它能在失败报告中留下可复查的证据。

如果一个修改只让某个 task 过了，但无法推广，也无法写进这份文档的允许范围，就不能作为正式方法的一部分。
