# MannerCAP v2 启动交付

先读 [基线和真实证据](baseline_inventory.md)、[任务台账](task_ledger.md)。

## 4090 上的路径

- 源码：`/data/gaochuanhao/mannercap`
- 环境：`/data/gaochuanhao/mannercap_env`
- 离线依赖：`/data/gaochuanhao/mannercap_wheels`
- 全部开发尝试：`/data/gaochuanhao/mannercap_runs`

## 验证命令

```bash
cd /data/gaochuanhao/mannercap
PYTHONPATH=.:stagecap_robosuite_libero_repro_20260714 \
  /data/gaochuanhao/mannercap_env/bin/python -m pytest -q \
  tests stagecap_robosuite_libero_repro_20260714/tests/test_v2_evaluation_contract.py

# --output 必须是新目录，不覆盖已有尝试。默认 pyroki 后端需要 8116 服务。
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /data/gaochuanhao/mannercap_env/bin/python scripts/gate0_smoke.py \
  --ik-backend mujoco-local --output /data/gaochuanhao/mannercap_runs/NEW_UNIQUE_RUN

/data/gaochuanhao/mannercap_env/bin/python scripts/gate0_smoke.py \
  --audit /data/gaochuanhao/mannercap_runs/gate0_20260916_attempt06/lift/transitions.jsonl

/data/gaochuanhao/mannercap_env/bin/python scripts/gate0_recompute_task.py \
  /data/gaochuanhao/mannercap_runs/gate0_20260916_attempt06/lift
```

Stack 将最后路径的 `lift` 替换为 `stack`。状态重算不等于动作回放。

## 接口工作

- [独立评测审计](evaluation_audit.md)
- [请求编排及方法规格初版](synthesis_method_spec.md)
- [动作和数据接口审计](action_interface_audit.md)

这些模块是未接入正式生产的原型；不能把单元测试数当成物理校准、数据发布或学生学习结果。
