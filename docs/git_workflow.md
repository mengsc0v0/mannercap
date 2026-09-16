# MannerCAP Git 工作流

## 仓库和分支

- 上游仓库：`https://github.com/mengsc0v0/mannercap.git`
- v2 启动基线：`2fa18c4cf6b750b9f8539e3fd760ddc2f00ddaf8`
- 当前集成分支：`work/synthesis-v2-gate0-20260915`
- 4090 工作副本：`/data/gaochuanhao/mannercap`

开发先进入 `work/<scope>-<gate>-<date>` 分支。每个提交只包含一个可审查的目的，并在提交前运行对应测试。Gate 的协议、阈值或数据划分冻结后用带注释标签，例如 `synthesis-v2-gate1`；没有通过关口时不创建完成标签。

## Git 跟踪范围

Git 跟踪源码、测试、配置、实施方案、审计报告、环境锁和证据哈希。原始轨迹、视频、模型、虚拟环境、离线 wheelhouse 和运行日志保存在 `/data/gaochuanhao`，通过清单、SHA-256、代码提交和命令关联。大文件后续如需跨机器版本化，应使用项目批准的对象存储或 Git LFS，并在 `data_refs` 中登记；不能直接提交普通 Git blob。

敏感凭据不进入 Git。环境模板只能保存变量名和占位符。

## 4090 同步

服务器不能直连 GitHub 时，以本地可联网工作副本作为 Git 传输端：

1. 在本地分支完成并验证提交。
2. 生成包含该分支的 Git bundle，复制到服务器。
3. 在服务器仓库 fetch bundle，并将同名分支快进到提交；运行数据目录不参与 checkout。
4. 在服务器重跑测试，把结果和证据哈希提交回同一分支。
5. 从本地将分支推送到 GitHub，并由项目负责人按关口决定是否合并或打标签。

服务器分支不得使用 `reset --hard` 覆盖未登记的工作。同步前先检查 `git status`；有服务器独有修改时先提交到单独分支或导出补丁。

## 提交前检查

```bash
git diff --check
PYTHONPATH=.:stagecap_robosuite_libero_repro_20260714 \
  /data/gaochuanhao/mannercap_env/bin/python -m pytest -q \
  tests stagecap_robosuite_libero_repro_20260714/tests/test_v2_evaluation_contract.py
git status --short
```

真实运行必须使用新的输出目录，保留失败尝试。提交报告时同时更新 `reports/gate0/evidence_manifest.sha256` 或对应关口清单。
