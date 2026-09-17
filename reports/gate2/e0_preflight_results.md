# Gate 2 / E0 无 API 预检结果

日期：2026-09-17  
运行目录：`/data/gaochuanhao/mannercap_runs/gate2_e0_preflight_20260917`

对 Gate 1 的实际 draft manifest 执行离线预检：

- request：14；
- 方法/request lane：70；
- G0/G1/G2/G3：建立 deterministic `pending` 槽位；
- M：`skipped/api_not_configured`；
- 实际 AttemptRecord：0；
- 计费 attempt：0；
- task/manner/safety/overall：全部 `unknown`；
- simulator、controller、model API、external API、final audit：全部未调用。

输出文件 SHA-256：
`190ce58ad2e394e8de90f02d0a62226160a398fc93d44df9fe0c8d582f5359fc`。
内部 canonical report hash：
`b1d32cef06e81b9249883c89d7a2f17f3c0f25d48254baf11b090e12feaa7d64`。

该结果确认五方法矩阵、共同预算、M 的无 API 跳过和 unknown 边界可以从真实
draft manifest 确定性生成。它没有运行候选生成、仿真或最终审计，不是 E0
成功率或质量结果。
