# Mimo validation report (2026-07-14)

## Credential and API validation

The original uploaded `stagecap_repro_20260712.tar.gz` contains the same
plaintext `sk-...` credential in two experiment scripts. The token is not
reproduced here and is not copied into generated configs, reports, or the
sanitized reproduction archive.

Validated endpoint and model:

```text
https://api.xiaomimimo.com/v1/chat/completions
mimo-v2.5
```

Checks completed successfully:

- direct OpenAI-compatible text request: HTTP 200;
- `capx.llm.client.query_model` text request;
- 384 x 216 image input through the same client;
- a fresh authentication probe immediately before the fair baseline run.

## Baseline-label correction

The historical directories below were initially called Agent0, but their
generated YAML files bind `FrankaRobosuite*StageApi`:

```text
outputs/mimo_archive_key_agent0_validation_20260714
outputs/mimo_agent0_all7_1trial_20260714
outputs/mimo_agent0_failed2_retry_1trial_20260714
```

They disable the stage runner, but still expose StageCaP audited skills. Their
correct label is **StageCaP API without stage loop ablation**, not Agent0.

The historical seven-task API/no-loop run produced `5/7` at a 4096-token
budget. A corrected-budget retry succeeded on Spill Wipe and still failed on
Nut Assembly. These runs remain useful API and model-client smoke tests, but
must not be used as the official baseline.

## Fair privileged Agent0

Agent0 was rerun with the following enforced protocol:

- `env.cfg.privileged: true` for all seven tasks;
- only CapX's original `*PrivilegedApi` for object-state input and robot motion;
- no `FrankaRobosuite*StageApi`;
- no stage machine, verifier, local retry/replan, or deterministic fast path;
- `mimo-v2.5`, `max_tokens=20480`, temperature 1.0, one worker;
- one trial per task, video disabled.

| Task | Result | Reward | Model query |
| --- | ---: | ---: | ---: |
| Cube Lifting | 1/1 | 1.000 | 19.30 s |
| Cube Restack | 0/1 | 0.024 | 42.92 s |
| Nut Assembly | 0/1 | 0.353 | 43.18 s |
| Cube Stack | 1/1 | 1.000 | 30.85 s |
| Two-Arm Lift | 0/1 | 0.015 | 59.23 s |
| Spill Wipe | 1/1 | 1.000 | 41.98 s |
| Two-Arm Handover | 0/1 | 0.094 | 76.28 s |

Aggregate: `3/7 (43%)`.

Raw report:

```text
outputs/mimo_agent0_privileged_all7_1trial_20260714_v3/robosuite_stagecap_report.json
```

Every generated task config was audited after the run: all seven use
`privileged: true`, bind an original CapX privileged API, set
`use_stage_verifiable_runner: false`, and contain no StageCaP API.

## StageCaP model-path validation

With deterministic action fast paths disabled, Mimo generated executable
StageCaP actions for Nut Assembly (`1/1`) and Spill Wipe (`1/1` after the AST
reserved-local repair). The successful wipe reduced remaining markers
`72 -> 42 -> 19 -> 0` through three local replans without resetting the
environment. These are StageCaP model-path tests, not Agent0 runs.

## Scope

The one-trial seven-task runs establish configuration isolation and executable
end-to-end behavior. They are smoke evidence, not a success-rate estimate.
Statistical comparison requires matched seeds and repeated trials for every
condition. The plaintext token in the original uploaded archive should be
revoked before that archive is shared.
