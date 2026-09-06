# Bundle validation report (2026-07-14)

Validation was performed with the isolated runtime paths used by the final v2
archive.

## Source isolation

- Standalone RoboSuite import: `1.5.1`, loaded from
  `capx/third_party/robosuite`.
- LIBERO RoboSuite import: `1.4.0`, loaded from
  `capx/third_party/libero_dependencies/robosuite`.
- LIBERO benchmark import exposed the four base suites used by the 40-task
  runner.
- The bundled LIBERO RoboSuite cleanup patch matches the runtime source used
  by the original experiments.

## Automated tests

```text
42 passed, 3 warnings in 7.75s
```

Two warnings are PyRoKi / jaxls factory deprecations; the third is a Python
3.12 invalid-escape `SyntaxWarning` in vendored RoboSuite ASCII art. They do
not affect the test result. The tests include seven-task configuration
isolation and reject Agent0 configs that omit privileged observations or bind
StageCaP APIs.

## Privileged Agent0 audit

All seven generated configs from the fair Mimo run were checked for:

```text
env.cfg.privileged: true
API profile: original CapX *PrivilegedApi
use_stage_verifiable_runner: false
FrankaRobosuite*StageApi count: 0
```

The seven-task result was `3/7`; see
`ROBOSUITE_FAIR_COMPARISON_20260714.md`. The older `5/7` run is explicitly
labeled as a StageCaP API without stage loop ablation.

## Simulator smoke test

Command condition: privileged deterministic StageCaP, Cube Lifting, one trial,
video disabled, model endpoint deliberately set to an unreachable local port.

```text
success_trials: 1/1
reward: 1.0
task_completed: true
sandbox_rc: 0
model requests: 0
```

The current full StageCaP contract was also run once on all seven task
families and completed `7/7` without model fallback.

## PyRoKi service

The bundled server code initialized the Panda model and reached the FastAPI
ready state on an isolated validation port (`18116`). The normal experiment
port remains `8116`.

## Secret handling

No API key is stored in source configs, compact reports, or the v2 archive.
The original uploaded archive contains a plaintext credential and must not be
shared.
