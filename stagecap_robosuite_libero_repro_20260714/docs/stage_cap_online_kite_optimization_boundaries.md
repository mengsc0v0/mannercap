# Stage-CAP + Online KITE Optimization Boundaries

This note defines what we count as method improvement during LIBERO40 evaluation.

## Formal Evaluation Constraints

- `deterministic fast-path` must be disabled.
- Each task may use at most 10 total stage attempts.
- The model used for the stage loop should support image inputs when Online KITE feedback images are enabled.
- Results must report the exact condition, including whether SAM3 perception is enabled.

## Allowed Method Improvements

These changes are part of the reproducible method:

- Stage decomposition and stage ordering, when expressed as generic task-family rules.
- Stage boundary conditions, such as drawer open fraction, handle contact, object held state, transport arrival, and BDDL relation checks.
- Generic API contracts, including required arguments, return-value shape, and serialization rules.
- Generic action contracts, such as using one audited pick skill per pick-lift attempt, verifying `held` and `lift_delta`, and switching grasp strategy based on object geometry.
- Generic Online KITE feedback templates that convert observed stage failure into reusable repair context.
- Generic preflight checks that reject unsafe or invalid generated code before it mutates the simulator.

## Disallowed Shortcuts

These must not be used for formal results:

- Task-specific success code.
- Fixed coordinates for a particular object, handle, drawer, or receptacle.
- Hidden oracle trajectories or replayed demonstration actions.
- Per-task prompt hacks or manually written feedback text for one failure case.
- Restoring deterministic fast-path code and reporting it as the method result.

## Current Optimization Logic

The method should improve by repeatedly tightening generic contracts:

1. Plan the task into verifiable stages.
2. Generate code for only the current stage.
3. Execute the stage and collect structured evidence.
4. If the stage fails, Online KITE extracts the key failure frame and diagnostics.
5. The replanner receives the failure evidence and revises only the current stage strategy.
6. The verifier decides whether to retry the stage, replan the remaining suffix, or proceed.

For pick-place tasks, the current generic rule is:

- Ground the object and target first.
- Pick and lift-test before any transport.
- Use object geometry from `get_object_grasp_affordance`.
- Use measured geometry as the primary grasp-strategy signal.
- For tall/narrow objects whose z extent is clearly larger than xy extent, switch to side grasp early.
- For flat/squat/wide/cube-like objects, use geometry-based grasp.
- For bowl/rim/cup objects, use low-side or rim-aware grasp.
- If an audited pick skill already reports `held=True` and sufficient `lift_delta`, the pick stage should stop immediately; extra lift/probing motion is treated as unsafe because it can drop the object before transport.
- Transport only after held/lift evidence is verified.
- Release only after target arrival is verified.

## Reporting Rules

Every experiment should record:

- Output directory.
- Task suite and task ids.
- Model name.
- SAM3 enabled or disabled.
- Fast-path enabled or disabled.
- Max total stage attempts.
- Success count and per-task status.
- Main failure stage and evidence for failed tasks.
