# Step 22: T8 (put both moka pots on the stove) ordinary onboarding, round 1

Config frozen at commit aecc897 before any T8 execution (identity: `round1/identity.json`, sha256 of config, scripts, demo file).
Plan: GripPlan(1.0, 100.0, settle_after_open=1), approach check on, no guard, no manner request, dev inits 0-3.

## Demo reference (pre-set T5 rule)
- Rule: lowest-index demo replay-successful in two independent replays, used as is. Result: demo_0.
- Checked: 8 demos (0-7), 16 replays (rule needed only demo_0; the rest were run as a batch). Single replays of demos 1, 3, 7 and 5
  disagreed or failed (A False / B True for 1, 3, 7; 5 failed twice): replay success is not deterministic for every demo.
- Correction before any execution: the first draft of the extraction docstring added an unapproved "rise >= 5 cm in 80 steps" condition
  (it would have selected demo_4). It was removed before the T8 config was frozen; the column is kept as information only.
- Objects: moka_pot_2 (first, close step 111) and moka_pot_1 (second, step 349). Target: site `flat_stove_1_cook_region`
  (half-size 0.075 x 0.075 m, stove already on at init). Reference pose and target constants: `TaskSpec 8` in `scripts/mt/s7_task_config.py`
  and `demo_selection.json`.

## Round 1 result: 0 / 4 initial states succeed
| init | status | pot_first (moka_pot_2) grasp/lift/transport/place | pot_second grasp/lift/transport/place | task_ok | v1.1 clean | steps / wall |
|---|---|---|---|---|---|---|
| 0 | stage_timeout:pot_first.pick_1.descend | no/no/no/no | no/no/no/no | no | no (3 unexcused) | 671 / 29 s |
| 1 | same | no/no/no/no | no/no/no/no | no | no (3) | 666 / 27 s |
| 2 | same | no/no/no/no | no/no/no/no | no | no (3) | 662 / 27 s |
| 3 | same | no/no/no/no | no/no/no/no | no | no (3) | 670 / 26 s |
Max pot rise: pot_first 0.4 mm, pot_second 8-11 mm. `grasp_reached_band` is true for both pots (a weak criterion: the 1-100 N band is met
by a 6 N one-finger contact); the stability test fails (right finger force 0). Not counted as success.

## First deviation (identical in all four inits)
1. Planner picks the single reference candidate (yaw -89.2 deg, clearance +2.9 mm; the planner's hand envelope says feasible).
2. In `pot_first.pick_1.descend` (control steps ~59-62 on) the left finger lands on the pot body wall (pot geom g5, 57 N peak) and the
   hand stops 3.2 cm above and 1.4 cm beside the planned grasp pose for the remaining ~250 steps; the stage times out (init 0 diagnostic:
   `round1/diag_init0.json`). The pot moves < 2 mm.
3. Cause visible in the demo: the demo hand is TILTED at both grasps (24.1 and 30.3 deg from vertical; hand body also touches the pot
   handle g12, fingers on opposite body walls g5/g8). The executor's grasp orientation is the yaw-aligned vertical quaternion only; the
   reference converts the demo hand POSITION but cannot carry the tilt. A vertical hand at the demo position lands on the wall.

## Decision (round 1) and conclusion as reworded in Step 23
Round 1 stopped at "gap" and described the cause as a missing tilted-hand primitive.  That wording is withdrawn.
Current wording: the official demonstration grasps with a tilted hand; the adapter only passed reference position and yaw, so the
tilt component was lost.  With that simplified reference, all four dev inits were blocked by contact during the descent.
Not claimed: that no vertical grasp exists; that keeping the demo tilt will succeed; that a new primitive is needed.  A demo success does
not replace the contact audit (the demo hand body rests on the pot handle, which the contact policy may still reject).
Not done in round 1: no vertical-grasp re-search, no per-init change.  Follow-up: `reports/step23_shared_fixes/REPORT.md`.

## Cost
`onboarding_cost.json`. Shared code: `synthesis_v2` unchanged. `scripts/mt/s7_task_config.py` +24 lines (new optional field
`object_target_params` + loop in `make_program` + TaskSpec 8); programs of tasks 0, 1, 4, 5, 7 are identical before/after; 585 tests pass.
What T4/T5 did give: two-object per-target config, site targets, demo-reference grasp, plan, contact audit, v1.1 policy and the
pot/target pipeline ran unchanged to the first grasp. What was missing: a tilted-hand grasp.
