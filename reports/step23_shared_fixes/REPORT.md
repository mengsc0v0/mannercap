# Step 23: shared fixes, full-pose grasp reference, T8 first-grasp check

## A. Shared fixes (committed e3c196e; no three-arm rerun, old results unchanged)
1. Snapshot restore with no force target. Cause: `restore_state` only acted `if st.get("force_target")`, so a snapshot without a
   target kept the target of the state it was restored from. Fix: `NativeSession._restore_force_target` (key absent: untouched;
   truthy: set; empty: clear / re-resolve the hook-time default binding, wipe excluded). Tests `tests/test_restore_force_target.py`
   (target -> none; object A -> object B; 4 tests). Scope audited earlier (step 21): 20/32 arrays differ, 1000 rows, only in
   `pick_1.approach/descend` before the first grasp, 0 rows in hold windows, v0/v1/`task_ok` identical for 48/48.
   Old results are NOT upgraded to "all records identical"; the defect is fixed for future runs only.
2. Required stage timeout. Confirmed: after an approach/descend timeout the episode kept closing and moving (671 steps, status only
   at the end) and a single finger (left 5.98 N, right 0) satisfied the force band. Opt-in flags (defaults off, plan hashes unchanged):
   `stop_on_stage_timeout` (approach, descend, lift, transport), `require_bilateral_grasp` (both fingers must carry load; a single
   finger in the wide band does not establish a grasp). Tests `tests/test_stop_on_stage_timeout.py` (3). T5 good grasps pass the
   bilateral check (~9 N per finger, 4 runs). T8 ran with the guard off: it is only a base-execution test without guard.

## B. Full-pose grasp reference (bounded shared extension, no task branch)
Change points (yaw-only before): demo extraction stored only `hand_yaw_rel_deg`; `reference_to_candidate` rebuilt the quaternion from
yaw; `plan_approach` used `yaw_quat(...)`; the executor default quaternion is the yaw-aligned vertical one. No stage overrides the
chosen quaternion (one `quat` is used for approach, descend, lift, transport, place, retreat; `target_yaw_*` re-aim is not used by T8).
Planner: `HandEnvelope` works with any rotation; traverse checks only the endpoint rotations; feasibility needs clearance > 0 along
traverse/descent and for the open hand at the grasp pose.
Convention: `hand_quat_obj_xyzw` = hand orientation in the object yaw frame (xyzw); world = Rz(object yaw) * reference; same frame as
`rel_obj`. Optional: references without it follow the old path (`tests/test_full_pose_reference.py`, 3 tests; old refs identical).
Validation without simulation episodes: round trip reference -> candidate -> demo pot frame: position error <= 5e-5 m, rotation error
<= 8e-5 deg; four dev inits keep the 24.05 / 30.28 deg tilt, hand position identical to the yaw-only one (`t8_full_pose_check.json`).

## C. First grasp, fixed demo_0 reference, T8 init 0 (stop rule applied)
Result: blocked BEFORE any motion by the planner pre-check (`t8_first_grasp_init0_fullpose.json`): descent min clearance -0.25 mm and
open hand at the grasp pose -2.17 mm against moka_pot_2 (the target itself). Not the old location (descent contact), a new one.
Attribution (`demo_close_clearance.json`, read-only replay of demo_0): when the demo starts to close, its own open hand already touches
the pot: moka_pot_2 palm vs handle g12 -0.72 mm, vs g11 -0.21 mm, finger1 vs g5 -0.06 mm; moka_pot_1 palm vs g12 -1.05 mm, vs g13
-0.54 mm. The strict "clearance > 0" criterion therefore rejects the official grasp pose by construction; the transform is exact, so the
veto is not a conversion artefact. Per the stop rule: no tilt search, no pre-check bypass, no contact-policy change, four inits NOT run.
Not proven: whether the tilted descent would then be free of contact; whether the contact policy would excuse palm-on-handle contact
(it probably would not); whether another grasp could pass. Decision needed: see final report.

## D. Cost (adds to round 1; earlier failures still count)
Shared code: `approach_planner.py` +5 lines (optional full quaternion per candidate); `program_executor.py` two opt-in plan flags,
full-pose branch in `reference_to_candidate`; `native_osc_env.py` / `libero_backend.py` restore fix; `s7_task_config.py` +quaternion
fields in SPECS[8]. Tests +10 (595 pass). New scripts `s23_*`. Simulation runs: 1 init-0 stop check, 1 bilateral-only check, 4 T5
bilateral runs, 1 first-grasp attempt (planner veto, 0 steps), 1 offline demo replay (2 closes).
