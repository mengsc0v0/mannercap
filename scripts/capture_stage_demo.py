#!/usr/bin/env python3
"""Capture per-stage (approach/descend/grasp/lift/transport/final/retreat)
demo video clips for the CVPR method figure.

Fixed demo case (chosen via a mock-LLM seed sweep, see repo memory):
    task=stack, tag=phased_5stage, seed=1
This trial exhibits exactly one genuine stage-level repair (the "final"
placement phase is initially too fast for the phased manner spec, gets
diagnosed + rolled back + retried slower, then succeeds) and the whole
episode reaches outcome=both_pass. It is a self-contained re-run of the
exact same stage loop as run_stages_with_rollback() in
run_manner_ablation.py, except it additionally records main/wrist video
frame-index boundaries around each stage call so the single recorded
video can be losslessly sliced into per-stage clips (only the FINAL
successful pass of a retried stage is kept, since frame boundaries are
overwritten on the retry).

Usage:
    cd <STAGECAP_ROOT>
    MUJOCO_GL=egl /workdata/cap-x/.venv/bin/python3 ../scripts/capture_stage_demo.py
"""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio

BASE = os.path.dirname(os.path.abspath(__file__))
NEW_ROOT = os.path.abspath(os.path.join(BASE, ".."))
STAGECAP_ROOT = os.path.join(NEW_ROOT, "stagecap_robosuite_libero_repro_20260714")
sys.path.insert(0, NEW_ROOT)
sys.path.insert(0, STAGECAP_ROOT)

import run_manner_ablation as rma  # noqa: E402
from capx.envs.manner_validator import MannerValidator, StageStatus  # noqa: E402
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi  # noqa: E402

TASK = "stack"
TAG = "phased_5stage"
SEED = 1
OUT_DIR = os.path.join(NEW_ROOT, "repro_results", "robosuite", "stage_demo")
FPS = 20


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    entry = rma.TASK_REGISTRY[TASK]
    sim_cls = rma.load_sim_cls(entry["sim_cls_path"])
    env = sim_cls(privileged=True, enable_render=True,
                  max_steps=entry["max_steps"], seed=SEED)
    api = FrankaControlPrivilegedApi(env)
    validator = MannerValidator()

    spec = rma.make_mock_spec(0, TAG)
    params = rma.spec_to_exec_params(spec, task=TASK, tag=TAG)
    print(f"[spec] label={spec.label} phase_velocity={spec.phase_velocity}")

    env.reset(seed=SEED)
    env.enable_video_capture(True, clear=True, wrist_camera=True)
    env.begin_manner_recording(clear=True)

    stages = rma.STAGE_BUILDERS[TASK](env, api)
    phase_list = [p for p, _ in stages]
    entry_ckpts: dict = {}
    stage_frame_ranges: dict = {}
    cur_params = rma._copy_params(params)
    st: dict = {}
    repairs_used = 0
    max_repairs = 2
    i = 0
    while i < len(stages):
        phase, fn = stages[i]
        entry_ckpts[phase] = env.capture_checkpoint()
        main_start = env.get_video_frame_count()
        wrist_start = len(env.get_wrist_video_frames())
        tele_start = len(env._manner_telemetry)

        fn(cur_params, st)

        main_end = env.get_video_frame_count()
        wrist_end = len(env.get_wrist_video_frames())
        seg = env._manner_telemetry[tele_start:]
        manner_ok = rma._validate_stage_manner(seg, phase, spec, validator, StageStatus)
        task_ok = rma._stage_task_predicate(env, api, TASK, phase, st)
        print(f"  stage={phase:10s} manner_ok={manner_ok!s:5s} task_ok={task_ok!s:5s} "
              f"frames main[{main_start}:{main_end}] wrist[{wrist_start}:{wrist_end}]")

        if manner_ok and task_ok:
            stage_frame_ranges[phase] = (main_start, main_end, wrist_start, wrist_end)
            i += 1
            continue

        cause = "manner" if not manner_ok else "task"
        plan = None if repairs_used >= max_repairs else \
            rma._stage_repair_plan(TASK, phase, cause, cur_params, spec)
        if plan is None:
            stage_frame_ranges[phase] = (main_start, main_end, wrist_start, wrist_end)
            i += 1
            continue

        repairs_used += 1
        print(f"    -> repair: {plan['name']}  rollback_to={plan['rollback_to']}")
        target = plan["rollback_to"]
        env.restore_checkpoint(entry_ckpts[target])
        cur_params = plan["params"]
        i = phase_list.index(target)

    env.end_manner_recording()
    exec_result = rma._finalize_staged_result(TASK, env, api, st)
    print(f"[result] task_completed={exec_result['task_completed']} "
          f"reward={exec_result['reward']:.2f} z_gap={exec_result.get('z_gap')} "
          f"repairs_used={repairs_used}")

    main_frames = env.get_video_frames()
    wrist_frames = env.get_wrist_video_frames()
    print(f"[frames] main total={len(main_frames)} wrist total={len(wrist_frames)}")

    for phase in phase_list:
        if phase not in stage_frame_ranges:
            print(f"  [skip] {phase}: no successful pass recorded")
            continue
        ms, me, ws, we = stage_frame_ranges[phase]
        main_clip = main_frames[ms:me]
        wrist_clip = wrist_frames[ws:we]
        if main_clip:
            fpath = os.path.join(OUT_DIR, f"{TASK}_{TAG}_seed{SEED}_{phase}_main.mp4")
            imageio.mimsave(fpath, main_clip, fps=FPS, macro_block_size=None)
            print(f"  [saved] {fpath}  ({len(main_clip)} frames)")
        if wrist_clip:
            fpath = os.path.join(OUT_DIR, f"{TASK}_{TAG}_seed{SEED}_{phase}_wrist.mp4")
            imageio.mimsave(fpath, wrist_clip, fps=FPS, macro_block_size=None)
            print(f"  [saved] {fpath}  ({len(wrist_clip)} frames)")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
