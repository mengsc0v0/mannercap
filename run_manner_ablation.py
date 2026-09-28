"""
run_manner_ablation.py — Manner-conditioned trajectory ablation study
=====================================================================

回答的问题
----------
1. L1 任务成功：加了 manner 约束（听人话）后，任务能否完成？
2. L2 语气忠实：轨迹是否真的按人话执行？（MannerValidator SATISFIED）
3. L3 合理性：manner 约束是否让轨迹更合理？—— 与 baseline（无约束）对比
   峰值抓取力、速度、jerk、路径长度。

同时展示 "listening ≠ blindly obeying"：
  - Strategy A: 当 gentle spec 会导致抓不住时，clip 到最小可用夹持力并记录
  - Strategy B: 当 L2 满足但 L1 失败时，归类为 manner_infeasible（诚实拒绝）

任务
----
支持 --tasks lift stack （spill_wipe 后续扩展）。
Baseline 通过传入 MannerSpec()（无约束）复用同一执行器，保证公平对照。

用法
----
Smoke（mock LLM + 单 seed）:

    /workdata/cap-x/.venv/bin/python3 ../run_manner_ablation.py \
        --tasks lift stack --seeds 1 --mock-llm

Full（真实 LLM + 5 seed）:

    export AIHUBMIX_API_KEY=sk-...
    /workdata/cap-x/.venv/bin/python3 ../run_manner_ablation.py \
        --tasks lift stack --seeds 5 \
        --output repro_results/robosuite/manner_ablation.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Callable

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "stagecap_robosuite_libero_repro_20260714")
sys.path.insert(0, BASE)

import numpy as np

from synthesis_v2.experience_library import ExperienceLibrary, GLOBAL_PHASE, flatten_trial_params

# ──────────────────────────────────────────────────────────────────────────────
# 经验库校准层（Phase 3）：(task, phase, dimension, level) -> 历史 both_pass 分位数
# ──────────────────────────────────────────────────────────────────────────────
# 惰性加载 + 按绝对路径缓存，避免同一批次内每个 trial 都重新读一次 jsonl。
_EXPERIENCE_LIBRARY_CACHE: dict[str, ExperienceLibrary] = {}


def _get_experience_library(path: str) -> ExperienceLibrary:
    key = os.path.abspath(path)
    lib = _EXPERIENCE_LIBRARY_CACHE.get(key)
    if lib is None:
        lib = ExperienceLibrary()
        lib.load(path)
        _EXPERIENCE_LIBRARY_CACHE[key] = lib
    return lib


def _calibrate_value(lib, task, phase, dimension, level, fallback, min_samples=3):
    """None-safe wrapper around ExperienceLibrary.calibrate(): when lib is
    None (calibration disabled) or task/level are unknown, always returns
    (fallback, False) unchanged -- this is what preserves the exact current
    preset behaviour when --use-experience-calibration is not passed."""
    if lib is None or not task or not level:
        return fallback, False
    return lib.calibrate(task, phase, dimension, level, fallback,
                          min_samples=min_samples)


def _calibrate_repair_ratio(lib, task, phase, cause, dimension, fallback_ratio, min_samples=3):
    """None-safe wrapper around ExperienceLibrary.calibrate_repair_ratio():
    when lib is None or task is unknown, always returns (fallback_ratio,
    False) unchanged -- preserves today's hardcoded MANNER_REPAIR_LIBRARY
    adjustment ratios when --use-experience-calibration is not passed or the
    (task, phase, cause, dimension) bucket is cold."""
    if lib is None or not task:
        return fallback_ratio, False
    return lib.calibrate_repair_ratio(task, phase, cause, dimension, fallback_ratio,
                                       min_samples=min_samples)


def _write_back_experience(lib, path, task, tag, seed, run_stamp, params, outcome):
    """Phase 5 self-evolving write-back: flatten a both_pass trial's final
    (post-repair-if-any) params into per-(phase, dimension) experience
    samples. Appended both to the in-memory ``lib`` (so later trials in the
    same batch process can immediately retrieve them via ``calibrate()``)
    and persisted to ``path`` (so future/other processes see them too).
    None-safe: no-op when the experience library is not enabled."""
    if lib is None or not task:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for dimension, phase_values in flatten_trial_params(params).items():
        for phase, value in phase_values.items():
            lib.add_experience(
                task, phase, dimension, tag, value, outcome,
                source_run_stamp=run_stamp, seed=seed, ts=ts)
            lib.append_record(path, lib.experiences[-1])


def _write_back_repair(lib, path, task, run_stamp, cause, before_params, after_params, outcome_after):
    """Phase 5 self-evolving write-back for a repair retry that reached
    both_pass: for every (phase, dimension) whose value actually changed
    between the pre-repair and post-repair params, record a genuine
    before/after RepairRecord (real in-process values, unlike the historical
    population script which cannot reconstruct them). None-safe no-op when
    the experience library is not enabled."""
    if lib is None or not task:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    before_flat = flatten_trial_params(before_params)
    after_flat = flatten_trial_params(after_params)
    for dimension, before_phases in before_flat.items():
        after_phases = after_flat.get(dimension, {})
        for phase, before_value in before_phases.items():
            after_value = after_phases.get(phase)
            if after_value is None or after_value == before_value:
                continue
            lib.add_repair(
                task, phase, cause, dimension, before_value, after_value,
                outcome_after, source_run_stamp=run_stamp, ts=ts)
            lib.append_record(path, lib.repairs[-1])


# ──────────────────────────────────────────────────────────────────────────────
# 测试用例（与已归档的 test_llm_manner_e2e.py 一致，现由本脚本 --mock-llm 模式覆盖）
# ──────────────────────────────────────────────────────────────────────────────
TEST_CASES: list[tuple[str, str]] = [
    ("像捧着装满水的玻璃杯，轻轻把方块放到目标位置上", "gentle"),
    ("小心翼翼，像在排除炸弹一样慢慢靠近",              "ultra_careful"),
    ("像哄婴儿睡觉一样，极其轻柔缓慢地拿起方块",        "ultra_gentle"),
    ("正常操作就行，随便",                             "normal"),
    ("迅速抓起来，越快越好",                           "fast"),
    ("用力把它压进去，确保固定住不会松动",              "firm"),
    ("先快速靠近，到最后5厘米时慢慢轻触",              "fast_then_slow"),
    ("Place the block gently onto the platform",        "gentle_en"),
    ("Grab it as fast as possible",                     "fast_en"),
    ("先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开", "phased_5stage"),
]

BASELINE_TAG = "baseline"


# ──────────────────────────────────────────────────────────────────────────────
# MannerSpec → 执行参数（Strategy A: clip 硬性物理下限）
# ──────────────────────────────────────────────────────────────────────────────
GRIP_FLOOR = 0.60  # 实测诊断：force<0.5 时夹爪几乎不闭合，需>=0.62 才能稳定抬起 5cm cube
V_MIN_DEFAULT = 0.02
KP_FLOOR = 250.0


def _derive_velocity(max_bound, min_bound, default: float) -> float:
    if max_bound is not None and min_bound is not None:
        return (max_bound * 0.85 + min_bound * 1.05) / 2
    elif max_bound is not None:
        return max_bound * 0.85
    elif min_bound is not None:
        return min_bound * 1.10
    return default


def spec_to_exec_params(spec, task=None, tag=None, experience_lib=None, min_samples=3) -> dict:
    clipped: list[str] = []
    clipped_dims: list[str] = []

    v_default = 0.12
    v_max = _derive_velocity(spec.max_joint_vel, spec.min_joint_vel, v_default)
    if v_max < V_MIN_DEFAULT:
        clipped.append(f"v_max({v_max:.3f}<{V_MIN_DEFAULT})")
        clipped_dims.append("vel")
        v_max = V_MIN_DEFAULT
    a_max = max(0.01, v_max * 0.7)

    _phase_defaults = {
        "approach":  (1.0, 1.0),
        "descend":   (0.6, 0.5),
        "lift":      (1.0, 1.0),
        "transport": (0.5, 0.4),
        "final":     (0.5, 0.4),
        "retreat":   (1.0, 1.0),
    }
    phase_overrides = getattr(spec, "phase_velocity", None) or {}
    phase_v_max: dict[str, float] = {}
    phase_a_max: dict[str, float] = {}
    calibrated: list[str] = []
    for name, (ratio_v, ratio_a) in _phase_defaults.items():
        ov = phase_overrides.get(name)
        if ov is not None and (ov.max_joint_vel is not None or ov.min_joint_vel is not None):
            v = _derive_velocity(ov.max_joint_vel, ov.min_joint_vel, v_max)
            v, used = _calibrate_value(experience_lib, task, name, "v_max", tag, v, min_samples)
            if used:
                calibrated.append(f"{name}.v_max")
            if v < V_MIN_DEFAULT:
                clipped.append(f"{name}.v_max({v:.3f}<{V_MIN_DEFAULT})")
                clipped_dims.append("vel")
                v = V_MIN_DEFAULT
            phase_v_max[name] = v
            a = max(0.01, v * 0.7)
            a, used = _calibrate_value(experience_lib, task, name, "a_max", tag, a, min_samples)
            if used:
                calibrated.append(f"{name}.a_max")
            phase_a_max[name] = a
        else:
            v = max(0.02, v_max * ratio_v)
            v, used = _calibrate_value(experience_lib, task, name, "v_max", tag, v, min_samples)
            if used:
                calibrated.append(f"{name}.v_max")
            phase_v_max[name] = v
            a = max(0.01, a_max * ratio_a)
            a, used = _calibrate_value(experience_lib, task, name, "a_max", tag, a, min_samples)
            if used:
                calibrated.append(f"{name}.a_max")
            phase_a_max[name] = a

    jerk_limit = spec.max_jerk

    kp_default = 1000.0
    if spec.kp_max is not None and spec.kp_min is not None:
        kp_pos = (spec.kp_max * 0.85 + spec.kp_min * 1.05) / 2
    elif spec.kp_max is not None:
        kp_pos = spec.kp_max * 0.85
    elif spec.kp_min is not None:
        kp_pos = spec.kp_min * 1.10
    else:
        kp_pos = kp_default
    kp_pos, used = _calibrate_value(experience_lib, task, GLOBAL_PHASE, "kp_pos", tag, kp_pos, min_samples)
    if used:
        calibrated.append("kp_pos")
    if kp_pos < KP_FLOOR:
        clipped.append(f"kp_pos({kp_pos:.0f}<{KP_FLOOR:.0f})")
        clipped_dims.append("kp")
        kp_pos = KP_FLOOR

    grip_default = 0.88
    if spec.grip_fraction_max is not None and spec.grip_fraction_min is not None:
        grip = (spec.grip_fraction_max * 0.9 + spec.grip_fraction_min * 1.05) / 2
    elif spec.grip_fraction_max is not None:
        grip = spec.grip_fraction_max * 0.90
    elif spec.grip_fraction_min is not None:
        grip = min(1.0, spec.grip_fraction_min * 1.10)
    else:
        grip = grip_default
    grip, used = _calibrate_value(experience_lib, task, GLOBAL_PHASE, "grip", tag, grip, min_samples)
    if used:
        calibrated.append("grip")
    if grip < GRIP_FLOOR:
        clipped.append(f"grip({grip:.2f}<{GRIP_FLOOR:.2f})")
        clipped_dims.append("grip")
        grip = GRIP_FLOOR

    return {
        "v_max":         float(v_max),
        "a_max":         float(a_max),
        "jerk_limit":    float(jerk_limit) if jerk_limit is not None else None,
        "kp_pos":        float(kp_pos),
        "grip":          float(grip),
        "phase_v_max":   {k: float(v) for k, v in phase_v_max.items()},
        "phase_a_max":   {k: float(v) for k, v in phase_a_max.items()},
        "_clipped":      clipped,
        "_clipped_dims": clipped_dims,
        "_calibrated":   calibrated,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 任务执行器
# ──────────────────────────────────────────────────────────────────────────────
def _pump(env, n: int) -> None:
    for _ in range(n):
        env._step_once()


def _grasp_close(env, api, force: float, steps: int = 50, n_tries: int = 3) -> None:
    """低力目标(<0.70)时分多次收紧-回稳，而非一次到位（灵活抓取策略）。

    低夹持力单次闭合常因接触面小、切向摩擦不足而打滑。分多次收紧-回稳，
    在不抬高峰值夹持力（不违反 Manner C 上界，L2 仍合规）的前提下，让指尖
    多次重新对齐接触面，提升抓取成功率（L1）。`n_tries` 可由经验库
    （见 MANNER_REPAIR_LIBRARY 的 "grip_slip" 条目）在重试时调高。
    """
    if force < 0.70:
        n_tries = max(1, n_tries)
        per_steps = max(15, steps // n_tries)
        for _ in range(n_tries):
            api.close_gripper_partial(force=force, steps=per_steps)
            _pump(env, 4)
    else:
        api.close_gripper_partial(force=force, steps=steps)


def run_cube_lift(env, api, params: dict) -> dict:
    cube_pos, _, _ = api.get_object_pose("red cube")
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])
    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})

    approach_pos = cube_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, grasp_quat,
        v_max=pv.get("approach", params["v_max"]), a_max=pa.get("approach", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="approach",
    )
    _pump(env, 8)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    grasp_pos = cube_pos.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, grasp_quat,
        v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
        a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
        jerk_limit=params["jerk_limit"], phase="descend",
    )
    _pump(env, 8)

    _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.18
    api.goto_pose_with_profile(
        lift_pos, grasp_quat,
        v_max=pv.get("lift", params["v_max"]), a_max=pa.get("lift", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="lift",
    )
    _pump(env, 10)

    cube_pos_after, _, _ = api.get_object_pose("red cube")
    return {
        "reward": float(env.compute_reward()),
        "cube_delta_z": float(cube_pos_after[2] - cube_pos[2]),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }


def run_cube_stack(env, api, params: dict) -> dict:
    """Red cube → placed on Green cube. Success = _check_success()."""
    red_pos,   _, _ = api.get_object_pose("red cube")
    green_pos, _, _ = api.get_object_pose("green cube")
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})

    approach_pos = red_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, grasp_quat,
        v_max=pv.get("approach", params["v_max"]), a_max=pa.get("approach", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="approach",
    )
    _pump(env, 6)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    grasp_pos = red_pos.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, grasp_quat,
        v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
        a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
        jerk_limit=params["jerk_limit"], phase="descend",
    )
    _pump(env, 6)

    _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    api.goto_pose_with_profile(
        lift_pos, grasp_quat,
        v_max=pv.get("lift", params["v_max"]), a_max=pa.get("lift", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="lift",
    )
    _pump(env, 6)

    over_green = green_pos.copy(); over_green[2] = lift_pos[2]
    api.goto_pose_with_profile(
        over_green, grasp_quat,
        v_max=pv.get("transport", params["v_max"]), a_max=pa.get("transport", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="transport",
    )
    _pump(env, 6)

    # green top = green_z + 0.025; red half-side = 0.025; small gap 0.008
    stack_pos = green_pos.copy(); stack_pos[2] = green_pos[2] + 0.058
    api.goto_pose_with_profile(
        stack_pos, grasp_quat,
        v_max=pv.get("final", max(0.02, params["v_max"] * 0.5)),
        a_max=pa.get("final", max(0.01, params["a_max"] * 0.4)),
        jerk_limit=params["jerk_limit"], phase="final",
    )
    _pump(env, 14)

    api.open_gripper()
    _pump(env, 15)

    retreat_pos = stack_pos.copy(); retreat_pos[2] += 0.20
    api.goto_pose_with_profile(
        retreat_pos, grasp_quat,
        v_max=pv.get("retreat", params["v_max"]), a_max=pa.get("retreat", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="retreat",
    )
    _pump(env, 20)

    red_after,   _, _ = api.get_object_pose("red cube")
    green_after, _, _ = api.get_object_pose("green cube")
    return {
        "reward": float(env.compute_reward()),
        "red_delta_z": float(red_after[2] - red_pos[2]),
        "z_gap": float(red_after[2] - green_after[2]),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }


def run_pour(env, api, params: dict) -> dict:
    """给领导倒水（pour）一次执行：抓起水杯，稳稳运送到目标点，轻放不洒。

    动作序列 (对应 5 阶段 manner):
        1. approach above cup   (快速接近)
        2. set stiffness (Manner B)
        3. descend to cup (re-fetch pose after settle) + grasp (轻轻拿起)
        4. lift up + transport to target (稳稳运送)
        5. lower to table (缓缓放下)
        6. open gripper (release)
        7. retreat (快速离开)

    关键修复（已实测验证）：descend 阶段的抓取目标必须在 approach 完成、
    杯子经过短暂沉降之后**重新查询** cup 位姿，不能复用 approach 之前缓存的
    旧位姿——即便只有几毫米的沉降误差，也足以让指尖对不准杯壁导致抓空
    （表现为杯子在整个 episode 里位置完全冻结不动）。
    """
    cup_pos, _, _ = api.get_object_pose("water cup")
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])
    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})

    # 1. approach above cup
    approach_pos = cup_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, grasp_quat,
        v_max=pv.get("approach", params["v_max"]), a_max=pa.get("approach", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="approach",
    )
    _pump(env, 8)

    # 2. stiffness
    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    # 3. descend（重新获取杯子位姿，见函数 docstring 的关键修复说明）
    cup_pos2, _, _ = api.get_object_pose("water cup")
    grasp_pos = cup_pos2.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, grasp_quat,
        v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
        a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
        jerk_limit=params["jerk_limit"], phase="descend",
    )
    _pump(env, 8)

    # 4. grasp（低力目标走多次收紧，见 _grasp_close）
    _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    # 5. lift
    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    api.goto_pose_with_profile(
        lift_pos, grasp_quat,
        v_max=pv.get("lift", params["v_max"]), a_max=pa.get("lift", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="lift",
    )
    _pump(env, 10)

    # 6. transport to target spot（稳稳运送，速度低于 approach，避免洒水）
    target_pos, _, _ = api.get_object_pose("target spot")
    transport_pos = target_pos.copy(); transport_pos[2] = lift_pos[2]
    api.goto_pose_with_profile(
        transport_pos, grasp_quat,
        v_max=pv.get("transport", max(0.02, params["v_max"] * 0.5)),
        a_max=pa.get("transport", max(0.01, params["a_max"] * 0.4)),
        jerk_limit=params["jerk_limit"], phase="transport",
    )
    _pump(env, 15)

    # 7. putdown（缓缓放下）
    putdown_pos = transport_pos.copy(); putdown_pos[2] -= 0.13
    api.goto_pose_with_profile(
        putdown_pos, grasp_quat,
        v_max=pv.get("final", max(0.02, params["v_max"] * 0.4)),
        a_max=pa.get("final", max(0.01, params["a_max"] * 0.3)),
        jerk_limit=params["jerk_limit"], phase="final",
    )
    _pump(env, 10)

    # 8. release
    api.open_gripper()
    _pump(env, 10)

    # 9. retreat（快速离开）
    retreat_pos = putdown_pos.copy(); retreat_pos[2] += 0.20
    api.goto_pose_with_profile(
        retreat_pos, grasp_quat,
        v_max=pv.get("retreat", params["v_max"]), a_max=pa.get("retreat", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="retreat",
    )
    _pump(env, 10)

    cup_after, _, _ = api.get_object_pose("water cup")
    return {
        "reward": float(env.compute_reward()),
        "cup_delta_z": float(cup_after[2] - cup_pos[2]),
        "spill_ratio": float(env.spill_ratio()),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 剩余任务接线：spill_wipe / nut_assembly / cube_restack
# (two_arm_handover 暂缓，见下方 TASK_REGISTRY 注释)
# ──────────────────────────────────────────────────────────────────────────────
_WIPE_DOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])


def _wipe_tool_center_goto(env):
    """Wipe 专用工具中心点 IK：擦拭工具固连在夹爪上，不是抓取姿态，因此不能走
    FrankaControlPrivilegedApi.goto_pose_with_profile（该 API 内含抓取用的 TCP
    偏移 + pyroki 逆解，是为抓取姿态标定的）。改为对夹爪 4 个 corner 几何体的
    中心点直接做 least-squares 关节空间逆解，再复用 env.move_to_joints_profiled
    执行（manner 速度/加速度曲线依旧生效）。已在独立探针脚本中验证可行
    （/tmp/wipe_manner_probe.py），此处原样接入正式管线。
    """
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    rs = env.robosuite_env
    sim = rs.sim
    model = sim.model._model
    scratch = mujoco.MjData(model)
    joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")
        for i in range(1, 8)
    ]
    qadr = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    eef = env.gripper_link_idx
    base = env.base_link_idx
    arm = rs.robots[0].arms[0]
    corner_ids = np.asarray(
        [sim.model.geom_name2id(n) for n in rs.robots[0].gripper[arm].important_geoms["corners"]],
        dtype=int,
    )

    def goto(pos_base, quat_wxyz, v_max, a_max, phase):
        q_before = sim.data.qpos.copy()
        scratch.qpos[:] = q_before
        scratch.qvel[:] = sim.data.qvel.copy()
        brot = Rotation.from_matrix(sim.data.xmat[base].reshape(3, 3))
        tgt = sim.data.xpos[base] + brot.apply(np.asarray(pos_base, float).reshape(3))
        q = np.asarray(quat_wxyz, float).reshape(4)
        trot = brot * Rotation.from_quat(q[[1, 2, 3, 0]])
        q0 = np.clip(q_before[qadr], lower + 1e-6, upper - 1e-6)

        def resid(qq):
            scratch.qpos[qadr] = qq
            mujoco.mj_forward(model, scratch)
            c = scratch.geom_xpos[corner_ids].mean(axis=0)
            re = (trot * Rotation.from_matrix(scratch.xmat[eef].reshape(3, 3)).inv()).as_rotvec()
            return np.r_[c - tgt, 0.2 * re, 1e-4 * (qq - q0)]

        sol = least_squares(
            resid, q0, bounds=(lower + 1e-6, upper - 1e-6),
            max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        env.move_to_joints_profiled(sol.x, v_max=v_max, a_max=a_max, phase=phase)

    return goto


def _wipe_marker_positions(env, remaining_only=True):
    """剩余 (未擦除) spill marker 在机器人基座系下的 xyz。"""
    rs = env.robosuite_env
    sim = rs.sim
    markers = list(rs.model.mujoco_arena.markers)
    if remaining_only:
        wiped = {id(m) for m in rs.wiped_markers}
        markers = [m for m in markers if id(m) not in wiped]
    if not markers:
        return np.zeros((0, 3))
    world = np.stack(
        [sim.data.body_xpos[sim.model.body_name2id(m.root_body)].copy() for m in markers]
    )
    b = np.asarray(env.base_link_wxyz_xyz, float)
    qw, qx, qy, qz = b[:4]
    t = b[4:7]
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return (world - t) @ R


def _wipe_nn_path(pts, start):
    rem = list(range(len(pts)))
    order = []
    cur = np.asarray(start[:2])
    while rem:
        d = [np.linalg.norm(pts[i][:2] - cur) for i in rem]
        k = rem[int(np.argmin(d))]
        order.append(k)
        cur = pts[k][:2]
        rem.remove(k)
    return pts[order]


def run_wipe(env, api, params: dict) -> dict:
    """擦拭任务：把桌上所有 brown spill marker 擦除。

    与 lift/stack/pour 不同，本任务不抓取物体——机械臂末端固连擦拭工具，用
    工具中心点 (4 个 corner 几何体均值) 直接逆解 (见 _wipe_tool_center_goto)。

    manner 映射：
        approach/descend 速度 <- spec 的 approach/descend 阶段速度
        sweep (擦拭平移) 速度 <- spec 的 transport 阶段速度
        下压深度 (penetration) <- 由 kp_pos (刚度/力度 manner 轴) 映射，裁剪在
            实测安全区间 [0.003, 0.012] 米内 (过深会穿透桌面法向, 过浅擦不到)。
    """
    goto = _wipe_tool_center_goto(env)
    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})
    v_max = params["v_max"]

    approach_v = pv.get("approach", v_max)
    descend_v = pv.get("descend", max(0.02, v_max * 0.6))
    sweep_v = pv.get("transport", max(0.02, v_max * 0.6))
    sweep_a = pa.get("transport", max(0.01, params["a_max"] * 0.5))
    kp_ref = 1000.0
    penetration = float(np.clip(0.006 * (params["kp_pos"] / kp_ref), 0.003, 0.012))

    markers0 = _wipe_marker_positions(env)
    num_markers0 = len(markers0)
    if num_markers0 == 0:
        return {
            "reward": float(env.compute_reward()),
            "sim_steps": int(env._sim_step_count),
            "task_completed": bool(env.task_completed()),
            "markers_remaining": 0,
            "num_markers": 0,
        }

    c0 = markers0[0].copy()
    goto(c0 + np.array([0.0, 0.0, 0.08]), _WIPE_DOWN_QUAT, approach_v, max(0.01, approach_v * 0.8), "approach")
    goto(c0 - np.array([0.0, 0.0, penetration]), _WIPE_DOWN_QUAT, descend_v, max(0.01, descend_v * 0.8), "descend")
    _pump(env, 3)

    max_iters = int(params.get("wipe_iters", 8))
    try:
        for _ in range(max_iters):
            remaining = _wipe_marker_positions(env)
            if len(remaining) == 0 or env.task_completed():
                break
            path = _wipe_nn_path(remaining, remaining[0])
            for w in path:
                target = w.copy()
                target[2] -= penetration
                goto(target, _WIPE_DOWN_QUAT, sweep_v, sweep_a, "transport")
                if env.task_completed():
                    break
            if env.task_completed():
                break
    except Exception as e:
        print(f"    [wipe stop] {type(e).__name__}: {str(e)[:120]}")

    remaining_final = _wipe_marker_positions(env)
    return {
        "reward": float(env.compute_reward()),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
        "markers_remaining": int(len(remaining_final)),
        "num_markers": int(num_markers0),
    }


def run_nut_assembly(env, api, params: dict) -> dict:
    """方形螺母套插销 (nut assembly)：抓取螺母 handle，运送并套入插销。

    关键修复（已实测验证）：方形螺母中心是**空心孔洞**（套在插销上的孔），
    若像 stack/pour 那样直接抓"螺母中心"点，夹爪只会在空气中合拢，完全抓
    不住任何实体材料——表现为夹爪路径/关节角完全正确、但螺母全程位置纹丝
    不动 (delta_z=0)。必须改为抓取 `square_nut_handle` (螺母侧边实体把手)，
    且使用 `get_object_pose` 已经算好的"抓取就绪"四元数 (内部通过
    rotate_by_180 @ invert_grasp_pose 抵消了螺母初始随机朝向)，而不是固定
    俯视四元数——否则夹爪合拢方向可能与把手长轴平行而抓空。

    由于抓取参考点变成了 handle 而非螺母中心，放置阶段需要把"螺母中心置于
    插销正上方"换算成"handle 应该去哪"：在夹爪刚抓稳 (`_grasp_close` 之后)
    立即测量 handle→螺母中心 的偏移向量 `center_offset`；只要抓取姿态在
    搬运全程保持不变 (未曾重新定向)，这个偏移向量在该固定姿态下保持恒定，
    因此 handle 的搬运/下插目标位置 = peg_pos - center_offset。

    另一处必要修复：`FrankaRobosuiteNutAssembly.reset()` 会故意把手臂折叠到
    摄像头视野外的隐藏姿态 (`qpos[:7]=[0,-1.585,0,-2.645,0,1,0.785]`)，并非
    常规的"就绪"姿态。若不先归位就直接求解 approach 的 IK，warm-start 起点
    (`prev_cfg`) 来自这个非常规折叠姿态，容易让 IK 落入与其它任务不一致的
    局部解，导致同一 phase_v_max/grip 参数下不同 tag 的抓取稳定性差异很大
    （实测 normal tag 的 nut_xy_error 远大于 gentle/fast）。因此在任何操作
    前先调用一次 move_to_joints_blocking(env.home_joint_position) 归位。

    成功判据由 `robosuite_env._check_success()` 内的 `on_peg()` 决定
    (螺母 xy 与插销 xy 误差 < 3cm 且 z 低于桌面+5cm 且已松开)。
    """
    env.move_to_joints_blocking(env.home_joint_position, max_steps=200)
    nut_pos, _, _ = api.get_object_pose("square nut")
    handle_pos, handle_quat, _ = api.get_object_pose("square nut handle")
    peg_pos, _, _ = api.get_object_pose("square peg")
    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})

    approach_pos = handle_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, handle_quat,
        v_max=pv.get("approach", params["v_max"]), a_max=pa.get("approach", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="approach",
    )
    _pump(env, 6)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    # descend（重新获取 handle 位姿，见函数 docstring 的关键修复说明）
    handle_pos2, handle_quat2, _ = api.get_object_pose("square nut handle")
    grasp_pos = handle_pos2.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, handle_quat2,
        v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
        a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
        jerk_limit=params["jerk_limit"], phase="descend",
    )
    _pump(env, 6)

    # 方形螺母 handle 很薄，接触面小；共享的 `_grasp_close` 只有在
    # force<0.70 时才走"多次收紧-回稳"分支，force>=0.70 (normal/fast 常见档位)
    # 会退化成一次性 close_gripper_partial，对薄 handle 经常一次抓空/打滑
    # (实测 normal/fast 两档 nut_delta_z 恒为 0，即从未真正抓起螺母)。
    # 因此这里不复用阈值分支，始终做 3 次收紧-回稳。
    _nut_tries = 3
    _nut_per_steps = max(15, 60 // _nut_tries)
    for _ in range(_nut_tries):
        api.close_gripper_partial(force=params["grip"], steps=_nut_per_steps)
        _pump(env, 4)

    # 抓稳后立即测量 handle -> 螺母中心 偏移向量（后续姿态不变，偏移恒定）
    nut_pos_grasped, _, _ = api.get_object_pose("square nut")
    handle_pos_grasped, _, _ = api.get_object_pose("square nut handle")
    center_offset = nut_pos_grasped - handle_pos_grasped

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    api.goto_pose_with_profile(
        lift_pos, handle_quat2,
        v_max=pv.get("lift", params["v_max"]), a_max=pa.get("lift", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="lift",
    )
    _pump(env, 6)

    # 把螺母中心搬到插销正上方 -> 换算成 handle 该去的位置
    over_peg = (peg_pos - center_offset).copy()
    over_peg[2] = lift_pos[2]
    api.goto_pose_with_profile(
        over_peg, handle_quat2,
        v_max=pv.get("transport", params["v_max"]), a_max=pa.get("transport", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="transport",
    )
    _pump(env, 6)

    final_pos = (peg_pos - center_offset).copy()
    final_pos[2] = grasp_pos[2] + 0.01
    api.goto_pose_with_profile(
        final_pos, handle_quat2,
        v_max=pv.get("final", max(0.02, params["v_max"] * 0.5)),
        a_max=pa.get("final", max(0.01, params["a_max"] * 0.4)),
        jerk_limit=params["jerk_limit"], phase="final",
    )
    _pump(env, 14)

    # 闭环 xy 微调：normal/fast 档更高速度下，final 阶段轨迹跟踪残余误差会让
    # 螺母中心偏离插销中心超过成功阈值 (3cm)。松爪前用实测螺母位置反推出还
    # 差多少，做一次慢速小幅修正（不受 tag 速度影响，始终用保守的
    # v_max=0.02，避免修正本身引入新的 manner 违规）。
    nut_before_release, _, _ = api.get_object_pose("square nut")
    xy_residual = peg_pos[:2] - nut_before_release[:2]
    if float(np.linalg.norm(xy_residual)) > 0.015:
        correction_pos = final_pos.copy()
        correction_pos[:2] += xy_residual
        api.goto_pose_with_profile(
            correction_pos, handle_quat2,
            v_max=0.02, a_max=0.01,
            jerk_limit=params["jerk_limit"], phase="final_correction",
        )
        _pump(env, 10)

    api.open_gripper()
    _pump(env, 15)

    retreat_pos = final_pos.copy(); retreat_pos[2] += 0.20
    api.goto_pose_with_profile(
        retreat_pos, handle_quat2,
        v_max=pv.get("retreat", params["v_max"]), a_max=pa.get("retreat", params["a_max"]),
        jerk_limit=params["jerk_limit"], phase="retreat",
    )
    _pump(env, 20)

    nut_after, _, _ = api.get_object_pose("square nut")
    nut_xy_error = float(np.linalg.norm(nut_after[:2] - peg_pos[:2]))
    return {
        "reward": float(env.compute_reward()),
        "nut_delta_z": float(nut_after[2] - nut_pos[2]),
        "nut_xy_error": nut_xy_error,
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }


def run_cube_restack(env, api, params: dict) -> dict:
    """两方块拆位重摞 (cube restack)：开局 green 方块已摞在 red 方块上方
    (由 `StackedObjectRandomSampler` 保证)。Robosuite Stack 环境的成功判据
    (`staged_rewards` 里的 r_stack) 固定要求 **cubeA(red) 被举起且与
    cubeB(green) 接触、此时夹爪未抓持** —— 即"红摞在绿上"。初始状态恰好
    相反 (绿在上、红在下)，因此必须先把 green 拆下移到旁侧空位 (拆位)，
    腾出 red，再抓起 red 摞到 green 新位置上方 (重摞)，才能满足
    env.task_completed()。比标准 stack 多一次完整抓取-放置循环，放大
    manner (尤其速度/力度) 在两次连续抓放动作间的一致性/漂移。
    """
    red_pos, _, _ = api.get_object_pose("red cube")
    green_pos, _, _ = api.get_object_pose("green cube")
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])
    pv = params.get("phase_v_max", {})
    pa = params.get("phase_a_max", {})

    def _goto(pos, phase_name, v_ratio=1.0, a_ratio=1.0):
        api.goto_pose_with_profile(
            pos, grasp_quat,
            v_max=pv.get(phase_name, max(0.02, params["v_max"] * v_ratio)),
            a_max=pa.get(phase_name, max(0.01, params["a_max"] * a_ratio)),
            jerk_limit=params["jerk_limit"], phase=phase_name,
        )

    # ---- 1. 拆位：抓取顶部 green 方块，放到旁侧临时位置 ----
    approach_pos = green_pos.copy(); approach_pos[2] += 0.20
    _goto(approach_pos, "approach")
    _pump(env, 6)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    green_pos2, _, _ = api.get_object_pose("green cube")
    grasp_pos = green_pos2.copy(); grasp_pos[2] -= 0.005
    _goto(grasp_pos, "descend", 0.6, 0.5)
    _pump(env, 6)

    _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    _goto(lift_pos, "lift")
    _pump(env, 6)

    stage_pos = red_pos.copy(); stage_pos[0] += 0.12; stage_pos[2] = lift_pos[2]
    _goto(stage_pos, "transport")
    _pump(env, 6)

    stage_down = stage_pos.copy(); stage_down[2] = red_pos[2]
    _goto(stage_down, "final", 0.5, 0.4)
    _pump(env, 10)

    api.open_gripper()
    _pump(env, 10)

    retreat1 = stage_down.copy(); retreat1[2] += 0.18
    _goto(retreat1, "retreat")
    _pump(env, 8)

    # ---- 2. 重摞：red 已空出，抓起 red 摞到 green 新位置上方 ----
    red_pos2, _, _ = api.get_object_pose("red cube")
    approach2 = red_pos2.copy(); approach2[2] += 0.20
    _goto(approach2, "approach")
    _pump(env, 6)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    red_pos3, _, _ = api.get_object_pose("red cube")
    grasp_pos2 = red_pos3.copy(); grasp_pos2[2] -= 0.005
    _goto(grasp_pos2, "descend", 0.6, 0.5)
    _pump(env, 6)

    _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos2 = grasp_pos2.copy(); lift_pos2[2] += 0.15
    _goto(lift_pos2, "lift")
    _pump(env, 6)

    green_staged, _, _ = api.get_object_pose("green cube")
    over_green = green_staged.copy(); over_green[2] = lift_pos2[2]
    _goto(over_green, "transport")
    _pump(env, 6)

    stack_pos = green_staged.copy(); stack_pos[2] = green_staged[2] + 0.058
    _goto(stack_pos, "final", 0.5, 0.4)
    _pump(env, 14)

    api.open_gripper()
    _pump(env, 15)

    retreat2 = stack_pos.copy(); retreat2[2] += 0.20
    _goto(retreat2, "retreat")
    _pump(env, 20)

    red_after, _, _ = api.get_object_pose("red cube")
    green_after, _, _ = api.get_object_pose("green cube")
    return {
        "reward": float(env.compute_reward()),
        "red_delta_z": float(red_after[2] - red_pos[2]),
        "z_gap": float(red_after[2] - green_after[2]),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 分阶段可回滚执行引擎 (技术 #1 stage-snapshot rollback + #2 stage-level dual-channel validation)
#
# 把 run_cube_lift / run_cube_stack / run_pour 的单体动作序列拆成有序的
# [(phase, stage_fn), ...] 阶段列表。每个阶段执行后立即做「双通道」判定：
#   通道 A (manner)：该阶段 telemetry 切片是否满足该 phase 的 MannerSpec 约束
#   通道 B (task)  ：该阶段是否达成物理子目标 (抓稳 / 抬起 / 对齐堆叠 …)
# 任一通道失败 -> 回滚到某个已保存的阶段入口快照 (env.restore_checkpoint)，
# 对该阶段做局部参数修复后原地重试，避免整条 episode 重跑。
# ──────────────────────────────────────────────────────────────────────────────
_MOVE_STAGE_PHASES = frozenset(
    {"approach", "descend", "lift", "transport", "final", "retreat"}
)


def _eef_pos(env) -> "np.ndarray":
    """当前末端执行器 (夹爪 link) 世界坐标 xyz。gripper_link_wxyz_xyz = [qw qx qy qz x y z]。"""
    return np.asarray(env.gripper_link_wxyz_xyz, dtype=float)[4:7]


def build_lift_stages(env, api):
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

    def approach(params, st):
        cube_pos, _, _ = api.get_object_pose("red cube")
        st["cube_pos0"] = cube_pos.copy()
        st["cube_pos"] = cube_pos.copy()
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        ap = cube_pos.copy(); ap[2] += 0.20
        api.goto_pose_with_profile(
            ap, grasp_quat,
            v_max=pv.get("approach", params["v_max"]),
            a_max=pa.get("approach", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="approach")
        _pump(env, 8)

    def descend(params, st):
        api.set_impedance_stiffness(kp_pos=params["kp_pos"])
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        grasp_pos = st["cube_pos"].copy(); grasp_pos[2] -= 0.005
        st["grasp_pos"] = grasp_pos
        api.goto_pose_with_profile(
            grasp_pos, grasp_quat,
            v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
            a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
            jerk_limit=params["jerk_limit"], phase="descend")
        _pump(env, 8)

    def grasp(params, st):
        _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    def lift(params, st):
        api.set_impedance_stiffness(kp_pos=1000.0)
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        lift_pos = st["grasp_pos"].copy(); lift_pos[2] += 0.18
        api.goto_pose_with_profile(
            lift_pos, grasp_quat,
            v_max=pv.get("lift", params["v_max"]),
            a_max=pa.get("lift", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="lift")
        _pump(env, 10)

    return [("approach", approach), ("descend", descend),
            ("grasp", grasp), ("lift", lift)]


def build_stack_stages(env, api):
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

    def approach(params, st):
        red_pos, _, _ = api.get_object_pose("red cube")
        green_pos, _, _ = api.get_object_pose("green cube")
        st["red_pos0"] = red_pos.copy()
        st["red_pos"] = red_pos.copy()
        st["green_pos"] = green_pos.copy()
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        ap = red_pos.copy(); ap[2] += 0.20
        api.goto_pose_with_profile(
            ap, grasp_quat,
            v_max=pv.get("approach", params["v_max"]),
            a_max=pa.get("approach", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="approach")
        _pump(env, 6)

    def descend(params, st):
        api.set_impedance_stiffness(kp_pos=params["kp_pos"])
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        grasp_pos = st["red_pos"].copy(); grasp_pos[2] -= 0.005
        st["grasp_pos"] = grasp_pos
        api.goto_pose_with_profile(
            grasp_pos, grasp_quat,
            v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
            a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
            jerk_limit=params["jerk_limit"], phase="descend")
        _pump(env, 6)

    def grasp(params, st):
        _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    def lift(params, st):
        api.set_impedance_stiffness(kp_pos=1000.0)
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        lift_pos = st["grasp_pos"].copy(); lift_pos[2] += 0.15
        st["lift_pos"] = lift_pos
        api.goto_pose_with_profile(
            lift_pos, grasp_quat,
            v_max=pv.get("lift", params["v_max"]),
            a_max=pa.get("lift", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="lift")
        _pump(env, 6)

    def transport(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        over_green = st["green_pos"].copy(); over_green[2] = st["lift_pos"][2]
        api.goto_pose_with_profile(
            over_green, grasp_quat,
            v_max=pv.get("transport", params["v_max"]),
            a_max=pa.get("transport", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="transport")
        _pump(env, 6)

    def final(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        stack_pos = st["green_pos"].copy(); stack_pos[2] = st["green_pos"][2] + 0.058
        api.goto_pose_with_profile(
            stack_pos, grasp_quat,
            v_max=pv.get("final", max(0.02, params["v_max"] * 0.5)),
            a_max=pa.get("final", max(0.01, params["a_max"] * 0.4)),
            jerk_limit=params["jerk_limit"], phase="final")
        _pump(env, 14)
        api.open_gripper()
        _pump(env, 15)

    def retreat(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        stack_pos = st["green_pos"].copy(); stack_pos[2] = st["green_pos"][2] + 0.058
        retreat_pos = stack_pos.copy(); retreat_pos[2] += 0.20
        api.goto_pose_with_profile(
            retreat_pos, grasp_quat,
            v_max=pv.get("retreat", params["v_max"]),
            a_max=pa.get("retreat", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="retreat")
        _pump(env, 20)

    return [("approach", approach), ("descend", descend), ("grasp", grasp),
            ("lift", lift), ("transport", transport), ("final", final),
            ("retreat", retreat)]


def build_pour_stages(env, api):
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

    def approach(params, st):
        cup_pos, _, _ = api.get_object_pose("water cup")
        st["cup_pos0"] = cup_pos.copy()
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        ap = cup_pos.copy(); ap[2] += 0.20
        api.goto_pose_with_profile(
            ap, grasp_quat,
            v_max=pv.get("approach", params["v_max"]),
            a_max=pa.get("approach", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="approach")
        _pump(env, 8)

    def descend(params, st):
        api.set_impedance_stiffness(kp_pos=params["kp_pos"])
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        cup_pos2, _, _ = api.get_object_pose("water cup")
        grasp_pos = cup_pos2.copy(); grasp_pos[2] -= 0.005
        st["grasp_pos"] = grasp_pos
        api.goto_pose_with_profile(
            grasp_pos, grasp_quat,
            v_max=pv.get("descend", max(0.02, params["v_max"] * 0.6)),
            a_max=pa.get("descend", max(0.01, params["a_max"] * 0.5)),
            jerk_limit=params["jerk_limit"], phase="descend")
        _pump(env, 8)

    def grasp(params, st):
        _grasp_close(env, api, params["grip"], n_tries=params.get("grasp_tries", 3))

    def lift(params, st):
        api.set_impedance_stiffness(kp_pos=1000.0)
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        lift_pos = st["grasp_pos"].copy(); lift_pos[2] += 0.15
        st["lift_pos"] = lift_pos
        api.goto_pose_with_profile(
            lift_pos, grasp_quat,
            v_max=pv.get("lift", params["v_max"]),
            a_max=pa.get("lift", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="lift")
        _pump(env, 10)

    def transport(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        target_pos, _, _ = api.get_object_pose("target spot")
        tp = target_pos.copy(); tp[2] = st["lift_pos"][2]
        st["transport_pos"] = tp
        api.goto_pose_with_profile(
            tp, grasp_quat,
            v_max=pv.get("transport", max(0.02, params["v_max"] * 0.5)),
            a_max=pa.get("transport", max(0.01, params["a_max"] * 0.4)),
            jerk_limit=params["jerk_limit"], phase="transport")
        _pump(env, 15)

    def final(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        putdown_pos = st["transport_pos"].copy(); putdown_pos[2] -= 0.13
        st["putdown_pos"] = putdown_pos
        api.goto_pose_with_profile(
            putdown_pos, grasp_quat,
            v_max=pv.get("final", max(0.02, params["v_max"] * 0.4)),
            a_max=pa.get("final", max(0.01, params["a_max"] * 0.3)),
            jerk_limit=params["jerk_limit"], phase="final")
        _pump(env, 10)
        api.open_gripper()
        _pump(env, 10)

    def retreat(params, st):
        pv, pa = params.get("phase_v_max", {}), params.get("phase_a_max", {})
        retreat_pos = st["putdown_pos"].copy(); retreat_pos[2] += 0.20
        api.goto_pose_with_profile(
            retreat_pos, grasp_quat,
            v_max=pv.get("retreat", params["v_max"]),
            a_max=pa.get("retreat", params["a_max"]),
            jerk_limit=params["jerk_limit"], phase="retreat")
        _pump(env, 10)

    return [("approach", approach), ("descend", descend), ("grasp", grasp),
            ("lift", lift), ("transport", transport), ("final", final),
            ("retreat", retreat)]


STAGE_BUILDERS = {
    "lift":  build_lift_stages,
    "stack": build_stack_stages,
    "pour":  build_pour_stages,
}


def _validate_stage_manner(seg, phase, spec, validator, StageStatus):
    """通道 A：对单个阶段 telemetry 切片做 manner 判定。

    把 MannerSpec 摊平成「只针对该 phase 相关维度」的子 spec：move 类阶段只查
    该 phase 的速度/jerk 上下界，grasp 阶段只查夹持/刚度界。无约束即视为通过。
    UNCERTAIN(证据不足) 不算失败，只有明确 NOT_SATISFIED 才判失败。
    """
    if not seg:
        return True
    from dataclasses import replace as _dc_replace
    if phase in _MOVE_STAGE_PHASES:
        vmax, vmin = spec.vel_bounds_for(phase)
        if vmax is None and vmin is None and spec.max_jerk is None:
            return True
        sspec = _dc_replace(
            spec, max_joint_vel=vmax, min_joint_vel=vmin, phase_velocity=None,
            kp_max=None, kp_min=None, grip_fraction_max=None,
            grip_fraction_min=None, max_jerk=None)
    elif phase in ("grasp", "grip"):
        if (spec.grip_fraction_max is None and spec.grip_fraction_min is None
                and spec.kp_max is None and spec.kp_min is None):
            return True
        sspec = _dc_replace(
            spec, max_joint_vel=None, min_joint_vel=None, phase_velocity=None,
            max_jerk=None)
    else:
        return True
    if not sspec.has_any_constraint():
        return True
    try:
        ev = validator.validate(seg, sspec)
        return ev.status != StageStatus.NOT_SATISFIED
    except Exception as e:
        print(f"    [stage manner-validate error @ {phase}] {e}")
        return True


def _stage_task_predicate(env, api, task, phase, st):
    """通道 B：单个阶段的物理子目标是否达成 (特权真值判定)。

    只对能廉价可靠判定的关键阶段设谓词 (grasp 抓稳 / lift 抬起 / stack 对齐)，
    其余阶段返回 True (交由后续阶段或最终 task_completed 兜底)。
    """
    try:
        if task == "lift":
            if phase == "lift":
                cube, _, _ = api.get_object_pose("red cube")
                return bool(cube[2] - st["cube_pos0"][2] > 0.05)
            return True
        if task == "stack":
            if phase == "lift":
                red, _, _ = api.get_object_pose("red cube")
                return bool(red[2] - st["red_pos0"][2] > 0.05)
            if phase in ("final", "retreat"):
                return bool(env.task_completed())
            return True
        if task == "pour":
            if phase == "lift":
                cup, _, _ = api.get_object_pose("water cup")
                return bool(cup[2] - st["cup_pos0"][2] > 0.05)
            if phase in ("final", "retreat"):
                sr = getattr(env, "spill_ratio", None)
                if callable(sr):
                    return bool(env.spill_ratio() > 0.999) and bool(env.task_completed())
                return bool(env.task_completed())
            return True
    except Exception as e:
        print(f"    [stage task-predicate error @ {task}/{phase}] {e}")
        return True
    return True


def _copy_params(params):
    out = dict(params)
    out["phase_v_max"] = dict(params.get("phase_v_max", {}))
    out["phase_a_max"] = dict(params.get("phase_a_max", {}))
    return out


def _stage_repair_plan(task, phase, cause, params, spec):
    """针对 (task, phase, 失败通道) 给出局部修复方案。

    返回 {name, rollback_to, params}；None 表示无已知修复 (诚实放弃该阶段)。
    - 物理抓取失败 (grasp/lift)：加收紧-回稳次数 + 抬夹持力，回滚到 descend 重新对位再抓。
    - stack 对齐失败 (final/retreat)：降 final 速度让方块稳落，回滚到 transport。
    - manner 越界 (move 阶段)：超上界->减速，低于下界->加速，原地重试该阶段。
    """
    if cause == "task" and phase in ("grasp", "lift"):
        np_ = _copy_params(params)
        np_["grasp_tries"] = min(6, int(params.get("grasp_tries", 3)) + 2)
        np_["grip"] = float(min(0.95, max(params["grip"], GRIP_FLOOR) + 0.05))
        pv = np_["phase_v_max"]
        pv["descend"] = max(0.015, pv.get("descend", params["v_max"] * 0.6) * 0.8)
        return {"name": "grip_reseat", "rollback_to": "descend", "params": np_}

    if cause == "task" and task == "stack" and phase in ("final", "retreat"):
        np_ = _copy_params(params)
        pv, pa = np_["phase_v_max"], np_["phase_a_max"]
        pv["final"] = max(0.012, pv.get("final", params["v_max"] * 0.5) * 0.6)
        pa["final"] = max(0.006, pa.get("final", params["a_max"] * 0.4) * 0.6)
        return {"name": "stack_settle", "rollback_to": "transport", "params": np_}

    if cause == "task" and task == "pour" and phase in ("final", "retreat"):
        np_ = _copy_params(params)
        pv, pa = np_["phase_v_max"], np_["phase_a_max"]
        for ph in ("transport", "final"):
            pv[ph] = max(0.012, pv.get(ph, params["v_max"] * 0.4) * 0.75)
            pa[ph] = max(0.006, pa.get(ph, params["a_max"] * 0.3) * 0.75)
        return {"name": "pour_slow", "rollback_to": "transport", "params": np_}

    if cause == "manner" and phase in _MOVE_STAGE_PHASES:
        vmax, vmin = spec.vel_bounds_for(phase)
        np_ = _copy_params(params)
        pv, pa = np_["phase_v_max"], np_["phase_a_max"]
        cur_v = pv.get(phase, params["v_max"])
        cur_a = pa.get(phase, params["a_max"])
        if vmax is not None:
            pv[phase] = max(0.01, min(cur_v, vmax * 0.8))
            pa[phase] = max(0.005, cur_a * 0.6)
        elif vmin is not None:
            pv[phase] = max(cur_v, vmin * 1.3)
            pa[phase] = cur_a * 1.3
        else:
            return None
        return {"name": f"manner_{phase}", "rollback_to": phase, "params": np_}

    return None


def _finalize_staged_result(task, env, api, st):
    exec_result = {
        "reward": float(env.compute_reward()),
        "sim_steps": int(env._sim_step_count),
        "task_completed": bool(env.task_completed()),
    }
    if task == "lift":
        cube_after, _, _ = api.get_object_pose("red cube")
        exec_result["cube_delta_z"] = float(cube_after[2] - st.get("cube_pos0", cube_after)[2])
    elif task == "stack":
        red_after, _, _ = api.get_object_pose("red cube")
        green_after, _, _ = api.get_object_pose("green cube")
        exec_result["red_delta_z"] = float(red_after[2] - st.get("red_pos0", red_after)[2])
        exec_result["z_gap"] = float(red_after[2] - green_after[2])
    elif task == "pour":
        cup_after, _, _ = api.get_object_pose("water cup")
        exec_result["cup_delta_z"] = float(cup_after[2] - st.get("cup_pos0", cup_after)[2])
        sr = getattr(env, "spill_ratio", None)
        if callable(sr):
            exec_result["spill_ratio"] = float(env.spill_ratio())
    return exec_result


def run_stages_with_rollback(env, api, stages, params, spec, validator,
                             StageStatus, task, max_repairs=2):
    """技术 #1+#2 的核心执行引擎：逐阶段执行 + 双通道判定 + 快照回滚 + 局部修复。

    与旧的单体 runner 相比：失败时只回滚并重跑「失败的那个阶段(及必要的前置阶段)」，
    而非整条 episode reset 重跑，显著降低浪费的仿真步数，并把修复定位到具体阶段。
    返回的 exec_result 字段与对应单体 runner 保持一致，额外带 _stage_log/_repairs_used。
    """
    st = {}
    phase_list = [p for p, _ in stages]
    entry_ckpts = {}
    stage_log = []
    cur_params = _copy_params(params)
    repairs_used = 0
    i = 0
    while i < len(stages):
        phase, fn = stages[i]
        entry_ckpts[phase] = env.capture_checkpoint()
        tele_start = len(env._manner_telemetry)
        fn(cur_params, st)
        seg = env._manner_telemetry[tele_start:]
        manner_ok = _validate_stage_manner(seg, phase, spec, validator, StageStatus)
        task_ok = _stage_task_predicate(env, api, task, phase, st)
        rec = {"stage": phase, "manner_ok": bool(manner_ok),
               "task_ok": bool(task_ok), "repaired": False}
        if manner_ok and task_ok:
            stage_log.append(rec)
            i += 1
            continue
        cause = "manner" if not manner_ok else "task"
        plan = None if repairs_used >= max_repairs else \
            _stage_repair_plan(task, phase, cause, cur_params, spec)
        if plan is None:
            rec["gave_up"] = True
            rec["cause"] = cause
            stage_log.append(rec)
            i += 1  # 无修复：带着失败继续后续阶段，最终由 task_completed 汇总判定
            continue
        repairs_used += 1
        rec["repaired"] = True
        rec["cause"] = cause
        rec["repair"] = plan["name"]
        rec["rollback_to"] = plan["rollback_to"]
        stage_log.append(rec)
        target = plan["rollback_to"]
        env.restore_checkpoint(entry_ckpts[target])
        cur_params = plan["params"]
        i = phase_list.index(target)

    exec_result = _finalize_staged_result(task, env, api, st)
    exec_result["_stage_log"] = stage_log
    exec_result["_repairs_used"] = int(repairs_used)
    return exec_result



TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "lift": {
        "sim_cls_path": "capx.envs.simulators.robosuite_cube_lift.FrankaRobosuiteCubeLiftLowLevel",
        "runner": run_cube_lift,
        "max_steps": 40000,
    },
    "stack": {
        "sim_cls_path": "capx.envs.simulators.robosuite_cubes.FrankaRobosuiteCubesLowLevel",
        "runner": run_cube_stack,
        "max_steps": 60000,
    },
    "pour": {
        "sim_cls_path": "capx.envs.simulators.robosuite_pour.FrankaRobosuitePourLowLevel",
        "runner": run_pour,
        "max_steps": 60000,
    },
    "wipe": {
        "sim_cls_path": "capx.envs.simulators.robosuite_spill_wipe.FrankaRobosuiteSpillWipeLowLevel",
        "runner": run_wipe,
        "max_steps": 6000,
    },
    "nut_assembly": {
        "sim_cls_path": "capx.envs.simulators.robosuite_nut_assembly.FrankaRobosuiteNutAssembly",
        "runner": run_nut_assembly,
        "max_steps": 40000,
    },
    "restack": {
        "sim_cls_path": "capx.envs.simulators.robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel",
        "runner": run_cube_restack,
        "max_steps": 60000,
    },
    # two_arm_handover 暂缓：其仿真器 RobosuiteHandoverEnv 继承的是 capx.envs.base.BaseEnv
    # 而非 lift/stack/pour/wipe/nut_assembly/restack 共用的 RobosuiteBaseEnv，且用双臂
    # (robots=["Panda","Panda"])，goto_pose_with_profile/move_to_joints_profiled/
    # begin_manner_recording 等单臂 manner 遥测基础设施均不兼容，需要单独扩展双臂支持，
    # 超出本次 Phase 1 接线范围，故意跳过（不强行造一个不可靠的接线）。
}


def load_sim_cls(dotted: str):
    mod_path, cls_name = dotted.rsplit(".", 1)
    __import__(mod_path)
    return getattr(sys.modules[mod_path], cls_name)


# ──────────────────────────────────────────────────────────────────────────────
# L3 指标
# ──────────────────────────────────────────────────────────────────────────────
def extract_metrics(telemetry: list) -> dict[str, float]:
    if not telemetry:
        return {}

    joint_speeds = np.array([np.linalg.norm(s.joint_vel) for s in telemetry])
    cfrcs = np.array([float(s.cfrc_norm) for s in telemetry])
    grip_fracs = np.array([float(s.gripper_fraction) for s in telemetry])
    accs = np.stack([np.asarray(s.joint_acc, dtype=np.float64) for s in telemetry], axis=0)

    if len(accs) >= 2:
        jerk = np.diff(accs, axis=0)
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    else:
        jerk_rms = 0.0

    grip_mask = np.array([s.phase == "grip" for s in telemetry])
    if grip_mask.any():
        peak_grip_force = float(np.max(cfrcs[grip_mask]))
    else:
        peak_grip_force = float(np.max(cfrcs))

    # finger separation during "move" phase AFTER first "grip" phase begins
    # (proxy for "cube stayed between fingers during transport")
    first_grip_idx = None
    for i, sn in enumerate(telemetry):
        if sn.phase == "grip":
            first_grip_idx = i
            break
    finger_sep_hold_mean = float("nan")
    finger_sep_hold_min  = float("nan")
    grip_active_ratio    = float("nan")
    # grip_active := finger separation < ~cube size AND contact force present
    # (both fingers squeezed on something, not just arm bumping table)
    GRIP_ACTIVE_SEP_THRESHOLD = 0.060   # cube side ~0.05m; leave small margin
    GRIP_ACTIVE_CFRC_THRESHOLD = 0.05
    if first_grip_idx is not None:
        seps = []
        active = []
        for sn in telemetry[first_grip_idx:]:
            if sn.phase == "move":
                sep = None
                if sn.gripper_qpos is not None and len(sn.gripper_qpos) >= 2:
                    sep = abs(float(sn.gripper_qpos[0]) - float(sn.gripper_qpos[1]))
                    seps.append(sep)
                cfrc = float(sn.cfrc_norm)
                is_active = (
                    sep is not None
                    and sep < GRIP_ACTIVE_SEP_THRESHOLD
                    and cfrc > GRIP_ACTIVE_CFRC_THRESHOLD
                )
                active.append(1 if is_active else 0)
        if seps:
            finger_sep_hold_mean = float(np.mean(seps))
            finger_sep_hold_min  = float(np.min(seps))
        if active:
            grip_active_ratio = float(np.mean(active))

    return {
        "mean_joint_speed": float(np.mean(joint_speeds)),
        "peak_joint_speed": float(np.max(joint_speeds)),
        "mean_cfrc":       float(np.mean(cfrcs)),
        "peak_cfrc":       float(np.max(cfrcs)),
        "peak_grip_force": peak_grip_force,
        "min_grip_frac":   float(np.min(grip_fracs)),
        "jerk_rms":        jerk_rms,
        "n_steps":         int(len(telemetry)),
        # New: cube-held-during-transport proxies
        "finger_sep_hold_mean": finger_sep_hold_mean,
        "finger_sep_hold_min":  finger_sep_hold_min,
        "grip_active_ratio":    grip_active_ratio,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mock spec
# ──────────────────────────────────────────────────────────────────────────────
def make_mock_spec(idx: int, tag: str):
    from capx.envs.manner_validator import MannerSpec, PhaseVelocity
    _mocks = {
        "gentle":        MannerSpec(max_joint_vel=0.04, kp_max=400.0, grip_fraction_max=0.30, max_jerk=0.8, label="mock_gentle"),
        "ultra_careful": MannerSpec(max_joint_vel=0.02, kp_max=300.0, grip_fraction_max=0.20, max_jerk=0.5, label="mock_ultra"),
        "ultra_gentle":  MannerSpec(max_joint_vel=0.02, kp_max=280.0, grip_fraction_max=0.15, max_jerk=0.5, label="mock_baby"),
        "normal":        MannerSpec(label="mock_normal"),
        "fast":          MannerSpec(min_joint_vel=0.30, label="mock_fast"),
        "firm":          MannerSpec(kp_min=1200.0, grip_fraction_min=0.85, label="mock_firm"),
        "fast_then_slow":MannerSpec(
            phase_velocity={
                "approach": PhaseVelocity(min_joint_vel=0.30),
                "descend":  PhaseVelocity(max_joint_vel=0.02),
                "lift":     PhaseVelocity(max_joint_vel=0.05),
            },
            kp_max=500.0, grip_fraction_max=0.30, max_jerk=1.2,
            label="mock_fast_then_slow_phased",
        ),
        "gentle_en":     MannerSpec(max_joint_vel=0.04, kp_max=380.0, grip_fraction_max=0.30, label="mock_gentle_en"),
        "fast_en":       MannerSpec(min_joint_vel=0.28, label="mock_fast_en"),
        "phased_5stage": MannerSpec(
            phase_velocity={
                "approach":  PhaseVelocity(min_joint_vel=0.30),
                "descend":   PhaseVelocity(max_joint_vel=0.02),
                "lift":      PhaseVelocity(max_joint_vel=0.04),
                "transport": PhaseVelocity(max_joint_vel=0.04),
                "final":     PhaseVelocity(max_joint_vel=0.015),
                "retreat":   PhaseVelocity(min_joint_vel=0.28),
            },
            grip_fraction_max=0.62, max_jerk=1.0,
            label="mock_phased_5stage",
        ),
    }
    return _mocks.get(tag, MannerSpec(label=f"mock_{tag}"))


# ──────────────────────────────────────────────────────────────────────────────
# Trial 数据结构与 outcome 分类（Strategy B）
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrialResult:
    task: str
    tag: str
    seed: int
    prompt: str
    label: str
    spec_json: dict
    params: dict
    clipped: list
    task_ok: Any
    manner_ok: Any
    outcome: str
    reward: Any
    exec_result: dict
    metrics: dict
    error: Any
    exec_time_s: float
    traj_file: Any = None   # 逐步物理量轨迹 npz 文件名（相对 traj_dir），None=未保存
    wrist_video_file: Any = None  # wrist-camera mp4 filename (relative to traj_dir), None=not captured
    repair_attempts: list = field(default_factory=list)  # bounded-retry log triggered by MANNER_REPAIR_LIBRARY
    calibrated: list = field(default_factory=list)  # phase/dimension keys overridden by experience-library calibration
    alignment_score: Any = None  # AlignmentReport (asdict-serialized): continuous per-phase
    # language-alignment score in [0,1] from compute_alignment_score(), complements the
    # binary manner_ok gate with a smooth "how well did the action match the instruction"
    # metric per phase. None = no constraint to score (baseline) or no telemetry.


def classify_outcome(task_ok, manner_ok, error_msg: str | None = None) -> str:
    if error_msg is not None:
        # Strategy B (extended): timeout = manner too extreme to complete within horizon
        low = error_msg.lower()
        if "terminated episode" in low or "horizon" in low or "max_step" in low:
            return "manner_infeasible_timeout"
        if "ik" in low or "solver" in low or "solve" in low:
            return "manner_infeasible_ik"
        return "error"
    if task_ok is None or manner_ok is None:
        return "error"
    if task_ok and manner_ok:
        return "both_pass"
    if task_ok and not manner_ok:
        return "task_only"
    if not task_ok and manner_ok:
        return "manner_infeasible"   # Strategy B (physics)
    return "both_fail"


# ------------------------------------------------------------------------------
# MANNER_REPAIR_LIBRARY -- failure-mode -> repair hint + local param adjustment
# ------------------------------------------------------------------------------
# Inspired by vlm_planner_cap/pattern_priors.py's repair_hints design: distill
# previously-diagnosed failure root-causes into reusable "experience" entries
# instead of re-deriving the fix from scratch every time. Unlike pattern_priors
# (which injects hint text into an LLM re-planning prompt), there is no LLM in
# the loop here -- each hint is paired with a concrete parameter-adjustment
# function, applied as a single bounded retry inside run_single_trial (never
# unbounded, so genuine manner_infeasible "honest refusals" are preserved).
def _adjust_grip_slip(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    # grasp_tries is an integer retry count, not a physical dimension tracked
    # by the experience library (flatten_trial_params only knows v_max/a_max/
    # grip/kp_pos/jerk_limit) -- experience_lib/task accepted only for a
    # uniform call signature across MANNER_REPAIR_LIBRARY["adjust"] entries.
    p = dict(params)
    p["grasp_tries"] = min(5, int(p.get("grasp_tries", 3)) + 2)
    return p


def _adjust_pour_spill(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    p = dict(params)
    pv = dict(p.get("phase_v_max", {}))
    pa = dict(p.get("phase_a_max", {}))
    calibrated: list[str] = list(p.get("_repair_calibrated", []))
    for name in ("transport", "final"):
        ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, name, "spill", "v_max", 0.75, min_samples)
        ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, name, "spill", "a_max", 0.75, min_samples)
        if name in pv:
            pv[name] = max(0.01, pv[name] * ratio_v)
            if used_v:
                calibrated.append(f"{name}.v_max")
        if name in pa:
            pa[name] = max(0.005, pa[name] * ratio_a)
            if used_a:
                calibrated.append(f"{name}.a_max")
    p["phase_v_max"] = pv
    p["phase_a_max"] = pa
    p["_repair_calibrated"] = calibrated
    return p


def _adjust_timeout(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    """Manner too slow -> steps approach/exceed horizon. Speed up only the
    slowest phase slightly, keeping it still the slowest (preserves the
    manner's qualitative fast/slow ordering) to cut convergence steps."""
    p = dict(params)
    pv = dict(p.get("phase_v_max", {}))
    pa = dict(p.get("phase_a_max", {}))
    calibrated: list[str] = list(p.get("_repair_calibrated", []))
    if pv:
        slowest = min(pv, key=lambda k: pv[k])
        ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, slowest, "timeout", "v_max", 1.25, min_samples)
        pv[slowest] = pv[slowest] * ratio_v
        if used_v:
            calibrated.append(f"{slowest}.v_max")
        if slowest in pa:
            ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, slowest, "timeout", "a_max", 1.25, min_samples)
            pa[slowest] = pa[slowest] * ratio_a
            if used_a:
                calibrated.append(f"{slowest}.a_max")
        p["phase_v_max"] = pv
        p["phase_a_max"] = pa
    else:
        ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, GLOBAL_PHASE, "timeout", "v_max", 1.25, min_samples)
        ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, GLOBAL_PHASE, "timeout", "a_max", 1.25, min_samples)
        p["v_max"] = p.get("v_max", 0.12) * ratio_v
        p["a_max"] = p.get("a_max", 0.08) * ratio_a
        if used_v:
            calibrated.append("v_max")
        if used_a:
            calibrated.append("a_max")
    p["_repair_calibrated"] = calibrated
    return p


def _adjust_stack_misalign(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    p = dict(params)
    pv = dict(p.get("phase_v_max", {}))
    pa = dict(p.get("phase_a_max", {}))
    calibrated: list[str] = list(p.get("_repair_calibrated", []))
    ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, "final", "stack_misalign", "v_max", 0.7, min_samples)
    ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, "final", "stack_misalign", "a_max", 0.7, min_samples)
    if "final" in pv:
        pv["final"] = max(0.01, pv["final"] * ratio_v)
        if used_v:
            calibrated.append("final.v_max")
    if "final" in pa:
        pa["final"] = max(0.005, pa["final"] * ratio_a)
        if used_a:
            calibrated.append("final.a_max")
    p["phase_v_max"] = pv
    p["phase_a_max"] = pa
    p["_repair_calibrated"] = calibrated
    return p


def _adjust_nut_misalign(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    p = dict(params)
    pv = dict(p.get("phase_v_max", {}))
    pa = dict(p.get("phase_a_max", {}))
    calibrated: list[str] = list(p.get("_repair_calibrated", []))
    ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, "final", "nut_misalign", "v_max", 0.7, min_samples)
    ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, "final", "nut_misalign", "a_max", 0.7, min_samples)
    if "final" in pv:
        pv["final"] = max(0.01, pv["final"] * ratio_v)
        if used_v:
            calibrated.append("final.v_max")
    if "final" in pa:
        pa["final"] = max(0.005, pa["final"] * ratio_a)
        if used_a:
            calibrated.append("final.a_max")
    p["phase_v_max"] = pv
    p["phase_a_max"] = pa
    p["_repair_calibrated"] = calibrated
    return p


def _adjust_wipe_incomplete(params: dict, experience_lib=None, task=None, min_samples: int = 3) -> dict:
    p = dict(params)
    pv = dict(p.get("phase_v_max", {}))
    pa = dict(p.get("phase_a_max", {}))
    calibrated: list[str] = list(p.get("_repair_calibrated", []))
    ratio_v, used_v = _calibrate_repair_ratio(experience_lib, task, "transport", "wipe_incomplete", "v_max", 0.75, min_samples)
    ratio_a, used_a = _calibrate_repair_ratio(experience_lib, task, "transport", "wipe_incomplete", "a_max", 0.75, min_samples)
    if "transport" in pv:
        pv["transport"] = max(0.01, pv["transport"] * ratio_v)
        if used_v:
            calibrated.append("transport.v_max")
    if "transport" in pa:
        pa["transport"] = max(0.005, pa["transport"] * ratio_a)
        if used_a:
            calibrated.append("transport.a_max")
    p["phase_v_max"] = pv
    p["phase_a_max"] = pa
    p["wipe_iters"] = min(16, int(p.get("wipe_iters", 8)) + 4)
    p["_repair_calibrated"] = calibrated
    return p


MANNER_REPAIR_LIBRARY: dict[str, dict[str, Any]] = {
    "grip_slip": {
        "hint": "Grip slipped (fingers never held object stably); without "
                "raising the peak grip-force ceiling, increase the number of "
                "tighten-settle cycles so fingertips can re-seat, then retry.",
        "adjust": _adjust_grip_slip,
    },
    "spill": {
        "hint": "Transport/putdown too fast, causing spill; lower transport/"
                "final phase speed 25% (still respecting the manner's "
                "qualitative trend) then retry.",
        "adjust": _adjust_pour_spill,
    },
    "timeout": {
        "hint": "Manner too slow, run approached/exceeded the horizon step "
                "budget; slightly speed up only the single slowest phase, "
                "then retry.",
        "adjust": _adjust_timeout,
    },
    "stack_misalign": {
        "hint": "Stack placement misaligned (likely overshoot/slide); lower "
                "the final-phase placement speed, then retry.",
        "adjust": _adjust_stack_misalign,
    },
    "nut_misalign": {
        "hint": "Nut not seated on peg within tolerance; lower the "
                "final-phase placement speed, then retry.",
        "adjust": _adjust_nut_misalign,
    },
    "wipe_incomplete": {
        "hint": "Markers remain unwiped; slow the sweep/transport phase and "
                "allow more sweep iterations, then retry.",
        "adjust": _adjust_wipe_incomplete,
    },
}


def diagnose_failure(task: str, outcome: str, exec_result: dict, metrics: dict) -> str | None:
    """Map (task, outcome, exec_result, metrics) to a MANNER_REPAIR_LIBRARY key.

    Returns None when no known failure signature matches -- this intentionally
    does NOT trigger a retry, preserving the "honest refusal" semantics of
    manner_infeasible for genuinely unclassified failures.
    """
    if outcome == "manner_infeasible_timeout":
        return "timeout"
    if task == "pour" and exec_result.get("spill_ratio") is not None \
            and exec_result["spill_ratio"] < 0.999 and outcome in ("both_fail", "manner_infeasible"):
        return "spill"
    if task in ("stack", "restack") and outcome in ("both_fail", "manner_infeasible") \
            and exec_result.get("z_gap") is not None and abs(exec_result["z_gap"] - 0.05) > 0.02:
        return "stack_misalign"
    if task == "nut_assembly" and outcome in ("both_fail", "manner_infeasible") \
            and exec_result.get("nut_xy_error") is not None and exec_result["nut_xy_error"] > 0.03:
        return "nut_misalign"
    if task == "wipe" and outcome in ("both_fail", "manner_infeasible") \
            and exec_result.get("markers_remaining") is not None and exec_result["markers_remaining"] > 0:
        return "wipe_incomplete"
    if outcome in ("both_fail", "manner_infeasible") \
            and metrics.get("grip_active_ratio") is not None \
            and metrics["grip_active_ratio"] < 0.5:
        return "grip_slip"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["lift"],
                   choices=list(TASK_REGISTRY.keys()))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--mock-llm", action="store_true")
    p.add_argument("--include-baseline", action="store_true", default=True)
    p.add_argument("--no-baseline", dest="include_baseline", action="store_false")
    p.add_argument("--output", type=str,
                   default="repro_results/robosuite/manner_ablation.jsonl")
    p.add_argument("--traj-dir", type=str,
                   default="repro_results/robosuite/traj",
                   help="逐步物理量轨迹 npz 输出目录（相对仓库根，同 --output 语义）；传空字符串禁用")
    p.add_argument("--tags", nargs="+", default=None,
                   help="Subset of TEST_CASES tags to run (default: all). Baseline is controlled separately by --include-baseline/--no-baseline.")
    p.add_argument("--capture-video", action="store_true",
                   help="Enable wrist-camera rendering + per-trial mp4 export into --traj-dir.")
    p.add_argument("--staged-rollback", dest="staged_rollback",
                   action="store_true",
                   help="Use stage-snapshot rollback engine (per-stage "
                        "checkpoint + dual-channel validate + local repair) "
                        "instead of the monolithic runner + single global retry.")
    p.add_argument("--use-experience-calibration", dest="use_experience_calibration",
                   action="store_true",
                   help="Calibrate phase_v_max/phase_a_max/kp_pos/grip against "
                        "historical both_pass samples in the experience library "
                        "(synthesis_v2.experience_library) instead of the fixed "
                        "ratio presets, whenever a (task, phase, dimension, tag) "
                        "bucket has >= --experience-min-samples matching records. "
                        "Falls back to the existing preset when the bucket is "
                        "cold (no regression vs. today's behaviour).")
    p.add_argument("--experience-library-path", type=str,
                   default="repro_results/robosuite/experience_library.jsonl",
                   help="Path to the experience library jsonl (see "
                        "scripts/build_experience_library.py to populate it "
                        "from historical trial jsonl).")
    p.add_argument("--experience-min-samples", type=int, default=3,
                   help="Minimum matching historical samples required before "
                        "the calibration layer overrides a preset value.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def save_trajectory_npz(telemetry: list, traj_dir: str, task: str, tag: str,
                         seed: int, run_stamp: str) -> str:
    """把整条 episode 的逐步 MannerStepSnapshot 序列落盘为 .npz。

    数组形状约定（T = 该 episode 的 manner-recording 步数）：
        t (T,) s；phase (T,) str；joint_pos/joint_vel/joint_acc (T,7)；
        kp (T,)；gripper_fraction (T,)；gripper_qpos (T,2)；cfrc_norm (T,)；
        eef_pos (T,3)；eef_quat (T,4) wxyz。
        object_names (N,) str（该 episode 内追踪的物体名，如 cube/cubeA/cubeB/cup）；
        object_pos (T,N,3)；object_quat (T,N,4) wxyz — 物体真值位姿随时间的变化
        （特权信息，来自仿真器原始 observable，非视觉估计）。

    Returns:
        文件名（不含目录），供 jsonl 记录里的 traj_file 字段引用。
    """
    t = np.array([s.t for s in telemetry], dtype=np.float64)
    phase = np.array([s.phase for s in telemetry], dtype="<U16")
    joint_vel = np.stack([np.asarray(s.joint_vel, dtype=np.float64) for s in telemetry])
    joint_acc = np.stack([np.asarray(s.joint_acc, dtype=np.float64) for s in telemetry])
    joint_pos = np.stack([np.asarray(getattr(s, "joint_pos", np.zeros(7)), dtype=np.float64) for s in telemetry])
    kp = np.array([s.kp for s in telemetry], dtype=np.float64)
    gripper_fraction = np.array([s.gripper_fraction for s in telemetry], dtype=np.float64)
    gripper_qpos = np.stack([np.asarray(s.gripper_qpos, dtype=np.float64) for s in telemetry])
    cfrc_norm = np.array([s.cfrc_norm for s in telemetry], dtype=np.float64)
    grasp_contact_force = np.array([getattr(s, "grasp_contact_force", 0.0) for s in telemetry], dtype=np.float64)
    eef_pos = np.stack([np.asarray(getattr(s, "eef_pos", np.zeros(3)), dtype=np.float64) for s in telemetry])
    eef_quat = np.stack([np.asarray(getattr(s, "eef_quat", np.zeros(4)), dtype=np.float64) for s in telemetry])

    # object ground-truth poses (privileged): object_names is constant across
    # one episode (same task -> same tracked objects), so take it from step 0.
    object_names = getattr(telemetry[0], "object_names", ()) if telemetry else ()
    n_obj = len(object_names)
    object_pos = np.stack([
        np.asarray(getattr(s, "object_pos", np.zeros((n_obj, 3))), dtype=np.float64)
        for s in telemetry
    ]) if n_obj else np.zeros((len(telemetry), 0, 3))
    object_quat = np.stack([
        np.asarray(getattr(s, "object_quat", np.zeros((n_obj, 4))), dtype=np.float64)
        for s in telemetry
    ]) if n_obj else np.zeros((len(telemetry), 0, 4))

    fname = f"{task}_{tag}_seed{seed}_{run_stamp}.npz"
    fpath = os.path.join(traj_dir, fname)
    np.savez_compressed(
        fpath,
        t=t, phase=phase,
        joint_pos=joint_pos, joint_vel=joint_vel, joint_acc=joint_acc,
        kp=kp, gripper_fraction=gripper_fraction, gripper_qpos=gripper_qpos,
        cfrc_norm=cfrc_norm, grasp_contact_force=grasp_contact_force,
        eef_pos=eef_pos, eef_quat=eef_quat,
        object_names=np.array(object_names, dtype="<U16"),
        object_pos=object_pos, object_quat=object_quat,
        task=task, tag=tag, seed=seed,
    )
    return fname


def run_single_trial(task, tag, prompt, seed, spec, args,
                     env, api, validator, StageStatus, runner,
                     traj_dir=None, run_stamp="", capture_video=False):
    from capx.envs.manner_validator import compute_alignment_score
    t0 = time.time()
    experience_lib = None
    if getattr(args, "use_experience_calibration", False):
        experience_lib = _get_experience_library(args.experience_library_path)
    params = spec_to_exec_params(
        spec, task=task, tag=tag, experience_lib=experience_lib,
        min_samples=getattr(args, "experience_min_samples", 3))
    spec_json = {
        "max_joint_vel":     spec.max_joint_vel,
        "min_joint_vel":     spec.min_joint_vel,
        "kp_max":            spec.kp_max,
        "kp_min":            spec.kp_min,
        "grip_fraction_max": spec.grip_fraction_max,
        "grip_fraction_min": spec.grip_fraction_min,
        "max_jerk":          spec.max_jerk,
        "label":             spec.label,
    }

    def _attempt(attempt_params):
        """Run one execution attempt against the sim; return raw result fields
        (no TrialResult wrapping yet -- lets us run this twice for the bounded
        MANNER_REPAIR_LIBRARY retry without duplicating the whole function)."""
        try:
            env.reset(seed=seed)
            if capture_video:
                env.enable_video_capture(True, clear=True, wrist_camera=True)
            env.begin_manner_recording(clear=True)
            if getattr(args, "staged_rollback", False) and task in STAGE_BUILDERS:
                stages = STAGE_BUILDERS[task](env, api)
                exec_result = run_stages_with_rollback(
                    env, api, stages, attempt_params, spec, validator,
                    StageStatus, task)
            else:
                exec_result = runner(env, api, attempt_params)
            env.end_manner_recording()
        except Exception as e:
            emsg = f"{type(e).__name__}: {e}"
            outcome = classify_outcome(None, None, error_msg=emsg)
            return {
                "task_ok": None, "manner_ok": None, "outcome": outcome,
                "reward": None, "exec_result": {}, "metrics": {},
                "error": f"{emsg}\n{traceback.format_exc(limit=4)}",
                "telemetry": None, "alignment_score": None,
            }

        task_ok = bool(exec_result.get("task_completed", False))
        reward = float(exec_result.get("reward", 0.0))

        # Fix (2): if Strategy A clipped a physical dim (e.g. grip forced up to
        # GRIP_FLOOR because spec said grip<0.30 which cannot hold a 5cm cube),
        # validate against the *clipped_spec* -- the effective bounds we actually
        # attempted to honor. This turns "physically infeasible + L1 fail" into
        # `manner_infeasible` (Strategy B) instead of the ambiguous `both_fail`.
        from dataclasses import replace as _dc_replace
        _dim_to_fields = {
            "vel":  ("max_joint_vel", "min_joint_vel"),
            "kp":   ("kp_max", "kp_min"),
            "grip": ("grip_fraction_max", "grip_fraction_min"),
        }
        clipped_dims = attempt_params.get("_clipped_dims", [])
        if clipped_dims:
            overrides = {}
            for d in clipped_dims:
                for fn in _dim_to_fields.get(d, ()):
                    overrides[fn] = None
            clipped_spec = _dc_replace(spec, **overrides)
        else:
            clipped_spec = spec

        manner_ok = None
        telemetry = env.get_manner_telemetry()
        if telemetry and clipped_spec.has_any_constraint():
            try:
                ev = validator.validate(telemetry, clipped_spec)
                manner_ok = (ev.status == StageStatus.SATISFIED)
            except Exception as e:
                print(f"    [validator error] {e}")
                manner_ok = None
        elif not clipped_spec.has_any_constraint():
            manner_ok = True   # baseline / normal / fully-clipped
        metrics = extract_metrics(telemetry) if telemetry else {}

        # Continuous, per-phase language-alignment score -- complements the
        # binary manner_ok gate above with a [0,1] margin score (reporting /
        # ablation only, does not affect outcome classification or repair).
        alignment_score = None
        if telemetry and clipped_spec.has_any_constraint():
            try:
                from dataclasses import asdict as _asdict
                alignment_score = _asdict(
                    compute_alignment_score(telemetry, clipped_spec)
                )
            except Exception as e:
                print(f"    [alignment score error] {e}")
                alignment_score = None

        return {
            "task_ok": task_ok, "manner_ok": manner_ok,
            "outcome": classify_outcome(task_ok, manner_ok),
            "reward": reward, "exec_result": exec_result, "metrics": metrics,
            "error": None, "telemetry": telemetry,
            "alignment_score": alignment_score,
        }

    result = _attempt(params)
    repair_attempts = []
    pre_repair_params = None
    repair_cause = None

    # Bounded (<=1) MANNER_REPAIR_LIBRARY retry: only for outcomes that mean
    # the task itself failed, and only when diagnose_failure() matches a known
    # failure signature. Unmatched failures are left alone (honest refusal).
    if result["outcome"] in ("both_fail", "manner_infeasible", "manner_infeasible_timeout"):
        diagnosis = diagnose_failure(task, result["outcome"], result["exec_result"], result["metrics"])
        if diagnosis is not None and diagnosis in MANNER_REPAIR_LIBRARY:
            entry = MANNER_REPAIR_LIBRARY[diagnosis]
            adjusted_params = entry["adjust"](
                params, experience_lib=experience_lib, task=task,
                min_samples=getattr(args, "experience_min_samples", 3))
            adjusted_params["_clipped"] = params.get("_clipped", [])
            adjusted_params["_clipped_dims"] = params.get("_clipped_dims", [])
            repair_attempts.append({
                "attempt": 1, "diagnosis": diagnosis, "hint": entry["hint"],
                "prior_outcome": result["outcome"],
                "calibrated_dims": adjusted_params.get("_repair_calibrated", []),
            })
            pre_repair_params = dict(params)
            repair_cause = diagnosis
            retry_result = _attempt(adjusted_params)
            params = adjusted_params
            result = retry_result

    if result["error"] is not None:
        return TrialResult(
            task=task, tag=tag, seed=seed, prompt=prompt, label=spec.label,
            spec_json=spec_json,
            params={k: v for k, v in params.items() if k not in ("_clipped", "_clipped_dims", "_calibrated", "_repair_calibrated")},
            clipped=params["_clipped"],
            task_ok=None, manner_ok=None, outcome=result["outcome"],
            reward=None, exec_result={}, metrics={},
            error=result["error"],
            exec_time_s=time.time() - t0,
            traj_file=None,
            repair_attempts=repair_attempts,
            alignment_score=result.get("alignment_score"),
            calibrated=params.get("_calibrated", []),
        )

    telemetry = result["telemetry"]
    traj_file = None
    if traj_dir and telemetry:
        try:
            traj_file = save_trajectory_npz(telemetry, traj_dir, task, tag, seed, run_stamp)
        except Exception as e:
            print(f"    [traj save error] {e}")
            traj_file = None

    wrist_video_file = None
    if capture_video and traj_dir:
        try:
            frames = env.get_wrist_video_frames()
            if frames:
                import imageio
                fname = f"{task}_{tag}_seed{seed}_{run_stamp}_wrist.mp4"
                fpath = os.path.join(traj_dir, fname)
                imageio.mimsave(fpath, frames, fps=20, macro_block_size=None)
                wrist_video_file = fname
        except Exception as e:
            print(f"    [video save error] {e}")
            wrist_video_file = None

    # Phase 5: self-evolving write-back. Only for genuinely successful trials,
    # and only when the experience library is enabled (--use-experience-
    # calibration) -- same gate as the read side, so the closed loop is fully
    # opt-in and the default (flag off) path is byte-identical to pre-Phase-5.
    if experience_lib is not None and result["outcome"] == "both_pass":
        try:
            _write_back_experience(
                experience_lib, args.experience_library_path, task, tag, seed,
                run_stamp, params, "both_pass")
            if pre_repair_params is not None:
                _write_back_repair(
                    experience_lib, args.experience_library_path, task,
                    run_stamp, repair_cause, pre_repair_params, params,
                    "both_pass")
        except Exception as e:
            print(f"    [experience write-back error] {e}")

    return TrialResult(
        task=task, tag=tag, seed=seed, prompt=prompt, label=spec.label,
        spec_json=spec_json,
        params={k: v for k, v in params.items() if k not in ("_clipped", "_clipped_dims", "_calibrated", "_repair_calibrated")},
        clipped=params["_clipped"],
        task_ok=result["task_ok"], manner_ok=result["manner_ok"],
        outcome=result["outcome"],
        reward=result["reward"], exec_result=result["exec_result"], metrics=result["metrics"],
        error=None, exec_time_s=time.time() - t0,
        traj_file=traj_file,
        wrist_video_file=wrist_video_file,
        repair_attempts=repair_attempts,
        alignment_score=result.get("alignment_score"),
        calibrated=params.get("_calibrated", []),
    )


def main() -> int:
    args = parse_args()

    from capx.envs.manner_validator import (
        MannerValidator, MannerSpec, StageStatus,
    )

    parser = None
    _spec_cache: dict = {}
    if not args.mock_llm:
        from capx.llm.manner_parser import LLMMannerParser
        parser = LLMMannerParser()
        print("[LLM] 初始化 Gemini 解析器 ✓")
    else:
        print("[LLM] --mock-llm：使用固定 MannerSpec")

    def _get_spec(idx: int, prompt: str, tag: str):
        if tag == BASELINE_TAG:
            return MannerSpec(label="baseline_no_constraint")
        if prompt in _spec_cache:
            return _spec_cache[prompt]
        if parser is None:
            spec = make_mock_spec(idx, tag)
        else:
            spec, _reasoning = parser.parse_with_reasoning(prompt)
        _spec_cache[prompt] = spec
        return spec

    cases: list[tuple[str, str]] = list(TEST_CASES)
    if args.tags:
        tagset = set(args.tags)
        cases = [(p, t) for (p, t) in cases if t in tagset]
    if args.include_baseline:
        cases = [("(no manner constraint)", BASELINE_TAG)] + cases

    if args.dry_run:
        print("=== DRY RUN ===")
        for task in args.tasks:
            for seed_i in range(args.seeds):
                seed = args.seed_start + seed_i
                for idx, (prompt, tag) in enumerate(cases):
                    print(f"  {task}  seed={seed}  [{idx:02d}] {tag}  <- {prompt[:40]}")
        return 0

    out_path = os.path.abspath(os.path.join(BASE, "..", args.output)) \
        if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "a", encoding="utf-8")
    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    print(f"[OUT] JSONL → {out_path}  (run_stamp={run_stamp})")

    traj_dir_abs = None
    if args.traj_dir:
        traj_dir_abs = os.path.abspath(os.path.join(BASE, "..", args.traj_dir)) \
            if not os.path.isabs(args.traj_dir) else args.traj_dir
        os.makedirs(traj_dir_abs, exist_ok=True)
        print(f"[OUT] Trajectories → {traj_dir_abs}")

    validator = MannerValidator()
    all_results: list[TrialResult] = []

    for task in args.tasks:
        entry = TASK_REGISTRY[task]
        sim_cls = load_sim_cls(entry["sim_cls_path"])
        runner = entry["runner"]

        print(f"\n{'=' * 72}\n  TASK = {task}\n{'=' * 72}")
        env = sim_cls(
            privileged=True, enable_render=args.capture_video,
            max_steps=entry["max_steps"], seed=args.seed_start,
        )
        from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
        api = FrankaControlPrivilegedApi(env)

        for seed_i in range(args.seeds):
            seed = args.seed_start + seed_i
            print(f"\n--- seed = {seed} ---")

            for idx, (prompt, tag) in enumerate(cases):
                print(f"[{idx+1:02d}/{len(cases):02d}] {task}/{tag}  seed={seed}")

                try:
                    spec = _get_spec(idx, prompt, tag)
                except Exception as e:
                    print(f"  ✗ spec 解析失败: {e}")
                    continue

                print(f"  spec: label={spec.label}  "
                      f"vel[{spec.min_joint_vel},{spec.max_joint_vel}]  "
                      f"kp[{spec.kp_min},{spec.kp_max}]  "
                      f"grip[{spec.grip_fraction_min},{spec.grip_fraction_max}]  "
                      f"jerk={spec.max_jerk}")

                r = run_single_trial(
                    task, tag, prompt, seed, spec, args,
                    env, api, validator, StageStatus, runner,
                    traj_dir_abs, run_stamp,
                    capture_video=args.capture_video,
                )

                if r.clipped:
                    print(f"  [clip A] {', '.join(r.clipped)}")

                if r.error:
                    print(f"  ✗ error: {r.error.splitlines()[0]}")
                else:
                    task_s = "✓" if r.task_ok else "✗"
                    manner_s = ("✓" if r.manner_ok is True else
                                "✗" if r.manner_ok is False else "?")
                    print(f"  task={task_s} reward={r.reward:.2f}  manner={manner_s}  "
                          f"→ {r.outcome}  [{r.exec_time_s:.1f}s]")
                    if r.metrics:
                        m = r.metrics
                        print(f"  L3: speed(mean/peak)={m['mean_joint_speed']:.3f}/{m['peak_joint_speed']:.3f}  "
                              f"cfrc(peak)={m['peak_cfrc']:.1f}  "
                              f"grip_force={m['peak_grip_force']:.1f}  "
                              f"jerk_rms={m['jerk_rms']:.2f}  "
                              f"grip_active={m.get('grip_active_ratio', float('nan')):.2f}  "
                              f"finger_sep={m.get('finger_sep_hold_mean', float('nan')):.4f}")
                    if r.alignment_score and r.alignment_score.get("overall_score") is not None:
                        a = r.alignment_score
                        phases = " ".join(
                            f"{p}={s:.2f}" for p, s in a.get("phase_scores", {}).items()
                        )
                        print(f"  align: overall={a['overall_score']:.3f}  [{phases}]")

                rec = asdict(r)
                rec["run_stamp"] = run_stamp
                fout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fout.flush()
                all_results.append(r)

        env.close()

    fout.close()
    print_summary(all_results)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# 汇总（含 vs baseline delta）
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(results: list) -> None:
    print(f"\n{'=' * 90}\n  SUMMARY\n{'=' * 90}")
    if not results:
        print("  (no results)")
        return

    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.task, r.tag)].append(r)

    header = (f"{'task':<6} {'tag':<16} {'n':>3} {'L1':>7} {'L2':>7} {'both':>7}   "
              f"{'align':>7} "
              f"{'speed':>8} {'cfrc':>8} {'gripF':>8} {'jerk':>7} {'grip_act':>9}   outcome_mix")
    print(header)
    print("-" * len(header))

    baseline_metrics: dict = {}
    for (task, tag), rs in groups.items():
        if tag == BASELINE_TAG:
            baseline_metrics[task] = _mean_metrics(rs)

    def _delta(m, base, key):
        v = m.get(key)
        b = base.get(key) if base else None
        if v is None:
            return "   -   "
        if b is None:
            return f"{v:7.2f}"
        d = v - b
        sign = "+" if d >= 0 else "-"
        return f"{v:5.2f}{sign}{abs(d):.2f}"

    def _mean_alignment(rs: list) -> str:
        scores = [
            r.alignment_score["overall_score"] for r in rs
            if r.alignment_score and r.alignment_score.get("overall_score") is not None
        ]
        if not scores:
            return "   -   "
        return f"{sum(scores) / len(scores):7.3f}"

    for (task, tag), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] != BASELINE_TAG, x[0][1])):
        n = len(rs)
        l1 = sum(1 for r in rs if r.task_ok is True)
        l2 = sum(1 for r in rs if r.manner_ok is True)
        both = sum(1 for r in rs if r.task_ok and r.manner_ok)
        m = _mean_metrics(rs)
        base = baseline_metrics.get(task, {}) if tag != BASELINE_TAG else None

        outcomes = defaultdict(int)
        for r in rs:
            outcomes[r.outcome] += 1
        mix = " ".join(f"{k}:{v}" for k, v in outcomes.items())

        print(f"{task:<6} {tag:<16} {n:>3} "
              f"{l1:>2}/{n:<4} {l2:>2}/{n:<4} {both:>2}/{n:<4}   "
              f"{_mean_alignment(rs)} "
              f"{_delta(m, base, 'mean_joint_speed'):>8} "
              f"{_delta(m, base, 'peak_cfrc'):>8} "
              f"{_delta(m, base, 'peak_grip_force'):>8} "
              f"{_delta(m, base, 'jerk_rms'):>7} "
              f"{_delta(m, base, 'grip_active_ratio'):>9}   {mix}")

    print("\n说明:")
    print("  L1  = 任务成功  L2  = manner 忠实  both = 双通过")
    print("  align = 平均语言对齐分数（compute_alignment_score，连续[0,1]，1=完全对齐；'-'=该组无约束/无telemetry可打分）")
    print("  数字后 +x.xx / -x.xx 表示与该任务 baseline 的差值（同 seed 集合平均）")
    print("  outcome: both_pass / task_only / manner_only / manner_infeasible / both_fail / error")


def _mean_metrics(rs: list) -> dict:
    keys = ["mean_joint_speed", "peak_joint_speed", "peak_cfrc", "peak_grip_force", "jerk_rms",
            "grip_active_ratio", "finger_sep_hold_mean", "finger_sep_hold_min"]
    out = {}
    for k in keys:
        vs = [r.metrics.get(k) for r in rs if r.metrics.get(k) is not None]
        if vs:
            out[k] = float(np.mean(vs))
    return out


if __name__ == "__main__":
    sys.exit(main())
