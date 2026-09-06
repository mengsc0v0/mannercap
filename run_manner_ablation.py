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
from dataclasses import dataclass, asdict
from typing import Any, Callable

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "stagecap_robosuite_libero_repro_20260714")
sys.path.insert(0, BASE)

import numpy as np


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
]

BASELINE_TAG = "baseline"


# ──────────────────────────────────────────────────────────────────────────────
# MannerSpec → 执行参数（Strategy A: clip 硬性物理下限）
# ──────────────────────────────────────────────────────────────────────────────
GRIP_FLOOR = 0.60  # 实测诊断：force<0.5 时夹爪几乎不闭合，需>=0.62 才能稳定抬起 5cm cube
V_MIN_DEFAULT = 0.02
KP_FLOOR = 250.0


def spec_to_exec_params(spec) -> dict:
    clipped: list[str] = []
    clipped_dims: list[str] = []

    v_default = 0.12
    if spec.max_joint_vel is not None and spec.min_joint_vel is not None:
        v_max = (spec.max_joint_vel * 0.85 + spec.min_joint_vel * 1.05) / 2
    elif spec.max_joint_vel is not None:
        v_max = spec.max_joint_vel * 0.85
    elif spec.min_joint_vel is not None:
        v_max = spec.min_joint_vel * 1.10
    else:
        v_max = v_default
    if v_max < V_MIN_DEFAULT:
        clipped.append(f"v_max({v_max:.3f}<{V_MIN_DEFAULT})")
        clipped_dims.append("vel")
        v_max = V_MIN_DEFAULT
    a_max = max(0.01, v_max * 0.7)

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
        "_clipped":      clipped,
        "_clipped_dims": clipped_dims,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 任务执行器
# ──────────────────────────────────────────────────────────────────────────────
def _pump(env, n: int) -> None:
    for _ in range(n):
        env._step_once()


def _grasp_close(env, api, force: float, steps: int = 50) -> None:
    """低力目标(<0.70)时分多次收紧-回稳，而非一次到位（灵活抓取策略）。

    低夹持力单次闭合常因接触面小、切向摩擦不足而打滑。分 3 次收紧-回稳，
    在不抬高峰值夹持力（不违反 Manner C 上界，L2 仍合规）的前提下，让指尖
    多次重新对齐接触面，提升抓取成功率（L1）。
    """
    if force < 0.70:
        n_tries = 3
        per_steps = max(15, steps // n_tries)
        for _ in range(n_tries):
            api.close_gripper_partial(force=force, steps=per_steps)
            _pump(env, 4)
    else:
        api.close_gripper_partial(force=force, steps=steps)


def run_cube_lift(env, api, params: dict) -> dict:
    cube_pos, _, _ = api.get_object_pose("red cube")
    grasp_quat = np.array([0.0, 0.0, 1.0, 0.0])

    approach_pos = cube_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 8)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    grasp_pos = cube_pos.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, grasp_quat,
        v_max=max(0.02, params["v_max"] * 0.6),
        a_max=max(0.01, params["a_max"] * 0.5),
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 8)

    _grasp_close(env, api, params["grip"])

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.18
    api.goto_pose_with_profile(
        lift_pos, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
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

    approach_pos = red_pos.copy(); approach_pos[2] += 0.20
    api.goto_pose_with_profile(
        approach_pos, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 6)

    api.set_impedance_stiffness(kp_pos=params["kp_pos"])

    grasp_pos = red_pos.copy(); grasp_pos[2] -= 0.005
    api.goto_pose_with_profile(
        grasp_pos, grasp_quat,
        v_max=max(0.02, params["v_max"] * 0.6),
        a_max=max(0.01, params["a_max"] * 0.5),
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 6)

    _grasp_close(env, api, params["grip"])

    api.set_impedance_stiffness(kp_pos=1000.0)
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    api.goto_pose_with_profile(
        lift_pos, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 6)

    over_green = green_pos.copy(); over_green[2] = lift_pos[2]
    api.goto_pose_with_profile(
        over_green, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 6)

    # green top = green_z + 0.025; red half-side = 0.025; small gap 0.008
    stack_pos = green_pos.copy(); stack_pos[2] = green_pos[2] + 0.058
    api.goto_pose_with_profile(
        stack_pos, grasp_quat,
        v_max=max(0.02, params["v_max"] * 0.5),
        a_max=max(0.01, params["a_max"] * 0.4),
        jerk_limit=params["jerk_limit"],
    )
    _pump(env, 14)

    api.open_gripper()
    _pump(env, 15)

    retreat_pos = stack_pos.copy(); retreat_pos[2] += 0.20
    api.goto_pose_with_profile(
        retreat_pos, grasp_quat,
        v_max=params["v_max"], a_max=params["a_max"],
        jerk_limit=params["jerk_limit"],
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
    from capx.envs.manner_validator import MannerSpec
    _mocks = {
        "gentle":        MannerSpec(max_joint_vel=0.04, kp_max=400.0, grip_fraction_max=0.30, max_jerk=0.8, label="mock_gentle"),
        "ultra_careful": MannerSpec(max_joint_vel=0.02, kp_max=300.0, grip_fraction_max=0.20, max_jerk=0.5, label="mock_ultra"),
        "ultra_gentle":  MannerSpec(max_joint_vel=0.02, kp_max=280.0, grip_fraction_max=0.15, max_jerk=0.5, label="mock_baby"),
        "normal":        MannerSpec(label="mock_normal"),
        "fast":          MannerSpec(min_joint_vel=0.30, label="mock_fast"),
        "firm":          MannerSpec(kp_min=1200.0, grip_fraction_min=0.85, label="mock_firm"),
        "fast_then_slow":MannerSpec(max_joint_vel=0.06, kp_max=500.0, max_jerk=2.0, label="mock_mixed"),
        "gentle_en":     MannerSpec(max_joint_vel=0.04, kp_max=380.0, grip_fraction_max=0.30, label="mock_gentle_en"),
        "fast_en":       MannerSpec(min_joint_vel=0.28, label="mock_fast_en"),
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
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_single_trial(task, tag, prompt, seed, spec, args,
                     env, api, validator, StageStatus, runner):
    t0 = time.time()
    params = spec_to_exec_params(spec)
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

    try:
        env.reset(seed=seed)
        env.begin_manner_recording(clear=True)
        exec_result = runner(env, api, params)
        env.end_manner_recording()
    except Exception as e:
        emsg = f"{type(e).__name__}: {e}"
        outcome = classify_outcome(None, None, error_msg=emsg)
        return TrialResult(
            task=task, tag=tag, seed=seed, prompt=prompt, label=spec.label,
            spec_json=spec_json,
            params={k: v for k, v in params.items() if k not in ("_clipped", "_clipped_dims")},
            clipped=params["_clipped"],
            task_ok=None, manner_ok=None, outcome=outcome,
            reward=None, exec_result={}, metrics={},
            error=f"{emsg}\n{traceback.format_exc(limit=4)}",
            exec_time_s=time.time() - t0,
        )

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
    clipped_dims = params.get("_clipped_dims", [])
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

    return TrialResult(
        task=task, tag=tag, seed=seed, prompt=prompt, label=spec.label,
        spec_json=spec_json,
        params={k: v for k, v in params.items() if k not in ("_clipped", "_clipped_dims")},
        clipped=params["_clipped"],
        task_ok=task_ok, manner_ok=manner_ok,
        outcome=classify_outcome(task_ok, manner_ok),
        reward=reward, exec_result=exec_result, metrics=metrics,
        error=None, exec_time_s=time.time() - t0,
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

    validator = MannerValidator()
    all_results: list[TrialResult] = []

    for task in args.tasks:
        entry = TASK_REGISTRY[task]
        sim_cls = load_sim_cls(entry["sim_cls_path"])
        runner = entry["runner"]

        print(f"\n{'=' * 72}\n  TASK = {task}\n{'=' * 72}")
        env = sim_cls(
            privileged=True, enable_render=False,
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
              f"{_delta(m, base, 'mean_joint_speed'):>8} "
              f"{_delta(m, base, 'peak_cfrc'):>8} "
              f"{_delta(m, base, 'peak_grip_force'):>8} "
              f"{_delta(m, base, 'jerk_rms'):>7} "
              f"{_delta(m, base, 'grip_active_ratio'):>9}   {mix}")

    print("\n说明:")
    print("  L1  = 任务成功  L2  = manner 忠实  both = 双通过")
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
