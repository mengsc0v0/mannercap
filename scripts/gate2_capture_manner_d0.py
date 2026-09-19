"""Plan or capture a zero-API, same-initial-state D0 manner pilot.

The real execution path reuses the existing Lift / Stack teacher but records
the v2 simulator snapshot, compiled MJB, per-transition actions, controller
side channels, contacts, phase markers, language, and lineage.  Legacy manner
validation is not used for qualification; formal manner and safety remain
unknown until independent thresholds are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from synthesis_v2.d0_manner import (  # noqa: E402
    DEFAULT_TAGS,
    DEFAULT_TASKS,
    PRESETS,
    audit_results,
    build_plan,
    load_results,
    validate_plan,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_finite(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _make_spec(tag: str) -> Any:
    from capx.envs.manner_validator import MannerSpec, PhaseVelocity

    preset = PRESETS[tag]
    phases = {
        name: PhaseVelocity(**bounds)
        for name, bounds in preset["phase_velocity"].items()
    }
    return MannerSpec(
        **preset["global"],
        phase_velocity=phases,
        label=f"d0_{tag}",
    )


def _spec_json(spec: Any) -> dict[str, Any]:
    return {
        "max_joint_vel": spec.max_joint_vel,
        "min_joint_vel": spec.min_joint_vel,
        "kp_max": spec.kp_max,
        "kp_min": spec.kp_min,
        "grip_fraction_max": spec.grip_fraction_max,
        "grip_fraction_min": spec.grip_fraction_min,
        "max_jerk": spec.max_jerk,
        "phase_velocity": {
            name: {
                "max_joint_vel": value.max_joint_vel,
                "min_joint_vel": value.min_joint_vel,
            }
            for name, value in (spec.phase_velocity or {}).items()
        },
        "label": spec.label,
    }


def _phase_metrics(telemetry: list[Any]) -> dict[str, Any]:
    import numpy as np

    grouped: dict[str, list[float]] = {}
    for sample in telemetry:
        speed = float(np.linalg.norm(np.asarray(sample.joint_vel, dtype=np.float64)))
        grouped.setdefault(str(sample.phase), []).append(speed)
    return {
        phase: {
            "sample_count": len(values),
            "mean_joint_speed_rad_s": sum(values) / len(values),
            "peak_joint_speed_rad_s": max(values),
        }
        for phase, values in sorted(grouped.items())
        if values
    }


def _prepare_state_reader(env: Any) -> Any:
    import mujoco
    import numpy as np

    sim = env.robosuite_env.sim
    raw_model = getattr(sim.model, "_model", sim.model)
    kinematic_data = mujoco.MjData(raw_model)

    def state() -> dict[str, Any]:
        data = sim.data
        kinematic_data.qpos[:] = data.qpos
        if kinematic_data.mocap_pos.size:
            kinematic_data.mocap_pos[:] = data.mocap_pos
            kinematic_data.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(raw_model, kinematic_data)
        return {
            "time_s": float(data.time),
            "qpos": np.asarray(data.qpos).tolist(),
            "qvel": np.asarray(data.qvel).tolist(),
            "qacc_warmstart": np.asarray(data.qacc_warmstart).tolist(),
            "act": np.asarray(data.act).tolist(),
            "actuator_ctrl": np.asarray(data.ctrl).tolist(),
            "eef_position_m": np.asarray(kinematic_data.xpos[env.gripper_link_idx]).tolist(),
            "eef_quaternion_wxyz": np.asarray(kinematic_data.xquat[env.gripper_link_idx]).tolist(),
        }

    return state


def _install_physics_counter(env: Any) -> tuple[list[int], dict[str, Any]]:
    sim = env.robosuite_env.sim
    counter = [0]
    originals: dict[str, Any] = {}
    for step_name in ("step", "step2"):
        original = getattr(sim, step_name)
        originals[step_name] = original

        def counted_step(_original=original):
            value = _original()
            counter[0] += 1
            return value

        setattr(sim, step_name, counted_step)
    return counter, originals


def _restore_physics_steps(env: Any, originals: Mapping[str, Any]) -> None:
    sim = env.robosuite_env.sim
    for name, value in originals.items():
        setattr(sim, name, value)


def _create_pair_environment(record: Mapping[str, Any]) -> Any:
    import numpy as np
    import run_manner_ablation as old

    task = str(record["task"])
    seed = int(record["seed"])
    random.seed(seed)
    np.random.seed(seed)
    entry = old.TASK_REGISTRY[task]
    env = old.load_sim_cls(entry["sim_cls_path"])(
        privileged=True,
        enable_render=False,
        max_steps=entry["max_steps"],
        seed=seed,
    )
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    return env


def _run_episode(
    record: Mapping[str, Any],
    output: Path,
    *,
    pair_reference: Path | None = None,
    base_env: Any | None = None,
) -> dict[str, Any]:
    target = output / "episodes" / str(record["episode_id"])
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu_started = time.process_time()
    result: dict[str, Any] = {
        "schema": "mannercap-d0-manner-episode-result-v1",
        **dict(record),
        "status": "started",
        "task_completed": None,
        "api_calls": 0,
        "outer_candidate_attempts": 1,
        "formal_manner_status": "unknown",
        "formal_safety_status": "unknown",
        "promotion_allowed": False,
        "pair_execution_isolation": (
            "posix-fork-copy-on-write" if base_env is not None else "fresh-process"
        ),
    }
    env = base_env
    owns_env = env is None
    counter = [0]
    physics_originals: dict[str, Any] = {}
    original_control_step: Any | None = None
    try:
        import numpy as np
        import run_manner_ablation as old
        from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
        from gate0_smoke import audit_transitions
        from synthesis_v2.mujoco_contacts import (
            capture_object_contacts,
            resolve_task_object_registry,
        )
        from synthesis_v2.simulator_state import (
            capture_controller_command_state,
            capture_simulator_snapshot,
            save_compiled_model_binary,
        )

        task = str(record["task"])
        seed = int(record["seed"])
        entry = old.TASK_REGISTRY[task]
        if env is None:
            env = _create_pair_environment(record)
        state = _prepare_state_reader(env)
        sim = env.robosuite_env.sim
        reference_initial: dict[str, Any] | None = None
        if pair_reference is not None:
            reference_initial = json.loads(
                (pair_reference / "initial_state.json").read_text(encoding="utf-8")
            )
            fresh_state_hash = _canonical_sha256(state())
            reference_state_hash = reference_initial["initial_state_sha256"]
            if fresh_state_hash != reference_state_hash:
                raise RuntimeError(
                    "episode initial state differs from pair "
                    f"reference: {fresh_state_hash} != {reference_state_hash}"
                )
            candidate_model = target / "candidate_model.mjb"
            candidate_hash = save_compiled_model_binary(env, candidate_model)
            candidate_model.unlink()
            reference_hash = reference_initial["compiled_model"]["sha256"]
            if candidate_hash != reference_hash:
                raise RuntimeError(
                    "episode compiled model differs from pair reference: "
                    f"{candidate_hash} != {reference_hash}"
                )
        counter, physics_originals = _install_physics_counter(env)
        initial_state = state()
        initial_state_sha256 = _canonical_sha256(initial_state)
        xml = sim.model.get_xml()
        object_registry = resolve_task_object_registry(env.robosuite_env, task)
        if pair_reference is None:
            (target / "model.xml").write_text(xml, encoding="utf-8")
            compiled_sha256 = save_compiled_model_binary(env, target / "model.mjb")
        else:
            shutil.copy2(pair_reference / "model.xml", target / "model.xml")
            shutil.copy2(pair_reference / "model.mjb", target / "model.mjb")
            compiled_sha256 = hashlib.sha256((target / "model.mjb").read_bytes()).hexdigest()
        _write_json(
            target / "initial_state.json",
            {
                "schema": "mannercap-d0-initial-state-v1",
                "episode_id": record["episode_id"],
                "pair_id": record["pair_id"],
                "state": initial_state,
                "initial_state_sha256": initial_state_sha256,
                "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                "compiled_model": {
                    "path": "model.mjb",
                    "format": "MuJoCo compiled MJB",
                    "sha256": compiled_sha256,
                    "required_for_strict_replay": True,
                },
                "physics_timestep_s": float(sim.model.opt.timestep),
                "eef_pose_derivation": "scratch_mjdata_mj_kinematics_from_qpos",
                "object_registry": object_registry,
                "snapshot": capture_simulator_snapshot(env),
            },
        )
        result["initial_state_sha256"] = initial_state_sha256
        result["compiled_model_sha256"] = compiled_sha256

        spec = _make_spec(str(record["manner_tag"]))
        if "candidate_preset" in record:
            from capx.envs.manner_validator import MannerSpec, PhaseVelocity
            candidate = record["candidate_preset"]
            spec = MannerSpec(**candidate["global"], phase_velocity={
                name: PhaseVelocity(**bounds)
                for name, bounds in candidate["phase_velocity"].items()
            }, label="e0_development_candidate")
        params = old.spec_to_exec_params(spec)
        result["requested_spec"] = _spec_json(spec)
        result["executed_parameters"] = {
            key: value for key, value in params.items() if not key.startswith("_")
        }
        result["parameter_clipping"] = list(params.get("_clipped", []))
        _write_json(target / "executed_parameters.json", _json_finite(result["executed_parameters"]))

        original_control_step = env._do_robosuite_step
        try:
            with (target / "transitions.jsonl").open("x", encoding="utf-8") as stream:

                def record_step(action: Any) -> None:
                    if counter[0] >= record.get("physics_substep_cap", float("inf")):
                        raise RuntimeError("development physics substep budget exhausted")
                    before = state()
                    controller_command = capture_controller_command_state(env)
                    substeps_before = counter[0]
                    original_control_step(action)
                    row = {
                        "state_t": before,
                        "action_t": np.asarray(action).tolist(),
                        "controller_command_t": controller_command,
                        "action_slice": env._ACTION_SLICE,
                        "phase_program_marker": env._manner_phase,
                        "physics_substeps": counter[0] - substeps_before,
                        "state_t_plus_1": state(),
                        "contact_evidence_t_plus_1": capture_object_contacts(sim, object_registry),
                        "task_success_after": bool(env.task_completed()),
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

                env._do_robosuite_step = record_step
                env.begin_manner_recording(clear=True)
                api = FrankaControlPrivilegedApi(env)
                if record.get("ik_backend") == "mujoco-local":
                    from gate0_local_ik import install_local_ik

                    result["ik_diagnostics"] = []
                    install_local_ik(api, env, result["ik_diagnostics"])
                execution = entry["runner"](env, api, params)
                env.end_manner_recording()
        finally:
            env._do_robosuite_step = original_control_step

        telemetry = env.get_manner_telemetry()
        result["execution"] = _json_finite(execution)
        result["task_completed"] = bool(execution.get("task_completed", False))
        result["descriptive_metrics"] = _json_finite(old.extract_metrics(telemetry))
        result["descriptive_phase_metrics"] = _json_finite(_phase_metrics(telemetry))
        result["status"] = "executed"
        result["offline_recomputation"] = audit_transitions(target / "transitions.jsonl")
    except Exception:
        result["status"] = "execution_error"
        result["error"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.perf_counter() - started
        result["cpu_time_s"] = time.process_time() - cpu_started
        result["observed_execution_physics_substeps"] = counter[0]
        _write_json(target / "result.json", _json_finite(result))
        if env is not None and physics_originals:
            _restore_physics_steps(env, physics_originals)
        if env is not None and owns_env:
            env.robosuite_env.close()
    print(json.dumps(_json_finite(result), ensure_ascii=False), flush=True)
    return result


def _write_audit(output: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    audit = audit_results(plan, load_results(output, plan))
    _write_json(output / "audit.json", audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument(
        "--ik-backend",
        choices=("mujoco-local", "pyroki"),
        default="mujoco-local",
    )
    parser.add_argument("--run", action="store_true", help="execute real simulator episodes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = args.seed if args.seed is not None else [42, 43]
    requested = build_plan(args.tasks, seeds, args.tags, ik_backend=args.ik_backend)
    output = args.output.expanduser().resolve()
    plan_path = output / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_plan(plan)
        if plan["plan_sha256"] != requested["plan_sha256"]:
            raise ValueError("arguments do not match the existing immutable D0 plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty when creating a D0 plan")
        output.mkdir(parents=True, exist_ok=True)
        plan = requested
        _write_json(plan_path, plan)
        _write_json(
            output / "run_manifest.json",
            {
                "schema": "mannercap-d0-manner-run-manifest-v1",
                "plan_sha256": plan["plan_sha256"],
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ),
                "api_budget": 0,
                "formal_manner_thresholds_frozen": False,
            },
        )

    if not args.run:
        audit = _write_audit(output, plan)
        print(json.dumps({"output": str(output), "execution_requested": False, **audit}))
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path[:0] = [str(ROOT), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")
    if not hasattr(os, "fork"):
        raise RuntimeError("real D0 capture requires POSIX fork isolation")

    execution_errors = 0
    pair_ids = list(dict.fromkeys(str(item["pair_id"]) for item in plan["episodes"]))
    for pair_id in pair_ids:
        episodes = [item for item in plan["episodes"] if str(item["pair_id"]) == pair_id]
        result_paths = [
            output / "episodes" / str(item["episode_id"]) / "result.json"
            for item in episodes
        ]
        existing = [path.is_file() for path in result_paths]
        if all(existing):
            continue
        if any(existing):
            raise RuntimeError(
                f"partial pair {pair_id} cannot be resumed under fork isolation; "
                "use a new output directory"
            )

        base_env = _create_pair_environment(episodes[0])
        pair_reference: Path | None = None
        try:
            for episode, result_path in zip(episodes, result_paths):
                pid = os.fork()
                if pid == 0:
                    result = _run_episode(
                        episode,
                        output,
                        pair_reference=pair_reference,
                        base_env=base_env,
                    )
                    os._exit(int(result.get("status") == "execution_error"))
                _, status = os.waitpid(pid, 0)
                if not result_path.is_file():
                    raise RuntimeError(
                        f"forked episode {episode['episode_id']} exited without result.json"
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                child_failed = not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0
                execution_errors += int(
                    child_failed or result.get("status") == "execution_error"
                )
                pair_reference = result_path.parent
                _write_audit(output, plan)
        finally:
            base_env.robosuite_env.close()
    audit = _write_audit(output, plan)
    print(json.dumps({"output": str(output), "execution_requested": True, **audit}))
    return 1 if execution_errors or audit["missing_episode_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
