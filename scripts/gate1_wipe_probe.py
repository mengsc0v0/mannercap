"""Probe the RoboSuite Wipe task and run one bounded local-IK attempt.

The probe is deliberately independent of any LLM.  It records the benchmark's
own marker progress and success predicate, named MuJoCo contacts, and a compact
control trace.  A successful task attempt is evidence that the third task is
operable; it is not force-sensor calibration or permission to promote Wipe into
the supported task set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def marker_status(robosuite_env: Any) -> dict[str, Any]:
    total = int(robosuite_env.num_markers)
    wiped = int(len(robosuite_env.wiped_markers))
    return {
        "total_markers": total,
        "wiped_markers": wiped,
        "remaining_markers": total - wiped,
        "wiped_fraction": float(wiped / max(total, 1)),
        "reward": float(robosuite_env.reward()),
        "task_completed": bool(robosuite_env._check_success()),
    }


def named_contacts(sim: Any) -> list[dict[str, Any]]:
    import mujoco
    import numpy as np

    mujoco.mj_rnePostConstraint(sim.model._model, sim.data._data)
    rows: list[dict[str, Any]] = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, index, wrench)
        rows.append(
            {
                "contact_id": index,
                "geom1": sim.model.geom_id2name(geom1_id) or "",
                "geom2": sim.model.geom_id2name(geom2_id) or "",
                "body1": sim.model.body_id2name(int(sim.model.geom_bodyid[geom1_id])) or "",
                "body2": sim.model.body_id2name(int(sim.model.geom_bodyid[geom2_id])) or "",
                "distance_m": float(contact.dist),
                "position_world_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "force_contact_local_n": np.asarray(wrench[:3], dtype=np.float64).tolist(),
                "torque_contact_local_nm": np.asarray(wrench[3:], dtype=np.float64).tolist(),
                "force_norm_n": float(np.linalg.norm(wrench[:3])),
            }
        )
    return rows


def marker_registry(robosuite_env: Any) -> list[dict[str, Any]]:
    registry = []
    for index, marker in enumerate(robosuite_env.model.mujoco_arena.markers):
        body_name = str(marker.root_body)
        registry.append(
            {
                "marker_index": index,
                "body_name": body_name,
                "body_id": int(robosuite_env.sim.model.body_name2id(body_name)),
                "contact_geoms": [str(name) for name in getattr(marker, "contact_geoms", ())],
                "position_world_m": [
                    float(value)
                    for value in robosuite_env.sim.data.body_xpos[
                        robosuite_env.sim.model.body_name2id(body_name)
                    ]
                ],
            }
        )
    return registry


def inspect_task_attributes(robosuite_env: Any) -> dict[str, Any]:
    names = [
        name
        for name in dir(robosuite_env)
        if any(token in name.lower() for token in ("wipe", "marker", "contact"))
        and not name.startswith("__")
    ]
    simple: dict[str, Any] = {}
    for name in sorted(names):
        try:
            value = getattr(robosuite_env, name)
        except Exception:
            continue
        if isinstance(value, (str, bool, int, float)) or value is None:
            simple[name] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 100:
            converted = []
            for item in value:
                if isinstance(item, (str, bool, int, float)) or item is None:
                    converted.append(item)
                else:
                    converted.append(type(item).__name__)
            simple[name] = converted
        else:
            simple[name] = {"type": type(value).__name__}
    return simple


def tool_state(env: Any) -> dict[str, Any]:
    import numpy as np

    robosuite_env = env.robosuite_env
    sim = robosuite_env.sim
    arm = robosuite_env.robots[0].arms[0]
    important = robosuite_env.robots[0].gripper[arm].important_geoms
    geometry_positions: dict[str, list[dict[str, Any]]] = {}
    for group, names in important.items():
        geometry_positions[str(group)] = []
        for name in names:
            geom_id = int(sim.model.geom_name2id(name))
            geometry_positions[str(group)].append(
                {
                    "name": str(name),
                    "geom_id": geom_id,
                    "position_world_m": np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).tolist(),
                }
            )
    cartesian = np.asarray(env.get_observation()["robot_cartesian_pos"], dtype=np.float64)
    return {
        "arm": str(arm),
        "eef_position_world_m": np.asarray(robosuite_env._get_eef_xpos(arm), dtype=np.float64).tolist(),
        "wrapper_tcp_position_robot_base_m": cartesian[:3].tolist(),
        "important_geometries": geometry_positions,
    }


def eef_wrench_state(robosuite_env: Any) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    robot = robosuite_env.robots[0]
    for arm in robot.arms:
        values = np.asarray(robot.recent_ee_forcetorques[arm].current, dtype=np.float64)
        rows.append({
            "arm": str(arm),
            "force_n": values[:3].tolist(),
            "torque_nm": values[3:6].tolist(),
            "force_norm_n": float(np.linalg.norm(values[:3])),
            "torque_norm_nm": float(np.linalg.norm(values[3:6])),
            "qualification": "uncalibrated_robosuite_sensor",
        })
    return rows


def run_probe(
    output: Path,
    seed: int,
    execute_wipe: bool,
    wipe_attempts: int,
    surface_penetration_m: float,
) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")

    import numpy as np
    from scipy.spatial.transform import Rotation

    from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
    from capx.integrations.base_api import ApiBase
    from capx.integrations.franka.robosuite_stage import FrankaRobosuiteSpillWipeStageApi
    from capx.integrations.franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi

    started = time.perf_counter()
    env = None
    trace_stream = None
    original_goto = FrankaControlSpillWipePrivilegedApi.goto_pose
    result: dict[str, Any] = {
        "schema": "mannercap-gate1-wipe-probe-v1",
        "task": "wipe",
        "seed": seed,
        "llm_calls": 0,
        "execute_wipe_requested": execute_wipe,
        "surface_penetration_m": surface_penetration_m,
        "status": "started",
        "qualification": "candidate",
        "promotion_allowed": False,
    }
    try:
        np.random.seed(seed)
        env = FrankaRobosuiteSpillWipeLowLevel(
            privileged=True,
            enable_render=False,
            max_steps=4000,
            seed=seed,
        )
        # RoboSuite's Wipe placement sampler uses NumPy's global RNG during
        # reset; the wrapper seed currently covers only its private generator.
        np.random.seed(seed)
        _obs, reset_info = env.reset(seed=seed)
        robosuite_env = env.robosuite_env
        sim = robosuite_env.sim
        xml = sim.model.get_xml()
        (output / "model.xml").write_text(xml, encoding="utf-8")
        low, high = robosuite_env.action_spec
        markers = marker_registry(robosuite_env)
        initial = {
            "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "physics_timestep_s": float(sim.model.opt.timestep),
            "state": {
                "time_s": float(sim.data.time),
                "qpos": np.asarray(sim.data.qpos, dtype=np.float64).tolist(),
                "qvel": np.asarray(sim.data.qvel, dtype=np.float64).tolist(),
                "actuator_ctrl": np.asarray(sim.data.ctrl, dtype=np.float64).tolist(),
            },
            "reset_info": reset_info,
            "action_spec": {
                "low": np.asarray(low, dtype=np.float64).tolist(),
                "high": np.asarray(high, dtype=np.float64).tolist(),
                "robosuite_action_dimension": int(np.asarray(low).size),
                "wrapper_action_slice": int(env._ACTION_SLICE),
                "wrapper_built_action_dimension": int(env._build_action().size),
            },
            "marker_registry": markers,
            "marker_status": marker_status(robosuite_env),
            "task_attributes": inspect_task_attributes(robosuite_env),
            "tool_state": tool_state(env),
            "named_contacts": named_contacts(sim),
        }
        write_json(output / "initial_state.json", initial)

        trace_stream = (output / "transitions.jsonl").open("x", encoding="utf-8")
        original_step = env._do_robosuite_step
        transition_count = 0
        max_contact_force_n = 0.0
        max_eef_force_n = 0.0
        max_eef_torque_nm = 0.0
        unique_pairs: set[tuple[str, str]] = set()

        def recording_step(action: Any) -> None:
            nonlocal transition_count, max_contact_force_n, max_eef_force_n, max_eef_torque_nm
            before = marker_status(robosuite_env)
            time_before = float(sim.data.time)
            original_step(action)
            contacts = named_contacts(sim)
            eef_wrenches = eef_wrench_state(robosuite_env)
            for contact in contacts:
                max_contact_force_n = max(max_contact_force_n, contact["force_norm_n"])
                unique_pairs.add(tuple(sorted((contact["geom1"], contact["geom2"]))))
            for wrench in eef_wrenches:
                max_eef_force_n = max(max_eef_force_n, wrench["force_norm_n"])
                max_eef_torque_nm = max(max_eef_torque_nm, wrench["torque_norm_nm"])
            transition_count += 1
            trace_stream.write(
                json.dumps(
                    {
                        "transition": transition_count,
                        "time_before_s": time_before,
                        "time_after_s": float(sim.data.time),
                        "action": np.asarray(action, dtype=np.float64).tolist(),
                        "markers_before": before,
                        "markers_after": marker_status(robosuite_env),
                        "contacts": contacts,
                        "eef_wrenches": eef_wrenches,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

        env._do_robosuite_step = recording_step
        env._step_once()
        result["hold_control"] = {
            "executed": True,
            "marker_status": marker_status(robosuite_env),
            "contacts": named_contacts(sim),
        }

        if execute_wipe:
            api = FrankaRobosuiteSpillWipeStageApi.__new__(FrankaRobosuiteSpillWipeStageApi)
            ApiBase.__init__(api, env)
            api._TCP_OFFSET = np.array([0.0, 0.0, -0.0158], dtype=np.float64)
            api.cfg = None
            ik_diagnostics: list[dict[str, Any]] = []

            import mujoco
            from scipy.optimize import least_squares
            from types import MethodType

            model = sim.model._model
            scratch = mujoco.MjData(model)
            joints = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")
                for i in range(1, 8)
            ]
            if any(joint < 0 for joint in joints):
                raise ValueError("Wipe local IK requires the seven named Panda joints")
            qadr = model.jnt_qposadr[joints]
            lower, upper = model.jnt_range[joints].T
            eef = env.gripper_link_idx
            base = env.base_link_idx
            arm = robosuite_env.robots[0].arms[0]
            corner_names = robosuite_env.robots[0].gripper[arm].important_geoms["corners"]
            corner_ids = np.asarray([sim.model.geom_name2id(name) for name in corner_names], dtype=int)

            def goto_tool_center(
                self: Any,
                position: Any,
                quaternion_wxyz: Any,
                *,
                v_max: float = 0.12,
                a_max: float = 0.10,
                phase: str = "wipe_probe",
            ) -> None:
                q_before = sim.data.qpos.copy()
                v_before = sim.data.qvel.copy()
                time_before = float(sim.data.time)
                checked_fields = ("qpos", "qvel", "ctrl", "act", "qacc", "mocap_pos", "mocap_quat")
                state_before = {
                    name: np.array(getattr(sim.data, name), copy=True) for name in checked_fields
                }
                scratch.qpos[:] = q_before
                scratch.qvel[:] = v_before
                base_rotation = Rotation.from_matrix(sim.data.xmat[base].reshape(3, 3))
                target_position = sim.data.xpos[base] + base_rotation.apply(
                    np.asarray(position, dtype=np.float64).reshape(3)
                )
                quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
                target_rotation = base_rotation * Rotation.from_quat(quat[[1, 2, 3, 0]])
                q0 = np.clip(q_before[qadr], lower + 1e-6, upper - 1e-6)

                def residual(q: Any) -> Any:
                    scratch.qpos[qadr] = q
                    mujoco.mj_forward(model, scratch)
                    center = np.asarray(scratch.geom_xpos[corner_ids], dtype=np.float64).mean(axis=0)
                    rotation_error = (
                        target_rotation
                        * Rotation.from_matrix(scratch.xmat[eef].reshape(3, 3)).inv()
                    ).as_rotvec()
                    return np.r_[center - target_position, 0.2 * rotation_error, 1e-4 * (q - q0)]

                solved = least_squares(
                    residual,
                    q0,
                    bounds=(lower + 1e-6, upper - 1e-6),
                    max_nfev=300,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                error = residual(solved.x)
                pos_error = float(np.linalg.norm(error[:3]))
                rot_error = float(np.linalg.norm(error[3:6]) / 0.2)
                untouched = (
                    all(
                        np.array_equal(value, getattr(sim.data, name))
                        for name, value in state_before.items()
                    )
                    and time_before == float(sim.data.time)
                )
                diagnostic = {
                    "phase": phase,
                    "position_error_m": pos_error,
                    "orientation_error_rad": rot_error,
                    "nfev": int(solved.nfev),
                    "checked_live_state_unchanged_during_ik": untouched,
                    "target_geometry": "mean_wiping_corner_position",
                    "controller_snapshot_verified": False,
                }
                ik_diagnostics.append(diagnostic)
                if not untouched or pos_error > 0.002 or rot_error > 0.02:
                    raise RuntimeError(
                        f"Wipe local IK failed: position={pos_error}, rotation={rot_error}, "
                        f"untouched={untouched}"
                    )
                env.move_to_joints_profiled(
                    solved.x,
                    v_max=v_max,
                    a_max=a_max,
                    phase=phase,
                )
                actual_center = np.asarray(sim.data.geom_xpos[corner_ids], dtype=np.float64).mean(axis=0)
                diagnostic["post_motion_tool_center_error_m"] = float(
                    np.linalg.norm(actual_center - target_position)
                )
                diagnostic["post_motion_orientation_error_rad"] = float(
                    np.linalg.norm(
                        (
                            target_rotation
                            * Rotation.from_matrix(sim.data.xmat[eef].reshape(3, 3)).inv()
                        ).as_rotvec()
                    )
                )
                if diagnostic["post_motion_tool_center_error_m"] > 0.005:
                    raise RuntimeError("Controller did not reach the Wipe tool-center target")

            api.goto_pose_with_profile = MethodType(goto_tool_center, api)

            def local_goto(self: Any, position: Any, quaternion_wxyz: Any) -> None:
                target = np.asarray(position, dtype=np.float64).reshape(3).copy()
                # RoboSuite recognizes a marker only when it lies strictly
                # above the wiping-corner plane (0 < distance < 2 cm).  Put
                # surface targets slightly below the marker centers; leave
                # approach/lift waypoints unchanged.
                if target[2] < 0.0:
                    target[2] -= surface_penetration_m
                self.goto_pose_with_profile(
                    target,
                    np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4),
                    v_max=0.12,
                    a_max=0.10,
                    phase="wipe_probe",
                )

            FrankaControlSpillWipePrivilegedApi.goto_pose = local_goto
            attempt_results = []
            try:
                attempt_results.append(api.approach_spill_surface(stage_attempt=1))
                for attempt in range(1, wipe_attempts + 1):
                    if env.task_completed():
                        break
                    attempt_results.append(api.wipe_spill_and_verify(stage_attempt=attempt))
            except Exception:
                attempt_results.append(
                    {
                        "skill": "bounded_wipe_probe",
                        "success": False,
                        "error": traceback.format_exc(),
                        "marker_status": marker_status(robosuite_env),
                    }
                )
            result["wipe_attempt"] = {
                "backend": "mujoco_local_ik_development_only",
                "attempt_results": attempt_results,
                "ik_diagnostics": ik_diagnostics,
                "final_marker_status": marker_status(robosuite_env),
                "final_tool_state": tool_state(env),
            }

        result["status"] = "executed"
        result["task_operable"] = bool(
            result["hold_control"]["executed"]
            and (
                not execute_wipe
                or result.get("wipe_attempt", {}).get("final_marker_status", {}).get("task_completed")
            )
        )
        result["promotion_blockers"] = [
            "wipe force/contact channels are not calibrated",
            "wipe constraints and thresholds are not approved in the public registry",
        ]
        if execute_wipe and not result["task_operable"]:
            result["promotion_blockers"].append("bounded task execution did not reach benchmark success")
        result["trace_summary"] = {
            "control_transitions": transition_count,
            "max_named_contact_force_n": max_contact_force_n,
            "max_eef_force_n": max_eef_force_n,
            "max_eef_torque_nm": max_eef_torque_nm,
            "eef_wrench_qualification": "uncalibrated_robosuite_sensor",
            "unique_geom_pairs": [list(pair) for pair in sorted(unique_pairs)],
        }
    except Exception:
        result["status"] = "execution_error"
        result["task_operable"] = False
        result["error"] = traceback.format_exc()
    finally:
        FrankaControlSpillWipePrivilegedApi.goto_pose = original_goto
        if trace_stream is not None:
            trace_stream.close()
        if env is not None:
            env.robosuite_env.close()
        result["wall_s"] = time.perf_counter() - started
        write_json(output / "probe.json", result)
    print(json.dumps({
        "status": result.get("status"),
        "task_operable": result.get("task_operable"),
        "trace_summary": result.get("trace_summary"),
        "final_marker_status": result.get("wipe_attempt", {}).get("final_marker_status"),
        "promotion_blockers": result.get("promotion_blockers"),
        "output": str(output),
    }, ensure_ascii=False, allow_nan=False, default=json_default), flush=True)
    return bool(result.get("status") == "executed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute-wipe", action="store_true")
    parser.add_argument("--wipe-attempts", type=int, default=3)
    parser.add_argument("--surface-penetration-m", type=float, default=0.002)
    args = parser.parse_args()
    if args.wipe_attempts < 0:
        parser.error("--wipe-attempts must be non-negative")
    if not 0.0 <= args.surface_penetration_m < 0.02:
        parser.error("--surface-penetration-m must be in [0, 0.02)")
    return 0 if run_probe(
        args.output.resolve(),
        args.seed,
        args.execute_wipe,
        args.wipe_attempts,
        args.surface_penetration_m,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
