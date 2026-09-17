"""Run an independent Wipe MuJoCo controlled-load channel probe.

The probe applies known world-frame force/torque levels directly through
``sim.data.xfrc_applied`` to the Wipe wiping-gripper body.  It records paired
no-load/load windows from both the raw MuJoCo ``sensordata`` sensors and
RoboSuite's ``recent_ee_forcetorques`` channel.  It does not call a Wipe skill
or IK method.  Each sample uses the wrapper's normal hold-controller step so
the robot remains stable and RoboSuite refreshes the same recent channel used
by the Wipe probe.

The output is a simulator sensor-channel response audit.  It intentionally
does not claim a contact-model calibration, a real-hardware calibration, or a
Wipe safety-threshold approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis_v2.wipe_controlled_load import (  # noqa: E402
    RAW_CHANNEL,
    RECENT_CHANNEL,
    ControlledLoadSpec,
    Decision,
    audit_controlled_load_suite,
)


def json_default(value: Any) -> Any:
    """Convert NumPy scalar/array values without importing NumPy at import time."""

    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def _vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must be a finite 3-vector")
    return values  # type: ignore[return-value]


def _unit_vector(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = _vector3(value, name=name)
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _validate_levels(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    levels = tuple(float(value) for value in values)
    if not levels or any(not math.isfinite(value) or value <= 0.0 for value in levels):
        raise ValueError(f"{name} must contain positive finite values")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return levels


def _model_name(model: Any, kind: Any, object_id: int) -> str:
    try:
        name = model.id2name(object_id, kind)
    except Exception:
        name = None
    return str(name or "")


def _resolve_id(model: Any, method_name: str, name: str) -> int:
    method = getattr(model, method_name, None)
    if method is None:
        raise RuntimeError(f"model has no {method_name} lookup")
    object_id = int(method(name))
    if object_id < 0:
        raise ValueError(f"model lookup failed for {method_name}({name!r})")
    return object_id


def resolve_binding(
    sim: Any,
    *,
    body_name: str,
    site_name: str,
    force_sensor_name: str,
    torque_sensor_name: str,
    expected_body_id: int | None,
    mujoco: Any,
) -> dict[str, Any]:
    """Resolve and record object/body/geom/site/sensor identity from the model."""

    model = sim.model
    body_id = _resolve_id(model, "body_name2id", body_name)
    if body_id == 0:
        raise ValueError("world body cannot be a controlled-load target")
    if expected_body_id is not None and body_id != expected_body_id:
        raise ValueError(
            f"Wipe model body id changed: {body_name!r} resolved to {body_id}, expected {expected_body_id}"
        )
    site_id = _resolve_id(model, "site_name2id", site_name)
    force_sensor_id = _resolve_id(model, "sensor_name2id", force_sensor_name)
    torque_sensor_id = _resolve_id(model, "sensor_name2id", torque_sensor_name)

    force_dim = int(model.sensor_dim[force_sensor_id])
    torque_dim = int(model.sensor_dim[torque_sensor_id])
    if force_dim != 3 or torque_dim != 3:
        raise ValueError(f"Wipe force/torque sensor dimensions must be 3, got {force_dim}/{torque_dim}")
    force_objid = int(model.sensor_objid[force_sensor_id])
    torque_objid = int(model.sensor_objid[torque_sensor_id])
    if force_objid != site_id or torque_objid != site_id:
        raise ValueError(
            "force/torque sensors must both be attached to the declared Wipe FT site "
            f"{site_name!r}, got objids {force_objid}/{torque_objid} vs {site_id}"
        )

    geom_ids: list[int] = []
    geom_names: list[str] = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) == body_id:
            geom_ids.append(geom_id)
            try:
                geom_names.append(str(model.geom_id2name(geom_id) or ""))
            except Exception:
                geom_names.append("")
    if not geom_ids or any(not name for name in geom_names):
        raise ValueError(f"target body {body_name!r} has no complete geom name binding")

    # Sensor type is recorded for auditability.  The exact enum check is kept
    # diagnostic because mujoco Python versions expose enum values differently.
    sensor_type_names: dict[str, str] = {}
    for label, sensor_id in (("force", force_sensor_id), ("torque", torque_sensor_id)):
        try:
            sensor_type_names[label] = str(model.sensor_type[sensor_id])
        except Exception:
            sensor_type_names[label] = "unknown"

    return {
        "body": {
            "id": body_id,
            "name": body_name,
            "geom_ids": geom_ids,
            "geom_names": geom_names,
        },
        "sensor": {
            "site_id": site_id,
            "site_name": site_name,
            "force_sensor_id": force_sensor_id,
            "force_sensor_name": force_sensor_name,
            "force_sensor_dim": force_dim,
            "torque_sensor_id": torque_sensor_id,
            "torque_sensor_name": torque_sensor_name,
            "torque_sensor_dim": torque_dim,
            "sensor_type": sensor_type_names,
            "recent_channel": {
                "source": "robot.recent_ee_forcetorques",
                "arm": "right",
                "layout": "force_n_then_torque_nm",
                "qualification": "separate_filtered_channel",
            },
        },
        "application": {
            "api": "sim.data.xfrc_applied",
            "body_id": body_id,
            "frame": "world",
            "point": "target_body_com",
            "layout": "force_world_n_then_torque_world_nm",
        },
    }


def _capture_state(data: Any, np: Any) -> dict[str, Any]:
    fields = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qfrc_applied")
    captured: dict[str, Any] = {name: np.array(getattr(data, name), copy=True) for name in fields if hasattr(data, name)}
    captured["time"] = float(data.time)
    return captured


def _restore_state(data: Any, snapshot: Mapping[str, Any], np: Any) -> None:
    for name, value in snapshot.items():
        if name == "time":
            data.time = float(value)
        elif name != "xfrc_applied" and hasattr(data, name):
            target = getattr(data, name)
            target[...] = value
    if hasattr(data, "xfrc_applied"):
        data.xfrc_applied[...] = 0.0


def _state_equal(data: Any, snapshot: Mapping[str, Any], np: Any) -> bool:
    if float(data.time) != float(snapshot.get("time")):
        return False
    for name, value in snapshot.items():
        if name == "time" or not hasattr(data, name):
            continue
        if not np.array_equal(getattr(data, name), value):
            return False
    return True


def _contact_rows(sim: Any, body_id: int, mujoco: Any, np: Any) -> list[dict[str, Any]]:
    """Return contacts involving the selected body with explicit pair identity."""

    model = sim.model
    data = sim.data
    rows: list[dict[str, Any]] = []
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        if body_id not in (body1_id, body2_id):
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model._model, data._data, contact_id, wrench)
        rows.append(
            {
                "contact_id": contact_id,
                "geom1_id": geom1_id,
                "geom1_name": str(model.geom_id2name(geom1_id) or ""),
                "body1_id": body1_id,
                "body1_name": str(model.body_id2name(body1_id) or ""),
                "geom2_id": geom2_id,
                "geom2_name": str(model.geom_id2name(geom2_id) or ""),
                "body2_id": body2_id,
                "body2_name": str(model.body_id2name(body2_id) or ""),
                "distance_m": float(contact.dist),
                "position_world_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "force_contact_local_n": np.asarray(wrench[:3], dtype=np.float64).tolist(),
                "torque_contact_local_nm": np.asarray(wrench[3:], dtype=np.float64).tolist(),
                "force_norm_n": float(np.linalg.norm(wrench[:3])),
            }
        )
    return rows


def _sensor_sample(
    sim: Any,
    *,
    binding: Mapping[str, Any],
    robot: Any,
    arm: str,
    body_id: int,
    mujoco: Any,
    np: Any,
) -> dict[str, Any]:
    """Read raw and recent channels at one actual simulator timestamp."""

    model = sim.model
    data = sim.data
    sensor = binding["sensor"]
    force_id = int(sensor["force_sensor_id"])
    torque_id = int(sensor["torque_sensor_id"])
    force_adr = int(model.sensor_adr[force_id])
    torque_adr = int(model.sensor_adr[torque_id])
    raw_force = np.asarray(data.sensordata[force_adr : force_adr + 3], dtype=np.float64).copy()
    raw_torque = np.asarray(data.sensordata[torque_adr : torque_adr + 3], dtype=np.float64).copy()
    recent_values = np.asarray(robot.recent_ee_forcetorques[arm].current, dtype=np.float64).reshape(-1)
    if recent_values.size < 6:
        raise ValueError(f"recent_ee_forcetorques[{arm!r}] must expose force+torque six-vector")
    recent_force = recent_values[:3].copy()
    recent_torque = recent_values[3:6].copy()
    rotation = np.asarray(data.site_xmat[int(sensor["site_id"])], dtype=np.float64).reshape(3, 3).copy()
    contacts = _contact_rows(sim, body_id, mujoco, np)
    mujoco.mj_rnePostConstraint(model._model, data._data)
    cfrc = np.asarray(data.cfrc_ext[body_id], dtype=np.float64).reshape(6).copy()
    return {
        "timestamp_s": float(data.time),
        "reset_id": "",  # filled by run_window so one identifier is shared across paired windows
        "sensor_valid": bool(np.all(np.isfinite(raw_force)) and np.all(np.isfinite(raw_torque)) and np.all(np.isfinite(recent_values[:6]))),
        "site_rotation_world": rotation.tolist(),
        "channels": {
            RAW_CHANNEL: {
                "force_n": raw_force.tolist(),
                "torque_nm": raw_torque.tolist(),
            },
            RECENT_CHANNEL: {
                "force_n": recent_force.tolist(),
                "torque_nm": recent_torque.tolist(),
            },
        },
        "selected_body_contact_count": len(contacts),
        "selected_body_contacts": contacts,
        "diagnostic_cfrc_ext": {
            "layout": "torque_nm_then_force_n",
            "force_n": cfrc[3:].tolist(),
            "torque_nm": cfrc[:3].tolist(),
            "qualification": "diagnostic_only_not_sensor_calibration",
        },
    }


def _run_window(
    env: Any,
    sim: Any,
    *,
    binding: Mapping[str, Any],
    robot: Any,
    arm: str,
    body_id: int,
    force_world: Sequence[float],
    torque_world: Sequence[float],
    steps: int,
    window: int,
    reset_id: str,
    mujoco: Any,
    np: Any,
) -> list[dict[str, Any]]:
    """Advance through the normal hold controller and retain measured samples."""

    if steps < 1 or window < 1 or window > steps:
        raise ValueError("steps/window must satisfy 1 <= window <= steps")
    data = sim.data
    model = sim.model
    data.xfrc_applied[...] = 0.0
    data.xfrc_applied[body_id, :3] = np.asarray(force_world, dtype=np.float64)
    data.xfrc_applied[body_id, 3:] = np.asarray(torque_world, dtype=np.float64)
    samples: list[dict[str, Any]] = []
    for index in range(steps):
        # RoboSuite updates recent_ee_forcetorques only on its normal control
        # path.  Holding the saved joint target keeps the robot stable while
        # xfrc_applied remains the sole changed physical input.
        env._step_once()
        if index >= steps - window:
            sample = _sensor_sample(
                sim,
                binding=binding,
                robot=robot,
                arm=arm,
                body_id=body_id,
                mujoco=mujoco,
                np=np,
            )
            sample["reset_id"] = reset_id
            samples.append(sample)
    data.xfrc_applied[...] = 0.0
    return samples


def _case_record(
    *,
    case_id: str,
    case_type: str,
    level: float | None,
    direction_world: Sequence[float] | None,
    force_world: Sequence[float],
    torque_world: Sequence[float],
    binding: Mapping[str, Any],
    reset_id: str,
    baseline_samples: Sequence[Mapping[str, Any]],
    loaded_samples: Sequence[Mapping[str, Any]],
    state_restored: bool,
    settle_steps: int,
    load_steps: int,
    window: int,
) -> dict[str, Any]:
    load = {
        "force_world_n": list(force_world),
        "torque_world_nm": list(torque_world),
        "force_unit": "N",
        "torque_unit": "N*m",
        "application_frame": "world",
        "sensor_frame": "site",
    }
    if level is not None:
        load["level"] = float(level)
    if direction_world is not None:
        load["direction_world"] = list(direction_world)
    return {
        "case_id": case_id,
        "case_type": case_type,
        "executed": True,
        "reset_id": reset_id,
        "known_load": load,
        "sensor_transform": "-R_site.T @ applied_world",
        "binding": binding,
        "baseline_samples": list(baseline_samples),
        "loaded_samples": list(loaded_samples),
        "state_restored": bool(state_restored),
        "sampling": {
            "settle_steps": int(settle_steps),
            "load_steps": int(load_steps),
            "window_samples": int(window),
            "time_source": "sim.data.time",
            "fixed_dt_assumption": False,
        },
    }


def run_probe(
    output: Path,
    *,
    seed: int,
    body_name: str,
    site_name: str,
    force_sensor_name: str,
    torque_sensor_name: str,
    expected_body_id: int | None,
    force_levels_n: Sequence[float],
    force_directions_world: Sequence[Sequence[float]],
    torque_levels_nm: Sequence[float],
    torque_directions_world: Sequence[Sequence[float]],
    settle_steps: int,
    load_steps: int,
    window: int,
    max_gap_s: float,
    force_tolerance_n: float,
    torque_tolerance_nm: float,
    cross_force_tolerance_n: float,
    cross_torque_tolerance_nm: float,
) -> int:
    """Run the paired load probe and write an auditable JSON package."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(ROOT), str(ROOT / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")

    import mujoco
    import numpy as np

    from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
    from synthesis_v2.simulator_state import capture_simulator_snapshot, restore_simulator_snapshot

    force_levels = _validate_levels(force_levels_n, name="force_levels_n")
    torque_levels = _validate_levels(torque_levels_nm, name="torque_levels_nm")
    force_directions = tuple(_unit_vector(direction, name="force_direction_world") for direction in force_directions_world)
    torque_directions = tuple(_unit_vector(direction, name="torque_direction_world") for direction in torque_directions_world)
    if settle_steps < window or load_steps < window:
        raise ValueError("settle_steps and load_steps must each be >= window")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive and finite")

    spec = ControlledLoadSpec(
        expected_body_id=expected_body_id,
        expected_body_name=body_name,
        expected_site_name=site_name,
        expected_force_sensor_name=force_sensor_name,
        expected_torque_sensor_name=torque_sensor_name,
        min_window_samples=window,
        max_gap_s=max_gap_s,
        force_tolerance_n=force_tolerance_n,
        torque_tolerance_nm=torque_tolerance_nm,
        cross_force_tolerance_n=cross_force_tolerance_n,
        cross_torque_tolerance_nm=cross_torque_tolerance_nm,
        force_levels_n=tuple(force_levels),
        force_directions_world=tuple(force_directions),
        torque_levels_nm=tuple(torque_levels),
        torque_directions_world=tuple(torque_directions),
    )

    env = None
    cases: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema": "mannercap-gate1-wipe-controlled-load-v1",
        "task": "wipe",
        "seed": int(seed),
        "status": "started",
        "qualification": "simulator_sensor_channel_response_only",
        "physical_qualification": "unknown",
        "promotion_allowed": False,
        "sensor_calibration_verified": False,
        "contact_model_calibration_verified": False,
        "real_hardware_calibration_verified": False,
        "load_protocol": {
            "settle_steps": int(settle_steps),
            "load_steps": int(load_steps),
            "window_samples": int(window),
            "force_levels_n": list(force_levels),
            "force_directions_world": [list(direction) for direction in force_directions],
            "torque_levels_nm": list(torque_levels),
            "torque_directions_world": [list(direction) for direction in torque_directions],
            "force_tolerance_n": float(force_tolerance_n),
            "torque_tolerance_nm": float(torque_tolerance_nm),
            "cross_force_tolerance_n": float(cross_force_tolerance_n),
            "cross_torque_tolerance_nm": float(cross_torque_tolerance_nm),
            "max_gap_s": float(max_gap_s),
            "known_load_source": "sim.data.xfrc_applied[body_id, :3/3:]",
            "known_load_force_unit": "N",
            "known_load_torque_unit": "N*m",
            "known_load_frame": "world",
            "sensor_frame": "site",
            "sensor_transform": "-R_site.T @ applied_world",
            "no_contact_required": True,
            "hold_controller_step_called": True,
            "controller_command_source": "wrapper_saved_joint_target",
            "wipe_skill_or_ik_called": False,
            "method_api_called": False,
        },
    }
    try:
        np.random.seed(seed)
        env = FrankaRobosuiteSpillWipeLowLevel(
            privileged=True,
            enable_render=False,
            max_steps=max(4000, settle_steps + load_steps + 10),
            seed=seed,
        )
        np.random.seed(seed)
        _obs, reset_info = env.reset(seed=seed)
        robosuite_env = env.robosuite_env
        sim = robosuite_env.sim
        model = sim.model
        xml = model.get_xml()
        model_sha256 = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        (output / "model.xml").write_text(xml, encoding="utf-8")
        binding = resolve_binding(
            sim,
            body_name=body_name,
            site_name=site_name,
            force_sensor_name=force_sensor_name,
            torque_sensor_name=torque_sensor_name,
            expected_body_id=expected_body_id,
            mujoco=mujoco,
        )
        body_id = int(binding["body"]["id"])
        site_id = int(binding["sensor"]["site_id"])
        robot = robosuite_env.robots[0]
        if "right" not in tuple(robot.arms):
            raise ValueError(f"Wipe robot arms do not expose required right arm: {robot.arms!r}")
        arm = "right"
        initial_state = _capture_state(sim.data, np)
        initial_snapshot = capture_simulator_snapshot(env)
        mujoco.mj_forward(model._model, sim.data._data)
        initial_contacts = _contact_rows(sim, body_id, mujoco, np)
        reset_id = f"wipe-seed-{seed}-model-{model_sha256[:16]}"
        result["model"] = {
            "sha256": model_sha256,
            "physics_timestep_s": float(model.opt.timestep),
            "reset_info": reset_info,
        }
        result["binding"] = binding
        result["initial"] = {
            "time_s": float(sim.data.time),
            "reset_id": reset_id,
            "body_position_world_m": np.asarray(sim.data.xpos[body_id], dtype=np.float64).tolist(),
            "site_position_world_m": np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).tolist(),
            "site_rotation_world": np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).tolist(),
            "selected_body_contact_count": len(initial_contacts),
            "selected_body_contacts": initial_contacts,
            "state_arrays_recorded": sorted(name for name in initial_state if name != "time"),
        }

        def run_one(
            case_id: str,
            case_type: str,
            level: float | None,
            direction: Sequence[float] | None,
            force: Sequence[float],
            torque: Sequence[float],
        ) -> None:
            restore_simulator_snapshot(env, initial_snapshot)
            baseline = _run_window(
                env,
                sim,
                binding=binding,
                robot=robot,
                arm=arm,
                body_id=body_id,
                force_world=(0.0, 0.0, 0.0),
                torque_world=(0.0, 0.0, 0.0),
                steps=settle_steps,
                window=window,
                reset_id=reset_id,
                mujoco=mujoco,
                np=np,
            )
            restore_simulator_snapshot(env, initial_snapshot)
            loaded = _run_window(
                env,
                sim,
                binding=binding,
                robot=robot,
                arm=arm,
                body_id=body_id,
                force_world=force,
                torque_world=torque,
                steps=load_steps,
                window=window,
                reset_id=reset_id,
                mujoco=mujoco,
                np=np,
            )
            restore_simulator_snapshot(env, initial_snapshot)
            restored = _state_equal(sim.data, initial_state, np) and bool(np.allclose(sim.data.xfrc_applied, 0.0))
            cases.append(
                _case_record(
                    case_id=case_id,
                    case_type=case_type,
                    level=level,
                    direction_world=direction,
                    force_world=force,
                    torque_world=torque,
                    binding=binding,
                    reset_id=reset_id,
                    baseline_samples=baseline,
                    loaded_samples=loaded,
                    state_restored=restored,
                    settle_steps=settle_steps,
                    load_steps=load_steps,
                    window=window,
                )
            )

        run_one("no_contact_zero", "no_contact", None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for direction_index, direction in enumerate(force_directions):
            for level in force_levels:
                force = tuple(level * item for item in direction)
                run_one(
                    f"force_world_dir{direction_index}_level{level:g}n",
                    "force_level",
                    level,
                    direction,
                    force,
                    (0.0, 0.0, 0.0),
                )
        for direction_index, direction in enumerate(torque_directions):
            for level in torque_levels:
                torque = tuple(level * item for item in direction)
                run_one(
                    f"torque_world_dir{direction_index}_level{level:g}nm",
                    "torque_level",
                    level,
                    direction,
                    (0.0, 0.0, 0.0),
                    torque,
                )

        suite = audit_controlled_load_suite(cases, spec=spec)
        result["status"] = "executed"
        result["audit"] = suite.as_dict()
        result["cases"] = cases
        result["sensor_channel_response_verified"] = suite.decision is Decision.PASS
        result["physical_qualification"] = "unknown"
        result["promotion_allowed"] = False
        result["promotion_blockers"] = [
            "xfrc_applied response checks are simulator channel evidence, not a real sensor calibration",
            "no-contact response does not calibrate Wipe contact geometry or friction",
            "force/torque thresholds remain development tolerances pending independent sign-off",
        ]
        output_file = output / "controlled_load_calibration.json"
        write_json(output_file, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(output_file),
                    "case_count": len(cases),
                    "audit_decision": suite.decision.value,
                    "sensor_channel_response_verified": result["sensor_channel_response_verified"],
                    "physical_qualification": result["physical_qualification"],
                    "promotion_allowed": result["promotion_allowed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if suite.decision is Decision.PASS else 1
    except Exception:
        result["status"] = "execution_error"
        result["error"] = traceback.format_exc()
        result["physical_qualification"] = "unknown"
        result["promotion_allowed"] = False
        write_json(output / "controlled_load_calibration.json", result)
        print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        if env is not None:
            try:
                env.robosuite_env.sim.data.xfrc_applied[...] = 0.0
            except Exception:
                pass
            env.robosuite_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--body-name", default="gripper0_right_wiping_gripper")
    parser.add_argument("--site-name", default="gripper0_right_ft_frame")
    parser.add_argument("--force-sensor-name", default="gripper0_right_force_ee")
    parser.add_argument("--torque-sensor-name", default="gripper0_right_torque_ee")
    parser.add_argument("--expected-body-id", type=int, default=114)
    parser.add_argument("--force-levels-n", nargs="+", type=float, default=(2.0, 5.0, 10.0))
    parser.add_argument(
        "--force-direction-world",
        nargs=3,
        type=float,
        action="append",
        dest="force_directions_world",
        metavar=("FX", "FY", "FZ"),
        help="unit direction; repeat for additional world axes (default: +X,+Y,+Z)",
    )
    parser.add_argument("--torque-levels-nm", nargs="+", type=float, default=(0.05, 0.1))
    parser.add_argument(
        "--torque-direction-world",
        nargs=3,
        type=float,
        action="append",
        dest="torque_directions_world",
        metavar=("TX", "TY", "TZ"),
        help="unit direction; repeat for additional world axes (default: +Z)",
    )
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--load-steps", type=int, default=60)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--max-gap-s", type=float, default=0.1)
    parser.add_argument("--force-tolerance-n", type=float, default=0.01)
    parser.add_argument("--torque-tolerance-nm", type=float, default=1e-4)
    parser.add_argument("--cross-force-tolerance-n", type=float, default=0.01)
    parser.add_argument("--cross-torque-tolerance-nm", type=float, default=1e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force_directions = args.force_directions_world or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    torque_directions = args.torque_directions_world or ((0.0, 0.0, 1.0),)
    if args.settle_steps < args.window or args.load_steps < args.window:
        raise SystemExit("--settle-steps and --load-steps must each be >= --window")
    return run_probe(
        args.output,
        seed=args.seed,
        body_name=args.body_name,
        site_name=args.site_name,
        force_sensor_name=args.force_sensor_name,
        torque_sensor_name=args.torque_sensor_name,
        expected_body_id=args.expected_body_id,
        force_levels_n=args.force_levels_n,
        force_directions_world=force_directions,
        torque_levels_nm=args.torque_levels_nm,
        torque_directions_world=torque_directions,
        settle_steps=args.settle_steps,
        load_steps=args.load_steps,
        window=args.window,
        max_gap_s=args.max_gap_s,
        force_tolerance_n=args.force_tolerance_n,
        torque_tolerance_nm=args.torque_tolerance_nm,
        cross_force_tolerance_n=args.cross_force_tolerance_n,
        cross_torque_tolerance_nm=args.cross_torque_tolerance_nm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
