"""Capture and restore the MuJoCo state needed for deterministic action replay.

The snapshot is JSON-compatible.  It includes mutable MuJoCo inputs that are
not present in robosuite's legacy ``MjSimState`` plus wrapper counters and RNG
state. Controller and interpolator caches are included, but replay must still
compare every resulting transition instead of trusting restoration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import random
from typing import Any


DATA_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "plugin_state",
    "eq_active",
)

WRAPPER_SCALAR_FIELDS = (
    "_step_count",
    "_sim_step_count",
    "_gripper_fraction",
    "_manner_phase",
)

ROBOSUITE_SCALAR_FIELDS = ("timestep", "cur_time", "done")


def _simple_object_state(obj: Any) -> dict[str, Any]:
    """Return JSON-safe scalar / array attributes without following references."""
    state: dict[str, Any] = {}
    for name, value in vars(obj).items():
        if isinstance(value, (str, int, float, bool, type(None), list, tuple)) or hasattr(value, "tolist"):
            state[name] = _jsonable(value)
    return state


def _restore_simple_object_state(obj: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        if not hasattr(obj, name):
            continue
        current = getattr(obj, name)
        if hasattr(current, "shape") and isinstance(value, (list, int, float)):
            import numpy as np

            setattr(obj, name, np.asarray(value, dtype=current.dtype))
        else:
            setattr(obj, name, deepcopy(value))


def _capture_controller_state(env: Any) -> dict[str, Any]:
    robots: list[dict[str, Any]] = []
    for robot in getattr(env.robosuite_env, "robots", []):
        parts: dict[str, Any] = {}
        composite = getattr(robot, "composite_controller", None)
        for name, controller in getattr(composite, "part_controllers", {}).items():
            interpolator = getattr(controller, "interpolator", None)
            parts[name] = {
                "controller": _simple_object_state(controller),
                "interpolator": None if interpolator is None else _simple_object_state(interpolator),
            }
        robots.append({"robot": _simple_object_state(robot), "parts": parts})
    return {"robots": robots}


def _restore_controller_state(env: Any, state: dict[str, Any]) -> None:
    live_robots = getattr(env.robosuite_env, "robots", [])
    saved_robots = state.get("robots", [])
    if len(live_robots) != len(saved_robots):
        raise ValueError("controller robot count differs from snapshot")
    for robot, saved in zip(live_robots, saved_robots):
        _restore_simple_object_state(robot, saved["robot"])
        composite = getattr(robot, "composite_controller", None)
        live_parts = getattr(composite, "part_controllers", {})
        if set(live_parts) != set(saved["parts"]):
            raise ValueError("controller part names differ from snapshot")
        for name, part_state in saved["parts"].items():
            controller = live_parts[name]
            _restore_simple_object_state(controller, part_state["controller"])
            interpolator_state = part_state["interpolator"]
            if interpolator_state is not None:
                interpolator = getattr(controller, "interpolator", None)
                if interpolator is None:
                    raise ValueError(f"missing interpolator for controller part {name}")
                _restore_simple_object_state(interpolator, interpolator_state)


def _array_to_list(value: Any) -> list[Any]:
    converted = value.tolist() if hasattr(value, "tolist") else list(value)
    return converted if isinstance(converted, list) else [converted]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _nested_tuple(value: Any) -> Any:
    return tuple(_nested_tuple(item) for item in value) if isinstance(value, list) else value


def capture_simulator_snapshot(env: Any) -> dict[str, Any]:
    sim = env.robosuite_env.sim
    data = sim.data
    arrays: dict[str, list[Any]] = {}
    missing: list[str] = []
    for name in DATA_ARRAY_FIELDS:
        if hasattr(data, name):
            arrays[name] = _array_to_list(getattr(data, name))
        else:
            missing.append(name)

    wrapper = {
        name: deepcopy(getattr(env, name))
        for name in WRAPPER_SCALAR_FIELDS
        if hasattr(env, name)
    }
    if hasattr(env, "_current_joints"):
        wrapper["_current_joints"] = _array_to_list(env._current_joints)

    robosuite_state = {
        name: deepcopy(getattr(env.robosuite_env, name))
        for name in ROBOSUITE_SCALAR_FIELDS
        if hasattr(env.robosuite_env, name)
    }

    numpy_legacy = None
    try:
        import numpy as np

        state = np.random.get_state()
        numpy_legacy = {
            "bit_generator": state[0],
            "keys": state[1].tolist(),
            "position": int(state[2]),
            "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }
    except Exception:
        pass

    env_rng = None
    if hasattr(env, "_rng") and hasattr(env._rng, "bit_generator"):
        env_rng = _jsonable(env._rng.bit_generator.state)

    return {
        "schema": "mannercap-mujoco-snapshot-v2",
        "time_s": float(data.time),
        "data_arrays": arrays,
        "missing_optional_data_fields": missing,
        "wrapper_state": _jsonable(wrapper),
        "robosuite_state": _jsonable(robosuite_state),
        "python_random_state": _jsonable(random.getstate()),
        "numpy_legacy_random_state": numpy_legacy,
        "env_rng_state": env_rng,
        "controller_state": _capture_controller_state(env),
        "controller_object_state_captured": True,
    }


def restore_simulator_snapshot(env: Any, snapshot: dict[str, Any]) -> None:
    if snapshot.get("schema") not in {"mannercap-mujoco-snapshot-v1", "mannercap-mujoco-snapshot-v2"}:
        raise ValueError("unsupported simulator snapshot schema")
    sim = env.robosuite_env.sim
    data = sim.data
    arrays = snapshot["data_arrays"]

    # Kinematic inputs must be in place before forward computes derived state.
    for name in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
        if name in arrays and hasattr(data, name):
            getattr(data, name)[:] = arrays[name]
    data.time = float(snapshot["time_s"])
    sim.forward()

    # Forward may update solver fields; restore all captured mutable inputs.
    for name, value in arrays.items():
        if hasattr(data, name):
            target = getattr(data, name)
            if getattr(target, "size", len(value)) != 0:
                target[:] = value
    data.time = float(snapshot["time_s"])

    for name, value in snapshot.get("wrapper_state", {}).items():
        if hasattr(env, name):
            current = getattr(env, name)
            if hasattr(current, "shape") and isinstance(value, list):
                import numpy as np

                setattr(env, name, np.asarray(value, dtype=current.dtype))
            else:
                setattr(env, name, value)
    for name, value in snapshot.get("robosuite_state", {}).items():
        if hasattr(env.robosuite_env, name):
            setattr(env.robosuite_env, name, value)

    random.setstate(_nested_tuple(snapshot["python_random_state"]))
    numpy_state = snapshot.get("numpy_legacy_random_state")
    if numpy_state is not None:
        import numpy as np

        np.random.set_state((
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
    if snapshot.get("env_rng_state") is not None and hasattr(env, "_rng"):
        env._rng.bit_generator.state = snapshot["env_rng_state"]
    if snapshot.get("controller_state") is not None:
        _restore_controller_state(env, snapshot["controller_state"])


def save_compiled_model_binary(env: Any, path: str | Path) -> str:
    """Save the exact compiled MuJoCo model and return its SHA-256.

    MuJoCo XML serialization is not lossless for compiled runtime fields such
    as inferred inertias and armatures.  Strict replay therefore needs the
    binary ``.mjb`` model in addition to the human-readable XML provenance.
    """

    import mujoco

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapped_model = env.robosuite_env.sim.model
    model = getattr(wrapped_model, "_model", wrapped_model)
    mujoco.mj_saveModel(model, str(target), None)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if target.stat().st_size <= 0:
        raise RuntimeError("compiled MuJoCo model binary is empty")
    return digest


def reset_robosuite_from_model_binary(env: Any, path: str | Path) -> None:
    """Replace robosuite's simulator with an exact compiled ``.mjb`` model.

    This mirrors ``reset_from_xml_string`` while avoiding XML recompilation.
    Robosuite's normal deterministic reset is still run so robots,
    controllers, observables, and cached simulator references bind to the new
    ``MjSim`` before the recorded snapshot is restored.
    """

    import mujoco
    from robosuite.utils.binding_utils import MjSim

    inner = env.robosuite_env
    inner.close()
    inner.deterministic_reset = True
    try:
        compiled_model = mujoco.MjModel.from_binary_path(str(Path(path)))
        inner.sim = MjSim(compiled_model)
        inner.sim.forward()
        inner.initialize_time(inner.control_freq)
        inner.reset()
    finally:
        inner.deterministic_reset = False


__all__ = [
    "capture_simulator_snapshot",
    "reset_robosuite_from_model_binary",
    "restore_simulator_snapshot",
    "save_compiled_model_binary",
]
