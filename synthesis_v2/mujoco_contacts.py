"""MuJoCo object wrench and contact-pair recording for Gate 1 evidence."""
from __future__ import annotations

from typing import Any, Iterable


def split_body_spatial_wrench(row: Iterable[float]) -> dict[str, list[float]]:
    """Split cfrc_ext's [rotation, translation] spatial-vector layout."""
    values = [float(value) for value in row]
    if len(values) != 6:
        raise ValueError("cfrc_ext body row must contain six values")
    return {"torque_nm": values[:3], "force_n": values[3:]}


def split_contact_wrench(row: Iterable[float]) -> dict[str, list[float]]:
    """Split mj_contactForce's documented [force, torque] layout."""
    values = [float(value) for value in row]
    if len(values) != 6:
        raise ValueError("mj_contactForce result must contain six values")
    return {"force_n": values[:3], "torque_nm": values[3:]}


def resolve_task_object_registry(robosuite_env: Any, task: str) -> list[dict[str, Any]]:
    """Resolve stable object/body/geom identities from a supported task."""
    if task == "lift":
        objects = [("primary", robosuite_env.cube)]
    elif task == "stack":
        objects = [("primary", robosuite_env.cubeA), ("secondary", robosuite_env.cubeB)]
    else:
        raise ValueError(f"unsupported task for object contact recording: {task}")

    model = robosuite_env.sim.model
    registry: list[dict[str, Any]] = []
    for object_id, obj in objects:
        body_name = str(obj.root_body)
        geom_names = tuple(str(name) for name in obj.contact_geoms)
        registry.append({
            "object_id": object_id,
            "body_id": int(model.body_name2id(body_name)),
            "body_name": body_name,
            "geom_ids": [int(model.geom_name2id(name)) for name in geom_names],
            "geom_names": list(geom_names),
            "body_wrench_frame": "mujoco_subtree_com",
            "body_wrench_layout": "torque_nm_then_force_n",
        })
    return registry


def capture_object_contacts(sim: Any, registry: list[dict[str, Any]]) -> dict[str, Any]:
    """Capture body aggregates and object-involving contact-pair wrenches."""
    import mujoco
    import numpy as np

    model = sim.model
    data = sim.data
    # cfrc_ext is only guaranteed after the post-constraint RNE stage.
    mujoco.mj_rnePostConstraint(model._model, data._data)

    objects = []
    object_geoms: dict[int, str] = {}
    for binding in registry:
        for geom_id in binding["geom_ids"]:
            object_geoms[int(geom_id)] = binding["object_id"]
        row = split_body_spatial_wrench(np.asarray(data.cfrc_ext[binding["body_id"]]))
        objects.append({
            "object_id": binding["object_id"],
            "body_id": binding["body_id"],
            "body_name": binding["body_name"],
            "frame": binding["body_wrench_frame"],
            **row,
        })

    contacts = []
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        involved = sorted({object_geoms[g] for g in (geom1, geom2) if g in object_geoms})
        if not involved:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model._model, data._data, contact_id, wrench)
        contacts.append({
            "contact_id": contact_id,
            "object_ids": involved,
            "geom1_id": geom1,
            "geom1_name": model.geom_id2name(geom1),
            "geom2_id": geom2,
            "geom2_name": model.geom_id2name(geom2),
            "body1_id": int(model.geom_bodyid[geom1]),
            "body2_id": int(model.geom_bodyid[geom2]),
            "distance_m": float(contact.dist),
            "position_world_m": np.asarray(contact.pos).tolist(),
            "contact_frame_rows_world": np.asarray(contact.frame).reshape(3, 3).tolist(),
            "wrench_frame": "contact_local",
            "force_direction_attribution": "not_qualified",
            **split_contact_wrench(wrench),
        })
    return {
        "sim_time_s": float(data.time),
        "sensor_valid": True,
        "body_aggregates": objects,
        "object_contacts": contacts,
    }


__all__ = [
    "capture_object_contacts",
    "resolve_task_object_registry",
    "split_body_spatial_wrench",
    "split_contact_wrench",
]
