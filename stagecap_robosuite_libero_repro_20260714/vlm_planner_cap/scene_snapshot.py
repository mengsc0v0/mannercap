"""Readonly scene snapshot helpers for Stage-Verifiable CaP.

The planner and stage code agent are allowed to use environment facts that do
not mutate simulator state. This module collects a compact JSON-safe snapshot
from exposed readonly APIs and passes it downstream as grounding/evidence.
"""

from __future__ import annotations

from typing import Any, Callable

from .schema import StagePlan


MAX_NAMES = 80
MAX_CONTACTS = 24
MAX_DIAGNOSTICS = 24
MAX_OBJECTS_OF_INTEREST = 4
MAX_TARGET_SITES = 4
MAX_GRASP_CANDIDATES = 4


def collect_readonly_scene_snapshot(
    api_functions: dict[str, Callable[..., Any]],
    stage: StagePlan | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Collect a compact scene snapshot using only exposed readonly APIs."""
    snapshot: dict[str, Any] = {
        "task": task or "",
        "stage_id": stage.stage_id if stage else None,
        "stage_targets": list(stage.target_references) if stage else [],
        "errors": {},
    }

    for key, fn_name, limit in (
        ("task_status", "get_task_status", None),
        ("robosuite_task_context", "get_robosuite_task_context", None),
        ("bddl_goal_diagnostics", "get_bddl_goal_diagnostics", None),
        ("ee_pose", "get_ee_pose", None),
        ("gripper_state", "get_gripper_state", None),
        ("joint_positions", "get_joint_positions", None),
        ("instance_object_names", "get_instance_object_names", MAX_NAMES),
        ("scene_site_names", "get_scene_site_names", MAX_NAMES),
        ("scene_geom_names", "get_scene_geom_names", MAX_NAMES),
        ("scene_joint_names", "get_scene_joint_names", MAX_NAMES),
        ("all_object_poses", "get_all_object_poses", None),
    ):
        if fn_name in api_functions:
            _capture(snapshot, api_functions, key, fn_name, limit=limit)
    for key, fn_name in (
        ("cube_lift_diagnostic", "get_cube_lift_diagnostic"),
        ("stack_diagnostic", "get_stack_diagnostic"),
        ("restack_diagnostic", "get_restack_diagnostic"),
        ("nut_peg_diagnostic", "get_nut_peg_diagnostic"),
        ("bimanual_lift_diagnostic", "get_bimanual_lift_diagnostic"),
        ("wipe_diagnostic", "get_wipe_diagnostic"),
        ("hammer_handle_geometry", "get_hammer_handle_geometry"),
        ("handover_diagnostic", "get_handover_diagnostic"),
    ):
        if fn_name in api_functions:
            _capture(snapshot, api_functions, key, fn_name)

    stage_text = " ".join(stage.target_references if stage else [])
    task_text = task or ""
    lower = " ".join([stage_text, task_text]).lower()
    object_names = _candidate_object_names(snapshot, stage_text.lower(), task_text.lower())
    target_sites = _candidate_site_names(snapshot, stage_text.lower(), task_text.lower())

    object_snapshots = {}
    for object_name in object_names[:MAX_OBJECTS_OF_INTEREST]:
        object_snapshots[object_name] = {}
        if "get_object_pose" in api_functions:
            object_snapshots[object_name]["pose"] = _call(
                api_functions, "get_object_pose", object_name
            )
        grasp_state_supported, affordance_supported = _grasp_snapshot_channels(
            snapshot, object_name
        )
        if grasp_state_supported and "measure_object_grasp_state" in api_functions:
            object_snapshots[object_name]["grasp_state"] = _call(
                api_functions, "measure_object_grasp_state", object_name
            )
        if affordance_supported and "get_object_grasp_affordance" in api_functions:
            affordance = _call(api_functions, "get_object_grasp_affordance", object_name)
            object_snapshots[object_name]["grasp_affordance"] = _compact_grasp_affordance(affordance)
    if object_snapshots:
        snapshot["objects_of_interest"] = object_snapshots

    site_snapshots = {}
    for site_name in target_sites[:MAX_TARGET_SITES]:
        site_snapshots[site_name] = {
            "pose": _call(api_functions, "get_site_pose", site_name),
        }
    if site_snapshots:
        snapshot["target_sites_of_interest"] = site_snapshots

    if "drawer" in lower or any("drawer" in str(ref).lower() for ref in (stage.target_references if stage else [])):
        drawer = {}
        for level in ("top", "middle", "bottom"):
            drawer[level] = {
                "handle_pose": _call(api_functions, "get_cabinet_drawer_handle_pose", level),
                "progress": _call(api_functions, "get_cabinet_drawer_progress", level),
                "sleeve_hook_poses": _compact_hook_poses(_call(api_functions, "get_drawer_sleeve_hook_poses", level)),
                "hook_alignment": _call(api_functions, "measure_drawer_hook_alignment", level),
                "contacts": _call(api_functions, "measure_drawer_gripper_contacts", level),
            }
        snapshot["drawers"] = drawer

    if "stove" in lower:
        snapshot["stove_diagnostics"] = {
            "object_state": _call(api_functions, "get_object_state_diagnostic", "flat_stove_1"),
            "geom_diagnostics": _limit(_call(api_functions, "get_geom_diagnostics", "stove"), MAX_DIAGNOSTICS),
        }

    if "get_contact_diagnostics" in api_functions:
        snapshot["contact_diagnostics"] = _limit(
            _call(api_functions, "get_contact_diagnostics"),
            MAX_CONTACTS,
        )
    if "get_joint_diagnostics" in api_functions:
        snapshot["joint_diagnostics"] = _limit(
            _call(api_functions, "get_joint_diagnostics"),
            MAX_DIAGNOSTICS,
        )
    if not snapshot.get("errors"):
        snapshot.pop("errors", None)
    return _json_safe(snapshot)


def flatten_api_functions(apis: dict[str, Any]) -> dict[str, Callable[..., Any]]:
    """Flatten API objects into a function-name map."""
    functions: dict[str, Callable[..., Any]] = {}
    for api in apis.values():
        try:
            api_functions = api.functions()
        except Exception:
            continue
        for name, fn in api_functions.items():
            if callable(fn):
                functions[name] = fn
    return functions


def snapshot_for_prompt(snapshot: dict[str, Any], max_chars: int = 9000) -> str:
    """Serialize a scene snapshot compactly for prompt injection."""
    import json

    text = json.dumps(_json_safe(snapshot), ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <snapshot truncated>"


def _capture(
    snapshot: dict[str, Any],
    api_functions: dict[str, Callable[..., Any]],
    key: str,
    fn_name: str,
    *args: Any,
    limit: int | None = None,
) -> None:
    value = _call(api_functions, fn_name, *args)
    if isinstance(value, dict) and "error" in value:
        snapshot.setdefault("errors", {})[key] = value["error"]
        return
    snapshot[key] = _limit(value, limit) if limit is not None else value


def _call(api_functions: dict[str, Callable[..., Any]], fn_name: str, *args: Any) -> Any:
    fn = api_functions.get(fn_name)
    if fn is None:
        return {"error": f"API not available: {fn_name}"}
    try:
        return fn(*args)
    except Exception as exc:
        return {"error": repr(exc)}


def _candidate_object_names(snapshot: dict[str, Any], lower_stage_text: str, lower_task_text: str = "") -> list[str]:
    names: list[str] = []
    all_poses = snapshot.get("all_object_poses")
    if isinstance(all_poses, dict):
        names.extend(str(name) for name in all_poses)
    exact_names = snapshot.get("instance_object_names")
    if isinstance(exact_names, list):
        names.extend(str(name) for name in exact_names)
    names = list(dict.fromkeys(names))
    scored = sorted(
        names,
        key=lambda name: (
            _name_match_rank(name, lower_stage_text),
            _semantic_name_rank(name, lower_stage_text),
            _name_match_rank(name, lower_task_text),
            name,
        ),
    )
    return scored


def _candidate_site_names(snapshot: dict[str, Any], lower_stage_text: str, lower_task_text: str = "") -> list[str]:
    names = [str(name) for name in snapshot.get("scene_site_names", []) if isinstance(name, str)]
    target_words = ("region", "contain", "cook", "plate", "basket", "microwave", "stove", "drawer", "rack", "caddy", "bowl")
    filtered = [name for name in names if any(word in name.lower() for word in target_words)]
    return sorted(
        filtered,
        key=lambda name: (
            _name_match_rank(name, lower_stage_text),
            _semantic_name_rank(name, lower_stage_text),
            _name_match_rank(name, lower_task_text),
            name,
        ),
    )


def _name_matches(name: str, text: str) -> bool:
    return _name_match_rank(name, text) < 100


def _name_match_rank(name: str, text: str) -> int:
    normalized = name.lower().replace("_", " ")
    text_normalized = text.lower().replace("_", " ")
    compact_name = normalized.replace(" ", "")
    compact_text = text_normalized.replace(" ", "")
    if compact_name and compact_name in compact_text:
        return 0
    parts = [part for part in normalized.split() if part]
    if parts and all(part in text_normalized for part in parts):
        return 1
    if any(part and part in text_normalized for part in parts):
        return 10
    return 100


def _semantic_name_rank(name: str, text: str) -> int:
    normalized = name.lower().replace("_", " ")
    text_normalized = text.lower().replace("_", " ")
    rank = 0
    if "white mug" in text_normalized and "porcelain mug" in normalized:
        rank -= 50
    if (
        ("yellow and white mug" in text_normalized or "yellow white mug" in text_normalized)
        and "white yellow mug" in normalized
    ):
        rank -= 50
    if "left plate" in text_normalized and "plate left" in normalized:
        rank -= 50
    if "right plate" in text_normalized and "plate right" in normalized:
        rank -= 50
    if "back caddy compartment" in text_normalized and "back contain" in normalized:
        rank -= 50
    if "front caddy compartment" in text_normalized and "front contain" in normalized:
        rank -= 50
    if "front of stove" in text_normalized and "stove front" in normalized:
        rank -= 60
    if "back of stove" in text_normalized and "stove back" in normalized:
        rank -= 60
    return rank


def _looks_movable_object(name: str) -> bool:
    lower = name.lower()
    fixture_words = ("cabinet", "stove", "table", "floor", "wall", "region", "site")
    return not any(word in lower for word in fixture_words)


def _grasp_snapshot_channels(snapshot: dict[str, Any], object_name: str) -> tuple[bool, bool]:
    """Return whether grasp-state and affordance calls are meaningful for an object."""
    if not _looks_movable_object(object_name):
        return False, False

    context = snapshot.get("robosuite_task_context")
    if not isinstance(context, dict):
        return True, True

    family = str(context.get("task_family", "")).lower()
    name = object_name.lower()
    if family == "peg_insertion":
        graspable = "nut" in name and "block" not in name
        return graspable, graspable
    if family == "bimanual_lift":
        return "pot" in name, "handle" in name
    if family == "wipe":
        return False, False
    if family == "bimanual_handover":
        return True, "handle" in name
    return True, True


def _compact_grasp_affordance(value: Any) -> Any:
    if not isinstance(value, dict) or "error" in value:
        return value
    result = dict(value)
    candidates = result.get("candidates")
    if isinstance(candidates, list):
        result["candidates"] = candidates[:MAX_GRASP_CANDIDATES]
        result["candidate_count"] = len(candidates)
    return result


def _compact_hook_poses(value: Any) -> Any:
    if not isinstance(value, dict) or "error" in value:
        return value
    result = dict(value)
    for key in ("sleeve_candidates", "hook_candidates", "side_hook_candidates"):
        candidates = result.get(key)
        if isinstance(candidates, list):
            result[key] = candidates[:MAX_GRASP_CANDIDATES]
            result[f"{key}_count"] = len(candidates)
    return result


def _limit(value: Any, limit: int | None) -> Any:
    if limit is None:
        return value
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, tuple):
        return list(value[:limit])
    return value


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
