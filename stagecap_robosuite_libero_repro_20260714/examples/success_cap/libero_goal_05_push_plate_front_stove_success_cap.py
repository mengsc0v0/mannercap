# Stage code block 0
STAGE_ID = "obj1_plate_ground_object_target"
STAGE_ATTEMPT = 1
import json
import numpy as np

object_name = 'plate'
predicate = 'on'
semantic_target_name = 'front of stove'
object_pos, object_quat = get_object_pose(object_name)
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_name = target.get("target_name") if isinstance(target, dict) else None
target_position = target.get("target_position") if isinstance(target, dict) else None
target_valid = target_position is not None and bool(np.all(np.isfinite(np.asarray(target_position, dtype=float))))
object_valid = object_pos is not None and bool(np.all(np.isfinite(np.asarray(object_pos, dtype=float))))
status = "satisfied" if object_valid and target_valid else "needs_repair"
evidence = {
    "object_name": object_name,
    "predicate": predicate,
    "semantic_target_name": semantic_target_name,
    "target_name": target_name,
    "object_position": object_pos,
    "target_position": target_position,
    "object_valid": object_valid,
    "target_valid": target_valid,
}
repair = "" if status == "satisfied" else "Grounding failed; retry with the exact movable object and BDDL target region from the scene snapshot."
print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 1
STAGE_ID = "obj1_plate_push_to_region_verify"
STAGE_ATTEMPT = 1
import json

stage_id = 'obj1_plate_push_to_region_verify'
object_name = 'plate'
target_name = 'front of stove'
predicate = 'on'

try:
    before_goal = get_bddl_goal_diagnostics()
    print("Push-region BDDL before:", before_goal)
    result = push_object_to_target_region(
        object_name,
        target_name=target_name or None,
        predicate=predicate,
        max_iters=5,
        settle_steps=55,
    )
    print("Push-region result:", result)
    after_goal = get_bddl_goal_diagnostics()
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    relation = result.get("final_relation", {}) if isinstance(result, dict) else {}
    best_relation = result.get("best_relation", {}) if isinstance(result, dict) else {}
    relation_satisfied = bool(isinstance(result, dict) and result.get("relation_satisfied", False))
    all_goals = bool(isinstance(after_goal, dict) and after_goal.get("all_goals_satisfied", False))
    success = bool(relation_satisfied or completed or reward >= 1.0 or all_goals)
    records = result.get("records", []) if isinstance(result, dict) else []
    last_record = records[-1] if isinstance(records, list) and records else {}
    status = "satisfied" if success else "needs_repair"
    evidence = (
        f"success={success}, relation_satisfied={relation_satisfied}, reward={reward:.3f}, "
        f"task_completed={completed}, all_goals={all_goals}, object={result.get('object_name', object_name) if isinstance(result, dict) else object_name}, "
        f"target={result.get('target_name', target_name) if isinstance(result, dict) else target_name}, predicate={predicate}, "
        f"final_relation={relation}, best_relation={best_relation}, last_record={last_record}, reason={result.get('reason') if isinstance(result, dict) else None}"
    )
    repair = "" if success else "Push-region relation remains unsatisfied; retry with refreshed local_delta/contact side or re-ground target region."
except Exception as exc:
    status = "needs_repair"
    evidence = f"push-region fast-path exception: {exc}"
    repair = "Check object alias, target region grounding, and push_object_to_target_region availability."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))


# Stage code block 2
STAGE_ID = "obj1_plate_push_to_region_verify"
STAGE_ATTEMPT = 2
import json

stage_id = 'obj1_plate_push_to_region_verify'
object_name = 'plate'
target_name = 'front of stove'
predicate = 'on'

try:
    before_goal = get_bddl_goal_diagnostics()
    print("Push-region BDDL before:", before_goal)
    result = push_object_to_target_region(
        object_name,
        target_name=target_name or None,
        predicate=predicate,
        max_iters=5,
        settle_steps=55,
    )
    print("Push-region result:", result)
    after_goal = get_bddl_goal_diagnostics()
    task_status = get_task_status()
    reward = float(task_status.get("reward", 0.0) or 0.0)
    completed = bool(task_status.get("task_completed", False))
    relation = result.get("final_relation", {}) if isinstance(result, dict) else {}
    best_relation = result.get("best_relation", {}) if isinstance(result, dict) else {}
    relation_satisfied = bool(isinstance(result, dict) and result.get("relation_satisfied", False))
    all_goals = bool(isinstance(after_goal, dict) and after_goal.get("all_goals_satisfied", False))
    success = bool(relation_satisfied or completed or reward >= 1.0 or all_goals)
    records = result.get("records", []) if isinstance(result, dict) else []
    last_record = records[-1] if isinstance(records, list) and records else {}
    status = "satisfied" if success else "needs_repair"
    evidence = (
        f"success={success}, relation_satisfied={relation_satisfied}, reward={reward:.3f}, "
        f"task_completed={completed}, all_goals={all_goals}, object={result.get('object_name', object_name) if isinstance(result, dict) else object_name}, "
        f"target={result.get('target_name', target_name) if isinstance(result, dict) else target_name}, predicate={predicate}, "
        f"final_relation={relation}, best_relation={best_relation}, last_record={last_record}, reason={result.get('reason') if isinstance(result, dict) else None}"
    )
    repair = "" if success else "Push-region relation remains unsatisfied; retry with refreshed local_delta/contact side or re-ground target region."
except Exception as exc:
    status = "needs_repair"
    evidence = f"push-region fast-path exception: {exc}"
    repair = "Check object alias, target region grounding, and push_object_to_target_region availability."

print("STAGE_STATUS " + json.dumps({
    "stage_id": stage_id,
    "status": status,
    "evidence": evidence,
    "repair": repair,
}, default=str))
