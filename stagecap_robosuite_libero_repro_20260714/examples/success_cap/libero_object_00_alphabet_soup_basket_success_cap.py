# Stage code block 0
STAGE_ID = "obj1_alphabet_soup_ground_object_target"
STAGE_ATTEMPT = 1
import json
import numpy as np


# Resolve object and target names from the scene snapshot
object_name = "alphabet_soup"
target_predicate = "in"

# Get the object pose
object_pose = get_object_pose(object_name)
if object_pose is None:
    # Try with instance name from BDDL diagnostics
    object_name = "alphabet_soup_1"
    object_pose = get_object_pose(object_name)
    if object_pose is None:
        evidence = f"object '{object_name}' not found"
        repair = "list visible objects with get_observation or check scene"
        status = "not_satisfied"
        print("STAGE_STATUS " + json.dumps({
            "stage_id": STAGE_ID,
            "status": status,
            "evidence": evidence,
            "repair": repair
        }, default=str))
        raise SystemExit

object_pos, object_quat = object_pose
evidence_obj = {
    "object_name": object_name,
    "position": object_pos,
    "quaternion": object_quat
}

# Get target region pose for the alphabet soup with predicate "in"
target = get_goal_target_region_pose(object_name, predicate=target_predicate)
if target is None:
    evidence = f"target region for '{object_name}' with predicate '{target_predicate}' not found"
    repair = "check BDDL goal or scene for basket region"
    status = "not_satisfied"
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))
    raise SystemExit

target_name = target["target_name"]
target_pos = target["target_position"]
evidence_target = {
    "target_name": target_name,
    "target_position": target_pos
}

# Check that the target is a basket (semantic consistency)
is_basket = "basket" in target_name.lower()
if not is_basket:
    evidence = f"target '{target_name}' does not appear to be a basket"
    repair = "verify task language and BDDL goal"
    status = "not_satisfied"
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))
    raise SystemExit

# Check reachability: both object and target positions should be within plausible workspace
# Assuming typical RoboMimic workspace boundaries (adjust if known)
workspace_min = np.array([-1.0, -1.0, 0.0])
workspace_max = np.array([1.0, 1.0, 1.0])
object_reachable = np.all((object_pos >= workspace_min) & (object_pos <= workspace_max))
target_reachable = np.all((target_pos >= workspace_min) & (target_pos <= workspace_max))

if not object_reachable or not target_reachable:
    evidence = f"object reachable: {object_reachable}, target reachable: {target_reachable}"
    repair = "check object and target positions relative to robot workspace"
    status = "not_satisfied"
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))
    raise SystemExit

# Optional: check grasp affordance to ensure the object is graspable
affordance = get_object_grasp_affordance(object_name)
graspable = len(affordance.get("candidates", [])) > 0 if affordance else False
evidence_grasp = {
    "graspable": graspable,
    "num_candidates": len(affordance.get("candidates", [])) if affordance else 0
}

# All checks passed
evidence = {
    "object": evidence_obj,
    "target": evidence_target,
    "grasp": evidence_grasp,
    "reachable": True,
    "is_basket": True
}
status = "satisfied"
repair = ""

print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair
}, default=str))

# Stage code block 1
STAGE_ID = "obj1_alphabet_soup_pick_lift_verify"
STAGE_ATTEMPT = 1
import json
import numpy as np

# Current stage information

# Define object name from the current stage target references
object_name = "alphabet_soup"  # Base name; the system should handle instance resolution

# Execute the audited pick contract exactly as specified in the oracle-readonly contract
grasp = verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1)

# Extract evidence fields from the grasp result
held = bool(grasp.get("held") or grasp.get("transport_stable_held"))
lift_delta = float(grasp.get("lift_delta", 0.0) or 0.0)
final_grasp_state = grasp.get("final_grasp_state", {})
final_contact_supported = grasp.get("final_contact_supported", False)
skill_used = grasp.get("skill", "unknown")
stage_pick_contract = grasp.get("stage_pick_contract", {})

# Determine stage status based on contract criteria
status = "satisfied" if held and lift_delta >= 0.05 else "needs_repair"

# Prepare evidence for logging
evidence = {
    "object_name": object_name,
    "held": held,
    "transport_stable_held": grasp.get("transport_stable_held", False),
    "lift_delta": lift_delta,
    "final_grasp_state": final_grasp_state,
    "final_contact_supported": final_contact_supported,
    "skill_used": skill_used,
    "stage_pick_contract": stage_pick_contract
}

# Repair message if needed
repair = "" if status == "satisfied" else "retry audited pick contract on next stage attempt"

# Print final status
print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair
}, default=str))

# Stage code block 2
STAGE_ID = "obj1_alphabet_soup_transport_guarded"
STAGE_ATTEMPT = 1
import numpy as np
import json

# Stage parameters
object_name = "alphabet_soup"
predicate = "in"

# Step 1: Verify the object is still held from previous stage
grasp_state = measure_object_grasp_state(object_name)
held_before = bool(grasp_state.get("held") or grasp_state.get("strict_held") or 
                   grasp_state.get("held_by_contact") or grasp_state.get("elevated_contact"))
if not held_before:
    status = "needs_repair"
    evidence = {"held_before": False, "grasp_state": grasp_state}
    repair = "Object not held at start of transport; return to pick_lift_verify stage"
    print("STAGE_STATUS " + json.dumps({"stage_id": STAGE_ID, "status": status, 
                                       "evidence": evidence, "repair": repair}, default=str))
    raise SystemExit

# Step 2: Get target region for the basket
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_name = target.get("target_name")
target_position = target.get("target_position")
print(f"Target region: {target_name}, position: {target_position}")

# Step 3: Execute guarded transport to the target region
transport_result = guarded_transport_held_object_to_target(
    object_name, predicate=predicate, target_name=target_name
)
print(f"Transport result: {transport_result}")

# Step 4: Verify transport success
held_after = bool(transport_result.get("held_after", False))
arrived = bool(transport_result.get("arrived", False))
xy_distance = float(transport_result.get("xy_distance_to_target", float('inf')))
xy_tolerance = float(transport_result.get("xy_tolerance", 0.1))

# Check additional grasp state after transport
post_grasp = measure_object_grasp_state(object_name)
still_held = bool(post_grasp.get("held") or post_grasp.get("strict_held") or 
                  post_grasp.get("held_by_contact") or post_grasp.get("elevated_contact"))

# Get object and end-effector poses for evidence
ee_pos, ee_quat = get_ee_pose()
obj_pos, obj_quat = get_object_pose(object_name)

# Determine stage status
if held_after and arrived and still_held and xy_distance <= xy_tolerance:
    status = "satisfied"
    evidence = {
        "held_before": held_before,
        "held_after": held_after,
        "arrived": arrived,
        "xy_distance_to_target": xy_distance,
        "xy_tolerance": xy_tolerance,
        "post_grasp_state": post_grasp,
        "ee_pose": {"position": ee_pos, "quaternion": ee_quat},
        "object_pose": {"position": obj_pos, "quaternion": obj_quat},
        "transport_result": transport_result
    }
    repair = ""
else:
    status = "not_satisfied"
    evidence = {
        "held_before": held_before,
        "held_after": held_after,
        "arrived": arrived,
        "xy_distance_to_target": xy_distance,
        "xy_tolerance": xy_tolerance,
        "still_held": still_held,
        "post_grasp_state": post_grasp,
        "ee_pose": {"position": ee_pos, "quaternion": ee_quat},
        "object_pose": {"position": obj_pos, "quaternion": obj_quat},
        "transport_result": transport_result
    }
    # Determine repair based on failure mode
    if not held_after or not still_held:
        repair = "Object slipped during transport; return to pick_lift_verify stage"
    elif not arrived:
        repair = "Transport did not reach target region; retry transport or replan waypoint"
    else:
        repair = f"xy_distance {xy_distance:.3f} exceeds tolerance {xy_tolerance:.3f}; retry transport"

# Final status output
print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair
}, default=str))

# Stage code block 3
STAGE_ID = "obj1_alphabet_soup_low_release_verify"
STAGE_ATTEMPT = 1
import json
import numpy as np

# Stage-specific configuration
object_name = "alphabet_soup"
predicate = "in"
target_name = "basket_1_contain_region"  # From BDDL goal diagnostics

# Get current state
print("=== Pre-release observations ===")
ee_pos, ee_quat = get_ee_pose()
print(f"End-effector pose: {ee_pos}, {ee_quat}")

# Get object grasp state to confirm it's still held
grasp_state = measure_object_grasp_state(object_name)
held = bool(grasp_state.get("held", False) or grasp_state.get("transport_stable_held", False))
print(f"Object grasp state: {grasp_state}")

# Get target region pose for grounding
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_pos = target["target_position"]
target_site = target["target_name"]
print(f"Target region: {target_site}, position: {target_pos}")

# Get current object pose for diagnostic
obj_pos, obj_quat = get_object_pose(object_name)
print(f"Current object position: {obj_pos}")

# Check current relation diagnostic before release
relation_before = get_site_in_diagnostic(object_name, target_site)
print(f"Relation diagnostic before release: {relation_before}")

# Verify we are above the target region (transport should have arrived)
xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
print(f"XY distance to target center: {xy_dist:.4f}")

if not held:
    print("ERROR: Object not held before release attempt")
    STAGE_STATUS = {
        "stage_id": STAGE_ID,
        "status": "needs_repair",
        "evidence": {"held": held, "grasp_state": grasp_state},
        "repair": "Object lost during transport, return to pick_lift_verify stage"
    }
    print("STAGE_STATUS " + json.dumps(STAGE_STATUS, default=str))
    raise SystemExit

# Perform low release to target
print("\n=== Performing low release ===")
release_result = low_release_object_to_target(
    object_name=object_name,
    predicate=predicate,
    target_name=target_name,
    release_z=0.1,  # Release height above target
    approach=0.05,  # Approach distance
    retreat=0.08,   # Retreat distance after release
    settle_steps=50
)

print(f"Release result: {release_result}")

# Post-release observations
print("\n=== Post-release observations ===")
obj_pos_after, obj_quat_after = get_object_pose(object_name)
print(f"Object position after release: {obj_pos_after}")

# Check relation diagnostic after release
relation_after = get_site_in_diagnostic(object_name, target_site)
print(f"Relation diagnostic after release: {relation_after}")

# Check if object is no longer held
grasp_after = measure_object_grasp_state(object_name)
held_after = bool(grasp_after.get("held", False) or grasp_after.get("transport_stable_held", False))
print(f"Object held after release: {held_after}")

# Verify placement relation
relation_satisfied = bool(relation_after.get("in", False))
print(f"Relation 'in' satisfied: {relation_satisfied}")

# If relation not satisfied, try bounded local placement closure
if not relation_satisfied and not held_after:
    print("\n=== Attempting local placement closure ===")
    closure_result = local_placement_closure(
        object_name=object_name,
        predicate=predicate,
        target_name=target_name,
        max_iters=2
    )
    print(f"Closure result: {closure_result}")
    
    # Re-check relation after closure
    relation_final = get_site_in_diagnostic(object_name, target_site)
    relation_satisfied = bool(relation_final.get("in", False))
    print(f"Relation after closure: {relation_final}")

# Final status determination
if held_after:
    # Object still held after release attempt - release failed
    status = "needs_repair"
    evidence = {
        "held_after": held_after,
        "release_result": release_result,
        "relation_after": relation_after,
        "reason": "Object still held after release attempt"
    }
    repair = "Release failed, may need lower release height or different approach"
elif relation_satisfied:
    status = "satisfied"
    evidence = {
        "relation_satisfied": relation_satisfied,
        "held_after": held_after,
        "release_result": release_result,
        "relation_after": relation_after,
        "final_object_pose": [obj_pos_after[0], obj_pos_after[1], obj_pos_after[2]],
        "target_position": [target_pos[0], target_pos[1], target_pos[2]]
    }
    repair = ""
else:
    status = "not_satisfied"
    evidence = {
        "relation_satisfied": relation_satisfied,
        "held_after": held_after,
        "release_result": release_result,
        "relation_after": relation_after,
        "final_object_pose": [obj_pos_after[0], obj_pos_after[1], obj_pos_after[2]],
        "target_position": [target_pos[0], target_pos[1], target_pos[2]],
        "reason": "Placement relation not satisfied after release and closure attempts"
    }
    repair = "Object not in target region, may need repositioning or different release strategy"

# Print final stage status
STAGE_STATUS = {
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair
}
print("\nSTAGE_STATUS " + json.dumps(STAGE_STATUS, default=str))