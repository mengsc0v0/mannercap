# Stage code block 0
STAGE_ID = "obj1_wine_bottle_ground_object_target"
STAGE_ATTEMPT = 1
import json
import numpy as np


# Step 1: Get observations and grounding
obs = get_observation()
task_status = get_task_status()
print(f"[{STAGE_ID}] Observing scene...")
print(f"  Task: {obs.get('task', 'unknown')}")
print(f"  Task completed: {task_status.get('task_completed', False)}")
print(f"  Reward: {task_status.get('reward', 0.0)}")

# Step 2: Find and ground the wine bottle
wine_bottle_name = None
all_objects = get_all_object_poses()
for obj_name, obj_pose in all_objects.items():
    if "wine" in obj_name.lower() and "bottle" in obj_name.lower():
        wine_bottle_name = obj_name
        pos, quat = obj_pose
        print(f"  Found wine bottle: {obj_name}")
        print(f"    Position: {pos}")
        print(f"    Quaternion: {quat}")
        break

if wine_bottle_name is None:
    # Try alternative naming
    for obj_name in all_objects.keys():
        if "wine_bottle" in obj_name.lower():
            wine_bottle_name = obj_name
            pos, quat = all_objects[obj_name]
            print(f"  Found wine bottle (alt name): {obj_name}")
            print(f"    Position: {pos}")
            print(f"    Quaternion: {quat}")
            break

# Step 3: Ground the target rack using BDDL goal
bddl_goal = get_bddl_goal_diagnostics()
target_info = None
for goal in bddl_goal.get("goal_state", []):
    if "wine_bottle" in goal.get("object_name", ""):
        target_info = goal
        break

# Step 4: Also get the target pose from API
try:
    target_pose_data = get_goal_target_region_pose(wine_bottle_name, predicate="on")
    target_name_from_api = target_pose_data.get("target_name")
    target_pos_from_api = target_pose_data.get("target_position")
    print(f"  Target from API: {target_name_from_api}")
    print(f"    Position: {target_pos_from_api}")
except Exception as e:
    print(f"  Error getting target pose: {e}")
    target_name_from_api = None
    target_pos_from_api = None

# Step 5: Compare and verify consistency
evidence_parts = []
repair_msg = ""
status = "satisfied"

if wine_bottle_name is None:
    status = "not_satisfied"
    evidence_parts.append("wine_bottle_not_found")
    repair_msg = "Cannot find wine bottle in scene"
else:
    # Check if wine bottle is in the BDDL goal
    if target_info is not None:
        evidence_parts.append(f"bddl_object={target_info.get('object_name')}")
        evidence_parts.append(f"bddl_target={target_info.get('target_name')}")
        evidence_parts.append(f"bddl_predicate={target_info.get('predicate')}")
    else:
        evidence_parts.append("no_bddl_goal_for_wine_bottle")
        status = "needs_repair"
        repair_msg = "Wine bottle not found in BDDL goal"
    
    # Check if target from API matches BDDL target
    if target_name_from_api is not None:
        evidence_parts.append(f"api_target={target_name_from_api}")
        if target_info is not None and target_name_from_api != target_info.get("target_name"):
            status = "needs_repair"
            repair_msg = f"Target mismatch: API={target_name_from_api}, BDDL={target_info.get('target_name')}"
    else:
        evidence_parts.append("api_target_not_found")
        status = "needs_repair"
        repair_msg = "Could not get target from API"
    
    # Check reachability
    if target_pos_from_api is not None and wine_bottle_name is not None:
        bottle_pos = all_objects.get(wine_bottle_name, [None, None])[0]
        if bottle_pos is not None:
            distance = np.linalg.norm(np.array(bottle_pos) - np.array(target_pos_from_api))
            evidence_parts.append(f"distance_to_target={distance:.3f}")
            if distance > 2.0:  # Unreasonably far
                status = "needs_repair"
                repair_msg = f"Wine bottle too far from target: {distance:.2f}m"

# Step 6: Check relation predicate
relation = None
if target_info:
    relation = target_info.get("predicate")
    evidence_parts.append(f"relation_predicate={relation}")

# Print evidence
evidence = "; ".join(evidence_parts)
print(f"  Evidence: {evidence}")
print(f"  Status: {status}")
if repair_msg:
    print(f"  Repair hint: {repair_msg}")

# Step 7: Final status
print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair_msg
}, default=str))

# Stage code block 1
STAGE_ID = "obj1_wine_bottle_pick_lift_verify"
STAGE_ATTEMPT = 1
import json
import numpy as np

# Get initial object pose for evidence
object_name = "wine_bottle_1"
initial_obj_pos, initial_obj_quat = get_object_pose(object_name)
print(f"Initial object pose: {initial_obj_pos}")

# Get initial gripper pose for evidence
initial_ee_pos, initial_ee_quat = get_ee_pose()
print(f"Initial gripper pose: {initial_ee_pos}")

# Perform the pick-lift operation using the unified contract
# The stage_attempt is provided as STAGE_ATTEMPT (global)
grasp = verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1)

# Get post-grasp evidence
post_grasp_obj_pos, post_grasp_obj_quat = get_object_pose(object_name)
post_grasp_ee_pos, post_grasp_ee_quat = get_ee_pose()
grasp_state = measure_object_grasp_state(object_name)

# Build evidence string
evidence = (
    f"object={object_name}; "
    f"initial_obj_z={initial_obj_pos[2]:.3f}; "
    f"post_grasp_obj_z={post_grasp_obj_pos[2]:.3f}; "
    f"z_lift={post_grasp_obj_pos[2] - initial_obj_pos[2]:.3f}; "
    f"grasp_held={grasp.get('held')}; "
    f"lift_delta={grasp.get('lift_delta'):.3f}; "
    f"transport_stable={grasp.get('transport_stable_held')}; "
    f"grasp_state_held={grasp_state.get('held')}; "
    f"final_grasp_skill={grasp.get('skill')}"
)

# Check success conditions
if (grasp.get('held') and grasp.get('lift_delta', 0) >= 0.05) or grasp.get('transport_stable_held'):
    # Pick-lift successful
    status = "satisfied"
    repair = ""
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))
else:
    # Pick-lift failed - needs repair
    status = "needs_repair"
    repair = "Pick-lift verification failed. Let the stage loop retry with next attempt."
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))

# Stage code block 2
STAGE_ID = "obj1_wine_bottle_transport_guarded"
STAGE_ATTEMPT = 1
import json
import numpy as np

# Current stage constants
object_name = "wine_bottle_1"
predicate = "on"  # from BDDL goal

# 1. Get current grasp state to confirm object is held
grasp_state = measure_object_grasp_state(object_name)
held = bool(grasp_state.get("held", False) or grasp_state.get("held_by_contact", False) or grasp_state.get("elevated_contact", False))
print("Initial grasp state:", json.dumps(grasp_state, default=str))

if not held:
    status = "needs_repair"
    evidence = f"object_held={held}"
    repair = "Object is not held; return to pick_lift_verify stage"
    print("STAGE_STATUS " + json.dumps({
        "stage_id": STAGE_ID,
        "status": status,
        "evidence": evidence,
        "repair": repair
    }, default=str))
    raise SystemExit

# 2. Get target region pose for the wine bottle with predicate 'on'
target = get_goal_target_region_pose(object_name, predicate=predicate)
target_name = target["target_name"]
target_pos = target["target_position"]
print("Target info:", json.dumps(target, default=str))

# 3. Check current EE and object positions for grounding
ee_pos, ee_quat = get_ee_pose()
obj_pos, obj_quat = get_object_pose(object_name)
print("EE pose:", json.dumps({"pos": ee_pos, "quat": ee_quat}, default=str))
print("Object pose:", json.dumps({"pos": obj_pos, "quat": obj_quat}, default=str))

# 4. Perform guarded transport to target
# The transport skill returns: held_before, held_after, arrived, xy_distance_to_target, reward, task_completed
transport_result = guarded_transport_held_object_to_target(
    object_name,
    predicate=predicate,
    target_name=target_name
)
print("Transport result:", json.dumps(transport_result, default=str))

# 5. Extract key transport results
held_before = bool(transport_result.get("held_before", False))
held_after = bool(transport_result.get("held_after", False))
arrived = bool(transport_result.get("arrived", False))
xy_distance = float(transport_result.get("xy_distance_to_target", float('inf')))

# 6. Verify transport stage success conditions
# Success: object remained held, arrived at target (within xy_tolerance), even if not yet placed.
if held_before and held_after and arrived:
    # Even if reward=0 and task_completed=False, transport stage is satisfied.
    status = "satisfied"
    evidence = (f"held_before={held_before}; held_after={held_after}; arrived={arrived}; "
                f"xy_distance_to_target={xy_distance:.4f}; target_name={target_name}")
    repair = ""
else:
    status = "needs_repair"
    evidence = (f"held_before={held_before}; held_after={held_after}; arrived={arrived}; "
                f"xy_distance_to_target={xy_distance:.4f}")
    if not held_after:
        repair = "Object slipped during transport; return to pick_lift_verify stage"
    else:
        repair = "Transport did not arrive at target; retry or replan waypoint"

# 7. Final status print
print("STAGE_STATUS " + json.dumps({
    "stage_id": STAGE_ID,
    "status": status,
    "evidence": evidence,
    "repair": repair
}, default=str))

# Stage code block 3
STAGE_ID = "obj1_wine_bottle_low_release_verify"
STAGE_ATTEMPT = 1
import json
import numpy as np

# Get current state
print("Current state observations:")
ee_pos, ee_quat = get_ee_pose()
obj_pos, obj_quat = get_object_pose("wine_bottle_1")
task_status = get_task_status()

print(f"End-effector position: {ee_pos}")
print(f"Object position: {obj_pos}")
print(f"Task status: {task_status}")

# Get target region
target = get_goal_target_region_pose("wine_bottle_1", predicate="on")
print(f"Target region: {target}")

# Use the rack-specific release function for rack/caddy targets
target_name = target["target_name"]
print(f"Attempting rack_supported_release to {target_name}")

# Call the rack_supported release
result = rack_supported_release_object_to_target("wine_bottle_1", target_name=target_name)
print(f"Release result: {result}")

# Check post-release state
obj_pose_after, obj_quat_after = get_object_pose("wine_bottle_1")
ee_pose_after, ee_quat_after = get_ee_pose()
print(f"Post-release object position: {obj_pose_after}")
print(f"Post-release end-effector position: {ee_pose_after}")

# Check placement relation
on_diagnostic = get_on_diagnostic("wine_bottle_1", "wine_rack_1_top_region")
print(f"On diagnostic: {on_diagnostic}")

# Check if object is held
grasp_state = measure_object_grasp_state("wine_bottle_1")
print(f"Grasp state after release: {grasp_state}")

# Verify success conditions
object_released = not grasp_state.get("held", True) if grasp_state else True
relation_satisfied = on_diagnostic.get("on", False) if on_diagnostic else False

# Check task completion
completed = bool(task_status.get("task_completed", False))

# Determine stage status
if object_released and relation_satisfied:
    status = "satisfied"
    evidence = f"object_released={object_released}; relation_satisfied={relation_satisfied}; task_completed={completed}"
    repair = ""
elif not object_released:
    status = "needs_repair"
    evidence = f"object_released={object_released}; grasp_state={grasp_state}"
    repair = "Object still held after release attempt"
else:
    status = "not_satisfied"
    evidence = f"object_released={object_released}; relation_satisfied={relation_satisfied}"
    repair = "Placement relation not satisfied, try lower release height"

print(f"Stage status: {status}")
print(f"Evidence: {evidence}")

# Final status output
print("STAGE_STATUS " + json.dumps({
    "stage_id": "obj1_wine_bottle_low_release_verify",
    "status": status,
    "evidence": evidence,
    "repair": repair
}, default=str))