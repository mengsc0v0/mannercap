# Code block 0
import numpy as np
from your_api import *  # Replace with actual import statements if needed

# Step 1: Arm 0 picks up the hammer
hammer_pos, hammer_quat = get_hammer_pose()
print(f"Hammer position: {hammer_pos}")

# Calculate gripper approach position (above hammer)
approach_pos = hammer_pos.copy()
approach_pos[2] += 0.15  # 15cm above hammer

# Arm 0 approaches with gripper facing down, opening along X-axis (to grasp handle from sides)
approach_quat = np.array([0, 0.707, 0.707, 0])  # WXYZ

# Move to approach position first
goto_pose_arm0(approach_pos, approach_quat)

# Descend to grasp position
grasp_pos = hammer_pos.copy()
grasp_pos[2] += 0.02  # 2cm above table (safe grasp height)
goto_pose_arm0(grasp_pos, approach_quat)
close_gripper_arm0()

# Lift hammer to handover height
lift_pos = grasp_pos.copy()
lift_pos[2] = 0.175  # Midpoint of required range
goto_pose_arm0(lift_pos, approach_quat)

# Step 2: Prepare for handover
# Calculate handover position (midpoint between arms, safe distance from table gap)
handover_x = (0.44 + 1.18) / 2  # Midpoint of arm x-coordinates
handover_y = 0.0  # Keep centered
handover_z = 0.175  # Within required range

# Orientation: rotate hammer handle to be accessible to Arm 1
# Arm 1's default orientation opens along Y-axis, so rotate hammer to be perpendicular
# Current hammer orientation: aligned with Y-axis (handle forward)
# Desired: rotate 90° around Z-axis so handle points toward +X (toward Arm 1)
handover_quat = np.array([0.707, 0, 0, 0.707])  # 90° rotation around Z-axis

# Move to handover position
handover_pos = np.array([handover_x, handover_y, handover_z])
goto_pose_arm0(handover_pos, handover_quat)

# Step 3: Arm 1 grasps hammer
# Arm 1's gripper needs to be oriented to grasp handle from sides
# Use Arm 1's reference orientation: facing down, opening along Y-axis
arm1_quat = np.array([0, 0, 1, 0])  # WXYZ

# Calculate Arm 1 approach position
arm1_approach_pos = handover_pos.copy()
arm1_approach_pos[2] += 0.10  # 10cm above handover position

# Arm 1 moves to approach position
goto_pose_arm1(arm1_approach_pos, arm1_quat)

# Arm 1 descends to grasp position (aligned with hammer handle)
arm1_grasp_pos = handover_pos.copy()
# Offset slightly toward handle (handle is toward +X from center)
arm1_grasp_pos[0] += 0.05  # Grasp 5cm from center along handle
goto_pose_arm1(arm1_grasp_pos, arm1_quat)
close_gripper_arm1()

# Step 4: Complete handover
# Arm 0 releases hammer
open_gripper_arm0()

# Arm 0 moves away to avoid collision
arm0_retreat_pos = lift_pos.copy()
arm0_retreat_pos[0] -= 0.2  # Move away from handover zone
goto_pose_arm0(arm0_retreat_pos, approach_quat)

# Arm 1 lifts hammer slightly to confirm successful grasp
arm1_lift_pos = arm1_grasp_pos.copy()
arm1_lift_pos[2] += 0.10  # Lift 10cm
goto_pose_arm1(arm1_lift_pos, arm1_quat)

# Verify hammer is in correct height range
final_hammer_pos, _ = get_hammer_pose()
if 0.15 <= final_hammer_pos[2] <= 0.20:
    print("SUCCESS: Handover completed with hammer in height range")
else:
    print(f"WARNING: Hammer at height {final_hammer_pos[2]}m (outside 0.15-0.20m range)")