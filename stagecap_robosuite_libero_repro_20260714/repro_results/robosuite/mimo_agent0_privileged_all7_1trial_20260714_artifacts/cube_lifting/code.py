# Code block 0
import numpy as np

# Step 1: Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose("red cube")

# Step 2: Approach from above with a small offset (0.1 meters)
approach_offset = 0.1
goto_pose(grasp_position, grasp_quaternion, z_approach=approach_offset)

# Step 3: Open gripper to prepare for grasping
open_gripper()

# Step 4: Close gripper to grasp the cube
close_gripper()

# Step 5: Lift the cube by moving upward (add 0.2m to current Z position)
lift_offset = 0.2
lift_position = grasp_position.copy()
lift_position[2] += lift_offset
# Use a safe orientation (gripper down) for lifting
lift_quaternion = np.array([0.0, 0.0, 1.0, 0.0])  # WXYZ
goto_pose(lift_position, lift_quaternion)