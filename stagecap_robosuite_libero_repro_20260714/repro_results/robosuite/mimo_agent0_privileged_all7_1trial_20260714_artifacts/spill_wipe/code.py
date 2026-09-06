# Code block 0
import numpy as np

# Get the spill pose and bounding box extents
spill_position, spill_quaternion, bbox_extent = get_object_pose('brown spill', return_bbox_extent=True)

# Calculate wiping bounds using spill extents (axis-aligned with table)
min_x = spill_position[0] - bbox_extent[0]/2
max_x = spill_position[0] + bbox_extent[0]/2
min_y = spill_position[1] - bbox_extent[1]/2
max_y = spill_position[1] + bbox_extent[1]/2

# Set wiping parameters
wipe_z = 0.0
step_size = 0.05  # 5cm steps to avoid large motions
orientation = np.array([0.0, 0.0, 1.0, 0.0])  # wxyz downward orientation

# Generate grid of wiping points
x_points = np.arange(min_x, max_x + step_size, step_size)
y_points = np.arange(min_y, max_y + step_size, step_size)

# Execute back-and-forth wiping pattern
for i, y in enumerate(y_points):
    # Alternate direction each row to minimize large motions
    row_x = x_points if i % 2 == 0 else x_points[::-1]
    for x in row_x:
        position = np.array([x, y, wipe_z])
        goto_pose(position, orientation)