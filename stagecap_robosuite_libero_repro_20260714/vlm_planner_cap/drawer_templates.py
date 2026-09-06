"""Stage templates for drawer manipulation."""

from __future__ import annotations

from .schema import StageCheck, StagePlan, TaskPlan


def build_drawer_open_plan(task: str, drawer_name: str = "target_drawer") -> TaskPlan:
    """Return a default stage plan for clean drawer opening."""
    stages = [
        StagePlan(
            stage_id="align_handle_height",
            name="Align with handle height",
            intent="Move the gripper center to the same height and lateral line as the drawer handle.",
            allowed_primitives=["goto_pose_checked", "goto_finger_midpoint_checked", "move_linear_delta"],
            target_references=[drawer_name, "drawer_handle_pose", "gripper_pose"],
            checks=[
                StageCheck(
                    name="same_horizontal_line",
                    description="The gripper center is visually and geometrically aligned with the handle line.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper vertical offset from handle is below threshold",
                        "gripper is not above or below the handle in the image",
                    ],
                ),
            ],
            repair_hints=[
                "Adjust height before moving forward toward the handle.",
                "Avoid contacting the drawer face while aligning.",
            ],
        ),
        StagePlan(
            stage_id="approach_handle",
            name="Approach handle",
            intent="Move the gripper close to the handle from the front while keeping the handle between the fingers.",
            allowed_primitives=["goto_pose_checked", "goto_finger_midpoint_checked", "move_linear_delta"],
            target_references=[drawer_name, "drawer_handle_pose", "approach_axis"],
            checks=[
                StageCheck(
                    name="near_handle",
                    description="The gripper is close enough to close around the handle.",
                    check_type="hybrid",
                    success_criteria=[
                        "distance from gripper center to handle is below threshold",
                        "handle appears between or directly in front of the fingers",
                    ],
                ),
            ],
            repair_hints=[
                "If the gripper is offset left or right, translate laterally before closing.",
                "If the gripper is too far, move forward in small increments.",
            ],
        ),
        StagePlan(
            stage_id="close_on_handle",
            name="Close on handle",
            intent="Close the gripper to capture the drawer handle.",
            allowed_primitives=["close_gripper"],
            target_references=[drawer_name, "drawer_handle_pose", "gripper_width"],
            checks=[
                StageCheck(
                    name="handle_captured",
                    description="The handle remains between the gripper fingers after closing.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper width decreased",
                        "handle is still visually inside the gripper aperture",
                        "small test motion does not immediately lose the handle",
                    ],
                ),
            ],
            repair_hints=[
                "If the handle is outside the fingers, reopen and return to approach_handle.",
            ],
        ),
        StagePlan(
            stage_id="test_capture",
            name="Test handle capture",
            intent="Apply a small pull to verify that the drawer follows the gripper.",
            allowed_primitives=["move_linear_delta", "move_along_direction", "move_finger_midpoint_delta"],
            target_references=[drawer_name, "drawer_open_axis", "drawer_progress"],
            checks=[
                StageCheck(
                    name="drawer_responds",
                    description="A small pull changes drawer progress while the gripper stays near the handle.",
                    check_type="hybrid",
                    success_criteria=[
                        "drawer progress increases after the small pull",
                        "relative gripper-handle offset remains small",
                    ],
                ),
            ],
            repair_hints=[
                "If drawer progress does not change, improve the grasp before pulling farther.",
            ],
        ),
        StagePlan(
            stage_id="pull_drawer_open",
            name="Pull drawer open",
            intent="Move along the drawer opening axis until the drawer is open enough for the task.",
            allowed_primitives=["move_linear_delta", "move_along_direction", "move_finger_midpoint_delta"],
            target_references=[drawer_name, "drawer_open_axis", "drawer_progress"],
            checks=[
                StageCheck(
                    name="drawer_open_enough",
                    description="The drawer opening is visibly and geometrically sufficient.",
                    check_type="hybrid",
                    success_criteria=[
                        "drawer progress exceeds the task threshold",
                        "the drawer interior is visible enough for downstream placement if needed",
                    ],
                ),
            ],
            repair_hints=[
                "If the gripper slips, return to approach_handle and close_on_handle.",
                "Pull in multiple small increments rather than one large motion.",
            ],
        ),
    ]
    return TaskPlan(
        task=task,
        stages=stages,
        notes=[
            "This plan assumes coordinates are read-only grounding.",
            "All drawer motion must be produced through gripper motion, not by writing simulator state.",
        ],
    )
