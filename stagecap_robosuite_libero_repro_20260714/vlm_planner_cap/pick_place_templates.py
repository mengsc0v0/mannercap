"""Generic stage templates for pick/place style LIBERO tasks."""

from __future__ import annotations

from .schema import StageCheck, StagePlan, TaskPlan


def build_pick_place_plan(task: str) -> TaskPlan:
    """Return a conservative generic pick/place stage plan."""
    stages = [
        StagePlan(
            stage_id="localize_object_and_target",
            name="Localize object and target",
            intent="Identify the exact task object(s) and target region or receptacle before moving.",
            allowed_primitives=["get_observation"],
            target_references=["task_objects", "task_target", "gripper_pose"],
            checks=[
                StageCheck(
                    name="object_target_grounded",
                    description="The task object and target are both grounded in the current scene.",
                    check_type="hybrid",
                    success_criteria=[
                        "the selected object matches the task language",
                        "the target/receptacle/region matches the task language",
                        "read-only coordinates or perception points are plausible",
                    ],
                ),
            ],
            repair_hints=[
                "If object names are ambiguous, print visible objects and choose the exact named object.",
                "Reject points outside the table workspace or with impossible height.",
            ],
        ),
        StagePlan(
            stage_id="approach_object",
            name="Approach object",
            intent="Move the gripper to a safe pre-grasp pose near the object.",
            allowed_primitives=["goto_pose_checked", "move_linear_delta", "solve_ik", "move_to_joints"],
            target_references=["object_pose", "pregrasp_pose"],
            checks=[
                StageCheck(
                    name="near_object",
                    description="The gripper is close enough and properly oriented for grasping.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper-object distance is below a pre-grasp threshold",
                        "the gripper is not colliding with the table or target",
                    ],
                ),
            ],
        ),
        StagePlan(
            stage_id="grasp_object",
            name="Grasp object",
            intent="Open, descend or advance to the object, close the gripper, and confirm the object is held.",
            allowed_primitives=["open_gripper", "close_gripper", "goto_pose_checked", "move_linear_delta"],
            target_references=["object_pose", "gripper_width"],
            checks=[
                StageCheck(
                    name="object_grasped",
                    description="The object moves with the gripper after closing.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper closes around the object",
                        "object remains near the gripper after a small lift",
                    ],
                ),
            ],
            repair_hints=[
                "If the object slips, reopen, re-approach with a lower or more centered grasp, and retry.",
            ],
        ),
        StagePlan(
            stage_id="lift_object",
            name="Lift object",
            intent="Lift the grasped object high enough to clear obstacles.",
            allowed_primitives=["move_linear_delta", "goto_pose_checked"],
            target_references=["gripper_pose", "object_pose"],
            checks=[
                StageCheck(
                    name="object_lifted",
                    description="The object is held above its original support surface.",
                    check_type="hybrid",
                    success_criteria=[
                        "object height increases",
                        "object remains close to the gripper",
                    ],
                ),
            ],
        ),
        StagePlan(
            stage_id="transport_to_target",
            name="Transport to target",
            intent="Move the grasped object to a pre-place pose above or inside the target.",
            allowed_primitives=["goto_pose_checked", "move_linear_delta", "solve_ik", "move_to_joints"],
            target_references=["target_pose", "object_pose", "gripper_pose"],
            checks=[
                StageCheck(
                    name="above_target",
                    description="The held object is positioned above or inside the target region.",
                    check_type="hybrid",
                    success_criteria=[
                        "object projection lies inside the target/receptacle footprint",
                        "object is not colliding with irrelevant objects",
                    ],
                ),
            ],
        ),
        StagePlan(
            stage_id="release_and_verify",
            name="Release and verify",
            intent="Release the object, retreat, and verify the target relation.",
            allowed_primitives=["open_gripper", "move_linear_delta", "goto_pose_checked"],
            target_references=["target_pose", "object_pose"],
            checks=[
                StageCheck(
                    name="placement_satisfied",
                    description="The object remains at the target after release and gripper retreat.",
                    check_type="hybrid",
                    success_criteria=[
                        "object stays in/on/near the specified target",
                        "gripper is no longer holding the object",
                    ],
                ),
            ],
            repair_hints=[
                "If the object is near but outside the target, regrasp and retry with a lower release height.",
            ],
        ),
    ]
    return TaskPlan(
        task=task,
        stages=stages,
        notes=[
            "This is a generic pick/place template; exact object and target names must come from the task.",
            "All movement must be performed through clean robot action primitives.",
        ],
    )

