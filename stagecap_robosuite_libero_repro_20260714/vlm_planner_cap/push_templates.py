"""Stage templates for push-to-region tasks."""

from __future__ import annotations

from .schema import StageCheck, StagePlan, TaskPlan


def build_push_plan(task: str) -> TaskPlan:
    """Return a generic push-to-region stage plan."""
    stages = [
        StagePlan(
            stage_id="localize_push_object_and_region",
            name="Localize push object and target region",
            intent="Identify the object to push, the target region, and the desired push direction.",
            allowed_primitives=["get_observation"],
            target_references=["object_pose", "target_region_pose", "gripper_pose"],
            checks=[
                StageCheck(
                    name="push_goal_grounded",
                    description="The object and target region are both grounded before contact.",
                    check_type="hybrid",
                    success_criteria=[
                        "object coordinates are plausible",
                        "target region coordinates are plausible",
                        "push direction points from object toward target region",
                    ],
                ),
            ],
        ),
        StagePlan(
            stage_id="approach_push_contact",
            name="Approach push contact",
            intent="Move the gripper to a contact pose behind the object relative to the target direction.",
            allowed_primitives=["goto_pose_checked", "move_linear_delta"],
            target_references=["object_pose", "push_direction", "contact_pose"],
            checks=[
                StageCheck(
                    name="contact_pose_reached",
                    description="The gripper is at a safe pushing contact pose without grasping the object.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper is behind the object along the push direction",
                        "gripper is near the object at table height",
                    ],
                ),
            ],
        ),
        StagePlan(
            stage_id="push_to_region",
            name="Push to target region",
            intent="Push the object along the target direction until it reaches the specified region.",
            allowed_primitives=["move_linear_delta", "move_along_direction", "goto_pose_checked"],
            target_references=["push_direction", "target_region_pose", "object_pose"],
            checks=[
                StageCheck(
                    name="object_in_region",
                    description="The object is inside or sufficiently close to the target region after pushing.",
                    check_type="hybrid",
                    success_criteria=[
                        "object center moves toward the target region",
                        "object remains on the table",
                        "object ends inside or near the target footprint",
                    ],
                ),
            ],
            repair_hints=[
                "If the object misses laterally, reset contact point and apply a smaller corrected push.",
                "Use several short pushes instead of one large push.",
            ],
        ),
        StagePlan(
            stage_id="retreat_and_verify_push",
            name="Retreat and verify push result",
            intent="Retreat the gripper and verify the object remains in the target region.",
            allowed_primitives=["move_linear_delta", "goto_pose_checked"],
            target_references=["object_pose", "target_region_pose"],
            checks=[
                StageCheck(
                    name="push_relation_stable",
                    description="The pushed object remains at the target after the gripper retreats.",
                    check_type="hybrid",
                    success_criteria=[
                        "object does not move away after retreat",
                        "object still satisfies the target spatial relation visually",
                    ],
                ),
            ],
        ),
    ]
    return TaskPlan(
        task=task,
        stages=stages,
        notes=[
            "This is a generic push template.",
            "The object must be moved by gripper contact, not by writing simulator state.",
        ],
    )

