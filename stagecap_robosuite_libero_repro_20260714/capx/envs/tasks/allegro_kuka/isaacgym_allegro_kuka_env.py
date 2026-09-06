from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling an IsaacGymEnvs Kuka arm with an Allegro dexterous hand.
Goal: lift the block and reorient it so object keypoints align with the goal keypoints.
Use high-level AllegroKuka APIs first: read state, open hand, move palm to object, align fingertips, close hand, lift, rotate object to goal, and verify. This environment is currently headless; use numeric state feedback rather than images.
Only write executable Python code and do not wrap it in code fences.
"""


ORACLE_CODE = """
open_hand(steps=2)
move_palm_to_object(steps=4)
align_fingertips_to_object(steps=3)
close_hand_on_object(steps=4)
lift_object(steps=4)
rotate_object_to_goal(steps=5)
RESULT = get_task_state()
print("max_keypoint_dist", RESULT["max_keypoint_dist"])
print("success", verify_object_reorientation())
"""


class AllegroKukaCodeEnv(CodeExecutionEnvBase):
    """High-level code execution environment for IsaacGymEnvs AllegroKuka."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["AllegroKukaCodeEnv"]
