from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a fixed-base Kuka arm with an Allegro dexterous hand in a nut-bolt screw task.
Goal: first move from a high standoff pose to the table-mounted nut, close the Allegro hand around the nut, then perform repeated screw-turn and regrasp cycles until the nut has turned at least one full turn.
Use the high-level KukaAllegroNutBolt APIs first. Prefer plan_nut_contact_approach and approach_nut_contact_pose for fine-grained target-object approach; then use close_on_nut, screw_turn, regrasp, and verify_screwed.
This environment tracks unwrapped nut_joint command progress from the validated Isaac Gym demo; use numeric state feedback for CAP and ViFailback diagnosis.
Only write executable Python code and do not wrap it in code fences.
"""


ORACLE_CODE = """
move_to_standoff()
approach_nut_contact_pose(side="front", close_for_contact=False)
close_on_nut()
for _ in range(3):
    screw_turn(angle_delta_rad=3.1)
    if verify_screwed():
        break
    regrasp()
    approach_nut_contact_pose(side="front", close_for_contact=False)
    close_on_nut()
RESULT = get_task_state()
print("phase", RESULT["phase"])
print("approach_feedback", RESULT["approach_feedback"])
print("turns", RESULT["nut_joint_progress_turns"])
print("min_fingertip_distance", RESULT["min_fingertip_nut_distance_m"])
print("success", verify_screwed())
"""


class KukaAllegroNutBoltCodeEnv(CodeExecutionEnvBase):
    """High-level code execution environment for Kuka+Allegro nut-bolt screwing."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["KukaAllegroNutBoltCodeEnv"]
