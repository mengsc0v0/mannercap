from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling MFR_benchmark's Victor/iiwa-style 7-DoF arm with an Allegro dexterous hand.
Goal: first move the hand/tool into the screwdriver pregrasp/contact pose, then grasp and turn the screwdriver while keeping the screwdriver top close to its nominal axis.
Use the high-level MFR screwdriver APIs first: inspect state, move to standoff, approach the screwdriver, align fingertips, grasp, settle/insert the tool tip, turn, and verify.
MFR starts in a near-contact pregrasp; if the screwdriver top is already close to the nominal axis, prefer the guarded axis_preserving_finger_pronation primitive before any large approach or wrist-turn motion.
This environment is currently headless; use numeric state feedback rather than images.
Only write executable Python code and do not wrap it in code fences.
"""


ORACLE_CODE = """
before = get_task_state()
print("before_angle_delta", before["screwdriver_angle_delta"])
print("before_top_error", before["screwdriver_top_error"])
axis_preserving_finger_pronation(steps=1, joint_delta=0.35)
RESULT = get_task_state()
print("after_angle_delta", RESULT["screwdriver_angle_delta"])
print("after_top_error", RESULT["screwdriver_top_error"])
print("success", verify_screwdriver_turning())
"""


class MfrScrewdriverCodeEnv(CodeExecutionEnvBase):
    """High-level code execution environment for MFR screwdriver turning."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["MfrScrewdriverCodeEnv"]
