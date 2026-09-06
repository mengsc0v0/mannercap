from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling DexScrew's XHand dexterous hand on a nut-bolt task.
Goal: rotate the nut through physical fingertip contact. You may command only the 12 XHand joints; do not set object pose, nut pose, simulator state, or nut_joint state directly.
Use the high-level DexScrew APIs first: inspect state, run_contact_policy in short chunks, verify_contact_driven_rotation, and diagnose_nutbolt_failure for feedback.
The current controlled container is headless because Isaac Gym camera sensors segfault there, so use numeric state feedback: nut_joint angle delta, nut contact force, fingertip contact force, and audit booleans.
Each task should use at most three feedback rounds.
Only write executable Python code and do not wrap it in code fences.
"""


ORACLE_CODE = """
for round_idx in range(3):
    run_contact_policy(steps=60, stop_on_success=True)
    state = get_task_state()
    print("round", round_idx + 1)
    print("nut_delta_rad", state["nut_dof_delta_rad"])
    print("nut_contact_max", state["nut_contact_force_max"])
    print("fingertip_contact_max", state["fingertip_contact_force_max"])
    print("commanded_nut_joint_action_max", state["commanded_nut_joint_action_max"])
    if verify_contact_driven_rotation():
        break
RESULT = get_task_state()
print("success", verify_contact_driven_rotation())
print("diagnosis", diagnose_nutbolt_failure()["issues"])
"""


class DexScrewXHandNutBoltCodeEnv(CodeExecutionEnvBase):
    """High-level code execution environment for DexScrew XHand nut-bolt."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["DexScrewXHandNutBoltCodeEnv"]
