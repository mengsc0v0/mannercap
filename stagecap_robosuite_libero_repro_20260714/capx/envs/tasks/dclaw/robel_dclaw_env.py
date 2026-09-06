from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a ROBEL D'Claw dexterous hand with the APIs described below.
Goal: rotate the screw/object so the signed target_error is close to zero and verify success.
Use the higher-level DClaw APIs first. For screw tasks, `run_reference_screw_policy` is the validated reference primitive; for repairs, use contact force, tripod rolling, target-error tracking, and re-check task state after each motion.
You may write python code comments for reasoning but ONLY write executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, import it explicitly.
"""


ORACLE_CODE = """
run_reference_screw_policy(steps=160, action_mode="mean_scaled_tanh", scale=0.5)
RESULT = get_task_state()
print("target_error", RESULT["target_error"])
print("success", verify_screw_rotation(threshold=0.2))
"""


class DClawCodeEnv(CodeExecutionEnvBase):
    """High-level code execution environment for ROBEL D'Claw tasks."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["DClawCodeEnv"]
