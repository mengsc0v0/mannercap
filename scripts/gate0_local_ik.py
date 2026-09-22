"""Explicit development-only IK backend using a separate MuJoCo data buffer.

Targets the actual gripper EEF body, expressed in the wrapper's base frame.
This is a different backend from PyRoKi, not a reproduction of its solver.
No writes to the live simulation state are made while solving IK.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from types import MethodType


def install_local_ik(api, env, diagnostics):
    sim = env.robosuite_env.sim
    model = sim.model._model
    scratch = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    if any(j < 0 for j in joints):
        raise ValueError("Local IK requires the seven named Panda joints")
    qadr = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    eef = env.gripper_link_idx
    base = env.base_link_idx

    def goto(self, position, quaternion_wxyz, z_approach=0.0, *,
             v_max=0.15, a_max=0.10, jerk_limit=None, phase="move"):
        if z_approach:
            raise ValueError("Gate0 local IK does not implement z_approach")
        q_before = sim.data.qpos.copy()
        v_before = sim.data.qvel.copy()
        time_before = float(sim.data.time)
        checked_fields = ("qpos", "qvel", "ctrl", "act", "qacc", "mocap_pos", "mocap_quat")
        state_before = {name: np.array(getattr(sim.data, name), copy=True) for name in checked_fields}
        scratch.qpos[:] = q_before
        scratch.qvel[:] = v_before
        base_rot = Rotation.from_matrix(sim.data.xmat[base].reshape(3, 3))
        target_pos = sim.data.xpos[base] + base_rot.apply(np.asarray(position))
        wxyz = np.asarray(quaternion_wxyz)
        target_rot = base_rot * Rotation.from_quat(wxyz[[1, 2, 3, 0]])
        q0 = np.clip(q_before[qadr], lower + 1e-6, upper - 1e-6)

        def residual(q):
            scratch.qpos[qadr] = q
            mujoco.mj_forward(model, scratch)
            rotation_error = (target_rot * Rotation.from_matrix(scratch.xmat[eef].reshape(3, 3)).inv()).as_rotvec()
            return np.r_[scratch.xpos[eef] - target_pos, .2 * rotation_error, 1e-4 * (q - q0)]

        solved = least_squares(residual, q0, bounds=(lower + 1e-6, upper - 1e-6),
                               max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solved.x)
        pos_error = float(np.linalg.norm(error[:3]))
        rot_error = float(np.linalg.norm(error[3:6]) / .2)
        untouched = (all(np.array_equal(value, getattr(sim.data, name)) for name, value in state_before.items())
                     and time_before == float(sim.data.time))
        diagnostics.append({"phase": phase, "position_error_m": pos_error,
                            "orientation_error_rad": rot_error, "nfev": int(solved.nfev),
                            "checked_live_state_unchanged_during_ik": untouched,
                            "checked_fields": list(checked_fields) + ["time"],
                            "target_body": "gripper0_right_eef",
                            "controller_snapshot_verified": False})
        if not untouched or pos_error > .002 or rot_error > .02:
            raise RuntimeError(f"Local IK failed: position={pos_error}, rotation={rot_error}, untouched={untouched}")
        env.move_to_joints_profiled(solved.x, v_max=v_max, a_max=a_max,
                                   jerk_limit=jerk_limit, phase=phase)
        diagnostics[-1]["post_motion_position_error_m"] = float(np.linalg.norm(sim.data.xpos[eef] - target_pos))
        diagnostics[-1]["post_motion_orientation_error_rad"] = float(np.linalg.norm(
            (target_rot * Rotation.from_matrix(sim.data.xmat[eef].reshape(3, 3)).inv()).as_rotvec()))
        if (diagnostics[-1]["post_motion_position_error_m"] > .005
                or diagnostics[-1]["post_motion_orientation_error_rad"] > .02):
            raise RuntimeError("Controller did not reach the local IK target within the Gate 0 tolerance")

    api.goto_pose_with_profile = MethodType(goto, api)
