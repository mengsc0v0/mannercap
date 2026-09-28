from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.motion.pyroki import init_pyroki


# ------------------------------- Control API ------------------------------
class FrankaControlPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv, multi_turn: bool = False) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None
        self.multi_turn = multi_turn

    def functions(self) -> dict[str, Any]:
        base_functions = {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            # "home_pose": self.home_pose,
        }
        # if self.multi_turn:
        #     base_functions["breakpoint_code_block"] = self.breakpoint_code_block
        return base_functions

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        obs = self._env.get_observation()

        if (
            "red" in object_name and "cube" in object_name
        ):  # TODO: Slightly problematic that these are hardcoded language descriptions
            # Could just expose a low level obs dict item with the same object name from the low level env
            return (
                obs["cube_poses"]["primary"][:3],
                obs["cube_poses"]["primary"][3:],
                np.array([0.05, 0.05, 0.05]),
            )
        elif "green" in object_name and "cube" in object_name:
            return (
                obs["cube_poses"]["secondary"][:3],
                obs["cube_poses"]["secondary"][3:],
                np.array([0.05, 0.05, 0.05]),
            )
        elif "cup" in object_name or "mug" in object_name or "water" in object_name:
            return (
                obs["cup_poses"]["primary"][:3],
                obs["cup_poses"]["primary"][3:],
                np.array([0.07, 0.07, 0.06]),
            )
        elif "target" in object_name or "desk" in object_name or "spot" in object_name or "boss" in object_name:
            return (
                obs["target_poses"]["primary"][:3],
                obs["target_poses"]["primary"][3:],
                np.array([0.12, 0.12, 0.002]),
            )
        elif "peg" in object_name:
            return (
                obs["nut_poses"]["square_peg"][:3],
                obs["nut_poses"]["square_peg"][3:],
                None,
            )
        elif "nut" in object_name:
            key = "square_nut_handle" if "handle" in object_name else "square_nut"
            return (
                obs["nut_poses"][key][:3],
                obs["nut_poses"][key][3:],
                np.array([0.03, 0.03, 0.01]),
            )
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        if "red" in object_name and "cube" in object_name:
            return obs["cube_poses"]["primary"][:3], np.array([0, 0, 1, 0])
        elif "green" in object_name and "cube" in object_name:
            return obs["cube_poses"]["secondary"][:3], np.array([0, 0, 1, 0])
        elif "cup" in object_name or "mug" in object_name or "water" in object_name:
            return obs["cup_poses"]["primary"][:3], np.array([0, 0, 1, 0])
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        rot = SciRotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        )

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            self._solve_and_move(quat_wxyz, z_offset_pos)

        self._solve_and_move(quat_wxyz, offset_pos)

    def _solve_and_move(self, quat_wxyz: np.ndarray, target_pos: np.ndarray) -> None:
        """Solve IK and move to target position (helper to reduce goto_pose duplication)."""
        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=40)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=60)


    def goto_pose_with_profile(
        self,
        position,
        quaternion_wxyz,
        z_approach=0.0,
        *,
        v_max=0.15,
        a_max=0.10,
        jerk_limit=None,
        phase="move",
    ):
        """Manner-aware goto_pose for RoboSuite environments (Manner A + D).

        Args:
            position:        (3,) XYZ in metres (robot-base frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach:      Z offset pre-approach (same as goto_pose).
            v_max:           Max joint-space speed rad/s.
                             缓缓靠近 -> 0.03, 正常 -> 0.15, 迅速 -> 0.25.
            a_max:           Max joint-space acceleration rad/s^2.
            jerk_limit:      Minimum-jerk quintic when set (Manner D: 平稳).
                             None -> trapezoidal. Typical: 0.5 smooth – 5.0.
        """
        import numpy as _np
        from scipy.spatial.transform import Rotation as _Rot

        pos = _np.asarray(position, dtype=_np.float64).reshape(3)
        quat_wxyz = _np.asarray(quaternion_wxyz, dtype=_np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        rot = _Rot.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(_np.array([0.0, 0.0, -z_approach]))
            self._solve_and_move_profiled(quat_wxyz, z_offset_pos, v_max, a_max, jerk_limit, phase)

        self._solve_and_move_profiled(quat_wxyz, offset_pos, v_max, a_max, jerk_limit, phase)

    def _solve_and_move_profiled(self, quat_wxyz, target_pos, v_max, a_max, jerk_limit, phase="move"):
        """Solve IK then execute with velocity profile (internal helper)."""
        import numpy as _np
        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=_np.concatenate([quat_wxyz, target_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=_np.concatenate([quat_wxyz, target_pos]),
                prev_cfg=self.cfg,
            )
        joints = _np.asarray(self.cfg[:-1], dtype=_np.float64).reshape(7)
        self._env.move_to_joints_profiled(
            joints, v_max=v_max, a_max=a_max, jerk_limit=jerk_limit, phase=phase
        )

    def set_impedance_stiffness(self, kp_pos=150.0, kp_ori=None):
        """Adjust arm compliance (Manner B: 阻抗刚度).

        Args:
            kp_pos:  Joint position gain.  300-600 -> 轻轻, 800-1200 -> 正常,
                     1300-1500 -> 用力.
            kp_ori:  Accepted for API symmetry; not separately applied.
        """
        self._env.set_joint_stiffness(kp_pos)

    def close_gripper_partial(self, *, force=None, width=None, steps=60):
        """Grip with partial force or to a target width (Manner C: 夹持力矩).

        Args:
            force:  Normalised grip force [0, 1]. 0->虚握, 1->紧握.
            width:  Finger-gap metres. Width overrides force.
            steps:  Simulation steps to hold at target.
        """
        self._env.close_gripper_partial(force=force, width=width, steps=steps)

    def home_pose(self) -> None:
        """
        Move the robot to a safe home pose.
        Args:
            None
        Returns:
            None
        """

        # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ]
        )
        self._env.move_to_joints_blocking(joints)
