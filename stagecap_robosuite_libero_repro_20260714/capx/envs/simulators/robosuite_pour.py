"""Low-level Robosuite Franka environment for the "serve water to the boss"
pour/carry task, compatible with FrankaControlPrivilegedApi.

Mirrors the structure of robosuite_cube_lift.py / robosuite_cubes.py, but
wraps the custom PourEnv (capx.envs.simulators.robosuite_pour_env) instead of
a native robosuite Lift/Stack environment, since no built-in pour/liquid task
exists in robosuite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_pour_env import PourEnv


class FrankaRobosuitePourLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka "serve water" (carry-cup-without-spilling) environment."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )

        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = PourEnv(
                    robots=["Panda"],
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                )
            else:
                self.robosuite_env = PourEnv(
                    robots=["Panda"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                )
        else:
            self.robosuite_env = PourEnv(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=load_composite_controller_config(controller=self.controller_cfg),
                horizon=max_steps,
                reward_shaping=True,
            )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        # Adjust initial orientation (same Panda home-pose offset convention
        # used by the lift/stack wrappers, robot-specific not task-specific).
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {
            "task_prompt": "Carry the water cup to the marked spot on the table without spilling it. Quaternions are WXYZ."
        }
        return obs, info

    def _cup_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get cup pose in robot base frame."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cup_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cup_quat"], robosuite_obs["cup_pos"]])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cup_robot_base = base_transform @ cup_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cup_robot_base.translation(), cup_robot_base.rotation().wxyz]
                )
            ],
        }

    def _target_pose_dict(self) -> dict[str, list[float]]:
        """Get the target placement spot's pose in robot base frame (fixed
        orientation, identity quat -- it's just a marked table region)."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        target_world = vtf.SE3(
            wxyz_xyz=np.concatenate(
                [np.array([1.0, 0.0, 0.0, 0.0]), self.robosuite_env.target_world_pos()]
            )
        )
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        target_robot_base = base_transform @ target_world
        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [target_robot_base.translation(), target_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        return self.robosuite_env._check_success()

    def spill_ratio(self) -> float:
        return float(self.robosuite_env._spill_ratio())

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cup_pose_dict(robosuite_obs)
        cup_pose_array = np.stack(
            [np.asarray(pose_dict["primary"], dtype=np.float32)], axis=0
        )

        robosuite_obs["cup_poses"] = {"primary": cup_pose_array[0]}

        target_pose_dict = self._target_pose_dict()
        target_pose_array = np.stack(
            [np.asarray(target_pose_dict["primary"], dtype=np.float32)], axis=0
        )
        robosuite_obs["target_poses"] = {"primary": target_pose_array[0]}

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs


__all__ = ["FrankaRobosuitePourLowLevel"]
