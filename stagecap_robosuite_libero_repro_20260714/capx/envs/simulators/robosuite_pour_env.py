"""PourEnv: "serve water to the boss" manipulation task.

A cup containing several small free-body "beads" (stand-ins for water, since
MuJoCo has no native liquid) must be picked up, carried across the table to a
marked target region, and set down there WITHOUT spilling the beads out of
the cup. This is the physical grounding for MannerCap's phase-aware manner
instructions:

    approach (快速接近) -> grasp (轻轻拿起) -> transport (稳稳运送)
        -> final/put-down (缓缓放下) -> retreat (快速离开)

Unlike lift/stack (where manner only affects HOW gently the task is done),
here manner directly gates task SUCCESS: shaking the cup (high jerk / fast
acceleration during grasp or transport) physically flings beads out, so
"spill_ratio" is an objective, task-intrinsic proxy for manner compliance,
independent of the rule-based MannerValidator.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.models.objects.cup_object import CupObject


class PourEnv(ManipulationEnv):
    """Serve-water task: carry a bead-filled cup to a target region without spilling."""

    N_BEADS = 8
    BEAD_RADIUS = 0.006
    CUP_OUTER_RADIUS = 0.030
    CUP_INNER_RADIUS = 0.024
    CUP_HEIGHT = 0.06
    CUP_NGEOMS = 16
    SPILL_SUCCESS_RATIO = 0.75
    TARGET_XY_TOLERANCE = 0.07
    CATCH_RADIUS = 0.06  # bead-to-cup-center xy distance considered "still in cup"

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # Target region ("boss's desk spot") in table-local xy (relative to
        # table_offset), fixed across resets so every trial carries the cup
        # the same distance -- only the cup's own start pose is randomized.
        self.target_xy_offset = np.array([0.05, 0.20])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Model / references / observables
    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="cup_ceramic_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.cup = CupObject(
            name="cup",
            outer_radius=self.CUP_OUTER_RADIUS,
            inner_radius=self.CUP_INNER_RADIUS,
            height=self.CUP_HEIGHT,
            ngeoms=self.CUP_NGEOMS,
            rgba=[0.75, 0.75, 0.78, 1.0],
            material=ceramic,
        )

        self.beads = [
            BallObject(
                name=f"bead_{i}",
                size=(self.BEAD_RADIUS,),
                density=500.0,
                rgba=[0.15, 0.45, 0.95, 1.0],
            )
            for i in range(self.N_BEADS)
        ]

        # Target marker: fixed flat visual disc, not part of the dynamic
        # object list (no placement sampler / no free joint).
        self.target_marker = BoxObject(
            name="target_marker",
            size=[0.06, 0.06, 0.001],
            rgba=[0.2, 0.85, 1.0, 0.55],
            joints=None,
            obj_type="visual",
        )
        target_body = self.target_marker.get_obj()
        target_pos = self.table_offset + np.array(
            [self.target_xy_offset[0], self.target_xy_offset[1], 0.026]
        )
        target_body.set("pos", " ".join(str(x) for x in target_pos))
        mujoco_arena.worldbody.append(target_body)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cup)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cup,
                x_range=[-0.06, 0.06],
                y_range=[-0.22, -0.10],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.05,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cup] + self.beads,
        )
        self.model.merge_assets(self.target_marker)

    def _setup_references(self):
        super()._setup_references()
        self.cup_body_id = self.sim.model.body_name2id(self.cup.root_body)
        self.bead_body_ids = [
            self.sim.model.body_name2id(bead.root_body) for bead in self.beads
        ]

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cup_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cup_body_id])

            @sensor(modality=modality)
            def cup_quat(obs_cache):
                from robosuite.utils.transform_utils import convert_quat
                return convert_quat(np.array(self.sim.data.body_xquat[self.cup_body_id]), to="xyzw")

            @sensor(modality=modality)
            def beads_in_cup_ratio(obs_cache):
                return np.array([self._spill_ratio()])

            sensors = [cup_pos, cup_quat, beads_in_cup_ratio]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cup_pos", f"{arm_pf}gripper_to_cup_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            cup_pos = cup_quat = None
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )
                if obj is self.cup:
                    cup_pos, cup_quat = np.array(obj_pos), np.array(obj_quat)

            if cup_pos is not None:
                self._place_beads_in_cup(cup_pos)

    def _place_beads_in_cup(self, cup_xy_pos: np.ndarray) -> None:
        """Drop beads inside the cup cavity in a small ring pattern so gravity
        settles them at the bottom without initial overlap."""
        n = self.N_BEADS
        ring_r = max(0.0, self.CUP_INNER_RADIUS - self.BEAD_RADIUS - 0.003)
        for i, bead in enumerate(self.beads):
            angle = 2.0 * np.pi * i / n
            # alternate two z-layers so nearby beads don't spawn overlapping
            layer = i % 2
            local_xy = np.array([ring_r * 0.6 * np.cos(angle), ring_r * 0.6 * np.sin(angle)])
            pos = cup_xy_pos.copy()
            pos[0] += local_xy[0]
            pos[1] += local_xy[1]
            pos[2] += 0.01 + layer * (2.2 * self.BEAD_RADIUS) + 0.002
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.sim.data.set_joint_qpos(bead.joints[0], np.concatenate([pos, quat]))

    # ------------------------------------------------------------------
    # Success / reward
    # ------------------------------------------------------------------
    def target_world_pos(self) -> np.ndarray:
        """World-frame xyz of the target placement spot (the marked region on
        the table the cup should be carried to and set down at)."""
        return self.table_offset + np.array(
            [self.target_xy_offset[0], self.target_xy_offset[1], 0.026]
        )

    def _spill_ratio(self) -> float:
        """Fraction of beads still within CATCH_RADIUS of the cup's xy center."""
        cup_xy = self.sim.data.body_xpos[self.cup_body_id][:2]
        n_in = 0
        for bid in self.bead_body_ids:
            bead_xy = self.sim.data.body_xpos[bid][:2]
            if np.linalg.norm(bead_xy - cup_xy) < self.CATCH_RADIUS:
                n_in += 1
        return n_in / max(1, len(self.bead_body_ids))

    def _cup_upright(self) -> bool:
        mat = self.sim.data.body_xmat[self.cup_body_id].reshape(3, 3)
        return bool(mat[2, 2] > 0.80)

    def _cup_near_target(self) -> bool:
        cup_xy = self.sim.data.body_xpos[self.cup_body_id][:2]
        target_xy = self.table_offset[:2] + self.target_xy_offset
        return bool(np.linalg.norm(cup_xy - target_xy) < self.TARGET_XY_TOLERANCE)

    def _check_success(self) -> bool:
        return self._cup_near_target() and self._cup_upright() and (
            self._spill_ratio() >= self.SPILL_SUCCESS_RATIO
        )

    def reward(self, action=None):
        reward = 0.0
        cup_pos = self.sim.data.body_xpos[self.cup_body_id]
        table_h = self.table_offset[2]

        if self.reward_shaping:
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cup, target_type="body", return_distance=True
            )
            reward += 0.15 * (1.0 - np.tanh(5.0 * dist))
            if cup_pos[2] > table_h + 0.03:
                reward += 0.15
                target_xy = self.table_offset[:2] + self.target_xy_offset
                d_target = np.linalg.norm(cup_pos[:2] - target_xy)
                reward += 0.3 * (1.0 - np.tanh(3.0 * d_target))
            reward += 0.4 * self._spill_ratio()
        elif self._check_success():
            reward = 1.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0
        return reward

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cup)
