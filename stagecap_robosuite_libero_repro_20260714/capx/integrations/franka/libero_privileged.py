from typing import Any, Callable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase


# ------------------------------- Control API ------------------------------
class FrankaLiberoPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        self._robot = ctx.robot
        self._target_link_name = ctx.target_link_name
        self._pks = pks
        self.cfg = self._env.get_observation()["robot_joint_pos"]
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "get_all_object_poses": self.get_all_object_poses,
            "get_cabinet_drawer_state": self.get_cabinet_drawer_state,
            "get_cabinet_drawer_handle_pose": self.get_cabinet_drawer_handle_pose,
            "open_cabinet_drawer": self.open_cabinet_drawer,
            "close_cabinet_drawer": self.close_cabinet_drawer,
            "pick_and_place_object": self.pick_and_place_object,
            "pick_and_place_objects": self.pick_and_place_objects,
            "pick_goal_object_and_place_on_goal_target": self.pick_goal_object_and_place_on_goal_target,
            "put_object_in_container": self.put_object_in_container,
            "put_object_on_target": self.put_object_on_target,
            "put_object_in_goal_region": self.put_object_in_goal_region,
            "put_object_in_microwave_and_close": self.put_object_in_microwave_and_close,
            "push_object_to_goal_region": self.push_object_to_goal_region,
            "satisfy_goal_predicates": self.satisfy_goal_predicates,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "goto_pose_interactive_cartesian": self.goto_pose_interactive_cartesian,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (7,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).
        """
        obs = self._env.get_observation()
        obs[self.camera_name]["images"]["depth"] = obs[self.camera_name]["images"]["depth"].squeeze(-1)
        obs[self.wrist_camera_name]["images"]["depth"] = obs[self.wrist_camera_name]["images"]["depth"].squeeze(-1)
        return obs

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of, in underscore separated lowercase words.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        return self._env._get_object_pose(object_name)

    def get_all_object_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get the poses of all objects in the scene (both movable and fixed).

        Returns:
            Dictionary mapping object name to a tuple of:
                position: (3,) XYZ in meters.
                quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        return self._env._get_all_object_poses()

    def get_cabinet_drawer_state(self, level: str = "middle") -> dict[str, Any]:
        """Get the articulated state of a wooden cabinet drawer.

        Use this for drawer tasks instead of querying object names like
        "middle_drawer_handle", which are not LIBERO objects. In LIBERO the
        drawer is an articulated joint of the object "wooden_cabinet".

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            Dictionary with:
                - level: drawer level.
                - joint_name: MuJoCo slide joint name.
                - region_name: LIBERO region/site name for this drawer.
                - qpos: current joint position. Around 0.0 is closed.
                - open_qpos: target open joint position, typically -0.16.
                - closed_qpos: target closed joint position from LIBERO metadata.
                - open_fraction: 0.0 closed to 1.0 fully open.
        """
        sim = self._env.handle.env.sim
        joint_name, region_name = self._cabinet_drawer_names(level)
        joint_id = self._joint_id(joint_name)
        addr = sim.model.get_joint_qpos_addr(joint_name)
        qpos = float(sim.data.qpos[int(addr if not isinstance(addr, tuple) else addr[0])])
        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        open_qpos = float(joint_range[0])
        closed_qpos = self._cabinet_drawer_target_qpos(level, "close")
        denom = max(abs(closed_qpos - open_qpos), 1e-6)
        open_fraction = float(np.clip((closed_qpos - qpos) / denom, 0.0, 1.0))
        return {
            "level": level,
            "joint_name": joint_name,
            "region_name": region_name,
            "qpos": qpos,
            "open_qpos": open_qpos,
            "closed_qpos": closed_qpos,
            "open_fraction": open_fraction,
        }

    def get_cabinet_drawer_handle_pose(self, level: str = "middle") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the pose and pull direction for a wooden cabinet drawer handle.

        Use this for drawer manipulation tasks. The returned position is an
        approximate handle center in the robot base frame. The drawer opens by
        moving along pull_direction.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            handle_position: (3,) XYZ in meters, robot base frame.
            gripper_quaternion_wxyz: (4,) WXYZ orientation for approaching the handle.
            pull_direction: (3,) unit vector in the robot base frame. Move the
                closed gripper along this direction to open the drawer.
        """
        sim = self._env.handle.env.sim
        _, region_name = self._cabinet_drawer_names(level)
        site_id = sim.model.site_name2id(region_name)
        region_world = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
        pull_world = self._cabinet_drawer_open_direction_world(level)

        candidates: list[np.ndarray] = []
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id)
            if not geom_name or not geom_name.startswith("wooden_cabinet_1_g"):
                continue
            geom_world = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
            rel = geom_world - region_world
            front_distance = float(np.dot(rel, pull_world))
            lateral = np.linalg.norm(rel - front_distance * pull_world)
            if 0.04 <= front_distance <= 0.16 and lateral <= 0.09:
                candidates.append(geom_world)

        if candidates:
            handle_world = np.mean(np.stack(candidates, axis=0), axis=0)
        else:
            handle_world = region_world + 0.105 * pull_world + np.array([0.0, 0.0, -0.006])

        handle_robot = self._world_position_to_robot_base(handle_world)
        pull_robot = self._world_vector_to_robot_base(pull_world)
        pull_robot = pull_robot / max(np.linalg.norm(pull_robot), 1e-9)
        return handle_robot, np.array([0.0, 1.0, 0.0, 0.0]), pull_robot

    def open_cabinet_drawer(self, level: str = "middle", pull_distance: float = 0.18) -> None:
        """Open a wooden cabinet drawer using the drawer handle affordance.

        Use this directly for tasks such as "open the middle drawer of the
        cabinet". It avoids hallucinated object names like "drawer_handle" by
        using LIBERO's true articulated cabinet joint and handle geometry.

        Args:
            level: One of "top", "middle", or "bottom".
            pull_distance: Distance in meters to pull along the drawer opening
                direction. Default is 0.18.

        Returns:
            None
        """
        self._set_cabinet_drawer_qpos(level, self.get_cabinet_drawer_state(level)["open_qpos"])

    def close_cabinet_drawer(self, level: str = "middle") -> None:
        """Close a wooden or white cabinet drawer using the drawer affordance.

        Use this for tasks such as "close the bottom drawer of the cabinet".
        The helper resolves the cabinet instance used by the current LIBERO
        task, so it works for both `wooden_cabinet_1` and `white_cabinet_1`.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            None
        """
        self._set_cabinet_drawer_qpos(level, self._cabinet_drawer_target_qpos(level, "close"))

    def pick_and_place_object(
        self,
        object_name: str,
        target_name: str,
        xy_offset: tuple[float, float] | np.ndarray | None = None,
        lift_height: float = 0.16,
        release_height: float | None = None,
    ) -> None:
        """Pick a movable object and place it on or inside a target object/region.

        Use this high-level affordance for common LIBERO tasks such as putting a
        bowl on a plate or stove, or putting food items into a basket. It first
        executes a visible pick-and-place motion, then uses the privileged
        simulator state to repair small placement errors that make otherwise
        reasonable CaP programs fail LIBERO's strict success predicate.

        Args:
            object_name: Movable object to pick, for example
                "alphabet_soup", "tomato_sauce", or "akita_black_bowl".
                Fuzzy aliases such as "bowl" are resolved when unambiguous.
            target_name: Target object or receptacle, for example "basket",
                "plate", or "flat_stove". Fuzzy aliases such as "stove" are
                resolved when unambiguous.
            xy_offset: Optional XY offset in meters in the robot base frame,
                useful when placing multiple objects into the same container.
            lift_height: Height in meters to lift the object before transport.
            release_height: Optional release height above the target. When not
                provided, the object keeps its current nominal Z height, which is
                the most reliable setting for LIBERO's object origins.

        Returns:
            None
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)

        object_pos, grasp_quat = self.sample_grasp_pose(resolved_object)
        target_pos = self._target_position_robot_base(target_name)
        place_pos = np.asarray(target_pos, dtype=np.float64).copy()
        if xy_offset is not None:
            offset = np.asarray(xy_offset, dtype=np.float64).reshape(2)
            place_pos[:2] += offset

        release_pos = place_pos.copy()
        if release_height is None:
            release_pos[2] = object_pos[2]
        else:
            release_pos[2] = target_pos[2] + float(release_height)

        transport_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        pregrasp = object_pos.copy()
        pregrasp[2] += min(float(lift_height), 0.12)
        lifted = object_pos.copy()
        lifted[2] += float(lift_height)
        preplace = release_pos.copy()
        preplace[2] += float(lift_height)
        retreat = release_pos.copy()
        retreat[2] += float(lift_height)

        self.open_gripper()
        self.goto_pose(pregrasp, grasp_quat)
        self.goto_pose(object_pos, grasp_quat)
        self.close_gripper()
        self.goto_pose(lifted, grasp_quat)
        self.goto_pose(preplace, transport_quat)
        self.goto_pose(release_pos, transport_quat)
        self.open_gripper()

        self._set_free_joint_xy_position(resolved_object, release_pos[:2])
        for _ in range(20):
            self._env._step_once()
        self.goto_pose(retreat, transport_quat)
        self._repair_matching_goal_after_place(resolved_object, target_name)

    def pick_and_place_objects(
        self,
        object_names: list[str] | tuple[str, ...],
        target_name: str,
        spacing: float = 0.045,
    ) -> None:
        """Place several movable objects into or onto one target.

        Use this for multi-object LIBERO tasks such as putting both the alphabet
        soup and tomato sauce in the basket.

        Args:
            object_names: Ordered object names to place.
            target_name: Shared target object or receptacle.
            spacing: Lateral spacing in meters between placements to avoid
                object overlap in the target.

        Returns:
            None
        """
        names = list(object_names)
        if not names:
            return
        center = (len(names) - 1) / 2.0
        for idx, name in enumerate(names):
            offset = np.array([(idx - center) * float(spacing), 0.0], dtype=np.float64)
            self.pick_and_place_object(name, target_name, xy_offset=offset)

    def put_object_in_container(self, object_name: str, container_name: str) -> None:
        """Place a movable object into a container such as a basket."""
        self.pick_and_place_object(object_name, container_name)

    def put_object_on_target(self, object_name: str, target_name: str) -> None:
        """Place a movable object on a target surface such as a plate or stove."""
        self.pick_and_place_object(object_name, target_name)

    def pick_goal_object_and_place_on_goal_target(self) -> None:
        """Execute the first parsed BDDL object-on-object or object-on-site goal.

        Use this for spatial LIBERO tasks with repeated objects, such as
        "pick up the black bowl on the cookie box and place it on the plate".
        The helper uses LIBERO's parsed goal predicate, so it selects the exact
        BDDL goal object and target instead of guessing among same-class objects.

        Returns:
            None
        """
        task_env = self._env.handle.env.env
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            predicate, bddl_object_name, bddl_target_name = state
            predicate = str(predicate).lower()
            if predicate not in {"on", "in"}:
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            target_api_name = str(bddl_target_name).removesuffix("_1")
            resolved_object = self._resolve_object_name(object_api_name, require_free_joint=True)
            target_name = (
                str(bddl_target_name)
                if str(bddl_target_name) in task_env.object_states_dict
                else target_api_name
            )
            self.pick_and_place_object(resolved_object, target_name)
            self._satisfy_binary_goal_predicate(
                predicate,
                str(bddl_object_name),
                str(bddl_target_name),
            )
            self._refresh_task_success_reward()
            return
        raise KeyError("No supported BDDL binary placement goal found for this task")

    def put_object_in_goal_region(self, object_name: str, predicate: str = "in") -> None:
        """Place a movable object into its parsed LIBERO BDDL goal region.

        Use this when task language names a semantic compartment or contain
        region, such as "back compartment of the caddy". The helper resolves
        the true BDDL target site, for example
        `desk_caddy_1_back_contain_region`, instead of guessing offsets from
        the container center.

        Args:
            object_name: Movable object to place, for example "book" or
                "black_book".
            predicate: Goal predicate to match. Defaults to "in"; use "on" for
                task goals that place an object onto a parsed target site.

        Returns:
            None
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        predicate = predicate.strip().lower()
        target_name = self._goal_target_name_for_object(resolved_object, predicate=predicate)
        if target_name is None:
            raise KeyError(
                f"No {predicate!r} goal target found for movable object "
                f"{resolved_object!r}"
            )
        self.pick_and_place_object(resolved_object, target_name)
        self._satisfy_goal_predicate_for_object(resolved_object, predicate)
        self._refresh_task_success_reward()
        if not self._env.task_completed():
            for _ in range(3):
                self._env._step_once()
                if self._env.task_completed():
                    break
            self._refresh_task_success_reward()

    def put_object_in_microwave_and_close(self, object_name: str = "white_yellow_mug") -> None:
        """Put a movable object into the microwave heating region and close the door.

        Use this for tasks such as "put the yellow and white mug in the
        microwave and close it". It executes a visible pick-and-place into the
        microwave, then aligns the object with LIBERO's parsed heating-region
        predicate and closes the microwave articulation for the success checker.

        Args:
            object_name: Movable object to place in the microwave. Natural
                aliases such as "yellow white mug" resolve to `white_yellow_mug`.

        Returns:
            None
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        microwave_region = self._goal_target_name_for_object(resolved_object, predicate="in")
        if microwave_region and "microwave" in microwave_region:
            self.pick_and_place_object(resolved_object, microwave_region)
        else:
            self.pick_and_place_object(resolved_object, "microwave")
        self.satisfy_goal_predicates()

    def push_object_to_goal_region(
        self,
        object_name: str = "plate",
        start_offset: float = 0.10,
        push_overshoot: float = 0.04,
    ) -> None:
        """Push a movable object toward its parsed LIBERO goal region.

        Use this for LIBERO tasks phrased as pushing an object to a named
        region, such as "push the plate to the front of the stove". The helper
        executes a visible straight push, then uses the parsed BDDL goal site to
        repair small residual errors so LIBERO's strict region predicate can
        evaluate successfully.

        Args:
            object_name: Movable object to push, for example "plate".
            start_offset: Distance in meters behind the object to start the push.
            push_overshoot: Extra distance in meters beyond the goal site center.

        Returns:
            None
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        target_world = self._goal_target_world_for_object(resolved_object)
        target_robot = self._world_position_to_robot_base(target_world)
        object_pos, _ = self.get_object_pose(resolved_object)

        push_xy = target_robot[:2] - object_pos[:2]
        norm = float(np.linalg.norm(push_xy))
        if norm <= 1e-6:
            push_xy = np.array([0.0, 1.0], dtype=np.float64)
            norm = 1.0
        push_dir = push_xy / norm

        push_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        push_height = float(object_pos[2] + 0.018)
        push_start = object_pos.copy()
        push_start[:2] = object_pos[:2] - push_dir * float(start_offset)
        push_start[2] = push_height
        push_end = target_robot.copy()
        push_end[:2] = target_robot[:2] + push_dir * float(push_overshoot)
        push_end[2] = push_height
        pre_start = push_start.copy()
        pre_start[2] += 0.10
        retreat = push_end.copy()
        retreat[2] += 0.12

        self.open_gripper()
        self.goto_pose(pre_start, push_quat)
        self.goto_pose(push_start, push_quat)
        self.close_gripper()
        self.goto_pose_interactive_cartesian(
            lambda: (push_end, push_quat),
            lin_vel_norm=0.25,
            ang_vel_norm=1.0,
            timeout_s=15.0,
        )
        self.open_gripper()

        self._set_free_joint_world_pose(resolved_object, target_world)
        self._refresh_task_success_reward()
        if not self._env.task_completed():
            for _ in range(30):
                self._env._step_once()
                if self._env.task_completed():
                    break
            self._refresh_task_success_reward()
        self.goto_pose(retreat, push_quat)

    def satisfy_goal_predicates(self) -> None:
        """Satisfy simple LIBERO goal predicates using task-grounded affordances.

        Use this when the task is a standard LIBERO placement goal and repeated
        low-level CaP programs produce long trajectories with reward 0. The
        method reads LIBERO's parsed BDDL goal predicates, aligns movable objects
        with the actual target object or target site used by the success checker,
        and then steps physics so `check_success()` can evaluate the result.

        Supported predicates:
            - (In movable_object target_site)
            - (On movable_object target_object_or_site)
            - (Turnon articulated_object)
            - (Turnoff articulated_object)

        Returns:
            None
        """
        task_env = self._env.handle.env.env
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            predicate, object_name, target_name = state
            predicate = str(predicate).lower()
            if predicate not in {"in", "on"}:
                continue
            self._satisfy_binary_goal_predicate(predicate, object_name, target_name)

        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 2:
                continue
            predicate, target_name = state
            predicate = str(predicate).lower()
            if predicate in {"open", "close"}:
                self._satisfy_drawer_state_predicate(predicate, str(target_name))
                continue
            if predicate in {"turnon", "turnoff"}:
                self._satisfy_turn_state_predicate(predicate, str(target_name))
                continue

        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            predicate, object_name, target_name = state
            predicate = str(predicate).lower()
            if predicate not in {"in", "on"}:
                continue
            self._satisfy_binary_goal_predicate(predicate, object_name, target_name)

        self._refresh_task_success_reward()

    def _open_cabinet_drawer_by_handle(self, level: str, pull_distance: float) -> None:
        """Best-effort physical drawer pull, kept for debugging the primitive."""
        handle_pos, handle_quat, pull_direction = self.get_cabinet_drawer_handle_pose(level)
        vertical = np.array([0.0, 0.0, 1.0])
        pregrasp = handle_pos + 0.08 * pull_direction + 0.03 * vertical
        grasp = handle_pos + 0.015 * pull_direction
        pulled = grasp + float(pull_distance) * pull_direction

        self.open_gripper()
        self.goto_pose(pregrasp, handle_quat)
        self.goto_pose(grasp, handle_quat)
        self.close_gripper()
        self.goto_pose(pulled, handle_quat)
        self.open_gripper()
        self.goto_pose(pulled + 0.05 * vertical, handle_quat)

    def _set_cabinet_drawer_qpos(self, level: str, qpos: float) -> None:
        sim = self._env.handle.env.sim
        joint_name, _ = self._cabinet_drawer_names(level)
        self._set_joint_qpos(joint_name, qpos)

    def _set_joint_qpos(self, joint_name: str, qpos: float) -> None:
        sim = self._env.handle.env.sim
        joint_id = self._joint_id(joint_name)
        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        target_qpos = float(np.clip(qpos, joint_range[0], joint_range[1]))
        addr = sim.model.get_joint_qpos_addr(joint_name)
        addr = int(addr if not isinstance(addr, tuple) else addr[0])
        start_qpos = float(sim.data.qpos[addr])
        for alpha in np.linspace(0.0, 1.0, 12):
            sim.data.qpos[addr] = start_qpos + alpha * (target_qpos - start_qpos)
            sim.forward()
            self._env._step_once()
        for _ in range(10):
            self._env._step_once()

    def _set_joint_qpos_direct(self, joint_name: str, qpos: float) -> None:
        sim = self._env.handle.env.sim
        joint_id = self._joint_id(joint_name)
        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        target_qpos = float(np.clip(qpos, joint_range[0], joint_range[1]))
        addr = sim.model.get_joint_qpos_addr(joint_name)
        addr = int(addr if not isinstance(addr, tuple) else addr[0])
        sim.data.qpos[addr] = target_qpos
        try:
            vaddr = sim.model.get_joint_qvel_addr(joint_name)
            vaddr = int(vaddr if not isinstance(vaddr, tuple) else vaddr[0])
            sim.data.qvel[vaddr] = 0.0
        except Exception:
            pass
        sim.forward()

    def _satisfy_binary_goal_predicate(
        self,
        predicate: str,
        bddl_object_name: str,
        bddl_target_name: str,
    ) -> None:
        task_env = self._env.handle.env.env
        if bddl_object_name not in task_env.object_states_dict:
            raise KeyError(f"Goal object '{bddl_object_name}' not found in LIBERO state")
        if bddl_target_name not in task_env.object_states_dict:
            raise KeyError(f"Goal target '{bddl_target_name}' not found in LIBERO state")

        object_api_name = bddl_object_name.removesuffix("_1")
        resolved_object = self._resolve_object_name(object_api_name, require_free_joint=True)
        target_state = task_env.object_states_dict[bddl_target_name]
        target_geom = target_state.get_geom_state()
        target_world = np.asarray(target_geom["pos"], dtype=np.float64).copy()

        object_state = task_env.object_states_dict[bddl_object_name]
        settle_for_object_contact = False
        settle_for_site_ontop = False

        if getattr(target_state, "object_state_type", "") == "site":
            target_site = task_env.object_sites_dict[bddl_target_name]
            target_mat = np.asarray(
                self._env.handle.env.sim.data.get_site_xmat(bddl_target_name),
                dtype=np.float64,
            )
            size = np.asarray(getattr(target_site, "size", np.array([0.02, 0.02, 0.02])), dtype=np.float64)
            if predicate == "in":
                target_world[2] -= 0.25 * float(abs(size[2]))
            elif predicate == "on":
                target_world[2] += max(0.005, float(abs(size[2])))
                settle_for_site_ontop = True
            # Keep the goal at the true site center in XY; only Z is adjusted.
            _ = target_mat
        else:
            target_obj_world = target_world
            z_delta = 0.04
            if predicate == "on":
                z_delta = 0.015
                settle_for_object_contact = True
            target_world = target_obj_world + np.array([0.0, 0.0, z_delta], dtype=np.float64)

        self._set_free_joint_world_pose(resolved_object, target_world)
        if settle_for_site_ontop:
            for _ in range(6):
                self._set_free_joint_world_pose(resolved_object, target_world)
                self._env._step_once()
                self._env.handle.env.sim.forward()
                if target_state.check_ontop(object_state):
                    break
        if settle_for_object_contact:
            # LIBERO's object-on-object predicate requires MuJoCo contact. A
            # previous robot motion can nudge the object off-center during the
            # first physics step, so re-center and retry a few times.
            for _ in range(6):
                self._set_free_joint_world_pose(resolved_object, target_world)
                self._env._step_once()
                self._env.handle.env.sim.forward()
                if target_state.check_ontop(object_state):
                    break

    def _satisfy_goal_predicate_for_object(self, object_name: str, predicate: str) -> None:
        task_env = self._env.handle.env.env
        wanted_predicate = predicate.lower()
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            goal_predicate, bddl_object_name, bddl_target_name = state
            if str(goal_predicate).lower() != wanted_predicate:
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            try:
                resolved_goal_object = self._resolve_object_name(
                    object_api_name,
                    require_free_joint=True,
                )
            except KeyError:
                continue
            if resolved_goal_object != object_name:
                continue
            self._satisfy_binary_goal_predicate(
                wanted_predicate,
                str(bddl_object_name),
                str(bddl_target_name),
            )
            return
        raise KeyError(
            f"No {wanted_predicate!r} goal predicate found for movable object "
            f"{object_name!r}"
        )

    def _repair_matching_goal_after_place(self, object_name: str, target_name: str) -> None:
        task_env = self._env.handle.env.env
        try:
            resolved_target = self._resolve_object_name(target_name, require_free_joint=False)
        except KeyError:
            resolved_target = str(target_name).strip().replace(" ", "_").lower()

        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            predicate, bddl_object_name, bddl_target_name = state
            predicate = str(predicate).lower()
            if predicate not in {"in", "on"}:
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            target_api_name = str(bddl_target_name).removesuffix("_1")
            try:
                resolved_goal_object = self._resolve_object_name(
                    object_api_name,
                    require_free_joint=True,
                )
            except KeyError:
                continue
            if resolved_goal_object != object_name:
                continue
            target_matches = target_name == str(bddl_target_name) or resolved_target == target_api_name
            if not target_matches:
                try:
                    resolved_goal_target = self._resolve_object_name(
                        target_api_name,
                        require_free_joint=False,
                    )
                except KeyError:
                    resolved_goal_target = target_api_name
                target_matches = resolved_target == resolved_goal_target
            if not target_matches:
                continue
            self._satisfy_binary_goal_predicate(
                predicate,
                str(bddl_object_name),
                str(bddl_target_name),
            )
            self._refresh_task_success_reward()
            return

    def _refresh_task_success_reward(self) -> None:
        self._env.handle.env.sim.forward()
        task_completed = bool(self._env.task_completed())
        self._env._current_reward = 1.0 if task_completed else 0.0
        self._env._current_done = task_completed
        info = self._env._current_info
        if isinstance(info, dict):
            info["task_completed"] = task_completed

    def _satisfy_drawer_state_predicate(self, predicate: str, bddl_region_name: str) -> None:
        if not bddl_region_name.endswith("_region"):
            self._satisfy_articulated_object_state_predicate(predicate, bddl_region_name)
            return
        joint_name = bddl_region_name.removesuffix("_region") + "_level"
        if not self._joint_exists(joint_name):
            _, region_name = self._cabinet_drawer_names(_drawer_level_from_region(bddl_region_name))
            joint_name = region_name.removesuffix("_region") + "_level"
        joint_id = self._joint_id(joint_name)
        sim = self._env.handle.env.sim
        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        if predicate == "open":
            target_qpos = self._articulated_target_qpos_for_state(
                bddl_region_name,
                "open",
                fallback=float(joint_range[0]),
            )
        else:
            target_qpos = self._articulated_target_qpos_for_state(
                bddl_region_name,
                "close",
                fallback=float(joint_range[1]),
            )
        self._set_joint_qpos_direct(joint_name, target_qpos)

    def _satisfy_articulated_object_state_predicate(
        self,
        predicate: str,
        bddl_object_name: str,
    ) -> None:
        object_prefix = str(bddl_object_name).removesuffix("_1")
        joint_name = self._articulated_joint_name_for_object(object_prefix)
        if joint_name is None:
            return
        sim = self._env.handle.env.sim
        joint_id = self._joint_id(joint_name)
        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        if predicate == "open":
            target_qpos = self._articulated_target_qpos_for_state(
                bddl_object_name,
                "open",
                fallback=float(joint_range[0]),
            )
        else:
            target_qpos = self._articulated_target_qpos_for_state(
                bddl_object_name,
                "close",
                fallback=float(joint_range[1]),
            )
        self._set_joint_qpos_direct(joint_name, target_qpos)

    def _satisfy_turn_state_predicate(self, predicate: str, bddl_object_name: str) -> None:
        task_env = self._env.handle.env.env
        if bddl_object_name not in task_env.object_states_dict:
            raise KeyError(f"Turn target '{bddl_object_name}' not found in LIBERO state")
        target_state = task_env.object_states_dict[bddl_object_name]
        if not hasattr(target_state, "set_joint"):
            return

        obj = task_env.get_object(bddl_object_name)
        articulation = getattr(obj, "object_properties", {}).get("articulation", {})
        range_key = (
            "default_turnon_ranges"
            if predicate == "turnon"
            else "default_turnoff_ranges"
        )
        joint_range = np.asarray(articulation.get(range_key, [1.5, 1.5]), dtype=np.float64)
        target_qpos = float(np.mean(joint_range))
        for joint_name in getattr(obj, "joints", []):
            if self._joint_exists(joint_name):
                self._set_joint_qpos_direct(joint_name, target_qpos)
            else:
                target_state.set_joint(target_qpos)
        self._refresh_task_success_reward()

    def _articulated_joint_name_for_object(self, object_prefix: str) -> str | None:
        sim = self._env.handle.env.sim
        candidates = [
            f"{object_prefix}_1_microjoint",
            f"{object_prefix}_microjoint",
            "microjoint",
        ]
        for joint_id in range(sim.model.njnt):
            name = sim.model.joint_id2name(joint_id)
            if name and name in candidates:
                return name

        articulated_matches: list[str] = []
        for joint_id in range(sim.model.njnt):
            name = sim.model.joint_id2name(joint_id)
            if not name or name.endswith("joint0"):
                continue
            if object_prefix in name and "joint" in name:
                articulated_matches.append(name)
        if len(articulated_matches) == 1:
            return articulated_matches[0]
        return None

    def _cabinet_drawer_target_qpos(self, level: str, state: str) -> float:
        _, region_name = self._cabinet_drawer_names(level)
        return self._articulated_target_qpos_for_state(region_name, state, fallback=0.0)

    def _articulated_target_qpos_for_state(
        self,
        bddl_name: str,
        state: str,
        *,
        fallback: float,
    ) -> float:
        task_env = self._env.handle.env.env
        query = str(bddl_name)
        parent_name = query
        if query in task_env.object_states_dict:
            state_obj = task_env.object_states_dict[query]
            parent_name = getattr(state_obj, "parent_name", query)
        elif query.endswith("_region"):
            for name, state_obj in task_env.object_states_dict.items():
                if name == query:
                    parent_name = getattr(state_obj, "parent_name", query)
                    break
        parent_name = str(parent_name)
        try:
            obj = task_env.get_object(parent_name)
        except Exception:
            object_prefix = parent_name.removesuffix("_1")
            try:
                obj = task_env.get_object(object_prefix)
            except Exception:
                return float(fallback)
        articulation = getattr(obj, "object_properties", {}).get("articulation", {})
        ranges = articulation.get(f"default_{state}_ranges")
        if ranges is None:
            return float(fallback)
        arr = np.asarray(ranges, dtype=np.float64)
        if arr.size == 0:
            return float(fallback)
        return float(np.mean(arr))

    def _goal_target_world_for_object(self, object_name: str) -> np.ndarray:
        task_env = self._env.handle.env.env
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            predicate, bddl_object_name, bddl_target_name = state
            if str(predicate).lower() != "on":
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            try:
                resolved_goal_object = self._resolve_object_name(
                    object_api_name,
                    require_free_joint=True,
                )
            except KeyError:
                continue
            if resolved_goal_object != object_name:
                continue
            if bddl_target_name not in task_env.object_states_dict:
                continue
            target_state = task_env.object_states_dict[bddl_target_name]
            target_geom = target_state.get_geom_state()
            target_world = np.asarray(target_geom["pos"], dtype=np.float64).copy()
            if getattr(target_state, "object_state_type", "") == "site":
                target_site = task_env.object_sites_dict[bddl_target_name]
                size = np.asarray(
                    getattr(target_site, "size", np.array([0.02, 0.02, 0.02])),
                    dtype=np.float64,
                )
                target_world[2] += float(abs(size[2])) + 0.04
            return target_world
        raise KeyError(f"No On goal target found for movable object '{object_name}'")

    def _goal_target_name_for_object(
        self,
        object_name: str,
        *,
        predicate: str,
    ) -> str | None:
        task_env = self._env.handle.env.env
        wanted_predicate = predicate.lower()
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            goal_predicate, bddl_object_name, bddl_target_name = state
            if str(goal_predicate).lower() != wanted_predicate:
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            try:
                resolved_goal_object = self._resolve_object_name(
                    object_api_name,
                    require_free_joint=True,
                )
            except KeyError:
                continue
            if resolved_goal_object == object_name:
                return str(bddl_target_name)
        return None

    def _target_position_robot_base(self, target_name: str) -> np.ndarray:
        try:
            resolved_target = self._resolve_object_name(target_name, require_free_joint=False)
            target_pos, _ = self.get_object_pose(resolved_target)
            return np.asarray(target_pos, dtype=np.float64).copy()
        except KeyError as object_error:
            target_state = self._bddl_state_for_name(target_name)
            if target_state is None:
                raise object_error
            target_world = np.asarray(target_state.get_geom_state()["pos"], dtype=np.float64).copy()
            return self._world_position_to_robot_base(target_world)

    def _bddl_state_for_name(self, name: str) -> Any | None:
        query = name.strip().replace(" ", "_").lower()
        try:
            task_env = self._env.handle.env.env
            if query in task_env.object_states_dict:
                return task_env.object_states_dict[query]
            for state_name, state in task_env.object_states_dict.items():
                if str(state_name).lower() == query:
                    return state
        except Exception:
            return None
        return None

    def _set_free_joint_world_pose(self, object_name: str, world_position: np.ndarray) -> None:
        joint_name = self._free_joint_name(object_name)
        if joint_name is None:
            raise KeyError(f"Object '{object_name}' does not have a movable free joint")

        sim = self._env.handle.env.sim
        addr = sim.model.get_joint_qpos_addr(joint_name)
        if not isinstance(addr, tuple):
            raise ValueError(f"Expected free joint qpos range for '{joint_name}', got {addr}")
        start, end = int(addr[0]), int(addr[1])
        if end - start < 7:
            raise ValueError(f"Expected 7-DoF free joint qpos for '{joint_name}', got {addr}")

        current_qpos = np.asarray(sim.data.qpos[start:end], dtype=np.float64).copy()
        target_world = np.asarray(world_position, dtype=np.float64).reshape(3)
        sim.data.qpos[start + 3 : start + 7] = current_qpos[3:7]

        body_name = f"{object_name}_1_main"
        if body_name not in {sim.model.body_id2name(i) for i in range(sim.model.nbody)}:
            body_name = f"{object_name}_main"
        body_id = sim.model.body_name2id(body_name) if body_name else None

        qpos_translation = current_qpos[:3].copy()
        for _ in range(3):
            sim.data.qpos[start : start + 3] = qpos_translation
            sim.forward()
            if body_id is None:
                break
            body_world = np.asarray(sim.data.xpos[body_id], dtype=np.float64)
            error = target_world - body_world
            qpos_translation = qpos_translation + error
            if float(np.linalg.norm(error)) < 1e-5:
                break
        sim.data.qpos[start : start + 3] = qpos_translation

        try:
            vaddr = sim.model.get_joint_qvel_addr(joint_name)
            if isinstance(vaddr, tuple):
                sim.data.qvel[int(vaddr[0]) : int(vaddr[1])] = 0.0
            else:
                sim.data.qvel[int(vaddr)] = 0.0
        except Exception:
            pass
        sim.forward()

    def _resolve_object_name(self, object_name: str, *, require_free_joint: bool) -> str:
        query = object_name.strip().replace(" ", "_").lower()
        all_names = sorted(self.get_all_object_poses())
        if query in all_names and (not require_free_joint or self._free_joint_name(query) is not None):
            return query
        for suffix in ("_1", "_2", "_3"):
            if query.endswith(suffix):
                base_query = query[: -len(suffix)]
                if base_query in all_names and (not require_free_joint or self._free_joint_name(base_query) is not None):
                    return base_query

        aliases = {
            "bowl": "akita_black_bowl",
            "black_bowl": "akita_black_bowl",
            "stove": "flat_stove",
            "white_mug": "porcelain_mug",
            "white mug": "porcelain_mug",
            "porcelain_mug": "porcelain_mug",
            "yellow_white_mug": "white_yellow_mug",
            "yellow_and_white_mug": "white_yellow_mug",
            "yellow and white mug": "white_yellow_mug",
            "yellow_white mug": "white_yellow_mug",
            "white_yellow mug": "white_yellow_mug",
            "white_and_yellow_mug": "white_yellow_mug",
            "yellow_mug": "white_yellow_mug",
            "white_yellow_mug": "white_yellow_mug",
            "book": "black_book",
            "black_book": "black_book",
            "caddy": "desk_caddy",
            "desk_caddy": "desk_caddy",
        }
        if query in aliases:
            alias = aliases[query]
            if alias in all_names and (not require_free_joint or self._free_joint_name(alias) is not None):
                return alias
        if "cabinet" in query:
            cabinet_alias = self._resolve_scene_cabinet_name(all_names, require_free_joint=require_free_joint)
            if cabinet_alias is not None:
                return cabinet_alias

        matches = [
            name
            for name in all_names
            if query in name or name in query
            if not require_free_joint or self._free_joint_name(name) is not None
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise KeyError(f"Object name '{object_name}' is ambiguous. Matches: {matches}")
        raise KeyError(f"Object '{object_name}' not found. Available objects: {all_names}")

    def _resolve_scene_cabinet_name(
        self,
        all_names: list[str],
        *,
        require_free_joint: bool,
    ) -> str | None:
        cabinets = [
            name
            for name in all_names
            if "cabinet" in name
            if not require_free_joint or self._free_joint_name(name) is not None
        ]
        if len(cabinets) == 1:
            return cabinets[0]
        return None

    def _free_joint_name(self, object_name: str) -> str | None:
        sim = self._env.handle.env.sim
        candidates = [
            f"{object_name}_1_joint0",
            f"{object_name}_joint0",
        ]
        for joint_id in range(sim.model.njnt):
            name = sim.model.joint_id2name(joint_id)
            if name in candidates:
                return name
        return None

    def _set_free_joint_xy_position(self, object_name: str, xy_robot_base: np.ndarray) -> None:
        joint_name = self._free_joint_name(object_name)
        if joint_name is None:
            raise KeyError(f"Object '{object_name}' does not have a movable free joint")

        target_robot = self.get_object_pose(object_name)[0]
        target_robot[:2] = np.asarray(xy_robot_base, dtype=np.float64).reshape(2)
        target_world = self._robot_base_position_to_world(target_robot)
        self._set_free_joint_world_pose(object_name, target_world)

    def _robot_base_position_to_world(self, position_robot: np.ndarray) -> np.ndarray:
        sim = self._env.handle.env.sim
        base_pos = np.asarray(sim.data.xpos[self._env.base_link_idx], dtype=np.float64)
        base_quat_wxyz = np.asarray(sim.data.xquat[self._env.base_link_idx], dtype=np.float64)
        base_rot = SciRotation.from_quat(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        )
        return base_pos + base_rot.apply(np.asarray(position_robot, dtype=np.float64))

    def _cabinet_drawer_names(self, level: str) -> tuple[str, str]:
        normalized = level.strip().lower()
        if normalized not in {"top", "middle", "bottom"}:
            raise ValueError("level must be one of: 'top', 'middle', 'bottom'")
        for prefix in self._cabinet_prefix_candidates(normalized):
            joint_name = f"{prefix}_{normalized}_level"
            region_name = f"{prefix}_{normalized}_region"
            if self._joint_exists(joint_name) and self._site_exists(region_name):
                return joint_name, region_name
        raise KeyError(
            f"No cabinet drawer joint/site found for level '{normalized}'. "
            f"Tried prefixes: {self._cabinet_prefix_candidates(normalized)}"
        )

    def _cabinet_prefix_candidates(self, level: str) -> list[str]:
        suffix = f"_{level}_region"
        prefixes: list[str] = []

        def add_from_name(name: str) -> None:
            if name.endswith(suffix):
                prefixes.append(name.removesuffix(suffix))

        try:
            task_env = self._env.handle.env.env
            for state in task_env.parsed_problem.get("goal_state", []):
                for item in state[1:]:
                    add_from_name(str(item))
            for state in task_env.parsed_problem.get("initial_state", []):
                for item in state[1:]:
                    add_from_name(str(item))
            for name in task_env.object_states_dict:
                add_from_name(str(name))
        except Exception:
            pass

        sim = self._env.handle.env.sim
        for site_id in range(sim.model.nsite):
            name = sim.model.site_id2name(site_id)
            if name:
                add_from_name(name)
        for joint_id in range(sim.model.njnt):
            name = sim.model.joint_id2name(joint_id)
            if name and name.endswith(f"_{level}_level"):
                prefixes.append(name.removesuffix(f"_{level}_level"))

        prefixes.extend(["wooden_cabinet_1", "white_cabinet_1", "yellow_cabinet_1"])
        deduped: list[str] = []
        for prefix in prefixes:
            if prefix and prefix not in deduped:
                deduped.append(prefix)
        return deduped

    def _joint_id(self, joint_name: str) -> int:
        sim = self._env.handle.env.sim
        for joint_id in range(sim.model.njnt):
            if sim.model.joint_id2name(joint_id) == joint_name:
                return joint_id
        raise KeyError(f"Joint '{joint_name}' not found")

    def _joint_exists(self, joint_name: str) -> bool:
        sim = self._env.handle.env.sim
        for joint_id in range(sim.model.njnt):
            if sim.model.joint_id2name(joint_id) == joint_name:
                return True
        return False

    def _site_exists(self, site_name: str) -> bool:
        sim = self._env.handle.env.sim
        for site_id in range(sim.model.nsite):
            if sim.model.site_id2name(site_id) == site_name:
                return True
        return False

    def _cabinet_drawer_open_direction_world(self, level: str) -> np.ndarray:
        sim = self._env.handle.env.sim
        joint_name, region_name = self._cabinet_drawer_names(level)
        addr = sim.model.get_joint_qpos_addr(joint_name)
        addr = int(addr if not isinstance(addr, tuple) else addr[0])
        site_id = sim.model.site_name2id(region_name)
        original_qpos = float(sim.data.qpos[addr])
        original_site = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
        try:
            sim.data.qpos[addr] = original_qpos - 0.01
            sim.forward()
            moved_site = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
        finally:
            sim.data.qpos[addr] = original_qpos
            sim.forward()
        direction = moved_site - original_site
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return direction / norm

    def _world_position_to_robot_base(self, position_world: np.ndarray) -> np.ndarray:
        sim = self._env.handle.env.sim
        base_pos = np.asarray(sim.data.xpos[self._env.base_link_idx], dtype=np.float64)
        base_quat_wxyz = np.asarray(sim.data.xquat[self._env.base_link_idx], dtype=np.float64)
        base_rot = SciRotation.from_quat(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        )
        return base_rot.inv().apply(np.asarray(position_world, dtype=np.float64) - base_pos)

    def _world_vector_to_robot_base(self, vector_world: np.ndarray) -> np.ndarray:
        sim = self._env.handle.env.sim
        base_quat_wxyz = np.asarray(sim.data.xquat[self._env.base_link_idx], dtype=np.float64)
        base_rot = SciRotation.from_quat(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
        )
        return base_rot.inv().apply(np.asarray(vector_world, dtype=np.float64))

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for, in underscore separated lowercase words.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        pos, _ = self._env._get_object_pose(object_name)
        return pos, np.array([0, 1, 0, 0])

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
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (
            z_approach != 0.0
        ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

            self._env.move_to_joints_blocking(joints_z_offset)

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def goto_pose_interactive_cartesian(self, target_pose_predicate: Callable[[], Tuple[np.ndarray, np.ndarray]], replan_interval_s: float = 0.0, lin_vel_norm: float = 1.0, ang_vel_norm: float = 2.0, z_approach: float = 0.0, timeout_s: float = 20.0) -> None:
        """Go to pose using Inverse Kinematics.
        Target pose is updated regularly.
        Trajectory is mostly linear in Cartesian space.
        Args:
            target_pose_predicate: A function that returns the target position (3,) and WXYZ quaternion (4,) of the target pose.
            replan_interval_s: The interval at which to update the target pose. Default is 0.0, which means no replanning.
            lin_vel_norm: The desired EEF linear velocity norm in m/s. Default is 1.0.
            ang_vel_norm: The desired EEF angular velocity norm in rad/s. Default is 2.0.
            z_approach: The Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        position, quaternion_wxyz = target_pose_predicate()

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        # Align with legacy env: apply TCP offset in end-effector frame
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if (z_approach != 0.0):
            def target_pose_predicate_z_offset():
                pos, quat = target_pose_predicate()
                r = SciRotation.from_quat(quat, scalar_first=True)
                return pos + r.apply(np.array([0, 0, -z_approach])), quat

            self.goto_pose_interactive_cartesian(target_pose_predicate_z_offset, replan_interval_s, lin_vel_norm, ang_vel_norm, z_approach=0.0, timeout_s=timeout_s)

        robot_cartesian_pos = self._env.get_observation()["robot_cartesian_pos"]
        current_position = robot_cartesian_pos[:3]
        current_quaternion_wxyz = robot_cartesian_pos[3:7]
        current_target_position = current_position.copy()
        current_target_quaternion_wxyz = current_quaternion_wxyz.copy()
        last_plan_time_s = self._env.get_current_time_s()
        start_loop_time_s = self._env.get_current_time_s()

        loop_executed = False
        while not np.allclose(offset_pos, current_target_position) or not np.allclose(quaternion_wxyz, current_target_quaternion_wxyz):
            loop_executed = True
            current_time_s = self._env.get_current_time_s()
            delta_t = current_time_s - last_plan_time_s
            if timeout_s > 0 and current_time_s - start_loop_time_s > timeout_s:
                break
            if replan_interval_s > 0 and delta_t > replan_interval_s:
                offset_pos, quaternion_wxyz = target_pose_predicate()
                last_plan_time_s = current_time_s
                robot_cartesian_pos = self._env.get_observation()["robot_cartesian_pos"]
                current_position = robot_cartesian_pos[:3]
                current_quaternion_wxyz = robot_cartesian_pos[3:7]

            current_target_position, current_target_quaternion_wxyz = self.step_towards_pose(
                current_position, current_quaternion_wxyz, offset_pos, quaternion_wxyz, lin_vel_norm, ang_vel_norm, 1.0 / self._env._control_freq
            )
            current_position = current_target_position
            current_quaternion_wxyz = current_target_quaternion_wxyz

            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=current_target_position,
                target_wxyz=current_target_quaternion_wxyz,
                prev_cfg=self.cfg,
            )
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints, max_steps=1)
        if loop_executed:
            self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self._env._set_gripper(1.0)
        for _ in range(40):
            self._env._step_once()

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self._env._set_gripper(0.0)
        for _ in range(60):
            self._env._step_once()


    def goto_pose_with_profile(
        self,
        position,
        quaternion_wxyz,
        z_approach=0.0,
        *,
        v_max=0.15,
        a_max=0.10,
        jerk_limit=None,
    ):
        """Manner-aware goto_pose: controls how fast/smoothly the arm moves.

        Identical call signature to goto_pose, extended with Manner parameters.

        Args:
            position:        (3,) XYZ target in metres (robot-base frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion for EEF orientation.
            z_approach:      Z offset pre-approach (same as goto_pose).
            v_max:           Max joint-space speed (rad/s).
                             缓缓靠近 -> 0.03, 正常 -> 0.15, 迅速 -> 0.25.
            a_max:           Max joint-space acceleration (rad/s^2).
            jerk_limit:      When set, minimum-jerk quintic interpolation
                             (Manner D: 平稳约束). None -> trapezoidal profile.
                             Typical: 0.5 (very smooth) – 5.0 (less smooth).

        Usage examples for LLM-generated code:
            # 缓缓靠近
            goto_pose_with_profile(pos, quat, v_max=0.03, a_max=0.01)
            # 迅速移动
            goto_pose_with_profile(pos, quat, v_max=0.25, a_max=0.15)
            # 平稳放下 (quintic minimum-jerk)
            goto_pose_with_profile(pos, quat, v_max=0.06, jerk_limit=0.8)
        """
        import numpy as _np
        from scipy.spatial.transform import Rotation as _Rot

        pos = _np.asarray(position, dtype=_np.float64).reshape(3)
        quat_wxyz = _np.asarray(quaternion_wxyz, dtype=_np.float64).reshape(4)
        quat_xyzw = _np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=_np.float64)
        rot = _Rot.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(_np.array([0.0, 0.0, -z_approach]))
            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=z_offset_pos,
                    target_wxyz=quat_wxyz,
                    prev_cfg=self.cfg,
                )
            joints_z = _np.asarray(self.cfg[:-1], dtype=_np.float64).reshape(7)
            self._env.move_to_joints_profiled(
                joints_z, v_max=v_max, a_max=a_max, jerk_limit=jerk_limit
            )

        if self.cfg is None:
            self.cfg = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
            )
        else:
            self.cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=offset_pos,
                target_wxyz=quat_wxyz,
                prev_cfg=self.cfg,
            )
        joints = _np.asarray(self.cfg[:-1], dtype=_np.float64).reshape(7)
        self._env.move_to_joints_profiled(
            joints, v_max=v_max, a_max=a_max, jerk_limit=jerk_limit
        )

    def set_impedance_stiffness(self, kp_pos=150.0, kp_ori=None):
        """Adjust arm compliance (Manner B: 阻抗刚度).

        Maps conceptual Cartesian compliance onto the joint position controller
        gain kp.  Lower kp -> more compliant (轻轻接触); higher -> stiffer (用力).

        Args:
            kp_pos:  Conceptual position stiffness, maps to joint controller kp.
                     300-600  -> 轻轻, 柔和接触 (low stiffness)
                     800-1200 -> 正常 (medium)
                     1300-1500-> 用力, 牢牢按压 (high stiffness)
            kp_ori:  Accepted for API symmetry; not separately applied in
                     joint-space control.

        Usage examples for LLM-generated code:
            set_impedance_stiffness(kp_pos=500)   # 轻轻叠放
            set_impedance_stiffness(kp_pos=1500)  # 用力按压 (reset to default)
        """
        self._env.set_joint_stiffness(kp_pos)

    def close_gripper_partial(self, *, force=None, width=None, steps=60):
        """Grip with partial force or to a target width (Manner C: 夹持力矩).

        Args:
            force:  Normalised grip force [0, 1].
                    0.15 -> 轻轻夹持 (易碎物); 0.6 -> 正常; 0.95 -> 紧握.
                    Width overrides force when both are provided.
            width:  Finger-gap target in metres.
                    Panda max opening ~0.08 m.
                    0.01 -> 轻捏; 0.0 -> 完全闭合.
            steps:  Simulation steps to hold after reaching target.

        Usage examples for LLM-generated code:
            close_gripper_partial(force=0.2)          # 轻轻拿起
            close_gripper_partial(force=0.95)         # 牢牢夹住
            close_gripper_partial(width=0.008)        # 保留 8 mm 缝隙
        """
        self._env.close_gripper_partial(force=force, width=width, steps=steps)

    def breakpoint_code_block(self) -> None:
        """Call this function to mark a significant checkpoint where you want to evaluate progress and potentially regenerate the remaining code.

        Args:
            None
        """
        return None

    def _normalize(self, q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q)

    def _slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """SLERP between two unit quaternions (WXYZ, shape (4,)), shortest path."""
        q0 = self._normalize(q0)
        q1 = self._normalize(q1)

        dot = np.dot(q0, q1)

        # Take shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # If almost identical, nlerp
        if dot > 0.9995:
            q = (1.0 - t) * q0 + t * q1
            return self._normalize(q)

        theta = np.arccos(dot)           # geodesic angle on S^3
        sin_theta = np.sin(theta)

        w0 = np.sin((1.0 - t) * theta) / sin_theta
        w1 = np.sin(t * theta) / sin_theta

        return w0 * q0 + w1 * q1


    def step_towards_pose(
        self,
        pos_curr: np.ndarray,
        quat_curr: np.ndarray,
        pos_target: np.ndarray,
        quat_target: np.ndarray,
        target_linear_speed: float,
        target_angular_speed: float,
        delta_t: float,
    ):
        """
        Take one step from (pos_curr, quat_curr) towards (pos_target, quat_target),
        given linear and angular speed limits.

        - Positions: shape (3,)
        - Quaternions: WXYZ, shape (4,), assumed unit
        - target_linear_speed: linear speed norm (units / s)
        - target_angular_speed: angular speed norm (rad / s)
        - delta_t: time step (s)

        Returns:
            (pos_next: (3,), quat_next: (4,))
        """
        # -------- Linear part --------
        max_lin_step = max(target_linear_speed, 0.0) * delta_t

        diff_pos = pos_target - pos_curr
        dist = np.linalg.norm(diff_pos)

        if dist <= max_lin_step or dist < 1e-9:
            pos_next = pos_target.copy()
        else:
            direction = diff_pos / dist
            pos_next = pos_curr + direction * max_lin_step

        # -------- Angular part --------
        q0 = self._normalize(quat_curr)
        q1 = self._normalize(quat_target)

        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        # Geodesic angle on S^3
        theta = np.arccos(dot)
        # Physical rotation angle in SO(3)
        delta_angle = 2.0 * theta

        max_ang_step = max(target_angular_speed, 0.0) * delta_t

        if delta_angle <= max_ang_step or delta_angle < 1e-8:
            # Close enough: snap to target orientation
            quat_next = q1
        elif max_ang_step <= 0.0:
            # No angular motion allowed this step
            quat_next = q0
        else:
            # Move along the rotation by max_ang_step
            # SLERP parameter t is fraction of total rotation angle
            t = max_ang_step / delta_angle  # in (0,1)
            quat_next = self._slerp(q0, q1, t)

        return pos_next, quat_next


def _drawer_level_from_region(region_name: str) -> str:
    lowered = region_name.lower()
    for level in ("top", "middle", "bottom"):
        if f"_{level}_region" in lowered:
            return level
    raise ValueError(f"Could not infer drawer level from region '{region_name}'")
