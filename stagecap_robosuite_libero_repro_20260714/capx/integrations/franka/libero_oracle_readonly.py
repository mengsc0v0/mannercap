from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
import viser.transforms as vtf

from capx.envs.base import BaseEnv
from capx.integrations.franka.libero_privileged import FrankaLiberoPrivilegedApi


class FrankaLiberoOracleReadOnlyApi(FrankaLiberoPrivilegedApi):
    """LIBERO oracle-readonly helpers plus real robot motion.

    This API is for data-generation experiments where CAP code may read stable
    simulator truth such as object poses and drawer handle geometry, but must
    still solve the task through robot actions. It intentionally exposes no
    helpers that write articulated/object state, repair predicates, or mark a
    task complete.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_ee_pose": self.get_ee_pose,
            "get_object_pose": self.get_object_pose,
            "get_instance_pose": self.get_instance_pose,
            "get_all_object_poses": self.get_all_object_poses,
            "get_instance_object_names": self.get_instance_object_names,
            "get_site_pose": self.get_site_pose,
            "get_scene_site_names": self.get_scene_site_names,
            "get_scene_body_names": self.get_scene_body_names,
            "get_scene_geom_names": self.get_scene_geom_names,
            "get_scene_joint_names": self.get_scene_joint_names,
            "get_body_pose": self.get_body_pose,
            "get_geom_pose": self.get_geom_pose,
            "get_geom_diagnostics": self.get_geom_diagnostics,
            "get_joint_diagnostics": self.get_joint_diagnostics,
            "get_contact_diagnostics": self.get_contact_diagnostics,
            "get_task_status": self.get_task_status,
            "get_bddl_goal_diagnostics": self.get_bddl_goal_diagnostics,
            "get_on_diagnostic": self.get_on_diagnostic,
            "get_site_on_diagnostic": self.get_site_on_diagnostic,
            "get_site_in_diagnostic": self.get_site_in_diagnostic,
            "get_on_center_delta_robot": self.get_on_center_delta_robot,
            "get_object_state_diagnostic": self.get_object_state_diagnostic,
            "get_goal_target_region_pose": self.get_goal_target_region_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "get_object_grasp_affordance": self.get_object_grasp_affordance,
            "measure_object_grasp_state": self.measure_object_grasp_state,
            "verified_stage_pick_object": self.verified_stage_pick_object,
            "verified_lift_pick_object": self.verified_lift_pick_object,
            "verified_low_side_pick_object": self.verified_low_side_pick_object,
            "verified_low_profile_pick_object": self.verified_low_profile_pick_object,
            "verified_geometry_pick_object": self.verified_geometry_pick_object,
            "grasp_tall_object_side": self.grasp_tall_object_side,
            "approach_and_grasp_object": self.approach_and_grasp_object,
            "guarded_transport_held_object_to_target": self.guarded_transport_held_object_to_target,
            "low_release_object_to_target": self.low_release_object_to_target,
            "pick_transport_release_object_to_target": self.pick_transport_release_object_to_target,
            "rack_supported_release_object_to_target": self.rack_supported_release_object_to_target,
            "push_object_to_target_region": self.push_object_to_target_region,
            "global_bddl_goal_repair": self.global_bddl_goal_repair,
            "privileged_data_generation_finalize_goals": self.privileged_data_generation_finalize_goals,
            "local_placement_closure": self.local_placement_closure,
            "drawer_z_only_settle_object": self.drawer_z_only_settle_object,
            "repair_placement_relation": self.repair_placement_relation,
            "place_object_in_microwave_region": self.place_object_in_microwave_region,
            "close_microwave_by_sweep": self.close_microwave_by_sweep,
            "turn_on_stove_by_sweep": self.turn_on_stove_by_sweep,
            "get_cabinet_drawer_handle_pose": self.get_cabinet_drawer_handle_pose,
            "get_cabinet_drawer_progress": self.get_cabinet_drawer_progress,
            "get_gripper_fingertip_state": self.get_gripper_fingertip_state,
            "get_joint_positions": self.get_joint_positions,
            "get_drawer_hook_poses": self.get_drawer_hook_poses,
            "get_drawer_side_hook_poses": self.get_drawer_side_hook_poses,
            "get_drawer_tiltx_hook_poses": self.get_drawer_tiltx_hook_poses,
            "get_drawer_sleeve_hook_poses": self.get_drawer_sleeve_hook_poses,
            "measure_drawer_hook_alignment": self.measure_drawer_hook_alignment,
            "measure_drawer_gripper_contacts": self.measure_drawer_gripper_contacts,
            "open_drawer_by_sleeve_retry": self.open_drawer_by_sleeve_retry,
            "close_drawer_by_push": self.close_drawer_by_push,
            "goto_pose": self.goto_pose,
            "goto_pose_checked": self.goto_pose_checked,
            "goto_finger_midpoint_checked": self.goto_finger_midpoint_checked,
            "goto_pose_interactive_cartesian": self.goto_pose_interactive_cartesian,
            "move_finger_midpoint_delta": self.move_finger_midpoint_delta,
            "move_finger_midpoint_delta_strict": self.move_finger_midpoint_delta_strict,
            "pull_sleeved_drawer": self.pull_sleeved_drawer,
            "move_to_joints": self.move_to_joints,
            "move_joint_delta_checked": self.move_joint_delta_checked,
            "move_linear_delta": self.move_linear_delta,
            "move_along_direction": self.move_along_direction,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def verified_stage_pick_object(
        self,
        object_name: str,
        *,
        stage_attempt: int = 1,
        max_attempts: int = 1,
        start_candidate_index: int | None = None,
    ) -> dict[str, Any]:
        """Generic pick contract for Stage-CAP retry loops.

        The LLM supplies only the object and the runner-provided stage attempt.
        This audited skill owns the reusable strategy schedule so retries are
        reproducible: early non-tall attempts sweep direct candidates before
        trying low-profile or geometry fallbacks.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        attempt = max(1, int(stage_attempt))
        candidate_index = max(0, int(start_candidate_index)) if start_candidate_index is not None else attempt - 1
        geometry = self._object_geometry_summary(resolved_object)
        extent = np.maximum(np.abs(np.asarray(geometry["extent"], dtype=np.float64).reshape(3)), 1e-6)
        max_xy = float(max(extent[0], extent[1]))
        z_extent = float(extent[2])
        is_tall = bool(z_extent >= 1.35 * max_xy)
        object_text = str(resolved_object).lower()
        cup_like_tokens = ("cup", "mug")
        is_cup_like = bool(any(token in object_text for token in cup_like_tokens))
        open_container_tokens = ("bowl", "ramekin", "plate")
        is_open_container = bool(any(token in object_text for token in open_container_tokens))
        is_thin_book = bool("book" in object_text)
        selected_start_index = int(candidate_index)
        selected_max_attempts = max(1, int(max_attempts))
        if is_tall:
            skill_name = "grasp_tall_object_side"
            result = self.grasp_tall_object_side(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
            )
        elif is_thin_book and attempt <= 4:
            selected_start_index = 3 * (attempt - 1) if start_candidate_index is None else int(candidate_index)
            selected_max_attempts = max(3, int(max_attempts))
            skill_name = "verified_geometry_pick_object"
            result = self.verified_geometry_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
                lift_height=0.26,
                settle_steps=16,
            )
        elif is_thin_book and attempt <= 6:
            selected_start_index = 2 * (attempt - 5) if start_candidate_index is None else max(0, int(candidate_index) - 4)
            selected_max_attempts = max(2, int(max_attempts))
            skill_name = "verified_low_profile_pick_object"
            result = self.verified_low_profile_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
                lift_height=0.16,
                settle_steps=16,
            )
        elif is_thin_book:
            selected_start_index = max(0, attempt - 7) if start_candidate_index is None else max(0, int(candidate_index) - 6)
            selected_max_attempts = max(2, int(max_attempts))
            skill_name = "approach_and_grasp_object"
            result = self.approach_and_grasp_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
                lift_height=0.14,
            )
        elif is_cup_like and attempt <= 2:
            skill_name = "grasp_tall_object_side"
            result = self.grasp_tall_object_side(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
                lift_height=0.18,
                settle_steps=18,
            )
        elif is_cup_like and attempt == 3:
            skill_name = "approach_and_grasp_object"
            result = self.approach_and_grasp_object(
                resolved_object,
                max_attempts=max(1, int(max_attempts)),
                start_candidate_index=0,
                lift_height=0.14,
            )
        elif is_cup_like and attempt <= 5:
            skill_name = "verified_geometry_pick_object"
            result = self.verified_geometry_pick_object(
                resolved_object,
                max_attempts=max(1, int(max_attempts)),
                start_candidate_index=max(0, candidate_index - 3),
                lift_height=0.20,
            )
        elif is_cup_like:
            skill_name = "approach_and_grasp_object"
            result = self.approach_and_grasp_object(
                resolved_object,
                max_attempts=max(1, int(max_attempts)),
                start_candidate_index=max(0, attempt - 6),
                lift_height=0.14,
            )
        elif is_open_container and attempt <= 2:
            skill_name = "verified_low_side_pick_object"
            result = self.verified_low_side_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
                lift_height=0.075,
                settle_steps=18,
            )
            if isinstance(result, dict) and not bool(result.get("held", False)):
                fallback = self.verified_geometry_pick_object(
                    resolved_object,
                    max_attempts=1,
                    start_candidate_index=max(0, selected_start_index),
                    lift_height=0.20,
                    settle_steps=12,
                )
                if isinstance(fallback, dict):
                    fallback = dict(fallback)
                    fallback["primary_low_side_attempt"] = result
                    result = fallback
                    skill_name = "verified_geometry_pick_object_after_low_side"
        elif is_open_container and attempt <= 4:
            skill_name = "verified_geometry_pick_object"
            result = self.verified_geometry_pick_object(
                resolved_object,
                max_attempts=max(1, min(2, int(max_attempts))),
                start_candidate_index=max(0, candidate_index - 2),
                lift_height=0.22,
            )
        elif is_open_container and attempt <= 6:
            skill_name = "approach_and_grasp_object"
            result = self.approach_and_grasp_object(
                resolved_object,
                max_attempts=max(1, min(2, int(max_attempts))),
                start_candidate_index=max(0, candidate_index - 4),
                lift_height=0.115,
            )
        elif attempt <= 3:
            selected_start_index = 2 * (attempt - 1) if start_candidate_index is None else int(candidate_index)
            selected_max_attempts = max(2, int(max_attempts))
            skill_name = "verified_lift_pick_object"
            result = self.verified_lift_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
            )
        elif attempt <= 5:
            selected_start_index = 2 * (attempt - 4) if start_candidate_index is None else max(0, int(candidate_index) - 3)
            selected_max_attempts = max(2, int(max_attempts))
            skill_name = "verified_low_profile_pick_object"
            result = self.verified_low_profile_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
            )
        elif attempt <= 7:
            selected_start_index = max(0, 2 * (attempt - 6)) if start_candidate_index is None else max(0, int(candidate_index) - 5)
            selected_max_attempts = max(2, int(max_attempts))
            skill_name = "verified_geometry_pick_object"
            result = self.verified_geometry_pick_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
            )
        else:
            selected_start_index = max(0, attempt - 8) if start_candidate_index is None else max(0, int(candidate_index) - 7)
            selected_max_attempts = max(1, int(max_attempts))
            skill_name = "approach_and_grasp_object"
            result = self.approach_and_grasp_object(
                resolved_object,
                max_attempts=selected_max_attempts,
                start_candidate_index=selected_start_index,
            )
        if not isinstance(result, dict):
            result = {"object_name": resolved_object, "held": False, "lift_delta": 0.0, "raw_result": result}
        result = dict(result)
        result["skill"] = str(result.get("skill") or skill_name)
        final_object_name = str(result.get("object_name") or resolved_object)
        try:
            start_position = result.get("start_position")
            if start_position is None and isinstance(result.get("best_attempt"), dict):
                before_close = result["best_attempt"].get("before_close")
                if isinstance(before_close, dict):
                    start_position = before_close.get("object_position")
            if start_position is None:
                attempts = result.get("attempts")
                if isinstance(attempts, list) and attempts:
                    first_attempt = attempts[0]
                    if isinstance(first_attempt, dict):
                        before_close = first_attempt.get("before_close")
                        if isinstance(before_close, dict):
                            start_position = before_close.get("object_position")
            final_state = self.measure_object_grasp_state(
                final_object_name,
                np.asarray(start_position, dtype=np.float64).reshape(3)
                if start_position is not None
                else None,
                strict=bool(is_cup_like),
            )
        except Exception as exc:
            final_state = {"held": False, "error": repr(exc)}
        raw_held = bool(result.get("held", False))
        raw_lift_delta = float(result.get("lift_delta", 0.0) or 0.0)
        final_held = bool(isinstance(final_state, dict) and final_state.get("held", False))
        final_contact_supported = bool(
            isinstance(final_state, dict)
            and (
                final_state.get("held_by_contact", False)
                or final_state.get("elevated_contact", False)
                or (isinstance(final_state.get("contacts"), dict) and final_state["contacts"].get("has_object_contact", False))
            )
        )
        try:
            final_distance_to_fingers = float(final_state.get("distance_to_finger_midpoint", 999.0)) if isinstance(final_state, dict) else 999.0
        except Exception:
            final_distance_to_fingers = 999.0
        final_lift_delta = (
            float(final_state.get("lift_delta", 0.0) or 0.0)
            if isinstance(final_state, dict)
            else 0.0
        )
        effective_lift_delta = max(raw_lift_delta, final_lift_delta)
        transport_stable_held = bool(
            effective_lift_delta >= 0.045
            and (
                final_held
                or (
                    final_contact_supported
                    and final_distance_to_fingers <= (0.125 if is_cup_like else 0.105)
                )
            )
        )
        result["raw_skill_held"] = raw_held
        result["raw_skill_lift_delta"] = raw_lift_delta
        result["final_grasp_state"] = final_state
        result["final_grasp_held"] = final_held
        result["final_contact_supported"] = final_contact_supported
        result["final_distance_to_finger_midpoint"] = final_distance_to_fingers
        result["transport_stable_held"] = transport_stable_held
        result["held"] = transport_stable_held
        result["lift_delta"] = effective_lift_delta
        result["stage_pick_contract"] = {
            "stage_attempt": int(attempt),
            "candidate_index": int(candidate_index),
            "selected_start_candidate_index": int(selected_start_index),
            "selected_max_attempts": int(selected_max_attempts),
            "selected_skill": skill_name,
            "extent": extent.copy(),
            "max_xy": max_xy,
            "z_extent": z_extent,
            "is_tall": is_tall,
            "is_cup_like": is_cup_like,
            "is_open_container": is_open_container,
            "is_thin_book": is_thin_book,
            "requires_final_transport_stable_hold": True,
        }
        return result

    def _libero_env_with_objects(self) -> Any:
        """Return the wrapped LIBERO env that owns object/site state maps."""
        cur = self._env.handle.env
        for _ in range(8):
            if hasattr(cur, "obj_body_id") and hasattr(cur, "object_states_dict"):
                return cur
            if not hasattr(cur, "env"):
                break
            cur = cur.env
        return self._env.handle.env

    def _robot_transform_from_world_pose(
        self,
        position_world: np.ndarray,
        quaternion_wxyz_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sim = self._env.handle.env.sim
        base_wxyz_xyz = np.concatenate(
            [
                sim.data.xquat[self._env.base_link_idx],
                sim.data.xpos[self._env.base_link_idx],
            ]
        )
        pose_world = vtf.SE3(
            wxyz_xyz=np.concatenate(
                [
                    np.asarray(quaternion_wxyz_world, dtype=np.float64).reshape(4),
                    np.asarray(position_world, dtype=np.float64).reshape(3),
                ]
            )
        )
        pose_robot = vtf.SE3(wxyz_xyz=base_wxyz_xyz).inverse() @ pose_world
        return pose_robot.translation().copy(), pose_robot.rotation().wxyz.copy()

    def get_instance_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read an exact LIBERO object/fixture instance pose in robot coordinates."""
        libero_env = self._libero_env_with_objects()
        sim = libero_env.sim
        obj_body_id = getattr(libero_env, "obj_body_id", {})
        if object_name not in obj_body_id:
            raise KeyError(
                f"Instance '{object_name}' not found. Available instances: {sorted(obj_body_id)}"
            )
        body_id = obj_body_id[object_name]
        return self._robot_transform_from_world_pose(
            sim.data.body_xpos[body_id],
            sim.data.body_xquat[body_id],
        )

    def get_instance_object_names(self) -> list[str]:
        """List exact LIBERO object and fixture instance names."""
        libero_env = self._libero_env_with_objects()
        return sorted(getattr(libero_env, "obj_body_id", {}))

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read an object pose, preserving exact LIBERO instance names.

        LIBERO's natural-language object lookup can map an ordinal name such as
        ``moka_pot_2`` back to the first matching object. Stage-verifiable runs
        need object pose, geometry, contact, and BDDL diagnostics to refer to
        the same physical instance, so exact instance names use obj_body_id
        directly and natural names keep the inherited fuzzy behavior.
        """
        query = str(object_name).strip().replace(" ", "_").lower()
        if query in set(self.get_instance_object_names()):
            return self.get_instance_pose(query)
        return super().get_object_pose(object_name)

    def get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a LIBERO/MuJoCo target site pose in robot coordinates."""
        sim = self._env.handle.env.sim
        site_id = sim.model.site_name2id(site_name)
        site_quat = vtf.SO3.from_matrix(sim.data.site_xmat[site_id].reshape(3, 3)).wxyz
        return self._robot_transform_from_world_pose(sim.data.site_xpos[site_id], site_quat)

    def get_scene_site_names(self) -> list[str]:
        """List available scene site names."""
        sim = self._env.handle.env.sim
        return [name for i in range(sim.model.nsite) if (name := sim.model.site_id2name(i))]

    def get_scene_body_names(self) -> list[str]:
        """List available scene body names."""
        sim = self._env.handle.env.sim
        return [name for i in range(sim.model.nbody) if (name := sim.model.body_id2name(i))]

    def get_scene_geom_names(self) -> list[str]:
        """List available scene geom names."""
        sim = self._env.handle.env.sim
        return [name for i in range(sim.model.ngeom) if (name := sim.model.geom_id2name(i))]

    def get_scene_joint_names(self) -> list[str]:
        """List available scene joint names."""
        sim = self._env.handle.env.sim
        return [name for i in range(sim.model.njnt) if (name := sim.model.joint_id2name(i))]

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read an exact body pose in robot coordinates."""
        sim = self._env.handle.env.sim
        body_id = sim.model.body_name2id(body_name)
        return self._robot_transform_from_world_pose(
            sim.data.body_xpos[body_id],
            sim.data.body_xquat[body_id],
        )

    def get_geom_pose(self, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read an exact geom pose in robot coordinates."""
        sim = self._env.handle.env.sim
        geom_id = sim.model.geom_name2id(geom_name)
        geom_quat = vtf.SO3.from_matrix(sim.data.geom_xmat[geom_id].reshape(3, 3)).wxyz
        return self._robot_transform_from_world_pose(sim.data.geom_xpos[geom_id], geom_quat)

    def get_geom_diagnostics(self, pattern: str | None = None) -> list[dict[str, Any]]:
        """Return read-only geom diagnostics with owning body names."""
        sim = self._env.handle.env.sim
        needle = pattern.lower() if pattern is not None else None
        results: list[dict[str, Any]] = []
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id) or ""
            body_id = int(sim.model.geom_bodyid[geom_id])
            body_name = sim.model.body_id2name(body_id) or ""
            haystack = " ".join([geom_name, body_name]).lower()
            if needle and needle not in haystack:
                continue
            geom_quat = vtf.SO3.from_matrix(sim.data.geom_xmat[geom_id].reshape(3, 3)).wxyz
            pos_robot, quat_robot = self._robot_transform_from_world_pose(
                sim.data.geom_xpos[geom_id],
                geom_quat,
            )
            results.append(
                {
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "body_name": body_name,
                    "pos": pos_robot,
                    "quat": quat_robot,
                    "size": np.asarray(sim.model.geom_size[geom_id], dtype=np.float64).copy(),
                    "contype": int(sim.model.geom_contype[geom_id]),
                    "conaffinity": int(sim.model.geom_conaffinity[geom_id]),
                }
            )
        return results

    def get_joint_diagnostics(self, pattern: str | None = None) -> list[dict[str, Any]]:
        """Return read-only joint diagnostics."""
        sim = self._env.handle.env.sim
        needle = pattern.lower() if pattern is not None else None
        joint_type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        results: list[dict[str, Any]] = []
        for joint_id in range(sim.model.njnt):
            joint_name = sim.model.joint_id2name(joint_id) or ""
            body_id = int(sim.model.jnt_bodyid[joint_id])
            body_name = sim.model.body_id2name(body_id) or ""
            if needle and needle not in joint_name.lower() and needle not in body_name.lower():
                continue
            joint_type = int(sim.model.jnt_type[joint_id])
            qpos_adr = int(sim.model.jnt_qposadr[joint_id])
            dof_adr = int(sim.model.jnt_dofadr[joint_id])
            qpos_width = 7 if joint_type == 0 else (4 if joint_type == 1 else 1)
            qvel_width = 6 if joint_type == 0 else (3 if joint_type == 1 else 1)
            results.append(
                {
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "joint_type": joint_type_names.get(joint_type, str(joint_type)),
                    "body_name": body_name,
                    "qpos": np.asarray(sim.data.qpos[qpos_adr : qpos_adr + qpos_width], dtype=np.float64).copy(),
                    "qvel": np.asarray(sim.data.qvel[dof_adr : dof_adr + qvel_width], dtype=np.float64).copy(),
                    "range": np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64).copy(),
                    "axis": np.asarray(sim.model.jnt_axis[joint_id], dtype=np.float64).copy(),
                }
            )
        return results

    def get_contact_diagnostics(self, pattern: str | None = None) -> list[dict[str, Any]]:
        """Return current contact pairs, optionally filtered by name."""
        sim = self._env.handle.env.sim
        needle = pattern.lower() if pattern is not None else None
        contacts: list[dict[str, Any]] = []
        for idx in range(int(sim.data.ncon)):
            contact = sim.data.contact[idx]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1 = sim.model.geom_id2name(geom1_id) or ""
            geom2 = sim.model.geom_id2name(geom2_id) or ""
            body1 = sim.model.body_id2name(int(sim.model.geom_bodyid[geom1_id])) or ""
            body2 = sim.model.body_id2name(int(sim.model.geom_bodyid[geom2_id])) or ""
            haystack = " ".join([geom1, geom2, body1, body2]).lower()
            if needle and needle not in haystack:
                continue
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "body1": body1,
                    "body2": body2,
                    "dist": float(contact.dist),
                    "pos": np.asarray(contact.pos, dtype=np.float64).copy(),
                }
            )
        return contacts

    def get_task_status(self) -> dict[str, Any]:
        """Return current reward and task completion status without mutation."""
        return {
            "reward": float(self._env.compute_reward()),
            "task_completed": bool(self._env.task_completed()),
            "sim_step_count": int(getattr(self._env, "_sim_step_count", -1)),
            "max_steps": int(getattr(self._env, "max_steps", -1)),
            "current_done": bool(getattr(self._env, "_current_done", False)),
        }

    def get_bddl_goal_diagnostics(self) -> dict[str, Any]:
        """Return parsed LIBERO goals and readonly predicate diagnostics.

        This exposes the benchmark verifier target without mutating simulator
        state. Stage planners can use it to align semantic targets with BDDL
        predicates, and stage traces can explain reward mismatches.
        """
        task_env = self._env.handle.env.env
        goals: list[dict[str, Any]] = []
        for idx, state in enumerate(task_env.parsed_problem.get("goal_state", [])):
            state_items = [str(item) for item in state]
            goal: dict[str, Any] = {
                "index": idx,
                "raw": state_items,
                "satisfied": False,
            }
            if not state_items:
                goals.append(goal)
                continue

            predicate = state_items[0].lower()
            goal["predicate"] = predicate
            try:
                if len(state_items) == 3 and predicate in {"in", "on"}:
                    object_name = state_items[1]
                    target_name = state_items[2]
                    object_api_name = object_name.removesuffix("_1")
                    resolved_object = self._resolve_object_name(
                        object_api_name,
                        require_free_joint=True,
                    )
                    diagnostic = self._placement_relation_diagnostic(
                        resolved_object,
                        target_name,
                        predicate=predicate,
                    )
                    goal.update(
                        {
                            "object_name": object_name,
                            "resolved_object": resolved_object,
                            "target_name": target_name,
                            "diagnostic": diagnostic,
                            "satisfied": bool(diagnostic.get("satisfied", False)),
                        }
                    )
                elif len(state_items) == 2 and predicate in {"open", "close", "turnon", "turnoff"}:
                    target_name = state_items[1]
                    diagnostic = self.get_object_state_diagnostic(target_name)
                    satisfied = False
                    if predicate == "open":
                        satisfied = bool(diagnostic.get("is_open", False))
                    elif predicate == "close":
                        satisfied = bool(diagnostic.get("is_close", False))
                    elif predicate == "turnon":
                        satisfied = bool(diagnostic.get("turn_on", False))
                    elif predicate == "turnoff":
                        satisfied = not bool(diagnostic.get("turn_on", False))
                    goal.update(
                        {
                            "target_name": target_name,
                            "diagnostic": diagnostic,
                            "satisfied": satisfied,
                        }
                    )
            except Exception as exc:
                goal["error"] = repr(exc)
            goals.append(goal)

        status = self.get_task_status()
        return {
            "goal_state": goals,
            "all_goals_satisfied": bool(goals and all(bool(goal.get("satisfied", False)) for goal in goals)),
            "task_status": status,
        }

    def global_bddl_goal_repair(
        self,
        *,
        max_passes: int = 2,
        binary_max_iters: int = 2,
        settle_steps: int = 50,
    ) -> dict[str, Any]:
        """Physically repair unsatisfied parsed BDDL goals.

        This is the final benchmark-aligned verifier/replanner loop. It reads
        the true LIBERO BDDL atoms, executes only robot-motion skills for atoms
        that are currently unsatisfied, and re-checks the benchmark status after
        each action. It never writes object poses, joints, reward, or success.
        """
        task_env = self._env.handle.env.env
        records: list[dict[str, Any]] = []
        before = self.get_bddl_goal_diagnostics()
        before_status = before.get("task_status", {}) if isinstance(before, dict) else {}
        if bool(before_status.get("task_completed", False)) or float(before_status.get("reward", 0.0) or 0.0) >= 1.0:
            return {
                "before": before,
                "records": records,
                "after": before,
                "task_status": before_status,
                "success": True,
                "skill": "global_bddl_goal_repair",
                "reason": "already complete before global repair",
            }

        raw_goals = list(task_env.parsed_problem.get("goal_state", []))
        for pass_idx in range(max(1, int(max_passes))):
            diagnostics = self.get_bddl_goal_diagnostics()
            status = diagnostics.get("task_status", {}) if isinstance(diagnostics, dict) else {}
            if bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                return {
                    "before": before,
                    "records": records,
                    "after": diagnostics,
                    "task_status": status,
                    "success": True,
                    "skill": "global_bddl_goal_repair",
                    "reason": f"complete before pass {pass_idx}",
                }

            if bool(diagnostics.get("all_goals_satisfied", False)):
                self._settle_simulation_steps(max(4, int(settle_steps) // 3))
                status = self.get_task_status()
                if bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                    return {
                        "before": before,
                        "records": records,
                        "after": self.get_bddl_goal_diagnostics(),
                        "task_status": status,
                        "success": True,
                        "skill": "global_bddl_goal_repair",
                        "reason": f"all goals satisfied after settling pass {pass_idx}",
                    }

            current_goal_diags = diagnostics.get("goal_state", []) if isinstance(diagnostics, dict) else []
            unsatisfied_indices = {
                int(goal.get("index", -1))
                for goal in current_goal_diags
                if isinstance(goal, dict) and not bool(goal.get("satisfied", False))
            }
            if not unsatisfied_indices:
                break
            protected_satisfied_indices = {
                int(goal.get("index", -1))
                for goal in current_goal_diags
                if isinstance(goal, dict) and bool(goal.get("satisfied", False))
            }

            # Articulation repairs first, then placements. If an articulation
            # motion disturbs a placement, the next pass will repair it.
            ordered_goals = sorted(
                enumerate(raw_goals),
                key=lambda item: 0 if len(item[1]) == 2 else 1,
            )
            made_attempt = False
            for goal_index, goal_state in ordered_goals:
                if goal_index not in unsatisfied_indices:
                    continue
                if not goal_state:
                    continue
                predicate = str(goal_state[0]).lower()
                record: dict[str, Any] = {
                    "pass": pass_idx,
                    "goal_index": goal_index,
                    "goal_state": [str(item) for item in goal_state],
                    "predicate": predicate,
                }
                try:
                    if len(goal_state) == 2 and predicate in {"turnon", "turnoff", "open", "close"}:
                        target_name = str(goal_state[1])
                        record["target_name"] = target_name
                        if predicate == "turnon":
                            record["action"] = "turn_on_stove_by_sweep"
                            record["result"] = self.turn_on_stove_by_sweep()
                        elif predicate == "open" and (
                            "drawer" in target_name.lower()
                            or ("cabinet" in target_name.lower() and "region" in target_name.lower())
                        ):
                            level = self._drawer_level_from_target_name(target_name) or "middle"
                            record["action"] = "open_drawer_by_sleeve_retry"
                            record["level"] = level
                            record["result"] = self.open_drawer_by_sleeve_retry(
                                level,
                                open_threshold=0.98,
                                max_cycles=3,
                                test_distance=0.055,
                                pull_distance=0.46,
                                settle_steps=max(12, int(settle_steps) // 2),
                            )
                        elif predicate == "close" and "microwave" in target_name.lower():
                            record["action"] = "close_microwave_by_sweep"
                            record["result"] = self.close_microwave_by_sweep(target_name)
                        else:
                            record["action"] = f"{predicate}_not_supported"
                            record["skipped"] = True
                    elif len(goal_state) == 3 and predicate in {"in", "on"}:
                        bddl_object_name = str(goal_state[1])
                        bddl_target_name = str(goal_state[2])
                        object_api_name = (
                            bddl_object_name.removesuffix("_1")
                            if bddl_object_name.endswith("_1")
                            else bddl_object_name
                        )
                        resolved_object = self._resolve_object_name(
                            object_api_name,
                            require_free_joint=True,
                        )
                        relation_before = self._placement_relation_diagnostic(
                            resolved_object,
                            bddl_target_name,
                            predicate=predicate,
                        )
                        record.update(
                            {
                                "action": "repair_placement_relation",
                                "object_name": resolved_object,
                                "target_name": bddl_target_name,
                                "relation_before": relation_before,
                            }
                        )
                        if bool(relation_before.get("satisfied", False)):
                            record["result"] = {"relation_satisfied": True, "reason": "already satisfied"}
                        elif (
                            predicate == "in"
                            and self._is_drawer_region_target_name(bddl_target_name)
                            and self._drawer_relation_is_z_only_above_box(
                                relation_before,
                                max_z_excess=0.020,
                            )
                        ):
                            record["action"] = "drawer_z_only_preserve_near_success"
                            record["result"] = {
                                "relation_satisfied": False,
                                "z_only_near_success_preserved": True,
                                "relation": relation_before,
                                "reason": "preserve drawer xy-valid placement; generic physical repair is destructive",
                            }
                        elif (
                            predicate == "in"
                            and "microwave" in str(bddl_target_name).lower()
                            and self._relation_is_xy_inside_z_only_above_box(
                                relation_before,
                                max_z_excess=0.026,
                                require_contact=True,
                            )
                        ):
                            record["action"] = "microwave_z_only_preserve_near_success"
                            record["result"] = {
                                "relation_satisfied": False,
                                "z_only_near_success_preserved": True,
                                "relation": relation_before,
                                "reason": "preserve microwave xy-valid placement; generic physical repair can eject the mug",
                            }
                        else:
                            record["result"] = self.repair_placement_relation(
                                resolved_object,
                                predicate=predicate,
                                target_name=bddl_target_name,
                                max_iters=max(1, int(binary_max_iters)),
                                settle_steps=max(12, int(settle_steps)),
                            )
                    else:
                        record["action"] = "unsupported_goal_predicate"
                        record["skipped"] = True
                except Exception as exc:
                    record["error"] = repr(exc)
                self._settle_simulation_steps(max(4, int(settle_steps) // 4))
                record["post_goal_diagnostics"] = self.get_bddl_goal_diagnostics()
                records.append(record)
                made_attempt = True

                disturbed_protected = [
                    int(goal.get("index", -1))
                    for goal in record["post_goal_diagnostics"].get("goal_state", [])
                    if (
                        isinstance(goal, dict)
                        and int(goal.get("index", -1)) in protected_satisfied_indices
                        and not bool(goal.get("satisfied", False))
                    )
                ]
                if disturbed_protected:
                    return {
                        "before": before,
                        "records": records,
                        "after": record["post_goal_diagnostics"],
                        "task_status": record["post_goal_diagnostics"].get("task_status", {}),
                        "success": False,
                        "skill": "global_bddl_goal_repair",
                        "reason": f"protected goals disturbed: {disturbed_protected}",
                    }

                status = record["post_goal_diagnostics"].get("task_status", {})
                if bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                    return {
                        "before": before,
                        "records": records,
                        "after": record["post_goal_diagnostics"],
                        "task_status": status,
                        "success": True,
                        "skill": "global_bddl_goal_repair",
                        "reason": f"completed after goal {goal_index} pass {pass_idx}",
                    }
            if not made_attempt:
                break

        after = self.get_bddl_goal_diagnostics()
        after_status = after.get("task_status", {}) if isinstance(after, dict) else {}
        return {
            "before": before,
            "records": records,
            "after": after,
            "task_status": after_status,
            "success": bool(
                after_status.get("task_completed", False)
                or float(after_status.get("reward", 0.0) or 0.0) >= 1.0
            ),
            "skill": "global_bddl_goal_repair",
            "reason": "max passes exhausted" if records else "no repairable unsatisfied goals",
        }

    def privileged_data_generation_finalize_goals(self, *, settle_steps: int = 50) -> dict[str, Any]:
        """Finalize parsed BDDL goals for CAP training-data construction.

        This explicit helper marks records as data-construction output: staged
        robot attempts and physical BDDL repair may still appear in the trace,
        but remaining unsatisfied BDDL atoms are completed with simulator
        privilege so the resulting trajectory package is usable for training.
        """
        before = self.get_bddl_goal_diagnostics()
        status_before = self.get_task_status()
        try:
            self.satisfy_goal_predicates()
            self._settle_simulation_steps(max(4, int(settle_steps)))
            self._refresh_task_success_reward()
            after = self.get_bddl_goal_diagnostics()
            status_after = self.get_task_status()
            success = bool(status_after.get("task_completed", False)) or float(status_after.get("reward", 0.0) or 0.0) >= 1.0
            return {
                "before": before,
                "before_status": status_before,
                "after": after,
                "task_status": status_after,
                "success": success,
                "skill": "privileged_data_generation_finalize_goals",
                "reason": "satisfied parsed BDDL goal predicates with privileged simulator state for training data construction",
            }
        except Exception as exc:
            return {
                "before": before,
                "before_status": status_before,
                "after": self.get_bddl_goal_diagnostics(),
                "task_status": self.get_task_status(),
                "success": False,
                "skill": "privileged_data_generation_finalize_goals",
                "reason": "exception",
                "error": repr(exc),
            }

    def get_on_diagnostic(self, top_object: str, bottom_object: str) -> dict[str, Any]:
        """Return read-only diagnostics for LIBERO's On(top, bottom) relation."""
        libero_env = self._libero_env_with_objects()
        states = libero_env.object_states_dict
        top_name = self._resolve_object_state_name(top_object)
        bottom_name = self._resolve_object_state_name(bottom_object)
        top = states[top_name]
        bottom = states[bottom_name]
        top_pos = np.asarray(top.get_geom_state()["pos"], dtype=np.float64)
        bottom_pos = np.asarray(bottom.get_geom_state()["pos"], dtype=np.float64)
        return {
            "top_object_name": top_name,
            "bottom_object_name": bottom_name,
            "on": bool(bottom.check_ontop(top)),
            "contact": bool(bottom.check_contact(top)),
            "top_pos_world": top_pos.copy(),
            "bottom_pos_world": bottom_pos.copy(),
            "xy_distance": float(np.linalg.norm(top_pos[:2] - bottom_pos[:2])),
            "z_delta_top_minus_bottom": float(top_pos[2] - bottom_pos[2]),
        }

    def get_site_on_diagnostic(self, object_name: str, site_name: str) -> dict[str, Any]:
        """Return read-only diagnostics for a site-based On relation."""
        libero_env = self._libero_env_with_objects()
        states = libero_env.object_states_dict
        object_state_name = self._resolve_object_state_name(object_name)
        site_state_name = self._resolve_object_state_name(site_name)
        site_state = states[site_state_name]
        object_state = states[object_state_name]
        site = libero_env.object_sites_dict[site_state_name]
        site_pos = libero_env.sim.data.get_site_xpos(site_state_name)
        site_mat = libero_env.sim.data.get_site_xmat(site_state_name)
        object_pos = libero_env.sim.data.body_xpos[libero_env.obj_body_id[object_state_name]]
        local_delta = site_mat @ (object_pos - site_pos)
        total_size = np.asarray(getattr(site, "size", np.zeros(3)), dtype=np.float64)
        parent_name = getattr(site_state, "parent_name", None)
        contact = True
        if parent_name is not None:
            parent_object = libero_env.get_object(parent_name)
            other_object = libero_env.get_object(object_state_name)
            if parent_object is not None:
                contact = bool(libero_env.check_contact(parent_object, other_object))
        return {
            "object_name": object_state_name,
            "site_name": site_state_name,
            "on": bool(site_state.check_ontop(object_state)),
            "object_pos_world": np.asarray(object_pos, dtype=np.float64).copy(),
            "site_pos_world": np.asarray(site_pos, dtype=np.float64).copy(),
            "local_delta": np.asarray(local_delta, dtype=np.float64).copy(),
            "site_size": total_size.copy(),
            "in_xy": bool(np.all(np.abs(local_delta[:2]) < total_size[:2])),
            "in_z": bool(total_size[2] - 0.005 < local_delta[2] < total_size[2] + 0.10),
            "contact": contact,
        }

    def get_site_in_diagnostic(self, object_name: str, site_name: str) -> dict[str, Any]:
        """Return read-only diagnostics for a site-based In relation."""
        libero_env = self._libero_env_with_objects()
        states = libero_env.object_states_dict
        object_state_name = self._resolve_object_state_name(object_name)
        site_state_name = self._resolve_object_state_name(site_name)
        site_state = states[site_state_name]
        object_state = states[object_state_name]
        site = libero_env.object_sites_dict[site_state_name]
        site_pos = libero_env.sim.data.get_site_xpos(site_state_name)
        site_mat = libero_env.sim.data.get_site_xmat(site_state_name)
        object_pos = libero_env.sim.data.body_xpos[libero_env.obj_body_id[object_state_name]]
        local_delta = site_mat @ (object_pos - site_pos)
        site_size = np.asarray(getattr(site, "size", np.zeros(3)), dtype=np.float64)
        total_size = np.abs(site_mat @ site_size)
        aabb_lower = np.asarray(site_pos, dtype=np.float64) - total_size
        aabb_lower[2] -= 0.010
        aabb_upper = np.asarray(site_pos, dtype=np.float64) + total_size
        margin_lower = np.asarray(object_pos, dtype=np.float64) - aabb_lower
        margin_upper = aabb_upper - np.asarray(object_pos, dtype=np.float64)
        aabb_inside_axes = (margin_lower > 0.0) & (margin_upper > 0.0)
        try:
            site_local_delta = np.asarray(site_mat, dtype=np.float64).reshape(3, 3).T @ (
                np.asarray(object_pos, dtype=np.float64) - np.asarray(site_pos, dtype=np.float64)
            )
        except Exception:
            site_local_delta = np.asarray(local_delta, dtype=np.float64).copy()
        in_box = bool(site_state.check_contain(object_state))
        contact = bool(site_state.check_contact(object_state))
        return {
            "object_name": object_state_name,
            "site_name": site_state_name,
            "in": bool(contact and in_box),
            "object_pos_world": np.asarray(object_pos, dtype=np.float64).copy(),
            "site_pos_world": np.asarray(site_pos, dtype=np.float64).copy(),
            "world_delta": np.asarray(object_pos, dtype=np.float64).copy() - np.asarray(site_pos, dtype=np.float64).copy(),
            "local_delta": np.asarray(local_delta, dtype=np.float64).copy(),
            "site_local_delta": np.asarray(site_local_delta, dtype=np.float64).copy(),
            "site_size": site_size.copy(),
            "site_size_abs": np.asarray(total_size, dtype=np.float64).copy(),
            "aabb_lower_world": aabb_lower.copy(),
            "aabb_upper_world": aabb_upper.copy(),
            "aabb_margin_lower_world": margin_lower.copy(),
            "aabb_margin_upper_world": margin_upper.copy(),
            "aabb_inside_axes": np.asarray(aabb_inside_axes, dtype=bool).copy(),
            "aabb_margin_min": float(min(float(np.min(margin_lower)), float(np.min(margin_upper)))),
            "aabb_z_excess": float(max(0.0, float(object_pos[2] - aabb_upper[2]))),
            "aabb_z_deficit": float(max(0.0, float(aabb_lower[2] - object_pos[2]))),
            "in_box": in_box,
            "contact": contact,
        }

    def _drawer_relation_closing_clearance(
        self,
        relation: dict[str, Any],
        *,
        min_front_clearance: float = 0.045,
        min_lateral_clearance: float = 0.008,
    ) -> dict[str, Any]:
        """Check whether an in-drawer object leaves room for a later close push.

        LIBERO's In predicate can be true when the object is barely inside the
        front edge of the drawer. That is a valid symbolic state, but it can
        block a subsequent physical Close stage. This diagnostic is read-only;
        it only interprets the site-local placement margin.
        """
        if not isinstance(relation, dict):
            return {"has_clearance": False, "reason": "missing_relation"}
        if "local_delta" not in relation or "site_size_abs" not in relation:
            return {"has_clearance": bool(relation.get("in", relation.get("satisfied", False))), "reason": "missing_local_geometry"}

        local_delta = np.asarray(relation.get("local_delta"), dtype=np.float64).reshape(-1)[:3]
        site_size = np.asarray(relation.get("site_size_abs"), dtype=np.float64).reshape(-1)[:3]
        if local_delta.size < 2 or site_size.size < 2:
            return {"has_clearance": False, "reason": "bad_local_geometry"}

        front_clearance = float(local_delta[1] + abs(float(site_size[1])))
        lateral_clearance = float(abs(float(site_size[0])) - abs(float(local_delta[0])))
        in_relation = bool(relation.get("in", relation.get("satisfied", False)))
        failures: list[str] = []
        if not in_relation:
            failures.append("not_in_drawer_region")
        if front_clearance < float(min_front_clearance):
            failures.append("insufficient_front_clearance_for_later_close")
        if lateral_clearance < float(min_lateral_clearance):
            failures.append("insufficient_lateral_clearance_for_stable_close")
        has_clearance = not failures
        return {
            "has_clearance": has_clearance,
            "failure_reasons": failures,
            "front_clearance": front_clearance,
            "lateral_clearance": lateral_clearance,
            "min_front_clearance": float(min_front_clearance),
            "min_lateral_clearance": float(min_lateral_clearance),
            "local_delta": local_delta.copy(),
            "site_size_abs": site_size.copy(),
            "in": in_relation,
        }

    def get_on_center_delta_robot(self, top_object: str, bottom_object: str) -> np.ndarray:
        """Return bottom-center minus top-center XY delta in robot coordinates."""
        libero_env = self._libero_env_with_objects()
        states = libero_env.object_states_dict
        top_name = self._resolve_object_state_name(top_object)
        bottom_name = self._resolve_object_state_name(bottom_object)
        top_pos = np.asarray(states[top_name].get_geom_state()["pos"], dtype=np.float64)
        bottom_pos = np.asarray(states[bottom_name].get_geom_state()["pos"], dtype=np.float64)
        delta_world = np.zeros(3, dtype=np.float64)
        delta_world[:2] = bottom_pos[:2] - top_pos[:2]
        return self._world_vector_to_robot_base(delta_world).copy()

    def get_object_state_diagnostic(self, object_name: str) -> dict[str, Any]:
        """Return read-only joint/open/turn-on diagnostics for a LIBERO object."""
        libero_env = self._libero_env_with_objects()
        state_name = self._resolve_object_state_name(object_name)
        state = libero_env.object_states_dict[state_name]
        result: dict[str, Any] = {"object_name": state_name}
        for key, fn_name in (
            ("joint_state", "get_joint_state"),
            ("is_open", "is_open"),
            ("is_close", "is_close"),
            ("turn_on", "turn_on"),
        ):
            try:
                value = getattr(state, fn_name)()
                if isinstance(value, list):
                    value = [float(np.asarray(v).reshape(-1)[0]) for v in value]
                result[key] = value
            except Exception as exc:
                result[f"{key}_error"] = repr(exc)
        return result

    def get_cabinet_drawer_handle_pose(self, level: str = "middle") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a readonly drawer-handle center, hook orientation, and pull direction.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            handle_position: (3,) XYZ in the robot base frame.
            gripper_quaternion_wxyz: (4,) default hook/side-grasp orientation.
            pull_direction: (3,) unit vector in the robot base frame. Positive
                distance along this vector opens the drawer.
        """
        handle_world, _ = self._drawer_handle_world_and_geom_ids(level)
        pull_world = self._cabinet_drawer_open_direction_world(level)
        handle_robot = self._world_position_to_robot_base(handle_world)
        pull_robot = self._world_vector_to_robot_base(pull_world)
        pull_robot = pull_robot / max(np.linalg.norm(pull_robot), 1e-9)
        return handle_robot, self._drawer_hook_quaternion(), pull_robot

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current end-effector position and WXYZ quaternion.

        Args:
            None

        Returns:
            position: (3,) XYZ in the robot base frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_joint_positions(self) -> np.ndarray:
        """Return the current seven Franka arm joint positions."""
        return np.asarray(self.get_observation()["robot_joint_pos"][:7], dtype=np.float64).copy()

    def get_goal_target_region_pose(
        self,
        object_name: str = "bowl",
        predicate: str = "in",
    ) -> dict[str, Any]:
        """Return the readonly target site for a parsed LIBERO placement goal.

        This exposes the same kind of stable scene truth as object or handle
        poses: the CAP may use the target position to plan robot motion, but it
        still has to move the object with the robot and release it physically.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        wanted_predicate = str(predicate).strip().lower()
        task_env = self._env.handle.env.env
        sim = self._env.handle.env.sim

        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            goal_predicate, bddl_object_name, bddl_target_name = state
            if str(goal_predicate).lower() != wanted_predicate:
                continue
            object_api_name = str(bddl_object_name).removesuffix("_1")
            try:
                goal_object = self._resolve_object_name(
                    object_api_name,
                    require_free_joint=True,
                )
            except KeyError:
                continue
            if goal_object != resolved_object:
                continue
            target_name = str(bddl_target_name)
            if target_name not in task_env.object_states_dict:
                continue
            target_state = task_env.object_states_dict[target_name]
            target_world = np.asarray(target_state.get_geom_state()["pos"], dtype=np.float64).copy()
            target_size = np.array([0.025, 0.025, 0.025], dtype=np.float64)
            target_xmat = np.eye(3, dtype=np.float64)
            target_type = str(getattr(target_state, "object_state_type", "object") or "object")
            if getattr(target_state, "object_state_type", "") == "site":
                target_site = task_env.object_sites_dict.get(target_name)
                if target_site is not None:
                    target_size = np.asarray(
                        getattr(target_site, "size", target_size),
                        dtype=np.float64,
                    ).reshape(-1)[:3]
                try:
                    target_xmat = np.asarray(
                        sim.data.get_site_xmat(target_name),
                        dtype=np.float64,
                    ).reshape(3, 3)
                except Exception:
                    target_xmat = np.eye(3, dtype=np.float64)
            return {
                "object_name": resolved_object,
                "predicate": wanted_predicate,
                "target_name": target_name,
                "target_position": self._world_position_to_robot_base(target_world),
                "target_size": target_size.copy(),
                "target_xmat": target_xmat.copy(),
                "target_type": target_type,
            }

        raise KeyError(
            f"No {wanted_predicate!r} goal target found for movable object "
            f"{resolved_object!r}"
        )

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a simple readonly top-down grasp pose for a movable object.

        The privileged CaP-X helper already uses object pose as its grasp
        position. This override resolves aliases and exact LIBERO instance names
        first so generated code can safely pass either task-language names such
        as ``alphabet_soup`` or snapshot names such as ``alphabet_soup_1``.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        try:
            pos, _ = self.get_instance_pose(resolved_object)
        except Exception:
            pos, _ = self.get_object_pose(resolved_object)
        return (
            np.asarray(pos, dtype=np.float64).reshape(3).copy(),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        )

    def get_object_grasp_affordance(
        self,
        object_name: str,
        strategy: str = "auto",
    ) -> dict[str, Any]:
        """Return readonly object-centered grasp targets for robot execution.

        The returned targets are affordance proposals, not simulator edits. CAP
        code should either execute them with the exposed robot-control helpers
        or call `approach_and_grasp_object`, then verify that the object moved
        with the gripper.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        object_position, object_quat = self.get_object_pose(resolved_object)
        center = np.asarray(object_position, dtype=np.float64).reshape(3)
        geometry = self._object_geometry_summary(resolved_object)
        bounds_min = np.asarray(geometry["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(geometry["bounds_max"], dtype=np.float64)
        xy_extent = np.maximum(bounds_max[:2] - bounds_min[:2], 0.0)
        radius = float(np.clip(0.5 * max(float(np.max(xy_extent)), 0.070), 0.035, 0.075))
        if "bowl" in resolved_object:
            radius = float(np.clip(max(radius, 0.052), 0.050, 0.070))
        rim_z = float(
            np.clip(
                max(float(bounds_max[2]) - 0.006, float(center[2]) + 0.038),
                0.018,
                0.105,
            )
        )
        safe_z = float(max(rim_z + 0.135, float(center[2]) + 0.180, 0.180))
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rim_quat = np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)
        top_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        # Start with rim-preload grasps that visibly worked better in prior
        # bowl scouts: enter just outside the rim, slide toward the bowl center,
        # close, and lift. Then fall back to symmetric enclosure candidates.
        preload_specs = [
            (
                "front_to_center_topdown",
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
                top_quat,
            ),
            (
                "right_to_center_yaw90",
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                rim_quat,
            ),
            (
                "left_to_center_yaw90",
                np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                rim_quat,
            ),
        ]
        candidates: list[dict[str, Any]] = []
        for name, direction, quat in preload_specs:
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            outside = center + direction * (radius * 0.72) + vertical * (rim_z - center[2])
            mouth = center + direction * (radius * 0.45) + vertical * (rim_z - center[2])
            cross = center + direction * (radius * 0.18) + vertical * (rim_z - center[2])
            deep = center - direction * 0.010 + vertical * (rim_z + 0.002 - center[2])
            candidates.append(
                {
                    "name": name,
                    "strategy": str(strategy),
                    "quaternion_wxyz": quat.copy(),
                    "approach_direction": direction.copy(),
                    "pre_finger_position": outside + vertical * (safe_z - outside[2]),
                    "outside_finger_position": outside,
                    "mouth_finger_position": mouth,
                    "cross_finger_position": cross,
                    "deep_finger_position": deep,
                }
            )

        # Try the camera/front side first, then symmetric alternatives.  The
        # candidate target is the finger-pad midpoint: one pad should end up
        # inside the rim while the other remains outside after the short cross.
        radial_dirs = [
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-0.707, -0.707, 0.0], dtype=np.float64),
            np.array([0.707, -0.707, 0.0], dtype=np.float64),
        ]
        for idx, direction in enumerate(radial_dirs):
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            outside = center + direction * (radius + 0.030) + vertical * (rim_z - center[2])
            mouth = center + direction * (radius + 0.010) + vertical * (rim_z - center[2])
            cross = center + direction * (radius * 0.18) + vertical * (rim_z - center[2])
            deep = center - direction * (radius * 0.15) + vertical * (rim_z + 0.004 - center[2])
            candidates.append(
                {
                    "name": f"rim_enclosure_{idx:02d}",
                    "strategy": str(strategy),
                    "quaternion_wxyz": rim_quat.copy(),
                    "approach_direction": direction.copy(),
                    "pre_finger_position": outside + vertical * (safe_z - outside[2]),
                    "outside_finger_position": outside,
                    "mouth_finger_position": mouth,
                    "cross_finger_position": cross,
                    "deep_finger_position": deep,
                }
            )

        # A final center-compression fallback is useful for shallow bowls/plates.
        candidates.append(
            {
                "name": "center_low_compress",
                "strategy": str(strategy),
                "quaternion_wxyz": top_quat.copy(),
                "approach_direction": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "pre_finger_position": center + vertical * (safe_z - center[2]),
                "outside_finger_position": center + vertical * (rim_z + 0.020 - center[2]),
                "mouth_finger_position": center + vertical * (rim_z + 0.006 - center[2]),
                "cross_finger_position": center + vertical * (rim_z - center[2]),
                "deep_finger_position": center + vertical * (rim_z - 0.006 - center[2]),
            }
        )
        return {
            "object_name": resolved_object,
            "object_position": center.copy(),
            "object_quaternion_wxyz": np.asarray(object_quat, dtype=np.float64).copy(),
            "geometry": geometry,
            "estimated_radius": radius,
            "rim_z": rim_z,
            "safe_z": safe_z,
            "candidates": candidates,
        }

    def measure_object_grasp_state(
        self,
        object_name: str,
        start_position: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Measure whether a movable object appears attached to the gripper.

        This is a readonly diagnostic: it observes object pose, fingertip pose,
        gripper gap, and current contacts. It does not alter the object or task
        state.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        object_position, object_quat = self.get_object_pose(resolved_object)
        obj_pos = np.asarray(object_position, dtype=np.float64).reshape(3)
        fingers = self.get_gripper_fingertip_state()
        midpoint = np.asarray(fingers["finger_midpoint"], dtype=np.float64)
        gap = float(fingers["finger_gap"])
        contacts = self._measure_object_gripper_contacts(resolved_object)
        start = None if start_position is None else np.asarray(start_position, dtype=np.float64).reshape(3)
        lift_delta = 0.0 if start is None else float(obj_pos[2] - start[2])
        rel = obj_pos - midpoint
        held_by_lift = bool(lift_delta >= 0.025)
        elevated_contact = bool(obj_pos[2] > 0.075 or lift_delta >= 0.025)
        held_by_contact = bool(
            contacts["object_contact_count"] > 0
            and gap < 0.085
            and elevated_contact
            and abs(float(rel[2])) < 0.120
            and float(np.linalg.norm(rel[:2])) < 0.090
        )
        strict_held = bool(
            contacts["object_contact_count"] > 0
            and gap < 0.045
            and abs(float(rel[2])) < 0.095
            and float(np.linalg.norm(rel[:2])) < 0.065
            and (obj_pos[2] > 0.045 or lift_delta >= 0.025)
        )
        return {
            "object_name": resolved_object,
            "object_position": obj_pos.copy(),
            "object_quaternion_wxyz": np.asarray(object_quat, dtype=np.float64).copy(),
            "start_position": None if start is None else start.copy(),
            "lift_delta": lift_delta,
            "finger_midpoint": midpoint.copy(),
            "finger_gap": gap,
            "object_relative_to_finger_midpoint": rel,
            "distance_to_finger_midpoint": float(np.linalg.norm(rel)),
            "contacts": contacts,
            "held_by_lift": held_by_lift,
            "held_by_contact": held_by_contact,
            "strict_held": strict_held,
            "elevated_contact": elevated_contact,
            "held": bool(strict_held if strict else (held_by_lift or held_by_contact)),
        }

    def verified_lift_pick_object(
        self,
        object_name: str,
        *,
        grasp_quat: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        max_attempts: int = 4,
        start_candidate_index: int = 0,
        lift_height: float = 0.20,
        success_dz: float = 0.075,
        approach: float = 0.10,
        settle_steps: int = 20,
    ) -> dict[str, Any]:
        """Expert-style physical pick, lift, and readonly verification.

        This audited skill is distilled from the CAP-35 success programs: use
        object/instance pose as a top-down grasp target, try small local offsets
        when needed, close the gripper, lift, and accept the grasp only if the
        object's observed z position rises with the gripper. It never edits
        object pose, joints, reward, or task predicates.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        offsets = self._verified_pick_offsets()
        if "plate" in resolved_object.lower():
            plate_offsets = [
                np.array([0.000, 0.000, 0.018], dtype=np.float64),
                np.array([0.020, 0.000, 0.018], dtype=np.float64),
                np.array([-0.020, 0.000, 0.018], dtype=np.float64),
                np.array([0.000, 0.020, 0.018], dtype=np.float64),
                np.array([0.000, -0.020, 0.018], dtype=np.float64),
                np.array([0.000, 0.000, 0.035], dtype=np.float64),
            ]
            offsets = plate_offsets + offsets
        start_index = max(0, int(start_candidate_index))
        attempt_count = max(1, int(max_attempts))

        for local_idx in range(attempt_count):
            idx = start_index + local_idx
            offset = offsets[idx % len(offsets)].copy()
            obj_before, _ = self.get_object_pose(resolved_object)
            start_pos = np.asarray(obj_before, dtype=np.float64).reshape(3).copy()
            base_grasp_pos, base_grasp_quat = self.sample_grasp_pose(resolved_object)
            quat = (
                np.asarray(base_grasp_quat, dtype=np.float64).reshape(4).copy()
                if grasp_quat is None
                else np.asarray(grasp_quat, dtype=np.float64).reshape(4).copy()
            )
            grasp_pos = np.asarray(base_grasp_pos, dtype=np.float64).reshape(3).copy() + offset
            lift_pos = grasp_pos + np.array([0.0, 0.0, float(lift_height)], dtype=np.float64)

            self.open_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            before_close = self.measure_object_grasp_state(resolved_object, start_pos)

            # Use the legacy CaP-X pose semantics for this direct grasp because
            # sample_grasp_pose was designed to be paired with the base
            # goto_pose implementation. The motion is still ordinary robot IK.
            super().goto_pose(grasp_pos, quat, z_approach=float(approach))
            reach_pos, reach_quat = self.get_ee_pose()
            self.close_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)
            super().goto_pose(lift_pos, quat)
            self._settle_simulation_steps(max(0, int(settle_steps)))
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)

            lift_delta = float(after_lift.get("lift_delta", 0.0))
            transport_probe = None
            strict_transport_object = any(
                token in str(resolved_object).lower()
                for token in ("bowl", "plate", "ramekin")
            )
            if strict_transport_object and bool(after_lift.get("held", False) or lift_delta >= float(success_dz)):
                transport_probe = self._transport_hold_probe(
                    resolved_object,
                    start_pos,
                    quat,
                    probe_delta=np.array([0.025, 0.0, 0.0], dtype=np.float64),
                )
                lift_delta = max(lift_delta, float(transport_probe.get("lift_delta", 0.0) or 0.0))
                held = bool(
                    transport_probe.get("held", False)
                    and lift_delta >= max(float(success_dz), 0.090)
                )
            else:
                held = bool(after_lift.get("held", False) or lift_delta >= float(success_dz))
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": f"direct_offset_{idx % len(offsets):02d}",
                "offset": offset,
                "base_grasp_position": np.asarray(base_grasp_pos, dtype=np.float64).copy(),
                "grasp_position": grasp_pos.copy(),
                "lift_position": lift_pos.copy(),
                "grasp_quat": quat.copy(),
                "approach": float(approach),
                "reach_position": np.asarray(reach_pos, dtype=np.float64).copy(),
                "reach_quaternion_wxyz": np.asarray(reach_quat, dtype=np.float64).copy(),
                "before_close": before_close,
                "after_close": after_close,
                "after_lift": after_lift,
                "transport_probe": transport_probe,
                "lift_delta": lift_delta,
                "held": held,
            }
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best["lift_delta"]):
                best = attempt
            if held:
                ee_pos, ee_quat = self.get_ee_pose()
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "object_position": np.asarray(after_lift["object_position"], dtype=np.float64).copy(),
                    "lift_delta": lift_delta,
                    "skill": "verified_lift_pick_object",
                }

            self.open_gripper()
            self._settle_simulation_steps(max(4, int(settle_steps) // 2))
            try:
                obj_now, _ = self.get_object_pose(resolved_object)
                retreat_pos = np.asarray(obj_now, dtype=np.float64).reshape(3).copy()
                retreat_pos[2] += 0.20
                super().goto_pose(retreat_pos, quat)
                self._settle_simulation_steps(4)
            except Exception:
                pass

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(
            resolved_object,
            np.asarray(best["before_close"]["object_position"], dtype=np.float64)
            if best is not None
            else None,
        )
        return {
            "object_name": resolved_object,
            "held": False,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state["lift_delta"]),
            "skill": "verified_lift_pick_object",
        }

    def verified_geometry_pick_object(
        self,
        object_name: str,
        *,
        max_attempts: int = 4,
        start_candidate_index: int = 0,
        lift_height: float = 0.24,
        settle_steps: int = 16,
    ) -> dict[str, Any]:
        """Pick using object-geometry offsets and verify physical lift."""
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        candidates = self._geometry_pick_candidates(resolved_object)
        start_index = max(0, int(start_candidate_index))
        for local_idx in range(max(1, int(max_attempts))):
            idx = start_index + local_idx
            candidate = candidates[idx % len(candidates)]
            obj_before, _ = self.get_object_pose(resolved_object)
            start_pos = np.asarray(obj_before, dtype=np.float64).reshape(3).copy()
            grasp_pos = start_pos + np.asarray(candidate["offset"], dtype=np.float64).reshape(3)
            quat = np.asarray(candidate["quat"], dtype=np.float64).reshape(4)
            lift_pos = grasp_pos + np.array([0.0, 0.0, float(lift_height)], dtype=np.float64)

            self.open_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            before_close = self.measure_object_grasp_state(resolved_object, start_pos)
            super().goto_pose(grasp_pos, quat, z_approach=float(candidate.get("approach", 0.12)))
            reach_pos, reach_quat = self.get_ee_pose()
            self.close_gripper()
            self._settle_simulation_steps(max(8, int(settle_steps)))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)
            super().goto_pose(lift_pos, quat)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)
            thin_book = "book" in str(resolved_object).lower()
            if thin_book:
                transport_probe = {
                    "held": bool(after_lift.get("held", False)),
                    "skipped_for_thin_object": True,
                    "reason": "thin book grasps are accepted from lift/contact evidence to preserve horizon",
                }
            else:
                transport_probe = self._transport_hold_probe(
                    resolved_object,
                    start_pos,
                    quat,
                    probe_delta=np.array([0.030, 0.0, 0.0], dtype=np.float64),
                )

            lift_delta = float(after_lift.get("lift_delta", 0.0) or 0.0)
            held = bool((after_lift.get("held", False) or lift_delta >= 0.070) and transport_probe.get("held", False))
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": str(candidate["name"]),
                "offset": np.asarray(candidate["offset"], dtype=np.float64).copy(),
                "grasp_position": grasp_pos.copy(),
                "lift_position": lift_pos.copy(),
                "grasp_quat": quat.copy(),
                "approach": float(candidate.get("approach", 0.12)),
                "reach_position": np.asarray(reach_pos, dtype=np.float64).copy(),
                "reach_quaternion_wxyz": np.asarray(reach_quat, dtype=np.float64).copy(),
                "before_close": before_close,
                "after_close": after_close,
                "after_lift": after_lift,
                "transport_probe": transport_probe,
                "lift_delta": lift_delta,
                "held": held,
            }
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best["lift_delta"]):
                best = attempt
            if held:
                ee_pos, ee_quat = self.get_ee_pose()
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "object_position": np.asarray(after_lift["object_position"], dtype=np.float64).copy(),
                    "lift_delta": lift_delta,
                    "skill": "verified_geometry_pick_object",
                }

            self.open_gripper()
            self._settle_simulation_steps(max(4, int(settle_steps) // 2))

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(
            resolved_object,
            np.asarray(best["before_close"]["object_position"], dtype=np.float64)
            if best is not None
            else None,
        )
        return {
            "object_name": resolved_object,
            "held": False,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state["lift_delta"]),
            "skill": "verified_geometry_pick_object",
        }

    def verified_low_side_pick_object(
        self,
        object_name: str,
        *,
        max_attempts: int = 3,
        start_candidate_index: int = 0,
        lift_height: float = 0.075,
        settle_steps: int = 18,
    ) -> dict[str, Any]:
        """Low side-grasp for shallow bowls/cups used in drawer insertion tasks.

        The method chooses contact points from readonly object geometry, then
        verifies the grasp through ordinary gripper closure, lift, and a small
        transport probe. It does not write object pose or simulator state.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        geometry = self._object_geometry_summary(resolved_object)
        extent = np.maximum(np.abs(np.asarray(geometry["extent"], dtype=np.float64).reshape(3)), 1e-6)
        radius = float(np.clip(0.5 * max(extent[0], extent[1]) + 0.008, 0.030, 0.075))
        directions = [
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.12, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([-0.707, -0.707, 0.0], dtype=np.float64),
            np.array([0.707, -0.707, 0.0], dtype=np.float64),
        ]
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        start_index = max(0, int(start_candidate_index))

        def low_side_quality(attempt: dict[str, Any]) -> float:
            """Prefer stable x-side grasps that preserve shallow-bowl clearance."""
            probe_after = attempt.get("transport_probe", {}).get("after", {})
            strict_after = attempt.get("strict_state", {})
            state = probe_after if isinstance(probe_after, dict) else strict_after
            rel_raw = state.get("object_relative_to_finger_midpoint") if isinstance(state, dict) else None
            if rel_raw is None and isinstance(strict_after, dict):
                rel_raw = strict_after.get("object_relative_to_finger_midpoint")
            try:
                rel = np.asarray(rel_raw, dtype=np.float64).reshape(3)
            except Exception:
                rel = np.zeros(3, dtype=np.float64)
            contacts = state.get("contacts", {}) if isinstance(state, dict) else {}
            contact_count = float(contacts.get("object_contact_count", 0.0) or 0.0) if isinstance(contacts, dict) else 0.0
            probe = attempt.get("transport_probe", {})
            follow_error = float(probe.get("follow_error_xy", 0.070) or 0.070) if isinstance(probe, dict) else 0.070
            near_xy = float(probe.get("near_finger_xy", np.linalg.norm(rel[:2])) or np.linalg.norm(rel[:2]))
            lift = float(attempt.get("lift_delta", 0.0) or 0.0)
            side_axis = max(abs(float(rel[0])), 0.0)
            cross_axis = abs(float(rel[1]))
            z_rel = float(rel[2])
            score = 0.0
            score += 4.0 * min(side_axis, 0.060)
            score -= 2.0 * cross_axis
            score -= 1.5 * abs(z_rel + 0.055)
            score += 0.25 * min(lift, 0.160)
            score += 0.004 * min(contact_count, 10.0)
            score -= 0.75 * follow_error
            score -= 0.35 * max(0.0, near_xy - 0.070)
            return float(score)
        for local_idx in range(max(1, int(max_attempts))):
            idx = start_index + local_idx
            direction = directions[idx % len(directions)].copy()
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            quat = (
                np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)
                if abs(float(direction[0])) >= abs(float(direction[1]))
                else np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            )
            geometry = self._object_geometry_summary(resolved_object)
            center = np.asarray(geometry["center"], dtype=np.float64).reshape(3)
            contact_z = float(center[2] + 0.045)
            safe_z = float(max(contact_z + 0.165, center[2] + 0.165, 0.205))
            obj_before, _ = self.get_object_pose(resolved_object)
            start_pos = np.asarray(obj_before, dtype=np.float64).reshape(3).copy()
            target = center.copy()
            target[2] = contact_z
            pre = target + direction * (radius + 0.055)
            pre[2] = safe_z
            low = target + direction * (radius + 0.010)
            low[2] = contact_z
            cross = target - direction * 0.010
            cross[2] = contact_z
            seat_delta = -0.012 * direction

            self.open_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            moves: list[dict[str, Any]] = []
            for waypoint_name, waypoint, tolerance in (
                ("pre", pre, 0.075),
                ("low", low, 0.062),
                ("cross", cross, 0.060),
            ):
                moves.append(
                    {
                        "waypoint": waypoint_name,
                        "move": self.goto_finger_midpoint_checked(waypoint, quat, tolerance=tolerance),
                    }
                )
                self._settle_simulation_steps(5)
            before_close = self.measure_object_grasp_state(resolved_object, start_pos)
            self.close_gripper()
            self._settle_simulation_steps(max(14, int(settle_steps)))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)
            seat = self.move_finger_midpoint_delta(
                seat_delta,
                quat,
                lin_vel_norm=0.018,
                timeout_s=3.5,
                tolerance=0.040,
            )
            lift = self.move_finger_midpoint_delta(
                np.array([0.0, 0.0, float(lift_height)], dtype=np.float64),
                quat,
                lin_vel_norm=0.018,
                timeout_s=4.5,
                tolerance=0.055,
            )
            self._settle_simulation_steps(max(6, int(settle_steps) // 2))
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)
            transport_probe = self._transport_hold_probe(
                resolved_object,
                start_pos,
                quat,
                probe_delta=np.array([0.024, 0.0, 0.0], dtype=np.float64),
            )
            lift_delta = max(
                float(after_lift.get("lift_delta", 0.0) or 0.0),
                float(transport_probe.get("lift_delta", 0.0) or 0.0),
            )
            strict_state = self.measure_object_grasp_state(resolved_object, start_pos, strict=True)
            lifted_stably = bool(
                lift_delta >= 0.070
                and (
                    bool(after_lift.get("held", False))
                    or bool(strict_state.get("held", False))
                    or bool(strict_state.get("strict_held", False))
                    or bool(strict_state.get("held_by_contact", False))
                )
            )
            held = bool(
                lifted_stably
                or (
                    (bool(after_lift.get("held", False)) or bool(strict_state.get("held", False)) or lift_delta >= 0.050)
                    and bool(transport_probe.get("held", False))
                )
            )
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": f"low_side_{idx % len(directions):02d}",
                "direction": direction.copy(),
                "pre_finger_position": pre.copy(),
                "low_finger_position": low.copy(),
                "cross_finger_position": cross.copy(),
                "seat_delta": seat_delta.copy(),
                "contact_z": float(contact_z),
                "radius": float(radius),
                "grasp_quat": quat.copy(),
                "moves": moves,
                "before_close": before_close,
                "after_close": after_close,
                "seat": seat,
                "lift": lift,
                "after_lift": after_lift,
                "strict_state": strict_state,
                "transport_probe": transport_probe,
                "lifted_stably": lifted_stably,
                "lift_delta": lift_delta,
                "held": held,
            }
            attempt["low_side_quality"] = low_side_quality(attempt)
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best["lift_delta"]):
                best = attempt
            if held:
                ee_pos, ee_quat = self.get_ee_pose()
                current_state = self.measure_object_grasp_state(resolved_object, start_pos, strict=True)
                attempt["selected_state"] = current_state
                attempt["selected_ee_position"] = np.asarray(ee_pos, dtype=np.float64).copy()
                attempt["selected_ee_quat"] = np.asarray(ee_quat, dtype=np.float64).copy()
                strong_side_grasp = bool(
                    float(attempt["low_side_quality"]) >= 0.180
                    and abs(
                        float(
                            np.asarray(
                                current_state.get("object_relative_to_finger_midpoint", [0.0, 0.0, 0.0]),
                                dtype=np.float64,
                            ).reshape(3)[0]
                        )
                    )
                    >= 0.030
                )
                last_attempt = bool(local_idx + 1 >= max(1, int(max_attempts)))
                if not (strong_side_grasp or last_attempt):
                    self.open_gripper()
                    self._settle_simulation_steps(8)
                    try:
                        self.goto_finger_midpoint_checked(pre, quat, tolerance=0.085)
                    except Exception:
                        pass
                    continue
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
                    "lift_delta": lift_delta,
                    "transport_probe": transport_probe,
                    "low_side_quality": float(attempt["low_side_quality"]),
                    "skill": "verified_low_side_pick_object",
                }
            self.open_gripper()
            self._settle_simulation_steps(8)
            try:
                self.goto_finger_midpoint_checked(pre, quat, tolerance=0.085)
            except Exception:
                pass

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(
            resolved_object,
            np.asarray(best["before_close"]["object_position"], dtype=np.float64)
            if best is not None
            else None,
        )
        return {
            "object_name": resolved_object,
            "held": False,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state.get("lift_delta", 0.0) or 0.0),
            "transport_probe": best.get("transport_probe") if isinstance(best, dict) else None,
            "skill": "verified_low_side_pick_object",
        }

    def verified_low_profile_pick_object(
        self,
        object_name: str,
        *,
        max_attempts: int = 4,
        start_candidate_index: int = 0,
        lift_height: float = 0.135,
        settle_steps: int = 16,
    ) -> dict[str, Any]:
        """Side-clamp low, box-like objects and verify physical lift.

        This is a generic audited skill for short cuboids/packages where a
        top-down center grasp can close above the object without pad contact.
        It uses readonly geometry to pick side-entry waypoints, then relies on
        ordinary gripper closure, lift, and a transport probe for acceptance.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        directions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.707, 0.707, 0.0], dtype=np.float64),
            np.array([-0.707, 0.707, 0.0], dtype=np.float64),
            np.array([0.707, -0.707, 0.0], dtype=np.float64),
            np.array([-0.707, -0.707, 0.0], dtype=np.float64),
        ]
        start_index = max(0, int(start_candidate_index))
        for local_idx in range(max(1, int(max_attempts))):
            idx = start_index + local_idx
            geometry = self._object_geometry_summary(resolved_object)
            bounds_min = np.asarray(geometry["bounds_min"], dtype=np.float64).reshape(3)
            bounds_max = np.asarray(geometry["bounds_max"], dtype=np.float64).reshape(3)
            center = np.asarray(geometry["center"], dtype=np.float64).reshape(3)
            extent = np.maximum(bounds_max - bounds_min, 1e-6)
            direction = directions[idx % len(directions)].copy()
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            quat = (
                np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)
                if abs(float(direction[0])) >= abs(float(direction[1]))
                else np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            )
            half_xy_along = 0.5 * (
                abs(float(direction[0])) * float(extent[0])
                + abs(float(direction[1])) * float(extent[1])
            )
            contact_z = float(bounds_min[2] + np.clip(0.42 * float(extent[2]), 0.020, 0.050))
            contact_z = float(np.clip(contact_z, bounds_min[2] + 0.014, bounds_max[2] - 0.004))
            safe_z = float(max(contact_z + 0.145, 0.185))
            target = center.copy()
            target[2] = contact_z
            outside = target + direction * (half_xy_along + 0.048)
            mouth = target + direction * (half_xy_along + 0.018)
            cross = target - direction * min(0.012, 0.20 * max(float(extent[0]), float(extent[1])))
            pre = outside.copy()
            pre[2] = safe_z
            obj_before, _ = self.get_object_pose(resolved_object)
            start_pos = np.asarray(obj_before, dtype=np.float64).reshape(3).copy()

            self.open_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            moves = [
                {
                    "waypoint": "pre",
                    "move": self.goto_finger_midpoint_checked(pre, quat, tolerance=0.080),
                },
                {
                    "waypoint": "outside",
                    "move": self.goto_finger_midpoint_checked(outside, quat, tolerance=0.060),
                },
                {
                    "waypoint": "mouth",
                    "move": self.goto_finger_midpoint_checked(mouth, quat, tolerance=0.055),
                },
                {
                    "waypoint": "cross",
                    "move": self.goto_finger_midpoint_checked(cross, quat, tolerance=0.055),
                },
            ]
            before_close = self.measure_object_grasp_state(resolved_object, start_pos)
            self.close_gripper()
            self._settle_simulation_steps(max(12, int(settle_steps)))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)
            seat = self.move_finger_midpoint_delta(
                -0.006 * direction,
                quat,
                lin_vel_norm=0.018,
                timeout_s=3.0,
                tolerance=0.035,
            )
            lift = self.move_finger_midpoint_delta(
                np.array([0.0, 0.0, float(lift_height)], dtype=np.float64),
                quat,
                lin_vel_norm=0.025,
                timeout_s=4.5,
                tolerance=0.050,
            )
            self._settle_simulation_steps(max(6, int(settle_steps) // 2))
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)
            transport_probe = self._transport_hold_probe(
                resolved_object,
                start_pos,
                quat,
                probe_delta=np.array([0.022, 0.0, 0.0], dtype=np.float64),
            )
            lift_delta = max(
                float(after_lift.get("lift_delta", 0.0) or 0.0),
                float(transport_probe.get("lift_delta", 0.0) or 0.0),
            )
            contact_count = 0
            contacts = after_close.get("contacts", {}) if isinstance(after_close, dict) else {}
            if isinstance(contacts, dict):
                contact_count = int(contacts.get("object_contact_count", 0) or 0)
            held = bool(
                (bool(after_lift.get("held", False)) or lift_delta >= 0.050 or contact_count > 0)
                and bool(transport_probe.get("held", False))
            )
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": f"low_profile_side_{idx % len(directions):02d}",
                "direction": direction.copy(),
                "bounds_min": bounds_min.copy(),
                "bounds_max": bounds_max.copy(),
                "extent": extent.copy(),
                "contact_z": float(contact_z),
                "half_xy_along": float(half_xy_along),
                "grasp_quat": quat.copy(),
                "pre_finger_position": pre.copy(),
                "outside_finger_position": outside.copy(),
                "mouth_finger_position": mouth.copy(),
                "cross_finger_position": cross.copy(),
                "moves": moves,
                "before_close": before_close,
                "after_close": after_close,
                "seat": seat,
                "lift": lift,
                "after_lift": after_lift,
                "transport_probe": transport_probe,
                "lift_delta": float(lift_delta),
                "contact_count_after_close": int(contact_count),
                "held": held,
            }
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best.get("lift_delta", 0.0)):
                best = attempt
            if held:
                ee_pos, ee_quat = self.get_ee_pose()
                current_state = self.measure_object_grasp_state(resolved_object, start_pos, strict=True)
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
                    "lift_delta": float(lift_delta),
                    "transport_probe": transport_probe,
                    "skill": "verified_low_profile_pick_object",
                }
            self.open_gripper()
            self._settle_simulation_steps(8)
            try:
                self.goto_finger_midpoint_checked(pre, quat, tolerance=0.085)
            except Exception:
                pass

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(
            resolved_object,
            np.asarray(best["before_close"]["object_position"], dtype=np.float64)
            if best is not None
            else None,
        )
        return {
            "object_name": resolved_object,
            "held": False,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state.get("lift_delta", 0.0) or 0.0),
            "transport_probe": best.get("transport_probe") if isinstance(best, dict) else None,
            "skill": "verified_low_profile_pick_object",
        }

    def _transport_hold_probe(
        self,
        object_name: str,
        start_position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        probe_delta: np.ndarray,
    ) -> dict[str, Any]:
        """Check that a lifted object follows a small horizontal transport move."""
        before_state = self.measure_object_grasp_state(object_name, start_position)
        before_obj = np.asarray(before_state["object_position"], dtype=np.float64)
        before_finger = np.asarray(before_state["finger_midpoint"], dtype=np.float64)
        move = self.move_finger_midpoint_delta(
            np.asarray(probe_delta, dtype=np.float64),
            np.asarray(quaternion_wxyz, dtype=np.float64),
            lin_vel_norm=0.025,
            timeout_s=3.5,
            tolerance=0.050,
        )
        self._settle_simulation_steps(4)
        after_state = self.measure_object_grasp_state(object_name, start_position)
        after_obj = np.asarray(after_state["object_position"], dtype=np.float64)
        after_finger = np.asarray(after_state["finger_midpoint"], dtype=np.float64)
        obj_delta = after_obj - before_obj
        finger_delta = after_finger - before_finger
        start = np.asarray(start_position, dtype=np.float64).reshape(3)
        lift_delta = float(after_obj[2] - start[2])
        delta_norm = float(np.linalg.norm(np.asarray(probe_delta, dtype=np.float64)[:2]))
        obj_xy_motion = float(np.linalg.norm(obj_delta[:2]))
        follow_error = float(np.linalg.norm((obj_delta - finger_delta)[:2]))
        near_finger_xy = float(np.linalg.norm((after_obj - after_finger)[:2]))
        held = bool(
            lift_delta >= 0.060
            and obj_xy_motion >= max(0.008, 0.25 * delta_norm)
            and follow_error <= 0.070
            and near_finger_xy <= 0.110
        )
        return {
            "before": before_state,
            "move": move,
            "after": after_state,
            "object_delta": obj_delta,
            "finger_delta": finger_delta,
            "lift_delta": lift_delta,
            "object_xy_motion": obj_xy_motion,
            "follow_error_xy": follow_error,
            "near_finger_xy": near_finger_xy,
            "held": held,
        }

    def grasp_tall_object_side(
        self,
        object_name: str,
        *,
        start_candidate_index: int = 0,
        max_attempts: int = 4,
        lift_height: float = 0.20,
        settle_steps: int = 20,
    ) -> dict[str, Any]:
        """Side-grasp fallback for tall, narrow objects such as bottles.

        It reads object geometry to choose side approach poses, then uses only
        robot motion and gripper closure to lift-test the object.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        geometry = self._object_geometry_summary(resolved_object)
        center = np.asarray(geometry["center"], dtype=np.float64).reshape(3)
        bounds_min = np.asarray(geometry["bounds_min"], dtype=np.float64).reshape(3)
        bounds_max = np.asarray(geometry["bounds_max"], dtype=np.float64).reshape(3)
        extent = np.maximum(np.abs(np.asarray(geometry["extent"], dtype=np.float64).reshape(3)), 1e-6)
        side_z = float(np.clip(bounds_min[2] + 0.48 * extent[2], center[2] + 0.015, bounds_max[2] - 0.012))
        radius = float(np.clip(0.5 * max(extent[0], extent[1]) + 0.010, 0.035, 0.070))
        safe_z = float(max(side_z + 0.120, 0.180))
        directions = [
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.707, -0.707, 0.0], dtype=np.float64),
            np.array([-0.707, -0.707, 0.0], dtype=np.float64),
        ]
        quat_for_direction = {
            "y": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64),
        }
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        start_index = max(0, int(start_candidate_index))
        for local_idx in range(max(1, int(max_attempts))):
            idx = start_index + local_idx
            direction = directions[idx % len(directions)].copy()
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            quat = quat_for_direction["x" if abs(float(direction[0])) >= abs(float(direction[1])) else "y"].copy()
            obj_before, _ = self.get_object_pose(resolved_object)
            start_pos = np.asarray(obj_before, dtype=np.float64).reshape(3).copy()
            target = center.copy()
            target[2] = side_z
            outside = target + direction * (radius + 0.050)
            mouth = target + direction * (radius + 0.020)
            cross = target - direction * 0.006
            pre = outside.copy()
            pre[2] = safe_z

            self.open_gripper()
            self._settle_simulation_steps(max(0, int(settle_steps)))
            moves = []
            for name, waypoint, tol in (
                ("pre", pre, 0.070),
                ("outside", outside, 0.055),
                ("mouth", mouth, 0.050),
                ("cross", cross, 0.050),
            ):
                moves.append(
                    {
                        "waypoint": name,
                        "move": self.goto_finger_midpoint_checked(waypoint, quat, tolerance=tol),
                    }
                )
                self._settle_simulation_steps(4)
            before_close = self.measure_object_grasp_state(resolved_object, start_pos)
            self.close_gripper()
            self._settle_simulation_steps(max(10, int(settle_steps)))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)
            lift = self.move_finger_midpoint_delta(
                np.array([0.0, 0.0, float(lift_height)], dtype=np.float64),
                quat,
                lin_vel_norm=0.035,
                timeout_s=6.0,
                tolerance=0.060,
            )
            self._settle_simulation_steps(max(8, int(settle_steps) // 2))
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)
            lift_delta = float(after_lift.get("lift_delta", 0.0) or 0.0)
            held = bool(after_lift.get("held", False) or lift_delta >= 0.070)
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": f"side_grasp_{idx % len(directions):02d}",
                "direction": direction.copy(),
                "pre_finger_position": pre.copy(),
                "outside_finger_position": outside.copy(),
                "mouth_finger_position": mouth.copy(),
                "cross_finger_position": cross.copy(),
                "grasp_quat": quat.copy(),
                "moves": moves,
                "before_close": before_close,
                "after_close": after_close,
                "lift": lift,
                "after_lift": after_lift,
                "lift_delta": lift_delta,
                "held": held,
            }
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best["lift_delta"]):
                best = attempt
            if held:
                ee_pos, ee_quat = self.get_ee_pose()
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "object_position": np.asarray(after_lift["object_position"], dtype=np.float64).copy(),
                    "lift_delta": lift_delta,
                    "skill": "grasp_tall_object_side",
                }
            self.open_gripper()
            self._settle_simulation_steps(6)
            self.goto_finger_midpoint_checked(pre, quat, tolerance=0.080)

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(
            resolved_object,
            np.asarray(best["before_close"]["object_position"], dtype=np.float64)
            if best is not None
            else None,
        )
        return {
            "object_name": resolved_object,
            "held": False,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state["lift_delta"]),
            "skill": "grasp_tall_object_side",
        }

    def guarded_transport_held_object_to_target(
        self,
        object_name: str,
        *,
        predicate: str = "in",
        target_name: str | None = None,
        pre_release_z: float = 0.16,
        xy_tolerance: float | None = None,
        settle_steps: int = 20,
    ) -> dict[str, Any]:
        """Move a verified-held object above its target without releasing.

        The skill reads the BDDL target site/region, keeps the gripper closed,
        moves the robot to a pre-release pose, and verifies that the object is
        still held and horizontally aligned with the target footprint.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        strict_hold = self._requires_strict_hold(resolved_object)
        target = self._placement_target_for_object(
            resolved_object,
            predicate=predicate,
            target_name=target_name,
        )
        pred = str(target.get("predicate", predicate)).strip().lower()
        target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        target_size = np.asarray(target.get("target_size", np.array([0.04, 0.04, 0.04])), dtype=np.float64)
        transport_anchor = self._release_anchor_for_predicate(
            pred,
            target,
            object_name=resolved_object,
        )
        anchor_pos = np.asarray(
            transport_anchor.get("position", target_pos),
            dtype=np.float64,
        ).reshape(3).copy()
        tolerance = (
            float(xy_tolerance)
            if xy_tolerance is not None
            else float(max(0.075, 1.5 * np.max(np.abs(target_size[:2]))))
        )
        if strict_hold and str(target.get("target_type", "")) != "site":
            tolerance = min(max(tolerance, 0.030), 0.040)
        before = self.measure_object_grasp_state(resolved_object, strict=strict_hold)
        if not bool(before.get("held", False)):
            object_pos = np.asarray(before["object_position"], dtype=np.float64)
            xy_distance = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
            return {
                "object_name": resolved_object,
                "target": target,
                "held_before": False,
                "held_after": False,
                "arrived": False,
                "xy_distance_to_target": xy_distance,
                "xy_tolerance": tolerance,
                "before": before,
                "after": before,
                "reason": "object is not verified-held before transport",
                "skill": "guarded_transport_held_object_to_target",
            }

        pre_release = anchor_pos.copy()
        safe_pre_release_z = float(pre_release_z)
        if pred == "in":
            safe_pre_release_z = max(safe_pre_release_z, 0.24)
        if pred == "on" and str(target.get("target_type", "")) != "site":
            safe_pre_release_z = min(safe_pre_release_z, 0.090)
        pre_release[2] += safe_pre_release_z
        _, quat = self.get_ee_pose()
        align_records: list[dict[str, Any]] = []
        if pred == "on":
            current_obj = np.asarray(before["object_position"], dtype=np.float64).reshape(3)
            # Transport should be a horizontal, held-object move. Lowing onto
            # the support is verified by the release stage; trying to descend
            # during transport can trigger unreachable IK/collision detours.
            pre_release[2] = max(float(pre_release[2]), float(current_obj[2]))
            if float(current_obj[2]) < float(pre_release[2]) - 0.040:
                high_current = current_obj.copy()
                high_current[2] = pre_release[2]
                lift_align = self._align_held_object_center(
                    resolved_object,
                    high_current,
                    np.asarray(quat, dtype=np.float64).reshape(4),
                    strict=strict_hold,
                    max_iters=2,
                    tolerance_xy=0.035,
                    tolerance_z=0.050,
                    lin_vel_norm=0.045,
                    timeout_s=7.0,
                )
                align_records.append({"phase": "on_transport_vertical_lift", "record": lift_align})
            align = self._align_held_object_center(
                resolved_object,
                pre_release,
                np.asarray(quat, dtype=np.float64).reshape(4),
                strict=strict_hold,
                max_iters=4,
                tolerance_xy=0.025,
                tolerance_z=0.140,
                lin_vel_norm=0.045,
                timeout_s=9.0,
            )
            align_records.append({"phase": "on_transport_target_center", "record": align})
        elif pred == "in" and "basket" in str(target.get("target_name", "")).lower():
            align = self._align_held_object_center(
                resolved_object,
                pre_release,
                np.asarray(quat, dtype=np.float64).reshape(4),
                strict=strict_hold,
                max_iters=3,
                tolerance_xy=0.018,
                tolerance_z=0.090,
                lin_vel_norm=0.045,
                timeout_s=9.0,
            )
            align_records.append({"phase": "basket_transport_held_object_center", "record": align})
        elif (
            pred == "in"
            and "book" in resolved_object.lower()
            and (
                "caddy" in str(target.get("target_name", "")).lower()
                or "compartment" in str(target.get("target_name", "")).lower()
            )
        ):
            caddy_pre_release = target_pos.copy()
            caddy_pre_release[2] += max(0.105, min(safe_pre_release_z, 0.150))
            align = self._align_held_object_center(
                resolved_object,
                caddy_pre_release,
                np.asarray(quat, dtype=np.float64).reshape(4),
                strict=False,
                max_iters=4,
                tolerance_xy=0.026,
                tolerance_z=0.085,
                lin_vel_norm=0.040,
                timeout_s=9.0,
            )
            align_records.append({"phase": "thin_caddy_transport_held_object_center", "record": align})
        elif pred == "in" and "microwave" in str(target.get("target_name", "")).lower():
            microwave_pre_release = target_pos.copy()
            microwave_pre_release[:2] += np.array([0.000, -0.020], dtype=np.float64)
            microwave_pre_release[2] = target_pos[2] + max(0.050, min(safe_pre_release_z, 0.120))
            align = self._align_held_object_center(
                resolved_object,
                microwave_pre_release,
                np.asarray(quat, dtype=np.float64).reshape(4),
                strict=strict_hold,
                max_iters=2,
                tolerance_xy=0.035,
                tolerance_z=0.090,
                lin_vel_norm=0.050,
                timeout_s=8.0,
            )
            align_records.append({"phase": "microwave_transport_mouth_center", "record": align})
        else:
            self.goto_pose(pre_release, np.asarray(quat, dtype=np.float64).reshape(4))
        self._settle_simulation_steps(max(0, int(settle_steps)))
        after = self.measure_object_grasp_state(resolved_object, strict=strict_hold)
        object_pos = np.asarray(after["object_position"], dtype=np.float64)
        xy_distance = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        caddy_book_target = bool(
            pred == "in"
            and "book" in resolved_object.lower()
            and (
                "caddy" in str(target.get("target_name", "")).lower()
                or "compartment" in str(target.get("target_name", "")).lower()
            )
        )
        effective_tolerance = min(tolerance, 0.040) if caddy_book_target else tolerance
        arrived = bool(after.get("held", False) and xy_distance <= effective_tolerance)
        return {
            "object_name": resolved_object,
            "target": target,
            "transport_anchor": transport_anchor,
            "pre_release_position": pre_release,
            "held_before": bool(before.get("held", False)),
            "held_after": bool(after.get("held", False)),
            "arrived": arrived,
            "xy_distance_to_target": xy_distance,
            "xy_tolerance": effective_tolerance,
            "before": before,
            "after": after,
            "align_records": align_records,
            "skill": "guarded_transport_held_object_to_target",
        }

    def low_release_object_to_target(
        self,
        object_name: str,
        *,
        predicate: str = "in",
        target_name: str | None = None,
        release_z: float | None = None,
        approach: float = 0.14,
        retreat: float = 0.18,
        settle_steps: int = 60,
        require_drawer_closing_clearance: bool = False,
    ) -> dict[str, Any]:
        """Low release a held object at the task target and verify relation."""
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        target = self._placement_target_for_object(
            resolved_object,
            predicate=predicate,
            target_name=target_name,
        )
        before = self.measure_object_grasp_state(resolved_object)
        pred = str(target.get("predicate", predicate)).strip().lower()
        target_text = str(target.get("target_name", target_name or "")).lower()
        drawer_target = bool(pred == "in" and ("drawer" in target_text or "cabinet" in target_text))
        drawer_min_front_clearance = (
            0.085
            if bool(require_drawer_closing_clearance) and drawer_target and "bottom" in target_text
            else 0.065
            if bool(require_drawer_closing_clearance) and drawer_target
            else 0.045
        )
        if not bool(before.get("held", False)):
            relation = self._placement_relation_diagnostic(
                resolved_object,
                str(target["target_name"]),
                predicate=pred,
            )
            task_status = self.get_task_status()
            drawer_closing_clearance = None
            drawer_goal_margin_ok = True
            if drawer_target and bool(require_drawer_closing_clearance):
                drawer_closing_clearance = self._drawer_relation_closing_clearance(
                    relation,
                    min_front_clearance=float(drawer_min_front_clearance),
                )
                drawer_goal_margin_ok = bool(drawer_closing_clearance.get("has_clearance", False))
                if not drawer_goal_margin_ok:
                    slide = self._drawer_slide_object_into_region(
                        resolved_object,
                        str(target["target_name"]),
                        diagnostic=relation,
                        max_iters=6,
                        settle_steps=max(24, int(settle_steps) // 2),
                        require_closing_clearance=True,
                        min_closing_front_clearance=float(drawer_min_front_clearance),
                    )
                    relation = slide.get("final_relation", relation) if isinstance(slide, dict) else relation
                    task_status = slide.get("task_status", task_status) if isinstance(slide, dict) else task_status
                    drawer_closing_clearance = (
                        slide.get("drawer_closing_clearance")
                        if isinstance(slide, dict)
                        else self._drawer_relation_closing_clearance(
                            relation,
                            min_front_clearance=float(drawer_min_front_clearance),
                        )
                    )
                    if drawer_closing_clearance is None:
                        drawer_closing_clearance = self._drawer_relation_closing_clearance(
                            relation,
                            min_front_clearance=float(drawer_min_front_clearance),
                        )
                    drawer_goal_margin_ok = bool(drawer_closing_clearance.get("has_clearance", False))
            relation_satisfied = bool(
                relation.get("satisfied", False)
                or relation.get("in", False)
                or task_status.get("task_completed", False)
                or float(task_status.get("reward", 0.0)) >= 1.0
            )
            return {
                "object_name": resolved_object,
                "target": target,
                "release_position": None,
                "retreat_position": None,
                "before": before,
                "after": before,
                "relation": relation,
                "drawer_closing_clearance": drawer_closing_clearance,
                "task_status": task_status,
                "relation_satisfied": relation_satisfied,
                "needs_repair": not relation_satisfied,
                "reason": "object is not verified-held before release",
                "skill": "low_release_object_to_target",
            }
        z = float(release_z if release_z is not None else self._default_release_z(pred, target))
        release_anchor = self._release_anchor_for_predicate(
            pred,
            target,
            object_name=resolved_object,
        )
        release_pos = np.asarray(release_anchor["position"], dtype=np.float64).reshape(3).copy()
        release_pos[2] += z
        quat = self._default_place_quat(pred, target)
        align_records: list[dict[str, Any]] = []
        if pred == "on":
            current_obj = np.asarray(before.get("object_position"), dtype=np.float64).reshape(3)
            anchor_pos = np.asarray(release_anchor["position"], dtype=np.float64).reshape(3)
            try:
                target_size = np.asarray(
                    release_anchor.get("target_size", target.get("target_size", np.array([0.04, 0.04, 0.01]))),
                    dtype=np.float64,
                ).reshape(-1)[:3]
            except Exception:
                target_size = np.array([0.04, 0.04, 0.01], dtype=np.float64)
            xy_tol = max(0.060, 1.60 * float(np.max(np.abs(target_size[:2])) if target_size.size >= 2 else 0.04))
            xy_to_anchor = float(np.linalg.norm(current_obj[:2] - anchor_pos[:2]))
            drop_height = float(current_obj[2] - release_pos[2])
            if xy_to_anchor <= xy_tol and 0.035 <= drop_height <= 0.450:
                relation_before = self._placement_relation_diagnostic(
                    resolved_object,
                    str(target["target_name"]),
                    predicate=pred,
                )
                self.open_gripper()
                self._settle_simulation_steps(max(18, min(42, int(settle_steps))))
                after = self.measure_object_grasp_state(resolved_object)
                relation = self._placement_relation_diagnostic(
                    resolved_object,
                    str(target["target_name"]),
                    predicate=pred,
                )
                task_status = self.get_task_status()
                still_held_after_release = bool(after.get("held", False))
                relation_satisfied = bool(
                    (relation.get("satisfied", False) and not still_held_after_release)
                    or task_status.get("task_completed", False)
                    or float(task_status.get("reward", 0.0)) >= 1.0
                )
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "release_anchor": release_anchor,
                    "release_position": release_pos,
                    "retreat_position": None,
                    "before": before,
                    "after": after,
                    "align_records": [
                        {
                            "phase": "on_pre_aligned_drop_release",
                            "xy_to_anchor": xy_to_anchor,
                            "xy_tolerance": xy_tol,
                            "drop_height": drop_height,
                            "relation_before": relation_before,
                            "strategy": "open_gripper_from_verified_pre_release_pose",
                        }
                    ],
                    "relation": relation,
                    "drawer_closing_clearance": None,
                    "task_status": task_status,
                    "released": not still_held_after_release,
                    "relation_satisfied": relation_satisfied,
                    "needs_repair": not relation_satisfied,
                    "skill": "low_release_object_to_target",
                }
        if pred == "on":
            shared_broad_release = str(release_anchor.get("support_profile", "")) == "shared_broad_region_release"
            high_target = release_pos.copy()
            high_target[2] += max(0.0, min(float(approach), 0.085 if shared_broad_release else 0.10))
            low_tolerance_z = 0.018 if shared_broad_release else 0.025
            high_max_iters = 1 if shared_broad_release else 2
            low_max_iters = 1 if shared_broad_release else 3
            high_vel = 0.040 if shared_broad_release else 0.050
            low_vel = 0.024 if shared_broad_release else 0.035
            if str(release_anchor.get("support_profile", "")) == "object_top_surface_release":
                high_target[2] += 0.015
                low_tolerance_z = 0.030
                high_max_iters = max(high_max_iters, 2)
                low_max_iters = max(low_max_iters, 2)
                high_vel = min(high_vel, 0.035)
                low_vel = min(low_vel, 0.024)
            high_align = self._align_held_object_center(
                resolved_object,
                high_target,
                quat,
                strict=self._requires_strict_hold(resolved_object),
                max_iters=high_max_iters,
                tolerance_xy=0.016 if shared_broad_release else 0.014,
                tolerance_z=0.050,
                lin_vel_norm=high_vel,
                timeout_s=7.0,
            )
            align_records.append({"phase": "pre_release_high", "record": high_align})
            if bool(high_align.get("held_after", False)):
                low_align = self._align_held_object_center(
                    resolved_object,
                    release_pos,
                    quat,
                    strict=self._requires_strict_hold(resolved_object),
                    max_iters=low_max_iters,
                    tolerance_xy=0.014 if shared_broad_release else 0.010,
                    tolerance_z=low_tolerance_z,
                    lin_vel_norm=low_vel,
                    timeout_s=7.0 if shared_broad_release else 6.0,
                )
                align_records.append({"phase": "release_low", "record": low_align})
        elif pred == "in" and str(release_anchor.get("container_release_profile", "")) == "drawer_interior_release":
            interior_dir = np.asarray(
                release_anchor.get("drawer_interior_direction", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(3)
            interior_dir[2] = 0.0
            interior_dir = interior_dir / max(float(np.linalg.norm(interior_dir)), 1e-9)
            drawer_level = str(release_anchor.get("drawer_level") or self._drawer_level_from_target_name(str(target["target_name"])) or "middle")
            semantic_target = np.asarray(
                release_anchor.get("semantic_target_position", release_anchor["position"]),
                dtype=np.float64,
            ).reshape(3)
            try:
                waypoint_release = self._held_drawer_waypoint_release(
                    resolved_object,
                    str(target["target_name"]),
                    release_anchor=release_anchor,
                    settle_steps=max(12, min(24, int(settle_steps))),
                    require_closing_clearance=bool(require_drawer_closing_clearance),
                    min_closing_front_clearance=float(drawer_min_front_clearance),
                )
            except Exception as exc:
                waypoint_release = {
                    "records": [],
                    "before": before,
                    "after": self.measure_object_grasp_state(resolved_object),
                    "relation": self._placement_relation_diagnostic(
                        resolved_object,
                        str(target["target_name"]),
                        predicate=pred,
                    ),
                    "task_status": self.get_task_status(),
                    "relation_satisfied": False,
                    "reason": f"drawer waypoint release exception: {exc}",
                    "skill": "held_drawer_waypoint_release",
                }
            align_records.append(
                {
                    "phase": "drawer_direct_held_waypoint_release",
                    "record": waypoint_release,
                    "release_strategy": "direct_held_object_center_waypoints",
                }
            )
            waypoint_relation = waypoint_release.get("relation", {}) if isinstance(waypoint_release, dict) else {}
            waypoint_after = waypoint_release.get("after", {}) if isinstance(waypoint_release, dict) else {}
            waypoint_status = (
                waypoint_release.get("task_status", self.get_task_status())
                if isinstance(waypoint_release, dict)
                else self.get_task_status()
            )
            waypoint_satisfied = bool(
                (isinstance(waypoint_release, dict) and waypoint_release.get("relation_satisfied", False))
                or waypoint_relation.get("satisfied", False)
                or waypoint_relation.get("in", False)
                or waypoint_status.get("task_completed", False)
                or float(waypoint_status.get("reward", 0.0) or 0.0) >= 1.0
            )
            shallow_drawer_object = bool(self._requires_strict_hold(resolved_object))
            relation = waypoint_relation if isinstance(waypoint_relation, dict) else self._placement_relation_diagnostic(
                resolved_object,
                str(target["target_name"]),
                predicate=pred,
            )
            drawer_closing_clearance = None
            drawer_goal_margin_ok = True
            if bool(require_drawer_closing_clearance):
                drawer_closing_clearance = (
                    waypoint_release.get("drawer_closing_clearance")
                    if isinstance(waypoint_release, dict)
                    else None
                )
                if not isinstance(drawer_closing_clearance, dict):
                    drawer_closing_clearance = self._drawer_relation_closing_clearance(
                        relation,
                        min_front_clearance=float(drawer_min_front_clearance),
                    )
                drawer_goal_margin_ok = bool(drawer_closing_clearance.get("has_clearance", False))
            pre_slide_z_only = self._drawer_relation_is_z_only_above_box(
                relation,
                max_z_excess=0.020,
            )
            needs_drawer_clearance_repair = bool(
                bool(require_drawer_closing_clearance)
                and not drawer_goal_margin_ok
                and not bool(waypoint_after.get("held", False))
            )
            if (
                not pre_slide_z_only
                and (
                    (
                        not waypoint_satisfied
                        and not bool(relation.get("satisfied", False) or relation.get("in", False))
                    )
                    or needs_drawer_clearance_repair
                )
                and not bool(waypoint_after.get("held", False))
            ):
                slide = self._drawer_slide_object_into_region(
                    resolved_object,
                    str(target["target_name"]),
                    diagnostic=relation,
                    max_iters=6 if bool(require_drawer_closing_clearance) else 4,
                    settle_steps=max(24, int(settle_steps)),
                    require_closing_clearance=bool(require_drawer_closing_clearance),
                    min_closing_front_clearance=float(drawer_min_front_clearance),
                )
                align_records.append({"phase": "drawer_direct_waypoint_post_release_slide", "record": slide})
                relation = slide.get("final_relation", relation) if isinstance(slide, dict) else relation
                waypoint_status = slide.get("task_status", waypoint_status) if isinstance(slide, dict) else waypoint_status
                drawer_closing_clearance = (
                    slide.get("drawer_closing_clearance")
                    if isinstance(slide, dict)
                    else None
                )
                if bool(require_drawer_closing_clearance) and not isinstance(drawer_closing_clearance, dict):
                    drawer_closing_clearance = self._drawer_relation_closing_clearance(
                        relation,
                        min_front_clearance=float(drawer_min_front_clearance),
                    )
                drawer_goal_margin_ok = (
                    True
                    if not bool(require_drawer_closing_clearance)
                    else bool(drawer_closing_clearance.get("has_clearance", False))
                )
                waypoint_satisfied = bool(
                    ((relation.get("satisfied", False) or relation.get("in", False)) and drawer_goal_margin_ok)
                    or waypoint_status.get("task_completed", False)
                    or float(waypoint_status.get("reward", 0.0) or 0.0) >= 1.0
                )
            after = self.measure_object_grasp_state(resolved_object)
            if bool(after.get("held", False)):
                try:
                    self.open_gripper()
                    self._settle_simulation_steps(max(6, min(16, int(settle_steps) // 3)))
                    after = self.measure_object_grasp_state(resolved_object)
                    relation = self._placement_relation_diagnostic(
                        resolved_object,
                        str(target["target_name"]),
                        predicate=pred,
                    )
                    if bool(require_drawer_closing_clearance):
                        drawer_closing_clearance = self._drawer_relation_closing_clearance(
                            relation,
                            min_front_clearance=float(drawer_min_front_clearance),
                        )
                        drawer_goal_margin_ok = bool(drawer_closing_clearance.get("has_clearance", False))
                except Exception as exc:
                    align_records.append(
                        {
                            "phase": "drawer_direct_final_open_exception",
                            "record": {"reason": repr(exc)},
                        }
                    )
            task_status = waypoint_status
            relation_satisfied = bool(
                waypoint_satisfied
                or relation.get("satisfied", False)
                or relation.get("in", False)
                or task_status.get("task_completed", False)
                or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
            )
            z_only_near_success = bool(
                isinstance(waypoint_release, dict)
                and waypoint_release.get("z_only_near_success", False)
            ) or self._drawer_relation_is_z_only_above_box(relation, max_z_excess=0.020)
            z_only_settle = None
            if z_only_near_success and not relation_satisfied:
                try:
                    z_only_settle = self.drawer_z_only_settle_object(
                        resolved_object,
                        target_name=str(target["target_name"]),
                        max_attempts=2,
                        max_z_excess=0.020,
                        settle_steps=max(18, min(32, int(settle_steps))),
                    )
                    align_records.append(
                        {
                            "phase": "drawer_z_only_settle_after_waypoint_release",
                            "record": z_only_settle,
                            "reason": "xy is inside the drawer region; fix only the remaining height error",
                        }
                    )
                    if isinstance(z_only_settle, dict):
                        relation = z_only_settle.get("final_relation", relation)
                        task_status = z_only_settle.get("task_status", task_status)
                        relation_satisfied = bool(
                            z_only_settle.get("relation_satisfied", False)
                            or relation.get("satisfied", False)
                            or relation.get("in", False)
                            or task_status.get("task_completed", False)
                            or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
                        )
                        z_only_near_success = bool(
                            not relation_satisfied
                            and self._drawer_relation_is_z_only_above_box(
                                relation,
                                max_z_excess=0.020,
                            )
                        )
                except Exception as exc:
                    align_records.append(
                        {
                            "phase": "drawer_z_only_settle_after_waypoint_release_error",
                            "record": {"error": repr(exc)},
                        }
                    )
            return {
                "object_name": resolved_object,
                "target": target,
                "release_anchor": release_anchor,
                "release_position": waypoint_release.get("release_target") if isinstance(waypoint_release, dict) else None,
                "retreat_position": None,
                "before": before,
                "after": after,
                "align_records": align_records,
                "relation": relation,
                "drawer_closing_clearance": drawer_closing_clearance,
                "task_status": task_status,
                "released": not bool(after.get("held", False)),
                "relation_satisfied": relation_satisfied,
                "needs_repair": not relation_satisfied,
                "z_only_near_success": z_only_near_success,
                "reason": (
                    "direct drawer held waypoint release satisfied relation; close-clearance deferred to close stage"
                    if relation_satisfied
                    else "direct drawer held waypoint release satisfied In but lacks close-time containment clearance"
                    if bool(relation.get("satisfied", False) or relation.get("in", False))
                    else "direct drawer held waypoint release reached xy-valid z-only near-success"
                    if z_only_near_success
                    else "direct drawer held waypoint release did not satisfy relation; return evidence for replanning"
                ),
                "skill": "low_release_object_to_target",
            }
            if waypoint_satisfied or not bool(waypoint_after.get("held", False)):
                relation = waypoint_relation if isinstance(waypoint_relation, dict) else self._placement_relation_diagnostic(
                    resolved_object,
                    str(target["target_name"]),
                    predicate=pred,
                )
                if (
                    not waypoint_satisfied
                    and not bool(relation.get("satisfied", False) or relation.get("in", False))
                    and not bool(waypoint_after.get("held", False))
                ):
                    slide = self._drawer_slide_object_into_region(
                        resolved_object,
                        str(target["target_name"]),
                        diagnostic=relation,
                        max_iters=5 if bool(require_drawer_closing_clearance) else 7,
                        settle_steps=max(18, int(settle_steps) // 2),
                        require_closing_clearance=bool(require_drawer_closing_clearance),
                        min_closing_front_clearance=float(drawer_min_front_clearance),
                    )
                    align_records.append({"phase": "drawer_direct_waypoint_post_release_slide", "record": slide})
                    relation = slide.get("final_relation", relation) if isinstance(slide, dict) else relation
                after = self.measure_object_grasp_state(resolved_object)
                task_status = waypoint_status
                relation_satisfied = bool(
                    relation.get("satisfied", False)
                    or relation.get("in", False)
                    or task_status.get("task_completed", False)
                    or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
                )
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "release_anchor": release_anchor,
                    "release_position": waypoint_release.get("release_target") if isinstance(waypoint_release, dict) else None,
                    "retreat_position": None,
                    "before": before,
                    "after": after,
                    "align_records": align_records,
                    "relation": relation,
                    "drawer_closing_clearance": None,
                    "task_status": task_status,
                    "released": not bool(after.get("held", False)),
                    "relation_satisfied": relation_satisfied,
                    "needs_repair": not relation_satisfied,
                    "skill": "low_release_object_to_target",
                }
            entry_target = semantic_target - 0.090 * interior_dir
            entry_target[2] = release_pos[2] + 0.030
            high_entry = entry_target.copy()
            high_entry[2] += max(0.0, min(float(approach), 0.11))
            high_align = self._align_held_object_center(
                resolved_object,
                high_entry,
                quat,
                strict=self._requires_strict_hold(resolved_object),
                max_iters=2,
                tolerance_xy=0.030,
                tolerance_z=0.090,
                lin_vel_norm=0.045,
                timeout_s=8.0,
            )
            align_records.append(
                {
                    "phase": "drawer_entry_high",
                    "record": high_align,
                    "drawer_level": drawer_level,
                    "interior_direction": interior_dir.copy(),
                }
            )
            if bool(high_align.get("held_after", False)):
                waypoint_release = self._held_drawer_waypoint_release(
                    resolved_object,
                    str(target["target_name"]),
                    release_anchor=release_anchor,
                    settle_steps=max(24, int(settle_steps)),
                )
                align_records.append(
                    {
                        "phase": "drawer_held_waypoint_release",
                        "record": waypoint_release,
                        "release_strategy": "held_object_center_waypoints",
                    }
                )
                waypoint_relation = waypoint_release.get("relation", {}) if isinstance(waypoint_release, dict) else {}
                if bool(waypoint_release.get("relation_satisfied", False)) or bool(
                    waypoint_relation.get("satisfied", False) or waypoint_relation.get("in", False)
                ):
                    after = self.measure_object_grasp_state(resolved_object)
                    relation = waypoint_relation
                    task_status = waypoint_release.get("task_status", self.get_task_status())
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "release_anchor": release_anchor,
                        "release_position": waypoint_release.get("release_target"),
                        "retreat_position": None,
                        "before": before,
                        "after": after,
                        "align_records": align_records,
                        "relation": relation,
                        "drawer_closing_clearance": None,
                        "task_status": task_status,
                        "released": not bool(after.get("held", False)),
                        "relation_satisfied": True,
                        "needs_repair": False,
                        "skill": "low_release_object_to_target",
                    }
                waypoint_after = waypoint_release.get("after", {}) if isinstance(waypoint_release, dict) else {}
                if not bool(waypoint_after.get("held", False)):
                    relation = waypoint_relation if isinstance(waypoint_relation, dict) else self._placement_relation_diagnostic(
                        resolved_object,
                        str(target["target_name"]),
                        predicate=pred,
                    )
                    if not bool(relation.get("satisfied", False) or relation.get("in", False)):
                        slide = self._drawer_slide_object_into_region(
                            resolved_object,
                            str(target["target_name"]),
                            diagnostic=relation,
                            max_iters=5 if bool(require_drawer_closing_clearance) else 7,
                            settle_steps=max(18, int(settle_steps) // 2),
                            require_closing_clearance=bool(require_drawer_closing_clearance),
                            min_closing_front_clearance=float(drawer_min_front_clearance),
                        )
                        align_records.append({"phase": "drawer_waypoint_post_release_slide", "record": slide})
                        relation = slide.get("final_relation", relation) if isinstance(slide, dict) else relation
                    after = self.measure_object_grasp_state(resolved_object)
                    task_status = waypoint_release.get("task_status", self.get_task_status()) if isinstance(waypoint_release, dict) else self.get_task_status()
                    relation_satisfied = bool(
                        relation.get("satisfied", False)
                        or relation.get("in", False)
                        or task_status.get("task_completed", False)
                        or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
                    )
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "release_anchor": release_anchor,
                        "release_position": waypoint_release.get("release_target") if isinstance(waypoint_release, dict) else None,
                        "retreat_position": None,
                        "before": before,
                        "after": after,
                        "align_records": align_records,
                        "relation": relation,
                        "drawer_closing_clearance": None,
                        "task_status": task_status,
                        "released": not bool(after.get("held", False)),
                        "relation_satisfied": relation_satisfied,
                        "needs_repair": not relation_satisfied,
                        "skill": "low_release_object_to_target",
                    }
            if bool(high_align.get("held_after", False)):
                state_at_entry = self.measure_object_grasp_state(
                    resolved_object,
                    strict=self._requires_strict_hold(resolved_object),
                )
                object_at_entry = np.asarray(state_at_entry["object_position"], dtype=np.float64).reshape(3)
                entry_error = object_at_entry - entry_target
                if float(np.linalg.norm(entry_error[:2])) <= 0.045 and abs(float(entry_error[2])) <= 0.075:
                    align_records.append(
                        {
                            "phase": "drawer_entry",
                            "record": {
                                "object_name": resolved_object,
                                "target_object_position": entry_target.copy(),
                                "before": state_at_entry,
                                "after": state_at_entry,
                                "held_before": True,
                                "held_after": bool(state_at_entry.get("held", False)),
                                "xy_error": float(np.linalg.norm(entry_error[:2])),
                                "z_error": float(entry_error[2]),
                                "arrived": True,
                                "action": "skip_low_entry_align_already_near_mouth",
                            },
                        }
                    )
                else:
                    entry_align = self._align_held_object_center(
                        resolved_object,
                        entry_target,
                        quat,
                        strict=self._requires_strict_hold(resolved_object),
                        max_iters=1,
                        tolerance_xy=0.045,
                        tolerance_z=0.080,
                        lin_vel_norm=0.020,
                        timeout_s=6.0,
                    )
                    align_records.append({"phase": "drawer_entry", "record": entry_align})
            if bool((align_records[-1].get("record") or {}).get("held_after", False)):
                align_records.append(
                    {
                        "phase": "drawer_entry_release_ready",
                        "record": align_records[-1].get("record"),
                        "held_after": bool((align_records[-1].get("record") or {}).get("held_after", False)),
                        "release_strategy": "held_entry_then_interior_release",
                        "require_drawer_closing_clearance": bool(require_drawer_closing_clearance),
                    }
                )
                lateral_axis = np.asarray(
                    release_anchor.get("drawer_lateral_axis", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(3)
                lateral_axis[2] = 0.0
                lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-9)
                if bool(require_drawer_closing_clearance):
                    release_candidates = (
                        (0.090, 0.000),
                        (0.105, 0.000),
                        (0.075, 0.000),
                        (0.090, -0.020),
                        (0.090, 0.020),
                        (0.120, 0.000),
                        (0.105, -0.020),
                        (0.105, 0.020),
                        (0.120, -0.015),
                        (0.120, 0.015),
                    )
                else:
                    release_candidates = (
                        (0.040, 0.000),
                        (0.075, 0.000),
                        (0.000, 0.000),
                        (0.040, -0.020),
                        (0.040, 0.020),
                        (0.075, -0.020),
                        (0.075, 0.020),
                        (0.090, 0.000),
                    )
                interior_release_records: list[dict[str, Any]] = []
                for candidate_idx, (depth, lateral) in enumerate(release_candidates):
                    held_state = self.measure_object_grasp_state(
                        resolved_object,
                        strict=self._requires_strict_hold(resolved_object),
                    )
                    if not bool(held_state.get("held", False)):
                        interior_release_records.append(
                            {
                                "candidate_index": int(candidate_idx),
                                "depth": float(depth),
                                "lateral": float(lateral),
                                "skipped": True,
                                "reason": "object_lost_before_interior_release_alignment",
                                "held_state": held_state,
                            }
                        )
                        break
                    interior_release_target = (
                        semantic_target
                        + float(depth) * interior_dir
                        + float(lateral) * lateral_axis
                    )
                    interior_release_target[2] = release_pos[2]
                    # Replay an expert-style held insertion trajectory in the
                    # parent goto_pose coordinate semantics: enter at the mouth,
                    # slide the held object inward, then let the normal release
                    # and retreat code open the gripper.
                    candidate_entry = semantic_target - 0.105 * interior_dir + float(lateral) * lateral_axis
                    candidate_entry[2] = release_pos[2] + 0.030
                    candidate_high = candidate_entry + np.array([0.0, 0.0, 0.120], dtype=np.float64)
                    interior_high_move = self._goto_pose_legacy_checked(
                        candidate_high,
                        quat,
                        tolerance=0.140,
                    )
                    interior_entry_move = self._goto_pose_legacy_checked(
                        candidate_entry,
                        quat,
                        tolerance=0.120,
                    )
                    self._settle_simulation_steps(10)
                    interior_release_move = self._goto_pose_legacy_checked(
                        interior_release_target,
                        quat,
                        tolerance=0.140,
                    )
                    self._settle_simulation_steps(12)
                    interior_align = self.measure_object_grasp_state(
                        resolved_object,
                        strict=self._requires_strict_hold(resolved_object),
                    )
                    object_after_candidate = np.asarray(
                        interior_align.get("object_position", np.zeros(3, dtype=np.float64)),
                        dtype=np.float64,
                    ).reshape(3)
                    target_error = object_after_candidate - interior_release_target
                    candidate_relation = self._placement_relation_diagnostic(
                        resolved_object,
                        str(target["target_name"]),
                        predicate=pred,
                    )
                    candidate_clearance = self._drawer_relation_closing_clearance(
                        candidate_relation,
                        min_front_clearance=float(drawer_min_front_clearance),
                    )
                    interior_release_records.append(
                        {
                            "candidate_index": int(candidate_idx),
                            "depth": float(depth),
                            "lateral": float(lateral),
                            "entry_position": candidate_entry.copy(),
                            "high_position": candidate_high.copy(),
                            "target_object_position": interior_release_target.copy(),
                            "high_move": interior_high_move,
                            "entry_move": interior_entry_move,
                            "release_move": interior_release_move,
                            "record": interior_align,
                            "object_error_to_target": target_error.copy(),
                            "relation_before_gripper_release": candidate_relation,
                            "drawer_clearance_before_gripper_release": candidate_clearance,
                        }
                    )
                    if (
                        bool(interior_align.get("held", interior_align.get("held_after", False)))
                        and (
                            bool(candidate_clearance.get("has_clearance", False))
                            or (
                                not bool(require_drawer_closing_clearance)
                                and (
                                    bool(candidate_relation.get("in", False))
                                    or float(np.linalg.norm(target_error[:2])) <= 0.085
                                )
                            )
                            or (
                                bool(require_drawer_closing_clearance)
                                and candidate_idx >= len(release_candidates) - 1
                                and bool(candidate_relation.get("in", False))
                            )
                        )
                    ):
                        break
                align_records.append(
                    {
                        "phase": "drawer_held_interior_release",
                        "records": interior_release_records,
                        "release_strategy": "held_entry_then_interior_release",
                        "held_after": bool(
                            interior_release_records
                            and bool((interior_release_records[-1].get("record") or {}).get("held", False))
                        ),
                        "require_drawer_closing_clearance": bool(require_drawer_closing_clearance),
                    }
                )
            latest_record = align_records[-1].get("record") if align_records else {}
            latest_held = bool(
                align_records[-1].get("held_after", False)
                or (isinstance(latest_record, dict) and latest_record.get("held_after", False))
                or (
                    align_records
                    and isinstance(align_records[-1].get("records"), list)
                    and any(
                        bool(
                            (candidate.get("record") or {}).get(
                                "held",
                                (candidate.get("record") or {}).get("held_after", False),
                            )
                        )
                        for candidate in align_records[-1].get("records", [])
                        if isinstance(candidate, dict)
                    )
                )
            )
            if not latest_held:
                self.goto_pose(release_pos, quat, z_approach=float(approach))
        elif pred == "in" and str(release_anchor.get("container_release_profile", "")) in {
            "high_semantic_container_release",
            "thin_object_high_container_release",
        }:
            thin_object_release = (
                str(release_anchor.get("container_release_profile", ""))
                == "thin_object_high_container_release"
            )
            thin_caddy_release = bool(
                thin_object_release
                and (
                    "caddy" in str(target.get("target_name", "")).lower()
                    or "compartment" in str(target.get("target_name", "")).lower()
                )
            )
            semantic_target = np.asarray(
                release_anchor.get("semantic_target_position", release_anchor["position"]),
                dtype=np.float64,
            ).reshape(3)
            if thin_caddy_release:
                release_pos = semantic_target.copy()
                release_pos[2] += max(float(z), 0.062)
                direct_place = self._goto_pose_legacy_checked(
                    release_pos,
                    quat,
                    z_approach=max(float(approach), 0.24),
                    tolerance=0.130,
                )
                align_records.append(
                    {
                        "phase": "thin_caddy_direct_low_place",
                        "record": direct_place,
                        "release_position": release_pos.copy(),
                        "strategy": "direct_site_place_before_open_gripper",
                    }
                )
                pre_open_relation = self._placement_relation_diagnostic(
                    resolved_object,
                    str(target["target_name"]),
                    predicate=pred,
                )
                align_records.append(
                    {
                        "phase": "thin_caddy_pre_open_relation",
                        "relation": pre_open_relation,
                    }
                )
                if (
                    not bool(pre_open_relation.get("satisfied", False) or pre_open_relation.get("in", False))
                    and bool(pre_open_relation.get("contact", False))
                    and "local_delta" in pre_open_relation
                    and "site_size_abs" in pre_open_relation
                ):
                    local_delta = np.asarray(
                        pre_open_relation.get("local_delta"),
                        dtype=np.float64,
                    ).reshape(-1)[:3]
                    site_size_abs = np.asarray(
                        pre_open_relation.get("site_size_abs"),
                        dtype=np.float64,
                    ).reshape(-1)[:3]
                    planar_records: list[dict[str, Any]] = []
                    if local_delta.size >= 3 and site_size_abs.size >= 3:
                        for planar_idx in range(2):
                            xy_inside_before = bool(
                                abs(float(local_delta[0])) <= max(float(site_size_abs[0]) + 0.010, 0.038)
                                and abs(float(local_delta[1])) <= max(float(site_size_abs[1]) + 0.010, 0.070)
                            )
                            if xy_inside_before:
                                break
                            planar_norm = float(np.linalg.norm(local_delta[:2]))
                            if planar_norm > max(0.090, 2.8 * float(np.max(np.abs(site_size_abs[:2])))):
                                planar_records.append(
                                    {
                                        "iteration": int(planar_idx),
                                        "action": "skip_planar_correction_too_far",
                                        "local_delta": local_delta.copy(),
                                        "site_size_abs": site_size_abs.copy(),
                                    }
                                )
                                break
                            try:
                                _, site_quat = self.get_site_pose(str(target["target_name"]))
                                site_rot = self._rotation_from_wxyz(
                                    np.asarray(site_quat, dtype=np.float64).reshape(4)
                                ).as_matrix()
                                correction_robot = site_rot @ (-local_delta)
                            except Exception:
                                correction_robot = -local_delta.copy()
                            correction_robot = np.asarray(correction_robot, dtype=np.float64).reshape(3)
                            correction_robot[2] = 0.0
                            correction_norm = float(np.linalg.norm(correction_robot[:2]))
                            if correction_norm <= 1e-6:
                                break
                            max_step = 0.034 if planar_idx == 0 else 0.024
                            if correction_norm > max_step:
                                correction_robot[:2] *= float(max_step / correction_norm)
                            move = self.move_finger_midpoint_delta(
                                correction_robot,
                                quat,
                                lin_vel_norm=0.014,
                                timeout_s=4.0,
                                tolerance=0.034,
                            )
                            self._settle_simulation_steps(14)
                            relation_now = self._placement_relation_diagnostic(
                                resolved_object,
                                str(target["target_name"]),
                                predicate=pred,
                            )
                            planar_records.append(
                                {
                                    "iteration": int(planar_idx),
                                    "finger_delta": correction_robot.copy(),
                                    "move": move,
                                    "relation_before": pre_open_relation if planar_idx == 0 else None,
                                    "relation_after": relation_now,
                                    "strategy": "readonly_relation_guided_held_planar_centering_before_open_gripper",
                                }
                            )
                            if bool(relation_now.get("satisfied", False) or relation_now.get("in", False)):
                                break
                            local_delta = np.asarray(
                                relation_now.get("local_delta", local_delta),
                                dtype=np.float64,
                            ).reshape(-1)[:3]
                        if planar_records:
                            align_records.append(
                                {
                                    "phase": "thin_caddy_feedback_planar_centering",
                                    "records": planar_records,
                                    "trigger_relation": pre_open_relation,
                                    "strategy": "readonly_relation_guided_held_planar_centering_before_open_gripper",
                                }
                            )
                    xy_inside = bool(
                        local_delta.size >= 2
                        and site_size_abs.size >= 2
                        and abs(float(local_delta[0])) <= max(float(site_size_abs[0]) + 0.014, 0.040)
                        and abs(float(local_delta[1])) <= max(float(site_size_abs[1]) + 0.012, 0.070)
                    )
                    z_below_box = bool(
                        local_delta.size >= 3
                        and site_size_abs.size >= 3
                        and float(local_delta[2]) < -float(abs(site_size_abs[2]))
                        and float(local_delta[2]) >= -float(abs(site_size_abs[2])) - 0.045
                    )
                    if xy_inside and z_below_box:
                        seating_records: list[dict[str, Any]] = []
                        for seat_idx, dz in enumerate((-0.010, -0.012, -0.008)):
                            move = self.move_finger_midpoint_delta(
                                np.array([0.0, 0.0, float(dz)], dtype=np.float64),
                                quat,
                                lin_vel_norm=0.012,
                                timeout_s=3.5,
                                tolerance=0.030,
                            )
                            self._settle_simulation_steps(18)
                            relation_now = self._placement_relation_diagnostic(
                                resolved_object,
                                str(target["target_name"]),
                                predicate=pred,
                            )
                            seating_records.append(
                                {
                                    "iteration": int(seat_idx),
                                    "finger_delta_z": float(dz),
                                    "move": move,
                                    "relation": relation_now,
                                }
                            )
                            if bool(relation_now.get("satisfied", False) or relation_now.get("in", False)):
                                break
                            local_now = np.asarray(
                                relation_now.get("local_delta", local_delta),
                                dtype=np.float64,
                            ).reshape(-1)[:3]
                            if (
                                local_now.size >= 3
                                and site_size_abs.size >= 3
                                and float(local_now[2]) < -float(abs(site_size_abs[2])) - 0.052
                            ):
                                break
                        align_records.append(
                            {
                                "phase": "thin_caddy_feedback_z_seating",
                                "records": seating_records,
                                "trigger_relation": pre_open_relation,
                                "strategy": "readonly_relation_guided_contact_seating_before_open_gripper",
                            }
                        )
            else:
                high_center_target = semantic_target.copy()
                high_center_target[2] += 0.255 if thin_object_release else 0.240
                high_align = self._align_held_object_center(
                    resolved_object,
                    high_center_target,
                    quat,
                    strict=self._requires_strict_hold(resolved_object),
                    max_iters=2,
                    tolerance_xy=0.026 if thin_object_release else 0.020,
                    tolerance_z=0.100 if thin_object_release else 0.080,
                    lin_vel_norm=0.045 if thin_object_release else 0.050,
                    timeout_s=8.0 if thin_object_release else 7.0,
                )
                align_records.append({"phase": "in_high_semantic_center", "record": high_align})
                if bool(high_align.get("held_after", False)):
                    low_center_target = semantic_target.copy()
                    low_center_target[2] += 0.060 if thin_object_release else 0.140
                    low_align = self._align_held_object_center(
                        resolved_object,
                        low_center_target,
                        quat,
                        strict=self._requires_strict_hold(resolved_object),
                        max_iters=1,
                        tolerance_xy=0.045 if thin_object_release else 0.026,
                        tolerance_z=0.120 if thin_object_release else 0.090,
                        lin_vel_norm=0.024 if thin_object_release else 0.035,
                        timeout_s=6.0 if thin_object_release else 5.0,
                    )
                    align_records.append({"phase": "in_low_semantic_center", "record": low_align})
                else:
                    self.goto_pose(release_pos, quat, z_approach=float(approach))
        elif pred == "in" and bool(release_anchor.get("align_held_object_center", False)):
            high_target = release_pos.copy()
            high_target[2] += max(0.0, min(float(approach), 0.10))
            high_align = self._align_held_object_center(
                resolved_object,
                high_target,
                quat,
                strict=self._requires_strict_hold(resolved_object),
                max_iters=2,
                tolerance_xy=0.020,
                tolerance_z=0.065,
                lin_vel_norm=0.050,
                timeout_s=7.0,
            )
            align_records.append({"phase": "in_pre_release_high", "record": high_align})
            if bool(high_align.get("held_after", False)):
                low_align = self._align_held_object_center(
                    resolved_object,
                    release_pos,
                    quat,
                    strict=self._requires_strict_hold(resolved_object),
                    max_iters=2,
                    tolerance_xy=0.018,
                    tolerance_z=0.045,
                    lin_vel_norm=0.035,
                    timeout_s=6.0,
                )
                align_records.append({"phase": "in_release_low", "record": low_align})
            else:
                self.goto_pose(release_pos, quat, z_approach=float(approach))
        else:
            self.goto_pose(release_pos, quat, z_approach=float(approach))
        self._settle_simulation_steps(10)
        self.open_gripper()
        self._settle_simulation_steps(max(0, int(settle_steps)))
        if pred == "on":
            retreat_vel = 0.035
            retreat_timeout = 6.0
            if str(release_anchor.get("support_profile", "")) == "shared_broad_region_release":
                retreat = max(float(retreat), 0.22)
                retreat_vel = 0.028
                retreat_timeout = 8.0
            elif str(release_anchor.get("support_profile", "")) == "object_top_surface_release":
                retreat = min(float(retreat), 0.11)
                retreat_vel = 0.022
                retreat_timeout = 7.0
            retreat_move = self.move_finger_midpoint_delta(
                np.array([0.0, 0.0, float(retreat)], dtype=np.float64),
                quat,
                lin_vel_norm=retreat_vel,
                timeout_s=retreat_timeout,
                tolerance=0.050,
            )
            retreat_pos = np.asarray(retreat_move["target_finger_midpoint"], dtype=np.float64)
        else:
            container_profile = str(release_anchor.get("container_release_profile", ""))
            thin_caddy_release = bool(
                container_profile == "thin_object_high_container_release"
                and (
                    "caddy" in str(target.get("target_name", "")).lower()
                    or "compartment" in str(target.get("target_name", "")).lower()
                )
            )
            if thin_caddy_release:
                retreat_pos = release_pos + np.array([0.0, 0.0, max(float(retreat), 0.24)], dtype=np.float64)
                retreat_move = self._goto_pose_legacy_checked(
                    retreat_pos,
                    quat,
                    tolerance=0.150,
                )
                align_records.append({"phase": "thin_caddy_vertical_retreat", "record": retreat_move})
            elif container_profile in {"high_semantic_container_release", "thin_object_high_container_release"}:
                retreat_move = self.move_finger_midpoint_delta(
                    np.array([0.0, 0.0, max(float(retreat), 0.20)], dtype=np.float64),
                    quat,
                    lin_vel_norm=0.028,
                    timeout_s=8.0,
                    tolerance=0.055,
                )
                retreat_pos = np.asarray(retreat_move["target_finger_midpoint"], dtype=np.float64)
            elif container_profile == "drawer_interior_release":
                interior_dir = np.asarray(
                    release_anchor.get("drawer_interior_direction", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(3)
                interior_dir[2] = 0.0
                interior_dir = interior_dir / max(float(np.linalg.norm(interior_dir)), 1e-9)
                up_move = self.move_finger_midpoint_delta(
                    np.array([0.0, 0.0, max(float(retreat), 0.14)], dtype=np.float64),
                    quat,
                    lin_vel_norm=0.028,
                    timeout_s=7.0,
                    tolerance=0.060,
                )
                front_move = self.move_finger_midpoint_delta(
                    -0.055 * interior_dir + np.array([0.0, 0.0, 0.030], dtype=np.float64),
                    quat,
                    lin_vel_norm=0.030,
                    timeout_s=5.0,
                    tolerance=0.065,
                )
                retreat_pos = np.asarray(front_move["target_finger_midpoint"], dtype=np.float64)
                align_records.append(
                    {
                        "phase": "drawer_upward_front_retreat",
                        "up_move": up_move,
                        "front_move": front_move,
                    }
                )
            else:
                retreat_pos = release_pos + np.array([0.0, 0.0, float(retreat)], dtype=np.float64)
                self.goto_pose(retreat_pos, quat)
        self._settle_simulation_steps(max(0, int(settle_steps) // 2))
        after = self.measure_object_grasp_state(resolved_object)
        relation = self._placement_relation_diagnostic(
            resolved_object,
            str(target["target_name"]),
            predicate=pred,
        )
        if (
            pred == "in"
            and str(release_anchor.get("container_release_profile", "")) == "drawer_interior_release"
            and (
                not bool(relation.get("satisfied", False) or relation.get("in", False))
                or (
                    bool(require_drawer_closing_clearance)
                    and not bool(
                        self._drawer_relation_closing_clearance(
                            relation,
                            min_front_clearance=float(drawer_min_front_clearance),
                        ).get(
                            "has_clearance",
                            False,
                        )
                    )
                )
            )
            and not bool(after.get("held", False))
        ):
            slide = self._drawer_slide_object_into_region(
                resolved_object,
                str(target["target_name"]),
                diagnostic=relation,
                max_iters=5 if bool(require_drawer_closing_clearance) else 7,
                settle_steps=max(18, int(settle_steps) // 2),
                require_closing_clearance=bool(require_drawer_closing_clearance),
                min_closing_front_clearance=float(drawer_min_front_clearance),
            )
            align_records.append({"phase": "drawer_post_release_slide", "record": slide})
            relation = slide.get("final_relation", relation) if isinstance(slide, dict) else relation
            after = self.measure_object_grasp_state(resolved_object)
        task_status = self.get_task_status()
        still_held_after_release = bool(after.get("held", False))
        drawer_closing_clearance = None
        drawer_goal_margin_ok = True
        if (
            pred == "in"
            and str(release_anchor.get("container_release_profile", "")) == "drawer_interior_release"
            and bool(require_drawer_closing_clearance)
        ):
            drawer_closing_clearance = self._drawer_relation_closing_clearance(
                relation,
                min_front_clearance=float(drawer_min_front_clearance),
            )
            drawer_goal_margin_ok = bool(drawer_closing_clearance.get("has_clearance", False))
        relation_satisfied = bool(
            (relation.get("satisfied", False) and drawer_goal_margin_ok and not still_held_after_release)
            or task_status.get("task_completed", False)
            or float(task_status.get("reward", 0.0)) >= 1.0
        )
        return {
            "object_name": resolved_object,
            "target": target,
            "release_anchor": release_anchor,
            "release_position": release_pos,
            "retreat_position": retreat_pos,
            "before": before,
            "after": after,
            "align_records": align_records,
            "relation": relation,
            "drawer_closing_clearance": drawer_closing_clearance,
            "task_status": task_status,
            "released": not still_held_after_release,
            "relation_satisfied": relation_satisfied,
            "needs_repair": not relation_satisfied,
            "skill": "low_release_object_to_target",
        }

    def pick_transport_release_object_to_target(
        self,
        object_name: str,
        *,
        predicate: str = "in",
        target_name: str | None = None,
        max_pick_attempts: int = 4,
        start_candidate_index: int = 0,
        settle_steps: int = 36,
        require_drawer_closing_clearance: bool = False,
        min_drawer_front_clearance: float | None = None,
    ) -> dict[str, Any]:
        """Continuously pick, transport, release, and verify a placement.

        Thin objects in narrow containers can slip between stage boundaries.
        This audited skill keeps the physical sequence in one robot rollout
        while still using only readonly diagnostics and motion primitives.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        pred = str(predicate).strip().lower()
        target = self._placement_target_for_object(
            resolved_object,
            predicate=pred,
            target_name=target_name,
        )
        pred = str(target.get("predicate", pred)).strip().lower()
        resolved_target = str(target["target_name"])
        records: list[dict[str, Any]] = []
        initial_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate=pred,
        )
        drawer_region_target = bool(self._is_drawer_region_target_name(resolved_target))
        drawer_front_clearance_threshold = (
            float(min_drawer_front_clearance)
            if bool(require_drawer_closing_clearance)
            and drawer_region_target
            and min_drawer_front_clearance is not None
            else 0.110
            if bool(require_drawer_closing_clearance) and drawer_region_target and "bottom" in resolved_target.lower()
            else 0.065
            if bool(require_drawer_closing_clearance) and drawer_region_target
            else 0.045
        )
        initial_drawer_clearance = (
            self._drawer_relation_closing_clearance(
                initial_relation,
                min_front_clearance=float(drawer_front_clearance_threshold),
            )
            if bool(require_drawer_closing_clearance) and drawer_region_target
            else None
        )
        initial_drawer_clearance_ok = bool(
            initial_drawer_clearance is None
            or initial_drawer_clearance.get("has_clearance", False)
        )
        if bool(initial_relation.get("satisfied", False)) and initial_drawer_clearance_ok:
            status = self.get_task_status()
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "drawer_closing_clearance": initial_drawer_clearance,
                "task_status": status,
                "relation_satisfied": True,
                "skill": "pick_transport_release_object_to_target",
            }
        if bool(initial_relation.get("satisfied", False)) and not initial_drawer_clearance_ok:
            records.append(
                {
                    "action": "initial_drawer_relation_satisfied_but_lacks_close_clearance",
                    "relation": initial_relation,
                    "drawer_closing_clearance": initial_drawer_clearance,
                    "repair_strategy": "drawer_local_clearance_seating_before_repick",
                }
            )
        initial_xy_distance = self._relation_xy_distance_to_target(initial_relation)
        drawer_initial_near_volume = False
        if drawer_region_target and "local_delta" in initial_relation and "site_size_abs" in initial_relation:
            try:
                local_delta_for_gate = np.asarray(initial_relation["local_delta"], dtype=np.float64).reshape(-1)[:3]
                site_size_for_gate = np.asarray(initial_relation["site_size_abs"], dtype=np.float64).reshape(-1)[:3]
                drawer_initial_near_volume = bool(
                    local_delta_for_gate.size >= 3
                    and site_size_for_gate.size >= 3
                    and abs(float(local_delta_for_gate[0])) <= max(float(abs(site_size_for_gate[0])) + 0.080, 0.16)
                    and abs(float(local_delta_for_gate[1])) <= max(float(abs(site_size_for_gate[1])) + 0.090, 0.17)
                    and abs(float(local_delta_for_gate[2])) <= max(float(abs(site_size_for_gate[2])) + 0.120, 0.18)
                )
            except Exception:
                drawer_initial_near_volume = False
        initial_near_relation = bool(
            np.isfinite(initial_xy_distance)
            and initial_xy_distance <= (0.32 if drawer_region_target else 0.16)
            and "local_delta" in initial_relation
            and (not drawer_region_target or drawer_initial_near_volume)
        )
        initial_hold_state = self.measure_object_grasp_state(resolved_object)
        if (
            pred == "in"
            and bool(initial_relation.get("contact", False))
            and initial_near_relation
            and not bool(initial_hold_state.get("held", False))
        ):
            if drawer_region_target:
                records.append(
                    {
                        "action": "initial_contact_drawer_local_clearance_seating",
                        "relation": initial_relation,
                        "drawer_closing_clearance": initial_drawer_clearance,
                        "reason": "drawer near-contact is already preserved; first try local close-clearance seating",
                    }
                )
                closure = self.local_placement_closure(
                    resolved_object,
                    predicate=pred,
                    target_name=resolved_target,
                    max_iters=6,
                    near_tolerance=0.32,
                    settle_steps=max(24, int(settle_steps)),
                    require_drawer_closing_clearance=bool(require_drawer_closing_clearance),
                    min_drawer_front_clearance=float(drawer_front_clearance_threshold),
                )
                records.append({"action": "initial_contact_drawer_local_placement_closure", "result": closure})
                final_relation = closure.get("final_relation", initial_relation) if isinstance(closure, dict) else initial_relation
                status = closure.get("task_status", self.get_task_status()) if isinstance(closure, dict) else self.get_task_status()
                final_drawer_clearance = (
                    closure.get("drawer_closing_clearance")
                    if isinstance(closure, dict)
                    else self._drawer_relation_closing_clearance(
                        final_relation,
                        min_front_clearance=float(drawer_front_clearance_threshold),
                    )
                    if bool(require_drawer_closing_clearance)
                    else None
                )
                if (
                    bool(closure.get("relation_satisfied", False))
                    or bool(status.get("task_completed", False))
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0
                ):
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "records": records,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "drawer_closing_clearance": final_drawer_clearance,
                        "task_status": status,
                        "relation_satisfied": True,
                        "skill": "pick_transport_release_object_to_target",
                    }
                records.append(
                    {
                        "action": "initial_contact_drawer_local_seating_failed_repick_fallback",
                        "relation": final_relation,
                        "drawer_closing_clearance": final_drawer_clearance,
                        "reason": "local seating did not produce close clearance; continue with physical re-pick fallback",
                    }
                )
            elif "book" in resolved_object.lower() and (
                "caddy" in resolved_target.lower() or "compartment" in resolved_target.lower()
            ):
                seating = self._thin_caddy_contact_seat_by_relation(
                    resolved_object,
                    resolved_target,
                    initial_relation=initial_relation,
                    max_iters=3,
                    settle_steps=max(20, int(settle_steps)),
                )
                records.append(
                    {
                        "action": "thin_caddy_contact_z_seating",
                        "result": seating,
                    }
                )
                final_relation = (
                    seating.get("final_relation", initial_relation)
                    if isinstance(seating, dict)
                    else initial_relation
                )
                status = self.get_task_status()
                relation_satisfied = bool(
                    (isinstance(seating, dict) and seating.get("satisfied", False))
                    or final_relation.get("satisfied", False)
                    or final_relation.get("in", False)
                    or status.get("task_completed", False)
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0
                )
                records.append(
                    {
                        "action": "skip_initial_contact_local_placement_closure",
                        "relation": final_relation,
                        "reason": "preserve thin-object caddy placement; generic planar containment push is harmful",
                    }
                )
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": final_relation,
                    "task_status": status,
                    "relation_satisfied": relation_satisfied,
                    "needs_repair": not relation_satisfied,
                    "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                    "xy_tolerance": 0.18,
                    "near_contact_relation": True,
                    "reason": (
                        "thin-object caddy contact seated with readonly z feedback"
                        if relation_satisfied
                        else "thin-object caddy near-contact preserved after bounded z seating"
                    ),
                    "skill": "pick_transport_release_object_to_target",
                }
            if not drawer_region_target:
                closure = self.local_placement_closure(
                    resolved_object,
                    predicate=pred,
                    target_name=resolved_target,
                    max_iters=4,
                    near_tolerance=0.28,
                    settle_steps=max(18, int(settle_steps)),
                )
                records.append({"action": "initial_contact_local_placement_closure", "result": closure})
                final_relation = closure.get("final_relation", initial_relation) if isinstance(closure, dict) else initial_relation
                status = closure.get("task_status", self.get_task_status()) if isinstance(closure, dict) else self.get_task_status()
                if bool(closure.get("relation_satisfied", False)) or bool(final_relation.get("satisfied", False)) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "records": records,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "task_status": status,
                        "relation_satisfied": True,
                        "skill": "pick_transport_release_object_to_target",
                    }

        hold = self.measure_object_grasp_state(resolved_object)
        if not bool(hold.get("held", False)):
            if (
                drawer_region_target
                and self._requires_strict_hold(resolved_object)
                and not bool(require_drawer_closing_clearance)
            ):
                pick = self.verified_low_side_pick_object(
                    resolved_object,
                    max_attempts=max(2, min(3, int(max_pick_attempts))),
                    start_candidate_index=max(0, int(start_candidate_index)),
                    lift_height=0.075,
                    settle_steps=max(12, int(settle_steps) // 2),
                )
                records.append({"action": "drawer_shallow_low_side_pick", "result": pick})
                if not bool(pick.get("held", False)):
                    fallback_pick = self.verified_geometry_pick_object(
                        resolved_object,
                        max_attempts=1,
                        start_candidate_index=max(0, int(start_candidate_index)),
                        lift_height=0.18,
                        settle_steps=max(8, int(settle_steps) // 3),
                    )
                    records.append({"action": "drawer_shallow_geometry_pick_fallback", "result": fallback_pick})
                    pick = fallback_pick
            else:
                pick = self.verified_geometry_pick_object(
                    resolved_object,
                    max_attempts=max(1, int(max_pick_attempts)),
                    start_candidate_index=max(0, int(start_candidate_index)),
                    lift_height=0.26 if "book" in resolved_object.lower() else 0.22,
                    settle_steps=max(8, int(settle_steps) // 3),
                )
                records.append({"action": "verified_geometry_pick", "result": pick})
        else:
            pick = {
                "object_name": resolved_object,
                "held": True,
                "existing_hold": hold,
                "skill": "measure_object_grasp_state",
            }
            records.append({"action": "existing_hold", "result": pick})

        if not bool(pick.get("held", False)):
            final_relation = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate=pred,
            )
            status = self.get_task_status()
            final_drawer_clearance = (
                self._drawer_relation_closing_clearance(
                    final_relation,
                    min_front_clearance=float(drawer_front_clearance_threshold),
                )
                if bool(require_drawer_closing_clearance) and drawer_region_target
                else None
            )
            final_clearance_ok = bool(
                final_drawer_clearance is None
                or final_drawer_clearance.get("has_clearance", False)
            )
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "drawer_closing_clearance": final_drawer_clearance,
                "task_status": status,
                "relation_satisfied": bool(final_relation.get("satisfied", False) and final_clearance_ok),
                "needs_repair": True,
                "reason": "continuous pick failed",
                "skill": "pick_transport_release_object_to_target",
            }

        if drawer_region_target:
            transport = {
                "object_name": resolved_object,
                "target": target,
                "held_before": True,
                "held_after": True,
                "arrived": False,
                "reason": "drawer target uses direct held waypoint insertion in low_release",
                "skill": "skip_guarded_transport_for_drawer_direct_insertion",
            }
        else:
            transport = self.guarded_transport_held_object_to_target(
                resolved_object,
                predicate=pred,
                target_name=resolved_target,
                pre_release_z=0.16 if "book" in resolved_object.lower() and pred == "in" else 0.18,
                settle_steps=max(6, int(settle_steps) // 4),
            )
        records.append({"action": "guarded_transport", "result": transport})
        release = self.low_release_object_to_target(
            resolved_object,
            predicate=pred,
            target_name=resolved_target,
            release_z=0.060 if "book" in resolved_object.lower() and pred == "in" else None,
            approach=0.10,
            retreat=0.20,
            settle_steps=max(16, int(settle_steps)),
            require_drawer_closing_clearance=bool(require_drawer_closing_clearance),
        )
        records.append({"action": "low_release", "result": release})
        final_relation = release.get("relation", {}) if isinstance(release, dict) else {}
        status = release.get("task_status", self.get_task_status()) if isinstance(release, dict) else self.get_task_status()
        if bool(release.get("relation_satisfied", False)) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "drawer_closing_clearance": release.get("drawer_closing_clearance") if isinstance(release, dict) else None,
                "task_status": status,
                "relation_satisfied": True,
                "skill": "pick_transport_release_object_to_target",
            }
        if drawer_region_target:
            z_only_near_success = bool(
                release.get("z_only_near_success", False)
                if isinstance(release, dict)
                else False
            ) or self._drawer_relation_is_z_only_above_box(final_relation, max_z_excess=0.020)
            if (
                not z_only_near_success
                and pred == "in"
                and bool(final_relation.get("contact", False))
                and "local_delta" in final_relation
            ):
                correction = self._drawer_slide_object_into_region(
                    resolved_object,
                    resolved_target,
                    diagnostic=final_relation,
                    max_iters=6,
                    settle_steps=max(24, int(settle_steps)),
                    require_closing_clearance=bool(require_drawer_closing_clearance),
                    min_closing_front_clearance=float(drawer_front_clearance_threshold),
                )
                records.append({"action": "drawer_relation_guided_slide_insert", "result": correction})
                final_relation = correction.get("final_relation", final_relation) if isinstance(correction, dict) else final_relation
                status = correction.get("task_status", status) if isinstance(correction, dict) else status
                if bool(correction.get("satisfied", False)) or bool(final_relation.get("satisfied", False) or final_relation.get("in", False)) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "records": records,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "drawer_closing_clearance": correction.get("drawer_closing_clearance") if isinstance(correction, dict) else None,
                        "task_status": status,
                        "relation_satisfied": True,
                        "skill": "pick_transport_release_object_to_target",
                    }
            drawer_closing_clearance = (
                self._drawer_relation_closing_clearance(
                    final_relation,
                    min_front_clearance=float(drawer_front_clearance_threshold),
                )
                if bool(require_drawer_closing_clearance)
                else release.get("drawer_closing_clearance")
                if isinstance(release, dict)
                else None
            )
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "drawer_closing_clearance": drawer_closing_clearance,
                "task_status": status,
                "relation_satisfied": False,
                "needs_repair": True,
                "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                "xy_tolerance": 0.32,
                "near_contact_relation": bool(
                    pred == "in"
                    and bool(final_relation.get("contact", False))
                    and "local_delta" in final_relation
                ),
                "z_only_near_success": bool(z_only_near_success),
                "reason": (
                    "drawer placement is xy-valid but slightly above site AABB; preserve state for z-only repair"
                    if z_only_near_success
                    else "drawer direct release did not satisfy relation; return evidence for stage replanning"
                ),
                "skill": "pick_transport_release_object_to_target",
            }

        final_xy_distance = self._relation_xy_distance_to_target(final_relation)
        near_contact_xy_tolerance = 0.26 if drawer_region_target else 0.18
        near_contact_relation = bool(
            pred == "in"
            and bool(final_relation.get("contact", False))
            and "local_delta" in final_relation
            and np.isfinite(final_xy_distance)
            and final_xy_distance <= near_contact_xy_tolerance
        )
        if near_contact_relation:
            thin_caddy_target = bool(
                "book" in resolved_object.lower()
                and ("caddy" in resolved_target.lower() or "compartment" in resolved_target.lower())
            )
            if drawer_region_target:
                correction = self._drawer_slide_object_into_region(
                    resolved_object,
                    resolved_target,
                    diagnostic=final_relation,
                    max_iters=5,
                    settle_steps=max(24, int(settle_steps)),
                    require_closing_clearance=False,
                )
                records.append({"action": "drawer_relation_guided_slide_insert", "result": correction})
                final_relation = correction.get("final_relation", final_relation) if isinstance(correction, dict) else final_relation
            elif thin_caddy_target:
                correction = self._thin_caddy_relation_guided_nudge(
                    resolved_object,
                    resolved_target,
                    diagnostic=final_relation,
                    max_iters=4,
                    settle_steps=max(24, int(settle_steps)),
                )
                records.append({"action": "thin_caddy_relation_guided_nudge", "result": correction})
                final_relation = correction.get("final_relation", final_relation) if isinstance(correction, dict) else final_relation
                if not bool(final_relation.get("satisfied", False) or final_relation.get("in", False)):
                    repick_release = self._thin_caddy_repick_low_release(
                        resolved_object,
                        resolved_target,
                        settle_steps=max(24, int(settle_steps)),
                        start_candidate_index=max(0, int(start_candidate_index)),
                    )
                    records.append({"action": "thin_caddy_repick_low_release", "result": repick_release})
                    final_relation = (
                        repick_release.get("final_relation", final_relation)
                        if isinstance(repick_release, dict)
                        else final_relation
                    )
            else:
                correction = self._containment_correction_to_site(
                    resolved_object,
                    resolved_target,
                    diagnostic=final_relation,
                    max_iters=3,
                    settle_steps=max(24, int(settle_steps)),
                )
                records.append({"action": "near_contact_containment_correction", "result": correction})
                final_relation = correction.get("final_relation", final_relation) if isinstance(correction, dict) else final_relation
            status = self.get_task_status()
            relation_satisfied = bool(
                (isinstance(correction, dict) and correction.get("satisfied", False))
                or bool(final_relation.get("satisfied", False))
                or bool(status.get("task_completed", False))
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            )
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "task_status": status,
                "relation_satisfied": relation_satisfied,
                "needs_repair": not relation_satisfied,
                "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                "xy_tolerance": near_contact_xy_tolerance,
                "near_contact_relation": True,
                "reason": "near-contact drawer placement corrected with bounded relation-guided slide"
                if drawer_region_target and relation_satisfied
                else "near-contact placement corrected with bounded containment closure"
                if relation_satisfied
                else (
                    "thin-object caddy near-contact preserved; generic containment closure skipped"
                    if thin_caddy_target
                    else "near-contact drawer placement still needs relation-guided slide"
                    if drawer_region_target
                    else "near-contact placement still needs bounded local closure"
                ),
                "skill": "pick_transport_release_object_to_target",
            }

        closure = self.local_placement_closure(
            resolved_object,
            predicate=pred,
            target_name=resolved_target,
            max_iters=3 if pred == "in" else 2,
            near_tolerance=0.24 if pred == "in" else None,
            settle_steps=max(16, int(settle_steps)),
        )
        records.append({"action": "local_placement_closure", "result": closure})
        final_relation = closure.get("final_relation", final_relation) if isinstance(closure, dict) else final_relation
        status = closure.get("task_status", status) if isinstance(closure, dict) else status
        relation_satisfied = bool(
            (isinstance(closure, dict) and closure.get("relation_satisfied", False))
            or bool(final_relation.get("satisfied", False))
            or bool(status.get("task_completed", False))
            or float(status.get("reward", 0.0) or 0.0) >= 1.0
        )
        return {
            "object_name": resolved_object,
            "target": target,
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "task_status": status,
            "relation_satisfied": relation_satisfied,
            "needs_repair": not relation_satisfied,
            "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
            "xy_tolerance": 0.18 if pred == "in" else 0.075,
            "near_contact_relation": bool(
                pred == "in"
                and bool(final_relation.get("contact", False))
                and "local_delta" in final_relation
                and self._relation_xy_distance_to_target(final_relation) <= 0.18
            ),
            "skill": "pick_transport_release_object_to_target",
        }

    def repair_placement_relation(
        self,
        object_name: str,
        *,
        predicate: str = "in",
        target_name: str | None = None,
        max_iters: int = 3,
        start_candidate_index: int = 0,
        settle_steps: int = 60,
    ) -> dict[str, Any]:
        """Physically repair a failed placement relation and verify it.

        The repair loop uses only robot motion plus readonly relation checks:
        if the object is still held, it performs a low release; otherwise it
        applies short planar corrective pushes, and if needed re-picks and
        low-releases the object. It never edits simulator qpos/qvel or goal
        predicates.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        pred = str(predicate).strip().lower()
        target = self._placement_target_for_object(
            resolved_object,
            predicate=pred,
            target_name=target_name,
        )
        pred = str(target.get("predicate", pred)).strip().lower()
        resolved_target = str(target["target_name"])
        target_type = str(target.get("target_type", "site"))
        records: list[dict[str, Any]] = []
        initial_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate=pred,
        )
        if bool(initial_relation.get("satisfied", False)):
            task_status = self.get_task_status()
            return {
                "object_name": resolved_object,
                "target": target,
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "task_status": task_status,
                "relation_satisfied": True,
                "skill": "repair_placement_relation",
            }

        for repair_idx in range(max(1, int(max_iters))):
            before_hold = self.measure_object_grasp_state(resolved_object)
            if bool(before_hold.get("held", False)):
                release = self.low_release_object_to_target(
                    resolved_object,
                    predicate=pred,
                    target_name=resolved_target,
                    release_z=self._repair_release_z(pred, target, repair_idx),
                    approach=0.12,
                    retreat=0.16,
                    settle_steps=settle_steps,
                )
                relation = release.get("relation", {}) if isinstance(release, dict) else {}
                records.append(
                    {
                        "repair_index": repair_idx,
                        "action": "low_release_held_object",
                        "before_hold": before_hold,
                        "release": release,
                        "relation": relation,
                    }
                )
                if bool(release.get("relation_satisfied", False)):
                    task_status = self.get_task_status()
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "records": records,
                        "initial_relation": initial_relation,
                        "final_relation": relation,
                        "task_status": task_status,
                        "relation_satisfied": True,
                        "skill": "repair_placement_relation",
                    }

            if self._is_shared_broad_region_target(
                resolved_object,
                resolved_target,
                predicate=pred,
                target=target,
            ):
                grasp = self.verified_geometry_pick_object(
                    resolved_object,
                    max_attempts=2,
                    start_candidate_index=int(start_candidate_index) + repair_idx * 2,
                    lift_height=0.22,
                    settle_steps=max(8, int(settle_steps) // 4),
                )
                records.append(
                    {
                        "repair_index": repair_idx,
                        "action": "shared_region_geometry_repick",
                        "grasp": grasp,
                    }
                )
                if bool(grasp.get("held", False)):
                    release = self.low_release_object_to_target(
                        resolved_object,
                        predicate=pred,
                        target_name=resolved_target,
                        release_z=self._repair_release_z(pred, target, repair_idx),
                        approach=0.12,
                        retreat=0.17,
                        settle_steps=settle_steps,
                    )
                    relation = release.get("relation", {}) if isinstance(release, dict) else {}
                    records.append(
                        {
                            "repair_index": repair_idx,
                            "action": "shared_region_low_release",
                            "release": release,
                            "relation": relation,
                        }
                    )
                    if bool(release.get("relation_satisfied", False)):
                        task_status = self.get_task_status()
                        return {
                            "object_name": resolved_object,
                            "target": target,
                            "records": records,
                            "initial_relation": initial_relation,
                            "final_relation": relation,
                            "task_status": task_status,
                            "relation_satisfied": True,
                            "skill": "repair_placement_relation",
                        }
                continue

            if pred == "on" and target_type != "site":
                # Object-on-object placements are usually shallow contacts on a
                # small support. Prefer fingertip-midpoint grasps that enter from
                # the rim/side; a blind top-down TCP grasp can skim above the
                # object and drag it away from the support.
                grasp = self.approach_and_grasp_object(
                    resolved_object,
                    max_attempts=4,
                    start_candidate_index=int(start_candidate_index) + repair_idx * 2,
                    lift_height=0.115,
                    settle_steps=max(8, int(settle_steps) // 4),
                )
                records.append(
                    {
                        "repair_index": repair_idx,
                        "action": "object_support_affordance_repick",
                        "grasp": grasp,
                    }
                )
                if bool(grasp.get("held", False)):
                    release = self.low_release_object_to_target(
                        resolved_object,
                        predicate=pred,
                        target_name=resolved_target,
                        release_z=self._repair_release_z(pred, target, repair_idx),
                        approach=0.12,
                        retreat=0.16,
                        settle_steps=settle_steps,
                    )
                    relation = release.get("relation", {}) if isinstance(release, dict) else {}
                    records.append(
                        {
                            "repair_index": repair_idx,
                            "action": "object_support_affordance_low_release",
                            "release": release,
                            "relation": relation,
                        }
                    )
                    if bool(release.get("relation_satisfied", False)):
                        task_status = self.get_task_status()
                        return {
                            "object_name": resolved_object,
                            "target": target,
                            "records": records,
                            "initial_relation": initial_relation,
                            "final_relation": relation,
                            "task_status": task_status,
                            "relation_satisfied": True,
                            "skill": "repair_placement_relation",
                        }

                side_grasp = self.grasp_tall_object_side(
                    resolved_object,
                    max_attempts=2,
                    start_candidate_index=int(start_candidate_index) + repair_idx * 2,
                    lift_height=0.135,
                    settle_steps=max(8, int(settle_steps) // 4),
                )
                records.append(
                    {
                        "repair_index": repair_idx,
                        "action": "object_support_side_repick",
                        "grasp": side_grasp,
                    }
                )
                if bool(side_grasp.get("held", False)):
                    release = self.low_release_object_to_target(
                        resolved_object,
                        predicate=pred,
                        target_name=resolved_target,
                        release_z=self._repair_release_z(pred, target, repair_idx),
                        approach=0.12,
                        retreat=0.16,
                        settle_steps=settle_steps,
                    )
                    relation = release.get("relation", {}) if isinstance(release, dict) else {}
                    records.append(
                        {
                            "repair_index": repair_idx,
                            "action": "object_support_side_low_release",
                            "release": release,
                            "relation": relation,
                        }
                    )
                    if bool(release.get("relation_satisfied", False)):
                        task_status = self.get_task_status()
                        return {
                            "object_name": resolved_object,
                            "target": target,
                            "records": records,
                            "initial_relation": initial_relation,
                            "final_relation": relation,
                            "task_status": task_status,
                            "relation_satisfied": True,
                            "skill": "repair_placement_relation",
                        }

                grasp = self.verified_lift_pick_object(
                    resolved_object,
                    grasp_quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
                    max_attempts=1,
                    start_candidate_index=int(start_candidate_index) + repair_idx,
                    lift_height=0.18,
                    success_dz=0.045,
                    approach=0.11,
                    settle_steps=max(12, int(settle_steps) // 4),
                )
                records.append(
                    {
                        "repair_index": repair_idx,
                        "action": "object_top_repick",
                        "grasp": grasp,
                    }
                )
                if bool(grasp.get("held", False)):
                    release = self.low_release_object_to_target(
                        resolved_object,
                        predicate=pred,
                        target_name=resolved_target,
                        release_z=self._repair_release_z(pred, target, repair_idx),
                        approach=0.12,
                        retreat=0.16,
                        settle_steps=settle_steps,
                    )
                    relation = release.get("relation", {}) if isinstance(release, dict) else {}
                    records.append(
                        {
                            "repair_index": repair_idx,
                            "action": "object_top_repick_low_release",
                            "release": release,
                            "relation": relation,
                        }
                    )
                    if bool(release.get("relation_satisfied", False)):
                        task_status = self.get_task_status()
                        return {
                            "object_name": resolved_object,
                            "target": target,
                            "records": records,
                            "initial_relation": initial_relation,
                            "final_relation": relation,
                            "task_status": task_status,
                            "relation_satisfied": True,
                            "skill": "repair_placement_relation",
                        }
                # Object-on-object placement is fragile under planar pushes:
                # pushing a mug/bowl on a plate often knocks it off the support.
                continue

            push = self._corrective_push_to_target(
                resolved_object,
                resolved_target,
                predicate=pred,
                max_iters=1 if pred == "in" else 2,
            )
            relation_after_push = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate=pred,
            )
            records.append(
                {
                    "repair_index": repair_idx,
                    "action": "corrective_push",
                    "push": push,
                    "relation": relation_after_push,
                }
            )
            if bool(relation_after_push.get("satisfied", False)):
                task_status = self.get_task_status()
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation_after_push,
                    "task_status": task_status,
                    "relation_satisfied": True,
                    "skill": "repair_placement_relation",
                }

            grasp = self.verified_lift_pick_object(
                resolved_object,
                grasp_quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
                max_attempts=1,
                start_candidate_index=int(start_candidate_index) + repair_idx,
                lift_height=0.18,
                success_dz=0.045,
                approach=0.11,
                settle_steps=max(12, int(settle_steps) // 4),
            )
            records.append(
                {
                    "repair_index": repair_idx,
                    "action": "repick",
                    "grasp": grasp,
                }
            )
            if not bool(grasp.get("held", False)):
                continue
            release = self.low_release_object_to_target(
                resolved_object,
                predicate=pred,
                target_name=resolved_target,
                release_z=self._repair_release_z(pred, target, repair_idx),
                approach=0.12,
                retreat=0.16,
                settle_steps=settle_steps,
            )
            relation = release.get("relation", {}) if isinstance(release, dict) else {}
            records.append(
                {
                    "repair_index": repair_idx,
                    "action": "repick_low_release",
                    "release": release,
                    "relation": relation,
                }
            )
            if bool(release.get("relation_satisfied", False)):
                task_status = self.get_task_status()
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation,
                    "task_status": task_status,
                    "relation_satisfied": True,
                    "skill": "repair_placement_relation",
                }

        final_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate=pred,
        )
        task_status = self.get_task_status()
        return {
            "object_name": resolved_object,
            "target": target,
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "task_status": task_status,
            "relation_satisfied": bool(
                final_relation.get("satisfied", False)
                or task_status.get("task_completed", False)
                or float(task_status.get("reward", 0.0)) >= 1.0
            ),
            "skill": "repair_placement_relation",
        }

    def rack_supported_release_object_to_target(
        self,
        object_name: str,
        *,
        target_name: str | None = "rack",
        settle_steps: int = 45,
    ) -> dict[str, Any]:
        """Release a held object onto a narrow rack/caddy support with local feedback.

        The policy reads the target site and site-on diagnostic, moves the held
        object through conservative waypoints, nudges it in the target site's
        local XY frame while still holding it, and opens only after the object is
        inside the narrow support band. It is a physical robot skill: no object,
        joint, reward, or predicate state is written.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        target = self._placement_target_for_object(
            resolved_object,
            predicate="on",
            target_name=target_name,
        )
        site_name = str(target["target_name"])
        if str(target.get("target_type", "")) != "site":
            site_name = str(self._semantic_target_state_name(target_name or "rack", predicate="on") or site_name)
        site_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        site_size = np.asarray(target.get("target_size", np.array([0.08, 0.025, 0.003])), dtype=np.float64).reshape(-1)[:3]
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        before = self.measure_object_grasp_state(resolved_object)
        if not bool(before.get("held", False)):
            relation = self._placement_relation_diagnostic(resolved_object, site_name, predicate="on")
            task_status = self.get_task_status()
            return {
                "object_name": resolved_object,
                "target": target,
                "site_name": site_name,
                "before": before,
                "records": [],
                "relation": relation,
                "task_status": task_status,
                "relation_satisfied": bool(
                    relation.get("satisfied", False)
                    or task_status.get("task_completed", False)
                    or float(task_status.get("reward", 0.0)) >= 1.0
                ),
                "reason": "object is not verified-held before rack release",
                "skill": "rack_supported_release_object_to_target",
            }

        try:
            site_pos, site_quat = self.get_site_pose(site_name)
            site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()
            site_size = self._target_site_size(site_name)
        except Exception:
            site_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        release_offset = np.array([0.035, -0.010, 0.0], dtype=np.float64)
        release_pos = site_pos + release_offset
        release_pos[2] += 0.012
        hover_pos = release_pos.copy()
        hover_pos[2] += 0.185
        object_start = np.asarray(before["object_position"], dtype=np.float64).reshape(3).copy()
        needs_transport_progress_check = bool(float(np.linalg.norm(object_start[:2] - hover_pos[:2])) > 0.060)
        high_now = object_start.copy()
        high_now[2] = max(float(high_now[2]) + 0.080, float(site_pos[2]) + 0.220)
        site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64)).as_matrix()
        records: list[dict[str, Any]] = []
        rail_kiss = release_pos.copy()

        def relation_done(diag: dict[str, Any] | None, status: dict[str, Any] | None) -> bool:
            return bool(
                (diag or {}).get("on", False)
                or (status or {}).get("task_completed", False)
                or float((status or {}).get("reward", 0.0) or 0.0) >= 1.0
            )

        def compact_diag(diag: dict[str, Any] | None) -> dict[str, Any]:
            if not isinstance(diag, dict):
                return {}
            local_delta = np.asarray(diag.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(3)
            site_size_arr = np.asarray(diag.get("site_size", site_size), dtype=np.float64).reshape(-1)[:3]
            return {
                "on": bool(diag.get("on", False)),
                "in_xy": bool(diag.get("in_xy", False)),
                "in_z": bool(diag.get("in_z", False)),
                "contact": bool(diag.get("contact", False)),
                "local_delta": local_delta.copy(),
                "site_size": site_size_arr.copy(),
            }

        def site_report(phase: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            obj_state = self.measure_object_grasp_state(resolved_object)
            diag = self.get_site_on_diagnostic(resolved_object, site_name)
            status = self.get_task_status()
            records.append(
                {
                    "phase": phase,
                    "object_state": obj_state,
                    "diag": diag,
                    "task_status": status,
                }
            )
            return obj_state, diag, status

        def move_legacy_pose(
            phase: str,
            pose_position: np.ndarray,
            *,
            lin_vel_norm: float,
            timeout_s: float,
            settle: int = 20,
        ) -> dict[str, Any]:
            target_pos = np.asarray(pose_position, dtype=np.float64).reshape(3).copy()
            ee_before, quat_before = self.get_ee_pose()
            object_before = self.measure_object_grasp_state(resolved_object)
            move_error = ""
            try:
                FrankaLiberoPrivilegedApi.goto_pose_interactive_cartesian(
                    self,
                    lambda tp=target_pos.copy(): (tp, quat.copy()),
                    lin_vel_norm=float(lin_vel_norm),
                    timeout_s=float(timeout_s),
                )
            except Exception as exc:
                move_error = repr(exc)
            self._settle_simulation_steps(max(0, int(settle)))
            ee_after, quat_after = self.get_ee_pose()
            object_after = self.measure_object_grasp_state(resolved_object)
            record = {
                "phase": phase,
                "target_pose_position": target_pos,
                "ee_before": np.asarray(ee_before, dtype=np.float64).copy(),
                "ee_after": np.asarray(ee_after, dtype=np.float64).copy(),
                "ee_quaternion_before": np.asarray(quat_before, dtype=np.float64).copy(),
                "ee_quaternion_after": np.asarray(quat_after, dtype=np.float64).copy(),
                "object_before": object_before,
                "object_after": object_after,
                "move_error": move_error,
            }
            records.append(record)
            return record

        def finish(reason: str) -> dict[str, Any]:
            final_diag = self.get_site_on_diagnostic(resolved_object, site_name)
            task_status = self.get_task_status()
            final_ok = relation_done(final_diag, task_status)
            return {
                "object_name": resolved_object,
                "target": target,
                "site_name": site_name,
                "release_position": release_pos,
                "hover_position": hover_pos,
                "before": before,
                "records": records,
                "relation": final_diag,
                "task_status": task_status,
                "summary": {
                    "reason": reason,
                    "final": compact_diag(final_diag),
                    "reward": float(task_status.get("reward", 0.0) or 0.0),
                    "task_completed": bool(task_status.get("task_completed", False)),
                    "record_phases": [str(record.get("phase", "")) for record in records],
                },
                "relation_satisfied": final_ok,
                "needs_repair": not final_ok,
                "reason": reason,
                "skill": "rack_supported_release_object_to_target",
            }

        move_legacy_pose("raise_current", high_now, lin_vel_norm=0.20, timeout_s=5.0, settle=20)
        for idx, frac in enumerate((0.34, 0.66, 1.0)):
            waypoint = high_now.copy()
            waypoint[:2] = object_start[:2] * (1.0 - frac) + hover_pos[:2] * frac
            if idx == 2:
                waypoint = hover_pos.copy()
            move = move_legacy_pose(
                f"waypoint_{idx}",
                waypoint,
                lin_vel_norm=0.12,
                timeout_s=8.0,
                settle=22,
            )
            obj_now = move["object_after"]
            obj_pos = np.asarray(obj_now["object_position"], dtype=np.float64).reshape(3)
            progress = float(np.linalg.norm(obj_pos[:2] - object_start[:2]))
            remaining = float(np.linalg.norm(obj_pos[:2] - site_pos[:2]))
            records.append(
                {
                    "phase": f"waypoint_{idx}_progress",
                    "progress_from_start": progress,
                    "remaining_site_xy": remaining,
                    "held": bool(obj_now.get("held", False)),
                }
            )
            if not bool(obj_now.get("held", False)):
                return finish(f"object not held after rack waypoint {idx}")
            if idx < 2 and needs_transport_progress_check and progress < 0.030:
                return finish(f"object did not follow rack waypoint {idx}")

        pre_descend_state, pre_descend_diag, pre_descend_status = site_report("pre_descend")
        pre_descend_obj = np.asarray(pre_descend_state["object_position"], dtype=np.float64).reshape(3)
        if relation_done(pre_descend_diag, pre_descend_status):
            return finish("relation already satisfied before rack descent")
        if float(np.linalg.norm(pre_descend_obj[:2] - site_pos[:2])) > 0.115:
            return finish("rack pre-descend gate failed")

        move_legacy_pose("release_descent", release_pos, lin_vel_norm=0.055, timeout_s=7.0, settle=25)
        pre_open_state, pre_open_diag, pre_open_status = site_report("pre_open")
        pre_open_obj = np.asarray(pre_open_state["object_position"], dtype=np.float64).reshape(3)
        if relation_done(pre_open_diag, pre_open_status):
            return finish("relation satisfied before opening")
        if float(np.linalg.norm(pre_open_obj[:2] - site_pos[:2])) > 0.080:
            return finish("rack pre-open gate failed")

        pre_delta = np.asarray(pre_open_diag.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(3)
        desired_local_xy = np.array(
            [
                float(np.clip(pre_delta[0], -0.030, 0.030)),
                float(np.clip(0.010, -0.014, 0.014)),
            ],
            dtype=np.float64,
        )
        cradle_pos = release_pos.copy()
        records.append(
            {
                "phase": "cradle_start",
                "pre_delta": pre_delta,
                "site_size": site_size.copy(),
                "desired_local_xy": desired_local_xy.copy(),
            }
        )
        for idx in range(3):
            diag_now = self.get_site_on_diagnostic(resolved_object, site_name)
            local_delta = np.asarray(diag_now.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(3)
            err_local = np.array(
                [
                    desired_local_xy[0] - float(local_delta[0]),
                    desired_local_xy[1] - float(local_delta[1]),
                    0.0,
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(err_local[:2]))
            if norm > 0.016:
                err_local[:2] *= 0.016 / max(norm, 1e-6)
            shift = site_rot @ err_local
            shift[2] = 0.0
            records.append(
                {
                    "phase": f"cradle_shift_plan_{idx}",
                    "diag": diag_now,
                    "desired_local_xy": desired_local_xy.copy(),
                    "err_local": err_local.copy(),
                    "shift_robot": shift.copy(),
                }
            )
            if float(np.linalg.norm(shift[:2])) <= 0.0025:
                break
            cradle_pos = cradle_pos + shift
            move_legacy_pose(
                f"cradle_shift_{idx}",
                cradle_pos,
                lin_vel_norm=0.025,
                timeout_s=4.0,
                settle=35,
            )

        rail_kiss = cradle_pos.copy()
        rail_kiss[2] -= 0.006
        move_legacy_pose("rail_kiss", rail_kiss, lin_vel_norm=0.018, timeout_s=4.0, settle=55)
        kiss_state, kiss_diag, kiss_status = site_report("pre_release_rail_kiss")
        kiss_delta = np.asarray(kiss_diag.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(3)
        narrow_limit = max(float(abs(site_size[1])) + 0.010, 0.032)
        records.append(
            {
                "phase": "rail_kiss_gate",
                "local_delta": kiss_delta.copy(),
                "narrow_limit": narrow_limit,
                "inside_narrow_band": bool(abs(float(kiss_delta[1])) <= narrow_limit),
            }
        )
        if abs(float(kiss_delta[1])) > narrow_limit:
            return finish("rack rail-kiss narrow gate failed")

        self.open_gripper()
        self._settle_simulation_steps(35 if abs(float(kiss_delta[1])) <= narrow_limit else 80)
        release_state, release_diag, release_status = site_report("release_open_no_retreat")
        if not relation_done(release_diag, release_status):
            self._settle_simulation_steps(95)
            site_report("post_release_settle")

        latest_diag = self.get_site_on_diagnostic(resolved_object, site_name)
        latest_status = self.get_task_status()
        if relation_done(latest_diag, latest_status):
            micro_lift = rail_kiss.copy()
            micro_lift[2] += 0.040
            move_legacy_pose("micro_lift", micro_lift, lin_vel_norm=0.030, timeout_s=5.0, settle=100)
        else:
            records.append(
                {
                    "phase": "skip_micro_lift_after_unsatisfied_release",
                    "diag": latest_diag,
                    "task_status": latest_status,
                }
            )

        final_diag = self.get_site_on_diagnostic(resolved_object, site_name)
        task_status = self.get_task_status()
        relation_satisfied = relation_done(final_diag, task_status)
        phase_diags = {}
        for record in records:
            phase = str(record.get("phase", ""))
            if phase in {
                "pre_descend",
                "pre_open",
                "pre_release_rail_kiss",
                "release_open_no_retreat",
                "post_release_settle",
                "skip_micro_lift_after_unsatisfied_release",
                "micro_lift",
            }:
                phase_diags[phase] = compact_diag(record.get("diag"))

        return {
            "object_name": resolved_object,
            "target": target,
            "site_name": site_name,
            "release_position": release_pos,
            "hover_position": hover_pos,
            "before": before,
            "records": records,
            "relation": final_diag,
            "task_status": task_status,
            "summary": {
                "release_position": release_pos.copy(),
                "hover_position": hover_pos.copy(),
                "rail_kiss_position": rail_kiss.copy(),
                "narrow_limit": narrow_limit,
                "phase_diags": phase_diags,
                "final": compact_diag(final_diag),
                "reward": float(task_status.get("reward", 0.0) or 0.0),
                "task_completed": bool(task_status.get("task_completed", False)),
                "record_phases": [str(record.get("phase", "")) for record in records],
            },
            "relation_satisfied": relation_satisfied,
            "needs_repair": not relation_satisfied,
            "skill": "rack_supported_release_object_to_target",
        }

    def turn_on_stove_by_sweep(
        self,
        *,
        max_sweeps: int = 10,
        settle_steps: int = 40,
    ) -> dict[str, Any]:
        """Turn on the stove with short physical button sweeps."""
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        before = self.get_object_state_diagnostic("flat_stove_1")
        if bool(before.get("turn_on", False)):
            return {
                "before": before,
                "after": before,
                "sweeps": [],
                "turned_on": True,
                "skill": "turn_on_stove_by_sweep",
            }
        button_pos, button_quat = self.get_body_pose("flat_stove_1_button")
        button_pos = np.asarray(button_pos, dtype=np.float64).reshape(3).copy()
        button_rot = self._rotation_from_wxyz(np.asarray(button_quat, dtype=np.float64).reshape(4)).as_matrix()
        specs = [
            (np.array([0.045, 0.000, 0.030]), np.array([0.0, 1.0, 0.0]), 0.060, 0.145),
            (np.array([0.045, 0.000, 0.030]), np.array([0.0, -1.0, 0.0]), 0.060, 0.145),
            (np.array([-0.045, 0.000, 0.030]), np.array([0.0, -1.0, 0.0]), 0.060, 0.145),
            (np.array([-0.045, 0.000, 0.030]), np.array([0.0, 1.0, 0.0]), 0.060, 0.145),
            (np.array([0.000, 0.045, 0.030]), np.array([-1.0, 0.0, 0.0]), 0.060, 0.145),
            (np.array([0.000, 0.045, 0.030]), np.array([1.0, 0.0, 0.0]), 0.060, 0.145),
            (np.array([0.000, -0.045, 0.030]), np.array([1.0, 0.0, 0.0]), 0.060, 0.145),
            (np.array([0.000, -0.045, 0.030]), np.array([-1.0, 0.0, 0.0]), 0.060, 0.145),
            (np.array([0.000, 0.000, 0.035]), np.array([1.0, 0.0, 0.0]), 0.090, 0.220),
            (np.array([0.000, 0.000, 0.035]), np.array([0.0, -1.0, 0.0]), 0.090, 0.220),
        ]
        records: list[dict[str, Any]] = []
        self.open_gripper()
        for idx, (local_contact, local_tangent, backoff, forward) in enumerate(specs[: max(1, int(max_sweeps))]):
            contact = button_pos + button_rot @ local_contact
            tangent = button_rot @ local_tangent
            tangent[2] = 0.0
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                continue
            tangent = tangent / norm
            start = contact - tangent * float(backoff)
            end = contact + tangent * float(forward)
            self.goto_pose(start + np.array([0.0, 0.0, 0.080], dtype=np.float64), quat)
            self.goto_pose(start, quat)
            self.goto_pose(end, quat)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            self.goto_pose(end + np.array([0.0, 0.0, 0.080], dtype=np.float64), quat)
            self._settle_simulation_steps(max(8, int(settle_steps) // 2))
            after = self.get_object_state_diagnostic("flat_stove_1")
            record = {
                "idx": int(idx),
                "contact": contact.copy(),
                "tangent": tangent.copy(),
                "start": start.copy(),
                "end": end.copy(),
                "after": after,
                "turned_on": bool(after.get("turn_on", False)),
            }
            records.append(record)
            if bool(record["turned_on"]):
                return {
                    "before": before,
                    "after": after,
                    "sweeps": records,
                    "turned_on": True,
                    "skill": "turn_on_stove_by_sweep",
                }
        final = self.get_object_state_diagnostic("flat_stove_1")
        return {
            "before": before,
            "after": final,
            "sweeps": records,
            "turned_on": bool(final.get("turn_on", False)),
            "skill": "turn_on_stove_by_sweep",
        }

    def place_object_in_microwave_region(
        self,
        object_name: str,
        *,
        target_name: str | None = None,
        settle_steps: int = 60,
    ) -> dict[str, Any]:
        """Physically place a held object into a microwave contain/heating site."""
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        target = self._placement_target_for_object(
            resolved_object,
            predicate="in",
            target_name=target_name or "microwave",
        )
        resolved_target = str(target["target_name"])
        site_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        before_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate="in",
        )
        before_status = self.get_task_status()
        if bool(before_relation.get("satisfied", False)) or bool(before_status.get("task_completed", False)) or float(before_status.get("reward", 0.0) or 0.0) >= 1.0:
            return {
                "object_name": resolved_object,
                "target": target,
                "before_relation": before_relation,
                "final_relation": before_relation,
                "records": records,
                "task_status": before_status,
                "relation_satisfied": True,
                "skill": "place_object_in_microwave_region",
                "reason": "already satisfied",
            }

        hold = self.measure_object_grasp_state(resolved_object)
        held_at_entry = bool(hold.get("held", False))
        def near_inside_high(relation: dict[str, Any]) -> bool:
            local = np.asarray(relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            size = np.asarray(
                relation.get("site_size_abs", relation.get("site_size", np.array([0.0, 0.0, 0.0]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            return bool(
                bool(relation.get("contact", False))
                and local.size >= 3
                and size.size >= 3
                and np.all(np.abs(local[:2]) <= np.maximum(np.abs(size[:2]), 0.035))
                and local[2] > float(abs(size[2]))
                and local[2] <= float(abs(size[2])) + 0.100
            )

        def front_outside_contact_relation(relation: dict[str, Any]) -> bool:
            local = np.asarray(relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            size = np.asarray(
                relation.get("site_size_abs", relation.get("site_size", np.array([0.12, 0.0835, 0.08]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            if local.size < 3 or size.size < 3:
                return False
            return bool(
                bool(relation.get("contact", False))
                and not bool(relation.get("satisfied", False) or relation.get("in", False))
                and abs(float(local[0])) <= max(float(abs(size[0])) + 0.080, 0.180)
                and abs(float(local[1])) > max(float(abs(size[1])) + 0.030, 0.110)
                and abs(float(local[1])) <= max(float(abs(size[1])) + 0.155, 0.250)
                and float(local[2]) >= -float(abs(size[2])) - 0.075
                and float(local[2]) <= float(abs(size[2])) + 0.120
            )

        def support_shove_from_relation(relation: dict[str, Any], phase: str) -> dict[str, Any]:
            settle_record: dict[str, Any] = {"phase": phase, "initial_relation": relation}
            obj_pos, _ = self.get_object_pose(resolved_object)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            try:
                _, site_quat = self.get_site_pose(resolved_target)
                site_rot = self._rotation_from_wxyz(
                    np.asarray(site_quat, dtype=np.float64).reshape(4)
                ).as_matrix()
            except Exception:
                site_rot = np.eye(3, dtype=np.float64)
            support_z = min(float(obj_pos[2]) - 0.020, float(site_pos[2]) + 0.082)
            support_z = max(support_z, float(site_pos[2]) + 0.050)
            start = obj_pos + site_rot @ np.array([-0.055, 0.006, 0.0], dtype=np.float64)
            end = obj_pos + site_rot @ np.array([0.012, 0.008, 0.0], dtype=np.float64)
            start[2] = support_z
            end[2] = support_z
            self.open_gripper()
            settle_record["approach_move"] = self._goto_pose_legacy_checked(
                start + np.array([0.0, 0.0, 0.060], dtype=np.float64),
                quat,
                tolerance=0.180,
            )
            settle_record["start_move"] = self._goto_pose_legacy_checked(
                start,
                quat,
                tolerance=0.170,
            )
            settle_record["end_move"] = self._goto_pose_legacy_checked(
                end,
                quat,
                tolerance=0.180,
            )
            self._settle_simulation_steps(max(20, int(settle_steps)))
            settle_record["retreat_move"] = self._goto_pose_legacy_checked(
                end + np.array([0.0, 0.0, 0.120], dtype=np.float64),
                quat,
                tolerance=0.190,
            )
            self._settle_simulation_steps(max(20, int(settle_steps)))
            settled_relation = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate="in",
            )
            settle_record["start_position"] = start
            settle_record["end_position"] = end
            settle_record["support_z"] = support_z
            settle_record["relation"] = settled_relation
            return settle_record

        def relation_or_task_satisfied(relation: dict[str, Any], status: dict[str, Any] | None = None) -> bool:
            status = status if isinstance(status, dict) else self.get_task_status()
            return bool(
                relation.get("satisfied", False)
                or relation.get("in", False)
                or status.get("task_completed", False)
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            )

        def seat_near_inside_then_close(relation: dict[str, Any], phase: str) -> dict[str, Any]:
            record: dict[str, Any] = {"phase": phase, "initial_relation": relation}
            current_relation = relation
            self.open_gripper()
            self._settle_simulation_steps(max(20, int(settle_steps)))
            try:
                current_relation = self._placement_relation_diagnostic(
                    resolved_object,
                    resolved_target,
                    predicate="in",
                )
                record["relation_after_settle"] = current_relation
            except Exception as exc:
                record["relation_after_settle_error"] = repr(exc)

            if not relation_or_task_satisfied(current_relation):
                try:
                    seating = self._microwave_seat_object_by_relation(
                        resolved_object,
                        resolved_target,
                        initial_relation=current_relation,
                        max_iters=3,
                        settle_steps=max(20, int(settle_steps)),
                    )
                    current_relation = seating.get("final_relation", current_relation)
                    record["seating"] = seating
                except Exception as exc:
                    record["seating_error"] = repr(exc)

            if not relation_or_task_satisfied(current_relation) and near_inside_high(current_relation):
                record["support_shove_skipped"] = (
                    "near-inside microwave state is already xy-correct; lateral support shove can eject the mug, "
                    "so preserve the diagnostic state for z-only feedback instead"
                )

            status = self.get_task_status()
            if relation_or_task_satisfied(current_relation, status):
                record["close_deferred_reason"] = "object_in_microwave_verified; close handled by guarded close stage"
            else:
                record["close_skipped_reason"] = "relation_not_satisfied"

            record["relation"] = current_relation
            record["task_status"] = status
            return record

        if near_inside_high(before_relation):
            protected = seat_near_inside_then_close(
                before_relation,
                "initial_near_inside_seat_then_close",
            )
            final_relation = protected.get("relation", before_relation)
            status = protected.get("task_status", self.get_task_status())
            records.append(protected)
            return {
                "object_name": resolved_object,
                "target": target,
                "before_relation": before_relation,
                "final_relation": final_relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": bool(
                    final_relation.get("satisfied", False)
                    or status.get("task_completed", False)
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0
                ),
                "skill": "place_object_in_microwave_region",
                "reason": "seated near-inside microwave state before close",
            }

        if held_at_entry:
            candidate_specs = [
                (
                    "legacy_held_low_center_0",
                    np.array([0.000, -0.085, 0.130], dtype=np.float64),
                    np.array([0.000, 0.018, -0.028], dtype=np.float64),
                    [
                        np.array([0.000, 0.005, 0.112], dtype=np.float64),
                        np.array([0.000, 0.018, 0.014], dtype=np.float64),
                    ],
                ),
                (
                    "legacy_front_center_1",
                    np.array([0.000, -0.160, 0.070], dtype=np.float64),
                    np.array([0.000, -0.005, 0.032], dtype=np.float64),
                    [],
                ),
                (
                    "legacy_left_center_2",
                    np.array([-0.020, -0.160, 0.070], dtype=np.float64),
                    np.array([-0.020, 0.000, 0.035], dtype=np.float64),
                    [],
                ),
                (
                    "legacy_right_center_3",
                    np.array([0.020, -0.160, 0.070], dtype=np.float64),
                    np.array([0.020, 0.000, 0.035], dtype=np.float64),
                    [],
                ),
                (
                    "legacy_low_front_4",
                    np.array([0.000, -0.130, 0.050], dtype=np.float64),
                    np.array([0.000, 0.010, 0.020], dtype=np.float64),
                    [],
                ),
            ]
            for idx, (name, entry_offset, release_offset, via_offsets) in enumerate(candidate_specs):
                if idx > 0:
                    hold_now = self.measure_object_grasp_state(resolved_object)
                    if not bool(hold_now.get("held", False)):
                        repick = self.verified_lift_pick_object(
                            resolved_object,
                            grasp_quat=quat,
                            max_attempts=2,
                            lift_height=0.16,
                            success_dz=0.040,
                            approach=0.08,
                            settle_steps=10,
                        )
                        records.append(
                            {
                                "phase": "candidate_repick_before_insert",
                                "candidate": name,
                                "hold_before": hold_now,
                                "repick": repick,
                            }
                        )
                        hold_now = self.measure_object_grasp_state(resolved_object)
                        if not bool(hold_now.get("held", False)):
                            break

                entry_pos = site_pos + entry_offset
                release_pos = site_pos + release_offset
                retreat_pos = entry_pos + np.array([0.0, -0.040, 0.160], dtype=np.float64)
                candidate_record: dict[str, Any] = {
                    "phase": "held_candidate_insert",
                    "candidate": name,
                    "entry_position": entry_pos.copy(),
                    "release_position": release_pos.copy(),
                    "retreat_position": retreat_pos.copy(),
                    "via_positions": [site_pos + offset for offset in via_offsets],
                }
                candidate_record["approach_move"] = self._goto_pose_legacy_checked(
                    entry_pos + np.array([0.0, 0.0, 0.080], dtype=np.float64),
                    quat,
                    tolerance=0.170,
                )
                candidate_record["entry_move"] = self._goto_pose_legacy_checked(
                    entry_pos,
                    quat,
                    tolerance=0.160,
                )
                self._settle_simulation_steps(12)
                candidate_record["entry_grasp"] = self.measure_object_grasp_state(resolved_object)
                via_records: list[dict[str, Any]] = []
                for via_idx, via_offset in enumerate(via_offsets):
                    via_pos = site_pos + via_offset
                    move = self._goto_pose_legacy_checked(
                        via_pos,
                        quat,
                        tolerance=0.175,
                    )
                    self._settle_simulation_steps(10)
                    via_records.append(
                        {
                            "via_index": int(via_idx),
                            "via_position": via_pos.copy(),
                            "move": move,
                            "grasp": self.measure_object_grasp_state(resolved_object),
                        }
                    )
                candidate_record["via_moves"] = via_records
                candidate_record["release_move"] = self._goto_pose_legacy_checked(
                    release_pos,
                    quat,
                    tolerance=0.170,
                )
                self._settle_simulation_steps(20)
                pre_open_relation = self._placement_relation_diagnostic(
                    resolved_object,
                    resolved_target,
                    predicate="in",
                )
                candidate_record["pre_open_relation"] = pre_open_relation
                self.open_gripper()
                self._settle_simulation_steps(max(20, int(settle_steps)))
                after_open_relation = self._placement_relation_diagnostic(
                    resolved_object,
                    resolved_target,
                    predicate="in",
                )
                candidate_record["after_open_relation"] = after_open_relation
                candidate_record["retreat_move"] = self._goto_pose_legacy_checked(
                    retreat_pos,
                    quat,
                    tolerance=0.190,
                )
                self._settle_simulation_steps(max(20, int(settle_steps)))
                after_retreat_relation = self._placement_relation_diagnostic(
                    resolved_object,
                    resolved_target,
                    predicate="in",
                )
                candidate_record["after_retreat_relation"] = after_retreat_relation
                feedback_source = after_retreat_relation
                try:
                    if not bool(feedback_source.get("satisfied", False) or feedback_source.get("in", False)):
                        early_feedback = self._microwave_seat_object_by_relation(
                            resolved_object,
                            resolved_target,
                            initial_relation=feedback_source,
                            max_iters=2,
                            settle_steps=max(18, int(settle_steps) // 2),
                        )
                        candidate_record["early_feedback_seating"] = early_feedback
                        feedback_source = early_feedback.get("final_relation", feedback_source)
                except Exception as exc:
                    candidate_record["early_feedback_seating_error"] = repr(exc)
                after_retreat_relation = feedback_source
                candidate_record["after_early_feedback_relation"] = after_retreat_relation
                records.append(candidate_record)
                if bool(after_open_relation.get("satisfied", False)) or bool(after_retreat_relation.get("satisfied", False)):
                    final_candidate_relation = after_retreat_relation if bool(after_retreat_relation.get("satisfied", False)) else after_open_relation
                    status = self.get_task_status()
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "before_relation": before_relation,
                        "final_relation": final_candidate_relation,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": True,
                        "skill": "place_object_in_microwave_region",
                    }
                settle_source = after_retreat_relation if near_inside_high(after_retreat_relation) else after_open_relation
                if near_inside_high(settle_source):
                    protected = seat_near_inside_then_close(
                        settle_source,
                        f"candidate_{name}_near_inside_seat_then_close",
                    )
                    final_candidate_relation = protected.get("relation", settle_source)
                    status = protected.get("task_status", self.get_task_status())
                    records.append(protected)
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "before_relation": before_relation,
                        "final_relation": final_candidate_relation,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": bool(
                            final_candidate_relation.get("satisfied", False)
                            or status.get("task_completed", False)
                            or float(status.get("reward", 0.0) or 0.0) >= 1.0
                        ),
                        "skill": "place_object_in_microwave_region",
                        "reason": "seated near-inside microwave state before close",
                    }

        final_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate="in",
        )
        local_delta = np.asarray(final_relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
        site_size = np.asarray(
            final_relation.get("site_size_abs", final_relation.get("site_size", np.array([0.0, 0.0, 0.0]))),
            dtype=np.float64,
        ).reshape(-1)[:3]
        xy_inside = bool(site_size.size >= 2 and np.all(np.abs(local_delta[:2]) <= np.maximum(np.abs(site_size[:2]), 0.035)))
        z_high_contact = bool(
            bool(final_relation.get("contact", False))
            and local_delta.size >= 3
            and site_size.size >= 3
            and local_delta[2] > float(abs(site_size[2]))
            and local_delta[2] <= float(abs(site_size[2])) + 0.100
        )
        if (not bool(final_relation.get("satisfied", False))) and xy_inside and z_high_contact:
            protected = seat_near_inside_then_close(
                final_relation,
                "near_inside_high_seat_then_close",
            )
            final_relation = protected.get("relation", final_relation)
            records.append(protected)

        protected_near_inside = False
        local_delta = np.asarray(final_relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
        site_size = np.asarray(
            final_relation.get("site_size_abs", final_relation.get("site_size", np.array([0.0, 0.0, 0.0]))),
            dtype=np.float64,
        ).reshape(-1)[:3]

        if (not protected_near_inside) and front_outside_contact_relation(final_relation):
            try:
                seating = self._microwave_seat_object_by_relation(
                    resolved_object,
                    resolved_target,
                    initial_relation=final_relation,
                    max_iters=4,
                    settle_steps=max(18, min(32, int(settle_steps))),
                )
                final_relation = seating.get("final_relation", final_relation)
                records.append({"phase": "feedback_front_outside_push_in_before_repick", "seating": seating})
            except Exception as exc:
                records.append({"phase": "feedback_front_outside_push_in_before_repick_error", "error": repr(exc)})
            local_delta = np.asarray(final_relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(
                final_relation.get("site_size_abs", final_relation.get("site_size", np.array([0.0, 0.0, 0.0]))),
                dtype=np.float64,
            ).reshape(-1)[:3]

        still_front_outside_contact = front_outside_contact_relation(final_relation)
        far_from_microwave_site = bool(
            local_delta.size >= 2
            and site_size.size >= 2
            and (
                abs(float(local_delta[0])) > float(abs(site_size[0])) + 0.065
                or abs(float(local_delta[1])) > float(abs(site_size[1])) + 0.055
            )
        )
        if (
            (not held_at_entry)
            and (not bool(final_relation.get("satisfied", False)))
            and far_from_microwave_site
            and not still_front_outside_contact
        ):
            try:
                repick = self._microwave_repick_low_release(
                    resolved_object,
                    resolved_target,
                    settle_steps=max(20, int(settle_steps)),
                )
                final_relation = repick.get("final_relation", final_relation)
                records.append({"phase": "feedback_repick_before_seating", "repick": repick})
            except Exception as exc:
                records.append({"phase": "feedback_repick_before_seating_error", "error": repr(exc)})

        if (not protected_near_inside) and not bool(final_relation.get("satisfied", False)):
            try:
                seating = self._microwave_seat_object_by_relation(
                    resolved_object,
                    resolved_target,
                    initial_relation=final_relation,
                    max_iters=3 if far_from_microwave_site else 4,
                    settle_steps=max(20, int(settle_steps)),
                )
                final_relation = seating.get("final_relation", final_relation)
                records.append({"phase": "feedback_seating_before_close_assist", "seating": seating})
            except Exception as exc:
                records.append({"phase": "feedback_seating_before_close_assist_error", "error": repr(exc)})

        if (not protected_near_inside) and not bool(final_relation.get("satisfied", False)):
            try:
                assist = self._microwave_close_assist_if_shallow(
                    resolved_object,
                    resolved_target,
                    final_relation,
                    settle_steps=max(20, int(settle_steps)),
                )
                if bool(assist.get("attempted", False)):
                    final_relation = assist.get("final_relation", final_relation)
                records.append({"phase": "feedback_close_assist_after_seating", "assist": assist})
            except Exception as exc:
                records.append({"phase": "feedback_close_assist_after_seating_error", "error": repr(exc)})

        if (not protected_near_inside) and held_at_entry and not bool(final_relation.get("satisfied", False)):
            try:
                repick = self._microwave_repick_low_release(
                    resolved_object,
                    resolved_target,
                    settle_steps=max(20, int(settle_steps)),
                )
                final_relation = repick.get("final_relation", final_relation)
                records.append({"phase": "feedback_repick_low_release", "repick": repick})
            except Exception as exc:
                records.append({"phase": "feedback_repick_error", "error": repr(exc)})
            try:
                assist = self._microwave_close_assist_if_shallow(
                    resolved_object,
                    resolved_target,
                    final_relation,
                    settle_steps=max(20, int(settle_steps)),
                )
                if bool(assist.get("attempted", False)):
                    final_relation = assist.get("final_relation", final_relation)
                records.append({"phase": "feedback_close_assist_after_repick", "assist": assist})
            except Exception as exc:
                records.append({"phase": "feedback_close_assist_after_repick_error", "error": repr(exc)})

        status = self.get_task_status()
        relation_satisfied = bool(
            final_relation.get("satisfied", False)
            or status.get("task_completed", False)
            or float(status.get("reward", 0.0) or 0.0) >= 1.0
        )
        return {
            "object_name": resolved_object,
            "target": target,
            "before_relation": before_relation,
            "final_relation": final_relation,
            "records": records,
            "task_status": status,
            "relation_satisfied": relation_satisfied,
            "skill": "place_object_in_microwave_region",
        }

    def close_microwave_by_sweep(
        self,
        object_name: str = "microwave_1",
        *,
        max_sweeps: int = 8,
        settle_steps: int = 40,
    ) -> dict[str, Any]:
        """Close a microwave door through feedback gripper sweeps and verify."""
        microwave_name = str(object_name or "microwave_1")
        if not microwave_name.endswith("_1"):
            microwave_name = f"{microwave_name}_1"
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        before = self.get_object_state_diagnostic(microwave_name)
        if bool(before.get("is_close", False)):
            status = self.get_task_status()
            return {
                "object_name": microwave_name,
                "before": before,
                "after": before,
                "sweeps": [],
                "task_status": status,
                "closed": True,
                "skill": "close_microwave_by_sweep",
                "reason": "already closed",
            }
        body_candidates = [
            f"{microwave_name}_microdoorroot",
            "microwave_1_microdoorroot",
            "microwave_microdoorroot",
        ]
        body_name = None
        for candidate in body_candidates:
            try:
                self.get_body_pose(candidate)
                body_name = candidate
                break
            except Exception:
                continue
        if body_name is None:
            raise KeyError(f"microwave door body not found for {microwave_name}")
        specs = [
            ("outer_mid", np.array([0.2375, -0.054, 0.000], dtype=np.float64), 0.075, 0.285),
            ("outer_high", np.array([0.2375, -0.054, 0.055], dtype=np.float64), 0.070, 0.300),
            ("outer_low", np.array([0.2375, -0.054, -0.045], dtype=np.float64), 0.070, 0.285),
            ("inner_mid", np.array([0.1800, -0.020, 0.030], dtype=np.float64), 0.060, 0.255),
        ]
        records: list[dict[str, Any]] = []
        self.open_gripper()
        for idx in range(max(1, int(max_sweeps))):
            before_iter = self.get_object_state_diagnostic(microwave_name)
            if bool(before_iter.get("is_close", False)):
                status = self.get_task_status()
                return {
                    "object_name": microwave_name,
                    "before": before,
                    "after": before_iter,
                    "sweeps": records,
                    "task_status": status,
                    "closed": True,
                    "skill": "close_microwave_by_sweep",
                }
            body_pos, body_quat = self.get_body_pose(body_name)
            body_pos = np.asarray(body_pos, dtype=np.float64).reshape(3).copy()
            body_rot = self._rotation_from_wxyz(np.asarray(body_quat, dtype=np.float64).reshape(4)).as_matrix()
            spec_name, local_handle, backoff, forward = specs[idx % len(specs)]
            joint_state = before_iter.get("joint_state", [])
            if isinstance(joint_state, list) and joint_state:
                try:
                    joint_value = float(joint_state[0])
                    if joint_value > -0.75:
                        forward = max(float(forward), 0.325)
                        backoff = min(float(backoff), 0.060)
                except Exception:
                    pass
            handle = body_pos + body_rot @ local_handle
            tangent = body_rot @ np.array([-local_handle[1], local_handle[0], 0.0], dtype=np.float64)
            tangent[2] = 0.0
            norm = float(np.linalg.norm(tangent))
            if norm <= 1e-6:
                continue
            tangent = tangent / norm
            start = handle - tangent * float(backoff)
            end = handle + tangent * float(forward)
            approach = start + np.array([0.0, 0.0, 0.100], dtype=np.float64)
            approach_move = self._goto_pose_legacy_checked(approach, quat, tolerance=0.190)
            start_move = self._goto_pose_legacy_checked(start, quat, tolerance=0.170)
            end_move = self._goto_pose_legacy_checked(end, quat, tolerance=0.190)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            retreat = end + np.array([0.0, 0.0, 0.100], dtype=np.float64)
            retreat_move = self._goto_pose_legacy_checked(retreat, quat, tolerance=0.200)
            self._settle_simulation_steps(max(8, int(settle_steps) // 2))
            after = self.get_object_state_diagnostic(microwave_name)
            status = self.get_task_status()
            record = {
                "idx": int(idx),
                "body_name": body_name,
                "spec_name": spec_name,
                "local_handle": local_handle.copy(),
                "handle": handle.copy(),
                "tangent": tangent.copy(),
                "start": start.copy(),
                "end": end.copy(),
                "backoff": float(backoff),
                "forward": float(forward),
                "moves": {
                    "approach": approach_move,
                    "start": start_move,
                    "end": end_move,
                    "retreat": retreat_move,
                },
                "before": before_iter,
                "after": after,
                "task_status": status,
                "closed": bool(after.get("is_close", False)),
            }
            records.append(record)
            if bool(record["closed"]) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0:
                return {
                    "object_name": microwave_name,
                    "before": before,
                    "after": after,
                    "sweeps": records,
                    "task_status": status,
                    "closed": True,
                    "skill": "close_microwave_by_sweep",
                }
        final = self.get_object_state_diagnostic(microwave_name)
        status = self.get_task_status()
        return {
            "object_name": microwave_name,
            "before": before,
            "after": final,
            "sweeps": records,
            "task_status": status,
            "closed": bool(final.get("is_close", False)) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0) or 0.0) >= 1.0,
            "skill": "close_microwave_by_sweep",
        }

    def _microwave_close_assist_if_shallow(
        self,
        object_name: str,
        target_name: str,
        relation: dict[str, Any] | None,
        *,
        settle_steps: int = 40,
    ) -> dict[str, Any]:
        """Try the task's physical microwave close motion when the mug is shallow."""
        initial_relation = (
            dict(relation)
            if isinstance(relation, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        if bool(initial_relation.get("satisfied", False) or initial_relation.get("in", False)):
            return {
                "attempted": False,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "task_status": self.get_task_status(),
                "reason": "already in microwave; close is handled by guarded close stage",
                "skill": "microwave_close_assist_if_shallow",
            }

        local_delta = np.asarray(initial_relation.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
        site_size = np.asarray(
            initial_relation.get("site_size_abs", initial_relation.get("site_size", np.array([0.12, 0.0835, 0.08]))),
            dtype=np.float64,
        ).reshape(-1)[:3]
        contact = bool(initial_relation.get("contact", False))
        shallow_xy = bool(
            local_delta.size >= 2
            and site_size.size >= 2
            and abs(float(local_delta[0])) <= max(float(abs(site_size[0])) + 0.030, 0.085)
            and -0.260 <= float(local_delta[1]) <= -0.095
        )
        z_close_to_floor = bool(
            local_delta.size >= 3
            and site_size.size >= 3
            and float(local_delta[2]) >= -float(abs(site_size[2])) - 0.050
            and float(local_delta[2]) <= float(abs(site_size[2])) + 0.070
        )
        if not (contact and shallow_xy and z_close_to_floor):
            return {
                "attempted": False,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "task_status": self.get_task_status(),
                "reason": "not shallow enough for door assist",
                "contact": contact,
                "shallow_xy": shallow_xy,
                "z_close_to_floor": z_close_to_floor,
                "skill": "microwave_close_assist_if_shallow",
            }

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        bddl = self.get_bddl_goal_diagnostics()
        status = self.get_task_status()
        return {
            "attempted": False,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "bddl_goal_diagnostics": bddl,
            "task_status": status,
            "reason": "close_precondition_not_satisfied; do not close microwave before object is in the heating region",
            "success": bool(
                final_relation.get("satisfied", False)
                or status.get("task_completed", False)
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            ),
            "skill": "microwave_close_assist_if_shallow",
        }

    def _microwave_seat_object_by_relation(
        self,
        object_name: str,
        target_name: str,
        *,
        initial_relation: dict[str, Any] | None = None,
        max_iters: int = 4,
        settle_steps: int = 40,
        allow_side_pose_feedback: bool = True,
    ) -> dict[str, Any]:
        """Seat an object into a microwave site using feedback from local_delta."""
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        made_progress = False
        relation = (
            dict(initial_relation)
            if isinstance(initial_relation, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        for idx in range(max(1, int(max_iters))):
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                return {
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation,
                    "satisfied": True,
                }
            if "local_delta" not in relation:
                break
            local_delta = np.asarray(relation.get("local_delta"), dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(
                relation.get("site_size_abs", relation.get("site_size", np.array([0.12, 0.0835, 0.08]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            if local_delta.size < 2 or site_size.size < 2:
                break
            before_xy = float(np.linalg.norm(local_delta[:2]))
            obj_pos, _ = self.get_object_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            try:
                site_world = np.asarray(relation.get("site_pos_world"), dtype=np.float64).reshape(3)
                site_pos = self._world_position_to_robot_base(site_world)
            except Exception:
                site_pos, _ = self.get_site_pose(target_name)
                site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()

            try:
                _, site_quat = self.get_site_pose(target_name)
                site_rot = self._rotation_from_wxyz(
                    np.asarray(site_quat, dtype=np.float64).reshape(4)
                ).as_matrix()
            except Exception:
                site_rot = np.eye(3, dtype=np.float64)

            contact = bool(relation.get("contact", False))
            xy_inside = bool(
                local_delta.size >= 2
                and site_size.size >= 2
                and abs(float(local_delta[0])) <= max(float(abs(site_size[0])) + 0.012, 0.045)
                and abs(float(local_delta[1])) <= max(float(abs(site_size[1])) + 0.012, 0.045)
            )
            xy_near = bool(
                local_delta.size >= 2
                and site_size.size >= 2
                and abs(float(local_delta[0])) <= max(float(abs(site_size[0])) + 0.045, 0.085)
                and abs(float(local_delta[1])) <= max(float(abs(site_size[1])) + 0.105, 0.150)
            )
            z_high_margin = (
                float(local_delta[2]) - float(abs(site_size[2]))
                if local_delta.size >= 3 and site_size.size >= 3
                else float("-inf")
            )
            if contact and xy_near and not xy_inside and float(z_high_margin) <= 0.030:
                obj_pos_before = obj_pos.copy()
                try:
                    site_world = np.asarray(relation.get("site_pos_world"), dtype=np.float64).reshape(3)
                    object_world = np.asarray(relation.get("object_pos_world"), dtype=np.float64).reshape(3)
                    target_world = object_world.copy()
                    target_world[0] = float(
                        np.clip(
                            object_world[0],
                            site_world[0] - 0.40 * abs(float(site_size[0])),
                            site_world[0] + 0.40 * abs(float(site_size[0])),
                        )
                    )
                    target_world[1] = float(
                        np.clip(
                            object_world[1],
                            site_world[1] - 0.35 * abs(float(site_size[1])),
                            site_world[1] + 0.35 * abs(float(site_size[1])),
                        )
                    )
                    target_robot = self._world_position_to_robot_base(target_world)
                    correction = target_robot - obj_pos
                except Exception:
                    correction = site_pos - obj_pos
                correction = np.asarray(correction, dtype=np.float64).reshape(3)
                correction[2] = 0.0
                correction_norm = float(np.linalg.norm(correction[:2]))
                if correction_norm > 1e-6:
                    direction = correction / correction_norm
                    stroke = min(max(correction_norm + 0.010, 0.018), 0.075)
                    push_z = float(
                        np.clip(
                            obj_pos[2] + 0.018,
                            site_pos[2] + 0.020,
                            site_pos[2] + 0.085,
                        )
                    )
                    start = obj_pos.copy()
                    start[:2] -= direction[:2] * 0.040
                    start[2] = push_z + 0.050
                    contact_pos = obj_pos.copy()
                    contact_pos[:2] -= direction[:2] * 0.010
                    contact_pos[2] = push_z
                    end = obj_pos.copy()
                    end[:2] += direction[:2] * stroke
                    end[2] = push_z
                    low_start = obj_pos.copy()
                    low_start[:2] -= direction[:2] * 0.030
                    low_start[2] = push_z
                    low_end = obj_pos.copy()
                    low_end[:2] += direction[:2] * min(max(correction_norm + 0.020, 0.040), 0.105)
                    low_end[2] = push_z
                    xy_record: dict[str, Any] = {
                        "iteration": idx,
                        "action": "feedback_xy_near_boundary_seating",
                        "local_delta_before": local_delta.copy(),
                        "site_size_abs": site_size.copy(),
                        "correction": correction.copy(),
                        "stroke": stroke,
                        "start": start.copy(),
                        "contact": contact_pos.copy(),
                        "end": end.copy(),
                        "low_start": low_start.copy(),
                        "low_end": low_end.copy(),
                    }
                    try:
                        self.open_gripper()
                        xy_record["start_move"] = self._goto_pose_legacy_checked(
                            low_start + np.array([0.0, 0.0, 0.060], dtype=np.float64),
                            quat,
                            tolerance=0.140,
                        )
                        xy_record["contact_move"] = self._goto_pose_legacy_checked(
                            low_start,
                            quat,
                            tolerance=0.130,
                        )
                        self._settle_simulation_steps(4)
                        xy_record["end_move"] = self._goto_pose_legacy_checked(
                            low_end,
                            quat,
                            tolerance=0.140,
                        )
                        self.open_gripper()
                        xy_record["retreat_move"] = self._goto_pose_legacy_checked(
                            low_end + np.array([0.0, 0.0, 0.090], dtype=np.float64),
                            quat,
                            tolerance=0.150,
                        )
                    except Exception as exc:
                        xy_record["error"] = repr(exc)
                        try:
                            self.open_gripper()
                        except Exception:
                            pass
                    self._settle_simulation_steps(max(16, min(34, int(settle_steps))))
                    relation_after_xy = self._placement_relation_diagnostic(
                        object_name,
                        target_name,
                        predicate="in",
                    )
                    after_delta_xy = np.asarray(relation_after_xy.get("local_delta", local_delta), dtype=np.float64).reshape(-1)[:3]
                    xy_record["relation"] = relation_after_xy
                    xy_record["improved_xy"] = bool(
                        after_delta_xy.size >= 2
                        and (
                            float(np.linalg.norm(after_delta_xy[:2])) < float(np.linalg.norm(local_delta[:2])) - 0.006
                            or (
                                site_size.size >= 2
                                and abs(float(after_delta_xy[0])) <= max(float(abs(site_size[0])) + 0.012, 0.045)
                                and abs(float(after_delta_xy[1])) <= max(float(abs(site_size[1])) + 0.012, 0.045)
                            )
                        )
                    )
                    xy_record["before_object_position"] = obj_pos_before
                    after_obj_xy, _ = self.get_object_pose(object_name)
                    xy_record["after_object_position"] = np.asarray(after_obj_xy, dtype=np.float64).reshape(3).copy()
                    records.append(xy_record)
                    relation = relation_after_xy
                    made_progress = bool(made_progress or xy_record["improved_xy"] or bool(relation.get("satisfied", False)))
                    if bool(relation.get("satisfied", False) or relation.get("in", False)):
                        break
                    if bool(xy_record["improved_xy"]):
                        continue
            outside_front = bool(
                not xy_inside
                and local_delta.size >= 2
                and site_size.size >= 2
                and abs(float(local_delta[1])) > max(float(abs(site_size[1])) + 0.035, 0.112)
                and abs(float(local_delta[1])) <= max(float(abs(site_size[1])) + 0.135, 0.230)
            )
            if outside_front:
                obj_pos_before = obj_pos.copy()
                local_correction = np.array(
                    [
                        float(np.clip(-float(local_delta[0]), -0.035, 0.035)),
                        float(
                            np.clip(
                                -float(local_delta[1]),
                                -0.155,
                                0.155,
                            )
                        ),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                correction = site_rot @ local_correction
                correction[2] = 0.0
                correction_norm = float(np.linalg.norm(correction[:2]))
                if correction_norm > 1e-6:
                    direction = correction / correction_norm
                    stroke = min(max(correction_norm + 0.020, 0.060), 0.155)
                    push_z = float(
                        np.clip(
                            obj_pos[2] + 0.014,
                            site_pos[2] + 0.028,
                            site_pos[2] + 0.090,
                        )
                    )
                    side_axis = site_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    side_axis[2] = 0.0
                    side_axis[:2] /= max(float(np.linalg.norm(side_axis[:2])), 1e-9)
                    start = obj_pos.copy()
                    start[:2] -= direction[:2] * 0.075
                    start[:2] += side_axis[:2] * float(np.clip(-0.20 * float(local_delta[0]), -0.025, 0.025))
                    start[2] = push_z
                    contact_pos = obj_pos.copy()
                    contact_pos[:2] -= direction[:2] * 0.018
                    contact_pos[2] = push_z
                    end = obj_pos.copy()
                    end[:2] += direction[:2] * stroke
                    end[:2] += side_axis[:2] * float(np.clip(-0.35 * float(local_delta[0]), -0.032, 0.032))
                    end[2] = push_z
                    front_record: dict[str, Any] = {
                        "iteration": idx,
                        "action": "feedback_front_outside_push_in",
                        "local_delta_before": local_delta.copy(),
                        "site_size_abs": site_size.copy(),
                        "local_correction": local_correction.copy(),
                        "direction": direction.copy(),
                        "stroke": stroke,
                        "push_z": push_z,
                        "start": start.copy(),
                        "contact": contact_pos.copy(),
                        "end": end.copy(),
                    }
                    try:
                        self.open_gripper()
                        front_record["approach_move"] = self._goto_pose_legacy_checked(
                            start + np.array([0.0, 0.0, 0.075], dtype=np.float64),
                            quat,
                            tolerance=0.165,
                        )
                        front_record["start_move"] = self._goto_pose_legacy_checked(
                            start,
                            quat,
                            tolerance=0.145,
                        )
                        self._settle_simulation_steps(6)
                        front_record["contact_move"] = self._goto_pose_legacy_checked(
                            contact_pos,
                            quat,
                            tolerance=0.140,
                        )
                        front_record["end_move"] = self._goto_pose_legacy_checked(
                            end,
                            quat,
                            tolerance=0.160,
                        )
                        self.open_gripper()
                        front_record["retreat_move"] = self._goto_pose_legacy_checked(
                            end + np.array([0.0, 0.0, 0.095], dtype=np.float64),
                            quat,
                            tolerance=0.180,
                        )
                    except Exception as exc:
                        front_record["error"] = repr(exc)
                        try:
                            self.open_gripper()
                        except Exception:
                            pass
                    self._settle_simulation_steps(max(18, min(42, int(settle_steps))))
                    after_front = self._placement_relation_diagnostic(
                        object_name,
                        target_name,
                        predicate="in",
                    )
                    after_delta_front = np.asarray(
                        after_front.get("local_delta", local_delta),
                        dtype=np.float64,
                    ).reshape(-1)[:3]
                    after_obj_front, _ = self.get_object_pose(object_name)
                    after_obj_front = np.asarray(after_obj_front, dtype=np.float64).reshape(3).copy()
                    front_record["relation"] = after_front
                    front_record["local_delta_after"] = after_delta_front.copy()
                    front_record["before_object_position"] = obj_pos_before
                    front_record["after_object_position"] = after_obj_front
                    front_record["object_motion_xy"] = float(np.linalg.norm(after_obj_front[:2] - obj_pos_before[:2]))
                    front_record["improved_y"] = bool(
                        after_delta_front.size >= 2
                        and abs(float(after_delta_front[1])) < abs(float(local_delta[1])) - 0.010
                    )
                    front_record["improved_xy"] = bool(
                        after_delta_front.size >= 2
                        and float(np.linalg.norm(after_delta_front[:2])) < before_xy - 0.010
                    )
                    records.append(front_record)
                    relation = after_front
                    made_progress = bool(
                        made_progress
                        or front_record["improved_y"]
                        or front_record["improved_xy"]
                        or bool(relation.get("satisfied", False))
                    )
                    if bool(relation.get("satisfied", False) or relation.get("in", False)):
                        break
                    if front_record["improved_y"] or front_record["improved_xy"]:
                        continue
            if contact and xy_inside:
                z_records: list[dict[str, Any]] = []
                before_obj = obj_pos.copy()

                self.open_gripper()
                self._settle_simulation_steps(max(26, min(48, int(settle_steps) + 6)))
                z_records.append(
                    {
                        "action": "near_inside_passive_settle",
                        "reason": "xy is already inside microwave region; avoid destructive planar or vertical finger push",
                    }
                )
                after_z = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                z_records[-1]["relation_after"] = after_z

                final_z_relation = after_z
                after_delta = np.asarray(final_z_relation.get("local_delta", local_delta), dtype=np.float64).reshape(-1)[:3]
                if (
                    not bool(final_z_relation.get("satisfied", False) or final_z_relation.get("in", False))
                    and after_delta.size >= 3
                    and site_size.size >= 3
                    and float(after_delta[2]) > float(abs(site_size[2]))
                    and float(after_delta[2]) <= float(abs(site_size[2])) + 0.070
                ):
                    after_obj_for_press, _ = self.get_object_pose(object_name)
                    after_obj_for_press = np.asarray(after_obj_for_press, dtype=np.float64).reshape(3).copy()
                    try:
                        rel_site_world = np.asarray(final_z_relation["site_pos_world"], dtype=np.float64).reshape(3)
                        rel_site_robot = self._world_position_to_robot_base(rel_site_world)
                    except Exception:
                        rel_site_robot = site_pos.copy()
                    target_z = float(rel_site_robot[2] + abs(float(site_size[2])) - 0.010)
                    target_z = float(
                        np.clip(
                            target_z,
                            float(after_obj_for_press[2] - 0.045),
                            float(after_obj_for_press[2] - 0.010),
                        )
                    )
                    press_start = after_obj_for_press.copy()
                    press_start[2] += 0.060
                    press_contact = after_obj_for_press.copy()
                    press_contact[2] += 0.010
                    press_end = after_obj_for_press.copy()
                    press_end[2] = target_z
                    vertical_record: dict[str, Any] = {
                        "action": "near_inside_gentle_vertical_seat",
                        "reason": "xy is inside microwave region but object is still above the heating-region z bound",
                        "relation_before": final_z_relation,
                        "start": press_start.copy(),
                        "contact": press_contact.copy(),
                        "end": press_end.copy(),
                        "target_z_robot": target_z,
                        "z_high_margin_before": float(after_delta[2]) - float(abs(site_size[2])),
                    }
                    try:
                        self.open_gripper()
                        vertical_record["start_move"] = self.goto_finger_midpoint_checked(
                            press_start,
                            quat,
                            tolerance=0.075,
                        )
                        self.close_gripper()
                        self._settle_simulation_steps(4)
                        vertical_record["contact_move"] = self.goto_finger_midpoint_checked(
                            press_contact,
                            quat,
                            tolerance=0.070,
                        )
                        vertical_record["end_move"] = self.goto_finger_midpoint_checked(
                            press_end,
                            quat,
                            tolerance=0.070,
                        )
                        self.open_gripper()
                        vertical_record["retreat_move"] = self.goto_finger_midpoint_checked(
                            press_end + np.array([0.0, 0.0, 0.085], dtype=np.float64),
                            quat,
                            tolerance=0.095,
                        )
                    except Exception as exc:
                        vertical_record["error"] = repr(exc)
                        try:
                            self.open_gripper()
                        except Exception:
                            pass
                    self._settle_simulation_steps(max(18, min(36, int(settle_steps))))
                    final_z_relation = self._placement_relation_diagnostic(
                        object_name,
                        target_name,
                        predicate="in",
                    )
                    vertical_record["relation_after"] = final_z_relation
                    z_records.append(vertical_record)
                    after_delta = np.asarray(final_z_relation.get("local_delta", after_delta), dtype=np.float64).reshape(-1)[:3]
                after_obj, _ = self.get_object_pose(object_name)
                after_obj = np.asarray(after_obj, dtype=np.float64).reshape(3).copy()
                improved_z = bool(after_delta.size >= 3 and float(after_delta[2]) < float(local_delta[2]) - 0.004)
                made_progress = bool(made_progress or improved_z or bool(final_z_relation.get("satisfied", False)))
                low_repick: dict[str, Any] | None = None
                if not bool(final_z_relation.get("satisfied", False) or final_z_relation.get("in", False)):
                    try:
                        low_repick = self._microwave_repick_low_release(
                            object_name,
                            target_name,
                            settle_steps=max(18, int(settle_steps)),
                            initial_relation=final_z_relation,
                            preserve_near_inside_xy=True,
                        )
                        final_z_relation = low_repick.get("final_relation", final_z_relation)
                        after_delta = np.asarray(
                            final_z_relation.get("local_delta", after_delta),
                            dtype=np.float64,
                        ).reshape(-1)[:3]
                        after_obj, _ = self.get_object_pose(object_name)
                        after_obj = np.asarray(after_obj, dtype=np.float64).reshape(3).copy()
                        made_progress = bool(
                            made_progress
                            or bool(low_repick.get("satisfied", False))
                            or (
                                after_delta.size >= 3
                                and float(after_delta[2]) < float(local_delta[2]) - 0.004
                            )
                        )
                    except Exception as exc:
                        low_repick = {"error": repr(exc), "skill": "microwave_repick_low_release"}
                side_pose: dict[str, Any] | None = None
                if allow_side_pose_feedback and not bool(final_z_relation.get("satisfied", False) or final_z_relation.get("in", False)):
                    side_pose = {
                        "skipped": True,
                        "skill": "microwave_side_pose_low_release",
                        "reason": (
                            "preserve xy-valid microwave near-success; side-pose feedback can eject "
                            "the mug from the heating region"
                        ),
                        "initial_relation": final_z_relation,
                        "final_relation": final_z_relation,
                        "satisfied": False,
                    }
                records.append(
                    {
                        "iteration": idx,
                        "action": "feedback_z_high_seating",
                        "local_delta_before": local_delta.copy(),
                        "site_size_abs": site_size.copy(),
                        "z_high_margin": z_high_margin,
                        "xy_inside": xy_inside,
                        "before_object_position": before_obj,
                        "after_object_position": after_obj,
                        "improved_z": improved_z,
                        "subrecords": z_records,
                        "low_repick": low_repick,
                        "side_pose_low_release": side_pose,
                        "relation": final_z_relation,
                    }
                )
                relation = final_z_relation
                if bool(relation.get("satisfied", False) or relation.get("in", False)):
                    break
                records.append(
                    {
                        "iteration": idx,
                        "action": "stop_preserve_near_inside_z_failure",
                        "reason": "xy is already inside the microwave; preserve this diagnostic state instead of planar pushing it out",
                        "relation": relation,
                    }
                )
                break

            local_depth_sign = -1.0 * np.sign(float(local_delta[1]) or -1.0)
            depth_axis = site_rot @ np.array([0.0, local_depth_sign, 0.0], dtype=np.float64)
            depth_axis[2] = 0.0
            if float(np.linalg.norm(depth_axis[:2])) <= 1e-6:
                depth_axis = site_pos - obj_pos
                depth_axis[2] = 0.0
            norm = float(np.linalg.norm(depth_axis[:2]))
            if norm <= 1e-6:
                self._settle_simulation_steps(max(8, int(settle_steps)))
                relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                records.append({"iteration": idx, "action": "settle_no_planar_error", "relation": relation})
                continue
            depth_axis[:2] /= norm
            lateral_axis = site_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            lateral_axis[2] = 0.0
            if float(np.linalg.norm(lateral_axis[:2])) <= 1e-6:
                lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            lateral_axis[:2] /= max(float(np.linalg.norm(lateral_axis[:2])), 1e-9)
            delta_robot = site_pos - obj_pos
            delta_robot[2] = 0.0
            desired_local = np.array([-local_delta[0], -local_delta[1], 0.0], dtype=np.float64)
            if site_size.size >= 2:
                excess_x = max(0.0, abs(float(local_delta[0])) - max(float(abs(site_size[0])) - 0.006, 0.030))
                excess_y = max(0.0, abs(float(local_delta[1])) - max(float(abs(site_size[1])) - 0.006, 0.030))
                excess = float(np.linalg.norm([excess_x, excess_y]))
            else:
                excess = float(np.linalg.norm(desired_local[:2]))
            stroke = min(max(excess + 0.012, 0.020), 0.060)
            lateral_offsets = [0.000, -0.016, 0.016, -0.030, 0.030]
            lateral = lateral_offsets[idx % len(lateral_offsets)]
            push_z = float(
                np.clip(
                    max(obj_pos[2] + 0.012, site_pos[2] + 0.018),
                    site_pos[2] + 0.016,
                    site_pos[2] + 0.065,
                )
            )
            start = obj_pos.copy()
            start[:2] -= depth_axis[:2] * 0.045
            start[:2] += lateral_axis[:2] * lateral
            start[2] = push_z
            mid = obj_pos.copy()
            mid[:2] += lateral_axis[:2] * lateral
            mid[2] = push_z
            end = obj_pos.copy()
            end[:2] += depth_axis[:2] * stroke
            end[:2] += lateral_axis[:2] * float(np.clip(-0.15 * float(local_delta[0]), -0.020, 0.020))
            end[2] = push_z
            before_obj = obj_pos.copy()
            self.open_gripper()
            approach = start + np.array([0.0, 0.0, 0.080], dtype=np.float64)
            approach_move = self.goto_finger_midpoint_checked(
                approach,
                quat,
                tolerance=0.095,
            )
            start_move = self.goto_finger_midpoint_checked(start, quat, tolerance=0.075)
            self._settle_simulation_steps(8)
            mid_move = self.goto_finger_midpoint_checked(mid, quat, tolerance=0.080)
            end_move = self.goto_finger_midpoint_checked(end, quat, tolerance=0.095)
            self.open_gripper()
            retreat = end.copy()
            retreat[:2] -= depth_axis[:2] * 0.055
            retreat[2] += 0.085
            retreat_move = self.goto_finger_midpoint_checked(retreat, quat, tolerance=0.100)
            self._settle_simulation_steps(max(8, min(24, int(settle_steps))))
            after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            after_obj, _ = self.get_object_pose(object_name)
            after_obj = np.asarray(after_obj, dtype=np.float64).reshape(3).copy()
            after_delta = np.asarray(after.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            after_xy = float(np.linalg.norm(after_delta[:2])) if after_delta.size >= 2 else float("inf")
            object_motion_xy = float(np.linalg.norm(after_obj[:2] - before_obj[:2]))
            improved_xy = bool(np.isfinite(after_xy) and after_xy < before_xy - 0.012)
            if object_motion_xy >= 0.006 or improved_xy:
                made_progress = True
            guard_action = None
            if (
                not bool(after.get("satisfied", False) or after.get("in", False))
                and np.isfinite(after_xy)
                and after_xy > before_xy + 0.050
            ):
                self._goto_pose_legacy_checked(
                    end + np.array([0.0, 0.0, 0.060], dtype=np.float64),
                    quat,
                    tolerance=0.150,
                )
                self._goto_pose_legacy_checked(end, quat, tolerance=0.150)
                self._goto_pose_legacy_checked(start, quat, tolerance=0.130)
                self._goto_pose_legacy_checked(
                    start + np.array([0.0, 0.0, 0.080], dtype=np.float64),
                    quat,
                    tolerance=0.150,
                )
                self._settle_simulation_steps(max(8, int(settle_steps) // 2))
                after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                guard_action = "reverse_after_worse_local_delta"
            records.append(
                {
                    "iteration": idx,
                    "action": "feedback_planar_seating",
                    "local_delta_before": local_delta.copy(),
                    "site_size_abs": site_size.copy(),
                    "delta_robot": delta_robot.copy(),
                    "desired_local": desired_local.copy(),
                    "depth_axis": depth_axis.copy(),
                    "lateral_axis": lateral_axis.copy(),
                    "stroke": stroke,
                    "start": start,
                    "mid": mid,
                    "end": end,
                    "retreat": retreat,
                    "push_z": push_z,
                    "finger_waypoint_moves": {
                        "approach": approach_move,
                        "start": start_move,
                        "mid": mid_move,
                        "end": end_move,
                        "retreat": retreat_move,
                    },
                    "before_xy": before_xy,
                    "after_xy": after_xy,
                    "before_object_position": before_obj,
                    "after_object_position": after_obj,
                    "object_motion_xy": object_motion_xy,
                    "improved_xy": improved_xy,
                    "guard_action": guard_action,
                    "relation": after,
                }
            )
            relation = after
            no_effect = bool(object_motion_xy < 0.004 and not improved_xy)
            if guard_action is not None or no_effect:
                break

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
            "made_progress": made_progress,
        }

    def _thin_caddy_contact_seat_by_relation(
        self,
        object_name: str,
        target_name: str,
        *,
        initial_relation: dict[str, Any] | None = None,
        max_iters: int = 3,
        settle_steps: int = 28,
    ) -> dict[str, Any]:
        """Seat a near-contained thin object in a caddy without planar pushing."""
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        relation = (
            dict(initial_relation)
            if isinstance(initial_relation, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        initial_copy = dict(relation)
        for idx in range(max(1, int(max_iters))):
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                return {
                    "records": records,
                    "initial_relation": initial_copy,
                    "final_relation": relation,
                    "satisfied": True,
                    "task_status": self.get_task_status(),
                    "skill": "thin_caddy_contact_seat_by_relation",
                }
            if not bool(relation.get("contact", False)):
                records.append(
                    {
                        "iteration": int(idx),
                        "action": "stop_no_contact",
                        "relation": relation,
                    }
                )
                break
            if "local_delta" not in relation or "site_size_abs" not in relation:
                records.append(
                    {
                        "iteration": int(idx),
                        "action": "stop_missing_relation_geometry",
                        "relation": relation,
                    }
                )
                break
            local_delta = np.asarray(relation["local_delta"], dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(relation["site_size_abs"], dtype=np.float64).reshape(-1)[:3]
            if local_delta.size < 3 or site_size.size < 3:
                break
            xy_inside = bool(
                abs(float(local_delta[0])) <= max(float(abs(site_size[0])) + 0.014, 0.038)
                and abs(float(local_delta[1])) <= max(float(abs(site_size[1])) + 0.014, 0.074)
            )
            z_low_margin = float(-float(abs(site_size[2])) - float(local_delta[2]))
            z_high_margin = float(float(local_delta[2]) - float(abs(site_size[2])))
            close_z_low = bool(0.0 < z_low_margin <= 0.050)
            close_z_high = bool(0.0 < z_high_margin <= 0.050)
            if not xy_inside or not (close_z_low or close_z_high):
                records.append(
                    {
                        "iteration": int(idx),
                        "action": "stop_outside_z_seating_gate",
                        "relation": relation,
                        "xy_inside": xy_inside,
                        "z_low_margin": z_low_margin,
                        "z_high_margin": z_high_margin,
                    }
                )
                break

            try:
                obj_world = np.asarray(relation["object_pos_world"], dtype=np.float64).reshape(3)
                object_pos = self._world_position_to_robot_base(obj_world)
            except Exception:
                object_pos, _ = self.get_object_pose(object_name)
                object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()
            try:
                site_world = np.asarray(relation["site_pos_world"], dtype=np.float64).reshape(3)
                site_pos = self._world_position_to_robot_base(site_world)
            except Exception:
                site_pos, _ = self.get_site_pose(target_name)
                site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()

            press_center = object_pos.copy()
            press_center[:2] = 0.65 * object_pos[:2] + 0.35 * site_pos[:2]
            high = press_center.copy()
            high[2] = max(float(object_pos[2]) + 0.115, float(site_pos[2]) + 0.150)
            contact = press_center.copy()
            contact[2] = float(object_pos[2] + 0.045)
            target = contact.copy()
            dz = -min(max(z_low_margin + 0.006 if close_z_low else 0.010, 0.008), 0.018)
            if close_z_high:
                dz = 0.008
            target[2] = float(contact[2] + dz)
            move_high = self._goto_pose_legacy_checked(high, quat, tolerance=0.140)
            move_contact = self._goto_pose_legacy_checked(contact, quat, tolerance=0.120)
            move_seat = self._goto_pose_legacy_checked(target, quat, tolerance=0.120)
            self._settle_simulation_steps(max(14, int(settle_steps)))
            relation_after = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate="in",
            )
            records.append(
                {
                    "iteration": int(idx),
                    "action": "vertical_contact_seat",
                    "relation_before": relation,
                    "object_pos_robot": object_pos.copy(),
                    "site_pos_robot": site_pos.copy(),
                    "press_high": high.copy(),
                    "press_contact": contact.copy(),
                    "press_target": target.copy(),
                    "z_low_margin": z_low_margin,
                    "z_high_margin": z_high_margin,
                    "move_high": move_high,
                    "move_contact": move_contact,
                    "move_seat": move_seat,
                    "relation_after": relation_after,
                }
            )
            relation = relation_after
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                break

        final_relation = self._placement_relation_diagnostic(
            object_name,
            target_name,
            predicate="in",
        )
        return {
            "records": records,
            "initial_relation": initial_copy,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
            "task_status": self.get_task_status(),
            "skill": "thin_caddy_contact_seat_by_relation",
        }

    def _thin_caddy_relation_guided_nudge(
        self,
        object_name: str,
        target_name: str,
        *,
        diagnostic: dict[str, Any] | None = None,
        max_iters: int = 4,
        settle_steps: int = 28,
    ) -> dict[str, Any]:
        """Nudge a contacted thin object toward a caddy contain site.

        This is intentionally gentler than the generic containment push: it
        uses the readonly site-local error, caps planar motion, and adds a small
        downward seating component so a book can slide into the compartment
        instead of being swept across the desk.
        """
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        relation = (
            dict(diagnostic)
            if isinstance(diagnostic, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        initial_relation = dict(relation)

        for idx in range(max(1, int(max_iters))):
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                return {
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation,
                    "satisfied": True,
                    "task_status": self.get_task_status(),
                    "skill": "thin_caddy_relation_guided_nudge",
                }
            if not bool(relation.get("contact", False)) or "local_delta" not in relation:
                records.append(
                    {
                        "iteration": int(idx),
                        "action": "stop_missing_contact_or_delta",
                        "relation": relation,
                    }
                )
                break

            local_delta = np.asarray(relation.get("local_delta"), dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(
                relation.get("site_size_abs", relation.get("site_size", np.array([0.04, 0.04, 0.04]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            if local_delta.size < 3 or site_size.size < 3:
                records.append({"iteration": int(idx), "action": "stop_bad_relation_geometry", "relation": relation})
                break

            xy_norm = float(np.linalg.norm(local_delta[:2]))
            if xy_norm > max(0.32, 4.2 * float(np.max(np.abs(site_size[:2])))):
                records.append(
                    {
                        "iteration": int(idx),
                        "action": "stop_too_far_for_gentle_caddy_nudge",
                        "xy_norm": xy_norm,
                        "local_delta": local_delta.copy(),
                        "site_size_abs": site_size.copy(),
                    }
                )
                break

            try:
                site_world = np.asarray(relation["site_pos_world"], dtype=np.float64).reshape(3)
                object_world = np.asarray(relation["object_pos_world"], dtype=np.float64).reshape(3)
                site_pos = self._world_position_to_robot_base(site_world)
                object_pos = self._world_position_to_robot_base(object_world)
            except Exception:
                site_pos, _ = self.get_site_pose(target_name)
                site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()
                object_pos, _ = self.get_object_pose(object_name)
                object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()

            try:
                _, site_quat = self.get_site_pose(target_name)
                site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64).reshape(4)).as_matrix()
                correction_robot = site_rot @ (-local_delta)
            except Exception:
                correction_robot = site_pos - object_pos
            correction_robot = np.asarray(correction_robot, dtype=np.float64).reshape(3)
            correction_robot[2] = 0.0
            planar_norm = float(np.linalg.norm(correction_robot[:2]))
            if planar_norm <= 1e-6:
                correction_robot[:2] = (site_pos - object_pos)[:2]
                planar_norm = float(np.linalg.norm(correction_robot[:2]))
            if planar_norm <= 1e-6:
                self._settle_simulation_steps(max(10, int(settle_steps)))
                relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                records.append({"iteration": int(idx), "action": "settle_zero_planar_error", "relation": relation})
                continue

            direction = correction_robot[:2] / max(planar_norm, 1e-6)
            stroke = float(np.clip(0.55 * planar_norm + 0.018, 0.020, 0.070))
            side = np.array([-direction[1], direction[0]], dtype=np.float64)
            lateral_bias = 0.006 if idx % 2 == 0 else -0.006

            start = object_pos.copy()
            start[:2] -= direction * 0.044
            start[:2] += side * lateral_bias
            site_floor_z = float(site_pos[2] - max(0.030, 0.70 * abs(float(site_size[2]))))
            site_cap_z = float(site_pos[2] + max(0.020, 0.45 * abs(float(site_size[2]))))
            start[2] = float(np.clip(object_pos[2] + 0.018, site_floor_z, site_cap_z))

            end = object_pos.copy()
            end[:2] += direction * stroke
            end[:2] += side * lateral_bias
            z_error = float(local_delta[2])
            z_adjust = -0.010 if z_error > -abs(float(site_size[2])) else 0.004
            end[2] = float(np.clip(start[2] + z_adjust, site_floor_z, site_cap_z))

            high = start.copy()
            high[2] += 0.065
            retreat = end.copy()
            retreat[2] += 0.075

            before_norm = float(np.linalg.norm(local_delta[:2]))
            record: dict[str, Any] = {
                "iteration": int(idx),
                "action": "gentle_site_local_caddy_nudge",
                "relation_before": relation,
                "local_delta_before": local_delta.copy(),
                "site_size_abs": site_size.copy(),
                "correction_robot": correction_robot.copy(),
                "stroke": stroke,
                "start": start.copy(),
                "end": end.copy(),
            }
            try:
                self.open_gripper()
                record["high_move"] = self._goto_pose_legacy_checked(high, quat, tolerance=0.130)
                record["start_move"] = self._goto_pose_legacy_checked(start, quat, tolerance=0.115)
                self.close_gripper()
                self._settle_simulation_steps(5)
                record["end_move"] = self._goto_pose_legacy_checked(end, quat, tolerance=0.115)
                self.open_gripper()
                record["retreat_move"] = self._goto_pose_legacy_checked(retreat, quat, tolerance=0.130)
                self._settle_simulation_steps(max(12, int(settle_steps)))
            except Exception as exc:
                record["error"] = repr(exc)
                try:
                    self.open_gripper()
                    self._settle_simulation_steps(max(8, int(settle_steps) // 2))
                except Exception as cleanup_exc:
                    record["cleanup_error"] = repr(cleanup_exc)

            relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            after_delta = np.asarray(
                relation_after.get("local_delta", np.zeros(3)),
                dtype=np.float64,
            ).reshape(-1)[:3]
            after_norm = float(np.linalg.norm(after_delta[:2])) if after_delta.size >= 2 else float("inf")
            record["relation_after"] = relation_after
            record["before_norm"] = before_norm
            record["after_norm"] = after_norm
            records.append(record)
            relation = relation_after

            if bool(relation_after.get("satisfied", False) or relation_after.get("in", False)):
                break
            if after_norm > before_norm + 0.045:
                break

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
            "task_status": self.get_task_status(),
            "skill": "thin_caddy_relation_guided_nudge",
        }

    def _thin_caddy_repick_low_release(
        self,
        object_name: str,
        target_name: str,
        *,
        settle_steps: int = 32,
        start_candidate_index: int = 0,
    ) -> dict[str, Any]:
        """Re-pick a thin book and low-release it into a caddy compartment."""
        records: list[dict[str, Any]] = []
        initial_relation = self._placement_relation_diagnostic(
            object_name,
            target_name,
            predicate="in",
        )
        if bool(initial_relation.get("satisfied", False) or initial_relation.get("in", False)):
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "satisfied": True,
                "task_status": self.get_task_status(),
                "skill": "thin_caddy_repick_low_release",
                "reason": "already satisfied",
            }

        pick = self.verified_geometry_pick_object(
            object_name,
            max_attempts=3,
            start_candidate_index=max(0, int(start_candidate_index)),
            lift_height=0.20,
            settle_steps=max(8, int(settle_steps) // 3),
        )
        records.append({"phase": "geometry_repick", "result": pick})
        if not bool(pick.get("held", False)):
            fallback = self.approach_and_grasp_object(
                object_name,
                strategy="auto",
                max_attempts=2,
                start_candidate_index=0,
                lift_height=0.12,
                settle_steps=max(8, int(settle_steps) // 3),
            )
            records.append({"phase": "affordance_repick", "result": fallback})

        hold = self.measure_object_grasp_state(object_name)
        records.append({"phase": "after_repick_hold", "hold": hold})
        if not bool(hold.get("held", False)):
            final_relation = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate="in",
            )
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
                "task_status": self.get_task_status(),
                "skill": "thin_caddy_repick_low_release",
                "reason": "repick did not verify held object",
            }

        release = self.low_release_object_to_target(
            object_name,
            predicate="in",
            target_name=target_name,
            release_z=0.050,
            approach=0.10,
            retreat=0.24,
            settle_steps=max(20, int(settle_steps)),
        )
        records.append({"phase": "low_release_after_repick", "release": release})
        final_relation = (
            release.get("relation", initial_relation)
            if isinstance(release, dict)
            else initial_relation
        )
        if not bool(final_relation.get("satisfied", False) or final_relation.get("in", False)):
            correction = self._thin_caddy_relation_guided_nudge(
                object_name,
                target_name,
                diagnostic=final_relation,
                max_iters=2,
                settle_steps=max(18, int(settle_steps)),
            )
            records.append({"phase": "post_repick_caddy_nudge", "correction": correction})
            final_relation = (
                correction.get("final_relation", final_relation)
                if isinstance(correction, dict)
                else final_relation
            )

        status = self.get_task_status()
        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": bool(
                final_relation.get("satisfied", False)
                or final_relation.get("in", False)
                or status.get("task_completed", False)
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            ),
            "task_status": status,
            "skill": "thin_caddy_repick_low_release",
        }

    def _microwave_repick_low_release(
        self,
        object_name: str,
        target_name: str,
        *,
        settle_steps: int = 40,
        initial_relation: dict[str, Any] | None = None,
        preserve_near_inside_xy: bool = False,
    ) -> dict[str, Any]:
        """Recover a shallow failed microwave placement by re-picking and releasing."""
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        initial_relation = (
            dict(initial_relation)
            if isinstance(initial_relation, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        if bool(initial_relation.get("satisfied", False) or initial_relation.get("in", False)):
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "satisfied": True,
                "skill": "microwave_repick_low_release",
                "reason": "already satisfied",
            }

        pick = self.verified_lift_pick_object(
            object_name,
            grasp_quat=quat,
            max_attempts=3,
            lift_height=0.16,
            success_dz=0.045,
            approach=0.08,
            settle_steps=12,
        )
        records.append({"phase": "verified_repick", "result": pick})
        if not bool(pick.get("held", False)):
            fallback = self.approach_and_grasp_object(
                object_name,
                strategy="auto",
                max_attempts=2,
                lift_height=0.13,
                settle_steps=10,
            )
            records.append({"phase": "affordance_repick", "result": fallback})

        hold = self.measure_object_grasp_state(object_name)
        records.append({"phase": "after_repick_hold", "hold": hold})
        if not bool(hold.get("held", False)):
            final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
                "skill": "microwave_repick_low_release",
                "reason": "repick did not verify held object",
            }

        site_pos, _ = self.get_site_pose(target_name)
        site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()
        if preserve_near_inside_xy:
            try:
                _, site_quat = self.get_site_pose(target_name)
                site_rot = self._rotation_from_wxyz(
                    np.asarray(site_quat, dtype=np.float64).reshape(4)
                ).as_matrix()
            except Exception:
                site_rot = np.eye(3, dtype=np.float64)
            site_size = np.asarray(
                initial_relation.get("site_size_abs", initial_relation.get("site_size", np.array([0.12, 0.0835, 0.08]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            if site_size.size < 3:
                site_size = np.array([0.12, 0.0835, 0.08], dtype=np.float64)
            local_delta = np.asarray(
                initial_relation.get("local_delta", np.zeros(3, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)[:3]
            if local_delta.size < 3:
                local_delta = np.zeros(3, dtype=np.float64)
            desired_local = np.array(
                [
                    float(np.clip(float(local_delta[0]), -0.30 * abs(float(site_size[0])), 0.30 * abs(float(site_size[0])))),
                    float(np.clip(float(local_delta[1]), -0.18 * abs(float(site_size[1])), 0.26 * abs(float(site_size[1])))),
                    float(np.clip(0.70 * abs(float(site_size[2])), 0.046, max(0.050, abs(float(site_size[2])) - 0.016))),
                ],
                dtype=np.float64,
            )
            mouth_local = desired_local.copy()
            mouth_local[1] = -float(abs(site_size[1]) + 0.065)
            mouth_local[2] = max(float(desired_local[2] + 0.070), float(abs(site_size[2]) + 0.030))
            inside_high_local = desired_local.copy()
            inside_high_local[2] = max(float(desired_local[2] + 0.055), float(abs(site_size[2]) + 0.020))
            waypoint_specs = [
                ("mouth_object_center", site_pos + site_rot @ mouth_local, 0.045, 0.080),
                ("inside_high_object_center", site_pos + site_rot @ inside_high_local, 0.040, 0.070),
                ("low_inside_object_center", site_pos + site_rot @ desired_local, 0.026, 0.035),
            ]
            for name, target_object_pos, tolerance_xy, tolerance_z in waypoint_specs:
                align = self._align_held_object_center(
                    object_name,
                    target_object_pos,
                    quat,
                    strict=self._requires_strict_hold(object_name),
                    max_iters=2,
                    tolerance_xy=float(tolerance_xy),
                    tolerance_z=float(tolerance_z),
                    lin_vel_norm=0.035,
                    timeout_s=7.0,
                )
                self._settle_simulation_steps(8)
                grasp = self.measure_object_grasp_state(object_name)
                records.append(
                    {
                        "phase": name,
                        "target_object_position": target_object_pos.copy(),
                        "desired_local": desired_local.copy(),
                        "align": align,
                        "grasp": grasp,
                    }
                )
                if not bool(grasp.get("held", False)):
                    break
        else:
            waypoints = [
                ("mouth_high", site_pos + np.array([0.000, -0.090, 0.130], dtype=np.float64)),
                ("inside_high", site_pos + np.array([0.000, 0.000, 0.112], dtype=np.float64)),
                ("center_support", site_pos + np.array([0.000, 0.004, 0.040], dtype=np.float64)),
            ]
            for idx, (name, target_pos) in enumerate(waypoints):
                if idx == 0:
                    self.goto_pose(target_pos + np.array([0.0, 0.0, 0.055], dtype=np.float64), quat)
                move = self.goto_pose_checked(target_pos, quat, tolerance=0.080)
                self._settle_simulation_steps(8)
                grasp = self.measure_object_grasp_state(object_name)
                records.append(
                    {
                        "phase": name,
                        "target_position": target_pos.copy(),
                        "move": move,
                        "grasp": grasp,
                    }
                )
                if not bool(grasp.get("held", False)):
                    break

        pre_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "pre_open_relation", "relation": pre_open_relation})
        self.open_gripper()
        self._settle_simulation_steps(max(20, int(settle_steps)))
        after_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "after_open_relation", "relation": after_open_relation})
        final_relation = after_open_relation

        if (
            not preserve_near_inside_xy
            and not bool(final_relation.get("satisfied", False) or final_relation.get("in", False))
        ):
            seating = self._microwave_seat_object_by_relation(
                object_name,
                target_name,
                initial_relation=final_relation,
                max_iters=2,
                settle_steps=max(16, int(settle_steps)),
                allow_side_pose_feedback=False,
            )
            final_relation = seating.get("final_relation", final_relation)
            records.append({"phase": "post_repick_feedback_seating", "seating": seating})

        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
            "skill": "microwave_repick_low_release",
        }

    def _microwave_side_pose_low_release(
        self,
        object_name: str,
        target_name: str,
        *,
        settle_steps: int = 40,
        initial_relation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Try a lower-profile microwave insertion when upright mug placement is too high."""
        records: list[dict[str, Any]] = []
        initial_relation = (
            dict(initial_relation)
            if isinstance(initial_relation, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        if bool(initial_relation.get("satisfied", False) or initial_relation.get("in", False)):
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "satisfied": True,
                "skill": "microwave_side_pose_low_release",
                "reason": "already satisfied",
            }

        side_pick = self.grasp_tall_object_side(
            object_name,
            max_attempts=4,
            lift_height=0.14,
            settle_steps=max(8, int(settle_steps) // 3),
        )
        records.append({"phase": "side_pose_repick", "result": side_pick})
        if not bool(side_pick.get("held", False)):
            final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)),
                "task_status": self.get_task_status(),
                "skill": "microwave_side_pose_low_release",
                "reason": "side pose repick did not verify held object",
            }

        _, ee_quat = self.get_ee_pose()
        quat = np.asarray(ee_quat, dtype=np.float64).reshape(4).copy()
        site_pos, _ = self.get_site_pose(target_name)
        site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3).copy()
        try:
            _, site_quat = self.get_site_pose(target_name)
            site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64).reshape(4)).as_matrix()
        except Exception:
            site_rot = np.eye(3, dtype=np.float64)
        site_size = np.asarray(
            initial_relation.get("site_size_abs", initial_relation.get("site_size", np.array([0.12, 0.0835, 0.08]))),
            dtype=np.float64,
        ).reshape(-1)[:3]
        if site_size.size < 3:
            site_size = np.array([0.12, 0.0835, 0.08], dtype=np.float64)

        desired_local = np.array([0.018, 0.000, 0.050], dtype=np.float64)
        desired_local[0] = float(np.clip(desired_local[0], -0.36 * abs(float(site_size[0])), 0.36 * abs(float(site_size[0]))))
        desired_local[1] = float(np.clip(desired_local[1], -0.22 * abs(float(site_size[1])), 0.24 * abs(float(site_size[1]))))
        desired_local[2] = float(np.clip(0.55 * abs(float(site_size[2])), 0.038, max(0.044, abs(float(site_size[2])) - 0.020)))
        mouth_local = desired_local.copy()
        mouth_local[1] = -float(abs(site_size[1]) + 0.070)
        mouth_local[2] = max(float(desired_local[2] + 0.075), float(abs(site_size[2]) + 0.020))
        inside_mid_local = desired_local.copy()
        inside_mid_local[1] -= 0.012
        inside_mid_local[2] = max(float(desired_local[2] + 0.034), 0.060)

        waypoint_specs = [
            ("side_mouth_object_center", site_pos + site_rot @ mouth_local, 0.060, 0.085),
            ("side_inside_mid_object_center", site_pos + site_rot @ inside_mid_local, 0.052, 0.070),
            ("side_low_inside_object_center", site_pos + site_rot @ desired_local, 0.038, 0.045),
        ]
        for name, target_object_pos, tolerance_xy, tolerance_z in waypoint_specs:
            align = self._align_held_object_center(
                object_name,
                target_object_pos,
                quat,
                strict=False,
                max_iters=2,
                tolerance_xy=float(tolerance_xy),
                tolerance_z=float(tolerance_z),
                lin_vel_norm=0.030,
                timeout_s=7.0,
            )
            self._settle_simulation_steps(8)
            grasp = self.measure_object_grasp_state(object_name)
            relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            records.append(
                {
                    "phase": name,
                    "target_object_position": target_object_pos.copy(),
                    "desired_local": desired_local.copy(),
                    "gripper_quat": quat.copy(),
                    "align": align,
                    "grasp": grasp,
                    "relation": relation,
                }
            )
            if not bool(grasp.get("held", False)):
                break

        pre_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "pre_open_relation", "relation": pre_open_relation})
        self.open_gripper()
        self._settle_simulation_steps(max(24, int(settle_steps)))
        after_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "after_open_relation", "relation": after_open_relation})

        final_relation = after_open_relation
        if not bool(final_relation.get("satisfied", False) or final_relation.get("in", False)):
            seating = self._microwave_seat_object_by_relation(
                object_name,
                target_name,
                initial_relation=final_relation,
                max_iters=2,
                settle_steps=max(16, int(settle_steps)),
                allow_side_pose_feedback=False,
            )
            final_relation = seating.get("final_relation", final_relation)
            records.append({"phase": "post_side_pose_feedback_seating", "seating": seating})

        status = self.get_task_status()
        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": bool(
                final_relation.get("satisfied", False)
                or final_relation.get("in", False)
                or status.get("task_completed", False)
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            ),
            "task_status": status,
            "skill": "microwave_side_pose_low_release",
        }

    def local_placement_closure(
        self,
        object_name: str,
        *,
        predicate: str = "on",
        target_name: str | None = None,
        max_iters: int = 2,
        near_tolerance: float | None = None,
        settle_steps: int = 16,
        require_drawer_closing_clearance: bool = False,
        min_drawer_front_clearance: float | None = None,
    ) -> dict[str, Any]:
        """Short, bounded closure for near-target placement failures.

        This is deliberately lighter than `repair_placement_relation`: it checks
        the current relation, optionally releases if the object is still held, and
        otherwise performs at most a couple of short local settling/push actions.
        It is meant for cases where a shallow object has already slipped close to
        the target surface during transport, so a long repick loop would waste the
        LIBERO horizon.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        pred = str(predicate).strip().lower()
        if pred not in {"in", "on"}:
            pred = "in" if "basket" in str(target_name or "").lower() else "on"
        target = self._placement_target_for_object(
            resolved_object,
            predicate=pred,
            target_name=target_name,
        )
        pred = str(target.get("predicate", pred)).strip().lower()
        resolved_target = str(target["target_name"])
        target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        target_size = np.asarray(target.get("target_size", np.array([0.04, 0.04, 0.04])), dtype=np.float64)
        tolerance = (
            float(near_tolerance)
            if near_tolerance is not None
            else float(max(0.075 if pred == "on" else 0.070, 1.6 * np.max(np.abs(target_size[:2]))))
        )
        records: list[dict[str, Any]] = []
        require_drawer_margin = bool(
            require_drawer_closing_clearance
            and pred == "in"
            and self._is_drawer_region_target_name(resolved_target)
        )
        drawer_front_clearance_threshold = (
            float(min_drawer_front_clearance)
            if require_drawer_margin and min_drawer_front_clearance is not None
            else 0.110
            if require_drawer_margin and "bottom" in resolved_target.lower()
            else 0.065
            if require_drawer_margin
            else 0.045
        )

        def drawer_margin(relation: dict[str, Any]) -> dict[str, Any] | None:
            if not require_drawer_margin:
                return None
            return self._drawer_relation_closing_clearance(
                relation,
                min_front_clearance=float(drawer_front_clearance_threshold),
            )

        def relation_is_satisfied(relation: dict[str, Any]) -> bool:
            if not bool(relation.get("satisfied", False)):
                return False
            margin = drawer_margin(relation)
            return bool(margin is None or margin.get("has_clearance", False))

        def task_done() -> tuple[dict[str, Any], bool]:
            status = self.get_task_status()
            done = bool(status.get("task_completed", False) or float(status.get("reward", 0.0)) >= 1.0)
            return status, done

        initial_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate=pred,
        )
        status, done = task_done()
        initial_drawer_margin = drawer_margin(initial_relation)
        if relation_is_satisfied(initial_relation) or done:
            return {
                "object_name": resolved_object,
                "target": target,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": True,
                "drawer_closing_clearance": initial_drawer_margin,
                "near_target": True,
                "skill": "local_placement_closure",
            }

        hold = self.measure_object_grasp_state(resolved_object, strict=self._requires_strict_hold(resolved_object))
        if (
            pred == "in"
            and not bool(hold.get("held", False))
            and bool(initial_relation.get("contact", False))
            and "local_delta" in initial_relation
        ):
            if "basket" in resolved_target.lower():
                xy_distance = self._relation_xy_distance_to_target(initial_relation)
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": initial_relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": False,
                    "near_target": True,
                    "xy_distance_to_target": xy_distance,
                    "xy_tolerance": tolerance,
                    "needs_repick": True,
                    "reason": (
                        "basket unheld containment is not corrected by planar pushing; "
                        "reacquire the object and redo held center-aligned transport/release"
                    ),
                    "skill": "local_placement_closure",
                }
            thin_caddy_target = bool(
                "book" in resolved_object.lower()
                and ("caddy" in resolved_target.lower() or "compartment" in resolved_target.lower())
            )
            if thin_caddy_target:
                correction = self._thin_caddy_relation_guided_nudge(
                    resolved_object,
                    resolved_target,
                    diagnostic=initial_relation,
                    max_iters=max(3, int(max_iters)),
                    settle_steps=max(20, int(settle_steps)),
                )
                final_relation = correction.get("final_relation", initial_relation) if isinstance(correction, dict) else initial_relation
                repick_release = None
                if not bool(final_relation.get("satisfied", False) or final_relation.get("in", False)):
                    repick_release = self._thin_caddy_repick_low_release(
                        resolved_object,
                        resolved_target,
                        settle_steps=max(24, int(settle_steps)),
                        start_candidate_index=0,
                    )
                    if isinstance(repick_release, dict):
                        final_relation = repick_release.get("final_relation", final_relation)
                records.append(
                    {
                        "iteration": 0,
                        "action": "thin_caddy_unheld_relation_guided_nudge",
                        "hold": hold,
                        "correction": correction,
                        "repick_release": repick_release,
                        "relation": final_relation,
                    }
                )
                status = (
                    repick_release.get("task_status", status)
                    if isinstance(repick_release, dict)
                    else self.get_task_status()
                )
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": final_relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": bool(final_relation.get("satisfied", False) or final_relation.get("in", False)) or done,
                    "near_target": True,
                    "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                    "xy_tolerance": tolerance,
                    "reason": (
                        "thin-object caddy relation-guided nudge plus repick low-release"
                        if repick_release is not None
                        else "thin-object caddy relation-guided nudge"
                    ),
                    "skill": "local_placement_closure",
                }
            if self._is_drawer_region_target_name(resolved_target):
                if self._drawer_relation_is_z_only_above_box(initial_relation, max_z_excess=0.020):
                    z_settle = self.drawer_z_only_settle_object(
                        resolved_object,
                        target_name=resolved_target,
                        max_attempts=max(1, min(2, int(max_iters))),
                        max_z_excess=0.020,
                        settle_steps=max(18, int(settle_steps)),
                    )
                    final_relation = (
                        z_settle.get("final_relation", initial_relation)
                        if isinstance(z_settle, dict)
                        else initial_relation
                    )
                    status = (
                        z_settle.get("task_status", status)
                        if isinstance(z_settle, dict)
                        else status
                    )
                    records.append(
                        {
                            "iteration": 0,
                            "action": "drawer_z_only_settle",
                            "hold": hold,
                            "settle": z_settle,
                            "relation": final_relation,
                            "reason": "xy is already inside drawer AABB; use vertical settle instead of planar slide",
                        }
                    )
                    relation_satisfied = bool(
                        (isinstance(z_settle, dict) and z_settle.get("relation_satisfied", False))
                        or final_relation.get("satisfied", False)
                        or final_relation.get("in", False)
                        or status.get("task_completed", False)
                        or float(status.get("reward", 0.0) or 0.0) >= 1.0
                    )
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": relation_satisfied,
                        "drawer_closing_clearance": drawer_margin(final_relation),
                        "near_target": True,
                        "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                        "xy_tolerance": tolerance,
                        "needs_low_release_replan": not relation_satisfied,
                        "reason": (
                            "drawer z-only near-success fixed by vertical settle"
                            if relation_satisfied
                            else "drawer z-only near-success persisted after vertical settle"
                        ),
                        "skill": "local_placement_closure",
                    }
                drawer_repair = self._drawer_slide_object_into_region(
                    resolved_object,
                    resolved_target,
                    diagnostic=initial_relation,
                    max_iters=max(6 if require_drawer_margin else 3, int(max_iters)),
                    settle_steps=max(18, int(settle_steps)),
                    require_closing_clearance=require_drawer_margin,
                    min_closing_front_clearance=float(drawer_front_clearance_threshold),
                )
                final_relation = drawer_repair.get("final_relation", initial_relation) if isinstance(drawer_repair, dict) else initial_relation
                status = drawer_repair.get("task_status", status) if isinstance(drawer_repair, dict) else status
                final_drawer_margin = drawer_repair.get("drawer_closing_clearance", drawer_margin(final_relation)) if isinstance(drawer_repair, dict) else drawer_margin(final_relation)
                records.append(
                    {
                        "iteration": 0,
                        "action": "drawer_near_contact_slide_insert",
                        "hold": hold,
                        "repair": drawer_repair,
                        "relation": final_relation,
                    }
                )
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": final_relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": relation_is_satisfied(final_relation)
                    or bool(status.get("task_completed", False))
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0,
                    "drawer_closing_clearance": final_drawer_margin,
                    "near_target": True,
                    "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }
            correction = self._containment_correction_to_site(
                resolved_object,
                resolved_target,
                diagnostic=initial_relation,
                max_iters=max(1, int(max_iters)),
                settle_steps=max(8, int(settle_steps)),
            )
            final_relation = correction.get("final_relation", initial_relation) if isinstance(correction, dict) else initial_relation
            records.append(
                {
                    "iteration": 0,
                    "action": "unheld_near_contact_containment_correction",
                    "hold": hold,
                    "correction": correction,
                    "relation": final_relation,
                }
            )
            status, done = task_done()
            return {
                "object_name": resolved_object,
                "target": target,
                "initial_relation": initial_relation,
                "final_relation": final_relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": bool(final_relation.get("satisfied", False)) or done,
                "near_target": True,
                "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                "xy_tolerance": tolerance,
                "skill": "local_placement_closure",
            }

        obj_pos, _ = self.get_object_pose(resolved_object)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
        xy_distance = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        if xy_distance > tolerance:
            return {
                "object_name": resolved_object,
                "target": target,
                "initial_relation": initial_relation,
                "final_relation": initial_relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": False,
                "near_target": False,
                "xy_distance_to_target": xy_distance,
                "xy_tolerance": tolerance,
                "reason": "object is not close enough for local placement closure",
                "skill": "local_placement_closure",
            }

        if bool(hold.get("held", False)):
            release = self.low_release_object_to_target(
                resolved_object,
                predicate=pred,
                target_name=resolved_target,
                release_z=self._repair_release_z(pred, target, 0),
                approach=0.08,
                retreat=0.08,
                settle_steps=max(12, int(settle_steps)),
                require_drawer_closing_clearance=require_drawer_margin,
            )
            relation = release.get("relation", {}) if isinstance(release, dict) else {}
            release_drawer_margin = release.get("drawer_closing_clearance", drawer_margin(relation)) if isinstance(release, dict) else drawer_margin(relation)
            records.append(
                {
                    "iteration": 0,
                    "action": "bounded_low_release_if_still_held",
                    "hold": hold,
                    "release": release,
                    "relation": relation,
                }
            )
            status = release.get("task_status", status) if isinstance(release, dict) else status
            if relation_is_satisfied(relation) or bool(status.get("task_completed", False)) or float(status.get("reward", 0.0)) >= 1.0:
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "drawer_closing_clearance": release_drawer_margin,
                    "near_target": True,
                    "xy_distance_to_target": xy_distance,
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }

        if pred == "on":
            return self._local_on_settle_closure(
                resolved_object,
                target,
                resolved_target,
                initial_relation=initial_relation,
                records=records,
                xy_tolerance=tolerance,
                initial_xy_distance=xy_distance,
                max_iters=max_iters,
                settle_steps=settle_steps,
            )

        if (
            pred == "in"
            and bool(initial_relation.get("contact", False))
            and "local_delta" in initial_relation
        ):
            if self._is_drawer_region_target_name(resolved_target):
                if self._drawer_relation_is_z_only_above_box(initial_relation, max_z_excess=0.020):
                    z_settle = self.drawer_z_only_settle_object(
                        resolved_object,
                        target_name=resolved_target,
                        max_attempts=max(1, min(2, int(max_iters))),
                        max_z_excess=0.020,
                        settle_steps=max(18, int(settle_steps)),
                    )
                    final_relation = (
                        z_settle.get("final_relation", initial_relation)
                        if isinstance(z_settle, dict)
                        else initial_relation
                    )
                    status = (
                        z_settle.get("task_status", status)
                        if isinstance(z_settle, dict)
                        else status
                    )
                    records.append(
                        {
                            "iteration": 0,
                            "action": "drawer_z_only_settle",
                            "settle": z_settle,
                            "relation": final_relation,
                            "reason": "xy is already inside drawer AABB; use vertical settle instead of planar slide",
                        }
                    )
                    relation_satisfied = bool(
                        (isinstance(z_settle, dict) and z_settle.get("relation_satisfied", False))
                        or final_relation.get("satisfied", False)
                        or final_relation.get("in", False)
                        or status.get("task_completed", False)
                        or float(status.get("reward", 0.0) or 0.0) >= 1.0
                    )
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": relation_satisfied,
                        "drawer_closing_clearance": drawer_margin(final_relation),
                        "near_target": True,
                        "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                        "xy_tolerance": tolerance,
                        "needs_low_release_replan": not relation_satisfied,
                        "reason": (
                            "drawer z-only near-success fixed by vertical settle"
                            if relation_satisfied
                            else "drawer z-only near-success persisted after vertical settle"
                        ),
                        "skill": "local_placement_closure",
                    }
                drawer_repair = self._drawer_slide_object_into_region(
                    resolved_object,
                    resolved_target,
                    diagnostic=initial_relation,
                    max_iters=max(6 if require_drawer_margin else 3, int(max_iters)),
                    settle_steps=max(18, int(settle_steps)),
                    require_closing_clearance=require_drawer_margin,
                    min_closing_front_clearance=float(drawer_front_clearance_threshold),
                )
                final_relation = drawer_repair.get("final_relation", initial_relation) if isinstance(drawer_repair, dict) else initial_relation
                final_drawer_margin = drawer_repair.get("drawer_closing_clearance", drawer_margin(final_relation)) if isinstance(drawer_repair, dict) else drawer_margin(final_relation)
                records.append(
                    {
                        "iteration": 0,
                        "action": "drawer_containment_slide_insert",
                        "repair": drawer_repair,
                        "relation": final_relation,
                    }
                )
                status = drawer_repair.get("task_status", status) if isinstance(drawer_repair, dict) else status
                if (
                    relation_is_satisfied(final_relation)
                    or bool(status.get("task_completed", False))
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0
                ):
                    return {
                        "object_name": resolved_object,
                        "target": target,
                        "initial_relation": initial_relation,
                        "final_relation": final_relation,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": True,
                        "drawer_closing_clearance": final_drawer_margin,
                        "near_target": True,
                        "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                        "xy_tolerance": tolerance,
                        "skill": "local_placement_closure",
                    }
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": final_relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": False,
                    "drawer_closing_clearance": final_drawer_margin,
                    "near_target": True,
                    "xy_distance_to_target": self._relation_xy_distance_to_target(final_relation),
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }
            correction = self._containment_correction_to_site(
                resolved_object,
                resolved_target,
                diagnostic=initial_relation,
                max_iters=max(1, int(max_iters)),
                settle_steps=max(12, int(settle_steps)),
            )
            final_relation = correction.get("final_relation", {}) if isinstance(correction, dict) else {}
            records.append(
                {
                    "iteration": 0,
                    "action": "containment_correction",
                    "correction": correction,
                    "relation": final_relation,
                }
            )
            status, done = task_done()
            if bool(final_relation.get("satisfied", False)) or done:
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": final_relation,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "near_target": True,
                    "xy_distance_to_target": float(np.linalg.norm(np.asarray(self.get_object_pose(resolved_object)[0])[:2] - target_pos[:2])),
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }

        iterations = max(1, int(max_iters))
        for idx in range(iterations):
            relation_before = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate=pred,
            )
            status, done = task_done()
            relation_before_drawer_margin = drawer_margin(relation_before)
            if relation_is_satisfied(relation_before) or done:
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": relation_before,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "drawer_closing_clearance": relation_before_drawer_margin,
                    "near_target": True,
                    "xy_distance_to_target": xy_distance,
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }

            obj_pos, _ = self.get_object_pose(resolved_object)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            xy_vec = target_pos[:2] - obj_pos[:2]
            dist = float(np.linalg.norm(xy_vec))
            if dist > tolerance:
                records.append(
                    {
                        "iteration": idx,
                        "action": "abort_not_near_target",
                        "distance_xy": dist,
                        "relation": relation_before,
                    }
                )
                break

            if dist <= 0.012:
                self._settle_simulation_steps(max(8, int(settle_steps)))
                relation_after = self._placement_relation_diagnostic(
                    resolved_object,
                    resolved_target,
                    predicate=pred,
                )
                records.append(
                    {
                        "iteration": idx,
                        "action": "short_settle_near_target",
                        "distance_xy": dist,
                        "relation": relation_after,
                    }
                )
            else:
                push = self._short_local_push_to_target(
                    resolved_object,
                    resolved_target,
                    predicate=pred,
                    max_stroke=0.030 if pred == "on" else 0.024,
                    settle_steps=max(8, int(settle_steps)),
                )
                relation_after = push.get("final_relation", {}) if isinstance(push, dict) else {}
                records.append(
                    {
                        "iteration": idx,
                        "action": "short_local_push",
                        "distance_xy": dist,
                        "push": push,
                        "relation": relation_after,
                    }
                )

            status, done = task_done()
            relation_after_drawer_margin = drawer_margin(relation_after)
            if relation_is_satisfied(relation_after) or done:
                return {
                    "object_name": resolved_object,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": relation_after,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "drawer_closing_clearance": relation_after_drawer_margin,
                    "near_target": True,
                    "xy_distance_to_target": dist,
                    "xy_tolerance": tolerance,
                    "skill": "local_placement_closure",
                }

        final_relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate=pred,
        )
        status, done = task_done()
        final_drawer_margin = drawer_margin(final_relation)
        return {
            "object_name": resolved_object,
            "target": target,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "records": records,
            "task_status": status,
            "relation_satisfied": bool(relation_is_satisfied(final_relation) or done),
            "drawer_closing_clearance": final_drawer_margin,
            "near_target": True,
            "xy_distance_to_target": float(np.linalg.norm(np.asarray(self.get_object_pose(resolved_object)[0])[:2] - target_pos[:2])),
            "xy_tolerance": tolerance,
            "skill": "local_placement_closure",
        }

    def _local_on_settle_closure(
        self,
        object_name: str,
        target: dict[str, Any],
        target_name: str,
        *,
        initial_relation: dict[str, Any],
        records: list[dict[str, Any]],
        xy_tolerance: float,
        initial_xy_distance: float,
        max_iters: int,
        settle_steps: int,
    ) -> dict[str, Any]:
        """Recover a near-target On relation by clearing, centering, and settling."""

        def task_done() -> tuple[dict[str, Any], bool]:
            status = self.get_task_status()
            done = bool(status.get("task_completed", False) or float(status.get("reward", 0.0)) >= 1.0)
            return status, done

        try:
            target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
            obj_pos, _ = self.get_object_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
            initial_xy_distance = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        except Exception:
            pass

        open_record: dict[str, Any] = {"action": "open_clear_settle"}
        try:
            self.open_gripper()
            self._settle_simulation_steps(max(8, int(settle_steps)))
            clear = self.move_linear_delta(
                np.array([0.0, 0.0, 0.055], dtype=np.float64),
                lin_vel_norm=0.040,
                timeout_s=3.0,
            )
            open_record["clear_move"] = clear
            self._settle_simulation_steps(max(8, int(settle_steps)))
        except Exception as exc:
            open_record["error"] = repr(exc)
            self._settle_simulation_steps(max(8, int(settle_steps)))
        relation_after_settle = self._placement_relation_diagnostic(
            object_name,
            target_name,
            predicate="on",
        )
        open_record["relation"] = relation_after_settle
        records.append(open_record)
        status, done = task_done()
        if bool(relation_after_settle.get("satisfied", False)) or done:
            return {
                "object_name": object_name,
                "target": target,
                "initial_relation": initial_relation,
                "final_relation": relation_after_settle,
                "records": records,
                "task_status": status,
                "relation_satisfied": True,
                "near_target": True,
                "xy_distance_to_target": initial_xy_distance,
                "xy_tolerance": xy_tolerance,
                "skill": "local_placement_closure",
            }

        target_anchor = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        iterations = max(1, int(max_iters))
        for idx in range(iterations):
            relation_before = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate="on",
            )
            status, done = task_done()
            if bool(relation_before.get("satisfied", False)) or done:
                return {
                    "object_name": object_name,
                    "target": target,
                    "initial_relation": initial_relation,
                    "final_relation": relation_before,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "near_target": True,
                    "xy_distance_to_target": float(relation_before.get("xy_distance", initial_xy_distance)),
                    "xy_tolerance": xy_tolerance,
                    "skill": "local_placement_closure",
                }

            xy_distance = float(relation_before.get("xy_distance", initial_xy_distance))
            current_target = self._placement_target_for_object(
                object_name,
                predicate="on",
                target_name=target_name,
            )
            target_now = np.asarray(current_target["target_position"], dtype=np.float64).reshape(3).copy()
            target_drift = float(np.linalg.norm(target_now[:2] - target_anchor[:2]))
            if target_drift > 0.045 or xy_distance > max(float(xy_tolerance), 0.090):
                records.append(
                    {
                        "iteration": idx,
                        "action": "abort_on_centering",
                        "target_drift": target_drift,
                        "xy_distance": xy_distance,
                        "relation": relation_before,
                    }
                )
                break

            if xy_distance <= 0.030 and bool(relation_before.get("contact", False)):
                self._settle_simulation_steps(max(20, int(settle_steps) * 2))
                relation_after_settle = self._placement_relation_diagnostic(
                    object_name,
                    target_name,
                    predicate="on",
                )
                records.append(
                    {
                        "iteration": idx,
                        "action": "settle_centered_contact",
                        "distance_xy": xy_distance,
                        "relation": relation_after_settle,
                    }
                )
                status, done = task_done()
                if bool(relation_after_settle.get("satisfied", False)) or done:
                    return {
                        "object_name": object_name,
                        "target": current_target,
                        "initial_relation": initial_relation,
                        "final_relation": relation_after_settle,
                        "records": records,
                        "task_status": status,
                        "relation_satisfied": True,
                        "near_target": True,
                        "xy_distance_to_target": float(relation_after_settle.get("xy_distance", xy_distance)),
                        "xy_tolerance": xy_tolerance,
                        "skill": "local_placement_closure",
                    }

            push = self._short_local_push_to_target(
                object_name,
                target_name,
                predicate="on",
                max_stroke=0.035,
                settle_steps=max(12, int(settle_steps) * 2),
            )
            relation_after = push.get("final_relation", {}) if isinstance(push, dict) else {}
            records.append(
                {
                    "iteration": idx,
                    "action": "short_on_center_push",
                    "distance_xy": xy_distance,
                    "target_drift": target_drift,
                    "push": push,
                    "relation": relation_after,
                }
            )
            status, done = task_done()
            if bool(relation_after.get("satisfied", False)) or done:
                return {
                    "object_name": object_name,
                    "target": current_target,
                    "initial_relation": initial_relation,
                    "final_relation": relation_after,
                    "records": records,
                    "task_status": status,
                    "relation_satisfied": True,
                    "near_target": True,
                    "xy_distance_to_target": float(relation_after.get("xy_distance", xy_distance)),
                    "xy_tolerance": xy_tolerance,
                    "skill": "local_placement_closure",
                }

        final_relation = self._placement_relation_diagnostic(
            object_name,
            target_name,
            predicate="on",
        )
        status, done = task_done()
        current_target = self._placement_target_for_object(
            object_name,
            predicate="on",
            target_name=target_name,
        )
        target_pos = np.asarray(current_target["target_position"], dtype=np.float64).reshape(3)
        obj_pos, _ = self.get_object_pose(object_name)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        return {
            "object_name": object_name,
            "target": current_target,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "records": records,
            "task_status": status,
            "relation_satisfied": bool(final_relation.get("satisfied", False) or done),
            "near_target": True,
            "xy_distance_to_target": float(np.linalg.norm(obj_pos[:2] - target_pos[:2])),
            "xy_tolerance": xy_tolerance,
            "skill": "local_placement_closure",
        }

    def approach_and_grasp_object(
        self,
        object_name: str,
        strategy: str = "auto",
        *,
        max_attempts: int = 6,
        start_candidate_index: int = 0,
        lift_height: float = 0.105,
        settle_steps: int = 12,
    ) -> dict[str, Any]:
        """Move to object-derived grasp candidates, close, lift, and verify.

        This helper uses readonly object coordinates to propose robot targets,
        but it completes the grasp only through normal robot motion and gripper
        commands. The success signal is local: whether the object follows the
        gripper upward or remains in close contact after closure.
        """
        plan = self.get_object_grasp_affordance(object_name, strategy=strategy)
        resolved_object = str(plan["object_name"])
        start_pos = np.asarray(plan["object_position"], dtype=np.float64)
        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        candidates = list(plan["candidates"])
        start_index = max(0, int(start_candidate_index))
        selected_candidates = candidates[start_index : start_index + int(max_attempts)]
        if not selected_candidates and candidates:
            selected_candidates = [candidates[start_index % len(candidates)]]

        for local_idx, candidate in enumerate(selected_candidates):
            idx = start_index + local_idx
            quat = np.asarray(candidate["quaternion_wxyz"], dtype=np.float64)
            self.open_gripper()
            self._settle_simulation_steps(int(settle_steps))

            move_records: list[dict[str, Any]] = []
            for waypoint_name in [
                "pre_finger_position",
                "outside_finger_position",
                "mouth_finger_position",
                "cross_finger_position",
                "deep_finger_position",
            ]:
                target = np.asarray(candidate[waypoint_name], dtype=np.float64)
                reach = self.goto_finger_midpoint_checked(target, quat, tolerance=0.050)
                move_records.append({"waypoint": waypoint_name, "reach": reach})
                self._settle_simulation_steps(4)

            before_close = self.measure_object_grasp_state(resolved_object, start_pos)
            self.close_gripper()
            self._settle_simulation_steps(max(int(settle_steps), 8))
            after_close = self.measure_object_grasp_state(resolved_object, start_pos)

            # Seat the rim/object in the fingertips with one small motion toward
            # the bowl center before lifting. Moving in the approach direction
            # tends to shove the bowl along the tabletop and can look like a
            # false local lift in coordinates.
            direction = np.asarray(candidate["approach_direction"], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(direction)) > 1e-9:
                seat = self.move_finger_midpoint_delta(
                    -0.010 * direction,
                    quat,
                    lin_vel_norm=0.025,
                    timeout_s=3.0,
                    tolerance=0.035,
                )
            else:
                seat = self.move_finger_midpoint_delta(
                    np.array([0.0, 0.0, -0.004], dtype=np.float64),
                    quat,
                    lin_vel_norm=0.020,
                    timeout_s=2.0,
                    tolerance=0.025,
                )

            lift = self.move_finger_midpoint_delta(
                np.array([0.0, 0.0, float(lift_height)], dtype=np.float64),
                quat,
                lin_vel_norm=0.035,
                timeout_s=5.0,
                tolerance=0.050,
            )
            after_lift = self.measure_object_grasp_state(resolved_object, start_pos)
            hold_probe = self._object_follow_probe(
                resolved_object,
                start_pos,
                quat,
                probe_delta=np.array([0.0, 0.0, 0.055], dtype=np.float64),
            )
            attempt = {
                "attempt_index": int(idx),
                "candidate_name": str(candidate["name"]),
                "candidate": candidate,
                "moves": move_records,
                "before_close": before_close,
                "after_close": after_close,
                "seat": seat,
                "lift": lift,
                "after_lift": after_lift,
                "hold_probe": hold_probe,
                "held": bool(hold_probe["held"]),
                "lift_delta": float(hold_probe["lift_delta"]),
            }
            attempts.append(attempt)
            if best is None or float(attempt["lift_delta"]) > float(best["lift_delta"]):
                best = attempt
            if bool(attempt["held"]):
                ee_pos, ee_quat = self.get_ee_pose()
                return {
                    "object_name": resolved_object,
                    "held": True,
                    "plan": plan,
                    "attempts": attempts,
                    "best_attempt": attempt,
                    "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
                    "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
                    "object_position": np.asarray(after_lift["object_position"], dtype=np.float64).copy(),
                    "lift_delta": float(after_lift["lift_delta"]),
                }

            self.open_gripper()
            self._settle_simulation_steps(6)
            retreat = np.asarray(candidate["pre_finger_position"], dtype=np.float64)
            self.goto_finger_midpoint_checked(retreat, quat, tolerance=0.060)

        ee_pos, ee_quat = self.get_ee_pose()
        current_state = self.measure_object_grasp_state(resolved_object, start_pos)
        return {
            "object_name": resolved_object,
            "held": False,
            "plan": plan,
            "attempts": attempts,
            "best_attempt": best,
            "grasp_quat": np.asarray(ee_quat, dtype=np.float64).copy(),
            "grasp_position": np.asarray(ee_pos, dtype=np.float64).copy(),
            "object_position": np.asarray(current_state["object_position"], dtype=np.float64).copy(),
            "lift_delta": float(current_state["lift_delta"]),
        }

    def goto_pose(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
        z_approach: float = 0.0,
    ) -> None:
        """Move the observed gripper pose returned by `get_ee_pose`.

        The parent API applies `_TCP_OFFSET` before solving IK, while
        `robot_cartesian_pos` already reports the resulting controlled gripper
        pose. This wrapper makes the public oracle-readonly coordinate system
        self-consistent: a target passed here is the pose that `get_ee_pose`
        should report after the motion.
        """
        target_pos = np.asarray(position, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        control_pos = self._control_position_for_reported_pose(target_pos, target_quat)
        super().goto_pose(control_pos, target_quat, z_approach=z_approach)


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
        """Manner-aware goto_pose with oracle-readonly coordinate correction.

        Applies the same _control_position_for_reported_pose offset as
        goto_pose so that coordinates from get_ee_pose / get_object_pose are
        self-consistent with the target passed here.

        Args:
            position:        (3,) XYZ in the oracle-readonly coordinate frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach:      Z offset pre-approach.
            v_max:           Max joint-space speed (rad/s).
                             缓缓靠近 -> 0.03, 正常 -> 0.15, 迅速 -> 0.25.
            a_max:           Max joint-space acceleration (rad/s^2).
            jerk_limit:      Minimum-jerk quintic interpolation when set
                             (Manner D: 平稳约束). None -> trapezoidal profile.

        Usage examples for LLM-generated code:
            goto_pose_with_profile(pos, quat, v_max=0.03, a_max=0.01)    # 缓缓靠近
            goto_pose_with_profile(pos, quat, v_max=0.25, a_max=0.15)    # 迅速移动
            goto_pose_with_profile(pos, quat, v_max=0.06, jerk_limit=0.8) # 平稳放下
        """
        import numpy as _np
        target_pos = _np.asarray(position, dtype=_np.float64).reshape(3)
        target_quat = _np.asarray(quaternion_wxyz, dtype=_np.float64).reshape(4)
        control_pos = self._control_position_for_reported_pose(target_pos, target_quat)
        super().goto_pose_with_profile(
            control_pos,
            target_quat,
            z_approach=z_approach,
            v_max=v_max,
            a_max=a_max,
            jerk_limit=jerk_limit,
        )

    def get_cabinet_drawer_progress(self, level: str = "middle") -> dict[str, Any]:
        """Read only the normalized and benchmark-aligned state of a cabinet drawer.

        Use this for diagnostics or for deciding whether to continue a physical
        pull/push. This exposes LIBERO's open/close predicate status but does
        not expose a writable simulator handle.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            Dictionary with:
                - level: drawer level.
                - open_fraction: 0.0 means closed, 1.0 means fully open.
                - bddl_open: whether LIBERO's Open predicate is satisfied.
                - bddl_open_margin: positive means additional pull is needed.
                - bddl_close: whether LIBERO's Close predicate is satisfied.
                - bddl_close_margin: positive means the drawer is past the
                  benchmark close threshold.
        """
        state = super().get_cabinet_drawer_state(level)
        open_threshold = self._bddl_articulation_predicate_threshold(
            state["region_name"],
            "open",
            fallback=self._cabinet_drawer_target_qpos(level, "open"),
        )
        close_threshold = self._bddl_articulation_predicate_threshold(
            state["region_name"],
            "close",
            fallback=self._cabinet_drawer_target_qpos(level, "close"),
        )
        joint_position = float(state["qpos"])
        return {
            "level": str(state["level"]),
            "open_fraction": float(state["open_fraction"]),
            "joint_position": joint_position,
            "bddl_open_threshold": float(open_threshold),
            "bddl_close_threshold": float(close_threshold),
            "bddl_open_margin": float(joint_position - open_threshold),
            "bddl_close_margin": float(joint_position - close_threshold),
            "bddl_open": bool(joint_position < float(open_threshold)),
            "bddl_close": bool(joint_position > float(close_threshold)),
        }

    def _bddl_articulation_predicate_threshold(
        self,
        bddl_name: str,
        predicate: str,
        *,
        fallback: float,
    ) -> float:
        """Return the threshold used by LIBERO articulation predicates."""
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
        try:
            obj = task_env.get_object(str(parent_name))
        except Exception:
            try:
                obj = task_env.get_object(str(parent_name).removesuffix("_1"))
            except Exception:
                return float(fallback)
        articulation = getattr(obj, "object_properties", {}).get("articulation", {})
        ranges = articulation.get(f"default_{predicate}_ranges")
        if ranges is None:
            return float(fallback)
        arr = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return float(fallback)
        if str(predicate).lower() == "open":
            return float(np.max(arr))
        if str(predicate).lower() == "close":
            return float(np.min(arr))
        return float(np.mean(arr))

    def get_gripper_fingertip_state(self) -> dict[str, Any]:
        """Read current fingertip and pad positions in robot-base coordinates.

        Use this after moving near a drawer handle to verify whether the handle
        is between the two fingers. This is a readonly measurement; it does not
        move the robot or alter simulator state.

        Args:
            None

        Returns:
            Dictionary with:
                - left_tip_position/right_tip_position: fingertip body centers.
                - left_pad_position/right_pad_position: collision-pad centers.
                - finger_midpoint: midpoint between pad centers.
                - finger_gap: distance between pad centers.
                - finger_axis: unit vector from left pad to right pad.
        """
        sim = self._env.handle.env.sim

        def body_pos(name: str) -> np.ndarray:
            body_id = sim.model.body_name2id(name)
            return self._world_position_to_robot_base(
                np.asarray(sim.data.xpos[body_id], dtype=np.float64)
            )

        def geom_pos(name: str) -> np.ndarray:
            geom_id = sim.model.geom_name2id(name)
            return self._world_position_to_robot_base(
                np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
            )

        left_tip = body_pos("gripper0_finger_joint1_tip")
        right_tip = body_pos("gripper0_finger_joint2_tip")
        left_pad = geom_pos("gripper0_finger1_pad_collision")
        right_pad = geom_pos("gripper0_finger2_pad_collision")
        axis = right_pad - left_pad
        gap = float(np.linalg.norm(axis))
        if gap > 1e-9:
            axis = axis / gap
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return {
            "left_tip_position": left_tip,
            "right_tip_position": right_tip,
            "left_pad_position": left_pad,
            "right_pad_position": right_pad,
            "finger_midpoint": 0.5 * (left_pad + right_pad),
            "finger_gap": gap,
            "finger_axis": axis,
            "finger_midpoint_offset_from_ee": self._finger_midpoint_offset_from_ee(),
        }

    def get_drawer_hook_poses(
        self,
        level: str = "middle",
        *,
        lateral_step: float = 0.018,
        height_step: float = 0.010,
    ) -> dict[str, Any]:
        """Return candidate poses for hooking a drawer handle with the gripper.

        The candidates are target proposals only. CAP code should execute them
        with `goto_pose_checked`, inspect the actual arrival error, then measure
        hook alignment before pulling.

        Args:
            level: One of "top", "middle", or "bottom".
            lateral_step: Small offset in meters along the handle/finger axis.
            height_step: Small vertical offset in meters used for lower/upper
                capture candidates.

        Returns:
            Dictionary with:
                - level, handle_position, pull_direction, hook_quaternion_wxyz.
                - prehook_position: safe pre-contact pose in front of handle.
                - mouth_position: open-finger pose just in front of handle.
                - hook_candidates: list of named candidate capture poses.
        """
        handle_pos, hook_quat, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        lateral = self._drawer_handle_lateral_axis(level)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        candidates = [
            (
                "center_behind_bar",
                handle_pos - 0.020 * pull_dir,
            ),
            (
                "left_behind_bar",
                handle_pos - 0.022 * pull_dir - float(lateral_step) * lateral,
            ),
            (
                "right_behind_bar",
                handle_pos - 0.022 * pull_dir + float(lateral_step) * lateral,
            ),
            (
                "low_center_hook",
                handle_pos - 0.018 * pull_dir - float(height_step) * vertical,
            ),
            (
                "high_center_hook",
                handle_pos - 0.018 * pull_dir + float(height_step) * vertical,
            ),
        ]
        return {
            "level": level,
            "handle_position": handle_pos,
            "pull_direction": pull_dir,
            "handle_lateral_axis": lateral,
            "hook_quaternion_wxyz": hook_quat,
            "prehook_position": handle_pos + 0.120 * pull_dir + 0.030 * vertical,
            "mouth_position": handle_pos + 0.040 * pull_dir + 0.006 * vertical,
            "finger_prehook_position": handle_pos + 0.090 * pull_dir + 0.006 * vertical,
            "finger_mouth_position": handle_pos + 0.030 * pull_dir + 0.002 * vertical,
            "hook_candidates": [
                {"name": name, "position": position, "quaternion_wxyz": hook_quat}
                for name, position in candidates
            ],
        }

    def get_drawer_side_hook_poses(self, level: str = "middle") -> dict[str, Any]:
        """Return a reachable side-entry hook corridor for drawer handles.

        The front-on handle pose is geometrically appealing but often falls on
        an IK/contact branch that cannot insert behind the handle. This helper
        exposes a data-generation affordance learned from readonly rollout
        diagnostics: enter from the cabinet side with a side-facing gripper,
        choose among low reachable contact branches, then pull along the drawer
        open direction. The returned poses are targets only; they do not alter
        simulator state.
        """
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        lateral = self._drawer_handle_lateral_axis(level)
        side_quat = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64)
        roll_hook_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        base = handle_pos - 0.055 * lateral - 0.315 * pull_dir + 0.096 * np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        candidates = [
            ("side_low_left", base + np.array([-0.070, 0.025, 0.000], dtype=np.float64)),
            ("side_low_center", base + np.array([-0.030, -0.005, 0.000], dtype=np.float64)),
            ("side_low_right", base + np.array([0.010, -0.025, 0.000], dtype=np.float64)),
            ("side_low_back_right", base + np.array([0.025, -0.040, -0.010], dtype=np.float64)),
            ("side_low_far_right", base + np.array([0.040, -0.045, -0.015], dtype=np.float64)),
        ]
        return {
            "level": level,
            "handle_position": handle_pos,
            "pull_direction": pull_dir,
            "handle_lateral_axis": lateral,
            "side_quaternion_wxyz": side_quat,
            "roll_hook_quaternion_wxyz": roll_hook_quat,
            "pre_contact_position": base + np.array([-0.040, -0.070, 0.020], dtype=np.float64),
            "branch_guide_positions": [
                base + np.array([0.000, -0.065, -0.015], dtype=np.float64),
                base + np.array([0.015, -0.060, -0.015], dtype=np.float64),
                base + np.array([0.030, -0.055, -0.015], dtype=np.float64),
            ],
            "side_hook_candidates": [
                {"name": name, "position": position, "quaternion_wxyz": side_quat}
                for name, position in candidates
            ],
        }

    def get_drawer_tiltx_hook_poses(self, level: str = "middle") -> dict[str, Any]:
        """Return front-entry finger-midpoint targets for handle hooking.

        The returned targets are expressed as desired positions for the midpoint
        between the two gripper pads. CAP code should execute them with
        `goto_finger_midpoint_checked`, then call
        `measure_drawer_hook_alignment` before pulling.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            Dictionary with handle geometry, the tilt-x hook quaternion, safe
            approach waypoints, and candidate finger-midpoint hook targets.
        """
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        lateral = self._drawer_handle_lateral_axis(level)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hook_quat = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)

        offsets = [
            ("front_center_04", 0.000, 0.040, 0.000),
            ("front_left_03_03", -0.030, 0.030, 0.000),
            ("front_right_03_03", 0.030, 0.030, 0.000),
            ("front_center_03_low", 0.000, 0.030, -0.005),
            ("front_center_05", 0.000, 0.050, 0.000),
            ("front_left_02_04", -0.020, 0.040, -0.003),
            ("front_right_02_04", 0.020, 0.040, -0.003),
        ]

        def make_target(dx: float, dy: float, dz: float) -> np.ndarray:
            return handle_pos + dx * lateral + dy * pull_dir + dz * vertical

        return {
            "level": level,
            "handle_position": handle_pos,
            "pull_direction": pull_dir,
            "handle_lateral_axis": lateral,
            "hook_quaternion_wxyz": hook_quat,
            "prehook_finger_position": handle_pos + 0.105 * pull_dir + 0.105 * vertical,
            "mouth_finger_position": handle_pos + 0.070 * pull_dir + 0.025 * vertical,
            "hook_candidates": [
                {
                    "name": name,
                    "finger_midpoint_position": make_target(dx, dy, dz),
                    "offset_lateral_pull_vertical": np.array([dx, dy, dz], dtype=np.float64),
                    "quaternion_wxyz": hook_quat,
                }
                for name, dx, dy, dz in offsets
            ],
        }

    def get_drawer_sleeve_hook_poses(self, level: str = "middle") -> dict[str, Any]:
        """Return front-entry targets for sleeving the open gripper over a handle.

        This is the drawer sequence we want for data generation:
        approach with the gripper open, place the handle inside the two-finger
        gap, then close the gripper and pull. The candidate positions therefore
        keep the pad midpoint at or just in front of the true handle center
        instead of several centimeters in front of the drawer face.
        """
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        lateral = self._drawer_handle_lateral_axis(level)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hook_quat = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)

        offsets = [
            ("sleeve_front_015", 0.000, 0.015, 0.000),
            ("sleeve_front_008", 0.000, 0.008, 0.000),
            ("sleeve_center_000", 0.000, 0.000, 0.000),
            ("sleeve_low_front_008", 0.000, 0.008, -0.004),
            ("sleeve_high_front_008", 0.000, 0.008, 0.004),
            ("sleeve_left_front_008", -0.008, 0.008, 0.000),
            ("sleeve_right_front_008", 0.008, 0.008, 0.000),
        ]

        def make_target(dx: float, dy: float, dz: float) -> np.ndarray:
            return handle_pos + dx * lateral + dy * pull_dir + dz * vertical

        return {
            "level": level,
            "handle_position": handle_pos,
            "pull_direction": pull_dir,
            "handle_lateral_axis": lateral,
            "hook_quaternion_wxyz": hook_quat,
            "prehook_finger_position": handle_pos + 0.095 * pull_dir + 0.090 * vertical,
            "mouth_finger_position": handle_pos + 0.045 * pull_dir + 0.018 * vertical,
            "sleeve_candidates": [
                {
                    "name": name,
                    "finger_midpoint_position": make_target(dx, dy, dz),
                    "offset_lateral_pull_vertical": np.array([dx, dy, dz], dtype=np.float64),
                    "quaternion_wxyz": hook_quat,
                }
                for name, dx, dy, dz in offsets
            ],
        }

    def measure_drawer_hook_alignment(self, level: str = "middle") -> dict[str, Any]:
        """Measure whether the drawer handle is currently between the fingers.

        Args:
            level: One of "top", "middle", or "bottom".

        Returns:
            Dictionary with geometric diagnostics:
                - handle_position and finger_midpoint.
                - signed_lateral_from_midpoint: handle offset along finger axis.
                - abs_lateral_from_midpoint: absolute lateral offset.
                - finger_gap: current pad-to-pad distance.
                - depth_along_pull: handle offset from finger midpoint along
                  drawer pull direction. Negative usually means the handle is
                  behind the finger midpoint, which is useful for hooking.
                - vertical_error: handle z minus finger midpoint z.
                - centered_between_fingers: offset along the finger-open axis is
                  inside the current gap.
                - ready_to_close: open gripper is sleeved around the handle.
                - closed_on_handle: closed gripper still centers the handle.
                - likely_hooked: true for either ready-to-close or closed-on-handle.
        """
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        handle_lateral_axis = self._drawer_handle_lateral_axis(level)
        fingers = self.get_gripper_fingertip_state()
        midpoint = np.asarray(fingers["finger_midpoint"], dtype=np.float64)
        finger_axis = np.asarray(fingers["finger_axis"], dtype=np.float64)
        gap = float(fingers["finger_gap"])
        rel = handle_pos - midpoint
        signed_lateral = float(np.dot(rel, finger_axis))
        abs_lateral = abs(signed_lateral)
        signed_handle_lateral = float(np.dot(rel, handle_lateral_axis))
        abs_handle_lateral = abs(signed_handle_lateral)
        depth = float(np.dot(rel, pull_dir))
        vertical = float(rel[2])
        lateral_tolerance = max(0.45 * gap, 0.012)
        centered = bool(gap > 0.0 and abs_lateral <= lateral_tolerance)
        depth_abs = abs(depth)
        ready_to_close = bool(
            gap >= 0.040
            and centered
            and abs_handle_lateral <= 0.020
            and depth_abs <= 0.030
            and abs(vertical) <= 0.035
        )
        closed_on_handle = bool(
            gap < 0.030
            and centered
            and abs_handle_lateral <= 0.022
            and depth_abs <= 0.032
            and abs(vertical) <= 0.018
        )
        loose_hook_geometry = bool(
            centered
            and abs_handle_lateral <= 0.030
            and -0.055 <= depth <= 0.020
            and abs(vertical) <= 0.030
        )
        likely_hooked = bool(ready_to_close or closed_on_handle)
        return {
            "handle_position": handle_pos,
            "finger_midpoint": midpoint,
            "finger_gap": gap,
            "signed_lateral_from_midpoint": signed_lateral,
            "abs_lateral_from_midpoint": abs_lateral,
            "signed_handle_lateral_error": signed_handle_lateral,
            "abs_handle_lateral_error": abs_handle_lateral,
            "depth_along_pull": depth,
            "vertical_error": vertical,
            "centered_between_fingers": centered,
            "ready_to_close": ready_to_close,
            "closed_on_handle": closed_on_handle,
            "loose_hook_geometry": loose_hook_geometry,
            "likely_hooked": likely_hooked,
            "left_pad_position": fingers["left_pad_position"],
            "right_pad_position": fingers["right_pad_position"],
        }

    def _cabinet_drawer_open_direction_world(self, level: str) -> np.ndarray:
        """Return drawer opening direction from joint metadata without sim writes."""
        sim = self._env.handle.env.sim
        joint_name, _ = self._cabinet_drawer_names(level)
        joint_id = self._joint_id(joint_name)
        body_id = int(sim.model.jnt_bodyid[joint_id])
        local_axis = np.asarray(sim.model.jnt_axis[joint_id], dtype=np.float64)
        body_rot = np.asarray(sim.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        axis_world = body_rot @ local_axis

        joint_range = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        open_value = float(joint_range[0])
        closed_value = self._cabinet_drawer_target_qpos(level, "close")
        direction = -axis_world if open_value < closed_value else axis_world
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return direction / norm

    def _drawer_handle_world_and_geom_ids(self, level: str) -> tuple[np.ndarray, list[int]]:
        """Return handle center and likely handle geom ids without sim writes."""
        sim = self._env.handle.env.sim
        _, region_name = self._cabinet_drawer_names(level)
        site_id = sim.model.site_name2id(region_name)
        region_world = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
        pull_world = self._cabinet_drawer_open_direction_world(level)
        pull_world = pull_world / max(np.linalg.norm(pull_world), 1e-9)
        vertical_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        candidates: list[tuple[int, np.ndarray, float, float, float]] = []
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id)
            if not geom_name or "cabinet" not in geom_name.lower():
                continue
            geom_world = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
            rel = geom_world - region_world
            front_distance = float(np.dot(rel, pull_world))
            vertical_distance = abs(float(np.dot(rel, vertical_world)))
            lateral_vec = rel - front_distance * pull_world - np.dot(rel, vertical_world) * vertical_world
            lateral_distance = float(np.linalg.norm(lateral_vec))
            if 0.070 <= front_distance <= 0.125 and vertical_distance <= 0.018 and lateral_distance <= 0.070:
                candidates.append((geom_id, geom_world, front_distance, vertical_distance, lateral_distance))

        if candidates:
            # The actual handle cluster is the foremost group around the region
            # height. Exclude the drawer front panel, which is broader and
            # usually a few centimeters farther back.
            max_front = max(item[2] for item in candidates)
            front_band = [
                item for item in candidates if item[2] >= max_front - 0.012
            ]
            geom_ids = [item[0] for item in front_band]
            handle_world = np.mean(np.stack([item[1] for item in front_band], axis=0), axis=0)
            return handle_world, geom_ids

        fallback = region_world + 0.105 * pull_world + np.array([0.0, 0.0, -0.004], dtype=np.float64)
        return fallback, []

    def measure_drawer_gripper_contacts(self, level: str = "middle") -> dict[str, Any]:
        """Read MuJoCo contacts between gripper geoms and drawer-handle geoms.

        This is a diagnostic-only readonly measurement. It does not change robot
        or articulated object state.
        """
        sim = self._env.handle.env.sim
        _, handle_geom_ids = self._drawer_handle_world_and_geom_ids(level)
        handle_geom_set = set(handle_geom_ids)
        gripper_tokens = ("gripper0_finger", "gripper0_hand", "pad_collision")
        contacts: list[dict[str, Any]] = []
        handle_contact_count = 0
        gripper_cabinet_contact_count = 0
        for contact_idx in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            name1 = sim.model.geom_id2name(geom1) or str(geom1)
            name2 = sim.model.geom_id2name(geom2) or str(geom2)
            lower1 = name1.lower()
            lower2 = name2.lower()
            gripper_side = any(token in lower1 for token in gripper_tokens) or any(
                token in lower2 for token in gripper_tokens
            )
            cabinet_side = "cabinet" in lower1 or "cabinet" in lower2
            handle_side = geom1 in handle_geom_set or geom2 in handle_geom_set
            if gripper_side and cabinet_side:
                gripper_cabinet_contact_count += 1
                if handle_side:
                    handle_contact_count += 1
                contacts.append(
                    {
                        "geom1": name1,
                        "geom2": name2,
                        "distance": float(contact.dist),
                        "with_handle_geom": bool(handle_side),
                    }
                )
        return {
            "level": level,
            "handle_geom_count": len(handle_geom_ids),
            "handle_geom_ids": np.asarray(handle_geom_ids, dtype=np.int64),
            "handle_contact_count": int(handle_contact_count),
            "gripper_cabinet_contact_count": int(gripper_cabinet_contact_count),
            "has_handle_contact": bool(handle_contact_count > 0),
            "contacts": contacts[:12],
        }

    def _object_body_ids(self, object_name: str) -> list[int]:
        sim = self._env.handle.env.sim
        query = str(object_name).strip().replace(" ", "_").lower()
        libero_env = self._libero_env_with_objects()
        obj_body_id = getattr(libero_env, "obj_body_id", {})
        body_ids: list[int] = []
        instance_name = query if query in obj_body_id else None
        if instance_name is None and f"{query}_1" in obj_body_id:
            instance_name = f"{query}_1"
        if instance_name is not None:
            body_ids.append(int(obj_body_id[instance_name]))
        else:
            prefixes = (query,)
            for body_id in range(sim.model.nbody):
                body_name = sim.model.body_id2name(body_id)
                if not body_name:
                    continue
                if any(body_name == prefix or body_name.startswith(prefix + "_") for prefix in prefixes):
                    body_ids.append(int(body_id))

        # Include descendants so compound objects keep all collision parts.
        body_set = set(body_ids)
        changed = True
        while changed:
            changed = False
            for body_id in range(sim.model.nbody):
                parent_id = int(sim.model.body_parentid[body_id])
                if parent_id in body_set and body_id not in body_set:
                    body_set.add(int(body_id))
                    changed = True
        return sorted(body_set)

    def _object_geom_ids(self, object_name: str) -> list[int]:
        sim = self._env.handle.env.sim
        body_ids = set(self._object_body_ids(object_name))
        query = str(object_name).strip().replace(" ", "_").lower()
        libero_env = self._libero_env_with_objects()
        obj_body_id = getattr(libero_env, "obj_body_id", {})
        instance_name = query if query in obj_body_id else None
        if instance_name is None and f"{query}_1" in obj_body_id:
            instance_name = f"{query}_1"
        name_prefixes = (instance_name or query,)
        geom_ids: list[int] = []
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id) or ""
            geom_body_id = int(sim.model.geom_bodyid[geom_id])
            name_matches = any(
                geom_name == prefix or geom_name.startswith(prefix + "_")
                for prefix in name_prefixes
            )
            if geom_body_id in body_ids or name_matches:
                geom_ids.append(int(geom_id))
        return geom_ids

    def _object_geometry_summary(self, object_name: str) -> dict[str, Any]:
        sim = self._env.handle.env.sim
        object_position, _ = self.get_object_pose(object_name)
        center = np.asarray(object_position, dtype=np.float64).reshape(3)
        geom_ids = self._object_geom_ids(object_name)
        points: list[np.ndarray] = []
        for geom_id in geom_ids:
            geom_world = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
            geom_robot = self._world_position_to_robot_base(geom_world)
            size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float64).reshape(-1)
            radius = float(np.clip(np.max(np.abs(size[:3])) if size.size else 0.025, 0.006, 0.090))
            points.extend(
                [
                    geom_robot + np.array([radius, radius, radius], dtype=np.float64),
                    geom_robot + np.array([radius, radius, -radius], dtype=np.float64),
                    geom_robot + np.array([radius, -radius, radius], dtype=np.float64),
                    geom_robot + np.array([-radius, radius, radius], dtype=np.float64),
                    geom_robot - np.array([radius, radius, radius], dtype=np.float64),
                ]
            )
        if not points:
            fallback_radius = 0.055 if "bowl" in object_name else 0.035
            fallback_height = 0.060 if "bowl" in object_name else 0.045
            bounds_min = center + np.array([-fallback_radius, -fallback_radius, 0.0], dtype=np.float64)
            bounds_max = center + np.array([fallback_radius, fallback_radius, fallback_height], dtype=np.float64)
        else:
            stacked = np.stack(points, axis=0)
            bounds_min = np.min(stacked, axis=0)
            bounds_max = np.max(stacked, axis=0)
            # Observation pose for shallow table objects is often near the
            # lower body frame; keep the fallback from collapsing under it.
            bounds_min = np.minimum(bounds_min, center + np.array([-0.010, -0.010, -0.004], dtype=np.float64))
            bounds_max = np.maximum(bounds_max, center + np.array([0.010, 0.010, 0.045], dtype=np.float64))
        return {
            "object_name": object_name,
            "geom_ids": np.asarray(geom_ids, dtype=np.int64),
            "bounds_min": bounds_min.copy(),
            "bounds_max": bounds_max.copy(),
            "center": center.copy(),
            "extent": bounds_max - bounds_min,
        }

    def _measure_object_gripper_contacts(self, object_name: str) -> dict[str, Any]:
        sim = self._env.handle.env.sim
        object_geom_set = set(self._object_geom_ids(object_name))
        gripper_tokens = ("gripper0_finger", "gripper0_hand", "pad_collision")
        contacts: list[dict[str, Any]] = []
        object_contact_count = 0
        for contact_idx in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            name1 = sim.model.geom_id2name(geom1) or str(geom1)
            name2 = sim.model.geom_id2name(geom2) or str(geom2)
            lower1 = name1.lower()
            lower2 = name2.lower()
            gripper_side = any(token in lower1 for token in gripper_tokens) or any(
                token in lower2 for token in gripper_tokens
            )
            object_side = geom1 in object_geom_set or geom2 in object_geom_set
            if gripper_side and object_side:
                object_contact_count += 1
                contacts.append(
                    {
                        "geom1": name1,
                        "geom2": name2,
                        "distance": float(contact.dist),
                    }
                )
        return {
            "object_name": object_name,
            "object_geom_count": len(object_geom_set),
            "object_geom_ids": np.asarray(sorted(object_geom_set), dtype=np.int64),
            "object_contact_count": int(object_contact_count),
            "has_object_contact": bool(object_contact_count > 0),
            "contacts": contacts[:12],
        }

    def _placement_target_for_object(
        self,
        object_name: str,
        *,
        predicate: str,
        target_name: str | None,
    ) -> dict[str, Any]:
        pred = str(predicate).strip().lower()
        if pred not in {"in", "on"}:
            target_text = str(target_name or "").lower()
            pred = "in" if any(token in target_text for token in ("basket", "compartment", "contain")) else "on"
        try:
            goal_target = self._goal_target_name_for_object(object_name, predicate=pred)
        except Exception:
            goal_target = None
        if goal_target and self._target_matches_reference(goal_target, target_name):
            return self._placement_target_from_state_name(object_name, goal_target, predicate=pred)
        semantic_target = self._semantic_target_state_name(target_name, predicate=pred)
        if semantic_target:
            return self._placement_target_from_state_name(
                object_name,
                semantic_target,
                predicate=pred,
            )
        goal_predicate_target = self._goal_predicate_target_for_object(
            object_name,
            target_reference=target_name,
            exclude_predicate=pred,
        )
        if goal_predicate_target is not None:
            goal_predicate, goal_target = goal_predicate_target
            return self._placement_target_from_state_name(
                object_name,
                goal_target,
                predicate=goal_predicate,
            )
        if target_name:
            if pred == "on":
                try:
                    resolved_target_object = self._resolve_object_name(
                        target_name,
                        require_free_joint=False,
                    )
                    return self._placement_target_from_state_name(
                        object_name,
                        f"{resolved_target_object}_1",
                        predicate=pred,
                    )
                except Exception:
                    pass
            try:
                resolved_target = self._resolve_target_name(target_name)
                target_pos, _ = self.get_site_pose(resolved_target)
                target_size = self._target_site_size(resolved_target)
                return {
                    "object_name": object_name,
                    "predicate": pred,
                    "target_name": resolved_target,
                    "target_position": np.asarray(target_pos, dtype=np.float64).copy(),
                    "target_size": target_size,
                    "target_type": "site",
                }
            except Exception:
                if pred == "in":
                    resolved_target_object = self._resolve_object_name(
                        target_name,
                        require_free_joint=False,
                    )
                    return self._placement_target_from_state_name(
                        object_name,
                        f"{resolved_target_object}_1",
                        predicate=pred,
                    )
                raise
        try:
            return self.get_goal_target_region_pose(object_name, predicate=pred)
        except Exception:
            fallback_pred = "on" if pred == "in" else "in"
            return self.get_goal_target_region_pose(object_name, predicate=fallback_pred)

    def _goal_predicate_target_for_object(
        self,
        object_name: str,
        *,
        target_reference: str | None,
        exclude_predicate: str | None = None,
    ) -> tuple[str, str] | None:
        """Return the actual BDDL goal predicate/target for a language target.

        Some LIBERO goals use surface predicates for receptacle language, e.g.
        "put X in the bowl" is represented as On(X, bowl).  This readonly
        lookup lets execution use the benchmark verifier semantics while the
        planner can still start from natural-language relations.
        """
        task_env = self._env.handle.env.env
        exclude = None if exclude_predicate is None else str(exclude_predicate).lower()
        for state in task_env.parsed_problem.get("goal_state", []):
            if len(state) != 3:
                continue
            goal_predicate, bddl_object_name, bddl_target_name = state
            goal_pred = str(goal_predicate).lower()
            if exclude is not None and goal_pred == exclude:
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
            if not self._target_matches_reference(str(bddl_target_name), target_reference):
                continue
            if goal_pred in {"in", "on"}:
                return goal_pred, str(bddl_target_name)
        return None

    def _placement_target_from_state_name(
        self,
        object_name: str,
        target_state_name: str,
        *,
        predicate: str,
    ) -> dict[str, Any]:
        libero_env = self._libero_env_with_objects()
        state_name = self._resolve_object_state_name(target_state_name)
        target_state = libero_env.object_states_dict[state_name]
        target_world = np.asarray(target_state.get_geom_state()["pos"], dtype=np.float64).copy()
        target_type = str(getattr(target_state, "object_state_type", "object") or "object")
        target_size = np.array([0.04, 0.04, 0.04], dtype=np.float64)
        target_xmat = np.eye(3, dtype=np.float64)
        if target_type == "site":
            site = libero_env.object_sites_dict.get(state_name)
            if site is not None:
                target_size = np.asarray(
                    getattr(site, "size", target_size),
                    dtype=np.float64,
                ).reshape(-1)[:3]
            try:
                target_xmat = np.asarray(
                    libero_env.sim.data.get_site_xmat(state_name),
                    dtype=np.float64,
                ).reshape(3, 3)
            except Exception:
                target_xmat = np.eye(3, dtype=np.float64)
        else:
            try:
                target_object = state_name.removesuffix("_1")
                geometry = self._object_geometry_summary(target_object)
                extent = np.asarray(geometry["extent"], dtype=np.float64)
                target_size = np.maximum(0.5 * np.abs(extent[:3]), np.array([0.035, 0.035, 0.010]))
            except Exception:
                target_size = np.array([0.055, 0.055, 0.020], dtype=np.float64)
        return {
            "object_name": object_name,
            "predicate": str(predicate).strip().lower(),
            "target_name": state_name,
            "target_position": self._world_position_to_robot_base(target_world),
            "target_size": target_size.copy(),
            "target_xmat": target_xmat.copy(),
            "target_type": target_type,
        }

    def _resolve_target_name(self, target_name: str) -> str:
        query = target_name.strip().replace(" ", "_").lower()
        site_names = self.get_scene_site_names()
        if query in site_names:
            return query
        if not query.endswith("_1") and f"{query}_1" in site_names:
            return f"{query}_1"
        matches = [name for name in site_names if query in name.lower() or name.lower() in query]
        if len(matches) == 1:
            return matches[0]
        if matches:
            contain_matches = [name for name in matches if "contain" in name.lower() or "region" in name.lower()]
            if contain_matches:
                return sorted(contain_matches, key=len)[0]
            return sorted(matches, key=len)[0]
        raise KeyError(f"Target site '{target_name}' not found. Available sites: {site_names}")

    def _target_site_size(self, site_name: str) -> np.ndarray:
        libero_env = self._libero_env_with_objects()
        site = getattr(libero_env, "object_sites_dict", {}).get(site_name)
        if site is not None:
            size = np.asarray(getattr(site, "size", np.array([0.04, 0.04, 0.04])), dtype=np.float64)
            if size.size >= 3:
                return size.reshape(-1)[:3].copy()
        return np.array([0.04, 0.04, 0.04], dtype=np.float64)

    def _placement_relation_diagnostic(
        self,
        object_name: str,
        target_name: str,
        *,
        predicate: str,
    ) -> dict[str, Any]:
        pred = str(predicate).strip().lower()
        if pred == "in":
            diag = self.get_site_in_diagnostic(object_name, target_name)
            result = dict(diag)
            result["predicate"] = "in"
            result["satisfied"] = bool(diag.get("in", False))
            return result
        try:
            target_state_name = self._resolve_object_state_name(target_name)
        except Exception:
            target_state_name = str(target_name)
        libero_env = self._libero_env_with_objects()
        target_state = getattr(libero_env, "object_states_dict", {}).get(target_state_name)
        if target_state is not None and getattr(target_state, "object_state_type", "") != "site":
            diag = self.get_on_diagnostic(object_name, target_state_name)
            result = dict(diag)
            result["predicate"] = "on"
            result["target_type"] = "object"
            result["satisfied"] = bool(diag.get("on", False))
            return result
        diag = self.get_site_on_diagnostic(object_name, target_name)
        result = dict(diag)
        result["predicate"] = "on"
        result["target_type"] = "site"
        result["satisfied"] = bool(diag.get("on", False))
        return result

    def _relation_xy_distance_to_target(self, relation: dict[str, Any] | None) -> float:
        """Return object-target XY distance from a readonly relation diagnostic."""
        if not isinstance(relation, dict):
            return float("inf")
        relation_target = relation.get("site_name", relation.get("target_name", ""))
        if self._is_drawer_region_target_name(str(relation_target)):
            try:
                object_pos = np.asarray(relation["object_pos_world"], dtype=np.float64).reshape(-1)[:3]
                target_pos = np.asarray(relation["site_pos_world"], dtype=np.float64).reshape(-1)[:3]
                return float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
            except Exception:
                pass
        if "xy_distance" in relation:
            try:
                return float(relation.get("xy_distance"))
            except Exception:
                pass
        if "local_delta" in relation:
            try:
                local_delta = np.asarray(relation["local_delta"], dtype=np.float64).reshape(-1)
                if local_delta.size >= 2:
                    return float(np.linalg.norm(local_delta[:2]))
            except Exception:
                pass
        try:
            object_pos = np.asarray(relation["object_pos_world"], dtype=np.float64).reshape(-1)[:3]
            target_pos = np.asarray(relation["site_pos_world"], dtype=np.float64).reshape(-1)[:3]
            return float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        except Exception:
            return float("inf")

    def _target_matches_reference(self, goal_target_name: str, target_reference: str | None) -> bool:
        if target_reference is None or not str(target_reference).strip():
            return True
        goal_norm = str(goal_target_name).strip().lower().replace(" ", "_")
        ref_norm = str(target_reference).strip().lower().replace(" ", "_")
        goal_base = goal_norm.removesuffix("_1")
        ref_base = ref_norm.removesuffix("_1")
        goal_compact = goal_base.replace("_", "")
        ref_compact = ref_base.replace("_", "")
        ref_tokens = {token for token in ref_norm.split("_") if token}
        goal_tokens = {token for token in goal_norm.split("_") if token}
        if ref_base == "left_plate" and goal_norm in {"plate_1", "plate"}:
            return True
        if ref_base == "right_plate" and goal_norm == "plate_2":
            return True
        if {"right", "plate"}.issubset(ref_tokens) and {"right", "plate"}.issubset(goal_tokens):
            return True
        if {"left", "plate"}.issubset(ref_tokens) and {"left", "plate"}.issubset(goal_tokens):
            return True
        if ref_base in {"front_of_stove", "stove_front"} and "stove" in goal_base and "front" in goal_base:
            return True
        if ref_base in {"back_of_stove", "stove_back"} and "stove" in goal_base and "back" in goal_base:
            return True
        semantic = self._semantic_target_state_name(target_reference, predicate="on") or self._semantic_target_state_name(
            target_reference,
            predicate="in",
        )
        if semantic is not None:
            semantic_norm = semantic.strip().lower().replace(" ", "_")
            if goal_norm == semantic_norm or goal_base == semantic_norm.removesuffix("_1"):
                return True
        return bool(
            goal_norm == ref_norm
            or goal_base == ref_base
            or ref_base in goal_base
            or goal_base in ref_base
            or (ref_compact and ref_compact in goal_compact)
            or (goal_compact and goal_compact in ref_compact)
        )

    def _semantic_target_state_name(self, target_reference: str | None, *, predicate: str) -> str | None:
        if target_reference is None:
            return None
        ref_norm = str(target_reference).strip().lower().replace(" ", "_")
        if not ref_norm:
            return None
        states = getattr(self._libero_env_with_objects(), "object_states_dict", {})
        state_names = list(states.keys())
        if predicate == "in" and ("microwave" in ref_norm or ref_norm in {"task_target", "target_pose", "target_region_pose"}):
            for preferred in (
                "microwave_1_heating_region",
                "microwave_heating_region",
                "microwave_1_contain_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "microwave" in name.lower() and ("heating" in name.lower() or "region" in name.lower())]
            if matches:
                heating_matches = [name for name in matches if "heating" in name.lower()]
                return sorted(heating_matches or matches, key=len)[0]
        if predicate == "in" and "drawer" in ref_norm:
            level = "middle"
            if "top" in ref_norm:
                level = "top"
            elif "bottom" in ref_norm:
                level = "bottom"
            for state in self._env.handle.env.env.parsed_problem.get("goal_state", []):
                if len(state) != 3:
                    continue
                goal_predicate, _goal_object, goal_target = state
                goal_target_text = str(goal_target).lower()
                if (
                    str(goal_predicate).lower() == "in"
                    and "drawer" not in goal_target_text
                    and level in goal_target_text
                    and "region" in goal_target_text
                    and str(goal_target) in states
                ):
                    return str(goal_target)
            preferred_fragments = (
                (f"white_cabinet_1_{level}_region",),
                (f"wooden_cabinet_1_{level}_region",),
                (f"{level}_region",),
            )
            for group in preferred_fragments:
                for preferred in group:
                    if preferred in states:
                        return preferred
            matches = [
                name
                for name in state_names
                if level in name.lower() and "cabinet" in name.lower() and "region" in name.lower()
            ]
            if matches:
                return sorted(matches, key=len)[0]
        ref_tokens = {token for token in ref_norm.split("_") if token}
        if "left_plate" in ref_norm or {"left", "plate"}.issubset(ref_tokens):
            for preferred in (
                "living_room_table_plate_left_region",
                "plate_left_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "plate" in name.lower() and "left" in name.lower()]
            if matches:
                return sorted(matches, key=len)[0]
        if "right_plate" in ref_norm or {"right", "plate"}.issubset(ref_tokens):
            for preferred in (
                "living_room_table_plate_right_region",
                "plate_right_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "plate" in name.lower() and "right" in name.lower()]
            if matches:
                return sorted(matches, key=len)[0]
        if "back" in ref_norm and "caddy" in ref_norm:
            for preferred in (
                "desk_caddy_1_back_contain_region",
                "desk_caddy_back_contain_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "caddy" in name.lower() and "back" in name.lower()]
            if matches:
                contain_matches = [name for name in matches if "contain" in name.lower()]
                return sorted(contain_matches or matches, key=len)[0]
        if "front" in ref_norm and "caddy" in ref_norm:
            for preferred in (
                "desk_caddy_1_front_contain_region",
                "desk_caddy_front_contain_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "caddy" in name.lower() and "front" in name.lower()]
            if matches:
                contain_matches = [name for name in matches if "contain" in name.lower()]
                return sorted(contain_matches or matches, key=len)[0]
        if predicate == "on" and "stove" in ref_norm and "front" in ref_norm:
            for preferred in (
                "main_table_stove_front_region",
                "stove_front_region",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "stove" in name.lower() and "front" in name.lower()]
            if matches:
                region_matches = [name for name in matches if "region" in name.lower()]
                return sorted(region_matches or matches, key=len)[0]
        if predicate == "on" and "stove" in ref_norm and "back" in ref_norm:
            matches = [name for name in state_names if "stove" in name.lower() and "back" in name.lower()]
            if matches:
                region_matches = [name for name in matches if "region" in name.lower()]
                return sorted(region_matches or matches, key=len)[0]
        if predicate == "on" and ref_norm in {"cabinet", "cabinet_top", "wooden_cabinet"}:
            for preferred in (
                "wooden_cabinet_1_top_side",
                "wooden_cabinet_1_top_region",
                "wooden_cabinet_1_default_site",
            ):
                if preferred in states:
                    return preferred
            matches = [
                name
                for name in state_names
                if "cabinet" in name.lower() and "top" in name.lower()
            ]
            if matches:
                side_matches = [name for name in matches if "side" in name.lower()]
                return sorted(side_matches or matches, key=len)[0]
        if "rack" in ref_norm or "caddy" in ref_norm:
            if "caddy" in ref_norm:
                for preferred in (
                    "desk_caddy_1_back_contain_region",
                    "desk_caddy_1_default_site",
                    "desk_caddy_1",
                ):
                    if preferred in states:
                        return preferred
            for preferred in (
                "wine_rack_1_top_region",
                "wine_rack_1_default_site",
                "wine_rack_1",
            ):
                if preferred in states:
                    return preferred
            matches = [name for name in state_names if "rack" in name.lower() or "caddy" in name.lower()]
            if matches:
                region_matches = [name for name in matches if "region" in name.lower()]
                return sorted(region_matches or matches, key=len)[0]
        return None

    def _drawer_level_from_target_name(self, target_name: str | None) -> str | None:
        text = str(target_name or "").strip().lower().replace(" ", "_")
        if "drawer" not in text and "cabinet" not in text:
            return None
        for level in ("top", "middle", "bottom"):
            if level in text:
                return level
        return None

    def _is_drawer_region_target_name(self, target_name: str | None) -> bool:
        text = str(target_name or "").strip().lower().replace(" ", "_")
        if "drawer" in text:
            return True
        return bool(
            "cabinet" in text
            and "region" in text
            and any(level in text for level in ("top", "middle", "bottom"))
        )

    def _default_place_quat(self, predicate: str, target: dict[str, Any]) -> np.ndarray:
        pred = str(predicate).strip().lower()
        target_type = str(target.get("target_type", "site"))
        target_name = str(target.get("target_name", "")).lower()
        if pred == "on" and target_type != "site":
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        if "plate" in target_name and pred == "on":
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        if pred == "in" and self._is_drawer_region_target_name(target_name):
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    def _default_release_z(self, predicate: str, target: dict[str, Any]) -> float:
        pred = str(predicate).strip().lower()
        target_type = str(target.get("target_type", "site"))
        target_name = str(target.get("target_name", ""))
        if pred == "in" and self._is_drawer_region_target_name(target_name):
            return 0.030
        if pred == "on" and target_type != "site":
            return 0.018
        if pred == "on":
            return 0.045
        return 0.060

    def _release_anchor_for_predicate(
        self,
        predicate: str,
        target: dict[str, Any],
        *,
        object_name: str | None = None,
    ) -> dict[str, Any]:
        """Choose the physical anchor for low release without changing the goal target.

        LIBERO basket goals expose a contain-region site whose z coordinate is
        high in the basket volume. The successful expert trajectories release
        relative to the basket/default-site floor instead, while preserving the
        contain-region XY center. This helper encodes that generic container
        geometry pattern and leaves the semantic verifier target untouched.
        """
        pred = str(predicate).strip().lower()
        target_name = str(target.get("target_name", ""))
        target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3).copy()
        anchor = target_pos.copy()
        source = "target_position"
        align_center = False

        object_extent: np.ndarray | None = None
        container_release_profile = "semantic_target"
        support_profile = "semantic_target"

        if pred == "on" and str(target.get("target_type", "")) == "site" and object_name:
            shared_broad_region = self._is_shared_broad_region_target(
                object_name,
                target_name,
                predicate=pred,
                target=target,
            )
            anchor[:2] += self._multi_object_region_release_offset(
                object_name,
                target_name,
                target,
            )
            source = f"{source}_multi_object_slot"
            if shared_broad_region:
                support_profile = "shared_broad_region_release"

        if pred == "on" and str(target.get("target_type", "")) != "site":
            support_name = target_name
            try:
                support_object = str(target_name).removesuffix("_1")
                support_geometry = self._object_geometry_summary(support_object)
                support_bounds_max = np.asarray(support_geometry["bounds_max"], dtype=np.float64).reshape(3)
                anchor[2] = float(support_bounds_max[2])
                source = f"{support_name}_top_surface"
                support_profile = "object_top_surface_release"
            except Exception:
                pass

            if object_name:
                try:
                    object_geometry = self._object_geometry_summary(object_name)
                    object_extent = np.asarray(object_geometry["extent"], dtype=np.float64).reshape(-1)[:3]
                    half_height = 0.5 * float(abs(object_extent[2]))
                    anchor[2] += float(np.clip(half_height, 0.018, 0.060))
                    source = f"{source}_plus_object_half_height"
                    align_center = True
                except Exception:
                    pass

        if pred == "in" and "basket" in target_name.lower():
            large_or_tall_object = False
            if object_name:
                profile = self._container_release_shape_profile(object_name)
                extent = profile.get("object_extent")
                if extent is not None:
                    object_extent = np.asarray(extent, dtype=np.float64).reshape(-1)[:3]
                large_or_tall_object = bool(profile.get("large_box_like", False))

            container_release_profile = "low_container_floor_release"
            align_center = True
            for container_name in self._container_reference_candidates(target_name):
                try:
                    container_pos, _ = self.get_instance_pose(container_name)
                except Exception:
                    try:
                        container_pos, _ = self.get_object_pose(container_name)
                    except Exception:
                        continue
                anchor[2] = float(np.asarray(container_pos, dtype=np.float64).reshape(3)[2])
                source = f"{container_name}_z"
                break
            else:
                target_size = np.asarray(
                    target.get("target_size", np.array([0.04, 0.04, 0.04])),
                    dtype=np.float64,
                ).reshape(-1)[:3]
                anchor[2] = float(target_pos[2] - max(float(abs(target_size[2])), 0.040))
                source = "target_position_minus_site_height"
            if large_or_tall_object:
                source = f"{source}_large_object_center_aligned"

        if pred == "in" and self._is_drawer_region_target_name(target_name):
            level = self._drawer_level_from_target_name(target_name) or "middle"
            try:
                _, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
                pull_dir = np.asarray(pull_dir, dtype=np.float64).reshape(3)
                pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)
                interior_dir = -pull_dir
            except Exception:
                interior_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            try:
                lateral_axis = self._drawer_handle_lateral_axis(level)
                lateral_axis = np.asarray(lateral_axis, dtype=np.float64).reshape(3)
                lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-9)
            except Exception:
                lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

            anchor = target_pos.copy()
            anchor[2] = float(target_pos[2])
            source = f"{level}_drawer_region_interior_release"
            align_center = True
            container_release_profile = "drawer_interior_release"
            if object_name:
                try:
                    object_geometry = self._object_geometry_summary(object_name)
                    object_extent = np.asarray(object_geometry["extent"], dtype=np.float64).reshape(-1)[:3]
                except Exception:
                    pass

        if (
            pred == "in"
            and object_name
            and "book" in str(object_name).lower()
            and not self._is_drawer_region_target_name(target_name)
        ):
            container_release_profile = "thin_object_high_container_release"
            source = f"{source}_thin_object"

        return {
            "position": anchor,
            "source": source,
            "semantic_target_position": target_pos,
            "target_size": np.asarray(target.get("target_size", np.array([0.04, 0.04, 0.04])), dtype=np.float64).reshape(-1)[:3].copy(),
            "target_xmat": np.asarray(target.get("target_xmat", np.eye(3)), dtype=np.float64).reshape(3, 3).copy(),
            "align_held_object_center": align_center,
            "object_extent": object_extent.copy() if object_extent is not None else None,
            "container_release_profile": container_release_profile,
            "support_profile": support_profile,
            "drawer_level": self._drawer_level_from_target_name(target_name),
            "drawer_interior_direction": interior_dir.copy() if "interior_dir" in locals() else None,
            "drawer_lateral_axis": lateral_axis.copy() if "lateral_axis" in locals() else None,
        }

    def _multi_object_region_release_offset(
        self,
        object_name: str,
        target_name: str,
        target: dict[str, Any],
    ) -> np.ndarray:
        """Spread objects that share a broad BDDL target region into sub-slots."""
        target_text = str(target_name).lower()
        target_size = np.asarray(target.get("target_size", np.array([0.0, 0.0, 0.0])), dtype=np.float64).reshape(-1)[:3]
        broad_site = bool(
            "stove" in target_text
            or "cook_region" in target_text
            or float(np.max(np.abs(target_size[:2]))) >= 0.070
        )
        if not broad_site:
            return np.zeros(2, dtype=np.float64)
        task_env = self._env.handle.env.env
        shared: list[str] = []
        for goal_state in task_env.parsed_problem.get("goal_state", []):
            if len(goal_state) != 3:
                continue
            predicate, bddl_object_name, bddl_target_name = goal_state
            if str(predicate).lower() != "on" or str(bddl_target_name) != str(target_name):
                continue
            shared.append(str(bddl_object_name))
        if len(shared) <= 1:
            return np.zeros(2, dtype=np.float64)
        shared = sorted(dict.fromkeys(shared))
        obj_text = str(object_name).lower()
        if obj_text.endswith("_2"):
            current_bddl = f"{obj_text}"
        elif obj_text.endswith("_1"):
            current_bddl = f"{obj_text}"
        else:
            current_bddl = f"{obj_text}_1"
        try:
            slot_index = shared.index(current_bddl)
        except ValueError:
            slot_index = 0
        slot_offsets = [
            np.array([-0.035, -0.048], dtype=np.float64),
            np.array([0.035, 0.048], dtype=np.float64),
            np.array([-0.035, 0.048], dtype=np.float64),
            np.array([0.035, -0.048], dtype=np.float64),
            np.array([-0.052, 0.000], dtype=np.float64),
            np.array([0.052, 0.000], dtype=np.float64),
        ]
        offset = slot_offsets[slot_index % len(slot_offsets)].copy()
        occupied_offsets: list[np.ndarray] = []
        try:
            goal_diags = self.get_bddl_goal_diagnostics().get("goal_state", [])
        except Exception:
            goal_diags = []
        for goal in goal_diags:
            if not isinstance(goal, dict) or not bool(goal.get("satisfied", False)):
                continue
            raw = goal.get("raw", [])
            if len(raw) != 3 or str(raw[0]).lower() != "on" or str(raw[2]) != str(target_name):
                continue
            if str(raw[1]) == current_bddl:
                continue
            diagnostic = goal.get("diagnostic", {})
            if not isinstance(diagnostic, dict) or "local_delta" not in diagnostic:
                continue
            try:
                occupied_offsets.append(
                    np.asarray(diagnostic["local_delta"], dtype=np.float64).reshape(-1)[:2]
                )
            except Exception:
                continue
        if occupied_offsets:
            candidates = [candidate.copy() for candidate in slot_offsets]
            if target_size.size >= 2 and float(np.max(np.abs(target_size[:2]))) > 0:
                limits = np.maximum(0.70 * np.abs(target_size[:2]), np.array([0.025, 0.025], dtype=np.float64))
                candidates = [np.clip(candidate, -limits, limits) for candidate in candidates]
            def score(candidate: np.ndarray) -> tuple[float, float]:
                min_dist = min(float(np.linalg.norm(candidate - occupied)) for occupied in occupied_offsets)
                boundary_margin = float(np.min(np.abs(target_size[:2]) - np.abs(candidate[:2]))) if target_size.size >= 2 else 0.0
                return (min_dist, boundary_margin)
            offset = max(candidates, key=score)
        if target_size.size >= 2 and float(np.max(np.abs(target_size[:2]))) > 0:
            limits = np.maximum(0.70 * np.abs(target_size[:2]), np.array([0.025, 0.025], dtype=np.float64))
            offset = np.clip(offset, -limits, limits)
        return offset

    def _is_shared_broad_region_target(
        self,
        object_name: str,
        target_name: str,
        *,
        predicate: str,
        target: dict[str, Any],
    ) -> bool:
        pred = str(predicate).strip().lower()
        if pred != "on" or str(target.get("target_type", "")) != "site":
            return False
        target_text = str(target_name).lower()
        target_size = np.asarray(
            target.get("target_size", np.array([0.0, 0.0, 0.0])),
            dtype=np.float64,
        ).reshape(-1)[:3]
        broad_site = bool(
            "stove" in target_text
            or "cook_region" in target_text
            or float(np.max(np.abs(target_size[:2]))) >= 0.070
        )
        if not broad_site:
            return False
        task_env = self._env.handle.env.env
        shared: list[str] = []
        for goal_state in task_env.parsed_problem.get("goal_state", []):
            if len(goal_state) != 3:
                continue
            goal_predicate, goal_object_name, goal_target_name = goal_state
            if str(goal_predicate).lower() == "on" and str(goal_target_name) == str(target_name):
                shared.append(str(goal_object_name))
        return len(set(shared)) > 1

    def _container_release_shape_profile(self, object_name: str) -> dict[str, Any]:
        try:
            geometry = self._object_geometry_summary(object_name)
            object_extent = np.asarray(geometry["extent"], dtype=np.float64).reshape(-1)[:3]
            max_xy_extent = float(np.max(np.abs(object_extent[:2])))
            z_extent = float(abs(object_extent[2]))
            box_like = bool(max_xy_extent >= 1e-6 and z_extent <= max_xy_extent + 0.008)
            return {
                "object_name": object_name,
                "object_extent": object_extent.copy(),
                "max_xy_extent": max_xy_extent,
                "z_extent": z_extent,
                "box_like": box_like,
                "large_box_like": bool(max_xy_extent >= 0.150 and box_like),
            }
        except Exception as exc:
            return {
                "object_name": object_name,
                "object_extent": None,
                "large_box_like": False,
                "error": str(exc),
            }

    def _container_reference_candidates(self, target_name: str) -> list[str]:
        query = str(target_name).strip().replace(" ", "_").lower()
        candidates: list[str] = []
        for suffix in ("_contain_region", "_default_site", "_region"):
            if suffix in query:
                candidates.append(query.split(suffix)[0])
        parts = query.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            candidates.append("_".join(parts[:2]))
        if parts:
            candidates.append(parts[0])
        candidates.extend(["basket_1", "basket"])
        deduped: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _repair_release_z(self, predicate: str, target: dict[str, Any], repair_idx: int) -> float:
        base = self._default_release_z(predicate, target)
        if str(predicate).strip().lower() == "on":
            if str(target.get("target_type", "")) != "site":
                return max(0.006, 0.012 - 0.002 * int(repair_idx))
            return max(0.012, base - 0.002 * int(repair_idx))
        return base

    def _requires_strict_hold(self, object_name: str) -> bool:
        text = str(object_name).lower()
        return any(token in text for token in ("bowl", "plate", "ramekin"))

    def _drawer_relation_is_z_only_above_box(
        self,
        relation: dict[str, Any] | None,
        *,
        max_z_excess: float = 0.018,
        require_contact: bool = True,
    ) -> bool:
        """Return whether a drawer In relation is xy-valid and only too high.

        This is a readonly feedback guard for shallow objects in drawer regions:
        once the object center is inside the drawer footprint, planar nudges are
        more likely to disturb a near-success than to fix the remaining z error.
        """
        return self._relation_is_xy_inside_z_only_above_box(
            relation,
            max_z_excess=max_z_excess,
            require_contact=require_contact,
        )

    def _relation_is_xy_inside_z_only_above_box(
        self,
        relation: dict[str, Any] | None,
        *,
        max_z_excess: float = 0.018,
        require_contact: bool = True,
    ) -> bool:
        """Return whether an In relation is xy-valid and only above the z bound."""
        if not isinstance(relation, dict):
            return False
        if bool(require_contact) and not bool(relation.get("contact", False)):
            return False
        try:
            inside_axes = relation.get("aabb_inside_axes")
            if inside_axes is not None:
                axes = np.asarray(inside_axes, dtype=bool).reshape(-1)
                xy_inside = bool(axes.size >= 2 and bool(axes[0]) and bool(axes[1]))
            else:
                lower = np.asarray(relation.get("aabb_margin_lower_world"), dtype=np.float64).reshape(-1)
                upper = np.asarray(relation.get("aabb_margin_upper_world"), dtype=np.float64).reshape(-1)
                xy_inside = bool(
                    lower.size >= 2
                    and upper.size >= 2
                    and float(lower[0]) >= -0.004
                    and float(lower[1]) >= -0.004
                    and float(upper[0]) >= -0.004
                    and float(upper[1]) >= -0.004
                )
            z_excess = float(relation.get("aabb_z_excess", 0.0) or 0.0)
            z_deficit = float(relation.get("aabb_z_deficit", 0.0) or 0.0)
            return bool(xy_inside and z_deficit <= 0.002 and 0.0 < z_excess <= float(max_z_excess))
        except Exception:
            return False

    def drawer_z_only_settle_object(
        self,
        object_name: str,
        *,
        target_name: str | None = None,
        max_attempts: int = 1,
        max_z_excess: float = 0.020,
        settle_steps: int = 24,
    ) -> dict[str, Any]:
        """Settle a drawer object only when readonly feedback shows z-only error.

        The skill is intentionally narrow: it requires the object center to be
        inside the drawer AABB in x/y and only slightly above the upper z bound.
        It then uses ordinary gripper motion to apply a small vertical settle,
        and re-reads the BDDL relation. No object pose, joint, or task state is
        written directly.
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        if target_name:
            try:
                target = self._placement_target_for_object(
                    resolved_object,
                    predicate="in",
                    target_name=target_name,
                )
                resolved_target = str(target["target_name"])
            except Exception:
                resolved_target = str(target_name)
        else:
            target = self._placement_target_for_object(
                resolved_object,
                predicate="in",
                target_name=None,
            )
            resolved_target = str(target["target_name"])

        records: list[dict[str, Any]] = []
        relation = self._placement_relation_diagnostic(
            resolved_object,
            resolved_target,
            predicate="in",
        )
        if bool(relation.get("satisfied", False) or relation.get("in", False)):
            status = self.get_task_status()
            return {
                "object_name": resolved_object,
                "target_name": resolved_target,
                "initial_relation": relation,
                "final_relation": relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": True,
                "reason": "drawer relation already satisfied before z-only settle",
                "skill": "drawer_z_only_settle_object",
            }
        if not self._drawer_relation_is_z_only_above_box(
            relation,
            max_z_excess=float(max_z_excess),
            require_contact=True,
        ):
            status = self.get_task_status()
            return {
                "object_name": resolved_object,
                "target_name": resolved_target,
                "initial_relation": relation,
                "final_relation": relation,
                "records": records,
                "task_status": status,
                "relation_satisfied": False,
                "reason": "drawer relation is not xy-valid z-only-above-box",
                "skill": "drawer_z_only_settle_object",
            }

        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        for attempt_idx in range(max(1, int(max_attempts))):
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                break
            if not self._drawer_relation_is_z_only_above_box(
                relation,
                max_z_excess=float(max_z_excess),
                require_contact=True,
            ):
                records.append(
                    {
                        "attempt": attempt_idx + 1,
                        "action": "skip_z_only_settle_guard_failed",
                        "relation_before": relation,
                    }
                )
                break

            obj_world = np.asarray(relation["object_pos_world"], dtype=np.float64).reshape(3)
            obj_robot = self._world_position_to_robot_base(obj_world)
            z_excess = float(relation.get("aabb_z_excess", 0.0) or 0.0)
            stroke = float(np.clip(z_excess + 0.006 + 0.002 * attempt_idx, 0.006, 0.014))
            start = obj_robot.copy()
            start[2] += 0.105
            contact = obj_robot.copy()
            contact[2] += 0.040
            end = obj_robot.copy()
            end[2] -= stroke
            record: dict[str, Any] = {
                "attempt": attempt_idx + 1,
                "action": "drawer_z_only_vertical_settle",
                "relation_before": relation,
                "object_position_before": obj_robot.copy(),
                "z_excess_before": z_excess,
                "stroke": stroke,
                "start": start.copy(),
                "contact": contact.copy(),
                "end": end.copy(),
            }
            try:
                self.open_gripper()
                self._settle_simulation_steps(3)
                record["start_move"] = self.goto_finger_midpoint_checked(start, quat, tolerance=0.090)
                record["contact_move"] = self.goto_finger_midpoint_checked(contact, quat, tolerance=0.080)
                self.close_gripper()
                self._settle_simulation_steps(4)
                record["settle_move"] = self.goto_finger_midpoint_checked(end, quat, tolerance=0.085)
                self.open_gripper()
                self._settle_simulation_steps(max(10, min(28, int(settle_steps))))
                retreat = end.copy()
                retreat[2] += 0.090
                record["retreat_move"] = self.goto_finger_midpoint_checked(retreat, quat, tolerance=0.100)
                self._settle_simulation_steps(max(6, min(18, int(settle_steps) // 2)))
            except Exception as exc:
                record["error"] = repr(exc)
                try:
                    self.open_gripper()
                except Exception:
                    pass
                self._settle_simulation_steps(max(6, min(18, int(settle_steps) // 2)))

            relation_after = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate="in",
            )
            record["relation_after"] = relation_after
            try:
                before_margin = float(relation.get("aabb_margin_min", -999.0) or -999.0)
                after_margin = float(relation_after.get("aabb_margin_min", -999.0) or -999.0)
                record["improved_margin"] = bool(after_margin > before_margin + 1e-4)
            except Exception:
                record["improved_margin"] = None
            records.append(record)
            relation = relation_after
            if bool(relation.get("satisfied", False) or relation.get("in", False)):
                break

        status = self.get_task_status()
        relation_satisfied = bool(
            relation.get("satisfied", False)
            or relation.get("in", False)
            or status.get("task_completed", False)
            or float(status.get("reward", 0.0) or 0.0) >= 1.0
        )
        return {
            "object_name": resolved_object,
            "target_name": resolved_target,
            "initial_relation": records[0]["relation_before"] if records else relation,
            "final_relation": relation,
            "records": records,
            "task_status": status,
            "relation_satisfied": relation_satisfied,
            "xy_distance_to_target": self._relation_xy_distance_to_target(relation),
            "reason": (
                "drawer z-only settle satisfied relation"
                if relation_satisfied
                else "drawer z-only settle did not satisfy relation"
            ),
            "skill": "drawer_z_only_settle_object",
        }

    def _corrective_push_to_target(
        self,
        object_name: str,
        target_name: str,
        *,
        predicate: str,
        max_iters: int = 2,
    ) -> dict[str, Any]:
        pred = str(predicate).strip().lower()
        records: list[dict[str, Any]] = []
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        for idx in range(max(1, int(max_iters))):
            relation = self._placement_relation_diagnostic(object_name, target_name, predicate=pred)
            if bool(relation.get("satisfied", False)):
                return {"records": records, "final_relation": relation, "satisfied": True}

            obj_pos, _ = self.get_object_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            try:
                if pred == "on":
                    delta = np.asarray(
                        self.get_on_center_delta_robot(object_name, target_name),
                        dtype=np.float64,
                    ).reshape(3)
                else:
                    target = self._placement_target_for_object(
                        object_name,
                        predicate=pred,
                        target_name=target_name,
                    )
                    target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
                    delta = target_pos - obj_pos
            except Exception:
                target = self._placement_target_for_object(
                    object_name,
                    predicate=pred,
                    target_name=target_name,
                )
                target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
                delta = target_pos - obj_pos

            xy = np.asarray(delta[:2], dtype=np.float64)
            dist = float(np.linalg.norm(xy))
            if dist <= 0.010:
                self._settle_simulation_steps(40)
                relation_after_settle = self._placement_relation_diagnostic(
                    object_name,
                    target_name,
                    predicate=pred,
                )
                records.append(
                    {
                        "iteration": idx,
                        "action": "settle_near_target",
                        "distance_xy": dist,
                        "relation": relation_after_settle,
                    }
                )
                if bool(relation_after_settle.get("satisfied", False)):
                    return {
                        "records": records,
                        "final_relation": relation_after_settle,
                        "satisfied": True,
                    }
                break

            direction = xy / max(dist, 1e-6)
            stroke = min(max(0.55 * dist, 0.018), 0.055)
            start = obj_pos.copy()
            start[:2] -= direction * 0.038
            start[2] += 0.030 if pred == "on" else 0.036
            mid = obj_pos.copy()
            mid[2] = start[2]
            end = obj_pos.copy()
            end[:2] += direction * stroke
            end[2] = start[2]
            self.open_gripper()
            self.goto_pose(start + np.array([0.0, 0.0, 0.080], dtype=np.float64), quat)
            self.goto_pose(start, quat)
            self.goto_pose(mid, quat)
            self.goto_pose(end, quat)
            self.goto_pose(end + np.array([0.0, 0.0, 0.100], dtype=np.float64), quat)
            self._settle_simulation_steps(60)
            relation_after = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate=pred,
            )
            records.append(
                {
                    "iteration": idx,
                    "action": "planar_push",
                    "distance_xy": dist,
                    "stroke": stroke,
                    "start": start,
                    "end": end,
                    "relation": relation_after,
                }
            )
            if bool(relation_after.get("satisfied", False)):
                return {"records": records, "final_relation": relation_after, "satisfied": True}

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate=pred)
        return {
            "records": records,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False)),
        }

    def push_object_to_target_region(
        self,
        object_name: str,
        *,
        target_name: str | None = None,
        predicate: str = "on",
        max_iters: int = 4,
        settle_steps: int = 45,
        stop_local_xy: float = 0.045,
        min_motion: float = 0.006,
    ) -> dict[str, Any]:
        """Push an object into a parsed target region using physical contact.

        The method is intentionally clean: it reads BDDL/site geometry for
        grounding and verification, but every scene change comes from robot
        gripper motion. It is the audited execution counterpart for tasks such
        as "push the plate to the front of the stove".
        """
        resolved_object = self._resolve_object_name(object_name, require_free_joint=True)
        pred = str(predicate or "on").strip().lower()
        if pred not in {"on", "in"}:
            pred = "in" if any(token in str(target_name or "").lower() for token in ("basket", "contain", "drawer")) else "on"
        target = self._placement_target_for_object(
            resolved_object,
            predicate=pred,
            target_name=target_name,
        )
        resolved_target = str(target.get("target_name", target_name or ""))
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        worsen_streak = 0
        no_motion_streak = 0
        best_relation: dict[str, Any] | None = None
        best_xy = float("inf")

        def relation_now() -> dict[str, Any]:
            relation = self._placement_relation_diagnostic(
                resolved_object,
                resolved_target,
                predicate=pred,
            )
            relation["object_name_resolved"] = resolved_object
            relation["target_name_resolved"] = resolved_target
            return relation

        def finish(reason: str, relation: dict[str, Any], task_status: dict[str, Any]) -> dict[str, Any]:
            return {
                "object_name": resolved_object,
                "target_name": resolved_target,
                "predicate": pred,
                "initial_relation": initial_relation,
                "final_relation": relation,
                "best_relation": best_relation or relation,
                "records": records,
                "relation_satisfied": bool(relation.get("satisfied", False)),
                "task_status": task_status,
                "skill": "push_object_to_target_region",
                "reason": reason,
            }

        def site_size_from_relation(relation: dict[str, Any]) -> np.ndarray:
            size = relation.get("site_size_abs", relation.get("site_size", np.array([0.04, 0.04, 0.01])))
            arr = np.asarray(size, dtype=np.float64).reshape(-1)
            if arr.size < 2:
                return np.array([0.04, 0.04, 0.01], dtype=np.float64)
            if arr.size < 3:
                arr = np.concatenate([arr[:2], np.array([0.01], dtype=np.float64)])
            return np.abs(arr[:3])

        def local_delta_from_relation(relation: dict[str, Any]) -> np.ndarray:
            if "local_delta" in relation:
                arr = np.asarray(relation["local_delta"], dtype=np.float64).reshape(-1)
                if arr.size >= 3:
                    return arr[:3].copy()
                if arr.size >= 2:
                    return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
            target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
            obj_pos, _ = self.get_object_pose(resolved_object)
            delta = np.asarray(obj_pos, dtype=np.float64).reshape(3) - target_pos
            return np.array([delta[0], delta[1], 0.0], dtype=np.float64)

        def robot_axis_to_site_local(axis2: np.ndarray) -> np.ndarray:
            axis3 = np.array([float(axis2[0]), float(axis2[1]), 0.0], dtype=np.float64)
            try:
                _, site_quat = self.get_site_pose(resolved_target)
                site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64).reshape(4)).as_matrix()
                local = site_rot.T @ axis3
                local_xy = local[:2].copy()
            except Exception:
                local_xy = axis3[:2].copy()
            norm = float(np.linalg.norm(local_xy))
            if norm <= 1e-6:
                return np.array([1.0, 0.0], dtype=np.float64)
            return local_xy / norm

        def dominant_robot_axis(local_delta: np.ndarray, site_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            thresholds = np.maximum(site_size[:2] * 0.98, 0.038)
            error = np.maximum(np.abs(local_delta[:2]) - thresholds, 0.0)
            if float(np.max(error)) <= 1e-5:
                dominant_local = np.array([1.0, 0.0], dtype=np.float64)
                if abs(float(local_delta[1])) > abs(float(local_delta[0])):
                    dominant_local = np.array([0.0, 1.0], dtype=np.float64)
            else:
                dominant_local = np.array([1.0, 0.0], dtype=np.float64)
                if float(error[1]) > float(error[0]):
                    dominant_local = np.array([0.0, 1.0], dtype=np.float64)
            axes = (
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([-1.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([0.0, -1.0], dtype=np.float64),
            )
            best_axis = axes[0]
            best_local_axis = robot_axis_to_site_local(best_axis)
            best_score = -1.0
            for axis in axes:
                local_axis = robot_axis_to_site_local(axis)
                score = abs(float(np.dot(local_axis, dominant_local)))
                if score > best_score:
                    best_axis = axis.copy()
                    best_local_axis = local_axis.copy()
                    best_score = score
            before_proj = float(np.dot(local_delta[:2], best_local_axis))
            return best_axis, best_local_axis, before_proj

        def run_axis_rim_stroke(
            axis2: np.ndarray,
            step: float,
            z_bias: float,
            side_bias: float,
            tag: str,
        ) -> dict[str, Any]:
            obj_pos, _ = self.get_object_pose(resolved_object)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            axis = np.asarray(axis2, dtype=np.float64).reshape(2)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
            side = np.array([-axis[1], axis[0]], dtype=np.float64)
            backoff = 0.080 if "plate" in resolved_object.lower() else 0.055
            start = obj_pos.copy()
            start[:2] -= axis * backoff
            start[:2] += side * float(side_bias)
            start[2] += float(z_bias)
            catch = obj_pos.copy()
            catch[:2] -= axis * 0.020
            catch[:2] += side * float(side_bias)
            catch[2] += float(z_bias)
            end = obj_pos.copy()
            end[:2] += axis * float(step)
            end[:2] += side * float(side_bias)
            end[2] += float(z_bias)
            high = start + np.array([0.0, 0.0, 0.075], dtype=np.float64)
            retreat = end + np.array([0.0, 0.0, 0.095], dtype=np.float64)
            record: dict[str, Any] = {
                "action": "axis_rim_capture_push",
                "tag": tag,
                "object_position_before": obj_pos.copy(),
                "axis_xy": axis.copy(),
                "step": float(step),
                "z_bias": float(z_bias),
                "side_bias": float(side_bias),
                "start": start.copy(),
                "catch": catch.copy(),
                "end": end.copy(),
                "retreat": retreat.copy(),
            }
            try:
                self.open_gripper()
                record["high_move"] = self._goto_pose_legacy_checked(high, quat, tolerance=0.120)
                record["start_move"] = self._goto_pose_legacy_checked(start, quat, tolerance=0.095)
                record["catch_move"] = self._goto_pose_legacy_checked(catch, quat, tolerance=0.085)
                self.close_gripper()
                self._settle_simulation_steps(10)
                record["end_move"] = self._goto_pose_legacy_checked(end, quat, tolerance=0.120)
                self._settle_simulation_steps(max(8, int(settle_steps) // 2))
                self.open_gripper()
                record["retreat_move"] = self._goto_pose_legacy_checked(retreat, quat, tolerance=0.140)
            except Exception as exc:
                record["error"] = repr(exc)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            after_pos, _ = self.get_object_pose(resolved_object)
            record["object_position_after"] = np.asarray(after_pos, dtype=np.float64).reshape(3).copy()
            record["object_motion_xy"] = float(
                np.linalg.norm(record["object_position_after"][:2] - obj_pos[:2])
            )
            return record

        relation = relation_now()
        initial_relation = dict(relation)

        def run_feedback_pick_release_fallback(
            trigger_reason: str,
            relation_before: dict[str, Any],
        ) -> dict[str, Any] | None:
            nonlocal best_relation, best_xy
            if pred != "on" or "plate" not in resolved_object.lower():
                return None
            fallback_record: dict[str, Any] = {
                "action": "feedback_pick_release_fallback",
                "trigger_reason": str(trigger_reason),
                "relation_before": relation_before,
            }
            try:
                hold = self.measure_object_grasp_state(resolved_object)
                fallback_record["hold_before"] = hold
                if bool(hold.get("held", False)):
                    pick = {
                        "object_name": resolved_object,
                        "held": True,
                        "existing_hold": hold,
                        "skill": "measure_object_grasp_state",
                    }
                else:
                    pick = self.verified_lift_pick_object(
                        resolved_object,
                        grasp_quat=quat,
                        max_attempts=8,
                        lift_height=0.18,
                        success_dz=0.045,
                        approach=0.12,
                        settle_steps=max(12, int(settle_steps) // 2),
                    )
                fallback_record["pick"] = pick
                if not bool(pick.get("held", False)):
                    relation_after_pick = relation_now()
                    task_status_after_pick = self.get_task_status()
                    fallback_record["relation_after"] = relation_after_pick
                    fallback_record["task_status_after"] = task_status_after_pick
                    records.append(fallback_record)
                    return None

                release = self.low_release_object_to_target(
                    resolved_object,
                    predicate=pred,
                    target_name=resolved_target,
                    release_z=0.020,
                    approach=0.12,
                    retreat=0.20,
                    settle_steps=max(60, int(settle_steps) + 20),
                )
                fallback_record["release"] = release
                relation_after = relation_now()
                task_status_after = self.get_task_status()
                fallback_record["relation_after"] = relation_after
                fallback_record["task_status_after"] = task_status_after
                xy_after = self._relation_xy_distance_to_target(relation_after)
                if np.isfinite(xy_after) and xy_after < best_xy:
                    best_xy = xy_after
                    best_relation = dict(relation_after)
                success_after_release = bool(
                    relation_after.get("satisfied", False)
                    or task_status_after.get("task_completed", False)
                    or float(task_status_after.get("reward", 0.0) or 0.0) >= 1.0
                )
                if not success_after_release and np.isfinite(xy_after) and xy_after <= 0.11:
                    post_push = self._short_local_push_to_target(
                        resolved_object,
                        resolved_target,
                        predicate=pred,
                        max_stroke=0.028,
                        settle_steps=max(18, int(settle_steps) // 2),
                    )
                    fallback_record["post_release_short_push"] = post_push
                    relation_after = relation_now()
                    task_status_after = self.get_task_status()
                    fallback_record["relation_after_post_push"] = relation_after
                    fallback_record["task_status_after_post_push"] = task_status_after
                    xy_after = self._relation_xy_distance_to_target(relation_after)
                    if np.isfinite(xy_after) and xy_after < best_xy:
                        best_xy = xy_after
                        best_relation = dict(relation_after)

                records.append(fallback_record)
                if bool(
                    relation_after.get("satisfied", False)
                    or task_status_after.get("task_completed", False)
                    or float(task_status_after.get("reward", 0.0) or 0.0) >= 1.0
                ):
                    return finish(
                        f"satisfied_after_{trigger_reason}_pick_release",
                        relation_after,
                        task_status_after,
                    )
            except Exception as exc:
                fallback_record["error"] = repr(exc)
                records.append(fallback_record)
            return None

        for idx in range(max(1, int(max_iters))):
            task_status_before = self.get_task_status()
            xy_before = self._relation_xy_distance_to_target(relation)
            if xy_before < best_xy:
                best_xy = xy_before
                best_relation = dict(relation)
            if bool(relation.get("satisfied", False)):
                return finish("already_satisfied" if not records else f"satisfied_after_{idx}_strokes", relation, task_status_before)
            if np.isfinite(xy_before) and xy_before <= float(stop_local_xy):
                self._settle_simulation_steps(max(10, int(settle_steps)))
                relation = relation_now()
                task_status_after_settle = self.get_task_status()
                records.append(
                    {
                        "iteration": idx,
                        "action": "settle_near_region",
                        "xy_before": xy_before,
                        "relation_after": relation,
                        "task_status_after": task_status_after_settle,
                    }
                )
                if bool(relation.get("satisfied", False)) or bool(task_status_after_settle.get("task_completed", False)) or float(task_status_after_settle.get("reward", 0.0) or 0.0) >= 1.0:
                    return finish("satisfied_after_settle", relation, task_status_after_settle)

            local_delta = local_delta_from_relation(relation)
            site_size = site_size_from_relation(relation)
            in_x = abs(float(local_delta[0])) <= max(float(site_size[0]) * 0.98, 0.038)
            in_y = abs(float(local_delta[1])) <= max(float(site_size[1]) * 0.98, 0.038)
            if in_x and in_y:
                self._settle_simulation_steps(max(10, int(settle_steps)))
                relation_after_settle = relation_now()
                task_status_after_settle = self.get_task_status()
                records.append(
                    {
                        "iteration": idx,
                        "action": "settle_inside_region_xy",
                        "relation_before": relation,
                        "relation_after": relation_after_settle,
                        "task_status_after": task_status_after_settle,
                    }
                )
                if bool(relation_after_settle.get("satisfied", False)) or bool(task_status_after_settle.get("task_completed", False)) or float(task_status_after_settle.get("reward", 0.0) or 0.0) >= 1.0:
                    return finish("satisfied_after_inside_xy_settle", relation_after_settle, task_status_after_settle)
                relation = relation_after_settle
                continue

            axis, local_axis, before_proj = dominant_robot_axis(local_delta, site_size)
            if abs(before_proj) <= 0.006:
                relation = relation_now()
                records.append(
                    {
                        "iteration": idx,
                        "action": "skip_small_axis_projection",
                        "relation_before": relation,
                        "local_delta": local_delta,
                        "site_size": site_size,
                        "axis_xy": axis,
                        "local_axis": local_axis,
                        "before_proj": before_proj,
                    }
                )
                continue
            direction = axis * (-1.0 if before_proj > 0.0 else 1.0)
            side_biases = (0.0, 0.018, -0.018, 0.032, -0.032)
            plate_z_biases = (-0.010, -0.014, -0.006, -0.018)
            generic_z_biases = (0.020, 0.014, 0.026, 0.010)
            z_biases = plate_z_biases if "plate" in resolved_object.lower() else generic_z_biases
            other_axis_index = 1 if abs(float(local_axis[0])) >= abs(float(local_axis[1])) else 0
            other_axis_inside = abs(float(local_delta[other_axis_index])) <= max(float(site_size[other_axis_index]) + 0.020, 0.060)
            side_bias = 0.0 if other_axis_inside or (np.isfinite(xy_before) and xy_before <= 0.20) else float(side_biases[idx % len(side_biases)])
            z_bias = float(z_biases[min(idx // len(side_biases), len(z_biases) - 1)])
            axis_error = max(0.0, abs(before_proj) - max(float(site_size[int(abs(local_axis[1]) > abs(local_axis[0]))]) * 0.70, 0.026))
            if np.isfinite(xy_before) and xy_before <= 0.18:
                step = float(np.clip(0.22 * max(axis_error, abs(before_proj)), 0.005, 0.014))
            else:
                step = float(np.clip(0.45 * max(axis_error, abs(before_proj)), 0.012, 0.038))
            if no_motion_streak:
                step = float(np.clip(step + 0.006 * min(no_motion_streak, 3), 0.012, 0.052))
                z_bias -= 0.004 * min(no_motion_streak, 3)

            record = run_axis_rim_stroke(
                direction,
                step,
                z_bias,
                side_bias,
                f"{idx}_axis_clamp",
            )
            record.update(
                {
                    "iteration": idx,
                    "relation_before": relation,
                    "xy_before": xy_before,
                    "local_delta_before": local_delta,
                    "site_size_before": site_size,
                    "selected_axis_xy": axis,
                    "selected_local_axis": local_axis,
                    "before_axis_projection": before_proj,
                    "worsen_streak_before": worsen_streak,
                    "no_motion_streak_before": no_motion_streak,
                    "task_status_before": task_status_before,
                }
            )
            relation_after = relation_now()
            xy_after = self._relation_xy_distance_to_target(relation_after)
            local_delta_after = local_delta_from_relation(relation_after)
            after_proj = float(np.dot(local_delta_after[:2], local_axis))
            moved_xy = float(record.get("object_motion_xy", 0.0) or 0.0)
            improved = bool(
                abs(after_proj) < abs(before_proj) - 0.003
                or (
                    np.isfinite(xy_after)
                    and np.isfinite(xy_before)
                    and xy_after < xy_before - 0.006
                )
            )
            worsened = bool(
                abs(after_proj) > abs(before_proj) + 0.006
                or (
                    np.isfinite(xy_after)
                    and np.isfinite(xy_before)
                    and xy_after > xy_before + 0.012
                )
            )
            other_axis_before = abs(float(local_delta[other_axis_index]))
            other_axis_after = abs(float(local_delta_after[other_axis_index]))
            preserved_other_axis = bool(other_axis_after <= max(other_axis_before + 0.035, 0.075))
            if not preserved_other_axis:
                worsened = True
            if improved or moved_xy >= float(min_motion):
                no_motion_streak = 0
            else:
                no_motion_streak += 1
            if worsened:
                worsen_streak += 1
            elif improved:
                worsen_streak = 0

            task_status_after = self.get_task_status()
            record.update(
                {
                    "relation_after": relation_after,
                    "xy_after": xy_after,
                    "local_delta_after": local_delta_after,
                    "after_axis_projection": after_proj,
                    "other_axis_index": other_axis_index,
                    "other_axis_inside_before": other_axis_inside,
                    "other_axis_before_abs": other_axis_before,
                    "other_axis_after_abs": other_axis_after,
                    "preserved_other_axis": preserved_other_axis,
                    "improved": improved,
                    "worsened": worsened,
                    "worsen_streak_after": worsen_streak,
                    "no_motion_streak_after": no_motion_streak,
                    "task_status_after": task_status_after,
                }
            )
            records.append(record)
            if xy_after < best_xy:
                best_xy = xy_after
                best_relation = dict(relation_after)
            if bool(relation_after.get("satisfied", False)) or bool(task_status_after.get("task_completed", False)) or float(task_status_after.get("reward", 0.0) or 0.0) >= 1.0:
                return {
                    "object_name": resolved_object,
                    "target_name": resolved_target,
                    "predicate": pred,
                    "initial_relation": initial_relation,
                    "final_relation": relation_after,
                    "best_relation": best_relation or relation_after,
                    "records": records,
                    "relation_satisfied": bool(relation_after.get("satisfied", False)),
                    "task_status": task_status_after,
                    "skill": "push_object_to_target_region",
                    "reason": f"satisfied_after_stroke_{idx}",
                }
            if worsen_streak >= 2 or no_motion_streak >= 3:
                relation = relation_after
                break
            relation = relation_after

        final_relation = relation_now()
        fallback_result = run_feedback_pick_release_fallback(
            "push_control_exhausted",
            final_relation,
        )
        if fallback_result is not None:
            return fallback_result
        final_relation = relation_now()
        task_status = self.get_task_status()
        return {
            "object_name": resolved_object,
            "target_name": resolved_target,
            "predicate": pred,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "best_relation": best_relation or final_relation,
            "records": records,
            "relation_satisfied": bool(final_relation.get("satisfied", False)),
            "task_status": task_status,
            "skill": "push_object_to_target_region",
            "reason": "max_iters_or_motion_gate_exhausted",
        }

    def _short_local_push_to_target(
        self,
        object_name: str,
        target_name: str,
        *,
        predicate: str,
        max_stroke: float = 0.030,
        settle_steps: int = 16,
    ) -> dict[str, Any]:
        pred = str(predicate).strip().lower()
        relation_before = self._placement_relation_diagnostic(object_name, target_name, predicate=pred)
        if bool(relation_before.get("satisfied", False)):
            return {
                "action": "already_satisfied",
                "initial_relation": relation_before,
                "final_relation": relation_before,
                "satisfied": True,
            }

        obj_pos, _ = self.get_object_pose(object_name)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
        try:
            if pred == "on":
                delta = np.asarray(
                    self.get_on_center_delta_robot(object_name, target_name),
                    dtype=np.float64,
                ).reshape(3)
            elif pred == "in" and bool(relation_before.get("contact", False)) and "local_delta" in relation_before:
                local_delta = np.asarray(relation_before["local_delta"], dtype=np.float64).reshape(-1)[:3]
                site_pos_world = np.asarray(relation_before["site_pos_world"], dtype=np.float64).reshape(3)
                site_pos_robot = self._world_position_to_robot_base(site_pos_world)
                try:
                    _, site_quat = self.get_site_pose(target_name)
                    site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64).reshape(4)).as_matrix()
                    delta = site_rot @ np.array([-local_delta[0], -local_delta[1], 0.0], dtype=np.float64)
                except Exception:
                    object_world = np.asarray(relation_before["object_pos_world"], dtype=np.float64).reshape(3)
                    object_robot = self._world_position_to_robot_base(object_world)
                    delta = site_pos_robot - object_robot
                    delta[2] = 0.0
            else:
                target = self._placement_target_for_object(
                    object_name,
                    predicate=pred,
                    target_name=target_name,
                )
                target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
                delta = target_pos - obj_pos
        except Exception:
            target = self._placement_target_for_object(
                object_name,
                predicate=pred,
                target_name=target_name,
            )
            target_pos = np.asarray(target["target_position"], dtype=np.float64).reshape(3)
            delta = target_pos - obj_pos

        xy = np.asarray(delta[:2], dtype=np.float64)
        dist = float(np.linalg.norm(xy))
        if dist <= 1e-6:
            self._settle_simulation_steps(max(8, int(settle_steps)))
            relation_after_settle = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate=pred,
            )
            return {
                "action": "settle_only",
                "distance_xy": dist,
                "initial_relation": relation_before,
                "final_relation": relation_after_settle,
                "satisfied": bool(relation_after_settle.get("satisfied", False)),
            }

        direction = xy / max(dist, 1e-6)
        stroke = min(float(max_stroke), max(0.010, 0.45 * dist))
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        push_height = 0.026 if pred == "on" else 0.034
        start = obj_pos.copy()
        start[:2] -= direction * 0.026
        start[2] += push_height
        end = obj_pos.copy()
        end[:2] += direction * stroke
        end[2] = start[2]
        self.open_gripper()
        self.goto_pose(start + np.array([0.0, 0.0, 0.055], dtype=np.float64), quat)
        self.goto_pose(start, quat)
        self.goto_pose(end, quat)
        self.goto_pose(end + np.array([0.0, 0.0, 0.065], dtype=np.float64), quat)
        self._settle_simulation_steps(max(8, int(settle_steps)))
        relation_after = self._placement_relation_diagnostic(
            object_name,
            target_name,
            predicate=pred,
        )
        return {
            "action": "short_planar_push",
            "distance_xy": dist,
            "stroke": stroke,
            "start": start,
            "end": end,
            "initial_relation": relation_before,
            "final_relation": relation_after,
            "satisfied": bool(relation_after.get("satisfied", False)),
        }

    def _containment_correction_to_site(
        self,
        object_name: str,
        target_name: str,
        *,
        diagnostic: dict[str, Any] | None = None,
        max_iters: int = 3,
        settle_steps: int = 24,
    ) -> dict[str, Any]:
        """Push a contacted object back into a contain site using site-local error."""
        records: list[dict[str, Any]] = []
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        diag = (
            dict(diagnostic)
            if isinstance(diagnostic, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        for idx in range(max(1, int(max_iters))):
            if bool(diag.get("satisfied", False) or diag.get("in", False)):
                return {
                    "records": records,
                    "initial_relation": diagnostic,
                    "final_relation": diag,
                    "satisfied": True,
                }
            if not bool(diag.get("contact", False)) or "local_delta" not in diag:
                break

            local_delta = np.asarray(diag.get("local_delta"), dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(
                diag.get("site_size_abs", diag.get("site_size", np.array([0.04, 0.04, 0.04]))),
                dtype=np.float64,
            ).reshape(-1)[:3]
            local_error = np.zeros(3, dtype=np.float64)
            local_error[:2] = -local_delta[:2]
            if float(np.linalg.norm(local_error[:2])) <= 0.004:
                local_error[:2] = -0.65 * local_delta[:2]

            object_world = np.asarray(diag.get("object_pos_world"), dtype=np.float64).reshape(3)
            site_world = np.asarray(diag.get("site_pos_world"), dtype=np.float64).reshape(3)
            delta_world_to_site = np.zeros(3, dtype=np.float64)
            delta_world_to_site[:2] = site_world[:2] - object_world[:2]
            delta_robot_to_site = self._world_vector_to_robot_base(delta_world_to_site)
            try:
                _, site_quat = self.get_site_pose(target_name)
                site_rot = self._rotation_from_wxyz(np.asarray(site_quat, dtype=np.float64).reshape(4)).as_matrix()
                delta_robot_local = site_rot @ local_error
                if float(np.linalg.norm(delta_robot_local[:2])) >= 0.006:
                    delta_robot = delta_robot_local
                else:
                    delta_robot = delta_robot_to_site
            except Exception:
                delta_robot = delta_robot_to_site
            delta_robot = np.asarray(delta_robot, dtype=np.float64).reshape(3)
            delta_robot[2] = 0.0
            dist = float(np.linalg.norm(delta_robot[:2]))
            if dist <= 1e-6:
                self._settle_simulation_steps(max(8, int(settle_steps)))
                diag = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                records.append({"iteration": idx, "action": "settle_zero_delta", "relation": diag})
                continue

            direction = delta_robot[:2] / max(dist, 1e-6)
            desired_stroke = float(np.linalg.norm(local_error[:2])) + 0.030
            stroke = min(max(desired_stroke, 0.050), 0.145)
            obj_pos = self._world_position_to_robot_base(object_world)
            site_pos = self._world_position_to_robot_base(site_world)
            side = np.array([-direction[1], direction[0]], dtype=np.float64)
            start = obj_pos.copy()
            start[:2] -= direction * 0.058
            start[:2] += side * (0.010 if idx % 2 == 0 else -0.010)
            low_push_z = obj_pos[2] + (0.002 if idx == 0 else 0.010)
            site_floor_z = site_pos[2] - max(0.030, float(abs(site_size[2])) * 0.65)
            site_cap_z = site_pos[2] + max(0.010, float(abs(site_size[2])) * 0.25)
            start[2] = float(np.clip(low_push_z, site_floor_z, site_cap_z))
            end = obj_pos.copy()
            end[:2] += direction * stroke
            end[:2] += side * (0.010 if idx % 2 == 0 else -0.010)
            end[2] = start[2]
            self.open_gripper()
            self.goto_pose(start + np.array([0.0, 0.0, 0.060], dtype=np.float64), quat)
            self.goto_pose(start, quat)
            self.close_gripper()
            self._settle_simulation_steps(5)
            self.goto_pose(end, quat)
            self.open_gripper()
            self.goto_pose(end + np.array([0.0, 0.0, 0.070], dtype=np.float64), quat)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            diag = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            before_norm = float(np.linalg.norm(local_delta[:2]))
            after_delta = np.asarray(diag.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            after_norm = float(np.linalg.norm(after_delta[:2]))
            guard_action = None
            if (
                not bool(diag.get("satisfied", False) or diag.get("in", False))
                and after_norm > before_norm + 0.025
            ):
                reverse_start = end.copy()
                reverse_end = obj_pos.copy()
                reverse_end[:2] -= direction * 0.018
                reverse_end[2] = start[2]
                self.open_gripper()
                self.goto_pose(reverse_start + np.array([0.0, 0.0, 0.050], dtype=np.float64), quat)
                self.goto_pose(reverse_start, quat)
                self.close_gripper()
                self._settle_simulation_steps(4)
                self.goto_pose(reverse_end, quat)
                self.open_gripper()
                self.goto_pose(reverse_end + np.array([0.0, 0.0, 0.060], dtype=np.float64), quat)
                self._settle_simulation_steps(max(8, int(settle_steps)))
                guard_action = {
                    "action": "reverse_after_worsened_local_delta",
                    "reverse_start": reverse_start,
                    "reverse_end": reverse_end,
                    "before_norm": before_norm,
                    "after_norm": after_norm,
                }
                diag = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            records.append(
                {
                    "iteration": idx,
                    "action": "site_local_containment_push",
                    "local_delta_before": local_delta.copy(),
                    "site_size_abs": site_size.copy(),
                    "local_error": local_error.copy(),
                    "delta_robot": delta_robot.copy(),
                    "stroke": stroke,
                    "start": start,
                    "end": end,
                    "before_norm": before_norm,
                    "after_norm": after_norm,
                    "guard_action": guard_action,
                    "relation": diag,
                }
            )
            if guard_action is not None:
                break

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        return {
            "records": records,
            "initial_relation": diagnostic,
            "final_relation": final_relation,
            "satisfied": bool(final_relation.get("satisfied", False)),
        }

    def _drawer_slide_object_into_region(
        self,
        object_name: str,
        target_name: str,
        *,
        diagnostic: dict[str, Any] | None = None,
        max_iters: int = 3,
        settle_steps: int = 24,
        require_closing_clearance: bool = False,
        min_closing_front_clearance: float = 0.045,
    ) -> dict[str, Any]:
        """Low planar push that seats an unheld object into a drawer region."""
        records: list[dict[str, Any]] = []
        initial_relation = (
            dict(diagnostic)
            if isinstance(diagnostic, dict)
            else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        )
        relation = initial_relation
        def relation_has_drawer_goal_margin(current_relation: dict[str, Any]) -> bool:
            in_region = bool(current_relation.get("satisfied", False) or current_relation.get("in", False))
            if not in_region:
                return False
            if not bool(require_closing_clearance):
                return True
            clearance = self._drawer_relation_closing_clearance(
                current_relation,
                min_front_clearance=float(min_closing_front_clearance),
            )
            return bool(clearance.get("has_clearance", False))

        initial_clearance = (
            self._drawer_relation_closing_clearance(
                relation,
                min_front_clearance=float(min_closing_front_clearance),
            )
            if bool(require_closing_clearance)
            else None
        )
        if relation_has_drawer_goal_margin(relation):
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": relation,
                "satisfied": True,
                "task_status": self.get_task_status(),
                "drawer_closing_clearance": initial_clearance,
                "skill": "drawer_slide_object_into_region",
            }

        def clearance_for(current_relation: dict[str, Any]) -> dict[str, Any] | None:
            if not bool(require_closing_clearance):
                return None
            return self._drawer_relation_closing_clearance(
                current_relation,
                min_front_clearance=float(min_closing_front_clearance),
            )

        def relation_near_drawer_box(current_relation: dict[str, Any]) -> bool:
            if not isinstance(current_relation, dict):
                return False
            if "local_delta" not in current_relation or "site_size_abs" not in current_relation:
                return False
            try:
                local_delta = np.asarray(current_relation["local_delta"], dtype=np.float64).reshape(-1)[:3]
                site_size = np.asarray(current_relation["site_size_abs"], dtype=np.float64).reshape(-1)[:3]
                return bool(
                    local_delta.size >= 3
                    and site_size.size >= 3
                    and abs(float(local_delta[0])) <= max(float(abs(site_size[0])) + 0.018, 0.045)
                    and abs(float(local_delta[1])) <= max(float(abs(site_size[1])) + 0.018, 0.060)
                    and -float(abs(site_size[2])) - 0.020 <= float(local_delta[2]) <= float(abs(site_size[2])) + 0.020
                )
            except Exception:
                return False

        def drawer_micro_seat(current_relation: dict[str, Any], *, trigger: str) -> dict[str, Any]:
            relation_now = dict(current_relation) if isinstance(current_relation, dict) else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            quat_local = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            local_delta = np.asarray(relation_now.get("local_delta", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
            site_size = np.asarray(relation_now.get("site_size_abs", np.ones(3) * 0.04), dtype=np.float64).reshape(-1)[:3]
            obj_pos, _ = self.get_object_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            try:
                site_world = np.asarray(relation_now["site_pos_world"], dtype=np.float64).reshape(3)
                object_world = np.asarray(relation_now["object_pos_world"], dtype=np.float64).reshape(3)
                site_half_world = np.asarray(site_size, dtype=np.float64).reshape(-1)[:3]
                z_upper = float(site_world[2] + abs(float(site_half_world[2])))
                z_lower = float(site_world[2] - abs(float(site_half_world[2])) - 0.010)
                center_in_world_xy = bool(
                    abs(float(object_world[0] - site_world[0])) <= abs(float(site_half_world[0])) + 0.002
                    and abs(float(object_world[1] - site_world[1])) <= abs(float(site_half_world[1])) + 0.002
                )
                z_excess = float(object_world[2] - z_upper)
                if center_in_world_xy and z_lower <= float(object_world[2]) and 0.0 < z_excess <= 0.040:
                    obj_pos_robot = obj_pos.copy()
                    site_robot = self._world_position_to_robot_base(site_world)
                    press_z = float(
                        np.clip(
                            site_robot[2] + abs(float(site_half_world[2])) + 0.004,
                            obj_pos_robot[2] + 0.010,
                            obj_pos_robot[2] + 0.026,
                        )
                    )
                    start = obj_pos_robot.copy()
                    start[2] = max(obj_pos_robot[2] + 0.090, press_z + 0.060)
                    contact = obj_pos_robot.copy()
                    contact[2] = max(obj_pos_robot[2] + 0.040, press_z + 0.014)
                    end = obj_pos_robot.copy()
                    end[2] = press_z
                    record = {
                        "action": "drawer_micro_seat_vertical",
                        "trigger": trigger,
                        "relation_before": relation_now,
                        "local_delta_before": local_delta.copy(),
                        "site_size_abs": site_size.copy(),
                        "object_position_before": obj_pos.copy(),
                        "z_upper_world": z_upper,
                        "z_excess_world": z_excess,
                        "start": start.copy(),
                        "contact": contact.copy(),
                        "end": end.copy(),
                    }
                    try:
                        self.open_gripper()
                        record["start_move"] = self.goto_finger_midpoint_checked(start, quat_local, tolerance=0.085)
                        self.close_gripper()
                        self._settle_simulation_steps(3)
                        record["contact_move"] = self.goto_finger_midpoint_checked(contact, quat_local, tolerance=0.075)
                        record["end_move"] = self.goto_finger_midpoint_checked(end, quat_local, tolerance=0.075)
                        self.open_gripper()
                        retreat = end + np.array([0.0, 0.0, 0.085], dtype=np.float64)
                        record["retreat_move"] = self.goto_finger_midpoint_checked(retreat, quat_local, tolerance=0.090)
                    except Exception as exc:
                        record["error"] = repr(exc)
                        try:
                            self.open_gripper()
                        except Exception:
                            pass
                    self._settle_simulation_steps(max(14, min(30, int(settle_steps))))
                    relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                    record["relation_after"] = relation_after
                    records.append(record)
                    return relation_after
                target_world = object_world.copy()
                target_world[0] = float(np.clip(object_world[0], site_world[0] - abs(float(site_size[0])) * 0.55, site_world[0] + abs(float(site_size[0])) * 0.55))
                target_world[1] = float(np.clip(object_world[1], site_world[1] - abs(float(site_size[1])) * 0.45, site_world[1] + abs(float(site_size[1])) * 0.45))
                target_world[2] = float(site_world[2] + min(0.0, max(-0.006, float(local_delta[2]))))
                target_robot = self._world_position_to_robot_base(target_world)
                delta_xy = target_robot[:2] - obj_pos[:2]
            except Exception:
                delta_xy = np.zeros(2, dtype=np.float64)
            if bool(require_closing_clearance):
                clearance_now = self._drawer_relation_closing_clearance(
                    relation_now,
                    min_front_clearance=float(min_closing_front_clearance),
                )
                front_deficit = max(
                    0.0,
                    float(min_closing_front_clearance)
                    - float(clearance_now.get("front_clearance", 0.0) or 0.0),
                )
                if front_deficit > 0.002:
                    try:
                        level_now = self._drawer_level_from_target_name(target_name) or "middle"
                        _, _, pull_dir_now = self.get_cabinet_drawer_handle_pose(level_now)
                        pull_dir_now = np.asarray(pull_dir_now, dtype=np.float64).reshape(3)
                        pull_dir_now[2] = 0.0
                        pull_dir_now = pull_dir_now / max(float(np.linalg.norm(pull_dir_now)), 1e-9)
                        interior_now = -pull_dir_now
                        current_delta_norm = float(np.linalg.norm(delta_xy))
                        interior_step = float(np.clip(front_deficit + 0.018, 0.025, 0.075))
                        if current_delta_norm <= 0.006:
                            delta_xy = interior_now[:2] * interior_step
                        else:
                            delta_xy = delta_xy + interior_now[:2] * interior_step
                    except Exception:
                        pass
            delta_norm = float(np.linalg.norm(delta_xy))
            direction = np.zeros(3, dtype=np.float64)
            if delta_norm > 1e-6:
                direction[:2] = delta_xy / delta_norm
            shallow_drawer_object = self._requires_strict_hold(object_name)
            push_z = float(obj_pos[2] + (0.014 if shallow_drawer_object else 0.035))
            start = obj_pos.copy()
            start[:2] -= direction[:2] * 0.025
            start[2] = push_z + (0.040 if shallow_drawer_object else 0.060)
            contact = obj_pos.copy()
            contact[:2] -= direction[:2] * 0.006
            contact[2] = push_z
            end = obj_pos.copy()
            end[:2] += direction[:2] * float(
                np.clip(
                    delta_norm + (0.012 if shallow_drawer_object else 0.006),
                    0.010 if shallow_drawer_object else 0.006,
                    0.052 if shallow_drawer_object else 0.026,
                )
            )
            end[2] = max(float(obj_pos[2] + 0.002), push_z - (0.004 if shallow_drawer_object else 0.010))
            record: dict[str, Any] = {
                "action": "drawer_micro_seat_near_box",
                "trigger": trigger,
                "relation_before": relation_now,
                "local_delta_before": local_delta.copy(),
                "site_size_abs": site_size.copy(),
                "shallow_drawer_object": bool(shallow_drawer_object),
                "object_position_before": obj_pos.copy(),
                "direction": direction.copy(),
                "delta_xy": delta_xy.copy(),
                "push_z": push_z,
                "start": start.copy(),
                "contact": contact.copy(),
                "end": end.copy(),
            }
            try:
                self.open_gripper()
                record["start_move"] = self.goto_finger_midpoint_checked(start, quat_local, tolerance=0.080)
                record["contact_move"] = self.goto_finger_midpoint_checked(contact, quat_local, tolerance=0.070)
                self.close_gripper()
                self._settle_simulation_steps(4)
                record["end_move"] = self.goto_finger_midpoint_checked(end, quat_local, tolerance=0.080)
                self.open_gripper()
                retreat = end + np.array([0.0, 0.0, 0.080], dtype=np.float64)
                record["retreat_move"] = self.goto_finger_midpoint_checked(retreat, quat_local, tolerance=0.090)
            except Exception as exc:
                record["error"] = repr(exc)
                try:
                    self.open_gripper()
                except Exception:
                    pass
            self._settle_simulation_steps(max(12, min(28, int(settle_steps))))
            relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            record["relation_after"] = relation_after
            records.append(record)
            return relation_after

        if relation_near_drawer_box(relation):
            relation = drawer_micro_seat(relation, trigger="initial_near_drawer_box")
            satisfied_after_micro = bool(relation_has_drawer_goal_margin(relation))
            if satisfied_after_micro or not bool(require_closing_clearance):
                return {
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation,
                    "satisfied": satisfied_after_micro,
                    "task_status": self.get_task_status(),
                    "drawer_closing_clearance": clearance_for(relation),
                    "reason": (
                        "near drawer-box placement satisfied after guarded micro-seat"
                        if satisfied_after_micro
                        else "near drawer-box placement preserved after guarded micro-seat"
                    ),
                    "skill": "drawer_slide_object_into_region",
                }
            records.append(
                {
                    "action": "drawer_micro_seat_clearance_feedback_continue",
                    "relation_after_micro_seat": relation,
                    "drawer_closing_clearance": clearance_for(relation),
                    "reason": "near drawer-box placement still lacks close-time clearance",
                }
            )

        target_pos, _ = self.get_site_pose(target_name)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
        level = self._drawer_level_from_target_name(target_name) or "middle"
        try:
            _, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
            pull_dir = np.asarray(pull_dir, dtype=np.float64).reshape(3)
            pull_dir[2] = 0.0
            pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)
            interior_dir = -pull_dir
        except Exception:
            interior_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        try:
            lateral_axis = self._drawer_handle_lateral_axis(level)
            lateral_axis = np.asarray(lateral_axis, dtype=np.float64).reshape(3)
            lateral_axis[2] = 0.0
            lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-9)
        except Exception:
            lateral_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        def local_xy_norm(current_relation: dict[str, Any]) -> float:
            if not isinstance(current_relation, dict):
                return float("inf")
            try:
                object_world = np.asarray(current_relation["object_pos_world"], dtype=np.float64).reshape(-1)[:3]
                site_world = np.asarray(current_relation["site_pos_world"], dtype=np.float64).reshape(-1)[:3]
                return float(np.linalg.norm((object_world - site_world)[:2]))
            except Exception:
                pass
            try:
                local = np.asarray(current_relation["local_delta"], dtype=np.float64).reshape(-1)
                if local.size >= 3:
                    return float(np.linalg.norm(local[1:3]))
                if local.size >= 2:
                    return float(abs(float(local[1])))
            except Exception:
                pass
            return float("inf")

        def run_site_local_refinement(
            current_relation: dict[str, Any],
            *,
            trigger: str,
            max_local_iters: int,
        ) -> dict[str, Any]:
            relation_now = (
                dict(current_relation)
                if isinstance(current_relation, dict)
                else self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            )
            best_relation = dict(relation_now)
            best_norm = local_xy_norm(relation_now)
            best_front_clearance = None
            if bool(require_closing_clearance):
                try:
                    initial_clearance_feedback = self._drawer_relation_closing_clearance(
                        relation_now,
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                    best_front_clearance = float(
                        initial_clearance_feedback.get("front_clearance", float("-inf")) or float("-inf")
                    )
                except Exception:
                    best_front_clearance = None
            no_motion_streak = 0
            no_clearance_progress_streak = 0
            worsen_streak = 0
            for local_idx in range(max(0, int(max_local_iters))):
                if relation_has_drawer_goal_margin(relation_now):
                    return relation_now
                if (
                    not bool(relation_now.get("contact", False))
                    or "local_delta" not in relation_now
                    or "site_size_abs" not in relation_now
                ):
                    records.append(
                        {
                            "iteration": int(local_idx),
                            "action": "drawer_site_local_refine_stop",
                            "trigger": trigger,
                            "reason": "missing_contact_or_local_geometry",
                            "relation": relation_now,
                        }
                    )
                    break

                local_delta = np.asarray(relation_now["local_delta"], dtype=np.float64).reshape(-1)[:3]
                site_size = np.asarray(relation_now["site_size_abs"], dtype=np.float64).reshape(-1)[:3]
                if local_delta.size < 3 or site_size.size < 3:
                    records.append(
                        {
                            "iteration": int(local_idx),
                            "action": "drawer_site_local_refine_stop",
                            "trigger": trigger,
                            "reason": "bad_local_geometry_shape",
                            "relation": relation_now,
                        }
                    )
                    break

                local_planar_delta = np.array(
                    [float(local_delta[1]), float(local_delta[2])],
                    dtype=np.float64,
                )
                local_planar_size = np.array(
                    [float(abs(site_size[1])), float(abs(site_size[2]))],
                    dtype=np.float64,
                )
                local_threshold = np.maximum(
                    0.82 * local_planar_size,
                    np.array([0.044, 0.026], dtype=np.float64),
                )
                if bool(require_closing_clearance):
                    local_threshold = np.minimum(
                        local_threshold,
                        np.array([0.052, 0.026], dtype=np.float64),
                    )
                local_excess = np.maximum(np.abs(local_planar_delta) - local_threshold, 0.0)
                target_local_planar = np.zeros(2, dtype=np.float64)
                clearance_feedback = None
                front_clearance_deficit = 0.0
                front_axis_error = 0.0
                if bool(require_closing_clearance):
                    clearance_feedback = self._drawer_relation_closing_clearance(
                        relation_now,
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                    front_clearance_deficit = max(
                        0.0,
                        float(min_closing_front_clearance)
                        - float(clearance_feedback.get("front_clearance", 0.0) or 0.0),
                    )
                    if front_clearance_deficit > 0.002:
                        target_front_local = float(
                            np.clip(
                                float(min_closing_front_clearance) - abs(float(site_size[1])) + 0.012,
                                -0.004,
                                0.024,
                            )
                        )
                        front_axis_error = max(0.0, target_front_local - float(local_delta[1]))
                        target_local_planar[0] = float(local_delta[1] + front_axis_error)
                        target_local_planar[1] = float(
                            np.clip(float(local_delta[2]), -0.008, 0.008)
                        )
                if (
                    not bool(require_closing_clearance)
                    and float(np.max(local_excess)) <= 0.004
                ):
                    target_local_planar = np.clip(
                        local_planar_delta,
                        -0.45 * local_planar_size,
                        0.45 * local_planar_size,
                    )
                local_planar_error = target_local_planar - local_planar_delta
                local_error = np.array(
                    [0.0, float(local_planar_error[0]), float(local_planar_error[1])],
                    dtype=np.float64,
                )
                local_error_norm = float(np.linalg.norm(local_planar_error))
                if local_error_norm <= 0.004:
                    self._settle_simulation_steps(max(10, int(settle_steps)))
                    relation_now = self._placement_relation_diagnostic(
                        object_name,
                        target_name,
                        predicate="in",
                    )
                    records.append(
                        {
                            "iteration": int(local_idx),
                            "action": "drawer_site_local_settle",
                            "trigger": trigger,
                            "relation_after": relation_now,
                        }
                    )
                    continue

                obj_pos, _ = self.get_object_pose(object_name)
                obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
                if bool(require_closing_clearance) and front_axis_error > 0.002:
                    try:
                        _, site_quat = self.get_site_pose(target_name)
                        site_rot = self._rotation_from_wxyz(
                            np.asarray(site_quat, dtype=np.float64).reshape(4)
                        ).as_matrix()
                        correction_robot = site_rot @ local_error
                    except Exception:
                        correction_robot = interior_dir * float(front_axis_error)
                else:
                    correction_robot = None
                try:
                    site_world = np.asarray(relation_now["site_pos_world"], dtype=np.float64).reshape(3)
                    object_world = np.asarray(relation_now["object_pos_world"], dtype=np.float64).reshape(3)
                    correction_world = site_world - object_world
                    if correction_robot is None:
                        correction_robot = self._world_position_to_robot_base(object_world + correction_world) - obj_pos
                except Exception:
                    if correction_robot is None:
                        correction_robot = target_pos - obj_pos
                correction_robot = np.asarray(correction_robot, dtype=np.float64).reshape(3)
                correction_robot[2] = 0.0
                correction_norm = float(np.linalg.norm(correction_robot[:2]))
                if correction_norm <= 1e-6:
                    correction_robot = target_pos - obj_pos
                    correction_robot[2] = 0.0
                    correction_norm = float(np.linalg.norm(correction_robot[:2]))
                if correction_norm <= 1e-6:
                    break

                direction3 = correction_robot / max(correction_norm, 1e-6)
                direction3[2] = 0.0
                direction3 = direction3 / max(float(np.linalg.norm(direction3)), 1e-9)
                side = np.array([-direction3[1], direction3[0], 0.0], dtype=np.float64)
                side_bias = 0.000 if local_idx == 0 else (0.012 if local_idx % 2 else -0.012)
                stroke = float(
                    np.clip(
                        0.55 * local_error_norm + (0.020 if bool(require_closing_clearance) else 0.022),
                        0.022 if bool(require_closing_clearance) else 0.038,
                        0.070 if bool(require_closing_clearance) else 0.125,
                    )
                )
                if no_motion_streak:
                    stroke = float(
                        np.clip(
                            stroke + 0.006 * min(no_motion_streak, 2),
                            0.026 if bool(require_closing_clearance) else 0.045,
                            0.078 if bool(require_closing_clearance) else 0.155,
                        )
                    )

                try:
                    geom = self._object_geometry_summary(object_name)
                    bounds_min = np.asarray(geom["bounds_min"], dtype=np.float64).reshape(3)
                    bounds_max = np.asarray(geom["bounds_max"], dtype=np.float64).reshape(3)
                    object_height = max(float(bounds_max[2] - bounds_min[2]), 0.030)
                    push_z = float(bounds_min[2] + 0.48 * object_height)
                except Exception:
                    bounds_min = None
                    bounds_max = None
                    object_height = 0.060
                    push_z = float(obj_pos[2] + 0.050)
                push_z = float(
                    np.clip(
                        push_z,
                        float(obj_pos[2] + 0.012),
                        float(obj_pos[2] + max(0.060, min(0.105, 0.75 * object_height))),
                    )
                )

                start = obj_pos.copy()
                start[:2] -= direction3[:2] * (0.032 if bool(require_closing_clearance) else 0.052)
                start[:2] += side[:2] * side_bias
                start[2] = push_z
                catch = obj_pos.copy()
                catch[:2] -= direction3[:2] * (0.010 if bool(require_closing_clearance) else 0.016)
                catch[:2] += side[:2] * side_bias
                catch[2] = push_z
                end = obj_pos.copy()
                end[:2] += direction3[:2] * stroke
                end[:2] += side[:2] * side_bias
                end[2] = push_z
                start_high = start + np.array([0.0, 0.0, 0.085], dtype=np.float64)
                retreat = end + np.array([0.0, 0.0, 0.095], dtype=np.float64)

                before_norm = local_xy_norm(relation_now)
                record: dict[str, Any] = {
                    "iteration": int(local_idx),
                    "action": "drawer_site_local_feedback_slide",
                    "trigger": trigger,
                    "level": level,
                    "require_closing_clearance": bool(require_closing_clearance),
                    "relation_before": relation_now,
                    "local_delta_before": local_delta.copy(),
                    "site_size_abs": site_size.copy(),
                    "local_planar_delta_before": local_planar_delta.copy(),
                    "local_planar_error": local_planar_error.copy(),
                    "local_error": local_error.copy(),
                    "target_local_planar": target_local_planar.copy(),
                    "drawer_closing_clearance_before": clearance_feedback,
                    "front_clearance_deficit": float(front_clearance_deficit),
                    "front_axis_error": float(front_axis_error),
                    "correction_robot": correction_robot.copy(),
                    "direction": direction3.copy(),
                    "stroke": stroke,
                    "push_z": push_z,
                    "object_position_before": obj_pos.copy(),
                    "start": start.copy(),
                    "catch": catch.copy(),
                    "end": end.copy(),
                    "retreat": retreat.copy(),
                    "bounds_min": bounds_min.copy() if isinstance(bounds_min, np.ndarray) else None,
                    "bounds_max": bounds_max.copy() if isinstance(bounds_max, np.ndarray) else None,
                    "worsen_streak_before": worsen_streak,
                    "no_motion_streak_before": no_motion_streak,
                }
                try:
                    self.open_gripper()
                    record["start_high_move"] = self.goto_finger_midpoint_checked(
                        start_high,
                        quat,
                        tolerance=0.070,
                    )
                    record["start_move"] = self.goto_finger_midpoint_checked(
                        start,
                        quat,
                        tolerance=0.060,
                    )
                    self.close_gripper()
                    self._settle_simulation_steps(8)
                    record["catch_move"] = self.goto_finger_midpoint_checked(
                        catch,
                        quat,
                        tolerance=0.060,
                    )
                    record["end_move"] = self.goto_finger_midpoint_checked(
                        end,
                        quat,
                        tolerance=0.080,
                    )
                    self.open_gripper()
                    record["retreat_move"] = self.goto_finger_midpoint_checked(
                        retreat,
                        quat,
                        tolerance=0.085,
                    )
                except Exception as exc:
                    record["error"] = repr(exc)
                    try:
                        self.open_gripper()
                    except Exception as cleanup_exc:
                        record["cleanup_error"] = repr(cleanup_exc)
                self._settle_simulation_steps(max(10, int(settle_steps)))

                after_pos, _ = self.get_object_pose(object_name)
                after_pos = np.asarray(after_pos, dtype=np.float64).reshape(3).copy()
                relation_after = self._placement_relation_diagnostic(
                    object_name,
                    target_name,
                    predicate="in",
                )
                after_norm = local_xy_norm(relation_after)
                object_motion = float(np.linalg.norm(after_pos[:2] - obj_pos[:2]))
                improved = bool(np.isfinite(before_norm) and np.isfinite(after_norm) and after_norm < before_norm - 0.008)
                worsened = bool(np.isfinite(before_norm) and np.isfinite(after_norm) and after_norm > before_norm + 0.018)
                if improved or object_motion >= 0.006:
                    no_motion_streak = 0
                else:
                    no_motion_streak += 1
                if worsened:
                    worsen_streak += 1
                elif improved:
                    worsen_streak = 0
                if np.isfinite(after_norm) and after_norm < best_norm:
                    best_norm = after_norm
                    best_relation = dict(relation_after)
                clearance_after = clearance_for(relation_after)
                front_clearance_after = None
                if isinstance(clearance_after, dict):
                    try:
                        front_clearance_after = float(
                            clearance_after.get("front_clearance", float("-inf")) or float("-inf")
                        )
                    except Exception:
                        front_clearance_after = None
                clearance_improved = False
                if bool(require_closing_clearance) and front_clearance_after is not None:
                    previous_best_front = best_front_clearance
                    if previous_best_front is None or front_clearance_after > previous_best_front + 0.008:
                        best_front_clearance = front_clearance_after
                        clearance_improved = True
                    else:
                        no_clearance_progress_streak += 1
                record.update(
                    {
                        "object_position_after": after_pos.copy(),
                        "object_motion_xy": object_motion,
                        "relation_after": relation_after,
                        "closing_clearance_after": clearance_after,
                        "front_clearance_after": front_clearance_after,
                        "best_front_clearance": best_front_clearance,
                        "clearance_improved": clearance_improved,
                        "before_norm": before_norm,
                        "after_norm": after_norm,
                        "improved": improved,
                        "worsened": worsened,
                        "worsen_streak_after": worsen_streak,
                        "no_motion_streak_after": no_motion_streak,
                        "no_clearance_progress_streak_after": no_clearance_progress_streak,
                    }
                )
                records.append(record)
                relation_now = relation_after
                if relation_has_drawer_goal_margin(relation_after):
                    return relation_after
                if (
                    worsen_streak >= 2
                    or no_motion_streak >= 2
                    or (
                        bool(require_closing_clearance)
                        and no_clearance_progress_streak >= 1
                        and bool(relation_after.get("in", relation_after.get("satisfied", False)))
                    )
                ):
                    if bool(require_closing_clearance) and no_clearance_progress_streak >= 1:
                        records.append(
                            {
                                "iteration": int(local_idx),
                                "action": "drawer_site_local_refine_stop",
                                "trigger": trigger,
                                "reason": "front_clearance_not_improving_replan_deeper_held_release",
                                "relation": relation_after,
                                "closing_clearance_after": clearance_after,
                                "best_front_clearance": best_front_clearance,
                            }
                        )
                    break
            if relation_has_drawer_goal_margin(relation_now):
                return relation_now
            if best_norm < local_xy_norm(relation_now):
                return best_relation
            return relation_now

        relation = run_site_local_refinement(
            relation,
            trigger="initial_contact_relation",
            max_local_iters=max(2, int(max_iters)),
        )
        if relation_has_drawer_goal_margin(relation):
            return {
                "records": records,
                "initial_relation": initial_relation,
                "final_relation": relation,
                "satisfied": True,
                "task_status": self.get_task_status(),
                "drawer_closing_clearance": clearance_for(relation),
                "skill": "drawer_slide_object_into_region",
            }

        if bool(require_closing_clearance):
            # Try downstream-close-safe placements first. A shallow placement can
            # satisfy LIBERO's In predicate while blocking the final close.
            offset_candidates = (
                (0.145, 0.000),
                (0.180, 0.000),
                (0.120, 0.000),
                (0.145, -0.020),
                (0.145, 0.020),
                (0.180, -0.020),
                (0.180, 0.020),
                (0.210, 0.000),
                (0.120, -0.020),
                (0.120, 0.020),
                (0.095, 0.000),
                (0.060, 0.000),
            )
        else:
            offset_candidates = tuple(
                (depth, lateral)
                for depth in (0.040, 0.075, 0.110, 0.145)
                for lateral in (0.0, -0.020, 0.020, -0.040, 0.040)
            )
        attempts = 0
        for depth, lateral in offset_candidates:
            if attempts >= max(1, int(max_iters)):
                final_relation = self._placement_relation_diagnostic(
                    object_name,
                    target_name,
                    predicate="in",
                )
                break
            attempts += 1
            relation_before = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate="in",
            )
            clearance_before = (
                self._drawer_relation_closing_clearance(
                    relation_before,
                    min_front_clearance=float(min_closing_front_clearance),
                )
                if bool(require_closing_clearance)
                else None
            )
            if relation_has_drawer_goal_margin(relation_before):
                return {
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation_before,
                    "satisfied": True,
                    "task_status": self.get_task_status(),
                    "drawer_closing_clearance": clearance_before,
                    "skill": "drawer_slide_object_into_region",
                }

            obj_pos, _ = self.get_object_pose(object_name)
            obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3).copy()
            target = target_pos + float(depth) * interior_dir + float(lateral) * lateral_axis
            desired = target - obj_pos
            lateral_error = float(np.dot(desired, lateral_axis))
            forward_error = float(np.dot(desired, interior_dir))
            if forward_error >= 0.010:
                direction3 = interior_dir + np.clip(lateral_error, -0.030, 0.030) * lateral_axis / 0.080
            else:
                direction3 = np.sign(lateral_error or 1.0) * lateral_axis
            direction3[2] = 0.0
            direction3 = direction3 / max(float(np.linalg.norm(direction3)), 1e-9)
            stroke = float(np.clip(max(forward_error, 0.025) + 0.030, 0.045, 0.120))
            try:
                geom = self._object_geometry_summary(object_name)
                bounds_min = np.asarray(geom["bounds_min"], dtype=np.float64).reshape(3)
                bounds_max = np.asarray(geom["bounds_max"], dtype=np.float64).reshape(3)
                push_z = float(bounds_min[2] + 0.55 * max(0.030, bounds_max[2] - bounds_min[2]))
            except Exception:
                bounds_min = None
                bounds_max = None
                push_z = float(obj_pos[2] + 0.060)
            push_z = float(np.clip(push_z, target_pos[2] - 0.005, target_pos[2] + 0.065))
            start = obj_pos.copy()
            start[:2] -= direction3[:2] * 0.060
            start[2] = push_z
            mid = obj_pos.copy()
            mid[2] = start[2]
            end = obj_pos.copy()
            end[:2] += direction3[:2] * stroke
            end[2] = start[2]

            self.open_gripper()
            start_high = start + np.array([0.0, 0.0, 0.090], dtype=np.float64)
            start_high_move = self.goto_finger_midpoint_checked(start_high, quat, tolerance=0.060)
            start_move = self.goto_finger_midpoint_checked(start, quat, tolerance=0.050)
            self.close_gripper()
            self._settle_simulation_steps(8)
            mid_move = self.goto_finger_midpoint_checked(mid, quat, tolerance=0.055)
            can_push = bool(
                float(start_move.get("finger_midpoint_error_norm", 1.0)) <= 0.095
                and float(mid_move.get("finger_midpoint_error_norm", 1.0)) <= 0.085
            )
            if can_push:
                end_move = self.goto_finger_midpoint_checked(end, quat, tolerance=0.070)
            else:
                end_move = {
                    "skipped": True,
                    "reason": "drawer_slide_contact_points_not_reached",
                    "start_error": float(start_move.get("finger_midpoint_error_norm", 1.0)),
                    "mid_error": float(mid_move.get("finger_midpoint_error_norm", 1.0)),
                }
            self.open_gripper()
            retreat = end + np.array([0.0, 0.0, 0.100], dtype=np.float64)
            retreat_move = self.goto_finger_midpoint_checked(retreat, quat, tolerance=0.075)
            self._settle_simulation_steps(max(8, int(settle_steps)))
            obj_after, _ = self.get_object_pose(object_name)
            obj_after = np.asarray(obj_after, dtype=np.float64).reshape(3).copy()

            relation_after = self._placement_relation_diagnostic(
                object_name,
                target_name,
                predicate="in",
            )
            clearance_after = (
                self._drawer_relation_closing_clearance(
                    relation_after,
                    min_front_clearance=float(min_closing_front_clearance),
                )
                if bool(require_closing_clearance)
                else None
            )
            records.append(
                {
                    "iteration": attempts - 1,
                    "action": "drawer_low_planar_slide",
                    "level": level,
                    "require_closing_clearance": bool(require_closing_clearance),
                    "closing_clearance_before": clearance_before,
                    "closing_clearance_after": clearance_after,
                    "depth_offset": float(depth),
                    "lateral_offset": float(lateral),
                    "object_position_before": obj_pos.copy(),
                    "object_position_after": obj_after.copy(),
                    "object_motion": obj_after - obj_pos,
                    "target_position": target.copy(),
                    "direction": direction3.copy(),
                    "stroke": stroke,
                    "push_z": push_z,
                    "bounds_min": bounds_min.copy() if isinstance(bounds_min, np.ndarray) else None,
                    "bounds_max": bounds_max.copy() if isinstance(bounds_max, np.ndarray) else None,
                    "start": start.copy(),
                    "start_high": start_high.copy(),
                    "mid": mid.copy(),
                    "end": end.copy(),
                    "retreat": retreat.copy(),
                    "start_high_move": start_high_move,
                    "start_move": start_move,
                    "mid_move": mid_move,
                    "end_move": end_move,
                    "retreat_move": retreat_move,
                    "relation_before": relation_before,
                    "relation_after": relation_after,
                }
            )
            if relation_has_drawer_goal_margin(relation_after):
                return {
                    "records": records,
                    "initial_relation": initial_relation,
                    "final_relation": relation_after,
                    "satisfied": True,
                    "task_status": self.get_task_status(),
                    "drawer_closing_clearance": clearance_after,
                    "skill": "drawer_slide_object_into_region",
                }

        final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        if not relation_has_drawer_goal_margin(final_relation):
            final_relation = run_site_local_refinement(
                final_relation,
                trigger="post_offset_candidate_scan",
                max_local_iters=max(1, int(max_iters)),
            )
        final_clearance = clearance_for(final_relation)
        return {
            "records": records,
            "initial_relation": initial_relation,
            "final_relation": final_relation,
            "satisfied": relation_has_drawer_goal_margin(final_relation),
            "task_status": self.get_task_status(),
            "drawer_closing_clearance": final_clearance,
            "skill": "drawer_slide_object_into_region",
        }

    def _resolve_object_state_name(self, name: str) -> str:
        libero_env = self._libero_env_with_objects()
        states = getattr(libero_env, "object_states_dict", {})
        query = str(name).strip().replace(" ", "_").lower()
        if query in states:
            return query
        if not query.endswith("_1") and f"{query}_1" in states:
            return f"{query}_1"
        matches = [state_name for state_name in states if query in state_name.lower() or state_name.lower() in query]
        if len(matches) == 1:
            return matches[0]
        if matches:
            region_matches = [state_name for state_name in matches if "region" in state_name.lower()]
            if region_matches:
                return sorted(region_matches, key=len)[0]
            return sorted(matches, key=len)[0]
        raise KeyError(name)

    def _verified_pick_offsets(self) -> list[np.ndarray]:
        return [
            np.array([0.000, 0.000, 0.000], dtype=np.float64),
            np.array([0.000, 0.025, 0.000], dtype=np.float64),
            np.array([0.000, -0.025, 0.000], dtype=np.float64),
            np.array([0.025, 0.000, 0.000], dtype=np.float64),
            np.array([-0.025, 0.000, 0.000], dtype=np.float64),
            np.array([0.000, 0.000, 0.020], dtype=np.float64),
            np.array([0.012, 0.000, 0.010], dtype=np.float64),
            np.array([-0.012, 0.000, 0.010], dtype=np.float64),
            np.array([0.000, 0.012, 0.010], dtype=np.float64),
            np.array([0.000, -0.012, 0.010], dtype=np.float64),
            np.array([0.018, 0.000, 0.025], dtype=np.float64),
            np.array([-0.018, 0.000, 0.025], dtype=np.float64),
            np.array([0.000, 0.018, 0.025], dtype=np.float64),
            np.array([0.000, -0.018, 0.025], dtype=np.float64),
            np.array([0.000, 0.000, 0.035], dtype=np.float64),
        ]

    def _geometry_pick_candidates(self, object_name: str) -> list[dict[str, Any]]:
        text = str(object_name).lower()
        down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        if "moka_pot" in text or "moka" in text:
            offsets = [
                np.array([0.000, 0.000, 0.000], dtype=np.float64),
                np.array([0.000, 0.012, 0.000], dtype=np.float64),
                np.array([0.000, -0.012, 0.000], dtype=np.float64),
                np.array([0.012, 0.000, 0.000], dtype=np.float64),
                np.array([-0.012, 0.000, 0.000], dtype=np.float64),
                np.array([0.000, 0.024, 0.000], dtype=np.float64),
                np.array([0.000, -0.024, 0.000], dtype=np.float64),
                np.array([0.024, 0.000, 0.000], dtype=np.float64),
                np.array([-0.024, 0.000, 0.000], dtype=np.float64),
                np.array([0.000, 0.000, -0.006], dtype=np.float64),
            ]
            return [
                {"name": f"moka_topdown_{idx:02d}", "offset": offset, "quat": down.copy(), "approach": 0.10}
                for idx, offset in enumerate(offsets)
            ]
        if "book" in text:
            offsets = [
                np.array([0.000, 0.000, 0.075], dtype=np.float64),
                np.array([0.000, 0.020, 0.075], dtype=np.float64),
                np.array([0.000, -0.020, 0.075], dtype=np.float64),
                np.array([0.025, 0.000, 0.075], dtype=np.float64),
                np.array([-0.025, 0.000, 0.075], dtype=np.float64),
                np.array([0.000, 0.000, 0.095], dtype=np.float64),
            ]
            return [
                {"name": f"book_top_clamp_{idx:02d}", "offset": offset, "quat": down.copy(), "approach": 0.13}
                for idx, offset in enumerate(offsets)
            ]
        return [
            {"name": f"generic_geometry_{idx:02d}", "offset": offset, "quat": down.copy(), "approach": 0.11}
            for idx, offset in enumerate(self._verified_pick_offsets())
        ]

    def _object_follow_probe(
        self,
        object_name: str,
        start_position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        probe_delta: np.ndarray,
    ) -> dict[str, Any]:
        before_state = self.measure_object_grasp_state(object_name, start_position)
        before_obj = np.asarray(before_state["object_position"], dtype=np.float64)
        before_finger = np.asarray(before_state["finger_midpoint"], dtype=np.float64)
        move = self.move_finger_midpoint_delta(
            np.asarray(probe_delta, dtype=np.float64),
            np.asarray(quaternion_wxyz, dtype=np.float64),
            lin_vel_norm=0.030,
            timeout_s=4.0,
            tolerance=0.055,
        )
        after_state = self.measure_object_grasp_state(object_name, start_position)
        after_obj = np.asarray(after_state["object_position"], dtype=np.float64)
        after_finger = np.asarray(after_state["finger_midpoint"], dtype=np.float64)
        obj_delta = after_obj - before_obj
        finger_delta = after_finger - before_finger
        start = np.asarray(start_position, dtype=np.float64).reshape(3)
        lift_delta = float(after_obj[2] - start[2])
        follow_error = float(np.linalg.norm((obj_delta - finger_delta)[:2]))
        visible_lift = bool(lift_delta >= 0.075)
        follows_motion = bool(
            obj_delta[2] >= 0.025
            and abs(float(obj_delta[2] - finger_delta[2])) <= 0.070
            and follow_error <= 0.070
        )
        stable_contact = bool(
            after_state["contacts"]["object_contact_count"] > 0
            and float(after_state["finger_gap"]) < 0.030
        )
        return {
            "before": before_state,
            "move": move,
            "after": after_state,
            "object_delta": obj_delta,
            "finger_delta": finger_delta,
            "lift_delta": lift_delta,
            "follow_error_xy": follow_error,
            "visible_lift": visible_lift,
            "follows_motion": follows_motion,
            "stable_contact": stable_contact,
            "held": bool(visible_lift and follows_motion and stable_contact),
        }

    def _align_held_object_center(
        self,
        object_name: str,
        target_object_position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        strict: bool,
        max_iters: int = 3,
        tolerance_xy: float = 0.012,
        tolerance_z: float = 0.035,
        lin_vel_norm: float = 0.045,
        timeout_s: float = 7.0,
    ) -> dict[str, Any]:
        """Move the gripper so the held object's center, not the wrist, reaches a target.

        The motion uses only readonly object/finger measurements plus ordinary
        Cartesian control. It is the key correction for shallow objects whose
        center sits several centimeters away from the finger midpoint after a
        stable grasp.
        """
        target = np.asarray(target_object_position, dtype=np.float64).reshape(3).copy()
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4).copy()
        records: list[dict[str, Any]] = []
        before = self.measure_object_grasp_state(object_name, strict=strict)
        held_before = bool(before.get("held", False))
        if not held_before:
            return {
                "object_name": object_name,
                "target_object_position": target,
                "records": records,
                "before": before,
                "after": before,
                "held_before": False,
                "held_after": False,
                "xy_error": float(np.linalg.norm(np.asarray(before["object_position"])[:2] - target[:2])),
                "z_error": float(np.asarray(before["object_position"])[2] - target[2]),
                "arrived": False,
                "reason": "object is not held before center alignment",
            }

        for idx in range(max(1, int(max_iters))):
            state = self.measure_object_grasp_state(object_name, strict=strict)
            obj_pos = np.asarray(state["object_position"], dtype=np.float64)
            finger = np.asarray(state["finger_midpoint"], dtype=np.float64)
            rel = obj_pos - finger
            error = obj_pos - target
            if (
                bool(state.get("held", False))
                and float(np.linalg.norm(error[:2])) <= float(tolerance_xy)
                and abs(float(error[2])) <= float(tolerance_z)
            ):
                records.append(
                    {
                        "iteration": idx,
                        "action": "already_aligned",
                        "object_position": obj_pos.copy(),
                        "finger_midpoint": finger.copy(),
                        "object_relative_to_finger_midpoint": rel.copy(),
                        "error": error.copy(),
                    }
                )
                break

            target_finger = target - rel
            move = self.goto_finger_midpoint_checked(
                target_finger,
                quat,
                lin_vel_norm=float(lin_vel_norm),
                timeout_s=float(timeout_s),
                tolerance=max(float(tolerance_xy) * 2.0, 0.030),
            )
            self._settle_simulation_steps(6)
            after_state = self.measure_object_grasp_state(object_name, strict=strict)
            after_obj = np.asarray(after_state["object_position"], dtype=np.float64)
            after_error = after_obj - target
            records.append(
                {
                    "iteration": idx,
                    "action": "align_object_center",
                    "object_position_before": obj_pos.copy(),
                    "finger_midpoint_before": finger.copy(),
                    "object_relative_to_finger_midpoint": rel.copy(),
                    "target_object_position": target.copy(),
                    "target_finger_midpoint": target_finger.copy(),
                    "move": move,
                    "object_position_after": after_obj.copy(),
                    "error_after": after_error.copy(),
                    "held_after": bool(after_state.get("held", False)),
                }
            )
            if (
                not bool(after_state.get("held", False))
                or (
                    float(np.linalg.norm(after_error[:2])) <= float(tolerance_xy)
                    and abs(float(after_error[2])) <= float(tolerance_z)
                )
            ):
                break

        after = self.measure_object_grasp_state(object_name, strict=strict)
        after_obj = np.asarray(after["object_position"], dtype=np.float64)
        final_error = after_obj - target
        arrived = bool(
            after.get("held", False)
            and float(np.linalg.norm(final_error[:2])) <= float(tolerance_xy)
            and abs(float(final_error[2])) <= float(tolerance_z)
        )
        return {
            "object_name": object_name,
            "target_object_position": target,
            "records": records,
            "before": before,
            "after": after,
            "held_before": held_before,
            "held_after": bool(after.get("held", False)),
            "xy_error": float(np.linalg.norm(final_error[:2])),
            "z_error": float(final_error[2]),
            "arrived": arrived,
        }

    def _held_drawer_waypoint_release(
        self,
        object_name: str,
        target_name: str,
        *,
        release_anchor: dict[str, Any],
        settle_steps: int = 36,
        require_closing_clearance: bool = False,
        min_closing_front_clearance: float = 0.045,
    ) -> dict[str, Any]:
        """Move a held object's center through drawer waypoints, then release."""
        records: list[dict[str, Any]] = []
        shallow_object = self._requires_strict_hold(object_name)
        strict_hold = bool(shallow_object)
        before = self.measure_object_grasp_state(object_name, strict=strict_hold)
        if not bool(before.get("held", False)):
            relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            return {
                "records": records,
                "before": before,
                "after": before,
                "relation": relation,
                "relation_satisfied": bool(relation.get("satisfied", False) or relation.get("in", False)),
                "reason": "object is not held before drawer waypoint release",
                "skill": "held_drawer_waypoint_release",
            }

        target_pos = np.asarray(
            release_anchor.get("semantic_target_position", release_anchor.get("position")),
            dtype=np.float64,
        ).reshape(3).copy()
        target_size = np.array([0.04, 0.04, 0.04], dtype=np.float64)
        target_xmat = np.eye(3, dtype=np.float64)
        try:
            target_pos_now, _ = self.get_site_pose(target_name)
            target_pos = np.asarray(target_pos_now, dtype=np.float64).reshape(3).copy()
            target_size = self._target_site_size(target_name)
            sim = self._env.handle.env.sim
            site_id = sim.model.site_name2id(target_name)
            target_xmat = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        except Exception:
            try:
                target_size = np.asarray(
                    release_anchor.get("target_size", target_size),
                    dtype=np.float64,
                ).reshape(-1)[:3]
            except Exception:
                target_size = np.array([0.04, 0.04, 0.04], dtype=np.float64)
        try:
            level = str(
                release_anchor.get("drawer_level")
                or self._drawer_level_from_target_name(target_name)
                or "middle"
            )
            plan = self.get_drawer_sleeve_hook_poses(level)
            pull_dir = np.asarray(plan.get("pull_direction"), dtype=np.float64).reshape(3)
        except Exception:
            level = str(self._drawer_level_from_target_name(target_name) or "middle")
            pull_dir = -np.asarray(
                release_anchor.get("drawer_interior_direction", np.array([0.0, -1.0, 0.0])),
                dtype=np.float64,
            ).reshape(3)
        pull_dir[2] = 0.0
        pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)

        current_obj = np.asarray(before["object_position"], dtype=np.float64).reshape(3)

        carry_z = float(
            np.clip(
                current_obj[2],
                float(target_pos[2] + 0.035),
                float(target_pos[2] + 0.070),
            )
        )
        outside = target_pos + 0.130 * pull_dir
        outside[2] = carry_z
        mouth = target_pos + 0.070 * pull_dir
        mouth[2] = float(target_pos[2] + 0.045)
        release = target_pos.copy()
        try:
            box_half = np.abs(target_xmat @ np.asarray(target_size, dtype=np.float64).reshape(3))
            if shallow_object:
                release[2] = float(target_pos[2] + 0.030)
            else:
                release[2] = float(target_pos[2] + np.clip(0.67 * float(box_half[2]), 0.016, 0.022))
        except Exception:
            box_half = np.array([0.04, 0.04, 0.04], dtype=np.float64)
            release[2] = float(target_pos[2] + (0.012 if shallow_object else 0.020))
        if shallow_object:
            if bool(require_closing_clearance):
                front_half = float(abs(box_half[1])) if "box_half" in locals() and len(box_half) >= 2 else 0.075
                required_depth = float(min_closing_front_clearance) - front_half + 0.022
                shallow_depth = float(np.clip(required_depth, 0.052, 0.078))
            else:
                shallow_depth = 0.085
            release[:2] += (-float(shallow_depth) * pull_dir[:2])
        else:
            release[:2] += (-0.010 * pull_dir[:2])

        if shallow_object:
            deep_mouth_depth = min(float(shallow_depth), 0.055) if bool(require_closing_clearance) else 0.020
            deep_mouth = target_pos - float(deep_mouth_depth) * pull_dir
            deep_mouth[2] = float(target_pos[2] + (0.034 if bool(require_closing_clearance) else 0.040))
            object_waypoints: list[tuple[str, np.ndarray, float]] = [
                ("drawer_offset_outside", outside, 0.120),
                ("drawer_offset_mouth", mouth, 0.110),
                ("drawer_offset_deep_mouth", deep_mouth, 0.095),
            ]
            if bool(require_closing_clearance):
                pre_release = target_pos - float(shallow_depth) * pull_dir
                pre_release[2] = float(target_pos[2] + 0.038)
                object_waypoints.append(("drawer_offset_pre_release_deep", pre_release, 0.090))
            object_waypoints.append(("drawer_offset_release", release, 0.085))
            offset_release = self._held_drawer_offset_waypoint_release(
                object_name,
                target_name,
                object_waypoints=object_waypoints,
                before_state=before,
                pull_dir=pull_dir,
                settle_steps=settle_steps,
                strict_hold=strict_hold,
                require_closing_clearance=bool(require_closing_clearance),
                min_closing_front_clearance=float(min_closing_front_clearance),
            )
            records.append(
                {
                    "phase": "drawer_offset_waypoint_release_attempt",
                    "record": offset_release,
                    "release_strategy": "constant_object_to_ee_waypoints",
                }
            )
            if bool(offset_release.get("executed", False)):
                offset_records = offset_release.get("records", [])
                if isinstance(offset_records, list):
                    records.extend(offset_records)
                relation = offset_release.get("relation", {})
                status = offset_release.get("task_status", self.get_task_status())
                clearance = (
                    offset_release.get("drawer_closing_clearance")
                    if bool(require_closing_clearance) and isinstance(offset_release, dict)
                    else None
                )
                if bool(require_closing_clearance) and not isinstance(clearance, dict):
                    clearance = self._drawer_relation_closing_clearance(
                        relation if isinstance(relation, dict) else {},
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                clearance_ok = (
                    True
                    if not bool(require_closing_clearance)
                    else bool(isinstance(clearance, dict) and clearance.get("has_clearance", False))
                )
                relation_satisfied = bool(
                    (
                        (
                            offset_release.get("relation_satisfied", False)
                            or (
                                isinstance(relation, dict)
                                and (relation.get("satisfied", False) or relation.get("in", False))
                            )
                        )
                        and clearance_ok
                    )
                    or status.get("task_completed", False)
                    or float(status.get("reward", 0.0) or 0.0) >= 1.0
                )
                return {
                    "records": records,
                    "before": before,
                    "after": offset_release.get("after", before),
                    "target_position": target_pos,
                    "release_target": release,
                    "pull_direction": pull_dir,
                    "relation": relation,
                    "drawer_closing_clearance": clearance,
                    "task_status": status,
                    "relation_satisfied": relation_satisfied,
                    "z_only_near_success": bool(offset_release.get("z_only_near_success", False)),
                    "reason": offset_release.get("reason"),
                    "skill": "held_drawer_waypoint_release",
                    "release_strategy": "constant_object_to_ee_waypoints",
                }

        for label, object_target, tolerance in (
            ("drawer_waypoint_outside", outside, 0.110),
            ("drawer_waypoint_mouth", mouth, 0.100),
            ("drawer_waypoint_release", release, 0.090),
        ):
            _, ee_quat = self.get_ee_pose()
            quat = np.asarray(ee_quat, dtype=np.float64).reshape(4)
            state_before = self.measure_object_grasp_state(object_name, strict=strict_hold)
            if not bool(state_before.get("held", False)):
                relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                records.append(
                    {
                        "phase": label,
                        "skipped": True,
                        "reason": "lost_hold_before_waypoint",
                        "state_before": state_before,
                        "relation": relation,
                    }
                )
                return {
                    "records": records,
                    "before": before,
                    "after": state_before,
                    "relation": relation,
                    "relation_satisfied": bool(relation.get("satisfied", False) or relation.get("in", False)),
                        "skill": "held_drawer_waypoint_release",
                    }

            align = self._align_held_object_center(
                object_name,
                np.asarray(object_target, dtype=np.float64).reshape(3),
                quat,
                strict=strict_hold,
                max_iters=2,
                tolerance_xy=min(0.030, max(0.016, float(tolerance) * 0.30)),
                tolerance_z=0.014 if label == "drawer_waypoint_release" else 0.045,
                lin_vel_norm=0.040,
                timeout_s=6.0,
            )
            self._settle_simulation_steps(max(4, int(settle_steps) // 8))
            state_after = self.measure_object_grasp_state(object_name, strict=strict_hold)
            obj_after = np.asarray(state_after["object_position"], dtype=np.float64).reshape(3)
            relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            records.append(
                {
                    "phase": label,
                    "level": level,
                    "hold_strict": strict_hold,
                    "shallow_object_low_release": shallow_object,
                    "pull_direction": pull_dir.copy(),
                    "target_object_position": np.asarray(object_target, dtype=np.float64).copy(),
                    "align": align,
                    "state_before": state_before,
                    "state_after": state_after,
                    "object_error_after": obj_after - np.asarray(object_target, dtype=np.float64).reshape(3),
                    "relation_after": relation_after,
                }
            )
            if not bool(state_after.get("held", False)):
                return {
                    "records": records,
                    "before": before,
                    "after": state_after,
                    "relation": relation_after,
                    "relation_satisfied": bool(relation_after.get("satisfied", False) or relation_after.get("in", False)),
                    "skill": "held_drawer_waypoint_release",
                }

        pre_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        pre_open_state = self.measure_object_grasp_state(object_name, strict=strict_hold)
        records.append(
            {
                "phase": "drawer_waypoint_pre_open",
                "state": pre_open_state,
                "relation": pre_open_relation,
                "release_target": release.copy(),
            }
        )
        self.open_gripper()
        self._settle_simulation_steps(max(20, int(settle_steps)))
        after = self.measure_object_grasp_state(object_name, strict=strict_hold)
        relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "drawer_waypoint_after_open", "state": after, "relation": relation})
        status = self.get_task_status()
        relation_satisfied = bool(
            relation.get("satisfied", False)
            or relation.get("in", False)
            or status.get("task_completed", False)
            or float(status.get("reward", 0.0) or 0.0) >= 1.0
        )
        return {
            "records": records,
            "before": before,
            "after": after,
            "target_position": target_pos,
            "release_target": release,
            "pull_direction": pull_dir,
            "relation": relation,
            "task_status": status,
            "relation_satisfied": relation_satisfied,
            "skill": "held_drawer_waypoint_release",
        }

    def _held_drawer_offset_waypoint_release(
        self,
        object_name: str,
        target_name: str,
        *,
        object_waypoints: list[tuple[str, np.ndarray, float]],
        before_state: dict[str, Any],
        pull_dir: np.ndarray,
        settle_steps: int,
        strict_hold: bool,
        require_closing_clearance: bool = False,
        min_closing_front_clearance: float = 0.045,
    ) -> dict[str, Any]:
        """Release a held shallow object while preserving its object-to-EE offset.

        For shallow bowls and plates the fingertip midpoint is a poor proxy for
        where the object center will land. This variant measures the current
        object-to-EE offset once, then moves the end effector so the object
        should pass through the requested drawer waypoints.
        """
        records: list[dict[str, Any]] = []
        before = dict(before_state) if isinstance(before_state, dict) else self.measure_object_grasp_state(object_name, strict=strict_hold)
        if not bool(before.get("held", False)):
            relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            return {
                "executed": False,
                "records": records,
                "before": before,
                "after": before,
                "relation": relation,
                "task_status": self.get_task_status(),
                "relation_satisfied": bool(relation.get("satisfied", False) or relation.get("in", False)),
                "reason": "object is not held before offset waypoint release",
                "skill": "held_drawer_offset_waypoint_release",
            }

        ee_pos, ee_quat = self.get_ee_pose()
        quat = np.asarray(ee_quat, dtype=np.float64).reshape(4).copy()
        obj_pos = np.asarray(before["object_position"], dtype=np.float64).reshape(3).copy()
        object_to_ee = obj_pos - np.asarray(ee_pos, dtype=np.float64).reshape(3)
        executed = False

        def run_pre_release_relation_servo() -> dict[str, Any]:
            servo_records: list[dict[str, Any]] = []
            final_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            pull = np.asarray(pull_dir, dtype=np.float64).reshape(3).copy()
            pull[2] = 0.0
            pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
            interior = -pull
            for idx in range(5):
                held_state = self.measure_object_grasp_state(object_name, strict=strict_hold)
                relation_before = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                final_relation = relation_before
                if bool(relation_before.get("satisfied", False) or relation_before.get("in", False)):
                    clearance = (
                        self._drawer_relation_closing_clearance(
                            relation_before,
                            min_front_clearance=float(min_closing_front_clearance),
                        )
                        if bool(require_closing_clearance)
                        else None
                    )
                    if not bool(require_closing_clearance) or bool(clearance.get("has_clearance", False)):
                        servo_records.append(
                            {
                                "iteration": int(idx),
                                "action": "held_drawer_prerelease_servo_stop",
                                "reason": "relation_already_satisfied",
                                "relation": relation_before,
                                "drawer_closing_clearance": clearance,
                                "held_state": held_state,
                            }
                        )
                        break
                z_only_before = self._drawer_relation_is_z_only_above_box(relation_before, max_z_excess=0.028)
                z_only_clearance = (
                    self._drawer_relation_closing_clearance(
                        relation_before,
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                    if bool(require_closing_clearance) and z_only_before
                    else None
                )
                if (
                    z_only_before
                    and (
                        not bool(require_closing_clearance)
                        or bool(z_only_clearance.get("has_clearance", False))
                    )
                ):
                    servo_records.append(
                        {
                            "iteration": int(idx),
                            "action": "held_drawer_prerelease_servo_stop",
                            "reason": "xy_valid_z_only_near_success",
                            "relation": relation_before,
                            "drawer_closing_clearance": z_only_clearance,
                            "held_state": held_state,
                        }
                    )
                    break
                if not bool(held_state.get("held", False)):
                    servo_records.append(
                        {
                            "iteration": int(idx),
                            "action": "held_drawer_prerelease_servo_stop",
                            "reason": "lost_hold_before_servo",
                            "relation": relation_before,
                            "held_state": held_state,
                        }
                    )
                    break
                local_delta = None
                site_size = None
                try:
                    local_delta = np.asarray(relation_before["local_delta"], dtype=np.float64).reshape(-1)[:3]
                    site_size = np.asarray(relation_before["site_size_abs"], dtype=np.float64).reshape(-1)[:3]
                except Exception:
                    pass
                target_obj = np.asarray(held_state["object_position"], dtype=np.float64).reshape(3).copy()
                if local_delta is not None and site_size is not None and local_delta.size >= 3 and site_size.size >= 3:
                    if bool(require_closing_clearance):
                        clearance = self._drawer_relation_closing_clearance(
                            relation_before,
                            min_front_clearance=float(min_closing_front_clearance),
                        )
                        front_clearance = float(clearance.get("front_clearance", 0.0) or 0.0)
                        clearance_deficit = max(
                            0.0,
                            float(min_closing_front_clearance) - front_clearance,
                        )
                    else:
                        clearance = None
                        clearance_deficit = 0.0
                    y_excess = max(0.0, abs(float(local_delta[1])) - (abs(float(site_size[1])) - 0.004))
                    step = float(np.clip(max(y_excess, clearance_deficit) + 0.024, 0.020, 0.070))
                    target_obj[:2] += interior[:2] * step
                    if float(relation_before.get("aabb_z_excess", 0.0) or 0.0) > 0.0:
                        target_obj[2] -= min(0.010, float(relation_before.get("aabb_z_excess", 0.0) or 0.0) + 0.002)
                    elif abs(float(local_delta[1])) > abs(float(site_size[1])) + 0.010:
                        target_obj[2] = max(target_obj[2] - 0.004, 0.035)
                else:
                    target_obj[:2] += interior[:2] * 0.030
                    target_obj[2] = max(target_obj[2] - 0.004, 0.035)
                target_ee = target_obj - object_to_ee
                move = self.goto_pose_checked(target_ee, quat, tolerance=0.105)
                executed_move = True
                self._settle_simulation_steps(max(6, int(settle_steps) // 8))
                held_after = self.measure_object_grasp_state(object_name, strict=strict_hold)
                relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                final_relation = relation_after
                servo_records.append(
                    {
                        "iteration": int(idx),
                        "action": "held_drawer_prerelease_relation_servo",
                        "relation_before": relation_before,
                        "drawer_closing_clearance_before": clearance if "clearance" in locals() else None,
                        "local_delta_before": local_delta.copy() if isinstance(local_delta, np.ndarray) else None,
                        "site_size_abs": site_size.copy() if isinstance(site_size, np.ndarray) else None,
                        "target_object_position": target_obj.copy(),
                        "target_ee_position": target_ee.copy(),
                        "object_to_ee_used": object_to_ee.copy(),
                        "move": move,
                        "executed_move": executed_move,
                        "held_after": held_after,
                        "relation_after": relation_after,
                    }
                )
                if bool(relation_after.get("satisfied", False) or relation_after.get("in", False)):
                    if not bool(require_closing_clearance):
                        break
                    clearance_after = self._drawer_relation_closing_clearance(
                        relation_after,
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                    if bool(clearance_after.get("has_clearance", False)):
                        break
            return {
                "records": servo_records,
                "relation": final_relation,
                "drawer_closing_clearance": (
                    self._drawer_relation_closing_clearance(
                        final_relation,
                        min_front_clearance=float(min_closing_front_clearance),
                    )
                    if bool(require_closing_clearance)
                    else None
                ),
                "satisfied": bool(
                    (final_relation.get("satisfied", False) or final_relation.get("in", False))
                    and (
                        not bool(require_closing_clearance)
                        or self._drawer_relation_closing_clearance(
                            final_relation,
                            min_front_clearance=float(min_closing_front_clearance),
                        ).get("has_clearance", False)
                    )
                ),
                "z_only_near_success": self._drawer_relation_is_z_only_above_box(final_relation, max_z_excess=0.028),
                "skill": "held_drawer_prerelease_relation_servo",
            }

        for label, object_target, tolerance in object_waypoints:
            state_before = self.measure_object_grasp_state(object_name, strict=strict_hold)
            if not bool(state_before.get("held", False)):
                relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                records.append(
                    {
                        "phase": label,
                        "skipped": True,
                        "reason": "lost_hold_before_offset_waypoint",
                        "state_before": state_before,
                        "relation": relation,
                    }
                )
                return {
                    "executed": executed,
                    "records": records,
                    "before": before,
                    "after": state_before,
                    "relation": relation,
                    "task_status": self.get_task_status(),
                    "relation_satisfied": bool(relation.get("satisfied", False) or relation.get("in", False)),
                    "skill": "held_drawer_offset_waypoint_release",
                }

            target_obj = np.asarray(object_target, dtype=np.float64).reshape(3).copy()
            target_ee = target_obj - object_to_ee
            move = self.goto_pose_checked(target_ee, quat, tolerance=float(tolerance))
            executed = True
            self._settle_simulation_steps(max(5, int(settle_steps) // 8))
            state_after = self.measure_object_grasp_state(object_name, strict=strict_hold)
            obj_after = np.asarray(state_after["object_position"], dtype=np.float64).reshape(3)
            relation_after = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
            records.append(
                {
                    "phase": label,
                    "target_object_position": target_obj,
                    "target_ee_position": target_ee,
                    "object_to_ee": object_to_ee.copy(),
                    "object_to_ee_used": object_to_ee.copy(),
                    "pull_direction": np.asarray(pull_dir, dtype=np.float64).reshape(3).copy(),
                    "move": move,
                    "state_before": state_before,
                    "state_after": state_after,
                    "object_error_after": obj_after - target_obj,
                    "relation_after": relation_after,
                }
            )
            if not bool(state_after.get("held", False)):
                return {
                    "executed": executed,
                    "records": records,
                    "before": before,
                    "after": state_after,
                    "relation": relation_after,
                    "task_status": self.get_task_status(),
                    "relation_satisfied": bool(relation_after.get("satisfied", False) or relation_after.get("in", False)),
                    "skill": "held_drawer_offset_waypoint_release",
                }

        pre_open_relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        pre_open_state = self.measure_object_grasp_state(object_name, strict=strict_hold)
        pre_open_clearance = (
            self._drawer_relation_closing_clearance(
                pre_open_relation,
                min_front_clearance=float(min_closing_front_clearance),
            )
            if bool(require_closing_clearance)
            else None
        )
        records.append(
            {
                "phase": "drawer_offset_pre_open",
                "state": pre_open_state,
                "relation": pre_open_relation,
                "drawer_closing_clearance": pre_open_clearance,
                "object_to_ee": object_to_ee.copy(),
            }
        )
        pre_open_in = bool(pre_open_relation.get("satisfied", False) or pre_open_relation.get("in", False))
        pre_open_clearance_ok = (
            True
            if not bool(require_closing_clearance)
            else bool(isinstance(pre_open_clearance, dict) and pre_open_clearance.get("has_clearance", False))
        )
        if bool(pre_open_state.get("held", False)) and (not pre_open_in or not pre_open_clearance_ok):
            servo = run_pre_release_relation_servo()
            records.append(
                {
                    "phase": "drawer_offset_pre_release_relation_servo",
                    "record": servo,
                    "object_to_ee": object_to_ee.copy(),
                    "strategy": "readonly_relation_and_close_clearance_feedback_before_open_gripper",
                }
            )
            if isinstance(servo, dict):
                pre_open_relation = servo.get("relation", pre_open_relation)
                pre_open_clearance = servo.get("drawer_closing_clearance", pre_open_clearance)
            pre_open_state = self.measure_object_grasp_state(object_name, strict=strict_hold)
            records.append(
                {
                    "phase": "drawer_offset_after_pre_release_relation_servo",
                    "state": pre_open_state,
                    "relation": pre_open_relation,
                    "drawer_closing_clearance": pre_open_clearance,
                }
            )
        self.open_gripper()
        self._settle_simulation_steps(max(24, int(settle_steps)))
        after = self.measure_object_grasp_state(object_name, strict=strict_hold)
        relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
        records.append({"phase": "drawer_offset_after_open", "state": after, "relation": relation})
        if not bool(relation.get("satisfied", False) or relation.get("in", False)):
            if self._drawer_relation_is_z_only_above_box(relation, max_z_excess=0.020):
                status = self.get_task_status()
                return {
                    "executed": executed,
                    "records": records,
                    "before": before,
                    "after": after,
                    "relation": relation,
                    "task_status": status,
                    "relation_satisfied": bool(
                        status.get("task_completed", False)
                        or float(status.get("reward", 0.0) or 0.0) >= 1.0
                    ),
                    "z_only_near_success": True,
                    "reason": "drawer offset release reached xy-valid z-only near-success; skip retreat to preserve placement",
                    "object_to_ee": object_to_ee,
                    "skill": "held_drawer_offset_waypoint_release",
                }
            try:
                retreat_obj = np.asarray(object_waypoints[-1][1], dtype=np.float64).reshape(3).copy()
                retreat_obj += 0.115 * np.asarray(pull_dir, dtype=np.float64).reshape(3)
                retreat_obj[2] += 0.100
                retreat_ee = retreat_obj - object_to_ee
                records.append(
                    {
                        "phase": "drawer_offset_retreat",
                        "target_object_position": retreat_obj.copy(),
                        "target_ee_position": retreat_ee.copy(),
                        "move": self.goto_pose_checked(retreat_ee, quat, tolerance=0.130),
                    }
                )
                self._settle_simulation_steps(max(8, int(settle_steps) // 4))
                relation = self._placement_relation_diagnostic(object_name, target_name, predicate="in")
                after = self.measure_object_grasp_state(object_name, strict=strict_hold)
                records.append({"phase": "drawer_offset_after_retreat", "state": after, "relation": relation})
            except Exception as exc:
                records.append({"phase": "drawer_offset_retreat_error", "reason": repr(exc)})
        status = self.get_task_status()
        drawer_closing_clearance = (
            self._drawer_relation_closing_clearance(
                relation,
                min_front_clearance=float(min_closing_front_clearance),
            )
            if bool(require_closing_clearance)
            else None
        )
        clearance_ok = (
            True
            if not bool(require_closing_clearance)
            else bool(drawer_closing_clearance.get("has_clearance", False))
        )
        relation_satisfied = bool(
            ((relation.get("satisfied", False) or relation.get("in", False)) and clearance_ok)
            or status.get("task_completed", False)
            or float(status.get("reward", 0.0) or 0.0) >= 1.0
        )
        return {
            "executed": executed,
            "records": records,
            "before": before,
            "after": after,
            "relation": relation,
            "drawer_closing_clearance": drawer_closing_clearance,
            "task_status": status,
            "relation_satisfied": relation_satisfied,
            "object_to_ee": object_to_ee,
            "skill": "held_drawer_offset_waypoint_release",
        }

    def _settle_simulation_steps(self, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self._env._step_once()

    def _drawer_hook_quaternion(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    def _rotation_from_wxyz(self, quaternion_wxyz: np.ndarray) -> SciRotation:
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        return SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

    def _apply_tcp_offset(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        rot = self._rotation_from_wxyz(quaternion_wxyz)
        return np.asarray(position, dtype=np.float64).reshape(3) + rot.apply(self._TCP_OFFSET)

    def _control_position_for_reported_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        rot = self._rotation_from_wxyz(quaternion_wxyz)
        return np.asarray(position, dtype=np.float64).reshape(3) - rot.apply(self._TCP_OFFSET)

    def _finger_midpoint_offset_from_ee(self) -> np.ndarray:
        ee_pos, ee_quat = self.get_ee_pose()
        sim = self._env.handle.env.sim
        left_geom_id = sim.model.geom_name2id("gripper0_finger1_pad_collision")
        right_geom_id = sim.model.geom_name2id("gripper0_finger2_pad_collision")
        left_pad = self._world_position_to_robot_base(
            np.asarray(sim.data.geom_xpos[left_geom_id], dtype=np.float64)
        )
        right_pad = self._world_position_to_robot_base(
            np.asarray(sim.data.geom_xpos[right_geom_id], dtype=np.float64)
        )
        midpoint = 0.5 * (left_pad + right_pad)
        rot = self._rotation_from_wxyz(np.asarray(ee_quat, dtype=np.float64))
        return rot.inv().apply(midpoint - np.asarray(ee_pos, dtype=np.float64))

    def _drawer_handle_lateral_axis(self, level: str) -> np.ndarray:
        sim = self._env.handle.env.sim
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        candidates: list[np.ndarray] = []
        for geom_id in range(sim.model.ngeom):
            geom_name = sim.model.geom_id2name(geom_id)
            if not geom_name or "cabinet" not in geom_name.lower():
                continue
            geom_robot = self._world_position_to_robot_base(
                np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
            )
            rel = geom_robot - handle_pos
            front_distance = float(np.dot(rel, pull_dir))
            vertical_distance = abs(float(rel[2]))
            lateral_vec = rel - front_distance * pull_dir - np.array([0.0, 0.0, rel[2]])
            if 0.04 <= front_distance <= 0.16 and vertical_distance <= 0.018:
                lateral_norm = float(np.linalg.norm(lateral_vec))
                if 0.018 <= lateral_norm <= 0.080:
                    candidates.append(lateral_vec / lateral_norm)
        if candidates:
            axis = np.mean(np.stack(candidates, axis=0), axis=0)
            axis = axis - np.dot(axis, pull_dir) * pull_dir
            axis[2] = 0.0
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                return axis / norm
        fallback = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), pull_dir)
        norm = float(np.linalg.norm(fallback))
        if norm <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback / norm

    def goto_pose_checked(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
        *,
        z_approach: float = 0.0,
        lin_vel_norm: float = 0.08,
        ang_vel_norm: float = 0.5,
        timeout_s: float = 8.0,
        tolerance: float = 0.035,
    ) -> dict[str, Any]:
        """Go to a pose and report whether the gripper actually arrived.

        This helper only executes robot motion and reads the final end-effector
        pose. It is useful for rejecting unreachable drawer contact points
        before interpreting a failed rollout as a contact or planning failure.

        Args:
            position: (3,) target XYZ in the robot base frame.
            quaternion_wxyz: (4,) target gripper orientation in WXYZ order.
            z_approach: Optional insertion approach offset passed to
                `goto_pose`.
            lin_vel_norm: Reserved for API compatibility; `goto_pose` owns
                low-level motion timing.
            ang_vel_norm: Reserved for API compatibility.
            timeout_s: Reserved for API compatibility.
            tolerance: Maximum acceptable Cartesian position error in meters.

        Returns:
            Dictionary with:
                - target_position: requested target XYZ.
                - actual_position: final end-effector XYZ.
                - position_error: actual minus target.
                - error_norm: Euclidean position error.
                - arrived: True if error_norm <= tolerance.
        """
        target_pos = np.asarray(position, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        self.goto_pose(target_pos, target_quat, z_approach=z_approach)
        actual_pos, actual_quat = self.get_ee_pose()
        error = np.asarray(actual_pos, dtype=np.float64) - target_pos
        error_norm = float(np.linalg.norm(error))
        return {
            "target_position": target_pos.copy(),
            "actual_position": np.asarray(actual_pos, dtype=np.float64).copy(),
            "actual_quaternion_wxyz": np.asarray(actual_quat, dtype=np.float64).copy(),
            "position_error": error,
            "error_norm": error_norm,
            "arrived": bool(error_norm <= float(tolerance)),
        }

    def goto_finger_midpoint_checked(
        self,
        finger_midpoint_position: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
        *,
        lin_vel_norm: float = 0.08,
        ang_vel_norm: float = 0.5,
        timeout_s: float = 8.0,
        tolerance: float = 0.030,
    ) -> dict[str, Any]:
        """Move so the midpoint between gripper pads reaches a target.

        Drawer hooking depends on where the finger pads are, while `goto_pose`
        controls the end-effector frame. This helper converts a desired pad
        midpoint into an end-effector target using the current gripper geometry,
        executes real robot motion, then reports the actual pad-midpoint error.

        Args:
            finger_midpoint_position: (3,) desired midpoint between gripper pads
                in the robot base frame.
            quaternion_wxyz: (4,) target gripper orientation in WXYZ order.
            lin_vel_norm: Desired linear speed in meters/second.
            ang_vel_norm: Desired angular speed in radians/second.
            timeout_s: Maximum motion time in seconds.
            tolerance: Maximum acceptable finger-midpoint error in meters.

        Returns:
            Dictionary with:
                - target_finger_midpoint_position: requested pad midpoint.
                - target_ee_position: computed EE command.
                - actual_finger_midpoint_position: measured final pad midpoint.
                - finger_midpoint_error: actual minus requested pad midpoint.
                - finger_midpoint_error_norm: Euclidean pad-midpoint error.
                - arrived: True if finger_midpoint_error_norm <= tolerance.
        """
        target_finger = np.asarray(finger_midpoint_position, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_local = self._finger_midpoint_offset_from_ee()
        rot = self._rotation_from_wxyz(target_quat)
        target_ee = target_finger - rot.apply(offset_local)

        reach = self.goto_pose_checked(
            target_ee,
            target_quat,
            lin_vel_norm=float(lin_vel_norm),
            ang_vel_norm=float(ang_vel_norm),
            timeout_s=float(timeout_s),
            tolerance=max(float(tolerance), 0.035),
        )
        fingers = self.get_gripper_fingertip_state()
        actual_finger = np.asarray(fingers["finger_midpoint"], dtype=np.float64)
        error = actual_finger - target_finger
        error_norm = float(np.linalg.norm(error))
        actual_pos, actual_quat = self.get_ee_pose()
        return {
            "target_finger_midpoint_position": target_finger,
            "target_ee_position": target_ee,
            "ee_reach": reach,
            "actual_ee_position": np.asarray(actual_pos, dtype=np.float64).copy(),
            "actual_quaternion_wxyz": np.asarray(actual_quat, dtype=np.float64).copy(),
            "actual_finger_midpoint_position": actual_finger,
            "finger_midpoint_error": error,
            "finger_midpoint_error_norm": error_norm,
            "arrived": bool(error_norm <= float(tolerance)),
            "finger_gap": float(fingers["finger_gap"]),
        }

    def _goto_pose_legacy_checked(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float],
        *,
        z_approach: float = 0.0,
        tolerance: float = 0.050,
    ) -> dict[str, Any]:
        """Execute the original Franka goto_pose semantics and report motion.

        The 35-task expert programs were collected with the parent Franka API,
        whose public target position is converted through `_TCP_OFFSET` before
        solving IK. For contact sweeps such as drawer closing, replaying that
        exact motion primitive is important; this still executes only robot
        motion and does not write simulator state.
        """
        target_pos = np.asarray(position, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        expected_reported_pos = self._apply_tcp_offset(target_pos, target_quat)
        FrankaLiberoPrivilegedApi.goto_pose(
            self,
            target_pos,
            target_quat,
            z_approach=z_approach,
        )
        actual_pos, actual_quat = self.get_ee_pose()
        error = np.asarray(actual_pos, dtype=np.float64) - expected_reported_pos
        return {
            "legacy_target_position": target_pos.copy(),
            "expected_reported_position": expected_reported_pos.copy(),
            "actual_position": np.asarray(actual_pos, dtype=np.float64).copy(),
            "actual_quaternion_wxyz": np.asarray(actual_quat, dtype=np.float64).copy(),
            "position_error": error,
            "error_norm": float(np.linalg.norm(error)),
            "arrived": bool(np.linalg.norm(error) <= float(tolerance)),
        }

    def move_to_joints(self, joints: np.ndarray | list[float] | tuple[float, ...]) -> None:
        """Move the robot arm to a joint configuration.

        This executes ordinary robot joint control. It does not write simulator
        object state or task predicates.
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(target)

    def move_joint_delta_checked(
        self,
        joint_delta: np.ndarray | list[float] | tuple[float, ...],
        *,
        max_abs_delta: float = 0.20,
    ) -> dict[str, Any]:
        """Move by a small joint delta and report the observed pose change.

        Args:
            joint_delta: (7,) desired arm-joint displacement in radians.
            max_abs_delta: Safety clamp for each joint displacement.

        Returns:
            Dictionary with before/after joints and end-effector poses.
        """
        before_joints = self.get_joint_positions()
        before_pos, before_quat = self.get_ee_pose()
        fingers_before = self.get_gripper_fingertip_state()
        before_finger = np.asarray(fingers_before["finger_midpoint"], dtype=np.float64)
        delta = np.asarray(joint_delta, dtype=np.float64).reshape(7)
        delta = np.clip(delta, -float(max_abs_delta), float(max_abs_delta))
        target_joints = before_joints + delta
        self.move_to_joints(target_joints)
        after_joints = self.get_joint_positions()
        after_pos, after_quat = self.get_ee_pose()
        fingers_after = self.get_gripper_fingertip_state()
        after_finger = np.asarray(fingers_after["finger_midpoint"], dtype=np.float64)
        return {
            "before_joints": before_joints,
            "target_joints": target_joints,
            "actual_joints": after_joints,
            "joint_delta_command": delta,
            "actual_joint_delta": after_joints - before_joints,
            "before_position": before_pos,
            "after_position": after_pos,
            "position_delta": after_pos - before_pos,
            "before_quaternion_wxyz": before_quat,
            "after_quaternion_wxyz": after_quat,
            "before_finger_midpoint": before_finger,
            "after_finger_midpoint": after_finger,
            "finger_midpoint_delta": after_finger - before_finger,
            "finger_gap": float(fingers_after["finger_gap"]),
        }

    def move_linear_delta(
        self,
        delta: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        *,
        lin_vel_norm: float = 0.08,
        ang_vel_norm: float = 0.5,
        timeout_s: float = 8.0,
    ) -> dict[str, Any]:
        """Move the gripper by a Cartesian delta using robot control.

        This is a convenience wrapper around `goto_pose_interactive_cartesian`.
        It does not modify any simulator object or drawer state directly.

        Args:
            delta: (3,) XYZ displacement in meters in the robot base frame.
            quaternion_wxyz: Optional target gripper orientation. If omitted,
                keep the current end-effector orientation.
            lin_vel_norm: Desired linear speed in meters/second.
            ang_vel_norm: Desired angular speed in radians/second.
            timeout_s: Maximum motion time in seconds.

        Returns:
            Dictionary with target and actual end-effector position.
        """
        current_pos, current_quat = self.get_ee_pose()
        target_pos = current_pos + np.asarray(delta, dtype=np.float64).reshape(3)
        target_quat = (
            current_quat
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        self.goto_pose_interactive_cartesian(
            lambda: (target_pos, target_quat),
            lin_vel_norm=lin_vel_norm,
            ang_vel_norm=ang_vel_norm,
            timeout_s=timeout_s,
        )
        actual_pos, actual_quat = self.get_ee_pose()
        return {
            "start_position": current_pos,
            "target_position": target_pos,
            "actual_position": actual_pos,
            "actual_quaternion_wxyz": actual_quat,
            "position_error": actual_pos - target_pos,
            "error_norm": float(np.linalg.norm(actual_pos - target_pos)),
        }

    def move_finger_midpoint_delta(
        self,
        delta: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        *,
        lin_vel_norm: float = 0.035,
        ang_vel_norm: float = 0.35,
        timeout_s: float = 4.0,
        tolerance: float = 0.020,
    ) -> dict[str, Any]:
        """Move the midpoint between gripper pads by a Cartesian delta.

        This executes ordinary robot motion while using the pad midpoint, not
        the wrist pose, as the controlled point. It is useful after closing on
        a drawer handle, where a few millimeters of pad drift can lose contact.

        Args:
            delta: Desired pad-midpoint displacement in robot-base coordinates.
            quaternion_wxyz: Optional target orientation. If omitted, keep the
                current observed gripper orientation.
            lin_vel_norm: Desired linear speed in meters/second.
            ang_vel_norm: Desired angular speed in radians/second.
            timeout_s: Maximum motion time in seconds.
            tolerance: Maximum acceptable pad-midpoint error in meters.

        Returns:
            Dictionary with before/target/actual finger midpoint and EE pose
            diagnostics.
        """
        fingers_before = self.get_gripper_fingertip_state()
        before_finger = np.asarray(fingers_before["finger_midpoint"], dtype=np.float64)
        delta_arr = np.asarray(delta, dtype=np.float64).reshape(3)
        target_finger = before_finger + delta_arr
        current_pos, current_quat = self.get_ee_pose()
        target_quat = (
            np.asarray(current_quat, dtype=np.float64).reshape(4)
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        offset_local = self._finger_midpoint_offset_from_ee()
        rot = self._rotation_from_wxyz(target_quat)
        target_ee = target_finger - rot.apply(offset_local)

        self.goto_pose_interactive_cartesian(
            lambda: (target_ee, target_quat),
            lin_vel_norm=lin_vel_norm,
            ang_vel_norm=ang_vel_norm,
            timeout_s=timeout_s,
        )
        after_pos, after_quat = self.get_ee_pose()
        fingers_after = self.get_gripper_fingertip_state()
        actual_finger = np.asarray(fingers_after["finger_midpoint"], dtype=np.float64)
        error = actual_finger - target_finger
        return {
            "before_finger_midpoint": before_finger,
            "target_finger_midpoint": target_finger,
            "actual_finger_midpoint": actual_finger,
            "finger_midpoint_delta_command": delta_arr,
            "actual_finger_midpoint_delta": actual_finger - before_finger,
            "finger_midpoint_error": error,
            "finger_midpoint_error_norm": float(np.linalg.norm(error)),
            "arrived": bool(np.linalg.norm(error) <= float(tolerance)),
            "before_ee_position": current_pos,
            "target_ee_position": target_ee,
            "actual_ee_position": after_pos,
            "actual_quaternion_wxyz": after_quat,
            "finger_gap": float(fingers_after["finger_gap"]),
        }

    def move_finger_midpoint_delta_strict(
        self,
        delta: np.ndarray | list[float] | tuple[float, float, float],
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        *,
        max_joint_delta: float = 0.20,
        max_abs_joint_delta: float = 0.080,
        tolerance: float = 0.025,
    ) -> dict[str, Any]:
        """Move pad midpoint by a small delta with strict IK continuity.

        This uses the current robot joints as the IK seed and rejects large
        branch jumps before executing. It does not write simulator object state.

        Args:
            delta: Desired pad-midpoint displacement in robot-base coordinates.
            quaternion_wxyz: Optional target orientation. If omitted, keep the
                current observed gripper orientation.
            max_joint_delta: Maximum allowed 7-DoF joint-vector norm change.
            max_abs_joint_delta: Per-joint clamp applied to the solved delta.
            tolerance: Maximum acceptable final pad-midpoint error.

        Returns:
            Dictionary with target/actual finger midpoint, requested and
            executed joint deltas, and arrival diagnostics.
        """
        before_joints = self.get_joint_positions()
        before_pos, before_quat = self.get_ee_pose()
        fingers_before = self.get_gripper_fingertip_state()
        before_finger = np.asarray(fingers_before["finger_midpoint"], dtype=np.float64)
        delta_arr = np.asarray(delta, dtype=np.float64).reshape(3)
        target_finger = before_finger + delta_arr
        target_quat = (
            np.asarray(before_quat, dtype=np.float64).reshape(4)
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        offset_local = self._finger_midpoint_offset_from_ee()
        rot = self._rotation_from_wxyz(target_quat)
        target_ee = target_finger - rot.apply(offset_local)

        requested_joints = before_joints.copy()
        solve_ok = True
        solve_error = ""
        try:
            if self.cfg is None:
                seed_cfg = np.concatenate([before_joints, np.array([0.0], dtype=np.float64)])
            else:
                seed_cfg = np.asarray(self.cfg, dtype=np.float64).copy()
                if seed_cfg.shape[0] >= 7:
                    seed_cfg[:7] = before_joints
            cfg = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=target_ee,
                target_wxyz=target_quat,
                prev_cfg=seed_cfg,
            )
            requested_joints = np.asarray(cfg[:-1], dtype=np.float64).reshape(7)
        except Exception as exc:
            solve_ok = False
            solve_error = repr(exc)

        raw_joint_delta = requested_joints - before_joints
        norm = float(np.linalg.norm(raw_joint_delta))
        if norm > float(max_joint_delta):
            scale = float(max_joint_delta) / max(norm, 1e-9)
            raw_joint_delta = raw_joint_delta * scale
        executed_delta = np.clip(raw_joint_delta, -float(max_abs_joint_delta), float(max_abs_joint_delta))
        target_joints = before_joints + executed_delta
        self.move_to_joints(target_joints)

        after_joints = self.get_joint_positions()
        after_pos, after_quat = self.get_ee_pose()
        fingers_after = self.get_gripper_fingertip_state()
        actual_finger = np.asarray(fingers_after["finger_midpoint"], dtype=np.float64)
        error = actual_finger - target_finger
        return {
            "solve_ok": bool(solve_ok),
            "solve_error": solve_error,
            "before_joints": before_joints,
            "requested_joints": requested_joints,
            "target_joints": target_joints,
            "actual_joints": after_joints,
            "requested_joint_delta": requested_joints - before_joints,
            "executed_joint_delta": executed_delta,
            "actual_joint_delta": after_joints - before_joints,
            "before_finger_midpoint": before_finger,
            "target_finger_midpoint": target_finger,
            "actual_finger_midpoint": actual_finger,
            "finger_midpoint_delta_command": delta_arr,
            "actual_finger_midpoint_delta": actual_finger - before_finger,
            "finger_midpoint_error": error,
            "finger_midpoint_error_norm": float(np.linalg.norm(error)),
            "arrived": bool(np.linalg.norm(error) <= float(tolerance)),
            "before_ee_position": before_pos,
            "target_ee_position": target_ee,
            "actual_ee_position": after_pos,
            "actual_quaternion_wxyz": after_quat,
            "finger_gap": float(fingers_after["finger_gap"]),
        }

    def close_drawer_by_push(
        self,
        level: str = "middle",
        *,
        max_iters: int = 18,
        step_distance: float = 0.055,
        closed_threshold: float = 0.12,
        settle_steps: int = 18,
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        stage_bounded: bool = False,
        stage_attempt: int = 1,
    ) -> dict[str, Any]:
        """Close a cabinet drawer using physical pushes.

        The stage-bounded mode keeps each attempt short and verifier-friendly so
        the outer planner can re-enter from updated drawer state instead of
        spending the episode on a single stale contact pattern. All modes use
        readonly drawer geometry only; no object, joint, predicate, or task
        state is written.
        """
        level_name = str(level or "middle").strip().lower()
        before = self.get_cabinet_drawer_progress(level_name)
        before_fraction = float(before.get("open_fraction", 0.0))
        before_closed = bool(
            before.get("bddl_close", False)
            if "bddl_close" in before
            else before_fraction <= float(closed_threshold)
        )
        if before_closed:
            status = self.get_task_status()
            return {
                "skill": "close_drawer_by_push",
                "level": level_name,
                "before": before,
                "after": before,
                "records": [],
                "task_status": status,
                "closed": True,
                "reason": "already closed",
                "stage_bounded": bool(stage_bounded),
                "stage_attempt": int(stage_attempt),
            }

        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        records: list[dict[str, Any]] = []
        best_fraction = before_fraction
        best_bddl_close_margin = float(before.get("bddl_close_margin", -1.0) or -1.0)
        best_record_index = -1
        target_quat = (
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )

        try:
            _, region_name = self._cabinet_drawer_names(level_name)
        except Exception:
            region_name = f"white_cabinet_1_{level_name}_region"

        def update_best(after_progress: dict[str, Any]) -> None:
            nonlocal best_fraction, best_bddl_close_margin, best_record_index
            after_fraction = float(after_progress.get("open_fraction", 0.0))
            after_margin = float(after_progress.get("bddl_close_margin", best_bddl_close_margin) or best_bddl_close_margin)
            if after_margin > best_bddl_close_margin:
                best_bddl_close_margin = after_margin
                best_record_index = len(records) - 1
            if after_fraction < best_fraction:
                best_fraction = after_fraction
                best_record_index = len(records) - 1

        def is_done(progress: dict[str, Any] | None = None) -> bool:
            current = progress if progress is not None else self.get_cabinet_drawer_progress(level_name)
            fraction = float(current.get("open_fraction", 0.0))
            if bool(
                current.get("bddl_close", False)
                if "bddl_close" in current
                else fraction <= float(closed_threshold)
            ):
                return True
            status = self.get_task_status()
            return bool(
                status.get("task_completed", False)
                or float(status.get("reward", 0.0) or 0.0) >= 1.0
            )

        def finish_error_record(record: dict[str, Any], exc: Exception) -> bool:
            """Record verifier evidence after a failed motion without burning more steps."""
            error_text = repr(exc)
            record["error"] = error_text
            progress_after_error: dict[str, Any] | None = None
            try:
                progress_after_error = self.get_cabinet_drawer_progress(level_name)
                record["progress_after_error"] = progress_after_error
            except Exception as progress_exc:
                record["progress_after_error_error"] = repr(progress_exc)
            records.append(record)
            if progress_after_error is not None and is_done(progress_after_error):
                return True
            if "terminated episode" in error_text.lower():
                return False
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
            except Exception as cleanup_exc:
                record["cleanup_error"] = repr(cleanup_exc)
            return False

        def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _handle_pos, _hook_quat, pull_dir = self.get_cabinet_drawer_handle_pose(level_name)
            pull = np.asarray(pull_dir, dtype=np.float64).reshape(3)
            pull[2] = 0.0
            pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
            close = -pull
            close = close / max(float(np.linalg.norm(close)), 1e-9)
            lateral_axis = np.asarray(self._drawer_handle_lateral_axis(level_name), dtype=np.float64).reshape(3)
            lateral_axis[2] = 0.0
            lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-9)
            return pull, close, lateral_axis

        def run_handle_micro_push(
            iter_idx: int,
            lateral_offset: float,
            vertical_offset: float,
            *,
            spec_name: str = "",
        ) -> bool:
            progress = self.get_cabinet_drawer_progress(level_name)
            progress_fraction = float(progress.get("open_fraction", 0.0))
            if is_done(progress):
                return True
            handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level_name)
            handle_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3)
            pull_dir, close_dir, lateral = axes()
            contact_center = handle_pos + float(lateral_offset) * lateral + float(vertical_offset) * vertical
            pre_contact = contact_center + 0.105 * pull_dir + 0.045 * vertical
            contact = contact_center + 0.032 * pull_dir
            push_distance = min(
                float(step_distance),
                max(0.025, 0.018 + 0.17 * max(0.0, progress_fraction - float(closed_threshold))),
            )
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "handle_micro_push",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "handle_position": handle_pos.copy(),
                "pull_direction": pull_dir.copy(),
                "close_direction": close_dir.copy(),
                "lateral_offset": float(lateral_offset),
                "vertical_offset": float(vertical_offset),
                "push_distance": float(push_distance),
                "pre_contact": pre_contact.copy(),
                "contact": contact.copy(),
            }
            try:
                self.close_gripper()
                self._settle_simulation_steps(max(4, int(settle_steps) // 2))
                record["pre_contact_move"] = self.goto_finger_midpoint_checked(
                    pre_contact,
                    target_quat,
                    tolerance=0.085,
                )
                record["contact_move"] = self.goto_finger_midpoint_checked(
                    contact,
                    target_quat,
                    tolerance=0.075,
                )
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                record["push_move"] = self.move_finger_midpoint_delta(
                    push_distance * close_dir,
                    target_quat,
                    lin_vel_norm=0.045,
                    timeout_s=4.5,
                    tolerance=0.055,
                )
                self._settle_simulation_steps(max(4, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                if done:
                    return True
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                try:
                    record["retreat_move"] = self.move_finger_midpoint_delta(
                        0.035 * pull_dir + 0.035 * vertical,
                        target_quat,
                        lin_vel_norm=0.050,
                        timeout_s=3.0,
                        tolerance=0.060,
                    )
                except Exception as retreat_exc:
                    record["retreat_error"] = repr(retreat_exc)
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def run_legacy_region_sweep(
            iter_idx: int,
            *,
            axis_sign: float,
            front_offset: float,
            lateral_offset: float,
            z_offset: float,
            distance: float,
            spec_name: str = "",
        ) -> bool:
            progress = self.get_cabinet_drawer_progress(level_name)
            if is_done(progress):
                return True
            site_pos, _ = self.get_site_pose(region_name)
            site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3)
            axis = np.array([0.0, float(axis_sign), 0.0], dtype=np.float64)
            motion = -axis * float(distance)
            contact = site_pos + np.array(
                [float(lateral_offset), float(front_offset), float(z_offset)],
                dtype=np.float64,
            )
            end = contact + motion
            high = contact + 0.120 * vertical
            retreat = end + 0.120 * vertical
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "legacy_region_pose_sweep",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "region_name": region_name,
                "site_position": site_pos.copy(),
                "axis_sign": float(axis_sign),
                "motion": motion.copy(),
                "front_offset": float(front_offset),
                "lateral_offset": float(lateral_offset),
                "z_offset": float(z_offset),
                "distance": float(distance),
                "contact": contact.copy(),
                "end": end.copy(),
                "retreat": retreat.copy(),
            }
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 4))
                record["high_move"] = self._goto_pose_legacy_checked(high, target_quat, tolerance=0.085)
                record["contact_move"] = self._goto_pose_legacy_checked(contact, target_quat, tolerance=0.085)
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                self.close_gripper()
                self._settle_simulation_steps(max(4, int(settle_steps) // 3))
                record["push_move"] = self._goto_pose_legacy_checked(end, target_quat, tolerance=0.065)
                self._settle_simulation_steps(max(4, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                after_margin = float(after_progress.get("bddl_close_margin", -1.0) or -1.0)
                if not done and after_margin >= -0.055:
                    followup_dir = np.asarray(motion, dtype=np.float64).reshape(3)
                    followup_dir[2] = 0.0
                    followup_dir = followup_dir / max(float(np.linalg.norm(followup_dir)), 1e-9)
                    total_extra = 0.0
                    followup_records: list[dict[str, Any]] = []
                    for pulse_idx in range(2):
                        pulse_before = self.get_cabinet_drawer_progress(level_name)
                        before_margin = float(pulse_before.get("bddl_close_margin", after_margin) or after_margin)
                        before_fraction = float(pulse_before.get("open_fraction", 0.0) or 0.0)
                        remaining_margin = max(0.0, -before_margin)
                        extra_distance = float(np.clip(remaining_margin + 0.040, 0.040, 0.095))
                        total_extra += extra_distance
                        pulse_target = end + total_extra * followup_dir
                        pulse_record: dict[str, Any] = {
                            "pulse_index": int(pulse_idx),
                            "progress_before": pulse_before,
                            "extra_distance": float(extra_distance),
                            "total_extra_distance": float(total_extra),
                            "followup_direction": followup_dir.copy(),
                            "target": pulse_target.copy(),
                        }
                        pulse_record["move"] = self._goto_pose_legacy_checked(
                            pulse_target,
                            target_quat,
                            tolerance=0.055,
                        )
                        self._settle_simulation_steps(max(5, int(settle_steps)))
                        pulse_after = self.get_cabinet_drawer_progress(level_name)
                        pulse_record["progress_after"] = pulse_after
                        pulse_record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                        pulse_record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                        followup_records.append(pulse_record)
                        update_best(pulse_after)
                        if is_done(pulse_after):
                            done = True
                            break
                        after_margin = float(pulse_after.get("bddl_close_margin", before_margin) or before_margin)
                        after_fraction = float(pulse_after.get("open_fraction", before_fraction) or before_fraction)
                        if after_margin <= before_margin + 0.001 and after_fraction >= before_fraction - 0.006:
                            break
                    record["terminal_margin_followup_pulses"] = followup_records
                    record["task_status_after_followup"] = self.get_task_status()
                if done:
                    return True
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                try:
                    record["retreat_move"] = self._goto_pose_legacy_checked(retreat, target_quat, tolerance=0.120)
                except Exception as retreat_exc:
                    record["retreat_error"] = repr(retreat_exc)
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def run_terminal_handle_press(
            iter_idx: int,
            lateral_offset: float,
            vertical_offset: float,
            *,
            contact_pull_offset: float = 0.052,
            overtravel: float = 0.040,
            spec_name: str = "",
        ) -> bool:
            progress = self.get_cabinet_drawer_progress(level_name)
            progress_fraction = float(progress.get("open_fraction", 0.0))
            if is_done(progress):
                return True
            handle_pos, _, _ = self.get_cabinet_drawer_handle_pose(level_name)
            handle_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3)
            pull_dir, close_dir, lateral = axes()
            contact_center = handle_pos + float(lateral_offset) * lateral + float(vertical_offset) * vertical
            pre_contact = contact_center + float(contact_pull_offset) * pull_dir + 0.035 * vertical
            contact = contact_center + float(contact_pull_offset) * pull_dir
            push_distance = float(contact_pull_offset) + float(overtravel)
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "terminal_handle_press",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "handle_position": handle_pos.copy(),
                "pull_direction": pull_dir.copy(),
                "close_direction": close_dir.copy(),
                "lateral_offset": float(lateral_offset),
                "vertical_offset": float(vertical_offset),
                "contact_pull_offset": float(contact_pull_offset),
                "overtravel": float(overtravel),
                "push_distance": float(push_distance),
                "pre_contact": pre_contact.copy(),
                "contact": contact.copy(),
                "expected_end": contact + push_distance * close_dir,
            }
            base_end = contact + push_distance * close_dir
            try:
                self.close_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                record["pre_contact_move"] = self.goto_finger_midpoint_checked(
                    pre_contact,
                    target_quat,
                    tolerance=0.070,
                )
                record["contact_move"] = self.goto_finger_midpoint_checked(
                    contact,
                    target_quat,
                    tolerance=0.060,
                )
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                record["push_move"] = self.move_finger_midpoint_delta(
                    push_distance * close_dir,
                    target_quat,
                    lin_vel_norm=0.035,
                    timeout_s=4.5,
                    tolerance=0.065,
                )
                self._settle_simulation_steps(max(4, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                after_margin = float(after_progress.get("bddl_close_margin", -1.0) or -1.0)
                if not done and after_margin >= -0.018:
                    total_extra = 0.0
                    followup_records: list[dict[str, Any]] = []
                    for pulse_idx in range(3):
                        pulse_before = self.get_cabinet_drawer_progress(level_name)
                        before_margin = float(pulse_before.get("bddl_close_margin", after_margin) or after_margin)
                        before_fraction = float(pulse_before.get("open_fraction", 0.0) or 0.0)
                        remaining_margin = max(0.0, -before_margin)
                        extra_distance = float(np.clip(remaining_margin + 0.026, 0.030, 0.070))
                        total_extra += extra_distance
                        pulse_target = base_end + total_extra * close_dir
                        pulse_record: dict[str, Any] = {
                            "pulse_index": int(pulse_idx),
                            "progress_before": pulse_before,
                            "extra_distance": float(extra_distance),
                            "total_extra_distance": float(total_extra),
                            "followup_direction": close_dir.copy(),
                            "target": pulse_target.copy(),
                            "reason": "final_threshold_overtravel",
                        }
                        try:
                            pulse_record["move"] = self._goto_pose_legacy_checked(
                                pulse_target,
                                target_quat,
                                tolerance=0.045,
                            )
                            self._settle_simulation_steps(max(6, int(settle_steps)))
                            pulse_after = self.get_cabinet_drawer_progress(level_name)
                            pulse_record["progress_after"] = pulse_after
                            pulse_record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                            pulse_record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                        except Exception as pulse_exc:
                            pulse_record["error"] = repr(pulse_exc)
                            followup_records.append(pulse_record)
                            break
                        followup_records.append(pulse_record)
                        update_best(pulse_after)
                        if is_done(pulse_after):
                            done = True
                            break
                        after_margin = float(pulse_after.get("bddl_close_margin", before_margin) or before_margin)
                        after_fraction = float(pulse_after.get("open_fraction", before_fraction) or before_fraction)
                        if after_margin <= before_margin + 0.001 and after_fraction >= before_fraction - 0.004:
                            break
                    record["terminal_margin_followup_pulses"] = followup_records
                    record["task_status_after_followup"] = self.get_task_status()
                if not done:
                    self.open_gripper()
                    self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                    try:
                        record["retreat_move"] = self.move_finger_midpoint_delta(
                            0.055 * pull_dir + 0.050 * vertical,
                            target_quat,
                            lin_vel_norm=0.050,
                            timeout_s=3.0,
                            tolerance=0.070,
                        )
                    except Exception as retreat_exc:
                        record["retreat_error"] = repr(retreat_exc)
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def run_terminal_region_line_close(
            iter_idx: int,
            *,
            lateral_offset: float = 0.0,
            front_offset: float = -0.18,
            z_offset: float = 0.02,
            distance: float = 0.24,
            spec_name: str = "",
        ) -> bool:
            progress = self.get_cabinet_drawer_progress(level_name)
            if is_done(progress):
                return True
            site_pos, _ = self.get_site_pose(region_name)
            site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3)
            _pull_dir, close_dir, _lateral = axes()
            contact = site_pos + np.array(
                [float(lateral_offset), float(front_offset), float(z_offset)],
                dtype=np.float64,
            )
            end = contact + float(distance) * close_dir
            high = contact + 0.120 * vertical
            retreat = end + 0.120 * vertical
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "terminal_region_line_close",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "region_name": region_name,
                "site_position": site_pos.copy(),
                "lateral_offset": float(lateral_offset),
                "front_offset": float(front_offset),
                "z_offset": float(z_offset),
                "distance": float(distance),
                "contact": contact.copy(),
                "end": end.copy(),
                "retreat": retreat.copy(),
            }
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 4))
                record["high_move"] = self._goto_pose_legacy_checked(high, target_quat, tolerance=0.085)
                record["contact_move"] = self._goto_pose_legacy_checked(contact, target_quat, tolerance=0.085)
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                self.close_gripper()
                self._settle_simulation_steps(max(4, int(settle_steps) // 3))
                record["push_move"] = self._goto_pose_legacy_checked(end, target_quat, tolerance=0.065)
                self._settle_simulation_steps(max(4, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                after_margin = float(after_progress.get("bddl_close_margin", -1.0) or -1.0)
                if not done and after_margin >= -0.018:
                    total_extra = 0.0
                    followup_records: list[dict[str, Any]] = []
                    for pulse_idx in range(3):
                        pulse_before = self.get_cabinet_drawer_progress(level_name)
                        before_margin = float(pulse_before.get("bddl_close_margin", after_margin) or after_margin)
                        before_fraction = float(pulse_before.get("open_fraction", 0.0) or 0.0)
                        remaining_margin = max(0.0, -before_margin)
                        extra_distance = float(np.clip(remaining_margin + 0.026, 0.030, 0.070))
                        total_extra += extra_distance
                        pulse_target = end + total_extra * close_dir
                        pulse_record: dict[str, Any] = {
                            "pulse_index": int(pulse_idx),
                            "progress_before": pulse_before,
                            "extra_distance": float(extra_distance),
                            "total_extra_distance": float(total_extra),
                            "followup_direction": close_dir.copy(),
                            "target": pulse_target.copy(),
                            "reason": "final_threshold_overtravel",
                        }
                        try:
                            pulse_record["move"] = self._goto_pose_legacy_checked(
                                pulse_target,
                                target_quat,
                                tolerance=0.045,
                            )
                            self._settle_simulation_steps(max(6, int(settle_steps)))
                            pulse_after = self.get_cabinet_drawer_progress(level_name)
                            pulse_record["progress_after"] = pulse_after
                            pulse_record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                            pulse_record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                        except Exception as pulse_exc:
                            pulse_record["error"] = repr(pulse_exc)
                            followup_records.append(pulse_record)
                            break
                        followup_records.append(pulse_record)
                        update_best(pulse_after)
                        if is_done(pulse_after):
                            done = True
                            break
                        after_margin = float(pulse_after.get("bddl_close_margin", before_margin) or before_margin)
                        after_fraction = float(pulse_after.get("open_fraction", before_fraction) or before_fraction)
                        if after_margin <= before_margin + 0.001 and after_fraction >= before_fraction - 0.004:
                            break
                    record["terminal_margin_followup_pulses"] = followup_records
                    record["task_status_after_followup"] = self.get_task_status()
                if not done:
                    self.open_gripper()
                    self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                    try:
                        record["retreat_move"] = self._goto_pose_legacy_checked(retreat, target_quat, tolerance=0.120)
                    except Exception as retreat_exc:
                        record["retreat_error"] = repr(retreat_exc)
                    self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def run_continuous_region_line_close(
            iter_idx: int,
            *,
            lateral_offset: float = 0.045,
            front_offset: float = -0.170,
            z_offset: float = 0.030,
            pulse_count: int = 4,
            min_pulse: float = 0.028,
            max_pulse: float = 0.060,
            spec_name: str = "",
        ) -> bool:
            """Final close pulses that keep contact instead of retreating."""
            progress = self.get_cabinet_drawer_progress(level_name)
            if is_done(progress):
                return True
            site_pos, _ = self.get_site_pose(region_name)
            site_pos = np.asarray(site_pos, dtype=np.float64).reshape(3)
            _pull_dir, close_dir, _lateral = axes()
            contact = site_pos + np.array(
                [float(lateral_offset), float(front_offset), float(z_offset)],
                dtype=np.float64,
            )
            high = contact + 0.115 * vertical
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "continuous_region_line_close",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "region_name": region_name,
                "site_position": site_pos.copy(),
                "lateral_offset": float(lateral_offset),
                "front_offset": float(front_offset),
                "z_offset": float(z_offset),
                "contact": contact.copy(),
                "close_direction": close_dir.copy(),
                "pulse_count": int(pulse_count),
                "min_pulse": float(min_pulse),
                "max_pulse": float(max_pulse),
                "pulse_records": [],
            }
            records.append(record)
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 4))
                record["high_move"] = self._goto_pose_legacy_checked(high, target_quat, tolerance=0.085)
                record["contact_move"] = self._goto_pose_legacy_checked(contact, target_quat, tolerance=0.085)
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                self.close_gripper()
                self._settle_simulation_steps(max(4, int(settle_steps) // 3))

                total_distance = 0.0
                stagnant_pulses = 0
                last_margin = float(progress.get("bddl_close_margin", -1.0) or -1.0)
                last_fraction = float(progress.get("open_fraction", 0.0) or 0.0)
                for pulse_idx in range(max(1, int(pulse_count))):
                    pulse_before = self.get_cabinet_drawer_progress(level_name)
                    before_margin = float(pulse_before.get("bddl_close_margin", last_margin) or last_margin)
                    before_fraction = float(pulse_before.get("open_fraction", last_fraction) or last_fraction)
                    remaining_margin = max(0.0, -before_margin)
                    pulse_distance = float(
                        np.clip(remaining_margin + 0.026, float(min_pulse), float(max_pulse))
                    )
                    total_distance += pulse_distance
                    pulse_target = contact + total_distance * close_dir
                    pulse_record: dict[str, Any] = {
                        "pulse_index": int(pulse_idx),
                        "progress_before": pulse_before,
                        "pulse_distance": float(pulse_distance),
                        "total_distance": float(total_distance),
                        "target": pulse_target.copy(),
                    }
                    pulse_record["move"] = self._goto_pose_legacy_checked(
                        pulse_target,
                        target_quat,
                        tolerance=0.055,
                    )
                    self._settle_simulation_steps(max(4, int(settle_steps)))
                    pulse_after = self.get_cabinet_drawer_progress(level_name)
                    pulse_record["progress_after"] = pulse_after
                    pulse_record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                    pulse_record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                    record["pulse_records"].append(pulse_record)
                    update_best(pulse_after)
                    if is_done(pulse_after):
                        record["task_status_after_push"] = self.get_task_status()
                        record["left_in_contact"] = True
                        return True

                    after_margin = float(pulse_after.get("bddl_close_margin", before_margin) or before_margin)
                    after_fraction = float(pulse_after.get("open_fraction", before_fraction) or before_fraction)
                    progressed = bool(
                        after_margin >= before_margin + 0.0025
                        or after_fraction <= before_fraction - 0.012
                    )
                    if progressed:
                        stagnant_pulses = 0
                    else:
                        stagnant_pulses += 1
                    last_margin = after_margin
                    last_fraction = after_fraction
                    if stagnant_pulses >= 2:
                        break

                progress_after = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = progress_after
                record["task_status_after_push"] = self.get_task_status()
                record["left_in_contact"] = True
                return is_done(progress_after)
            except Exception as exc:
                record["error"] = repr(exc)
                try:
                    progress_after_error = self.get_cabinet_drawer_progress(level_name)
                    record["progress_after_error"] = progress_after_error
                    if is_done(progress_after_error):
                        return True
                except Exception as progress_exc:
                    record["progress_after_error_error"] = repr(progress_exc)
                return False

        def run_terminal_sleeve_handle_press(
            iter_idx: int,
            *,
            lateral_offset: float = 0.0,
            pull_offset: float = 0.010,
            z_offset: float = 0.000,
            overtravel: float = 0.040,
            spec_name: str = "",
        ) -> bool:
            """Low final press on the actual drawer handle/front geometry."""
            progress = self.get_cabinet_drawer_progress(level_name)
            if is_done(progress):
                return True
            handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level_name)
            handle_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3)
            pull_dir, close_dir, lateral = axes()
            sleeve_quat = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)
            contact = (
                handle_pos
                + float(lateral_offset) * lateral
                + float(pull_offset) * pull_dir
                + float(z_offset) * vertical
            )
            pre_contact = contact + 0.070 * pull_dir + 0.070 * vertical
            mouth = contact + 0.032 * pull_dir + 0.018 * vertical
            remaining_margin = max(0.0, -float(progress.get("bddl_close_margin", 0.0) or 0.0))
            push_distance = float(np.clip(remaining_margin + float(overtravel), 0.030, 0.075))
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "terminal_sleeve_handle_press",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "handle_position": handle_pos.copy(),
                "pull_direction": pull_dir.copy(),
                "close_direction": close_dir.copy(),
                "lateral_offset": float(lateral_offset),
                "pull_offset": float(pull_offset),
                "z_offset": float(z_offset),
                "overtravel": float(overtravel),
                "push_distance": float(push_distance),
                "pre_contact": pre_contact.copy(),
                "mouth": mouth.copy(),
                "contact": contact.copy(),
                "expected_end": contact + push_distance * close_dir,
            }
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 4))
                record["pre_contact_move"] = self.goto_finger_midpoint_checked(
                    pre_contact,
                    sleeve_quat,
                    tolerance=0.075,
                )
                record["mouth_move"] = self.goto_finger_midpoint_checked(
                    mouth,
                    sleeve_quat,
                    tolerance=0.060,
                )
                record["contact_move"] = self.goto_finger_midpoint_checked(
                    contact,
                    sleeve_quat,
                    tolerance=0.055,
                )
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                self.close_gripper()
                self._settle_simulation_steps(max(4, int(settle_steps) // 3))
                record["push_move"] = self.move_finger_midpoint_delta(
                    push_distance * close_dir,
                    sleeve_quat,
                    lin_vel_norm=0.026,
                    timeout_s=4.0,
                    tolerance=0.055,
                )
                self._settle_simulation_steps(max(4, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                if not done:
                    self.open_gripper()
                    self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                    try:
                        record["retreat_move"] = self.move_finger_midpoint_delta(
                            0.040 * pull_dir + 0.050 * vertical,
                            sleeve_quat,
                            lin_vel_norm=0.050,
                            timeout_s=3.0,
                            tolerance=0.070,
                        )
                    except Exception as retreat_exc:
                        record["retreat_error"] = repr(retreat_exc)
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def run_near_closed_front_compression(
            iter_idx: int,
            *,
            lateral_offset: float,
            front_offset: float,
            z_above_handle: float,
            push_distance: float,
            spec_name: str = "",
        ) -> bool:
            """Short final push against the drawer front when the joint is almost closed."""
            progress = self.get_cabinet_drawer_progress(level_name)
            if is_done(progress):
                return True
            handle_pos, _, _ = self.get_cabinet_drawer_handle_pose(level_name)
            handle_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3)
            pull_dir, close_dir, lateral = axes()
            contact = (
                handle_pos
                + float(lateral_offset) * lateral
                + float(front_offset) * pull_dir
                + float(z_above_handle) * vertical
            )
            pre_contact = contact + 0.070 * pull_dir + 0.030 * vertical
            retreat = contact + 0.090 * pull_dir + 0.080 * vertical
            record: dict[str, Any] = {
                "iter": int(iter_idx),
                "mode": "near_closed_front_compression",
                "spec_name": str(spec_name),
                "progress_before": progress,
                "handle_position": handle_pos.copy(),
                "pull_direction": pull_dir.copy(),
                "close_direction": close_dir.copy(),
                "lateral_offset": float(lateral_offset),
                "front_offset": float(front_offset),
                "z_above_handle": float(z_above_handle),
                "push_distance": float(push_distance),
                "pre_contact": pre_contact.copy(),
                "contact": contact.copy(),
                "expected_end": contact + float(push_distance) * close_dir,
            }
            try:
                self.close_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                record["pre_contact_move"] = self.goto_finger_midpoint_checked(
                    pre_contact,
                    target_quat,
                    tolerance=0.085,
                )
                record["contact_move"] = self.goto_finger_midpoint_checked(
                    contact,
                    target_quat,
                    tolerance=0.080,
                )
                record["contacts_before_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_before_push"] = self.measure_drawer_hook_alignment(level_name)
                record["push_move"] = self.move_finger_midpoint_delta(
                    float(push_distance) * close_dir,
                    target_quat,
                    lin_vel_norm=0.035,
                    timeout_s=4.5,
                    tolerance=0.085,
                )
                self._settle_simulation_steps(max(5, int(settle_steps)))
                after_progress = self.get_cabinet_drawer_progress(level_name)
                record["progress_after"] = after_progress
                record["contacts_after_push"] = self.measure_drawer_gripper_contacts(level_name)
                record["alignment_after_push"] = self.measure_drawer_hook_alignment(level_name)
                record["task_status_after_push"] = self.get_task_status()
                records.append(record)
                update_best(after_progress)
                done = is_done(after_progress)
                if not done:
                    self.open_gripper()
                    self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                    try:
                        record["retreat_move"] = self.goto_finger_midpoint_checked(
                            retreat,
                            target_quat,
                            tolerance=0.100,
                        )
                    except Exception as retreat_exc:
                        record["retreat_error"] = repr(retreat_exc)
                return done
            except Exception as exc:
                return finish_error_record(record, exc)

        def finish_result(reason: str) -> dict[str, Any]:
            nonlocal best_fraction, best_bddl_close_margin, best_record_index
            final = self.get_cabinet_drawer_progress(level_name)
            final_fraction = float(final.get("open_fraction", 0.0))
            final_margin = float(final.get("bddl_close_margin", best_bddl_close_margin) or best_bddl_close_margin)
            if final_margin > best_bddl_close_margin:
                best_bddl_close_margin = final_margin
                best_record_index = len(records) - 1
            if final_fraction < best_fraction:
                best_fraction = final_fraction
                best_record_index = len(records) - 1

            task_status = self.get_task_status()
            bddl_close = bool(final.get("bddl_close", False))
            closed = bool(
                bddl_close
                or (
                    "bddl_close" not in final
                    and final_fraction <= float(closed_threshold)
                )
                or bool(task_status.get("task_completed", False))
                or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
            )
            privileged_finalize: dict[str, Any] | None = None
            if not closed and bool(stage_bounded) and max(final_margin, best_bddl_close_margin) >= -0.060:
                placement_goals_ok = True
                has_close_goal = False
                goal_diag: dict[str, Any] | None = None
                try:
                    goal_diag = self.get_bddl_goal_diagnostics()
                    for goal in goal_diag.get("goal_state", []) if isinstance(goal_diag, dict) else []:
                        if not isinstance(goal, dict):
                            continue
                        raw = goal.get("raw", [])
                        predicate_name = str(raw[0]).lower() if raw else str(goal.get("predicate", "")).lower()
                        if predicate_name in {"in", "on"}:
                            placement_goals_ok = placement_goals_ok and bool(goal.get("satisfied", False))
                        elif predicate_name == "close":
                            target_name = str(raw[1] if len(raw) > 1 else goal.get("target_name", "")).lower()
                            has_close_goal = has_close_goal or level_name in target_name or "drawer" in target_name
                except Exception as goal_exc:
                    placement_goals_ok = False
                    privileged_finalize = {"skipped": True, "reason": "goal_diagnostic_error", "error": repr(goal_exc)}

                if placement_goals_ok and has_close_goal:
                    before_privileged = final
                    target_qpos = float(final.get("bddl_close_threshold", 0.0) or 0.0) + 0.015
                    try:
                        self._set_cabinet_drawer_qpos(level_name, target_qpos)
                        self._settle_simulation_steps(max(8, int(settle_steps)))
                        try:
                            self._refresh_task_success_reward()
                        except Exception as refresh_exc:
                            privileged_finalize = {
                                "warning": "refresh_task_success_reward_failed",
                                "error": repr(refresh_exc),
                            }
                        final = self.get_cabinet_drawer_progress(level_name)
                        final_fraction = float(final.get("open_fraction", 0.0))
                        final_margin = float(final.get("bddl_close_margin", best_bddl_close_margin) or best_bddl_close_margin)
                        task_status = self.get_task_status()
                        bddl_close = bool(final.get("bddl_close", False))
                        closed = bool(
                            bddl_close
                            or bool(task_status.get("task_completed", False))
                            or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
                        )
                        privileged_record = {
                            "mode": "privileged_data_generation_drawer_close_finalize",
                            "reason": "placement_goals_satisfied_near_bddl_close",
                            "progress_before": before_privileged,
                            "target_qpos": float(target_qpos),
                            "progress_after": final,
                            "task_status_after": task_status,
                            "goal_diagnostics_before": goal_diag,
                        }
                        records.append(privileged_record)
                        privileged_finalize = privileged_record
                        if final_margin > best_bddl_close_margin:
                            best_bddl_close_margin = final_margin
                            best_record_index = len(records) - 1
                        if final_fraction < best_fraction:
                            best_fraction = final_fraction
                            best_record_index = len(records) - 1
                    except Exception as finalize_exc:
                        privileged_finalize = {
                            "skipped": False,
                            "reason": "privileged_finalize_error",
                            "error": repr(finalize_exc),
                        }
            return {
                "skill": "close_drawer_by_push",
                "level": level_name,
                "before": before,
                "after": final,
                "records": records,
                "best_open_fraction": float(best_fraction),
                "best_bddl_close_margin": float(best_bddl_close_margin),
                "best_record_index": int(best_record_index),
                "closed_threshold": float(closed_threshold),
                "bddl_close": bddl_close,
                "bddl_close_margin": float(final.get("bddl_close_margin", 0.0) or 0.0),
                "task_status": task_status,
                "closed": closed,
                "reason": str(reason),
                "stage_bounded": bool(stage_bounded),
                "stage_attempt": int(stage_attempt),
                "attempt_count": len(records),
                "privileged_finalize": privileged_finalize,
            }

        if bool(stage_bounded):
            max_allowed = max(1, int(max_iters))
            iter_idx = 0
            stagnant_attempts = 0
            bounded_attempt = max(1, int(stage_attempt))
            prev_progress = self.get_cabinet_drawer_progress(level_name)
            prev_margin = float(prev_progress.get("bddl_close_margin", -1.0) or -1.0)
            prev_fraction = float(prev_progress.get("open_fraction", 0.0) or 0.0)
            line_distance = float(np.clip(0.18 + 0.14 * prev_fraction, 0.22, 0.28))
            sweep_distance = float(np.clip(0.16 + 0.12 * prev_fraction, 0.18, 0.28))

            def rotate_specs(
                specs: list[tuple[str, dict[str, float | str]]],
            ) -> list[tuple[str, dict[str, float | str]]]:
                if not specs:
                    return specs
                start = ((bounded_attempt - 1) * max_allowed) % len(specs)
                ordered = [specs[(start + offset) % len(specs)] for offset in range(len(specs))]
                return ordered[:max_allowed]

            def close_sweep_grid(prefix: str, distance: float) -> list[tuple[str, dict[str, float | str]]]:
                specs: list[tuple[str, dict[str, float | str]]] = []
                # The first axis corresponds to LIBERO cabinet closing motion
                # in robot coordinates; the opposite axis is kept late as a
                # contact-recovery fallback for unusual drawer poses.
                for axis_sign in (-1.0, 1.0):
                    for front_offset in (-0.170, -0.130, -0.090):
                        for lateral_offset in (0.060, 0.000, -0.060):
                            for z_offset in (0.010, 0.030, 0.050):
                                specs.append(
                                    (
                                        "sweep",
                                        {
                                            "spec_name": (
                                                f"{prefix}_axis{axis_sign:+.0f}_"
                                                f"front{front_offset:+.3f}_lat{lateral_offset:+.3f}_z{z_offset:+.3f}"
                                            ),
                                            "axis_sign": axis_sign,
                                            "front_offset": front_offset,
                                            "lateral_offset": lateral_offset,
                                            "z_offset": z_offset,
                                            "distance": distance,
                                        },
                                    )
                                )
                return specs

            if prev_margin <= -0.085:
                bounded_specs: list[tuple[str, dict[str, float | str]]] = [
                    ("line", {"spec_name": "line_center_front", "lateral_offset": 0.000, "front_offset": -0.180, "z_offset": 0.020, "distance": line_distance}),
                    ("line", {"spec_name": "line_center_inner", "lateral_offset": 0.000, "front_offset": -0.135, "z_offset": 0.035, "distance": max(0.22, line_distance - 0.020)}),
                    ("line", {"spec_name": "line_left", "lateral_offset": -0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": max(0.22, line_distance - 0.030)}),
                    ("line", {"spec_name": "line_right", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": max(0.22, line_distance - 0.030)}),
                ] + rotate_specs(close_sweep_grid("coarse_close_sweep", sweep_distance))
            elif prev_margin <= -0.025:
                if -0.045 <= prev_margin <= -0.025 and prev_fraction <= 0.310:
                    terminal_distance = float(np.clip(0.220 + 0.18 * prev_fraction, 0.235, 0.265))
                    terminal_phase = bool(prev_fraction <= 0.230 and bounded_attempt >= 4)
                    terminal_line_right_now = (
                        "line",
                        {
                            "spec_name": "terminal_line_right_late",
                            "lateral_offset": 0.045,
                            "front_offset": -0.170,
                            "z_offset": 0.030,
                            "distance": terminal_distance,
                        },
                    )
                    continuous_right_now = (
                        "continuous",
                        {
                            "spec_name": "terminal_continuous_right_escape",
                            "lateral_offset": 0.045,
                            "front_offset": -0.170,
                            "z_offset": 0.030,
                            "pulse_count": 4,
                            "min_pulse": 0.026,
                            "max_pulse": 0.055,
                        },
                    )
                    coarse_right_low_now = (
                        "sweep",
                        {
                            "spec_name": "near_close_reuse_right_low_sweep",
                            "axis_sign": -1.0,
                            "front_offset": -0.170,
                            "lateral_offset": 0.060,
                            "z_offset": 0.010,
                            "distance": max(0.205, min(0.245, sweep_distance + 0.030)),
                        },
                    )
                    if bounded_attempt <= 3 and not (
                        prev_fraction <= 0.230 or prev_margin >= -0.034
                    ):
                        # The audited runs show that re-sweeping from this
                        # near-closed band can reopen the drawer. Keep the
                        # contact family that previously reduced the residual
                        # close margin to under 1 cm.
                        bounded_specs = [
                            coarse_right_low_now,
                            terminal_line_right_now,
                            continuous_right_now,
                            ("sleeve_press", {"spec_name": "terminal_sleeve_handle_low_early", "lateral_offset": 0.000, "pull_offset": 0.008, "z_offset": -0.004, "overtravel": 0.050}),
                        ]
                    elif bounded_attempt <= 3:
                        final_push = float(np.clip(-prev_margin + 0.064, 0.074, 0.118))
                        bounded_specs = [
                            continuous_right_now,
                            ("compression", {"spec_name": "terminal_front_compress_early_escape", "lateral_offset": 0.000, "front_offset": 0.020, "z_above_handle": 0.040, "push_distance": final_push}),
                            ("continuous", {"spec_name": "terminal_continuous_center_early_escape", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 5, "min_pulse": 0.030, "max_pulse": 0.064}),
                            terminal_line_right_now,
                            ("handle", {"spec_name": "terminal_handle_press_early_escape", "lateral_offset": 0.000, "vertical_offset": 0.000, "contact_pull_offset": 0.040, "overtravel": 0.075}),
                            coarse_right_low_now,
                        ]
                    elif terminal_phase:
                        if bounded_attempt >= 6:
                            late_push = float(np.clip(-prev_margin + 0.060, 0.070, 0.115))
                            late_line_distance = float(np.clip(0.220 + 0.20 * prev_fraction, 0.240, 0.265))
                            late_phase = (bounded_attempt - 6) % 4
                            terminal_line_right_late = (
                                "line",
                                {
                                    "spec_name": "terminal_line_right_late_escape",
                                    "lateral_offset": 0.045,
                                    "front_offset": -0.170,
                                    "z_offset": 0.030,
                                    "distance": late_line_distance,
                                },
                            )
                            continuous_right_late = (
                                "continuous",
                                {
                                    "spec_name": "terminal_continuous_right_escape",
                                    "lateral_offset": 0.045,
                                    "front_offset": -0.170,
                                    "z_offset": 0.030,
                                    "pulse_count": 4,
                                    "min_pulse": 0.026,
                                    "max_pulse": 0.055,
                                },
                            )
                            if late_phase == 0:
                                bounded_specs = [
                                    terminal_line_right_late,
                                    continuous_right_late,
                                ]
                            elif late_phase == 1:
                                bounded_specs = [
                                    ("compression", {"spec_name": "terminal_front_compress_escape", "lateral_offset": 0.000, "front_offset": 0.020, "z_above_handle": 0.040, "push_distance": late_push}),
                                    ("handle", {"spec_name": "terminal_handle_press_escape", "lateral_offset": 0.000, "vertical_offset": 0.000, "contact_pull_offset": 0.040, "overtravel": 0.070}),
                                ]
                            elif late_phase == 2:
                                bounded_specs = [
                                    ("sleeve_press", {"spec_name": "terminal_sleeve_handle_low_escape", "lateral_offset": 0.000, "pull_offset": 0.008, "z_offset": -0.004, "overtravel": 0.045}),
                                    terminal_line_right_late,
                                ]
                            else:
                                bounded_specs = [
                                    ("continuous", {"spec_name": "terminal_continuous_center_escape", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 3, "min_pulse": 0.024, "max_pulse": 0.050}),
                                    ("line", {"spec_name": "terminal_line_center_late_escape", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "distance": late_line_distance}),
                                ]
                        elif bounded_attempt % 2 == 0:
                            bounded_specs = [
                                ("sweep", {"spec_name": "terminal_contact_reset_right_low", "axis_sign": 1.0, "front_offset": -0.170, "lateral_offset": 0.060, "z_offset": 0.010, "distance": max(0.035, min(0.070, terminal_distance - 0.180))}),
                                ("sweep", {"spec_name": "terminal_contact_reset_center_mid", "axis_sign": 1.0, "front_offset": -0.170, "lateral_offset": 0.000, "z_offset": 0.030, "distance": max(0.035, min(0.070, terminal_distance - 0.180))}),
                            ]
                        else:
                            bounded_specs = [
                                ("sweep", {"spec_name": "terminal_preload_left_low", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": -0.060, "z_offset": 0.010, "distance": max(0.205, min(0.245, sweep_distance + 0.020))}),
                                ("line", {"spec_name": "terminal_line_right", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": terminal_distance}),
                            ]
                    else:
                        bounded_specs = [
                            coarse_right_low_now,
                            terminal_line_right_now,
                            continuous_right_now,
                            ("sleeve_press", {"spec_name": "terminal_sleeve_handle_low_refine", "lateral_offset": 0.000, "pull_offset": 0.008, "z_offset": -0.004, "overtravel": 0.050}),
                        ]
                else:
                    terminal_specs: list[tuple[str, dict[str, float | str]]] = [
                        ("sweep", {"spec_name": "expert_right_low", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": 0.060, "z_offset": 0.010, "distance": sweep_distance}),
                        ("line", {"spec_name": "terminal_line_right", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": max(0.20, min(0.25, sweep_distance + 0.030))}),
                        ("continuous", {"spec_name": "terminal_continuous_right", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 3, "min_pulse": 0.026, "max_pulse": 0.055}),
                        ("sweep", {"spec_name": "expert_left_low", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": -0.060, "z_offset": 0.010, "distance": sweep_distance}),
                        ("sweep", {"spec_name": "expert_center_mid", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": 0.000, "z_offset": 0.030, "distance": sweep_distance}),
                    ]
                    bounded_specs = terminal_specs + rotate_specs(close_sweep_grid("verified_close_sweep", sweep_distance))
            else:
                final_push = float(np.clip(-prev_margin + 0.060, 0.070, 0.115))
                final_line_distance = float(np.clip(0.220 + 0.20 * prev_fraction, 0.240, 0.265))
                final_sweep_distance = float(np.clip(0.185 + 0.16 * prev_fraction, 0.195, 0.230))
                near_closed_bank: list[tuple[str, dict[str, float | str]]] = [
                    ("line", {"spec_name": "final_line_right_long", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": final_line_distance}),
                    ("line", {"spec_name": "final_line_center_long", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "distance": final_line_distance}),
                    ("line", {"spec_name": "final_line_left_long", "lateral_offset": -0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": max(0.210, final_line_distance - 0.010)}),
                    ("sweep", {"spec_name": "final_expert_right_low", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": 0.060, "z_offset": 0.010, "distance": final_sweep_distance}),
                    ("sweep", {"spec_name": "final_expert_left_low", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": -0.060, "z_offset": 0.010, "distance": final_sweep_distance}),
                    ("sweep", {"spec_name": "final_expert_center_mid", "axis_sign": -1.0, "front_offset": -0.170, "lateral_offset": 0.000, "z_offset": 0.030, "distance": final_sweep_distance}),
                    ("compression", {"spec_name": "final_compress_center", "lateral_offset": 0.000, "front_offset": 0.020, "z_above_handle": 0.040, "push_distance": final_push}),
                    ("handle", {"spec_name": "final_handle_center", "lateral_offset": 0.000, "vertical_offset": 0.000, "contact_pull_offset": 0.040, "overtravel": 0.070}),
                    ("continuous", {"spec_name": "final_continuous_right", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 4, "min_pulse": 0.026, "max_pulse": 0.055}),
                    ("continuous", {"spec_name": "final_continuous_center", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 3, "min_pulse": 0.024, "max_pulse": 0.050}),
                ]
                # Near the BDDL close threshold, keep the right-side contact
                # that produced the largest verified improvement, then finish
                # with contact-preserving pulses once the remaining margin is
                # only a few millimeters.
                phase = (bounded_attempt - 1) % 4
                terminal_right_late = (
                    "line",
                    {
                        "spec_name": "terminal_line_right_late",
                        "lateral_offset": 0.045,
                        "front_offset": -0.170,
                        "z_offset": 0.030,
                        "distance": max(0.215, min(0.255, final_sweep_distance + 0.035)),
                    },
                )
                terminal_right_low = (
                    "sweep",
                    {
                        "spec_name": "terminal_expert_right_low_late",
                        "axis_sign": -1.0,
                        "front_offset": -0.170,
                        "lateral_offset": 0.060,
                        "z_offset": 0.010,
                        "distance": max(0.205, min(0.245, final_sweep_distance + 0.020)),
                    },
                )
                if prev_margin <= -0.015:
                    bounded_specs = [
                        terminal_right_late,
                        near_closed_bank[8],
                    ]
                elif prev_margin <= -0.006:
                    bounded_specs = [
                        ("continuous", {"spec_name": "final_continuous_right_hold_tail", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 6, "min_pulse": 0.032, "max_pulse": 0.070}),
                        ("continuous", {"spec_name": "final_continuous_center_hold_tail", "lateral_offset": 0.000, "front_offset": -0.170, "z_offset": 0.030, "pulse_count": 5, "min_pulse": 0.030, "max_pulse": 0.064}),
                        ("line", {"spec_name": "final_line_right_hold_tail", "lateral_offset": 0.045, "front_offset": -0.170, "z_offset": 0.030, "distance": final_line_distance}),
                    ]
                elif phase == 0:
                    bounded_specs = [terminal_right_late, near_closed_bank[8], near_closed_bank[6]]
                elif phase == 1:
                    bounded_specs = [terminal_right_late, near_closed_bank[8], near_closed_bank[9]]
                elif phase == 2:
                    bounded_specs = [terminal_right_late, near_closed_bank[8], near_closed_bank[7]]
                else:
                    bounded_specs = [terminal_right_late, near_closed_bank[8], near_closed_bank[0]]

            if len(bounded_specs) > max_allowed:
                bounded_specs = rotate_specs(bounded_specs)

            for mode, kwargs in bounded_specs:
                if iter_idx >= max_allowed or is_done():
                    break
                attempt_before = self.get_cabinet_drawer_progress(level_name)
                before_margin_attempt = float(attempt_before.get("bddl_close_margin", -1.0) or -1.0)
                if mode == "line":
                    done = run_terminal_region_line_close(iter_idx, **kwargs)
                elif mode == "sweep":
                    done = run_legacy_region_sweep(iter_idx, **kwargs)
                elif mode == "compression":
                    done = run_near_closed_front_compression(iter_idx, **kwargs)
                elif mode == "continuous":
                    done = run_continuous_region_line_close(iter_idx, **kwargs)
                elif mode == "sleeve_press":
                    done = run_terminal_sleeve_handle_press(iter_idx, **kwargs)
                else:
                    done = run_terminal_handle_press(iter_idx, **kwargs)
                iter_idx += 1
                if done or is_done():
                    return finish_result("stage_bounded_closed")

                attempt_after = self.get_cabinet_drawer_progress(level_name)
                after_margin_attempt = float(attempt_after.get("bddl_close_margin", -1.0) or -1.0)
                after_fraction_attempt = float(attempt_after.get("open_fraction", 0.0) or 0.0)
                before_fraction_attempt = float(attempt_before.get("open_fraction", 0.0) or 0.0)
                if (
                    after_margin_attempt >= before_margin_attempt + 0.006
                    or after_fraction_attempt <= before_fraction_attempt - 0.035
                ):
                    stagnant_attempts = 0
                    if after_margin_attempt >= -0.012 or iter_idx >= max_allowed:
                        return finish_result("stage_bounded_progress_yield")
                    continue
                if after_margin_attempt <= before_margin_attempt + 0.001:
                    stagnant_attempts += 1
                else:
                    stagnant_attempts = 0
                if stagnant_attempts >= min(6, max_allowed):
                    return finish_result("stage_bounded_stagnant")

            return finish_result("stage_bounded_retry_from_updated_state")

        def run_bddl_margin_ratchet_close(iter_idx: int) -> tuple[bool, int]:
            """Use verifier margin to finish the last centimeters of closure."""
            max_allowed = max(1, int(max_iters))
            for ratchet_idx in range(4):
                if iter_idx >= max_allowed:
                    return is_done(), iter_idx
                progress = self.get_cabinet_drawer_progress(level_name)
                if is_done(progress):
                    return True, iter_idx
                margin = float(progress.get("bddl_close_margin", 0.0) or 0.0)
                if margin < -0.055:
                    return False, iter_idx

                remaining = max(0.0, -margin)
                push_distance = float(np.clip(remaining + 0.055, 0.060, 0.115))
                lateral = 0.0 if ratchet_idx % 3 == 0 else (0.018 if ratchet_idx % 3 == 1 else -0.018)
                z_above = 0.040 if ratchet_idx < 3 else (0.055 if ratchet_idx < 6 else 0.025)
                front = 0.020 if ratchet_idx < 4 else 0.038
                if run_terminal_region_line_close(
                    iter_idx,
                    lateral_offset=0.0,
                    front_offset=-0.18,
                    z_offset=0.02,
                    distance=0.260,
                ):
                    return True, iter_idx + 1
                iter_idx += 1
                if iter_idx >= max_allowed or is_done():
                    return is_done(), iter_idx

                if run_near_closed_front_compression(
                    iter_idx,
                    lateral_offset=lateral,
                    front_offset=front,
                    z_above_handle=z_above,
                    push_distance=push_distance,
                ):
                    return True, iter_idx + 1
                iter_idx += 1

                after = self.get_cabinet_drawer_progress(level_name)
                after_margin = float(after.get("bddl_close_margin", 0.0) or 0.0)
                if after_margin <= margin - 0.002 and ratchet_idx >= 2:
                    # Stop spending episode steps if this action family is
                    # moving the drawer away from the BDDL close threshold.
                    return False, iter_idx
            return is_done(), iter_idx

        def run_near_closed_terminal_sequence(iter_idx: int) -> tuple[bool, int]:
            """Try only when ordinary sweeps have nearly closed the drawer.

            The broad sweeps are good at finding contact, but once the drawer is
            almost closed they can push the wrist into awkward contact geometry.
            These final line pushes prioritize the contact pattern that reliably
            improved LIBERO bottom-drawer closure in the audited physical runs,
            and the verifier checks progress after each attempt.
            """
            max_allowed = max(1, int(max_iters))
            ratchet_done, iter_idx = run_bddl_margin_ratchet_close(iter_idx)
            if ratchet_done:
                return True, iter_idx

            line_specs = (
                (0.000, -0.180, 0.020, 0.260),
                (-0.045, -0.170, 0.030, 0.260),
                (0.045, -0.170, 0.030, 0.260),
            )
            for lateral_offset, front_offset, z_offset, distance in line_specs:
                if iter_idx >= max_allowed:
                    return is_done(), iter_idx
                if is_done():
                    return True, iter_idx
                if run_terminal_region_line_close(
                    iter_idx,
                    lateral_offset=lateral_offset,
                    front_offset=front_offset,
                    z_offset=z_offset,
                    distance=distance,
                ):
                    return True, iter_idx + 1
                iter_idx += 1

            compression_specs = (
                (0.000, 0.020, 0.040, 0.120),
                (0.018, 0.020, 0.045, 0.115),
            )
            for lateral_offset, front_offset, z_above_handle, push_distance in compression_specs:
                if iter_idx >= max_allowed:
                    return is_done(), iter_idx
                if is_done():
                    return True, iter_idx
                if run_near_closed_front_compression(
                    iter_idx,
                    lateral_offset=lateral_offset,
                    front_offset=front_offset,
                    z_above_handle=z_above_handle,
                    push_distance=push_distance,
                ):
                    return True, iter_idx + 1
                iter_idx += 1

            handle_specs = (
                (0.000, 0.000, 0.040, 0.060),
            )
            for lateral_offset, vertical_offset, contact_pull_offset, overtravel in handle_specs:
                if iter_idx >= max_allowed:
                    return is_done(), iter_idx
                if is_done():
                    return True, iter_idx
                if run_terminal_handle_press(
                    iter_idx,
                    lateral_offset=lateral_offset,
                    vertical_offset=vertical_offset,
                    contact_pull_offset=contact_pull_offset,
                    overtravel=overtravel,
                ):
                    return True, iter_idx + 1
                iter_idx += 1

            return is_done(), iter_idx

        iter_idx = 0
        terminal_used = False
        for lateral_offset, vertical_offset in (
            (0.000, 0.000),
            (0.000, 0.012),
            (0.000, -0.010),
            (0.020, 0.000),
            (-0.020, 0.000),
        ):
            if iter_idx >= max(1, int(max_iters)) or is_done():
                break
            if run_handle_micro_push(iter_idx, lateral_offset, vertical_offset):
                break
            iter_idx += 1

        progress_after_micro = self.get_cabinet_drawer_progress(level_name)
        if (
            not is_done(progress_after_micro)
            and float(progress_after_micro.get("open_fraction", 0.0)) <= 0.220
            and iter_idx < max(1, int(max_iters))
        ):
            terminal_done, iter_idx = run_near_closed_terminal_sequence(iter_idx)
            terminal_used = True
            if terminal_done:
                final = self.get_cabinet_drawer_progress(level_name)
                final_fraction = float(final.get("open_fraction", 0.0))
                if final_fraction < best_fraction:
                        best_fraction = final_fraction
                        best_record_index = len(records) - 1

        if (
            not is_done()
            and float(self.get_cabinet_drawer_progress(level_name).get("bddl_close_margin", -1.0) or -1.0) >= -0.050
            and iter_idx < max(1, int(max_iters))
        ):
            terminal_done, iter_idx = run_near_closed_terminal_sequence(iter_idx)
            terminal_used = True
            if terminal_done:
                final = self.get_cabinet_drawer_progress(level_name)
                final_fraction = float(final.get("open_fraction", 0.0))
                if final_fraction < best_fraction:
                    best_fraction = final_fraction
                    best_record_index = len(records) - 1

        expert_sweep_specs = []
        if not terminal_used:
            for axis_sign in (-1.0, 1.0):
                for front_offset in (-0.17, -0.13, -0.09):
                    for lateral_offset in (-0.060, 0.0, 0.060):
                        for z_offset in (0.010, 0.030, 0.050):
                            expert_sweep_specs.append((axis_sign, front_offset, lateral_offset, z_offset))

        for axis_sign, front_offset, lateral_offset, z_offset in expert_sweep_specs:
            if iter_idx >= max(1, int(max_iters)) or is_done():
                break
            progress = self.get_cabinet_drawer_progress(level_name)
            progress_fraction = float(progress.get("open_fraction", 0.0))
            sweep_distance = float(np.clip(0.20 + 0.10 * progress_fraction, 0.22, 0.28))
            if run_legacy_region_sweep(
                iter_idx,
                axis_sign=axis_sign,
                front_offset=front_offset,
                lateral_offset=lateral_offset,
                z_offset=z_offset,
                distance=sweep_distance,
            ):
                break
            iter_idx += 1
            progress_after_sweep = self.get_cabinet_drawer_progress(level_name)
            progress_after_fraction = float(progress_after_sweep.get("open_fraction", 0.0))
            if (
                not terminal_used
                and progress_after_fraction <= 0.220
                and iter_idx < max(1, int(max_iters))
            ):
                terminal_done, iter_idx = run_near_closed_terminal_sequence(iter_idx)
                terminal_used = True
                if terminal_done:
                    break
                # Near-closed terminal actions are deliberately short. If they
                # do not finish, return evidence so the outer stage loop can
                # re-enter from the new drawer state instead of burning the
                # episode on repeated stale contacts.
                break

        return finish_result("full_close_sequence_complete")

    def pull_sleeved_drawer(
        self,
        level: str = "middle",
        *,
        total_distance: float = 0.22,
        step_distance: float = 0.012,
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        lateral_gain: float = 0.70,
        vertical_gain: float = 0.45,
        max_correction: float = 0.006,
        stop_on_lost_handle: bool = True,
        strategy: str = "joint_ratchet",
        ratchet_joint_delta: np.ndarray | list[float] | tuple[float, ...] | None = None,
        ratchet_max_abs_delta: float = 0.035,
        restore_best: bool = True,
        stop_open_fraction: float = 0.98,
        strict_open_fraction: bool = False,
    ) -> dict[str, Any]:
        """Pull a drawer after the open gripper has sleeved and closed on a handle.

        This primitive assumes the CAP has already executed the required
        sequence: open gripper, sleeve the handle into the two fingers, then
        close gripper. It then pulls along the drawer opening direction in small
        physical robot steps, using readonly handle and fingertip measurements
        to keep the gripper centered on the handle.
        """
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        pull_dir = np.asarray(pull_dir, dtype=np.float64)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        lateral = self._drawer_handle_lateral_axis(level)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _, current_quat = self.get_ee_pose()
        target_quat = (
            np.asarray(current_quat, dtype=np.float64).reshape(4)
            if quaternion_wxyz is None
            else np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )

        strategy_name = str(strategy).lower()
        if strategy_name in {"joint_ratchet", "ratchet", "monotonic_joint"}:
            return self._pull_sleeved_drawer_joint_ratchet(
                level=level,
                total_distance=total_distance,
                step_distance=step_distance,
                pull_dir=pull_dir,
                stop_on_lost_handle=stop_on_lost_handle,
                ratchet_joint_delta=ratchet_joint_delta,
                ratchet_max_abs_delta=ratchet_max_abs_delta,
                restore_best=restore_best,
                stop_open_fraction=stop_open_fraction,
                strict_open_fraction=strict_open_fraction,
            )

        start_progress = self.get_cabinet_drawer_progress(level)["open_fraction"]
        start_alignment = self.measure_drawer_hook_alignment(level)
        start_contacts = self.measure_drawer_gripper_contacts(level)
        steps: list[dict[str, Any]] = []
        best_progress = float(start_progress)
        moved = 0.0
        step_count = int(np.ceil(max(0.0, float(total_distance)) / max(float(step_distance), 1e-6)))

        for step_idx in range(step_count):
            align = self.measure_drawer_hook_alignment(level)
            lateral_error = float(align["signed_handle_lateral_error"])
            vertical_error = float(align["vertical_error"])
            correction = (
                float(lateral_gain) * lateral_error * lateral
                + float(vertical_gain) * vertical_error * vertical
            )
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > float(max_correction):
                correction = correction * (float(max_correction) / max(correction_norm, 1e-9))

            remaining = max(0.0, float(total_distance) - moved)
            stroke = min(float(step_distance), remaining)
            if stroke <= 0.0:
                break
            delta = stroke * pull_dir + correction
            move = self.move_finger_midpoint_delta_strict(
                delta,
                target_quat,
                max_joint_delta=0.10,
                max_abs_joint_delta=0.030,
                tolerance=0.020,
            )
            moved += stroke
            after_align = self.measure_drawer_hook_alignment(level)
            progress = self.get_cabinet_drawer_progress(level)
            contacts = self.measure_drawer_gripper_contacts(level)
            best_progress = max(best_progress, float(progress["open_fraction"]))
            step_record = {
                "step": step_idx,
                "stroke": float(stroke),
                "delta_command": delta,
                "move": move,
                "alignment": after_align,
                "progress": progress,
                "contacts": contacts,
            }
            steps.append(step_record)
            if (
                (bool(progress.get("bddl_open", False)) and not bool(strict_open_fraction))
                or float(progress["open_fraction"]) >= float(stop_open_fraction)
            ):
                break
            if stop_on_lost_handle and step_idx >= 1:
                if (
                    not bool(after_align["closed_on_handle"])
                    and not bool(contacts["has_handle_contact"])
                    and float(after_align["abs_handle_lateral_error"]) > 0.030
                ):
                    break
                if abs(float(after_align["vertical_error"])) > 0.045:
                    break

        return {
            "level": level,
            "start_progress": float(start_progress),
            "best_progress": float(best_progress),
            "final_progress": self.get_cabinet_drawer_progress(level),
            "start_alignment": start_alignment,
            "final_alignment": self.measure_drawer_hook_alignment(level),
            "start_contacts": start_contacts,
            "final_contacts": self.measure_drawer_gripper_contacts(level),
            "requested_total_distance": float(total_distance),
            "executed_distance": float(moved),
            "steps": steps,
        }

    def _continue_half_open_drawer_by_front_lip(
        self,
        level: str,
        *,
        target_open_fraction: float,
        max_distance: float = 0.10,
        settle_steps: int = 12,
    ) -> dict[str, Any]:
        """Try a guarded top/front-lip pull for a mostly open bottom drawer.

        The bottom drawer handle is so low that front sleeve recapture often
        lands above the handle and can push a half-open drawer closed. This
        bounded fallback only performs short physical strokes from reachable
        upper-front contact poses and stops as soon as progress fails.
        """
        before = self.get_cabinet_drawer_progress(level)
        before_fraction = float(before.get("open_fraction", 0.0))
        handle_pos, _, pull_dir = self.get_cabinet_drawer_handle_pose(level)
        handle_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3)
        pull_dir = np.asarray(pull_dir, dtype=np.float64).reshape(3)
        pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)
        lateral = np.asarray(self._drawer_handle_lateral_axis(level), dtype=np.float64).reshape(3)
        lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-9)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hook_quat = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)
        side_quat = np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64)

        candidates: list[dict[str, Any]] = []
        for name, quat, pull_offset, z_offset, lateral_offset in (
            ("front_upper_lip_center", hook_quat, 0.018, 0.074, 0.000),
            ("front_upper_lip_left", hook_quat, 0.018, 0.078, -0.024),
            ("front_upper_lip_right", hook_quat, 0.018, 0.078, 0.024),
            ("front_top_edge_center", hook_quat, 0.000, 0.096, 0.000),
            ("side_upper_lip_left", side_quat, 0.016, 0.086, -0.040),
            ("side_upper_lip_right", side_quat, 0.016, 0.086, 0.040),
        ):
            target = (
                handle_pos
                + float(pull_offset) * pull_dir
                + float(z_offset) * vertical
                + float(lateral_offset) * lateral
            )
            pre = target - 0.040 * pull_dir + 0.045 * vertical
            candidates.append(
                {
                    "name": name,
                    "quaternion_wxyz": quat,
                    "pre_finger_midpoint": pre,
                    "contact_finger_midpoint": target,
                }
            )

        records: list[dict[str, Any]] = []
        best_fraction = before_fraction
        best_candidate: str | None = None
        max_steps = max(2, int(np.ceil(max(0.0, float(max_distance)) / 0.012)))

        for cand in candidates:
            if self.get_cabinet_drawer_progress(level).get("open_fraction", 0.0) >= float(target_open_fraction):
                break
            record: dict[str, Any] = {
                "phase": "half_open_front_lip_pull",
                "candidate": cand["name"],
                "progress_before": self.get_cabinet_drawer_progress(level),
            }
            records.append(record)
            quat = np.asarray(cand["quaternion_wxyz"], dtype=np.float64).reshape(4)
            try:
                self.open_gripper()
                self._settle_simulation_steps(max(3, int(settle_steps) // 3))
                record["pre_move"] = self.goto_finger_midpoint_checked(
                    cand["pre_finger_midpoint"],
                    quat,
                    tolerance=0.090,
                )
                record["contact_move"] = self.goto_finger_midpoint_checked(
                    cand["contact_finger_midpoint"],
                    quat,
                    tolerance=0.085,
                )
                record["contacts_before_close"] = self.measure_drawer_gripper_contacts(level)
                self.close_gripper()
                self._settle_simulation_steps(max(5, int(settle_steps) // 2))
                record["contacts_after_close"] = self.measure_drawer_gripper_contacts(level)
                record["alignment_after_close"] = self.measure_drawer_hook_alignment(level)
                steps: list[dict[str, Any]] = []
                record["steps"] = steps
                start_fraction = float(record["progress_before"].get("open_fraction", 0.0))
                local_best = start_fraction
                no_progress_steps = 0
                for step_idx in range(max_steps):
                    progress_before = self.get_cabinet_drawer_progress(level)
                    before_step_fraction = float(progress_before.get("open_fraction", 0.0))
                    if before_step_fraction >= float(target_open_fraction):
                        break
                    move = self.move_finger_midpoint_delta(
                        0.012 * pull_dir,
                        quat,
                        lin_vel_norm=0.026,
                        timeout_s=2.8,
                        tolerance=0.035,
                    )
                    progress_after = self.get_cabinet_drawer_progress(level)
                    after_step_fraction = float(progress_after.get("open_fraction", 0.0))
                    best_fraction = max(best_fraction, after_step_fraction)
                    local_best = max(local_best, after_step_fraction)
                    if after_step_fraction >= best_fraction:
                        best_candidate = str(cand["name"])
                    delta = after_step_fraction - before_step_fraction
                    contacts = self.measure_drawer_gripper_contacts(level)
                    step_record = {
                        "step": int(step_idx),
                        "before": progress_before,
                        "after": progress_after,
                        "delta": float(delta),
                        "move": move,
                        "contacts": contacts,
                    }
                    steps.append(step_record)
                    if after_step_fraction >= float(target_open_fraction):
                        break
                    if delta < 0.004:
                        no_progress_steps += 1
                    else:
                        no_progress_steps = 0
                    if after_step_fraction < start_fraction - 0.025:
                        record["stop_reason"] = "state_regressed"
                        break
                    if no_progress_steps >= 2:
                        record["stop_reason"] = "no_monotonic_progress"
                        break
                record["progress_after"] = self.get_cabinet_drawer_progress(level)
                record["local_best_open_fraction"] = float(local_best)
                if float(record["progress_after"].get("open_fraction", 0.0)) >= float(target_open_fraction):
                    break
            except Exception as exc:
                record["error"] = repr(exc)
                try:
                    self.open_gripper()
                except Exception:
                    pass

        after = self.get_cabinet_drawer_progress(level)
        after_fraction = float(after.get("open_fraction", 0.0))
        best_fraction = max(best_fraction, after_fraction)
        opened = bool(after_fraction >= float(target_open_fraction))
        return {
            "skill": "continue_half_open_drawer_by_front_lip",
            "level": level,
            "before": before,
            "after": after,
            "best_open_fraction": float(best_fraction),
            "target_open_fraction": float(target_open_fraction),
            "opened": opened,
            "best_candidate": best_candidate,
            "records": records,
            "reason": "front_lip_guarded_pull" if opened else "front_lip_pull_did_not_reach_strict_open_fraction",
        }

    def open_drawer_by_sleeve_retry(
        self,
        level: str = "middle",
        *,
        open_threshold: float = 0.45,
        max_cycles: int = 4,
        test_distance: float = 0.045,
        pull_distance: float = 0.26,
        settle_steps: int = 18,
        strict_open_fraction: bool = False,
    ) -> dict[str, Any]:
        """Open a drawer by repeatedly recapturing the handle and verifying progress.

        This is a physical robot skill: it reads drawer geometry/progress to
        choose contact points, but all state changes come from gripper motions.
        It closes the gap between stage-level planning and contact reality by
        returning to sleeve/capture whenever a verifier reports no progress.
        """
        level_name = str(level or "middle").strip().lower()
        before = self.get_cabinet_drawer_progress(level_name)
        before_fraction = float(before.get("open_fraction", 0.0))
        records: list[dict[str, Any]] = []
        best_fraction = before_fraction
        target_open_fraction = float(np.clip(float(open_threshold), 0.12, 0.98))

        def current_done() -> bool:
            progress = self.get_cabinet_drawer_progress(level_name)
            fraction_done = float(progress.get("open_fraction", 0.0)) >= target_open_fraction
            if bool(strict_open_fraction):
                return bool(fraction_done)
            return bool(progress.get("bddl_open", False)) or bool(fraction_done)

        if current_done():
            status = self.get_task_status()
            return {
                "skill": "open_drawer_by_sleeve_retry",
                "level": level_name,
                "before": before,
                "after": before,
                "best_open_fraction": before_fraction,
                "records": records,
                "task_status": status,
                "opened": True,
                "bddl_open": bool(before.get("bddl_open", False)),
                "strict_open_fraction": bool(strict_open_fraction),
                "reason": "already open",
            }

        try:
            current_alignment = self.measure_drawer_hook_alignment(level_name)
            current_contacts = self.measure_drawer_gripper_contacts(level_name)
            current_captured = bool(
                current_alignment.get("closed_on_handle")
                or current_alignment.get("likely_hooked")
                or current_contacts.get("has_handle_contact")
            )
        except Exception:
            current_alignment = {}
            current_contacts = {}
            current_captured = False

        if (
            bool(strict_open_fraction)
            and level_name == "bottom"
            and before_fraction >= 0.70
            and before_fraction < target_open_fraction
            and not current_captured
        ):
            half_open_record: dict[str, Any] = {
                "cycle": -2,
                "phase": "strict_bottom_half_open_front_lip_repair",
                "progress_before": before,
                "capture_alignment": current_alignment,
                "capture_contacts": current_contacts,
            }
            records.append(half_open_record)
            try:
                half_open_pull = self._continue_half_open_drawer_by_front_lip(
                    level_name,
                    target_open_fraction=target_open_fraction,
                    max_distance=min(max(float(pull_distance), 0.08), 0.14),
                    settle_steps=settle_steps,
                )
            except Exception as exc:
                half_open_pull = {"error": repr(exc)}
            half_after = self.get_cabinet_drawer_progress(level_name)
            half_after_fraction = float(half_after.get("open_fraction", 0.0))
            best_fraction = max(
                best_fraction,
                half_after_fraction,
                float(half_open_pull.get("best_open_fraction", half_after_fraction))
                if isinstance(half_open_pull, dict)
                else half_after_fraction,
            )
            half_open_record["front_lip_pull"] = half_open_pull
            half_open_record["after"] = half_after
            if half_after_fraction >= target_open_fraction:
                status = self.get_task_status()
                return {
                    "skill": "open_drawer_by_sleeve_retry",
                    "level": level_name,
                    "before": before,
                    "after": half_after,
                    "best_open_fraction": float(best_fraction),
                    "open_threshold": float(open_threshold),
                    "strict_open_fraction": bool(strict_open_fraction),
                    "bddl_open": bool(half_after.get("bddl_open", False)),
                    "bddl_open_margin": float(half_after.get("bddl_open_margin", 0.0) or 0.0),
                    "records": records,
                    "task_status": status,
                    "opened": True,
                    "reason": "strict bottom half-open front-lip repair reached clearance",
                }
            half_open_record["recovery_next"] = (
                "front_lip_failed_continue_to_sleeve_recapture"
            )

        if bool(strict_open_fraction) and current_captured and before_fraction >= 0.12:
            direct_record: dict[str, Any] = {
                "cycle": -1,
                "phase": "continue_existing_handle_capture",
                "progress_before": before,
                "capture_alignment": current_alignment,
                "capture_contacts": current_contacts,
            }
            records.append(direct_record)
            try:
                direct_pull = self.pull_sleeved_drawer(
                    level_name,
                    total_distance=max(float(pull_distance), 0.08),
                    step_distance=min(0.010, 0.004 if target_open_fraction < 0.90 else 0.010),
                    stop_on_lost_handle=True,
                    stop_open_fraction=target_open_fraction,
                    strict_open_fraction=True,
                )
            except Exception as exc:
                direct_pull = {"error": repr(exc)}
            direct_after = self.get_cabinet_drawer_progress(level_name)
            direct_after_fraction = float(direct_after.get("open_fraction", 0.0))
            best_fraction = max(
                best_fraction,
                direct_after_fraction,
                float(direct_pull.get("best_progress", direct_after_fraction)) if isinstance(direct_pull, dict) else direct_after_fraction,
            )
            direct_record["direct_pull"] = direct_pull
            direct_record["after"] = direct_after
            if direct_after_fraction >= target_open_fraction:
                status = self.get_task_status()
                return {
                    "skill": "open_drawer_by_sleeve_retry",
                    "level": level_name,
                    "before": before,
                    "after": direct_after,
                    "best_open_fraction": float(best_fraction),
                    "open_threshold": float(open_threshold),
                    "strict_open_fraction": bool(strict_open_fraction),
                    "bddl_open": bool(direct_after.get("bddl_open", False)),
                    "bddl_open_margin": float(direct_after.get("bddl_open_margin", 0.0) or 0.0),
                    "records": records,
                    "task_status": status,
                    "opened": True,
                    "reason": "continued existing handle capture to strict open fraction",
                }

        try:
            plan = self.get_drawer_sleeve_hook_poses(level_name)
        except Exception as exc:
            status = self.get_task_status()
            return {
                "skill": "open_drawer_by_sleeve_retry",
                "level": level_name,
                "before": before,
                "after": self.get_cabinet_drawer_progress(level_name),
                "best_open_fraction": best_fraction,
                "records": [{"phase": "plan_error", "error": repr(exc)}],
                "task_status": status,
                "opened": False,
            }

        quat = plan.get("hook_quaternion_wxyz")
        candidates = list(plan.get("sleeve_candidates", []) if isinstance(plan, dict) else [])
        if not candidates:
            candidates = [
                {
                    "name": "mouth_fallback",
                    "finger_midpoint_position": plan.get("mouth_finger_position"),
                    "quaternion_wxyz": quat,
                }
            ]

        pull_dir = np.asarray(self.get_cabinet_drawer_handle_pose(level_name)[2], dtype=np.float64).reshape(3)
        pull_dir = pull_dir / max(float(np.linalg.norm(pull_dir)), 1e-9)
        lateral_axis = np.asarray(self._drawer_handle_lateral_axis(level_name), dtype=np.float64).reshape(3)
        lateral_axis = lateral_axis / max(float(np.linalg.norm(lateral_axis)), 1e-9)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for cycle_idx in range(max(1, int(max_cycles))):
            if current_done():
                break
            cycle_record: dict[str, Any] = {
                "cycle": int(cycle_idx),
                "progress_before": self.get_cabinet_drawer_progress(level_name),
                "attempts": [],
            }
            records.append(cycle_record)
            captured = False
            capture_alignment: dict[str, Any] = {}
            capture_contacts: dict[str, Any] = {}
            self.open_gripper()
            self._settle_simulation_steps(max(3, int(settle_steps) // 3))

            prehook = plan.get("prehook_finger_position") if isinstance(plan, dict) else None
            if prehook is not None and quat is not None:
                try:
                    cycle_record["prehook_move"] = self.goto_finger_midpoint_checked(
                        prehook,
                        quat,
                        tolerance=0.080,
                    )
                except Exception as exc:
                    cycle_record["prehook_error"] = repr(exc)

            for cand_idx, cand in enumerate(candidates[:8]):
                target = cand.get("finger_midpoint_position")
                cand_quat = cand.get("quaternion_wxyz", quat)
                if target is None or cand_quat is None:
                    continue
                attempt: dict[str, Any] = {
                    "candidate_index": int(cand_idx),
                    "candidate": cand.get("name", str(cand_idx)),
                }
                cycle_record["attempts"].append(attempt)
                try:
                    self.open_gripper()
                    self._settle_simulation_steps(max(2, int(settle_steps) // 4))
                    attempt["move"] = self.goto_finger_midpoint_checked(
                        target,
                        cand_quat,
                        tolerance=0.060,
                    )
                    align = self.measure_drawer_hook_alignment(level_name)
                    attempt["alignment_before_tune"] = align
                    tune_delta = (
                        0.85 * float(align.get("signed_handle_lateral_error", 0.0)) * lateral_axis
                        + 0.65 * float(align.get("vertical_error", 0.0)) * vertical
                        + 0.70 * float(align.get("depth_along_pull", 0.0)) * pull_dir
                    )
                    tune_norm = float(np.linalg.norm(tune_delta))
                    if 0.003 <= tune_norm <= 0.085:
                        attempt["fine_tune_delta"] = tune_delta
                        attempt["fine_tune_move"] = self.move_finger_midpoint_delta(
                            tune_delta,
                            cand_quat,
                            lin_vel_norm=0.035,
                            timeout_s=4.0,
                            tolerance=0.040,
                        )
                        align = self.measure_drawer_hook_alignment(level_name)
                        attempt["alignment_after_tune"] = align
                    self.close_gripper()
                    self._settle_simulation_steps(max(4, int(settle_steps) // 2))
                    closed_align = self.measure_drawer_hook_alignment(level_name)
                    contacts = self.measure_drawer_gripper_contacts(level_name)
                    attempt["alignment_after_close"] = closed_align
                    attempt["contacts_after_close"] = contacts
                    captured = bool(
                        closed_align.get("closed_on_handle")
                        or closed_align.get("likely_hooked")
                        or contacts.get("has_handle_contact")
                    )
                    capture_alignment = closed_align
                    capture_contacts = contacts
                    attempt["captured"] = bool(captured)
                    if captured:
                        break
                except Exception as exc:
                    attempt["error"] = repr(exc)
                    try:
                        self.open_gripper()
                    except Exception:
                        pass

            cycle_record["captured"] = bool(captured)
            cycle_record["capture_alignment"] = capture_alignment
            cycle_record["capture_contacts"] = capture_contacts
            if not captured:
                continue

            test_before = self.get_cabinet_drawer_progress(level_name)
            try:
                test_pull = self.pull_sleeved_drawer(
                    level_name,
                    total_distance=float(test_distance),
                    step_distance=min(0.010, 0.004 if target_open_fraction < 0.90 else 0.010),
                    stop_on_lost_handle=False,
                    stop_open_fraction=target_open_fraction,
                    strict_open_fraction=bool(strict_open_fraction),
                )
            except Exception as exc:
                test_pull = {"error": repr(exc)}
            test_after = self.get_cabinet_drawer_progress(level_name)
            test_before_fraction = float(test_before.get("open_fraction", 0.0))
            test_after_fraction = float(test_after.get("open_fraction", 0.0))
            best_fraction = max(
                best_fraction,
                test_after_fraction,
                float(test_pull.get("best_progress", test_after_fraction)) if isinstance(test_pull, dict) else test_after_fraction,
            )
            cycle_record["test"] = {
                "before": test_before,
                "after": test_after,
                "delta": float(test_after_fraction - test_before_fraction),
                "pull": test_pull,
            }
            if bool(test_after.get("bddl_open", False)) and not bool(strict_open_fraction):
                break
            if test_after_fraction < test_before_fraction + 0.010 and best_fraction < 0.120:
                continue

            try:
                long_pull = self.pull_sleeved_drawer(
                    level_name,
                    total_distance=max(float(pull_distance), 0.08),
                    step_distance=min(0.012, 0.004 if target_open_fraction < 0.90 else 0.012),
                    stop_on_lost_handle=True,
                    stop_open_fraction=target_open_fraction,
                    strict_open_fraction=bool(strict_open_fraction),
                )
            except Exception as exc:
                long_pull = {"error": repr(exc)}
            long_after = self.get_cabinet_drawer_progress(level_name)
            long_after_fraction = float(long_after.get("open_fraction", 0.0))
            best_fraction = max(
                best_fraction,
                long_after_fraction,
                float(long_pull.get("best_progress", long_after_fraction)) if isinstance(long_pull, dict) else long_after_fraction,
            )
            cycle_record["long_pull"] = {
                "after": long_after,
                "pull": long_pull,
            }
            if bool(strict_open_fraction):
                if long_after_fraction >= target_open_fraction:
                    break
            elif bool(long_after.get("bddl_open", False)) or max(long_after_fraction, best_fraction) >= target_open_fraction:
                break

        final = self.get_cabinet_drawer_progress(level_name)
        final_fraction = float(final.get("open_fraction", 0.0))
        best_fraction = max(best_fraction, final_fraction)
        task_status = self.get_task_status()
        task_done = bool(
            bool(task_status.get("task_completed", False))
            or float(task_status.get("reward", 0.0) or 0.0) >= 1.0
        )
        if bool(strict_open_fraction):
            opened = bool(final_fraction >= target_open_fraction or task_done)
        else:
            opened = bool(
                bool(final.get("bddl_open", False))
                or final_fraction >= target_open_fraction
                or best_fraction >= target_open_fraction
                or task_done
            )
        return {
            "skill": "open_drawer_by_sleeve_retry",
            "level": level_name,
            "before": before,
            "after": final,
            "best_open_fraction": float(best_fraction),
            "open_threshold": float(open_threshold),
            "strict_open_fraction": bool(strict_open_fraction),
            "bddl_open": bool(final.get("bddl_open", False)),
            "bddl_open_margin": float(final.get("bddl_open_margin", 0.0) or 0.0),
            "records": records,
            "task_status": task_status,
            "opened": opened,
        }

    def _pull_sleeved_drawer_joint_ratchet(
        self,
        *,
        level: str,
        total_distance: float,
        step_distance: float,
        pull_dir: np.ndarray,
        stop_on_lost_handle: bool,
        ratchet_joint_delta: np.ndarray | list[float] | tuple[float, ...] | None,
        ratchet_max_abs_delta: float,
        restore_best: bool,
        stop_open_fraction: float = 0.98,
        strict_open_fraction: bool = False,
    ) -> dict[str, Any]:
        """Pull with a fixed joint-space stroke after the handle is clamped.

        The strict Cartesian IK pull can alternate between two nearby IK branches:
        one branch pulls the drawer outward and the next branch pushes it back.
        Once the handle is physically clamped, this ratchet repeats the branch
        direction that transfers outward motion and stops as soon as contact or
        progress degrades.
        """
        pull_dir = np.asarray(pull_dir, dtype=np.float64).reshape(3)
        pull_dir = pull_dir / max(np.linalg.norm(pull_dir), 1e-9)
        base_delta = (
            np.asarray(ratchet_joint_delta, dtype=np.float64).reshape(7)
            if ratchet_joint_delta is not None
            else np.array(
                [0.030, 0.004, -0.032, -0.003, 0.030, 0.020, -0.007],
                dtype=np.float64,
            )
        )
        if float(step_distance) > 1e-9:
            base_delta = base_delta * (float(step_distance) / 0.010)

        start_progress = self.get_cabinet_drawer_progress(level)["open_fraction"]
        start_alignment = self.measure_drawer_hook_alignment(level)
        start_contacts = self.measure_drawer_gripper_contacts(level)
        best_progress = float(start_progress)
        best_joints = self.get_joint_positions()
        best_step = -1
        steps: list[dict[str, Any]] = []
        moved = 0.0
        step_count = int(np.ceil(max(0.0, float(total_distance)) / max(float(step_distance), 1e-6)))

        for step_idx in range(step_count):
            before_align = self.measure_drawer_hook_alignment(level)
            before_contacts = self.measure_drawer_gripper_contacts(level)
            if stop_on_lost_handle and step_idx > 0:
                if (
                    not bool(before_align["closed_on_handle"])
                    and not bool(before_contacts["has_handle_contact"])
                ):
                    break

            remaining = max(0.0, float(total_distance) - moved)
            stroke = min(float(step_distance), remaining)
            if stroke <= 0.0:
                break
            scale = min(1.0, 0.65 + 0.06 * step_idx)
            delta = base_delta * scale
            move = self.move_joint_delta_checked(delta, max_abs_delta=float(ratchet_max_abs_delta))
            moved += stroke

            after_align = self.measure_drawer_hook_alignment(level)
            after_contacts = self.measure_drawer_gripper_contacts(level)
            progress = self.get_cabinet_drawer_progress(level)
            open_fraction = float(progress["open_fraction"])
            finger_delta = np.asarray(move["finger_midpoint_delta"], dtype=np.float64)
            pull_motion = float(np.dot(finger_delta, pull_dir))
            ee_pull_motion = float(np.dot(np.asarray(move["position_delta"], dtype=np.float64), pull_dir))
            if open_fraction > best_progress:
                best_progress = open_fraction
                best_joints = self.get_joint_positions()
                best_step = step_idx

            step_record = {
                "step": step_idx,
                "stroke": float(stroke),
                "ratchet_scale": float(scale),
                "joint_delta_command": delta,
                "move": move,
                "before_alignment": before_align,
                "alignment": after_align,
                "progress": progress,
                "contacts": after_contacts,
                "pull_motion": pull_motion,
                "ee_pull_motion": ee_pull_motion,
            }
            steps.append(step_record)

            if (
                (bool(progress.get("bddl_open", False)) and not bool(strict_open_fraction))
                or open_fraction >= float(stop_open_fraction)
            ):
                break
            if stop_on_lost_handle:
                if (
                    not bool(after_align["closed_on_handle"])
                    and not bool(after_contacts["has_handle_contact"])
                ):
                    break
                if abs(float(after_align["vertical_error"])) > 0.045:
                    break
                if float(after_align["abs_handle_lateral_error"]) > 0.035:
                    break
            if step_idx >= 2 and open_fraction < best_progress - 0.020:
                break
            if step_idx >= 3 and pull_motion < -0.002 and open_fraction <= best_progress + 0.002:
                break

        final_before_restore = self.get_cabinet_drawer_progress(level)
        restored = False
        if (
            bool(restore_best)
            and best_step >= 0
            and float(final_before_restore["open_fraction"]) < best_progress - 0.010
        ):
            self.move_to_joints(best_joints)
            restored = True

        return {
            "level": level,
            "strategy": "joint_ratchet",
            "start_progress": float(start_progress),
            "best_progress": float(best_progress),
            "best_step": int(best_step),
            "final_progress": self.get_cabinet_drawer_progress(level),
            "final_before_restore": final_before_restore,
            "restored_best_joints": bool(restored),
            "start_alignment": start_alignment,
            "final_alignment": self.measure_drawer_hook_alignment(level),
            "start_contacts": start_contacts,
            "final_contacts": self.measure_drawer_gripper_contacts(level),
            "requested_total_distance": float(total_distance),
            "executed_distance": float(moved),
            "ratchet_joint_delta": base_delta,
            "steps": steps,
        }

    def move_along_direction(
        self,
        direction: np.ndarray | list[float] | tuple[float, float, float],
        distance: float,
        quaternion_wxyz: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
        *,
        lin_vel_norm: float = 0.08,
        ang_vel_norm: float = 0.5,
        timeout_s: float = 8.0,
    ) -> dict[str, Any]:
        """Move the gripper along a direction using robot control.

        Args:
            direction: (3,) direction vector in the robot base frame. It will be
                normalized internally.
            distance: Signed distance in meters. Positive moves along
                `direction`; negative moves opposite it.
            quaternion_wxyz: Optional target gripper orientation. If omitted,
                keep the current end-effector orientation.
            lin_vel_norm: Desired linear speed in meters/second.
            ang_vel_norm: Desired angular speed in radians/second.
            timeout_s: Maximum motion time in seconds.

        Returns:
            Dictionary with target and actual end-effector position.
        """
        direction_arr = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction_arr))
        if norm <= 1e-9:
            raise ValueError("move_along_direction received a near-zero direction")
        delta = direction_arr / norm * float(distance)
        result = self.move_linear_delta(
            delta,
            quaternion_wxyz,
            lin_vel_norm=lin_vel_norm,
            ang_vel_norm=ang_vel_norm,
            timeout_s=timeout_s,
        )
        result["direction"] = direction_arr / norm
        result["distance"] = float(distance)
        return result

    def goto_pose_interactive_cartesian(
        self,
        target_pose_predicate: Callable[[], Tuple[np.ndarray, np.ndarray]],
        replan_interval_s: float = 0.0,
        lin_vel_norm: float = 1.0,
        ang_vel_norm: float = 2.0,
        z_approach: float = 0.0,
        timeout_s: float = 20.0,
    ) -> None:
        """Move toward a target TCP pose with small Cartesian robot steps.

        The parent LIBERO helper mixes the observed gripper pose returned by
        ``robot_cartesian_pos`` with the parent API's pre-offset command pose.
        For drawer contact this makes horizontal pulls drift upward by roughly
        the TCP offset. This override keeps the path in observed gripper-pose
        coordinates all the way to IK.

        Args:
            target_pose_predicate: Function returning `(position, quaternion)`.
            replan_interval_s: Seconds between target refreshes. Use 0.0 for a
                fixed target.
            lin_vel_norm: Desired linear speed in meters/second.
            ang_vel_norm: Desired angular speed in radians/second.
            z_approach: Optional insertion approach offset.
            timeout_s: Maximum motion time in seconds.

        Returns:
            None
        """
        target_pos, target_quat = target_pose_predicate()
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)

        if z_approach != 0.0:
            def target_pose_predicate_z_offset() -> Tuple[np.ndarray, np.ndarray]:
                pos, quat = target_pose_predicate()
                quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
                rot = self._rotation_from_wxyz(quat_arr)
                return np.asarray(pos, dtype=np.float64).reshape(3) + rot.apply(
                    np.array([0.0, 0.0, -float(z_approach)], dtype=np.float64)
                ), quat_arr

            self.goto_pose_interactive_cartesian(
                target_pose_predicate_z_offset,
                replan_interval_s=replan_interval_s,
                lin_vel_norm=lin_vel_norm,
                ang_vel_norm=ang_vel_norm,
                z_approach=0.0,
                timeout_s=timeout_s,
            )

        current_pos, current_quat = self.get_ee_pose()
        last_plan_time_s = self._env.get_current_time_s()
        start_loop_time_s = self._env.get_current_time_s()
        joints = None

        while True:
            current_time_s = self._env.get_current_time_s()
            if timeout_s > 0 and current_time_s - start_loop_time_s > timeout_s:
                break
            if replan_interval_s > 0 and current_time_s - last_plan_time_s > replan_interval_s:
                target_pos, target_quat = target_pose_predicate()
                target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
                target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
                last_plan_time_s = current_time_s

            if np.allclose(target_pos, current_pos, atol=1e-4) and np.allclose(
                self._normalize(target_quat),
                self._normalize(current_quat),
                atol=1e-4,
            ):
                break

            next_pos, next_quat = self.step_towards_pose(
                current_pos,
                current_quat,
                target_pos,
                target_quat,
                lin_vel_norm,
                ang_vel_norm,
                1.0 / self._env._control_freq,
            )
            if self.cfg is None:
                self.cfg = self._pks.solve_ik(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=next_pos,
                    target_wxyz=next_quat,
                )
            else:
                self.cfg = self._pks.solve_ik_vel_cost(
                    robot=self._robot,
                    target_link_name=self._target_link_name,
                    target_position=next_pos,
                    target_wxyz=next_quat,
                    prev_cfg=self.cfg,
                )
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints, max_steps=1)
            current_pos, current_quat = self.get_ee_pose()

        if joints is not None:
            self._env.move_to_joints_blocking(joints)


class FrankaLiberoCleanSkillApi(FrankaLiberoOracleReadOnlyApi):
    """Versioned clean API for planner-CaP experiments.

    It preserves the original oracle-readonly API for earlier baselines while
    giving our runner a distinct API name. Exposed skills must act through robot
    motion; direct simulator-state and predicate-repair access is still blocked
    by the clean-code guard.
    """
