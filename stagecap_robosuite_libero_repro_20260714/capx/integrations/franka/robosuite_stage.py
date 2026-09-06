from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.franka.handover_privileged import FrankaHandoverPrivilegedApi
from capx.integrations.franka.nut_assembly_privileged import (
    FrankaControlNutAssemblyPrivilegedApi,
)
from capx.integrations.franka.spill_wipe_privileged import (
    FrankaControlSpillWipePrivilegedApi,
)
from capx.integrations.franka.two_arm_lift_privileged import (
    FrankaTwoArmLiftPrivilegedApi,
)


def _quaternion_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    a = np.asarray(q0, dtype=np.float64).reshape(4)
    b = np.asarray(q1, dtype=np.float64).reshape(4)
    a /= max(float(np.linalg.norm(a)), 1e-9)
    b /= max(float(np.linalg.norm(b)), 1e-9)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def _matrix_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    xyzw = SciRotation.from_matrix(matrix[:3, :3]).as_quat()
    return matrix[:3, 3].copy(), np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


class _RobosuitePrivilegedStageMixin:
    """Read-only diagnostics shared by privileged RoboSuite Stage-CAP APIs."""

    TASK_FAMILY = "robosuite"
    OBJECT_NAMES: tuple[str, ...] = ()

    def _common_functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_task_status": self.get_task_status,
            "get_robosuite_task_context": self.get_robosuite_task_context,
            "get_all_object_poses": self.get_all_object_poses,
            "get_instance_object_names": self.get_instance_object_names,
            "get_contact_diagnostics": self.get_contact_diagnostics,
            "get_gripper_state": self.get_gripper_state,
            "get_object_grasp_affordance": self.get_object_grasp_affordance,
            "measure_object_grasp_state": self.measure_object_grasp_state,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the current RoboSuite observation without modifying simulator state."""
        return self._env.get_observation()

    def get_task_status(self) -> dict[str, Any]:
        """Return privileged read-only reward and benchmark completion state."""
        return {
            "reward": float(self._env.compute_reward()),
            "task_completed": bool(self._env.task_completed()),
            "sim_step_count": int(getattr(self._env, "_sim_step_count", -1)),
            "max_steps": int(getattr(self._env, "max_steps", -1)),
        }

    def get_robosuite_task_context(self) -> dict[str, Any]:
        """Return the task family and exact object names used by this audited API."""
        return {
            "simulator": "robosuite",
            "task_family": self.TASK_FAMILY,
            "privileged_readonly": True,
            "objects": list(self.OBJECT_NAMES),
            "state_mutation_allowed": False,
        }

    def get_instance_object_names(self) -> list[str]:
        """List the exact semantic object names accepted by this API."""
        return list(self.OBJECT_NAMES)

    def get_contact_diagnostics(self, pattern: str | None = None) -> list[dict[str, Any]]:
        """Return current MuJoCo contacts, optionally filtered by geom or body name."""
        sim = self._env.robosuite_env.sim
        needle = str(pattern).lower() if pattern else None
        contacts: list[dict[str, Any]] = []
        for index in range(int(sim.data.ncon)):
            contact = sim.data.contact[index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1 = sim.model.geom_id2name(geom1_id) or ""
            geom2 = sim.model.geom_id2name(geom2_id) or ""
            body1 = sim.model.body_id2name(int(sim.model.geom_bodyid[geom1_id])) or ""
            body2 = sim.model.body_id2name(int(sim.model.geom_bodyid[geom2_id])) or ""
            text = " ".join((geom1, geom2, body1, body2)).lower()
            if needle and needle not in text:
                continue
            contacts.append(
                {
                    "index": index,
                    "geom1": geom1,
                    "geom2": geom2,
                    "body1": body1,
                    "body2": body2,
                    "distance": float(contact.dist),
                    "position_world": np.asarray(contact.pos, dtype=np.float64).copy(),
                }
            )
        return contacts

    def _contact_count(self, object_tokens: tuple[str, ...], robot_tokens: tuple[str, ...]) -> int:
        count = 0
        for contact in self.get_contact_diagnostics():
            left = " ".join(
                str(contact.get(key, "")).lower()
                for key in ("geom1", "body1")
            )
            right = " ".join(
                str(contact.get(key, "")).lower()
                for key in ("geom2", "body2")
            )
            object_left = any(token in left for token in object_tokens)
            object_right = any(token in right for token in object_tokens)
            robot_left = any(token in left for token in robot_tokens)
            robot_right = any(token in right for token in robot_tokens)
            if (object_left and robot_right) or (object_right and robot_left):
                count += 1
        return count

    def _settle(self, steps: int = 20) -> None:
        for _ in range(max(0, int(steps))):
            self._env._step_once()

    def _skill_result(
        self,
        skill: str,
        *,
        success: bool,
        reason: str,
        metrics: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> dict[str, Any]:
        status = self.get_task_status()
        effective_success = bool(success or status["task_completed"])
        return {
            "skill": skill,
            "success": effective_success,
            "recoverable": bool(
                False
                if effective_success
                else (not success if recoverable is None else recoverable)
            ),
            "reason": reason,
            "metrics": metrics,
            "evidence": evidence or {},
            "task_status": status,
        }


class FrankaRobosuiteStackStageApi(
    _RobosuitePrivilegedStageMixin,
    FrankaControlPrivilegedApi,
):
    """Privileged-readonly Stage-CAP API for RoboSuite cube stacking."""

    TASK_FAMILY = "stack"
    OBJECT_NAMES = ("red cube", "green cube")

    def __init__(self, env: BaseEnv) -> None:
        FrankaControlPrivilegedApi.__init__(self, env)

    def functions(self) -> dict[str, Any]:
        functions = FrankaControlPrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_ee_pose": self.get_ee_pose,
                "get_stack_diagnostic": self.get_stack_diagnostic,
                "goto_pose_checked": self.goto_pose_checked,
                "verified_pick_cube": self.verified_pick_cube,
                "transport_cube_above_target": self.transport_cube_above_target,
                "release_and_verify_stack": self.release_and_verify_stack,
            }
        )
        return functions

    @staticmethod
    def _cube_token(object_name: str) -> str:
        name = str(object_name).lower()
        if "red" in name or "primary" in name or "cubea" in name:
            return "cubea"
        if "green" in name or "secondary" in name or "cubeb" in name:
            return "cubeb"
        raise ValueError(f"Unknown cube: {object_name!r}")

    def _cube_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion, extent = self.get_object_pose(object_name, return_bbox_extent=True)
        return (
            np.asarray(position, dtype=np.float64).copy(),
            np.asarray(quaternion, dtype=np.float64).copy(),
            np.asarray(extent, dtype=np.float64).copy(),
        )

    def get_all_object_poses(self) -> dict[str, dict[str, Any]]:
        """Return exact red and green cube poses and full extents."""
        result: dict[str, dict[str, Any]] = {}
        for name in self.OBJECT_NAMES:
            position, quaternion, extent = self._cube_pose(name)
            result[name] = {
                "position": position,
                "quaternion_wxyz": quaternion,
                "extent": extent,
            }
        return result

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current gripper contact-center pose in robot-base coordinates."""
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_gripper_state(self) -> dict[str, Any]:
        """Return normalized gripper opening and current end-effector pose."""
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return {
            "opening": float(pose[-1]),
            "position": pose[:3].copy(),
            "quaternion_wxyz": pose[3:7].copy(),
        }

    def get_object_grasp_affordance(self, object_name: str) -> dict[str, Any]:
        """Return the exact top-down grasp candidate for a cube."""
        position, _quaternion, extent = self._cube_pose(object_name)
        return {
            "object_name": object_name,
            "extent": extent,
            "candidates": [
                {
                    "position": position,
                    "quaternion_wxyz": np.array([0.0, 0.0, 1.0, 0.0]),
                    "approach": 0.10,
                }
            ],
        }

    def measure_object_grasp_state(self, object_name: str) -> dict[str, Any]:
        """Measure cube lift, gripper proximity, opening, and contact evidence."""
        position, _quaternion, _extent = self._cube_pose(object_name)
        ee_position, _ee_quaternion = self.get_ee_pose()
        opening = float(self.get_gripper_state()["opening"])
        distance = float(np.linalg.norm(position - ee_position))
        contact_count = self._contact_count(
            (self._cube_token(object_name),),
            ("finger", "gripper0"),
        )
        lifted = bool(float(position[2]) > -0.070)
        near_gripper = bool(distance <= 0.135)
        elevated_contact = bool(lifted and contact_count > 0)
        held_by_contact = bool(elevated_contact and near_gripper)
        strict_held = bool(held_by_contact and distance <= 0.125 and opening < 0.98)
        held = bool(
            lifted
            and near_gripper
            and (held_by_contact or opening < 0.98)
        )
        return {
            "object_name": object_name,
            "held": held,
            "strict_held": strict_held,
            "held_by_contact": held_by_contact,
            "elevated_contact": elevated_contact,
            "lifted": lifted,
            "object_position": position,
            "ee_position": ee_position,
            "object_ee_distance": distance,
            "gripper_opening": opening,
            "gripper_contact_count": contact_count,
        }

    def get_stack_diagnostic(self) -> dict[str, Any]:
        """Return exact red-on-green alignment, contact, and benchmark status."""
        red, _red_quat, red_extent = self._cube_pose("red cube")
        green, _green_quat, green_extent = self._cube_pose("green cube")
        desired_red_z = float(green[2] + 0.5 * green_extent[2] + 0.5 * red_extent[2])
        xy_error = float(np.linalg.norm(red[:2] - green[:2]))
        z_error = float(red[2] - desired_red_z)
        cube_contact_count = self._contact_count(("cubea",), ("cubeb",))
        status = self.get_task_status()
        geometry_satisfied = bool(xy_error <= 0.035 and abs(z_error) <= 0.030)
        return {
            "red_position": red,
            "green_position": green,
            "desired_red_center_z": desired_red_z,
            "xy_error": xy_error,
            "z_error": z_error,
            "cube_contact_count": cube_contact_count,
            "geometry_satisfied": geometry_satisfied,
            "stacked": bool(status["task_completed"] or (geometry_satisfied and cube_contact_count > 0)),
            "task_status": status,
        }

    def goto_pose_checked(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        position_tolerance: float = 0.035,
    ) -> dict[str, Any]:
        """Execute a Cartesian move and return measured terminal pose errors."""
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        FrankaControlPrivilegedApi.goto_pose(
            self,
            target_position,
            target_quaternion,
            z_approach=float(z_approach),
        )
        actual_position, actual_quaternion = self.get_ee_pose()
        position_error = float(np.linalg.norm(actual_position - target_position))
        return {
            "success": bool(position_error <= float(position_tolerance)),
            "target_position": target_position,
            "actual_position": actual_position,
            "position_error": position_error,
            "orientation_error_rad": _quaternion_angle(actual_quaternion, target_quaternion),
        }

    def verified_pick_cube(
        self,
        object_name: str = "red cube",
        *,
        stage_attempt: int = 1,
        lift_height: float = 0.18,
    ) -> dict[str, Any]:
        """Pick a cube with a top-down grasp, lift it, and return hold evidence."""
        existing = self.measure_object_grasp_state(object_name)
        if existing["held"]:
            return self._skill_result(
                "verified_pick_cube",
                success=True,
                reason="cube already held at stage entry",
                metrics={"stage_attempt": int(stage_attempt), "lift_delta": 0.0},
                evidence={"after": existing},
            )

        before_position, _before_quat, _extent = self._cube_pose(object_name)
        pick_position, pick_quaternion = self.sample_grasp_pose(object_name)
        offsets = (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.006, 0.0, 0.0]),
            np.array([-0.006, 0.0, 0.0]),
        )
        offset = offsets[(max(1, int(stage_attempt)) - 1) % len(offsets)]
        pick_position = np.asarray(pick_position, dtype=np.float64).copy() + offset
        pick_quaternion = np.asarray(pick_quaternion, dtype=np.float64).copy()

        self.open_gripper()
        FrankaControlPrivilegedApi.goto_pose(
            self,
            pick_position,
            pick_quaternion,
            z_approach=0.10,
        )
        self.close_gripper()
        lift_position = pick_position + np.array([0.0, 0.0, float(lift_height)])
        FrankaControlPrivilegedApi.goto_pose(self, lift_position, pick_quaternion)

        after_position, _after_quat, _after_extent = self._cube_pose(object_name)
        after = self.measure_object_grasp_state(object_name)
        lift_delta = float(after_position[2] - before_position[2])
        success = bool(after["held"] and lift_delta >= 0.055)
        return self._skill_result(
            "verified_pick_cube",
            success=success,
            reason="cube lifted with verified gripper proximity" if success else "cube did not remain held after lift test",
            metrics={
                "stage_attempt": int(stage_attempt),
                "lift_delta": lift_delta,
                "object_ee_distance": after["object_ee_distance"],
                "contact_count": after["gripper_contact_count"],
            },
            evidence={"before_position": before_position, "after": after},
        )

    def transport_cube_above_target(
        self,
        object_name: str = "red cube",
        target_name: str = "green cube",
        *,
        clearance: float = 0.08,
    ) -> dict[str, Any]:
        """Move a verified-held cube above the target cube without releasing."""
        held_before = self.measure_object_grasp_state(object_name)
        if not held_before["held"]:
            return self._skill_result(
                "transport_cube_above_target",
                success=False,
                reason="source cube is not verified-held",
                metrics={"held_before": False},
                evidence={"grasp_state": held_before},
            )

        object_position, _object_quat, object_extent = self._cube_pose(object_name)
        target_position, _target_quat, target_extent = self._cube_pose(target_name)
        _ee_position, ee_quaternion = self.get_ee_pose()
        desired_object_position = target_position.copy()
        final_center_z = float(
            target_position[2] + 0.5 * target_extent[2] + 0.5 * object_extent[2]
        )
        desired_object_position[2] = max(
            final_center_z + float(clearance),
            float(object_position[2]),
        )
        FrankaControlPrivilegedApi.goto_pose(
            self,
            desired_object_position,
            ee_quaternion,
        )

        after_position, _after_quat, _after_extent = self._cube_pose(object_name)
        held_after = self.measure_object_grasp_state(object_name)
        xy_error = float(np.linalg.norm(after_position[:2] - target_position[:2]))
        vertical_clearance = float(
            after_position[2]
            - (target_position[2] + 0.5 * target_extent[2] + 0.5 * object_extent[2])
        )
        success = bool(held_after["held"] and xy_error <= 0.040 and vertical_clearance >= 0.035)
        return self._skill_result(
            "transport_cube_above_target",
            success=success,
            reason="held cube reached guarded pre-stack pose" if success else "guarded transport or hold verification failed",
            metrics={
                "xy_error": xy_error,
                "vertical_clearance": vertical_clearance,
                "held_before": held_before["held"],
                "held_after": held_after["held"],
            },
            evidence={
                "commanded_grasp_center": desired_object_position,
                "grasp_state": held_after,
            },
        )

    def release_and_verify_stack(
        self,
        object_name: str = "red cube",
        target_name: str = "green cube",
        *,
        settle_steps: int = 30,
    ) -> dict[str, Any]:
        """Lower a held cube onto the target, release, retreat, and verify stacking."""
        object_position, _object_quat, object_extent = self._cube_pose(object_name)
        target_position, _target_quat, target_extent = self._cube_pose(target_name)
        _ee_position, ee_quaternion = self.get_ee_pose()
        desired_object_position = target_position.copy()
        desired_object_position[2] = (
            target_position[2] + 0.5 * target_extent[2] + 0.5 * object_extent[2]
        )

        FrankaControlPrivilegedApi.goto_pose(
            self,
            desired_object_position,
            ee_quaternion,
            z_approach=0.10,
        )
        self.open_gripper()
        self._settle(settle_steps)
        retreat_position, retreat_quaternion = self.get_ee_pose()
        retreat_position = retreat_position + np.array([0.0, 0.0, 0.12])
        FrankaControlPrivilegedApi.goto_pose(self, retreat_position, retreat_quaternion)
        self._settle(8)

        diagnostic = self.get_stack_diagnostic()
        success = bool(diagnostic["stacked"])
        return self._skill_result(
            "release_and_verify_stack",
            success=success,
            reason="stack relation stable after release and retreat" if success else "released cubes do not satisfy stack geometry",
            metrics={
                "xy_error": diagnostic["xy_error"],
                "z_error": diagnostic["z_error"],
                "cube_contact_count": diagnostic["cube_contact_count"],
            },
            evidence={"stack_diagnostic": diagnostic},
            recoverable=not success,
        )


class FrankaRobosuiteCubeLiftStageApi(FrankaRobosuiteStackStageApi):
    """Privileged-readonly Stage-CAP API for RoboSuite cube lifting."""

    TASK_FAMILY = "cube_lift"
    OBJECT_NAMES = ("red cube",)

    def functions(self) -> dict[str, Any]:
        functions = FrankaControlPrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_ee_pose": self.get_ee_pose,
                "get_cube_lift_diagnostic": self.get_cube_lift_diagnostic,
                "goto_pose_checked": self.goto_pose_checked,
                "verified_pick_cube": self.verified_pick_cube,
            }
        )
        return functions

    def get_cube_lift_diagnostic(self) -> dict[str, Any]:
        """Return exact red-cube height, grasp evidence, and benchmark state."""
        position, _quaternion, extent = self._cube_pose("red cube")
        grasp_state = self.measure_object_grasp_state("red cube")
        status = self.get_task_status()
        return {
            "red_position": position,
            "red_extent": extent,
            "held": grasp_state["held"],
            "lifted": grasp_state["lifted"],
            "gripper_contact_count": grasp_state["gripper_contact_count"],
            "task_status": status,
        }


class FrankaRobosuiteRestackStageApi(FrankaRobosuiteStackStageApi):
    """Stage-CAP API for unstacking green before stacking red on green."""

    TASK_FAMILY = "restack"

    def functions(self) -> dict[str, Any]:
        functions = super().functions()
        functions.update(
            {
                "get_restack_diagnostic": self.get_restack_diagnostic,
                "relocate_covering_cube_and_verify": self.relocate_covering_cube_and_verify,
                "recover_and_verify_restack": self.recover_and_verify_restack,
            }
        )
        return functions

    def get_restack_diagnostic(self) -> dict[str, Any]:
        """Return initial occlusion, table separation, and final stack evidence."""
        red, _red_quaternion, red_extent = self._cube_pose("red cube")
        green, _green_quaternion, green_extent = self._cube_pose("green cube")
        xy_distance = float(np.linalg.norm(red[:2] - green[:2]))
        green_on_red_height = float(
            red[2] + 0.5 * red_extent[2] + 0.5 * green_extent[2]
        )
        green_on_red = bool(
            xy_distance <= 0.045
            and abs(float(green[2] - green_on_red_height)) <= 0.040
        )
        same_table_height = bool(abs(float(green[2] - red[2])) <= 0.040)
        ready_for_red_pick = bool(
            not green_on_red and same_table_height and xy_distance >= 0.090
        )
        return {
            "red_position": red,
            "green_position": green,
            "red_green_xy_distance": xy_distance,
            "green_on_red": green_on_red,
            "same_table_height": same_table_height,
            "ready_for_red_pick": ready_for_red_pick,
            "final_stack": self.get_stack_diagnostic(),
            "task_status": self.get_task_status(),
        }

    def _restack_relocation_target(self, red_position: np.ndarray) -> np.ndarray:
        low_level = self._env
        table_center = np.asarray(red_position, dtype=np.float64).copy()
        half_size = np.array([0.36, 0.36], dtype=np.float64)
        try:
            base = np.asarray(low_level.base_link_wxyz_xyz, dtype=np.float64)
            table_world = np.asarray(
                low_level.robosuite_env.table_offset,
                dtype=np.float64,
            )
            world_to_base = np.linalg.inv(_pose_matrix(base[4:7], base[:4]))
            table_center = (world_to_base @ np.r_[table_world, 1.0])[:3]
            half_size = 0.5 * np.asarray(
                low_level.robosuite_env.table_full_size[:2],
                dtype=np.float64,
            )
        except Exception:
            pass

        candidates = []
        for delta in (
            np.array([-0.17, 0.0]),
            np.array([0.17, 0.0]),
            np.array([0.0, -0.16]),
            np.array([0.0, 0.16]),
        ):
            xy = np.asarray(red_position[:2], dtype=np.float64) + delta
            lower = table_center[:2] - half_size + 0.07
            upper = table_center[:2] + half_size - 0.07
            if np.all(xy >= lower) and np.all(xy <= upper):
                candidates.append(xy)
        if not candidates:
            candidates.append(
                np.clip(
                    np.asarray(red_position[:2]) + np.array([-0.14, 0.0]),
                    table_center[:2] - half_size + 0.07,
                    table_center[:2] + half_size - 0.07,
                )
            )
        target_xy = max(
            candidates,
            key=lambda xy: float(np.linalg.norm(xy - red_position[:2])),
        )
        return np.array([target_xy[0], target_xy[1], red_position[2]], dtype=np.float64)

    def relocate_covering_cube_and_verify(
        self,
        *,
        stage_attempt: int = 1,
        settle_steps: int = 24,
    ) -> dict[str, Any]:
        """Move the initially covering green cube to a stable table location."""
        before = self.get_restack_diagnostic()
        if before["ready_for_red_pick"]:
            return self._skill_result(
                "relocate_covering_cube_and_verify",
                success=True,
                reason="green cube is already separated on the table",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )

        pick = self.verified_pick_cube(
            "green cube",
            stage_attempt=stage_attempt,
            lift_height=0.14,
        )
        if not pick["success"]:
            return self._skill_result(
                "relocate_covering_cube_and_verify",
                success=False,
                reason="covering green cube could not be lifted",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before, "pick": pick},
            )

        red_position, _red_quaternion, _red_extent = self._cube_pose("red cube")
        target = self._restack_relocation_target(red_position)
        _ee_position, ee_quaternion = self.get_ee_pose()
        guarded = target + np.array([0.0, 0.0, 0.11])
        FrankaControlPrivilegedApi.goto_pose(self, guarded, ee_quaternion)
        FrankaControlPrivilegedApi.goto_pose(
            self,
            target,
            ee_quaternion,
            z_approach=0.08,
        )
        self.open_gripper()
        self._settle(settle_steps)
        retreat, retreat_quaternion = self.get_ee_pose()
        FrankaControlPrivilegedApi.goto_pose(
            self,
            retreat + np.array([0.0, 0.0, 0.10]),
            retreat_quaternion,
        )
        self._settle(8)

        after = self.get_restack_diagnostic()
        success = bool(after["ready_for_red_pick"])
        return self._skill_result(
            "relocate_covering_cube_and_verify",
            success=success,
            reason="green cube relocated clear of red cube" if success else "green cube relocation is not stable or sufficiently separated",
            metrics={
                "stage_attempt": int(stage_attempt),
                "red_green_xy_distance": after["red_green_xy_distance"],
                "same_table_height": after["same_table_height"],
            },
            evidence={"before": before, "pick": pick, "target": target, "after": after},
            recoverable=not success,
        )

    def recover_and_verify_restack(
        self,
        *,
        stage_attempt: int = 1,
    ) -> dict[str, Any]:
        """Release the held red cube, or recover one failed final placement."""
        before = self.get_restack_diagnostic()
        if before["final_stack"]["stacked"]:
            return self._skill_result(
                "recover_and_verify_restack",
                success=True,
                reason="final red-on-green stack is already stable",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )

        recovery: dict[str, Any] = {}
        red_state = self.measure_object_grasp_state("red cube")
        if not red_state["held"]:
            pick = self.verified_pick_cube(
                "red cube",
                stage_attempt=stage_attempt,
            )
            recovery["pick"] = pick
            if not pick["success"]:
                return self._skill_result(
                    "recover_and_verify_restack",
                    success=False,
                    reason="dropped red cube could not be recovered",
                    metrics={"stage_attempt": int(stage_attempt)},
                    evidence={"before": before, "recovery": recovery},
                )
            transport = self.transport_cube_above_target("red cube", "green cube")
            recovery["transport"] = transport
            if not transport["success"]:
                return self._skill_result(
                    "recover_and_verify_restack",
                    success=False,
                    reason="recovered red cube did not reach the guarded stack pose",
                    metrics={"stage_attempt": int(stage_attempt)},
                    evidence={"before": before, "recovery": recovery},
                )

        release = FrankaRobosuiteStackStageApi.release_and_verify_stack(
            self,
            "red cube",
            "green cube",
        )
        success = bool(release["success"])
        return self._skill_result(
            "recover_and_verify_restack",
            success=success,
            reason="final restack verified after low release" if success else "red-on-green relation remains unsatisfied after bounded recovery",
            metrics={
                "stage_attempt": int(stage_attempt),
                "recovery_used": bool(recovery),
            },
            evidence={"before": before, "recovery": recovery, "release": release},
            recoverable=not success,
        )


class FrankaRobosuiteNutAssemblyStageApi(
    _RobosuitePrivilegedStageMixin,
    FrankaControlNutAssemblyPrivilegedApi,
):
    """Privileged-readonly Stage-CAP API for square-nut peg insertion."""

    TASK_FAMILY = "peg_insertion"
    OBJECT_NAMES = ("brown square nut", "brown square nut handle", "brown square block")

    def __init__(self, env: BaseEnv) -> None:
        FrankaControlNutAssemblyPrivilegedApi.__init__(self, env)

    def functions(self) -> dict[str, Any]:
        functions = FrankaControlNutAssemblyPrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_ee_pose": self.get_ee_pose,
                "get_nut_handle_to_center_offset": self.get_nut_handle_to_center_offset,
                "get_nut_peg_diagnostic": self.get_nut_peg_diagnostic,
                "goto_pose_checked": self.goto_pose_checked,
                "verified_grasp_nut_handle": self.verified_grasp_nut_handle,
                "move_nut_above_peg": self.move_nut_above_peg,
                "insert_nut_and_verify": self.insert_nut_and_verify,
            }
        )
        return functions

    def _nut_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self.get_object_pose(object_name)
        return (
            np.asarray(position, dtype=np.float64).copy(),
            np.asarray(quaternion, dtype=np.float64).copy(),
        )

    def get_all_object_poses(self) -> dict[str, dict[str, Any]]:
        """Return exact square-nut center, handle, and square-peg poses."""
        queries = {
            "brown square nut": "white hollow center of the brown square nut",
            "brown square nut handle": "extruded handle of the brown square nut",
            "brown square block": "square block",
        }
        result: dict[str, dict[str, Any]] = {}
        for name, query in queries.items():
            position, quaternion = self._nut_pose(query)
            result[name] = {"position": position, "quaternion_wxyz": quaternion}
        return result

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current gripper contact-center pose in robot-base coordinates."""
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_gripper_state(self) -> dict[str, Any]:
        """Return normalized gripper opening and current end-effector pose."""
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return {
            "opening": float(pose[-1]),
            "position": pose[:3].copy(),
            "quaternion_wxyz": pose[3:7].copy(),
        }

    def get_object_grasp_affordance(self, object_name: str) -> dict[str, Any]:
        """Return the exact handle grasp candidate for the square nut."""
        handle_position, handle_quaternion = self._nut_pose(
            "extruded handle of the brown square nut"
        )
        return {
            "object_name": object_name,
            "grasp_part": "brown square nut handle",
            "candidates": [
                {
                    "position": handle_position,
                    "quaternion_wxyz": self._canonical_handle_quaternion(
                        handle_position,
                        handle_quaternion,
                    ),
                    "approach": 0.05,
                }
            ],
        }

    def measure_object_grasp_state(self, object_name: str = "brown square nut") -> dict[str, Any]:
        """Measure nut lift, handle-to-gripper proximity, opening, and contact evidence."""
        nut_position, _nut_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        handle_position, _handle_quaternion = self._nut_pose(
            "extruded handle of the brown square nut"
        )
        ee_position, _ee_quaternion = self.get_ee_pose()
        opening = float(self.get_gripper_state()["opening"])
        handle_distance = float(np.linalg.norm(handle_position - ee_position))
        contact_count = self._contact_count(("squarenut",), ("finger", "gripper0"))
        lifted = bool(float(nut_position[2]) > -0.070)
        held = bool(lifted and handle_distance <= 0.145 and (contact_count > 0 or opening < 0.98))
        grasp_contact_stable = bool(
            handle_distance <= 0.145
            and contact_count >= 4
            and 0.28 <= opening <= 0.60
        )
        transport_stable_held = bool(
            lifted
            and grasp_contact_stable
        )
        return {
            "object_name": object_name,
            "held": held,
            "grasp_contact_stable": grasp_contact_stable,
            "transport_stable_held": transport_stable_held,
            "lifted": lifted,
            "nut_position": nut_position,
            "handle_position": handle_position,
            "ee_position": ee_position,
            "handle_ee_distance": handle_distance,
            "gripper_opening": opening,
            "gripper_contact_count": contact_count,
        }

    def get_nut_peg_diagnostic(self) -> dict[str, Any]:
        """Return exact square-nut to peg alignment, contact, and benchmark state."""
        nut_position, _nut_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        peg_position, _peg_quaternion = self._nut_pose("square block")
        xy_error = float(np.linalg.norm(nut_position[:2] - peg_position[:2]))
        z_offset = float(nut_position[2] - peg_position[2])
        contact_count = self._contact_count(("squarenut",), ("peg1",))
        status = self.get_task_status()
        return {
            "nut_position": nut_position,
            "peg_position": peg_position,
            "xy_error": xy_error,
            "z_offset": z_offset,
            "nut_peg_contact_count": contact_count,
            "aligned": bool(xy_error <= 0.025),
            "inserted": bool(status["task_completed"]),
            "task_status": status,
        }

    def goto_pose_checked(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        position_tolerance: float = 0.040,
    ) -> dict[str, Any]:
        """Execute a Cartesian move and return measured terminal pose errors."""
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        FrankaControlNutAssemblyPrivilegedApi.goto_pose(
            self,
            target_position,
            target_quaternion,
            z_approach=float(z_approach),
        )
        actual_position, actual_quaternion = self.get_ee_pose()
        position_error = float(np.linalg.norm(actual_position - target_position))
        return {
            "success": bool(position_error <= float(position_tolerance)),
            "target_position": target_position,
            "actual_position": actual_position,
            "position_error": position_error,
            "orientation_error_rad": _quaternion_angle(actual_quaternion, target_quaternion),
        }

    def _canonical_handle_quaternion(
        self,
        handle_position: np.ndarray,
        handle_quaternion: np.ndarray,
    ) -> np.ndarray:
        nut_position, _nut_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        quaternion = np.asarray(handle_quaternion, dtype=np.float64).reshape(4).copy()
        rotation = SciRotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        ).as_matrix()
        if float((nut_position - handle_position) @ rotation[:, 0]) < 0.0:
            rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
            xyzw = SciRotation.from_matrix(rotation).as_quat()
            quaternion = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return quaternion

    def _desired_handle_pose_for_peg(self) -> tuple[np.ndarray, np.ndarray]:
        handle_position, handle_quaternion = self._nut_pose(
            "extruded handle of the brown square nut"
        )
        nut_position, _nut_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        peg_position, peg_quaternion = self._nut_pose("square block")
        handle_quaternion = self._canonical_handle_quaternion(
            handle_position,
            handle_quaternion,
        )
        handle_to_center = np.linalg.inv(
            _pose_matrix(nut_position, handle_quaternion)
        ) @ _pose_matrix(handle_position, handle_quaternion)
        desired_handle = _pose_matrix(peg_position, peg_quaternion) @ handle_to_center
        return _matrix_pose(desired_handle)

    def _command_nut_gripper(self, opening_fraction: float, *, steps: int) -> None:
        self._env._set_gripper(float(opening_fraction))
        self._settle(steps)

    def verified_grasp_nut_handle(
        self,
        *,
        stage_attempt: int = 1,
    ) -> dict[str, Any]:
        """Grasp the square nut handle, lift vertically, and verify a low-drift hold."""
        existing = self.measure_object_grasp_state()
        if existing["transport_stable_held"]:
            return self._skill_result(
                "verified_grasp_nut_handle",
                success=True,
                reason="nut already held at stage entry",
                metrics={"stage_attempt": int(stage_attempt), "lift_delta": 0.0},
                evidence={"after": existing},
            )

        before_position, _before_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        ordered_scales = [0.004]
        candidate_attempts: list[dict[str, Any]] = []
        selected_close_state: dict[str, Any] | None = None
        selected_close_position: np.ndarray | None = None
        for candidate_index, radial_scale in enumerate(ordered_scales, start=1):
            handle_position, handle_quaternion = self.sample_grasp_pose(
                "extruded handle of the brown square nut"
            )
            handle_position = np.asarray(handle_position, dtype=np.float64).copy()
            nut_position, _nut_quaternion = self._nut_pose(
                "white hollow center of the brown square nut"
            )
            radial_direction = handle_position - np.asarray(nut_position, dtype=np.float64)
            radial_norm = float(np.linalg.norm(radial_direction))
            if radial_norm > 1e-6:
                radial_direction /= radial_norm
            else:
                radial_direction = np.zeros(3, dtype=np.float64)
            candidate_offset = radial_direction * float(radial_scale)
            handle_quaternion = self._canonical_handle_quaternion(
                handle_position,
                np.asarray(handle_quaternion, dtype=np.float64),
            )
            handle_position += candidate_offset

            if float(self.get_gripper_state()["opening"]) < 0.90:
                self._command_nut_gripper(1.0, steps=20)
            candidate_before_position, _candidate_before_quaternion = self._nut_pose(
                "white hollow center of the brown square nut"
            )
            FrankaControlNutAssemblyPrivilegedApi.goto_pose(
                self,
                handle_position,
                handle_quaternion,
                z_approach=0.10,
            )

            preclose_position, _preclose_quaternion = self._nut_pose(
                "white hollow center of the brown square nut"
            )
            approach_displacement = float(
                np.linalg.norm(preclose_position - candidate_before_position)
            )
            retargeted = False
            if approach_displacement > 0.020:
                rejected_state = self.measure_object_grasp_state()
                candidate_attempts.append(
                    {
                        "candidate_index": candidate_index,
                        "radial_scale": float(radial_scale),
                        "candidate_offset": candidate_offset,
                        "grasp_position": handle_position,
                        "approach_displacement": approach_displacement,
                        "close_displacement": 0.0,
                        "total_displacement": approach_displacement,
                        "retargeted_after_nudge": False,
                        "rejected_before_close": True,
                        "close_state": rejected_state,
                    }
                )
                if candidate_index == 1:
                    ordered_scales.append(0.0)
                continue
            if approach_displacement > 0.004:
                # The open fingers nudged the nut. Re-observe and approach the moved
                # handle once before closing instead of squeezing at a stale pose.
                current_handle, current_handle_quaternion = self._nut_pose(
                    "extruded handle of the brown square nut"
                )
                current_handle_quaternion = self._canonical_handle_quaternion(
                    current_handle,
                    current_handle_quaternion,
                )
                FrankaControlNutAssemblyPrivilegedApi.goto_pose(
                    self,
                    current_handle + candidate_offset,
                    current_handle_quaternion,
                )
                handle_position = current_handle + candidate_offset
                handle_quaternion = current_handle_quaternion
                preclose_position, _preclose_quaternion = self._nut_pose(
                    "white hollow center of the brown square nut"
                )
                retargeted = True

            self._command_nut_gripper(0.0, steps=35)
            postclose_position, _postclose_quaternion = self._nut_pose(
                "white hollow center of the brown square nut"
            )
            close_displacement = float(
                np.linalg.norm(postclose_position - preclose_position)
            )
            total_displacement = float(
                np.linalg.norm(postclose_position - candidate_before_position)
            )
            close_state = self.measure_object_grasp_state()
            candidate_attempts.append(
                {
                    "candidate_index": candidate_index,
                    "radial_scale": float(radial_scale),
                    "candidate_offset": candidate_offset,
                    "grasp_position": handle_position,
                    "approach_displacement": approach_displacement,
                    "close_displacement": close_displacement,
                    "total_displacement": total_displacement,
                    "retargeted_after_nudge": retargeted,
                    "close_state": close_state,
                }
            )
            if (
                close_state["grasp_contact_stable"]
                and close_displacement <= 0.018
            ):
                selected_close_state = close_state
                selected_close_position = postclose_position
                break
            if candidate_index == 1:
                opening = float(close_state["gripper_opening"])
                ordered_scales.append(0.003 if opening > 0.80 else 0.0)

        if selected_close_state is None:
            after_position, _after_quaternion = self._nut_pose(
                "white hollow center of the brown square nut"
            )
            after = self.measure_object_grasp_state()
            return self._skill_result(
                "verified_grasp_nut_handle",
                success=False,
                reason="no local handle candidate produced stable contact evidence",
                metrics={
                    "stage_attempt": int(stage_attempt),
                    "lift_delta": float(after_position[2] - before_position[2]),
                    "candidate_count": len(candidate_attempts),
                    "transport_stable_held": False,
                },
                evidence={"before_position": before_position, "attempts": candidate_attempts, "after": after},
            )

        ee_position, ee_quaternion = self.get_ee_pose()
        FrankaControlNutAssemblyPrivilegedApi.goto_pose(
            self,
            ee_position + np.array([0.0, 0.0, 0.10]),
            ee_quaternion,
        )

        after_position, _after_quaternion = self._nut_pose(
            "white hollow center of the brown square nut"
        )
        after = self.measure_object_grasp_state()
        lift_delta = float(after_position[2] - before_position[2])
        postclose_lift = float(
            after_position[2] - selected_close_position[2]
        ) if selected_close_position is not None else 0.0
        success = bool(
            after["transport_stable_held"]
            and lift_delta >= 0.040
            and postclose_lift >= 0.035
        )
        return self._skill_result(
            "verified_grasp_nut_handle",
            success=success,
            reason="nut handle grasp survived the vertical lift test" if success else "nut did not remain held after handle grasp",
            metrics={
                "stage_attempt": int(stage_attempt),
                "lift_delta": lift_delta,
                "postclose_lift": postclose_lift,
                "handle_ee_distance": after["handle_ee_distance"],
                "contact_count": after["gripper_contact_count"],
                "transport_stable_held": after["transport_stable_held"],
            },
            evidence={
                "before_position": before_position,
                "attempts": candidate_attempts,
                "selected_close_state": selected_close_state,
                "after": after,
            },
        )

    def move_nut_above_peg(self, *, clearance: float = 0.05) -> dict[str, Any]:
        """Move the held square nut to a guarded pose above the square peg."""
        held_before = self.measure_object_grasp_state()
        if not held_before["transport_stable_held"]:
            return self._skill_result(
                "move_nut_above_peg",
                success=False,
                reason="nut is not verified-held",
                metrics={"held_before": False},
                evidence={"grasp_state": held_before},
            )

        motions: list[dict[str, Any]] = []
        diagnostic = self.get_nut_peg_diagnostic()
        held_after = held_before
        for _correction in range(2):
            desired_position, desired_quaternion = self._desired_handle_pose_for_peg()
            guarded_position = desired_position + np.array([0.0, 0.0, float(clearance)])
            motion = self.goto_pose_checked(guarded_position, desired_quaternion)
            motions.append(motion)
            diagnostic = self.get_nut_peg_diagnostic()
            held_after = self.measure_object_grasp_state()
            if (
                held_after["transport_stable_held"]
                and diagnostic["xy_error"] <= 0.010
                and diagnostic["z_offset"] >= 0.020
            ):
                break
            if not held_after["transport_stable_held"]:
                break
        success = bool(
            held_after["transport_stable_held"]
            and diagnostic["xy_error"] <= 0.010
            and diagnostic["z_offset"] >= 0.020
        )
        return self._skill_result(
            "move_nut_above_peg",
            success=success,
            reason="held nut aligned above peg with clearance" if success else "nut alignment or hold verification failed",
            metrics={
                "xy_error": diagnostic["xy_error"],
                "z_offset": diagnostic["z_offset"],
                "held_after": held_after["held"],
                "transport_stable_held_after": held_after["transport_stable_held"],
                "motion_position_error": motions[-1]["position_error"],
                "correction_count": len(motions),
            },
            evidence={"motions": motions, "diagnostic": diagnostic, "grasp_state": held_after},
        )

    def insert_nut_and_verify(self, *, settle_steps: int = 30) -> dict[str, Any]:
        """Descend the square nut onto the peg, release, settle, and verify insertion."""
        desired_position, desired_quaternion = self._desired_handle_pose_for_peg()
        FrankaControlNutAssemblyPrivilegedApi.goto_pose(
            self,
            desired_position,
            desired_quaternion,
        )
        final_position = desired_position + np.array([0.0, 0.0, -0.02])
        FrankaControlNutAssemblyPrivilegedApi.goto_pose(
            self,
            final_position,
            desired_quaternion,
        )
        self.open_gripper()
        self._settle(settle_steps)

        diagnostic = self.get_nut_peg_diagnostic()
        success = bool(diagnostic["inserted"])
        return self._skill_result(
            "insert_nut_and_verify",
            success=success,
            reason="benchmark insertion verified after release" if success else "nut did not satisfy peg insertion after settling",
            metrics={
                "xy_error": diagnostic["xy_error"],
                "z_offset": diagnostic["z_offset"],
                "nut_peg_contact_count": diagnostic["nut_peg_contact_count"],
            },
            evidence={"diagnostic": diagnostic},
            recoverable=not success,
        )


class FrankaRobosuiteSpillWipeStageApi(
    _RobosuitePrivilegedStageMixin,
    FrankaControlSpillWipePrivilegedApi,
):
    """Privileged-readonly Stage-CAP API for the RoboSuite wipe task."""

    TASK_FAMILY = "wipe"
    OBJECT_NAMES = ("brown spill",)

    def __init__(self, env: BaseEnv) -> None:
        FrankaControlSpillWipePrivilegedApi.__init__(
            self,
            env,
            tcp_offset=[0.0, 0.0, -0.0158],
        )

    def functions(self) -> dict[str, Any]:
        functions = FrankaControlSpillWipePrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_ee_pose": self.get_ee_pose,
                "get_wipe_diagnostic": self.get_wipe_diagnostic,
                "approach_spill_surface": self.approach_spill_surface,
                "wipe_spill_and_verify": self.wipe_spill_and_verify,
            }
        )
        return functions

    def _marker_positions_robot_base(self, *, remaining_only: bool) -> np.ndarray:
        robosuite_env = self._env.robosuite_env
        markers = list(robosuite_env.model.mujoco_arena.markers)
        if remaining_only:
            wiped_ids = {id(marker) for marker in robosuite_env.wiped_markers}
            markers = [marker for marker in markers if id(marker) not in wiped_ids]
        if not markers:
            return np.empty((0, 3), dtype=np.float64)

        positions_world = np.stack(
            [
                np.asarray(
                    robosuite_env.sim.data.body_xpos[
                        robosuite_env.sim.model.body_name2id(marker.root_body)
                    ],
                    dtype=np.float64,
                )
                for marker in markers
            ]
        )
        base = np.asarray(self._env.base_link_wxyz_xyz, dtype=np.float64)
        world_to_base = np.linalg.inv(_pose_matrix(base[4:7], base[:4]))
        homogeneous = np.c_[positions_world, np.ones(len(positions_world))]
        return (world_to_base @ homogeneous.T).T[:, :3]

    def get_all_object_poses(self) -> dict[str, dict[str, Any]]:
        """Return the exact marker-cloud center and axis-aligned extent."""
        position, quaternion, extent = self.get_object_pose(
            "brown spill",
            return_bbox_extent=True,
        )
        return {
            "brown spill": {
                "position": np.asarray(position, dtype=np.float64),
                "quaternion_wxyz": np.asarray(quaternion, dtype=np.float64),
                "extent": np.asarray(extent, dtype=np.float64),
            }
        }

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(self.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_gripper_state(self) -> dict[str, Any]:
        position, quaternion = self.get_ee_pose()
        return {
            "position": position,
            "quaternion_wxyz": quaternion,
            "wiping_tool_attached": True,
        }

    def get_object_grasp_affordance(self, object_name: str) -> dict[str, Any]:
        return {
            "object_name": object_name,
            "graspable": False,
            "tool_attached": True,
            "candidates": [],
        }

    def measure_object_grasp_state(self, object_name: str) -> dict[str, Any]:
        return {
            "object_name": object_name,
            "held": False,
            "graspable": False,
            "tool_attached": True,
        }

    def get_wipe_diagnostic(self) -> dict[str, Any]:
        """Return exact remaining marker geometry and official wipe progress."""
        status = self.get_task_status()
        robosuite_env = self._env.robosuite_env
        remaining = self._marker_positions_robot_base(remaining_only=True)
        if len(remaining):
            centroid = remaining.mean(axis=0)
            bounds_min = remaining.min(axis=0)
            bounds_max = remaining.max(axis=0)
        else:
            centroid = np.zeros(3, dtype=np.float64)
            bounds_min = np.zeros(3, dtype=np.float64)
            bounds_max = np.zeros(3, dtype=np.float64)
        total = int(robosuite_env.num_markers)
        wiped = int(len(robosuite_env.wiped_markers))
        return {
            "total_markers": total,
            "wiped_markers": wiped,
            "remaining_markers": total - wiped,
            "wiped_fraction": float(wiped / max(total, 1)),
            "remaining_centroid": centroid,
            "remaining_bounds_min": bounds_min,
            "remaining_bounds_max": bounds_max,
            "task_status": status,
        }

    def approach_spill_surface(self, *, stage_attempt: int = 1) -> dict[str, Any]:
        """Approach an exact remaining marker and establish wiping contact."""
        before = self.get_wipe_diagnostic()
        if before["task_status"]["task_completed"]:
            return self._skill_result(
                "approach_spill_surface",
                success=True,
                reason="spill is already fully wiped",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )

        remaining = self._marker_positions_robot_base(remaining_only=True)
        target = remaining[0].copy()
        down_quaternion = np.array([0.0, 0.0, 1.0, 0.0])
        FrankaControlSpillWipePrivilegedApi.goto_pose(
            self,
            target + np.array([0.0, 0.0, 0.08]),
            down_quaternion,
        )
        FrankaControlSpillWipePrivilegedApi.goto_pose(
            self,
            target,
            down_quaternion,
        )
        self._settle(5)
        after = self.get_wipe_diagnostic()
        ee_position, _ee_quaternion = self.get_ee_pose()
        xy_error = float(np.linalg.norm(ee_position[:2] - target[:2]))
        contact_count = len(self.get_contact_diagnostics("gripper0"))
        success = bool(
            after["wiped_markers"] > before["wiped_markers"]
            or (xy_error <= 0.045 and contact_count > 0)
        )
        return self._skill_result(
            "approach_spill_surface",
            success=success,
            reason="wiping tool reached the spill surface" if success else "wiping contact was not established",
            metrics={
                "stage_attempt": int(stage_attempt),
                "xy_error": xy_error,
                "contact_count": contact_count,
                "newly_wiped": after["wiped_markers"] - before["wiped_markers"],
            },
            evidence={"before": before, "target": target, "after": after},
        )

    @staticmethod
    def _serpentine_wipe_path(points: np.ndarray, spacing: float) -> list[np.ndarray]:
        bounds_min = points[:, :2].min(axis=0) - 0.015
        bounds_max = points[:, :2].max(axis=0) + 0.015
        surface_z = float(np.median(points[:, 2]))
        path: list[np.ndarray] = []
        if bounds_max[0] - bounds_min[0] >= bounds_max[1] - bounds_min[1]:
            passes = np.arange(bounds_min[1], bounds_max[1] + 0.5 * spacing, spacing)
            for index, y in enumerate(passes):
                xs = (bounds_min[0], bounds_max[0]) if index % 2 == 0 else (bounds_max[0], bounds_min[0])
                path.extend(
                    [np.array([xs[0], y, surface_z]), np.array([xs[1], y, surface_z])]
                )
        else:
            passes = np.arange(bounds_min[0], bounds_max[0] + 0.5 * spacing, spacing)
            for index, x in enumerate(passes):
                ys = (bounds_min[1], bounds_max[1]) if index % 2 == 0 else (bounds_max[1], bounds_min[1])
                path.extend(
                    [np.array([x, ys[0], surface_z]), np.array([x, ys[1], surface_z])]
                )
        return path

    @staticmethod
    def _cluster_marker_targets(points: np.ndarray, radius: float) -> list[np.ndarray]:
        """Choose marker medoids whose tool footprints cover the marker cloud."""
        if not len(points):
            return []
        active = np.ones(len(points), dtype=bool)
        targets: list[np.ndarray] = []
        while np.any(active):
            active_indices = np.flatnonzero(active)
            active_points = points[active_indices, :2]
            pairwise = np.linalg.norm(
                active_points[:, None, :] - active_points[None, :, :],
                axis=2,
            )
            counts = np.sum(pairwise <= float(radius), axis=1)
            medoid_index = int(active_indices[int(np.argmax(counts))])
            target = points[medoid_index].copy()
            targets.append(target)
            distances = np.linalg.norm(points[:, :2] - target[:2], axis=1)
            active[distances <= float(radius)] = False
        return targets

    @staticmethod
    def _nearest_neighbor_path(
        points: list[np.ndarray],
        start_xy: np.ndarray,
    ) -> list[np.ndarray]:
        remaining = [np.asarray(point, dtype=np.float64).copy() for point in points]
        ordered: list[np.ndarray] = []
        cursor = np.asarray(start_xy, dtype=np.float64).reshape(2)
        while remaining:
            index = min(
                range(len(remaining)),
                key=lambda item: float(np.linalg.norm(remaining[item][:2] - cursor)),
            )
            waypoint = remaining.pop(index)
            ordered.append(waypoint)
            cursor = waypoint[:2]
        return ordered

    def wipe_spill_and_verify(self, *, stage_attempt: int = 1) -> dict[str, Any]:
        """Visit clustered remaining-marker targets and verify official completion."""
        before = self.get_wipe_diagnostic()
        if before["task_status"]["task_completed"]:
            return self._skill_result(
                "wipe_spill_and_verify",
                success=True,
                reason="spill is already fully wiped",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )

        remaining = self._marker_positions_robot_base(remaining_only=True)
        attempt = max(1, int(stage_attempt))
        cluster_radius = 0.014 if attempt == 1 else 0.008 if attempt == 2 else 1e-6
        clustered_targets = self._cluster_marker_targets(remaining, cluster_radius)
        ee_position, _ee_quaternion = self.get_ee_pose()
        path = self._nearest_neighbor_path(clustered_targets, ee_position[:2])
        down_quaternion = np.array([0.0, 0.0, 1.0, 0.0])
        if path:
            first = path[0]
            if float(np.linalg.norm(ee_position[:2] - first[:2])) > 0.06:
                FrankaControlSpillWipePrivilegedApi.goto_pose(
                    self,
                    first + np.array([0.0, 0.0, 0.06]),
                    down_quaternion,
                )
            FrankaControlSpillWipePrivilegedApi.goto_pose(self, first, down_quaternion)
            for waypoint in path[1:]:
                FrankaControlSpillWipePrivilegedApi.goto_pose(
                    self,
                    waypoint,
                    down_quaternion,
                )
        self._settle(6)
        after = self.get_wipe_diagnostic()
        if path:
            FrankaControlSpillWipePrivilegedApi.goto_pose(
                self,
                path[-1] + np.array([0.0, 0.0, 0.08]),
                down_quaternion,
            )
        success = bool(after["task_status"]["task_completed"])
        return self._skill_result(
            "wipe_spill_and_verify",
            success=success,
            reason="all spill markers were wiped" if success else "some spill markers remain after the bounded marker path",
            metrics={
                "stage_attempt": int(stage_attempt),
                "path_waypoints": len(path),
                "cluster_radius": cluster_radius,
                "wiped_before": before["wiped_markers"],
                "wiped_after": after["wiped_markers"],
                "remaining_markers": after["remaining_markers"],
            },
            evidence={"before": before, "after": after},
            recoverable=not success,
        )


class FrankaRobosuiteHandoverStageApi(
    _RobosuitePrivilegedStageMixin,
    FrankaHandoverPrivilegedApi,
):
    """Privileged-readonly Stage-CAP API for two-arm hammer handover."""

    TASK_FAMILY = "bimanual_handover"
    OBJECT_NAMES = ("hammer", "hammer handle")

    def __init__(self, env: BaseEnv) -> None:
        FrankaHandoverPrivilegedApi.__init__(self, env)
        self._guarded_handover_position: np.ndarray | None = None

    def functions(self) -> dict[str, Any]:
        functions = FrankaHandoverPrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_object_pose": self.get_object_pose,
                "get_arm0_gripper_pose": self.get_arm0_gripper_pose,
                "get_arm1_gripper_pose": self.get_arm1_gripper_pose,
                "get_hammer_handle_geometry": self.get_hammer_handle_geometry,
                "get_handover_diagnostic": self.get_handover_diagnostic,
                "verified_pick_hammer_arm0": self.verified_pick_hammer_arm0,
                "present_hammer_for_handover": self.present_hammer_for_handover,
                "capture_hammer_arm1": self.capture_hammer_arm1,
                "release_arm0_and_verify_handover": self.release_arm0_and_verify_handover,
            }
        )
        return functions

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        poses = self.get_observation()["hammer_poses"]
        key = "handle" if "handle" in str(object_name).lower() else "hammer"
        pose = np.asarray(poses[key], dtype=np.float64)
        if return_bbox_extent:
            extent = np.array([0.25, 0.12, 0.06]) if key == "hammer" else np.array([0.20, 0.03, 0.03])
            return pose[:3].copy(), pose[3:7].copy(), extent
        return pose[:3].copy(), pose[3:7].copy()

    def get_all_object_poses(self) -> dict[str, dict[str, Any]]:
        result = {}
        for name in self.OBJECT_NAMES:
            position, quaternion = self.get_object_pose(name)
            result[name] = {
                "position": position,
                "quaternion_wxyz": quaternion,
            }
        return result

    def get_gripper_state(self) -> dict[str, Any]:
        observation = self.get_observation()
        arm0 = np.asarray(observation["robot0_cartesian_pos"], dtype=np.float64)
        arm1 = np.asarray(observation["robot1_cartesian_pos"], dtype=np.float64)
        return {
            "arm0": {
                "position": arm0[:3].copy(),
                "quaternion_wxyz": arm0[3:7].copy(),
                "opening": float(arm0[-1]),
            },
            "arm1": {
                "position": arm1[:3].copy(),
                "quaternion_wxyz": arm1[3:7].copy(),
                "opening": float(arm1[-1]),
            },
        }

    def get_hammer_handle_geometry(self) -> dict[str, Any]:
        """Return the exact handle axis and half-length, oriented toward arm 1."""
        sim = self._env.robosuite_env.sim
        geom_id = int(sim.model.geom_name2id("hammer_handle"))
        axis_world = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)[:, 2]
        base = np.asarray(self._env.base_link_wxyz_xyz_0, dtype=np.float64)
        base_rotation_world = _pose_matrix(base[4:7], base[:4])[:3, :3]
        axis_robot0 = base_rotation_world.T @ axis_world
        axis_robot0 /= max(float(np.linalg.norm(axis_robot0)), 1e-9)
        handle_position, _handle_quaternion = self.get_object_pose("hammer handle")
        arm1_position, _arm1_quaternion = self.get_arm1_gripper_pose()
        if float(axis_robot0 @ (arm1_position - handle_position)) < 0.0:
            axis_robot0 *= -1.0
        return {
            "position": handle_position,
            "axis_toward_arm1": axis_robot0,
            "half_length": float(sim.model.geom_size[geom_id][2]),
            "half_width": float(max(sim.model.geom_size[geom_id][0:2])),
        }

    def get_object_grasp_affordance(self, object_name: str) -> dict[str, Any]:
        handle_position, _handle_quaternion = self.get_object_pose("hammer handle")
        geometry = self.get_hammer_handle_geometry()
        arm1_offset = max(
            0.035,
            min(0.095, 0.72 * float(geometry["half_length"])),
        )
        return {
            "object_name": object_name,
            "grasp_part": "hammer handle",
            "candidates": [
                {
                    "position": handle_position + np.array([0.0, -0.035, -0.025]),
                    "quaternion_wxyz": np.array([0.0, 0.70710678, 0.70710678, 0.0]),
                    "arm": 0,
                },
                {
                    "position": handle_position
                    + np.asarray(geometry["axis_toward_arm1"]) * arm1_offset
                    + np.array([0.0, 0.0, -0.01]),
                    "quaternion_wxyz": np.array([0.0, 0.0, 1.0, 0.0]),
                    "arm": 1,
                },
            ],
        }

    def measure_object_grasp_state(self, object_name: str = "hammer") -> dict[str, Any]:
        return self.get_handover_diagnostic()

    def get_handover_diagnostic(self) -> dict[str, Any]:
        """Return official grasp predicates plus exact hand/handle geometry."""
        env = self._env.robosuite_env
        arm0_grasp, arm1_handle_grasp, hammer_height, table_height = env._get_task_info()
        handle_position, _handle_quaternion = self.get_object_pose("hammer handle")
        arm0_position, _arm0_quaternion = self.get_arm0_gripper_pose()
        arm1_position, _arm1_quaternion = self.get_arm1_gripper_pose()
        state = self.get_gripper_state()
        status = self.get_task_status()
        return {
            "arm0_grasp_any": bool(arm0_grasp),
            "arm1_grasp_handle": bool(arm1_handle_grasp),
            "hammer_height": float(hammer_height),
            "table_height": float(table_height),
            "height_above_table": float(hammer_height - table_height),
            "handle_position": handle_position,
            "arm0_handle_distance": float(np.linalg.norm(arm0_position - handle_position)),
            "arm1_handle_distance": float(np.linalg.norm(arm1_position - handle_position)),
            "arm0_contact_count": self._contact_count(("hammer",), ("gripper0_right",)),
            "arm1_contact_count": self._contact_count(("hammer_handle",), ("gripper1_right",)),
            "arm0_opening": state["arm0"]["opening"],
            "arm1_opening": state["arm1"]["opening"],
            "task_status": status,
        }

    def verified_pick_hammer_arm0(self, *, stage_attempt: int = 1) -> dict[str, Any]:
        """Pick the exact handle with arm 0 and verify the official grasp predicate."""
        before = self.get_handover_diagnostic()
        if before["arm0_grasp_any"] and before["height_above_table"] > 0.06:
            return self._skill_result(
                "verified_pick_hammer_arm0",
                success=True,
                reason="arm 0 already holds the lifted hammer",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )

        arm0_home, _arm0_quaternion = self.get_arm0_gripper_pose()
        arm1_home, _arm1_quaternion = self.get_arm1_gripper_pose()
        if self._guarded_handover_position is None:
            self._guarded_handover_position = 0.5 * (arm0_home + arm1_home)
            self._guarded_handover_position[2] = 0.10

        handle_position, _handle_quaternion = self.get_object_pose("hammer handle")
        y_offsets = (-0.035, 0.0, 0.035)
        offset_y = y_offsets[(max(1, int(stage_attempt)) - 1) % len(y_offsets)]
        grasp_position = handle_position + np.array([0.0, offset_y, -0.025])
        grasp_quaternion = np.array([0.0, 0.70710678, 0.70710678, 0.0])
        self.open_gripper_arm0()
        self.goto_pose_arm0(grasp_position, grasp_quaternion, z_approach=0.15)
        self.close_gripper_arm0()
        self.goto_pose_arm0(
            grasp_position + np.array([0.0, 0.0, 0.14]),
            grasp_quaternion,
        )
        after = self.get_handover_diagnostic()
        success = bool(after["arm0_grasp_any"] and after["height_above_table"] > 0.06)
        return self._skill_result(
            "verified_pick_hammer_arm0",
            success=success,
            reason="arm 0 lifted the hammer with verified contact" if success else "arm 0 did not retain the hammer after lift",
            metrics={
                "stage_attempt": int(stage_attempt),
                "offset_y": offset_y,
                "height_above_table": after["height_above_table"],
                "arm0_contact_count": after["arm0_contact_count"],
            },
            evidence={"before": before, "grasp_position": grasp_position, "after": after},
        )

    def present_hammer_for_handover(self) -> dict[str, Any]:
        """Move arm 0 to the guarded midpoint while retaining the hammer."""
        before = self.get_handover_diagnostic()
        if not before["arm0_grasp_any"]:
            return self._skill_result(
                "present_hammer_for_handover",
                success=False,
                reason="arm 0 must hold the hammer before presentation",
                metrics={"arm0_grasp_any": False},
                evidence={"before": before},
            )
        if self._guarded_handover_position is None:
            arm0, _ = self.get_arm0_gripper_pose()
            arm1, _ = self.get_arm1_gripper_pose()
            self._guarded_handover_position = 0.5 * (arm0 + arm1)
            self._guarded_handover_position[2] = 0.10
        handover_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        self.goto_pose_arm0(self._guarded_handover_position, handover_quaternion)
        after = self.get_handover_diagnostic()
        success = bool(
            after["arm0_grasp_any"]
            and after["height_above_table"] > 0.10
            and abs(float(after["handle_position"][0] - self._guarded_handover_position[0])) <= 0.18
        )
        return self._skill_result(
            "present_hammer_for_handover",
            success=success,
            reason="hammer presented in guarded handover region" if success else "hammer presentation lost the grasp or missed the guarded region",
            metrics={
                "height_above_table": after["height_above_table"],
                "arm0_grasp_any": after["arm0_grasp_any"],
            },
            evidence={"before": before, "target": self._guarded_handover_position, "after": after},
        )

    def capture_hammer_arm1(self, *, stage_attempt: int = 1) -> dict[str, Any]:
        """Close arm 1 on a free handle segment while arm 0 keeps supporting it."""
        before = self.get_handover_diagnostic()
        if before["arm1_grasp_handle"]:
            return self._skill_result(
                "capture_hammer_arm1",
                success=True,
                reason="arm 1 already grasps the handle",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
            )
        if not before["arm0_grasp_any"]:
            return self._skill_result(
                "capture_hammer_arm1",
                success=False,
                reason="arm 0 is no longer supporting the hammer",
                metrics={"stage_attempt": int(stage_attempt)},
                evidence={"before": before},
                recoverable=False,
            )

        geometry = self.get_hammer_handle_geometry()
        handle_position = np.asarray(geometry["position"], dtype=np.float64)
        handle_axis = np.asarray(geometry["axis_toward_arm1"], dtype=np.float64)
        half_length = float(geometry["half_length"])
        fractions = (0.72, 0.58, 0.84)
        fraction = fractions[(max(1, int(stage_attempt)) - 1) % len(fractions)]
        handle_offset = max(0.035, min(0.095, fraction * half_length))
        capture_position = (
            handle_position
            + handle_axis * handle_offset
            + np.array([0.0, 0.0, -0.01])
        )
        capture_quaternion = np.array([0.0, 0.0, 1.0, 0.0])
        self.open_gripper_arm1()
        self.goto_pose_arm1(capture_position, capture_quaternion, z_approach=0.12)
        self.close_gripper_arm1()
        self._settle(8)
        after = self.get_handover_diagnostic()
        success = bool(after["arm1_grasp_handle"])
        return self._skill_result(
            "capture_hammer_arm1",
            success=success,
            reason="arm 1 handle grasp verified" if success else "arm 1 did not capture the handle",
            metrics={
                "stage_attempt": int(stage_attempt),
                "handle_half_length": half_length,
                "handle_axis": handle_axis,
                "handle_offset": handle_offset,
                "arm1_contact_count": after["arm1_contact_count"],
                "arm1_opening": after["arm1_opening"],
            },
            evidence={"before": before, "capture_position": capture_position, "after": after},
        )

    def release_arm0_and_verify_handover(self) -> dict[str, Any]:
        """Release arm 0, lift with arm 1, retreat, and verify official success."""
        before = self.get_handover_diagnostic()
        if not before["arm1_grasp_handle"]:
            return self._skill_result(
                "release_arm0_and_verify_handover",
                success=False,
                reason="arm 1 must verify its handle grasp before transfer",
                metrics={"arm1_grasp_handle": False},
                evidence={"before": before},
            )

        self.open_gripper_arm0()
        arm1_position, arm1_quaternion = self.get_arm1_gripper_pose()
        self.goto_pose_arm1(
            arm1_position + np.array([0.0, 0.0, 0.06]),
            arm1_quaternion,
        )
        arm0_position, arm0_quaternion = self.get_arm0_gripper_pose()
        self.goto_pose_arm0(
            arm0_position + np.array([-0.12, 0.0, 0.08]),
            arm0_quaternion,
        )
        self._settle(12)
        after = self.get_handover_diagnostic()
        success = bool(after["task_status"]["task_completed"])
        return self._skill_result(
            "release_arm0_and_verify_handover",
            success=success,
            reason="official two-arm handover predicate satisfied" if success else "hammer transfer did not satisfy the official predicate",
            metrics={
                "arm0_grasp_any": after["arm0_grasp_any"],
                "arm1_grasp_handle": after["arm1_grasp_handle"],
                "height_above_table": after["height_above_table"],
            },
            evidence={"before": before, "after": after},
            recoverable=bool(after["arm1_grasp_handle"] and not success),
        )


class FrankaRobosuiteTwoArmLiftStageApi(
    _RobosuitePrivilegedStageMixin,
    FrankaTwoArmLiftPrivilegedApi,
):
    """Privileged-readonly Stage-CAP API for coordinated two-arm pot lifting."""

    TASK_FAMILY = "bimanual_lift"
    OBJECT_NAMES = ("pot", "pot handle 0", "pot handle 1")

    def __init__(self, env: BaseEnv) -> None:
        FrankaTwoArmLiftPrivilegedApi.__init__(self, env)

    def functions(self) -> dict[str, Any]:
        functions = FrankaTwoArmLiftPrivilegedApi.functions(self)
        functions.update(self._common_functions())
        functions.update(
            {
                "get_object_pose": self.get_object_pose,
                "get_handle0_pose": self.get_handle0_pose,
                "get_handle1_pose": self.get_handle1_pose,
                "get_bimanual_lift_diagnostic": self.get_bimanual_lift_diagnostic,
                "verify_bimanual_handle_grasps": self.verify_bimanual_handle_grasps,
                "goto_pose_both_checked": self.goto_pose_both_checked,
                "prepare_bimanual_handle_grasps": self.prepare_bimanual_handle_grasps,
                "close_and_verify_bimanual_grasps": self.close_and_verify_bimanual_grasps,
                "synchronized_lift_and_verify": self.synchronized_lift_and_verify,
            }
        )
        return functions

    def get_handle0_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact handle-0 position and pot-frame quaternion in robot0 coordinates."""
        pose = np.asarray(self.get_observation()["pot_poses"]["handle0"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_handle1_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact handle-1 position and pot-frame quaternion in robot0 coordinates."""
        pose = np.asarray(self.get_observation()["pot_poses"]["handle1"], dtype=np.float64)
        return pose[:3].copy(), pose[3:7].copy()

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return exact pot or handle pose by semantic name."""
        name = str(object_name).lower().replace("_", " ")
        if "handle" in name and any(token in name for token in ("0", "zero", "left")):
            return self.get_handle0_pose()
        if "handle" in name and any(token in name for token in ("1", "one", "right")):
            return self.get_handle1_pose()
        if "pot" in name:
            position, quaternion = self.get_pot_pose()
            return (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(quaternion, dtype=np.float64).copy(),
            )
        raise ValueError(f"Unknown bimanual-lift object: {object_name!r}")

    def get_all_object_poses(self) -> dict[str, dict[str, Any]]:
        """Return exact pot and both handle poses."""
        result: dict[str, dict[str, Any]] = {}
        for name in self.OBJECT_NAMES:
            position, quaternion = self.get_object_pose(name)
            result[name] = {"position": position, "quaternion_wxyz": quaternion}
        return result

    @staticmethod
    def _handle_grasp_quaternion(handle_quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Match the privileged oracle's handle-frame 180-degree X rotation."""
        quaternion = np.asarray(handle_quaternion_wxyz, dtype=np.float64).reshape(4)
        handle_rotation = SciRotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        )
        grasp_rotation = handle_rotation * SciRotation.from_euler("x", np.pi)
        xyzw = grasp_rotation.as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    def get_gripper_state(self) -> dict[str, Any]:
        """Return both gripper poses and normalized openings in robot0 coordinates."""
        obs = self.get_observation()
        pose0 = np.asarray(obs["robot0_cartesian_pos"], dtype=np.float64)
        pose1 = np.asarray(obs["robot1_cartesian_pos"], dtype=np.float64)
        return {
            "arm0": {
                "position": pose0[:3].copy(),
                "quaternion_wxyz": pose0[3:7].copy(),
                "opening": float(pose0[-1]),
            },
            "arm1": {
                "position": pose1[:3].copy(),
                "quaternion_wxyz": pose1[3:7].copy(),
                "opening": float(pose1[-1]),
            },
        }

    def get_object_grasp_affordance(self, object_name: str) -> dict[str, Any]:
        """Return the assigned side-grasp candidate for a pot handle."""
        name = str(object_name).lower()
        if "handle" in name and any(token in name for token in ("0", "zero", "left")):
            position, handle_quaternion = self.get_handle0_pose()
            quaternion = self._handle_grasp_quaternion(handle_quaternion)
            arm = 0
        elif "handle" in name and any(token in name for token in ("1", "one", "right")):
            position, handle_quaternion = self.get_handle1_pose()
            quaternion = self._handle_grasp_quaternion(handle_quaternion)
            arm = 1
        else:
            position, quaternion = self.get_pot_pose()
            arm = None
        return {
            "object_name": object_name,
            "assigned_arm": arm,
            "candidates": [
                {
                    "position": np.asarray(position, dtype=np.float64).copy(),
                    "quaternion_wxyz": np.asarray(quaternion, dtype=np.float64).copy(),
                }
            ],
        }

    def measure_object_grasp_state(self, object_name: str) -> dict[str, Any]:
        """Return the corresponding handle grasp state, or both states for the pot."""
        diagnostic = self.get_bimanual_lift_diagnostic()
        name = str(object_name).lower()
        if "handle" in name and any(token in name for token in ("0", "zero", "left")):
            return {
                "object_name": object_name,
                "held": diagnostic["arm0_grasped"],
                "distance": diagnostic["arm0_handle_distance"],
                "contact_count": diagnostic["arm0_handle_contact_count"],
            }
        if "handle" in name and any(token in name for token in ("1", "one", "right")):
            return {
                "object_name": object_name,
                "held": diagnostic["arm1_grasped"],
                "distance": diagnostic["arm1_handle_distance"],
                "contact_count": diagnostic["arm1_handle_contact_count"],
            }
        return {
            "object_name": object_name,
            "held": diagnostic["both_grasped"],
            "arm0_grasped": diagnostic["arm0_grasped"],
            "arm1_grasped": diagnostic["arm1_grasped"],
        }

    def get_bimanual_lift_diagnostic(self) -> dict[str, Any]:
        """Return exact handle/gripper geometry, contacts, pot height, and task state."""
        handle0, _handle0_quaternion = self.get_handle0_pose()
        handle1, _handle1_quaternion = self.get_handle1_pose()
        gripper0, _gripper0_quaternion = self.get_arm0_gripper_pose()
        gripper1, _gripper1_quaternion = self.get_arm1_gripper_pose()
        pot_position, _pot_quaternion = self.get_pot_pose()
        gripper_state = self.get_gripper_state()
        distance0 = float(np.linalg.norm(handle0 - gripper0))
        distance1 = float(np.linalg.norm(handle1 - gripper1))
        tcp_reference_offset = np.array(
            [0.0, 0.0, abs(float(self._TCP_OFFSET[2]))], dtype=np.float64
        )
        alignment_error0 = float(
            np.linalg.norm(np.asarray(gripper0) - (handle0 + tcp_reference_offset))
        )
        alignment_error1 = float(
            np.linalg.norm(np.asarray(gripper1) - (handle1 + tcp_reference_offset))
        )
        contact0 = self._contact_count(("pot_handle0",), ("gripper0", "finger_joint1"))
        contact1 = self._contact_count(("pot_handle1",), ("gripper1", "finger_joint2"))
        opening0 = float(gripper_state["arm0"]["opening"])
        opening1 = float(gripper_state["arm1"]["opening"])
        grasp0 = bool(alignment_error0 <= 0.035 and contact0 > 0 and opening0 < 0.98)
        grasp1 = bool(alignment_error1 <= 0.035 and contact1 > 0 and opening1 < 0.98)
        status = self.get_task_status()
        return {
            "pot_position": np.asarray(pot_position, dtype=np.float64).copy(),
            "pot_height": float(pot_position[2]),
            "handle0_position": handle0,
            "handle1_position": handle1,
            "arm0_gripper_position": np.asarray(gripper0, dtype=np.float64).copy(),
            "arm1_gripper_position": np.asarray(gripper1, dtype=np.float64).copy(),
            "arm0_handle_distance": distance0,
            "arm1_handle_distance": distance1,
            "arm0_tcp_alignment_error": alignment_error0,
            "arm1_tcp_alignment_error": alignment_error1,
            "arm0_handle_contact_count": contact0,
            "arm1_handle_contact_count": contact1,
            "arm0_gripper_opening": opening0,
            "arm1_gripper_opening": opening1,
            "arm0_grasped": grasp0,
            "arm1_grasped": grasp1,
            "both_grasped": bool(grasp0 and grasp1),
            "lifted": bool(status["task_completed"] or float(pot_position[2]) >= 0.030),
            "task_status": status,
        }

    def verify_bimanual_handle_grasps(self) -> dict[str, Any]:
        """Return a read-only verdict for the simultaneous handle grasps."""
        return self.get_bimanual_lift_diagnostic()

    def goto_pose_both_checked(
        self,
        position0: np.ndarray,
        quaternion_wxyz0: np.ndarray,
        position1: np.ndarray,
        quaternion_wxyz1: np.ndarray,
        z_approach: float = 0.0,
        position_tolerance: float = 0.050,
    ) -> dict[str, Any]:
        """Move both arms together and return terminal position errors."""
        target0 = np.asarray(position0, dtype=np.float64).reshape(3)
        target1 = np.asarray(position1, dtype=np.float64).reshape(3)
        FrankaTwoArmLiftPrivilegedApi.goto_pose_both(
            self,
            target0,
            np.asarray(quaternion_wxyz0, dtype=np.float64).reshape(4),
            target1,
            np.asarray(quaternion_wxyz1, dtype=np.float64).reshape(4),
            z_approach=float(z_approach),
        )
        actual0, _actual_quaternion0 = self.get_arm0_gripper_pose()
        actual1, _actual_quaternion1 = self.get_arm1_gripper_pose()
        error0 = float(np.linalg.norm(np.asarray(actual0) - target0))
        error1 = float(np.linalg.norm(np.asarray(actual1) - target1))
        return {
            "success": bool(max(error0, error1) <= float(position_tolerance)),
            "arm0_position_error": error0,
            "arm1_position_error": error1,
            "target0": target0,
            "target1": target1,
            "actual0": np.asarray(actual0, dtype=np.float64).copy(),
            "actual1": np.asarray(actual1, dtype=np.float64).copy(),
        }

    def prepare_bimanual_handle_grasps(
        self,
        *,
        stage_attempt: int = 1,
    ) -> dict[str, Any]:
        """Open both grippers and place them at their assigned side-grasp poses."""
        handle0, handle0_quaternion = self.get_handle0_pose()
        handle1, handle1_quaternion = self.get_handle1_pose()
        grasp_quaternion0 = self._handle_grasp_quaternion(handle0_quaternion)
        grasp_quaternion1 = self._handle_grasp_quaternion(handle1_quaternion)
        self.open_gripper_arm0()
        self.open_gripper_arm1()
        FrankaTwoArmLiftPrivilegedApi.goto_pose_arm0(
            self,
            handle0,
            grasp_quaternion0,
            z_approach=0.20,
        )
        FrankaTwoArmLiftPrivilegedApi.goto_pose_arm1(
            self,
            handle1,
            grasp_quaternion1,
            z_approach=0.20,
        )
        FrankaTwoArmLiftPrivilegedApi.goto_pose_both(
            self,
            handle0,
            grasp_quaternion0,
            handle1,
            grasp_quaternion1,
        )

        diagnostic = self.get_bimanual_lift_diagnostic()
        ready = bool(
            diagnostic["arm0_tcp_alignment_error"] <= 0.025
            and diagnostic["arm1_tcp_alignment_error"] <= 0.025
            and diagnostic["arm0_gripper_opening"] >= 0.90
            and diagnostic["arm1_gripper_opening"] >= 0.90
        )
        return self._skill_result(
            "prepare_bimanual_handle_grasps",
            success=ready,
            reason="both grippers reached assigned handle pregrasps" if ready else "one or both handle pregrasps are out of tolerance",
            metrics={
                "stage_attempt": int(stage_attempt),
                "arm0_handle_distance": diagnostic["arm0_handle_distance"],
                "arm1_handle_distance": diagnostic["arm1_handle_distance"],
                "arm0_tcp_alignment_error": diagnostic["arm0_tcp_alignment_error"],
                "arm1_tcp_alignment_error": diagnostic["arm1_tcp_alignment_error"],
            },
            evidence={"diagnostic": diagnostic},
        )

    def close_and_verify_bimanual_grasps(self) -> dict[str, Any]:
        """Close both grippers on their assigned handles and verify both grasps."""
        self.close_gripper_arm0()
        self.close_gripper_arm1()
        self._settle(8)
        diagnostic = self.get_bimanual_lift_diagnostic()
        success = bool(diagnostic["both_grasped"])
        return self._skill_result(
            "close_and_verify_bimanual_grasps",
            success=success,
            reason="both handle grasps verified" if success else "only one or neither handle grasp is verified",
            metrics={
                "arm0_handle_distance": diagnostic["arm0_handle_distance"],
                "arm1_handle_distance": diagnostic["arm1_handle_distance"],
                "arm0_tcp_alignment_error": diagnostic["arm0_tcp_alignment_error"],
                "arm1_tcp_alignment_error": diagnostic["arm1_tcp_alignment_error"],
                "arm0_contact_count": diagnostic["arm0_handle_contact_count"],
                "arm1_contact_count": diagnostic["arm1_handle_contact_count"],
                "arm0_opening": diagnostic["arm0_gripper_opening"],
                "arm1_opening": diagnostic["arm1_gripper_opening"],
            },
            evidence={"diagnostic": diagnostic},
        )

    def synchronized_lift_and_verify(self, *, lift_height: float = 0.20) -> dict[str, Any]:
        """Lift both handles synchronously and verify pot displacement and task success."""
        before = self.get_bimanual_lift_diagnostic()
        if not before["both_grasped"]:
            return self._skill_result(
                "synchronized_lift_and_verify",
                success=False,
                reason="both handle grasps are required before synchronized lift",
                metrics={"both_grasped": False},
                evidence={"before": before},
            )

        handle0, handle0_quaternion = self.get_handle0_pose()
        handle1, handle1_quaternion = self.get_handle1_pose()
        target0 = handle0 + np.array([0.0, 0.0, float(lift_height)])
        target1 = handle1 + np.array([0.0, 0.0, float(lift_height)])
        motion = self.goto_pose_both_checked(
            target0,
            self._handle_grasp_quaternion(handle0_quaternion),
            target1,
            self._handle_grasp_quaternion(handle1_quaternion),
        )
        after = self.get_bimanual_lift_diagnostic()
        lift_delta = float(after["pot_height"] - before["pot_height"])
        success = bool(after["task_status"]["task_completed"] or lift_delta >= 0.080)
        return self._skill_result(
            "synchronized_lift_and_verify",
            success=success,
            reason="pot lift verified with coordinated handles" if success else "pot height did not increase enough",
            metrics={
                "lift_delta": lift_delta,
                "arm0_handle_distance": after["arm0_handle_distance"],
                "arm1_handle_distance": after["arm1_handle_distance"],
                "motion_arm0_error": motion["arm0_position_error"],
                "motion_arm1_error": motion["arm1_position_error"],
            },
            evidence={"before": before, "motion": motion, "after": after},
            recoverable=bool(after["both_grasped"] and not success),
        )


__all__ = [
    "FrankaRobosuiteCubeLiftStageApi",
    "FrankaRobosuiteHandoverStageApi",
    "FrankaRobosuiteStackStageApi",
    "FrankaRobosuiteRestackStageApi",
    "FrankaRobosuiteNutAssemblyStageApi",
    "FrankaRobosuiteSpillWipeStageApi",
    "FrankaRobosuiteTwoArmLiftStageApi",
]
