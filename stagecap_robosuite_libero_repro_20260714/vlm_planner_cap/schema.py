"""Shared data structures for the VLM planner-verifier loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class StageStatus(str, Enum):
    """Verifier decision for a planned stage."""

    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    NEEDS_REPAIR = "needs_repair"
    UNCERTAIN = "uncertain"


class TaskFamily(str, Enum):
    """Manipulation families distilled from the CAP-35 success corpus."""

    PICK_PLACE = "pick_place"
    CONTAINER_PLACE = "container_place"
    SURFACE_PLACE = "surface_place"
    SPATIAL_RELATION = "spatial_relation"
    DRAWER = "drawer"
    STOVE_BUTTON = "stove_button"
    RACK_OR_CADDY = "rack_or_caddy"
    MULTI_OBJECT = "multi_object"
    PUSH_REPAIR = "push_repair"
    CUBE_LIFT = "cube_lift"
    STACK = "stack"
    RESTACK = "restack"
    PEG_INSERTION = "peg_insertion"
    WIPE = "wipe"
    BIMANUAL_LIFT = "bimanual_lift"
    BIMANUAL_HANDOVER = "bimanual_handover"


class StageDecision(str, Enum):
    """Control decision after stage verification."""

    PROCEED = "proceed"
    RETRY_STAGE = "retry_stage"
    REPLAN_STAGE = "replan_stage"
    REPLAN_SUFFIX = "replan_suffix"
    FAIL_TASK = "fail_task"


@dataclass(frozen=True)
class CoordinateObservation:
    """Read-only geometric state exposed to the planner and verifier."""

    gripper_pose: dict[str, Any] | None = None
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    articulated: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class StageCheck:
    """A visual/geometric condition that determines stage completion."""

    name: str
    description: str
    check_type: Literal["visual", "geometry", "hybrid"]
    success_criteria: list[str]


@dataclass(frozen=True)
class StagePlan:
    """One step in a task-level manipulation state machine."""

    stage_id: str
    name: str
    intent: str
    allowed_primitives: list[str]
    target_references: list[str]
    checks: list[StageCheck]
    repair_hints: list[str] = field(default_factory=list)
    allowed_observations: list[str] = field(default_factory=list)
    family: TaskFamily | None = None
    max_attempts: int = 2


@dataclass(frozen=True)
class TaskPlan:
    """A complete stage-level plan for a manipulation task."""

    task: str
    stages: list[StagePlan]
    notes: list[str] = field(default_factory=list)
    families: list[TaskFamily] = field(default_factory=list)


@dataclass(frozen=True)
class CleanApiPolicy:
    """Policy boundary for generated CaP code."""

    allowed_primitives: tuple[str, ...]
    allowed_observations: tuple[str, ...]
    allowed_skills: tuple[str, ...]
    forbidden_symbols: tuple[str, ...]
    profile: str = "default"

    def find_forbidden_symbols(self, code: str) -> list[str]:
        """Return forbidden symbols present in generated code."""
        return [symbol for symbol in self.forbidden_symbols if symbol in code]


DEFAULT_CLEAN_API_POLICY = CleanApiPolicy(
    allowed_primitives=(
        "goto_pose",
        "goto_pose_checked",
        "goto_finger_midpoint_checked",
        "open_gripper",
        "close_gripper",
        "move_linear_delta",
        "move_along_direction",
        "move_finger_midpoint_delta",
        "move_finger_midpoint_delta_strict",
        "move_joint_delta_checked",
        "move_to_joints",
    ),
    allowed_observations=(
        "get_observation",
        "get_ee_pose",
        "get_object_pose",
        "get_instance_pose",
        "get_all_object_poses",
        "get_instance_object_names",
        "get_site_pose",
        "get_scene_site_names",
        "get_scene_body_names",
        "get_scene_geom_names",
        "get_scene_joint_names",
        "get_body_pose",
        "get_geom_pose",
        "get_geom_diagnostics",
        "get_joint_diagnostics",
        "get_contact_diagnostics",
        "get_task_status",
        "get_bddl_goal_diagnostics",
        "get_on_diagnostic",
        "get_site_on_diagnostic",
        "get_site_in_diagnostic",
        "get_on_center_delta_robot",
        "get_object_state_diagnostic",
        "get_goal_target_region_pose",
        "sample_grasp_pose",
        "get_object_grasp_affordance",
        "measure_object_grasp_state",
        "get_cabinet_drawer_handle_pose",
        "get_cabinet_drawer_progress",
        "get_gripper_fingertip_state",
        "get_joint_positions",
        "get_drawer_hook_poses",
        "get_drawer_side_hook_poses",
        "get_drawer_tiltx_hook_poses",
        "get_drawer_sleeve_hook_poses",
        "measure_drawer_hook_alignment",
        "measure_drawer_gripper_contacts",
    ),
    allowed_skills=(
        "verified_lift_pick_object",
        "verified_stage_pick_object",
        "verified_low_side_pick_object",
        "verified_low_profile_pick_object",
        "verified_geometry_pick_object",
        "grasp_tall_object_side",
        "approach_and_grasp_object",
        "guarded_transport_held_object_to_target",
        "low_release_object_to_target",
        "pick_transport_release_object_to_target",
        "rack_supported_release_object_to_target",
        "push_object_to_target_region",
        "global_bddl_goal_repair",
        "local_placement_closure",
        "drawer_z_only_settle_object",
        "repair_placement_relation",
        "place_object_in_microwave_region",
        "close_microwave_by_sweep",
        "turn_on_stove_by_sweep",
        "open_drawer_by_sleeve_retry",
        "pull_sleeved_drawer",
        "close_drawer_by_push",
    ),
    forbidden_symbols=(
        "satisfy_goal_predicates",
        "open_cabinet_drawer",
        "close_cabinet_drawer",
        "pick_and_place_object",
        "pick_and_place_objects",
        "put_object_in_container",
        "put_object_on_target",
        "put_object_in_goal_region",
        "push_object_to_goal_region",
        "sim.data.qpos",
        "sim.data.qvel",
        ".data.qpos",
        ".data.qvel",
        "set_state",
        "set_success",
        "set_task_success",
        "set_object_pose",
        "set_instance_pose",
        "set_joint",
        "set_qpos",
        "_refresh_task_success_reward",
        "_check_success",
        "task_completed =",
    ),
    profile="libero_privileged_readonly",
)


SAM3_OBSERVATION_APIS = (
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "depth_to_point_cloud",
    "mask_to_world_points",
    "pixel_to_world_point",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points",
)


SAM3_CLEAN_API_POLICY = CleanApiPolicy(
    allowed_primitives=DEFAULT_CLEAN_API_POLICY.allowed_primitives,
    allowed_observations=tuple(
        dict.fromkeys(DEFAULT_CLEAN_API_POLICY.allowed_observations + SAM3_OBSERVATION_APIS)
    ),
    allowed_skills=DEFAULT_CLEAN_API_POLICY.allowed_skills,
    forbidden_symbols=DEFAULT_CLEAN_API_POLICY.forbidden_symbols,
    profile="libero_privileged_sam3",
)


CAPX_AGENT0_API_POLICY = CleanApiPolicy(
    allowed_primitives=(
        "goto_pose",
        "open_gripper",
        "close_gripper",
        "solve_ik",
        "solve_ik_strict",
        "move_to_joints",
        "execute_joint_trajectory",
        "goto_home_joint_position",
        "plan_grasp_trajectory",
        "plan_with_grasped_object",
    ),
    allowed_observations=(
        "get_observation",
        "get_ee_pose",
        "segment_sam3_text_prompt",
        "segment_sam3_point_prompt",
        "point_prompt_molmo",
        "plan_grasp",
        "plan_grasp_from_point_clouds",
        "get_oriented_bounding_box_from_3d_points",
        "depth_to_point_cloud",
        "mask_to_world_points",
        "pixel_to_world_point",
        "rotation_matrix_to_quaternion",
        "decompose_transform",
        "transform_points",
        "interpolate_segment",
        "normalize_vector",
        "select_top_down_grasp",
        "subsample_point_cloud",
        "filter_noise",
        "parse_grasp_poses_for_curobo",
    ),
    allowed_skills=(),
    forbidden_symbols=tuple(
        dict.fromkeys(
            DEFAULT_CLEAN_API_POLICY.forbidden_symbols
            + (
                "FrankaLiberoOracleReadOnlyApi",
                "FrankaLiberoCleanSkillApi",
                "FrankaLiberoPrivilegedApi",
                "get_object_pose",
                "get_all_object_poses",
                "get_goal_target_region_pose",
                "get_cabinet_drawer_handle_pose",
                "get_cabinet_drawer_progress",
                "get_bddl_goal_diagnostics",
                "get_task_status",
                "measure_object_grasp_state",
                "get_object_grasp_affordance",
                "verified_lift_pick_object",
                "verified_stage_pick_object",
                "guarded_transport_held_object_to_target",
                "low_release_object_to_target",
                "local_placement_closure",
                "repair_placement_relation",
                "open_drawer_by_sleeve_retry",
                "close_drawer_by_push",
            )
        )
    ),
    profile="capx_agent0",
)


ROBOSUITE_PRIVILEGED_PRIMITIVES = (
    "goto_pose",
    "goto_pose_checked",
    "goto_home_joint_position",
    "open_gripper",
    "close_gripper",
    "goto_pose_arm0",
    "goto_pose_arm1",
    "goto_pose_both",
    "goto_pose_both_checked",
    "open_gripper_arm0",
    "close_gripper_arm0",
    "open_gripper_arm1",
    "close_gripper_arm1",
)

ROBOSUITE_PRIVILEGED_OBSERVATIONS = (
    "get_observation",
    "get_task_status",
    "get_robosuite_task_context",
    "get_ee_pose",
    "get_gripper_state",
    "get_object_pose",
    "get_all_object_poses",
    "get_instance_object_names",
    "get_contact_diagnostics",
    "get_object_grasp_affordance",
    "measure_object_grasp_state",
    "sample_grasp_pose",
    "get_cube_lift_diagnostic",
    "get_stack_diagnostic",
    "get_restack_diagnostic",
    "get_nut_peg_diagnostic",
    "get_nut_handle_to_center_offset",
    "get_handle0_pos",
    "get_handle1_pos",
    "get_handle0_pose",
    "get_handle1_pose",
    "get_arm0_gripper_pose",
    "get_arm1_gripper_pose",
    "get_bimanual_lift_diagnostic",
    "verify_bimanual_handle_grasps",
    "get_wipe_diagnostic",
    "get_hammer_pose",
    "get_hammer_handle_geometry",
    "get_handover_diagnostic",
)

ROBOSUITE_PRIVILEGED_SKILLS = (
    "verified_pick_cube",
    "relocate_covering_cube_and_verify",
    "recover_and_verify_restack",
    "transport_cube_above_target",
    "release_and_verify_stack",
    "verified_grasp_nut_handle",
    "move_nut_above_peg",
    "insert_nut_and_verify",
    "prepare_bimanual_handle_grasps",
    "close_and_verify_bimanual_grasps",
    "synchronized_lift_and_verify",
    "approach_spill_surface",
    "wipe_spill_and_verify",
    "verified_pick_hammer_arm0",
    "present_hammer_for_handover",
    "capture_hammer_arm1",
    "release_arm0_and_verify_handover",
)


def build_bound_api_policy(
    api_functions: dict[str, Any] | set[str],
    *,
    privileged: bool,
    sam3_enabled: bool = False,
) -> CleanApiPolicy:
    """Build the prompt and preflight policy from functions actually bound to the env."""
    bound = set(api_functions)
    is_robosuite_stagecap = "get_robosuite_task_context" in bound

    if is_robosuite_stagecap:
        primitive_catalog = ROBOSUITE_PRIVILEGED_PRIMITIVES
        observation_catalog = ROBOSUITE_PRIVILEGED_OBSERVATIONS
        skill_catalog = ROBOSUITE_PRIVILEGED_SKILLS
        forbidden = DEFAULT_CLEAN_API_POLICY.forbidden_symbols
        profile = "robosuite_privileged_readonly"
    else:
        base = (
            CAPX_AGENT0_API_POLICY
            if not privileged
            else SAM3_CLEAN_API_POLICY
            if sam3_enabled
            else DEFAULT_CLEAN_API_POLICY
        )
        primitive_catalog = base.allowed_primitives
        observation_catalog = base.allowed_observations
        skill_catalog = base.allowed_skills
        forbidden = base.forbidden_symbols
        profile = base.profile

    return CleanApiPolicy(
        allowed_primitives=tuple(name for name in primitive_catalog if name in bound),
        allowed_observations=tuple(name for name in observation_catalog if name in bound),
        allowed_skills=tuple(name for name in skill_catalog if name in bound),
        forbidden_symbols=forbidden,
        profile=profile,
    )


@dataclass(frozen=True)
class VerificationEvidence:
    """Evidence used to judge one stage after execution."""

    stage_id: str
    status: StageStatus
    rationale: str
    visual_evidence: list[str] = field(default_factory=list)
    geometry_evidence: dict[str, Any] = field(default_factory=dict)
    log_evidence: list[str] = field(default_factory=list)
    repair_hint: str | None = None


@dataclass(frozen=True)
class StageExecutionRecord:
    """Trace record for stage-level execution and replanning."""

    task: str
    stage_id: str
    attempt: int
    code_path: str | None
    stdout: str = ""
    stderr: str = ""
    evidence: VerificationEvidence | None = None
    forbidden_symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageLoopState:
    """Mutable-by-replacement state for a stage-level execution loop."""

    task: str
    plan: TaskPlan
    current_stage_index: int = 0
    execution_records: list[StageExecutionRecord] = field(default_factory=list)
    repair_counts: dict[str, int] = field(default_factory=dict)

    @property
    def current_stage(self) -> StagePlan | None:
        if 0 <= self.current_stage_index < len(self.plan.stages):
            return self.plan.stages[self.current_stage_index]
        return None
