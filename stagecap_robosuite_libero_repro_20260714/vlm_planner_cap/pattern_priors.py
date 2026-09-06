"""Pattern priors distilled from the CAP-35 successful LIBERO programs.

The priors in this file are abstract execution schemas, not retrieved expert
programs. They encode what the success corpus and API audit taught us about
reliable manipulation loops: verify a lift before transport, release low before
retreat, test drawer contact before pulling far, and gate long-horizon tasks by
stage evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import StageCheck, StagePlan, TaskFamily, TaskPlan


@dataclass(frozen=True)
class ParsedTask:
    """Lightweight task interpretation for zero-shot stage planning."""

    task: str
    families: tuple[TaskFamily, ...]
    objects: tuple[str, ...]
    targets: tuple[str, ...]
    predicates: tuple[str, ...]
    drawer_level: str | None = None


_KNOWN_OBJECTS = (
    "alphabet soup",
    "tomato sauce",
    "cream cheese",
    "salad dressing",
    "bbq sauce",
    "ketchup",
    "chocolate pudding",
    "pudding",
    "orange juice",
    "wine bottle",
    "moka pot",
    "moka pots",
    "butter",
    "milk",
    "white mug",
    "yellow and white mug",
    "yellow white mug",
    "bowl",
    "plate",
    "mug",
    "mugs",
    "book",
)

_KNOWN_TARGETS = (
    "basket",
    "microwave",
    "plate",
    "plates",
    "stove",
    "cabinet top",
    "cabinet",
    "rack",
    "caddy",
    "drawer",
    "bowl",
)


def build_pattern_plan(task: str) -> TaskPlan:
    """Build a CAP-35 pattern-informed stage plan from task language."""
    parsed = parse_task(task)
    stages: list[StagePlan] = []

    if TaskFamily.CUBE_LIFT in parsed.families:
        return _task_plan_with_notes(task, _robosuite_cube_lift_stages(), parsed)
    if TaskFamily.RESTACK in parsed.families:
        return _task_plan_with_notes(task, _robosuite_restack_stages(), parsed)
    if TaskFamily.STACK in parsed.families:
        return _task_plan_with_notes(task, _robosuite_stack_stages(), parsed)
    if TaskFamily.PEG_INSERTION in parsed.families:
        return _task_plan_with_notes(task, _robosuite_peg_insertion_stages(), parsed)
    if TaskFamily.BIMANUAL_LIFT in parsed.families:
        return _task_plan_with_notes(task, _robosuite_bimanual_lift_stages(), parsed)
    if TaskFamily.WIPE in parsed.families:
        return _task_plan_with_notes(task, _robosuite_wipe_stages(), parsed)
    if TaskFamily.BIMANUAL_HANDOVER in parsed.families:
        return _task_plan_with_notes(task, _robosuite_bimanual_handover_stages(), parsed)

    if TaskFamily.PUSH_REPAIR in parsed.families:
        stages.extend(_push_region_stages(parsed))
        return _task_plan_with_notes(task, stages, parsed)

    if TaskFamily.DRAWER in parsed.families and _needs_drawer_first(parsed):
        stages.extend(_drawer_stages(parsed, mode="open"))

    if TaskFamily.STOVE_BUTTON in parsed.families:
        stages.extend(_stove_button_stages(parsed))

    movable_objects = parsed.objects or _fallback_objects(task)
    targets = parsed.targets or _fallback_targets(task)
    object_target_pairs = _pair_objects_and_targets(movable_objects, targets, parsed)
    object_target_pairs = _order_pairs_by_interference_risk(object_target_pairs, parsed)

    for index, (object_name, target_name, predicate) in enumerate(object_target_pairs, start=1):
        if object_name == "stove":
            continue
        prefix = f"obj{index}_{_slug(object_name)}"
        stages.extend(_pick_lift_transport_release_stages(prefix, object_name, target_name, predicate, parsed))

    if TaskFamily.DRAWER in parsed.families and _needs_drawer_after_object(parsed):
        stages.extend(_drawer_stages(parsed, mode="close"))

    if _needs_microwave_close(parsed):
        stages.extend(_microwave_close_stages())

    if not stages and TaskFamily.STOVE_BUTTON not in parsed.families:
        stages.extend(
            _pick_lift_transport_release_stages(
                "obj1_task_object",
                "task object",
                "task target",
                "on_or_in",
                parsed,
            )
        )

    return _task_plan_with_notes(task, stages, parsed)


def _task_plan_with_notes(task: str, stages: list[StagePlan], parsed: ParsedTask) -> TaskPlan:
    notes = [
        "Pattern priors are distilled from CAP-35 successful code as abstract schemas only.",
        "Evaluation must not retrieve or insert task-specific expert code.",
        "All scene changes must come from robot motion; readonly coordinates are grounding and verification signals.",
    ]
    if TaskFamily.MULTI_OBJECT in parsed.families:
        notes.append("Use gated sequential execution: verify each object before starting the next one.")
    if TaskFamily.DRAWER in parsed.families:
        notes.append("For drawer stages, verify contact/progress before large pull or downstream placement.")
    if _needs_microwave_close(parsed):
        notes.append("For microwave close stages, verify the readonly close diagnostic after physical contact sweeps.")
    if TaskFamily.PUSH_REPAIR in parsed.families:
        notes.append("For push-region goals, use physical contact pushes and verify with BDDL/site geometry after every stroke.")
    if any(
        family in parsed.families
        for family in (
            TaskFamily.CUBE_LIFT,
            TaskFamily.STACK,
            TaskFamily.RESTACK,
            TaskFamily.PEG_INSERTION,
            TaskFamily.WIPE,
            TaskFamily.BIMANUAL_LIFT,
            TaskFamily.BIMANUAL_HANDOVER,
        )
    ):
        notes.append(
            "RoboSuite privileged state is read-only grounding and verification; every scene change must still come from robot motion."
        )
    return TaskPlan(task=task, stages=stages, notes=notes, families=list(parsed.families))


def parse_task(task: str) -> ParsedTask:
    """Parse enough task structure to select zero-shot manipulation priors."""
    lowered = _normalize(task)
    targets = _extract_targets(lowered)
    objects = _extract_objects(lowered, targets)
    predicates: list[str] = []
    if " in " in f" {lowered} " or "inside" in lowered or "to basket" in lowered:
        predicates.append("in")
    if (" on " in f" {lowered} " and not lowered.startswith("turn on ")) or "onto" in lowered:
        predicates.append("on")
    if "right of" in lowered:
        predicates.append("right_of")
    if "next to" in lowered:
        predicates.append("next_to")

    families: list[TaskFamily] = []
    is_restack = "restack" in lowered or "re-stack" in lowered
    is_cube_stack = bool(
        "cube" in lowered
        and ("stack" in lowered or "on top of" in lowered)
        and not is_restack
    )
    is_cube_lift = bool(
        "cube" in lowered
        and "lift" in lowered
        and not is_cube_stack
        and not is_restack
    )
    is_peg_insertion = bool(
        "nut" in lowered
        and ("insert" in lowered or "assembly" in lowered)
        and any(term in lowered for term in ("peg", "block"))
    )
    is_bimanual_lift = bool(
        any(term in lowered for term in ("two arm", "two-arm", "both arm", "bimanual"))
        and "lift" in lowered
        and any(term in lowered for term in ("pot", "handle"))
        and not any(term in lowered for term in ("handover", "hand over", "hand it over"))
    )
    is_wipe = bool("wipe" in lowered and any(term in lowered for term in ("spill", "dirt", "marker")))
    is_bimanual_handover = bool(
        any(term in lowered for term in ("handover", "hand over", "hand it over"))
        and any(term in lowered for term in ("hammer", "arm 0", "arm0"))
    )
    if is_cube_lift:
        families.append(TaskFamily.CUBE_LIFT)
    if is_restack:
        families.append(TaskFamily.RESTACK)
    if is_cube_stack:
        families.append(TaskFamily.STACK)
    if is_peg_insertion:
        families.append(TaskFamily.PEG_INSERTION)
    if is_bimanual_lift:
        families.append(TaskFamily.BIMANUAL_LIFT)
    if is_wipe:
        families.append(TaskFamily.WIPE)
    if is_bimanual_handover:
        families.append(TaskFamily.BIMANUAL_HANDOVER)
    if any(target in lowered for target in ("basket", "bowl", "microwave")):
        families.append(TaskFamily.CONTAINER_PLACE)
    if any(target in lowered for target in ("plate", "stove", "cabinet top")):
        families.append(TaskFamily.SURFACE_PLACE)
    if any(rel in lowered for rel in ("next to", "right of", "between")):
        families.append(TaskFamily.SPATIAL_RELATION)
    if "push" in lowered and any(word in lowered for word in ("front", "back", "left", "right", "region", "side")):
        families.append(TaskFamily.PUSH_REPAIR)
    if any(term in lowered for term in ("drawer", "cabinet drawer")):
        families.append(TaskFamily.DRAWER)
    if "turn on stove" in lowered or "turn on the stove" in lowered or "turn the stove" in lowered:
        families.append(TaskFamily.STOVE_BUTTON)
    if "rack" in lowered or "caddy" in lowered:
        families.append(TaskFamily.RACK_OR_CADDY)
    if _looks_multi_object(lowered, objects) and not (is_bimanual_lift or is_bimanual_handover or is_restack):
        families.append(TaskFamily.MULTI_OBJECT)
    if (
        TaskFamily.PUSH_REPAIR not in families
        and not any(
            family in families
            for family in (
                TaskFamily.CUBE_LIFT,
                TaskFamily.STACK,
                TaskFamily.RESTACK,
                TaskFamily.PEG_INSERTION,
                TaskFamily.WIPE,
                TaskFamily.BIMANUAL_LIFT,
                TaskFamily.BIMANUAL_HANDOVER,
            )
        )
        and (
            not families
            or any(
                family in families
                for family in (TaskFamily.CONTAINER_PLACE, TaskFamily.SURFACE_PLACE, TaskFamily.RACK_OR_CADDY)
            )
        )
    ):
        families.insert(0, TaskFamily.PICK_PLACE)

    drawer_level = None
    if "top drawer" in lowered:
        drawer_level = "top"
    elif "bottom drawer" in lowered:
        drawer_level = "bottom"
    elif "drawer" in lowered:
        drawer_level = "middle"

    return ParsedTask(
        task=task,
        families=tuple(dict.fromkeys(families)),
        objects=objects,
        targets=targets,
        predicates=tuple(dict.fromkeys(predicates)) or ("on_or_in",),
        drawer_level=drawer_level,
    )


def _extract_objects(lowered: str, targets: tuple[str, ...]) -> tuple[str, ...]:
    """Extract movable object mentions while excluding target-only containers."""
    objects: list[str] = []
    negated_or_ignored: set[str] = set()
    for obj in _KNOWN_OBJECTS:
        forms = {obj, obj.replace(" ", "_")}
        if obj == "plates":
            forms.add("plate")
        for form in forms:
            if (
                f"not on the {form}" in lowered
                or f"not on {form}" in lowered
                or f"not in the {form}" in lowered
                or f"not in {form}" in lowered
                or f"ignore the {form}" in lowered
                or f"ignoring the {form}" in lowered
            ):
                negated_or_ignored.add("plate" if obj == "plates" else obj)
    if "moka pots" in lowered or "both moka pot" in lowered:
        objects.extend(["moka pot", "moka pot 2"])
    if "two mugs" in lowered:
        objects.extend(["mug", "mug"])
    mug_residual = lowered
    if "yellow and white mug" in lowered:
        objects.append("yellow and white mug")
        mug_residual = mug_residual.replace("yellow and white mug", "")
    elif "yellow white mug" in lowered:
        objects.append("yellow white mug")
        mug_residual = mug_residual.replace("yellow white mug", "")
    if "white mug" in mug_residual:
        objects.append("white mug")

    skip_aliases = {"moka pots", "mugs"}
    if any(obj == "moka pot" for obj in objects):
        skip_aliases.add("moka pot")
    if any(obj == "mug" for obj in objects):
        skip_aliases.add("mug")
    if any(obj in {"white mug", "yellow and white mug", "yellow white mug"} for obj in objects):
        skip_aliases.add("mug")
        skip_aliases.add("white mug")
        skip_aliases.add("yellow and white mug")
        skip_aliases.add("yellow white mug")
    if "chocolate pudding" in lowered:
        skip_aliases.add("pudding")

    for obj in _KNOWN_OBJECTS:
        if obj in skip_aliases:
            continue
        if obj in negated_or_ignored:
            continue
        if obj in lowered:
            objects.append(obj)

    objects.sort(key=lambda name: lowered.find(name) if name in lowered else len(lowered))

    # If a term is clearly the target receptacle/surface and another movable
    # object exists, keep it only as a target. This fixes tasks like
    # "cream cheese to bowl" while preserving "bowl to plate".
    if len(objects) > 1:
        target_set = set(targets)
        objects = [obj for obj in objects if obj not in target_set]
        if any("mug" in obj for obj in objects) and any("plate" in target for target in target_set):
            objects = [obj for obj in objects if obj not in {"plate", "plates"}]

    return tuple(objects)


def _extract_targets(lowered: str) -> tuple[str, ...]:
    """Extract target surfaces/receptacles from the goal phrase."""
    negated_targets: set[str] = set()
    for target in _KNOWN_TARGETS:
        forms = {target, target.replace(" ", "_")}
        if target == "plates":
            forms.add("plate")
        for form in forms:
            if (
                f"not on the {form}" in lowered
                or f"not on {form}" in lowered
                or f"not in the {form}" in lowered
                or f"not in {form}" in lowered
                or f"ignore the {form}" in lowered
                or f"ignoring the {form}" in lowered
            ):
                negated_targets.add("plate" if target == "plates" else target)
    if "front of the stove" in lowered or "front of stove" in lowered:
        return ("front of stove",)
    if "back of the stove" in lowered or "back of stove" in lowered:
        return ("back of stove",)
    if "drawer" in lowered:
        if "top drawer" in lowered:
            return ("top drawer",)
        if "bottom drawer" in lowered:
            return ("bottom drawer",)
        return ("middle drawer",)
    if "left plate" in lowered or "right plate" in lowered:
        targets: list[str] = []
        if "left plate" in lowered:
            targets.append("left plate")
        if "right plate" in lowered:
            targets.append("right plate")
        return tuple(targets)
    if "back compartment" in lowered and "caddy" in lowered:
        return ("back caddy compartment",)

    target_phrase = lowered
    if "cabinet" in lowered and (
        "top of the cabinet" in lowered
        or "top of the wooden cabinet" in lowered
        or "wooden cabinet top" in lowered
        or "cabinet top" in lowered
    ):
        target_phrase = "cabinet top"
    elif " to " in lowered:
        target_phrase = lowered.rsplit(" to ", 1)[1]
    elif " on it" in lowered and "stove" in lowered:
        target_phrase = "stove"
    elif " on " in lowered and not lowered.startswith("turn on "):
        target_phrase = lowered.rsplit(" on ", 1)[1]
    elif "right of" in lowered:
        target_phrase = lowered.rsplit("right of", 1)[1]
    elif "next to" in lowered:
        target_phrase = lowered.rsplit("next to", 1)[1]

    if "two plates" in lowered or "two plate" in lowered:
        return ("plate", "plate")

    targets: list[str] = []
    for target in _KNOWN_TARGETS:
        if target == "plates":
            continue
        canonical = "plate" if target == "plates" else target
        if canonical in negated_targets:
            continue
        if target in target_phrase:
            targets.append("plate" if target == "plates" else target)

    if not targets:
        for target in _KNOWN_TARGETS:
            if target == "plates":
                continue
            canonical = "plate" if target == "plates" else target
            if canonical in negated_targets:
                continue
            if target in lowered:
                targets.append("plate" if target == "plates" else target)
    return tuple(dict.fromkeys(targets))


def _robosuite_ground_stage(
    *,
    stage_id: str,
    name: str,
    intent: str,
    references: list[str],
    family: TaskFamily,
    diagnostic_api: str,
) -> StagePlan:
    return StagePlan(
        stage_id=stage_id,
        name=name,
        intent=intent,
        allowed_primitives=[],
        allowed_observations=[
            "get_robosuite_task_context",
            "get_all_object_poses",
            "get_instance_object_names",
            "get_object_pose",
            "get_object_grasp_affordance",
            diagnostic_api,
            "get_task_status",
        ],
        target_references=references,
        family=family,
        max_attempts=1,
        checks=[
            StageCheck(
                name="privileged_grounding_consistent",
                description="Exact read-only task geometry resolves every object and target required by the stage machine.",
                check_type="geometry",
                success_criteria=[
                    "task context matches the requested RoboSuite task family",
                    "all required object or handle poses are present",
                    "the initial task diagnostic is finite and physically plausible",
                ],
            )
        ],
        repair_hints=[
            "Use exact names from get_instance_object_names; do not invoke perception or guess coordinates.",
            "Privileged state is read-only and must never be used to mutate simulator state.",
        ],
    )


def _robosuite_action_stage(
    *,
    stage_id: str,
    name: str,
    intent: str,
    skill: str,
    observations: list[str],
    references: list[str],
    family: TaskFamily,
    criteria: list[str],
    max_attempts: int = 2,
    repair_hint: str,
) -> StagePlan:
    return StagePlan(
        stage_id=stage_id,
        name=name,
        intent=intent,
        allowed_primitives=[skill],
        allowed_observations=observations,
        target_references=references,
        family=family,
        max_attempts=max_attempts,
        checks=[
            StageCheck(
                name=f"{stage_id}_verified",
                description=intent,
                check_type="geometry",
                success_criteria=criteria,
            )
        ],
        repair_hints=[repair_hint],
    )


def _robosuite_cube_lift_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_ee_pose",
        "get_gripper_state",
        "measure_object_grasp_state",
        "get_cube_lift_diagnostic",
        "get_contact_diagnostics",
    ]
    family = TaskFamily.CUBE_LIFT
    return [
        _robosuite_ground_stage(
            stage_id="cube_lift_ground_object_target",
            name="Ground the red cube",
            intent="Read the exact red-cube pose and current benchmark state.",
            references=["red cube"],
            family=family,
            diagnostic_api="get_cube_lift_diagnostic",
        ),
        _robosuite_action_stage(
            stage_id="cube_lift_pick_verify",
            name="Pick and lift the red cube",
            intent="Grasp the red cube, lift it physically, and verify that it remains held.",
            skill="verified_pick_cube",
            observations=observations,
            references=["red cube", "gripper"],
            family=family,
            criteria=[
                "official task completion or sufficient cube lift is observed",
                "the cube remains near the gripper with contact evidence",
            ],
            max_attempts=3,
            repair_hint="Retry the audited cube pick with the next bounded lateral offset.",
        ),
    ]


def _robosuite_restack_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_ee_pose",
        "get_gripper_state",
        "measure_object_grasp_state",
        "get_restack_diagnostic",
        "get_stack_diagnostic",
        "get_contact_diagnostics",
    ]
    family = TaskFamily.RESTACK
    return [
        _robosuite_ground_stage(
            stage_id="restack_ground_object_target",
            name="Ground the initially stacked cubes",
            intent="Read both cube poses and verify that green obstructs the red cube.",
            references=["green cube", "red cube", "on"],
            family=family,
            diagnostic_api="get_restack_diagnostic",
        ),
        _robosuite_action_stage(
            stage_id="restack_relocate_covering_cube",
            name="Relocate the green cube",
            intent="Lift the covering green cube and place it stably on a clear table region.",
            skill="relocate_covering_cube_and_verify",
            observations=observations,
            references=["green cube", "red cube", "table"],
            family=family,
            criteria=[
                "green is no longer on red",
                "both cubes are at table height and separated in XY",
            ],
            max_attempts=3,
            repair_hint="Retry relocation before attempting to grasp the obstructed red cube.",
        ),
        _robosuite_action_stage(
            stage_id="restack_pick_lift_verify",
            name="Pick the exposed red cube",
            intent="Lift-test the red cube after the covering cube has been moved.",
            skill="verified_pick_cube",
            observations=observations,
            references=["red cube", "gripper"],
            family=family,
            criteria=["red cube remains held after a physical lift"],
            max_attempts=3,
            repair_hint="Retry the audited red-cube grasp only while green remains clear.",
        ),
        _robosuite_action_stage(
            stage_id="restack_transport_guarded",
            name="Transport red above green",
            intent="Move the held red cube above the relocated green cube without releasing.",
            skill="transport_cube_above_target",
            observations=observations,
            references=["red cube", "green cube", "pre-stack pose"],
            family=family,
            criteria=["red is centered above green and remains held"],
            repair_hint="Keep the gripper closed and retry guarded transport.",
        ),
        _robosuite_action_stage(
            stage_id="restack_low_release_verify",
            name="Release and verify the final stack",
            intent="Lower red onto green, release, retreat, and verify official completion.",
            skill="recover_and_verify_restack",
            observations=observations,
            references=["red cube", "green cube", "on"],
            family=family,
            criteria=["red-on-green stack remains stable after release"],
            repair_hint="Retry only a bounded final placement while the red cube remains recoverable.",
        ),
    ]


def _robosuite_stack_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_ee_pose",
        "get_gripper_state",
        "measure_object_grasp_state",
        "get_stack_diagnostic",
        "get_contact_diagnostics",
    ]
    return [
        _robosuite_ground_stage(
            stage_id="stack_ground_object_target",
            name="Ground the red and green cubes",
            intent="Read the exact cube poses, extents, and current stack relation before moving.",
            references=["red cube", "green cube", "on"],
            family=TaskFamily.STACK,
            diagnostic_api="get_stack_diagnostic",
        ),
        StagePlan(
            stage_id="stack_pick_lift_verify",
            name="Pick and lift-test the red cube",
            intent="Physically grasp the red cube, lift it clear of the table, and verify that it follows the gripper.",
            allowed_primitives=["verified_pick_cube"],
            allowed_observations=observations,
            target_references=["red cube", "gripper"],
            family=TaskFamily.STACK,
            max_attempts=3,
            checks=[
                StageCheck(
                    name="red_cube_held_after_lift",
                    description="The red cube rises with the gripper and remains grasped.",
                    check_type="geometry",
                    success_criteria=[
                        "red cube lift_delta exceeds the minimum",
                        "cube-to-gripper distance remains bounded",
                        "gripper/contact evidence is consistent with a grasp",
                    ],
                )
            ],
            repair_hints=["Retry verified_pick_cube with the current STAGE_ATTEMPT; do not transport an unheld cube."],
        ),
        StagePlan(
            stage_id="stack_transport_guarded",
            name="Transport the red cube above the green cube",
            intent="Move the verified-held red cube to a guarded pre-stack pose without releasing it.",
            allowed_primitives=["transport_cube_above_target"],
            allowed_observations=observations,
            target_references=["red cube", "green cube", "pre-stack pose"],
            family=TaskFamily.STACK,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="red_cube_above_green_cube",
                    description="The held red cube is centered above the green cube with vertical clearance.",
                    check_type="geometry",
                    success_criteria=[
                        "red-green XY error is within tolerance",
                        "red cube remains held",
                        "the cube is still above the final contact height",
                    ],
                )
            ],
            repair_hints=["Keep the gripper closed and retry transport; never release from the guarded pose."],
        ),
        StagePlan(
            stage_id="stack_low_release_verify",
            name="Lower, release, and verify the stack",
            intent="Lower the red cube to the exact support height, release gently, retreat, and verify stable stacking.",
            allowed_primitives=["release_and_verify_stack"],
            allowed_observations=observations,
            target_references=["red cube", "green cube", "on"],
            family=TaskFamily.STACK,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="stack_stable_after_release",
                    description="The red cube remains centered on and in contact with the green cube after retreat.",
                    check_type="geometry",
                    success_criteria=[
                        "benchmark task status is complete or stack geometry is satisfied",
                        "red-green XY and Z errors are within tolerance",
                        "the gripper has released and retreated",
                    ],
                )
            ],
            repair_hints=["Use only release_and_verify_stack; let its returned metrics determine whether a local retry is recoverable."],
        ),
    ]


def _robosuite_peg_insertion_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_ee_pose",
        "get_gripper_state",
        "measure_object_grasp_state",
        "get_nut_peg_diagnostic",
        "get_contact_diagnostics",
    ]
    return [
        _robosuite_ground_stage(
            stage_id="peg_ground_object_target",
            name="Ground the square nut handle and peg",
            intent="Read exact nut, handle, and peg poses and establish the rigid insertion geometry.",
            references=["brown square nut", "brown square block", "insert"],
            family=TaskFamily.PEG_INSERTION,
            diagnostic_api="get_nut_peg_diagnostic",
        ),
        StagePlan(
            stage_id="peg_pick_lift_verify",
            name="Grasp the square nut by its handle",
            intent="Physically grasp the nut handle, move to a manipulable posture, and verify the nut remains held.",
            allowed_primitives=["verified_grasp_nut_handle"],
            allowed_observations=observations,
            target_references=["brown square nut handle", "brown square nut"],
            family=TaskFamily.PEG_INSERTION,
            max_attempts=3,
            checks=[
                StageCheck(
                    name="nut_held_by_handle",
                    description="The nut is lifted and remains rigidly associated with the gripper.",
                    check_type="geometry",
                    success_criteria=[
                        "nut lift_delta exceeds the minimum",
                        "handle-to-gripper distance remains bounded",
                        "nut/gripper contact evidence remains present",
                    ],
                )
            ],
            repair_hints=["Retry the audited handle grasp; do not align or insert unless held is verified."],
        ),
        StagePlan(
            stage_id="peg_align_guarded",
            name="Align the nut above the square peg",
            intent="Move the held nut to a collision-safe pose directly above the peg while preserving its rigid transform.",
            allowed_primitives=["move_nut_above_peg"],
            allowed_observations=observations,
            target_references=["brown square nut", "brown square block", "pre-insertion pose"],
            family=TaskFamily.PEG_INSERTION,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="nut_hole_aligned_above_peg",
                    description="The nut center is aligned in XY with the peg and remains above contact height.",
                    check_type="geometry",
                    success_criteria=[
                        "nut-peg XY error is within tolerance",
                        "nut remains held",
                        "positive insertion clearance remains",
                    ],
                )
            ],
            repair_hints=["Retry guarded alignment without opening the gripper."],
        ),
        StagePlan(
            stage_id="peg_insert_verify",
            name="Insert and verify the square nut",
            intent="Descend along the peg axis, release the nut, settle, and verify benchmark insertion success.",
            allowed_primitives=["insert_nut_and_verify"],
            allowed_observations=observations,
            target_references=["brown square nut", "brown square block", "insert"],
            family=TaskFamily.PEG_INSERTION,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="nut_inserted_on_peg",
                    description="The released square nut satisfies the peg insertion relation.",
                    check_type="geometry",
                    success_criteria=[
                        "task_completed or equivalent insertion diagnostic is true",
                        "nut-peg lateral error remains small after settling",
                        "the gripper is open and no longer carrying the nut",
                    ],
                )
            ],
            repair_hints=["Use the returned insertion metrics to decide whether one bounded retry is possible."],
        ),
    ]


def _robosuite_bimanual_lift_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_handle0_pose",
        "get_handle1_pose",
        "get_arm0_gripper_pose",
        "get_arm1_gripper_pose",
        "get_gripper_state",
        "get_bimanual_lift_diagnostic",
        "verify_bimanual_handle_grasps",
        "get_contact_diagnostics",
    ]
    return [
        _robosuite_ground_stage(
            stage_id="bimanual_ground_object_target",
            name="Ground both pot handles",
            intent="Read exact pot and handle poses and verify that each arm is assigned to its corresponding handle.",
            references=["pot handle 0", "pot handle 1", "pot"],
            family=TaskFamily.BIMANUAL_LIFT,
            diagnostic_api="get_bimanual_lift_diagnostic",
        ),
        StagePlan(
            stage_id="bimanual_pregrasp_handles",
            name="Move both grippers to handle pregrasps",
            intent="Open both grippers and position them at collision-safe side grasps around their assigned handles.",
            allowed_primitives=["prepare_bimanual_handle_grasps"],
            allowed_observations=observations,
            target_references=["arm 0", "pot handle 0", "arm 1", "pot handle 1"],
            family=TaskFamily.BIMANUAL_LIFT,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="both_grippers_at_handles",
                    description="Each open gripper reaches its assigned handle without moving the pot significantly.",
                    check_type="geometry",
                    success_criteria=[
                        "arm0-handle0 and arm1-handle1 distances are within tolerance",
                        "both grippers remain open",
                        "the pot remains supported by the table",
                    ],
                )
            ],
            repair_hints=["Retry the audited pregrasp skill; do not close one gripper early."],
        ),
        StagePlan(
            stage_id="bimanual_grasp_verify",
            name="Close and verify both handle grasps",
            intent="Close both grippers on their handles and verify symmetric grasp evidence before lifting.",
            allowed_primitives=["close_and_verify_bimanual_grasps"],
            allowed_observations=observations,
            target_references=["pot handle 0", "pot handle 1", "both grippers"],
            family=TaskFamily.BIMANUAL_LIFT,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="both_handles_grasped",
                    description="Both assigned handle grasps are present at the same time.",
                    check_type="geometry",
                    success_criteria=[
                        "both gripper-handle distances remain bounded",
                        "both grippers close around nonzero handle width",
                        "contact evidence is symmetric enough for a lift",
                    ],
                )
            ],
            repair_hints=["Never begin the lift when only one handle is verified."],
        ),
        StagePlan(
            stage_id="bimanual_lift_verify",
            name="Synchronously lift and verify the pot",
            intent="Move both grasped handles upward together and verify pot height and benchmark success.",
            allowed_primitives=["synchronized_lift_and_verify"],
            allowed_observations=observations,
            target_references=["pot", "both handles", "synchronized lift"],
            family=TaskFamily.BIMANUAL_LIFT,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="pot_lifted_by_both_arms",
                    description="The pot rises while both handle grasps remain coordinated.",
                    check_type="geometry",
                    success_criteria=[
                        "pot lift_delta exceeds the success threshold",
                        "both grippers remain near their assigned handles",
                        "task_completed is true or the lift diagnostic is satisfied",
                    ],
                )
            ],
            repair_hints=["Use one synchronized retry only if both grasps remain valid; otherwise return to pregrasp."],
        ),
    ]


def _robosuite_wipe_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_ee_pose",
        "get_gripper_state",
        "get_wipe_diagnostic",
        "get_contact_diagnostics",
    ]
    family = TaskFamily.WIPE
    return [
        _robosuite_ground_stage(
            stage_id="wipe_ground_object_target",
            name="Ground the spill marker cloud",
            intent="Read exact spill bounds, remaining marker count, and tool pose.",
            references=["brown spill"],
            family=family,
            diagnostic_api="get_wipe_diagnostic",
        ),
        _robosuite_action_stage(
            stage_id="wipe_approach_surface",
            name="Establish wiping contact",
            intent="Approach an exact remaining marker with the attached tool and verify surface contact.",
            skill="approach_spill_surface",
            observations=observations,
            references=["brown spill", "wiping tool"],
            family=family,
            criteria=["the tool reaches table height with contact or wipes at least one marker"],
            max_attempts=2,
            repair_hint="Retarget the current remaining marker centroid before executing the raster.",
        ),
        _robosuite_action_stage(
            stage_id="wipe_execute_verify",
            name="Wipe and verify the marker set",
            intent="Execute a bounded clustered path over remaining privileged marker geometry.",
            skill="wipe_spill_and_verify",
            observations=observations,
            references=["brown spill", "remaining markers"],
            family=family,
            criteria=["official marker count reaches zero and task completion is true"],
            max_attempts=3,
            repair_hint="Recompute the path from only the remaining markers with tighter spacing.",
        ),
    ]


def _robosuite_bimanual_handover_stages() -> list[StagePlan]:
    observations = [
        "get_task_status",
        "get_object_pose",
        "get_arm0_gripper_pose",
        "get_arm1_gripper_pose",
        "get_gripper_state",
        "get_hammer_handle_geometry",
        "measure_object_grasp_state",
        "get_handover_diagnostic",
        "get_contact_diagnostics",
    ]
    family = TaskFamily.BIMANUAL_HANDOVER
    return [
        _robosuite_ground_stage(
            stage_id="handover_ground_object_target",
            name="Ground hammer and both grippers",
            intent="Read exact handle, hammer, and two-gripper geometry before motion.",
            references=["hammer handle", "hammer"],
            family=family,
            diagnostic_api="get_handover_diagnostic",
        ),
        _robosuite_action_stage(
            stage_id="handover_arm0_pick_verify",
            name="Arm 0 picks and lifts the hammer",
            intent="Grasp the handle with arm 0 and verify the official arm-0 grasp predicate after lift.",
            skill="verified_pick_hammer_arm0",
            observations=observations,
            references=["hammer handle", "arm 0"],
            family=family,
            criteria=["arm 0 grasps the hammer and lifts it above the table"],
            max_attempts=3,
            repair_hint="Retry a bounded offset along the handle while the hammer remains on the source table.",
        ),
        _robosuite_action_stage(
            stage_id="handover_present_guarded",
            name="Present hammer in the guarded midpoint",
            intent="Move the arm-0-held hammer into the central handover region with clearance.",
            skill="present_hammer_for_handover",
            observations=observations,
            references=["hammer", "handover midpoint", "arm 0"],
            family=family,
            criteria=["arm 0 retains the hammer above threshold in the guarded region"],
            repair_hint="Preserve arm-0 contact and retry the guarded presentation pose.",
        ),
        _robosuite_action_stage(
            stage_id="handover_arm1_capture_verify",
            name="Arm 1 captures the handle",
            intent="Close arm 1 on a free handle segment while arm 0 continues support.",
            skill="capture_hammer_arm1",
            observations=observations,
            references=["hammer handle", "arm 1", "arm 0"],
            family=family,
            criteria=["the official arm-1 handle grasp predicate becomes true"],
            max_attempts=3,
            repair_hint="Retry a bounded handle-axis offset without releasing arm 0.",
        ),
        _robosuite_action_stage(
            stage_id="handover_release_verify",
            name="Release arm 0 and verify transfer",
            intent="Release and retreat arm 0, then verify arm 1 alone holds the lifted hammer.",
            skill="release_arm0_and_verify_handover",
            observations=observations,
            references=["hammer handle", "arm 1", "official handover"],
            family=family,
            criteria=["arm 1 grasps the handle, arm 0 does not, and hammer height exceeds threshold"],
            max_attempts=2,
            repair_hint="Keep arm 1 closed and perform only a bounded lift or arm-0 retreat repair.",
        ),
    ]


def _push_region_stages(parsed: ParsedTask) -> list[StagePlan]:
    object_name = parsed.objects[0] if parsed.objects else "task object"
    target_name = parsed.targets[0] if parsed.targets else "task target"
    predicate = _predicate_for_target("on", target_name)
    if predicate not in {"on", "in"}:
        predicate = "on"
    prefix = f"obj1_{_slug(object_name)}"
    observations = [
        "get_observation",
        "get_object_pose",
        "get_goal_target_region_pose",
        "get_site_pose",
        "get_site_on_diagnostic",
        "get_site_in_diagnostic",
        "get_task_status",
        "get_bddl_goal_diagnostics",
    ]
    return [
        StagePlan(
            stage_id=f"{prefix}_ground_object_target",
            name=f"Ground push object {object_name} and region {target_name}",
            intent=(
                f"Resolve the movable object '{object_name}', the target region '{target_name}', "
                "and the benchmark predicate before contact."
            ),
            allowed_primitives=[],
            allowed_observations=observations,
            target_references=[object_name, target_name, predicate],
            family=TaskFamily.PUSH_REPAIR,
            max_attempts=1,
            checks=[
                StageCheck(
                    name="push_grounding_consistent",
                    description="The push object and target region are grounded by readonly scene geometry.",
                    check_type="hybrid",
                    success_criteria=[
                        "selected object name matches the task language",
                        "target region matches the task language and parsed BDDL goal",
                        "object-to-region displacement gives a plausible push direction",
                    ],
                )
            ],
            repair_hints=[
                "Resolve generic language such as task object to the movable object named in the task.",
                "Prefer parsed BDDL target sites over guessed offsets for front/back/left/right region goals.",
            ],
        ),
        StagePlan(
            stage_id=f"{prefix}_push_to_region_verify",
            name=f"Push {object_name} to {target_name}",
            intent=(
                f"Use gripper contact to push '{object_name}' toward '{target_name}', "
                "checking object motion and target-region relation after each stroke."
            ),
            allowed_primitives=[
                "push_object_to_target_region",
                "open_gripper",
                "close_gripper",
                "goto_pose_checked",
                "move_linear_delta",
                "move_along_direction",
            ],
            allowed_observations=observations,
            target_references=[object_name, target_name, predicate],
            family=TaskFamily.PUSH_REPAIR,
            max_attempts=2,
            checks=[
                StageCheck(
                    name="object_in_push_region",
                    description="The pushed object satisfies the parsed target-region predicate after retreat.",
                    check_type="hybrid",
                    success_criteria=[
                        "object moves toward the target site rather than away from it",
                        "site relation diagnostic is satisfied after settling",
                        "task status or BDDL diagnostics agrees with the target-region relation",
                    ],
                )
            ],
            repair_hints=[
                "Use short feedback strokes and remeasure local_delta after every push.",
                "If the object does not move, lower the contact height or shift to the rim/side.",
                "If local_delta worsens, invert the push direction or re-ground the target site.",
            ],
        ),
    ]


def _pick_lift_transport_release_stages(
    prefix: str,
    object_name: str,
    target_name: str,
    predicate: str,
    parsed: ParsedTask,
) -> list[StagePlan]:
    family = _placement_family(target_name, predicate, parsed)
    observations = [
        "get_observation",
        "get_object_pose",
        "get_goal_target_region_pose",
        "get_object_grasp_affordance",
        "measure_object_grasp_state",
        "get_ee_pose",
    ]
    stages = [
        StagePlan(
            stage_id=f"{prefix}_ground_object_target",
            name=f"Ground {object_name} and {target_name}",
            intent=(
                f"Resolve the movable object '{object_name}', the target '{target_name}', "
                "and the intended relation before acting."
            ),
            allowed_primitives=[],
            allowed_observations=observations,
            target_references=[object_name, target_name, predicate],
            family=family,
            max_attempts=1,
            checks=[
                StageCheck(
                    name="grounding_consistent",
                    description="Object and target references match task language and current scene.",
                    check_type="hybrid",
                    success_criteria=[
                        "selected object name matches the task language",
                        "selected target/receptacle/surface matches the task language",
                        "readonly object and target coordinates are plausible and reachable",
                    ],
                )
            ],
            repair_hints=[
                "If grounding is ambiguous, list visible/free objects and choose the exact task object.",
                "If target parsing fails, infer predicate from task words: basket/bowl means in, plate/stove/cabinet top means on.",
            ],
        ),
        StagePlan(
            stage_id=f"{prefix}_pick_lift_verify",
            name=f"Pick and lift-test {object_name}",
            intent=(
                f"Grasp '{object_name}', lift it, and verify that the object follows the gripper "
                "before any transport."
            ),
            allowed_primitives=[
                "verified_stage_pick_object",
            ],
            allowed_observations=observations,
            target_references=[object_name, "object_grasp_affordance", "gripper_pose"],
            family=TaskFamily.PICK_PLACE,
            max_attempts=8,
            checks=[
                StageCheck(
                    name="object_held_after_lift",
                    description="The object is lifted and remains close to the gripper.",
                    check_type="hybrid",
                    success_criteria=[
                        "object z increases after lift-test",
                        "object remains near the fingertip midpoint",
                        "measure_object_grasp_state reports held or contact/lift evidence",
                    ],
                )
            ],
            repair_hints=[
                "Call only verified_stage_pick_object(object_name, stage_attempt=STAGE_ATTEMPT, max_attempts=1) in this stage; the audited contract owns all geometry, cup/rim/tall-object strategy switching, and candidate scheduling.",
                "Verify the returned held/lift_delta/skill/stage_pick_contract evidence. If held=True and lift_delta is sufficient, print satisfied immediately and do not add another lift or probing motion in the same attempt.",
                "Do not transport if lift-test fails.",
                "If pick fails, print needs_repair and let the stage loop retry the same stage with the next STAGE_ATTEMPT; do not call lower-level pick skills yourself.",
            ],
        ),
        StagePlan(
            stage_id=f"{prefix}_transport_guarded",
            name=f"Guarded transport of {object_name}",
            intent=(
                f"Move the verified-held '{object_name}' to a pre-release pose for '{target_name}' "
                "without releasing early."
            ),
            allowed_primitives=[
                "guarded_transport_held_object_to_target",
                "goto_pose_checked",
                "move_linear_delta",
                "move_along_direction",
                "move_to_joints",
            ],
            allowed_observations=["get_observation", "get_object_pose", "measure_object_grasp_state", "get_ee_pose"],
            target_references=[object_name, target_name, "pre_release_pose"],
            family=family,
            max_attempts=3,
            checks=[
                StageCheck(
                    name="held_object_above_target",
                    description="The held object reaches a safe pre-release pose for the target.",
                    check_type="hybrid",
                    success_criteria=[
                        "object remains held during transport",
                        "object xy projection is within or just above the target footprint",
                        "transport action reports arrival before release",
                        "object may still be high above the support because the low_release stage owns descent, contact, and final relation verification",
                    ],
                )
            ],
            repair_hints=[
                "Prefer guarded_transport_held_object_to_target(object_name) to move a verified-held object to the target pre-release pose.",
                "A transport result with held_after=True, arrived=True, and small xy_distance_to_target is satisfied even when BDDL reward is still 0 or a diagnostic shows large positive z distance above the target.",
                "Do not require on/in contact, low height, reward=1, or task_completed=True in this stage; those are release-stage checks.",
                "If arrival fails, do not release; retreat or replan a safer waypoint.",
                "If object slips during transport, return to pick_lift_verify.",
            ],
        ),
        StagePlan(
            stage_id=f"{prefix}_low_release_verify",
            name=f"Low release and verify {object_name}",
            intent=(
                f"Lower '{object_name}' close to '{target_name}', release, retreat, and verify "
                f"the '{predicate}' relation."
            ),
            allowed_primitives=[
                "low_release_object_to_target",
                "rack_supported_release_object_to_target",
                "local_placement_closure",
                "repair_placement_relation",
                "open_gripper",
                "move_linear_delta",
                "goto_pose_checked",
            ],
            allowed_observations=[
                "get_observation",
                "get_object_pose",
                "get_goal_target_region_pose",
                "measure_object_grasp_state",
                "get_on_diagnostic",
                "get_site_on_diagnostic",
                "get_site_in_diagnostic",
            ],
            target_references=[object_name, target_name, predicate],
            family=family,
            max_attempts=3,
            checks=[
                StageCheck(
                    name="placement_relation_stable",
                    description="The released object remains in/on/near the target after gripper retreat.",
                    check_type="hybrid",
                    success_criteria=[
                        "gripper is no longer holding the object",
                        "object-target relation matches the task predicate",
                        "object does not move away after retreat",
                    ],
                )
            ],
            repair_hints=[
                "Prefer low_release_object_to_target(object_name, predicate=predicate, target_name=target['target_name']) to release and verify the placement relation.",
                "For rack/caddy targets with predicate on or in_or_on, prefer rack_supported_release_object_to_target(object_name, target_name=target_name) because ordinary low release often drops objects off narrow supports.",
                "Use only numeric release distances: approach, retreat, and release_z must be floats or omitted, never strings such as 'linear'.",
                "If the object is already near the target but relation is not satisfied, call bounded local_placement_closure with the same predicate and target_name before any long repair loop.",
                "Use repair_placement_relation only after bounded local placement closure fails and there is enough horizon.",
                "Do not re-pick or import helper modules in this release stage; if the object is no longer transportable, report needs_repair and let the stage loop return to pick_lift_verify.",
                "If release height caused drift, retry with lower release height and slower retreat.",
                "For rack/caddy, prefer guarded low insertion and retreat along the approach direction.",
            ],
        ),
    ]
    return stages


def _drawer_stages(parsed: ParsedTask, mode: str = "open") -> list[StagePlan]:
    level = parsed.drawer_level or "middle"
    if mode == "close":
        return [
            StagePlan(
                stage_id=f"{level}_drawer_close_guarded",
                name=f"Close {level} drawer by physical push",
                intent="Close the drawer using gripper contact while preserving completed object placement.",
                allowed_primitives=[
                    "goto_finger_midpoint_checked",
                    "move_finger_midpoint_delta",
                    "move_linear_delta",
                    "move_along_direction",
                ],
                allowed_observations=[
                    "get_cabinet_drawer_progress",
                    "get_cabinet_drawer_handle_pose",
                    "measure_drawer_hook_alignment",
                ],
                target_references=[f"{level}_drawer", "drawer_handle_pose", "drawer_progress"],
                family=TaskFamily.DRAWER,
                max_attempts=4,
                checks=[
                    StageCheck(
                        name="drawer_closed_enough",
                        description="Drawer progress decreases to the closed range through physical pushing.",
                        check_type="hybrid",
                        success_criteria=[
                            "drawer open_fraction decreases after push",
                            "final drawer open_fraction is below the task threshold",
                        ],
                    )
                ],
                repair_hints=[
                    "Use short pushes along the negative open direction.",
                    "If contact slips, re-establish handle/front contact before pushing farther.",
                ],
            )
        ]

    return [
        StagePlan(
            stage_id=f"{level}_drawer_ground_handle",
            name=f"Ground {level} drawer handle",
            intent="Read drawer handle geometry and choose a reachable sleeve/hook contact strategy.",
            allowed_primitives=[],
            allowed_observations=[
                "get_cabinet_drawer_handle_pose",
                "get_drawer_sleeve_hook_poses",
                "get_drawer_tiltx_hook_poses",
                "get_drawer_side_hook_poses",
                "get_cabinet_drawer_progress",
            ],
            target_references=[f"{level}_drawer", "drawer_handle_pose", "pull_direction"],
            family=TaskFamily.DRAWER,
            max_attempts=1,
            checks=[
                StageCheck(
                    name="drawer_handle_grounded",
                    description="Handle pose, pull direction, and initial drawer progress are available.",
                    check_type="geometry",
                    success_criteria=[
                        "handle position is finite and reachable",
                        "pull direction is a unit vector",
                        "drawer progress is below the desired open threshold",
                    ],
                )
            ],
            repair_hints=[
                "If front sleeve candidates are unreachable, switch to side-entry hook candidates.",
            ],
        ),
        StagePlan(
            stage_id=f"{level}_drawer_sleeve_or_hook_handle",
            name=f"Sleeve or hook {level} drawer handle",
            intent="Place the open gripper around or behind the handle, then close and verify capture.",
            allowed_primitives=[
                "open_gripper",
                "close_gripper",
                "goto_finger_midpoint_checked",
                "move_finger_midpoint_delta",
                "move_linear_delta",
            ],
            allowed_observations=[
                "get_drawer_sleeve_hook_poses",
                "get_drawer_tiltx_hook_poses",
                "get_drawer_side_hook_poses",
                "measure_drawer_hook_alignment",
                "get_gripper_fingertip_state",
            ],
            target_references=[f"{level}_drawer", "drawer_handle_pose", "finger_midpoint"],
            family=TaskFamily.DRAWER,
            max_attempts=5,
            checks=[
                StageCheck(
                    name="handle_captured_or_hooked",
                    description="The handle is centered in the gripper aperture or hook geometry after closure.",
                    check_type="hybrid",
                    success_criteria=[
                        "measure_drawer_hook_alignment reports ready_to_close, closed_on_handle, or loose_hook_geometry",
                        "finger midpoint is near handle height and lateral line",
                        "gripper does not close far away from the handle",
                    ],
                )
            ],
            repair_hints=[
                "Try sleeve candidates before side-entry hook candidates.",
                "Correct lateral and vertical error before closing.",
                "If closure misses, reopen and return to mouth/prehook pose.",
            ],
        ),
        StagePlan(
            stage_id=f"{level}_drawer_test_progress",
            name=f"Test {level} drawer capture",
            intent="Apply a small pull and verify drawer progress before committing to a long pull.",
            allowed_primitives=[
                "move_finger_midpoint_delta",
                "move_finger_midpoint_delta_strict",
                "move_along_direction",
                "move_linear_delta",
            ],
            allowed_observations=[
                "get_cabinet_drawer_progress",
                "measure_drawer_hook_alignment",
                "measure_drawer_gripper_contacts",
            ],
            target_references=[f"{level}_drawer", "pull_direction", "drawer_progress"],
            family=TaskFamily.DRAWER,
            max_attempts=4,
            checks=[
                StageCheck(
                    name="drawer_progress_after_test_pull",
                    description="Small test motion physically increases drawer open fraction.",
                    check_type="hybrid",
                    success_criteria=[
                        "drawer open_fraction increases after test pull",
                        "handle remains near the gripper after the test",
                        "progress is not inferred from gripper displacement alone",
                    ],
                )
            ],
            repair_hints=[
                "If progress does not improve, recapture the handle instead of pulling farther.",
                "If progress moves opposite direction, reverse pull direction and retest.",
            ],
        ),
        StagePlan(
            stage_id=f"{level}_drawer_pull_open",
            name=f"Pull {level} drawer open",
            intent="Pull in short verified strokes until the drawer is open enough for the task.",
            allowed_primitives=[
                "move_finger_midpoint_delta",
                "move_finger_midpoint_delta_strict",
                "move_along_direction",
                "move_linear_delta",
            ],
            allowed_observations=[
                "get_cabinet_drawer_progress",
                "measure_drawer_hook_alignment",
                "measure_drawer_gripper_contacts",
            ],
            target_references=[f"{level}_drawer", "pull_direction", "drawer_progress"],
            family=TaskFamily.DRAWER,
            max_attempts=6,
            checks=[
                StageCheck(
                    name="drawer_open_enough",
                    description="Drawer open fraction reaches the task-required threshold.",
                    check_type="hybrid",
                    success_criteria=[
                        "drawer open_fraction exceeds threshold",
                        "drawer interior or target object is accessible if needed",
                    ],
                )
            ],
            repair_hints=[
                "Use several short strokes and remeasure progress after each stroke.",
                "If handle slips, return to sleeve_or_hook_handle.",
            ],
        ),
    ]


def _stove_button_stages(parsed: ParsedTask) -> list[StagePlan]:
    del parsed
    return [
        StagePlan(
            stage_id="stove_ground_button",
            name="Ground stove button",
            intent="Identify the stove control contact point and a safe push direction.",
            allowed_primitives=[],
            allowed_observations=["get_observation", "get_all_object_poses", "get_object_pose", "get_ee_pose"],
            target_references=["stove", "button_or_knob_pose", "push_direction"],
            family=TaskFamily.STOVE_BUTTON,
            max_attempts=1,
            checks=[
                StageCheck(
                    name="stove_contact_grounded",
                    description="The stove contact target is localized before physical pressing.",
                    check_type="hybrid",
                    success_criteria=[
                        "stove pose is available",
                        "button/knob contact point is reachable from current robot side",
                    ],
                )
            ],
            repair_hints=[
                "If button geometry is not explicit, use the visible stove front/knob side as contact target.",
            ],
        ),
        StagePlan(
            stage_id="stove_press_and_verify",
            name="Press stove button and verify",
            intent="Use short physical contact strokes to turn on the stove before continuing.",
            allowed_primitives=["goto_pose_checked", "move_linear_delta", "move_along_direction"],
            allowed_observations=["get_observation", "get_object_pose", "get_ee_pose"],
            target_references=["stove", "button_or_knob_pose", "stove_state_visual"],
            family=TaskFamily.STOVE_BUTTON,
            max_attempts=4,
            checks=[
                StageCheck(
                    name="stove_on_or_progress",
                    description="The stove control has visibly or diagnostically changed after pressing.",
                    check_type="hybrid",
                    success_criteria=[
                        "press motion contacts the stove control area",
                        "post-press observation indicates the stove is on or the control has moved",
                    ],
                )
            ],
            repair_hints=[
                "If no state change appears, adjust the contact point and repeat a short press.",
                "Do not place cookware until the stove stage is verified or marked uncertain with evidence.",
            ],
        ),
    ]


def _microwave_close_stages() -> list[StagePlan]:
    return [
        StagePlan(
            stage_id="microwave_close_guarded",
            name="Close microwave by physical sweep",
            intent="Close the microwave door using contact sweeps, then verify the microwave close diagnostic.",
            allowed_primitives=[
                "open_gripper",
                "goto_pose_checked",
                "move_linear_delta",
                "move_along_direction",
            ],
            allowed_observations=[
                "get_object_state_diagnostic",
                "get_body_pose",
                "get_task_status",
                "get_bddl_goal_diagnostics",
            ],
            target_references=["microwave", "microwave_door", "close"],
            family=TaskFamily.CONTAINER_PLACE,
            max_attempts=4,
            checks=[
                StageCheck(
                    name="microwave_closed",
                    description="The microwave close predicate is satisfied after physical contact motion.",
                    check_type="hybrid",
                    success_criteria=[
                        "get_object_state_diagnostic reports is_close for the microwave",
                        "task status or BDDL diagnostics indicate the close goal is satisfied",
                    ],
                )
            ],
            repair_hints=[
                "Use a short door-handle sweep and remeasure the close diagnostic before retrying.",
                "Do not write the microwave joint state directly.",
            ],
        )
    ]


def _placement_family(target_name: str, predicate: str, parsed: ParsedTask) -> TaskFamily:
    lowered = _normalize(target_name)
    if "rack" in lowered or "caddy" in lowered:
        return TaskFamily.RACK_OR_CADDY
    if "basket" in lowered or "microwave" in lowered or predicate == "in":
        return TaskFamily.CONTAINER_PLACE
    if any(family == TaskFamily.SPATIAL_RELATION for family in parsed.families):
        return TaskFamily.SPATIAL_RELATION
    return TaskFamily.SURFACE_PLACE


def _pair_objects_and_targets(
    objects: tuple[str, ...],
    targets: tuple[str, ...],
    parsed: ParsedTask,
) -> list[tuple[str, str, str]]:
    if not objects:
        return []
    task_text = _normalize(parsed.task)
    if parsed.drawer_level and any("drawer" in target for target in targets):
        drawer_target = f"{parsed.drawer_level} drawer"
        return [(obj, drawer_target, "in") for obj in objects]
    if "right of" in task_text and "plate" in task_text:
        pairs: list[tuple[str, str, str]] = []
        first_target = "plate"
        for index, obj in enumerate(objects):
            if index == 0:
                pairs.append((obj, first_target, "on"))
            else:
                pairs.append((obj, "right of plate", "right_of"))
        return pairs
    usable_targets = tuple(target for target in targets if target not in objects)
    if not usable_targets:
        usable_targets = ("task target",)
    predicate = parsed.predicates[0] if parsed.predicates else "on_or_in"
    pairs: list[tuple[str, str, str]] = []
    if len(usable_targets) >= len(objects):
        for obj, target in zip(objects, usable_targets, strict=False):
            pairs.append((obj, target, _predicate_for_target(predicate, target)))
    else:
        for obj in objects:
            pairs.append((obj, usable_targets[-1], _predicate_for_target(predicate, usable_targets[-1])))
    return pairs


def _order_pairs_by_interference_risk(
    pairs: list[tuple[str, str, str]],
    parsed: ParsedTask,
) -> list[tuple[str, str, str]]:
    """Place robust site/region goals before fragile object-support goals."""
    if len(pairs) <= 1:
        return pairs
    if TaskFamily.MULTI_OBJECT not in parsed.families:
        return pairs
    indexed = list(enumerate(pairs))
    indexed.sort(key=lambda item: (_placement_interference_rank(item[1]), item[0]))
    return [pair for _, pair in indexed]


def _placement_interference_rank(pair: tuple[str, str, str]) -> int:
    _object_name, target_name, predicate = pair
    target = _normalize(target_name)
    pred = str(predicate).strip().lower()
    if pred == "in" or any(token in target for token in ("basket", "drawer", "compartment", "microwave")):
        return 0
    if any(token in target for token in ("left plate", "right plate", "region", "side")):
        return 1
    if pred == "on" and any(token in target for token in ("plate", "stove", "bowl", "cabinet")):
        return 2
    return 1


def _predicate_for_target(default: str, target: str) -> str:
    lowered = _normalize(target)
    if lowered in {"basket", "bowl", "drawer", "microwave"} or "compartment" in lowered:
        return "in"
    if lowered in {"plate", "left plate", "right plate", "stove", "cabinet top", "cabinet", "rack", "caddy"}:
        return "on" if lowered not in {"rack", "caddy"} else "in_or_on"
    return default


def _looks_multi_object(lowered: str, objects: tuple[str, ...]) -> bool:
    if len(objects) >= 2:
        return True
    return bool(re.search(r"\b(two|both)\b", lowered))


def _needs_drawer_first(parsed: ParsedTask) -> bool:
    text = _normalize(parsed.task)
    return bool(
        "open" in text
        or "in top drawer" in text
        or "in the top drawer" in text
        or "in middle drawer" in text
        or "in the middle drawer" in text
        or "in bottom drawer" in text
        or "in the bottom drawer" in text
        or (
            "drawer" in text
            and any(token in text for token in ("put", "place", "inside", "into", " in "))
        )
    )


def _needs_drawer_after_object(parsed: ParsedTask) -> bool:
    text = _normalize(parsed.task)
    return "close" in text and "drawer" in text


def _needs_microwave_close(parsed: ParsedTask) -> bool:
    text = _normalize(parsed.task)
    return "microwave" in text and "close" in text


def _fallback_objects(task: str) -> tuple[str, ...]:
    lowered = _normalize(task)
    if "turn on stove" in lowered or "turn on the stove" in lowered:
        return ()
    if "object" in lowered:
        return ("task object",)
    return ()


def _fallback_targets(task: str) -> tuple[str, ...]:
    lowered = _normalize(task)
    for target in _KNOWN_TARGETS:
        if target in lowered:
            return (target,)
    return ()


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize(text)).strip("_") or "item"
