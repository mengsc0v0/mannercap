"""Offline M proposal boundary. Contains no network client or credentials.

The first model smoke test shares the finite velocity domain of G1/G2.
This contract tests explicit instruction planning only, not physical-context
reasoning. Calibration thresholds and final-audit outputs are never supplied.
"""
from __future__ import annotations

import json
from copy import deepcopy

from synthesis_v2.e0_proposers import SPEED_BOUNDS

TASK_PHASES = {
    "lift": ("approach", "descend", "lift"),
    "stack": ("approach", "descend", "lift", "transport", "final", "retreat"),
}

V2_CONTRACT_VERSION = "e0-m-velocity-plan-v2"
UNSUPPORTED_REQUEST_CONSTRAINTS = (
    "max_jerk",
    "stiffness",
    "grip_force",
    "contact_force",
)
REQUEST_FIELD_UNSUPPORTED_NAME = {
    "max_jerk": "max_jerk",
    "kp_max": "stiffness",
    "kp_min": "stiffness",
    "grip_fraction_max": "grip_force",
    "grip_fraction_min": "grip_force",
}


def apply_explicit_neutral_phases(params: dict, candidate: dict, neutral_params: dict) -> dict:
    """An explicit empty phase resets speed/acceleration to the neutral profile.

    Omitted phases inherit global settings. Only phase velocity and acceleration
    are reset; this does not invent force or safety semantics.
    """
    result = deepcopy(params)
    for phase, bounds in candidate['phase_velocity'].items():
        if bounds == {}:
            for field in ('phase_v_max', 'phase_a_max'):
                result[field][phase] = neutral_params[field][phase]
    return result


def build_prompt(request: dict) -> dict:
    task = request["task"]
    if task not in TASK_PHASES:
        raise ValueError("unsupported task")
    return {
        "contract_version": "e0-m-velocity-plan-v1",
        "task": task,
        "instruction": request["raw_language"],
        "public_phase_names": list(TASK_PHASES[task]),
        "public_candidate_domain": deepcopy(SPEED_BOUNDS),
        "required_response": {
            "global_speed": "one of slow, neutral, fast",
            "phase_speeds": "object mapping applicable phase names to slow, neutral, fast",
        },
        "instructions": (
            "Return only a JSON object with global_speed and phase_speeds. "
            "Choose a global speed and optional overrides for task phases to follow "
            "the instruction. These are planning parameters, not safety guarantees."
        ),
    }


def build_prompt_v2(request: dict) -> dict:
    """Build the revised prompt with explicit scope and unsupported fields.

    The v1 prompt is retained for provenance of the cached calls.  New calls
    should use this contract: omitted phases inherit the global profile, while
    a phase value of ``neutral`` is an explicit neutral override.  A model must
    also declare request semantics that the velocity-only candidate cannot
    represent; this prevents silent loss of jerk/contact constraints.
    """
    prompt = build_prompt(request)
    prompt["contract_version"] = V2_CONTRACT_VERSION
    prompt["required_response"] = {
        "global_speed": "one of slow, neutral, fast",
        "phase_speeds": (
            "object containing only phases explicitly assigned a speed in the instruction; "
            "omit phases that should inherit global_speed"
        ),
        "unsupported_request_constraints": (
            "array of zero or more names from the public unsupported constraint domain"
        ),
    }
    prompt["unsupported_constraint_domain"] = list(UNSUPPORTED_REQUEST_CONSTRAINTS)
    prompt["phase_scope_semantics"] = {
        "omitted_phase": "inherit_global_speed",
        "neutral_phase": "explicit_neutral_override",
        "do_not_repeat_global": True,
    }
    prompt["instructions"] = (
        "Return only a JSON object with exactly global_speed, phase_speeds, and "
        "unsupported_request_constraints. Set global_speed from the instruction. "
        "Include a phase in phase_speeds only when the instruction explicitly assigns "
        "that phase a speed; omit unmentioned phases so they inherit global_speed. "
        "Use neutral for an explicitly neutral phase, which is different from omission. "
        "List every requested constraint outside this velocity-only contract (for example "
        "max_jerk, stiffness, grip_force, or contact_force) in "
        "unsupported_request_constraints; use [] only when none is requested. "
        "These are planning parameters, not safety guarantees."
    )
    return prompt


def parse_response(text: str, task: str) -> dict:
    """Reject malformed, extra-field, and out-of-domain model output."""
    if task not in TASK_PHASES:
        raise ValueError("unsupported task")
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {"global_speed", "phase_speeds"}:
        raise ValueError("response must contain exactly global_speed and phase_speeds")
    speed = value["global_speed"]
    phases = value["phase_speeds"]
    if not isinstance(speed, str) or speed not in SPEED_BOUNDS or not isinstance(phases, dict):
        raise ValueError("invalid speed or phase mapping")
    for phase, setting in phases.items():
        if phase not in TASK_PHASES[task] or not isinstance(setting, str) or setting not in SPEED_BOUNDS:
            raise ValueError("phase or speed outside public domain")
    return {"description": "M explicit instruction velocity smoke candidate",
            "global": deepcopy(SPEED_BOUNDS[speed]),
            "phase_velocity": {key: deepcopy(SPEED_BOUNDS[val]) for key, val in phases.items()}}


def parse_response_v2(text: str, task: str) -> dict:
    """Parse the revised contract without changing historical v1 semantics."""
    if task not in TASK_PHASES:
        raise ValueError("unsupported task")

    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    value = json.loads(text, object_pairs_hook=unique_object)
    required = {"global_speed", "phase_speeds", "unsupported_request_constraints"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("response must contain exactly the v2 contract fields")
    speed = value["global_speed"]
    phases = value["phase_speeds"]
    unsupported = value["unsupported_request_constraints"]
    if not isinstance(speed, str) or speed not in SPEED_BOUNDS or not isinstance(phases, dict):
        raise ValueError("invalid speed or phase mapping")
    if (
        not isinstance(unsupported, list)
        or any(not isinstance(item, str) for item in unsupported)
        or len(set(unsupported)) != len(unsupported)
    ):
        raise ValueError("unsupported_request_constraints must be a unique array")
    if any(item not in UNSUPPORTED_REQUEST_CONSTRAINTS for item in unsupported):
        raise ValueError("unsupported constraint outside public domain")
    for phase, setting in phases.items():
        if phase not in TASK_PHASES[task] or not isinstance(setting, str) or setting not in SPEED_BOUNDS:
            raise ValueError("phase or speed outside public domain")
    return {
        "description": "M explicit instruction velocity plan v2",
        "contract_version": V2_CONTRACT_VERSION,
        "global": deepcopy(SPEED_BOUNDS[speed]),
        "phase_velocity": {key: deepcopy(SPEED_BOUNDS[val]) for key, val in phases.items()},
        "unsupported_request_constraints": list(unsupported),
        "phase_scope_policy": "omit_unmentioned_inherit_global",
    }


def expected_v2_unsupported_constraints(request: dict) -> list[str]:
    """Return contract names required by structured request constraints."""
    requested = request.get("requested_manner")
    if not isinstance(requested, dict):
        raise ValueError("request.requested_manner is required for v2 coverage validation")
    fields: set[str] = set()
    global_spec = requested.get("global", {})
    phase_spec = requested.get("phase_velocity", {})
    if not isinstance(global_spec, dict) or not isinstance(phase_spec, dict):
        raise ValueError("request requested_manner fields must be objects")
    fields.update(global_spec)
    for phase, bounds in phase_spec.items():
        if not isinstance(bounds, dict):
            raise ValueError(f"request phase_velocity.{phase} must be an object")
        fields.update(bounds)
    names = set()
    for field in fields:
        if field in {"max_joint_vel", "min_joint_vel"}:
            continue
        if field not in REQUEST_FIELD_UNSUPPORTED_NAME:
            raise ValueError(f"request field {field!r} has no v2 unsupported-constraint name")
        names.add(REQUEST_FIELD_UNSUPPORTED_NAME[field])
    return sorted(names)


def validate_v2_coverage(candidate: dict, request: dict) -> dict:
    """Reject silent loss or invention of constraints outside velocity scope."""
    if candidate.get("contract_version") != V2_CONTRACT_VERSION:
        raise ValueError("candidate does not use the v2 contract")
    declared = candidate.get("unsupported_request_constraints")
    if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
        raise ValueError("candidate unsupported constraint declaration is invalid")
    expected = expected_v2_unsupported_constraints(request)
    if sorted(set(declared)) != expected or len(declared) != len(set(declared)):
        raise ValueError(
            "candidate unsupported constraints do not cover the structured request: "
            f"expected {expected}, got {declared}"
        )
    return candidate
