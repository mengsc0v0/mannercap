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
