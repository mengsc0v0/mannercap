"""Zero-API development candidate generators; these are not qualified baselines.

The finite speed domain is a candidate search domain, never an acceptance threshold.
Only public request fields are used. No simulator outcomes enter proposals.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from typing import Any, Mapping

from synthesis_v2.d0_manner import PRESETS, TASK_PROMPTS

METHODS = ("G0", "G1", "G2", "G3", "M")
SPEED_CHOICES = ("slow", "neutral", "fast")
SPEED_BOUNDS = {
    "slow": {"max_joint_vel": 0.08},
    "neutral": {},
    "fast": {"min_joint_vel": 0.18},
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def propose(
    request: Mapping[str, Any], method_id: str, attempt_index: int = 1,
) -> dict[str, Any]:
    """Generate a reproducible candidate without mutating the original request.

    G1 ignores instruction content when sampling. G2 reads only the requested
    phase scopes (or the global scope), and samples their speed independently.
    G3 maps the explicit public manner tag to the declared D0 preset. This does
    not test adaptation to physical context: no physical property is varied.
    Attempts are one-based; a new attempt need not yield a unique candidate.
    """
    if method_id not in METHODS:
        raise ValueError(f"unsupported method_id: {method_id}")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        raise ValueError("attempt_index must be a positive integer")
    task, seed = request.get("task"), request.get("seed")
    if task not in TASK_PROMPTS:
        raise ValueError("unsupported task")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    tag = request.get("manner_tag")
    if tag not in PRESETS:
        raise ValueError("unsupported manner_tag")
    requested = request.get("requested_manner")
    if not isinstance(requested, Mapping) or requested != PRESETS[tag]:
        raise ValueError("requested_manner must match the declared D0 manner_tag")
    if not isinstance(request.get("raw_language"), str) or not request["raw_language"].strip():
        raise ValueError("raw_language must be non-empty")

    # Deliberately exclude manner, language, request ID and outcomes from RNG input.
    proposal_seed = int(_sha(["e0-development-v1", task, seed, method_id, attempt_index])[:16], 16)
    rng = random.Random(proposal_seed)
    candidate: dict[str, Any] | None
    selected: dict[str, str] = {}
    if method_id == "G0":
        candidate = deepcopy(PRESETS["neutral"])
        source = "development_fixed_neutral"
    elif method_id == "G1":
        choice = rng.choice(SPEED_CHOICES)
        candidate = deepcopy(PRESETS[choice])
        selected["global"] = choice
        source = "development_global_random_instruction_blind"
    elif method_id == "G2":
        scopes = sorted(requested["phase_velocity"])
        candidate = {"description": "development stage random velocity", "global": {}, "phase_velocity": {}}
        for scope in scopes or ["global"]:
            choice = rng.choice(SPEED_CHOICES)
            selected[scope] = choice
            bounds = deepcopy(SPEED_BOUNDS[choice])
            if scope == "global":
                candidate["global"] = bounds
            else:
                candidate["phase_velocity"][scope] = bounds
        source = "development_random_in_public_request_scopes"
    elif method_id == "G3":
        candidate = deepcopy(PRESETS[tag])
        source = "development_public_manner_rule_no_physical_context_variation"
    else:
        candidate = None
        source = "llm_api_not_configured"
    skipped = method_id == "M"
    return {
        "schema": "mannercap-e0-development-proposal-v1",
        "request_id": request.get("request_id", request.get("episode_id")),
        "request_sha256": _sha(dict(request)),
        "requested_manner_sha256": _sha(requested),
        "method_id": method_id,
        "attempt_index": attempt_index,
        "proposal_seed": proposal_seed,
        "status": "skipped" if skipped else "proposed",
        "proposal_source": source,
        "candidate_preset": candidate,
        "selected_speed_choices": selected,
        "call_record": {"api_calls": 0, "status": "skipped" if skipped else "not_required",
                        "reason": "api_not_configured" if skipped else "pure_local_proposer"},
        "development_only": True,
        "formal_thresholds_frozen": False,
        "promotion_allowed": False,
        "physical_context_variation": False,
        "context_effect_claim_allowed": False,
    }


__all__ = ["METHODS", "SPEED_CHOICES", "propose"]
