"""Pure planning and audit helpers for the zero-API D0 manner pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLAN_SCHEMA = "mannercap-d0-manner-plan-v1"
AUDIT_SCHEMA = "mannercap-d0-manner-audit-v1"
DEFAULT_TASKS = ("lift", "stack")
DEFAULT_TAGS = ("neutral", "slow", "fast", "fast_approach_slow_contact")

PRESETS: dict[str, dict[str, Any]] = {
    "neutral": {
        "description": "task-only neutral instruction",
        "global": {},
        "phase_velocity": {},
    },
    "slow": {
        "description": "global slow execution",
        "global": {"max_joint_vel": 0.08, "max_jerk": 2.0},
        "phase_velocity": {},
    },
    "fast": {
        "description": "global fast execution",
        "global": {"min_joint_vel": 0.18},
        "phase_velocity": {},
    },
    "fast_approach_slow_contact": {
        "description": "fast approach followed by slow contact and manipulation",
        "global": {"max_jerk": 2.0},
        "phase_velocity": {
            "approach": {"min_joint_vel": 0.18},
            "descend": {"max_joint_vel": 0.06},
            "lift": {"max_joint_vel": 0.08},
            "transport": {"max_joint_vel": 0.08},
            "final": {"max_joint_vel": 0.06},
        },
    },
}

TASK_PROMPTS = {
    "lift": {
        "neutral": "Lift the cube.",
        "slow": "Lift the cube slowly.",
        "fast": "Lift the cube quickly.",
        "fast_approach_slow_contact": (
            "Approach the cube quickly, then contact it and lift it slowly."
        ),
    },
    "stack": {
        "neutral": "Stack the red cube on the green cube.",
        "slow": "Stack the red cube on the green cube slowly.",
        "fast": "Stack the red cube on the green cube quickly.",
        "fast_approach_slow_contact": (
            "Approach the red cube quickly, then contact it and stack it slowly."
        ),
    },
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_strings(values: Iterable[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def _unique_seeds(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError("seeds must contain integers")
    if any(value < 0 or value >= 2**32 for value in result):
        raise ValueError("seeds must fit NumPy's [0, 2**32) range")
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return result


def build_plan(
    tasks: Iterable[str] = DEFAULT_TASKS,
    seeds: Iterable[int] = (42, 43),
    tags: Iterable[str] = DEFAULT_TAGS,
) -> dict[str, Any]:
    task_values = _unique_strings(tasks, "tasks")
    seed_values = _unique_seeds(seeds)
    tag_values = _unique_strings(tags, "tags")
    unknown_tasks = sorted(set(task_values) - set(TASK_PROMPTS))
    unknown_tags = sorted(set(tag_values) - set(PRESETS))
    if unknown_tasks:
        raise ValueError(f"unsupported tasks: {unknown_tasks}")
    if unknown_tags:
        raise ValueError(f"unsupported manner tags: {unknown_tags}")

    episodes: list[dict[str, Any]] = []
    for task in task_values:
        for seed in seed_values:
            pair_id = f"d0-{task}-seed-{seed}-modifier"
            for tag in tag_values:
                episodes.append(
                    {
                        "episode_id": f"d0-{task}-seed-{seed}-{tag}",
                        "task": task,
                        "seed": seed,
                        "manner_tag": tag,
                        "raw_language": TASK_PROMPTS[task][tag],
                        "requested_manner": PRESETS[tag],
                        "pair_id": pair_id,
                        "pair_type": "modifier_word",
                        "split": "development",
                        "api_calls": 0,
                        "student_action_expressibility": "osc_pose_speed_timing_only",
                    }
                )
    payload = {
        "schema": PLAN_SCHEMA,
        "purpose": "zero_api_d0_manner_pilot",
        "tasks": list(task_values),
        "seeds": list(seed_values),
        "manner_tags": list(tag_values),
        "episode_count": len(episodes),
        "episodes": episodes,
        "api_budget": 0,
        "outer_candidate_attempts_per_episode": 1,
        "formal_manner_thresholds_frozen": False,
        "generalization_claim_allowed": False,
        "excluded_conditions": {
            "gentle": "requires stiffness/grip-force side channels outside the fixed LA4VLA 7D action",
            "firm": "requires stiffness/grip-force side channels outside the fixed LA4VLA 7D action",
        },
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported D0 plan schema")
    saved = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if saved != _canonical_sha256(payload):
        raise ValueError("D0 plan SHA-256 does not match its contents")
    episodes = plan.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != plan.get("episode_count"):
        raise ValueError("D0 episode count mismatch")
    ids = [episode.get("episode_id") for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("D0 episode ids must be unique")


def audit_results(plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit execution completeness and same-seed initial-state pairing."""

    validate_plan(plan)
    by_id = {str(result.get("episode_id")): result for result in results}
    expected_ids = {str(episode["episode_id"]) for episode in plan["episodes"]}
    unexpected = sorted(set(by_id) - expected_ids)
    missing = sorted(expected_ids - set(by_id))
    pair_groups: dict[str, list[Mapping[str, Any]]] = {}
    for episode in plan["episodes"]:
        result = by_id.get(str(episode["episode_id"]))
        if result is not None:
            pair_groups.setdefault(str(episode["pair_id"]), []).append(result)

    pair_audit: list[dict[str, Any]] = []
    for pair_id, group in sorted(pair_groups.items()):
        state_hashes = {result.get("initial_state_sha256") for result in group}
        model_hashes = {result.get("compiled_model_sha256") for result in group}
        exact = (
            len(group) == len(plan["manner_tags"])
            and None not in state_hashes
            and None not in model_hashes
            and len(state_hashes) == 1
            and len(model_hashes) == 1
        )
        pair_audit.append(
            {
                "pair_id": pair_id,
                "observed_episode_count": len(group),
                "expected_episode_count": len(plan["manner_tags"]),
                "initial_state_hashes": sorted(str(value) for value in state_hashes),
                "compiled_model_hashes": sorted(str(value) for value in model_hashes),
                "same_initial_state_exact": exact,
            }
        )

    executed = [result for result in results if result.get("status") == "executed"]
    task_pass = [result for result in executed if result.get("task_completed") is True]
    return {
        "schema": AUDIT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "planned_episode_count": len(plan["episodes"]),
        "observed_result_count": len(results),
        "executed_count": len(executed),
        "execution_error_count": sum(result.get("status") == "execution_error" for result in results),
        "task_success_count": len(task_pass),
        "task_failure_count": len(executed) - len(task_pass),
        "missing_episode_ids": missing,
        "unexpected_episode_ids": unexpected,
        "same_initial_state_pair_count": sum(item["same_initial_state_exact"] for item in pair_audit),
        "pair_count": len(pair_audit),
        "pairs": pair_audit,
        "api_calls": sum(int(result.get("api_calls", 0)) for result in results),
        "formal_manner_qualification": "unknown",
        "promotion_allowed": False,
    }


def load_results(root: Path | str, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    root_path = Path(root)
    results: list[dict[str, Any]] = []
    for episode in plan.get("episodes", []):
        path = root_path / "episodes" / str(episode["episode_id"]) / "result.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                results.append(value)
    return results


__all__ = [
    "AUDIT_SCHEMA",
    "DEFAULT_TAGS",
    "DEFAULT_TASKS",
    "PLAN_SCHEMA",
    "PRESETS",
    "TASK_PROMPTS",
    "audit_results",
    "build_plan",
    "load_results",
    "validate_plan",
]
