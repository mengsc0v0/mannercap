"""Single versioned stage table shared by request, execution, telemetry and audit.

The stage ids are exactly those of ``configs/constraint_registry.yaml``.  Legacy
spellings (``final``, ``release``...) are accepted *only* through ``ALIASES`` and
are always rewritten to the canonical id; an unknown name is never guessed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Final, Optional

STAGE_TABLE_VERSION: Final[str] = "stage_table.v1"

STAGES: Final[dict[str, tuple[str, ...]]] = {
    "lift": ("approach", "descend", "grasp", "lift"),
    "stack": ("approach", "descend", "grasp", "lift", "transport", "place", "retreat"),
    "nut_assembly": ("approach", "descend", "grasp", "lift", "transport", "place", "retreat"),
    # wiping tool (no fingers): hover over the dirt, press down to contact, sweep the dirt path, leave
    "wipe": ("approach", "descend", "sweep", "retreat"),
}

# Stages in which each manner dimension can be requested / measured.
DIMENSION_STAGES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "speed": {
        "lift": ("approach", "descend", "lift"),
        "stack": ("approach", "descend", "lift", "transport", "place", "retreat"),
        "nut_assembly": ("approach", "descend", "lift", "transport", "place", "retreat"),
        # `descend` ends in a fixed-speed contact-seek; approach/retreat are repositioning at a fixed speed.
        # The wipe speed manner is the speed of the wiping stroke itself.
        "wipe": ("sweep",),
    },
    "grip": {"lift": ("grasp",), "stack": ("grasp",), "nut_assembly": ("grasp",)},
    # total tool-vs-table normal force while sweeping (wipe has no gripper)
    "press": {"wipe": ("sweep",)},
}

ALIASES: Final[dict[str, str]] = {
    "final": "place",
    "release": "place",
    "open": "place",
    "carry": "transport",
    "reach": "approach",
    "pick": "grasp",
    "close": "grasp",
    "raise": "lift",
    "leave": "retreat",
}


def stage_ids(task: str) -> tuple[str, ...]:
    try:
        return STAGES[task]
    except KeyError as exc:
        raise ValueError(f"unsupported task {task!r}") from exc


def canonical_stage(task: str, name: str) -> Optional[str]:
    """Return the canonical stage id, or ``None`` when the name is not defined."""
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    return key if key in stage_ids(task) else None


def dimension_stages(task: str, dimension: str) -> tuple[str, ...]:
    return DIMENSION_STAGES[dimension][task]


def stage_table_hash() -> str:
    blob = json.dumps(
        {"version": STAGE_TABLE_VERSION, "stages": STAGES, "dimension_stages": DIMENSION_STAGES, "aliases": ALIASES},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()
