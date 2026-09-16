"""Strict, side-effect free validation for the proposed MannerCAP v2 steps.

The current recorder and exporters use a useful *shape* convention, but they do
not yet carry enough information to prove that a window belongs to one reset
and one source attempt.  This module deliberately lives outside the old
recorder/exporter modules.  It can therefore be used as a gate while those
modules remain unchanged.

The contract treats one row as one transition::

    observation.state -- action --> next_observation.state

``timestamp`` is the start time of that transition and ``dt`` is its measured
duration.  A window must have contiguous frame indices, strictly increasing
timestamps, one reset id, and one immutable lineage.  Image references are
checked for a usable path/reference shape; they are only checked on disk when
``image_root`` is supplied.

This validator does not execute an environment and does not claim that an
action is replayable.  It only checks the evidence needed before attempting a
replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_ACTION_FIELD = "lerobot_action_candidate"
DEFAULT_STATE_DIM = 8
DEFAULT_RAW_ACTION_DIM = 8
DEFAULT_ACTION_DIM = 7

# These names are intentionally explicit.  In particular, ``action`` and
# ``action.raw_joint_gripper`` are retained separately so a downstream export
# cannot silently replace a raw controller command with a derived pose delta.
REQUIRED_STEP_FIELDS: Tuple[str, ...] = (
    "observation.state",
    "next_observation.state",
    "action",
    "action.raw_joint_gripper",
    "timestamp",
    "dt",
    "frame_index",
    "sim_step_index",
    "episode_index",
    "task_index",
)
REQUIRED_LINEAGE_FIELDS: Tuple[str, ...] = (
    "trajectory_id",
    "source_attempt_id",
    "lineage_id",
    "reset_id",
)
IMAGE_FIELDS: Tuple[str, ...] = (
    "observation.images.image",
    "observation.images.image2",
)


@dataclass(frozen=True)
class ContractIssue:
    """One actionable contract violation."""

    code: str
    index: Optional[int]
    field: Optional[str]
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for reports/manifests."""

        return {
            "code": self.code,
            "index": self.index,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class TrajectoryContractReport:
    """Validation result returned by :func:`validate_trajectory`."""

    step_count: int
    issues: Tuple[ContractIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    # ``ok`` is convenient for smoke gates and keeps callers from depending on
    # one particular spelling.
    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def errors(self) -> Tuple[ContractIssue, ...]:
        return self.issues

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def has_code(self, code: str) -> bool:
        return any(issue.code == code for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "step_count": self.step_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def raise_if_invalid(self) -> "TrajectoryContractReport":
        if not self.valid:
            details = "; ".join(issue.message for issue in self.issues[:8])
            if len(self.issues) > 8:
                details += f"; ... ({len(self.issues) - 8} more)"
            raise TrajectoryContractError(details)
        return self


class TrajectoryContractError(ValueError):
    """Raised by ``raise_if_invalid`` for callers that want fail-fast gating."""


def validate_window(
    rows: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> TrajectoryContractReport:
    """Validate one training/replay window.

    A window uses the same rules as a trajectory and therefore cannot include
    a reset marker or a change of ``reset_id``.  The alias makes the intended
    call site clear to dataset builders.
    """

    return validate_trajectory(rows, **kwargs)


def validate_trajectory(
    rows: Iterable[Mapping[str, Any]],
    *,
    action_field: str = DEFAULT_ACTION_FIELD,
    state_dim: int = DEFAULT_STATE_DIM,
    raw_action_dim: int = DEFAULT_RAW_ACTION_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    require_images: bool = True,
    require_lineage: bool = True,
    require_actual_dt: bool = True,
    image_root: Optional[str | Path] = None,
    expected_lineage: Optional[Mapping[str, Any]] = None,
    time_tolerance: float = 1e-6,
) -> TrajectoryContractReport:
    """Check a sequence of transition records without mutating it.

    Parameters are deliberately explicit so a future action schema can be
    checked without changing the old modules.  The defaults match the current
    LIBERO recorder's state/action dimensions (8/8 raw/7 candidate) while
    requiring the additional v2 timing and lineage evidence.

    ``expected_lineage`` may contain any of ``trajectory_id``,
    ``source_attempt_id``, ``lineage_id``, ``reset_id``, ``episode_index`` or
    ``task_index``.  Values in every row must match those expectations.
    """

    issues: list[ContractIssue] = []
    try:
        materialized = list(rows)
    except TypeError:
        return TrajectoryContractReport(
            step_count=0,
            issues=(
                ContractIssue(
                    "invalid_rows",
                    None,
                    None,
                    "trajectory rows must be an iterable of mappings",
                ),
            ),
        )

    if not materialized:
        return TrajectoryContractReport(
            step_count=0,
            issues=(ContractIssue("empty_trajectory", None, None, "trajectory has no steps"),),
        )

    def add(code: str, index: Optional[int], field: Optional[str], message: str) -> None:
        issues.append(ContractIssue(code, index, field, message))

    normalized: list[Optional[Mapping[str, Any]]] = []
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            add("not_mapping", index, None, f"step {index} is not a mapping")
            normalized.append(None)
        else:
            normalized.append(row)

    required = list(REQUIRED_STEP_FIELDS)
    if not require_actual_dt and "dt" in required:
        required.remove("dt")
    if require_lineage:
        required.extend(REQUIRED_LINEAGE_FIELDS)
    if require_images:
        required.extend(IMAGE_FIELDS)
    # A caller may intentionally validate raw-only data by selecting action as
    # the export action.  Avoid duplicate missing-field diagnostics then.
    required_actions = {"action", "action.raw_joint_gripper", action_field}
    for index, row in enumerate(normalized):
        if row is None:
            continue
        for field in required:
            if field not in row or row[field] is None:
                add("missing_field", index, field, f"step {index} is missing required field {field!r}")
        for field in required_actions:
            if field not in row or row[field] is None:
                add("missing_field", index, field, f"step {index} is missing required action field {field!r}")

        for field in ("observation.state", "next_observation.state"):
            if field in row and row[field] is not None:
                _validate_numeric_vector(
                    row[field],
                    expected=state_dim,
                    index=index,
                    field=field,
                    add=add,
                )
        for field, expected in (
            ("action", raw_action_dim),
            ("action.raw_joint_gripper", raw_action_dim),
            (action_field, action_dim),
        ):
            if field in row and row[field] is not None:
                _validate_numeric_vector(
                    row[field],
                    expected=expected,
                    index=index,
                    field=field,
                    add=add,
                )

        if require_images:
            for field in IMAGE_FIELDS:
                if field in row and row[field] is not None:
                    _validate_image_ref(
                        row[field],
                        index=index,
                        field=field,
                        image_root=image_root,
                        add=add,
                    )

        for field in ("timestamp", "dt"):
            if field in row and row[field] is not None:
                value = _number(row[field])
                if value is None:
                    add(
                        "non_numeric_time",
                        index,
                        field,
                        f"step {index} field {field!r} must be a finite number",
                    )
                elif field == "dt" and value <= 0:
                    add(
                        "invalid_dt",
                        index,
                        field,
                        f"step {index} dt must be > 0, got {value!r}",
                    )

        for field in ("frame_index", "sim_step_index"):
            if field in row and row[field] is not None and not _is_integer(row[field]):
                add(
                    "invalid_index",
                    index,
                    field,
                    f"step {index} field {field!r} must be an integer",
                )

        if require_lineage:
            for field in REQUIRED_LINEAGE_FIELDS:
                if field in row and row[field] is not None and not _nonempty_id(row[field]):
                    add(
                        "invalid_lineage",
                        index,
                        field,
                        f"step {index} field {field!r} must be a non-empty string/id",
                    )

        # A reset marker is evidence that the row is a boundary, not part of a
        # replay window.  Support the common spellings without prescribing a
        # new recorder implementation.
        for marker in ("reset", "reset_event", "is_reset"):
            if bool(row.get(marker, False)):
                add(
                    "reset_marker",
                    index,
                    marker,
                    f"step {index} contains {marker}=true; windows must stop before reset",
                )

    timestamps = [_number(row.get("timestamp")) if row is not None else None for row in normalized]
    dts = [_number(row.get("dt")) if row is not None else None for row in normalized]
    frame_indices = [row.get("frame_index") if row is not None else None for row in normalized]
    sim_indices = [row.get("sim_step_index") if row is not None else None for row in normalized]

    first_timestamp: Optional[float] = None
    for index, timestamp in enumerate(timestamps):
        if timestamp is None:
            continue
        if first_timestamp is None:
            first_timestamp = timestamp
            continue
        previous = timestamps[index - 1]
        if previous is None:
            continue
        delta = timestamp - previous
        if delta <= time_tolerance:
            add(
                "non_monotonic_time",
                index,
                "timestamp",
                f"step {index} timestamp must increase strictly; previous={previous!r}, current={timestamp!r}",
            )
        current_dt = dts[index - 1]
        if current_dt is not None and abs(delta - current_dt) > time_tolerance:
            add(
                "dt_mismatch",
                index - 1,
                "dt",
                f"step {index - 1} dt={current_dt!r} does not match timestamp delta={delta!r}",
            )

    # Validate the final row's duration independently; there is no following
    # timestamp for it, but it still represents a transition.
    for index, dt in enumerate(dts):
        if dt is not None and dt <= 0:
            # The per-row pass already emits this issue; avoid a duplicate.
            continue

    previous_frame: Optional[int] = None
    previous_sim: Optional[int] = None
    for index, frame in enumerate(frame_indices):
        if not _is_integer(frame):
            continue
        frame_int = int(frame)
        if previous_frame is None:
            if frame_int != 0:
                add(
                    "non_contiguous_frame",
                    index,
                    "frame_index",
                    f"first frame_index must be 0, got {frame_int}",
                )
        elif frame_int != previous_frame + 1:
            add(
                "non_contiguous_frame",
                index,
                "frame_index",
                f"frame_index must increase by 1; previous={previous_frame}, current={frame_int}",
            )
        previous_frame = frame_int

    for index, sim in enumerate(sim_indices):
        if not _is_integer(sim):
            continue
        sim_int = int(sim)
        if previous_sim is not None and sim_int <= previous_sim:
            add(
                "non_monotonic_sim_step",
                index,
                "sim_step_index",
                f"sim_step_index must increase; previous={previous_sim}, current={sim_int}",
            )
        previous_sim = sim_int

    if require_lineage:
        _validate_constant_fields(normalized, REQUIRED_LINEAGE_FIELDS, add)
    _validate_constant_fields(normalized, ("episode_index", "task_index"), add)

    if expected_lineage:
        for field, expected in expected_lineage.items():
            if field not in REQUIRED_LINEAGE_FIELDS and field not in ("episode_index", "task_index"):
                continue
            for index, row in enumerate(normalized):
                if row is not None and field in row and row[field] != expected:
                    add(
                        "lineage_metadata_mismatch",
                        index,
                        field,
                        f"step {index} field {field!r}={row[field]!r} does not match expected {expected!r}",
                    )

    return TrajectoryContractReport(step_count=len(materialized), issues=tuple(issues))


def _validate_numeric_vector(
    value: Any,
    *,
    expected: int,
    index: int,
    field: str,
    add: Any,
) -> None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        add("invalid_vector", index, field, f"step {index} field {field!r} must be a numeric sequence")
        return
    try:
        length = len(value)
        values = iter(value)
    except (TypeError, ValueError):
        add("invalid_vector", index, field, f"step {index} field {field!r} must be a numeric sequence")
        return
    if length != expected:
        add(
            "bad_dimension",
            index,
            field,
            f"step {index} field {field!r} has dimension {length}; expected {expected}",
        )
    for component_index, component in enumerate(values):
        if _number(component) is None:
            add(
                "nonfinite_value",
                index,
                field,
                f"step {index} field {field!r}[{component_index}] must be finite numeric",
            )


def _validate_image_ref(
    value: Any,
    *,
    index: int,
    field: str,
    image_root: Optional[str | Path],
    add: Any,
) -> None:
    path_value: Any
    if isinstance(value, Mapping):
        path_value = value.get("path")
        if path_value in (None, "") and value.get("bytes") in (None, b""):
            add(
                "invalid_image_ref",
                index,
                field,
                f"step {index} image reference {field!r} needs a path or bytes",
            )
            return
    elif isinstance(value, (str, Path)):
        path_value = value
    else:
        add(
            "invalid_image_ref",
            index,
            field,
            f"step {index} image reference {field!r} must be a path or mapping",
        )
        return

    if image_root is not None and path_value not in (None, ""):
        candidate = Path(image_root) / str(path_value)
        if not candidate.is_file():
            add(
                "missing_image",
                index,
                field,
                f"step {index} image path does not exist: {candidate}",
            )


def _validate_constant_fields(
    rows: Sequence[Optional[Mapping[str, Any]]],
    fields: Sequence[str],
    add: Any,
) -> None:
    for field in fields:
        reference: Any = None
        reference_index: Optional[int] = None
        for index, row in enumerate(rows):
            if row is None or field not in row or row[field] is None:
                continue
            if reference_index is None:
                reference = row[field]
                reference_index = index
            elif row[field] != reference:
                code = "cross_reset_window" if field == "reset_id" else "lineage_mismatch"
                add(
                    code,
                    index,
                    field,
                    f"step {index} field {field!r} changed from {reference!r} at step {reference_index}",
                )


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() != ""


__all__ = [
    "ContractIssue",
    "TrajectoryContractError",
    "TrajectoryContractReport",
    "DEFAULT_ACTION_FIELD",
    "REQUIRED_STEP_FIELDS",
    "REQUIRED_LINEAGE_FIELDS",
    "IMAGE_FIELDS",
    "validate_trajectory",
    "validate_window",
]
