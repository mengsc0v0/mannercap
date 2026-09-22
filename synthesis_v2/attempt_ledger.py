"""Pure-Python candidate-attempt and resource ledger for MannerCAP v2.

The ledger is deliberately separate from the simulator and model callers.  It
records one explicit candidate attempt at a time, including parse failures and
other candidates that never reach execution.  No retry loop is hidden here:
the caller must submit each attempt, all attempts consume the shared
``max_attempts`` budget, and a request becomes terminal after acceptance or
budget exhaustion.

The request snapshot is copied when it is registered and copied again into
each attempt record.  A later mutation of a caller-owned manifest object cannot
rewrite the parent request or its original constraints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, TypeAlias


JSONValue: TypeAlias = Any


class LedgerError(ValueError):
    """Raised when an attempt or resource record violates the ledger contract."""


class AttemptOutcome(str, Enum):
    """Every candidate outcome that consumes an attempt number."""

    PARSE_FAILURE = "parse_failure"
    PARSE_ERROR = "parse_failure"
    INVALID = "invalid"
    INVALID_PARAMETER = "invalid"
    EXECUTION_FAILURE = "execution_failure"
    EXECUTION_ERROR = "execution_failure"
    VALIDATION_FAILURE = "validation_failure"
    VALIDATION_ERROR = "validation_failure"
    DUPLICATE = "duplicate"
    DUPLICATE_CANDIDATE = "duplicate"
    ACCEPTED = "accepted"
    ACCEPT = "accepted"


class RequestState(str, Enum):
    """Lifecycle state of one registered request."""

    OPEN = "open"
    ACCEPTED = "accepted"
    BUDGET_EXHAUSTED = "budget_exhausted"


_OUTCOME_ALIASES: Final[dict[str, str]] = {
    "parse_failure": AttemptOutcome.PARSE_FAILURE.value,
    "parse_failed": AttemptOutcome.PARSE_FAILURE.value,
    "parse_error": AttemptOutcome.PARSE_FAILURE.value,
    "invalid": AttemptOutcome.INVALID.value,
    "invalid_parameter": AttemptOutcome.INVALID.value,
    "invalid_parameters": AttemptOutcome.INVALID.value,
    "illegal": AttemptOutcome.INVALID.value,
    "illegal_parameter": AttemptOutcome.INVALID.value,
    "execution_failure": AttemptOutcome.EXECUTION_FAILURE.value,
    "execution_failed": AttemptOutcome.EXECUTION_FAILURE.value,
    "execution_error": AttemptOutcome.EXECUTION_FAILURE.value,
    "validation_failure": AttemptOutcome.VALIDATION_FAILURE.value,
    "validation_failed": AttemptOutcome.VALIDATION_FAILURE.value,
    "validation_error": AttemptOutcome.VALIDATION_FAILURE.value,
    "duplicate": AttemptOutcome.DUPLICATE.value,
    "duplicate_candidate": AttemptOutcome.DUPLICATE.value,
    "accepted": AttemptOutcome.ACCEPTED.value,
    "accept": AttemptOutcome.ACCEPTED.value,
}

_LEDGER_SCHEMA_VERSION: Final[str] = "synthesis_v2.attempt.v1"
_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "physics_substeps",
    "api_calls",
    "api_tokens",
    "cpu_time_s",
    "gpu_time_s",
    "wall_time_s",
    "storage_bytes",
)
_RESOURCE_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "physics_substeps": ("physics_substeps", "sim_physics_substeps", "sim_steps"),
    "api_calls": ("api_calls", "model_calls"),
    "api_tokens": ("api_tokens", "tokens", "model_tokens"),
    "cpu_time_s": ("cpu_time_s", "cpu_seconds", "cpu_s"),
    "gpu_time_s": ("gpu_time_s", "gpu_seconds", "gpu_s"),
    "wall_time_s": ("wall_time_s", "wall_seconds", "wall_s"),
    "storage_bytes": ("storage_bytes", "storage", "bytes"),
}


def _canonical_json(value: JSONValue) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError("ledger values must be JSON-compatible") from exc


def _json_copy(value: JSONValue, *, field_name: str) -> JSONValue:
    try:
        copied = deepcopy(value)
        _canonical_json(copied)
    except (TypeError, ValueError, LedgerError) as exc:
        raise LedgerError(f"{field_name} must be JSON-compatible") from exc
    return copied


def _content_hash(value: JSONValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{field_name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise LedgerError(f"{field_name} must be non-negative")
    return value


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise LedgerError(f"{field_name} must be a non-negative finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LedgerError(f"{field_name} must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise LedgerError(f"{field_name} must be non-negative and finite")
    return number


def _normalize_outcome(value: str | AttemptOutcome | None) -> str:
    if isinstance(value, AttemptOutcome):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise LedgerError("outcome/status must be an explicit non-empty value")
    normalized = _OUTCOME_ALIASES.get(value.strip().lower())
    if normalized is None:
        allowed = ", ".join(item.value for item in AttemptOutcome)
        raise LedgerError(f"unknown attempt outcome {value!r}; expected one of {allowed}")
    return normalized


@dataclass(frozen=True)
class ResourceUsage:
    """Non-negative resource vector charged to exactly one candidate attempt."""

    physics_substeps: int
    api_calls: int
    api_tokens: int
    cpu_time_s: float
    gpu_time_s: float
    wall_time_s: float
    storage_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "physics_substeps", _nonnegative_int(self.physics_substeps, "physics_substeps")
        )
        object.__setattr__(self, "api_calls", _nonnegative_int(self.api_calls, "api_calls"))
        object.__setattr__(self, "api_tokens", _nonnegative_int(self.api_tokens, "api_tokens"))
        object.__setattr__(self, "cpu_time_s", _nonnegative_float(self.cpu_time_s, "cpu_time_s"))
        object.__setattr__(self, "gpu_time_s", _nonnegative_float(self.gpu_time_s, "gpu_time_s"))
        object.__setattr__(self, "wall_time_s", _nonnegative_float(self.wall_time_s, "wall_time_s"))
        object.__setattr__(
            self, "storage_bytes", _nonnegative_int(self.storage_bytes, "storage_bytes")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceUsage":
        """Build a vector while rejecting missing fields and negative values."""

        if not isinstance(value, Mapping):
            raise LedgerError("resources must be a mapping or ResourceUsage")
        missing = [
            field_name
            for field_name in _RESOURCE_FIELDS
            if not any(alias in value and value[alias] is not None for alias in _RESOURCE_ALIASES[field_name])
        ]
        if missing:
            raise LedgerError("resources missing required field(s): " + ", ".join(missing))

        def read(field_name: str) -> Any:
            for alias in _RESOURCE_ALIASES[field_name]:
                if alias in value and value[alias] is not None:
                    return value[alias]
            # ``missing`` above guarantees this branch is unreachable.
            raise LedgerError(f"resources missing required field {field_name!r}")

        return cls(*(read(field_name) for field_name in _RESOURCE_FIELDS))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceUsage":
        """Alias for callers that use dict-oriented schema helpers."""

        return cls.from_mapping(value)

    @property
    def sim_physics_substeps(self) -> int:
        return self.physics_substeps

    @property
    def cpu_seconds(self) -> float:
        return self.cpu_time_s

    @property
    def gpu_seconds(self) -> float:
        return self.gpu_time_s

    @property
    def wall_seconds(self) -> float:
        return self.wall_time_s

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "physics_substeps": self.physics_substeps,
            "api_calls": self.api_calls,
            "api_tokens": self.api_tokens,
            "cpu_time_s": self.cpu_time_s,
            "gpu_time_s": self.gpu_time_s,
            "wall_time_s": self.wall_time_s,
            "storage_bytes": self.storage_bytes,
        }

    as_dict = to_dict


@dataclass(frozen=True)
class RequestSnapshot:
    """Immutable-by-copy request binding used by the attempt ledger."""

    request_id: str
    request: JSONValue
    request_hash: str

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, JSONValue] | Any,
        *,
        request_id: str | None = None,
    ) -> "RequestSnapshot":
        if hasattr(value, "request_id") and not isinstance(value, Mapping):
            inferred_id = str(value.request_id)
            try:
                request_value = value.to_manifest_dict()
            except AttributeError:
                request_value = {
                    "request_id": inferred_id,
                    "request": repr(value),
                }
        elif isinstance(value, Mapping):
            request_value = _json_copy(value, field_name="request")
            inferred_id = request_value.get("request_id") if isinstance(request_value, Mapping) else None
        else:
            raise LedgerError("request snapshot must be a mapping or RequestDraft-like object")

        resolved_id = request_id if request_id is not None else inferred_id
        if resolved_id is None:
            raise LedgerError("request snapshot is missing request_id")
        resolved_id = _non_empty_string(resolved_id, "request_id")
        if isinstance(request_value, Mapping) and "request_id" in request_value:
            snapshot_id = _non_empty_string(request_value["request_id"], "request_id")
            if snapshot_id != resolved_id:
                raise LedgerError(
                    f"request_id {resolved_id!r} does not match snapshot ID {snapshot_id!r}"
                )
        request_copy = _json_copy(request_value, field_name="request")
        return cls(resolved_id, request_copy, _content_hash(request_copy))


@dataclass(frozen=True)
class AttemptRecord:
    """One explicit candidate attempt and its charged resources."""

    attempt_id: str
    parent_request_id: str
    attempt_index: int
    outcome: str
    proposal_source: str
    candidate_plan: JSONValue
    call_record: JSONValue
    resources: ResourceUsage
    reason: str
    request_snapshot: JSONValue
    request_snapshot_hash: str
    duplicate_of_attempt_id: str | None = None
    request_state_after: str = RequestState.OPEN.value
    metadata: JSONValue = None

    def __post_init__(self) -> None:
        _non_empty_string(self.attempt_id, "attempt_id")
        _non_empty_string(self.parent_request_id, "parent_request_id")
        attempt_index = _nonnegative_int(self.attempt_index, "attempt_index")
        if attempt_index == 0:
            raise LedgerError("attempt_index must start at 1")
        normalized_outcome = _normalize_outcome(self.outcome)
        proposal_source = _non_empty_string(self.proposal_source, "proposal_source")
        reason = _non_empty_string(self.reason, "reason")
        if not isinstance(self.call_record, Mapping):
            raise LedgerError("call_record must be a JSON object")
        resources = (
            self.resources
            if isinstance(self.resources, ResourceUsage)
            else ResourceUsage.from_mapping(self.resources)
        )
        request_snapshot = _json_copy(self.request_snapshot, field_name="request_snapshot")
        request_snapshot_hash = _non_empty_string(
            self.request_snapshot_hash, "request_snapshot_hash"
        )
        if _content_hash(request_snapshot) != request_snapshot_hash:
            raise LedgerError("request_snapshot_hash does not match request_snapshot")
        duplicate_id = self.duplicate_of_attempt_id
        if duplicate_id is not None:
            duplicate_id = _non_empty_string(duplicate_id, "duplicate_of_attempt_id")
        if normalized_outcome == AttemptOutcome.DUPLICATE.value and duplicate_id is None:
            raise LedgerError("duplicate outcome requires duplicate_of_attempt_id")
        if normalized_outcome != AttemptOutcome.DUPLICATE.value and duplicate_id is not None:
            raise LedgerError("duplicate_of_attempt_id is only valid for duplicate outcomes")
        state = _non_empty_string(self.request_state_after, "request_state_after")
        if state not in {item.value for item in RequestState}:
            raise LedgerError(f"unknown request_state_after {state!r}")
        object.__setattr__(self, "attempt_index", attempt_index)
        object.__setattr__(self, "outcome", normalized_outcome)
        object.__setattr__(self, "proposal_source", proposal_source)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "candidate_plan", _json_copy(self.candidate_plan, field_name="candidate_plan"))
        object.__setattr__(self, "call_record", _json_copy(dict(self.call_record), field_name="call_record"))
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "request_snapshot", request_snapshot)
        object.__setattr__(self, "request_snapshot_hash", request_snapshot_hash)
        object.__setattr__(self, "duplicate_of_attempt_id", duplicate_id)
        object.__setattr__(self, "request_state_after", state)
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="metadata"))

    @property
    def request_id(self) -> str:
        """Alias matching the request-manifest field name."""

        return self.parent_request_id

    @property
    def attempt_number(self) -> int:
        return self.attempt_index

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "parent_request_id": self.parent_request_id,
            "request_id": self.parent_request_id,
            "attempt_index": self.attempt_index,
            "attempt_number": self.attempt_index,
            "outcome": self.outcome,
            "status": self.outcome,
            "proposal_source": self.proposal_source,
            "candidate_plan": deepcopy(self.candidate_plan),
            "call_record": deepcopy(self.call_record),
            "resources": self.resources.to_dict(),
            "reason": self.reason,
            "request_snapshot": deepcopy(self.request_snapshot),
            "request_snapshot_hash": self.request_snapshot_hash,
            "duplicate_of_attempt_id": self.duplicate_of_attempt_id,
            "request_state_after": self.request_state_after,
            "metadata": deepcopy(self.metadata),
        }

    as_dict = to_dict


def attempt_id_for(request_id: str, attempt_index: int) -> str:
    """Compute the stable ID for one request/index pair."""

    request_id = _non_empty_string(request_id, "request_id")
    attempt_index = _nonnegative_int(attempt_index, "attempt_index")
    if attempt_index == 0:
        raise LedgerError("attempt_index must start at 1")
    digest = _content_hash({"request_id": request_id, "attempt_index": attempt_index})[:24]
    return "attempt_" + digest


def _validated_attempt_mapping(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Validate a pre-built JSON row before passing it to the writer."""

    if not isinstance(value, Mapping):
        raise LedgerError("attempt records must be mappings")

    def required(name: str, *aliases: str) -> Any:
        for key in (name, *aliases):
            if key in value and value[key] is not None:
                return value[key]
        raise LedgerError(f"attempt record missing required field {name!r}")

    attempt_id = _non_empty_string(required("attempt_id"), "attempt_id")
    parent_request_id = _non_empty_string(
        required("parent_request_id", "request_id"), "parent_request_id"
    )
    if "request_id" in value and value["request_id"] != parent_request_id:
        raise LedgerError("parent_request_id and request_id disagree")
    attempt_index = _nonnegative_int(
        required("attempt_index", "attempt_number"), "attempt_index"
    )
    if attempt_index == 0:
        raise LedgerError("attempt_index must start at 1")
    expected_id = attempt_id_for(parent_request_id, attempt_index)
    if attempt_id != expected_id:
        raise LedgerError(
            f"attempt_id {attempt_id!r} does not match deterministic ID {expected_id!r}"
        )

    outcome = value.get("outcome")
    status = value.get("status")
    if outcome is None and status is None:
        raise LedgerError("attempt record missing required field 'outcome'")
    normalized_outcome = _normalize_outcome(outcome if outcome is not None else status)
    if outcome is not None and status is not None and _normalize_outcome(status) != normalized_outcome:
        raise LedgerError("outcome and status disagree")

    duplicate_id = value.get("duplicate_of_attempt_id")
    if normalized_outcome == AttemptOutcome.DUPLICATE.value and duplicate_id is None:
        raise LedgerError("duplicate outcome requires duplicate_of_attempt_id")
    if normalized_outcome != AttemptOutcome.DUPLICATE.value and duplicate_id is not None:
        raise LedgerError("duplicate_of_attempt_id is only valid for duplicate outcomes")

    if "candidate_plan" not in value:
        raise LedgerError("attempt record missing required field 'candidate_plan'")
    candidate_plan = value["candidate_plan"]
    request_snapshot = required("request_snapshot")
    request_snapshot_hash = required("request_snapshot_hash")
    usage = ResourceUsage.from_mapping(required("resources"))
    record = AttemptRecord(
        attempt_id=attempt_id,
        parent_request_id=parent_request_id,
        attempt_index=attempt_index,
        outcome=normalized_outcome,
        proposal_source=required("proposal_source"),
        candidate_plan=candidate_plan,
        call_record=required("call_record"),
        resources=usage,
        reason=required("reason"),
        request_snapshot=request_snapshot,
        request_snapshot_hash=request_snapshot_hash,
        duplicate_of_attempt_id=duplicate_id,
        request_state_after=required("request_state_after"),
        metadata=value.get("metadata"),
    )
    return record.to_dict()


class AttemptLedger:
    """Bounded, append-only attempt ledger for registered requests.

    ``record_attempt`` performs exactly one append.  It never retries, changes
    the request, or invokes a service.  Callers must explicitly register every
    parse, invalid-parameter, execution, validation, duplicate, and accepted
    candidate outcome.
    """

    def __init__(
        self,
        requests: Iterable[Mapping[str, JSONValue] | Any] = (),
        *,
        max_attempts: int = 3,
    ) -> None:
        self._max_attempts = _nonnegative_int(max_attempts, "max_attempts")
        if self._max_attempts == 0:
            raise LedgerError("max_attempts must be greater than zero")
        self._requests: dict[str, RequestSnapshot] = {}
        self._attempts: dict[str, list[AttemptRecord]] = {}
        for request in self._iter_requests(requests):
            self.register_request(request)

    @staticmethod
    def _iter_requests(
        requests: Iterable[Mapping[str, JSONValue] | Any],
    ) -> Iterable[Mapping[str, JSONValue] | Any]:
        if isinstance(requests, Mapping):
            # A single manifest record has request_id itself.  Otherwise a
            # mapping of request_id -> snapshot is accepted for convenience.
            if "request_id" in requests:
                return (requests,)
            return (
                dict(snapshot, request_id=request_id)
                if isinstance(snapshot, Mapping) and "request_id" not in snapshot
                else snapshot
                for request_id, snapshot in requests.items()
            )
        try:
            return iter(requests)
        except TypeError as exc:
            raise LedgerError("requests must be an iterable of snapshots") from exc

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def register_request(
        self,
        request: Mapping[str, JSONValue] | Any,
        *,
        request_id: str | None = None,
    ) -> str:
        """Register one original request before any candidate is attempted."""

        snapshot = RequestSnapshot.from_value(request, request_id=request_id)
        if snapshot.request_id in self._requests:
            raise LedgerError(f"duplicate registered request_id {snapshot.request_id!r}")
        self._requests[snapshot.request_id] = snapshot
        self._attempts[snapshot.request_id] = []
        return snapshot.request_id

    add_request = register_request
    register = register_request

    def _lookup(self, request_id: str) -> RequestSnapshot:
        request_id = _non_empty_string(request_id, "request_id")
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise LedgerError(f"unknown request_id {request_id!r}") from exc

    def state(self, request_id: str) -> RequestState:
        self._lookup(request_id)
        attempts = self._attempts[request_id]
        if attempts and attempts[-1].outcome == AttemptOutcome.ACCEPTED.value:
            return RequestState.ACCEPTED
        if len(attempts) >= self._max_attempts:
            return RequestState.BUDGET_EXHAUSTED
        return RequestState.OPEN

    request_state = state

    def attempt_count(self, request_id: str) -> int:
        self._lookup(request_id)
        return len(self._attempts[request_id])

    def remaining_attempts(self, request_id: str) -> int:
        return max(0, self._max_attempts - self.attempt_count(request_id))

    def _check_request_snapshot(
        self,
        snapshot: RequestSnapshot,
        supplied: Mapping[str, JSONValue] | None,
    ) -> None:
        if supplied is None:
            return
        supplied_copy = _json_copy(supplied, field_name="request_snapshot")
        supplied_hash = _content_hash(supplied_copy)
        if supplied_hash != snapshot.request_hash:
            raise LedgerError(
                f"original request {snapshot.request_id!r} changed after registration"
            )

    def record_attempt(
        self,
        request_id: str,
        *,
        outcome: str | AttemptOutcome | None = None,
        status: str | AttemptOutcome | None = None,
        proposal_source: str,
        candidate_plan: JSONValue,
        call_record: Mapping[str, JSONValue],
        resources: ResourceUsage | Mapping[str, Any],
        reason: str,
        request_snapshot: Mapping[str, JSONValue] | None = None,
        attempt_id: str | None = None,
        attempt_index: int | None = None,
        duplicate_of_attempt_id: str | None = None,
        metadata: JSONValue = None,
    ) -> AttemptRecord:
        """Append one and only one explicit candidate attempt.

        ``candidate_plan`` may be ``None`` for a parse failure, but the field
        must still be supplied.  ``call_record`` may be an empty object for a
        non-service proposal; its presence makes that choice auditable.
        """

        snapshot = self._lookup(request_id)
        previous = self._attempts[request_id]
        current_state = self.state(request_id)
        if current_state == RequestState.ACCEPTED:
            raise LedgerError(
                f"request {request_id!r} was accepted; no further attempts are allowed"
            )
        if current_state == RequestState.BUDGET_EXHAUSTED:
            raise LedgerError(
                f"request {request_id!r} exhausted max_attempts={self._max_attempts}"
            )
        if outcome is None and status is None:
            raise LedgerError("each attempt requires outcome/status")
        if outcome is not None and status is not None:
            normalized_outcome = _normalize_outcome(outcome)
            normalized_status = _normalize_outcome(status)
            if normalized_outcome != normalized_status:
                raise LedgerError("outcome and status disagree")
        else:
            normalized_outcome = _normalize_outcome(outcome if outcome is not None else status)

        if not isinstance(call_record, Mapping):
            raise LedgerError("call_record must be a JSON object")
        proposal_source = _non_empty_string(proposal_source, "proposal_source")
        reason = _non_empty_string(reason, "reason")
        if duplicate_of_attempt_id is not None:
            duplicate_of_attempt_id = _non_empty_string(
                duplicate_of_attempt_id, "duplicate_of_attempt_id"
            )
        if normalized_outcome == AttemptOutcome.DUPLICATE.value and duplicate_of_attempt_id is None:
            raise LedgerError("duplicate outcome requires duplicate_of_attempt_id")
        if normalized_outcome != AttemptOutcome.DUPLICATE.value and duplicate_of_attempt_id is not None:
            raise LedgerError("duplicate_of_attempt_id is only valid for duplicate outcomes")
        if normalized_outcome == AttemptOutcome.ACCEPTED.value and candidate_plan is None:
            raise LedgerError("accepted outcome requires a candidate_plan")

        self._check_request_snapshot(snapshot, request_snapshot)
        if isinstance(resources, ResourceUsage):
            usage = resources
        else:
            usage = ResourceUsage.from_mapping(resources)

        expected_index = len(previous) + 1
        if attempt_index is not None:
            attempt_index = _nonnegative_int(attempt_index, "attempt_index")
            if attempt_index != expected_index:
                raise LedgerError(
                    f"attempt_index must be contiguous: expected {expected_index}, got {attempt_index}"
                )
        else:
            attempt_index = expected_index
        expected_id = attempt_id_for(request_id, attempt_index)
        if attempt_id is not None:
            attempt_id = _non_empty_string(attempt_id, "attempt_id")
            if attempt_id != expected_id:
                raise LedgerError(
                    f"attempt_id {attempt_id!r} does not match deterministic ID {expected_id!r}"
                )
        else:
            attempt_id = expected_id

        record_state = (
            RequestState.ACCEPTED.value
            if normalized_outcome == AttemptOutcome.ACCEPTED.value
            else RequestState.BUDGET_EXHAUSTED.value
            if expected_index >= self._max_attempts
            else RequestState.OPEN.value
        )
        record = AttemptRecord(
            attempt_id=attempt_id,
            parent_request_id=request_id,
            attempt_index=expected_index,
            outcome=normalized_outcome,
            proposal_source=proposal_source,
            candidate_plan=_json_copy(candidate_plan, field_name="candidate_plan"),
            call_record=_json_copy(dict(call_record), field_name="call_record"),
            resources=usage,
            reason=reason,
            request_snapshot=deepcopy(snapshot.request),
            request_snapshot_hash=snapshot.request_hash,
            duplicate_of_attempt_id=duplicate_of_attempt_id,
            request_state_after=record_state,
            metadata=_json_copy(metadata, field_name="metadata"),
        )
        self._attempts[request_id].append(record)
        return record

    append_attempt = record_attempt
    add_attempt = record_attempt

    def attempts_for(self, request_id: str) -> tuple[AttemptRecord, ...]:
        self._lookup(request_id)
        return tuple(self._attempts[request_id])

    def records(self) -> tuple[AttemptRecord, ...]:
        return tuple(
            record
            for request_id in sorted(self._attempts)
            for record in self._attempts[request_id]
        )

    def to_dicts(self) -> list[dict[str, JSONValue]]:
        return [record.to_dict() for record in self.records()]

    as_dicts = to_dicts


def write_attempt_ledger(
    path: str | Path,
    records: AttemptLedger | Iterable[AttemptRecord | Mapping[str, JSONValue]],
) -> None:
    """Write explicit attempt records as stable UTF-8 JSONL."""

    if isinstance(records, AttemptLedger):
        rows = records.to_dicts()
    else:
        rows = []
        for record in records:
            if isinstance(record, AttemptRecord):
                rows.append(record.to_dict())
            elif isinstance(record, Mapping):
                rows.append(_validated_attempt_mapping(record))
            else:
                raise LedgerError("records must contain AttemptRecord objects or mappings")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# Friendly aliases for integrations that call the field a status or the
# resource vector a record.  They serialize to the same canonical schema.
AttemptStatus = AttemptOutcome
AttemptLedgerError = LedgerError
ResourceRecord = ResourceUsage


__all__ = [
    "AttemptLedger",
    "AttemptLedgerError",
    "AttemptOutcome",
    "AttemptRecord",
    "AttemptStatus",
    "LedgerError",
    "RequestSnapshot",
    "RequestState",
    "ResourceRecord",
    "ResourceUsage",
    "attempt_id_for",
    "write_attempt_ledger",
]
