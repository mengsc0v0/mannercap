"""Request-manifest orchestration for the first MannerCAP synthesis pass.

This module deliberately stays at the request-planning boundary.  It does not
run an environment, call a model, infer a physical threshold, or alter a
requested constraint.  A request is registered before candidate generation and
therefore remains in the manifest even when all later attempts fail.

The public helpers use JSON-compatible values so that the same objects can be
written to ``request_manifest.jsonl`` and replayed by a later generator.  The
only task values enabled by the first pass are ``lift`` and ``stack``.  A
future calibrated task must be added explicitly rather than entering the
manifest by accident.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Final, TypeAlias


JSONValue: TypeAlias = Any


class ManifestError(ValueError):
    """Raised when a request or a pair violates the manifest contract."""


class Split(str, Enum):
    """Named data uses in the v2 contract.

    ``development`` is intentionally explicit: it is the only split used by
    the convenience builder in this first pass.  The other names are accepted
    so that a later frozen manifest can use the same schema without changing
    the ID algorithm.
    """

    DEVELOPMENT = "development"
    GENERATOR_TEST = "generator_test"
    STUDENT_TRAIN = "student_train"
    STUDENT_VALIDATION = "student_validation"
    STUDENT_TEST = "student_test"


class PairType(str, Enum):
    """The three planned intervention families.

    Enum aliases keep the short names convenient without creating additional
    serialized values.
    """

    MODIFIER_WORD = "modifier_word"
    MANNER_WORD = "modifier_word"
    MODIFIER = "modifier_word"
    PHYSICAL_ATTRIBUTE = "physical_attribute"
    ATTRIBUTE = "physical_attribute"
    STAGE_SCOPE = "stage_scope"
    STAGE = "stage_scope"


DEVELOPMENT_SPLIT: Final[str] = Split.DEVELOPMENT.value
DEFAULT_DEVELOPMENT_TASKS: Final[tuple[str, ...]] = ("lift", "stack")
SUPPORTED_TASKS: Final[tuple[str, ...]] = DEFAULT_DEVELOPMENT_TASKS
SUPPORTED_SPLITS: Final[tuple[str, ...]] = tuple(item.value for item in Split)

_SPLIT_ALIASES: Final[dict[str, str]] = {
    "dev": DEVELOPMENT_SPLIT,
    "development": DEVELOPMENT_SPLIT,
    "generator_test": Split.GENERATOR_TEST.value,
    "generator-test": Split.GENERATOR_TEST.value,
    "student_train": Split.STUDENT_TRAIN.value,
    "student-train": Split.STUDENT_TRAIN.value,
    "student_validation": Split.STUDENT_VALIDATION.value,
    "student-validation": Split.STUDENT_VALIDATION.value,
    "student_test": Split.STUDENT_TEST.value,
    "student-test": Split.STUDENT_TEST.value,
}
_PAIR_TYPE_ALIASES: Final[dict[str, str]] = {
    "modifier_word": PairType.MODIFIER_WORD.value,
    "modifier-word": PairType.MODIFIER_WORD.value,
    "modifier": PairType.MODIFIER_WORD.value,
    "manner_word": PairType.MODIFIER_WORD.value,
    "manner-word": PairType.MODIFIER_WORD.value,
    "manner": PairType.MODIFIER_WORD.value,
    "physical_attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "physical-attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "attribute": PairType.PHYSICAL_ATTRIBUTE.value,
    "stage_scope": PairType.STAGE_SCOPE.value,
    "stage-scope": PairType.STAGE_SCOPE.value,
    "stage": PairType.STAGE_SCOPE.value,
    "scope": PairType.STAGE_SCOPE.value,
}

_REQUEST_SCHEMA_VERSION: Final[str] = "synthesis_v2.request.v1"
_MANIFEST_STATUS: Final[str] = "draft"


def _json_copy(value: JSONValue, *, field_name: str) -> JSONValue:
    """Copy and validate a JSON-compatible value without normalizing it."""

    try:
        copied = deepcopy(value)
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field_name} must be JSON-compatible") from exc
    return copied


def _canonical_json(value: JSONValue) -> str:
    """Return the stable JSON representation used for all content hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError("manifest values must be JSON-compatible") from exc


def _content_hash(value: JSONValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_split(value: str | Split) -> str:
    if isinstance(value, Split):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("split must be an explicit non-empty name")
    normalized = _SPLIT_ALIASES.get(value.strip().lower())
    if normalized is None:
        allowed = ", ".join(SUPPORTED_SPLITS)
        raise ManifestError(f"unknown split {value!r}; expected one of {allowed}")
    return normalized


def _normalize_pair_type(value: str | PairType) -> str:
    if isinstance(value, PairType):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("pair_type must be a non-empty name")
    normalized = _PAIR_TYPE_ALIASES.get(value.strip().lower())
    if normalized is None:
        allowed = ", ".join(item.value for item in PairType)
        raise ManifestError(f"unknown pair_type {value!r}; expected one of {allowed}")
    return normalized


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RequestDraft:
    """A request before pair metadata and its deterministic ID are attached.

    ``requested_constraints`` is copied at construction and emitted verbatim
    (modulo JSON formatting) under the same name in the manifest.  No clipping,
    threshold lookup, unit conversion, or inferred physical property is done
    here.  ``physical_config`` and ``attribute_intervention`` are opaque,
    caller-supplied JSON values for the calibrated task adapter.
    """

    task: str
    scene_id: str
    initial_state: JSONValue = field(default_factory=dict)
    target_object: str = ""
    physical_config: JSONValue = field(default_factory=dict)
    raw_language: str = ""
    requested_constraints: JSONValue = field(default_factory=dict)
    stage_scope: JSONValue = field(default_factory=list)
    allowed_control_channels: Sequence[str] = field(default_factory=tuple)
    budget: JSONValue = field(default_factory=dict)
    split: str | Split = DEVELOPMENT_SPLIT
    geometry_key: str | None = None
    manner_key: str | None = None
    attribute_intervention: JSONValue | None = None
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = _non_empty_string(self.task, "task").strip().lower()
        scene_id = _non_empty_string(self.scene_id, "scene_id")
        target_object = _non_empty_string(self.target_object, "target_object")
        raw_language = _non_empty_string(self.raw_language, "raw_language")
        split = _normalize_split(self.split)

        if self.geometry_key is not None:
            _non_empty_string(self.geometry_key, "geometry_key")
        if self.manner_key is not None:
            _non_empty_string(self.manner_key, "manner_key")

        if isinstance(self.allowed_control_channels, (str, bytes)):
            raise ManifestError("allowed_control_channels must be a sequence of channel names")
        controls: list[str] = []
        try:
            for channel in self.allowed_control_channels:
                controls.append(_non_empty_string(channel, "allowed_control_channels"))
        except TypeError as exc:
            raise ManifestError("allowed_control_channels must be a sequence") from exc

        # Copy every JSON field so the caller cannot change the registered raw
        # constraint or initial state by mutating its input after construction.
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "target_object", target_object)
        object.__setattr__(self, "raw_language", raw_language)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "allowed_control_channels", tuple(sorted(set(controls))))
        object.__setattr__(
            self,
            "initial_state",
            _json_copy(self.initial_state, field_name="initial_state"),
        )
        object.__setattr__(
            self,
            "physical_config",
            _json_copy(self.physical_config, field_name="physical_config"),
        )
        object.__setattr__(
            self,
            "requested_constraints",
            _json_copy(self.requested_constraints, field_name="requested_constraints"),
        )
        object.__setattr__(
            self,
            "stage_scope",
            _json_copy(self.stage_scope, field_name="stage_scope"),
        )
        object.__setattr__(self, "budget", _json_copy(self.budget, field_name="budget"))
        object.__setattr__(
            self,
            "attribute_intervention",
            _json_copy(self.attribute_intervention, field_name="attribute_intervention"),
        )
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="metadata"))

    @property
    def raw_constraints(self) -> JSONValue:
        """Compatibility name for the unmodified request constraints."""

        return deepcopy(self.requested_constraints)

    def payload(self) -> dict[str, JSONValue]:
        """Return the ID-bearing request content, excluding generated IDs."""

        return {
            "schema_version": _REQUEST_SCHEMA_VERSION,
            "task": self.task,
            "scene_id": self.scene_id,
            "split": self.split,
            "initial_state": deepcopy(self.initial_state),
            "target_object": self.target_object,
            "physical_config": deepcopy(self.physical_config),
            "raw_language": self.raw_language,
            "requested_constraints": deepcopy(self.requested_constraints),
            "stage_scope": deepcopy(self.stage_scope),
            "allowed_control_channels": list(self.allowed_control_channels),
            "budget": deepcopy(self.budget),
            "geometry_key": self.geometry_key,
            "manner_key": self.manner_key,
            "attribute_intervention": deepcopy(self.attribute_intervention),
            "metadata": deepcopy(self.metadata),
        }

    @property
    def request_id(self) -> str:
        """The stable ID for this request's full declared content."""

        return request_id_for(self)

    @property
    def physical_state_hash(self) -> str:
        """Hash of the complete caller-supplied state/configuration pair."""

        return _content_hash(
            {
                "initial_state": deepcopy(self.initial_state),
                "physical_config": deepcopy(self.physical_config),
            }
        )

    def to_manifest_dict(
        self,
        *,
        pair_id: str | None = None,
        pair_type: str | None = None,
        pair_member_role: str | None = None,
        pair_member_ids: Sequence[str] = (),
        pair_metadata: Mapping[str, JSONValue] | None = None,
    ) -> dict[str, JSONValue]:
        """Serialize one planned request, preserving all raw input fields."""

        record = self.payload()
        record.update(
            {
                "request_id": self.request_id,
                "physical_state_hash": self.physical_state_hash,
                "requested_constraints_hash": _content_hash(self.requested_constraints),
                "request_status": "planned",
                "manifest_status": _MANIFEST_STATUS,
                "pair_id": pair_id,
                "pair_type": pair_type,
                "pair_member_role": pair_member_role,
                "pair_member_ids": list(pair_member_ids),
                "pair_metadata": deepcopy(dict(pair_metadata or {})),
            }
        )
        return record


def _request_from_mapping(value: Mapping[str, JSONValue]) -> RequestDraft:
    """Construct a draft from a compact dictionary used by JSONL callers."""

    def get_required(name: str, *aliases: str) -> JSONValue:
        for key in (name, *aliases):
            if key in value:
                return value[key]
        raise ManifestError(f"request is missing required field {name!r}")

    known = {
        "task",
        "scene_id",
        "scene",
        "initial_state",
        "target_object",
        "object_id",
        "physical_config",
        "raw_language",
        "language",
        "requested_constraints",
        "constraints",
        "stage_scope",
        "scope",
        "allowed_control_channels",
        "allowed_controls",
        "budget",
        "split",
        "geometry_key",
        "common_geometry_id",
        "manner_key",
        "attribute_intervention",
        "metadata",
        # Generated fields are ignored when reading a record back for ID
        # checking; IDs are always recomputed from the request content.
        "request_id",
        "physical_state_hash",
        "requested_constraints_hash",
        "request_status",
        "manifest_status",
        "pair_id",
        "pair_type",
        "pair_member_role",
        "pair_member_ids",
        "pair_metadata",
        "schema_version",
    }
    metadata = deepcopy(value.get("metadata", {}))
    if not isinstance(metadata, Mapping):
        raise ManifestError("metadata must be a JSON object")
    extras = {key: deepcopy(item) for key, item in value.items() if key not in known}
    if extras:
        metadata = dict(metadata)
        metadata.setdefault("unmodeled_fields", {}).update(extras)

    geometry_key = value.get("geometry_key", value.get("common_geometry_id"))
    return RequestDraft(
        task=get_required("task"),
        scene_id=get_required("scene_id", "scene"),
        initial_state=get_required("initial_state"),
        target_object=get_required("target_object", "object_id"),
        physical_config=value.get("physical_config", {}),
        raw_language=get_required("raw_language", "language"),
        requested_constraints=value.get("requested_constraints", value.get("constraints", {})),
        stage_scope=value.get("stage_scope", value.get("scope", [])),
        allowed_control_channels=value.get(
            "allowed_control_channels", value.get("allowed_controls", ())
        ),
        budget=value.get("budget", {}),
        split=value.get("split", DEVELOPMENT_SPLIT),
        geometry_key=geometry_key,
        manner_key=value.get("manner_key"),
        attribute_intervention=value.get("attribute_intervention"),
        metadata=metadata,
    )


def coerce_request(value: RequestDraft | Mapping[str, JSONValue]) -> RequestDraft:
    """Return a validated draft from a dataclass or mapping."""

    if isinstance(value, RequestDraft):
        return value
    if isinstance(value, Mapping):
        return _request_from_mapping(value)
    raise ManifestError("requests must be RequestDraft objects or mappings")


def request_id_for(value: RequestDraft | Mapping[str, JSONValue]) -> str:
    """Compute a deterministic request ID from canonical request content."""

    draft = coerce_request(value)
    return "req_" + _content_hash(draft.payload())[:24]


@dataclass(frozen=True)
class PairPlan:
    """A planned two-member relation.

    Members may be request IDs, ``RequestDraft`` objects, or inline request
    mappings.  The builder resolves them against the request list, verifies the
    relation, and computes ``pair_id`` from the sorted member request IDs.
    """

    pair_type: str | PairType
    members: Sequence[str | RequestDraft | Mapping[str, JSONValue]]
    pair_id: str | None = None
    metadata: JSONValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalize_pair_type(self.pair_type)
        members = tuple(self.members)
        if len(members) != 2:
            raise ManifestError("each pair must have exactly two members")
        if members[0] == members[1]:
            raise ManifestError("a pair cannot contain the same member twice")
        if self.pair_id is not None:
            _non_empty_string(self.pair_id, "pair_id")
        object.__setattr__(self, "pair_type", normalized)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "metadata", _json_copy(self.metadata, field_name="metadata"))


def make_pair(
    left: str | RequestDraft | Mapping[str, JSONValue],
    right: str | RequestDraft | Mapping[str, JSONValue],
    pair_type: str | PairType,
    *,
    metadata: JSONValue | None = None,
) -> PairPlan:
    """Create a pair plan; validation occurs when it is attached to a manifest."""

    return PairPlan(pair_type, (left, right), metadata={} if metadata is None else metadata)


# A descriptive alias for callers that prefer the contract's wording.
pair_requests = make_pair


def pair_id_for(pair_type: str | PairType, request_ids: Iterable[str]) -> str:
    """Compute the order-independent deterministic ID of a two-member pair."""

    normalized_type = _normalize_pair_type(pair_type)
    ids = tuple(sorted(_non_empty_string(item, "request_id") for item in request_ids))
    if len(ids) != 2 or ids[0] == ids[1]:
        raise ManifestError("pair_id_for requires two distinct request IDs")
    return "pair_" + _content_hash({"pair_type": normalized_type, "request_ids": ids})[:24]


def _payload_without(draft: RequestDraft, *fields: str) -> dict[str, JSONValue]:
    payload = draft.payload()
    for field_name in fields:
        payload.pop(field_name, None)
    return payload


def _same_except(left: RequestDraft, right: RequestDraft, *fields: str) -> bool:
    return _canonical_json(_payload_without(left, *fields)) == _canonical_json(
        _payload_without(right, *fields)
    )


def validate_pair(
    left: RequestDraft | Mapping[str, JSONValue],
    right: RequestDraft | Mapping[str, JSONValue],
    pair_type: str | PairType,
) -> None:
    """Validate one of the three pair contracts.

    The checks are structural.  They never decide whether a physical value is
    sensible because the task properties and thresholds are still awaiting
    calibration.
    """

    first = coerce_request(left)
    second = coerce_request(right)
    relation = _normalize_pair_type(pair_type)

    if first.split != second.split:
        raise ManifestError(
            f"pair members must share a split; got {first.split!r} and {second.split!r}"
        )
    if first.task != second.task:
        raise ManifestError("pair members must use the same task")
    if first.scene_id != second.scene_id:
        raise ManifestError("pair members must use the same scene_id")

    if relation == PairType.MODIFIER_WORD.value:
        if first.raw_language == second.raw_language:
            raise ManifestError("modifier_word pair must change raw_language")
        if not _same_except(first, second, "raw_language", "requested_constraints", "manner_key"):
            raise ManifestError(
                "modifier_word pair may vary only raw_language and its declared constraints"
            )
        return

    if relation == PairType.PHYSICAL_ATTRIBUTE.value:
        if first.geometry_key is None or second.geometry_key is None:
            raise ManifestError(
                "physical_attribute pair requires a shared geometry_key/common_geometry_id"
            )
        if first.geometry_key != second.geometry_key:
            raise ManifestError("physical_attribute pair members must share geometry_key")
        if first.raw_language != second.raw_language:
            raise ManifestError("physical_attribute pair requires identical neutral language")
        if first.requested_constraints != second.requested_constraints:
            raise ManifestError(
                "physical_attribute pair requires identical requested_constraints"
            )
        if first.physical_state_hash == second.physical_state_hash:
            raise ManifestError("physical_attribute pair must change physical state/configuration")
        if not _same_except(
            first,
            second,
            "initial_state",
            "physical_config",
            "attribute_intervention",
            "metadata",
        ):
            raise ManifestError(
                "physical_attribute pair may vary only state/configuration and intervention metadata"
            )
        return

    if relation == PairType.STAGE_SCOPE.value:
        if first.stage_scope == second.stage_scope:
            raise ManifestError("stage_scope pair must change stage_scope")
        if first.manner_key != second.manner_key:
            raise ManifestError("stage_scope pair must keep the same manner_key")
        if not _same_except(
            first,
            second,
            "raw_language",
            "requested_constraints",
            "stage_scope",
        ):
            raise ManifestError(
                "stage_scope pair may vary only language, stage_scope, and its declared constraints"
            )
        return

    # _normalize_pair_type makes this unreachable, but retaining the guard
    # keeps future extensions from silently accepting a new relation.
    raise ManifestError(f"unsupported pair_type {relation!r}")


def _resolve_pair_member(
    member: str | RequestDraft | Mapping[str, JSONValue],
    requests_by_id: Mapping[str, RequestDraft],
) -> tuple[str, RequestDraft]:
    if isinstance(member, str):
        member_id = member
        if member_id not in requests_by_id:
            raise ManifestError(f"pair refers to unknown request_id {member_id!r}")
        return member_id, requests_by_id[member_id]

    draft = coerce_request(member)
    member_id = draft.request_id
    registered = requests_by_id.get(member_id)
    if registered is None:
        raise ManifestError(
            f"inline pair member {member_id!r} is not present in the request list"
        )
    return member_id, registered


def _coerce_pair(value: PairPlan | Mapping[str, JSONValue]) -> PairPlan:
    if isinstance(value, PairPlan):
        return value
    if not isinstance(value, Mapping):
        raise ManifestError("pairs must be PairPlan objects or mappings")
    pair_type = value.get("pair_type", value.get("type"))
    if pair_type is None:
        raise ManifestError("pair is missing pair_type")
    members = value.get("members", value.get("request_ids"))
    if members is None and "left" in value and "right" in value:
        members = (value["left"], value["right"])
    if members is None:
        raise ManifestError("pair is missing two members")
    return PairPlan(
        pair_type=pair_type,
        members=members,
        pair_id=value.get("pair_id"),
        metadata=value.get("metadata", value.get("pair_metadata", {})),
    )


def build_request_manifest(
    requests: Iterable[RequestDraft | Mapping[str, JSONValue]],
    pairs: Iterable[PairPlan | Mapping[str, JSONValue]] = (),
    *,
    allowed_tasks: Sequence[str] = SUPPORTED_TASKS,
) -> list[dict[str, JSONValue]]:
    """Build deterministic JSONL-ready request records.

    All requests are retained, including requests that only have a planned
    pair or that will later exhaust their candidate budget.  Pair membership is
    assigned per request because the request manifest is one JSON object per
    line.  A request is restricted to one pair in this initial schema, keeping
    the singular ``pair_id`` field unambiguous.
    """

    drafts = [coerce_request(item) for item in requests]
    allowed = { _non_empty_string(task, "allowed_tasks").strip().lower() for task in allowed_tasks }
    unknown_tasks = sorted({draft.task for draft in drafts if draft.task not in allowed})
    if unknown_tasks:
        raise ManifestError(
            "task(s) are outside the calibrated request space: " + ", ".join(unknown_tasks)
        )

    requests_by_id: dict[str, RequestDraft] = {}
    for draft in drafts:
        request_id = draft.request_id
        if request_id in requests_by_id:
            raise ManifestError(f"duplicate request_id {request_id!r}")
        requests_by_id[request_id] = draft

    pair_by_request: dict[str, dict[str, JSONValue]] = {}
    seen_pair_ids: set[str] = set()
    for raw_pair in pairs:
        pair = _coerce_pair(raw_pair)
        member_ids = []
        member_drafts = []
        for member in pair.members:
            member_id, member_draft = _resolve_pair_member(member, requests_by_id)
            member_ids.append(member_id)
            member_drafts.append(member_draft)
        if member_ids[0] == member_ids[1]:
            raise ManifestError("a pair cannot contain the same request twice")

        validate_pair(member_drafts[0], member_drafts[1], pair.pair_type)
        ordered_ids = tuple(sorted(member_ids))
        computed_pair_id = pair_id_for(pair.pair_type, ordered_ids)
        if pair.pair_id is not None and pair.pair_id != computed_pair_id:
            raise ManifestError(
                f"pair_id {pair.pair_id!r} does not match deterministic ID {computed_pair_id!r}"
            )
        if computed_pair_id in seen_pair_ids:
            raise ManifestError(f"duplicate pair {computed_pair_id!r}")
        seen_pair_ids.add(computed_pair_id)

        pair_metadata = deepcopy(pair.metadata)
        if not isinstance(pair_metadata, Mapping):
            raise ManifestError("pair metadata must be a JSON object")
        pair_metadata = dict(pair_metadata)
        if pair.pair_type == PairType.PHYSICAL_ATTRIBUTE.value:
            # This is a shared geometry identifier, not an inferred physical
            # quantity.  It is required for this pair type and is copied into
            # every member record for leakage and lineage checks.
            requested_geometry = pair_metadata.get("common_geometry_id")
            if (
                requested_geometry is not None
                and requested_geometry != member_drafts[0].geometry_key
            ):
                raise ManifestError(
                    "pair metadata common_geometry_id must match the members' geometry_key"
                )
            pair_metadata.setdefault("common_geometry_id", member_drafts[0].geometry_key)
            pair_metadata["physical_state_hashes"] = {
                ordered_ids[0]: requests_by_id[ordered_ids[0]].physical_state_hash,
                ordered_ids[1]: requests_by_id[ordered_ids[1]].physical_state_hash,
            }
            pair_metadata["attribute_intervention"] = {
                ordered_ids[0]: deepcopy(
                    requests_by_id[ordered_ids[0]].attribute_intervention
                ),
                ordered_ids[1]: deepcopy(
                    requests_by_id[ordered_ids[1]].attribute_intervention
                ),
            }

        for role, request_id in enumerate(ordered_ids):
            if request_id in pair_by_request:
                raise ManifestError(
                    f"request {request_id!r} is assigned to more than one pair; "
                    "the v1 manifest keeps pair_id singular"
                )
            pair_by_request[request_id] = {
                "pair_id": computed_pair_id,
                "pair_type": pair.pair_type,
                "pair_member_role": "a" if role == 0 else "b",
                "pair_member_ids": list(ordered_ids),
                "pair_metadata": deepcopy(pair_metadata),
            }

    records: list[dict[str, JSONValue]] = []
    for request_id in sorted(requests_by_id):
        draft = requests_by_id[request_id]
        relation = pair_by_request.get(request_id, {})
        records.append(draft.to_manifest_dict(**relation))
    return records


def build_development_manifest(
    requests: Iterable[RequestDraft | Mapping[str, JSONValue]],
    pairs: Iterable[PairPlan | Mapping[str, JSONValue]] = (),
) -> list[dict[str, JSONValue]]:
    """Build the first-pass manifest, explicitly restricted to development."""

    drafts = [coerce_request(item) for item in requests]
    non_development = [draft.request_id for draft in drafts if draft.split != DEVELOPMENT_SPLIT]
    if non_development:
        raise ManifestError(
            "build_development_manifest requires split='development' for every request"
        )
    return build_request_manifest(drafts, pairs, allowed_tasks=DEFAULT_DEVELOPMENT_TASKS)


# Short alias used by callers that treat the development manifest as the first
# version of the public builder.
build_manifest = build_request_manifest


def write_request_manifest(
    path: str | Path,
    records: Iterable[Mapping[str, JSONValue]],
) -> None:
    """Write records as stable UTF-8 JSON Lines, without invoking any service."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


__all__ = [
    "DEFAULT_DEVELOPMENT_TASKS",
    "DEVELOPMENT_SPLIT",
    "ManifestError",
    "PairPlan",
    "PairType",
    "RequestDraft",
    "SUPPORTED_SPLITS",
    "SUPPORTED_TASKS",
    "Split",
    "build_development_manifest",
    "build_manifest",
    "build_request_manifest",
    "coerce_request",
    "make_pair",
    "pair_id_for",
    "pair_requests",
    "request_id_for",
    "validate_pair",
    "write_request_manifest",
]
