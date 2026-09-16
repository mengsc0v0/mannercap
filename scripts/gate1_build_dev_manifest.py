"""Build a small, development-only request manifest from real initial states.

The entry point stays on the planning boundary.  It reads the two supplied
``initial_state.json`` files, derives stable initial-state references, and
passes explicit request drafts to :mod:`synthesis_v2.request_manifest`.  It
does not import an environment, call a model/API, execute candidates, or
silently repair incomplete requests.

Example::

    python scripts/gate1_build_dev_manifest.py \
        --lift-initial-state /run/lift/initial_state.json \
        --stack-initial-state /run/stack/initial_state.json \
        --output /run/request_manifest.jsonl

The generated examples are intentionally small: each of Lift and Stack gets
one pair of each of the three structural pair types.  Pending/incomplete
planning entries remain in the manifest with an explicit lifecycle marker;
the base request contract still serializes every entry as ``request_status``
``planned`` because this is a pre-execution manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Final, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis_v2.request_manifest import (
    DEVELOPMENT_SPLIT,
    ManifestError,
    PairType,
    RequestDraft,
    build_development_manifest,
    make_pair,
    write_request_manifest,
)
from synthesis_v2.task_space import TASK_SPACE_SCHEMA_VERSION, build_default_task_space


JSONValue: TypeAlias = Any
INITIAL_STATE_REF_SCHEMA_VERSION: Final[str] = "synthesis_v2.initial_state_ref.v1"
DEV_MANIFEST_SCHEMA_VERSION: Final[str] = "synthesis_v2.dev_manifest.v1"
SUPPORTED_DEVELOPMENT_TASKS: Final[tuple[str, ...]] = ("lift", "stack")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{64}$")


class DevelopmentManifestError(ValueError):
    """Raised when an input initial-state file cannot seed the manifest."""


def _canonical_json(value: JSONValue, field_name: str = "value") -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DevelopmentManifestError(f"{field_name} must be JSON-compatible") from exc


def _json_copy(value: JSONValue, field_name: str) -> JSONValue:
    try:
        copied = deepcopy(value)
        _canonical_json(copied, field_name)
    except DevelopmentManifestError:
        raise
    except Exception as exc:  # pragma: no cover - defensive deepcopy guard
        raise DevelopmentManifestError(f"{field_name} must be JSON-compatible") from exc
    return copied


def _sha256_json(value: JSONValue, field_name: str) -> str:
    return hashlib.sha256(_canonical_json(value, field_name).encode("utf-8")).hexdigest()


def _normalized_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip()):
        raise DevelopmentManifestError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return value.strip().lower()


def _normalized_task(task_id: str) -> str:
    if not isinstance(task_id, str) or not task_id.strip():
        raise DevelopmentManifestError("task_id must be a non-empty string")
    task = task_id.strip().lower()
    if task not in SUPPORTED_DEVELOPMENT_TASKS:
        raise DevelopmentManifestError(
            f"development manifest accepts only Lift/Stack, got {task_id!r}"
        )
    return task


def initial_state_id_for(
    task_id: str,
    model_sha256: str,
    initial_state_hash: str,
) -> str:
    """Return the stable ID for a task/model/initial-state content tuple.

    The source path and JSON whitespace are deliberately absent from the
    identity.  ``initial_state_hash`` is the canonical-content hash of the
    complete input document; the separate ``state_sha256`` exposed by
    :class:`InitialStateReference` identifies its dynamic ``state`` member.
    """

    task = _normalized_task(task_id)
    model_hash = _normalized_sha256(model_sha256, "model_sha256")
    state_hash = _normalized_sha256(initial_state_hash, "initial_state_hash")
    payload = {
        "schema_version": INITIAL_STATE_REF_SCHEMA_VERSION,
        "task": task,
        "model_sha256": model_hash,
        "initial_state_hash": state_hash,
    }
    digest = hashlib.sha256(_canonical_json(payload, "initial_state_id payload").encode("utf-8")).hexdigest()
    return "initial_" + digest[:24]


class InitialStateReference:
    """Validated identity and content loaded from one real initial-state file."""

    __slots__ = (
        "task_id",
        "source_path",
        "document",
        "model_sha256",
        "initial_state_hash",
        "state_sha256",
        "file_sha256",
        "initial_state_id",
    )

    def __init__(
        self,
        task_id: str,
        source_path: Path,
        document: Mapping[str, JSONValue],
        model_sha256: str,
        initial_state_hash: str,
        state_sha256: str,
        file_sha256: str,
        initial_state_id: str,
    ) -> None:
        self.task_id = task_id
        self.source_path = source_path
        self.document = document
        self.model_sha256 = model_sha256
        self.initial_state_hash = initial_state_hash
        self.state_sha256 = state_sha256
        self.file_sha256 = file_sha256
        self.initial_state_id = initial_state_id

    @property
    def manifest_initial_state(self) -> dict[str, JSONValue]:
        """Return the input document plus deterministic audit annotations.

        The original document values are copied unchanged.  The annotations
        are derived fields and let a manifest consumer find the stable ID
        without relying on a filesystem path.
        """

        result = _json_copy(dict(self.document), "initial_state document")
        result["initial_state_id"] = self.initial_state_id
        result["initial_state_hash"] = self.initial_state_hash
        result["state_sha256"] = self.state_sha256
        result["model_sha256"] = self.model_sha256
        return result

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": INITIAL_STATE_REF_SCHEMA_VERSION,
            "task_id": self.task_id,
            "initial_state_id": self.initial_state_id,
            "model_sha256": self.model_sha256,
            "initial_state_hash": self.initial_state_hash,
            "state_sha256": self.state_sha256,
            "file_sha256": self.file_sha256,
            "source_path": str(self.source_path),
        }


def load_initial_state_reference(
    path: str | Path,
    task_id: str,
) -> InitialStateReference:
    """Read and validate one task's actual ``initial_state.json`` file.

    A file must contain a JSON object, a 64-hex ``model_sha256`` and a
    ``state`` JSON object.  The full canonical document hash is used for
    ``initial_state_id`` so controller/snapshot content cannot be silently
    replaced by the shorter dynamic-state projection.
    """

    task = _normalized_task(task_id)
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise DevelopmentManifestError(f"initial-state file does not exist: {source_path}")
    if not source_path.is_file():
        raise DevelopmentManifestError(f"initial-state path is not a file: {source_path}")
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise DevelopmentManifestError(f"cannot read initial-state file: {source_path}") from exc
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentManifestError(f"initial-state file is not valid UTF-8 JSON: {source_path}") from exc
    if not isinstance(decoded, Mapping):
        raise DevelopmentManifestError("initial_state.json must contain a JSON object")
    document = _json_copy(dict(decoded), "initial_state document")
    model_sha256 = _normalized_sha256(document.get("model_sha256"), "model_sha256")
    state = document.get("state")
    if not isinstance(state, Mapping):
        raise DevelopmentManifestError("initial_state.json must contain a JSON object under 'state'")
    # Validate the state independently so state_sha256 is an explicit audit
    # field even when unrelated evidence metadata changes later.
    state_copy = _json_copy(dict(state), "initial_state.state")
    initial_state_hash = _sha256_json(document, "initial_state document")
    state_sha256 = _sha256_json(state_copy, "initial_state.state")
    initial_state_id = initial_state_id_for(task, model_sha256, initial_state_hash)
    return InitialStateReference(
        task_id=task,
        source_path=source_path,
        document=document,
        model_sha256=model_sha256,
        initial_state_hash=initial_state_hash,
        state_sha256=state_sha256,
        file_sha256=file_sha256,
        initial_state_id=initial_state_id,
    )


def _task_template(task_id: str, space: Any) -> Any:
    try:
        return space.get_task(task_id)
    except Exception as exc:  # pragma: no cover - default registry is tested separately
        raise DevelopmentManifestError(f"task-space registry lacks {task_id!r}") from exc


def _task_request_common(
    task_id: str,
    task_template: Any,
    initial_state: InitialStateReference,
    *,
    raw_language: str,
    requested_constraints: JSONValue,
    stage_scope: JSONValue,
    physical_config: JSONValue,
    metadata: Mapping[str, JSONValue],
) -> RequestDraft:
    objects = tuple(task_template.objects)
    if not objects:
        raise DevelopmentManifestError(f"task {task_id!r} has no target object declaration")
    stage_ids = [phase.stage_instance_id for phase in task_template.phases]
    if not stage_ids:
        raise DevelopmentManifestError(f"task {task_id!r} has no calibrated phases")
    controls = tuple(task_template.control_names())
    if not controls:
        raise DevelopmentManifestError(f"task {task_id!r} has no enabled control channels")
    return RequestDraft(
        task=task_id,
        scene_id=f"gate1-{task_id}-development-scene",
        initial_state=initial_state.manifest_initial_state,
        target_object=objects[0].object_id,
        physical_config=physical_config,
        raw_language=raw_language,
        requested_constraints=requested_constraints,
        stage_scope=stage_scope,
        allowed_control_channels=controls,
        budget={"max_attempts": 3},
        split=DEVELOPMENT_SPLIT,
        geometry_key=f"{task_id}:geometry:pending-calibration",
        manner_key="manner-neutral",
        metadata=dict(metadata),
    )


def _pair_examples(
    task_id: str,
    task_template: Any,
    initial_state: InitialStateReference,
) -> tuple[list[RequestDraft], list[Any]]:
    """Make one structural example of each PairType for one task."""

    phase_ids = [phase.stage_instance_id for phase in task_template.phases]
    first_stage = phase_ids[0]
    second_stage = phase_ids[-1]
    common_metadata = {
        "source": "gate1_dev_manifest_example",
        "task_space_schema_version": TASK_SPACE_SCHEMA_VERSION,
        "initial_state_id": initial_state.initial_state_id,
        "initial_state_hash": initial_state.initial_state_hash,
        "model_sha256": initial_state.model_sha256,
    }
    requests: list[RequestDraft] = []
    pairs: list[Any] = []

    modifier_metadata = {**common_metadata, "pair_example": "modifier_word"}
    modifier_a = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task",
        requested_constraints={"manner": "neutral", "raw_marker": "keep-a"},
        stage_scope=[{"stage_instance_id": first_stage}],
        physical_config={"configuration_ref": f"{task_id}:pending-calibration"},
        metadata=modifier_metadata,
    )
    modifier_b = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task carefully",
        requested_constraints={"manner": "careful", "raw_marker": "keep-b"},
        stage_scope=[{"stage_instance_id": first_stage}],
        physical_config={"configuration_ref": f"{task_id}:pending-calibration"},
        metadata=modifier_metadata,
    )
    requests.extend((modifier_a, modifier_b))
    pairs.append(make_pair(modifier_a, modifier_b, PairType.MODIFIER_WORD, metadata=modifier_metadata))

    attribute_metadata = {**common_metadata, "pair_example": "physical_attribute"}
    attribute_a = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task",
        requested_constraints={"manner": "neutral", "raw_marker": "attribute"},
        stage_scope=[{"stage_instance_id": first_stage}],
        physical_config={"configuration_ref": f"{task_id}:attribute-a-pending"},
        metadata=attribute_metadata,
    )
    attribute_b = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task",
        requested_constraints={"manner": "neutral", "raw_marker": "attribute"},
        stage_scope=[{"stage_instance_id": first_stage}],
        physical_config={"configuration_ref": f"{task_id}:attribute-b-pending"},
        metadata=attribute_metadata,
    )
    requests.extend((attribute_a, attribute_b))
    pairs.append(
        make_pair(
            attribute_a,
            attribute_b,
            PairType.PHYSICAL_ATTRIBUTE,
            metadata={**attribute_metadata, "attribute_intervention_ref": "pending-calibration"},
        )
    )

    stage_metadata = {**common_metadata, "pair_example": "stage_scope"}
    stage_a = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task",
        requested_constraints={"manner": "neutral", "raw_marker": "stage"},
        stage_scope=[{"stage_instance_id": first_stage}],
        physical_config={"configuration_ref": f"{task_id}:pending-calibration"},
        metadata=stage_metadata,
    )
    stage_b = _task_request_common(
        task_id,
        task_template,
        initial_state,
        raw_language=f"complete the {task_id} task",
        requested_constraints={"manner": "neutral", "raw_marker": "stage"},
        stage_scope=[{"stage_instance_id": second_stage}],
        physical_config={"configuration_ref": f"{task_id}:pending-calibration"},
        metadata=stage_metadata,
    )
    requests.extend((stage_a, stage_b))
    pairs.append(make_pair(stage_a, stage_b, PairType.STAGE_SCOPE, metadata=stage_metadata))
    return requests, pairs


def _pending_example(
    task_id: str,
    task_template: Any,
    initial_state: InitialStateReference,
) -> RequestDraft:
    objects = tuple(task_template.objects)
    return RequestDraft(
        task=task_id,
        scene_id=f"gate1-{task_id}-development-pending",
        initial_state=initial_state.manifest_initial_state,
        target_object=objects[0].object_id,
        physical_config={},
        raw_language=f"{task_id} request pending calibration",
        requested_constraints={},
        stage_scope=[],
        allowed_control_channels=(),
        budget={},
        split=DEVELOPMENT_SPLIT,
        geometry_key=f"{task_id}:geometry:pending-calibration",
        manner_key="manner-pending-calibration",
        metadata={
            "source": "gate1_dev_manifest_pending",
            "lifecycle_status": "pending_calibration",
            "applicability": "pending_calibration",
            "incomplete": True,
            "requests_enabled": False,
            "reason": "stage scope, property configuration, and legal domain remain pending calibration",
            "initial_state_id": initial_state.initial_state_id,
            "initial_state_hash": initial_state.initial_state_hash,
            "model_sha256": initial_state.model_sha256,
        },
    )


def build_development_manifest_from_initial_states(
    lift_initial_state_path: str | Path,
    stack_initial_state_path: str | Path,
    *,
    include_pending: bool = True,
) -> list[dict[str, JSONValue]]:
    """Build the small Lift/Stack development manifest.

    The two input paths are required and are never put into request IDs.  The
    output contains six paired requests per task (three pair types) and, by
    default, one explicit pending/incomplete request per task.  All records
    use ``split='development'`` and are returned in the stable order emitted
    by ``build_development_manifest``.
    """

    try:
        refs = {
            "lift": load_initial_state_reference(lift_initial_state_path, "lift"),
            "stack": load_initial_state_reference(stack_initial_state_path, "stack"),
        }
        task_space = build_default_task_space()
        requests: list[RequestDraft] = []
        pairs: list[Any] = []
        for task_id in SUPPORTED_DEVELOPMENT_TASKS:
            task_template = _task_template(task_id, task_space)
            task_requests, task_pairs = _pair_examples(task_id, task_template, refs[task_id])
            requests.extend(task_requests)
            pairs.extend(task_pairs)
            if include_pending:
                requests.append(_pending_example(task_id, task_template, refs[task_id]))
        return build_development_manifest(requests, pairs)
    except (DevelopmentManifestError, ManifestError):
        raise
    except Exception as exc:  # pragma: no cover - keep CLI failures actionable
        raise DevelopmentManifestError("could not build development manifest") from exc


def write_development_manifest(
    path: str | Path,
    records: Sequence[Mapping[str, JSONValue]],
) -> None:
    """Write already-built records through the existing stable JSONL writer."""

    if isinstance(records, (str, bytes)):
        raise DevelopmentManifestError("records must be a sequence of manifest mappings")
    try:
        write_request_manifest(path, records)
    except (TypeError, ValueError, OSError) as exc:
        raise DevelopmentManifestError("could not write development manifest") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lift-initial-state",
        type=Path,
        required=True,
        help="path to the real Lift initial_state.json",
    )
    parser.add_argument(
        "--stack-initial-state",
        type=Path,
        required=True,
        help="path to the real Stack initial_state.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="development request_manifest.jsonl output path",
    )
    parser.add_argument(
        "--exclude-pending",
        action="store_true",
        help="omit explicit pending/incomplete planning entries",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        records = build_development_manifest_from_initial_states(
            args.lift_initial_state,
            args.stack_initial_state,
            include_pending=not args.exclude_pending,
        )
        write_development_manifest(args.output, records)
    except (DevelopmentManifestError, ManifestError) as exc:
        _build_parser().error(str(exc))
    summary = {
        "schema_version": DEV_MANIFEST_SCHEMA_VERSION,
        "manifest_status": "draft",
        "split": DEVELOPMENT_SPLIT,
        "records": len(records),
        "tasks": sorted({record["task"] for record in records}),
        "pair_types": sorted({record["pair_type"] for record in records if record.get("pair_type")}),
        "pending_or_incomplete": sum(
            bool(record.get("metadata", {}).get("incomplete")) for record in records
        ),
        "output": str(Path(args.output).resolve()),
        "external_api_calls": 0,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke test
    raise SystemExit(main())


__all__ = [
    "DEV_MANIFEST_SCHEMA_VERSION",
    "DevelopmentManifestError",
    "InitialStateReference",
    "INITIAL_STATE_REF_SCHEMA_VERSION",
    "SUPPORTED_DEVELOPMENT_TASKS",
    "build_development_manifest_from_initial_states",
    "initial_state_id_for",
    "load_initial_state_reference",
    "main",
    "write_development_manifest",
]
