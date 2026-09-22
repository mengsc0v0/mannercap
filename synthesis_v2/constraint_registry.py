"""Versioned, calibration-gated constraint registry for MannerCAP v2.

The registry is deliberately independent of the generator and final evaluator.
It exposes only stage/metric references to the public generation section and
stores a reference to a separately sealed final-audit configuration.  Numeric
thresholds for the current Lift/Stack development registry remain ``null``
while calibration is pending.  A structurally valid pending registry is useful
for development but returns ``unknown`` and can never be promoted.

This module validates an already parsed mapping without mutating it.  The thin
``load_registry`` helper adds YAML file loading for the checked-in registry;
the validation and hash functions remain pure and standard-library based after
parsing.  No simulator, model, or physical calibration is run here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .contact_force_contract import Decision


SCHEMA_VERSION = "synthesis_v2.constraint_registry.v1"
HASH_ALGORITHM = "sha256"
REQUIRED_DECISION_DIMENSIONS: Tuple[str, ...] = ("task", "manner", "safety")
DECISION_STATES: Tuple[str, ...] = ("pass", "fail", "unknown")
REQUIRED_TASKS: Tuple[str, ...] = ("lift", "stack")
REQUIRED_THRESHOLD_KEYS: Tuple[str, ...] = ("lower", "upper", "tolerance")
METRIC_UNITS: Mapping[str, str] = {
    "joint_velocity": "rad/s",
    "gripper_fraction": "fraction",
    "force_norm": "N",
    "torque_norm": "N*m",
}


class RegistryError(ValueError):
    """Raised when a registry cannot be parsed as a mapping."""


class CalibrationStatus(str, Enum):
    PENDING = "pending_calibration"
    CALIBRATED = "calibrated"


class RegistryState(str, Enum):
    DEVELOPMENT_PENDING = "development_pending"
    FROZEN = "frozen"


@dataclass(frozen=True)
class RegistryIssue:
    """One actionable registry contract violation."""

    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class RegistryValidationReport:
    """Validation, promotion, and hash evidence for one registry document."""

    valid: bool
    promotion_allowed: bool
    decision: Decision
    registry_id: Optional[str]
    registry_version: Optional[str]
    declared_hash: Optional[str]
    computed_hash: Optional[str]
    issues: Tuple[RegistryIssue, ...] = ()

    @property
    def promotable(self) -> bool:
        return self.promotion_allowed

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def has_code(self, code: str) -> bool:
        return code in self.codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "promotion_allowed": self.promotion_allowed,
            "decision": self.decision.value,
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "declared_hash": self.declared_hash,
            "computed_hash": self.computed_hash,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class LoadedRegistry:
    """Parsed document and its pure validation report."""

    path: str
    document: Mapping[str, Any]
    report: RegistryValidationReport


def canonical_registry_hash(document: Mapping[str, Any]) -> str:
    """Hash canonical registry content while excluding its self-hash field."""

    if not isinstance(document, Mapping):
        raise RegistryError("registry document must be a mapping")
    copied = deepcopy(dict(document))
    copied.pop("registry_version_hash", None)
    try:
        payload = json.dumps(
            copied,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryError("registry values must be JSON-compatible and finite") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_registry_text(text: str) -> Mapping[str, Any]:
    """Parse YAML (or JSON-as-YAML) without validating the registry contract."""

    if not isinstance(text, str):
        raise RegistryError("registry text must be a string")
    try:
        import yaml  # type: ignore

        document = yaml.safe_load(text)
    except ImportError:
        try:
            document = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise RegistryError("PyYAML is unavailable and registry is not JSON") from exc
    except Exception as exc:
        raise RegistryError(f"registry YAML could not be parsed: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RegistryError("registry root must be a mapping")
    return document


def load_registry(path: str | Path) -> LoadedRegistry:
    """Load a checked-in YAML registry and return its validation report."""

    registry_path = Path(path)
    try:
        document = parse_registry_text(registry_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegistryError(f"could not read registry {registry_path}: {exc}") from exc
    return LoadedRegistry(
        path=str(registry_path),
        document=document,
        report=validate_registry(document),
    )


def public_generation_view(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return only the public generation section, excluding sealed contents."""

    report = validate_registry(document)
    if not report.valid:
        raise RegistryError("cannot expose public generation view of an invalid registry")
    public = document.get("public_generation")
    if not isinstance(public, Mapping):  # guarded by validate_registry
        raise RegistryError("registry has no public generation section")
    return deepcopy(dict(public))


def sealed_final_audit_reference(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return sealed metadata/reference without loading its separate artifact."""

    report = validate_registry(document)
    if not report.valid:
        raise RegistryError("cannot expose sealed audit reference of an invalid registry")
    sealed = document.get("sealed_final_audit")
    if not isinstance(sealed, Mapping):  # guarded by validate_registry
        raise RegistryError("registry has no sealed final-audit section")
    return deepcopy(dict(sealed))


def validate_registry(document: Mapping[str, Any]) -> RegistryValidationReport:
    """Validate schema, units, phase scope, tri-state policy, and promotion.

    The function has two separate outcomes.  ``valid`` means the pending
    development document is structurally honest.  ``promotion_allowed`` means
    every threshold and sealed-audit reference has independent calibration
    evidence.  Lift/Stack's checked-in development document should be valid but
    not promotable.
    """

    issues: list[RegistryIssue] = []
    if not isinstance(document, Mapping):
        return RegistryValidationReport(
            valid=False,
            promotion_allowed=False,
            decision=Decision.FAIL,
            registry_id=None,
            registry_version=None,
            declared_hash=None,
            computed_hash=None,
            issues=(RegistryIssue("root_not_mapping", "$", "registry root must be a mapping"),),
        )

    def add(code: str, path: str, message: str) -> None:
        issues.append(RegistryIssue(code, path, message))

    registry_id = _nonempty_string(document.get("registry_id"))
    registry_version = _nonempty_string(document.get("registry_version"))
    declared_hash = _nonempty_string(document.get("registry_version_hash"))
    if registry_id is None:
        add("missing_registry_id", "registry_id", "registry_id is required")
    if registry_version is None:
        add("missing_registry_version", "registry_version", "registry_version is required")
    if document.get("schema_version") != SCHEMA_VERSION:
        add("wrong_schema_version", "schema_version", f"schema_version must be {SCHEMA_VERSION!r}")
    if document.get("hash_algorithm") != HASH_ALGORITHM:
        add("wrong_hash_algorithm", "hash_algorithm", "hash_algorithm must be sha256")
    if not _is_hash(declared_hash):
        add("invalid_registry_hash", "registry_version_hash", "registry_version_hash must be 64 hex characters")

    try:
        computed_hash = canonical_registry_hash(document)
    except RegistryError as exc:
        computed_hash = None
        add("noncanonical_registry", "$", str(exc))
    if declared_hash is not None and computed_hash is not None and declared_hash.lower() != computed_hash:
        add("registry_hash_mismatch", "registry_version_hash", "declared registry hash does not match canonical content")

    state = document.get("status")
    if state not in {member.value for member in RegistryState}:
        add("invalid_registry_status", "status", "status must be development_pending or frozen")

    top_calibration = document.get("calibration")
    _validate_calibration_block(
        top_calibration,
        "calibration",
        add,
        require_evidence=isinstance(top_calibration, Mapping)
        and top_calibration.get("status") == CalibrationStatus.CALIBRATED.value,
    )
    if state == RegistryState.DEVELOPMENT_PENDING.value and isinstance(top_calibration, Mapping) and top_calibration.get("status") == CalibrationStatus.CALIBRATED.value:
        add("state_calibration_mismatch", "status", "development_pending registry cannot declare calibrated top-level evidence")
    if state == RegistryState.FROZEN.value and isinstance(top_calibration, Mapping) and top_calibration.get("status") != CalibrationStatus.CALIBRATED.value:
        add("frozen_registry_not_calibrated", "calibration.status", "frozen registry requires calibrated top-level evidence")

    _validate_decision_contract(document.get("decision_contract"), add)
    _validate_visibility_sections(document, add)
    task_constraint_ids = _validate_tasks(document.get("tasks"), top_calibration, add)
    _validate_public_refs(document.get("public_generation"), task_constraint_ids, add)

    valid = not issues
    promotion_allowed = valid and _promotion_ready(document)
    if valid and not promotion_allowed:
        decision = Decision.UNKNOWN
    elif valid:
        decision = Decision.PASS
    else:
        decision = Decision.FAIL
    return RegistryValidationReport(
        valid=valid,
        promotion_allowed=promotion_allowed,
        decision=decision,
        registry_id=registry_id,
        registry_version=registry_version,
        declared_hash=declared_hash,
        computed_hash=computed_hash,
        issues=tuple(issues),
    )


def _validate_decision_contract(value: Any, add) -> None:
    if not isinstance(value, Mapping):
        add("missing_decision_contract", "decision_contract", "decision_contract mapping is required")
        return
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping):
        add("missing_decision_dimensions", "decision_contract.dimensions", "task/manner/safety dimensions are required")
    else:
        for dimension in REQUIRED_DECISION_DIMENSIONS:
            block = dimensions.get(dimension)
            path = f"decision_contract.dimensions.{dimension}"
            if not isinstance(block, Mapping):
                add("missing_decision_dimension", path, f"{dimension} decision dimension is required")
                continue
            states = block.get("states")
            if not _exact_state_set(states):
                add("invalid_decision_states", f"{path}.states", "states must contain exactly pass, fail, unknown")
            if block.get("unknown_is_qualified") is not False:
                add("unknown_policy_allows_qualification", f"{path}.unknown_is_qualified", "unknown must never be qualified")
    if value.get("unknown_policy") != "not_qualified":
        add("invalid_unknown_policy", "decision_contract.unknown_policy", "unknown_policy must be not_qualified")
    joint_rule = value.get("joint_rule")
    expected_rule = {"any_fail": "fail", "all_pass": "pass", "otherwise": "unknown"}
    if not isinstance(joint_rule, Mapping) or any(joint_rule.get(k) != v for k, v in expected_rule.items()):
        add("invalid_joint_rule", "decision_contract.joint_rule", "joint rule must be any_fail=fail, all_pass=pass, otherwise=unknown")


def _validate_visibility_sections(document: Mapping[str, Any], add) -> None:
    public = document.get("public_generation")
    sealed = document.get("sealed_final_audit")
    if not isinstance(public, Mapping):
        add("missing_public_generation", "public_generation", "public generation section is required")
    else:
        if public.get("visibility") != "public":
            add("invalid_public_visibility", "public_generation.visibility", "public generation visibility must be public")
        if not _nonempty_string(public.get("config_id")):
            add("missing_public_config_id", "public_generation.config_id", "public config_id is required")
        refs = public.get("constraint_refs")
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs:
            add("missing_public_constraint_refs", "public_generation.constraint_refs", "public constraint references are required")
        for forbidden in ("constraints", "thresholds", "numeric_thresholds", "sealed_constraints"):
            if forbidden in public:
                add("public_inline_audit_data", f"public_generation.{forbidden}", "public generation cannot inline final-audit thresholds")
        audit_ref = public.get("final_audit_ref")
        if not isinstance(audit_ref, Mapping) or not _nonempty_string(audit_ref.get("config_id")):
            add("missing_public_audit_ref", "public_generation.final_audit_ref", "public section must carry only a reference to sealed audit config")
        elif audit_ref.get("visibility") not in (None, "sealed_reference"):
            add("invalid_public_audit_ref_visibility", "public_generation.final_audit_ref.visibility", "public audit reference must be sealed_reference")

    if not isinstance(sealed, Mapping):
        add("missing_sealed_final_audit", "sealed_final_audit", "sealed final-audit reference is required")
    else:
        if sealed.get("visibility") != "sealed":
            add("invalid_sealed_visibility", "sealed_final_audit.visibility", "sealed final audit must declare sealed visibility")
        if not _nonempty_string(sealed.get("config_id")):
            add("missing_sealed_config_id", "sealed_final_audit.config_id", "sealed config_id is required")
        if not _nonempty_string(sealed.get("config_ref")):
            add("missing_sealed_config_ref", "sealed_final_audit.config_ref", "sealed config_ref is required")
        status = sealed.get("status")
        if status not in {"sealed_pending", "sealed_ready"}:
            add("invalid_sealed_status", "sealed_final_audit.status", "sealed status must be sealed_pending or sealed_ready")
        if status == "sealed_ready" and not _is_hash(sealed.get("config_sha256")):
            add("missing_sealed_hash", "sealed_final_audit.config_sha256", "sealed_ready audit config needs a sha256")
        if status == "sealed_pending" and sealed.get("config_sha256") is not None:
            add("premature_sealed_hash", "sealed_final_audit.config_sha256", "pending sealed config cannot carry an asserted hash")
        for forbidden in ("constraints", "thresholds", "numeric_thresholds", "inline_config"):
            if forbidden in sealed:
                add("sealed_inline_data", f"sealed_final_audit.{forbidden}", "sealed audit must remain a separate referenced artifact")

    if isinstance(public, Mapping) and isinstance(sealed, Mapping):
        public_ref = public.get("final_audit_ref")
        if isinstance(public_ref, Mapping) and public_ref.get("config_id") != sealed.get("config_id"):
            add("audit_reference_mismatch", "public_generation.final_audit_ref.config_id", "public and sealed audit config IDs must match")
        if public.get("config_id") == sealed.get("config_id"):
            add("public_sealed_id_collision", "public_generation.config_id", "public generation config and sealed audit config must be distinct")


def _validate_tasks(value: Any, top_calibration: Any, add) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        add("missing_tasks", "tasks", "tasks must be a non-empty sequence")
        return []
    seen_tasks: set[str] = set()
    all_constraint_ids: list[str] = []
    for task_index, task in enumerate(value):
        path = f"tasks[{task_index}]"
        if not isinstance(task, Mapping):
            add("invalid_task", path, "task entry must be a mapping")
            continue
        task_id = _nonempty_string(task.get("task_id"))
        if task_id is None:
            add("missing_task_id", f"{path}.task_id", "task_id is required")
            continue
        if task_id in seen_tasks:
            add("duplicate_task_id", f"{path}.task_id", "task_id values must be unique")
        seen_tasks.add(task_id)
        if task_id not in REQUIRED_TASKS:
            add("unknown_task", f"{path}.task_id", f"only development tasks {REQUIRED_TASKS} are registered")
        if task.get("status") != "supported":
            add("invalid_task_status", f"{path}.status", "Lift/Stack development tasks must be supported")
        top_status = top_calibration.get("status") if isinstance(top_calibration, Mapping) else None
        task_status = task.get("calibration_status")
        if task_status not in {CalibrationStatus.PENDING.value, CalibrationStatus.CALIBRATED.value}:
            add("invalid_task_calibration_status", f"{path}.calibration_status", "task calibration_status must be pending_calibration or calibrated")
        elif top_status == CalibrationStatus.PENDING.value and task_status != CalibrationStatus.PENDING.value:
            add("task_calibration_mismatch", f"{path}.calibration_status", "pending top-level calibration requires pending task status")
        elif top_status == CalibrationStatus.CALIBRATED.value and task_status != CalibrationStatus.CALIBRATED.value:
            add("task_calibration_mismatch", f"{path}.calibration_status", "calibrated top-level registry requires calibrated task status")
        phases = task.get("phases")
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or not phases:
            add("missing_phases", f"{path}.phases", "each task needs at least one named phase")
            phase_ids: list[str] = []
        else:
            phase_ids = []
            for phase_index, phase in enumerate(phases):
                if not isinstance(phase, str) or not phase.strip():
                    add("invalid_phase", f"{path}.phases[{phase_index}]", "phase IDs must be non-empty strings")
                else:
                    phase_ids.append(phase)
            if len(phase_ids) != len(set(phase_ids)):
                add("duplicate_phase", f"{path}.phases", "phase IDs must be unique within a task")

        constraints = task.get("constraints")
        if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
            add("missing_constraints", f"{path}.constraints", "each task needs a constraint sequence")
            constraints = []
        if not constraints:
            add("missing_constraints", f"{path}.constraints", "an empty constraint list cannot qualify a task")
        seen_constraint_ids: set[str] = set()
        constrained_phases: set[str] = set()
        for constraint_index, constraint in enumerate(constraints):
            cpath = f"{path}.constraints[{constraint_index}]"
            if not isinstance(constraint, Mapping):
                add("invalid_constraint", cpath, "constraint entry must be a mapping")
                continue
            cid = _nonempty_string(constraint.get("constraint_id"))
            if cid is None:
                add("missing_constraint_id", f"{cpath}.constraint_id", "constraint_id is required")
            elif cid in seen_constraint_ids or cid in all_constraint_ids:
                add("duplicate_constraint_id", f"{cpath}.constraint_id", "constraint IDs must be globally unique")
            else:
                seen_constraint_ids.add(cid)
                all_constraint_ids.append(cid)
            phase = constraint.get("phase")
            if not isinstance(phase, str) or not phase.strip() or phase not in phase_ids:
                add("invalid_constraint_phase", f"{cpath}.phase", "constraint phase must reference a declared task phase")
            else:
                constrained_phases.add(phase)
            _validate_constraint(constraint, cpath, task, top_calibration, add)
        for phase in phase_ids:
            if phase not in constrained_phases:
                add("phase_without_constraint", f"{path}.phases", f"phase {phase!r} has no registered constraint")
    missing_tasks = [task_id for task_id in REQUIRED_TASKS if task_id not in seen_tasks]
    for task_id in missing_tasks:
        add("missing_task", "tasks", f"required development task {task_id!r} is missing")
    return all_constraint_ids


def _validate_constraint(constraint: Mapping[str, Any], path: str, task: Mapping[str, Any], top_calibration: Any, add) -> None:
    task_id = task.get("task_id")
    if constraint.get("task_id") != task_id:
        add("constraint_task_mismatch", f"{path}.task_id", "constraint task_id must match its containing task")
    metric = constraint.get("metric")
    unit = constraint.get("unit")
    if metric not in METRIC_UNITS:
        add("unknown_metric", f"{path}.metric", f"metric must be one of {tuple(METRIC_UNITS)}")
    elif unit != METRIC_UNITS[metric]:
        add("wrong_unit", f"{path}.unit", f"metric {metric!r} requires unit {METRIC_UNITS[metric]!r}")
    elif not isinstance(unit, str) or not unit.strip():
        add("missing_unit", f"{path}.unit", "unit is required")

    scope = constraint.get("scope")
    phase = constraint.get("phase")
    if not isinstance(scope, Mapping):
        add("missing_scope", f"{path}.scope", "stage scope mapping is required")
    else:
        if scope.get("kind") != "stage":
            add("invalid_scope_kind", f"{path}.scope.kind", "constraint scope kind must be stage")
        if scope.get("stage_instance_id") != phase:
            add("scope_phase_mismatch", f"{path}.scope.stage_instance_id", "scope stage_instance_id must equal phase")

    dimensions = constraint.get("decision_dimensions")
    try:
        dimensions_match = (
            isinstance(dimensions, Sequence)
            and not isinstance(dimensions, (str, bytes))
            and set(dimensions) == set(REQUIRED_DECISION_DIMENSIONS)
        )
    except TypeError:
        dimensions_match = False
    if not dimensions_match:
        add("invalid_constraint_dimensions", f"{path}.decision_dimensions", "each constraint must declare task, manner, and safety dimensions")
    if constraint.get("unknown_policy") != "not_qualified":
        add("invalid_constraint_unknown_policy", f"{path}.unknown_policy", "constraint unknown_policy must be not_qualified")
    top_status = top_calibration.get("status") if isinstance(top_calibration, Mapping) else None
    constraint_status = constraint.get("status")
    if constraint_status not in {CalibrationStatus.PENDING.value, CalibrationStatus.CALIBRATED.value}:
        add("invalid_constraint_status", f"{path}.status", "constraint status must be pending_calibration or calibrated")
    elif top_status == CalibrationStatus.PENDING.value and constraint_status != CalibrationStatus.PENDING.value:
        add("constraint_calibration_mismatch", f"{path}.status", "pending top-level calibration requires pending constraint status")
    elif top_status == CalibrationStatus.CALIBRATED.value and constraint_status != CalibrationStatus.CALIBRATED.value:
        add("constraint_calibration_mismatch", f"{path}.status", "calibrated top-level registry requires calibrated constraint status")

    thresholds = constraint.get("thresholds")
    threshold_values: list[Any] = []
    if not isinstance(thresholds, Mapping):
        add("missing_thresholds", f"{path}.thresholds", "threshold mapping with lower/upper/tolerance is required")
    else:
        for key in REQUIRED_THRESHOLD_KEYS:
            if key not in thresholds:
                add("missing_threshold_field", f"{path}.thresholds.{key}", "threshold key is required even when pending")
            else:
                value = thresholds.get(key)
                threshold_values.append(value)
                if value is not None and _finite_nonnegative(value) is None:
                    add("invalid_threshold", f"{path}.thresholds.{key}", "threshold must be null or finite non-negative")
        if any(value is not None for value in threshold_values) and constraint_status != CalibrationStatus.CALIBRATED.value:
            add("pending_threshold_value", f"{path}.thresholds", "uncalibrated development thresholds must remain null")
        if constraint_status == CalibrationStatus.CALIBRATED.value and any(value is None for value in threshold_values):
            add("missing_calibrated_threshold", f"{path}.thresholds", "calibrated constraints require numeric lower/upper/tolerance")
        lower = thresholds.get("lower")
        upper = thresholds.get("upper")
        if lower is not None and upper is not None and _finite_nonnegative(lower) is not None and _finite_nonnegative(upper) is not None and float(lower) > float(upper):
            add("reversed_thresholds", f"{path}.thresholds", "lower threshold cannot exceed upper threshold")

    calibration = constraint.get("calibration")
    _validate_calibration_block(
        calibration,
        f"{path}.calibration",
        add,
        require_evidence=constraint_status == CalibrationStatus.CALIBRATED.value,
    )
    if isinstance(calibration, Mapping) and constraint_status in {CalibrationStatus.PENDING.value, CalibrationStatus.CALIBRATED.value} and calibration.get("status") != constraint_status:
        add("constraint_calibration_status_mismatch", f"{path}.calibration.status", "constraint calibration status must match constraint status")
    if isinstance(top_calibration, Mapping) and top_calibration.get("status") == CalibrationStatus.PENDING.value and isinstance(calibration, Mapping) and any(value is not None for value in threshold_values):
        add("threshold_before_calibration", f"{path}.thresholds", "top-level calibration is pending; threshold cannot be promoted")


def _validate_calibration_block(value: Any, path: str, add, *, require_evidence: bool) -> None:
    if not isinstance(value, Mapping):
        add("missing_calibration_source", path, "calibration block is required")
        return
    status = value.get("status")
    if status not in {member.value for member in CalibrationStatus}:
        add("invalid_calibration_status", f"{path}.status", "calibration status must be pending_calibration or calibrated")
    if not _nonempty_string(value.get("source_ref")):
        add("missing_calibration_source", f"{path}.source_ref", "calibration source_ref is required")
    if not _nonempty_string(value.get("version")):
        add("missing_calibration_version", f"{path}.version", "calibration version is required")
    evidence = value.get("evidence_sha256")
    if require_evidence or status == CalibrationStatus.CALIBRATED.value:
        if not _is_hash(evidence):
            add("missing_calibration_evidence_hash", f"{path}.evidence_sha256", "calibrated block needs an evidence sha256")
    elif evidence is not None and not _is_hash(evidence):
        add("invalid_calibration_evidence_hash", f"{path}.evidence_sha256", "evidence_sha256 must be null or 64 hex characters")


def _validate_public_refs(public: Any, constraint_ids: Sequence[str], add) -> None:
    if not isinstance(public, Mapping):
        return
    refs = public.get("constraint_refs")
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        refs_tuple = tuple(refs)
        try:
            refs_are_unique = len(refs_tuple) == len(set(refs_tuple))
            refs_match = set(refs_tuple) == set(constraint_ids)
        except TypeError:
            refs_are_unique = False
            refs_match = False
        if not refs_are_unique:
            add("duplicate_public_constraint_ref", "public_generation.constraint_refs", "public references must be unique")
        if not refs_match:
            add("public_constraint_ref_mismatch", "public_generation.constraint_refs", "public refs must match task constraint IDs exactly")
    audit_ref = public.get("final_audit_ref")
    if isinstance(audit_ref, Mapping) and audit_ref.get("visibility") not in (None, "sealed_reference"):
        add("invalid_public_audit_visibility", "public_generation.final_audit_ref.visibility", "public audit ref must not expose sealed contents")


def _promotion_ready(document: Mapping[str, Any]) -> bool:
    top = document.get("calibration")
    sealed = document.get("sealed_final_audit")
    if not isinstance(top, Mapping) or top.get("status") != CalibrationStatus.CALIBRATED.value:
        return False
    if not isinstance(sealed, Mapping) or sealed.get("status") != "sealed_ready" or not _is_hash(sealed.get("config_sha256")):
        return False
    tasks = document.get("tasks")
    if not isinstance(tasks, Sequence):
        return False
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("calibration_status") != CalibrationStatus.CALIBRATED.value:
            return False
        constraints = task.get("constraints")
        if not isinstance(constraints, Sequence):
            return False
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                return False
            calibration = constraint.get("calibration")
            thresholds = constraint.get("thresholds")
            if not isinstance(calibration, Mapping) or calibration.get("status") != CalibrationStatus.CALIBRATED.value:
                return False
            if not isinstance(thresholds, Mapping) or any(thresholds.get(key) is None for key in REQUIRED_THRESHOLD_KEYS):
                return False
    return True


def _exact_state_set(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) == len(DECISION_STATES) and set(value) == set(DECISION_STATES)
    except TypeError:
        return False


def _nonempty_string(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _finite_nonnegative(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


__all__ = [
    "CalibrationStatus",
    "DECISION_STATES",
    "HASH_ALGORITHM",
    "LoadedRegistry",
    "METRIC_UNITS",
    "REQUIRED_DECISION_DIMENSIONS",
    "REQUIRED_TASKS",
    "REQUIRED_THRESHOLD_KEYS",
    "RegistryError",
    "RegistryIssue",
    "RegistryState",
    "RegistryValidationReport",
    "SCHEMA_VERSION",
    "canonical_registry_hash",
    "load_registry",
    "parse_registry_text",
    "public_generation_view",
    "sealed_final_audit_reference",
    "validate_registry",
]
