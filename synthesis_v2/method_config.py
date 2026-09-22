"""Versioned, calibration-gated contracts for the five synthesis methods.

The module validates method configuration at the planning boundary.  It keeps
the method difference in a small, explicit capability block while requiring
one shared request, task-space, public-constraint, executor, retry, stop and
resource contract.  It never loads a model, calls an API, runs a simulator or
reads a final-audit artifact.

The checked-in YAML files under ``configs/generation_methods`` are structural
development configurations.  Numeric constraint thresholds and resource
caps are represented by ``null`` until calibration and Gate 2 budgeting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, TypeAlias


JSONValue: TypeAlias = Any
SCHEMA_VERSION: Final[str] = "synthesis_v2.method_config.v1"
METHOD_IDS: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3", "M")
RESOURCE_DIMENSIONS: Final[tuple[str, ...]] = (
    "physics_substeps",
    "api_calls",
    "api_tokens",
    "cpu_time_s",
    "gpu_time_s",
    "wall_time_s",
    "storage_bytes",
)
ATTEMPT_OUTCOMES: Final[tuple[str, ...]] = (
    "parse_failure",
    "invalid",
    "execution_failure",
    "validation_failure",
    "duplicate",
    "accepted",
)
PAIR_TYPES: Final[tuple[str, ...]] = (
    "modifier_word",
    "physical_attribute",
    "stage_scope",
)
ALLOWED_ABLATIONS: Final[tuple[str, ...]] = (
    "no_phase_local_control",
    "no_context_attributes",
    "no_generation_manner_filter",
    "no_pair_orchestration",
    "no_failure_feedback",
)
_ABLATION_SWITCHES: Final[tuple[str, ...]] = (
    "phase_local_control",
    "context_attributes",
    "generation_manner_filter",
    "pair_orchestration",
    "failure_feedback",
)
_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "method_id", "method_name", "method_version", "shared", "method", "ablations", "metadata"}
)
_SHARED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "executor",
        "request_contract",
        "task_space_contract",
        "task_scope",
        "constraint_source",
        "attempt_policy",
        "budget",
        "validation",
        "reproducibility",
        "audit_access",
    }
)
_METHOD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "planner_kind",
        "proposal_source",
        "uses_language",
        "uses_stage_scope",
        "uses_object_attributes",
        "stage_local_control",
        "generation_manner_filter",
        "pair_orchestration",
        "failure_feedback",
        "sampling",
        "rules",
        "output_fields",
    }
)
_SAMPLING_KEYS: Final[frozenset[str]] = frozenset(
    {"kind", "scope", "seed_policy", "domain_ref", "numeric_domain"}
)
_RULE_KEYS: Final[frozenset[str]] = frozenset(
    {"visibility", "rule_refs", "inline_rules", "hidden_rules"}
)
_ABLATION_KEYS: Final[frozenset[str]] = frozenset({"active", "switches"})
_NUMERIC_DOMAIN_KEYS: Final[frozenset[str]] = frozenset({"lower", "upper"})
_THRESHOLD_KEYS: Final[frozenset[str]] = frozenset({"lower", "upper", "tolerance"})


class MethodConfigError(ValueError):
    """Raised when a method configuration violates the shared contract."""


@dataclass(frozen=True)
class MethodValidationReport:
    """Pure validation result, mirroring the constraint-registry interface."""

    valid: bool
    method_id: str | None
    issues: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return self.issues

    def as_dict(self) -> dict[str, JSONValue]:
        return {"valid": self.valid, "method_id": self.method_id, "issues": list(self.issues)}


@dataclass(frozen=True)
class MethodSuiteValidationReport:
    """Validation result for the complete five-method suite."""

    valid: bool
    method_ids: tuple[str, ...]
    issues: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return self.issues

    def as_dict(self) -> dict[str, JSONValue]:
        return {"valid": self.valid, "method_ids": list(self.method_ids), "issues": list(self.issues)}


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
        raise MethodConfigError(f"{field_name} must be JSON-compatible and finite") from exc


def _json_copy(value: JSONValue, field_name: str) -> JSONValue:
    copied = deepcopy(value)
    _canonical_json(copied, field_name)
    return copied


def canonical_config_hash(value: Mapping[str, JSONValue]) -> str:
    """Return the deterministic hash of a method config without mutating it."""

    if not isinstance(value, Mapping):
        raise MethodConfigError("method config must be a mapping")
    return hashlib.sha256(_canonical_json(dict(value), "method config").encode("utf-8")).hexdigest()


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MethodConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MethodConfigError(f"{field_name} must be boolean")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise MethodConfigError(f"{field_name} must be a mapping")
    return value


def _sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MethodConfigError(f"{field_name} must be a sequence")
    return tuple(value)


def _strings(value: Any, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = _sequence(value, field_name)
    result = tuple(_nonempty(item, field_name) for item in values)
    if not allow_empty and not result:
        raise MethodConfigError(f"{field_name} must not be empty")
    if len(result) != len(set(result)):
        raise MethodConfigError(f"{field_name} must not contain duplicates")
    return result


def _keys(mapping: Mapping[str, Any], allowed: frozenset[str], field_name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise MethodConfigError(f"{field_name} has unknown field(s): {', '.join(unknown)}")


def _no_forbidden_artifact(value: Any, path: str = "config") -> None:
    """Reject sealed/final-audit content and references recursively."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if (
                "sealed" in key_text
                or ("final_audit" in key_text and key_text != "final_audit_feedback")
                or key_text in {"finalaudit", "audit_answer"}
            ):
                raise MethodConfigError(f"{path}.{key} cannot reference or inline final-audit content")
            _no_forbidden_artifact(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _no_forbidden_artifact(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("sealed://") or "sealed_final_audit" in lowered or "final_audit" in lowered:
            raise MethodConfigError(f"{path} cannot contain a final-audit reference")


def _validate_thresholds(value: Any, field_name: str) -> None:
    thresholds = _mapping(value, field_name)
    _keys(thresholds, _THRESHOLD_KEYS, field_name)
    if set(thresholds) != _THRESHOLD_KEYS:
        raise MethodConfigError(f"{field_name} must explicitly contain lower, upper, and tolerance")
    if any(item is not None for item in thresholds.values()):
        raise MethodConfigError(f"{field_name} must keep unknown thresholds null")


def _validate_numeric_domain(value: Any, field_name: str) -> None:
    domain = _mapping(value, field_name)
    _keys(domain, _NUMERIC_DOMAIN_KEYS, field_name)
    if set(domain) != _NUMERIC_DOMAIN_KEYS:
        raise MethodConfigError(f"{field_name} must explicitly contain lower and upper")
    for key, item in domain.items():
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise MethodConfigError(f"{field_name}.{key} must be null or finite numeric")
        raise MethodConfigError(f"{field_name}.{key} must remain null before calibration")


@dataclass(frozen=True)
class MethodConfig:
    """Immutable-by-copy representation of one method YAML document."""

    schema_version: str
    method_id: str
    method_name: str
    method_version: str
    shared: Mapping[str, JSONValue]
    method: Mapping[str, JSONValue]
    ablations: Mapping[str, JSONValue]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _nonempty(self.schema_version, "schema_version"))
        object.__setattr__(self, "method_id", _nonempty(self.method_id, "method_id"))
        object.__setattr__(self, "method_name", _nonempty(self.method_name, "method_name"))
        object.__setattr__(self, "method_version", _nonempty(self.method_version, "method_version"))
        object.__setattr__(self, "shared", _json_copy(dict(_mapping(self.shared, "shared")), "shared"))
        object.__setattr__(self, "method", _json_copy(dict(_mapping(self.method, "method")), "method"))
        object.__setattr__(self, "ablations", _json_copy(dict(_mapping(self.ablations, "ablations")), "ablations"))
        object.__setattr__(self, "metadata", _json_copy(dict(_mapping(self.metadata, "metadata")), "metadata"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, JSONValue]) -> "MethodConfig":
        if not isinstance(value, Mapping):
            raise MethodConfigError("method config must be a mapping")
        _keys(value, _TOP_LEVEL_KEYS, "method config")
        return cls(
            schema_version=value.get("schema_version"),
            method_id=value.get("method_id"),
            method_name=value.get("method_name"),
            method_version=value.get("method_version"),
            shared=value.get("shared"),
            method=value.get("method"),
            ablations=value.get("ablations"),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "method_name": self.method_name,
            "method_version": self.method_version,
            "shared": deepcopy(dict(self.shared)),
            "method": deepcopy(dict(self.method)),
            "ablations": deepcopy(dict(self.ablations)),
            "metadata": deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class LoadedMethodConfig:
    """Path, parsed config and successful validation report."""

    path: str
    config: MethodConfig
    report: MethodValidationReport


_METHOD_PROFILES: Final[dict[str, dict[str, Any]]] = {
    "G0": {
        "planner_kind": "fixed",
        "proposal_source": "fixed_script",
        "uses_language": False,
        "uses_stage_scope": False,
        "uses_object_attributes": False,
        "stage_local_control": False,
        "sampling_kind": "fixed",
        "sampling_scope": "global",
        "rule_refs": (),
    },
    "G1": {
        "planner_kind": "global_random",
        "proposal_source": "global_random",
        "uses_language": False,
        "uses_stage_scope": False,
        "uses_object_attributes": False,
        "stage_local_control": False,
        "sampling_kind": "random",
        "sampling_scope": "global",
        "rule_refs": (),
    },
    "G2": {
        "planner_kind": "stage_random",
        "proposal_source": "stage_random",
        "uses_language": True,
        "uses_stage_scope": True,
        "uses_object_attributes": False,
        "stage_local_control": True,
        "sampling_kind": "random",
        "sampling_scope": "stage",
        "rule_refs": (),
    },
    "G3": {
        "planner_kind": "context_rules",
        "proposal_source": "context_rules",
        "uses_language": True,
        "uses_stage_scope": True,
        "uses_object_attributes": True,
        "stage_local_control": True,
        "sampling_kind": "rules",
        "sampling_scope": "stage",
        "rule_refs": ("task_space.public.v1", "constraint_registry.public_generation.v1"),
    },
    "M": {
        "planner_kind": "context_stage_plan",
        "proposal_source": "context_stage_plan",
        "uses_language": True,
        "uses_stage_scope": True,
        "uses_object_attributes": True,
        "stage_local_control": True,
        "sampling_kind": "context_stage_plan",
        "sampling_scope": "stage",
        "rule_refs": ("task_space.public.v1", "constraint_registry.public_generation.v1"),
    },
}


def _validate_shared(shared: Mapping[str, JSONValue]) -> None:
    _keys(shared, _SHARED_KEYS, "shared")
    executor = _mapping(shared.get("executor"), "shared.executor")
    if set(executor) != {"id", "version", "safety_layer"}:
        raise MethodConfigError("shared.executor must declare id, version, and safety_layer")
    _nonempty(executor.get("id"), "shared.executor.id")
    _nonempty(executor.get("version"), "shared.executor.version")
    _nonempty(executor.get("safety_layer"), "shared.executor.safety_layer")
    if executor.get("id") != "shared_executor.v1":
        raise MethodConfigError("all methods must use shared_executor.v1")
    if executor.get("safety_layer") != "shared_safety.v1":
        raise MethodConfigError("all methods must use shared_safety.v1")

    if shared.get("request_contract") != "synthesis_v2.request.v1":
        raise MethodConfigError("shared.request_contract must be synthesis_v2.request.v1")
    if shared.get("task_space_contract") != "synthesis_v2.task_space.v1":
        raise MethodConfigError("shared.task_space_contract must be synthesis_v2.task_space.v1")
    if tuple(shared.get("task_scope", ())) != ("lift", "stack"):
        raise MethodConfigError("shared.task_scope must be exactly lift and stack")

    source = _mapping(shared.get("constraint_source"), "shared.constraint_source")
    if set(source) != {"registry_ref", "visibility", "allowed_fields", "thresholds", "unknown_policy"}:
        raise MethodConfigError("shared.constraint_source must expose only the public structural contract")
    registry_ref = _nonempty(source.get("registry_ref"), "shared.constraint_source.registry_ref")
    if "sealed" in registry_ref.lower() or "final_audit" in registry_ref.lower():
        raise MethodConfigError("method config cannot reference sealed constraint content")
    if source.get("visibility") != "public":
        raise MethodConfigError("shared.constraint_source.visibility must be public")
    allowed_fields = _strings(source.get("allowed_fields"), "shared.constraint_source.allowed_fields")
    if "thresholds" in allowed_fields or "final_audit_results" in allowed_fields:
        raise MethodConfigError("public constraint fields cannot expose audit thresholds/results")
    _validate_thresholds(source.get("thresholds"), "shared.constraint_source.thresholds")
    if source.get("unknown_policy") != "not_qualified":
        raise MethodConfigError("shared.constraint_source.unknown_policy must be not_qualified")

    attempt = _mapping(shared.get("attempt_policy"), "shared.attempt_policy")
    required_attempt = {
        "max_attempts",
        "budget_scope",
        "outcomes",
        "hidden_retries",
        "stop_on_first_accepted",
        "acceptance_terminal",
        "preserve_original_request",
        "pair_orchestration_source",
    }
    if set(attempt) != required_attempt:
        raise MethodConfigError("shared.attempt_policy fields do not match the common attempt contract")
    if attempt.get("max_attempts") != 3 or isinstance(attempt.get("max_attempts"), bool):
        raise MethodConfigError("all methods must use max_attempts=3")
    if attempt.get("budget_scope") != "per_request_shared":
        raise MethodConfigError("attempt budget must be shared per request")
    outcomes = _strings(attempt.get("outcomes"), "shared.attempt_policy.outcomes")
    if outcomes != ATTEMPT_OUTCOMES:
        raise MethodConfigError("attempt outcomes must include every explicit failure and accepted state")
    if attempt.get("pair_orchestration_source") != "request_manifest":
        raise MethodConfigError("pair orchestration must come from request_manifest")
    for key in ("hidden_retries", "stop_on_first_accepted", "acceptance_terminal", "preserve_original_request"):
        if attempt.get(key) is not (False if key == "hidden_retries" else True):
            raise MethodConfigError(f"shared.attempt_policy.{key} violates the common stop/retry contract")

    budget = _mapping(shared.get("budget"), "shared.budget")
    if set(budget) != {"max_attempts", "resource_dimensions", "caps"}:
        raise MethodConfigError("shared.budget must declare max_attempts, resource_dimensions, and caps")
    if budget.get("max_attempts") != 3:
        raise MethodConfigError("shared.budget.max_attempts must be 3")
    resources = _strings(budget.get("resource_dimensions"), "shared.budget.resource_dimensions")
    if resources != RESOURCE_DIMENSIONS:
        raise MethodConfigError("shared.budget.resource_dimensions must match attempt ledger resources")
    caps = _mapping(budget.get("caps"), "shared.budget.caps")
    if tuple(caps) != RESOURCE_DIMENSIONS:
        raise MethodConfigError("shared.budget.caps must explicitly list every resource dimension")
    if any(value is not None for value in caps.values()):
        raise MethodConfigError("uncalibrated resource caps must remain null")

    validation = _mapping(shared.get("validation"), "shared.validation")
    if set(validation) != {"dimensions", "unknown_policy", "generation_filter", "final_audit_feedback"}:
        raise MethodConfigError("shared.validation must declare public tri-state and filter boundaries")
    dimensions = _strings(validation.get("dimensions"), "shared.validation.dimensions")
    if dimensions != ("task", "manner", "safety"):
        raise MethodConfigError("shared.validation dimensions must be task, manner, safety")
    if validation.get("unknown_policy") != "not_qualified":
        raise MethodConfigError("unknown cannot be accepted by generation validation")
    if validation.get("generation_filter") != "public_constraint_registry":
        raise MethodConfigError("generation filter must use the public constraint registry")
    if validation.get("final_audit_feedback") is not False:
        raise MethodConfigError("final-audit feedback is forbidden during generation")

    reproducibility = _mapping(shared.get("reproducibility"), "shared.reproducibility")
    if set(reproducibility) != {"request_order", "attempt_seed", "model_cache_key", "temperature", "service_version"}:
        raise MethodConfigError("shared.reproducibility fields do not match the reproducibility contract")
    if reproducibility.get("request_order") != "request_id_lexicographic":
        raise MethodConfigError("request order must be request_id_lexicographic")
    if reproducibility.get("attempt_seed") != "sha256(method_id,request_id,attempt_index)":
        raise MethodConfigError("attempt seed policy must use method/request/attempt identity")
    if reproducibility.get("model_cache_key") != "method_id,request_id,attempt_index":
        raise MethodConfigError("model cache key must be explicit and request-bound")
    if reproducibility.get("temperature") is not None or reproducibility.get("service_version") is not None:
        raise MethodConfigError("unknown model/service values must remain null")

    audit = _mapping(shared.get("audit_access"), "shared.audit_access")
    if set(audit) != {"allowed_visibility", "feedback_channel", "artifact_loading"}:
        raise MethodConfigError("shared.audit_access must declare the public-only boundary")
    if audit.get("allowed_visibility") != "public_generation":
        raise MethodConfigError("methods may use only public_generation visibility")
    if audit.get("feedback_channel") != "none":
        raise MethodConfigError("final-audit result feedback must be disabled")
    if audit.get("artifact_loading") is not False:
        raise MethodConfigError("method configs cannot load audit artifacts")


def _validate_rules(rules: Mapping[str, JSONValue], method_id: str) -> None:
    _keys(rules, _RULE_KEYS, f"method[{method_id}].rules")
    visibility = rules.get("visibility")
    if visibility != "public":
        raise MethodConfigError(f"method[{method_id}].rules.visibility must be public")
    refs = _strings(rules.get("rule_refs"), f"method[{method_id}].rules.rule_refs", allow_empty=True)
    if any(
        "sealed" in ref.lower()
        or "private" in ref.lower()
        or "final_audit" in ref.lower()
        or "final-audit" in ref.lower()
        for ref in refs
    ):
        raise MethodConfigError(
            f"method[{method_id}] cannot use sealed/final-audit or hidden rule references"
        )
    inline = _sequence(rules.get("inline_rules"), f"method[{method_id}].rules.inline_rules")
    if inline:
        raise MethodConfigError(f"method[{method_id}] cannot inline hidden or unversioned rules")
    if rules.get("hidden_rules") is not False:
        raise MethodConfigError(f"method[{method_id}] hidden_rules must be false")
    expected_refs = _METHOD_PROFILES[method_id]["rule_refs"]
    if refs != expected_refs:
        raise MethodConfigError(f"method[{method_id}] rule references exceed its declared public profile")


def _validate_method_block(config: MethodConfig) -> None:
    method_id = config.method_id
    if method_id not in METHOD_IDS:
        raise MethodConfigError(f"method_id must be one of {METHOD_IDS}")
    _keys(config.method, _METHOD_KEYS, f"method[{method_id}]")
    profile = _METHOD_PROFILES[method_id]
    method = config.method
    for key in ("planner_kind", "proposal_source"):
        if method.get(key) != profile[key]:
            raise MethodConfigError(f"method[{method_id}].{key} is outside the declared method difference")
    for key in ("uses_language", "uses_stage_scope"):
        if method.get(key) is not profile[key]:
            raise MethodConfigError(f"method[{method_id}].{key} is outside the declared method difference")

    ablations = config.ablations
    _keys(ablations, _ABLATION_KEYS, f"ablations[{method_id}]")
    active = _strings(ablations.get("active"), f"ablations[{method_id}].active", allow_empty=True)
    if any(item not in ALLOWED_ABLATIONS for item in active):
        raise MethodConfigError(f"ablations[{method_id}].active contains an unknown ablation")
    switches = _mapping(ablations.get("switches"), f"ablations[{method_id}].switches")
    if set(switches) != set(_ABLATION_SWITCHES):
        raise MethodConfigError(f"ablations[{method_id}].switches must explicitly declare every switch")
    for key, value in switches.items():
        _bool(value, f"ablations[{method_id}].switches.{key}")
    active_set = set(active)
    ablation_features = {
        "no_phase_local_control": ("phase_local_control", profile["stage_local_control"]),
        "no_context_attributes": ("context_attributes", profile["uses_object_attributes"]),
        # These two mechanisms are available to every full method profile.
        "no_generation_manner_filter": ("generation_manner_filter", True),
        "no_pair_orchestration": ("pair_orchestration", True),
        # Failure feedback is optional and disabled in every checked-in base
        # profile.  An explicit no-failure-feedback ablation is still a valid
        # no-op marker, but it may not turn the switch back on.
        "no_failure_feedback": ("failure_feedback", False),
    }
    for ablation_name, (switch_name, base_enabled) in ablation_features.items():
        if ablation_name in active_set and switches[switch_name]:
            raise MethodConfigError(
                f"ablations[{method_id}] {ablation_name} requires {switch_name}=false"
            )
        if base_enabled and not switches[switch_name] and ablation_name not in active_set:
            raise MethodConfigError(
                f"ablations[{method_id}] must record {ablation_name} when disabling {switch_name}"
            )

    expected_attributes = profile["uses_object_attributes"] and switches["context_attributes"]
    expected_local = profile["stage_local_control"] and switches["phase_local_control"]
    if method.get("uses_object_attributes") is not expected_attributes:
        raise MethodConfigError(f"method[{method_id}] object-attribute access is outside its profile/ablation")
    if method.get("stage_local_control") is not expected_local:
        raise MethodConfigError(f"method[{method_id}] phase-local control is outside its profile/ablation")
    if method.get("generation_manner_filter") is not switches["generation_manner_filter"]:
        raise MethodConfigError(f"method[{method_id}] generation filter switch is inconsistent")
    if method.get("pair_orchestration") is not switches["pair_orchestration"]:
        raise MethodConfigError(f"method[{method_id}] pair orchestration switch is inconsistent")
    if method.get("failure_feedback") is not switches["failure_feedback"]:
        raise MethodConfigError(f"method[{method_id}] failure-feedback switch is inconsistent")

    sampling = _mapping(method.get("sampling"), f"method[{method_id}].sampling")
    _keys(sampling, _SAMPLING_KEYS, f"method[{method_id}].sampling")
    if sampling.get("kind") != profile["sampling_kind"] or sampling.get("scope") != profile["sampling_scope"]:
        raise MethodConfigError(f"method[{method_id}] sampling is outside the declared method profile")
    _nonempty(sampling.get("seed_policy"), f"method[{method_id}].sampling.seed_policy")
    _nonempty(sampling.get("domain_ref"), f"method[{method_id}].sampling.domain_ref")
    _validate_numeric_domain(sampling.get("numeric_domain"), f"method[{method_id}].sampling.numeric_domain")

    _validate_rules(_mapping(method.get("rules"), f"method[{method_id}].rules"), method_id)
    output_fields = _strings(method.get("output_fields"), f"method[{method_id}].output_fields")
    required_outputs = (
        "stage_instance_id",
        "object_id",
        "control_plan",
        "predicted_plan",
        "executed_plan",
    )
    if output_fields != required_outputs:
        raise MethodConfigError(f"method[{method_id}] output fields must preserve planned and executed layers")


def validate_method_config(
    value: MethodConfig | Mapping[str, JSONValue],
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> MethodValidationReport:
    """Validate one config without mutating it or loading external artifacts."""

    method_id: str | None = None
    try:
        config = value if isinstance(value, MethodConfig) else MethodConfig.from_mapping(value)
        method_id = config.method_id
        if config.schema_version != SCHEMA_VERSION:
            raise MethodConfigError(f"schema_version must be {SCHEMA_VERSION}")
        _no_forbidden_artifact(config.to_dict())
        _validate_shared(config.shared)
        _validate_method_block(config)
        if registry_document is not None:
            _validate_registry_boundary(registry_document)
        return MethodValidationReport(True, method_id)
    except (MethodConfigError, TypeError, ValueError) as exc:
        return MethodValidationReport(False, method_id, (str(exc),))


def _validate_registry_boundary(registry_document: Mapping[str, JSONValue]) -> None:
    """Check only that a supplied registry exposes a valid public pending view."""

    if not isinstance(registry_document, Mapping):
        raise MethodConfigError("constraint registry document must be a mapping")
    # Import lazily so a caller validating pure config mappings does not need
    # a YAML parser or any environment package.
    from .constraint_registry import public_generation_view, validate_registry

    report = validate_registry(registry_document)
    if not report.valid:
        raise MethodConfigError("constraint registry is structurally invalid")
    public = public_generation_view(registry_document)
    if "constraints" in public or "thresholds" in public or "sealed_final_audit" in public:
        raise MethodConfigError("public generation view contains forbidden audit content")


def require_valid_method_config(
    value: MethodConfig | Mapping[str, JSONValue],
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> MethodConfig:
    """Return a validated config or raise a single actionable error."""

    config = value if isinstance(value, MethodConfig) else MethodConfig.from_mapping(value)
    report = validate_method_config(config, registry_document=registry_document)
    if not report.valid:
        raise MethodConfigError("; ".join(report.issues))
    return config


def _parse_text(text: str, path: str) -> Mapping[str, JSONValue]:
    if not isinstance(text, str):
        raise MethodConfigError(f"{path} must be text")
    try:
        import yaml  # type: ignore

        document = yaml.safe_load(text)
    except ImportError:
        try:
            document = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise MethodConfigError("PyYAML is unavailable and config is not JSON") from exc
    except Exception as exc:
        raise MethodConfigError(f"method config YAML could not be parsed: {exc}") from exc
    if not isinstance(document, Mapping):
        raise MethodConfigError(f"method config {path} must have a mapping root")
    return document


def load_method_config(
    path: str | Path,
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> MethodConfig:
    """Load and validate one YAML config; no service or simulator is touched."""

    config_path = Path(path)
    try:
        document = _parse_text(config_path.read_text(encoding="utf-8"), str(config_path))
    except OSError as exc:
        raise MethodConfigError(f"could not read method config {config_path}") from exc
    return require_valid_method_config(document, registry_document=registry_document)


def validate_method_suite(
    configs: Sequence[MethodConfig | Mapping[str, JSONValue]],
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> MethodSuiteValidationReport:
    """Validate all five configs and require an identical shared contract."""

    issues: list[str] = []
    parsed: list[MethodConfig] = []
    for index, value in enumerate(configs):
        report = validate_method_config(value, registry_document=registry_document)
        if not report.valid:
            issues.extend(f"config[{index}]: {issue}" for issue in report.issues)
        else:
            parsed.append(value if isinstance(value, MethodConfig) else MethodConfig.from_mapping(value))
    ids = tuple(config.method_id for config in parsed)
    if len(ids) != len(set(ids)):
        issues.append("method IDs must be unique")
    if set(ids) != set(METHOD_IDS):
        issues.append(f"method suite must contain exactly {METHOD_IDS}")
    if parsed:
        shared_hash = canonical_config_hash(parsed[0].shared)
        for config in parsed[1:]:
            if canonical_config_hash(config.shared) != shared_hash:
                issues.append("all methods must share one identical executor/constraint/attempt/budget contract")
                break
    return MethodSuiteValidationReport(not issues, ids, tuple(issues))


def require_valid_method_suite(
    configs: Sequence[MethodConfig | Mapping[str, JSONValue]],
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> tuple[MethodConfig, ...]:
    parsed = tuple(value if isinstance(value, MethodConfig) else MethodConfig.from_mapping(value) for value in configs)
    report = validate_method_suite(parsed, registry_document=registry_document)
    if not report.valid:
        raise MethodConfigError("; ".join(report.issues))
    return parsed


def load_method_suite(
    directory: str | Path,
    *,
    registry_document: Mapping[str, JSONValue] | None = None,
) -> tuple[MethodConfig, ...]:
    """Load exactly the five checked-in ``*.yaml`` method configs."""

    root = Path(directory)
    paths = sorted(root.glob("*.yaml"))
    if not paths:
        raise MethodConfigError(f"no method YAML files found under {root}")
    configs = tuple(load_method_config(path, registry_document=registry_document) for path in paths)
    return require_valid_method_suite(configs, registry_document=registry_document)


_ABLATION_TO_SWITCH: Final[dict[str, str]] = {
    "no_phase_local_control": "phase_local_control",
    "no_context_attributes": "context_attributes",
    "no_generation_manner_filter": "generation_manner_filter",
    "no_pair_orchestration": "pair_orchestration",
    "no_failure_feedback": "failure_feedback",
}


def apply_ablation(config: MethodConfig, ablation: str) -> MethodConfig:
    """Return a validated config with one named mechanism switch disabled."""

    if not isinstance(config, MethodConfig):
        raise MethodConfigError("apply_ablation expects a MethodConfig")
    if not isinstance(ablation, str) or ablation not in ALLOWED_ABLATIONS:
        raise MethodConfigError(f"unknown ablation {ablation!r}; expected one of {ALLOWED_ABLATIONS}")
    document = config.to_dict()
    active = set(document["ablations"]["active"])
    active.add(ablation)
    document["ablations"]["active"] = sorted(active)
    switch = _ABLATION_TO_SWITCH[ablation]
    document["ablations"]["switches"][switch] = False
    if ablation == "no_phase_local_control":
        document["method"]["stage_local_control"] = False
    elif ablation == "no_context_attributes":
        document["method"]["uses_object_attributes"] = False
    elif ablation == "no_generation_manner_filter":
        document["method"]["generation_manner_filter"] = False
    elif ablation == "no_pair_orchestration":
        document["method"]["pair_orchestration"] = False
    elif ablation == "no_failure_feedback":
        document["method"]["failure_feedback"] = False
    return require_valid_method_config(document)


__all__ = [
    "ALLOWED_ABLATIONS",
    "ATTEMPT_OUTCOMES",
    "LoadedMethodConfig",
    "METHOD_IDS",
    "MethodConfig",
    "MethodConfigError",
    "MethodSuiteValidationReport",
    "MethodValidationReport",
    "PAIR_TYPES",
    "RESOURCE_DIMENSIONS",
    "SCHEMA_VERSION",
    "apply_ablation",
    "canonical_config_hash",
    "load_method_config",
    "load_method_suite",
    "require_valid_method_config",
    "require_valid_method_suite",
    "validate_method_config",
    "validate_method_suite",
]
