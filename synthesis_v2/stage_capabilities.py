"""Capability registry of the shared stage-program executor (Step 24, interface items 5 and 8 of the gap table).

What a stage program may ask for is declared HERE, in one place, from what ``program_executor`` actually reads:
  * ``PARAM_SPECS``: every instance parameter the executor consumes, the primitive(s) that consume it, its shape and its
    physical domain.  The domain is the range of the workspace/controller (a unit check), never an acceptance threshold.
  * a key that is not declared is not silently ignored: it is reported as ``capability_unsupported:<instance>:<key>``.
  * ``EXECUTOR_PRIMITIVES``: the primitives the native executor can run (``sweep`` / ``articulate`` belong to other producers).
  * a key that only has an effect under a plan flag (the grasp placement keys need ``approach_check``) is refused when the flag
    is off, instead of being dropped.

This module imports nothing from the program/executor modules (duck-typed) so both can import it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

CAPABILITY_VERSION = "stage_capabilities.v2"

EXECUTOR_PRIMITIVES = ("approach", "descend", "grasp", "lift", "transport", "place", "insert", "retreat")


@dataclass(frozen=True)
class ParamSpec:
    key: str
    effective_on: tuple          # primitives whose executor branch (or subtask geometry) reads the key
    kind: str                    # "float" | "vec3" | "enum" | "dict" | "ref_id"
    lo: Optional[float] = None
    hi: Optional[float] = None
    enum: tuple = ()
    needs_plan_flag: Optional[str] = None
    doc: str = ""
    inert_ok_on: tuple = ()      # primitives where the task configs historically co-apply the key; accepted, reported inert
    proposer_settable: bool = True   # False: valid in a program (legacy configs) but a patch may not set it (ungrounded values)


# Geometry keys are merged by the executor from every instance of the subtask that has a target role (transport, place, insert).
_GEOM = ("transport", "place", "insert")

PARAM_SPECS: dict[str, ParamSpec] = {s.key: s for s in (
    ParamSpec("lift_dz", ("lift",), "float", 0.0, 0.35, doc="eef height above the grasp pose during lift (m)"),
    ParamSpec("transport_dz", ("transport",), "float", 0.0, 0.35, inert_ok_on=("place",),
              doc="eef height above the target while carrying (m)"),
    ParamSpec("place_dz", ("place",), "float", -0.05, 0.25, inert_ok_on=("transport",),
              doc="eef height above the target at release (m)"),
    ParamSpec("place_tol_m", ("place",), "float", 0.002, 0.03, inert_ok_on=("transport",),
              doc="position tolerance of the place servo (m)"),
    ParamSpec("target_offset_site", _GEOM, "vec3", -0.3, 0.3, doc="object target offset in the site frame (m)"),
    ParamSpec("target_align", _GEOM, "enum", enum=("collision_center",), doc="align the object's collision centre to the site"),
    ParamSpec("target_yaw_deg", _GEOM, "float", -360.0, 360.0, doc="object yaw relative to the site (deg)"),
    ParamSpec("target_yaw_symmetry_deg", _GEOM, "float", 1.0, 360.0, doc="rotation after which the object looks the same (deg)"),
    ParamSpec("grasp_reference", ("approach",), "dict", needs_plan_flag="approach_check", proposer_settable=False,
              doc="legacy inline reference: rel_obj (3, object yaw frame, m), hand_yaw_rel_deg, optional hand_quat_obj_xyzw "
                  "(4, unit); a patch selects a registered reference with grasp_reference_id instead of writing numbers"),
    ParamSpec("grasp_reference_id", ("approach",), "ref_id", needs_plan_flag="approach_check",
              doc="id of a registered grasp reference (configs/grasp_references.v1.json); resolved against the program's "
                  "scene and the instance's object; unknown / incompatible ids are refused; excludes grasp_reference"),
    ParamSpec("grasp_candidate", ("approach",), "dict", needs_plan_flag="approach_check",
              doc="one placement: yaw_deg, lateral_m, dz_m, longitudinal_m"),
    ParamSpec("grasp_grid", ("approach",), "dict", needs_plan_flag="approach_check",
              doc="candidate grid: yaws_deg, laterals_m, dzs_m (lists)"),
)}

_REF_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")

_DICT_FIELDS = {
    "grasp_reference": {"rel_obj": ("vec3", -0.3, 0.3), "hand_yaw_rel_deg": ("float", -360.0, 360.0),
                        "hand_quat_obj_xyzw": ("quat", None, None)},
    "grasp_candidate": {"yaw_deg": ("float", -360.0, 360.0), "lateral_m": ("float", -0.1, 0.1),
                        "dz_m": ("float", -0.1, 0.15), "longitudinal_m": ("float", -0.1, 0.1)},
    "grasp_grid": {"yaws_deg": ("list", -360.0, 360.0), "laterals_m": ("list", -0.1, 0.1), "dzs_m": ("list", -0.1, 0.15)},
}
_DICT_REQUIRED = {"grasp_reference": ("rel_obj", "hand_yaw_rel_deg"), "grasp_candidate": ("yaw_deg",),
                  "grasp_grid": ("yaws_deg", "laterals_m", "dzs_m")}


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return float(x)


def _check_field(kind: str, lo, hi, v: Any) -> Optional[str]:
    if kind == "float":
        x = _num(v)
        return "not_a_finite_number" if x is None else (f"out_of_range[{lo},{hi}]:{x}" if x < lo or x > hi else None)
    if kind in ("vec3", "list"):
        if not isinstance(v, (list, tuple)) or (kind == "vec3" and len(v) != 3) or (kind == "list" and not v):
            return "bad_length"
        for e in v:
            x = _num(e)
            if x is None:
                return "not_a_finite_number"
            if x < lo or x > hi:
                return f"out_of_range[{lo},{hi}]:{x}"
        return None
    if kind == "quat":
        if not isinstance(v, (list, tuple)) or len(v) != 4 or any(_num(e) is None for e in v):
            return "bad_length"
        n = math.sqrt(sum(float(e) ** 2 for e in v))
        return None if abs(n - 1.0) <= 1e-2 else f"not_unit:{n:.4f}"
    return "unknown_kind"


def check_value(spec: ParamSpec, value: Any) -> Optional[str]:
    """None when ``value`` is acceptable for ``spec``; else a short reason."""
    if spec.kind in ("float", "vec3"):
        return _check_field(spec.kind, spec.lo, spec.hi, value)
    if spec.kind == "enum":
        return None if value in spec.enum else f"not_in:{list(spec.enum)}"
    if spec.kind == "ref_id":
        return None if isinstance(value, str) and _REF_ID_RE.match(value) else "not_a_reference_id"
    if spec.kind == "dict":
        if not isinstance(value, Mapping):
            return "not_a_mapping"
        fields = _DICT_FIELDS[spec.key]
        for k in value:
            if k not in fields:
                return f"unknown_field:{k}"
        for k in _DICT_REQUIRED[spec.key]:
            if k not in value:
                return f"missing_field:{k}"
        for k, v in value.items():
            kind, lo, hi = fields[k]
            why = _check_field(kind, lo, hi, v)
            if why:
                return f"{k}:{why}"
        return None
    return "unknown_kind"


def param_issues(program, plan: Any = None) -> tuple[list[str], list[str]]:
    """(errors, notes) for the instance params of ``program``.

    errors: ``capability_unsupported:<id>:<key>`` (key not declared / declared only for other primitives),
            ``capability_requires:<id>:<key>:<plan flag>`` (plan given and the flag is off),
            ``param_invalid:<id>:<key>:<reason>`` (type / domain).
    notes:  ``param_inert:<id>:<key>`` for a declared key that this primitive does not read (historical co-application).
    ``plan`` is optional: the plan-flag check only runs when it is passed."""
    errs: list[str] = []
    notes: list[str] = []
    for it in program.instances:
        for key, value in dict(it.params).items():
            spec = PARAM_SPECS.get(key)
            if spec is None:
                errs.append(f"capability_unsupported:{it.instance_id}:{key}")
                continue
            if it.primitive not in spec.effective_on:
                if it.primitive in spec.inert_ok_on:
                    notes.append(f"param_inert:{it.instance_id}:{key}")
                else:
                    errs.append(f"capability_unsupported:{it.instance_id}:{key}:not_read_by_{it.primitive}")
                    continue
            why = check_value(spec, value)
            if why:
                errs.append(f"param_invalid:{it.instance_id}:{key}:{why}")
            if plan is not None and spec.needs_plan_flag and not getattr(plan, spec.needs_plan_flag, False):
                errs.append(f"capability_requires:{it.instance_id}:{key}:{spec.needs_plan_flag}")
    return errs, notes


def primitive_issues(program) -> list[str]:
    """Primitives that are valid program vocabulary but that the native executor cannot run."""
    return [f"capability_unsupported:{it.instance_id}:primitive:{it.primitive}"
            for it in program.instances if it.primitive not in EXECUTOR_PRIMITIVES]


def describe() -> dict:
    """The capability table as plain data (what a proposer is shown)."""
    return {"version": CAPABILITY_VERSION, "primitives": list(EXECUTOR_PRIMITIVES),
            "params": {k: {"effective_on": list(s.effective_on), "kind": s.kind, "lo": s.lo, "hi": s.hi,
                           "enum": list(s.enum), "needs_plan_flag": s.needs_plan_flag, "doc": s.doc,
                           "proposer_settable": s.proposer_settable,
                           **({"fields": {f: list(v[:1]) + [v[1], v[2]] for f, v in _DICT_FIELDS[k].items()},
                               "required": list(_DICT_REQUIRED[k])} if s.kind == "dict" else {})}
                       for k, s in PARAM_SPECS.items()}}
