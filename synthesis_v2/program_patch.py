"""Bounded local patch of a stage program (Step 24, interface item 8 of the gap table).

A proposer (a person, a rule, an LLM) may change the INSTANCE PARAMETERS of a stage program, nothing else:
  * the patch is data (``ops`` of ``set_param`` / ``unset_param``) plus a stated ``hypothesis``; any other top-level field or op is
    refused (``patch_forbidden_field`` / ``capability_unsupported:op:*``), so a patch cannot rewrite the task, the manner request,
    an acceptance threshold or the contact policy;
  * the ORIGINAL request, manner request, acceptance thresholds, contact-policy version, guard and plan are frozen as an anchor
    (name -> sha256); a patch is applied only while the current anchor equals the frozen one;
  * every value goes through ``stage_capabilities`` (declared key, declared primitive, physical domain);
  * the result is a structured ``PatchResult`` (never an exception and never a silent default): the new program, the exact diff,
    the parent / child program hashes and the skeleton hash (everything except params, must be unchanged).
Structure edits (new waypoints, a replaced primitive) are NOT supported yet and are refused as ``capability_unsupported``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from .stage_capabilities import PARAM_SPECS, check_value, param_issues, primitive_issues

PATCH_VERSION = "program_patch.v1"
MAX_OPS = 8
_PATCH_FIELDS = {"version", "hypothesis", "ops"}
_OP_FIELDS = {"op", "instance_id", "key", "value"}
_OPS = ("set_param", "unset_param")


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(x: Any) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()


def make_anchor(**named: Any) -> dict[str, str]:
    """Freeze named parts (request text, manner request, thresholds, policy version, guard, plan ...) as sha256 digests."""
    return {k: _sha(v) for k, v in sorted(named.items())}


def anchor_diff(frozen: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """Names whose digest differs, is missing or was added."""
    return sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))


def skeleton_hash(program) -> str:
    """Hash of the program without instance params (identity of what a patch is not allowed to change)."""
    d = program.to_dict()
    d["instances"] = [{k: v for k, v in i.items() if k != "params"} for i in d["instances"]]
    d.pop("language", None)
    return _sha(d)


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    refusal: Optional[str]                  # None | "anchor_changed" | "invalid_patch" | "capability_unsupported" | "invalid_program"
    errors: tuple = ()
    program: Any = None                     # the patched program when ok
    diff: tuple = ()                        # ({"instance_id", "key", "old", "new"}, ...)
    parent_hash: str = ""
    child_hash: str = ""
    skeleton_hash: str = ""
    hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"version": PATCH_VERSION, "ok": self.ok, "refusal": self.refusal, "errors": list(self.errors),
                "diff": list(self.diff), "parent_hash": self.parent_hash, "child_hash": self.child_hash,
                "skeleton_hash": self.skeleton_hash, "hypothesis": self.hypothesis}


def _refuse(kind: str, errors, program, hypothesis: str = "") -> PatchResult:
    return PatchResult(False, kind, tuple(errors), None, (), program.program_hash(), "", skeleton_hash(program), hypothesis)


def apply_patch(program, patch: Any, *, frozen_anchor: Mapping[str, str], current_anchor: Mapping[str, str],
                plan: Any = None) -> PatchResult:
    """Apply ``patch`` to ``program`` or refuse with a structured reason.  ``plan`` (optional) enables the plan-flag check."""
    changed = anchor_diff(frozen_anchor, current_anchor)
    if changed:
        return _refuse("anchor_changed", [f"anchor_changed:{n}" for n in changed], program)
    if not isinstance(patch, Mapping):
        return _refuse("invalid_patch", ["patch_not_a_mapping"], program)
    errs = [f"patch_forbidden_field:{k}" for k in patch if k not in _PATCH_FIELDS]
    hyp = patch.get("hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        errs.append("patch_missing_hypothesis")
    ops = patch.get("ops")
    if not isinstance(ops, (list, tuple)) or not ops:
        errs.append("patch_has_no_ops")
        ops = []
    if len(ops) > MAX_OPS:
        errs.append(f"patch_too_large:{len(ops)}>{MAX_OPS}")
    if errs:
        return _refuse("invalid_patch", errs, program, hyp if isinstance(hyp, str) else "")
    by_id = {i.instance_id: i for i in program.instances}
    new_params: dict[str, dict] = {}
    diff: list[dict] = []
    unsupported: list[str] = []
    bad: list[str] = []
    seen: set = set()
    for n, op in enumerate(ops):
        if not isinstance(op, Mapping):
            bad.append(f"op_not_a_mapping:{n}")
            continue
        bad += [f"patch_forbidden_field:ops[{n}].{k}" for k in op if k not in _OP_FIELDS]
        name, iid, key = op.get("op"), op.get("instance_id"), op.get("key")
        if name not in _OPS:
            unsupported.append(f"capability_unsupported:op:{name}")
            continue
        if iid not in by_id:
            bad.append(f"unknown_instance:{iid}")
            continue
        if key not in PARAM_SPECS:
            unsupported.append(f"capability_unsupported:{iid}:{key}")
            continue
        if (iid, key) in seen:
            bad.append(f"duplicate_op:{iid}:{key}")
            continue
        seen.add((iid, key))
        cur = new_params.setdefault(iid, dict(by_id[iid].params))
        old = cur.get(key)
        if name == "set_param":
            if "value" not in op:
                bad.append(f"set_param_without_value:{iid}:{key}")
                continue
            why = check_value(PARAM_SPECS[key], op["value"])
            if why:
                bad.append(f"param_invalid:{iid}:{key}:{why}")
                continue
            cur[key] = op["value"]
            diff.append({"instance_id": iid, "key": key, "old": old, "new": op["value"]})
        else:
            if key not in cur:
                bad.append(f"unset_missing_param:{iid}:{key}")
                continue
            del cur[key]
            diff.append({"instance_id": iid, "key": key, "old": old, "new": None})
    if unsupported:
        return _refuse("capability_unsupported", unsupported + bad, program, hyp)
    if bad:
        return _refuse("invalid_patch", bad, program, hyp)
    child = replace(program, instances=tuple(replace(i, params=new_params[i.instance_id]) if i.instance_id in new_params else i
                                              for i in program.instances))
    perr = list(child.validate()) + primitive_issues(child) + [e for e in param_issues(child, plan)[0]]
    perr = list(dict.fromkeys(perr))
    if perr:
        return _refuse("invalid_program", perr, program, hyp)
    sk = skeleton_hash(child)
    assert sk == skeleton_hash(program), "a params-only patch changed the program skeleton"
    return PatchResult(True, None, (), child, tuple(diff), program.program_hash(), child.program_hash(), sk, hyp)
