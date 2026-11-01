"""Registered grasp references (Step 24, interface item 1 of the gap table).

A grasp reference is the hand pose of an official demonstration at a gripper close, expressed in the object's frame.  Until now a
reference was an inline dict in a task config; a proposer could neither name it nor tell where it came from.  Here every
reference is a library entry with
  * a stable ``reference_id`` (append-only; the entry content is pinned by ``content_sha256``, a changed value is a new id);
  * provenance: the demonstration file (name + sha256), demo id, the capture instant (control step of the close), the
    derivation record(s) in this repository (path + sha256) and the extraction script(s) (path + sha256);
  * the coordinate-frame definition of every value (``frame``);
  * the scene and the objects it is valid for (``scene``, ``compatible_objects`` + the stated basis);
  * the demo's own contact evidence at the close when a record exists (``demo_contact_at_close``), otherwise ``not_recorded``.

Resolution never falls back: an unknown id, another scene, or an object the entry is not declared for is refused with a
structured reason (``reference_not_found`` / ``reference_incompatible``).  A legacy inline ``grasp_reference`` keeps working
unchanged (same numbers, same program hash); ``match_inline`` only attaches the id it is registered under, for provenance.

This module imports nothing from the program / executor modules (duck-typed on ``env_task`` / ``objects`` / ``instances``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

LIBRARY_VERSION = "grasp_references.v1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "configs" / "grasp_references.v1.json"

FRAME_DEFINITION = {
    "rel_obj": "hand (end-effector body) position minus object origin, in the object's yaw-only frame Rz(object_yaw) "
               "(world z up), metres",
    "hand_yaw_rel_deg": "yaw about world z of the demonstration hand relative to the executor's yaw-aligned grasp quaternion "
                        "q0 = Rz(wrap_pi/4(object_yaw)) * q_down (q_down = the reset-pose end-effector quaternion)",
    "hand_quat_obj_xyzw": "full hand orientation in the object's yaw-only frame: Rz(object_yaw)^-1 * R_hand, xyzw, w >= 0 "
                          "(absent: the executor uses a vertical hand, yaw only)",
}


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(x: Any) -> str:
    return hashlib.sha256(_canon(x).encode()).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def entry_digest(entry: Mapping[str, Any]) -> str:
    """Digest pinning an entry's content (everything except the digest itself)."""
    return _sha({k: v for k, v in entry.items() if k != "content_sha256"})


def tilt_deg(hand_quat_obj_xyzw) -> Optional[float]:
    """Angle between the hand's approach axis and straight down (the yaw-only object frame keeps tilt unchanged)."""
    if hand_quat_obj_xyzw is None:
        return None
    x, y, z, w = (float(v) for v in hand_quat_obj_xyzw)
    zz = 1.0 - 2.0 * (x * x + y * y)      # z component of the hand z axis (R[2, 2] of the unit quaternion)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, -zz)))))


@dataclass(frozen=True)
class GraspReference:
    reference_id: str
    scene: str
    object_name: str
    object_class: str
    compatible_objects: tuple
    compatibility_basis: str
    kind: str                       # "single_demo_close" | "median_over_demos"
    value: Mapping[str, Any]        # exactly what the executor consumes as ``grasp_reference``
    frame: Mapping[str, str]
    source: Mapping[str, Any]
    demo_contact_at_close: Mapping[str, Any]
    content_sha256: str

    def value_copy(self) -> dict:
        return copy.deepcopy(dict(self.value))

    def summary(self) -> dict:
        """What a proposer is shown to choose from (no free numbers to edit)."""
        q = self.value.get("hand_quat_obj_xyzw")
        return {"reference_id": self.reference_id, "kind": self.kind, "object": self.object_name,
                "executed_as": "tilted_hand_full_pose" if q is not None else "vertical_hand_yaw_only",
                "tilt_deg": None if q is None else round(tilt_deg(q), 1),
                "capture": {k: self.source.get(k) for k in ("demo", "close_step", "n_demos") if k in self.source},
                "demo_contact_at_close": self.demo_contact_at_close.get("status"),
                "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class Resolution:
    ok: bool
    reference: Optional[GraspReference]
    refusal: Optional[str]          # None | "reference_not_found:<id>" | "reference_incompatible:<id>:scene|object:..."

    @property
    def value(self) -> Optional[dict]:
        return None if self.reference is None else self.reference.value_copy()


class LibraryError(ValueError):
    """The library file itself is unusable (never a silent partial load)."""


class ReferenceLibrary:
    def __init__(self, entries: dict[str, GraspReference], library_sha256: str, path: Optional[Path] = None):
        self.entries, self.library_sha256, self.path = entries, library_sha256, path

    def ids(self) -> list[str]:
        return sorted(self.entries)

    def resolve(self, reference_id: str, scene: str, backend_object: str) -> Resolution:
        e = self.entries.get(reference_id)
        if e is None:
            return Resolution(False, None, f"reference_not_found:{reference_id}")
        if e.scene != scene:
            return Resolution(False, None, f"reference_incompatible:{reference_id}:scene:{scene}!={e.scene}")
        if backend_object not in e.compatible_objects:
            return Resolution(False, None, f"reference_incompatible:{reference_id}:object:{backend_object}"
                                           f"_not_in_{list(e.compatible_objects)}")
        return Resolution(True, e, None)

    def available(self, scene: str, backend_object: str) -> list[GraspReference]:
        return [e for _, e in sorted(self.entries.items()) if e.scene == scene and backend_object in e.compatible_objects]

    def match_inline(self, scene: str, backend_object: str, value: Mapping[str, Any]) -> Optional[GraspReference]:
        """The registered entry whose numbers equal an inline reference exactly (provenance for legacy configs)."""
        for e in self.available(scene, backend_object):
            if _canon(dict(e.value)) == _canon(dict(value)):
                return e
        return None


def _from_dict(d: Mapping[str, Any]) -> GraspReference:
    need = ("reference_id", "scene", "object_name", "object_class", "compatible_objects", "compatibility_basis", "kind",
            "value", "frame", "source", "demo_contact_at_close", "content_sha256")
    miss = [k for k in need if k not in d]
    if miss:
        raise LibraryError(f"entry {d.get('reference_id')!r} missing {miss}")
    if entry_digest(d) != d["content_sha256"]:
        raise LibraryError(f"entry {d['reference_id']!r}: content_sha256 does not match its content")
    if d["object_name"] not in d["compatible_objects"]:
        raise LibraryError(f"entry {d['reference_id']!r}: the source object is not in compatible_objects")
    return GraspReference(d["reference_id"], d["scene"], d["object_name"], d["object_class"], tuple(d["compatible_objects"]),
                          d["compatibility_basis"], d["kind"], copy.deepcopy(d["value"]), copy.deepcopy(d["frame"]),
                          copy.deepcopy(d["source"]), copy.deepcopy(d["demo_contact_at_close"]), d["content_sha256"])


def load_library(path: Optional[Path] = None) -> ReferenceLibrary:
    return _load(str(path or DEFAULT_PATH))


@lru_cache(maxsize=4)
def _load(path: str) -> ReferenceLibrary:
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    if d.get("version") != LIBRARY_VERSION:
        raise LibraryError(f"unsupported library version {d.get('version')!r}")
    entries: dict[str, GraspReference] = {}
    for e in d.get("entries", ()):
        g = _from_dict(e)
        if g.reference_id in entries:
            raise LibraryError(f"duplicate reference_id {g.reference_id!r}")
        entries[g.reference_id] = g
    return ReferenceLibrary(entries, hashlib.sha256(raw).hexdigest(), Path(path))


def _instance_object(program, instance) -> Optional[str]:
    return program.objects.get(instance.object_role)


def reference_issues(program, library: Optional[ReferenceLibrary] = None) -> list[str]:
    """Errors for every ``grasp_reference_id`` of the program (empty when none is used).

    ``reference_conflict:<iid>``      both ``grasp_reference`` and ``grasp_reference_id`` are set (no silent precedence);
    ``reference_not_found:<iid>:...`` / ``reference_incompatible:<iid>:...`` from the resolution."""
    errs: list[str] = []
    ids = [it for it in program.instances if "grasp_reference_id" in it.params]
    if not ids:
        return errs
    lib = library or load_library()
    for it in ids:
        rid = it.params["grasp_reference_id"]
        if "grasp_reference" in it.params:
            errs.append(f"reference_conflict:{it.instance_id}")
            continue
        obj = _instance_object(program, it)
        if obj is None:
            errs.append(f"reference_incompatible:{it.instance_id}:{rid}:object_role_unbound")
            continue
        r = lib.resolve(rid, program.env_task, obj)
        if not r.ok:
            errs.append(f"{r.refusal.split(':', 1)[0]}:{it.instance_id}:{r.refusal.split(':', 1)[1]}")
    return errs


def instance_reference(program, instance, library: Optional[ReferenceLibrary] = None):
    """(value, provenance, refusal) of the grasp reference an approach instance carries.

    (None, None, None)       the instance carries no reference;
    (value, provenance, None) resolved; ``provenance`` is what is recorded with the run;
    (None, None, refusal)     unresolvable (never a fallback)."""
    lib = library or load_library()
    p = instance.params
    if "grasp_reference_id" in p:
        if "grasp_reference" in p:
            return None, None, f"reference_conflict:{instance.instance_id}"
        obj = _instance_object(program, instance)
        r = lib.resolve(p["grasp_reference_id"], program.env_task, obj) if obj else \
            Resolution(False, None, f"reference_incompatible:{p['grasp_reference_id']}:object_role_unbound")
        if not r.ok:
            return None, None, r.refusal
        return r.value, {"reference_id": r.reference.reference_id, "origin": "grasp_reference_id",
                         "content_sha256": r.reference.content_sha256, "library_sha256": lib.library_sha256}, None
    if "grasp_reference" in p:
        obj = _instance_object(program, instance)
        m = lib.match_inline(program.env_task, obj, p["grasp_reference"]) if obj else None
        return dict(p["grasp_reference"]), {
            "reference_id": None if m is None else m.reference_id, "origin": "inline_grasp_reference",
            "registered": m is not None, "content_sha256": None if m is None else m.content_sha256,
            "library_sha256": lib.library_sha256}, None
    return None, None, None


def verify_library(library: Optional[ReferenceLibrary] = None, *, repo_root: Path = ROOT,
                   dataset_dir: Optional[Path] = None) -> list[dict]:
    """Re-check the recorded provenance: derivation record / script hashes in the repository and, when the dataset directory is
    given and the file exists, the demonstration file hash.  Per check: ``verified`` | ``mismatch`` | ``missing`` | ``skipped``."""
    lib = library or load_library()
    out = []
    for rid in lib.ids():
        src = lib.entries[rid].source
        for key in ("derivation_records", "extraction_scripts"):
            refs = src.get(key) or []
            if not refs:
                out.append({"reference_id": rid, "check": key, "status": "missing"})
            for ref in refs:
                f = repo_root / ref["path"]
                st = "missing" if not f.exists() else ("verified" if file_sha256(f) == ref["sha256"] else "mismatch")
                out.append({"reference_id": rid, "check": key, "path": ref["path"], "status": st})
        ds = src.get("dataset_file") or {}
        if dataset_dir is None:
            out.append({"reference_id": rid, "check": "dataset_file", "status": "skipped"})
        else:
            f = Path(dataset_dir) / ds.get("name", "")
            st = "missing" if not f.exists() else ("verified" if file_sha256(f) == ds.get("sha256") else "mismatch")
            out.append({"reference_id": rid, "check": "dataset_file", "name": ds.get("name"), "status": st})
    return out
