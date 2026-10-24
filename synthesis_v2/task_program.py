"""Typed task programs: object-scoped, repeatable stage instances (design v3 §2.6 / §8.1).

A *stage type* (``grasp``) is what a primitive does; a *stage instance*
(``cup_A.pick_1.grasp``) is one concrete execution of it on one object.  Manner
requirements, plans, measurement windows and resume points bind to instances,
so the same stage type may appear several times and each occurrence can carry
its own requirement.

Legacy single-object tasks (lift / stack / nut_assembly / wipe) are expressed as
programs whose instance ``label`` equals the legacy stage id, so the recorded
``stage`` column and every downstream consumer stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional

TASK_PROGRAM_VERSION = "task_program.v1"

# primitive -> (default stage type label, manner dimensions it supports, contact kind)
PRIMITIVES: dict[str, dict[str, Any]] = {
    "approach": {"dims": ("speed",), "contact": None, "needs": ("object_or_target",)},
    "descend": {"dims": ("speed",), "contact": None, "needs": ("object_or_target",)},
    "grasp": {"dims": ("grip",), "contact": "finger_object", "needs": ("object",)},
    "lift": {"dims": ("speed",), "contact": None, "needs": ("object",)},
    "transport": {"dims": ("speed",), "contact": None, "needs": ("object", "target")},
    "place": {"dims": ("speed",), "contact": "object_support", "needs": ("object", "target")},
    "insert": {"dims": ("speed",), "contact": "object_peg", "needs": ("object", "target")},
    "retreat": {"dims": ("speed",), "contact": None, "needs": ()},
    "sweep": {"dims": ("speed", "press"), "contact": "tool_surface", "needs": ("target",)},
    "articulate": {"dims": ("speed",), "contact": "hand_articulated", "needs": ("target",)},
}

# Stage *types* that hold the gripper force set at the latest grasp of the same object
# (the grasp parameter's scope).  Used for parameter-dependency / resume computation.
GRIP_HOLD_TYPES = ("lift", "transport", "place", "insert")

# Natural-language stage words -> candidate primitives.  Unlike the legacy global ALIASES
# ("close" -> grasp), resolution depends on which primitives the program actually has and
# on the bound object: "close" means the gripper only if no articulated close exists.
STAGE_WORDS: dict[str, tuple[str, ...]] = {
    "reach": ("approach",), "approach": ("approach",),
    "descend": ("descend",), "lower": ("descend", "place"),
    "pick": ("grasp",), "grasp": ("grasp",), "grip": ("grasp",), "hold": ("grasp",),
    "lift": ("lift",), "raise": ("lift",),
    "carry": ("transport",), "transport": ("transport",), "move": ("transport",),
    "place": ("place", "insert"), "put": ("place", "insert"), "release": ("place",), "final": ("place",),
    "insert": ("insert",),
    "leave": ("retreat",), "retreat": ("retreat",),
    "wipe": ("sweep",), "sweep": ("sweep",),
    "close": ("articulate", "grasp"), "open": ("articulate",), "push": ("articulate",),
}

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*$")


@dataclass(frozen=True)
class StageInstance:
    instance_id: str             # e.g. "cup_A.pick_1.grasp"; legacy programs: "cube.pick_1.grasp"
    primitive: str               # key of PRIMITIVES
    label: str                   # value written into the per-step ``stage`` column
    object_role: Optional[str] = None
    target_role: Optional[str] = None
    subtask: str = "main"        # e.g. "cup_A.pick_1"
    occurrence: int = 1
    preds: tuple[str, ...] = ()  # instance ids that must complete first (ordered stage graph)
    params: Mapping[str, Any] = field(default_factory=dict)  # primitive geometry options (offsets...)

    @property
    def dims(self) -> tuple[str, ...]:
        return PRIMITIVES[self.primitive]["dims"]

    def to_dict(self) -> dict[str, Any]:
        return {"instance_id": self.instance_id, "primitive": self.primitive, "label": self.label,
                "object_role": self.object_role, "target_role": self.target_role, "subtask": self.subtask,
                "occurrence": self.occurrence, "preds": list(self.preds), "params": dict(self.params)}


@dataclass(frozen=True)
class TaskProgram:
    name: str
    backend: str                         # "robosuite_native" | "libero"
    env_task: str                        # environment-level task id
    objects: Mapping[str, str]           # role -> backend object name
    instances: tuple[StageInstance, ...]
    goal: tuple[str, ...] = ()           # symbolic goal relations, e.g. "On(cup_A, plate_left)"
    source: str = "hand_written"         # hand_written | llm | demo
    language: str = ""

    # ---- lookup ---------------------------------------------------------
    def instance(self, instance_id: str) -> StageInstance:
        for it in self.instances:
            if it.instance_id == instance_id:
                return it
        raise KeyError(instance_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(it.instance_id for it in self.instances)

    def index(self, instance_id: str) -> int:
        return self.ids().index(instance_id)

    def of_primitive(self, primitive: str, object_role: Optional[str] = None) -> tuple[StageInstance, ...]:
        return tuple(it for it in self.instances if it.primitive == primitive
                     and (object_role is None or it.object_role == object_role))

    def grip_scope(self, grasp_id: str) -> tuple[str, ...]:
        """Instances during which the grip parameter of ``grasp_id`` is in effect (grasp .. release)."""
        g = self.instance(grasp_id)
        if g.primitive != "grasp":
            raise ValueError(grasp_id)
        out = [grasp_id]
        for it in self.instances[self.index(grasp_id) + 1:]:
            if it.object_role != g.object_role or it.primitive not in GRIP_HOLD_TYPES:
                break
            out.append(it.instance_id)
            if it.primitive in ("place", "insert"):
                break
        return tuple(out)

    # ---- validation -----------------------------------------------------
    def validate(self) -> list[str]:
        """Structural problems (empty list == structurally valid).  Not a feasibility proof."""
        errs: list[str] = []
        seen_labels: dict[str, str] = {}
        for it in self.instances:  # the recorded ``stage`` column is the label, so it must identify one instance
            if len(it.label) > 48:
                errs.append(f"label_too_long:{it.instance_id}:{len(it.label)}")
            if it.label in seen_labels:
                errs.append(f"duplicate_label:{seen_labels[it.label]}:{it.instance_id}:{it.label}")
            seen_labels[it.label] = it.instance_id
        seen: set[str] = set()
        for k, it in enumerate(self.instances):
            if not _ID_RE.match(it.instance_id):
                errs.append(f"bad_instance_id:{it.instance_id}")
            if it.instance_id in seen:
                errs.append(f"duplicate_instance:{it.instance_id}")
            if it.primitive not in PRIMITIVES:
                errs.append(f"unknown_primitive:{it.instance_id}:{it.primitive}")
                continue
            need = PRIMITIVES[it.primitive]["needs"]
            if "object" in need and it.object_role not in self.objects:
                errs.append(f"unbound_object:{it.instance_id}:{it.object_role}")
            if "object_or_target" in need and it.object_role not in self.objects and it.target_role not in self.objects:
                errs.append(f"unbound_object:{it.instance_id}:{it.object_role}")
            if "target" in need and it.target_role not in self.objects:
                errs.append(f"unbound_target:{it.instance_id}:{it.target_role}")
            for p in it.preds:
                if p not in seen:
                    errs.append(f"pred_not_before:{it.instance_id}<-{p}")
            seen.add(it.instance_id)
        # an object must be grasped before it is lifted / carried / placed
        held: Optional[str] = None
        for it in self.instances:
            if it.primitive == "grasp":
                if held is not None:
                    errs.append(f"grasp_while_holding:{it.instance_id}:{held}")
                held = it.object_role
            elif it.primitive in GRIP_HOLD_TYPES:
                if held != it.object_role:
                    errs.append(f"not_holding:{it.instance_id}:{it.object_role}")
                if it.primitive in ("place", "insert"):
                    held = None
        labels = [it.label for it in self.instances]
        if self.backend == "robosuite_native" and len(set(labels)) != len(labels) and not self.multi_label_ok:
            errs.append("duplicate_labels_require_instance_column")
        return errs

    @property
    def multi_label_ok(self) -> bool:
        return self.source != "legacy"

    # ---- identity -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": TASK_PROGRAM_VERSION, "name": self.name, "backend": self.backend,
                "env_task": self.env_task, "objects": dict(self.objects),
                "instances": [it.to_dict() for it in self.instances], "goal": list(self.goal),
                "source": self.source, "language": self.language}

    def program_hash(self) -> str:
        blob = json.dumps({k: v for k, v in self.to_dict().items() if k != "language"},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TaskProgram":
        insts = tuple(StageInstance(
            instance_id=str(x["instance_id"]), primitive=str(x["primitive"]),
            label=str(x.get("label") or x["primitive"]), object_role=x.get("object_role"),
            target_role=x.get("target_role"), subtask=str(x.get("subtask", "main")),
            occurrence=int(x.get("occurrence", 1)), preds=tuple(x.get("preds", ())),
            params=dict(x.get("params", {}))) for x in d["instances"])
        return TaskProgram(name=str(d["name"]), backend=str(d["backend"]), env_task=str(d["env_task"]),
                           objects=dict(d["objects"]), instances=insts, goal=tuple(d.get("goal", ())),
                           source=str(d.get("source", "hand_written")), language=str(d.get("language", "")))


# ---- builders -------------------------------------------------------------
def pick_place_instances(obj: str, target: Optional[str], k: int = 1, *, with_retreat: bool = True,
                         place_primitive: str = "place", lift_only: bool = False,
                         label_legacy: bool = False) -> list[StageInstance]:
    """Shared template: approach, descend, grasp, lift[, transport, place][, retreat] for one object."""
    sub = f"{obj}.pick_{k}"
    seq = ["approach", "descend", "grasp", "lift"]
    if not lift_only:
        seq += ["transport", place_primitive]
    if with_retreat and not lift_only:
        seq.append("retreat")
    out: list[StageInstance] = []
    for prim in seq:
        legacy_label = "place" if prim == "insert" else prim
        iid = f"{sub}.{legacy_label}"
        out.append(StageInstance(
            instance_id=iid, primitive=prim, label=legacy_label if label_legacy else iid,
            object_role=obj if prim != "retreat" else None,
            target_role=target if prim in ("transport", "place", "insert") else None,
            subtask=sub, occurrence=k, preds=(out[-1].instance_id,) if out else ()))
    return out


def chain(*groups: Iterable[StageInstance]) -> tuple[StageInstance, ...]:
    """Concatenate per-subtask instance groups, linking each group to the end of the previous one.

    Lets a multi-object task be declared as a sequence of shared ``pick_place_instances`` calls
    instead of a new execution branch (design v3 §2.5).
    """
    out: list[StageInstance] = []
    for g in groups:
        g = list(g)
        if out and g:
            g[0] = replace(g[0], preds=(out[-1].instance_id,))
        out.extend(g)
    return tuple(out)


def legacy_program(task: str) -> TaskProgram:
    """Programs equivalent to the pre-v3 fixed stage tables (labels == legacy stage ids)."""
    if task == "lift":
        inst = pick_place_instances("cube", None, lift_only=True, label_legacy=True)
        return TaskProgram("lift", "robosuite_native", "lift", {"cube": "cube"}, tuple(inst),
                           goal=("Lifted(cube)",), source="legacy")
    if task == "stack":
        inst = pick_place_instances("cubeA", "cubeB", label_legacy=True)
        return TaskProgram("stack", "robosuite_native", "stack", {"cubeA": "cubeA", "cubeB": "cubeB"},
                           tuple(inst), goal=("On(cubeA, cubeB)",), source="legacy")
    if task == "nut_assembly":
        inst = pick_place_instances("nut", "peg", place_primitive="insert", label_legacy=True)
        return TaskProgram("nut_assembly", "robosuite_native", "nut_assembly",
                           {"nut": "SquareNut", "peg": "peg1"}, tuple(inst), goal=("On(nut, peg)",),
                           source="legacy")
    if task == "wipe":
        mk = lambda iid, prim, tgt: StageInstance(f"table.wipe_1.{iid}", prim, iid, None, tgt, "table.wipe_1")
        inst = [mk("approach", "approach", None), mk("descend", "descend", None),
                mk("sweep", "sweep", "table"), mk("retreat", "retreat", None)]
        # approach/descend of the tool are object-free free-space moves toward the dirt path
        inst = [StageInstance(i.instance_id, i.primitive, i.label, None, "table" if i.primitive != "retreat" else None,
                              i.subtask, 1, (inst[k - 1].instance_id,) if k else ()) for k, i in enumerate(inst)]
        return TaskProgram("wipe", "robosuite_native", "wipe", {"table": "table"}, tuple(inst),
                           goal=("Clean(table)",), source="legacy")
    raise ValueError(f"no legacy program for {task!r}")


# ---- language-side references ----------------------------------------------
def resolve_stage_ref(program: TaskProgram, word: str, object_role: Optional[str] = None,
                      occurrence: Optional[int] = None) -> tuple[tuple[str, ...], str]:
    """Map a stage word (+ optional object / occurrence) to concrete instance ids.

    Returns ``(instance_ids, status)`` with status ``ok`` | ``ambiguous`` | ``unknown``.  Never guesses:
    "close" binds to an articulated close if the program has one, else to the grasp of the bound object;
    several matches without an object / occurrence qualifier are reported as ambiguous.
    """
    key = str(word).strip().lower()
    if key in program.ids():
        return (key,), "ok"
    prims = STAGE_WORDS.get(key)
    if prims is None:
        return (), "unknown"
    for prim in prims:  # first primitive family present in the program wins (ordered by specificity)
        cand = [it for it in program.instances if it.primitive == prim
                and (object_role is None or it.object_role == object_role or it.target_role == object_role)
                and (occurrence is None or it.occurrence == occurrence)]
        if cand:
            if len(cand) > 1 and object_role is None and occurrence is None:
                return tuple(c.instance_id for c in cand), "ambiguous"
            return tuple(c.instance_id for c in cand), "ok"
    return (), "unknown"


def dimension_instances(program: TaskProgram, dimension: str) -> tuple[str, ...]:
    return tuple(it.instance_id for it in program.instances if dimension in it.dims)


def labels(program: TaskProgram) -> tuple[str, ...]:
    return tuple(it.label for it in program.instances)


def iter_programs(tasks: Iterable[str]) -> Iterable[TaskProgram]:
    for t in tasks:
        yield legacy_program(t)
