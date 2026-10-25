"""Shared repair rule v0 (bounded, one mechanism): raise the carry height of a transport instance.

Mechanism with evidence (T0 soup init8, T7 soup init13; development period): a carried object touches the destination container /
a neighbour during the transport stage and the grasp then becomes one-sided.  The rule is task-id / seed free: it only sees the
program (stage instances, roles, parameters), the per-control-step stage labels, per-step left/right finger force of the grasp
target, and the carried-object contact events of the run.

  trigger     for one manipulated object: (a) a contact with a non-robot, non-support body above CONTACT_THR_N during a lift/transport
              stage of that object, AND (b) a one-sided grasp step (min(L, R) <= THR) of the same object inside
              [first contact step, last contact step + LAG] of the hold window.  Both are required (contact without grasp
              instability, e.g. tomato sauce touching the basket, does not fire).
  candidate   the object's transport instance gets transport_dz + DZ_STEP, only if (i) the contact occurred in that transport stage
              (the lift height is not a per-instance parameter) and (ii) new dz <= DZ_MAX.
  budget      one repair per request; a second trigger in the same request is reported, not repaired.
  recovery    resume at the START of the modified transport instance (prefix unchanged), re-verify the whole request.

DZ_MAX is the largest carry height exercised in simulation so far (0.175 + 0.03), not a measured workspace limit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from synthesis_v2.contact_audit import ContactRecorder, SUPPORT_PREFIXES, _is_robot

RULE_VERSION = "transport_repair_v0"
CONTACT_THR_N = 0.1
ONE_SIDED_THR_N = 0.1
LAG_STEPS = 5          # one-sided step may follow the last contact step by this many control steps (frozen before the fixed list)
TRIM = 2               # hold_trim_head_steps (same as the calibration config)
DZ_STEP = 0.03
DZ_MAX = 0.205
DEFAULT_TRANSPORT_DZ = 0.15   # program_executor.LIFT_DZ, used when the instance has no transport_dz


class CarriedContactRecorder(ContactRecorder):
    """Rollback-isolated events (object name, other body) -> [max N, {control steps}] for contacts of the named manipulated objects
    with any non-robot, non-support, non-self body (installed after reset, same restore/seed behaviour as ContactRecorder)."""

    def __init__(self, session, object_names: Sequence[str]) -> None:
        super().__init__(session)
        self.names = tuple(object_names)
        self._owner: Optional[dict] = None

    def _observe(self) -> None:
        import mujoco
        if self._owner is None:
            self._owner = {g: n for n in self.names for g in self.s.geom_ids_of(n)}
        m, d, buf = self.m, self.d, np.zeros(6)
        body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
        for ci in range(int(d.ncon)):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                n = self._owner.get(a)
                if n is None or self._owner.get(b) == n:
                    continue
                ob = body(b)
                if _is_robot(ob) or ob.startswith(SUPPORT_PREFIXES):
                    continue
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if f >= CONTACT_THR_N:          # only contact steps at/above the threshold are events (so every recorded step counts)
                    v = self.events.setdefault((n, ob), [0.0, set()])
                    v[0] = max(v[0], f); v[1].add(self.k)


@dataclass(frozen=True)
class Trigger:
    object_role: str
    contact_body: str
    contact_stage: str
    contact_first_step: int
    contact_last_step: int
    contact_peak_n_pair_all_stages: float   # max single-contact normal force of this (object, body) pair over the whole run
    contact_steps_in_transport: int
    first_non_bilateral_step: int     # first hold step with one_sided or no_contact inside the contact window
    non_bilateral_steps: int


@dataclass(frozen=True)
class Proposal:
    rule: str
    trigger: Trigger
    instance_label: str
    old_dz: float
    new_dz: float
    failure_step: int


def _by_label(program) -> dict:
    return {it.label: it for it in program.instances}


def hold_steps(program, stages: Sequence[str], role: str) -> np.ndarray:
    st = np.asarray([str(x) for x in stages])
    lift = [it.label for it in program.instances if it.primitive == "lift" and it.object_role == role]
    tr = [it.label for it in program.instances if it.primitive == "transport" and it.object_role == role]
    a = np.flatnonzero(np.isin(st, lift))[TRIM:]
    return np.concatenate([a, np.flatnonzero(np.isin(st, tr))]).astype(int)


GRASP_STATES = ("bilateral", "one_sided", "no_contact", "invalid")
CHECKS_VERSION = "hold_checks_v1"


def grasp_state(left, right, valid=None, thr: float = ONE_SIDED_THR_N) -> np.ndarray:
    """Per-step grasp state of the target: invalid (non-finite force or valid_fraction < 1) | bilateral (both > thr) |
    one_sided (exactly one side > thr) | no_contact (neither side > thr).  Invalid is decided first, so NaN is never read
    as 'not abnormal'."""
    L, R = np.asarray(left, float), np.asarray(right, float)
    ok = np.isfinite(L) & np.isfinite(R)
    if valid is not None:
        v = np.asarray(valid, float)
        ok &= np.isfinite(v) & (v >= 1.0 - 1e-9)
    with np.errstate(invalid="ignore"):
        lo, ro = L > thr, R > thr
    out = np.where(lo & ro, "bilateral", np.where(lo | ro, "one_sided", "no_contact")).astype(object)
    out[~ok] = "invalid"
    return out


def _hold_state(program, stages, left, right, valid, role):
    h = hold_steps(program, stages, role)
    v = None if valid is None else np.asarray(valid, float)[h]
    return h, grasp_state(np.asarray(left, float)[h], np.asarray(right, float)[h], v)


def hold_state_counts(program, stages, left, right, role, valid=None) -> dict:
    h, st = _hold_state(program, stages, left, right, valid, role)
    return {"n_hold": int(len(h)), **{k: int((st == k).sum()) for k in GRASP_STATES}}


def non_bilateral_steps(program, stages, left, right, role, valid=None) -> np.ndarray:
    """Hold steps with a finite, valid measurement in which not both fingers exceed the threshold (one_sided or no_contact).
    Same steps as the v0 'min(L, R) <= thr' for finite data; invalid steps are NOT in here, they are counted separately."""
    h, st = _hold_state(program, stages, left, right, valid, role)
    return h[(st == "one_sided") | (st == "no_contact")]


def diagnose(program, stages: Sequence[str], left, right, events: Mapping[tuple, Sequence], valid=None) -> list[Trigger]:
    """All triggers of one run, earliest first.  ``events``: (object name, other body) -> [peak N, {control steps}]."""
    lab = _by_label(program)
    name2role = {n: r for r, n in program.objects.items()}
    carried = {r: [it.label for it in program.instances if it.object_role == r and it.primitive in ("lift", "transport")] for r in program.objects}
    out: list[Trigger] = []
    for role in program.objects:
        bad = non_bilateral_steps(program, stages, left, right, role, valid)
        if not len(bad):
            continue
        for (obj, other), (peak, ks) in sorted(events.items()):
            if name2role.get(obj) != role or peak < CONTACT_THR_N:
                continue
            ks_c = sorted(k for k in ks if 0 <= k < len(stages) and str(stages[k]) in carried[role])
            if not ks_c:
                continue
            lo, hi = ks_c[0], ks_c[-1]
            hit = bad[(bad >= lo) & (bad <= hi + LAG_STEPS)]
            if not len(hit):
                continue
            tr_steps = [k for k in ks_c if lab[str(stages[k])].primitive == "transport"]
            out.append(Trigger(role, other, str(stages[lo]), int(lo), int(hi), float(peak), len(tr_steps), int(hit[0]), int(len(hit))))
    return sorted(out, key=lambda t: t.first_non_bilateral_step)


def propose(program, triggers: Sequence[Trigger]) -> tuple[Optional[Proposal], dict]:
    """At most one proposal per request.  Returns (proposal | None, decision record incl. reasons for every trigger)."""
    lab = _by_label(program)
    rec: dict[str, Any] = {"rule": RULE_VERSION, "n_triggers": len(triggers), "per_trigger": []}
    chosen: Optional[Proposal] = None
    for i, t in enumerate(triggers):
        tr = [it for it in program.instances if it.primitive == "transport" and it.object_role == t.object_role]
        why = None
        if t.contact_steps_in_transport == 0 or len(tr) != 1:
            why = "contact_not_in_a_transport_stage" if t.contact_steps_in_transport == 0 else "no_unique_transport_instance"
        else:
            old = float(tr[0].params.get("transport_dz", DEFAULT_TRANSPORT_DZ))
            new = round(old + DZ_STEP, 6)
            if new > DZ_MAX + 1e-9:
                why = f"infeasible_new_dz_{new}_gt_max_{DZ_MAX}"
        d = {"trigger": t.__dict__, "candidate": None if why else {"instance": tr[0].label, "old_dz": old, "new_dz": new}, "not_proposed_because": why}
        if why is None and chosen is None:
            chosen = Proposal(RULE_VERSION, t, tr[0].label, old, new, t.first_non_bilateral_step)
            d["selected"] = True
        elif why is None:
            d["not_proposed_because"] = "repair_budget_one_per_request"
        rec["per_trigger"].append(d)
    return chosen, rec


def apply(program, prop: Proposal):
    hit, inst = 0, []
    for it in program.instances:
        if it.label == prop.instance_label:
            it, hit = replace(it, params={**it.params, "transport_dz": prop.new_dz}), hit + 1
        inst.append(it)
    assert hit == 1
    return replace(program, instances=tuple(inst))


def request_checks(program, stages: Sequence[str], left, right, events, status: str, task_ok: bool, valid=None) -> dict:
    """LOCAL checks only: all stage instances executed, task end state, per-object hold-window grasp state (four classes,
    invalid => unknown, an empty hold window of a lifted object => unknown), no residual transport-repair trigger.
    NOT checked here: force band, speed limits, contact policy; the formal request verdict comes from the full audit."""
    seen = set(str(x) for x in stages)
    counts = {r: hold_state_counts(program, stages, left, right, r, valid) for r in program.objects}
    lifted = {it.object_role for it in program.instances if it.primitive == "lift"}
    unknown = [r for r, c in counts.items() if c["invalid"] > 0 or (r in lifted and c["n_hold"] == 0)]
    non_bil = {r: c["one_sided"] + c["no_contact"] for r, c in counts.items()}
    resid = diagnose(program, stages, left, right, events, valid)
    executed = all(it.label in seen for it in program.instances)
    return {"checks_version": CHECKS_VERSION, "status": status, "task_ok": bool(task_ok), "all_stage_instances_executed": executed,
            "hold_state_counts_by_object": counts, "non_bilateral_steps_by_object": non_bil,
            "unknown_objects": unknown, "residual_triggers": [t.__dict__ for t in resid],
            "task_and_hold_checks_passed": bool(status == "completed" and task_ok and executed and not unknown and not resid
                                                and all(v == 0 for v in non_bil.values())),
            "scope": "task end state + hold-window bilateral contact + no residual transport-repair trigger; not force band / speed / contact policy"}
