"""Step 24 (item 6): a structured, evidence-bearing failure report built ONLY from records the executor / audit already
produce.  It is the feedback a repair proposer sees; it is not a judge.

Design rules
* three failure classes are never merged: ``planning_rejected`` (no motion was executed), ``execution_failed`` (a stage did
  not do what the executor needs), ``audit_failed`` (the contact audit objects).  ``outcome_not_met`` lists measured
  per-object results (grasp/lift/transport/place) and the official task flag that did not hold.
* every failure names the instance/stage, the control step (if recorded), the requirement it violated and the record it came from.
* a quantity the existing records cannot supply is ``{"unknown": "<reason>"}`` -- never 0, never an estimate.  ``unknown_fields``
  lists every such path so a consumer can see the gaps at a glance.
* ``first_failure`` is the earliest by (stage index, control step): temporal order, NOT a causal claim.
* requirement ids are report labels (``exec.*``, ``contact_policy_v1.1.*``, ``outcome.*``, ``task.*``); they are not thresholds
  and not the ids of ``configs/constraint_registry.yaml``.  Nothing here can be edited by a proposer.
* ``resume_points`` only lists snapshots that exist.  Existing does not mean a given patch may resume from it; ``validity`` is
  ``unchecked`` until the dependency check of item 7.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

SCHEMA = "mannercap.failure_report.v1"
CLASSES = ("planning_rejected", "execution_failed", "audit_failed", "outcome_not_met")
_CLASS_RANK = {c: i for i, c in enumerate(CLASSES)}

REQUIREMENTS: dict[str, str] = {
    "exec.approach.feasible": "the planner found a candidate hand path with clearance > 0 (incl. the open hand at the grasp pose)",
    "exec.reference.resolved": "the grasp reference resolved against the registered library for this scene and object",
    "exec.stage_reached.<primitive>": "a required servo stage reached its goal within its step budget",
    "exec.grasp.force_band": "the grasp policy reached the force band",
    "exec.grasp.bilateral_contact": "both fingers carried the minimum contact force at the grasp",
    "exec.place.pose_reached": "the place pose was reached before release",
    "exec.guard.<reason>": "the execution guard (pre-grasp contact check) ended the episode",
    "exec.exception": "the executor raised",
    "contact_policy_v1.1.unexcused_contact": "a contact not excused by the frozen contact policy",
    "contact_policy_v1.1.support_contact_unexplained": "a support contact the policy could not explain",
    "outcome.grasp": "measured: the grasp was stable (force band, bilateral, in-hand spread)",
    "outcome.lift": "measured: the object rose at least the lift threshold",
    "outcome.transport": "measured: the object reached the planned target (xy) after transport",
    "outcome.place": "measured: the object rests in the target region after the retreat",
    "task.official_success": "the simulator's official task success flag",
}


def unknown(reason: str) -> dict:
    return {"unknown": reason}


def is_unknown(x: Any) -> bool:
    return isinstance(x, dict) and set(x) == {"unknown"}


def _find(program, instance_id: Optional[str] = None, label: Optional[str] = None,
          primitive: Optional[str] = None, role: Optional[str] = None):
    for i, it in enumerate(program.instances):
        if instance_id is not None and it.instance_id == instance_id:
            return i, it
        if label is not None and it.label == label:
            return i, it
        if instance_id is None and label is None and primitive == it.primitive and role == it.object_role:
            return i, it
    return None, None


def _loc(program, *, instance_id=None, label=None, primitive=None, role=None, stage_by_step=None, step=None):
    """instance id / stage index for a failure; unknown when the records do not say."""
    idx, it = _find(program, instance_id, label, primitive, role)
    if it is None and stage_by_step is not None and step is not None and 0 <= step < len(stage_by_step):
        idx, it = _find(program, label=str(stage_by_step[step]))
    return {"instance_id": it.instance_id if it else unknown("not recorded"),
            "stage_label": it.label if it else unknown("not recorded"),
            "primitive": it.primitive if it else unknown("not recorded"),
            "object_role": it.object_role if it else unknown("not recorded")}, (idx if idx is not None else None)


def _planner_evidence(meta: Mapping, subtask: Optional[str]) -> dict:
    ap = (meta or {}).get("approach_plan") or {}
    p = ap.get(subtask) if subtask else None
    if not p:
        return unknown("no approach_plan record for this subtask")
    cands = p.get("candidates") or []
    keep = ("yaw_deg", "lateral_m", "dz_m", "min_clearance_m", "blocking", "phase", "at_grasp_clearance_m", "feasible")
    return {"feasible": p.get("feasible"), "grid_source": p.get("grid_source"), "n_candidates": len(cands),
            "candidates": [{k: c.get(k) for k in keep} for c in cands[:8]],
            "reference": p.get("reference", unknown("no reference provenance recorded (record predates Step 24)")),
            "note": "clearance is signed metres between the hand and the named blocking body; > 0 is required"}


def _status_failures(program, status: str, meta: Mapping, task_ok, stage_by_step) -> list[dict]:
    out: list[dict] = []
    head, _, rest = (status or "").partition(":")
    meta = meta or {}
    if head == "completed":
        return out
    if head in ("approach_infeasible", "reference_unresolved"):
        iid = rest.split(":", 1)[0]
        loc, idx = _loc(program, instance_id=iid)
        it = _find(program, iid)[1]
        ev = _planner_evidence(meta, getattr(it, "subtask", None)) if head == "approach_infeasible" else {
            "resolution_error": rest.split(":", 1)[1] if ":" in rest else unknown("not recorded")}
        out.append({"class": "planning_rejected",
                    "requirement_id": "exec.approach.feasible" if head == "approach_infeasible" else "exec.reference.resolved",
                    **loc, "stage_index": idx, "control_step": 0,
                    "control_step_note": "rejected before any motion; no step was executed for this stage",
                    "pose_difference": unknown("planning rejection: no motion was executed, there is no reached pose"),
                    "evidence_source": "executor status + meta.approach_plan", "detail": ev})
        return out
    if head == "stage_timeout":
        for lab in [x for x in rest.split(",") if x]:
            loc, idx = _loc(program, label=lab)
            sf = (meta.get("stage_failure") or {}).get(loc["instance_id"]) if isinstance(loc["instance_id"], str) else None
            out.append({"class": "execution_failed", "requirement_id": f"exec.stage_reached.{loc['primitive'] if isinstance(loc['primitive'], str) else 'unknown'}",
                        **loc, "stage_index": idx,
                        "control_step": sf["control_step"] if sf else unknown("status names the stage only; meta.stage_failure absent"),
                        "pose_difference": (unknown("goal pose of the servo stage is not recorded; only the reached eef position is")
                                            if sf else unknown("neither goal nor reached pose recorded for this stage")),
                        "evidence_source": "executor status" + (" + meta.stage_failure" if sf else ""),
                        "detail": {"eef_pos_at_failure": sf["eef_pos"] if sf else unknown("not recorded"),
                                   "max_steps": sf["max_steps"] if sf else unknown("not recorded")}})
        return out
    iid = rest.split(":", 1)[0] if rest else None
    if head in ("grasp_band_not_reached", "grasp_not_bilateral"):
        loc, idx = _loc(program, instance_id=iid)
        gf = (meta.get("grasp_failure") or {}).get(iid) if iid else None
        pol = (meta.get("grasp_policy_by_instance") or {}).get(iid) if iid else None
        out.append({"class": "execution_failed",
                    "requirement_id": "exec.grasp.force_band" if head == "grasp_band_not_reached" else "exec.grasp.bilateral_contact",
                    **loc, "stage_index": idx,
                    "control_step": gf["control_step"] if gf else unknown("meta.grasp_failure absent"),
                    "pose_difference": unknown("a grasp has no goal pose; the hand pose at the grasp is not compared"),
                    "evidence_source": "executor status" + (" + meta.grasp_failure" if gf else "") + (" + meta.grasp_policy_by_instance" if pol else ""),
                    "detail": {"grasp_failure": gf if gf else unknown("not recorded"),
                               "grasp_policy": pol if pol else unknown("not recorded")}})
        return out
    if head == "place_not_reached":
        loc, idx = _loc(program, instance_id=iid)
        pf = (meta.get("place_failure") or {}).get(iid) if iid else None
        out.append({"class": "execution_failed", "requirement_id": "exec.place.pose_reached", **loc, "stage_index": idx,
                    "control_step": pf["control_step"] if pf else unknown("meta.place_failure absent"),
                    "pose_difference": ({"eef_error_m": pf["eef_error_m"], "tol_pos_m": pf["tol_pos_m"], "eef_pos": pf["eef_pos"],
                                         "eef_goal": pf["eef_goal"]} if pf else unknown("meta.place_failure absent")),
                    "evidence_source": "executor status" + (" + meta.place_failure" if pf else ""), "detail": {}})
        return out
    if head.startswith("guard_"):
        loc, idx = _loc(program, instance_id=iid)
        out.append({"class": "execution_failed", "requirement_id": f"exec.guard.{head[len('guard_'):]}", **loc, "stage_index": idx,
                    "control_step": unknown("not recorded in the status"),
                    "pose_difference": unknown("not applicable: the guard decision is about contact, not pose"),
                    "evidence_source": "executor status + meta.execution_guard",
                    "detail": {"execution_guard": (meta.get("execution_guard") or unknown("not recorded"))}})
        return out
    if head == "exception":
        out.append({"class": "execution_failed", "requirement_id": "exec.exception", "instance_id": unknown("not recorded"),
                    "stage_label": unknown("not recorded"), "primitive": unknown("not recorded"), "object_role": unknown("not recorded"),
                    "stage_index": None, "control_step": unknown("not recorded"),
                    "pose_difference": unknown("not recorded"), "evidence_source": "executor status", "detail": {"message": rest}})
        return out
    out.append({"class": "execution_failed", "requirement_id": "exec.unclassified_status", "instance_id": unknown("not recorded"),
                "stage_label": unknown("not recorded"), "primitive": unknown("not recorded"), "object_role": unknown("not recorded"),
                "stage_index": None, "control_step": unknown("not recorded"), "pose_difference": unknown("not recorded"),
                "evidence_source": "executor status", "detail": {"status": status}})
    return out


def _audit_failures(program, audit: Optional[Mapping], stage_by_step) -> list[dict]:
    if not audit:
        return []
    ver = audit.get("policy_version") or "v1.1"
    ver = str(ver).replace("contact_policy_", "").lstrip("_") or "v1.1"
    out: list[dict] = []
    for kind, key, req in (("unexcused", "unexcused", "unexcused_contact"), ("support", "support_unexplained", "support_contact_unexplained")):
        for row in audit.get(key) or audit.get(key + "_contacts") or []:
            rng = row.get("control_step_range")
            step = int(rng[0]) if rng else None
            prim_role = (row.get("stages") or [None])[0]
            prim, _, role = (prim_role or "").partition("@")
            loc, idx = _loc(program, stage_by_step=stage_by_step, step=step, primitive=prim or None, role=role or None)
            span = (int(rng[1]) - int(rng[0]) + 1) if rng else None
            out.append({"class": "audit_failed", "requirement_id": f"contact_policy_{ver}.{req}",
                        **loc, "stage_index": idx, "control_step": step if step is not None else unknown("not recorded"),
                        "pose_difference": unknown("a contact violation is not a pose error"),
                        "evidence_source": "contact audit (policy evaluate)",
                        "detail": {"robot_body": row.get("robot_body"), "other": row.get("other"), "max_n": row.get("max_n"),
                                   "control_step_range": rng if rng else unknown("not recorded"),
                                   "span_control_steps": span if span is not None else unknown("not recorded"),
                                   "n_substeps": row.get("n_substeps", unknown("not recorded")),
                                   "duration_s": unknown("only the control-step range / substep count is recorded; the contact need not be contiguous"),
                                   "reasons": row.get("v1_reasons", row.get("reasons", unknown("not recorded")))}})
    return out


def _outcome(program, outcome: Optional[Mapping]) -> tuple[dict, list[dict]]:
    objs: dict[str, Any] = {}
    fails: list[dict] = []
    for role in program.objects:
        o = (outcome or {}).get(role)
        if not o:
            objs[role] = {k: unknown("no per-object outcome record supplied") for k in ("grasp", "lift", "transport", "place", "slip")}
            continue
        gm = o.get("grasp_measure") or {}
        objs[role] = {
            "grasp": o.get("grasp", unknown("not recorded")), "lift": o.get("lift", unknown("not recorded")),
            "transport": o.get("transport", unknown("not recorded")), "place": o.get("place", unknown("not recorded")),
            "max_rise_lift_transport_m": o.get("max_rise_lift_transport_m", unknown("not recorded")),
            "final_xy_err_m": o.get("final_xy_err_m", unknown("not recorded")),
            "hold_rel_z_spread_m": gm.get("rel_z_spread_m", unknown("no grasp_measure recorded")),
            "hold_lr_imbalance": gm.get("lr_imbalance", unknown("no grasp_measure recorded")),
            "slip": unknown("no direct slip measurement is recorded; hold_rel_z_spread_m is the only related field")}
        for stage in ("grasp", "lift", "transport", "place"):
            if o.get(stage) is False:
                loc, idx = _loc(program, primitive=stage, role=role)
                fails.append({"class": "outcome_not_met", "requirement_id": f"outcome.{stage}", **loc, "stage_index": idx,
                              "control_step": unknown("an outcome is measured over a window, not at one step"),
                              "pose_difference": ({"final_xy_err_m": o["final_xy_err_m"]} if stage in ("transport", "place") and "final_xy_err_m" in o
                                                  else unknown("not recorded")),
                              "evidence_source": "per-object outcome record", "detail": {"object_role": role}})
                break   # the first unmet stage of this object; later ones are listed in `objects`
    return objs, fails


def _gaps(x: Any, path: str, acc: list[str]) -> None:
    if is_unknown(x):
        acc.append(path)
    elif isinstance(x, dict):
        for k, v in x.items():
            _gaps(v, f"{path}.{k}" if path else str(k), acc)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _gaps(v, f"{path}[{i}]", acc)


def resume_points(program, snapshots: Optional[Mapping], first_failure: Optional[Mapping]) -> list[dict]:
    """Existing snapshots only.  No claim about whether a patch may resume from one (item 7 decides that)."""
    out = []
    ff_idx = (first_failure or {}).get("stage_index")
    for lab, snap in (snapshots or {}).items():
        idx, it = _find(program, label=lab)
        out.append({"stage_label": lab, "instance_id": it.instance_id if it else unknown("label not in program"),
                    "stage_index": idx, "prefix_steps": int(snap.prefix_steps), "status_at_snapshot": snap.status,
                    "at_or_before_first_failure": (None if ff_idx is None or idx is None else bool(idx <= ff_idx)),
                    "validity": "unchecked",
                    "note": "a snapshot exists; whether a given patch may resume from it needs the item-7 dependency check"})
    out.sort(key=lambda r: (r["stage_index"] is None, r["stage_index"] if r["stage_index"] is not None else 0))
    return out


def build_failure_report(program, *, status: str, task_ok: Optional[bool] = None, meta: Optional[Mapping] = None,
                         audit: Optional[Mapping] = None, outcome: Optional[Mapping] = None,
                         snapshots: Optional[Mapping] = None, stage_by_step: Optional[list] = None,
                         record_gaps: Optional[list] = None) -> dict:
    """``audit``: ``{"policy_version", "unexcused": [rows], "support_unexplained": [rows]}`` (rows as in the policy output);
    ``outcome``: ``{role: per-object record}``; ``stage_by_step``: the per-control-step stage labels if available."""
    failures = _status_failures(program, status, meta or {}, task_ok, stage_by_step) + _audit_failures(program, audit, stage_by_step)
    objects, ofails = _outcome(program, outcome)
    failures += ofails
    if task_ok is False:
        failures.append({"class": "outcome_not_met", "requirement_id": "task.official_success", "instance_id": unknown("whole task"),
                         "stage_label": unknown("whole task"), "primitive": unknown("whole task"), "object_role": unknown("whole task"),
                         "stage_index": None, "control_step": unknown("whole task"),
                         "pose_difference": unknown("not recorded"), "evidence_source": "simulator task flag", "detail": {}})
    num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else float("inf")  # noqa: E731
    failures.sort(key=lambda f: (f["stage_index"] if f["stage_index"] is not None else 10 ** 9, num(f["control_step"]),
                                 _CLASS_RANK[f["class"]]))
    first = failures[0] if failures else None
    rep = {"schema": SCHEMA, "status_code": status, "task_ok": task_ok,
           "classes_present": [c for c in CLASSES if any(f["class"] == c for f in failures)],
           "first_failure": first, "failures": failures, "objects": objects,
           "resume_points": resume_points(program, snapshots, first),
           "requirements": {k: v for k, v in REQUIREMENTS.items()},
           "requirements_note": "report labels, not thresholds and not constraint_registry ids; the proposer cannot edit them",
           "ordering_note": "failures are ordered by (stage index, control step); this is temporal order, not a causal claim",
           "record_gaps": list(record_gaps or [])}
    gaps: list[str] = []
    for k in ("first_failure", "failures", "objects"):
        _gaps(rep[k], k, gaps)
    rep["unknown_fields"] = gaps
    return rep


def from_script_record(program, rec: Mapping, *, snapshots: Optional[Mapping] = None) -> dict:
    """Adapter for the s22/s23 per-run JSON (``status``, ``task_ok``, ``unexcused``, ``pots``, ``approach_plan``)."""
    audit = {"policy_version": "v1.1", "unexcused": rec.get("unexcused") or [], "support_unexplained": rec.get("support_unexplained") or []}
    gaps = []
    if "stage_failure" not in rec and str(rec.get("status", "")).startswith("stage_timeout"):
        gaps.append("script record has no meta.stage_failure: the control step and eef position of the timed-out stage are not available")
    return build_failure_report(program, status=rec["status"], task_ok=rec.get("task_ok"),
                                meta={"approach_plan": rec.get("approach_plan"), **{k: rec[k] for k in ("stage_failure", "grasp_failure", "place_failure") if k in rec}},
                                audit=audit, outcome=rec.get("pots") or rec.get("objects"), snapshots=snapshots, record_gaps=gaps)


def from_episode(program, ep, *, audit: Optional[Mapping] = None, outcome: Optional[Mapping] = None,
                 snapshots: Optional[Mapping] = None) -> dict:
    """Adapter for a ``RawEpisode`` (status, meta, per-step ``stage`` column, ``task_ok``)."""
    stage = ep.arrays.get("stage") if getattr(ep, "arrays", None) is not None else None
    return build_failure_report(program, status=ep.status, task_ok=getattr(ep, "task_ok", None), meta=ep.meta, audit=audit,
                                outcome=outcome, snapshots=snapshots,
                                stage_by_step=None if stage is None else [str(x) for x in stage])
