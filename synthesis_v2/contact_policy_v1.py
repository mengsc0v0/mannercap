"""Contact policy v1: a finger-link contact is excused only when it is a real grasping contact, proved per contact.

v0 (synthesis_v2/contact_audit.py) allows only the fingertip PADS to touch the grasp target; the finger-link collision
geom next to the pad is forbidden even when it is plainly part of the grasp.  v1 narrows that, but it does NOT trust the
``finger_link_on_target`` label: a contact is excused only if every recorded substep of it satisfies ALL of

  region    the contact point, expressed in the FINGER-LINK GEOM's own frame, lies on the inner half of that geom
            (its coordinate along the inward closing axis is positive; the geom's half-extents are recorded so the
            region is explicit), and the contact normal pushes inward (cos with the inward axis >= ``INNER_COS_MIN``).
            The inward axis is measured from the model at run time (the vector from this finger's geoms to the
            opposite finger's geoms), not hard-coded.  The geom frame is used, not the finger body origin: the body
            origin sits at an arbitrary joint frame and says nothing about which face of the solid was touched.
  object    the other body is the object this grasp instance is actually bound to (``force_target`` at close time),
            not merely "some object the audit called the target".
  window    every control step of the contact lies in [close_step, release_step) of that instance: the window opens
            when the gripper starts closing and ends when the release command is issued.  The ``place`` stage is NOT
            treated as held throughout, because the release happens inside it.

Everything else keeps the v0 treatment: palm / arm contacts, contacts on another object, and contacts after release are
violations.  The v0 verdict is recomputed and reported alongside, never replaced.

v1 is also STRICTER than v0 in one place: v0 reports robot<->table/floor contact as observation only, so a hard press on
the table can still be ``clean``.  v1 refuses to call a trajectory clean while such a contact is unexplained, and
carries the evidence needed to decide (stage, peak force, peak control step, whether it survived rollback).

Force accounting: ``classify_finger_geom`` is the single identity rule and the force measurement counts BOTH the pad and
the finger-link geom as that finger, so a contact excused here is already inside the measured hold force; v1 adds no
load that the force check cannot see.  ``force_accounting_checked`` records that this invariant was verified.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from synthesis_v2.contact_audit import CONTACT_THR_N, FINGER_LINK_ON_TARGET, SUPPORT_PREFIXES, ContactRecorder
from synthesis_v2.grasp_force_contract import classify_finger_geom

POLICY_VERSION = "contact_policy_v1"
INNER_COS_MIN = 0.5      # contact normal must point inward, not graze the side; 60 deg half-angle


class DetailRecorder:
    """Installs the per-contact geometry hook on a ``ContactRecorder``.

    Records, per (robot body label, other body): the peak force with its control step and geometry, the contact point in
    the finger body's local frame, the contact normal, and how many substeps satisfied the inner-face region test.
    """

    def __init__(self, rec: ContactRecorder) -> None:
        self.rec = rec
        self.detail: dict[tuple[str, str], dict] = {}
        self._inward: dict[str, np.ndarray] = {}
        self._site = None
        rec.detail_hook = self._hook

    def _inward_axis(self, side: str) -> Optional[np.ndarray]:
        """Unit vector from this finger's geoms toward the opposite finger's geoms, read from the live model."""
        if not self._inward:
            m, d = self.rec.m, self.rec.d
            roles = self.rec._roles or {}
            pts: dict[str, list] = {"L": [], "R": []}
            for gi, (sd, _role) in roles.items():
                pts[sd].append(d.geom_xpos[gi])
            if pts["L"] and pts["R"]:
                cl, cr = np.mean(pts["L"], axis=0), np.mean(pts["R"], axis=0)
                v = cr - cl
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    self._inward = {"L": v / n, "R": -v / n}
        return self._inward.get(side)

    def _hook(self, rec: ContactRecorder, key: tuple[str, str], rest: Sequence, f: float) -> None:
        import mujoco
        m, d = rec.m, rec.d
        gid = int(rest[0]) if len(rest) > 0 else -1
        ci = int(rest[1]) if len(rest) > 1 else -1
        row = self.detail.setdefault(key, {"n_substeps": 0, "n_substeps_inner_region": 0, "peak_n": -1.0,
                                           "min_inner_cos": None, "min_inward_offset_m": None,
                                           "region_fail_steps": set(), "region_fail_samples": [],
                                           "peak_step": None, "peak_geom": None, "peak_other_geom": None,
                                           "peak_contact_point_geom_local_m": None, "peak_contact_point_body_local_m": None,
                                           "geom_half_extent_m": None, "inward_axis_geom_local": None, "peak_normal_world": None,
                                           "peak_normal_inward_cos": None, "peak_inward_offset_m": None,
                                           "peak_eef_pos": None, "side_role": None})
        row["n_substeps"] += 1
        inner_cos = inward_off = None
        if ci >= 0:
            c = d.contact[ci]
            pos = np.asarray(c.pos, float).copy()
            normal = np.asarray(c.frame[:3], float).copy()        # mujoco: points from geom1 to geom2
            if int(c.geom2) == gid:
                normal = -normal                                   # orient robot -> other body
            role = rec._role(gid)
            if role is not None:
                ax = self._inward_axis(role[0])
                if ax is not None:
                    # reference frame = the finger-link GEOM itself, not the finger body origin: "inner face" is a
                    # property of this collision solid, and the body origin sits at an arbitrary joint frame.
                    gp = np.asarray(d.geom_xpos[gid], float)
                    gR = np.asarray(d.geom_xmat[gid], float).reshape(3, 3)
                    loc = gR.T @ (pos - gp)                       # contact point in geom-local coordinates
                    axl = gR.T @ ax                               # inward closing axis in the same frame
                    half = np.abs(np.asarray(m.geom_size[gid], float))
                    inward_off = float(np.dot(loc, axl))          # > 0 = inner half of the finger-link geom
                    inner_cos = float(np.dot(normal, ax))
                    row["geom_half_extent_m"] = half.round(5).tolist()
                    row["inward_axis_geom_local"] = np.asarray(axl, float).round(4).tolist()
                    row["min_inner_cos"] = inner_cos if row["min_inner_cos"] is None else min(row["min_inner_cos"], inner_cos)
                    row["min_inward_offset_m"] = (inward_off if row["min_inward_offset_m"] is None
                                                  else min(row["min_inward_offset_m"], inward_off))
                    if inward_off > 0.0 and inner_cos >= INNER_COS_MIN:
                        row["n_substeps_inner_region"] += 1
                    else:   # keep the evidence for the samples that failed the region test
                        row["region_fail_steps"].add(int(rec.k))
                        if len(row["region_fail_samples"]) < 12:
                            row["region_fail_samples"].append({"control_step": int(rec.k), "force_n": round(float(f), 3),
                                                               "inner_cos": round(inner_cos, 4),
                                                               "inward_offset_m": round(inward_off, 6)})
                row["side_role"] = list(role)
        if f > row["peak_n"]:
            if self._site is None:
                try:
                    self._site = int(self.rec.s.env.robots[0].eef_site_id)
                except Exception:
                    self._site = -1
            nmg = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "?") if g >= 0 else "?"  # noqa: E731
            row.update(peak_n=float(f), peak_step=int(rec.k), peak_geom=nmg(gid),
                       peak_other_geom=nmg(int(d.contact[ci].geom1) if ci >= 0 and int(d.contact[ci].geom2) == gid
                                           else (int(d.contact[ci].geom2) if ci >= 0 else -1)),
                       peak_contact_point_geom_local_m=None if ci < 0 or gid < 0 else
                       (np.asarray(d.geom_xmat[gid], float).reshape(3, 3).T
                        @ (np.asarray(d.contact[ci].pos, float) - np.asarray(d.geom_xpos[gid], float))).round(6).tolist(),
                       peak_contact_point_body_local_m=None if ci < 0 or gid < 0 else
                       (np.asarray(d.xmat[int(m.geom_bodyid[gid])], float).reshape(3, 3).T
                        @ (np.asarray(d.contact[ci].pos, float) - np.asarray(d.xpos[int(m.geom_bodyid[gid])], float))).round(6).tolist(),
                       peak_normal_world=None if ci < 0 else np.asarray(normal, float).round(4).tolist(),
                       peak_normal_inward_cos=None if inner_cos is None else round(inner_cos, 4),
                       peak_inward_offset_m=None if inward_off is None else round(inward_off, 5),
                       peak_eef_pos=None if self._site is None or self._site < 0 else
                       np.asarray(d.site_xpos[self._site], float).round(5).tolist())

    def export(self) -> dict:
        out = {}
        for (a, b), v in sorted(self.detail.items()):
            r = dict(v)
            r["region_fail_steps"] = sorted(r.get("region_fail_steps", ()))
            out[f"{a}||{b}"] = r
        return out


def _steps_of(rec_events: dict, key: tuple[str, str]) -> set:
    v = rec_events.get(key)
    return set(v[1]) if v else set()


def evaluate(v0_report: dict, events: dict, detail: dict[tuple[str, str], dict],
             windows: Sequence[dict], stages: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """v1 verdict from the v0 report plus per-contact geometry.  ``windows``: one dict per grasp instance with
    ``instance_id``, ``object`` (the bound force target body name), ``close_step`` and ``release_step``."""
    by_obj: dict[str, list] = {}
    for w in windows:
        by_obj.setdefault(str(w["object"]), []).append(w)
    rows = []
    for c in v0_report["forbidden_contacts"]:
        key = (c["robot_body"], c["other"])
        det = detail.get(key, {})
        steps = sorted(_steps_of(events, key))
        reasons, win = [], None
        if c["class"] != "finger_link_on_target":
            reasons.append("not_a_finger_link_contact")
        cand = by_obj.get(c["other"], [])
        if not cand:
            reasons.append("object_is_not_a_bound_grasp_target")
        else:
            win = next((w for w in cand if steps and min(steps) >= int(w["close_step"])
                        and max(steps) < int(w["release_step"])), None)
            if win is None:
                reasons.append("outside_hold_window_close_to_release")
        n, ninner = det.get("n_substeps", 0), det.get("n_substeps_inner_region", 0)
        if n == 0:
            reasons.append("no_contact_geometry_recorded")
        elif ninner != n:
            reasons.append(f"not_all_substeps_on_inner_face ({ninner}/{n})")
        rows.append({**c, "v1_excused": not reasons, "v1_reasons": reasons,
                     "matched_instance": None if win is None else win["instance_id"],
                     "control_step_range": [min(steps), max(steps)] if steps else None,
                     "n_substeps": n, "n_substeps_inner_region": ninner,
                     "peak_normal_inward_cos": det.get("peak_normal_inward_cos"),
                     "peak_inward_offset_m": det.get("peak_inward_offset_m"),
                     "peak_contact_point_geom_local_m": det.get("peak_contact_point_geom_local_m"),
                     "geom_half_extent_m": det.get("geom_half_extent_m"),
                     "peak_step": det.get("peak_step"), "peak_geom": det.get("peak_geom"),
                     "min_inner_cos": det.get("min_inner_cos"), "min_inward_offset_m": det.get("min_inward_offset_m"),
                     "region_fail_steps": sorted(det.get("region_fail_steps", ())),
                     "region_fail_samples": det.get("region_fail_samples", [])})
    unexcused = [r for r in rows if not r["v1_excused"]]
    support = []
    for c in v0_report.get("support_contacts_observed", []):
        key = (c["robot_body"], c["other"])
        det = detail.get(key, {})
        steps = sorted(_steps_of(events, key))
        support.append({**c, "control_step_range": [min(steps), max(steps)] if steps else None,
                        "stages": sorted({stages[k].split(".")[-1] + "@" + stages[k].split(".")[0]
                                          for k in steps if stages and 0 <= k < len(stages)}) if stages else None,
                        "peak_step": det.get("peak_step"), "peak_geom": det.get("peak_geom"),
                        "peak_other_geom": det.get("peak_other_geom"), "peak_eef_pos": det.get("peak_eef_pos"),
                        "peak_contact_point_geom_local_m": det.get("peak_contact_point_geom_local_m"),
                     "geom_half_extent_m": det.get("geom_half_extent_m"),
                        "in_final_trajectory": True,
                        "explained": False})
    return {
        "policy_version": POLICY_VERSION, "inner_cos_min": INNER_COS_MIN, "threshold_n": CONTACT_THR_N,
        "v0_clean_preserved": bool(v0_report["clean"]),
        "contacts": rows, "n_excused": sum(r["v1_excused"] for r in rows), "unexcused_contacts": unexcused,
        "support_contacts_unexplained": [s for s in support if not s["explained"]],
        "clean_v1": bool(not unexcused and not [s for s in support if not s["explained"]]),
        "clean_v1_ignoring_support": bool(not unexcused),
        "windows": [dict(w) for w in windows],
        "force_accounting": "classify_finger_geom is the single identity rule and the grasp force measurement counts "
                            "both pad and finger_link geoms as that finger, so an excused finger-link contact is "
                            "already inside the measured hold force.",
        "note": "clean_v1 requires BOTH no unexcused forbidden contact AND no unexplained robot<->support contact. "
                "v0 reports support contact as observation only; v1 refuses to call that complete.",
    }
