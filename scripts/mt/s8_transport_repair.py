"""Step 8: one bounded transport-repair case, three conditions on the SAME request / grasp plan / executor / criteria.

  A  original program (soup transport_dz 0.175), full run; stage snapshots are kept
  B  repair candidate (soup transport_dz = DZ only), full re-run from the initial state
  C  same candidate, new session, resumed from A's snapshot at the START of the soup transport stage

B and C use the identical program object (same program hash).  Recorders are rollback-isolated; the resumed session is
seeded with A's formal prefix (substeps, contact events) taken at the snapshot, so the three records are whole-episode records.

  s8_transport_repair.py TASK INIT ROLE [DZ]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_validate import hold_idx  # noqa: E402
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, CONTACT_THR_N, audit_role, cfg_path, grip_for, per_object, recorder_consistency  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402
from synthesis_v2.substep_recorder import SubstepRecorder  # noqa: E402
from synthesis_v2.transport_repair_rule import CarriedContactRecorder  # noqa: E402

THR = CONTACT_THR_N
_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731


def with_transport_dz(prog, role, dz):
    label = f"{role}.pick_1.transport"
    inst, hit = [], 0
    for it in prog.instances:
        if it.primitive == "transport" and it.label == label:
            it, hit = replace(it, params={**it.params, "transport_dz": dz}), hit + 1
        inst.append(it)
    assert hit == 1, "exactly the one named transport instance must change"
    return replace(prog, instances=tuple(inst))


def install(s, names):
    """Per physics substep, per object: (L, R, G palm/other-gripper, B basket) normal force on the object."""
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    holder = {}
    sub = SubstepRecorder(s, lambda sess: holder["read"]()).install_state_hooks()
    orig = s._hook

    def hook():
        orig()
        m, d = s._mj()
        tgt = {n: set(s.geom_ids_of(n)) for n in names}
        bind = {n: resolve_grasp_binding(m, mujoco, tgt[n], n) for n in names}
        left = {n: set(bind[n].left_finger_geom_ids) for n in names}
        right = {n: set(bind[n].right_finger_geom_ids) for n in names}
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""  # noqa: E731
        grip = {g for g in range(m.ngeom) if gn(g).startswith("gripper0")}
        basket = set(s.geom_ids_of("basket_1"))
        buf = np.zeros(6)

        def read():
            acc = {n: [0.0, 0.0, 0.0, 0.0] for n in names}
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                for n in names:
                    T = tgt[n]
                    if (g1 in T) == (g2 in T):
                        continue
                    o = g2 if g1 in T else g1
                    k = 0 if o in left[n] else 1 if o in right[n] else 2 if o in grip else 3 if o in basket else None
                    if k is None:
                        continue
                    mujoco.mj_contactForce(m, d, ci, buf)
                    acc[n][k] += abs(float(buf[0]))
            return tuple(x for n in names for x in acc[n])

        holder["read"] = read
        prev = s.env.sim.step

        def stepped(*a, **k):
            out = prev(*a, **k)
            sub.sample()
            return out

        s.env.sim.step = stepped

    s._hook = hook
    return sub


def run(t, init, prog, grips, *, snapshots=None, resume=None, seed=None):
    s = make_session(t, init, prog)
    names = [n for _, n in SPECS[t].objects]
    sub = install(s, names)
    box = {}

    def on_session(ss):                  # AFTER reset: reset rebuilds the sim, a wrapper installed earlier would be lost
        box["crec"] = ContactRecorder(ss).install()
        box["car"] = CarriedContactRecorder(ss, names).install()
        if resume is not None:
            box["crec"].seed(seed["crec"].events_at(resume.session), seed["crec"].companion_at(resume.session))
            box["car"].seed(seed["car"].events_at(resume.session), seed["car"].companion_at(resume.session))

    if resume is not None:
        sub.seed_prefix(*seed["sub"].rows_at(resume.session))
    plan = make_plan(t, prog, grip=grips, approach_check=EXECUTOR["approach_check"])
    t0 = time.time()
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s, on_session=on_session, snapshots=snapshots, resume=resume)
    wall = time.time() - t0
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 4 * len(names))
    crec = box["crec"]
    return {"ep": ep, "plan": plan, "sub": sub, "crec": crec, "car": box["car"], "ctrl": ctrl, "arr": arr, "names": names, "wall": wall,
            "prefix_rows": 0 if resume is None else len(seed["sub"].rows_at(resume.session)[0])}


def summarize(t, role, R, prog):
    ep, names, ctrl, arr = R["ep"], R["names"], R["ctrl"], R["arr"]
    a = ep.arrays
    st = np.asarray(a["stage"])
    roles = [r for r, _ in SPECS[t].objects]
    recx = {"ctrl": ctrl, "L": {n: arr[:, 4 * i] for i, n in enumerate(names)}, "R": {n: arr[:, 4 * i + 1] for i, n in enumerate(names)}}
    po = per_object(ep, prog, t, recx)
    stage_of = st[np.clip(ctrl, 0, len(st) - 1)]
    out = {"status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(len(st)), "program_hash": prog.program_hash(),
           "plan_hash": R["plan"].plan_hash(), "recorder_vs_telemetry": recorder_consistency(ep, recx), "objects": {}}
    for i, (r, n) in enumerate(zip(roles, names)):
        h = hold_idx(a, r)
        L, Rr = np.asarray(a["left_mean_n"])[h], np.asarray(a["right_mean_n"])[h]
        B = arr[:, 4 * i + 3]
        G = arr[:, 4 * i + 2]
        act = B > THR
        by_stage = {}
        for sg in sorted(set(stage_of[act])):
            m = act & (stage_of == sg)
            by_stage[str(sg)] = {"peak_n": float(B[m].max()), "substeps": int(m.sum()), "control_steps": int(len(set(ctrl[m]))),
                                 "first_control_step": int(ctrl[m].min())}
        cm = act & np.isin(stage_of, [f"{r}.pick_1.lift", f"{r}.pick_1.transport"])
        carried = {"peak_n": float(B[cm].max()) if cm.any() else 0.0, "substeps": int(cm.sum()), "control_steps": int(len(set(ctrl[cm]))),
                   "first_control_step": int(ctrl[cm].min()) if cm.any() else None, "first_stage": str(stage_of[cm][0]) if cm.any() else None}
        tr = np.flatnonzero(st == f"{r}.pick_1.transport")
        o = po[r]
        out["objects"][r] = {
            "grasp_stable": o.get("grasp_stable"), "measurement_valid": o.get("measurement_valid"), "hold_n": int(len(h)),
            "left_min_n": float(L.min()), "right_min_n": float(Rr.min()), "one_sided_control_steps": int((np.minimum(L, Rr) <= THR).sum()),
            "bilateral_strict": o.get("bilateral_strict"), "lr_imbalance": o.get("lr_imbalance"), "rel_z_spread_m": o.get("rel_z_spread_m"),
            "hold_total_min": o.get("hold_min"), "hold_total_max": o.get("hold_max"), "hold_total_mean": o.get("hold_mean"),
            "plan_band_1_22_all_hold_steps": bool(o.get("hold_min") is not None and o["hold_min"] >= 1.0 and o["hold_max"] <= 22.0),
            "peak_simultaneous_n": o.get("peak_simultaneous_n"),
            "basket_contact": {"carried_phase_lift_transport": carried, "by_stage_all": by_stage,
                               "note": "by_stage_all is NOT a verdict: contact in place/retreat/later stages is not auto-expected. Still to be separated: (a) intended support contact of the released object, (b) unintended contact (basket rim, neighbours), (c) excessive impact on an intended contact. Only the carried phase (lift/transport) is evaluated here"},
            "palm_or_other_gripper_geom_mean_n_in_hold": float(G[np.isin(ctrl, h)].mean()) if np.isin(ctrl, h).any() else None,
            "transport_eef_z_end_m": float(np.asarray(a["state"])[tr[-1], 2]) if len(tr) else None,
            "transport_eef_z_max_m": float(np.asarray(a["state"])[tr, 2].max()) if len(tr) else None, "transport_steps": int(len(tr))}
    stages = [str(x) for x in st]
    out["contact_policy"] = {k: R["crec"].report(stages)[k] for k in ("clean", "max_forbidden_n", "forbidden_contacts")}
    out["contact_policy"]["n_event_keys"] = len(R["crec"].events)
    c = R["sub"].cost()
    simulated_formal = c["formal_rows"] - R["prefix_rows"]
    out["cost"] = {"executed_substeps": c["executed_substeps"], "formal_substeps_simulated": simulated_formal,
                   "discarded_substeps_lookahead": c["executed_substeps"] - simulated_formal, "discarded_branches": c["discarded_branches"],
                   "wall_s": round(R["wall"], 1), "inherited_prefix_substeps": R["prefix_rows"]}
    return out


def per_step(R, nobj_cols):
    ctrl, arr = R["ctrl"], R["arr"]
    n = int(ctrl.max()) + 1
    cnt = np.bincount(ctrl, minlength=n).astype(float)
    return np.stack([np.bincount(ctrl, weights=arr[:, j], minlength=n) / np.maximum(cnt, 1) for j in range(nobj_cols)], axis=1)


def compare(X, Y, k0, ncols):
    """Compare two whole-episode records from control step k0 on (arrays, per-step substep means)."""
    ax, ay = X["ep"].arrays, Y["ep"].arrays
    res = {"n_steps": [int(len(ax["stage"])), int(len(ay["stage"]))], "stage_sequence_equal_from_k0": bool(np.array_equal(np.asarray(ax["stage"])[k0:], np.asarray(ay["stage"])[k0:]))}
    if len(ax["stage"]) == len(ay["stage"]):
        for key in ("actions", "state", "next_state", "left_mean_n", "right_mean_n"):
            d = np.abs(np.asarray(ax[key], float)[k0:] - np.asarray(ay[key], float)[k0:])
            res[f"max_abs_diff_{key}"] = float(np.nanmax(d))
        px, py = per_step(X, ncols)[k0:], per_step(Y, ncols)[k0:]
        res["substep_mean_force_max_abs_diff_n"] = float(np.abs(px - py).max())
        res["control_steps_with_force_diff_gt_1N"] = int((np.abs(px - py).max(axis=1) > 1.0).sum())
        res["compared_control_steps"] = int(len(px))
    return res


def main(t, init, role, dz):
    prog0 = make_program(t)
    prog1 = with_transport_dz(prog0, role, dz)
    names = dict(SPECS[t].objects)
    label = f"{role}.pick_1.transport"
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    diff = [(a.label, a.params.get("transport_dz"), b.params.get("transport_dz")) for a, b in zip(prog0.instances, prog1.instances) if dict(a.params) != dict(b.params)]
    snaps = {}
    A = run(t, init, prog0, grips, snapshots=snaps)
    snap = snaps[label]
    k0 = int(snap.session["control_step_index"])
    B = run(t, init, prog1, grips)
    C = run(t, init, prog1, grips, resume=snap, seed={"sub": A["sub"], "crec": A["crec"], "car": A["car"]})
    sa, sb, sc = (summarize(t, role, R, p) for R, p in ((A, prog0), (B, prog1), (C, prog1)))
    ncols = 4 * len(A["names"])
    stored = np.load(ROOT / "reports" / f"step8_t{t}_raw" / f"resp_init{init}_gentle.npz", allow_pickle=True)
    sub_cmp = {"substep_rows": [int(len(B["ctrl"])), int(len(C["ctrl"]))], "ctrl_index_equal": bool(np.array_equal(B["ctrl"], C["ctrl"])),
               "force_array_max_abs_diff_n": float(np.abs(B["arr"] - C["arr"]).max()) if B["arr"].shape == C["arr"].shape else "shape_differs",
               "force_array_bitwise_equal": bool(B["arr"].shape == C["arr"].shape and np.array_equal(B["arr"], C["arr"]))}
    out = {"task": t, "init": init, "role": role, "candidate": {"transport_dz_m": dz, "original_m": prog0.instances[[i.label for i in prog0.instances].index(label)].params["transport_dz"], "changed_instances": diff},
           "resume_point": {"stage": snap.stage, "control_step_index": k0, "prefix_steps": int(snap.prefix_steps), "meta_contains_transport_dz": "transport_dz" in json.dumps(snap.meta)},
           "A": sa, "B": sb, "C": sc,
           "checks": {"program_hash_A_ne_B": sa["program_hash"] != sb["program_hash"], "program_hash_B_eq_C": sb["program_hash"] == sc["program_hash"], "plan_hash_B_eq_C": sb["plan_hash"] == sc["plan_hash"],
                      "A_vs_stored_response_record": {key: float(np.nanmax(np.abs(np.asarray(A["ep"].arrays[key], float) - np.asarray(stored[key], float)))) if len(stored[key]) == len(A["ep"].arrays[key]) else "length_differs" for key in ("actions", "state")},
                      "prefix_A_vs_B_before_k0": {key: float(np.nanmax(np.abs(np.asarray(A["ep"].arrays[key], float)[:k0] - np.asarray(B["ep"].arrays[key], float)[:k0]))) for key in ("actions", "state", "left_mean_n", "right_mean_n")},
                      "B_vs_C_from_k0": compare(B, C, k0, ncols), "B_vs_C_all_substeps": sub_cmp, "A_vs_B_from_k0": compare(A, B, k0, ncols)}}
    totals = {"A_full": sa["cost"]["executed_substeps"], "B_extra_full_rerun": sb["cost"]["executed_substeps"], "C_extra_stage_resume": sc["cost"]["executed_substeps"]}
    totals["first_run_plus_one_repair_attempt_via_B_rerun"] = totals["A_full"] + totals["B_extra_full_rerun"]
    totals["first_run_plus_one_repair_attempt_via_C_resume"] = totals["A_full"] + totals["C_extra_stage_resume"]
    totals["wall_s"] = {"A": sa["cost"]["wall_s"], "B": sb["cost"]["wall_s"], "C": sc["cost"]["wall_s"]}
    out["cost_summary"] = totals
    d = ROOT / "reports" / "step8_transport_repair"
    d.mkdir(exist_ok=True)
    tag = f"T{t}_init{init}_{role}_dz{int(round(dz * 1000))}"
    (d / f"{tag}.json").write_text(json.dumps(out, indent=1, default=_J))
    np.savez_compressed(d / f"{tag}.npz", **{f"{c}_{k}": np.asarray(R["ep"].arrays[k]) for c, R in (("A", A), ("B", B), ("C", C)) for k in ("stage", "actions", "state", "next_state", "left_mean_n", "right_mean_n")},
                        **{f"{c}_sub_ctrl": R["ctrl"] for c, R in (("A", A), ("B", B), ("C", C))}, **{f"{c}_sub_arr": R["arr"] for c, R in (("A", A), ("B", B), ("C", C))})
    print(json.dumps(out, indent=1, default=_J))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.205)
