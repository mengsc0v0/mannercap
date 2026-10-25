"""Step 8 one-off local check: attribute the single-control-step 'right finger < 0.1 N' dropout (bilateral_strict=False).

Uses the rollback-isolated SubstepRecorder.  Per physics substep it stores, for the grasp target: L / R bound-finger force,
G (other gripper geoms, e.g. palm), number of finger<->target contacts incl. zero-force ones, min contact distance per side,
and the positions of the target and of the two finger bodies (opening / relative motion).
Candidate explanations of a dropout step: (A) force carried by geoms outside the L/R binding, (B) contact present but force
below the 0.1 N detection threshold, (C) true loss of contact (no contact / distance > 0), (D) relative motion in transport.

  s8_dropout_diag.py TASK INIT ROLE      (gentle plan, all objects gentle: same as the response run)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_validate import hold_idx  # noqa: E402
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for, recorder_consistency  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402
from synthesis_v2.substep_recorder import SubstepRecorder  # noqa: E402

THR = 0.1


def main(t, init, role, lvl="gentle"):
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    prog = make_program(t)
    s = make_session(t, init, prog)
    name = dict(SPECS[t].objects)[role]
    names = [n for _, n in SPECS[t].objects]
    holder = {}
    sub = SubstepRecorder(s, lambda sess: holder["read"]()).install_state_hooks()
    orig = s._hook

    def hook():
        orig()
        m, d = s._mj()
        tgt = set(s.geom_ids_of(name))
        b = resolve_grasp_binding(m, mujoco, tgt, name)
        left, right = set(b.left_finger_geom_ids), set(b.right_finger_geom_ids)
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
        grip = {g for g in range(m.ngeom) if gn(g).startswith("gripper0")}
        bodyL, bodyR = int(m.geom_bodyid[min(left)]), int(m.geom_bodyid[min(right)])
        tb = int(m.geom_bodyid[min(tgt)])
        buf = np.zeros(6)
        holder["names"] = {"left": sorted(gn(g) for g in left), "right": sorted(gn(g) for g in right)}

        def read():
            L = R = G = 0.0
            nl = nr = 0
            dl = dr = 9.0
            pairs = []
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if (g1 in tgt) != (g2 in tgt):                        # target vs a non-gripper geom: external disturbance
                    og = g2 if g1 in tgt else g1
                    if og not in grip and f > 0:
                        pairs.append(("EXT:" + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[og])) or "?"), f))
                        continue
                gg, o = (g1, g2) if g1 in grip else ((g2, g1) if g2 in grip else (None, None))
                if gg is None or o not in tgt:
                    continue
                if gg in left:
                    L += f; nl += 1; dl = min(dl, float(c.dist))
                elif gg in right:
                    R += f; nr += 1; dr = min(dr, float(c.dist))
                else:
                    G += f
                pairs.append((gn(gg), f))
            return (L, R, G, nl, nr, dl, dr, *d.xpos[tb], *d.xpos[bodyL], *d.xpos[bodyR], tuple(pairs))

        holder["read"] = read
        prev = s.env.sim.step

        def stepped(*a, **k):
            out = prev(*a, **k)
            sub.sample()
            return out

        s.env.sim.step = stepped

    s._hook = hook
    grips = {r: grip_for(PLANS[lvl], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    plan = make_plan(t, prog, grip=grips, approach_check=EXECUTOR["approach_check"])
    t0 = time.time()
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s)
    s.close()
    ctrl, rows = sub.formal_arrays()
    A = np.array([r[:-1] for r in rows], dtype=float)
    pairs = [r[-1] for r in rows]
    a = ep.arrays
    st = [str(x) for x in a["stage"]]
    # consistency with the executed telemetry (this check is what makes the recorder trustworthy)
    rec = {"ctrl": ctrl, "L": {name: A[:, 0]}, "R": {name: A[:, 1]}}
    cons = recorder_consistency(ep, rec)
    h = hold_idx(a, role)
    Lm, Rm = np.asarray(a["left_mean_n"]), np.asarray(a["right_mean_n"])
    low = [int(k) for k in h if min(Lm[k], Rm[k]) <= THR]
    out = {"task": t, "init": init, "role": role, "level": lvl, "status": ep.status, "task_ok": bool(ep.task_ok), "n_hold": int(len(h)),
           "rolled_back_cost": sub.rolled_back_cost(), "recorder_vs_telemetry": cons, "binding": holder["names"], "dropout_steps": [], "wall_s": round(time.time() - t0, 1)}
    def ext_summary(ks):
        agg = {}
        for k_ in ks:
            for i_ in np.flatnonzero(ctrl == k_):
                for n_, f_ in pairs[i_]:
                    if n_.startswith("EXT:"):
                        v = agg.setdefault(n_[4:], {"max_n": 0.0, "substeps": 0, "first_step": k_})
                        v["max_n"], v["substeps"] = max(v["max_n"], f_), v["substeps"] + 1
        return agg

    for k in low:
        w = np.flatnonzero(ctrl == k)
        nb = [int(x) for x in range(len(ctrl)) if False]
        L, R, G, nl, nr, dl, dr = (A[w, i] for i in range(7))
        weak = "R" if Rm[k] <= Lm[k] else "L"
        wf = (R if weak == "R" else L)
        nc = (nr if weak == "R" else nl)
        dist = (dr if weak == "R" else dl)
        opening = np.linalg.norm(A[w, 10:13] - A[w, 13:16], axis=1)
        rel = (A[w, 7:10] - 0.5 * (A[w, 10:13] + A[w, 13:16]))
        out["dropout_steps"].append({
            "control_step": k, "stage": st[k], "weak_side": weak, "telemetry_L_R_mean": [float(Lm[k]), float(Rm[k])],
            "n_substeps": int(len(w)), "weak_side_force": {"min": float(wf.min()), "max": float(wf.max()), "mean": float(wf.mean()),
                                                           "n_zero": int((wf <= 0).sum()), "n_lt_thr": int((wf < THR).sum()), "n_ge_thr": int((wf >= THR).sum())},
            "weak_side_contacts_per_substep": [int(nc.min()), int(nc.max())], "weak_side_min_dist_m": [float(dist.min()), float(dist.max())],
            "strong_side_force_mean": float((L if weak == "R" else R).mean()), "other_gripper_geoms_G": {"mean": float(G.mean()), "max": float(G.max())},
            "pair_names_seen": sorted({n for ps in (pairs[i] for i in w) for n, f in ps if not n.startswith("EXT:")}),
            "external_contacts_on_target_k-3..k+1": ext_summary(range(k - 3, k + 2)),
            "finger_opening_m": [float(opening.min()), float(opening.max())], "target_rel_to_finger_mid_m": {"min": [float(x) for x in rel.min(0)], "max": [float(x) for x in rel.max(0)]},
            "action_gripper": [float(np.asarray(a["actions"])[k][-1]), float(np.asarray(a["actions"])[max(k - 1, 0)][-1])],
            "eef_step_m": float(np.linalg.norm(np.asarray(a["next_state"])[k][:3] - np.asarray(a["state"])[k][:3]))})
    # context: 3 control steps either side of each dropout (control-step means)
    out["context"] = {int(k): {"L": [round(float(x), 2) for x in Lm[max(k - 3, 0):k + 4]], "R": [round(float(x), 2) for x in Rm[max(k - 3, 0):k + 4]],
                               "stage": st[max(k - 3, 0):k + 4]} for k in low}
    out["external_contacts_on_target_whole_transport"] = ext_summary([k_ for k_, x in enumerate(st) if x == f"{role}.pick_1.transport"])
    d = ROOT / "reports" / "step8_dropout_diag"
    d.mkdir(exist_ok=True)
    (d / f"T{t}_init{init}_{lvl}_{role}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
