"""Step 8 diagnostic: substep-level contact record for one grasp, to separate (a) measurement omission, (b) transient loss of
contact, (c) sustained one-sided support, and to compute the simultaneous two-finger force peak.

A recorder is chained after the session's own physics hook (does not change dynamics); per physics substep it logs
  L, R     summed normal force finger1-geoms / finger2-geoms vs the target (same definition as the telemetry)
  G        normal force of gripper0 geoms that are NOT bound finger geoms vs the target (palm / hand / unbound pad)
  S        normal force target vs non-gripper geoms (table, basket, neighbours: support / resting contact)
and the gripper-vs-target geom-name pair table.
Output: reports/step8_contact_diag/T{t}_init{i}_{lvl}_{role}.npz (raw substep arrays + stage rows) and .json summary.

  s8_contact_diag.py TASK INIT LEVEL ROLE
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_validate import hold_idx  # noqa: E402
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

THR = 0.1
_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731


def install_recorder(s, name, rec):
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    orig_hook = s._hook

    def hook():
        orig_hook()
        m, d = s._mj()
        tgt = set(s.geom_ids_of(name))
        b = resolve_grasp_binding(m, mujoco, tgt, name)
        left, right = set(b.left_finger_geom_ids), set(b.right_finger_geom_ids)
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
        grip = {g for g in range(m.ngeom) if gname(g).startswith("gripper0")}
        prev = s.env.sim.step
        buf = np.zeros(6)

        def stepped(*a, **k):
            out = prev(*a, **k)
            L = R = G = S = 0.0
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in tgt) == (g2 in tgt):
                    continue
                t, o = (g1, g2) if g1 in tgt else (g2, g1)
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if f <= 0:
                    continue
                if o in left:
                    L += f
                elif o in right:
                    R += f
                elif o in grip:
                    G += f
                else:
                    S += f
                if o in grip:
                    rec["pairs"][(gname(o), gname(t))] += 1
            rec["ctrl"].append(int(s.control_step_index))
            rec["L"].append(L); rec["R"].append(R); rec["G"].append(G); rec["S"].append(S)
            return out

        s.env.sim.step = stepped

    s._hook = hook


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def main():
    t, init, lvl, role = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    name = dict(SPECS[t].objects)[role]
    prog = make_program(t)
    s = make_session(t, init, prog)
    rec = {"ctrl": [], "L": [], "R": [], "G": [], "S": [], "pairs": collections.Counter()}
    install_recorder(s, name, rec)
    roles = [r for r, _ in SPECS[t].objects]
    plan = make_plan(t, prog, grip={r: grip_for(PLANS[lvl], EXECUTOR["settle_after_open"]) for r in roles}, approach_check=EXECUTOR["approach_check"])
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s)
    s.close()
    a, st = ep.arrays, np.asarray(ep.arrays["stage"])
    ctrl = np.asarray(rec["ctrl"])
    L, R, G, S = (np.asarray(rec[k]) for k in "LRGS")
    h = hold_idx(a, role)
    sub = np.isin(ctrl, h)
    grasp_win = np.flatnonzero(np.isin(st, [f"{role}.pick_1.{x}" for x in ("grasp", "lift", "transport")]))
    sub_w = np.isin(ctrl, grasp_win)
    Lm, Rm = np.asarray(a["left_mean_n"])[h], np.asarray(a["right_mean_n"])[h]
    one_sided = np.minimum(Lm, Rm) <= THR
    rl = runs(one_sided)
    longest = max(rl, key=lambda x: x[1]) if rl else None
    z = np.asarray(a["pos_" + name])[h, 2] - np.asarray(a["state"])[h, 2]
    ub = np.asarray(a["left_max_n"])[grasp_win] + np.asarray(a["right_max_n"])[grasp_win]
    mx = max(np.nanmax(np.asarray(a["left_max_n"])[grasp_win]), np.nanmax(np.asarray(a["right_max_n"])[grasp_win]))
    fr = lambda m_: float(np.mean(m_)) if sub.any() else None  # noqa: E731
    res = {"task": t, "init": init, "level": lvl, "role": role, "target": name, "task_ok": bool(ep.task_ok), "n_rows": int(len(st)),
           "ctrl_max_vs_rows": [int(ctrl.max()), int(len(st))],
           "hold_steps": int(len(h)), "hold_substeps": int(sub.sum()),
           "step_mean_bilateral_frac": float(np.mean(~one_sided)) if len(h) else None,
           "substep_bilateral_frac": fr((L[sub] > THR) & (R[sub] > THR)),
           "substep_left_only_frac": fr((L[sub] > THR) & (R[sub] <= THR)),
           "substep_right_only_frac": fr((L[sub] <= THR) & (R[sub] > THR)),
           "substep_neither_frac": fr((L[sub] <= THR) & (R[sub] <= THR)),
           "n_one_sided_runs": len(rl), "longest_one_sided_run_steps": int(longest[1]) if longest else 0,
           "longest_run_stage_rows": [str(st[h[longest[0]]]), str(st[h[longest[0] + longest[1] - 1]])] if longest else None,
           "longest_run_missing_side": (("left" if Lm[longest[0]:longest[0] + longest[1]].mean() < Rm[longest[0]:longest[0] + longest[1]].mean() else "right")
                                        if longest else None),
           "hold_L_mean": float(Lm.mean()) if len(h) else None, "hold_R_mean": float(Rm.mean()) if len(h) else None,
           "hold_L_min_step": float(Lm.min()) if len(h) else None, "hold_R_min_step": float(Rm.min()) if len(h) else None,
           "gripper_other_vs_target_hold_mean_n": float(G[sub].mean()) if sub.any() else None,
           "gripper_other_vs_target_hold_frac_gt_thr": fr(G[sub] > THR),
           "support_vs_nongripper_hold_mean_n": float(S[sub].mean()) if sub.any() else None,
           "support_vs_nongripper_hold_frac_gt_thr": fr(S[sub] > THR),
           "rel_z_spread_mm": float(np.ptp(z) * 1000) if len(h) else None,
           "peak_simultaneous_substep_total_n": float((L + R)[sub_w].max()) if sub_w.any() else None,
           "peak_upper_bound_sum_of_step_finger_max_n": float(np.nanmax(ub)) if len(ub) else None,
           "peak_lower_bound_max_single_finger_n": float(mx) if len(grasp_win) else None,
           "gripper_pairs_top": [[list(k), v] for k, v in rec["pairs"].most_common(8)]}
    d = ROOT / "reports" / "step8_contact_diag"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"T{t}_init{init}_{lvl}_{role}"
    np.savez_compressed(f"{stem}.npz", ctrl=ctrl, L=L, R=R, G=G, S=S, stage=st, actions=np.asarray(a["actions"]),
                        left_mean_n=np.asarray(a["left_mean_n"]), right_mean_n=np.asarray(a["right_mean_n"]),
                        left_max_n=np.asarray(a["left_max_n"]), right_max_n=np.asarray(a["right_max_n"]), hold_idx=h)
    Path(f"{stem}.json").write_text(json.dumps(res, indent=1, default=_J))
    print(json.dumps(res, default=_J))


if __name__ == "__main__":
    main()
