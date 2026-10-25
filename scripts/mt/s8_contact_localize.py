"""Step 8: read-only contact localisation.  Replays one already-run condition (same program, plan, force/motion parameters; nothing is
changed) with an extra observer that logs every robot<->alphabet_soup contact per physics substep: robot geom, soup geom, normal force,
signed distance, contact point in the hand (eef-site) frame and relative to the soup centre.  Used only to attribute the contact seen in
the audit; it does not alter the episode (the recorders only read ``sim.data``).

  s8_contact_localize.py TAG TASK INIT [yaw_deg lateral_m dz_m longitudinal_m]   (no candidate => the default grid)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for  # noqa: E402
from s8_transport_repair import run  # noqa: E402
import synthesis_v2.contact_audit as CA  # noqa: E402

LOG: list = []
K_LO, K_HI = 30, 95
_orig = CA.ContactRecorder._observe


def _observe(self):
    _orig(self)
    if not (K_LO <= self.k <= K_HI):
        return
    import mujoco
    m, d = self.m, self.d
    gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
    bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
    site = int(self.s.env.robots[0].eef_site_id)
    sp, sR = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
    buf = np.zeros(6)
    for ci in range(int(d.ncon)):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = bn(g1), bn(g2)
        if CA._is_robot(b1) == CA._is_robot(b2):
            continue
        rg, og = (g1, g2) if CA._is_robot(b1) else (g2, g1)
        if not bn(og).startswith("alphabet_soup"):
            continue
        mujoco.mj_contactForce(m, d, ci, buf)
        LOG.append({"k": int(self.k), "robot_body": bn(rg).replace("gripper0_", ""), "robot_geom": gn(rg), "robot_geom_type": int(m.geom_type[rg]),
                    "soup_geom": gn(og), "force_n": abs(float(buf[0])), "dist_m": float(c.dist),
                    "pos_in_hand_m": (sR.T @ (np.asarray(c.pos) - sp)).tolist(),
                    "pos_rel_soup_centre_m": (np.asarray(c.pos) - d.xpos[int(m.geom_bodyid[og])]).tolist(),
                    "eef_minus_soup_z_m": float(sp[2] - d.xpos[int(m.geom_bodyid[og])][2])})


CA.ContactRecorder._observe = _observe


def main():
    tag, t, init = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    prog = make_program(t)
    if len(sys.argv) > 4:
        gc = dict(zip(("yaw_deg", "lateral_m", "dz_m", "longitudinal_m"), map(float, sys.argv[4:8])))
        it0 = next(i for i in prog.instances if i.object_role == "alphabet_soup" and i.primitive == "approach")
        prog = replace(prog, instances=tuple(replace(i, params={**i.params, "grasp_candidate": gc}) if i.label == it0.label else i for i in prog.instances))
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    R = run(t, init, prog, grips)
    agg = defaultdict(list)
    for e in LOG:
        agg[(e["robot_body"], e["robot_geom"], e["soup_geom"])].append(e)
    rows = []
    for (rb, rg, og), v in sorted(agg.items()):
        f = np.array([x["force_n"] for x in v]); ks = sorted({x["k"] for x in v})
        rows.append({"robot_body": rb, "robot_geom": rg, "robot_geom_type": v[0]["robot_geom_type"], "soup_geom": og, "n_substep_contacts": len(v),
                     "control_steps": [ks[0], ks[-1], len(ks)], "max_force_n": round(float(f.max()), 2), "min_dist_m": round(min(x["dist_m"] for x in v), 5),
                     "mean_pos_in_hand_m": np.round(np.mean([x["pos_in_hand_m"] for x in v], 0), 4).tolist(),
                     "mean_pos_rel_soup_centre_m": np.round(np.mean([x["pos_rel_soup_centre_m"] for x in v], 0), 4).tolist(),
                     "mean_eef_minus_soup_z_m": round(float(np.mean([x["eef_minus_soup_z_m"] for x in v])), 4)})
    out = {"tag": tag, "task": t, "init": init, "candidate": None if len(sys.argv) <= 4 else gc, "task_ok": bool(R["ep"].task_ok), "status": R["ep"].status,
           "k_window": [K_LO, K_HI], "groups": rows, "n_logged": len(LOG)}
    p = ROOT / "reports" / "step8_grasp_candidates" / f"{tag}_T{t}_init{init}.contact_localize.json"
    json.dump(out, open(p, "w"), indent=1)
    for r in rows:
        print(json.dumps(r))
    print("task_ok", out["task_ok"], "status", out["status"], "n_logged", len(LOG))


main()
