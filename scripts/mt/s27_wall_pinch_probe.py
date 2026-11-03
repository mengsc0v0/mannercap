"""Step 27 probe (geometry only, no stepping): single-panel wall-pinch candidates for a thin-walled object.
For each thin wall panel of OBJ (box geom, smallest half-size <= 4 mm, horizontal normal) the eef site is put on the
panel centre line, closing axis (hand finger-row axis) parallel to the panel normal, at depth D below the panel top.
Returns planner-convention (yaw_deg, lateral_m, dz_m, longitudinal_m) and plan_approach feasibility.
Usage: s27_wall_pinch_probe.py TASK INIT OBJ [DEPTH_M ...]
"""
from __future__ import annotations

import json
import sys

import mujoco
import numpy as np


def main() -> int:
    t, init, obj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    depths = [float(x) for x in sys.argv[4:]] or [0.01, 0.015, 0.02]
    from synthesis_v2.approach_planner import plan_approach, quat_to_mat, scene_from_libero, yaw_quat
    from synthesis_v2.libero_backend import LiberoSession
    from synthesis_v2.native_producer import _yaw_aligned_quat
    s = LiberoSession(t, init, render=False, default_target=obj); s.reset()
    down = np.asarray(s.obs["robot0_eef_quat"], float)
    m_, d_, site_, env_, obst_, tgt_, _ = scene_from_libero(s, obj, return_steps=True)
    m, d = s._mj()
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj + "_main")
    opos = np.asarray(s.obs[f"{obj}_pos"], float)
    gq = _yaw_aligned_quat(down, np.asarray(s.obs[f"{obj}_quat"], float))
    # eef site vs finger collision geoms (current pose, hand pointing down)
    fing = [g for g in env_.gids if "finger" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    site_z = float(d.site_xpos[site_][2])
    tip_z = min(float(d.geom_xpos[g][2] - np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g] @ [0, 0, 1]) for g in fing) \
        if fing else None
    panels = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        sz = m.geom_size[g]; Rg = d.geom_xmat[g].reshape(3, 3); k = int(np.argmin(sz))
        n = Rg[:, k]
        if sz[k] > 0.004 or abs(n[2]) > 0.2:
            continue
        top = float(d.geom_xpos[g][2] + np.abs(Rg[2]) @ sz)
        panels.append({"geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "centre": d.geom_xpos[g].copy(),
                       "normal": n / np.linalg.norm(n[:2] + 1e-12), "half": sz.tolist(), "top_z": top,
                       "width_m": float(2 * sorted(sz)[1])})
    cen = np.mean([p["centre"] for p in panels], axis=0) if panels else opos
    out = {"task": t, "init": init, "obj": obj, "obj_pos": opos.tolist(), "site_z": site_z, "finger_tip_z": tip_z,
           "site_above_tip_m": None if tip_z is None else site_z - tip_z, "ring_centre": cen.tolist(), "rows": []}
    yaws = np.arange(-180.0, 180.0, 0.1)
    rows1 = [quat_to_mat(yaw_quat(gq, np.deg2rad(y))) @ env_.obs_to_site for y in yaws]
    for p in sorted(panels, key=lambda p: p["top_z"], reverse=True):
        nh = np.array([p["normal"][0], p["normal"][1], 0.0]); nh /= np.linalg.norm(nh)
        k = int(np.argmax([abs(R[:, 1] @ nh) for R in rows1]))
        y, R = float(yaws[k]), rows1[k]
        ang = float(np.degrees(np.arctan2(*(p["centre"][:2] - cen[:2])[::-1])))
        for D in depths:
            tgt = np.array([p["centre"][0], p["centre"][1], p["top_z"] - D + (out["site_above_tip_m"] or 0.0)])
            lon, lat, dz = np.linalg.solve(np.stack([R[:, 0], R[:, 1], [0, 0, 1.0]], axis=1), tgt - opos)
            pl = plan_approach(m_, d_, site_, env_, obst_, start_pos=np.asarray(s.obs["robot0_eef_pos"], float),
                               start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"], float), grasp_pos=opos,
                               grasp_quat_xyzw=gq, hover_dz=0.10, target=tgt_, grid=[(y, float(lat), float(dz), float(lon))])
            c = pl.candidates[0] if pl.candidates else None
            out["rows"].append({"geom": p["geom"], "angle_from_ring_centre_deg": round(ang, 1), "top_z": round(p["top_z"], 4),
                                "width_m": round(p["width_m"], 4), "depth_m": D, "yaw_deg": round(y, 2),
                                "lateral_m": round(float(lat), 5), "dz_m": round(float(dz), 5), "longitudinal_m": round(float(lon), 5),
                                "cos_closing_normal": round(float(abs(R[:, 1] @ nh)), 4),
                                "feasible": bool(c and c.feasible), "min_clearance_m": None if c is None else round(c.min_clearance_m, 4),
                                "blocking": None if c is None else c.blocking, "phase": None if c is None else c.phase})
    s.close()
    print(json.dumps(out, indent=0, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
