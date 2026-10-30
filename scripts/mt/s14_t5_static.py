"""T5 (book -> back compartment of desk caddy) static pre-check: geometry, masses, init spread, goal predicate.
No policy runs; only scene reset (+ SETTLE_STEPS idle).  usage: s14_t5_static.py INIT... -> reports/step14_t5_precheck/static.json"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT
from synthesis_v2.libero_backend import LiberoSession

OUT = ROOT / "reports" / "step14_t5_precheck"; OUT.mkdir(parents=True, exist_ok=True)
inits = [int(x) for x in sys.argv[1:]]
rows = []
for i in inits:
    s = LiberoSession(5, i, render=False, default_target="black_book_1"); o = s.reset()
    m, d = s._mj()
    rec = {"init": i}
    for nm in ("black_book_1", "desk_caddy_1", "white_yellow_mug_1"):
        obj = s.env.objects_dict.get(nm) or s.env.fixtures_dict.get(nm)
        gids = s.geom_ids_of(nm)
        # AABB of the contact geoms in the world frame
        lo = np.full(3, 1e9); hi = np.full(3, -1e9)
        for g in gids:
            sz = m.geom_size[g]; R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX: ext = np.abs(R) @ sz
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[a]:m.mesh_vertadr[a] + m.mesh_vertnum[a]]
                w = (v - m.mesh_pos[a]) @ R.T + p if False else v @ R.T + p  # mesh verts are in the mesh-local frame
                lo = np.minimum(lo, w.min(0)); hi = np.maximum(hi, w.max(0)); continue
            else: ext = np.full(3, sz[0])
            lo = np.minimum(lo, p - ext); hi = np.maximum(hi, p + ext)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm + "_main")
        w, x, y, z = d.xquat[bid]
        yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))))
        rec[nm] = {"pos": d.xpos[bid].round(4).tolist(), "yaw_deg": round(yaw, 1),
                   "aabb_size_m": (hi - lo).round(4).tolist(), "n_geoms": len(gids),
                   "mass_kg": round(float(m.body_subtreemass[bid]), 4) if bid >= 0 else None}
    # goal region (site) in world frame
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "desk_caddy_1_back_contain_region")
    rec["back_site"] = {"center": d.site_xpos[sid].round(4).tolist(), "half_size": m.site_size[sid].round(4).tolist(),
                        "xmat": d.site_xmat[sid].reshape(3, 3).round(3).tolist()}
    rec["book_pos_minus_site_xy"] = (d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "black_book_1_main")][:2] - d.site_xpos[sid][:2]).round(4).tolist()
    rec["success_at_init"] = s.success()
    rows.append(rec); s.close()
    print(i, rec["black_book_1"], rec["desk_caddy_1"]["pos"], rec["back_site"]["center"], flush=True)
(OUT / "static.json").write_text(json.dumps(rows, indent=1))
