"""Step 15 (2): offline check of the T5 target-region frame.  Rotation validity, axis meaning, local<->world round trip,
world-frame vs local-frame in-region test vs the environment's own SiteObject.in_box / In predicate, and where the book
ends up in the region frame in the committed runs.  usage: s15_frame_check.py  -> reports/step15_t5_place/frame_check.json"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as _R
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_session

SITE, BOOK, CADDY = "desk_caddy_1_back_contain_region", "black_book_1", "desk_caddy_1"
out = {"site": SITE}
prog = make_program(5)
s = make_session(5, 0, prog); s.reset(); m, d = s._mj()
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
p, R = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
site = s.env.object_sites_dict[SITE]
size = np.asarray(site.size, float)
out["origin_world"] = p.round(5).tolist(); out["R"] = R.round(4).tolist(); out["size_halfextent"] = size.tolist()
out["orthonormal_err"] = float(np.abs(R.T @ R - np.eye(3)).max()); out["det"] = float(np.linalg.det(R))
out["local_axes_in_world"] = {"x": R[:, 0].round(3).tolist(), "y": R[:, 1].round(3).tolist(), "z": R[:, 2].round(3).tolist()}
out["world_up_is_local"] = (R.T @ np.array([0, 0, 1.0])).round(3).tolist()
out["site_pose_api_equals_mujoco"] = bool(np.allclose(s.site_pose(SITE)[0], p) and np.allclose(s.site_pose(SITE)[1], R))
rng = np.random.default_rng(0); pts = p + rng.normal(0, 0.05, (200, 3))
loc = (R.T @ (pts - p).T).T; back = (R @ loc.T).T + p
out["roundtrip_max_err_m"] = float(np.abs(back - pts).max())
# the environment's own containment test on the BODY origin of the book, vs the local-frame rule on the same point
try:
    env_in = lambda q: bool(site.in_box(p, R, q))
except Exception as e:                                   # signature differs: fall back to the documented formula
    env_in = None; out["env_in_box_error"] = repr(e)
tot = np.abs(R @ size)
def mine_world(q):
    lb, ub = p - tot, p + tot; lb = lb.copy(); lb[2] -= 0.01
    return bool(np.all(q > lb) and np.all(q < ub))
def mine_local(q):                                       # same rule, in the region frame: |local| < size (z lower -0.01)
    l = R.T @ (q - p); return bool(np.all(np.abs(l[:2]) < size[:2]) and l[2] > -size[2] - 0.01 and l[2] < size[2])
cases = [p + rng.uniform(-1, 1, 3) * size * 1.4 for _ in range(500)]
out["in_box_agreement"] = {"n": len(cases), "world_rule_vs_local_rule": int(sum(mine_world(q) == mine_local(q) for q in cases)),
                           "env_vs_world_rule": None if env_in is None else int(sum(env_in(q) == mine_world(q) for q in cases))}
# book body-origin <-> collision geometry: the origin sits at the book's bottom face, the collision centre ~7 cm above it
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{BOOK}_main")
gc = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or g for g in range(m.ngeom) if m.geom_bodyid[g] == bb]
out["book_geom_centres_minus_body_origin_world"] = [(d.geom_xpos[g] - d.xpos[bb]).round(4).tolist() for g in range(m.ngeom) if m.geom_bodyid[g] == bb]
out["book_geom_half_sizes"] = [m.geom_size[g].round(4).tolist() for g in range(m.ngeom) if m.geom_bodyid[g] == bb]
# committed runs: final book body position in the region frame, both signs reported
runs = {}
for tag in ("d_demo0_ref", "d_demo0_ref_yawfix"):
    for i in range(4):
        j = json.load(open(ROOT / "reports/step14_t5_onboard" / tag / f"init{i}.json"))
        runs[f"{tag}/init{i}"] = {k: v for k, v in j.items() if k.startswith("book_final") or k in ("final_in_region", "env_success")}
out["committed_run_fields"] = runs
# demo: book position in the region frame as stored by the demo extract (success moment) vs recomputed here with R^T (p - origin)
demo = {}
for n in (0, 2, 4):
    j = json.load(open(ROOT / f"reports/step14_t5_onboard/demo_replay/demo_{n}.json"))
    sm = j["success_moment"]; sw = np.asarray(j["site"]["pos"], float)
    demo[f"demo_{n}"] = {"stored_book_pos_in_site_frame": sm["book_pos_in_site_frame"],
                         "recomputed_with_this_R": (R.T @ (np.asarray(sm["book_pos"]) - sw)).round(4).tolist()}
out["demo_success_moment_check"] = demo
# book geometry in the BODY frame (so 'book position' is not confused with 'book geometric centre')
Rb = d.xmat[bb].reshape(3, 3)
out["book_geoms_in_body_frame"] = [{"geom": int(g), "centre": (Rb.T @ (d.geom_xpos[g] - d.xpos[bb])).round(4).tolist(),
                                    "half_size": m.geom_size[g].round(4).tolist(), "geom_axes_in_body": (Rb.T @ d.geom_xmat[g].reshape(3, 3)).round(2).tolist()}
                                   for g in range(m.ngeom) if m.geom_bodyid[g] == bb and (m.geom_contype[g] or m.geom_conaffinity[g])]
OUT = ROOT / "reports" / "step15_t5_place"; OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "frame_check.json", "w"), indent=1)
for k in ("origin_world", "R", "size_halfextent", "orthonormal_err", "det", "local_axes_in_world", "world_up_is_local", "site_pose_api_equals_mujoco",
          "roundtrip_max_err_m", "in_box_agreement", "env_in_box_error", "book_geom_centres_minus_body_origin_world", "book_geom_half_sizes"):
    print("FC", k, out.get(k))
for k, v in runs.items(): print("FC", k, v)
print("FC demo", json.dumps(demo)[:900]); print("FC book_geoms_body", json.dumps(out["book_geoms_in_body_frame"]))
