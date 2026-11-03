"""Step 26 (P1): structured task context for LLM onboarding, read from the simulator and the BDDL problem only.

No per-task constant is written here.  For task t / init i the context holds: the instruction, the official goal
predicates and their exact semantics, every movable object (class, world pose, collision AABB), every goal target
(object or region site: pose, half size, the support surface below it), the executor capability registry, and the
grasp references available in the reference library for the involved object classes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRED_SEMANTICS = {  # transcribed from LIBERO object_states / site_object (the code the official success check runs)
    "on(obj, object)": "obj origin z >= target origin z AND obj in contact with target AND xy distance between the two ORIGINS < 0.03 m",
    "on(obj, region_site)": "d = site_R @ (obj_origin - site_pos): half_z - 0.005 < d_z < half_z + 0.10 AND |d_xy| < half_xy; "
                            "if the site has a parent body, obj must also touch that parent",
    "in(obj, region_site)": "obj origin strictly inside the site box: site_pos +- |site_R @ half_size| (lower z bound lowered by 0.01 m)",
}
ARTICULATED = ("open", "close", "turnon", "turnoff")


def _world_aabb(m, d, geoms):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        w = p + R @ c; e = np.abs(R) @ h
        lo, hi = np.minimum(lo, w - e), np.maximum(hi, w + e)
    return lo, hi


def _yaw(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def _wall_ring(m, d, gids, Rb):
    """Step 27: top ring of thin vertical wall panels (for grasp_wall_pinch), angles in the object's yaw frame."""
    from synthesis_v2.approach_planner import thin_wall_panels
    panels = thin_wall_panels(m, d, gids)
    if not panels:
        return None
    ztop = max(p["top_z"] for p in panels)
    ring = [p for p in panels if p["top_z"] > ztop - 0.005]
    cen = np.mean([p["centre"] for p in ring], axis=0)
    oyaw = np.arctan2(Rb[1, 0], Rb[0, 0])
    a = lambda v: round(float((np.degrees(np.arctan2(v[1], v[0]) - oyaw) + 180) % 360 - 180), 1)
    rad = float(np.mean([np.linalg.norm(p["centre"][:2] - cen[:2]) for p in ring]))
    other = [g for g in gids if g not in {p["gid"] for p in panels}
             and np.linalg.norm(d.geom_xpos[g][:2] - cen[:2]) > rad + 0.005]
    return {"ring_centre_world": cen.round(4).tolist(), "ring_radius_m": round(rad, 4), "top_z_world": round(ztop, 4),
            "panel_half_thickness_m": round(float(min(np.min(m.geom_size[p["gid"]]) for p in ring)), 4),
            "panels": [{"angle_deg_obj_frame": a(p["centre"][:2] - cen[:2]), "width_m": round(p["width_m"], 4)} for p in ring],
            "parts_outside_ring_angles_deg_obj_frame": sorted({a(d.geom_xpos[g][:2] - cen[:2]) for g in other})}


def task_context(t: int, init: int = 0, suite: str = "libero_10") -> dict:
    import mujoco
    from synthesis_v2.libero_backend import LiberoSession
    from synthesis_v2 import stage_capabilities as SC
    first = None
    s = LiberoSession(t, init, suite=suite, render=False, default_target="__probe__")
    pp = s.env.parsed_problem
    first = next(iter(s.env.objects_dict))
    s.close()
    s = LiberoSession(t, init, suite=suite, render=False, default_target=first); s.reset()
    m, d = s._mj()
    pp = s.env.parsed_problem
    cls = {n: c for c, ns in pp["objects"].items() for n in ns}
    objs = {}
    for n in s.env.objects_dict:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + "_main")
        lo, hi = _world_aabb(m, d, s.geom_ids_of(n))
        objs[n] = {"class": cls.get(n), "origin_world": d.xpos[bid].round(4).tolist(), "yaw_deg": round(_yaw(d.xmat[bid].reshape(3, 3)), 1),
                   "collision_aabb_world": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
                   "collision_size_m": (hi - lo).round(4).tolist(),
                   "origin_above_bottom_m": round(float(d.xpos[bid][2] - lo[2]), 4)}
        wall = _wall_ring(m, d, s.geom_ids_of(n), d.xmat[bid].reshape(3, 3))
        if wall:
            objs[n]["thin_wall_ring"] = wall
    goals, regions = [], {}
    sites = set(s.env.object_sites_dict) if hasattr(s.env, "object_sites_dict") else set()
    for g in pp["goal_state"]:
        pred, args = g[0].lower(), list(g[1:])
        goals.append({"predicate": pred, "args": args, "holds_at_init": bool(s.env._eval_predicate(g))})
        if pred == "on" and len(args) == 2 and args[1] in s.env.objects_dict:
            tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args[1] + "_main")
            goals[-1]["accepted_region_for_obj_origin"] = {"xy_within_m_of_target_origin": 0.03, "target_origin_world_now": d.xpos[tb].round(4).tolist(),
                                                           "z": "obj origin z >= target origin z, and obj must touch the target"}
        for a in args[1:]:
            if a in sites:
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, a)
                pos, R = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
                support = None
                for fx in list(getattr(s.env, "fixtures_dict", {})) + list(s.env.objects_dict):
                    gids = s.fixture_geom_ids(fx) if fx in getattr(s.env, "fixtures_dict", {}) else s.geom_ids_of(fx)
                    if not gids:
                        continue
                    lo, hi = _world_aabb(m, d, gids)
                    if lo[0] <= pos[0] <= hi[0] and lo[1] <= pos[1] <= hi[1] and fx not in [x for x in args[:1]]:
                        support = support or []
                        support.append({"body": fx, "aabb_top_z": round(float(hi[2]), 4), "aabb_min": lo.round(4).tolist(), "aabb_max": hi.round(4).tolist()})
                # surface directly below the site centre (ray cast down, any static or movable geom)
                gid = np.array([-1], dtype=np.int32)
                dist = mujoco.mj_ray(m, d, pos + [0, 0, 0.3], np.array([0, 0, -1.0]), None, 1, -1, gid)
                surf = None if dist < 0 else {"z": round(float(pos[2] + 0.3 - dist), 4),
                                              "geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))}
                hz = float(m.site_size[sid][2])
                accept = ({"origin_z_world_open_interval": [round(float(pos[2] + hz - 0.005), 4), round(float(pos[2] + hz + 0.10), 4)],
                           "origin_xy_world_box": [(pos[:2] - m.site_size[sid][:2]).round(4).tolist(), (pos[:2] + m.site_size[sid][:2]).round(4).tolist()],
                           "note": "on(): site-frame test; values here assume the site yaw shown"} if pred == "on" else
                          {"origin_box_world": [(pos - np.abs(R @ m.site_size[sid]) - [0, 0, 0.01]).round(4).tolist(),
                                                (pos + np.abs(R @ m.site_size[sid])).round(4).tolist()]})
                goals[-1]["accepted_region_for_obj_origin"] = accept
                regions[a] = {"kind": "region_site", "surface_below_site_center": surf, "pos_world": pos.round(4).tolist(), "yaw_deg": round(_yaw(R), 1),
                              "half_size_m": m.site_size[sid].round(4).tolist(), "bodies_under_site_xy": support}
    refs = json.loads((ROOT / "configs" / "grasp_references.v1.json").read_text())["entries"]
    involved = {cls.get(a) for g in goals for a in g["args"][:1]}
    avail = [{"reference_id": e["reference_id"], "object_class": e["object_class"], "compatible_objects": e["compatible_objects"],
              "kind": e["kind"], "value": e["value"]} for e in refs if e["object_class"] in involved]
    eef = np.asarray(s.env.sim.data.site_xpos[s.env.robots[0].eef_site_id]).round(4).tolist()
    s.close()
    return {"task": f"{suite}:{t}", "init": init, "instruction": pp["language_instruction"] if isinstance(pp["language_instruction"], str)
            else " ".join(pp["language_instruction"]),
            "goal": goals, "predicate_semantics": PRED_SEMANTICS,
            "unsupported_predicates_by_executor": list(ARTICULATED),
            "unsupported_note": "the executor cannot ACHIEVE these predicates; a goal whose holds_at_init is true needs no "
                                "action, it only has to stay true until the end",
            "objects": objs, "regions": regions, "fixtures": list(getattr(s.env, "fixtures_dict", {})),
            "eef_start_world": eef, "capabilities": SC.describe(), "grasp_references_available": avail}


if __name__ == "__main__":
    t = int(sys.argv[1]); i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ctx = task_context(t, i)
    out = ROOT / "reports" / "step26_llm_onboard" / "context" / f"t{t}_init{i}.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(ctx, indent=1, default=str))
    print(json.dumps({k: ctx[k] for k in ("instruction", "goal", "regions")}, default=str)[:3000])
    print({n: (o["class"], o["collision_size_m"]) for n, o in ctx["objects"].items()})
