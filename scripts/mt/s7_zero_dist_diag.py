"""One-scene diagnosis of mj_geomDistance == 0.0 in the approach planner (dev, read-only).
Usage: s7_zero_dist_diag.py TASK INIT DZ_CM   -> first pick only, lateral 0, single candidate.
For every (hand geom, object geom) query that returned exactly 0.0 at the at-grasp check: unrounded raw results for several
distmax, nearest points, bounding-sphere gap, and whether mj_collision finds a contact for the same pair (penetration depth)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
import synthesis_v2.approach_planner as ap
from s7_task_config import make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode

T, I, DZ = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) / 100
ap.candidate_grid = lambda *a, **k: [(0, 0.0, DZ)]
nm = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
bn = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"
state = {"n": 0, "ctx": None, "rows": [], "calls": 0}
orig_plan, orig_clr, orig_dist = ap.plan_approach, ap._clearance, ap._dist

def plan(*a, **k):
    state["n"] += 1; state["calls"] = 0
    return orig_plan(*a, **k)
def clr(m, mj, scratch, env, pos, R, obstacles, reach):
    state["ctx"] = ("at_grasp" if (len(obstacles) == 1 and abs(reach) == 0.0 and state.get("grasp_call")) else "path", pos.copy(), [o.name for o in obstacles])
    return orig_clr(m, mj, scratch, env, pos, R, obstacles, reach)
def dist(mj, m, data, g, og, fromto):
    r = orig_dist(mj, m, data, g, og, fromto)
    if state["n"] == 1 and r == 0.0 and len(state["rows"]) < 40:
        raw = {}
        for dm in (0.05, 0.035, 0.065, 0.2, 0.01, 0.002):
            ft = np.zeros(6); raw[dm] = (float(mj.mj_geomDistance(m, data, g, og, dm, ft)), np.round(ft, 5).tolist())
        mj.mj_collision(m, data)                  # contacts for the re-posed hand (geom_xpos was overwritten, not kinematics)
        cts = [(round(float(c.dist), 6), np.round(c.pos, 4).tolist()) for c in data.contact[:data.ncon] if {int(c.geom1), int(c.geom2)} == {g, og}]
        state["rows"].append({"ctx": state["ctx"][0] if state["ctx"] else None, "at": np.round(state["ctx"][1], 4).tolist() if state["ctx"] else None,
                              "obstacles": state["ctx"][2] if state["ctx"] else None,
                              "hand_geom": f"{bn(m, g)}/{nm(m, g)} type{int(m.geom_type[g])}", "obj_geom": f"{bn(m, og)}/{nm(m, og)} type{int(m.geom_type[og])}",
                              "centre_gap": round(float(np.linalg.norm(data.geom_xpos[g] - data.geom_xpos[og])), 5),
                              "rbound_sum": round(float(m.geom_rbound[g] + m.geom_rbound[og]), 5), "raw_by_distmax": raw, "mj_collision_contacts": cts})
    return r
ap.plan_approach, ap._clearance, ap._dist = plan, clr, dist
import synthesis_v2.program_executor as pe
prog = make_program(T); s = make_session(T, I, prog)
print("mujoco", mujoco.__version__, "ccd_iterations", s._mj()[0].opt.ccd_iterations, "ccd_tolerance", s._mj()[0].opt.ccd_tolerance, "noslip", s._mj()[0].opt.noslip_iterations)
ep = produce_program_episode(prog, I, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
print("status", ep.status, "task_ok", ep.task_ok)
first = list(ep.meta["approach_plan"].items())[0]
print("first plan", first[0], {k: v for k, v in first[1]["candidates"][0].items() if k not in ("offset",)})
print("n zero-distance queries logged (first plan only):", len(state["rows"]))
for r in state["rows"][:12]: print(json.dumps(r))
Path("reports/step7_onboard_diag").mkdir(parents=True, exist_ok=True)
json.dump(state["rows"], open(f"reports/step7_onboard_diag/zero_dist_t{T}_init{I}_dz{int(DZ*100)}.json", "w"), indent=1)
