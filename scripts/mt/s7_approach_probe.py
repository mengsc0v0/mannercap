"""Step 7.0 probe: whole-hand-envelope check of the executor's first approach+descent (planning only; the look-ahead
restores the session).  Usage: s7_approach_probe.py INIT... ; prints default vs chosen candidate."""
import json, sys
sys.path.insert(0, "scripts/mt")
import numpy as np
from s3_libero_task import make_program, make_session
from synthesis_v2.program_executor import _Geometry, _subtask_roles, neutral_program_plan, APPROACH_DZ
import os
from synthesis_v2.approach_planner import plan_approach, scene_from_libero, candidate_grid
GRID = candidate_grid(laterals_m=(0, .01, -.01, .02, -.02, .03, -.03, .04, -.04), dzs_m=(0, .01, .02, .03)) if os.environ.get('EXT') else None

out = {}
for init in [int(x) for x in sys.argv[1:]] or [17]:
    prog = make_program(2)
    s = make_session(init, prog); obs = s.reset()
    meta = {"down_quat_xyzw": np.asarray(obs["robot0_eef_quat"], float).tolist(), "subtasks": {}}
    first = prog.instances[0]
    geo = _Geometry(s, prog, True, meta)
    rec = geo.capture(first.subtask, *_subtask_roles(prog, first.subtask))
    nm = prog.objects[first.object_role]
    m, d, site, env, obs_, tgt = scene_from_libero(s, nm, settled=True)
    pos = np.asarray(s.obs["robot0_eef_pos"], float); quat0 = np.asarray(s.obs["robot0_eef_quat"], float)
    pl = plan_approach(m, d, site, env, obs_, start_pos=pos, start_quat_xyzw=quat0,
                       grasp_pos=np.asarray(rec["obj_pos0"], float), grasp_quat_xyzw=np.asarray(rec["grasp_quat_xyzw"], float),
                       hover_dz=APPROACH_DZ, target=tgt, grid=GRID)
    r = pl.to_dict(); out[init] = r
    c0 = r["candidates"][0]; ch = r["chosen"]
    print(f"init{init} {nm}: default clr={c0['min_clearance_m']:.4f} ({c0['blocking']},{c0['phase']}) feasible={c0['feasible']}"
          f" | chosen yaw={ch and ch["yaw_deg"]} lat={ch and ch["lateral_m"]} dz={ch and ch["dz_m"]} clr={ch and round(ch['min_clearance_m'],4)}"
          f" | n_cand={len(r['candidates'])}", flush=True)
json.dump(out, open("/tmp/s7_probe.json", "w"), indent=1)
