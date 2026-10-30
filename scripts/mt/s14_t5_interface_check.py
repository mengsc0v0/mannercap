"""Step 14 (B) acceptance: the shared target-region / fixture interface resolves T5's target consistently with the
environment's own success condition, the planner and the execution guard see the caddy, and the site is not an obstacle.
Usage: s14_t5_interface_check.py -> reports/step14_t5_onboard/interface_check.json (no episode is run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s7_task_config as C  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from synthesis_v2.approach_planner import scene_from_libero  # noqa: E402
from synthesis_v2.execution_guard import pre_close_check  # noqa: E402
from synthesis_v2.program_executor import _Geometry  # noqa: E402

OUT = Path("reports/step14_t5_onboard"); OUT.mkdir(parents=True, exist_ok=True)
JOINT = "black_book_1_joint0"


def put_book(s, pos, yaw):
    q = R.from_euler("z", yaw).as_quat()  # xyzw
    s.env.sim.data.set_joint_qpos(JOINT, np.concatenate([pos, [q[3], q[0], q[1], q[2]]]))
    s.env.sim.data.set_joint_qvel(JOINT, np.zeros(6))
    s.env.sim.forward()


res = []
for init in range(4):
    prog = C.make_program(5)
    s = C.make_session(5, init, prog); s.reset()
    meta = {"down_quat_xyzw": [1.0, 0.0, 0.0, 0.0], "subtasks": {}}
    g = _Geometry(s, prog, True, meta)
    rec = g.capture("book.pick_1", "book", "caddy_back")
    tgt = np.asarray(rec["target_pos0"]); delta = rec["target_yaw_delta_rad"]
    y0 = R.from_quat(s.obs["black_book_1_quat"]).as_euler("xyz")[2]
    pos_site, mat = s.site_pose("desk_caddy_1_back_contain_region")
    checks = {"initial_pose_success": bool(s.success())}
    put_book(s, tgt, y0 + delta); checks["resolved_target_pose_success"] = bool(s.success())
    for nm, d in (("x+4cm", [0.04, 0, 0]), ("y+8cm", [0, 0.08, 0]), ("z-12cm", [0, 0, -0.12]), ("z+12cm", [0, 0, 0.12])):
        put_book(s, tgt + np.array(d), y0 + delta); checks[f"displaced_{nm}_success"] = bool(s.success())
    s.reset()
    m, d_, site_, env_, obst_, tgt_ = scene_from_libero(s, "black_book_1")
    names = [o.name for o in obst_]
    chk = pre_close_check(s, "black_book_1")
    res.append({"init": init, "target_pos0": tgt.round(4).tolist(), "site_centre": pos_site.round(4).tolist(),
                "target_yaw_delta_deg": round(float(np.degrees(delta)), 1), "book_yaw_deg": round(float(np.degrees(y0)), 1),
                "success_checks": checks, "planner_obstacles": names, "planner_sees_caddy": "desk_caddy_1" in names,
                "guard_obstacle_keys": sorted(k for k in chk if "desk_caddy" in str(k)) or sorted(chk.keys())[:8],
                "guard_mentions_caddy": "desk_caddy_1" in json.dumps(chk, default=str),
                "site_is_obstacle": any("region" in n for n in names)})
    s.close()
ok = all(r["success_checks"]["resolved_target_pose_success"] and not r["success_checks"]["initial_pose_success"]
         and not any(v for k, v in r["success_checks"].items() if k.startswith("displaced")) and r["planner_sees_caddy"]
         and r["guard_mentions_caddy"] and not r["site_is_obstacle"] for r in res)
(OUT / "interface_check.json").write_text(json.dumps({"schema": "mannercap.t5_interface_check.v1", "all_pass": ok, "inits": res}, indent=1))
print(json.dumps({"all_pass": ok, "inits": res}, indent=1)[:3500])
