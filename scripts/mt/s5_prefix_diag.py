"""Step 5 close-out: why does the butter grasp transient depend on the cream-cheese level (prefix) in init16?
Replays a recorded trajectory; at every physical sub-step of the butter grasp window logs per-finger force on the butter,
the SAME-SUB-STEP L+R sum (vs the control-step sum-of-maxima the audit uses), finger/hand contacts with any other body, the
butter pose, the eef pose and the gripper opening.  At the start of the butter approach it also stores the pose of every
object in the scene and the controller-side per-instance bookkeeping (force target, hold band, grasp commands)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import ROOT, make_program, make_session  # noqa: E402

SRC = ROOT / "reports" / os.environ.get("SRC", "step5_libero_pairs_v2")
OUT = ROOT / "reports" / "step5_diag_20261025"
import mujoco

def run(init, combo, pre=3, post=14):
    a = dict(np.load(SRC / f"gen_init{init}_{combo}.npz", allow_pickle=True)); st = np.asarray(a["stage"])
    g = np.flatnonzero(st == "butter.pick_1.grasp"); g0 = int(g[0]); ap0 = int(np.flatnonzero(st == "butter.pick_1.approach")[0])
    k_lo, k_hi = g0 - pre, g0 + post
    s = make_session(init, make_program(2)); s.reset(); m, d = s._mj()
    bn = lambda gid: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) or "?"
    bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s.env.objects_dict["butter_1"].root_body)
    site = int(s.env.robots[0].eef_site_id)
    free = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.root_body) for n, o in s.env.objects_dict.items()}
    cur = {"k": -1}; rows = []; scene = {}; orig = s.env.sim.step; buf = np.zeros(6)
    def logged(*x, **kw):
        o = orig(*x, **kw)
        if k_lo <= cur["k"] <= k_hi and s.force_target_name == "butter_1" and s._binding is not None:
            ids = set(s.geom_ids_of("butter_1")); left, right = set(s._binding.left_finger_geom_ids), set(s._binding.right_finger_geom_ids)
            fl = fr = 0.0; other = {}
            for ci in range(int(d.ncon)):
                c = d.contact[ci]; g1, g2 = int(c.geom1), int(c.geom2)
                f_side = None
                for gg, oo in ((g1, g2), (g2, g1)):
                    if gg in left or gg in right:
                        f_side = ("L" if gg in left else "R", oo); break
                    if bn(gg).startswith("gripper0"):
                        f_side = ("H", oo); break
                if f_side is None: continue
                mujoco.mj_contactForce(m, d, ci, buf); f = abs(float(buf[0]))
                side, oo = f_side
                if oo in ids:
                    if side == "L": fl += f
                    elif side == "R": fr += f
                else:
                    nm = bn(oo); other[nm] = max(other.get(nm, 0.0), f)
            rows.append({"k": cur["k"], "fl": fl, "fr": fr, "sum": fl + fr, "other": other,
                         "butter": d.xpos[bb].tolist(), "butter_quat": d.xquat[bb].tolist(), "eef": d.site_xpos[site].tolist(),
                         "gripper_q": [float(v) for v in s.obs["robot0_gripper_qpos"]]})
        return o
    s.env.sim.step = logged
    ft = a["force_target"]
    for k in range(k_hi + 1):
        if k == ap0:
            scene = {n: d.xpos[i].tolist() for n, i in free.items()}
            scene["_eef"] = d.site_xpos[site].tolist()
        cur["k"] = k
        n = str(ft[k])
        if n and n != s.force_target_name: s.set_force_target(n)
        s.step(np.asarray(a["actions"][k], np.float32))
    s.close()
    per_k = {}
    for r in rows: per_k.setdefault(r["k"], []).append(r)
    summ = {int(k): {"max_sum_same_substep": max(x["sum"] for x in v), "max_L": max(x["fl"] for x in v), "max_R": max(x["fr"] for x in v),
                     "sum_of_maxima": max(x["fl"] for x in v) + max(x["fr"] for x in v),
                     "other_contacts": {b: round(f, 1) for x in v for b, f in x["other"].items() if f > 0.5},
                     "butter_xyz": np.round(v[-1]["butter"], 4).tolist(), "gripper_q": np.round(v[-1]["gripper_q"], 4).tolist()} for k, v in per_k.items()}
    ctl = {int(k): {"cmd": float(a["actions"][k][6]), "audit_L_max": float(a["left_max_n"][k]), "audit_R_max": float(a["right_max_n"][k]),
                    "hold_band": [float(a["hold_lo_n"][k]), float(a["hold_hi_n"][k])] if "hold_lo_n" in a else None,
                    "target": str(ft[k]), "hold_instance": str(a["hold_instance"][k]) if "hold_instance" in a else None} for k in range(k_lo, k_hi + 1)}
    return {"init": init, "combo": combo, "g0": g0, "approach_start": ap0, "scene_at_butter_approach_start": scene, "substep_summary": summ, "control": ctl}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[1:]:
        i, combo = c.split(":")
        r = run(int(i), combo)
        (OUT / f"prefix_init{i}_{combo}.json").write_text(json.dumps(r)); print("done", c, flush=True)
