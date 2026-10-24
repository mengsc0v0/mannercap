"""Step 5: list EVERY active contact pair (by body name) per control step near the dip, to test whether anything other
than the two fingers and the carried object touches the hand/object (e.g. arm link or other scene object)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import ROOT, make_program, make_session  # noqa: E402

D = ROOT / "reports" / "step4_libero_pairs_20261025_v1_frozen"
init, combo, lo, hi = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
import mujoco
a = dict(np.load(D / f"gen_init{init}_{combo}.npz", allow_pickle=True))
s = make_session(init, make_program(2)); s.reset(); m, d = s._mj()
nb = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
cur = {"k": -1}; res = {}; orig = s.env.sim.step; buf = np.zeros(6)

def st(*x, **kw):
    o = orig(*x, **kw)
    if lo <= cur["k"] <= hi:
        for ci in range(int(d.ncon)):
            c = d.contact[ci]; b1, b2 = nb(int(c.geom1)), nb(int(c.geom2))
            if "robot0_link" in b1 + b2 and "robot0_link" in (b1 if "robot0_link" in b1 else b2) and ("gripper" in b1 and "gripper" in b2):
                pass
            mujoco.mj_contactForce(m, d, ci, buf)
            key = " | ".join(sorted((b1, b2)))
            r = res.setdefault(cur["k"], {}); v = r.setdefault(key, [0, 0.0]); v[0] += 1; v[1] = max(v[1], abs(float(buf[0])))
    return o
s.env.sim.step = st
for k in range(hi + 1):
    cur["k"] = k; n = str(a["force_target"][k])
    if n and n != s.force_target_name: s.set_force_target(n)
    s.step(np.asarray(a["actions"][k], np.float32))
for k in sorted(res):
    print("k", k)
    for key, (n, f) in sorted(res[k].items()):
        print("   ", key, "substeps", n, "maxN %.1f" % f)
