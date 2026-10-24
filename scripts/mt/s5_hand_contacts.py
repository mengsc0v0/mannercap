"""Step 5: for each trajectory, every control step in which any robot/gripper body touches a non-target scene body
(table, basket, other objects) or touches the carried object with something other than the finger tips.
Reported per step: stage, body pair, max normal force over the physical sub-steps.  Replay-only (no re-generation)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import ROOT, make_program, make_session  # noqa: E402

SRC = ROOT / "reports" / sys.argv[1]
OUT = ROOT / "reports" / "step5_diag_20261025"; OUT.mkdir(exist_ok=True, parents=True)
import mujoco
for case in sys.argv[2:]:
    i, combo = case.split(":"); i = int(i)
    a = dict(np.load(SRC / f"gen_init{i}_{combo}.npz", allow_pickle=True))
    s = make_session(i, make_program(2)); s.reset(); m, d = s._mj()
    nb = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
    isr = lambda b: b.startswith(("robot0", "gripper0"))  # noqa: E731
    cur = {"k": -1}; ev = {}; orig = s.env.sim.step; buf = np.zeros(6)

    def st(*x, **kw):
        o = orig(*x, **kw)
        for ci in range(int(d.ncon)):
            c = d.contact[ci]; b1, b2 = nb(int(c.geom1)), nb(int(c.geom2))
            if isr(b1) == isr(b2):
                continue
            rb, ob = (b1, b2) if isr(b1) else (b2, b1)
            if "finger_joint" in rb and "tip" in rb and ob == f"{s.force_target_name}_main".replace("_1_1", "_1"):
                continue
            mujoco.mj_contactForce(m, d, ci, buf)
            f = abs(float(buf[0]))
            if f <= 0.0:
                continue
            v = ev.setdefault((cur["k"], rb, ob), [0.0, 0]); v[0] = max(v[0], f); v[1] += 1
        return o
    s.env.sim.step = st
    for k in range(len(a["actions"])):
        cur["k"] = k; n = str(a["force_target"][k])
        if n and n != s.force_target_name: s.set_force_target(n)
        s.step(np.asarray(a["actions"][k], np.float32))
    rows = [{"k": k, "stage": str(a["stage"][k]), "robot_body": rb, "other_body": ob, "max_n": round(v[0], 2), "n_contacts": v[1]}
            for (k, rb, ob), v in sorted(ev.items())]
    (OUT / f"handc_init{i}_{combo}{__import__('os').environ.get('TAG', '')}.json").write_text(json.dumps(rows))
    print("done", case, len(rows), flush=True)
