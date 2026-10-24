"""Step 5: sub-step physical evidence around the transport-end hold dip.
Replays a recorded trajectory's exported actions and logs, at EVERY MuJoCo sub-step in a window:
finger-vs-object normal force (L/R), all other bodies touching the object (with force), object pose in the end-effector
frame, end-effector speed, gripper joint positions.  The control-step mean (what the audit uses) and the last-sub-step
force (what cf_L/cf_R record) are reported side by side so the two are never mixed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import OBJECTS, ROOT, make_program, make_session  # noqa: E402
from s3_libero_validate import hold_idx  # noqa: E402

D = ROOT / "reports" / "step4_libero_pairs_20261025_v1_frozen"
OUT = ROOT / "reports" / "step5_diag_20261025"


def diag(init: int, combo: str, role: str = "cream_cheese", before: int = 22, after: int = 4, src: Path = D) -> dict:
    import mujoco
    nm = dict(OBJECTS)[role]
    a = dict(np.load(src / f"gen_init{init}_{combo}.npz", allow_pickle=True))
    st = np.asarray(a["stage"])
    tr = np.flatnonzero(st == f"{role}.pick_1.transport")
    k_lo, k_hi = int(tr[-1]) - before, int(tr[-1]) + after
    prog = make_program(2)
    s = make_session(init, prog)
    s.reset()
    m, d = s._mj()
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    obj_b = body(s.env.objects_dict[nm].root_body)
    bas_b = body(s.env.objects_dict["basket_1"].root_body)
    eef_site = int(s.env.robots[0].eef_site_id)
    ids = set(s.geom_ids_of(nm))
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or str(g)  # noqa: E731
    rows: list[dict] = []
    cur = {"k": -1}
    orig = s.env.sim.step
    buf = np.zeros(6)

    def logged(*args, **kw):
        out = orig(*args, **kw)
        k = cur["k"]
        if k_lo <= k <= k_hi and s._binding is not None:
            fl = fr = 0.0
            nl = nr = 0
            touch: dict[str, float] = {}
            left, right = set(s._binding.left_finger_geom_ids), set(s._binding.right_finger_geom_ids)
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 not in ids and g2 not in ids:
                    continue
                o = g2 if g1 in ids else g1
                if o in ids:
                    continue
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if o in left:
                    fl += f; nl += 1
                elif o in right:
                    fr += f; nr += 1
                else:
                    touch[gname(o)] = touch.get(gname(o), 0.0) + f
            eef = d.site_xpos[eef_site].copy()
            R = d.site_xmat[eef_site].reshape(3, 3)
            rel = R.T @ (d.xpos[obj_b] - eef)
            Ro = d.xmat[obj_b].reshape(3, 3)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Ro) - 1) / 2, -1, 1))))
            rows.append({"k": k, "t": float(d.time), "fl": fl, "fr": fr, "nl": nl, "nr": nr, "touch": touch, "eef": eef.tolist(),
                         "obj": d.xpos[obj_b].tolist(), "basket": d.xpos[bas_b].tolist(), "rel_eef_frame": rel.tolist(), "rel_ang_deg": ang})
        return out

    s.env.sim.step = logged
    ft = a["force_target"]
    for k in range(k_hi + 1):
        cur["k"] = k
        n = str(ft[k])
        if n and n != s.force_target_name:
            s.set_force_target(n)
        s.step(np.asarray(a["actions"][k], np.float32))
    s.close()
    h = hold_idx(a, role)
    return {"init": init, "combo": combo, "window": [k_lo, k_hi], "transport_last": int(tr[-1]),
            "place_first": int(np.flatnonzero(st == f"{role}.pick_1.place")[0]),
            "audit_mean_total": {int(k): float(a["left_mean_n"][k] + a["right_mean_n"][k]) for k in range(k_lo, k_hi + 1)},
            "recorded_last_substep_total": {int(k): float(a["cf_L_" + nm][k] + a["cf_R_" + nm][k]) for k in range(k_lo, k_hi + 1)},
            "valid_fraction": {int(k): float(a["valid_fraction"][k]) for k in range(k_lo, k_hi + 1)},
            "arm_action": {int(k): [round(float(x), 3) for x in a["actions"][k][:7]] for k in range(k_lo, k_hi + 1)},
            "substeps": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["5:GF", "7:GF", "0:GF", "2:GF"])
    ap.add_argument("--src", default=str(D))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for c in a.cases:
        i, combo = c.split(":")
        r = diag(int(i), combo, src=Path(a.src))
        (OUT / f"diag_init{i}_{combo}{a.tag}.json").write_text(json.dumps(r))
        print("done", c, len(r["substeps"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
