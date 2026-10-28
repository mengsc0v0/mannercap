"""Step 11.1b: static identity dump of every gripper collision geom, to decide whether the ``right_gripper`` contacts
reported by the audit are a CLASSIFICATION error (a finger geom whose body is not named *finger) or a REAL contact of a
non-finger body.  Model inspection only: the session is built and closed without stepping the simulator.

Usage: s11_gripper_geoms.py TASK INIT  ->  reports/step11_contact/gripper_geoms_t{T}.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, make_program, make_session  # noqa: E402
from synthesis_v2.grasp_force_contract import classify_finger_geom  # noqa: E402

OUT = ROOT / "reports" / "step11_contact"


def main() -> int:
    task, init = int(sys.argv[1]), int(sys.argv[2])
    import mujoco
    prog = make_program(task)
    s = make_session(task, init, prog)
    m, _ = s._mj()
    rows = []
    for gi in range(int(m.ngeom)):
        g = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        if not (g.startswith("gripper0") or b.startswith("gripper0")):
            continue
        r = classify_finger_geom(g, b)
        rows.append({"geom_id": gi, "geom": g, "body": b, "contype": int(m.geom_contype[gi]),
                     "conaffinity": int(m.geom_conaffinity[gi]), "group": int(m.geom_group[gi]),
                     "classified_as": None if r is None else {"side": r[0], "role": r[1]},
                     "audit_label_if_contacted": b.replace("gripper0_", "")})
    s.close()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"schema": "mannercap.gripper_geom_identity.v1", "task": task, "init": init,
           "n_gripper_geoms": len(rows), "geoms": rows,
           "note": "contype/conaffinity 0 means the geom cannot generate contacts (visual only)."}
    (OUT / f"gripper_geoms_t{task}.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        if r["contype"] == 0 and r["conaffinity"] == 0:
            continue
        print(f"  geom={r['geom']:44s} body={r['body']:28s} label={r['audit_label_if_contacted']:20s} "
              f"class={r['classified_as']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
