"""Step 8: physical-identity check of contact-policy violations.

Per physics substep, every robot-vs-movable-object contact above THR is listed with the geom/body names on both sides,
the object the force binding currently measures (``force_target_name``), and three independent verdicts:
  audit_allowed   the contact_audit rule: robot BODY name has 'finger_joint' and 'tip', and the other body is the current force target
  force_bound     the robot GEOM is one of the geoms the force measurement counts as left/right finger (resolve_grasp_binding of the
                  current force target)
  on_target       the other body is the current force target
Output: reports/step8_contact_identity/T{t}_init{i}_{role}_{A|B}.json (aggregated by geom pair x stage) + a text table.

  s8_contact_identity.py TASK INIT ROLE [DZ]     (A = original program, B = soup-only transport_dz candidate)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from s8_task_force import EXECUTOR, PLANS, grip_for  # noqa: E402
from s8_transport_repair import with_transport_dz  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

THR = 0.1


def run(t, init, prog):
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    s = make_session(t, init, prog)
    rec = collections.defaultdict(lambda: {"n_substeps": 0, "max_n": 0.0, "ctrl": set(), "off_pad_mm": 1e9})
    share = []  # per substep: ctrl, target-force by (L pad, L finger body, R pad, R finger body)
    cache = {}
    orig = s._hook

    def hook():
        orig()
        m, d = s._mj()
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"  # noqa: E731
        bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"  # noqa: E731
        is_robot = lambda b: b.startswith(("robot0", "gripper0"))  # noqa: E731
        prev = s.env.sim.step
        buf = np.zeros(6)

        def bind(name):
            if name not in cache:
                cache[name] = resolve_grasp_binding(m, mujoco, set(s.geom_ids_of(name)), name)
            return cache[name]

        pad_gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_finger{n}_pad_collision") for n in (1, 2)}

        def off_pad_mm(rg, pos):
            """distance (mm) of the contact point outside the same finger's pad box (0 = inside the pad box)."""
            for n in (1, 2):
                if gn(rg) == f"gripper0_finger{n}_collision":
                    g = pad_gid[n]
                    loc = np.asarray(d.geom_xmat[g]).reshape(3, 3).T @ (np.asarray(pos) - np.asarray(d.geom_xpos[g]))
                    return float(np.linalg.norm(np.maximum(np.abs(loc) - np.asarray(m.geom_size[g]), 0.0)) * 1000)
            return None

        def stepped(*a, **k):
            out = prev(*a, **k)
            tgt = s.force_target_name
            acc = [0.0, 0.0, 0.0, 0.0]
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                b1, b2 = bn(g1), bn(g2)
                if is_robot(b1) == is_robot(b2):
                    continue
                rg, og = (g1, g2) if is_robot(b1) else (g2, g1)
                rb, ob = bn(rg), bn(og)
                if ob.startswith(("living_room_table", "floor")):
                    continue
                mujoco.mj_contactForce(m, d, ci, buf)
                f = abs(float(buf[0]))
                if tgt and ob == f"{tgt}_main".replace("_1_1", "_1"):
                    nm = gn(rg)
                    if nm.endswith("_pad_collision"):
                        acc["12".index(nm[len("gripper0_finger")]) * 2] += f
                    elif nm in ("gripper0_finger1_collision", "gripper0_finger2_collision"):
                        acc["12".index(nm[len("gripper0_finger")]) * 2 + 1] += f
                if f <= THR:
                    continue
                side = "-"
                if tgt:
                    b = bind(tgt)
                    side = "L" if rg in b.left_finger_geom_ids else "R" if rg in b.right_finger_geom_ids else "-"
                tb = f"{tgt}_main".replace("_1_1", "_1") if tgt else None
                key = (gn(rg), rb.replace("gripper0_", ""), gn(og), ob, tgt, side,
                       bool("finger_joint" in rb and "tip" in rb and ob == tb), bool(ob == tb))
                v = rec[key]
                v["n_substeps"] += 1; v["max_n"] = max(v["max_n"], f); v["ctrl"].add(int(s.control_step_index))
                op = off_pad_mm(rg, c.pos)
                if op is not None:
                    v["off_pad_mm"] = min(v["off_pad_mm"], op)
            share.append((int(s.control_step_index), *acc))
            return out

        s.env.sim.step = stepped

    s._hook = hook
    grips = {r: grip_for(PLANS["gentle"], EXECUTOR["settle_after_open"]) for r, _ in SPECS[t].objects}
    plan = make_plan(t, prog, grip=grips, approach_check=EXECUTOR["approach_check"])
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s)
    s.close()
    return ep, rec, share


def main(t, init, role, dz):
    out = {}
    for tag, prog in (("A", make_program(t)), ("B", with_transport_dz(make_program(t), role, dz))):
        ep, rec, share = run(t, init, prog)
        stage = np.asarray(ep.arrays["stage"])
        rows = []
        agg = collections.OrderedDict()
        for (rg, rb, og, ob, tgt, side, allowed, on_t), v in rec.items():
            for k in sorted(v["ctrl"]):
                st = str(stage[min(k, len(stage) - 1)])
                a = agg.setdefault((rg, rb, og, ob, tgt, side, allowed, on_t, st), {"control_steps": 0, "max_n": 0.0, "min_dist_outside_pad_box_mm": None})
                a["control_steps"] += 1; a["max_n"] = max(a["max_n"], v["max_n"])
                a["min_dist_outside_pad_box_mm"] = None if v["off_pad_mm"] > 1e8 else round(v["off_pad_mm"], 2)
        for (rg, rb, og, ob, tgt, side, allowed, on_t, st), a in sorted(agg.items(), key=lambda kv: (kv[0][8], kv[0][0], kv[0][2])):
            rows.append({"stage_instance": st, "force_target": tgt, "robot_geom": rg, "robot_body": rb, "other_geom": og, "other_body": ob,
                         "force_bound_side": side, "on_current_target": on_t, "audit_allowed": allowed, **a, "max_n_over_geom_pair_all_stages": None})
        sh = np.asarray(share, float)
        shares = {}
        for st in sorted({str(x) for x in stage}):
            ks = [k for k in range(len(stage)) if str(stage[k]) == st]
            m_ = np.isin(sh[:, 0], ks)
            if m_.any() and sh[m_, 1:].sum() > 0:
                tot = sh[m_, 1:].sum(axis=0)
                shares[st] = {"force_sum_pad_L_bodyL_pad_R_bodyR": [round(float(x), 1) for x in tot], "finger_body_fraction": round(float((tot[1] + tot[3]) / tot.sum()), 4)}
        out[tag] = {"task_ok": bool(ep.task_ok), "rows": rows, "finger_body_force_share_by_stage": shares}
        print(f"== {tag}  task_ok={ep.task_ok}")
        for r in rows:
            print(f"  {r['stage_instance']:<28} tgt={r['force_target']!s:<16} {r['robot_geom']:<36} body={r['robot_body']:<22} <-> {r['other_geom']:<28} "
                  f"bound={r['force_bound_side']} on_tgt={r['on_current_target']!s:<5} audit_allowed={r['audit_allowed']!s:<5} n={r['control_steps']:>3} max={r['max_n']:.2f} off_pad_mm={r['min_dist_outside_pad_box_mm']}")
    d = ROOT / "reports" / "step8_contact_identity"
    d.mkdir(exist_ok=True)
    (d / f"T{t}_init{init}_{role}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.205)
