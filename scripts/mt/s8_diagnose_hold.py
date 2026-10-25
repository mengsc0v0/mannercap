"""Step 8: limited diagnosis of a hold-window failure from saved raw trajectories (no simulation).
For one request: four-state split per stage, first deviation, and a comparison with reference requests of the same task.

  s8_diagnose_hold.py TAG TASK ROLE FAILED_INIT REF_INIT [REF_INIT ...]  ->  reports/step8_diagnosis/{TAG}_T{TASK}_init{FAILED_INIT}_{ROLE}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT, SPECS, make_program  # noqa: E402
import synthesis_v2.transport_repair_rule as RULE  # noqa: E402

_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
STATES = RULE.GRASP_STATES


def load(tag, t, init):
    return np.load(ROOT / "reports" / "step8_shared_repair" / "trajectories" / f"{tag}_T{t}_init{init}_A.npz", allow_pickle=False)


def per_stage(z, prog, role, name):
    st = [str(x) for x in z["ep_stage"]]
    L, R, V = z["ep_left_mean_n"], z["ep_right_mean_n"], z["ep_valid_fraction"]
    state = RULE.grasp_state(L, R, V)
    pos = z[f"ep_pos_{name}"]
    eef = z["ep_state"][:, :3]
    z0 = pos[0, 2]
    rows = []
    for it in prog.instances:
        if it.object_role != role:
            continue
        idx = [k for k, s in enumerate(st) if s == it.label]
        if not idx:
            continue
        i = np.asarray(idx)
        rows.append({"stage": it.label.split(".")[-1], "steps": [int(i[0]), int(i[-1])], "n": int(len(i)),
                     **{k: int((state[i] == k).sum()) for k in STATES},
                     "L_mean_min_max": [round(float(np.nanmean(L[i])), 2), round(float(np.nanmin(L[i])), 2), round(float(np.nanmax(L[i])), 2)],
                     "R_mean_min_max": [round(float(np.nanmean(R[i])), 2), round(float(np.nanmin(R[i])), 2), round(float(np.nanmax(R[i])), 2)],
                     "other_max_n_max": round(float(np.nanmax(z["ep_other_max_n"][i])), 2),
                     "obj_z_rise_from_start_m": [round(float(pos[i[0], 2] - z0), 4), round(float(pos[i[-1], 2] - z0), 4)],
                     "obj_minus_eef_xy_m_end": round(float(np.linalg.norm(pos[i[-1], :2] - eef[i[-1], :2])), 4),
                     "obj_minus_eef_z_m_start_end": [round(float(pos[i[0], 2] - eef[i[0], 2]), 4), round(float(pos[i[-1], 2] - eef[i[-1], 2]), 4)]})
    return rows, state, st


def first_deviation(prog, role, state, st, z, name):
    h = RULE.hold_steps(prog, st, role)
    bad = [int(k) for k in h if state[k] in ("one_sided", "no_contact")]
    if not bad:
        return None
    k = bad[0]
    lo = max(0, k - 4)
    pos, eef = z[f"ep_pos_{name}"], z["ep_state"][:, :3]
    win = [{"k": j, "stage": st[j].split(".")[-1], "state": str(state[j]), "L": round(float(z["ep_left_mean_n"][j]), 2), "R": round(float(z["ep_right_mean_n"][j]), 2),
            "obj_minus_eef_z": round(float(pos[j, 2] - eef[j, 2]), 4), "obj_z": round(float(pos[j, 2]), 4), "other_max_n": round(float(z["ep_other_max_n"][j]), 2)}
           for j in range(lo, min(len(st), k + 6))]
    return {"first_non_bilateral_step": k, "stage": st[k], "n_non_bilateral": len(bad), "steps_window": win}


def substep_view(z, role_name, st, state):
    """Substep columns of the target: palm/other-gripper and basket share, and L/R per stage (substep means)."""
    cols = [str(c) for c in z["sub_columns"]]
    ctrl, arr = z["sub_ctrl"], z["sub_arr"]
    out = {}
    for c in ("L", "R", "palm_other_gripper", "basket"):
        out[c] = arr[:, cols.index(f"{role_name}.{c}")]
    res = {}
    for sg in sorted({s for s in st}):
        if role_name.split("_1")[0] not in sg and "soup" not in sg:
            continue
        m = np.isin(ctrl, [k for k, s in enumerate(st) if s == sg])
        if m.any():
            res[sg.split(".")[-1]] = {c: {"mean": round(float(out[c][m].mean()), 2), "max": round(float(out[c][m].max()), 2)} for c in out}
    return res


def main(tag, t, role, bad_init, refs):
    prog = make_program(t)
    name = prog.objects[role]
    rep = {"tag": tag, "task": t, "role": role, "object": name, "failed": f"init{bad_init}", "references": [f"init{r}" for r in refs]}
    for label, init in [("failed", bad_init)] + [("ref", r) for r in refs]:
        z = load(tag, t, init)
        rows, state, st = per_stage(z, prog, role, name)
        rep[f"init{init}"] = {"per_stage": rows, "first_deviation": first_deviation(prog, role, state, st, z, name),
                              "substep_by_stage": substep_view(z, name, st, state),
                              "task_success_last": bool(np.asarray(z["ep_success"])[-1])}
    d = ROOT / "reports" / "step8_diagnosis"
    d.mkdir(exist_ok=True)
    out = d / f"{tag}_T{t}_init{bad_init}_{role}.json"
    out.write_text(json.dumps(rep, indent=1, default=_J))
    for k in [f"init{bad_init}"] + [f"init{r}" for r in refs]:
        print("==", k, "success_last", rep[k]["task_success_last"])
        for r in rep[k]["per_stage"]:
            print("  ", {a: r[a] for a in ("stage", "n", "bilateral", "one_sided", "no_contact", "invalid", "L_mean_min_max", "R_mean_min_max", "obj_z_rise_from_start_m", "obj_minus_eef_z_m_start_end", "other_max_n_max")})
        fd = rep[k]["first_deviation"]
        print("   first_dev:", None if fd is None else {a: fd[a] for a in ("first_non_bilateral_step", "stage", "n_non_bilateral")})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), [int(x) for x in sys.argv[5:]])
