"""Step 31 diagnosis: re-execute a step-30 attempt command and dump the full contact_policy_v1 verdict per contact
(stage at each contact step, control-step range, peak force, geometry).  Read-only w.r.t. step-30 results."""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("S30_OUT", "reports/step31_t0_contact/diag")
SRC = Path(os.environ.get("S31_SRC", "reports/step30_manner_gen/formal/full"))
import s30_manner_gen as G
import synthesis_v2.contact_policy_v1 as CP

CAP = {}
_orig = CP.evaluate


def _cap(v0, events, detail, windows, stages=None):
    r = _orig(v0, events, detail, windows, stages)
    st = list(stages or [])
    for c in r["contacts"] + r["support_contacts_unexplained"]:
        key = (c["robot_body"], c["other"])
        steps = sorted(CP._steps_of(events, key))
        c["stages_hit"] = sorted({st[k] for k in steps if 0 <= k < len(st)})
    CAP["v1"] = r
    return r


CP.evaluate = _cap


def main(t, init, iid, k, out):
    run = json.loads((SRC / "runs" / f"t{t}_i{init}_{iid}.json").read_text())
    a = run["attempts"][k]
    cmd = {"speed": a["cmd"]["speed"], "grip": {r: tuple(v) for r, v in a["cmd"]["grip"].items()}}
    if os.environ.get("GRIP_OVERRIDE"):
        for kv in os.environ["GRIP_OVERRIDE"].split(";"):
            r, v = kv.split("=")
            cmd["grip"][r] = tuple(float(x) for x in v.split(","))
    prog, roles, names, make = G.program_of(t)
    if os.environ.get("DZ_ADD"):     # role=dz;...  raise lift_dz/transport_dz of that object's carry
        import dataclasses
        add = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in os.environ["DZ_ADD"].split(";")}
        from synthesis_v2.program_executor import LIFT_DZ
        inst = []
        for it in prog.instances:
            if it.object_role in add and it.primitive in ("lift", "transport"):
                k = it.primitive + "_dz"
                it = dataclasses.replace(it, params={**it.params, k: float(it.params.get(k, LIFT_DZ)) + add[it.object_role]})
            inst.append(it)
        prog = dataclasses.replace(prog, instances=tuple(inst))
    npz = G.OUT / "npz" / (Path(out).stem + ".npz")
    meas, info = G.execute(t, init, prog, roles, names, make, cmd, npz)
    ep = np.load(npz, allow_pickle=True)
    v1 = CAP["v1"]
    rows = [{k_: c.get(k_) for k_ in ("robot_body", "other", "class", "max_n", "v1_reasons", "control_step_range",
                                     "stages_hit", "peak_geom", "peak_step", "n_substeps", "n_substeps_inner_region")}
            for c in v1["contacts"] + v1["support_contacts_unexplained"]]
    res = {"task": t, "init": init, "iid": iid, "k": k, "cmd": {"speed": cmd["speed"], "grip": {r: list(v) for r, v in cmd["grip"].items()}},
           "task_ok": meas["task_ok"], "clean": meas["contact_clean"], "objects": meas["objects"], "contacts": rows,
           "windows": v1["windows"], "n_steps": int(len(ep["stage"]))}
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"clean": res["clean"], "task_ok": res["task_ok"], "contacts": [
        (c["robot_body"], c["other"], round(float(c["max_n"] or 0), 1), c["control_step_range"], c["stages_hit"]) for c in rows]}, default=str))


if __name__ == "__main__":
    t, init, iid, k, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    main(t, init, iid, k, out)
