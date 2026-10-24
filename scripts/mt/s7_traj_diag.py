"""Terminal-state diagnosis of one onboarding episode (read-only, same executor/config as s7_onboard 'after'):
per picked object and stage: object displacement vs its start, max finger forces on it, hand-object gap at stage end;
positions at lift end / before release / final; goal predicates (+ first stage where the object leaves the expected state).
Usage: s7_traj_diag.py TASK INIT...  -> reports/step7_onboard_diag/traj_t<T>_init<i>.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from s7_task_config import ROOT, SPECS, make_program, make_plan, make_session  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

T = int(sys.argv[1]); OUT = ROOT / "reports" / "step7_onboard_diag"; OUT.mkdir(parents=True, exist_ok=True)
r2 = lambda x: np.round(np.asarray(x, float), 3).tolist()  # noqa: E731
for i in [int(x) for x in sys.argv[2:]]:
    prog = make_program(T); s = make_session(T, i, prog)
    ep = produce_program_episode(prog, i, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
    a = ep.arrays; st = np.array([str(x) for x in a["stage"]]); out = {"task": T, "init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "objects": {}}
    inner = getattr(s.env, "env", s.env)
    out["predicates"] = [{"p": " ".join(map(str, g)), "true": bool(inner.eval_predicate(g))} for g in inner.parsed_problem["goal_state"]] \
        if hasattr(inner, "eval_predicate") else None
    basket = a["pos_basket_1"][-1] if "pos_basket_1" in a else None
    out["basket_final"] = r2(basket) if basket is not None else None
    for role, nm in SPECS[T].objects:
        key = nm if nm.endswith("_1") else nm + "_1"
        if "pos_" + key not in a: continue
        P = a["pos_" + key]; z0 = P[0]; rows = []
        for stg in dict.fromkeys(st):
            if not stg.startswith(role + "."): continue
            m = st == stg
            rows.append({"stage": stg.split(".")[-1], "n": int(m.sum()), "obj_dz_cm": round(float((P[m][-1] - z0)[2]) * 100, 1),
                         "obj_dxy_cm": round(float(np.linalg.norm((P[m][-1] - z0)[:2])) * 100, 1),
                         "obj_to_basket_dxy_cm": None if basket is None else round(float(np.linalg.norm(P[m][-1][:2] - a["pos_basket_1"][m][-1][:2])) * 100, 1),
                         "cf_L_max": round(float(a["cf_L_" + key][m].max()), 1), "cf_R_max": round(float(a["cf_R_" + key][m].max()), 1)})
        out["objects"][role] = {"start": r2(z0), "final": r2(P[-1]), "stages": rows}
    json.dump(out, open(OUT / f"traj_t{T}_init{i}.json", "w"), indent=1)
    print(f"== T{T} init{i} status={ep.status.split(':')[0]} task_ok={ep.task_ok} predicates={[(p['p'], p['true']) for p in out['predicates']] if out['predicates'] else None} basket_final={out['basket_final']}")
    for role, o in out["objects"].items():
        print(f"  {role} start={o['start']} final={o['final']}")
        for r in o["stages"]: print("    ", r)
    s.close()
