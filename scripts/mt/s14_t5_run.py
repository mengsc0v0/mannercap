"""Step 14 (D): T5 ordinary operation with the shared target-region / fixture interface, ONE official-demonstration grasp
reference, the execution guard ON and the failed-grasp stop ON.  Nothing here is tuned per init.
usage: s14_t5_run.py TAG INIT...  -> reports/step14_t5_onboard/<TAG>/init{N}.json
Reports per init, as separate facts: grasp established / book actually carried / entered the official target region
(environment's own In check, and the per-step geometric test used by it) / contact audit / guard / cost."""
import json, sys, time, math
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_plan, make_session
from synthesis_v2.contact_audit import ContactRecorder
from synthesis_v2.native_producer import GripPlan
from synthesis_v2.program_executor import produce_program_episode

GUARD = {"enabled": True, "min_clearance_m": 0.0, "contact_threshold_n": 0.1, "max_retries": 3}   # frozen identity
GRIP = GripPlan(1.0, 100.0, settle_after_open=1)                                                      # as the pre-check
SITE, BOOK = "desk_caddy_1_back_contain_region", "black_book_1"
TAG = sys.argv[1]; OUT = ROOT / "reports" / "step14_t5_onboard" / TAG; OUT.mkdir(parents=True, exist_ok=True)
CARRY_MIN_RISE_M = 0.10     # the book must leave its start height by this much while the hand carries it


def in_region(p, site_p, site_m, size):          # the geometry of LIBERO's SiteObject.in_box, on a per-step book position
    tot = np.abs(site_m @ size); ub, lb = site_p + tot, site_p - tot; lb[2] -= 0.01
    return bool(np.all(p > lb) and np.all(p < ub))


for i in [int(x) for x in sys.argv[2:]]:
    prog = make_program(5)
    plan = make_plan(5, prog, grip=GRIP, approach_check=True, stop_on_grasp_failure=True)
    s = make_session(5, i, prog)
    box = {}; t0 = time.time()
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, execution_guard=GUARD,
                                 on_session=lambda ss: box.update(r=ContactRecorder(ss).install()))
    wall = time.time() - t0
    a = ep.arrays; stages = np.array([str(x) for x in a["stage"]]); pb = np.asarray(a["pos_" + BOOK]); pa = np.asarray(a["pa_" + BOOK])
    m, d = s._mj(); sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    site_p, site_m = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    size = s.env.object_sites_dict[SITE].size
    last = lambda suffix: np.where(np.char.endswith(stages, suffix))[0]
    g = ep.meta.get("grasp_policy_by_instance") or {}
    grasp_reached = bool(g) and all(v["reached_band"] for v in g.values())
    lift_idx, tr_idx, pl_idx = last(".lift"), last(".transport"), last(".place")
    rise = pa[:, 2] - pb[0, 2]
    carried_rise = float(rise[np.concatenate([lift_idx, tr_idx])].max()) if len(lift_idx) + len(tr_idx) else 0.0
    end_tr = pa[tr_idx[-1]] if len(tr_idx) else None
    rel_end_tr = None if end_tr is None else (site_m.T @ (end_tr - site_p)).round(4).tolist()
    carried = bool(carried_rise >= CARRY_MIN_RISE_M and end_tr is not None and np.linalg.norm((end_tr - site_p)[:2]) < 0.06)
    ever_in = [int(k) for k in range(len(pa)) if in_region(pa[k], site_p, site_m, size)]
    book_end = pa[-1]
    audit = box["r"].report([str(x) for x in a["stage"]])
    gd = ep.meta.get("execution_guard", {})
    rec = {"init": i, "status": ep.status, "task_ok": bool(ep.task_ok), "env_success_now": bool(s.success()),
           "grasp_established": grasp_reached, "grasp_policy": {k: {kk: v[kk] for kk in ("reached_band", "reverse_pulses", "closing_steps", "final")} for k, v in g.items()},
           "grasp_failure": ep.meta.get("grasp_failure"),
           "stages_executed": [x for x in dict.fromkeys(stages.tolist())],
           "book_max_rise_during_lift_transport_m": round(carried_rise, 3), "book_carried": carried,
           "book_in_region_frame_at_end_of_transport_m": rel_end_tr,
           "entered_official_region_any_step": bool(ever_in), "first_step_in_region": ever_in[0] if ever_in else None,
           "book_final_in_region_frame_m": (site_m.T @ (book_end - site_p)).round(4).tolist(),
           "final_in_region": in_region(book_end, site_p, site_m, size),
           "back_region_half_size": [round(float(x), 4) for x in size],
           "contact_audit": {k: audit[k] for k in ("clean", "max_forbidden_n", "forbidden_contacts", "forbidden_by_class")},
           "guard": {"n_checks": len(gd.get("checks", [])), "n_retries": len(gd.get("retries", [])), "retries": gd.get("retries", []),
                     "chosen_candidate": gd.get("chosen_candidate")},
           "approach_plan": ep.meta.get("approach_plan"),
           "cost": {"control_steps": int(len(a["actions"])), "sim_wall_s": round(wall, 1)}}
    s.close()
    (OUT / f"init{i}.json").write_text(json.dumps(rec, default=str, indent=1))
    print(f"INIT {i} status={rec['status']} task_ok={rec['task_ok']} env_success={rec['env_success_now']} | grasp={rec['grasp_established']} "
          f"carried={rec['book_carried']}(rise={rec['book_max_rise_during_lift_transport_m']}) region_any={rec['entered_official_region_any_step']} "
          f"final_in_region={rec['final_in_region']} | clean={audit['clean']} max_forbidden_n={audit['max_forbidden_n']} "
          f"stages={rec['stages_executed']} wall={rec['cost']['sim_wall_s']}", flush=True)
