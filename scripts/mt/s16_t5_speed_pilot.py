"""Step 16: T5 place-speed paired pilot (structured request; a "manner EXECUTION pilot", not a language end-to-end test).

Same init, same grasp / carry plan.  Two conditions that differ ONLY in the speed level of the place-stage descent:
  N  normal  : every stage at the shared "normal" level (v_max 0.12 m/s) == the frozen ordinary plan, unchanged
  S  slow    : intent (stage=place, speed=slow); the place stage gets the shared "slow" default (v_max 0.06 m/s) and the
               shared rotation coupling w_max = ROT_PER_LIN * v (candidate_plan); every other stage is untouched
Speed levels come from synthesis_v2.manner_request.LEVEL_RANGES (shared); nothing is calibrated here except the T5 development
threshold derived from the dev inits (see derive_dev_calibration) which is a DEVELOPMENT calibration, not a validated one.

Windows (never merged):
  descent  = control steps of the place stage BEFORE the release command  [place_start, release_start)   <- speed window
  release  = [release_start, release_end)       (fingers opening; no hand motion is commanded)
  retreat  = the retreat stage
Labels kept separate: official_success_ever / instruction_completed / contact_clean_v1_1 / official_success_final.
usage: s16_t5_speed_pilot.py dev|new      -> reports/step16_t5_speed_pilot/<phase>/
"""
import dataclasses, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, make_program, make_plan, make_session
from s11_guarded_case import windows_from
from synthesis_v2.candidate_plan import MAX_ROT_RPS, ROT_PER_LIN
from synthesis_v2.contact_audit import ContactRecorder
from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1
from synthesis_v2.manner_calibration import eef_speed_mps
from synthesis_v2.manner_request import MannerIntent, speed_range
from synthesis_v2.native_producer import GripPlan
from synthesis_v2.program_executor import produce_program_episode

PHASE = sys.argv[1]
OUTROOT = ROOT / "reports" / "step16_t5_speed_pilot"
PROTO = json.loads((OUTROOT / "protocol.json").read_text())
OUT = OUTROOT / PHASE; (OUT / "traj").mkdir(parents=True, exist_ok=True)
INITS = PROTO["inits"][PHASE]
GUARD = {"enabled": True, "min_clearance_m": 0.0, "contact_threshold_n": 0.1, "max_retries": 3}   # frozen
GRIP = GripPlan(1.0, 100.0, settle_after_open=1)                                                     # frozen
SITE, BOOK, PLACE = "desk_caddy_1_back_contain_region", "black_book_1", "book.pick_1.place"
CARRY_MIN_RISE_M = 0.10
REQUEST = {"N": (), "S": (MannerIntent("place", "speed", "slow"),)}
SPEED_FAMILY = "stack"     # shared pick-and-place stage table, used only to read LEVEL_RANGES["speed"]
PREFIX_COLS = ("state", "actions", "next_state", "dt", "pos_black_book_1", "pa_black_book_1", "left_mean_n", "right_mean_n",
               "left_max_n", "right_max_n", "other_max_n", "joint_speed_mean", "joint_speed_max")


def in_region(p, site_p, site_m, size):          # geometry of LIBERO SiteObject.in_box on the book body origin (verified == env In)
    tot = np.abs(site_m @ size); ub, lb = site_p + tot, site_p - tot; lb[2] -= 0.01
    return bool(np.all(p > lb) and np.all(p < ub))


def plan_for(cond, prog):
    base = make_plan(5, prog, grip=GRIP, approach_check=True, stop_on_grasp_failure=True, stop_on_place_failure=True)
    stages = []
    for sp in base.stages:
        prim = sp.stage_id.split(".")[-1]                       # "book.pick_1.place" -> "place"
        lvl = next((i.level for i in REQUEST[cond] if i.dimension == "speed" and i.stage == prim), "normal")
        if lvl == "normal":
            stages.append(sp)                                    # untouched: bitwise the frozen ordinary plan
        else:
            v = speed_range(SPEED_FAMILY, lvl)[2]
            stages.append(dataclasses.replace(sp, v_max_mps=v, w_max_rps=min(MAX_ROT_RPS, ROT_PER_LIN * v)))
    return dataclasses.replace(base, stages=tuple(stages))


def win_stats(spd, dt, state, nxt, idx):
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return {"n_steps": 0}
    v = spd[idx]; v = v[np.isfinite(v)]
    seg = np.linalg.norm((nxt[idx, :3] - state[idx, :3]).astype(float), axis=1)
    return {"n_steps": int(len(idx)), "duration_s": round(float(dt[idx].sum()), 4), "mean_speed_mps": round(float(v.mean()), 5),
            "median_speed_mps": round(float(np.median(v)), 5), "p95_speed_mps": round(float(np.percentile(v, 95)), 5),
            "max_speed_mps": round(float(v.max()), 5), "path_length_m": round(float(seg.sum()), 5),
            "net_displacement_m": round(float(np.linalg.norm(nxt[idx[-1], :3].astype(float) - state[idx[0], :3].astype(float))), 5),
            "first_step": int(idx[0]), "last_step": int(idx[-1])}


def run_one(i, cond):
    prog = make_program(5); plan = plan_for(cond, prog)
    s = make_session(5, i, prog); box = {}; t0 = time.time()
    ep = produce_program_episode(prog, i, plan, keep_images=False, session=s, execution_guard=GUARD,
                                 on_session=lambda ss: box.update(r=(lambda c: (box.update(det=DetailRecorder(c)), c)[1])(ContactRecorder(ss).install())))
    wall = time.time() - t0
    a = ep.arrays; st = np.array([str(x) for x in a["stage"]]); pb = np.asarray(a["pos_" + BOOK]); pa = np.asarray(a["pa_" + BOOK])
    state, nxt, dt = np.asarray(a["state"], float), np.asarray(a["next_state"], float), np.asarray(a["dt"], float)
    spd = eef_speed_mps(a)
    m, d = s._mj(); sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    site_p, site_m = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy(); size = s.env.object_sites_dict[SITE].size
    g = ep.meta.get("grasp_policy_by_instance") or {}
    grasp_reached = bool(g) and all(v["reached_band"] for v in g.values())
    rs = next(iter((ep.meta.get("release_start_step_by_instance") or {}).values()), None)
    re_ = next(iter((ep.meta.get("release_end_step_by_instance") or {}).values()), None)
    pl = np.where(st == PLACE)[0]
    place_start = int(pl[0]) if len(pl) else None
    ever_in = [int(k) for k in range(len(pa)) if in_region(pa[k], site_p, site_m, size)]
    audit = box["r"].report([str(x) for x in st]); wins = windows_from(ep, 5)
    v11 = eval_v1(audit, box["r"].events, box["det"].detail, wins, [str(x) for x in st])
    book_end = pa[-1]; bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BOOK + "_main"); qq = d.xquat[bb]
    tot_ = np.abs(site_m @ size); lb_ = site_p - tot_; lb_[2] -= 0.01; ub_ = site_p + tot_
    after = sorted(k for (rb, ob), (f_, ks) in box["r"].events.items() if ob == BOOK + "_main" for k in ks if re_ is not None and k >= re_)
    post = {"official_in_final": in_region(book_end, site_p, site_m, size), "book_origin_final_world": book_end.round(4).tolist(),
            "z_margin_to_official_In_lower_bound_m": round(float(book_end[2] - lb_[2]), 4),
            "xy_inside_region_footprint": bool(np.all(book_end[:2] > lb_[:2]) and np.all(book_end[:2] < ub_[:2])),
            "tilt_deg": round(float(np.degrees(np.arccos(np.clip(1 - 2 * (qq[1] ** 2 + qq[2] ** 2), -1, 1)))), 1),
            "book_displacement_last_10_control_steps_m": round(float(np.linalg.norm(pa[-1] - pa[-11])), 5),
            "robot_book_contact_steps_after_release_end": after[:20], "detached_from_hand": bool(re_ is not None and not after)}
    desc = [] if place_start is None or rs is None else list(range(place_start, rs))
    rel = [] if rs is None or re_ is None else list(range(rs, re_))
    ret = list(np.where(st == "book.pick_1.retreat")[0])
    by_stage = {}
    for lab in dict.fromkeys(st.tolist()):
        idx = np.where(st == lab)[0]
        if lab == PLACE:
            idx = np.array(desc, dtype=int)
        by_stage[lab + ("#descent" if lab == PLACE else "")] = win_stats(spd, dt, state, nxt, idx)
    stages_run = list(dict.fromkeys(st.tolist()))
    official_ever = bool(ever_in)
    instr = bool(grasp_reached and ep.status == "completed" and not ep.meta.get("place_failure") and rs is not None and re_ is not None
                 and "book.pick_1.retreat" in stages_run)
    clean = bool(v11["clean_v1"])
    rec = {"init": i, "cond": cond, "request_intents": [x.to_dict() for x in REQUEST[cond]], "status": ep.status,
           "plan_stage_speeds": {x.stage_id.split(".")[-1]: {"v_max_mps": x.v_max_mps, "w_max_rps": x.w_max_rps, "max_steps": x.max_steps} for x in plan.stages},
           "official_success_ever": official_ever, "official_first_success_step": ever_in[0] if ever_in else None,
           "stage_at_official_first_success": str(st[ever_in[0]]) if ever_in else None,
           "instruction_completed": instr, "contact_clean_v1_1": clean, "contact_clean_v0": bool(audit["clean"]),
           "official_success_final": bool(s.success()), "ordinary_gate": bool(official_ever and instr and clean),
           "grasp_established": grasp_reached, "place_failure": ep.meta.get("place_failure"), "guard_retries": len(ep.meta.get("execution_guard", {}).get("retries", [])),
           "release_start_step": rs, "release_end_step": re_, "place_start_step": place_start,
           "release_issued_before_official_first_success": None if not ever_in or rs is None else bool(rs <= ever_in[0]),
           "in_region_at_release_start": None if rs is None else bool(in_region(pa[max(rs - 1, 0)], site_p, site_m, size)),
           "book_at_release_start_in_region_frame_m": None if rs is None else (site_m.T @ (pa[max(rs - 1, 0)] - site_p)).round(4).tolist(),
           "post_release_M3": post, "descent": by_stage.get(PLACE + "#descent"), "release_window": win_stats(spd, dt, state, nxt, rel),
           "retreat": win_stats(spd, dt, state, nxt, ret), "by_stage": by_stage, "stages_executed": stages_run,
           "v11_contacts": [{"body": r["robot_body"], "max_n": r["max_n"], "excused": r["v1_excused"], "steps": r["control_step_range"]} for r in v11["contacts"]],
           "cost": {"control_steps": int(len(a["actions"])), "sim_wall_s": round(wall, 1)}}
    s.close()
    cols = {k: np.asarray(a[k]) for k in PREFIX_COLS}; cols["stage"] = st; cols["eef_speed_mps"] = spd
    np.savez_compressed(OUT / "traj" / f"init{i}_{cond}.npz", release_start=-1 if rs is None else rs, release_end=-1 if re_ is None else re_, **cols)
    return rec, {k: np.asarray(a[k]) for k in PREFIX_COLS}, st


def pair_record(i, rn, rs_, arn, ars, stn, sts):
    ps = rn["place_start_step"]
    pre = {}
    if ps is not None and rs_["place_start_step"] == ps:
        for k in PREFIX_COLS:
            pre[k] = bool(np.array_equal(arn[k][:ps], ars[k][:ps], equal_nan=True))
        pre["state_at_place_start"] = bool(np.array_equal(arn["state"][ps], ars["state"][ps]))
        pre["stage_labels"] = bool(np.array_equal(stn[:ps], sts[:ps]))
    prefix_identical = bool(pre) and all(pre.values())
    dn, ds = rn["descent"], rs_["descent"]
    r = {"init": i, "place_start_step_equal": rn["place_start_step"] == rs_["place_start_step"], "prefix_bitwise_columns": pre, "prefix_identical": prefix_identical}
    if dn and ds and dn.get("n_steps") and ds.get("n_steps"):
        r["descent_N"], r["descent_S"] = dn, ds
        r["ratio_mean_speed_S_over_N"] = round(ds["mean_speed_mps"] / dn["mean_speed_mps"], 3)
        r["ratio_p95_speed_S_over_N"] = round(ds["p95_speed_mps"] / dn["p95_speed_mps"], 3)
        r["ratio_duration_S_over_N"] = round(ds["duration_s"] / dn["duration_s"], 3)
        r["ratio_path_length_S_over_N"] = round(ds["path_length_m"] / dn["path_length_m"], 3)
        crit = PROTO["criteria"]
        r["pair_manner_pass"] = bool(r["ratio_mean_speed_S_over_N"] <= crit["pair_ratio_mean_speed_max"] and r["ratio_duration_S_over_N"] >= crit["pair_ratio_duration_min"])
    else:
        r["pair_manner_pass"] = False; r["note"] = "descent window missing in at least one condition"
    other = {}
    for lab in rn["by_stage"]:
        if lab.endswith("#descent") or lab not in rs_["by_stage"]:
            continue
        a_, b_ = rn["by_stage"][lab], rs_["by_stage"][lab]
        other[lab.split(".")[-1]] = {"n_steps_N": a_.get("n_steps"), "n_steps_S": b_.get("n_steps"), "mean_speed_N": a_.get("mean_speed_mps"), "mean_speed_S": b_.get("mean_speed_mps")}
    other["release_window"] = {"n_steps_N": rn["release_window"].get("n_steps"), "n_steps_S": rs_["release_window"].get("n_steps"),
                               "mean_speed_N": rn["release_window"].get("mean_speed_mps"), "mean_speed_S": rs_["release_window"].get("mean_speed_mps")}
    r["other_stages_N_vs_S"] = other
    r["ordinary_gate_N"], r["ordinary_gate_S"] = rn["ordinary_gate"], rs_["ordinary_gate"]
    return r


def derive_dev_calibration(recs):
    """Same separation rule as synthesis_v2.manner_calibration (midpoint of the slow max and the normal min, only if separated),
    on the per-episode MEDIAN eef speed of the place descent.  DEVELOPMENT calibration on dev inits only; not validated."""
    med = lambda c: [r["descent"]["median_speed_mps"] for r in recs if r["cond"] == c and r.get("descent") and r["descent"].get("n_steps")]
    sl, no = med("S"), med("N")
    out = {"schema": "mannercap.t5_place_speed_dev_calibration.v1", "status": "development_calibration_not_validated", "n_slow": len(sl), "n_normal": len(no),
           "slow_medians_mps": sl, "normal_medians_mps": no, "dev_inits": PROTO["inits"]["dev"],
           "commanded_v_max_mps": {"slow": speed_range(SPEED_FAMILY, "slow")[2], "normal": speed_range(SPEED_FAMILY, "normal")[2]},
           "note": "the commanded v_max is NOT used as a threshold; the threshold comes from the realised eef speeds of the dev runs"}
    if sl and no and max(sl) < min(no):
        out["slow_upper_mps"] = round(0.5 * (max(sl) + min(no)), 5)
    else:
        out["slow_upper_mps"] = None; out["reason"] = "slow and normal realised median speeds overlap or are missing on the dev inits"
    return out


def main():
    cal = None
    if PHASE == "new":
        cal = json.loads((OUTROOT / "dev" / "dev_calibration.json").read_text())
    recs, pairs = [], []
    for i in INITS:
        rn, arn, stn = run_one(i, "N"); rs_, ars, sts = run_one(i, "S")
        for r in (rn, rs_):
            (OUT / f"init{i}_{r['cond']}.json").write_text(json.dumps(r, indent=1, default=str))
        recs += [rn, rs_]
        pr = pair_record(i, rn, rs_, arn, ars, stn, sts)
        if cal and cal.get("slow_upper_mps") is not None:
            u = cal["slow_upper_mps"]
            rn["abs_slow_by_dev_threshold"] = bool(rn["descent"]["median_speed_mps"] <= u)
            rs_["abs_slow_by_dev_threshold"] = bool(rs_["descent"]["median_speed_mps"] <= u)
            pr["abs_criterion"] = {"slow_upper_mps": u, "N_not_slow": not rn["abs_slow_by_dev_threshold"], "S_slow": rs_["abs_slow_by_dev_threshold"]}
            pr["pair_manner_pass_with_abs"] = bool(pr["pair_manner_pass"] and not rn["abs_slow_by_dev_threshold"] and rs_["abs_slow_by_dev_threshold"])
        pr["pair_joint_pass"] = bool(pr["pair_manner_pass"] and rn["ordinary_gate"] and rs_["ordinary_gate"])
        pairs.append(pr)
        (OUT / f"pair_init{i}.json").write_text(json.dumps(pr, indent=1, default=str))
        for r in (rn, rs_):
            d_ = r.get("descent") or {}
            print(f"RUN init{i} {r['cond']} status={r['status']} official_ever={r['official_success_ever']}(step {r['official_first_success_step']}) instr={r['instruction_completed']} "
                  f"clean_v1.1={r['contact_clean_v1_1']} final_official={r['official_success_final']} | descent n={d_.get('n_steps')} dur={d_.get('duration_s')}s "
                  f"mean={d_.get('mean_speed_mps')} p95={d_.get('p95_speed_mps')} | release {r['release_start_step']}-{r['release_end_step']} wall={r['cost']['sim_wall_s']}", flush=True)
        print(f"PAIR init{i} prefix_identical={pr['prefix_identical']} ratio_mean={pr.get('ratio_mean_speed_S_over_N')} ratio_p95={pr.get('ratio_p95_speed_S_over_N')} "
              f"ratio_dur={pr.get('ratio_duration_S_over_N')} ratio_path={pr.get('ratio_path_length_S_over_N')} pair_manner_pass={pr['pair_manner_pass']} pair_joint_pass={pr['pair_joint_pass']}", flush=True)
    if PHASE == "dev":
        cal = derive_dev_calibration(recs); (OUT / "dev_calibration.json").write_text(json.dumps(cal, indent=1)); print("DEVCAL", json.dumps(cal))
    (OUT / "summary.json").write_text(json.dumps({"phase": PHASE, "inits": INITS, "pairs": pairs, "n_runs": len(recs)}, indent=1, default=str))


if __name__ == "__main__":
    main()
