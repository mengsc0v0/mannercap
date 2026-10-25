"""Step 8 (v1 pipeline): object-dependent grip-force response, calibration and swapped pairs for LIBERO basket tasks T0 / T7.

Four separate questions are recorded and never merged into one flag:
  separability          do the plans give distinguishable hold-force distributions?
  measurement validity  was the force measured on the target object, finite, with a hold window?
  grasp stability       strict contact/stability criteria (both fingers on every hold step, L/R imbalance, object slip)
  calibration usability separable AND measurement-valid AND stable at every level that is used  -> bands are derived only then
Force statistics used for bands come only from records that pass measurement validity and stability.
Peaks: ``peak_simultaneous`` = max over physics substeps of (left+right) at the same substep; ``peak_upper_bound`` = max over
control steps of (left_max+right_max) (an upper bound).  The response-alert threshold derived from them is NOT a safety bound
(independent_safety_bound stays null / unverified).

The executor setting (EXECUTOR) is part of the config and its hash; every record carries a config key and existing files are
re-used only if the key matches.  Splits: calibration inits 8-13; pairs on dev inits 0-7 / 14-21 (22-39 reserved, 40-49 holdout).
Raw actions, stages, per-step telemetry and per-substep finger forces are saved per episode (reports/step8_t<T>_raw/*.npz).

  s8_task_force.py response TASK [--inits ...]
  s8_task_force.py derive TASK
  s8_task_force.py pairs TASK --inits ... --combos GF FG
  s8_task_force.py summary TASK
Not covered (reduced difficulty): the task-1 replay/resume/measure/event audits; this is a response probe + band audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_validate import hold_idx  # noqa: E402
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from synthesis_v2.contact_audit import ContactRecorder  # noqa: E402
from synthesis_v2.grasp_audit import audit_episode  # noqa: E402
from synthesis_v2.manner_calibration import DEFAULT_POLICY  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402
from synthesis_v2.substep_recorder import SubstepRecorder  # noqa: E402

SCHEMA = "s8.v1"
EXECUTOR = {"settle_after_open": 1, "approach_check": True}
PLANS = {"gentle": (1.0, 22.0), "normal": (1.0, 100.0), "firm": (35.0, 100.0)}
CAL_INITS = tuple(range(8, 14))
DEV_INITS = tuple(range(0, 8)) + tuple(range(14, 22))
LV = {"G": "gentle", "F": "firm"}
CONTACT_THR_N, LR_MAX, Z_MAX_M, VALID_MIN = 0.1, 0.10, 0.01, 0.99
_J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731


def _hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), default=_J).encode()).hexdigest()


RESP_KEY = _hash({"schema": SCHEMA, "executor": EXECUTOR, "plans": PLANS, "thr": [CONTACT_THR_N, LR_MAX, Z_MAX_M, VALID_MIN]})


def cfg_path(t):
    return ROOT / "configs" / f"manner_calibration_libero_t{t}.v3.json"


def out_dir(t, kind):
    d = ROOT / "reports" / f"step8_t{t}_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


# ---------------------------------------------------------------- physics-substep recorder (passive, rollback-isolated)
def install_recorder(s, names):
    """Per physics substep: left/right finger normal force per target object.  Rows recorded inside a save_state/restore_state
    branch (settled look-ahead, recovery) are moved to ``rolled_back`` (see synthesis_v2/substep_recorder.py)."""
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    holder = {}
    rec = SubstepRecorder(s, lambda sess: holder["read"]()).install_state_hooks()
    orig = s._hook

    def hook():
        orig()
        m, d = s._mj()
        bind = {n: resolve_grasp_binding(m, mujoco, s.geom_ids_of(n), n) for n in names}
        left, right = set(bind[names[0]].left_finger_geom_ids), set(bind[names[0]].right_finger_geom_ids)
        tgt = {n: set(b.target_geom_ids) for n, b in bind.items()}
        buf = np.zeros(6)

        def read():
            L = {n: 0.0 for n in names}
            R = {n: 0.0 for n in names}
            for ci in range(int(d.ncon)):
                c = d.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                for g, o in ((g1, g2), (g2, g1)):
                    side = "L" if g in left else ("R" if g in right else None)
                    if side is None or o in left or o in right:
                        continue
                    hit = [n for n in names if o in tgt[n]]
                    if hit:
                        mujoco.mj_contactForce(m, d, ci, buf)
                        f = abs(float(buf[0]))
                        if f > 0:
                            for n in hit:
                                (L if side == "L" else R)[n] += f
                    break
            return tuple(x for n in names for x in (L[n], R[n]))

        holder["read"] = read
        prev = s.env.sim.step

        def stepped(*a, **k):
            out = prev(*a, **k)
            rec.sample()
            return out

        s.env.sim.step = stepped

    s._hook = hook
    return rec


def run_episode(t, init, grips, raw_path=None):
    prog = make_program(t)
    s = make_session(t, init, prog)
    names = [n for _, n in SPECS[t].objects]
    sub = install_recorder(s, names)
    crec = ContactRecorder(s).install()          # frozen contact policy (rollback-isolated as well)
    t0 = time.time()
    plan = make_plan(t, prog, grip=grips, approach_check=EXECUTOR["approach_check"])
    ep = produce_program_episode(prog, init, plan, keep_images=False, session=s)
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 2 * len(names))
    rec = {"ctrl": ctrl, "L": {n: arr[:, 2 * i] for i, n in enumerate(names)}, "R": {n: arr[:, 2 * i + 1] for i, n in enumerate(names)},
           "rolled_back_cost": sub.rolled_back_cost(), "contact_audit": crec.report([str(x) for x in ep.arrays["stage"]])}
    if raw_path is not None:
        a = ep.arrays
        keep = {k: np.asarray(a[k]) for k in ("stage", "actions", "state", "next_state", "left_mean_n", "right_mean_n", "left_max_n", "right_max_n",
                                              "valid_fraction", "force_target", "hold_lo_n", "hold_hi_n") if k in a}
        for n in names:
            keep["sub_L_" + n], keep["sub_R_" + n] = rec["L"][n], rec["R"][n]
            if "pos_" + n in a:
                keep["pos_" + n] = np.asarray(a["pos_" + n])
        np.savez_compressed(raw_path, sub_ctrl=ctrl, **keep)
    return prog, ep, rec, time.time() - t0


def grip_for(plan, settle):
    return GripPlan(plan[0], plan[1], settle_after_open=settle)


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------- per-object record
def per_object(ep, prog, t, rec):
    a = ep.arrays
    st = np.asarray(a["stage"])
    try:
        audit = audit_episode(ep, prog.objects)
    except Exception as e:  # keep the request, mark unknown
        audit = {"_error": repr(e)}
    out = {}
    for role, name in SPECS[t].objects:
        h = hold_idx(a, role)
        win = np.flatnonzero(np.isin(st, [f"{role}.pick_1.{x}" for x in ("grasp", "lift", "transport")]))
        r = {"target": name, "hold_n": int(len(h))}
        g = audit.get(f"{role}.pick_1.grasp") if isinstance(audit, dict) else None
        r["grasp_flags"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in g.items() if k != "trace"} if isinstance(g, dict) else None
        if len(h) == 0:
            r.update({"status": "no_hold_window", "measurement_valid": False, "grasp_stable": None})
            out[role] = r
            continue
        L, R = np.asarray(a["left_mean_n"])[h], np.asarray(a["right_mean_n"])[h]
        tot = L + R
        vf = np.asarray(a["valid_fraction"])[h]
        tgt_ok = ("force_target" not in a) or set(np.asarray(a["force_target"])[h]) == {name}
        meas_valid = bool(np.all(np.isfinite(tot)) and vf.min() >= VALID_MIN and tgt_ok)
        r["measurement"] = {"finite": bool(np.all(np.isfinite(tot))), "valid_fraction_min": float(vf.min()), "target_ok": bool(tgt_ok)}
        r["measurement_valid"] = meas_valid
        if not meas_valid:
            r["status"] = "measurement_invalid"
            r["grasp_stable"] = None
            out[role] = r
            continue
        z = np.asarray(a["pos_" + name])[h, 2] - np.asarray(a["state"])[h, 2]
        both = np.minimum(L, R) > CONTACT_THR_N
        gaps = runs(~both)
        lr = float(np.mean(np.abs(L - R) / np.maximum(tot, 1e-9)))
        zs = float(np.ptp(z))
        bil_strict = bool(both.all())
        sub = np.isin(rec["ctrl"], h)
        subL, subR = rec["L"][name][sub], rec["R"][name][sub]
        w = np.isin(rec["ctrl"], win)
        sim = (rec["L"][name] + rec["R"][name])[w]
        pk_ub = np.asarray(a["left_max_n"])[win] + np.asarray(a["right_max_n"])[win]
        stable = bool(bil_strict and lr <= LR_MAX and zs <= Z_MAX_M)
        r.update({
            "status": "ok", "hold_total": [float(x) for x in tot], "hold_mean": float(tot.mean()), "hold_min": float(tot.min()), "hold_max": float(tot.max()),
            "lr_imbalance": lr, "rel_z_spread_m": zs,
            "bilateral_strict": bil_strict, "bilateral_step_fraction": float(both.mean()), "n_gap_runs": len(gaps),
            "longest_gap_steps": int(max((g_[1] for g_ in gaps), default=0)),
            "substep_bilateral_fraction": float(np.mean((subL > CONTACT_THR_N) & (subR > CONTACT_THR_N))) if sub.any() else None,
            "peak_simultaneous_n": float(sim.max()) if len(sim) else None,
            "peak_simultaneous_stage": str(st[rec["ctrl"][w][int(np.argmax((rec["L"][name] + rec["R"][name])[w]))]]) if len(sim) else None,
            "peak_upper_bound_n": float(np.nanmax(pk_ub)) if len(pk_ub) else None,
            "stability_components": {"bilateral_strict": bil_strict, "lr_ok": bool(lr <= LR_MAX), "z_ok": bool(zs <= Z_MAX_M)},
            "grasp_stable": stable})
        out[role] = r
    return out


def recorder_consistency(ep, rec):
    """Independent check that the formal substep rows are the executed trajectory: per control step the substep mean of the force on
    the object being measured must equal the executed telemetry, and the formal rows must hold exactly n_rows control steps."""
    a = ep.arrays
    n = len(a["stage"])
    ctrl = rec["ctrl"]
    ft = np.asarray(a["force_target"]) if "force_target" in a else None
    worst, cnt = 0.0, 0
    for k in range(n):
        if ft is None or ft[k] not in rec["L"]:
            continue
        sel = ctrl == k
        if not sel.any() or np.asarray(a["valid_fraction"])[k] < 1.0:
            continue
        d = max(abs(float(rec["L"][ft[k]][sel].mean()) - float(a["left_mean_n"][k])), abs(float(rec["R"][ft[k]][sel].mean()) - float(a["right_mean_n"][k])))
        worst, cnt = max(worst, d), cnt + 1
    per = np.bincount(ctrl, minlength=n)
    return {"checked_steps": cnt, "max_abs_diff_n": worst, "ctrl_range_ok": bool(len(ctrl) and ctrl.min() >= 0 and ctrl.max() == n - 1 or len(ctrl) == 0),
            "no_duplicate_control_steps": bool((per[per > 0] <= per[per > 0].min() * 2).all()) if (per > 0).any() else True,
            "n_substeps_per_step_min_max": [int(per[per > 0].min()), int(per.max())] if (per > 0).any() else None}


def _stamp(extra):
    return {"schema": SCHEMA, **extra}


def _valid_existing(f, key_field, key):
    if not f.exists():
        return False
    try:
        r = json.loads(f.read_text())
    except Exception:
        return False
    return r.get("schema") == SCHEMA and r.get(key_field) == key and r.get("status") != "error"


def _episode_record(base, t, init, grips, raw_path):
    try:
        prog, ep, rec, dt = run_episode(t, init, grips, raw_path)
        base.update({"status": ep.status, "task_ok": bool(ep.task_ok), "n_steps": int(ep.n_steps), "wall_s": round(dt, 1),
                     "raw": str(Path(raw_path).relative_to(ROOT)), "objects": per_object(ep, prog, t, rec),
                     "lookahead_rolled_back_cost": rec["rolled_back_cost"], "recorder_vs_telemetry": recorder_consistency(ep, rec),
                     "contact_policy": {k: rec["contact_audit"][k] for k in ("clean", "max_forbidden_n", "forbidden_contacts")},
                     "contact_policy_note": "frozen policy (contact_audit.py): any non-fingertip robot body <-> any movable object above 0.1 N is forbidden; "
                                            "support-surface contact observation-only"})
    except Exception as e:  # explicit failure record instead of losing the request
        base.update({"status": "error", "task_ok": None, "error": repr(e), "traceback": traceback.format_exc()[-1500:], "objects": {}})
    return base


def cmd_response(t, inits):
    d, raw = out_dir(t, "response"), out_dir(t, "raw")
    for i in inits:
        assert i in CAL_INITS, "calibration split only (8-13)"
        for lvl, plan in PLANS.items():
            f = d / f"resp_init{i}_{lvl}.json"
            if _valid_existing(f, "config_key", RESP_KEY):
                continue
            roles = [r for r, _ in SPECS[t].objects]
            base = _stamp({"task": t, "init": i, "level": lvl, "plan": list(plan), "executor": EXECUTOR, "config_key": RESP_KEY})
            rec = _episode_record(base, t, i, {r: grip_for(plan, EXECUTOR["settle_after_open"]) for r in roles}, raw / f"resp_init{i}_{lvl}.npz")
            f.write_text(json.dumps(rec, default=_J))
            print(f"T{t} i{i} {lvl} {rec['status']} ok={rec['task_ok']} " + " ".join(
                f"{r[:5]}:{v.get('status')}/stable={v.get('grasp_stable')}/m={v.get('hold_mean') and round(v['hold_mean'], 1)}" for r, v in rec["objects"].items()), flush=True)


# ---------------------------------------------------------------- calibration
def pct(v, q):
    return float(np.percentile(v, q))


def stats(h):
    return {"n": len(h), "p5": pct(h, 5), "p95": pct(h, 95), "p99": pct(h, 99), "min": float(min(h)), "max": float(max(h))} if len(h) else None


LEVELS_USED = ("gentle", "firm")   # two-level protocol: a development choice made after seeing the first response (gentle == normal);
#                                    versioned in the config (protocol), frozen before pairs.  normal is kept as a diagnostic reference only.


def cmd_derive(t):
    pol = dict(DEFAULT_POLICY)
    recs = {}
    for lvl in PLANS:
        for i in CAL_INITS:
            f = out_dir(t, "response") / f"resp_init{i}_{lvl}.json"
            assert _valid_existing(f, "config_key", RESP_KEY), f"missing/outdated response record {f.name}"
            recs[(lvl, i)] = json.loads(f.read_text())
    cal = {"schema": "mannercap.libero_calibration.v4", "task": f"libero_10/{t}", "dev_inits": list(CAL_INITS), "executor": EXECUTOR, "response_key": RESP_KEY,
           "protocol": {"levels_used": list(LEVELS_USED), "diagnostic_only": ["normal"], "origin": "development choice after observing gentle == normal; not pre-registered"},
           "criteria": {"contact_thr_n": CONTACT_THR_N, "lr_max": LR_MAX, "z_spread_max_m": Z_MAX_M, "valid_fraction_min": VALID_MIN,
                        "bilateral": "min(L,R)>thr on every hold control step"},
           "policy": {k: pol[k] for k in ("gentle_floor_fraction", "firm_cap_factor", "safety_margin_factor")},
           "plans": {k: {"target_lo_n": v[0], "target_hi_n": v[1]} for k, v in PLANS.items()},
           "independent_safety_bound": None, "objects": {}}
    n_dev = len(CAL_INITS)
    for role, _ in SPECS[t].objects:
        per = {lvl: [recs[(lvl, i)] for i in CAL_INITS] for lvl in PLANS}

        def obj(r):
            return r["objects"].get(role, {})

        cnt = {lvl: {"task_ok": sum(bool(r.get("task_ok")) for r in rs),
                     "measurement_valid": sum(bool(obj(r).get("measurement_valid")) for r in rs),
                     "stable": sum(bool(obj(r).get("grasp_stable")) for r in rs),
                     "eligible": sum(bool(r.get("task_ok") and obj(r).get("measurement_valid") and obj(r).get("grasp_stable")) for r in rs)}
               for lvl, rs in per.items()}
        diag = {lvl: stats([x for r in rs if r.get("task_ok") and obj(r).get("measurement_valid") for x in obj(r)["hold_total"]]) for lvl, rs in per.items()}
        elig = {lvl: [x for r in rs if r.get("task_ok") and obj(r).get("measurement_valid") and obj(r).get("grasp_stable") for x in obj(r)["hold_total"]]
                for lvl, rs in per.items()}
        used = list(LEVELS_USED)
        sep_gf = bool(diag["gentle"] and diag["firm"] and diag["gentle"]["p95"] < diag["firm"]["p5"])      # gentle vs firm, all task-ok valid records
        sep_normal_diag = ("normal_equals_gentle" if diag["normal"] and diag["gentle"] and abs(diag["normal"]["p5"] - diag["gentle"]["p5"]) < 0.5 and
                           abs(diag["normal"]["p95"] - diag["gentle"]["p95"]) < 0.5 else "normal_differs_from_gentle") if all(diag.values()) else None
        meas_ok = all(cnt[l]["measurement_valid"] == n_dev for l in used)
        stable_ok = all(cnt[l]["eligible"] == n_dev for l in used)
        usable = bool(sep_gf and meas_ok and stable_ok)
        o = {"levels_used": used, "separability_gentle_vs_firm": sep_gf, "normal_diagnostic": sep_normal_diag, "measurement_valid_used_levels": meas_ok,
             "grasp_stable_used_levels": stable_ok, "counts_per_level": cnt, "n_dev": n_dev, "diagnostic_hold_stats_all_task_ok": diag,
             "calibration_usable": usable,
             "unusable_reasons": [r_ for r_, bad in (("gentle_not_separable_from_firm", not sep_gf), ("measurement_invalid", not meas_ok), ("grasp_stability_failed", not stable_ok)) if bad]}
        pk_all = [obj(r).get("peak_simultaneous_n") for rs in per.values() for r in rs if obj(r).get("peak_simultaneous_n") is not None]
        o["diagnostic_peak_simultaneous_max_n_all"] = max(pk_all) if pk_all else None
        if usable:
            q = {lvl: stats(elig[lvl]) for lvl in used}
            mid = 0.5 * (q["gentle"]["p95"] + q["firm"]["p5"])
            o.update({"hold_stats": q, "gentle_lower": pol["gentle_floor_fraction"] * q["gentle"]["p5"], "gentle_upper": mid, "firm_lower": mid,
                      "firm_upper": pol["firm_cap_factor"] * q["firm"]["p99"]})
            pk = [obj(r)["peak_simultaneous_n"] for l in used for r in per[l] if obj(r).get("peak_simultaneous_n") is not None]
            o["response_alert_threshold_n"] = pol["safety_margin_factor"] * max(pk)
            o["response_alert_basis"] = "margin * max simultaneous two-finger substep peak over used-level eligible calibration records; NOT a safety bound"
        else:
            for k in ("gentle_lower", "gentle_upper", "firm_lower", "firm_upper", "response_alert_threshold_n"):
                o[k] = None
        cal["objects"][role] = o
    cal["content_hash"] = _hash(cal)
    cfg_path(t).write_text(json.dumps(cal, indent=1, sort_keys=True, default=_J))
    for r, o in cal["objects"].items():
        print(r, "sep(G vs F)=", o["separability_gentle_vs_firm"], "meas_ok=", o["measurement_valid_used_levels"], "stable_ok=", o["grasp_stable_used_levels"],
              "USABLE=", o["calibration_usable"], o["unusable_reasons"], {k: v["eligible"] for k, v in o["counts_per_level"].items()}, o["normal_diagnostic"])
    print("hash", cal["content_hash"][:16])


def load_cal(t, inits):
    cal = json.loads(cfg_path(t).read_text())
    h = cal.pop("content_hash")
    assert h == _hash(cal), "calibration modified"
    cal["content_hash"] = h
    assert not set(inits) & set(cal["dev_inits"]) and all(i in DEV_INITS for i in inits), "pairs only on dev inits, disjoint from calibration"
    assert cal["executor"] == EXECUTOR and cal["response_key"] == RESP_KEY, "executor/plans differ from the calibration config"
    return cal


def audit_role(ro, co, lvl):
    """Band audit of one object in one trajectory.  `qualified` = usable calibration, valid measurement, stable grasp, all hold steps in band.
    Per-trajectory force check only: no GF/FG pair logic, no safety verdict, no replay/recovery verification, no contact-policy verdict (separate field)."""
    reasons = []
    if not co["calibration_usable"] or lvl not in co["levels_used"]:
        reasons.append("calibration_unusable:" + "+".join(co["unusable_reasons"]))
    if ro.get("status") != "ok":
        reasons.append(ro.get("status", "missing"))
    elif not ro.get("grasp_stable"):
        reasons.append("grasp_unstable")
    lo, hi = co.get(f"{lvl}_lower"), co.get(f"{lvl}_upper")
    in_band = None
    if ro.get("status") == "ok" and lo is not None:
        in_band = bool(ro["hold_min"] >= lo and ro["hold_max"] <= hi)
        if not in_band:
            reasons.append("hold_force_out_of_band")
    alert = co.get("response_alert_threshold_n")
    pk = ro.get("peak_simultaneous_n")
    return {"level": lvl, "band": [lo, hi], "hold_min": ro.get("hold_min"), "hold_max": ro.get("hold_max"), "hold_mean": ro.get("hold_mean"),
            "all_hold_steps_in_band": in_band, "peak_simultaneous_n": pk, "peak_upper_bound_n": ro.get("peak_upper_bound_n"),
            "peak_le_response_alert": (None if alert is None or pk is None else bool(pk <= alert)),
            "grasp_stable": ro.get("grasp_stable"), "qualified": bool(not reasons and in_band), "reasons": reasons}


def cmd_pairs(t, inits, combos):
    d, raw = out_dir(t, "pairs"), out_dir(t, "raw")
    cal = load_cal(t, inits)
    for i in inits:
        for combo in combos:
            f = d / f"init{i}_{combo}.json"
            if _valid_existing(f, "calibration_hash", cal["content_hash"]):
                continue
            roles = [r for r, _ in SPECS[t].objects]
            grips = {r: grip_for((cal["plans"][LV[c]]["target_lo_n"], cal["plans"][LV[c]]["target_hi_n"]), cal["executor"]["settle_after_open"])
                     for r, c in zip(roles, combo)}
            base = _stamp({"task": t, "init": i, "combo": combo, "executor": cal["executor"], "calibration_hash": cal["content_hash"]})
            rec = _episode_record(base, t, i, grips, raw / f"pair_init{i}_{combo}_{cal['content_hash'][:8]}.npz")
            rec["role_audit"] = {r: audit_role(rec["objects"].get(r, {}), cal["objects"][r], LV[c]) for r, c in zip(roles, combo)}
            rec["force_check_passed_single_trajectory"] = bool(rec.get("task_ok") and all(v["qualified"] for v in rec["role_audit"].values()))
            f.write_text(json.dumps(rec, indent=1, default=_J))
            print(f"T{t} i{i} {combo} {rec['status']} ok={rec['task_ok']} force_check={rec['force_check_passed_single_trajectory']} " + " ".join(
                f"{r[:5]}:{','.join(v['reasons']) or 'ok'}" for r, v in rec["role_audit"].items()), flush=True)


def cmd_summary(t):
    cal = json.loads(cfg_path(t).read_text())
    fs = sorted(out_dir(t, "pairs").glob("init*_*.json"))
    recs = [json.loads(f.read_text()) for f in fs]
    recs = [r for r in recs if r.get("calibration_hash") == cal["content_hash"]]
    n = len(recs)
    s = {"task": t, "calibration_hash": cal["content_hash"], "n_requests": n, "n_error": sum(r["status"] == "error" for r in recs),
         "task_ok": sum(bool(r.get("task_ok")) for r in recs), "force_check_passed_single_trajectory": sum(r["force_check_passed_single_trajectory"] for r in recs), "contact_policy_clean": sum(bool(r.get("contact_policy", {}).get("clean")) for r in recs), "per_role": {}}
    for role in cal["objects"]:
        rows = [r["role_audit"][role] for r in recs if role in r.get("role_audit", {})]
        rs = {"n": len(rows), "calibration_usable": cal["objects"][role]["calibration_usable"], "stable": sum(bool(x["grasp_stable"]) for x in rows),
              "in_band": sum(x["all_hold_steps_in_band"] is True for x in rows), "qualified": sum(x["qualified"] for x in rows),
              "peak_le_response_alert": sum(x["peak_le_response_alert"] is True for x in rows), "peak_alert_known": sum(x["peak_le_response_alert"] is not None for x in rows)}
        reasons = {}
        for x in rows:
            for q in x["reasons"]:
                reasons[q] = reasons.get(q, 0) + 1
        rs["reasons"] = reasons
        s["per_role"][role] = rs
    (ROOT / "reports" / f"step8_t{t}_pairs_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("response", "derive", "pairs", "summary"))
    ap.add_argument("task", type=int)
    ap.add_argument("--inits", type=int, nargs="+", default=list(CAL_INITS))
    ap.add_argument("--combos", nargs="+", default=["GF", "FG"])
    a = ap.parse_args()
    assert a.task in SPECS and a.task in (0, 7)
    {"response": lambda: cmd_response(a.task, a.inits), "derive": lambda: cmd_derive(a.task), "pairs": lambda: cmd_pairs(a.task, a.inits, a.combos),
     "summary": lambda: cmd_summary(a.task)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
