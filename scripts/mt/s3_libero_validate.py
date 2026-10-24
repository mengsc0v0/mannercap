"""Step 3 validation: one LIBERO task through shared primitives on frozen test initial states.

Per initial state: completion, measurement credibility (per grasp), exported-action replay in a fresh env,
mid-hold snapshot resume, and state/contact-event stage identification (no generator labels as input)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_libero_task import OBJECTS, ROOT, make_program, make_session, run  # noqa: E402
from synthesis_v2.contact_events import CONTACT_THR_N, ContactEventTracker, run_tracker  # noqa: E402
from synthesis_v2.native_osc_env import state8  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402

DEV_INIT = tuple(range(8))
GRIP = GripPlan(8.0, 40.0)  # LIBERO gripper: ~10 N per close step; native gentle/firm bands do not transfer
TRIM = 2                    # hold_trim_head_steps (configs/manner_calibration.v3.json)
NAMES = tuple(n for _, n in OBJECTS)
FORCE_COLS = ("left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "other_max_n", "valid_fraction")
OUT = ROOT / "reports" / "step3_libero_20261025"


def eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def code_version() -> dict:
    def sh(*c):
        return subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout
    h = hashlib.sha256(sh("git", "diff", "HEAD").encode())
    for f in sorted(sh("git", "ls-files", "--others", "--exclude-standard").split()):
        if f.startswith(("synthesis_v2/", "scripts/mt/", "tests/", "configs/")) and f.endswith((".py", ".json", ".yaml")):
            h.update(f.encode()); h.update((ROOT / f).read_bytes())
    return {"head": sh("git", "rev-parse", "HEAD").strip(), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD").strip(),
            "working_tree_sha256": h.hexdigest()}


def roll(i, prog, actions, ft, k0=0, st=None, save_at=(), online=False):
    """Step exported actions in a fresh LIBERO session (reset, optionally restore a snapshot at k0)."""
    s = make_session(i, prog)
    s.reset()
    if st is not None:
        s.restore_state(st)
    import mujoco
    m, d = s._mj()
    aids = [j for j in range(m.nu) if "gripper0" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) or "")]
    cols, saved = defaultdict(list), {}
    tr = ContactEventTracker(NAMES) if online else None
    for k in range(k0, len(actions)):
        if k in save_at:
            saved[k] = s.save_state()
        name = str(ft[k])
        if name and name != s.force_target_name:
            s.set_force_target(name)
        before = {n: np.asarray(s.obs[n + "_pos"], float).copy() for n in NAMES}
        obs, tel = s.step(np.asarray(actions[k], np.float32))
        cf = s.object_contacts(NAMES)
        for c in FORCE_COLS:
            cols[c].append(getattr(tel, c))
        cols["next_state"].append(state8(obs).astype(np.float32))
        cols["success"].append(s.success())
        cols["af_total"].append(float(sum(abs(d.actuator_force[j]) for j in aids)))  # independent of contact reading
        for n in NAMES:
            cols["cf_L_" + n].append(cf[n][0]); cols["cf_R_" + n].append(cf[n][1])
            cols["pos_" + n].append(before[n]); cols["pa_" + n].append(np.asarray(obs[n + "_pos"], float).copy())
        if tr:
            tr.update(cf, {n: obs[n + "_pos"] for n in NAMES})
    return {k: np.asarray(v) for k, v in cols.items()}, saved, tr, s


def compare(a_gen: dict, cols: dict, k0: int = 0) -> dict:
    return {k: eq(np.asarray(a_gen[k])[k0:], cols[k]) for k in cols if k in a_gen}


def resume_points(a, prog) -> dict:
    st = np.asarray(a["stage"])
    pts = {}
    for role, _ in OBJECTS:
        lift, tr, pl = (np.flatnonzero(st == f"{role}.pick_1.{p}") for p in ("lift", "transport", "place"))
        pts[f"{role}:lift+1(in_trim)"] = int(lift[0]) + 1
        pts[f"{role}:lift+{TRIM + 3}(in_window)"] = int(lift[0]) + TRIM + 3
        pts[f"{role}:transport_mid"] = int(tr[len(tr) // 2])
        pts[f"{role}:transport_last-3"] = int(tr[-1]) - 3
        pts[f"{role}:place+3(opening)"] = int(pl[0]) + 3
    return pts


def hold_idx(a, role) -> np.ndarray:
    st = np.asarray(a["stage"])
    lift = np.flatnonzero(st == f"{role}.pick_1.lift")
    tr = np.flatnonzero(st == f"{role}.pick_1.transport")
    return np.concatenate([lift[TRIM:], tr])


def measure_checks(s, a, role, name, other) -> dict:
    import mujoco
    from synthesis_v2.grasp_force_contract import resolve_grasp_binding

    m = s._mj()[0]
    h = hold_idx(a, role)
    L, R = np.asarray(a["left_mean_n"])[h], np.asarray(a["right_mean_n"])[h]
    ids = s.geom_ids_of(name)
    b = resolve_grasp_binding(m, mujoco, ids, name)
    comp = b.is_complete() if callable(getattr(b, "is_complete", None)) else bool(b.is_complete)
    # independent physical consistency: friction capacity of the measured normal force vs object weight
    pads = list(b.left_finger_geom_ids) + list(b.right_finger_geom_ids)
    mu = min(max(float(m.geom_friction[p, 0]), float(m.geom_friction[o, 0])) for p in pads for o in ids)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s.env.objects_dict[name].root_body)
    weight = float(m.body_subtreemass[body]) * 9.81
    cap = mu * float(np.mean(L + R))
    z_rel = np.asarray(a["pos_" + name])[h, 2] - np.asarray(a["state"])[h, 2]
    pre = np.flatnonzero(np.isin(np.asarray(a["stage"]), [f"{role}.pick_1.approach", f"{role}.pick_1.descend"]))
    pre_f = np.nan_to_num(np.maximum(np.asarray(a["cf_L_" + name])[pre], np.asarray(a["cf_R_" + name])[pre]))
    cross = np.nan_to_num(np.maximum(np.asarray(a["cf_L_" + other])[h], np.asarray(a["cf_R_" + other])[h]))
    return {
        "binding_complete_and_correct_target": bool(comp and len(ids) > 0 and set(np.asarray(a["force_target"])[h]) == {name}),
        "valid_fraction_ge_0.99": bool(np.asarray(a["valid_fraction"])[h].min() >= 0.99),
        "bilateral_every_hold_step": bool(min(L.min(), R.min()) > CONTACT_THR_N),
        "left_right_balance_le_0.10": bool(np.mean(np.abs(L - R) / (L + R)) <= 0.10),
        "friction_capacity_ge_weight": bool(cap >= weight),
        "no_slip_rel_z_spread_le_1cm": bool(np.ptp(z_rel) <= 0.01),
        "no_force_before_grasp": bool(pre_f.max() <= CONTACT_THR_N),
        "no_crosstalk_other_object": bool(cross.max() <= CONTACT_THR_N),
        "_values": {"hold_n": int(len(h)), "total_force_mean_n": float(np.mean(L + R)), "lr_imbalance": float(np.mean(np.abs(L - R) / (L + R))),
                    "mu_min": mu, "weight_n": weight, "capacity_over_weight": cap / weight, "rel_z_spread_m": float(np.ptp(z_rel)),
                    "pre_grasp_max_n": float(pre_f.max()), "other_obj_max_n": float(cross.max())},
    }


PHASE_OF = {"approach": "free", "descend": "free", "retreat": "free", "grasp": "grasp", "lift": "carry", "transport": "carry", "place": None}


def event_checks(a) -> tuple[dict, tuple]:
    ph, ob, ev = run_tracker(a, NAMES)
    st = np.asarray(a["stage"])
    exp = [(n, e) for _, n in OBJECTS for e in ("contact_established", "lifted", "released")]
    got = [(e["object"], e["event"]) for e in ev]
    win_ok, delays = True, {}
    for role, n in OBJECTS:
        idx = lambda p: np.flatnonzero(st == f"{role}.pick_1.{p}")
        g, l, p_ = idx("grasp"), idx("lift"), idx("place")
        t = {e["event"]: e["step"] for e in ev if e["object"] == n}
        if len(t) == 3:
            ok = g[0] <= t["contact_established"] <= g[-1] + 2 and l[0] <= t["lifted"] <= l[0] + 10 and p_[0] <= t["released"] <= p_[-1] + 3
            delays[n] = {"contact_established_after_grasp_start": int(t["contact_established"] - g[0]),
                         "lifted_after_lift_start": int(t["lifted"] - l[0]), "released_after_place_start": int(t["released"] - p_[0])}
            win_ok &= bool(ok)
        else:
            win_ok = False
    exp_ph = np.array([PHASE_OF[x.rsplit(".", 1)[1]] or "?" for x in st])
    bad = np.zeros(len(st), bool)
    edge = np.flatnonzero(exp_ph[1:] != exp_ph[:-1]) + 1
    near = np.zeros(len(st), bool)
    for e in edge:
        near[max(0, e - 6): e + 6] = True
    bad = (exp_ph != "?") & (exp_ph != ph) & ~near
    return {"event_sequence_exact": got == exp, "events_inside_expected_generator_stage": bool(win_ok),
            "phase_mismatch_steps_outside_6step_boundary": int(bad.sum()), "_delays_steps": delays,
            "_events": [{k: (int(v) if isinstance(v, (np.integer, int)) else v) for k, v in e.items()} for e in ev]}, (ph, ob, ev)


def one_init(i: int, grip=GRIP, out=OUT, tag: str = "") -> dict:
    prog = make_program(2)
    box: dict = {}
    _, s, ep = run(i, 2, False, grip, on_session=lambda ss: box.update(st0=ss.save_state()))
    a = ep.arrays
    n = len(a["actions"])
    rec: dict = {"init_index": i, "n_steps": n, "status": ep.status, "task_ok": bool(ep.task_ok),
                 "final_success": bool(a["success"][-1]),
                 "grasp_policy": ep.meta.get("grasp_policy_by_instance")}
    rec["completion"] = bool(ep.status == "completed" and ep.task_ok and a["success"][-1])
    rec["measure"] = {r: measure_checks(s, a, r, nm, NAMES[1 - k]) for k, (r, nm) in enumerate(OBJECTS)}
    rec["measure_pass"] = {r: all(v for k, v in c.items() if not k.startswith("_")) for r, c in rec["measure"].items()}
    s.close()
    np.savez_compressed(out / f"gen_init{i}{tag}.npz", **{k: v for k, v in a.items() if k != "images"})
    ev, (ph, ob, evs) = event_checks(a)
    rec["events"] = ev
    pts = resume_points(a, prog)
    # replay in a fresh env from exported actions (also the continuous reference for resume), tracker run online
    cols, saved, tr, s2 = roll(i, prog, a["actions"], a["force_target"], save_at=set(pts.values()), online=True)
    s2.close()
    rep = compare(a, {k: v for k, v in cols.items() if k != "af_total"})
    for role, nm in OBJECTS:  # independent estimate: finger actuator force at quasi-static hold vs measured contact normal
        h = hold_idx(a, role)
        tot = np.asarray(cols["cf_L_" + nm])[h] + np.asarray(cols["cf_R_" + nm])[h]
        err = np.abs(np.asarray(cols["af_total"])[h] - tot) / tot
        ok = bool(np.nanmean(err) <= 0.10)
        rec["measure"][role]["actuator_force_cross_check_le_10pct"] = ok
        rec["measure"][role]["_values"]["actuator_vs_contact_rel_err_mean"] = float(np.nanmean(err))
        rec["measure"][role]["_values"]["actuator_total_mean_n"] = float(np.mean(np.asarray(cols["af_total"])[h]))
    rec["measure_pass"] = {r: all(v for k, v in c.items() if not k.startswith("_")) for r, c in rec["measure"].items()}
    rec["replay"] = {"bitwise_all_columns": bool(all(rep.values())), "columns_not_equal": [k for k, v in rep.items() if not v],
                     "n_columns": len(rep)}
    rec["online_equals_offline_events"] = bool(tr.events == evs)
    res = []
    for name, k in pts.items():
        c2, _, _, s3 = roll(i, prog, a["actions"], a["force_target"], k0=k, st=saved[k])
        s3.close()
        d = compare({kk: np.asarray(v)[k:] for kk, v in cols.items()}, c2)
        res.append({"point": name, "k": k, "suffix_len": n - k, "bitwise": bool(all(d.values())),
                    "columns_not_equal": [kk for kk, v in d.items() if not v]})
    rec["resume"] = res
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(DEV_INIT))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summary:
        return summarize()
    for i in args.inits:
        rec = one_init(i)
        (OUT / f"init{i}.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(i, "completion", rec["completion"], "measure", rec["measure_pass"], "replay", rec["replay"]["bitwise_all_columns"],
              "resume", sum(r["bitwise"] for r in rec["resume"]), "/", len(rec["resume"]),
              "events", rec["events"]["event_sequence_exact"], rec["events"]["events_inside_expected_generator_stage"],
              rec["events"]["phase_mismatch_steps_outside_6step_boundary"], "online==offline", rec["online_equals_offline_events"], flush=True)
    return 0


def summarize() -> int:
    recs = [json.loads((OUT / f"init{i}.json").read_text()) for i in DEV_INIT if (OUT / f"init{i}.json").exists()]
    N = len(recs)
    sub = defaultdict(lambda: [0, 0])
    for r in recs:
        for c in r["measure"].values():
            for k, v in c.items():
                if not k.startswith("_"):
                    sub[k][0] += bool(v); sub[k][1] += 1
    out = {"task": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket (libero_10 #1)",
           "test_initial_states": [r["init_index"] for r in recs], "code": code_version(),
           "completion": [sum(r["completion"] for r in recs), N],
           "measure_grasps_all_checks": [sum(v for r in recs for v in r["measure_pass"].values()), 2 * N],
           "measure_subchecks": {k: v for k, v in sub.items()},
           "replay_bitwise": [sum(r["replay"]["bitwise_all_columns"] for r in recs), N],
           "resume_bitwise_points": [sum(x["bitwise"] for r in recs for x in r["resume"]), sum(len(r["resume"]) for r in recs)],
           "event_sequence_exact": [sum(r["events"]["event_sequence_exact"] for r in recs), N],
           "events_inside_generator_stage": [sum(r["events"]["events_inside_expected_generator_stage"] for r in recs), N],
           "phase_zero_mismatch_outside_6step": [sum(r["events"]["phase_mismatch_steps_outside_6step_boundary"] == 0 for r in recs), N],
           "online_equals_offline_events": [sum(r["online_equals_offline_events"] for r in recs), N],
           "total_steps": [r["n_steps"] for r in recs]}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
