"""Step 23: copy of s22_t8_ordinary.py (same per-pot criteria, same GRIP, approach check on, NO guard) run with the FULL-POSE
grasp reference in SPECS[8] and the two opt-in stop flags (stop_on_stage_timeout, require_bilateral_grasp).  Output dir differs.
Step 22 text follows.
Step 22: T8 (libero_10 task 8, "put both moka pots on the stove") ORDINARY execution: no manner request, no force target,
GripPlan(1.0, 100.0, settle_after_open=1) for both pots (same plan as T4/T5 ordinary), approach check on, NO execution guard.
usage: s22_t8_ordinary.py TAG INIT...  -> reports/step22_t8_onboard/<TAG>/init{N}.json  (+ identity.json, written first)

Per pot (judged separately, thresholds fixed here before any run):
  grasp   = grasp policy reached the force band (reached_band) AND the hold window exists AND the s8 stability test holds
            (bilateral contact in every hold step, L/R imbalance <= 0.10, pot-in-hand z spread <= 1 cm);
  lift    = pot origin rises >= 0.05 m above its start during the lift/transport stages;
  transport = at the last transport step the pot origin is within 0.03 m (xy) of its planned target (site + target_offset_site);
  place   = after the retreat the pot origin is within 0.03 m (xy) of the planned target and within the stove cook-region site's
            half-extent in xy, and rests (z within 0.03 m of the planned target z).
Whole task: official success (ep.task_ok), status (any 'stage_timeout:*' is a failure), frozen contact policy v1.1 (clean_v1).
SUCCESS(init) = both pots have grasp, lift, transport, place AND task_ok AND completed (no timeout) AND clean_v1."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np, mujoco  # noqa: E402
from s7_task_config import ROOT, SPECS  # noqa: E402
import s8_task_force as S8  # noqa: E402
import s11_guarded_case as S11  # noqa: E402
import s7_task_config as S7  # noqa: E402
S8.make_plan = lambda t, prog, grip=None, **kw: S7.make_plan(t, prog, grip=grip, stop_on_stage_timeout=True, require_bilateral_grasp=True, **kw)
from synthesis_v2.contact_policy_v1 import POLICY_VERSION, evaluate as eval_v1  # noqa: E402

T = 8
TAG = sys.argv[1]
OUT = ROOT / "reports" / "step23_t8_fullpose" / TAG
OUT.mkdir(parents=True, exist_ok=True)
GRIP = S8.grip_for((1.0, 100.0), 1)
SITE = "flat_stove_1_cook_region"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
DEMO = "/workdata/cap-x/capx/third_party/LIBERO-PRO/libero/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
ident = {"git_head": git("rev-parse", "HEAD"), "git_dirty_files": [l for l in git("status", "--short").splitlines() if "upgrade.txt" not in l],
         "sha256": {p: sha(ROOT / p) for p in ("scripts/mt/s7_task_config.py", "scripts/mt/s22_t8_ordinary.py", "scripts/mt/s22_t8_select_demo.py",
                                                  "scripts/mt/s22_t8_demo_extract.py", "reports/step22_t8_onboard/demo_selection.json", "synthesis_v2/approach_planner.py", "scripts/mt/s23_full_pose_ref.py",
                                                  "synthesis_v2/program_executor.py", "synthesis_v2/contact_policy_v1.py")},
         "demo_file_sha256": sha(DEMO), "policy_version": POLICY_VERSION, "grip": [1.0, 100.0, "settle_after_open=1"],
         "approach_check": True, "execution_guard": None, "stop_on_stage_timeout": True, "require_bilateral_grasp": True, "spec": repr(SPECS[T])}
(OUT / "identity.json").write_text(json.dumps(ident, indent=1))
print("identity", ident["git_head"], ident["git_dirty_files"], flush=True)

CHECK = {"lift_min_rise_m": 0.05, "xy_tol_m": 0.03, "z_tol_m": 0.03}


def site_frame(init):
    """cook-region site pose (world) and half-size of this init (fixture: static); read from a fresh session of the same init."""
    from s7_task_config import make_program, make_session
    ss = make_session(T, init, make_program(T))
    pos, mat = ss.site_pose(SITE)
    m, _ = ss._mj()
    half = m.site_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)].copy()
    ss.close()
    return np.asarray(pos, float), np.asarray(mat, float).reshape(3, 3), half


for i in [int(x) for x in sys.argv[2:]]:
    sp, sm, half = site_frame(i)
    grips = {r: GRIP for r, _ in SPECS[T].objects}
    prog, ep, rec, crec, det, dt = S11.run(T, i, grips, None)
    a = ep.arrays
    st = np.asarray([str(x) for x in a["stage"]])
    per = S8.per_object(ep, prog, T, rec)
    wins = S11.windows_from(ep, T)
    v0 = rec["contact_audit"]
    v1 = eval_v1(v0, crec.events, det.detail, wins, st.tolist())
    pots, np_ = {}, {}
    for k, (role, name) in enumerate(SPECS[T].objects):
        P = np.asarray(a["pos_" + name])
        off = np.asarray(SPECS[T].object_target_params[k]["target_offset_site"], float)
        rel = (sm.T @ (P - sp).T).T                       # pot origin in the site frame, per step
        lab = lambda p: f"{role}.pick_1.{p}"               # noqa: E731
        sel = lambda *ps: np.flatnonzero(np.isin(st, [lab(p) for p in ps]))  # noqa: E731
        lt, tr = sel("lift", "transport"), sel("transport")
        rise = float(P[lt, 2].max() - P[0, 2]) if len(lt) else None
        g = (ep.meta.get("grasp_policy_by_instance") or {}).get(lab("grasp"), {})
        po = per[role]
        grasp_ok = bool(g.get("reached_band")) and po.get("status") == "ok" and bool(po.get("grasp_stable"))
        lift_ok = rise is not None and rise >= CHECK["lift_min_rise_m"]
        e_tr = None if not len(tr) else float(np.hypot(*(rel[tr[-1], :2] - off[:2])))
        transport_ok = e_tr is not None and e_tr <= CHECK["xy_tol_m"]
        e_pl = float(np.hypot(*(rel[-1, :2] - off[:2]))); dz_pl = float(rel[-1, 2] - off[2])
        in_region = bool(np.all(np.abs(rel[-1, :2]) <= half[:2]))
        place_ok = bool(e_pl <= CHECK["xy_tol_m"] and abs(dz_pl) <= CHECK["z_tol_m"] and in_region)
        pots[role] = {"object": name, "planned_target_site_frame": off.tolist(), "start_world": P[0].round(4).tolist(),
                      "final_site_frame": rel[-1].round(4).tolist(), "final_xy_err_m": round(e_pl, 4), "final_dz_m": round(dz_pl, 4),
                      "final_in_region_xy": in_region, "transport_end_xy_err_m": None if e_tr is None else round(e_tr, 4),
                      "max_rise_lift_transport_m": None if rise is None else round(rise, 4), "grasp_policy": g,
                      "grasp_measure": {k_: po.get(k_) for k_ in ("status", "measurement_valid", "grasp_stable", "hold_n", "hold_min", "hold_max",
                                                                   "lr_imbalance", "rel_z_spread_m", "stability_components")},
                      "grasp": grasp_ok, "lift": lift_ok, "transport": transport_ok, "place": place_ok}
        np_["pos_" + name] = P
    timeout = ep.status.startswith("stage_timeout")
    done_all = all(p[x] for p in pots.values() for x in ("grasp", "lift", "transport", "place"))
    success = bool(done_all and ep.task_ok and not timeout and ep.status == "completed" and v1["clean_v1"])
    out = {"schema": "mannercap.t8_ordinary.v1", "init": i, "SUCCESS": success, "status": ep.status, "task_ok": bool(ep.task_ok),
           "timeout": timeout, "n_steps": int(ep.n_steps), "wall_s": round(dt, 1), "checks": CHECK, "site_world": sp.tolist(),
           "site_half_size": half.tolist(), "v0_clean": v0["clean"], "v0_forbidden_by_class": v0["forbidden_by_class"],
           "clean_v1": v1["clean_v1"], "clean_v1_ignoring_support": v1["clean_v1_ignoring_support"], "n_excused": v1["n_excused"],
           "n_contacts": len(v1["contacts"]),
           "unexcused": [{k_: c.get(k_) for k_ in ("robot_body", "other", "max_n", "control_step_range", "stages", "v1_reasons")}
                         for c in v1["unexcused_contacts"]],
           "support_unexplained": v1["support_contacts_unexplained"], "windows": wins, "pots": pots,
           "approach_plan": ep.meta.get("approach_plan")}
    np.savez_compressed(OUT / f"init{i}_pots.npz", stage=st, **np_)
    (OUT / f"init{i}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"INIT {i} SUCCESS={success} status={ep.status} task_ok={ep.task_ok} steps={ep.n_steps} wall={dt:.0f}s clean_v1={v1['clean_v1']} "
          + " | ".join(f"{r}: g={p['grasp']} l={p['lift']} t={p['transport']} p={p['place']} err={p['final_xy_err_m']}" for r, p in pots.items()), flush=True)
