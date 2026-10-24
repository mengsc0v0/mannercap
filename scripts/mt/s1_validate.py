"""Step 1 validation: (1) exported-action dynamic replay, (2) resume-statistics consistency,
(3) fixed-configuration multi-seed run over the four object-level x order combinations.

Replay contract: a *fresh* session (different construction seed) restores the exported initial state,
then executes the exported 7D actions through the normal step interface.  The generator's contact
feedback is NOT consulted; stage labels / force-target names are evaluator metadata read from the
export (they decide what is measured, never what is commanded).  Forces are re-measured and everything
is re-scored with the same functions as the generation-time scoring.

Version discipline: the manifest (seeds, plans, calibration/code hashes, criteria) is written before
a phase starts and a phase refuses to overwrite an existing manifest without --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import s1_force_diag as fd  # noqa: E402
from synthesis_v2.manner_calibration import grip_bound  # noqa: E402
from synthesis_v2.native_osc_env import NativeSession, state8  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402

ORDERS = (("cubeA", "cubeB"), ("cubeB", "cubeA"))
LEVELS = ({"cubeA": "gentle", "cubeB": "firm"}, {"cubeA": "firm", "cubeB": "gentle"})
PACK_SEED = 0
DEV_SEEDS = list(range(100, 108))
REPLAY_SEED = 7777  # construction seed of the replay session; differs from every generation seed
STABLE_STEP_M = 1e-3  # settle-step displacement below which the settled reference counts as stable
FORCE_COLS = ("left_mean_n", "right_mean_n", "left_max_n", "right_max_n", "valid_fraction")
CODE_FILES = ["synthesis_v2/native_osc_env.py", "synthesis_v2/native_producer.py", "synthesis_v2/program_executor.py",
              "synthesis_v2/task_program.py", "synthesis_v2/grasp_force_contract.py", "synthesis_v2/manner_calibration.py",
              "configs/manner_calibration.v3.json", "scripts/mt/s1_force_diag.py", "scripts/mt/s1_validate.py"]


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sha(p) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]


def write_manifest(out: Path, phase: str, seeds: list, force: bool) -> None:
    path = out / "manifest.json"
    if path.exists() and not force:
        raise SystemExit(f"manifest exists: {path} (new config => new version dir, or --force)")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    man = {"phase": phase, "seeds": seeds, "orders": ORDERS, "levels": LEVELS, "replay_seed": REPLAY_SEED,
           "profile": "prod", "plans": {k: [v.target_lo_n, v.target_hi_n] for k, v in fd.LEVEL_PLAN.items()},
           "calibration": "configs/manner_calibration.v3.json", "hold_trim_head_steps": fd.TRIM,
           "task_criteria": {"xy_tol": fd.XY_TOL, "z_tol": fd.Z_TOL, "lift_min": fd.LIFT_MIN, "disturb_tol": fd.DISTURB_TOL,
                             "settle_ref_step": fd.SETTLE, "stable_step_m": STABLE_STEP_M},
           "safety": {"peak_cap_n": fd.CAL["safety"]["stack"]["peak_cap_n"], "def": "max over steps of left_max+right_max (NaN->0)"},
           "git_head": git("rev-parse", "HEAD"), "git_dirty_files": len(git("status", "--short").splitlines()),
           "file_sha256_16": {f: sha(f) for f in CODE_FILES}}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(path, "w"), indent=2, default=_j)


# ---------------------------------------------------------------- scoring
def hold_detail(a, prog, gid: str, level: str) -> dict:
    lo, hi, _ = grip_bound(fd.CAL, "stack", level)
    idx = fd.hold_idx(a, prog, gid)
    L, R = np.asarray(a["left_mean_n"])[idx], np.asarray(a["right_mean_n"])[idx]
    full = np.asarray(a["valid_fraction"])[idx] >= 1 - 1e-9
    tot = L + R
    fin = np.isfinite(tot) & full
    thr = fd.CAL["safety"]["stack"]["contact_threshold_n"]
    bil = fin & (L > thr) & (R > thr)
    d = {"gid": gid, "level": level, "band": [round(lo, 3), round(hi, 3)], "n_hold": int(len(idx)), "n_valid": int(full.sum()),
         "n_nan": int((~np.isfinite(tot)).sum()), "n_missing_contact": int((~bil).sum()),
         "n_below": int((fin & (tot < lo)).sum()), "n_above": int((fin & (tot > hi)).sum())}
    if fin.any():
        q = np.percentile(tot[fin], [5, 50, 95])
        d.update(min=round(float(tot[fin].min()), 3), max=round(float(tot[fin].max()), 3), p5=round(float(q[0]), 3),
                 p50=round(float(q[1]), 3), p95=round(float(q[2]), 3),
                 margin_lo=round(float(tot[fin].min() - lo), 3), margin_hi=round(float(hi - tot[fin].max()), 3))
    au = fd.grip_audit(a, prog, gid, level)
    d["audit_decision"], d["audit_reason"] = au["decision"], au["reason"]
    return d


def safety(a) -> dict:
    cap = fd.CAL["safety"]["stack"]["peak_cap_n"]
    pk = np.nan_to_num(np.asarray(a["left_max_n"], float), nan=0.0) + np.nan_to_num(np.asarray(a["right_max_n"], float), nan=0.0)
    return {"cap_n": cap, "peak_total_n": round(float(pk.max()), 3), "ok": bool(pk.max() <= cap),
            "argmax_stage": str(np.asarray(a["stage"])[int(pk.argmax())])}


def settle_info(a, prog) -> dict:
    st = np.asarray(a["stage"])
    grasp_labels = [i.label for i in prog.instances if i.primitive == "grasp"]
    fg = int(np.flatnonzero(np.isin(st, grasp_labels))[0])
    out = {"settle_ref_step": fd.SETTLE, "first_grasp_step": fg, "settle_before_first_grasp": bool(fg > fd.SETTLE), "ref_pos": {}}
    for r in ("cubeA", "cubeB"):
        p = np.asarray(a["pos_" + r])
        disp = float(np.linalg.norm(p[fd.SETTLE] - p[fd.SETTLE - 1]))
        out[r + "_settle_step_disp_m"] = round(disp, 6)
        out[r + "_stable"] = bool(disp <= STABLE_STEP_M)
        out["ref_pos"][r] = p[fd.SETTLE].tolist()
    return out


def score(a, prog, levels: dict, status: str, final_pos: dict) -> dict:
    ep = SimpleNamespace(arrays=a, meta={"final_obj_pos_by_role": final_pos})
    holds, bound = {}, True
    for r, lv in sorted(levels.items()):
        gid = f"{r}.pick_1.grasp"
        h = hold_detail(a, prog, gid, lv)
        b = fd.target_binding_check(a, prog, gid, r)
        h["binding"] = {k: b[k] for k in ("steps_in_scope", "steps_measuring_expected", "steps_hold_instance_is_self", "bound_ok")}
        bound &= b["bound_ok"]
        holds[r] = h
    task = fd.task_check(ep, prog)
    manner = all(h["audit_decision"] == "PASS" for h in holds.values())
    saf = safety(a)
    executed = status == "completed"
    return {"execution_end": executed, "task_ok": task["task_ok"], "manner_pass": manner, "safety_ok": saf["ok"],
            "bound_ok": bool(bound), "joint_pass": bool(executed and task["task_ok"] and manner and saf["ok"] and bound),
            "holds": holds, "task": task, "safety": saf, "settle": settle_info(a, prog)}


# ---------------------------------------------------------------- replay machinery
def make_init(seed: int):
    s = NativeSession("stack", seed, render=False)
    s.reset()
    st0, q0 = s.save_state(), np.asarray(s.initial["qpos"], float)
    s.close()
    return st0, q0


def run_actions(st_init, actions, ft, k0: int, k1: int, save_at=()):
    s = NativeSession("stack", REPLAY_SEED, render=False)
    s.reset()
    s.restore_state(st_init)
    cols = {k: [] for k in FORCE_COLS + ("next_state",)}
    pos = {r: [] for r in ("cubeA", "cubeB")}
    saved = {}
    for i in range(k0, k1):
        if i in save_at:
            saved[i] = s.save_state()
        name = str(ft[i])
        if name and name != s.force_target_name:
            s.set_force_target(name)
        for r in pos:
            pos[r].append(np.asarray(s.obs[f"{r}_pos"], float))
        obs, tel = s.step(np.asarray(actions[i], np.float32))
        for k in FORCE_COLS:
            cols[k].append(getattr(tel, k))
        cols["next_state"].append(state8(obs))
    final = {r: np.asarray(s.obs[f"{r}_pos"], float).tolist() for r in pos}
    s.close()
    return ({k: np.asarray(v, float) for k, v in cols.items()}, {r: np.asarray(v, float) for r, v in pos.items()}, final, saved)


def col_diff(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mism = int((np.isnan(x) != np.isnan(y)).sum())
    both = np.isfinite(x) & np.isfinite(y)
    return {"max_abs": float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0, "nan_mismatch": mism}


def replay_and_score(a_src: dict, st0, prog, levels) -> tuple[dict, dict]:
    n = len(a_src["actions"])
    cols, pos, final, _ = run_actions(st0, a_src["actions"], a_src["force_target"], 0, n)
    arr = dict(a_src)
    for k in FORCE_COLS:
        arr[k] = cols[k]
    for r in pos:
        arr["pos_" + r] = pos[r]
    diffs = {k: col_diff(a_src[k], cols[k]) for k in FORCE_COLS}
    diffs["next_state_max_abs"] = float(np.abs(np.asarray(a_src["next_state"], float) - cols["next_state"]).max())
    diffs["pos_max_abs"] = max(float(np.abs(np.asarray(a_src["pos_" + r], float) - pos[r]).max()) for r in pos)
    rec = score(arr, prog, levels, "completed", final)
    rec["replay_diff"] = diffs
    return rec, arr


def same_verdict(r1: dict, r2: dict) -> dict:
    keep = ("n_hold", "n_valid", "n_nan", "n_missing_contact", "n_below", "n_above", "audit_decision", "audit_reason")
    holds_eq = all(all(r1["holds"][r][k] == r2["holds"][r][k] for k in keep) for r in r1["holds"])
    return {"holds_equal": bool(holds_eq),
            **{k: r1[k] == r2[k] for k in ("task_ok", "manner_pass", "safety_ok", "bound_ok", "joint_pass")}}


# ---------------------------------------------------------------- resume check
def resume_points(a, prog, levels) -> dict:
    """Resume points inside each grasp's hold window.  Only stages that actually produced steps are used
    (the pick_place template's transport instance has no steps of its own in this program: lift carries it)."""
    st = np.asarray(a["stage"])
    pts = {}
    for r in sorted(levels):
        gid = f"{r}.pick_1.grasp"
        for i in prog.grip_scope(gid):
            it = prog.instance(i)
            if it.primitive not in ("lift", "transport"):
                continue
            ix = np.flatnonzero(st == it.label)
            if len(ix) < fd.TRIM + 6:
                continue
            nm = it.primitive
            pts[f"{r}:{nm}+1(in_trim)"] = int(ix[0]) + 1
            pts[f"{r}:{nm}+{fd.TRIM + 3}(in_window)"] = int(ix[0]) + fd.TRIM + 3
            pts[f"{r}:{nm}_mid"] = int(ix[len(ix) // 2])
            pts[f"{r}:{nm}_last-3"] = int(ix[-1]) - 3
    return pts


def naive_extra_dropped(a, prog, levels, k: int) -> dict:
    """Samples a suffix-only audit would drop because the trim restarts at the resume point."""
    st = np.asarray(a["stage"])
    out = {}
    for r in sorted(levels):
        gid = f"{r}.pick_1.grasp"
        cont = int((fd.hold_idx(a, prog, gid) >= k).sum())
        naive = sum(len(np.flatnonzero(st[k:] == prog.instance(i).label)[fd.TRIM:]) for i in prog.grip_scope(gid)
                    if prog.instance(i).primitive in ("lift", "transport"))
        out[r] = cont - naive
    return out


def resume_check(a_src: dict, st0, prog, levels) -> list:
    n = len(a_src["actions"])
    pts = resume_points(a_src, prog, levels)
    cols, pos, final, saved = run_actions(st0, a_src["actions"], a_src["force_target"], 0, n, save_at=set(pts.values()))
    base = dict(a_src)
    for k in FORCE_COLS:
        base[k] = cols[k]
    for r in pos:
        base["pos_" + r] = pos[r]
    ref = score(base, prog, levels, "completed", final)
    out = []
    for name, k in pts.items():
        c2, p2, f2, _ = run_actions(saved[k], a_src["actions"], a_src["force_target"], k, n)
        st_arr = dict(base)
        for key in FORCE_COLS:
            st_arr[key] = np.concatenate([cols[key][:k], c2[key]])
        for r in p2:
            st_arr["pos_" + r] = np.concatenate([pos[r][:k], p2[r]])
        rec = score(st_arr, prog, levels, "completed", f2)
        out.append({"point": name, "k": k, "suffix_len": n - k,
                    "force_cols_vs_continuous": {key: col_diff(cols[key][k:], c2[key]) for key in FORCE_COLS},
                    "verdict_vs_continuous": same_verdict(ref, rec),
                    "naive_suffix_only_extra_dropped": naive_extra_dropped(base, prog, levels, k)})
    return out


# ---------------------------------------------------------------- phases
def gen_one(seed: int, order, levels, outdir: Path, init) -> tuple[dict, dict, object, dict]:
    prog = fd.program(order)
    ep = produce_program_episode(prog, seed, fd.plan_for(prog, levels), keep_images=False)
    a = {k: v for k, v in ep.arrays.items() if k != "images"}
    tag = f"{order[0][-1]}{order[1][-1]}_{levels['cubeA']}A_{levels['cubeB']}B_s{seed}"
    np.savez_compressed(outdir / f"export_{tag}.npz", **a)
    st0, q0 = init
    gen = score(a, prog, levels, ep.status, ep.meta["final_obj_pos_by_role"])
    gen.update(tag=tag, seed=seed, order="->".join(order), levels=levels, n_steps=int(len(a["actions"])),
               init_qpos_matches_generator=float(np.abs(q0 - np.asarray(ep.meta["initial_state"]["qpos"])).max()))
    return gen, a, prog, {"tag": tag}


def phase_pack(out: Path) -> None:
    init = make_init(PACK_SEED)
    pickle.dump(init, open(out / f"init_s{PACK_SEED}.pkl", "wb"))
    recs = []
    for order in ORDERS:
        for lv in LEVELS:
            gen, a, prog, meta = gen_one(PACK_SEED, order, lv, out, init)
            exported = dict(np.load(out / f"export_{meta['tag']}.npz"))  # replay reads the exported file
            rp, _ = replay_and_score(exported, init[0], prog, lv)
            rs = resume_check(exported, init[0], prog, lv)
            rec = {"gen": gen, "replay": rp, "replay_same_verdict_as_gen": same_verdict(gen, rp), "resume": rs}
            recs.append(rec)
            ok_r = all(x["verdict_vs_continuous"]["holds_equal"] and x["verdict_vs_continuous"]["joint_pass"] == rp["joint_pass"] for x in rs)
            print(json.dumps({"tag": meta["tag"], "gen_joint": gen["joint_pass"], "replay_joint": rp["joint_pass"],
                              "replay_same": rec["replay_same_verdict_as_gen"], "replay_diff": rp["replay_diff"],
                              "resume_points": len(rs), "resume_all_consistent": ok_r}, default=_j), flush=True)
            json.dump(recs, open(out / "validation.json", "w"), indent=1, default=_j)
    n = len(recs)
    print(f"SUMMARY pack runs={n} gen_joint={sum(r['gen']['joint_pass'] for r in recs)} replay_joint={sum(r['replay']['joint_pass'] for r in recs)} "
          f"replay_same_verdict={sum(all(r['replay_same_verdict_as_gen'].values()) for r in recs)} "
          f"resume_points={sum(len(r['resume']) for r in recs)} resume_consistent="
          f"{sum(x['verdict_vs_continuous']['holds_equal'] for r in recs for x in r['resume'])}", flush=True)


def pair_ok(runs_same_order: list, cube: str) -> bool:
    g = [r for r in runs_same_order if r["levels"][cube] == "gentle"][0]
    f = [r for r in runs_same_order if r["levels"][cube] == "firm"][0]
    return bool(g["holds"][cube]["audit_decision"] == "PASS" and f["holds"][cube]["audit_decision"] == "PASS"
                and g["joint_pass"] and f["joint_pass"] and f["holds"][cube]["p50"] > g["holds"][cube]["p50"])


def phase_seeds(out: Path, seeds: list) -> None:
    recs = []
    for sd in seeds:
        init = make_init(sd)
        runs = []
        for order in ORDERS:
            for lv in LEVELS:
                gen, a, prog, meta = gen_one(sd, order, lv, out, init)
                exported = dict(np.load(out / f"export_{meta['tag']}.npz"))
                rp, _ = replay_and_score(exported, init[0], prog, lv)
                gen["replay"] = {k: rp[k] for k in ("execution_end", "task_ok", "manner_pass", "safety_ok", "bound_ok", "joint_pass")}
                gen["replay"]["diff"] = rp["replay_diff"]
                gen["replay"]["same_verdict_as_gen"] = same_verdict(gen, rp)
                runs.append(gen)
                h = gen["holds"]
                print(json.dumps({"tag": gen["tag"], "exec": gen["execution_end"], "task": gen["task_ok"], "manner": gen["manner_pass"],
                                  "safety": gen["safety_ok"], "bound": gen["bound_ok"], "joint": gen["joint_pass"],
                                  "replay_joint": rp["joint_pass"],
                                  "hold": {r: [h[r]["audit_decision"], h[r].get("min"), h[r].get("max"), h[r]["band"]] for r in h}}, default=_j), flush=True)
        ref = [r["settle"]["ref_pos"] for r in runs]
        spread = max(float(np.abs(np.asarray(x[c]) - np.asarray(ref[0][c])).max()) for x in ref for c in ("cubeA", "cubeB"))
        pairs = {f"{'->'.join(o)}:{c}": pair_ok([r for r in runs if r["order"] == "->".join(o)], c) for o in ORDERS for c in ("cubeA", "cubeB")}
        recs.append({"seed": sd, "all4_joint": all(r["joint_pass"] for r in runs), "pairs": pairs, "pairs_all": all(pairs.values()),
                     "settle_ref_spread_m": spread, "runs": runs})
        json.dump(recs, open(out / "validation.json", "w"), indent=1, default=_j)
    summarize(recs)


def summarize(recs: list) -> None:
    R = [r for s in recs for r in s["runs"]]
    n = len(R)
    c = lambda k: sum(bool(r[k]) for r in R)  # noqa: E731
    print(f"SUMMARY seeds={len(recs)} runs={n} exec={c('execution_end')} task={c('task_ok')} manner={c('manner_pass')} safety={c('safety_ok')} "
          f"bound={c('bound_ok')} joint={c('joint_pass')} replay_joint={sum(r['replay']['joint_pass'] for r in R)} "
          f"replay_same={sum(all(r['replay']['same_verdict_as_gen'].values()) for r in R)} "
          f"seeds_all4={sum(s['all4_joint'] for s in recs)}/{len(recs)} seeds_pairs_all={sum(s['pairs_all'] for s in recs)}/{len(recs)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("pack", "seeds", "aggregate"), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", help="subset of DEV_SEEDS; writes to dev_seeds/seed_<n> (parallel-friendly)")
    ap.add_argument("--out", default=str(ROOT / "reports/step1_validation_20261024"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.phase == "aggregate":
        recs = []
        for sd in DEV_SEEDS:
            f = Path(a.out) / "dev_seeds" / f"seed_{sd}" / "validation.json"
            if f.exists():
                recs += json.load(open(f))
        json.dump(recs, open(Path(a.out) / "dev_seeds" / "validation_all.json", "w"), indent=1)
        summarize(recs)
        return 0
    if a.phase == "pack":
        out = Path(a.out) / "pack_seed0"
        write_manifest(out, "pack", [PACK_SEED], a.force)
        phase_pack(out)
        return 0
    seeds = a.seeds or DEV_SEEDS
    assert set(seeds) <= set(DEV_SEEDS), "seeds must come from the frozen DEV_SEEDS list"
    out = Path(a.out) / "dev_seeds" / (f"seed_{seeds[0]}" if len(seeds) == 1 else "all")
    write_manifest(out, "seeds", seeds, a.force)
    phase_seeds(out, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
