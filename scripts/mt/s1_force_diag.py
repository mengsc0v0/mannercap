"""Step 1 follow-up: audit-definition force check, order-swap binding check, per-instance traces.

Question answered: is the per-instance grip *requirement* met on the hold window, with the same
definition the production audit uses (hold window = lift+transport minus head trim, every sample in
the calibrated band, both fingers in contact)?  The earlier 'mean over the grasp stage' number
included the closing ramp and is not the audited quantity.

Design: 2 grip assignments x 2 operation orders.  The requirement is bound to the OBJECT, so a bound
requirement must follow the object when the order changes; a requirement bound to the Nth grasp
would swap owners.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthesis_v2.grasp_force_contract import GraspState, summarize_grasp_force  # noqa: E402
from synthesis_v2.manner_calibration import grip_bound, hold_force_total  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode  # noqa: E402
from synthesis_v2.native_producer import StagePlan  # noqa: E402
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances  # noqa: E402

# "prod": what the production path builds (candidate_plan: GripPlan(lo, lo + GRIP_BAND_WIDTH_N), lo = LEVEL_RANGES[..][2]).
# "edge": reference stop thresholds equal to the audit lower edge -- kept as a sensitivity run.
PROFILES = {"prod": {"gentle": GripPlan(3.0, 9.0), "firm": GripPlan(28.0, 34.0)},
            "edge": {"gentle": GripPlan(2.0, 6.0), "firm": GripPlan(24.0, 34.0)}}
LEVEL_PLAN = dict(PROFILES["prod"])
CAL = json.load(open(ROOT / "configs/manner_calibration.v3.json"))
TRIM = int(CAL["policy"]["hold_trim_head_steps"])
CTRL_DT = 0.05
# Operational tolerances for the *task* check (not manner calibration); stated, not tuned.
XY_TOL, Z_TOL, LIFT_MIN, DISTURB_TOL = 0.02, 0.015, 0.10, 0.01
SETTLE = 12  # cubes spawn ~1 cm above the table and settle within 4 control steps; reference pose is the settled one


def program(order: tuple[str, str]) -> TaskProgram:
    groups = [pick_place_instances(r, "spot" + r[-1]) for r in order]
    return TaskProgram(f"two_cube_putback_{order[0]}_{order[1]}", "robosuite_native", "stack",
                       {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
                       chain(*groups), source="composed")


def plan_for(prog: TaskProgram, levels: dict[str, str]) -> ProgramPlan:
    stages = tuple(StagePlan(it.label, 0.12) for it in prog.instances)
    grips = {f"{r}.pick_1.grasp": LEVEL_PLAN[lv] for r, lv in levels.items()}
    return ProgramPlan(prog, stages, grips)


def hold_idx(arrays, prog: TaskProgram, gid: str) -> np.ndarray:
    """Audit definition: lift + transport of this instance's scope, minus the first TRIM steps of each."""
    st = np.asarray(arrays["stage"])
    out: list[int] = []
    for iid in prog.grip_scope(gid):
        it = prog.instance(iid)
        if it.primitive in ("lift", "transport"):
            out.extend(np.flatnonzero(st == it.label)[TRIM:].tolist())
    return np.asarray(sorted(out), dtype=int)


def grip_audit(arrays, prog, gid: str, level: str) -> dict:
    lo, hi, why = grip_bound(CAL, "stack", level)
    idx = hold_idx(arrays, prog, gid)
    rec = {"level": level, "band": [round(lo, 3), round(hi, 3)], "n_hold": int(len(idx))}
    mask = np.zeros(len(arrays["actions"]), bool)
    mask[idx] = True
    valid = np.asarray(arrays["valid_fraction"], float) >= 1 - 1e-9
    s = summarize_grasp_force(arrays["left_mean_n"], arrays["right_mean_n"], valid, mask=mask, dt_s=CTRL_DT,
                              contact_threshold_n=CAL["safety"]["stack"]["contact_threshold_n"],
                              min_bilateral_steps=len(idx))
    rec["grasp_state"] = s.state.value
    tot = hold_force_total(arrays, idx)
    rec.update(hold_min=round(float(tot.min()), 3), hold_median=round(float(np.median(tot)), 3),
               hold_max=round(float(tot.max()), 3))
    if s.state is not GraspState.BILATERAL:
        rec["decision"], rec["reason"] = "FAIL", f"hold_not_bilateral:{s.state.value}"
    elif tot.min() < lo:
        rec["decision"], rec["reason"] = "FAIL", "hold_force_below_band"
    elif tot.max() > hi:
        rec["decision"], rec["reason"] = "FAIL", "hold_force_above_band"
    else:
        rec["decision"], rec["reason"] = "PASS", ""
    return rec


def grasp_stage_stats(arrays, prog, gid: str) -> dict:
    """Closing-phase detail: the numbers behind the earlier 'mean' figure and the controller's view."""
    st = np.asarray(arrays["stage"])
    it = prog.instance(gid)
    idx = np.flatnonzero(st == it.label)
    tot = (np.asarray(arrays["left_mean_n"]) + np.asarray(arrays["right_mean_n"]))[idx]
    g = np.asarray(arrays["actions"])[idx, 6]
    return {"n_steps": int(len(idx)), "stage_mean_total": round(float(np.nanmean(tot)), 3),
            "stage_peak_total": round(float(np.nanmax(tot)), 3),
            "first_step_total_ge_lo": next((int(i) for i, v in enumerate(tot) if v >= LO_OF[gid]), None),
            "gripper_cmd_close/hold/open": [int((g > 0.5).sum()), int((np.abs(g) <= 0.5).sum()), int((g < -0.5).sum())]}


def target_binding_check(arrays, prog, gid: str, expected: str) -> dict:
    scope_labels = [prog.instance(i).label for i in prog.grip_scope(gid)]
    st = np.asarray(arrays["stage"])
    m = np.isin(st, scope_labels)
    ft = np.asarray(arrays["force_target"])[m]
    hi_ = np.asarray(arrays["hold_instance"])[m]
    return {"expected_target": expected, "steps_in_scope": int(m.sum()),
            "steps_measuring_expected": int((ft == expected).sum()),
            "steps_hold_instance_is_self": int((hi_ == gid).sum()),
            "bound_ok": bool((ft == expected).all() and (hi_ == gid).all())}


def task_check(ep, prog: TaskProgram) -> dict:
    a = ep.arrays
    st = np.asarray(a["stage"])
    out: dict = {"objects": {}}
    for role in ("cubeA", "cubeB"):
        pos = np.asarray(a["pos_" + role])
        own = np.isin(st, [i.label for i in prog.instances if i.subtask == f"{role}.pick_1"])
        first, last = int(np.flatnonzero(own)[0]), int(np.flatnonzero(own)[-1])
        p0, p1 = pos[SETTLE], np.asarray(ep.meta["final_obj_pos_by_role"][role])
        dxy, dz = float(np.linalg.norm((p1 - p0)[:2])), float(abs(p1[2] - p0[2]))
        rest_dev = max(float(np.linalg.norm(pos[SETTLE:first] - p0, axis=1).max(initial=0.0)),
                       float(np.linalg.norm(pos[last + 1:] - pos[last], axis=1).max(initial=0.0)))
        lift = float(pos[own][:, 2].max() - p0[2])
        out["objects"][role] = {
            "restored": bool(dxy <= XY_TOL and dz <= Z_TOL), "dxy": round(dxy, 4), "dz": round(dz, 4),
            "lifted": bool(lift >= LIFT_MIN), "max_lift_m": round(lift, 4),
            "undisturbed_outside_own_subtask": bool(rest_dev <= DISTURB_TOL), "rest_dev_m": round(rest_dev, 4)}
    out["task_ok"] = all(o["restored"] and o["lifted"] and o["undisturbed_outside_own_subtask"]
                         for o in out["objects"].values())
    return out


LO_OF: dict = {}


def plot(ep, prog, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = ep.arrays
    t = np.arange(len(a["actions"]))
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    tot = np.asarray(a["left_mean_n"]) + np.asarray(a["right_mean_n"])
    ax[0].plot(t, tot, label="total (L+R)", lw=1.2)
    ax[0].fill_between(t, a["hold_lo_n"], a["hold_hi_n"], alpha=0.2, color="g", label="controller target band")
    ax[0].set_ylabel("N"); ax[0].legend(loc="upper right")
    ax[1].plot(t, a["left_mean_n"], label="left"); ax[1].plot(t, a["right_mean_n"], label="right")
    ax[1].set_ylabel("N per finger"); ax[1].legend(loc="upper right")
    ax[2].plot(t, np.asarray(a["actions"])[:, 6], label="gripper cmd (+1 close, 0 hold, -1 open)")
    ft = np.asarray(a["force_target"])
    ax[2].step(t, (ft == "cubeB").astype(float) * 0.5 - 1.5, label="measuring cubeB (high) / cubeA (low)", color="r")
    ax[2].legend(loc="upper right"); ax[2].set_xlabel("control step")
    st = np.asarray(a["stage"])
    for i in np.flatnonzero(st[1:] != st[:-1]) + 1:
        for x in ax:
            x.axvline(i, color="k", lw=0.3, alpha=0.4)
    for it in prog.of_primitive("grasp"):
        i = int(np.flatnonzero(st == it.label)[0])
        ax[0].text(i, ax[0].get_ylim()[1] * 0.9, it.instance_id, fontsize=7, rotation=90)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)


def run_one(order, levels, seed, outdir: Path) -> dict:
    prog = program(order)
    assert not prog.validate()
    ep = produce_program_episode(prog, seed, plan_for(prog, levels), keep_images=False)
    gids = {r: f"{r}.pick_1.grasp" for r in order}
    for r, lv in levels.items():
        LO_OF[gids[r]] = LEVEL_PLAN[lv].target_lo_n
    rec = {"order": "->".join(order), "levels": levels, "seed": seed, "status": ep.status,
           "n_steps": int(ep.arrays["actions"].shape[0]), "grasps": {}}
    for r in order:
        g = gids[r]
        rec["grasps"][r] = {
            "controller_plan_band": [LEVEL_PLAN[levels[r]].target_lo_n, LEVEL_PLAN[levels[r]].target_hi_n],
            "controller_policy": ep.meta["grasp_policy_by_instance"][g],
            "binding": target_binding_check(ep.arrays, prog, g, r),
            "audit": grip_audit(ep.arrays, prog, g, levels[r]),
            "grasp_stage": grasp_stage_stats(ep.arrays, prog, g)}
    rec["task"] = task_check(ep, prog)
    rec["manner_pass"] = all(v["audit"]["decision"] == "PASS" for v in rec["grasps"].values())
    rec["bound_ok"] = all(v["binding"]["bound_ok"] for v in rec["grasps"].values())
    rec["joint_pass"] = bool(ep.status == "completed" and rec["task"]["task_ok"] and rec["manner_pass"])
    tag = f"{order[0][-1]}{order[1][-1]}_{levels['cubeA']}A_{levels['cubeB']}B_s{seed}"
    plot(ep, prog, outdir / f"force_trace_{tag}.png")
    np.savez_compressed(outdir / f"trace_{tag}.npz", **{k: v for k, v in ep.arrays.items() if k != "images"})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="prod")
    ap.add_argument("--out", default=str(ROOT / "reports/step1_followup_20261024"))
    a = ap.parse_args()
    LEVEL_PLAN.update(PROFILES[a.profile])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for order in (("cubeA", "cubeB"), ("cubeB", "cubeA")):
        for lv in ({"cubeA": "gentle", "cubeB": "firm"}, {"cubeA": "firm", "cubeB": "gentle"}):
            r = run_one(order, lv, a.seed, out)
            recs.append(r)
            print(json.dumps(r), flush=True)
    json.dump(recs, open(out / "force_diag.json", "w"), indent=2)
    print(f"SUMMARY runs={len(recs)} executed={sum(r['status']=='completed' for r in recs)} "
          f"bound_ok={sum(r['bound_ok'] for r in recs)} task_ok={sum(r['task']['task_ok'] for r in recs)} "
          f"manner_pass={sum(r['manner_pass'] for r in recs)} joint_pass={sum(r['joint_pass'] for r in recs)} of {len(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
