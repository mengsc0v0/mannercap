#!/usr/bin/env python3
"""Compute the manner physical-metric baseline table over recorded npz trajectories.

Applies synthesis_v2.manner_trajectory_metrics to every *.npz under the given
trajectory directories, groups by (task, tag), and emits:
  * <out>.per_trajectory.json  - full metrics per trajectory
  * <out>.summary.json         - per-(task,tag) mean/std of scalar metrics
  * <out>.cohens_d.json        - pairwise Cohen's d for requested field/groups
  * <out>.md                   - reviewer-facing markdown tables

Pure post-processing: never runs a simulator.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthesis_v2 import manner_trajectory_metrics as M

SCALAR_FIELDS = [
    "sparc", "ldlj", "sparc_joint", "eef_path_length",
    "integral_kinetic_energy", "peak_cfrc", "mean_cfrc",
    "contact_approach_velocity", "post_contact_force_slope",
    "mean_joint_speed", "peak_joint_speed", "peak_joint_speed_p95",
    "peak_grasp_force", "mean_grasp_force",
    "peak_grasp_force_grip", "mean_grasp_force_grip",
]

# semantic families for the kinematic (speed/smoothness) Cohen's d axis
FAST_TAGS = {"fast", "fast_en", "fast_then_slow", "normal", "baseline"}
GENTLE_TAGS = {"gentle", "gentle_en", "ultra_gentle", "ultra_careful"}

# semantic families for the grasp-force (contact) Cohen's d axis
FIRM_GRIP_TAGS = {"firm_grip", "firm"}
GENTLE_GRIP_TAGS = {"gentle_grip"}

# feature subsets for the manner classifier (leave-one-out AUC)
KINEMATIC_AUC_FIELDS = [
    "sparc", "ldlj", "sparc_joint", "mean_joint_speed",
    "peak_joint_speed", "peak_joint_speed_p95", "integral_kinetic_energy", "eef_path_length",
]
GRASP_AUC_FIELDS = [
    "peak_grasp_force", "mean_grasp_force",
    "peak_grasp_force_grip", "mean_grasp_force_grip",
]


def _scalar(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def load_traj(path):
    d = np.load(path, allow_pickle=True)
    traj = {}
    for k in ("t", "phase", "joint_vel", "eef_pos", "cfrc_norm", "grasp_contact_force"):
        if k in d.files:
            traj[k] = d[k]
    task = str(d["task"]) if "task" in d.files else "unknown"
    tag = str(d["tag"]) if "tag" in d.files else "unknown"
    seed = int(d["seed"]) if "seed" in d.files else -1
    return traj, task, tag, seed


def mean_std(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    paths = []
    for d in args.traj_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
    if not paths:
        print("no npz found", file=sys.stderr)
        return 1

    per_traj = []
    groups = defaultdict(list)  # (task, tag) -> [metrics]
    signatures = defaultdict(list)  # (task, tag) -> [fixed-length eef signature]
    for p in paths:
        traj, task, tag, seed = load_traj(p)
        m = M.compute_trajectory_metrics(traj)
        rec = {"file": os.path.basename(p), "task": task, "tag": tag, "seed": seed,
               "metrics": m}
        per_traj.append(rec)
        groups[(task, tag)].append(m)
        sig = M.trajectory_signature(traj)
        if sig is not None:
            signatures[(task, tag)].append(sig)

    # summary per (task, tag)
    summary = {}
    for (task, tag), ms in sorted(groups.items()):
        key = f"{task}/{tag}"
        entry = {"n_trajectories": len(ms)}
        for f in SCALAR_FIELDS:
            entry[f] = mean_std([_scalar(m.get(f)) for m in ms])
        # representative phase step ratio (mean of approach_descend_ratio)
        adr = [m.get("phase_step_ratios", {}).get("approach_descend_ratio")
               for m in ms]
        entry["approach_descend_ratio"] = mean_std([_scalar(x) for x in adr])
        mmr = [m.get("phase_step_ratios", {}).get("max_min_ratio") for m in ms]
        entry["max_min_speed_ratio"] = mean_std([_scalar(x) for x in mmr])
        summary[key] = entry

    # Cohen's d: fast-family vs gentle-family, per task
    cohens = {}
    tasks = sorted({t for (t, _) in groups.keys()})
    for task in tasks:
        fast_ms, gentle_ms = [], []
        for (t, tag), ms in groups.items():
            if t != task:
                continue
            if tag in FAST_TAGS:
                fast_ms.extend(ms)
            elif tag in GENTLE_TAGS:
                gentle_ms.extend(ms)
        if len(fast_ms) >= 2 and len(gentle_ms) >= 2:
            cohens[task] = M.group_separability(
                {"fast_family": fast_ms, "gentle_family": gentle_ms},
                SCALAR_FIELDS,
            )

    # Cohen's d: firm-grip vs gentle-grip (contact-force axis), per task
    cohens_grip = {}
    for task in tasks:
        firm_ms, soft_ms = [], []
        for (t, tag), ms in groups.items():
            if t != task:
                continue
            if tag in FIRM_GRIP_TAGS:
                firm_ms.extend(ms)
            elif tag in GENTLE_GRIP_TAGS:
                soft_ms.extend(ms)
        if len(firm_ms) >= 2 and len(soft_ms) >= 2:
            cohens_grip[task] = M.group_separability(
                {"firm_grip": firm_ms, "gentle_grip": soft_ms},
                SCALAR_FIELDS,
            )

    # Manner-classifier leave-one-out AUC (honest out-of-sample separability)
    auc_kin, auc_grip = {}, {}
    for task in tasks:
        fast_ms, gentle_ms = [], []
        firm_ms, soft_ms = [], []
        for (t, tag), ms in groups.items():
            if t != task:
                continue
            if tag in FAST_TAGS:
                fast_ms.extend(ms)
            elif tag in GENTLE_TAGS:
                gentle_ms.extend(ms)
            if tag in FIRM_GRIP_TAGS:
                firm_ms.extend(ms)
            elif tag in GENTLE_GRIP_TAGS:
                soft_ms.extend(ms)
        if len(fast_ms) >= 2 and len(gentle_ms) >= 2:
            auc_kin[task] = M.manner_classifier_auc(
                fast_ms, gentle_ms, KINEMATIC_AUC_FIELDS)
        if len(firm_ms) >= 2 and len(soft_ms) >= 2:
            auc_grip[task] = M.manner_classifier_auc(
                firm_ms, soft_ms, GRASP_AUC_FIELDS)

    # Trajectory-shape separability: silhouette (per axis) + DTW/Frechet of each
    # group's mean signature to the opposite group's mean signature.
    import numpy as _np

    def _axis_similarity(pos_tags, neg_tags):
        res = {}
        for task in tasks:
            pos_sigs, neg_sigs = [], []
            for (t, tag), sigs in signatures.items():
                if t != task:
                    continue
                if tag in pos_tags:
                    pos_sigs.extend(sigs)
                elif tag in neg_tags:
                    neg_sigs.extend(sigs)
            if len(pos_sigs) < 2 or len(neg_sigs) < 2:
                continue
            L = min(min(len(x) for x in pos_sigs),
                    min(len(x) for x in neg_sigs))
            P = _np.array([x[:L] for x in pos_sigs])
            N = _np.array([x[:L] for x in neg_sigs])
            X = _np.concatenate([P, N], axis=0)
            lab = _np.array([1] * len(P) + [0] * len(N))
            sil = M.silhouette_score(X, lab)
            # DTW/Frechet between the two group mean signatures (reshaped to path)
            d = 3
            pm = P.mean(axis=0).reshape(-1, d)
            nm = N.mean(axis=0).reshape(-1, d)
            res[task] = {
                "silhouette": sil,
                "dtw_between_means": M.dtw_distance(pm, nm),
                "frechet_between_means": M.discrete_frechet_distance(pm, nm),
                "n_pos": len(P), "n_neg": len(N),
            }
        return res

    sim_kin = _axis_similarity(FAST_TAGS, GENTLE_TAGS)
    sim_grip = _axis_similarity(FIRM_GRIP_TAGS, GENTLE_GRIP_TAGS)

    base = args.out
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + ".per_trajectory.json", "w") as f:
        json.dump(per_traj, f, indent=2, default=str)
    with open(base + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(base + ".cohens_d.json", "w") as f:
        json.dump({"kinematic_axis": cohens, "grasp_force_axis": cohens_grip}, f, indent=2)
    with open(base + ".classifier_auc.json", "w") as f:
        json.dump({"kinematic_axis": auc_kin, "grasp_force_axis": auc_grip}, f, indent=2)
    with open(base + ".similarity.json", "w") as f:
        json.dump({"kinematic_axis": sim_kin, "grasp_force_axis": sim_grip}, f, indent=2)

    # markdown
    lines = ["# Manner physical-metric baseline\n",
             f"Trajectories: {len(per_traj)} | groups: {len(groups)}\n",
             "\n## Per-group summary (mean +/- std)\n"]
    hdr = ["task/tag", "n", "SPARC", "LDLJ", "peak_jspeed", "app/desc ratio",
           "max/min ratio", "peak_grasp_grip", "mean_grasp_grip"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for key, e in summary.items():
        def cell(f):
            ms = e[f]
            if ms["mean"] is None:
                return "-"
            return f"{ms['mean']:.3f}+/-{ms['std']:.3f}"
        lines.append("| " + " | ".join([
            key, str(e["n_trajectories"]),
            cell("sparc"), cell("ldlj"), cell("peak_joint_speed"),
            cell("approach_descend_ratio"), cell("max_min_speed_ratio"),
            cell("peak_grasp_force_grip"), cell("mean_grasp_force_grip"),
        ]) + " |")

    if cohens:
        lines.append("\n## Cohen's d: fast-family vs gentle-family\n")
        lines.append("|d| > 0.8 = large effect (manner physically separable).\n")
        for task, fields in cohens.items():
            lines.append(f"\n### task = {task}\n")
            lines.append("| field | Cohen's d |")
            lines.append("|---|---|")
            for f, pair in fields.items():
                d = pair.get("fast_family|gentle_family")
                if d is not None:
                    lines.append(f"| {f} | {d:.3f} |")

    if cohens_grip:
        lines.append("\n## Cohen's d: firm-grip vs gentle-grip (contact-force axis)\n")
        lines.append("True finger-object normal contact force via mj_contactForce; |d| > 0.8 = large effect.\n")
        for task, fields in cohens_grip.items():
            lines.append(f"\n### task = {task}\n")
            lines.append("| field | Cohen's d |")
            lines.append("|---|---|")
            for f, pair in fields.items():
                d = pair.get("firm_grip|gentle_grip")
                if d is not None:
                    lines.append(f"| {f} | {d:.3f} |")

    if auc_kin or auc_grip:
        lines.append("\n## Manner-classifier leave-one-out AUC\n")
        lines.append("Out-of-sample AUC of a diagonal-LDA manner classifier "
                     "(LOO cross-validated). 1.0 = perfectly separable, 0.5 = chance.\n")

        def _auc_block(title, axis, pair_key):
            lines.append(f"\n### {title}\n")
            lines.append("| task | n(a/b) | multivariate LOO AUC | best single-feature AUC |")
            lines.append("|---|---|---|---|")
            for task, res in axis.items():
                uni = res.get("univariate_auc", {})
                if uni:
                    bf = max(uni, key=lambda k: abs(uni[k] - 0.5))
                    best = f"{bf}={uni[bf]:.3f}"
                else:
                    best = "-"
                lines.append(
                    f"| {task} | {res.get('n_a')}/{res.get('n_b')} | "
                    f"{res.get('loo_auc', 0.5):.3f} | {best} |")

        if auc_kin:
            _auc_block("Kinematic axis (fast vs gentle)", auc_kin, None)
        if auc_grip:
            _auc_block("Contact-force axis (firm-grip vs gentle-grip)", auc_grip, None)

    if sim_kin or sim_grip:
        lines.append("\n## Trajectory-shape separability (silhouette + DTW/Frechet)\n")
        lines.append("Silhouette on start-centred eef-path signatures: +1 = tight, "
                     "well-separated manner clusters, ~0 = overlapping. DTW/Frechet "
                     "are distances between the two groups' mean paths (larger = more "
                     "physically distinct shapes).\n")

        def _sim_block(title, axis):
            lines.append(f"\n### {title}\n")
            lines.append("| task | n(a/b) | silhouette | DTW(mean paths) | Frechet(mean paths) |")
            lines.append("|---|---|---|---|---|")
            for task, r in axis.items():
                lines.append(
                    f"| {task} | {r['n_pos']}/{r['n_neg']} | {r['silhouette']:.3f} | "
                    f"{r['dtw_between_means']:.3f} | {r['frechet_between_means']:.3f} |")

        if sim_kin:
            _sim_block("Kinematic axis (fast vs gentle)", sim_kin)
        if sim_grip:
            _sim_block("Contact-force axis (firm-grip vs gentle-grip)", sim_grip)
    with open(base + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {base}.md (+ .summary/.cohens_d/.per_trajectory json)")
    print(f"groups: {sorted(groups.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
