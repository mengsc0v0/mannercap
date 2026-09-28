"""Pure-numpy physical metrics for manner-sensitive trajectories.

No capx / robosuite / simulator dependency. Consumes the arrays saved by
run_manner_ablation.save_trajectory_npz (t, phase, joint_vel, eef_pos,
cfrc_norm, kp, gripper_fraction, ...) and returns reviewer-facing physical
metrics that evidence that manner was physically controlled:

  * SPARC  (spectral arc length, Balasubramanian 2015) - smoothness, scale/
    duration invariant.
  * LDLJ   (log dimensionless jerk) - smoothness, duration robust.
  * Per-phase speed profile + step ratios - normalises the raw
    approach-vs-descend speed-step intuition.
  * Contact approach velocity + post-contact force rise slope - gentle vs firm.
  * Path length + integral kinetic-energy proxy - economical vs aggressive.
  * Cohen's d - group separability (fast vs gentle vs firm) across conditions.

All functions accept plain numpy arrays / python floats. This module never runs
a simulator; it only post-processes recorded trajectory arrays.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "mannercap-manner-trajectory-metrics-v1"


def spectral_arc_length(speed, fs, *, padlevel=4, fc=10.0, amp_th=0.05):
    """SPARC of a 1-D speed profile (Balasubramanian et al. 2015).

    More negative == less smooth. Returns 0.0 for a degenerate profile.
    """
    speed = np.asarray(speed, dtype=np.float64).ravel()
    if speed.size < 4 or fs <= 0.0:
        return 0.0
    peak = np.max(np.abs(speed))
    if peak <= 0.0:
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(speed.size)) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    band = np.where(freq <= fc)[0]
    if band.size < 2:
        return 0.0
    f_sel = freq[band]
    m_sel = mag[band]
    inx = np.where(m_sel >= amp_th)[0]
    if inx.size < 2:
        return 0.0
    lo, hi = inx[0], inx[-1]
    f_sel = f_sel[lo : hi + 1]
    m_sel = m_sel[lo : hi + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 0.0:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(m_sel) ** 2)))


def dimensionless_jerk(speed, fs):
    """Dimensionless jerk of a speed profile (negative, scale/duration free)."""
    speed = np.asarray(speed, dtype=np.float64).ravel()
    if speed.size < 3 or fs <= 0.0:
        return 0.0
    dt = 1.0 / fs
    peak = np.max(np.abs(speed))
    if peak <= 0.0:
        return 0.0
    jerk = np.diff(speed, n=2) / (dt ** 2)
    duration = speed.size * dt
    scale = (duration ** 3) / (peak ** 2)
    return float(-scale * np.sum(jerk ** 2) * dt)


def log_dimensionless_jerk(speed, fs):
    """LDLJ = -ln(|dimensionless jerk|). Larger (less negative) == smoother."""
    dlj = dimensionless_jerk(speed, fs)
    if dlj == 0.0:
        return 0.0
    return float(-np.log(abs(dlj)))


def phase_mean_speeds(phase_labels, speed):
    """Mean speed per phase label, preserving first-seen order."""
    labels = np.asarray(phase_labels).ravel()
    speed = np.asarray(speed, dtype=np.float64).ravel()
    n = min(labels.size, speed.size)
    labels, speed = labels[:n], speed[:n]
    labels_s = labels.astype(str)
    out = {}
    order = []
    for lab in labels_s:
        if lab not in out:
            order.append(lab)
            out[lab] = 0.0
    for key in order:
        vals = speed[labels_s == key]
        out[key] = float(np.mean(vals)) if vals.size else 0.0
    return out


def phase_step_ratios(phase_speeds):
    """Reviewer-facing step ratios from per-phase mean speeds."""
    vals = np.asarray(list(phase_speeds.values()), dtype=np.float64)
    vals = vals[vals > 0.0]
    out = {}
    out["max_min_ratio"] = float(vals.max() / vals.min()) if vals.size >= 2 else 1.0

    def _ratio(a, b):
        va, vb = phase_speeds.get(a), phase_speeds.get(b)
        if va is None or vb is None or vb <= 0.0:
            return None
        return float(va / vb)

    for name, a, b in (
        ("approach_descend_ratio", "approach", "descend"),
        ("transport_final_ratio", "transport", "final"),
        ("approach_transport_ratio", "approach", "transport"),
    ):
        r = _ratio(a, b)
        if r is not None:
            out[name] = r
    return out


def path_length(positions):
    """Arc length of an (N, D) position trajectory."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def integral_kinetic_energy(joint_vel, dt):
    """Sum of squared joint speed * dt (mass-1 kinetic-energy proxy)."""
    jv = np.asarray(joint_vel, dtype=np.float64)
    if jv.ndim != 2 or jv.shape[0] < 1 or dt <= 0.0:
        return 0.0
    return float(np.sum(jv ** 2) * dt)


def contact_onset_index(cfrc_norm, *, threshold=1.0):
    """First index where contact force exceeds threshold (N)."""
    cf = np.asarray(cfrc_norm, dtype=np.float64).ravel()
    idx = np.where(cf > threshold)[0]
    return int(idx[0]) if idx.size else None


def contact_approach_velocity(cfrc_norm, speed, *, threshold=1.0):
    """Speed at the sample just before contact onset (gentle == low)."""
    onset = contact_onset_index(cfrc_norm, threshold=threshold)
    if onset is None:
        return 0.0
    sp = np.asarray(speed, dtype=np.float64).ravel()
    j = max(0, onset - 1)
    return float(sp[j]) if j < sp.size else 0.0


def post_contact_force_slope(cfrc_norm, dt, *, threshold=1.0, window=20):
    """Least-squares slope (N/s) of contact force over window samples after onset."""
    onset = contact_onset_index(cfrc_norm, threshold=threshold)
    if onset is None or dt <= 0.0:
        return 0.0
    cf = np.asarray(cfrc_norm, dtype=np.float64).ravel()
    seg = cf[onset : onset + window]
    if seg.size < 2:
        return 0.0
    t = np.arange(seg.size, dtype=np.float64) * dt
    return float(np.polyfit(t, seg, 1)[0])


def cohens_d(a, b):
    """Cohen's d effect size between two samples (pooled SD)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    na, nb = a.size, b.size
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def _sampling_frequency(t):
    if t.size < 2:
        return 0.0
    dt = np.median(np.diff(t))
    return float(1.0 / dt) if dt > 0.0 else 0.0


def compute_trajectory_metrics(traj, *, contact_threshold=1.0):
    """Compute the full physical-metric set for one recorded trajectory."""
    t = np.asarray(traj["t"], dtype=np.float64).ravel()
    fs = _sampling_frequency(t)
    dt = 1.0 / fs if fs > 0.0 else 0.0

    eef_pos_raw = traj.get("eef_pos")
    eef_pos = np.asarray(eef_pos_raw, dtype=np.float64) if eef_pos_raw is not None else None
    if eef_pos is not None and eef_pos.ndim == 2 and eef_pos.shape[0] >= 2 and dt > 0.0:
        eef_speed = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1) / dt
    else:
        eef_speed = np.zeros(0)

    jv_raw = traj.get("joint_vel")
    joint_vel = np.asarray(jv_raw, dtype=np.float64) if jv_raw is not None else None
    if joint_vel is not None and joint_vel.ndim == 2:
        joint_speed = np.linalg.norm(joint_vel, axis=1)
    else:
        joint_speed = np.zeros(0)

    phase = traj.get("phase")
    cfrc = traj.get("cfrc_norm")

    metrics = {"schema": SCHEMA}
    smooth_src = eef_speed if eef_speed.size >= 4 else joint_speed
    metrics["sparc"] = spectral_arc_length(smooth_src, fs)
    metrics["ldlj"] = log_dimensionless_jerk(smooth_src, fs)
    metrics["sparc_joint"] = spectral_arc_length(joint_speed, fs)

    if phase is not None and eef_speed.size:
        phase_arr = np.asarray(phase).ravel()
        aligned = phase_arr[1 : 1 + eef_speed.size]
        pspeeds = phase_mean_speeds(aligned, eef_speed)
        metrics["phase_mean_speeds"] = pspeeds
        metrics["phase_step_ratios"] = phase_step_ratios(pspeeds)
    else:
        metrics["phase_mean_speeds"] = {}
        metrics["phase_step_ratios"] = {}

    metrics["eef_path_length"] = path_length(eef_pos) if eef_pos is not None else 0.0
    metrics["integral_kinetic_energy"] = (
        integral_kinetic_energy(joint_vel, dt) if joint_vel is not None else 0.0
    )

    if cfrc is not None:
        cf = np.asarray(cfrc, dtype=np.float64).ravel()
        metrics["peak_cfrc"] = float(np.max(cf)) if cf.size else 0.0
        metrics["mean_cfrc"] = float(np.mean(cf)) if cf.size else 0.0
        metrics["contact_onset_index"] = contact_onset_index(cf, threshold=contact_threshold)
        metrics["contact_approach_velocity"] = contact_approach_velocity(
            cf, joint_speed, threshold=contact_threshold
        )
        metrics["post_contact_force_slope"] = post_contact_force_slope(
            cf, dt, threshold=contact_threshold
        )
    else:
        metrics["peak_cfrc"] = 0.0
        metrics["mean_cfrc"] = 0.0
        metrics["contact_onset_index"] = None
        metrics["contact_approach_velocity"] = 0.0
        metrics["post_contact_force_slope"] = 0.0

    grasp = traj.get("grasp_contact_force")
    if grasp is not None:
        gf = np.asarray(grasp, dtype=np.float64).ravel()
        metrics["peak_grasp_force"] = float(np.max(gf)) if gf.size else 0.0
        metrics["mean_grasp_force"] = float(np.mean(gf)) if gf.size else 0.0
        if phase is not None:
            ph = np.asarray(phase).ravel()
            n = min(ph.size, gf.size)
            gsel = gf[:n][ph[:n] == "grip"]
            metrics["peak_grasp_force_grip"] = float(np.max(gsel)) if gsel.size else 0.0
            metrics["mean_grasp_force_grip"] = float(np.mean(gsel)) if gsel.size else 0.0
        else:
            metrics["peak_grasp_force_grip"] = metrics["peak_grasp_force"]
            metrics["mean_grasp_force_grip"] = metrics["mean_grasp_force"]
    else:
        metrics["peak_grasp_force"] = 0.0
        metrics["mean_grasp_force"] = 0.0
        metrics["peak_grasp_force_grip"] = 0.0
        metrics["mean_grasp_force_grip"] = 0.0

    metrics["mean_joint_speed"] = float(np.mean(joint_speed)) if joint_speed.size else 0.0
    metrics["peak_joint_speed"] = float(np.max(joint_speed)) if joint_speed.size else 0.0
    # peak_joint_speed 已知伪影：move_to_joints_profiled() 在每次 profiled 运动结束后
    # 会调用未被速度剖面约束的 move_to_joints_blocking() 做"收尾追齐"（把残余位置误差在
    # 1-2 步内追平到 tolerance 以内），该追齐步的关节速度由位置伺服增益(kp)主导、与 v_max
    # 无关，导致不同 manner 条件下 peak_joint_speed 趋同到同一个由 kp 决定的常数，掩盖了
    # 真实的 manner 速度差异。用 95 分位数作为抗突刺的稳健峰值替代指标。
    metrics["peak_joint_speed_p95"] = (
        float(np.percentile(joint_speed, 95)) if joint_speed.size else 0.0
    )
    metrics["fs_hz"] = fs
    metrics["n_samples"] = int(t.size)
    return metrics


def group_separability(metrics_by_group, fields):
    """Pairwise Cohen's d between groups for each scalar field."""
    groups = list(metrics_by_group.keys())
    out = {}
    for field in fields:
        samples = {
            g: np.asarray(
                [m[field] for m in metrics_by_group[g]
                 if isinstance(m.get(field), (int, float))],
                dtype=np.float64,
            )
            for g in groups
        }
        field_out = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ga, gb = groups[i], groups[j]
                field_out[ga + "|" + gb] = cohens_d(samples[ga], samples[gb])
        out[field] = field_out
    return out

def roc_auc(scores, labels):
    """ROC AUC via the Mann-Whitney U statistic (rank-based, pure numpy).

    scores: predicted score per sample (higher => more likely positive).
    labels: 0/1 array. Returns AUC in [0, 1]; 0.5 == chance. NaN-safe.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # average ranks over all samples, ties averaged
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # resolve ties to average rank
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank_per_group = (start + csum + 1) / 2.0
    ranks = avg_rank_per_group[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _diag_lda_project(Xtr, ytr, xq):
    """Closed-form diagonal-LDA (naive-Gaussian) score for a query point.

    Direction w_j = (mu_a_j - mu_b_j) / pooled_var_j; score = (xq - mid) . w.
    Higher score => closer to class a. Closed-form => no iterative jackknife
    bias, stable for very small n. Returns a scalar.
    """
    a = Xtr[ytr == 1]; b = Xtr[ytr == 0]
    mu_a = a.mean(axis=0); mu_b = b.mean(axis=0)
    va = a.var(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
    vb = b.var(axis=0, ddof=1) if b.shape[0] > 1 else np.zeros(b.shape[1])
    pooled = (va + vb) / 2.0
    pooled = np.where(pooled > 0, pooled, 1.0)
    w = (mu_a - mu_b) / pooled
    mid = (mu_a + mu_b) / 2.0
    return float((xq - mid) @ w)


def _clean_feature_matrix(rows, fields):
    """Build (X, kept_fields) dropping non-finite / constant columns."""
    X = np.array(
        [[r.get(f) if isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool)
          else np.nan for f in fields] for r in rows],
        dtype=np.float64,
    )
    finite_col = np.all(np.isfinite(X), axis=0)
    nonconst_col = np.array([np.ptp(X[:, j]) > 0 for j in range(X.shape[1])])
    keep = finite_col & nonconst_col
    kept_fields = [f for f, k in zip(fields, keep) if k]
    return X[:, keep], kept_fields


def manner_classifier_auc(group_a, group_b, fields, *, l2=1.0, iters=800, lr=0.5):
    """Leave-one-out cross-validated AUC of a logistic manner classifier.

    group_a/group_b: lists of per-trajectory metric dicts (a=positive class).
    Returns dict with the multivariate LOO AUC (out-of-sample, honest for small
    n) plus per-field univariate AUC (rank-based, direction preserved).
    """
    rows = list(group_a) + list(group_b)
    y = np.array([1.0] * len(group_a) + [0.0] * len(group_b), dtype=np.float64)
    X, kept = _clean_feature_matrix(rows, fields)
    result = {"n_a": len(group_a), "n_b": len(group_b),
              "fields_used": kept, "loo_auc": 0.5, "univariate_auc": {}}
    # univariate AUC per usable field
    for j, f in enumerate(kept):
        result["univariate_auc"][f] = roc_auc(X[:, j], y)
    n = X.shape[0]
    if X.shape[1] == 0 or len(group_a) < 2 or len(group_b) < 2:
        return result
    oos = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        Xtr, ytr = X[tr], y[tr]
        if ytr.sum() < 2 or (ytr == 0).sum() < 2:
            continue
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
        Xtr_n = (Xtr - mu) / sd
        oos[i] = _diag_lda_project(Xtr_n, ytr, (X[i] - mu) / sd)
    result["loo_auc"] = roc_auc(oos, y)
    return result


def group_classifier_auc(metrics_by_group, fields, **kw):
    """Pairwise leave-one-out classifier AUC between every pair of groups."""
    groups = list(metrics_by_group.keys())
    out = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ga, gb = groups[i], groups[j]
            out[ga + "|" + gb] = manner_classifier_auc(
                metrics_by_group[ga], metrics_by_group[gb], fields, **kw)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory-similarity metrics (DTW / discrete Frechet) + cluster silhouette
# ──────────────────────────────────────────────────────────────────────────────
def _pairwise_dist(a, b):
    """Euclidean distance matrix between rows of a (n,d) and b (m,d)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def dtw_distance(a, b):
    """Dynamic Time Warping distance between two multivariate sequences.

    a: (n, d), b: (m, d) (1-D inputs treated as (n, 1)). Pure numpy DP,
    Euclidean local cost, standard 3-neighbour recursion. Returns the total
    warped cost (not length-normalized). NaN-safe on empty inputs.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    C = _pairwise_dist(a, b)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = C[i - 1]
        for j in range(1, m + 1):
            D[i, j] = ci[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def discrete_frechet_distance(a, b):
    """Discrete Frechet distance between two polylines (numpy DP, O(nm))."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    C = _pairwise_dist(a, b)
    n, m = C.shape
    ca = np.full((n, m), -1.0)
    ca[0, 0] = C[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], C[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], C[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                           C[i, j])
    return float(ca[n - 1, m - 1])


def _resample_sequence(x, num):
    """Linearly resample a (n,d) sequence to `num` rows along its time axis."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return np.zeros((num, x.shape[1]))
    if n == num:
        return x
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, num)
    return np.stack([np.interp(dst, src, x[:, j]) for j in range(x.shape[1])],
                    axis=1)


def trajectory_signature(traj, *, num=80):
    """Fixed-length (num*d,) descriptor of a trajectory for distance/clustering.

    Uses the end-effector path (eef_pos) resampled to `num` points and centred
    on its start, so distances reflect shape/scale rather than absolute table
    placement. Falls back to joint_vel if eef_pos is absent.
    """
    pos = traj.get("eef_pos")
    if pos is None:
        pos = traj.get("joint_vel")
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos[:, None]
    rs = _resample_sequence(pos, num)
    rs = rs - rs[0:1, :]
    return rs.ravel()


def similarity_to_reference(traj, reference, *, num=80):
    """DTW + discrete-Frechet distance of a trajectory's eef path to a reference.

    Both are time-resampled to `num` points and start-centred first, so the
    numbers are comparable across trajectories of different lengths.
    """
    def _prep(t):
        pos = t.get("eef_pos") if isinstance(t, Mapping) else t
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[:, None]
        rs = _resample_sequence(pos, num)
        return rs - rs[0:1, :]
    a = _prep(traj)
    b = _prep(reference)
    return {"dtw_to_reference": dtw_distance(a, b),
            "frechet_to_reference": discrete_frechet_distance(a, b)}


def silhouette_score(signatures, labels):
    """Mean silhouette coefficient for a labelled set of fixed-length vectors.

    signatures: (N, F) array; labels: length-N group ids. Pure numpy, Euclidean.
    s(i) = (b - a) / max(a, b); returns the mean over all samples in [-1, 1].
    +1 => tight, well-separated manner clusters; ~0 => overlapping; <0 => mixed.
    """
    X = np.asarray(signatures, dtype=np.float64)
    labels = np.asarray(labels)
    N = X.shape[0]
    uniq = np.unique(labels)
    if N < 2 or uniq.size < 2:
        return 0.0
    D = _pairwise_dist(X, X)
    sil = np.zeros(N)
    for i in range(N):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            sil[i] = 0.0
            continue
        a_i = D[i, same].mean()
        b_i = np.inf
        for g in uniq:
            if g == labels[i]:
                continue
            other = labels == g
            if other.sum() == 0:
                continue
            b_i = min(b_i, D[i, other].mean())
        denom = max(a_i, b_i)
        sil[i] = 0.0 if denom == 0 else (b_i - a_i) / denom
    return float(sil.mean())
