"""Command -> realised physical response analysis (continuity of a manner dimension).

Pure numpy.  Answers, for one controllable quantity (e.g. the commanded sweep speed or the
target hold force): is the *realised* value monotone and resolvable in the command, over what
range, and with how many distinguishable grades?  Only a quantity that passes
``classify_continuity`` may be described as "continuous"; otherwise it is "graded(k)".
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

RESPONSE_CURVE_VERSION = "response_curve.v1"

# decision thresholds (fixed before looking at results; see classify_continuity)
MIN_SPEARMAN = 0.95
MIN_R2 = 0.90
MIN_DISTINGUISHABLE = 5
SEPARABLE_AUC = 0.90
MAX_TIE_FRACTION = 0.20   # staircase (quantised) response: share of adjacent levels with identical realised value
TIE_REL_TOL = 0.005      # two level medians closer than 0.5 % of the realised range are "identical"


def _clean(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(xa) & np.isfinite(ya)
    return xa[m], ya[m]


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _clean(x, y)
    if len(xa) < 3:
        return math.nan
    rx, ry = _rank(xa), _rank(ya)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def linear_fit(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    xa, ya = _clean(x, y)
    if len(xa) < 3 or xa.std() == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "rmse": math.nan}
    slope, intercept = np.polyfit(xa, ya, 1)
    res = ya - (slope * xa + intercept)
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2),
            "rmse": float(np.sqrt((res ** 2).mean()))}


def auc_separation(a: Sequence[float], b: Sequence[float]) -> float:
    """P(b > a) + 0.5 P(b == a) (Mann-Whitney); 0.5 = inseparable, 1 = b entirely above a."""
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    aa, bb = aa[np.isfinite(aa)], bb[np.isfinite(bb)]
    if len(aa) == 0 or len(bb) == 0:
        return math.nan
    gt = (bb[:, None] > aa[None, :]).sum()
    eq = (bb[:, None] == aa[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(aa) * len(bb)))


def isotonic_increasing(y: Sequence[float], w: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing least-squares fit of ``y``."""
    ya = np.asarray(y, dtype=float)
    wa = np.ones(len(ya)) if w is None else np.asarray(w, dtype=float)
    blocks: list[list[float]] = []  # [value, weight, count]
    for v, wt in zip(ya, wa):
        blocks.append([float(v), float(wt), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([np.full(c, v) for v, _, c in blocks]) if blocks else ya


def per_level(x: Sequence[float], y: Sequence[float]) -> list[dict[str, Any]]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out = []
    for lv in sorted(set(xa[np.isfinite(xa)].tolist())):
        v = ya[(xa == lv)]
        n_all = int(len(v))
        v = v[np.isfinite(v)]
        rec: dict[str, Any] = {"command": lv, "n": n_all, "n_valid": int(len(v))}
        if len(v):
            rec.update(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                       p90=float(np.percentile(v, 90)), min=float(v.min()), max=float(v.max()))
        out.append(rec)
    return out


def distinguishable_levels(x: Sequence[float], y: Sequence[float], *, min_auc: float = SEPARABLE_AUC) -> dict[str, Any]:
    """Greedy count of grades: walk up the commands, keep a level only if its realised values are
    separable (AUC >= ``min_auc``) from the last kept level.  Also the adjacent-pair separability."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    levels = sorted(set(xa[np.isfinite(xa)].tolist()))
    groups = {lv: ya[xa == lv][np.isfinite(ya[xa == lv])] for lv in levels}
    levels = [lv for lv in levels if len(groups[lv])]
    if not levels:
        return {"n_distinguishable": 0, "kept_commands": [], "adjacent_auc": []}
    kept = [levels[0]]
    for lv in levels[1:]:
        if auc_separation(groups[kept[-1]], groups[lv]) >= min_auc:
            kept.append(lv)
    adj = [{"from": a, "to": b, "auc": auc_separation(groups[a], groups[b])} for a, b in zip(levels[:-1], levels[1:])]
    return {"n_distinguishable": len(kept), "kept_commands": kept, "adjacent_auc": adj}


def _trim_flat_ends(levels: list[float], med: list[float], min_levels: int, flat_ratio: float,
                    groups: Optional[Sequence[np.ndarray]] = None) -> tuple[int, int]:
    """Drop saturated / dead-zone end levels: remove an end while the slope of its 3 outermost level
    medians is below ``flat_ratio`` x the interval's overall slope, or (with ``groups``) while the end
    level's step slope is that low AND it is not separable (AUC < SEPARABLE_AUC) from its neighbour.
    Returns (i, j) offsets."""
    i, j = 0, len(levels) - 1

    def slope(a: int, b: int) -> float:
        xs, ys = np.asarray(levels[a: b + 1]), np.asarray(med[a: b + 1])
        return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 and xs.std() > 0 else 0.0

    while j - i + 1 > min_levels:
        overall = slope(i, j)
        if overall <= 0:
            break
        if slope(max(i, j - 2), j) < flat_ratio * overall:
            j -= 1
        elif slope(i, min(j, i + 2)) < flat_ratio * overall:
            i += 1
        elif groups is not None and slope(j - 1, j) < flat_ratio * overall \
                and auc_separation(groups[j - 1], groups[j]) < SEPARABLE_AUC:
            j -= 1
        elif groups is not None and slope(i, i + 1) < flat_ratio * overall \
                and auc_separation(groups[i], groups[i + 1]) < SEPARABLE_AUC:
            i += 1
        else:
            break
    return i, j


def monotone_range(x: Sequence[float], y: Sequence[float], *, min_rho: float = MIN_SPEARMAN,
                   min_levels: int = 4, flat_ratio: float = 0.5) -> dict[str, Any]:
    """Widest contiguous command interval (over the sorted levels) whose per-episode Spearman is
    >= ``min_rho``, then trimmed of saturated / dead-zone ends (see ``_trim_flat_ends``).  Returns the
    command interval and the realised interval (level medians) it maps to."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    levels = sorted(set(xa[np.isfinite(xa) & np.isfinite(ya)].tolist()))
    best: Optional[tuple[int, int, float]] = None
    for i in range(len(levels)):
        for j in range(i + min_levels - 1, len(levels)):
            m = (xa >= levels[i]) & (xa <= levels[j])
            rho = spearman(xa[m], ya[m])
            if rho == rho and rho >= min_rho and (best is None or (j - i) > (best[1] - best[0])
                                                  or ((j - i) == (best[1] - best[0]) and rho > best[2])):
                best = (i, j, rho)
    if best is None:
        return {"found": False}
    lv = levels[best[0]: best[1] + 1]
    med_all = [float(np.median(ya[(xa == v) & np.isfinite(ya)])) for v in lv]
    grp_all = [ya[(xa == v) & np.isfinite(ya)] for v in lv]
    a, b = _trim_flat_ends(lv, med_all, min_levels, flat_ratio, grp_all)
    lv, med = lv[a: b + 1], med_all[a: b + 1]
    m = (xa >= lv[0]) & (xa <= lv[-1])
    rho = spearman(xa[m], ya[m])
    if not (rho == rho and rho >= min_rho):
        return {"found": False}
    return {"found": True, "command_lo": lv[0], "command_hi": lv[-1], "n_levels": len(lv),
            "spearman": rho, "realised_lo": min(med), "realised_hi": max(med),
            "linear": linear_fit(xa[m], ya[m])}


def quantisation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Staircase detector: adjacent command levels whose realised medians coincide (|diff| <= TIE_REL_TOL x
    the span of the medians).  A quantised response (e.g. force set by an integer number of close steps) has
    many ties; a noisy continuous one has none."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lv = sorted(set(xa[np.isfinite(xa) & np.isfinite(ya)].tolist()))
    med = np.array([float(np.median(ya[(xa == v) & np.isfinite(ya)])) for v in lv])
    if len(med) < 2 or med.max() == med.min():
        return {"tie_fraction": math.nan, "n_distinct_medians": int(len(set(np.round(med, 9)))) if len(med) else 0}
    tol = TIE_REL_TOL * float(med.max() - med.min())
    ties = int(np.sum(np.abs(np.diff(med)) <= tol))
    distinct = 1 + int(np.sum(np.abs(np.diff(med)) > tol))
    return {"tie_fraction": ties / (len(med) - 1), "n_distinct_medians": distinct, "n_levels": int(len(med))}


def continuous_range(x: Sequence[float], y: Sequence[float], *, min_levels: int = 5) -> dict[str, Any]:
    """Widest contiguous command interval on which ALL continuity criteria hold at once (per-episode Spearman,
    linear R^2, >= MIN_DISTINGUISHABLE separable grades, tie fraction <= MAX_TIE_FRACTION), after trimming
    saturated / dead-zone ends.  Thresholds are the module constants; only the interval is searched."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    levels = sorted(set(xa[np.isfinite(xa) & np.isfinite(ya)].tolist()))
    best: Optional[dict[str, Any]] = None
    for i in range(len(levels)):
        for j in range(i + min_levels - 1, len(levels)):
            if best is not None and (j - i + 1) < best["n_levels"]:
                continue
            m = (xa >= levels[i]) & (xa <= levels[j])
            rho = spearman(xa[m], ya[m])
            lin = linear_fit(xa[m], ya[m])
            if not (rho == rho and rho >= MIN_SPEARMAN and lin["r2"] == lin["r2"] and lin["r2"] >= MIN_R2):
                continue
            k = distinguishable_levels(xa[m], ya[m])["n_distinguishable"]
            q = quantisation(xa[m], ya[m])
            tie = q.get("tie_fraction", math.nan)
            if k < MIN_DISTINGUISHABLE or (tie == tie and tie > MAX_TIE_FRACTION):
                continue
            meds = [float(np.median(ya[(xa == v) & np.isfinite(ya)])) for v in levels[i: j + 1]]
            grp = [ya[(xa == v) & np.isfinite(ya)] for v in levels[i: j + 1]]
            if _trim_flat_ends(levels[i: j + 1], meds, min_levels, 0.5, grp) != (0, j - i):
                continue  # a saturated / dead-zone end belongs to no continuous interval (a sub-interval is checked on its own)
            cand = {"found": True, "command_lo": levels[i], "command_hi": levels[j], "n_levels": j - i + 1,
                    "spearman": rho, "linear": lin, "n_grades": k, "tie_fraction": tie,
                    "realised_lo": min(meds), "realised_hi": max(meds)}
            if best is None or cand["n_levels"] > best["n_levels"] or (cand["n_levels"] == best["n_levels"]
                                                                       and rho > best["spearman"]):
                best = cand
    return best or {"found": False}


def inverse_map(x: Sequence[float], y: Sequence[float], rng: Mapping[str, Any]) -> Optional[dict[str, list[float]]]:
    """Monotone (isotonic) level-median curve over the monotone range, usable as realised -> command."""
    if not rng.get("found"):
        return None
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lv = [v for v in sorted(set(xa.tolist())) if rng["command_lo"] <= v <= rng["command_hi"]]
    med = np.array([float(np.median(ya[(xa == v) & np.isfinite(ya)])) for v in lv])
    cnt = np.array([float(np.sum((xa == v) & np.isfinite(ya))) for v in lv])
    iso = isotonic_increasing(med, cnt)
    # strictly increasing support points for np.interp
    keep = [0]
    for k in range(1, len(iso)):
        if iso[k] > iso[keep[-1]] + 1e-12:
            keep.append(k)
    return {"realised": [float(iso[k]) for k in keep], "command": [float(lv[k]) for k in keep]}


def command_for(inv: Mapping[str, Sequence[float]], realised_target: float) -> dict[str, Any]:
    r, c = np.asarray(inv["realised"], dtype=float), np.asarray(inv["command"], dtype=float)
    if len(r) < 2:
        return {"command": math.nan, "in_range": False}
    inside = bool(r[0] <= realised_target <= r[-1])
    return {"command": float(np.interp(realised_target, r, c)), "in_range": inside}


def classify_continuity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """'continuous' iff a command interval exists where every continuity criterion holds (see
    ``continuous_range``); then the label carries that interval.  Else 'graded(k)' (k = separable grades over the
    monotone range) with the criteria that failed there, or 'not_controllable' without any monotone range."""
    rng = summary.get("monotone_range", {})
    cr = summary.get("continuous_range", {})
    if cr.get("found"):
        return {"label": "continuous", "n_grades": int(cr["n_grades"]),
                "command_range": [cr["command_lo"], cr["command_hi"]],
                "realised_range": [cr["realised_lo"], cr["realised_hi"]], "failed_criteria": []}
    if not rng.get("found"):
        return {"label": "not_controllable", "n_grades": int(summary.get("distinguishable", {}).get("n_distinguishable", 0)),
                "failed_criteria": ["no_monotone_range"]}
    k = int(summary["monotone_distinguishable"]["n_distinguishable"])
    lin = rng["linear"]
    tie = summary.get("quantisation_in_range", {}).get("tie_fraction", math.nan)
    reasons = []
    if not rng["spearman"] >= MIN_SPEARMAN:
        reasons.append("spearman")
    if not (lin["r2"] == lin["r2"] and lin["r2"] >= MIN_R2):
        reasons.append("r2")
    if k < MIN_DISTINGUISHABLE:
        reasons.append("few_grades")
    if tie == tie and tie > MAX_TIE_FRACTION:
        reasons.append("staircase")
    return {"label": f"graded({k})", "n_grades": k, "failed_criteria": reasons or ["no_interval_meets_all_criteria"]}


def analyse_response(command: Sequence[float], realised: Sequence[float], *, name: str = "",
                     unit: str = "") -> dict[str, Any]:
    """One summary dict for a quantity.  ``command``/``realised`` are per-episode, NaN = failed or
    invalid episode (counted in ``n_invalid`` and excluded from the statistics)."""
    xa, ya = np.asarray(command, dtype=float), np.asarray(realised, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xv, yv = xa[valid], ya[valid]
    rng = monotone_range(xa, ya)
    crng = continuous_range(xa, ya)
    if rng.get("found"):
        m = (xa >= rng["command_lo"]) & (xa <= rng["command_hi"])
        mono_dist = distinguishable_levels(xa[m], ya[m])
        quant = quantisation(xa[m], ya[m])
    else:
        mono_dist = {"n_distinguishable": 0, "kept_commands": [], "adjacent_auc": []}
        quant = {}
    # staircase is a property of the actuator, not of a sub-interval: judged over the whole monotone range so that a
    # lucky sub-interval of a quantised response cannot be reported as continuous
    if crng.get("found") and quant.get("tie_fraction", math.nan) == quant.get("tie_fraction", math.nan) \
            and quant["tie_fraction"] > MAX_TIE_FRACTION:
        crng = {"found": False, "reason": "staircase_over_monotone_range"}
    out: dict[str, Any] = {
        "version": RESPONSE_CURVE_VERSION, "name": name, "unit": unit,
        "n": int(len(xa)), "n_invalid": int((~valid).sum()),
        "spearman_all": spearman(xv, yv),
        "linear_all": linear_fit(xv, yv),
        "levels": per_level(xa, ya),
        "distinguishable": distinguishable_levels(xa, ya),
        "monotone_range": rng,
        "continuous_range": crng,
        "monotone_distinguishable": mono_dist,
        "quantisation_in_range": quant,
        "quantisation_all": quantisation(xa, ya),
        "reachable_realised": ([float(np.min(yv)), float(np.max(yv))] if len(yv) else None),
        "usable_range": crng if crng.get("found") else rng,
        "inverse": inverse_map(xa, ya, crng if crng.get("found") else rng),
    }
    out["classification"] = classify_continuity(out)
    return out
