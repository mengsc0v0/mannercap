"""Analyse p11_continuity_sweep outputs: command -> realised response per manner quantity.

usage: p11_continuity_report.py DIR OUT_PREFIX
Reads DIR/{wipe_speed,wipe_press,lift_grip,stack_grip,lift_speed,stack_speed}.jsonl and writes
OUT_PREFIX.json and OUT_PREFIX.md.  An episode that failed the task (or has no valid measurement)
is NaN for the response statistics and is counted in n_invalid; per-level task-success rate is
reported so that failures at the range ends stay visible.
"""
import json, math, os, sys

import numpy as np

from synthesis_v2.response_curve import analyse_response
from synthesis_v2.stage_table import dimension_stages


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


MIN_LEVEL_SUCCESS = 0.5  # a command level at which fewer than half the episodes finish the task is "unusable"


def series(rows, getter):
    x = [r["level"] for r in rows]
    rate = {}
    for lv in set(x):
        rr = [r for r in rows if r["level"] == lv]
        rate[lv] = sum(1 for r in rr if r.get("task_ok")) / len(rr)
    y = []
    for r in rows:
        v = getter(r) if (r.get("task_ok") and rate[r["level"]] >= MIN_LEVEL_SUCCESS) else None
        y.append(math.nan if v is None else float(v))
    return x, y


def success_by_level(rows):
    out = {}
    for lv in sorted({r["level"] for r in rows}):
        rr = [r for r in rows if r["level"] == lv]
        out[lv] = {"n": len(rr), "task_ok": sum(1 for r in rr if r.get("task_ok"))}
    return out


def repeatability(summary):
    """median over levels of (p90-p10)/|median| of the realised value at a fixed command."""
    vals = [(l["p90"] - l["p10"]) / abs(l["median"]) for l in summary["levels"]
            if "median" in l and abs(l["median"]) > 1e-9 and l["n_valid"] >= 3]
    return float(np.median(vals)) if vals else None


def main():
    d, prefix = sys.argv[1], sys.argv[2]
    quantities = []
    wipe_speed = load(f"{d}/wipe_speed.jsonl")
    quantities.append(("wipe/sweep speed", "m/s", "command v_max", wipe_speed, lambda r: r.get("speed_med")))
    quantities.append(("wipe/press force", "N", "target mean tool force", load(f"{d}/wipe_press.jsonl"),
                       lambda r: r.get("press_med")))
    for task in ("lift", "stack"):
        quantities.append((f"{task}/grip hold force", "N", "target band lower bound", load(f"{d}/{task}_grip.jsonl"),
                           lambda r: r.get("hold_med") if r.get("hold_valid") else None))
        rows = load(f"{d}/{task}_speed.jsonl")
        for st in dimension_stages(task, "speed"):
            quantities.append((f"{task}/{st} speed", "m/s", "command v_max (all stages)", rows,
                               (lambda s: lambda r: (r.get("stage_speed_med") or {}).get(s))(st)))
    result = {}
    lines = ["# Command -> realised response (continuity audit)", "",
             "| quantity | label | grades | Spearman (all) | monotone command range | Spearman (range) | R2 (range) | slope | "
             "realised range in monotone range | reachable realised (all) | invalid/total | repeatability | tie fraction | failed criteria |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, unit, cmd, rows, getter in quantities:
        if not rows:
            continue
        x, y = series(rows, getter)
        s = analyse_response(x, y, name=name, unit=unit)
        s["command_meaning"] = cmd
        s["task_success_by_level"] = success_by_level(rows)
        s["repeatability_p90_p10_over_median"] = repeatability(s)
        result[name] = s
        r = s["usable_range"]
        f = lambda v, p=3: "-" if v is None or (isinstance(v, float) and v != v) else f"{v:.{p}g}"  # noqa: E731
        if r.get("found"):
            rng = f"{r['command_lo']:g}..{r['command_hi']:g} ({r['n_levels']} lv)"
            rrng = f"{f(r['realised_lo'])}..{f(r['realised_hi'])} {unit}"
            sp, r2, sl = f(r["spearman"]), f(r["linear"]["r2"]), f(r["linear"]["slope"])
        else:
            rng = rrng = sp = r2 = sl = "-"
        reach = s["reachable_realised"]
        lines.append(f"| {name} | **{s['classification']['label']}** | {s['classification']['n_grades']} | "
                     f"{f(s['spearman_all'])} | {rng} | {sp} | {r2} | {sl} | {rrng} | "
                     f"{f(reach[0]) + '..' + f(reach[1]) if reach else '-'} | {s['n_invalid']}/{s['n']} | "
                     f"{f(s['repeatability_p90_p10_over_median'])} | "
                     f"{f(s.get('quantisation_in_range', {}).get('tie_fraction'))} | "
                     f"{','.join(s['classification'].get('failed_criteria', [])) or '-'} |")
    json.dump(result, open(prefix + ".json", "w"), indent=1)
    lines += ["", "Labels: `continuous` = over the monotone range Spearman>=0.95, R2>=0.90 and >=5 separable grades (AUC>=0.9); "
              "`graded(k)` = monotone but only k separable grades; `not_controllable` = no monotone range. "
              "Invalid = task failed, no valid measurement, or the command level succeeded in < 50 % of episodes (unusable level). repeatability = median over levels of (p90-p10)/median across seeds."]
    open(prefix + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
