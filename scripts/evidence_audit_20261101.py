"""Step 1 (2026-11-01): corrected T5 cost table, recovery-cost attribution, init usage ledger.  No simulation.

Outputs (reports/archive_20261101/): t5_cost_summary.json, recovery_cost_by_selection.json,
init_usage_ledger.json, init_usage_ledger.md
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("reports/archive_20261101")
R = Path("reports")


def deep(o, key, depth=0):
    if isinstance(o, dict):
        if key in o and isinstance(o[key], (int, float)):
            return o[key]
        if depth < 2:
            for v in o.values():
                x = deep(v, key, depth + 1)
                if x is not None:
                    return x
    return None


def runs(d):
    return [f for f in sorted(glob.glob(f"{d}/init*.json")) if not re.search(r"pair|summary", Path(f).name)]


def t5_cost():
    groups = [
        ("step14_precheck_a_config_only", "pre-check (failed), counted separately", R / "step14_t5_precheck/a_config_only"),
        ("step14_precheck_b_adapter_obs", "pre-check (failed), counted separately", R / "step14_t5_precheck/b_adapter_obs"),
        ("step14_precheck_c_wide_grid_t4", "pre-check (failed), counted separately", R / "step14_t5_precheck/c_wide_grid_t4"),
        ("step14_onboard_demo0_ref", "first-time onboarding", R / "step14_t5_onboard/d_demo0_ref"),
        ("step14_onboard_demo0_ref_yawfix", "first-time onboarding", R / "step14_t5_onboard/d_demo0_ref_yawfix"),
        ("step15_p1_align", "first-time onboarding", R / "step15_t5_place/p1_align"),
        ("step15_p2_align_all", "first-time onboarding", R / "step15_t5_place/p2_align_all"),
        ("step15_p3_v11", "first-time onboarding", R / "step15_t5_place/p3_v11"),
        ("step15_p4_heldout", "first-time onboarding", R / "step15_t5_place/p4_heldout"),
        ("step16_dev", "added manner condition (speed)", R / "step16_t5_speed_pilot/dev"),
        ("step16_new", "added manner condition (speed)", R / "step16_t5_speed_pilot/new"),
    ]
    rows, cat = [], defaultdict(lambda: {"episodes": 0, "episodes_with_wall": 0, "sim_wall_s": 0.0, "control_steps": 0})
    for name, c, d in groups:
        fs = runs(d)
        wall = steps = nw = 0
        for f in fs:
            j = json.loads(Path(f).read_text())
            w = deep(j, "sim_wall_s")
            if w is None:
                w = deep(j, "wall_s")
            s = deep(j, "control_steps")
            if w is not None:
                wall += w
                nw += 1
            steps += s or 0
        rows.append({"group": name, "category": c, "episodes": len(fs), "episodes_with_wall_recorded": nw,
                     "sim_wall_s": round(wall, 1), "control_steps": steps})
        k = cat[c]
        k["episodes"] += len(fs); k["episodes_with_wall"] += nw; k["sim_wall_s"] = round(k["sim_wall_s"] + wall, 1)
        k["control_steps"] += steps
    ex = json.loads((R / "step17_t5_language/execution_records.json").read_text())
    rows.append({"group": "step17_language_execution", "category": "language entry (no new simulation timing recorded)",
                 "episodes": ex["n_runs"], "episodes_with_wall_recorded": 0, "sim_wall_s": None, "control_steps": None,
                 "note": "4 execution records, each bitwise-compared to step16 trajectories"})
    cat["language entry (executions; duration missing)"] = {"episodes": ex["n_runs"], "episodes_with_wall": 0,
                                                              "sim_wall_s": None, "control_steps": None}
    tot_ep = sum(v["episodes"] for v in cat.values())
    tot_wall_n = sum(v["episodes_with_wall"] for v in cat.values())
    tot_wall = round(sum(v["sim_wall_s"] or 0.0 for v in cat.values()), 1)
    out = {"schema": "mannercap.t5_cost_summary.v3", "rows": rows, "by_category": cat,
           "total": {"episodes": tot_ep, "episodes_with_wall_recorded": tot_wall_n, "episodes_wall_missing": tot_ep - tot_wall_n,
                     "sim_wall_s_of_recorded_episodes": tot_wall,
                     "note": "every execution counts toward the run total; the 4 step17 executions have no recorded duration, "
                             "so the wall sum covers only the episodes with a recorded duration and is a lower bound"},
           "human_hours": "未记录",
           "correction": "v3: step17's 4 executions are counted in the total run count (12 pre-check + 21 onboarding + 16 step16 + 4 step17 = 53), "
                         "durations marked missing, not excluded. The earlier 37 episodes / 261.9 s = onboarding (21) + step16 (16) only"}
    (OUT / "t5_cost_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def recovery_by_selection():
    cells = []
    for f in sorted(glob.glob("reports/step12_ablation/t*_init*.json")):
        cells += json.loads(Path(f).read_text())["cells"]
    res = {}
    for sel in ("rule", "retrieval"):
        for task in (None, 0, 7):
            row = {}
            for rec in ("rerun", "recover"):
                g = [c for c in cells if c["selection"] == sel and c["recovery"] == rec and (task is None or c["task"] == task)]
                row[rec] = {"n": len(g), "method_substeps": sum(c["substeps_method"] for c in g),
                            "joint_passed": sum(c["joint_passed"] for c in g)}
            a, b = row["rerun"]["method_substeps"], row["recover"]["method_substeps"]
            row["saving_fraction"] = round(1 - b / a, 4)
            res[f"{sel}|{'all' if task is None else 'T' + str(task)}"] = row
    out = {"schema": "mannercap.recovery_cost_by_selection.v1", "book": "method (cache hits charged at recorded cost)",
           "note": "27.2% belongs to selection=rule, 27.3% to selection=retrieval (NOT to T0/T7); per-task rows are listed for completeness",
           "rows": res}
    (OUT / "recovery_cost_by_selection.json").write_text(json.dumps(out, indent=1))
    return out


# directory -> (task, purpose).  Inits are scanned from the files; purposes are classified by hand from the git log/scripts.
LEDGER = [
    ("step3_libero_20261025", "T1", "dev: first LIBERO validation"),
    ("step4_libero_calib_20261025", "T1", "calibration (force response)"),
    ("step4_libero_pairs_20261025", "T1", "dev: pairs v1"),
    ("step4_libero_pairs_20261025_v1_frozen", "T1", "dev: pairs v1 frozen copy"),
    ("step5_devcheck_v1bands", "T1", "dev check"),
    ("step5_diag_20261025", "T1", "dev diagnosis"),
    ("step5_libero_calib_v2", "T1", "calibration v2"),
    ("step5_libero_pairs_v2", "T1", "dev: pairs v2"),
    ("step7_approach_v1", "T1", "dev: approach planner"), ("step7_approach_v2", "T1", "dev: approach planner"),
    ("step7_approach_v2b", "T1", "dev: approach planner"), ("step7_approach_v2c", "T1", "dev: approach planner"),
    ("step7_t1_regress", "T1", "dev: regression"), ("step7_t1_settle1", "T1", "dev: regression"),
    ("step7_dz_sweep", "T0", "dev: onboarding sweep"), ("step7_onboard_a", "T0", "dev: onboarding"),
    ("step7_onboard_b", "T0/T7", "dev: onboarding"), ("step7_onboard_c", "T0/T7", "dev: onboarding"),
    ("step7_onboard_diag", "T0/T7", "dev: diagnosis"),
    ("step7_onboard_t4a", "T4", "dev: onboarding"), ("step7_onboard_t4b", "T4", "dev: onboarding"),
    ("step7_onboard_t4c_ref", "T4", "dev: onboarding"),
    ("step8_t0_raw", "T0", "library source (raw) / dev pairs"), ("step8_t7_raw", "T7", "library source (raw) / dev pairs"),
    ("step8_t0_response", "T0", "calibration inits (library source)"), ("step8_t7_response", "T7", "calibration inits (library source)"),
    ("step8_t0_pairs", "T0", "dev pairs"), ("step8_t7_pairs", "T7", "dev pairs"),
    ("step8_t0_pairs_probe_v0", "T0", "dev probe"), ("step8_t7_pairs_probe_v0", "T7", "dev probe"),
    ("step8_t0_response_probe_v0", "T0", "calibration probe"), ("step8_t7_response_probe_v0", "T7", "calibration probe"),
    ("step8_cases", "T0/T7", "dev: failure cases / repair"), ("step8_diagnosis", "T7", "dev: diagnosis"),
    ("step8_grasp_candidates", "T7", "dev: grasp candidates"), ("step8_shared_repair", "T0/T7", "dev: shared repair"),
    ("step8_transport_repair", "T0/T7", "dev: transport repair"), ("step8_contact_diag", "T0/T7", "dev: contact diagnosis"),
    ("step8_contact_identity", "T7", "dev: contact identity"), ("step8_dropout_diag", "T0/T7", "dev: dropout diagnosis"),
    ("step9_t4_onboard", "T4", "dev: onboarding (paused)"),
    ("step10_experience", "T0/T7", "EVAL (E1-E4 trials, library vs fixed order); also used for later contact-policy tuning"),
    ("step11_contact", "T0/T7", "DEV-TUNING: contact policy v1.1 / guard fitted on inits 26-27 (re-uses eval inits)"),
    ("step12_ablation", "T0/T7", "EVAL (2x2 rule/retrieval x rerun/recover)"),
    ("step14_t5_precheck", "T5", "onboarding pre-check (failed)"), ("step14_t5_onboard", "T5", "onboarding"),
    ("step15_t5_place", "T5", "onboarding / place fitting; p4 'heldout' is dev-time"),
    ("step16_t5_speed_pilot/dev", "T5", "speed threshold dev (n=4)"),
    ("step16_t5_speed_pilot/new", "T5", "speed pilot new inits"),
    ("step17_t5_language", "T5", "language entry execution"),
]


def init_ledger():
    rows, used = [], defaultdict(set)
    for d, task, purpose in LEDGER:
        inits = set()
        for dp, _, fn in __import__("os").walk(R / d):
            for f in fn:
                m = re.search(r"init_?(\d+)", f)
                if m and f.endswith((".json", ".npz")):
                    inits.add(int(m.group(1)))
        if not inits:
            continue
        rows.append({"dir": f"reports/{d}", "task": task, "purpose": purpose, "inits": sorted(inits)})
        for t in task.split("/"):
            used[t] |= inits
    # step17 uses init 0/10 per file; step16 ok
    summary = {}
    for t, s in sorted(used.items()):
        summary[t] = {"used_inits": sorted(s), "never_used_inits": [i for i in range(50) if i not in s]}
    eval_burned = {"T0": [26, 27, 28, 29], "T7": [26, 27, 28, 29]}
    note = {
        "freeze_manifest_init_splits": "calibration 8-13; dev 0-7,14-21; reserved 22-39 (26-29 = this batch); holdout 40-49 untouched",
        "finding": "T0/T7 inits 22-25 were already used for repair/diagnosis (step8_cases, shared_repair, grasp_candidates), "
                   "and 26-29 are the evaluation batch that step11 also tuned on (26-27). "
                   "Untouched for T0/T7: 30-49 (verified from file names; runs that left no init-named file would not appear).",
        "recommended_split": {"step3_eval": [30, 31, 32, 33], "kept_as_final_holdout": list(range(40, 50)),
                              "spare": list(range(34, 40))},
        "limits": "scan keys on file names containing 'init<N>' under the listed directories; the native robosuite lines "
                  "(gate*/wipe*/continuity*/context*/prod_*/p08b) use seeds, not LIBERO init indices, and are not in this ledger",
    }
    out = {"schema": "mannercap.init_usage_ledger.v1", "rows": rows, "per_task": summary, "eval_inits_burned": eval_burned,
           "notes": note}
    (OUT / "init_usage_ledger.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = ["| 目录 | 任务 | 用途 | 初态 |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dir']} | {r['task']} | {r['purpose']} | {r['inits']} |")
    md += ["", "| 任务 | 已使用初态 | 从未使用 |", "|---|---|---|"]
    for t, s in summary.items():
        md.append(f"| {t} | {s['used_inits']} | {s['never_used_inits']} |")
    (OUT / "init_usage_ledger.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    t = t5_cost()
    print(json.dumps(t["by_category"], ensure_ascii=False))
    print(json.dumps({k: v["saving_fraction"] for k, v in recovery_by_selection()["rows"].items()}))
    L = init_ledger()
    print(json.dumps(L["per_task"], ensure_ascii=False)[:1200])
