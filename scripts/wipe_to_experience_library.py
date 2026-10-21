"""Append the wipe results (teacher grids, calibration dev records, production attempts) to the experience library.

usage: wipe_to_experience_library.py LIB.jsonl STAMP [--grids GLOB] [--dev GLOB] [--pipelines DIR1,DIR2]
Outcome vocabulary: both_pass (task + manner audited), task_only (task ok, manner band missed), task_fail.
Teacher-grid rows are not manner-labelled and only carry task_ok / task_fail.
"""
import argparse, glob, json
from pathlib import Path

from synthesis_v2.experience_library import ExperienceLibrary

ap = argparse.ArgumentParser()
ap.add_argument("lib")
ap.add_argument("stamp")
ap.add_argument("--grids", default="")
ap.add_argument("--dev", default="")
ap.add_argument("--pipelines", default="")
a = ap.parse_args()
lib = ExperienceLibrary()
n = 0


def add(dim, level, value, outcome, seed):
    global n
    lib.add_experience("wipe", "sweep", dim, level, value, outcome, source_run_stamp=a.stamp, seed=seed)
    n += 1


for p in sorted(glob.glob(a.grids)) if a.grids else []:
    for line in open(p):
        r = json.loads(line)
        add("v_cmd_mps", f"press{int(r['F'])}N", r["v"], "task_ok" if r["ok"] else "task_fail", r["seed"])
for p in sorted(glob.glob(a.dev)) if a.dev else []:
    for line in open(p):
        r = json.loads(line)
        st = r["stats"]
        out = "both_pass" if st.get("task_ok") else "task_fail"
        sp = st["speed"].get("sweep") or []
        if sp:
            import statistics
            add("v_realised_median_mps", r["speed_level"], statistics.median(sp), out, r["seed"])
        if st.get("press_median") is not None:
            add("tool_force_median_n", r["press_level"], st["press_median"], out, r["seed"])
for d in a.pipelines.split(",") if a.pipelines else []:
    for line in open(Path(d) / "attempts.jsonl"):
        t = json.loads(line)
        if t["outcome"] == "duplicate":
            continue
        m = t.get("metadata") or {}
        if t["outcome"] == "accepted":
            out = "both_pass"
        elif m.get("audit_task") == "pass":
            out = "task_only"
        else:
            out = "task_fail"
        cp = t["candidate_plan"]
        seed = t["request_snapshot"]["initial_state"]["seed"]
        sw = [s for s in cp["stages"] if s["stage_id"] == "sweep"]
        if sw:
            add("v_cmd_mps", sw[0]["level"], sw[0]["v_max_mps"], out, seed)
        add("press_target_n", cp["press"]["level"], cp["press"]["target_n"], out, seed)
with open(a.lib, "a", encoding="utf-8") as f:
    for rec in lib.experiences:
        from synthesis_v2.experience_library import _to_dict
        f.write(json.dumps(_to_dict(rec), ensure_ascii=False) + "\n")
print("appended", n, "records to", a.lib)
