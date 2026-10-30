"""Per-run M1/M2/M4 (+M3) fields recovered from RAW per-run records only (no back-filling from summaries).
Step15 ordinary runs (p3_v11 dev 0-3, p4_heldout 4-7) and Step16 pilot runs.  A field that the raw record cannot provide is None
and listed in `missing`.  Output: reports/step16_t5_speed_pilot/gate_fields_per_run.json"""
import json
from pathlib import Path
R15, R16 = Path("reports/step15_t5_place"), Path("reports/step16_t5_speed_pilot")
need = ("M1_official_success_ever", "M1_first_step", "M2_instruction_completed", "M4_clean_v1_1", "M4_clean_v0", "M3_official_in_final")
rows = []
for sub in ("p3_v11", "p4_heldout"):
    for f in sorted((R15 / sub).glob("init*.json"), key=lambda p: int(p.stem[4:])):
        r = json.load(open(f)); v = r.get("contact_policy_v1_1") or {}
        st = r.get("stages_executed") or []
        rows.append({"source": f"{f.parent.name}/{f.name}", "run": "ordinary", "init": r["init"],
            "M1_official_success_ever": r.get("entered_official_region_any_step"), "M1_first_step": r.get("first_step_in_region"),
            "M2_instruction_completed": None if "place_completed" not in r else bool(r["place_completed"] and r.get("status") == "completed" and not r.get("place_failure") and r.get("grasp_established") and any(s.endswith("retreat") for s in st)),
            "M4_clean_v1_1": v.get("clean_v1"), "M4_clean_v0": r.get("contact_policy_v0_clean"),
            "M3_official_in_final": (r.get("post_release") or {}).get("official_in_final", r.get("final_in_region"))})
for ph in ("dev", "new"):
    for f in sorted((R16 / ph).glob("init*_[NS].json"), key=lambda p: (int(p.stem[4:].split("_")[0]), p.stem[-1])):
        r = json.load(open(f))
        rows.append({"source": f"{ph}/{f.name}", "run": "pilot_" + r["cond"], "init": r["init"],
            "M1_official_success_ever": r.get("official_success_ever"), "M1_first_step": r.get("official_first_success_step"),
            "M2_instruction_completed": r.get("instruction_completed"), "M4_clean_v1_1": r.get("contact_clean_v1_1"), "M4_clean_v0": r.get("contact_clean_v0"),
            "M3_official_in_final": (r.get("post_release_M3") or {}).get("official_in_final")})
for x in rows:
    x["missing"] = [k for k in need if x.get(k) is None]
json.dump({"n_runs": len(rows), "rule": "fields derived from raw per-run JSON only; None = not recoverable; no value back-filled from a summary", "runs": rows}, open(R16 / "gate_fields_per_run.json", "w"), indent=1)
for x in rows: print(x["source"], x["run"], x["init"], "M1", x["M1_official_success_ever"], x["M1_first_step"], "M2", x["M2_instruction_completed"], "M4v1.1", x["M4_clean_v1_1"], "v0", x["M4_clean_v0"], "M3in", x["M3_official_in_final"], "missing", x["missing"])
print("ALL_GATE", sum(1 for x in rows if x["M1_official_success_ever"] and x["M2_instruction_completed"] and x["M4_clean_v1_1"]), "/", len(rows), "missing_any", sum(1 for x in rows if x["missing"]))
