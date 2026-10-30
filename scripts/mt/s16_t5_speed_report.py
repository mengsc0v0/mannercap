"""Aggregate the 16 pilot runs + draw the one speed-time figure (dev init 0, N vs S)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path("reports/step16_t5_speed_pilot"); rows = []; pairs = []
for ph in ("dev", "new"):
    for p in json.load(open(R / ph / "summary.json"))["pairs"]:
        pairs.append((ph, p))
        for c in "NS":
            r = json.load(open(R / ph / f"init{p['init']}_{c}.json")); rows.append((ph, r))
u = json.load(open(R / "freeze.json"))["slow_upper_mps"]
out = {"slow_upper_mps": u, "per_run": [], "per_pair": []}
for ph, r in rows:
    d = r["descent"]; m3 = r["post_release_M3"]
    out["per_run"].append({"phase": ph, "init": r["init"], "cond": r["cond"], "official_success_ever": r["official_success_ever"], "official_first_success_step": r["official_first_success_step"],
        "instruction_completed": r["instruction_completed"], "contact_clean_v1_1": r["contact_clean_v1_1"], "contact_clean_v0": r["contact_clean_v0"], "official_success_final": r["official_success_final"],
        "ordinary_gate": r["ordinary_gate"], "descent_mean": d["mean_speed_mps"], "descent_median": d["median_speed_mps"], "descent_p95": d["p95_speed_mps"], "descent_dur_s": d["duration_s"],
        "descent_steps": d["n_steps"], "abs_slow_by_dev_threshold": d["median_speed_mps"] <= u, "release": [r["release_start_step"], r["release_end_step"]],
        "M3": m3, "retreat_mean": r["retreat"]["mean_speed_mps"], "status": r["status"], "v11_excused": [c for c in r["v11_contacts"] if c["excused"]]})
for ph, p in pairs:
    n = [x for x in out["per_run"] if x["init"] == p["init"] and x["cond"] == "N"][0]; s = [x for x in out["per_run"] if x["init"] == p["init"] and x["cond"] == "S"][0]
    abs_ok = (not n["abs_slow_by_dev_threshold"]) and s["abs_slow_by_dev_threshold"]
    out["per_pair"].append({"phase": ph, "init": p["init"], "prefix_identical": p["prefix_identical"], "ratio_mean": p["ratio_mean_speed_S_over_N"], "ratio_p95": p["ratio_p95_speed_S_over_N"],
        "ratio_dur": p["ratio_duration_S_over_N"], "ratio_path": p["ratio_path_length_S_over_N"], "pair_manner_pass": p["pair_manner_pass"], "abs_criterion_pass": abs_ok,
        "pair_manner_pass_with_abs": bool(p["pair_manner_pass"] and abs_ok), "pair_joint_pass": bool(p["pair_manner_pass"] and abs_ok and n["ordinary_gate"] and s["ordinary_gate"]),
        "other_stage_max_abs_mean_speed_diff": max(abs((v["mean_speed_N"] or 0) - (v["mean_speed_S"] or 0)) for k, v in p["other_stages_N_vs_S"].items() if k != "release_window"),
        "other_stage_steps_equal": all(v["n_steps_N"] == v["n_steps_S"] for k, v in p["other_stages_N_vs_S"].items()),
        "release_mean_speed_N_S": [p["other_stages_N_vs_S"]["release_window"]["mean_speed_N"], p["other_stages_N_vs_S"]["release_window"]["mean_speed_S"]]})
json.dump(out, open(R / "pilot_summary.json", "w"), indent=1, default=str)
for k in ("official_success_ever", "instruction_completed", "contact_clean_v1_1", "official_success_final", "ordinary_gate"):
    for c in "NS": print("COUNT", k, c, sum(x[k] for x in out["per_run"] if x["cond"] == c), "/ 8")
print("PAIRS", "manner", sum(p["pair_manner_pass"] for p in out["per_pair"]), "abs", sum(p["abs_criterion_pass"] for p in out["per_pair"]), "joint", sum(p["pair_joint_pass"] for p in out["per_pair"]), "prefix", sum(p["prefix_identical"] for p in out["per_pair"]), "/ 8")
print("MAXOTHER", max(p["other_stage_max_abs_mean_speed_diff"] for p in out["per_pair"]), "stepsEqual", all(p["other_stage_steps_equal"] for p in out["per_pair"]))
for ph in ("dev", "new"):
    for c in "NS":
        xs = [x for x in out["per_run"] if x["phase"] == ph and x["cond"] == c]
        print("DESC", ph, c, "mean", round(np.mean([x["descent_mean"] for x in xs]), 4), "p95", round(np.mean([x["descent_p95"] for x in xs]), 4), "dur", {x["descent_dur_s"] for x in xs}, "median range", min(x["descent_median"] for x in xs), max(x["descent_median"] for x in xs))
print("M3", sorted({(x["M3"]["official_in_final"], x["M3"]["detached_from_hand"], x["M3"]["tilt_deg"], x["M3"]["z_margin_to_official_In_lower_bound_m"], x["M3"]["book_displacement_last_10_control_steps_m"]) for x in out["per_run"]}))
print("FIRSTSUCC_vs_RELEASE", [(x["init"], x["cond"], x["official_first_success_step"], x["release"][0]) for x in out["per_run"]][:4])
print("EXCUSED", sum(len(x["v11_excused"]) for x in out["per_run"]), "runs_with", sum(1 for x in out["per_run"] if x["v11_excused"]), "v0clean", sum(x["contact_clean_v0"] for x in out["per_run"]))
# figure: dev init 0
fig, ax = plt.subplots(figsize=(11, 4.6)); col = {"N": "tab:blue", "S": "tab:red"}
for c, name in (("N", "N normal place (0.12 m/s cmd)"), ("S", "S slow place (0.06 m/s cmd)")):
    z = np.load(R / "dev" / "traj" / f"init0_{c}.npz"); st = z["stage"]; v = z["eef_speed_mps"]; t = np.arange(len(v)) * 0.05
    ax.plot(t, v, color=col[c], lw=1.4, label=name)
    b = [0] + [k for k in range(1, len(st)) if st[k] != st[k - 1]]
    for k in b: ax.axvline(k * 0.05, color=col[c], ls=":", lw=0.7, alpha=0.6)
    for k0, k1 in zip(b, b[1:] + [len(st)]):
        if k0 == b[-2] or True: pass
    rs, re_ = int(z["release_start"]), int(z["release_end"])
    ax.axvspan(rs * 0.05, re_ * 0.05, color=col[c], alpha=0.12); ax.axvline(rs * 0.05, color=col[c], ls="-", lw=1.6)
    pk = [k for k in range(len(st)) if st[k].endswith(".place")]; ax.text(pk[0] * 0.05, 0.15 + (0.012 if c == "S" else 0), "place start " + c, color=col[c], fontsize=7, rotation=90, va="bottom")
ax.set_xlabel("time (s)"); ax.set_ylabel("EEF linear speed (m/s)"); ax.set_title("T5 dev init 0: same prefix, only place-stage speed differs (dotted = stage boundaries; solid + band = release start/window)")
ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(R / "speed_time_init0.png", dpi=140); print("FIG", (R / "speed_time_init0.png").stat().st_size)
