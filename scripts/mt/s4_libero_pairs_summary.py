"""Step 4d: aggregate the swapped-pair audit (denominators explicit)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "reports" / (sys.argv[1] if len(sys.argv) > 1 else "step4_libero_pairs_20261025")


def passes(r) -> dict:
    ba = r["band_audit"]
    return {"completion": r["completion"], "measure_all_checks": all(r["measure_pass"].values()),
            "replay_bitwise": r["replay"]["bitwise_all_columns"], "resume_all": all(x["bitwise"] for x in r["resume"]),
            "event_sequence": r["events"]["event_sequence_exact"], "events_in_stage": r["events"]["events_inside_expected_generator_stage"],
            "phase_zero_mismatch": r["events"]["phase_mismatch_steps_outside_6step_boundary"] == 0, "online_eq_offline": r["online_equals_offline_events"],
            "hold_in_band_both_objects": all(v["all_hold_steps_in_band"] for v in ba.values()),
            "peak_le_cap_both_objects": all(v["peak_le_cap"] for v in ba.values())}


def main() -> int:
    recs = {(int(p.stem.split("_")[0][4:]), p.stem.split("_")[1]): json.loads(p.read_text()) for p in sorted(D.glob("init*_*.json"))}
    out = {"n_trajectories": len(recs), "per_check": {}, "per_combo_all_pass": {}, "pairs": {}}
    P = {k: passes(r) for k, r in recs.items()}
    for chk in next(iter(P.values())):
        out["per_check"][chk] = [sum(p[chk] for p in P.values()), len(P)]
    for c in ("GF", "FG", "GG", "FF"):
        ks = [k for k in P if k[1] == c]
        out["per_combo_all_pass"][c] = [sum(all(P[k].values()) for k in ks), len(ks)]
    inits = sorted({k[0] for k in P})
    pairs = []
    for i in inits:
        if (i, "GF") in recs and (i, "FG") in recs:
            gf, fg = recs[(i, "GF")], recs[(i, "FG")]
            ok_traj = all(P[(i, "GF")].values()) and all(P[(i, "FG")].values())
            h = lambda r, o: r["band_audit"][o]["hold_mean"]
            sw = h(gf, "cream_cheese") < h(fg, "cream_cheese") and h(gf, "butter") > h(fg, "butter")
            # separation expressed against each object's own dev spread: gap between level bands
            pairs.append({"init": i, "both_trajectories_pass": ok_traj, "swap_direction_correct": bool(sw),
                          "cream_hold_GF_FG": [h(gf, "cream_cheese"), h(fg, "cream_cheese")], "butter_hold_GF_FG": [h(gf, "butter"), h(fg, "butter")],
                          "n_steps": [gf["n_steps"], fg["n_steps"]]})
    out["pairs"] = pairs
    out["pairs_all_pass"] = [sum(p["both_trajectories_pass"] and p["swap_direction_correct"] for p in pairs), len(pairs)]
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for p in pairs:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
