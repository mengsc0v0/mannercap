"""Apply the pre-registered T8 demo rule (see s22_t8_demo_extract.py docstring) to the two replay dirs; write the selection."""
import json
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "reports" / "step22_t8_onboard"
A = {p.stem: json.loads(p.read_text()) for p in sorted((D / "demo_replay_A").glob("demo_*.json"))}
B = {p.stem: json.loads(p.read_text()) for p in sorted((D / "demo_replay_B").glob("demo_*.json"))}
POTS = ("moka_pot_1", "moka_pot_2")
rows, chosen = [], None
for k in sorted(A, key=lambda s: int(s.split("_")[1])):
    ok = bool(A[k]["replay_success"] and B[k]["replay_success"])
    lifted = {p: [e for e in A[k]["close_events"] if e["lifted"] == p and e["lifted_ge_5cm"]] for p in POTS}
    both = all(lifted[p] for p in POTS)
    rows.append({"demo": k, "replay_A": A[k]["replay_success"], "replay_B": B[k]["replay_success"],
                 "informational_each_pot_rise_ge_5cm_in_80_steps": both, "n_close_events": len(A[k]["close_events"]),
                 "A_B_close_steps_equal": A[k]["close_steps"] == B[k]["close_steps"]})
    if chosen is None and ok:
        chosen = k
sel = {"rule": "lowest index replay-successful in BOTH independent replays (T5 rule); per pot the LAST close event whose lifted pot it is; no averaging",
       "n_demos_replayed": len(A), "n_replays": len(A) + len(B), "per_demo": rows, "chosen": chosen}
if chosen:
    r = A[chosen]
    ref, tgt = {}, {}
    for p in POTS:
        e = [x for x in r["close_events"] if x["lifted"] == p][-1]
        ref[p] = {"rel_obj": [round(v, 4) for v in e["rel_obj"]], "hand_yaw_rel_deg": round(e["hand_yaw_rel_deg"], 1), "close_step": e["close_step"]}
        tgt[p] = {"final_in_site_frame": r["final_pots_in_site_frame"][p]}
    sel.update(grasp_reference=ref, target_final_site_frame=tgt, site=r["site"], init_pots=r["init_pots"], close_order=[e["lifted"] for e in r["close_events"]])
(D / "demo_selection.json").write_text(json.dumps(sel, indent=1))
print(json.dumps({k: sel[k] for k in sel if k != "per_demo"}, indent=1)); [print(x) for x in rows]
