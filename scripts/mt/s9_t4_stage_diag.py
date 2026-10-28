"""Per-stage trace of a T4 run (reference grasp): eef position, object heights, grasp forces at each stage end.
usage: s9_t4_stage_diag.py INIT -> prints a table; writes reports/step9_t4_onboard/stage_diag_init{N}.json"""
import sys, json, numpy as np
sys.path.insert(0, "scripts/mt"); sys.path.insert(0, ".")
from s7_task_config import make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode
i = int(sys.argv[1])
prog = make_program(4); plan = make_plan(4, prog, approach_check=True); s = make_session(4, i, prog)
ep = produce_program_episode(prog, i, plan, keep_images=False, session=s)
a = ep.arrays
print("keys", sorted(a.keys()))
st = [str(x) for x in a["stage"]]
out = {"status": ep.status, "task_ok": bool(ep.task_ok), "approach_plan": {k: {"grid_source": v["grid_source"], "chosen": v["chosen"]} for k, v in ep.meta["approach_plan"].items()}, "stages": []}
seen = list(dict.fromkeys(st))
for g in seen:
    idx = [k for k, x in enumerate(st) if x == g]; e = idx[-1]
    row = {"stage": g, "n": len(idx), "yellow_pos_end": np.round(a["pos_white_yellow_mug_1"][e], 3).tolist(),
           "white_pos_end": np.round(a["pos_porcelain_mug_1"][e], 3).tolist(),
           "left_max_n_yellow": float(np.max(a["cf_L_white_yellow_mug_1"][idx])), "right_max_n_yellow": float(np.max(a["cf_R_white_yellow_mug_1"][idx])),
           "left_max_n_white": float(np.max(a["cf_L_porcelain_mug_1"][idx])), "right_max_n_white": float(np.max(a["cf_R_porcelain_mug_1"][idx]))}
    out["stages"].append(row)
    print(g.ljust(28), "n", len(idx), "Y", row["yellow_pos_end"], "W", row["white_pos_end"], "Y-force L/R max %.1f/%.1f" % (row["left_max_n_yellow"], row["right_max_n_yellow"]), "W %.1f/%.1f" % (row["left_max_n_white"], row["right_max_n_white"]))
json.dump(out, open(f"reports/step9_t4_onboard/stage_diag_init{i}.json", "w"), default=str, indent=1)
for inst, info in (ep.meta.get("grasp_policy_by_instance") or {}).items():
    print("GRASP", inst, "closing_steps", info["closing_steps"], "reached", info["reached_band"], "in_band_at", info["in_band_at"],
          "total_n", [round(t["total_n"], 1) for t in info["trace"]], "cmd", [t["cmd"] for t in info["trace"]][-3:])
print("state shape", a["state"].shape)
for g in ("white_mug.pick_1.descend", "white_mug.pick_1.grasp", "yellow_mug.pick_1.descend", "yellow_mug.pick_1.grasp", "yellow_mug.pick_1.lift"):
    idx = [k for k, x in enumerate(st) if x == g]
    print("STATE", g, "first", np.round(a["state"][idx[0]], 3).tolist(), "last", np.round(a["state"][idx[-1]], 3).tolist())
print("TARGET yellow hand", (np.array(ep.meta["approach_plan"]["yellow_mug.pick_1"]["chosen"]["offset"]) + np.array(ep.arrays["pos_white_yellow_mug_1"][0])).round(3).tolist())
print("TARGET white hand", (np.array(ep.meta["approach_plan"]["white_mug.pick_1"]["chosen"]["offset"]) + np.array(ep.arrays["pos_porcelain_mug_1"][0])).round(3).tolist())
for inst, info in (ep.meta.get("grasp_policy_by_instance") or {}).items():
    print("NOTES", inst, "reverse_pulses", info["reverse_pulses"], "reclose", info["reclose_pulses"], [(t["cmd"], round(t["total_n"], 1), t["note"]) for t in info["trace"][7:]])
