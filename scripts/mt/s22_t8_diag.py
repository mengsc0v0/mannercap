"""Step 22 diagnostic (read-only replay of the frozen round-1 config, no change): where does the hand stop in pot_first.descend and
what does it touch?  usage: s22_t8_diag.py INIT -> reports/step22_t8_onboard/round1/diag_init{N}.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from s7_task_config import ROOT, SPECS  # noqa: E402
import s8_task_force as S8, s11_guarded_case as S11  # noqa: E402
T, i = 8, int(sys.argv[1])
prog, ep, rec, crec, det, dt = S11.run(T, i, {r: S8.grip_for((1.0, 100.0), 1) for r, _ in SPECS[T].objects}, None)
a = ep.arrays; st = np.asarray([str(x) for x in a["stage"]])
S = np.asarray(a["state"]); d = np.flatnonzero(st == "pot_first.pick_1.descend")
P = np.asarray(a["pos_moka_pot_2"])
plan = ep.meta["approach_plan"]["pot_first.pick_1"]["chosen"]
want = P[0] + np.asarray(plan["offset"])        # planned hand position at grasp (pot origin + world offset)
ex = det.export()
rows = {k: {kk: v.get(kk) for kk in ("peak_n", "peak_step", "peak_geom", "peak_other_geom", "n_substeps", "n_substeps_inner_region",
                                    "peak_contact_point_body_local_m")} for k, v in ex.items() if "moka_pot_2" in k}
out = {"init": i, "descend_steps": int(len(d)), "hand_pos_at_descend_start": S[d[0], :3].round(4).tolist(),
       "hand_pos_at_descend_end": S[d[-1], :3].round(4).tolist(), "planned_grasp_hand_pos": want.round(4).tolist(),
       "hand_minus_planned_at_end_m": (S[d[-1], :3] - want).round(4).tolist(),
       "hand_z_trace_every_25": [(int(d[k]), round(float(S[d[k], 2]), 4)) for k in range(0, len(d), 25)],
       "pot_displacement_during_descend_m": (P[d[-1]] - P[0]).round(4).tolist(), "approach_chosen": plan, "contacts_with_pot_2": rows}
(ROOT / "reports" / "step22_t8_onboard" / "round1" / f"diag_init{i}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in out if k not in ("approach_chosen",)}, default=str)[:2500])
