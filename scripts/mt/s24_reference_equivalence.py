"""Step 24 (item 1): a legacy inline ``grasp_reference`` and the SAME reference selected by ``grasp_reference_id`` must give the
identical planner verdict and candidate; the stored Step 23 first-grasp record is the third witness.
T8, init 0, approach check on, no guard: the planner vetoes the first grasp before any motion (n_steps 0), so this runs no
episode and uses no budget of the Step 24 repair experiment.  Also records the refusals (unknown id, another task's reference).
usage: s24_reference_equivalence.py   -> reports/step24_reference_library/equivalence_T8_init0.json"""
import json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, SPECS, make_plan, make_program, make_session  # noqa: E402
from synthesis_v2.grasp_references import load_library  # noqa: E402
from synthesis_v2.native_producer import GripPlan  # noqa: E402
from synthesis_v2.program_executor import produce_program_episode  # noqa: E402
from synthesis_v2.program_patch import apply_patch, make_anchor  # noqa: E402

T, INIT = 8, 0
OUT = ROOT / "reports" / "step24_reference_library"; OUT.mkdir(parents=True, exist_ok=True)
lib = load_library()
base = make_program(T)
rid0 = "libero_10.t8.demo_0.close_111.moka_pot_2"
inst0 = "pot_first.pick_1.approach"
anchor = make_anchor(task="T8", plan="approach_check")
patch = {"hypothesis": "equivalence: select the registered reference the inline value was registered under",
         "ops": [{"op": "unset_param", "instance_id": inst0, "key": "grasp_reference"},
                 {"op": "set_param", "instance_id": inst0, "key": "grasp_reference_id", "value": rid0}]}


def run(prog):
    s = make_session(T, INIT, prog)
    plan = make_plan(T, prog, grip=GripPlan(1.0, 100.0, settle_after_open=1), approach_check=True,
                     stop_on_stage_timeout=True, require_bilateral_grasp=True)
    ep = produce_program_episode(prog, INIT, plan, keep_images=False, session=s)
    s.close()
    return {"status": ep.status, "n_steps": int(ep.n_steps), "approach_plan": ep.meta.get("approach_plan")}


res = {"patch": patch}
pr = apply_patch(base, patch, frozen_anchor=anchor, current_anchor=anchor, plan=type("P", (), {"approach_check": True})())
assert pr.ok, pr.to_dict()
res["patch_result"] = pr.to_dict()
legacy, by_id = run(base), run(pr.program)
res["legacy_inline"], res["by_reference_id"] = legacy, by_id
cand = lambda r: r["approach_plan"][inst0.rsplit(".", 1)[0]]["candidates"]  # noqa: E731
stored = json.load(open(ROOT / "reports/step23_shared_fixes/t8_first_grasp_init0_fullpose.json"))["approach_plan"]["pot_first.pick_1"]["candidates"]
res["equal"] = {"legacy_vs_id_candidates": cand(legacy) == cand(by_id), "legacy_vs_step23_record_candidates": cand(legacy) == stored,
                "status_equal": legacy["status"] == by_id["status"], "n_steps_equal": legacy["n_steps"] == by_id["n_steps"]}
res["provenance_recorded"] = {"legacy": legacy["approach_plan"]["pot_first.pick_1"].get("reference"),
                              "by_id": by_id["approach_plan"]["pot_first.pick_1"].get("reference")}
# refusals (no simulation)
bad = lambda v: apply_patch(base, {"hypothesis": "x", "ops": [{"op": "unset_param", "instance_id": inst0, "key": "grasp_reference"},  # noqa: E731
        {"op": "set_param", "instance_id": inst0, "key": "grasp_reference_id", "value": v}]},
        frozen_anchor=anchor, current_anchor=anchor, plan=type("P", (), {"approach_check": True})()).to_dict()
res["refusals"] = {"unknown_id": bad("libero_10.t8.demo_9.close_1.moka_pot_2"),
                   "other_task_object": bad("libero_10.t5.demo_0.close_57.black_book_1"),
                   "other_task_mug": bad("libero_10.t4.median_n40.porcelain_mug_1")}
(OUT / "equivalence_T8_init0.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({"equal": res["equal"], "status": legacy["status"], "provenance": res["provenance_recorded"],
                  "refusals": {k: (v["refusal"], v["errors"]) for k, v in res["refusals"].items()}}, indent=1))
