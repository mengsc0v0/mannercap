"""Step 4 (2026-11-01): read-only capability match of the not-yet-onboarded LIBERO-10 tasks against the CURRENT shared
capabilities.  Static facts only: reports/step6_libero10_spec_20261025 (BDDL goals/stages), synthesis_v2/program_executor.py,
scripts/mt/s7_task_config.py, local demo files.  No simulation, no executor change."""
import json
from pathlib import Path

OUT = Path("reports/step19_capability_match")
OUT.mkdir(parents=True, exist_ok=True)
DEMO_DIR = Path("/workdata/cap-x/capx/third_party/LIBERO-PRO/libero/datasets/libero_10")
demos_local = sorted(p.name.split("_demo")[0] for p in DEMO_DIR.glob("*_demo.hdf5")) if DEMO_DIR.exists() else []

shared_now = {
    "pick_place primitives (approach/grasp/lift/transport/place/release/retreat)": "T0 T1 T7 verified or tested; T4 T5 onboarded",
    "per-object targets (TaskSpec.targets)": "added by T4 (step 9)",
    "scene-region target '@site:<name>' with offset/yaw/symmetry/lift/transport/place heights": "added by T5 (step 14-15)",
    "collision-centre alignment + place_tol_m": "added by T5",
    "grasp_reference from ONE official demo (unaveraged)": "added by T5; needs the task's demo file",
    "grasp_grid override": "added by T4; T5 stopped using it (did not reach the grasp band)",
    "contact audit v1.1, execution guard, stage recovery": "task-independent",
    "NOT available": "articulated-object primitives (switch/knob/drawer/door), in-hand mug drift handling (T4 open)"}

tasks = {
    "T8": {"language": "put both moka pots on the stove", "stages": ["pick_place moka_pot_1 -> flat_stove_1_cook_region", "pick_place moka_pot_2 -> same"],
           "init_predicate": "(Turnon flat_stove_1) already true at init per BDDL (verify at runtime)",
           "needed": ["pick_place x2", "scene-region target", "two same-type objects"],
           "covered": ["pick_place", "@site region target (T5)", "per-object targets (T4)"],
           "gaps": ["moka-pot grasp pose / handle geometry unknown", "same-type objects: binding of 'moka_pot_1' vs '_2' in language", "no local demo file"],
           "new_work_expected": "config entry + (optional) demo-derived grasp_reference; no new primitive", "demo_local": False},
    "T6": {"language": "put the white mug on the plate and the chocolate pudding to the right of the plate",
           "stages": ["pick_place porcelain_mug_1 -> plate_1", "pick_place chocolate_pudding_1 -> living_room_table_plate_right_region"],
           "needed": ["pick_place x2", "object target (On plate)", "scene-region target"],
           "covered": ["pick_place", "@site region target (T5)", "mug-on-plate config path (T4)"],
           "gaps": ["mug in-hand drift unresolved in T4 (contact policy 0/8)", "pudding grasp pose", "no local demo file"],
           "new_work_expected": "config entry; mug stage inherits T4's open problem", "demo_local": False},
    "T3": {"language": "put the black bowl in the bottom drawer and close it", "stages": ["pick_place bowl -> open drawer region", "close_articulated drawer"],
           "needed": ["pick_place", "push-close drawer primitive"], "covered": ["pick_place (adapt)"], "gaps": ["drawer-closing primitive missing"],
           "new_work_expected": "new execution primitive", "demo_local": False},
    "T9": {"language": "put the mug in the microwave and close it", "stages": ["pick_place mug -> microwave region", "close_articulated microwave door"],
           "needed": ["pick_place", "door-close primitive"], "covered": ["pick_place (adapt)"], "gaps": ["door-closing primitive missing"],
           "new_work_expected": "new execution primitive", "demo_local": False},
    "T2": {"language": "turn on the stove and put the moka pot on it", "stages": ["turn_on stove knob", "pick_place moka_pot -> cook region"],
           "needed": ["knob/switch primitive", "pick_place"], "covered": ["pick_place (adapt)"], "gaps": ["knob primitive missing"],
           "new_work_expected": "new execution primitive", "demo_local": False},
}
selection = {
    "chosen": "T8", "second": "T6", "ranking_rule": "fewest missing primitives first, then most reuse of interfaces added during T4/T5, decided BEFORE any run",
    "why": ["T8 needs no new primitive (T2/T3/T9 each need an articulation primitive)",
            "both stages use exactly the interfaces added by T4/T5 (per-object targets, '@site' region target, optional demo grasp reference), so it tests whether those shared additions are reusable without editing shared code",
            "T6 is second: it shares the interfaces too, but its mug stage inherits T4's unresolved in-hand drift, which would confound the reuse question"],
    "reuse_measurement_preregistered": {
        "shared_code_lines_changed": "git diff numstat of synthesis_v2/ ; success criterion for 'reuse' = 0 lines changed in the executor/backend for the first config-only attempt",
        "task_specific_lines": "lines added to scripts/mt/s7_task_config.py",
        "cost": "episodes + sim_wall_s from cost.sim_wall_s, failed pre-checks listed separately; human hours: record start/end per phase (T5 has none)",
        "compare_with": "T5 first-time onboarding: 12 failed pre-check episodes (140.5 s) + 21 onboarding episodes (164.4 s), shared-code additions about +120 lines executor and +66 lines backend",
        "stop_rule": "one config-only attempt on 4 inits; at most ONE pre-registered adapter-level fix; NO widening of the grasp-candidate grid; if ordinary task success is still not reached, record the gap and stop",
        "dev_inits": "use inits that are not used for any held-out claim for T8 (none used so far for T8; suggest 0-3 for onboarding, keep 40-49 untouched)"},
    "open_question_for_user": "T8 has no local official demo (only T4 and T5 demos are in LIBERO-PRO datasets/libero_10). A demo-derived grasp_reference would require obtaining that demo file; the config-only attempt does not need it."}
out = {"schema": "mannercap.capability_match.v1", "read_only": True, "shared_capabilities_now": shared_now,
       "local_demo_files_for_tasks": demos_local, "tasks": tasks, "selection": selection,
       "matrix_note": "reports/step6_libero10_spec_20261025 predates T4/T5 onboarding; its 'missing'/'adapt' labels are kept for T2/T3/T9 and refreshed here for T6/T8"}
(OUT / "capability_match.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
md = ["| 任务 | 需要 | 现有共享能力覆盖 | 缺口 | 预计新增 | 本地示范 |", "|---|---|---|---|---|---|"]
for k, v in tasks.items():
    md.append(f"| {k} | {'; '.join(v['needed'])} | {'; '.join(v['covered'])} | {'; '.join(v['gaps'])} | {v['new_work_expected']} | {'有' if v['demo_local'] else '无'} |")
(OUT / "capability_match.md").write_text("\n".join(md))
print("demos local:", demos_local)
