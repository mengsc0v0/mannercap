"""Step 2/3 (2026-11-01): write the calibrated-lookup definition and the Step 3 evaluation PLAN (not run).  Offline.

configs/calibrated_lookup_c1.json   definition + equivalence outcome of the 'calibrated lookup' control
configs/selection_eval_e2.json      frozen plan for the fresh-init evaluation (selection x recovery); NOT executed
Both carry a content_hash over their canonical JSON (without the hash field)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def put(path, obj):
    body = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    obj["content_hash"] = hashlib.sha256(body.encode()).hexdigest()
    (ROOT / path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True))
    print(path, obj["content_hash"][:16])


eq = json.loads((ROOT / "reports/step18_lookup_equivalence/equivalence.json").read_text())
lib = json.loads((ROOT / "reports/step10_experience/library.json").read_text())
e1 = json.loads((ROOT / "configs/experience_retrieval_e1.json").read_text())

put("configs/calibrated_lookup_c1.json", {
    "schema": "mannercap.calibrated_lookup.v1",
    "status": "definition frozen; NOT a separate method (see decision)",
    "question": "is a simple calibration over the same historical data enough to match the experience library?",
    "definition": {
        "information_used": "historical measured response only (hold_min/hold_max per control candidate) of the SAME task, "
                            "SAME requested object, SAME stage window, measurement_valid and grasp_stable records",
        "not_used": "current init index, current object pose or any current-state matching",
        "per_candidate_summary": "records whose target_hi_n equals the candidate's hi (all same-object records if none); group by "
                                 "target_lo_n; MEDIAN of hold_min and of hold_max per lo; piecewise-linear in lo, clamped outside the measured lo range",
        "ranking": "same as order_B: candidates whose predicted interval lies inside the requested band first, by descending margin; "
                   "then by distance between the predicted-interval centre and the band centre; ties by pool order",
        "implementation": "scripts/mt/s18_lookup_equivalence.py:order_lookup"},
    "equivalence_check": {"file": "reports/step18_lookup_equivalence/equivalence.json", "ranking_level_only": True,
                          "n_requests": eq["n_requests"], "same_full_order": eq["same_full_order"],
                          "same_first_choice": eq["same_first_choice"],
                          "only_difference": "library retrieval aggregates by ENVELOPE (min hold_min / max hold_max) instead of the median; its task / all-record fallbacks are never triggered for the requested objects"},
    "coverage_fact": {"measured_candidates_in_library": [[1.0, 22.0], [1.0, 100.0], [35.0, 100.0]],
                      "pool_candidates_with_direct_measurement": [[1.0, 22.0], [1.0, 100.0]], "pool_size": 8,
                      "meaning": "6 of 8 pool candidates are ranked by interpolation / clamping between lo=1 and lo=35 anchors; stage matching is vacuous "
                                 "(one stage window in all 174 records)"},
    "decision": "The library retrieval IS a calibrated lookup in information terms. No separate 'calibrated lookup' arm is constructed or named. "
                "The method is positioned as measured-response parameter retrieval. The aggregation (median vs envelope) is a within-method "
                "sensitivity, pre-registered in selection_eval_e2.json as an offline-only comparison.",
    "library_content_hash": lib["content_hash"], "e1_content_hash": e1["content_hash"],
})

put("configs/selection_eval_e2.json", {
    "schema": "mannercap.selection_eval.v1", "status": "PLAN FROZEN FOR REVIEW; not run; run only after fairness confirmation and the logging patch below",
    "questions": ["does measured-response retrieval beat the face-value rule on the first candidate and on final joint pass within the fixed candidate budget?",
                  "does stage recovery keep candidates/verdicts identical to whole-episode rerun and raise qualified output per fixed physical budget?"],
    "no_preset_expectation": "no direction of improvement is assumed; null and negative results are reported with the same tables",
    "scope": {"tasks": [0, 7], "requested_objects": e1["scope"]["requested_object"],
              "inits": {"0": [30, 31, 32, 33], "7": [30, 31, 32, 33]},
              "init_basis": "reports/archive_20261101/init_usage_ledger.json: T0/T7 inits 30-49 never used in any file under reports/; 22-25 were used for repair, 26-29 for the previous evaluation (and tuning on 26-27)",
              "kept_untouched": {"final_holdout": list(range(40, 50)), "spare": list(range(34, 40))},
              "requests": e1["manner_requests"], "n_requests_per_arm": 32,
              "clusters": "8 (task x init); request-level pairs are NOT independent within an init"},
    "shared_across_arms": {"candidate_pool": e1["candidates"], "attempt_budget": e1["attempt_budget"], "executor": e1["executor"],
                           "guard": {"enabled": True, "contact_threshold_n": 0.1, "min_clearance_m": 0.0, "max_retries": 3},
                           "contact_policy": "contact_policy_v1.1", "pass_criterion": e1["pass_criterion"],
                           "other_object_fixed_plan": e1["scope"]["other_object_fixed_plan"],
                           "library": {"path": "reports/step10_experience/library.json", "content_hash": lib["content_hash"], "write_back": False},
                           "e1_config_hash": e1["content_hash"],
                           "episode_cache": "per (task, init, candidate, guard-state); cache hits are charged at recorded cost in the METHOD book"},
    "arms": {
        "phase_A_select_quality": [{"id": "rule_recover", "selection": "rule (face-value lower target, as in step12)", "recovery": "recover"},
                                   {"id": "retrieval_recover", "selection": "library order_B (envelope)", "recovery": "recover"}],
        "phase_B_recovery_effect": [{"id": "retrieval_rerun", "selection": "library order_B (envelope)", "recovery": "rerun"}],
        "not_repeated": "rule/rerun: development result exists (step12) and is not re-run on the new inits",
        "calibrated_lookup_arm": "none (equivalent, see configs/calibrated_lookup_c1.json). Median-aggregation ranking is computed OFFLINE only and reported as sensitivity; its candidates may lack executions and are then reported as 'not executed'"},
    "primary_outcomes": [
        "first-candidate task + manner(force band) + contact joint pass, per selection",
        "final joint pass within the fixed candidate budget (3 force candidates, guard max_retries 3 over the whole request)",
        "complete generation cost per request, failed requests included (METHOD book substeps; spent book reported separately and never used as per-arm cost)",
        "same selection: rerun vs recover agreement of candidate sequence, per-candidate verdicts, hold_min/hold_max (retrieval arm only, phase B)",
        "qualified requests within a fixed physical budget, computed offline from the logs: budgets = {0.25, 0.5, 0.75, 1.0} x total METHOD substeps of retrieval_rerun; requests ordered by (task, init, request id)"],
    "analysis": {"paired_by": "(task, init, request)", "interval": "cluster bootstrap over the 8 (task x init) clusters, 10000 resamples, seed 20261101",
                 "reported_separately": ["pair-level", "run-level (init-level)"],
                 "never": "no threshold, library, guard or policy change after seeing results; no write-back to the library"},
    "required_record_fields_per_refusal_and_execution": {
        "restored_to_stage": "instance id of the approach snapshot that was restored (rerun arm: 'episode_start')",
        "restore_state_identity": "sha256 of the serialised restored session state plus the snapshot's control step",
        "parameter_effective_stage": "first stage instance in which the modified parameter (grasp candidate / force candidate) takes effect",
        "executed_substeps_split": {"prefix_reused_not_executed": "int", "executed_after_restore": "int", "discarded_by_guard": "int"},
        "real_executed_substeps": "int per execution, excluding cache hits (cache hits listed separately with recorded cost)"},
    "prerequisites_before_running": [
        "implement and unit-test the logging fields above in synthesis_v2/program_executor.py (currently the refusal record has only check, control_step, refused_candidate, approach_instance, aborted_episode)",
        "full test suite must pass; re-hash executor/guard/contact-policy/planner/backend files into the freeze manifest and commit BEFORE any run",
        "add a post-run full-trajectory comparison of the final execution of recover vs rerun when the candidate sequence is equal"],
    "known_limits_to_state_with_results": [
        "library supports directly 2 of 8 pool candidates; others are interpolated between lo=1 and lo=35 anchors",
        "same-task same-object reuse only; no cross-task transfer claim",
        "guard and contact policy were tuned on T0/T7 inits 26-27, which are not in this evaluation",
        "rule baseline is a face-value rule; the library is information-equivalent to a calibrated lookup, so a 'better than calibration' claim is not tested"],
    "base_commit_at_freeze": "43e3c9b",
})
