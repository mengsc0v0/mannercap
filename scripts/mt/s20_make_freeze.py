"""Writes reports/step20_selection_eval/FREEZE_MANIFEST.json (hashes of every file the run depends on).  usage: s20_make_freeze.py "<pytest summary>" """
import hashlib
import json
import os
import subprocess
import sys

old = json.load(open("reports/step13_offline/FREEZE_MANIFEST.json"))
files = list(old["code_sha256"])
for f in ("scripts/mt/s20_selection_eval.py", "scripts/mt/s20_behavior_probe.py", "scripts/mt/s20_make_freeze.py",
          "tests/test_selection_eval_budget.py", "synthesis_v2/libero_backend.py", "synthesis_v2/native_osc_env.py",
          "synthesis_v2/native_producer.py", "configs/experience_retrieval_e1.json", "reports/step10_experience/library.json"):
    if f not in files:
        files.append(f)
sh = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
cfg = json.load(open("configs/selection_eval_e2.json"))
m = {"schema": "mannercap.freeze.v2",
     "scope": "T0/T7 x init30-33 x R1-R4, arms rule+recover, retrieval+recover, retrieval+rerun (96 requests)",
     "code_commit": head + " (parent of the freeze commit that adds this file; code files below are hashed in the working tree)",
     "branch": "work/experience-library-6phase-20260928", "tests": sys.argv[1] + " (full suite, after the logging patch and budget b2 tests)",
     "behavior_probe": "reports/step20_behavior_probe/compare_pre_vs_post.json: 16 executions bitwise identical before/after the logging patch and identical to stored step-12 results",
     "budget_version": "b2", "executor_guard": {"contact_threshold_n": 0.1, "enabled": True, "max_retries": 3, "min_clearance_m": 0.0},
     "contact_policy_version": "contact_policy_v1.1",
     "e1_config": {"path": "configs/experience_retrieval_e1.json", "content_hash": "6dfb9a519e7d93b2c7d78d694edbd9650f1d2df6280fd218168c9a4512ae08ef", "sha256_file": sh("configs/experience_retrieval_e1.json")},
     "experience_library": {"path": "reports/step10_experience/library.json", "content_hash": "177c053d241f7dbcbee06e1380fccd468f1460c44f7441f8ade38be0503971c4", "sha256_file": sh("reports/step10_experience/library.json")},
     "eval_config": {"path": "configs/selection_eval_e2.json", "content_hash": cfg["content_hash"], "sha256_file": sh("configs/selection_eval_e2.json")},
     "inits": cfg["scope"]["inits"], "requests": cfg["scope"]["requests"], "phases": cfg["phases"],
     "init_ledger": "reports/archive_20261101/init_usage_ledger.json (T0/T7 inits 30-49 unused before this run)",
     "code_sha256": {f: sh(f) for f in files},
     "frozen_during_run": "library, thresholds, guard, contact policy, candidate pool, requests; no online write-back",
     "cost_books": {"method": "executed_substeps of every execution, cache hits at recorded cost (used for every per-arm comparison)", "spent": "order-dependent, report only as total compute"}}
os.makedirs("reports/step20_selection_eval", exist_ok=True)
json.dump(m, open("reports/step20_selection_eval/FREEZE_MANIFEST.json", "w"), indent=1)
print(len(files), "files hashed")
