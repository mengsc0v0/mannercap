"""Write the freeze manifest for the step12 batch: code version, frozen configuration, library, policy/guard identity,
and a sha256 for every one of the 128-cell result files.  Run after the code and the results are committed."""
import glob
import hashlib
import json
import subprocess
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.check_output(["git", *a], text=True).strip()


E1 = json.loads(Path("configs/experience_retrieval_e1.json").read_text())
LIB = json.loads(Path("reports/step10_experience/library.json").read_text())
grids = sorted(glob.glob("reports/step12_ablation/t*_init*.json"))
n_cells = 0
heads = set()
for g in grids:
    d = json.loads(Path(g).read_text())
    n_cells += len(d["cells"])
    for c in d["cells"]:
        heads.add((c["e1_hash"], c["library_hash"], c["policy_version"], json.dumps(c["guard"], sort_keys=True)))
assert len(heads) == 1, "result files were produced under different frozen identities"
e1h, libh, pol, guard = next(iter(heads))
assert e1h == E1["content_hash"] and libh == LIB["content_hash"], "result identity != configuration on disk"

code = ["synthesis_v2/program_executor.py", "synthesis_v2/contact_policy_v1.py", "synthesis_v2/execution_guard.py",
        "synthesis_v2/contact_audit.py", "synthesis_v2/approach_planner.py", "synthesis_v2/substep_recorder.py",
        "scripts/mt/s11_guarded_case.py", "scripts/mt/s12_recovery_ablation.py", "scripts/mt/s10_retrieval_trial.py"]
dirty = [l for l in git("status", "--porcelain", "--", *code).splitlines()]
assert not dirty, f"frozen code has uncommitted changes: {dirty}"
man = {
    "schema": "mannercap.freeze.v1", "scope": "T0/T7 x init26-29 x R1-R4 (128 cells = 4 arms x 32 requests)",
    "code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "tests": "546 passed, 14 subtests (full suite, immediately before the commit)",
    "executor_guard": json.loads(guard), "contact_policy_version": pol,
    "e1_config": {"path": "configs/experience_retrieval_e1.json", "content_hash": E1["content_hash"], "sha256_file": sha("configs/experience_retrieval_e1.json"),
                  "attempt_budget": E1["attempt_budget"], "inits": E1["scope"]["inits"], "manner_requests": E1["manner_requests"]},
    "experience_library": {"path": "reports/step10_experience/library.json", "content_hash": LIB["content_hash"], "sha256_file": sha("reports/step10_experience/library.json")},
    "code_sha256": {p: sha(p) for p in code},
    "result_files": {g: sha(g) for g in grids}, "n_cells": n_cells,
    "analysis_files": {p: sha(p) for p in sorted(glob.glob("reports/step13_offline/*"))},
    "cost_books": {"spent": "executed_substeps of non-cached executions", "method": "executed_substeps of every execution, cache hits at recorded cost"},
    "init_splits": "calibration 8-13; dev 0-7,14-21; reserved 22-39 (26-29 = this batch); holdout 40-49 untouched",
}
assert n_cells == 128
Path("reports/step13_offline/FREEZE_MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
print("code_commit", man["code_commit"]); print("e1", E1["content_hash"][:16], "lib", LIB["content_hash"][:16], "policy", pol)
print("files hashed:", len(man["result_files"]), "result,", len(man["analysis_files"]), "analysis; cells", n_cells)
