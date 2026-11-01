"""Step 1 (2026-11-01): evidence archive + hash manifest, no simulation.

* copies the small ledgers/logs that exist only under /tmp (prod_v2, prod_v3, p08b) into
  reports/archive_20261101/tmp_logs (bulk arrays/videos are NOT copied; their absence is recorded);
* hashes every library source file, the step8 source trees and the copied logs;
* records what cannot be traced instead of inventing it.

Run from the repo root:  python scripts/evidence_archive_20261101.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ARCH = Path("reports/archive_20261101")
TMP_SETS = {"prod_v2": Path("/tmp/prod_v2"), "prod_v3": Path("/tmp/prod_v3"), "p08b": Path("/tmp/p08b")}
KEEP_SUFFIX = {".json", ".jsonl", ".log", ".txt", ".md"}
SOURCE_TREES = ["reports/step8_t0_raw", "reports/step8_t0_pairs", "reports/step8_t0_response",
                "reports/step8_t7_raw", "reports/step8_t7_pairs", "reports/step8_t7_response",
                "reports/step8_t7_pairs_summary.json",
                "configs/manner_calibration_libero_t0.v2.json", "configs/manner_calibration_libero_t7.v2.json"]
EXTRA_SCRIPTS = ["/tmp/recompute12.py", "/tmp/attstats.py"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tracked(p: Path) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode == 0


def files_of(p: Path):
    return [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())


def main() -> None:
    ARCH.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], {}
    for name, src in TMP_SETS.items():
        n_skip, skip_bytes, skip_ext = 0, 0, Counter()
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in KEEP_SUFFIX:
                n_skip += 1
                skip_bytes += f.stat().st_size
                skip_ext[f.suffix.lower() or "(none)"] += 1
                continue
            dst = ARCH / "tmp_logs" / name / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            entries.append({"role": f"tmp_log:{name}", "path": str(dst), "origin": str(f),
                            "sha256": sha(dst), "bytes": dst.stat().st_size, "git_tracked_before_archive": False})
        skipped[name] = {"origin_dir": str(src), "not_copied_files": n_skip, "not_copied_bytes": skip_bytes,
                         "not_copied_by_suffix": dict(skip_ext)}
    for s in EXTRA_SCRIPTS:
        sp = Path(s)
        if sp.exists():
            dst = ARCH / "derivation_scripts" / sp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dst)
            entries.append({"role": "derivation_script", "path": str(dst), "origin": s, "sha256": sha(dst),
                            "bytes": dst.stat().st_size, "git_tracked_before_archive": False})
    for t in SOURCE_TREES:
        for f in files_of(Path(t)):
            entries.append({"role": "library_source_tree", "path": str(f), "origin": str(f), "sha256": sha(f),
                            "bytes": f.stat().st_size, "git_tracked_before_archive": tracked(f)})
    lib_path = Path("reports/step10_experience/library.json")
    lib = json.loads(lib_path.read_text())
    miss, per_src = [], Counter()
    for r in lib["records"]:
        per_src[r["source"]] += 1
        if not Path(r["source"]).exists():
            miss.append(r["source"])
    entries.append({"role": "library", "path": str(lib_path), "sha256": sha(lib_path), "bytes": lib_path.stat().st_size,
                    "content_hash_field": lib["content_hash"], "git_tracked_before_archive": tracked(lib_path)})
    e1 = Path("configs/experience_retrieval_e1.json")
    entries.append({"role": "e1_config", "path": str(e1), "sha256": sha(e1), "bytes": e1.stat().st_size,
                    "git_tracked_before_archive": tracked(e1)})
    manifest = {
        "schema": "mannercap.evidence_archive.v1", "created_for": "2026-11-01 Step 1",
        "git_head_when_built": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "n_files": len(entries),
        "library": {"path": str(lib_path), "n_records": lib["n_records"], "content_hash": lib["content_hash"],
                    "distinct_source_files": len(per_src), "source_files_missing": sorted(set(miss)),
                    "source_files_all_present": not miss},
        "not_copied_bulk_data": skipped,
        "cannot_trace": [
            "prod_v2 / prod_v3 / p08b: no git commit, contact-policy version or calibration hash was written into the "
            "run outputs; the code state at run time is unknown (only file hashes of the copied ledgers are known)",
            "prod_v2 / prod_v3 / p08b bulk arrays (npz/mp4/...) stay only under /tmp and are not archived",
            "step8 raw sources: scripts committed at 3d427f2/59e236f per git log, but the run outputs record no commit; "
            "the link is by file time and commit order only",
            "human working hours for any onboarding: not recorded",
        ],
        "files": entries,
    }
    out = ARCH / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("files:", len(entries), "| library sources distinct:", len(per_src), "missing:", len(set(miss)), "->", out)
    print(json.dumps(skipped, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
