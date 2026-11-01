"""Step 3 prerequisite (2026-11-01): does the new recovery logging change execution behaviour?

Replays a fixed set of step-12 requests (both recovery modes) with the CURRENT code, hashes every per-execution array
and compares (a) with the stored step-12 result and (b) across code versions: run this file once BEFORE the logging
patch and once AFTER it (``--tag``) and compare the two output files with ``--compare``.
usage: s20_behavior_probe.py run TAG | compare TAG_A TAG_B"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

OUT = Path("reports/step20_behavior_probe")
# (selection, task, init, request id), fixed BEFORE any probe run from the stored step-12 cells:
#   retrieval T0 i26 R1: recover refuses twice in one episode, rerun needs 3 executions (2 refusals)
#   rule      T0 i27 R2: one refusal per force candidate, 3 force candidates
#   retrieval T7 i26 R1: one refusal (recover: one restore; rerun: 2 executions)
SET = [("retrieval", 0, 26, "R1"), ("rule", 0, 27, "R2"), ("retrieval", 7, 26, "R1")]


def digest_arrays(ep, rec):
    h = hashlib.sha256()
    for k in sorted(ep.arrays):
        v = np.asarray(ep.arrays[k])
        if v.dtype == object:
            h.update(repr(v.tolist()).encode())
        else:
            h.update(k.encode()); h.update(np.ascontiguousarray(v).tobytes())
    h.update(np.ascontiguousarray(rec["ctrl"]).tobytes())
    for n in sorted(rec["L"]):
        h.update(np.ascontiguousarray(rec["L"][n]).tobytes()); h.update(np.ascontiguousarray(rec["R"][n]).tobytes())
    return h.hexdigest()


def run(tag):
    import s12_recovery_ablation as S12
    orig = S12.run_guarded
    seen = []

    def wrapped(task, init, grips, guard):
        out = orig(task, init, grips, guard)
        prog, ep, rec, crec, det, dt = out
        seen.append({"arrays_sha256": digest_arrays(ep, rec), "array_keys": sorted(ep.arrays), "n_steps": int(ep.n_steps)})
        return out
    S12.run_guarded = wrapped
    res = []
    for sel, task, init, rid in SET:
        req = [r for r in S12.E1["manner_requests"] if r["id"] == rid][0]
        stored = {c["recovery"]: c for c in json.loads(Path(f"reports/step12_ablation/t{task}_init{init}.json").read_text())["cells"]
                  if c["selection"] == sel and c["request"]["id"] == rid}
        for rec_mode in ("recover", "rerun"):
            S12.CACHE.clear(); seen.clear()
            cell = S12.run_request(task, init, req, sel, rec_mode)
            ex = []
            for i, e in enumerate(cell["executions"]):
                st = stored[rec_mode]["executions"][i] if i < len(stored[rec_mode]["executions"]) else {}
                ex.append({"candidate": e["candidate"], "status": e["status"], "n_steps": e["n_steps"],
                           "executed_substeps": e["executed_substeps"], "formal_rows": e["formal_rows"],
                           "hold_min": e["measured"]["hold_min"], "hold_max": e["measured"]["hold_max"],
                           "joint": e["joint_passed"], "refused": [[r["check"], r["control_step"]] for r in e["guard_refused"]],
                           "arrays_sha256": seen[i]["arrays_sha256"] if i < len(seen) else None,
                           "matches_step12": {k: (e.get(k) == st.get(k)) for k in ("candidate", "status", "n_steps", "executed_substeps", "formal_rows", "joint_passed")}
                           if st else None,
                           "hold_equal_step12": (e["measured"]["hold_min"] == st.get("measured", {}).get("hold_min") and e["measured"]["hold_max"] == st.get("measured", {}).get("hold_max")) if st else None})
            res.append({"selection": sel, "task": task, "init": init, "request": rid, "recovery": rec_mode, "executions": ex})
            print(sel, task, init, rid, rec_mode, len(ex), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe_{tag}.json").write_text(json.dumps(res, indent=1))


def compare(a, b):
    A = json.loads((OUT / f"probe_{a}.json").read_text()); B = json.loads((OUT / f"probe_{b}.json").read_text())
    keys = ("candidate", "status", "n_steps", "executed_substeps", "formal_rows", "hold_min", "hold_max", "joint", "refused", "arrays_sha256")
    diffs, n = [], 0
    for x, y in zip(A, B):
        assert (x["selection"], x["task"], x["init"], x["request"], x["recovery"]) == (y["selection"], y["task"], y["init"], y["request"], y["recovery"])
        if len(x["executions"]) != len(y["executions"]):
            diffs.append(("n_executions", x["selection"], x["task"], x["init"], x["request"], x["recovery"])); continue
        for i, (e, f) in enumerate(zip(x["executions"], y["executions"])):
            n += 1
            for k in keys:
                if e[k] != f[k]:
                    diffs.append((k, x["selection"], x["task"], x["init"], x["request"], x["recovery"], i))
    s12 = [(e["matches_step12"], e["hold_equal_step12"]) for r in B for e in r["executions"]]
    ok12 = sum(1 for m, h in s12 if m and all(m.values()) and h)
    out = {"executions_compared": n, "differences_between_versions": diffs, "bitwise_identical_between_versions": not diffs,
           "executions_matching_stored_step12": ok12, "executions_total": len(s12)}
    (OUT / f"compare_{a}_vs_{b}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out)[:900])


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else compare(sys.argv[2], sys.argv[3])
