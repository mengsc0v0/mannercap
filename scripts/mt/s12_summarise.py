"""Fill the 2x2 table from the per-init grids, and audit one request's cost line by line.

Two cost books are reported side by side and never added together:
  spent  -- what this experiment paid; a cache hit costs nothing.
  method -- what the configuration would cost run on its own; a cache hit is charged its recorded cost.
Both are in physics substeps actually simulated (planner look-ahead and guard-discarded branches included).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("reports/step12_ablation")
ARMS = [(s, r) for s in ("rule", "retrieval") for r in ("rerun", "recover")]


def load():
    cells = []
    for f in sorted(OUT.glob("t*_init*.json")):
        cells += json.loads(f.read_text())["cells"]
    return cells


def table(cells):
    rows = []
    for sel, rec in ARMS:
        g = [c for c in cells if c["selection"] == sel and c["recovery"] == rec]
        rows.append({
            "selection": sel, "recovery": rec, "n_requests": len(g),
            "task_passed": sum(c["task_passed"] for c in g),
            "force_passed": sum(c["force_band_passed"] for c in g),
            "contact_passed": sum(c["contact_v1_passed"] for c in g),
            "joint_passed": sum(c["joint_passed"] for c in g),
            "substeps_spent": sum(c["substeps_spent"] for c in g),
            "substeps_method": sum(c["substeps_method"] for c in g),
            "wall_spent_s": round(sum(c["wall_spent_s"] for c in g), 1),
            "wall_method_s": round(sum(c["wall_method_s"] for c in g), 1),
            "executions_real": sum(c["n_actually_executed"] for c in g),
            "executions_cached": sum(c["n_cache_hits"] for c in g),
            "selection_s": round(sum(c["selection_seconds"] for c in g), 4)})
    return rows


def agreement(cells):
    """Within one selection rule, did rerun and recover reach the same candidate and the same verdict?"""
    out = []
    for sel in ("rule", "retrieval"):
        by = {}
        for c in cells:
            if c["selection"] == sel:
                by.setdefault((c["task"], c["init"], c["request"]["id"]), {})[c["recovery"]] = c
        for k, pair in sorted(by.items()):
            if len(pair) != 2:
                continue
            a, b = pair["rerun"], pair["recover"]
            fa = a["executions"][-1]["candidate"] if a["executions"] else None
            fb = b["executions"][-1]["candidate"] if b["executions"] else None
            ga = a["executions"][-1].get("guard_chosen") if a["executions"] else None
            gb = b["executions"][-1].get("guard_chosen") if b["executions"] else None
            out.append({"selection": sel, "task": k[0], "init": k[1], "request": k[2],
                        "same_force_candidate": fa == fb, "same_geometry_candidate": ga == gb,
                        "same_joint_verdict": a["joint_passed"] == b["joint_passed"],
                        "rerun": {"cand": fa, "geom": ga, "joint": a["joint_passed"]},
                        "recover": {"cand": fb, "geom": gb, "joint": b["joint_passed"]}})
    return out


def never_passed(cells):
    by = {}
    for c in cells:
        by.setdefault((c["task"], c["init"], c["request"]["id"]), []).append(c)
    out = []
    for k, g in sorted(by.items()):
        if len(g) == len(ARMS) and not any(c["joint_passed"] for c in g):
            last = {f'{c["selection"]}/{c["recovery"]}': {
                "task_ok": c["task_passed"], "force": c["force_band_passed"], "contact": c["contact_v1_passed"],
                "hold": (c["executions"][-1]["measured"] if c["executions"] else None),
                "unexcused": (c["executions"][-1]["v1_unexcused"] if c["executions"] else None),
                "status": (c["executions"][-1]["status"] if c["executions"] else None)} for c in g}
        else:
            continue
        out.append({"task": k[0], "init": k[1], "request": k[2], "band_n": g[0]["request"]["band_n"], "arms": last})
    return out


def audit(cells, task, init, rid):
    """Per-execution cost breakdown for one request, so the books can be checked against the raw episodes."""
    out = []
    for c in cells:
        if (c["task"], c["init"], c["request"]["id"]) != (task, init, rid):
            continue
        for e in c["executions"]:
            out.append({"selection": c["selection"], "recovery": c["recovery"], "candidate": e["candidate"],
                        "cached": e["reused_cached_episode"], "status": e["status"],
                        "executed_substeps": e["executed_substeps"], "formal_rows": e["formal_rows"],
                        "kept_control_steps": e["n_steps"],
                        "discarded_substeps": e["rolled_back_cost"]["substeps"],
                        "discarded_branches": e["rolled_back_cost"]["branches"],
                        "guard_refusals": len(e["guard_refused"]), "wall_s": e["wall_s"],
                        "joint": e.get("joint_passed")})
    return out


def main() -> int:
    cells = load()
    rep = {"schema": "mannercap.ablation_2x2_summary.v1", "n_cells": len(cells),
           "table": table(cells), "rerun_vs_recover_agreement": agreement(cells),
           "never_passed_in_any_arm": never_passed(cells),
           "audit_t7_init26_R1": audit(cells, 7, 26, "R1")}
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, default=float))
    for r in rep["table"]:
        print(f"{r['selection']:9s}/{r['recovery']:7s} n={r['n_requests']:2d} "
              f"task={r['task_passed']:2d} force={r['force_passed']:2d} contact={r['contact_passed']:2d} "
              f"joint={r['joint_passed']:2d} sub_spent={r['substeps_spent']:7d} sub_method={r['substeps_method']:7d} "
              f"wall_spent={r['wall_spent_s']:7.1f}s wall_method={r['wall_method_s']:7.1f}s "
              f"exec={r['executions_real']}+{r['executions_cached']}c")
    print("\nnever passed in any arm:", len(rep["never_passed_in_any_arm"]))
    for x in rep["never_passed_in_any_arm"]:
        print(" ", x["task"], x["init"], x["request"], x["band_n"])
    bad = [a for a in rep["rerun_vs_recover_agreement"] if not a["same_joint_verdict"]]
    print("rerun/recover verdict disagreements:", len(bad), "of", len(rep["rerun_vs_recover_agreement"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
