"""Step 2: what did repair rescue?  First branch (before ANY internal retry) vs the final in-budget result.

Where the first branch comes from.  In the recover arm the first function call already contains the repair, so its return
value is NOT the first-branch result.  In the rerun arm the first call IS the pre-repair branch (the guard aborts it).
The two arms are the same physical branch up to the first refusal; this script verifies that from the logs
(refusal check, control step, refused candidate, chain order) instead of assuming it, and reads the first branch from the
rerun arm's first execution.

Three levels per request:
  L0 first branch      : executions[0] of the rerun arm.
  L1 first force cand. : the completed branch of force attempt 1 (after geometry repair, before any force-candidate switch).
  L2 final in budget   : last execution.
Tri-state verdicts come from s13_components (aborted branch -> task/manner undetermined).
"""
import collections
import json
from pathlib import Path

from s13_components import components, is_aborted, key, load_cells

cells = load_cells()
pair = collections.defaultdict(dict)
for c in cells:
    pair[(c["selection"],) + key(c)][c["recovery"]] = c


def refusal_sig(r):
    return (r["check"], r["control_step"], tuple(r["refused_candidate"]), r["approach_instance"])


per = []
path_mismatch = []
for k, p in sorted(pair.items()):
    R, V = p["rerun"], p["recover"]
    band = R["request"]["band_n"]
    # chain of refusals in force attempt 1
    r_chain = [refusal_sig(e["guard_refused"][-1]) for e in R["executions"] if e["force_attempt"] == 1 and is_aborted(e)]
    v_chain = [refusal_sig(r) for r in V["executions"][0]["guard_refused"]]
    if r_chain != v_chain:
        path_mismatch.append((k, r_chain, v_chain))
    first = R["executions"][0]
    a1 = [e for e in R["executions"] if e["force_attempt"] == 1]
    l1 = a1[-1]
    final = R["executions"][-1]
    n_geo_all = sum(len(e["guard_refused"]) for e in V["executions"])
    n_switch = max(e["force_attempt"] for e in R["executions"]) - 1
    per.append({
        "selection": k[0], "task": k[1], "init": k[2], "request": k[3], "band_n": band,
        "L0": components(first, band), "L0_status": first["status"].split(":")[0],
        "L0_first_call_return_recover_arm": components(V["executions"][0], band),
        "L1": components(l1, band) if not is_aborted(l1) else {"task": "und", "manner": "und", "contact": "und", "joint": "und"},
        "L2": components(final, band), "L2_status": final["status"].split(":")[0],
        "geometry_repairs_total": n_geo_all, "geometry_repairs_attempt1": len(v_chain),
        "force_candidate_switches": n_switch,
        "refusal_chain_attempt1": [{"check": c[0], "control_step": c[1], "refused": list(c[2]), "approach": c[3]} for c in v_chain],
        "first_candidate": first["candidate"], "final_candidate": final["candidate"],
        "joint_at_force_attempt": R["joint_passed_at_force_attempt"],
        "first_branch_contact_evidence": ("support" if first["v1_support_unexplained"] else
                                          "non_finger_link" if any("not_a_finger_link_contact" in u["reasons"] for u in first["v1_unexcused"])
                                          else "none"),
        "recover_first_call_substeps": V["executions"][0]["executed_substeps"],
        "recover_first_call_restore_point_logged": all("restored_to_stage" in r for r in V["executions"][0]["guard_refused"]) if v_chain else None,
    })

print("refusal chain (check, control step, refused candidate, approach instance), rerun-vs-recover, mismatching requests:",
      len(path_mismatch), "of", len(pair))
for x in path_mismatch[:5]:
    print("  ", x)

# -------- table -------------------------------------------------------------------------------------------------
def tri(rows, level, comp):
    c = collections.Counter(r[level][comp] for r in rows)
    return f"{c['pass']}/{c['fail']}/{c['und']}"

print("\nformat pass/fail/und, n=32 per selection (first-branch fail counts only evidenced violations)")
hdr = "selection  n | task: L0  L1  L2 | manner: L0  L1  L2 | contact: L0  L1  L2 | joint: L0  L1  L2"
print(hdr)
table = {}
for sel in ("rule", "retrieval"):
    rows = [r for r in per if r["selection"] == sel]
    t = {comp: [tri(rows, lv, comp) for lv in ("L0", "L1", "L2")] for comp in ("task", "manner", "contact", "joint")}
    table[sel] = {"n": len(rows), **t}
    print(f"{sel:9s} {len(rows):2d} | " + " | ".join(f"{comp}: " + "  ".join(f"{v:>7s}" for v in t[comp]) for comp in t))

# first-call-return contamination in the recover arm
print("\nrecover arm, joint pass at the RETURN of the first call vs at the true first branch:")
for sel in ("rule", "retrieval"):
    rows = [r for r in per if r["selection"] == sel]
    a = sum(r["L0"]["joint"] == "pass" for r in rows)
    b = sum(r["L0_first_call_return_recover_arm"]["joint"] == "pass" for r in rows)
    print(f"  {sel:9s} true first branch {a}/{len(rows)}   first-call return (already repaired) {b}/{len(rows)}")

# -------- how many first branches were aborted, and with what evidence ---------------------------------------------
print("\nfirst branch aborted by the guard:")
for sel in ("rule", "retrieval"):
    rows = [r for r in per if r["selection"] == sel]
    ab = [r for r in rows if r["L0_status"] == "guard_refused_abort"]
    ev = collections.Counter(r["first_branch_contact_evidence"] for r in ab)
    chk = collections.Counter(r["refusal_chain_attempt1"][0]["check"] for r in ab)
    print(f"  {sel:9s} aborted {len(ab)}/{len(rows)}; contact evidence in the aborted branch's own log: {dict(ev)}; first refusing check: {dict(chk)}")

# -------- rescued requests ---------------------------------------------------------------------------------------
print("\nrescued = first-branch joint != pass AND final joint == pass")
resc_all = []
for sel in ("rule", "retrieval"):
    rows = [r for r in per if r["selection"] == sel]
    resc = [r for r in rows if r["L0"]["joint"] != "pass" and r["L2"]["joint"] == "pass"]
    cat = collections.Counter()
    mech = collections.Counter()
    for r in resc:
        flags = []
        if r["L0"]["contact"] == "fail" or r["geometry_repairs_attempt1"] > 0:
            flags.append("contact")
        if r["L1"]["manner"] == "fail":
            flags.append("manner(first force cand. missed the band)")
        if r["L0"]["task"] == "fail" or r["L1"]["task"] == "fail":
            flags.append("task")
        cat["+".join(flags) or "none"] += 1
        m = []
        if r["geometry_repairs_total"]:
            m.append(f"geometry-repair x{r['geometry_repairs_total']}")
        if r["force_candidate_switches"]:
            m.append(f"force-candidate-switch x{r['force_candidate_switches']}")
        mech["; ".join(m) or "none"] += 1
        resc_all.append({**r, "rescue_category": "+".join(flags) or "none", "rescue_mechanism": "; ".join(m) or "none"})
    print(f"  {sel:9s} rescued {len(resc)}/{len(rows)}   first-branch joint=pass already: {sum(r['L0']['joint']=='pass' for r in rows)}"
          f"   never passed: {sum(r['L2']['joint']!='pass' for r in rows)}")
    print("     by failing component:", dict(cat))
    print("     by repair mechanism :", dict(mech))

lost = [r for r in per if r["L0"]["joint"] == "pass" and r["L2"]["joint"] != "pass"]
print("first branch passed but final did not (should be 0):", len(lost))

out = Path("reports/step13_offline")
out.mkdir(parents=True, exist_ok=True)
(out / "step2_rescue.json").write_text(json.dumps({"table_pass_fail_und": table, "path_mismatch": len(path_mismatch),
                                                  "rescued": resc_all, "per_request": per}, indent=1, default=float))
