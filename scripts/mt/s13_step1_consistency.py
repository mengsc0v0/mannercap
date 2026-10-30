"""Step 1: rerun-vs-recover consistency of the three component verdicts, a determinism check, and a classification of
every contact row that mentions a tomato_sauce body (the 'object_is_not_a_bound_grasp_target' question)."""
import collections
import json

from s13_components import components, is_aborted, key, load_cells

cells = load_cells()
by = collections.defaultdict(dict)
for c in cells:
    by[(c["selection"],) + key(c)][c["recovery"]] = c

# ---- 1. recorded vs re-derived (final execution) --------------------------------------------------------------
mism = []
for c in cells:
    e = c["executions"][-1]
    d = components(e, c["request"]["band_n"])
    rec_force = bool(e["force_band_passed"])
    if rec_force != (d["task"] == "pass" and d["manner"] == "pass") or bool(e["joint_passed"]) != (d["joint"] == "pass"):
        mism.append((c["selection"], c["recovery"], key(c)))
print("final-execution records whose stored force/joint flag disagrees with the re-derivation:", len(mism), "of", len(cells))

# ---- 2. rerun vs recover: component agreement + numeric agreement ----------------------------------------------
agree = collections.Counter()
dmax = {"hold_min": 0.0, "hold_max": 0.0}
rows = []
for k, pair in sorted(by.items()):
    a, b = pair["rerun"], pair["recover"]
    ea, eb = a["executions"][-1], b["executions"][-1]
    ca, cb = components(ea, a["request"]["band_n"]), components(eb, b["request"]["band_n"])
    for comp in ("task", "manner", "contact", "joint"):
        agree[comp] += ca[comp] == cb[comp]
    for m in ("hold_min", "hold_max"):
        if ea["measured"][m] is not None and eb["measured"][m] is not None:
            dmax[m] = max(dmax[m], abs(ea["measured"][m] - eb["measured"][m]))
    rows.append({"selection": k[0], "task": k[1], "init": k[2], "request": k[3], "rerun": ca, "recover": cb,
                 "same": ca == cb, "force_attempts": [a["joint_passed_at_force_attempt"], b["joint_passed_at_force_attempt"]],
                 "final_candidate": [ea["candidate"], eb["candidate"]]})
n = len(by)
print(f"rerun vs recover, final execution, {n} paired requests:", {c: f"{agree[c]}/{n}" for c in agree})
print("max |rerun-recover| of final hold_min / hold_max (N):", dmax)
print("pairs whose (task,manner,contact) triple differs:", [(r['selection'], r['task'], r['init'], r['request']) for r in rows if not r["same"]])

# ---- 3. determinism: the same configuration run in both arms, when no repair happened -------------------------
same = diff = 0
worst = 0.0
for k, pair in by.items():
    a, b = pair["rerun"], pair["recover"]
    ea, eb = a["executions"][0], b["executions"][0]
    if is_aborted(ea) or eb["guard_refused"] or ea["guard_refused"]:
        continue
    if ea["candidate"] != eb["candidate"]:
        continue
    same += 1
    for m in ("hold_min", "hold_max", "hold_mean"):
        if ea["measured"][m] is not None:
            worst = max(worst, abs(ea["measured"][m] - eb["measured"][m]))
    diff += (ea["n_steps"] != eb["n_steps"]) or (ea["executed_substeps"] != eb["executed_substeps"])
print(f"determinism: {same} first executions with no repair in either arm; n_steps/substeps differ in {diff}; max measured diff {worst:.3g} N")

# ---- 4. tomato_sauce contact rows ---------------------------------------------------------------------------------
cls = collections.Counter()
detail = collections.defaultdict(set)
for c in cells:
    for e in c["executions"]:
        for u in e["v1_unexcused"]:
            if "tomato_sauce" not in u["other"]:
                continue
            if "object_is_not_a_bound_grasp_target" in u["reasons"]:
                cat = "A  non-target distractor (T7), hand body, closing on alphabet_soup"
            elif is_aborted(e):
                cat = "B  target-object palm contact in a guard-aborted (truncated) branch"
            else:
                cat = "C  finger-link on bound target after the permission window (place/retreat)"
            cls[(cat, c["task"], "aborted" if is_aborted(e) else "completed", "cached" if e["reused_cached_episode"] else "run")] += 1
            detail[cat].add((c["task"], c["init"], u["robot_body"], u["max_n"]))
for k, v in sorted(cls.items()):
    print(v, k)
for cat, s in sorted(detail.items()):
    print(cat, "->", sorted(s)[:8], "... n distinct (task,init,body,max_n) =", len(s))
Path = __import__("pathlib").Path
Path("reports/step13_offline/step1_consistency.json").write_text(json.dumps(
    {"final_flag_mismatches": len(mism), "n_pairs": n, "agree": dict(agree), "max_hold_diff_N": dmax,
     "determinism": {"n": same, "differ": int(diff), "max_measured_diff_N": worst}, "rows": rows,
     "tomato_rows": {" | ".join(map(str, k)): v for k, v in cls.items()}}, indent=1, default=float))
