"""Step 17, scope v2 (no simulator).  Three parts, written to reports/step17_t5_language/:
  (a) scope_v2_boundary.json : spec-v2 boundary sentences through the code rules only (no model, tests the RULES);
  (b) offline_check_v2.json  : the six step-17 sentences judged by the v2 gate.  Their model answers come from the SAVED llm_cache
      (a re-judgement of cached output, NOT a new model test).  The only live call is the single gate-feedback retry that
      compile_request(feedback_retry=True) makes when a cached answer was blocked by language_scope_mismatch (S1b).
usage: [env from ~/.config/mannercap/llm.env] s17_t5_language_v2_check.py [--offline]   (--offline: cache only, no API call)"""
import json, sys
from pathlib import Path
_argv = sys.argv; sys.argv = [_argv[0], "new"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import s16_t5_speed_pilot as s16
from s7_task_config import ROOT, make_program
from synthesis_v2.llm_interface import LLMClient, LLMConfig
from synthesis_v2.language_plan import compile_request
from synthesis_v2.language_scope import SCOPE_VERSION, resolve_speed_scope, scope_rules_hash
from synthesis_v2.manner_request import MannerIntent
sys.argv = _argv
OUT = ROOT / "reports" / "step17_t5_language"
SPEC = json.loads((OUT / "scope_spec_v2.json").read_text(encoding="utf-8"))
SPEEDS = ("approach", "descend", "lift", "transport", "place", "retreat")
REF = {"N": (), "S_place": (MannerIntent("place", "speed", "slow"),), "S_transport": (MannerIntent("transport", "speed", "slow"),),
       "S_global": tuple(MannerIntent(p, "speed", "slow") for p in SPEEDS)}


def ref_plan(key, prog):
    s16.REQUEST[key] = REF[key]
    return s16.plan_for(key, prog)


def short(eff):
    return {k.split(".")[-1]: v for k, v in eff.items()}


def boundary(prog):
    rows = []
    for c in SPEC["boundary_sentences"]:
        r = resolve_speed_scope(c["text"], prog)
        ok = r["status"] == c["expect_status"]
        if "expect_levels" in c:
            ok = ok and short(r["effective"]) == c["expect_levels"]
        if "expect_reason" in c:
            ok = ok and c["expect_reason"] in {x["reason"] for x in r["reasons"]}
        if "expect_exceptions" in c:
            ok = ok and [e.split(".")[-1] for e in r["exceptions"]] == c["expect_exceptions"]
        rows.append({"id": c["id"], "kind": c["kind"], "text": c["text"], "expect_status": c["expect_status"], "status": r["status"],
                     "ok": ok, "rules": [[h["word"], h["rule"], h.get("stage"), h.get("reason")] for h in r["hits"]],
                     "effective": short(r["effective"]), "exceptions": [e.split(".")[-1] for e in r["exceptions"]],
                     "reasons": r["reasons"], "confirmed_by": c["confirmed_by"]})
    return rows


def main():
    offline = "--offline" in _argv
    prog = make_program(5); base = ref_plan("N", prog)
    brows = boundary(prog)
    json.dump({"what": "code-rule check of the spec-v2 boundary sentences; no model call, no simulation", "scope_version": SCOPE_VERSION,
               "scope_rules_hash": scope_rules_hash(), "all_ok": all(r["ok"] for r in brows), "rows": brows},
              open(OUT / "scope_v2_boundary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in brows:
        print(r["id"], "OK " if r["ok"] else "BAD", r["status"], r["effective"] or r["reasons"][:1] and [x.get("reason", x.get("scope")) for x in r["reasons"]])
    client = LLMClient(LLMConfig.from_env(cache_dir=str(OUT / "llm_cache"), ledger_path=str(OUT / "llm_ledger.jsonl"), max_calls=2), offline=offline)
    rows = []
    for s in SPEC["sentences"]:
        try:
            r = compile_request(s["text"], prog, base, client, feedback_retry=True)
        except Exception as exc:
            rows.append({"id": s["id"], "text": s["text"], "error": f"{type(exc).__name__}: {exc}"})
            print(s["id"], "ERROR", type(exc).__name__, str(exc)[:200]); continue
        plan = r.pop("plan")
        row = {"id": s["id"], "text": s["text"], "expect_status": s["expect_status"], "first_attempt_status": r["attempts"][0]["status"],
               "final_status": r["status"], "status_ok": r["status"] == s["expect_status"], "needed_feedback_retry": r["needed_feedback_retry"],
               "needs_clarification": r["needs_clarification"], "attempts": r["attempts"], "guards": r["decision"]["guards"],
               "speed_scope": r["decision"].get("speed_scope"), "versions": r["versions"], "plan_hash": r["plan_hash"],
               "case_nature": s.get("case_nature")}
        if s["expect_structured_plan"]:
            ref = ref_plan(s["expect_structured_plan"], prog)
            row["ref_plan_hash"] = ref.plan_hash()
            row["plan_equals_structured"] = bool(plan is not None and plan.to_dict() == ref.to_dict())
        else:
            row["plan_is_none_as_expected"] = plan is None
        rows.append(row)
        print(s["id"], "first:", row["first_attempt_status"], "final:", row["final_status"], "retry:", row["needed_feedback_retry"],
              "plan==structured:", row.get("plan_equals_structured", row.get("plan_is_none_as_expected")))
    json.dump({"what": "RE-JUDGEMENT of the saved step-17 model answers with the scope-v2 gate (cache hits); not a new model test. "
                       "Only the S1b gate-feedback retry is a live call; its feedback names the instances that disagree.",
               "scope_version": SCOPE_VERSION, "scope_rules_hash": scope_rules_hash(), "offline_flag": offline,
               "llm_records": [{"status": r.status, "request_hash": r.request_hash, "usage": r.usage} for r in client.records], "rows": rows},
              open(OUT / "offline_check_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
