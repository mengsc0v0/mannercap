"""Step 17 (offline, no simulator): sentence -> LLM intake -> admit (structure + scope gate) -> production plan, compared with the
already validated structured plan.  Writes reports/step17_t5_language/offline_check.json.
usage: [env from ~/.config/mannercap/llm.env] s17_t5_language_check.py [--offline]   (--offline: cache only, no API call)"""
import json, sys
from pathlib import Path
_argv = sys.argv; sys.argv = [_argv[0], "new"]          # s16 reads argv[1] at import; only its plan_for / REQUEST are used here
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import s16_t5_speed_pilot as s16
from s7_task_config import ROOT, make_program
from synthesis_v2.llm_interface import LLMClient, LLMConfig
from synthesis_v2.language_plan import compile_request
from synthesis_v2.manner_request import MannerIntent, parse_manner_request
sys.argv = _argv
OUT = ROOT / "reports" / "step17_t5_language"
SPEC = json.loads((OUT / "scope_spec.json").read_text(encoding="utf-8"))
SPEEDS = ("approach", "descend", "lift", "transport", "place", "retreat")
REF = {"N": (), "S_place": (MannerIntent("place", "speed", "slow"),), "S_transport": (MannerIntent("transport", "speed", "slow"),),
       "S_global": tuple(MannerIntent(p, "speed", "slow") for p in SPEEDS)}


def ref_plan(key, prog):
    s16.REQUEST[key] = REF[key]
    return s16.plan_for(key, prog)


def table(plan):
    return {sp.stage_id.split(".")[-1]: {"v": sp.v_max_mps, "w": sp.w_max_rps, "max_steps": sp.max_steps} for sp in plan.stages}


def main():
    offline = "--offline" in _argv
    prog = make_program(5); base = ref_plan("N", prog)
    client = LLMClient(LLMConfig.from_env(cache_dir=str(OUT / "llm_cache"), ledger_path=str(OUT / "llm_ledger.jsonl"), max_calls=12),
                       offline=offline)
    rows, ok_all = [], True
    for s in SPEC["sentences"]:
        try:
            r = compile_request(s["text"], prog, base, client)
        except Exception as exc:
            rows.append({"id": s["id"], "text": s["text"], "error": f"{type(exc).__name__}: {exc}"}); ok_all = False
            print(s["id"], "ERROR", type(exc).__name__, str(exc)[:200]); continue
        plan = r.pop("plan")
        row = {"id": s["id"], "text": s["text"], "expect_status": s["expect_status"], "status": r["status"],
               "status_ok": r["status"] == s["expect_status"], "needs_clarification": r["needs_clarification"],
               "bindings": [{"instance_ids": b["instance_ids"], "dimension": b["dimension"], "level": b["level"], "clause": b["clause"]}
                            for b in r["intake"]["semantics"]["bindings"]],
               "bases": r["intake"]["bases"], "ambiguous": r["intake"]["semantics"]["ambiguous"], "guards": r["decision"]["guards"],
               "llm": r["llm"], "versions": r["versions"], "trace": r["trace"], "plan_hash": r["plan_hash"]}
        try:
            lex = parse_manner_request("stack", s["text"])
            row["lexicon_baseline_native_family_stack"] = {"status": lex.status, "intents": [[i.stage, i.dimension, i.level] for i in lex.intents]}
        except Exception as exc:
            row["lexicon_baseline_native_family_stack"] = {"error": str(exc)[:120]}
        if s["expect_structured_plan"]:
            ref = ref_plan(s["expect_structured_plan"], prog)
            row["ref_plan_hash"] = ref.plan_hash()
            row["plan_equals_structured"] = bool(plan is not None and plan.to_dict() == ref.to_dict())
            if plan is not None:
                row["plan_table"] = table(plan)
                row["levels_ok"] = {k: v["level"] for k, v in ((t["instance_id"].split(".")[-1], t) for t in r["trace"]) if v["level"] in ("slow", "normal", "fast")} \
                    == {k: v for k, v in s["expect_levels"].items()} if s["expect_levels"] else all(t["level"].startswith("neutral") for t in r["trace"])
            ok_all &= row["status_ok"] and row["plan_equals_structured"]
        else:
            row["plan"] = None if plan is None else "UNEXPECTED_PLAN"
            ok_all &= row["status_ok"] and plan is None
        rows.append(row)
        print(s["id"], row["status"], "status_ok=%s" % row["status_ok"], "plan==structured=%s" % row.get("plan_equals_structured"),
              "guards=%d" % len(row["guards"]), flush=True)
    rec = {"schema": "mannercap.t5_language_offline_check.v1", "all_ok": bool(ok_all), "spend": client.spend(), "model": client.config.identity(),
           "rows": rows}
    (OUT / "offline_check.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("ALL_OK", ok_all, json.dumps(client.spend()))


if __name__ == "__main__":
    main()
