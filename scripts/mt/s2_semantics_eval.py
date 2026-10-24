"""Step 2 acceptance: clause completeness, object swap, repeated stages, ambiguity.

The gold answers below were written by hand against the frozen stage-instance table and are checked
structurally at load time; they are NEVER put into the prompt.  Both the deterministic lexicon and the
model are scored with the same function, so the comparison is like for like.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis_v2.language_compiler import (  # noqa: E402
    MannerBinding, SEMANTICS_PROMPT_VERSION, Semantics, build_semantics_prompt, check_semantics,
    clause_coverage, lexicon_semantics, parse_semantics,
)
from synthesis_v2.llm_interface import LLMClient, LLMConfig  # noqa: E402
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances  # noqa: E402

A, B = "cubeA.pick_1", "cubeB.pick_1"


def two_cube_program() -> TaskProgram:
    return TaskProgram(
        "two_cube_putback", "robosuite_native", "stack",
        {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
        source="composed")


def bind(ids, dim, level, role=None, clause=""):
    return MannerBinding(tuple(ids), dim, level, role, clause, "gold")


# instruction -> verified answer.  Case names say what the case is meant to discriminate.
GOLD: list[tuple[str, str, Semantics]] = [
    ("clauses", "pick up cubeA gently and pick up cubeB firmly",
     Semantics("", (bind([f"{A}.grasp"], "grip", "gentle", "cubeA"),
                    bind([f"{B}.grasp"], "grip", "firm", "cubeB")), source="gold")),
    ("swap", "pick up cubeA firmly and pick up cubeB gently",
     Semantics("", (bind([f"{A}.grasp"], "grip", "firm", "cubeA"),
                    bind([f"{B}.grasp"], "grip", "gentle", "cubeB")), source="gold")),
    ("repeat_stage", "grasp cubeA gently, then grasp cubeB gently",
     Semantics("", (bind([f"{A}.grasp"], "grip", "gentle", "cubeA"),
                    bind([f"{B}.grasp"], "grip", "gentle", "cubeB")), source="gold")),
    ("mixed_dims", "approach cubeA slowly, grasp it gently, then transport cubeB fast",
     Semantics("", (bind([f"{A}.approach"], "speed", "slow", "cubeA"),
                    bind([f"{A}.grasp"], "grip", "gentle", "cubeA"),
                    bind([f"{B}.transport"], "speed", "fast", "cubeB")), source="gold")),
    ("ambiguous", "grasp it gently",
     Semantics("", (), ambiguous=({"clause": "grasp it gently",
                                   "candidates": [f"{A}.grasp", f"{B}.grasp"]},), source="gold")),
    ("unsupported", "open the drawer slowly",
     Semantics("", (), unsupported=({"clause": "open the drawer slowly",
                                     "reason": "no_such_stage_or_object"},), source="gold")),
]


def evaluate(parse_fn, program, name) -> list[dict]:
    out = []
    for case, text, gold in GOLD:
        rec: dict = {"parser": name, "case": case, "instruction": text}
        try:
            sem = parse_fn(text)
        except Exception as exc:
            rec.update(status=f"error:{type(exc).__name__}", detail=str(exc)[:200]); out.append(rec); continue
        errs = check_semantics(sem, program)
        rec["structural_errors"] = errs
        rec.update(clause_coverage(sem, gold))
        rec["status"] = "ok" if not errs else "structural_error"
        rec["bindings"] = [b.to_dict() for b in sem.bindings]
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="call the model as well as the lexicon")
    ap.add_argument("--cache", default="reports/llm_cache/semantics")
    ap.add_argument("--ledger", default="reports/llm_cache/ledger.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    program = two_cube_program()
    assert program.validate() == [], program.validate()
    for case, _text, gold in GOLD:
        errs = check_semantics(gold, program)
        assert errs == [], f"gold {case} is not structurally valid: {errs}"

    recs = evaluate(lambda t: lexicon_semantics(t, program), program, "lexicon")
    spend = None
    if a.live:
        client = LLMClient(LLMConfig.from_env(cache_dir=a.cache, ledger_path=a.ledger, max_calls=24))

        def llm_parse(text: str) -> Semantics:
            system, user = build_semantics_prompt(text, program)
            obj, _rec = client.complete_json(system, user, prompt_version=SEMANTICS_PROMPT_VERSION)
            return parse_semantics(obj, text, program)

        recs += evaluate(llm_parse, program, "llm")
        spend = client.spend()

    for r in recs:
        print(json.dumps({k: r[k] for k in ("parser", "case", "status", "n_gold", "matched",
                                            "missing", "extra", "exact", "ambiguous_got",
                                            "unsupported_got") if k in r}))
    for name in ("lexicon", "llm"):
        sub = [r for r in recs if r["parser"] == name]
        if sub:
            print(f"SUMMARY {name} exact={sum(bool(r.get('exact')) for r in sub)}/{len(sub)} "
                  f"structural_ok={sum(r.get('status') == 'ok' for r in sub)}/{len(sub)}")
    if spend:
        print("SPEND " + json.dumps(spend))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"records": recs, "spend": spend, "prompt_version": SEMANTICS_PROMPT_VERSION},
                  open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
