"""Step 6 intake evaluation.  Cases carry hand-written expected decisions; the expected answers are never shown to the model.
Sets: dev (used to develop the prompt/guards) and accept (written separately, run once on a frozen prompt).
Report: accept-correct / disambiguation-correct / unsupported-or-conflict reason-correct / false-accept / false-reject."""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "mt"))
import s6_libero10_formal as F  # noqa: E402
from synthesis_v2.language_compiler import binding_key, MannerBinding  # noqa: E402
from synthesis_v2.request_intake import (INTAKE_PROMPT_VERSION, admit, build_intake_prompt, lexicon_intake, parse_intake, program_steps)  # noqa: E402

OUT = ROOT / "reports" / "step6_libero10_spec_20261025" / "formal"
L = lambda t: F.TASKS[t]["language"]


def case(cid, t, text, expect, *, bind=(), steps=None, roles=(), code=None, why=""):
    return {"id": cid, "task_id": t, "text": text, "expect": expect, "gold_bindings": [list(b) for b in bind],
            "gold_steps": steps, "gold_roles": sorted(roles), "gold_code": code, "why": why}


def dev_cases() -> list[dict]:
    out = []
    for r in F.reqs():
        t = r["task_id"]
        out.append(case(r["request_id"], t, r["request_text"], "accept", bind=r["gold"], steps=[list(s) for s in program_steps(F.program(t))], why="templated request"))
    P = lambda i, t, txt, e, **k: out.append(case(f"P{i}", t, f"{L(t)}; {txt}", e, **k))
    P(0, 4, "hold the mug gently", "ambiguous", roles=["white_mug", "yellow_white_mug"], why="two mugs are manipulated")
    P(1, 6, "hold it firmly", "ambiguous", roles=["white_mug", "chocolate_pudding"], why="'it': two mentioned movable objects")
    P(2, 9, "hold the mug gently", "accept", bind=[("yellow_white_mug.pick_1.grasp", "grip", "gentle", "yellow_white_mug")], why="only one mug is a program role")
    P(3, 3, "close the drawer with gentle hold force", "unsupported", code="dimension_not_on_stage")
    P(4, 2, "turn on the stove gently", "unsupported", code="dimension_not_on_stage")
    P(5, 1, "touch the basket gently", "unsupported", code="quantity_not_measured")
    P(6, 1, "move slowly", "ambiguous", roles=["cream_cheese", "butter"])
    P(7, 0, "hold the ketchup gently", "unsupported", code="object_not_in_task")
    P(8, 5, "hold it with 5 N", "unsupported", code="numeric_level", why="'it' resolves (single object) but the value is not a level")
    P(9, 4, "put the red mug on the left plate", "conflict", code="object_not_in_program")
    out += [
     case("D1", 1, "pick up the butter and hold it gently", "accept", bind=[("butter.pick_1.grasp", "grip", "gentle", "butter")], steps=[], why="unique antecedent although two movables exist in the scene"),
     case("D2", 1, "put both the cream cheese box and the butter in the basket; hold it gently", "ambiguous", roles=["cream_cheese", "butter"]),
     case("D3", 4, "put the white mug on the left plate and put the yellow and white mug on the right plate; hold the white mug gently", "accept", bind=[("white_mug.pick_1.grasp", "grip", "gentle", "white_mug")], steps=[list(s) for s in program_steps(F.program(4))]),
     case("D4", 6, "put the white mug on the plate. Then put the chocolate pudding to the right of the plate. Hold the pudding firmly.", "accept", bind=[("chocolate_pudding.pick_1.grasp", "grip", "firm", "chocolate_pudding")], steps=[list(s) for s in program_steps(F.program(6))]),
     case("D5", 9, "put the yellow and white mug in the microwave and close it slowly", "accept", bind=[("microwave.close_1.articulate", "speed", "slow", "microwave")], steps=[list(s) for s in program_steps(F.program(9))], why="'it' = the microwave (the mug cannot be closed)"),
     case("D6", 3, "close the drawer slowly", "accept", bind=[("drawer.close_1.articulate", "speed", "slow", "drawer")], steps=[["close", "drawer", None]], why="task not restated beyond the close step"),
     case("D7", 1, "hold gently", "ambiguous", roles=["cream_cheese", "butter"], why="no object, two grasps"),
     case("D8", 5, "hold it gently", "accept", bind=[("book.pick_1.grasp", "grip", "gentle", "book")], steps=[], why="one manipulated object: single candidate"),
     case("D9", 1, "put the butter in the basket first and then the cream cheese box in the basket; hold the butter gently", "conflict", code="order_mismatch", why="program order is cream cheese, butter"),
     case("D10", 4, "put the white mug on the right plate; hold it gently", "conflict", code="target_mismatch", why="program puts the white mug on the left plate"),
    ]
    return out


def load_cases(name):
    return dev_cases() if name == "dev" else importlib.import_module("s6_intake_cases_accept").cases(case, F, program_steps)


def roles_of(prog, ids):
    m = {it.instance_id: (it.object_role or it.target_role) for it in prog.instances}
    return {m.get(i, i) for i in ids}


def score(c, intake, dec, prog) -> dict:
    sem = intake.semantics; got_b = {binding_key(b) for b in sem.bindings}
    gold_b = {binding_key(MannerBinding(tuple([b[0]]), b[1], b[2])) for b in c["gold_bindings"]}
    rec = {"status": dec["status"], "codes": dec["codes"], "guards": dec["guards"]}
    e = c["expect"]
    if e == "accept":
        ok = dec["status"] == "accept" and got_b == gold_b
        if ok and c["gold_steps"] is not None:
            got_s = {(s.action, s.object_role, s.target_role) for s in intake.steps if s.action != "pick"}
            gs = {tuple(s) for s in c["gold_steps"] if s[0] != "pick"}
            if intake.source == "llm" and not intake.restated and not gs:
                pass
            elif got_s != gs:
                ok = False; rec["steps_missing"] = sorted(map(list, gs - got_s)); rec["steps_extra"] = sorted(map(list, got_s - gs))
        rec.update(correct=ok, missing=sorted(map(list, gold_b - got_b)), extra=sorted(map(list, got_b - gold_b)))
    elif e == "ambiguous":
        cands = set()
        for a in sem.ambiguous:
            cands |= roles_of(prog, a.get("candidates", ()))
        rec.update(correct=dec["status"] == "ambiguous", cand_roles=sorted(cands, key=str), cand_exact=cands == set(c["gold_roles"]))
    else:  # unsupported | conflict
        rec.update(correct=dec["status"] == e and c["gold_code"] in dec["codes"], status_ok=dec["status"] == e)
    rec["false_accept"] = e != "accept" and dec["status"] == "accept"
    rec["false_reject"] = e == "accept" and dec["status"] != "accept"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="dev"); ap.add_argument("--live", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cases = load_cases(a.set); progs = {t: F.program(t) for t in F.CFG}
    client = None
    if a.live:
        from synthesis_v2.llm_interface import LLMClient, LLMConfig
        client = LLMClient(LLMConfig.from_env(cache_dir=str(ROOT / f"reports/llm_cache/step6_intake_{a.set}{a.tag}"), ledger_path=str(ROOT / "reports/llm_cache/ledger.jsonl"), max_calls=200))
    recs, review = [], []
    for parser in (("lexicon", "llm") if a.live else ("lexicon",)):
        for c in cases:
            prog = progs[c["task_id"]]; raw = None; err = None
            try:
                if parser == "lexicon":
                    it = lexicon_intake(c["text"], prog)
                else:
                    sysm, user = build_intake_prompt(c["text"], prog)
                    raw, _ = client.complete_json(sysm, user, prompt_version=INTAKE_PROMPT_VERSION)
                    it = parse_intake(raw, c["text"])
                dec = admit(it, prog); r = score(c, it, dec, prog)
            except Exception as exc:
                r = {"status": f"error:{type(exc).__name__}", "codes": [], "guards": [], "correct": False, "false_accept": False, "false_reject": c["expect"] == "accept", "detail": str(exc)[:200]}; it = None
            recs.append({"parser": parser, "id": c["id"], "expect": c["expect"], **r})
            review.append({"parser": parser, "case": c, "user_prompt": (build_intake_prompt(c["text"], prog)[1] if parser == "llm" else None), "raw_model_output": raw,
                           "parsed": it.to_dict() if it else None, "decision": r})
    summ = {}
    for parser in {r["parser"] for r in recs}:
        S = lambda f: [r for r in recs if r["parser"] == parser and f(r)]
        acc, amb, uns, con = S(lambda r: r["expect"] == "accept"), S(lambda r: r["expect"] == "ambiguous"), S(lambda r: r["expect"] == "unsupported"), S(lambda r: r["expect"] == "conflict")
        non = amb + uns + con
        summ[parser] = {"1_accept_correct": [sum(r["correct"] for r in acc), len(acc)], "2_ambiguity_correct": [sum(r["correct"] for r in amb), len(amb)],
                        "2b_ambiguity_candidates_exact": [sum(bool(r.get("cand_exact")) for r in amb), len(amb)],
                        "3_unsupported_reason_correct": [sum(r["correct"] for r in uns), len(uns)], "3b_conflict_reason_correct": [sum(r["correct"] for r in con), len(con)],
                        "4_false_accept": [sum(r["false_accept"] for r in non), len(non)], "5_false_reject": [sum(r["false_reject"] for r in acc), len(acc)],
                        "guards_triggered": sum(bool(r["guards"]) for r in S(lambda r: True)), "errors": sum(str(r["status"]).startswith("error") for r in S(lambda r: True))}
    sp = client.spend() if client else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"intake_{a.set}{a.tag}_scores.json").write_text(json.dumps({"prompt_version": INTAKE_PROMPT_VERSION, "summary": summ, "spend": sp, "records": recs}, indent=1, ensure_ascii=False))
    with (OUT / f"intake_{a.set}{a.tag}_review.jsonl").open("w") as fh:
        from synthesis_v2.request_intake import INTAKE_SYSTEM
        fh.write(json.dumps({"system_prompt": INTAKE_SYSTEM, "prompt_version": INTAKE_PROMPT_VERSION}, ensure_ascii=False) + "\n")
        for x in review:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(summ, indent=1)); print("spend", sp)
    bad = [r for r in recs if r["parser"] == ("llm" if a.live else "lexicon") and (not r["correct"] or r["false_accept"])]
    for r in bad:
        print("MISS", r["id"], r["expect"], r["status"], r["codes"], {k: r[k] for k in ("missing", "extra", "steps_missing", "steps_extra", "cand_roles", "guards", "detail") if r.get(k)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
