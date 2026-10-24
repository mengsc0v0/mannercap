"""Step 6 (formal): LIBERO-10 requests -> requirements -> stage instances -> admission status, and LLM parse vs
independently written intent.

What this file asserts and what it does not:
* PROGRAMS are typed TaskPrograms built from the shared pick_place_instances template (+ one ``articulate`` stage for
  the three articulation tasks).  Object roles are the sentence's own noun phrases; PHRASES records which BDDL object each
  role is and where that mapping comes from.
* Admission status of a request is derived, not asserted: a request is 'needs_new_primitive' iff its program contains a
  primitive that the executor sources do not implement; 'executable_verified' iff every pick-place stage it needs has
  executed + been validated (VERIFIED_SUBTASKS) AND the manner dimension has a calibrated band; everything else that the
  typed program can express is 'expressible_unverified'.  Ambiguous/unsupported manner clauses are scored on the
  probes, where the expected answer is "returned explicitly, never executed".
* GOLD is written by hand as literal instance ids and is never shown to the model.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synthesis_v2.language_compiler import (MannerBinding, SEMANTICS_PROMPT_VERSION, Semantics, build_semantics_prompt,  # noqa: E402
                                            check_program, check_semantics, clause_coverage, lexicon_semantics, parse_semantics)
from synthesis_v2.task_program import PRIMITIVES, StageInstance, TaskProgram, chain, pick_place_instances  # noqa: E402

OUT = ROOT / "reports" / "step6_libero10_spec_20261025" / "formal"
TASKS = {t["id"]: t for t in json.loads((ROOT / "reports/step6_libero10_spec_20261025/libero10_tasks.json").read_text())}

# role -> (BDDL object / region, how the phrase->object mapping is established)
# (role, backend name, phrase_source)
P_NAME = "object class name in BDDL (not rendered)"
P_INIT = "BDDL init region name"
P_RENDER = "BDDL class name + rendered scene check (formal/scene_t4.png)"
# each subtask: (object role, backend object, target role, backend target)
CFG = {
 0: dict(pp=[("alphabet_soup", "alphabet_soup_1", "basket", "basket_1_contain_region"), ("tomato_sauce", "tomato_sauce_1", "basket", "basket_1_contain_region")], art=None,
         phr={"alphabet_soup": P_NAME, "tomato_sauce": P_NAME, "basket": P_NAME}),
 1: dict(pp=[("cream_cheese", "cream_cheese_1", "basket", "basket_1_contain_region"), ("butter", "butter_1", "basket", "basket_1_contain_region")], art=None,
         phr={"cream_cheese": P_NAME, "butter": P_NAME, "basket": P_NAME}),
 7: dict(pp=[("alphabet_soup", "alphabet_soup_1", "basket", "basket_1_contain_region"), ("cream_cheese", "cream_cheese_1", "basket", "basket_1_contain_region")], art=None,
         phr={"alphabet_soup": P_NAME, "cream_cheese": P_NAME, "basket": P_NAME}),
 4: dict(pp=[("white_mug", "porcelain_mug_1", "left_plate", "plate_1"), ("yellow_white_mug", "white_yellow_mug_1", "right_plate", "plate_2")], art=None,
         phr={"white_mug": P_RENDER, "yellow_white_mug": P_RENDER, "left_plate": P_INIT + " (plate_1 in plate_left_region)", "right_plate": P_INIT + " (plate_2 in plate_right_region)"}),
 6: dict(pp=[("white_mug", "porcelain_mug_1", "plate", "plate_1"), ("chocolate_pudding", "chocolate_pudding_1", "plate_right_region", "living_room_table_plate_right_region")], art=None,
         phr={"white_mug": P_RENDER, "chocolate_pudding": P_NAME, "plate": P_NAME, "plate_right_region": "BDDL goal region name"}),
 8: dict(pp=[("right_moka_pot", "moka_pot_1", "stove", "flat_stove_1_cook_region"), ("left_moka_pot", "moka_pot_2", "stove", "flat_stove_1_cook_region")], art=None,
         phr={"right_moka_pot": P_INIT + " (moka_pot_1 in moka_pot_right_init_region)", "left_moka_pot": P_INIT + " (moka_pot_2 in moka_pot_left_init_region)", "stove": P_NAME}),
 5: dict(pp=[("book", "black_book_1", "caddy_back", "desk_caddy_1_back_contain_region")], art=None, phr={"book": P_NAME, "caddy_back": "BDDL goal region name"}),
 3: dict(pp=[("black_bowl", "akita_black_bowl_1", "drawer", "white_cabinet_1_bottom_region")], art=("drawer", "drawer", "close"), phr={"black_bowl": P_NAME, "drawer": "BDDL goal region name"}),
 9: dict(pp=[("yellow_white_mug", "white_yellow_mug_1", "microwave", "microwave_1_heating_region")], art=("microwave", "microwave", "close"), phr={"yellow_white_mug": P_RENDER, "microwave": P_NAME}),
 2: dict(pp=[("moka_pot", "moka_pot_1", "stove", "flat_stove_1_cook_region")], art=("stove", "stove", "turn_on"), art_first=True, phr={"moka_pot": P_NAME, "stove": P_NAME}),
}
PHRASE = {"white_mug": "white mug", "yellow_white_mug": "yellow and white mug", "alphabet_soup": "alphabet soup", "tomato_sauce": "tomato sauce",
          "cream_cheese": "cream cheese box", "butter": "butter", "right_moka_pot": "right moka pot", "left_moka_pot": "left moka pot",
          "moka_pot": "moka pot", "book": "book", "black_bowl": "black bowl", "chocolate_pudding": "chocolate pudding"}
ART_PHRASE = {3: "drawer", 9: "microwave door", 2: "stove knob"}
ART_CLAUSE = {3: "close the drawer {lv}", 9: "close the microwave door {lv}", 2: "turn on the stove {lv}"}
LV_ADV = {"slow": "slowly", "fast": "quickly"}

# executed + validated pick-place subtasks (Step 3-5, LIBERO-10 task 1, dev inits)
VERIFIED_SUBTASKS = {(1, "cream_cheese"), (1, "butter")}
CALIBRATED = {(1, "grip")}      # (task, dimension) with a frozen band (configs/manner_calibration.v1/v2.json)


def program(t: int) -> TaskProgram:
    c = CFG[t]; roles = {}
    groups = []
    for o, ob, tg, tb in c["pp"]:
        roles[o] = ob; roles[tg] = tb
        groups.append(pick_place_instances(o, tg))
    inst = list(chain(*groups))
    if c["art"]:
        r, bk, verb = c["art"]; roles[r] = bk
        sub = f"{r}.{verb}_1"
        a = StageInstance(f"{sub}.articulate", "articulate", f"{sub}.articulate", None, r, sub, 1)
        if c.get("art_first"):
            inst = [a] + list(chain(*groups))
            inst[1] = StageInstance(**{**inst[1].__dict__, "preds": (a.instance_id,)})
        else:
            inst.append(StageInstance(**{**a.__dict__, "preds": (inst[-1].instance_id,)}))
    return TaskProgram(f"libero10_t{t}", "libero", f"libero:libero_10:{t}", roles, tuple(inst), source="composed", language=TASKS[t]["language"])


def implemented_primitives() -> set[str]:
    src = "".join((ROOT / "synthesis_v2" / f).read_text() for f in ("program_executor.py", "native_producer.py"))
    return {p for p in PRIMITIVES if re.search(r'["\']%s["\']' % p, src)}


# evaluation windows: event definitions (stage identification downstream is by state/contact events, not demo labels)
WINDOWS = {
 ("grip", "hold"): "both finger-object normal forces >= 0.1 N on the grasped object (first 2 steps trimmed) -> first step either finger loses contact after the release command",
 ("speed", "approach"): "motion onset toward the object (end of previous stage release) -> first finger-object contact",
 ("speed", "place"): "object lifted clear of its support (support-contact loss) -> release (finger-object contact loss) after object-target contact",
 ("speed", "articulate"): "first gripper/hand contact with the articulated part -> articulated predicate satisfied or part at rest",
}

ALLOWED_CONTACTS = {  # registered contact relations per primitive; everything else touching the hand/fingers/object is a recorded non-target contact
 "approach": [], "descend": [("finger", "support_surface", "record_only")],
 "grasp": [("finger", "target_object", "allowed"), ("finger", "support_surface", "record_only"), ("target_object", "support_surface", "allowed")],
 "lift": [("finger", "target_object", "allowed"), ("target_object", "support_surface", "allowed_until_clear")],
 "transport": [("finger", "target_object", "allowed")],
 "place": [("finger", "target_object", "allowed"), ("target_object", "place_target", "allowed")],
 "retreat": [],
 "articulate": [("finger_or_hand", "articulated_part", "allowed")],
 "note": "everything not listed (hand/finger/arm vs other object, vs fixture rim/wall, target container by the fingers) is recorded with stage, pair, max force and classified unintended unless the task registers it; no severity score yet",
}


def reqs() -> list[dict]:
    out = []
    for t in sorted(CFG):
        c = CFG[t]; lang = TASKS[t]["language"]; pp = c["pp"]; names = [o for o, *_ in pp]; ph = [PHRASE[n] for n in names]
        def add(dim, texts, bind, kind):
            out.append({"request_id": f"L10-T{t}-{len(out):03d}", "task_id": t, "manner_axis": dim, "request_text": f"{lang}; {texts}", "kind": kind, "gold": bind})
        for k in ("gentle", "firm"):
            if len(pp) == 2:
                a, b = names; ka, kb = (("gentle", "firm") if k == "gentle" else ("firm", "gentle"))
                add("grip", f"hold the {ph[0]} {ka[:-1] if False else {'gentle':'gently','firm':'firmly'}[ka]} and the {ph[1]} {({'gentle':'gently','firm':'firmly'})[kb]}",
                    [(f"{a}.pick_1.grasp", "grip", ka, a), (f"{b}.pick_1.grasp", "grip", kb, b)], "following")
            else:
                add("grip", f"hold the {ph[0]} {({'gentle':'gently','firm':'firmly'})[k]}", [(f"{names[0]}.pick_1.grasp", "grip", k, names[0])], "following")
        for a1, a2 in (("fast", "slow"), ("slow", "fast")):
            n = names[0]
            add("speed", f"approach the {ph[0]} {LV_ADV[a1]} and place the {ph[0]} {LV_ADV[a2]}",
                [(f"{n}.pick_1.approach", "speed", a1, n), (f"{n}.pick_1.place", "speed", a2, n)], "following")
        if c["art"]:
            r, _, verb = c["art"]
            for lv in ("slow", "fast"):
                add("speed", ART_CLAUSE[t].format(lv=LV_ADV[lv]), [(f"{r}.{verb}_1.articulate", "speed", lv, r)], "following")
    return out


# probes: (task, text, expected, reason class); gold bindings only where the clause is resolvable
def probes() -> list[dict]:
    L = lambda t: TASKS[t]["language"]
    P = lambda t, txt, exp, why, bind=(): {"task_id": t, "text": f"{L(t)}; {txt}", "expected": exp, "why": why, "gold": list(bind)}
    return [
     P(4, "hold the mug gently", "ambiguous", "two mugs are manipulated; 'the mug' names none of them"),
     P(6, "hold it firmly", "ambiguous", "'it' has two candidate objects"),
     P(9, "hold the mug gently", "resolved", "only one mug is manipulated; distractor mug is not a program role",
       [("yellow_white_mug.pick_1.grasp", "grip", "gentle", "yellow_white_mug")]),
     P(3, "close the drawer with gentle hold force", "unsupported", "hold force is not defined for the articulated stage"),
     P(2, "turn on the stove gently", "unsupported", "no grip/press dimension on the articulated stage"),
     P(1, "touch the basket gently", "unsupported", "gentle contact (transient) is not measured; only hold-force levels are"),
     P(1, "move slowly", "ambiguous", "no stage/object given; several transport instances match"),
     P(0, "hold the ketchup gently", "unsupported", "object is not part of this task's program"),
     P(5, "hold it with 5 N", "unsupported", "numeric force is not a calibrated level"),
     P(4, "put the red mug on the left plate", "unsupported", "red mug is not part of the task goal"),
    ]


def sem_of(bind, text) -> Semantics:
    return Semantics(text, tuple(MannerBinding((i,), d, l, r, "", "gold") for i, d, l, r in bind), source="gold")


def status(t: int, r: dict, impl: set[str]) -> dict:
    prog = program(t)
    missing = sorted({it.primitive for it in prog.instances} - impl)
    dim = r["gold"][0][1] if r["gold"] else None
    roles = {b[3] for b in r["gold"]}
    verified_stage = all((t, ro) in VERIFIED_SUBTASKS for ro in roles if ro in [o for o, *_ in CFG[t]["pp"]])
    art_clause = any(".articulate" in b[0] for b in r["gold"])
    if missing:
        cls = "needs_new_primitive"
        why = f"program needs {missing}; no executor implementation" + ("; the manner clause itself is on the missing stage" if art_clause else "; the manner clause is on an implemented pick-place stage but the task cannot complete without the missing stage")
    elif all((t, o) in VERIFIED_SUBTASKS for o, *_ in CFG[t]["pp"]) and (t, dim) in CALIBRATED:
        cls, why = "executable_verified", "all stages executed+validated on dev inits of this task; band frozen (v1/v2); not all dev inits pass (see Step 5 close-out)"
    else:
        why = []
        if not all((t, o) in VERIFIED_SUBTASKS for o, *_ in CFG[t]["pp"]):
            why.append("pick-place on this object/target not yet run in this scene")
        if (t, dim) not in CALIBRATED:
            why.append(f"no calibrated response/band for {dim} on this task")
        cls, why = "expressible_unverified", "; ".join(why)
    return {"class": cls, "reason": why, "missing_primitives": missing}


def window_for(r: dict) -> list[dict]:
    out = []
    for iid, dim, lv, role in r["gold"]:
        stage = iid.split(".")[-1]
        key = (dim, "hold") if dim == "grip" else (dim, stage if stage in ("approach", "place", "articulate") else "place")
        out.append({"instance": iid, "dimension": dim, "level": lv, "event_window": WINDOWS[key]})
    return out


def llm_parse(client, text, prog):
    system, user = build_semantics_prompt(text, prog)
    obj, rec = client.complete_json(system, user, prompt_version=SEMANTICS_PROMPT_VERSION)
    return parse_semantics(obj, text, prog)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    impl = implemented_primitives()
    progs = {t: program(t) for t in CFG}
    prog_errs = {t: check_program(p, allowed_backends=("libero",)) for t, p in progs.items()}
    R = reqs(); PR = probes()
    for r in R:
        sem = sem_of(r["gold"], r["request_text"]); e = check_semantics(sem, progs[r["task_id"]])
        assert not e, (r["request_id"], e)
    for p in PR:
        e = check_semantics(sem_of(p["gold"], p["text"]), progs[p["task_id"]]); assert not e, (p["text"], e)
    table = []
    for r in R:
        s = status(r["task_id"], r, impl)
        table.append({"request_id": r["request_id"], "task_id": r["task_id"], "request_text": r["request_text"], "manner_axis": r["manner_axis"],
                      "requirements": [{"clause_object": b[3], "stage_instance": b[0], "dimension": b[1], "level": b[2]} for b in r["gold"]],
                      "base_task_requirement": TASKS[r["task_id"]]["goal"], "windows": window_for(r),
                      "target_band": ("configs/manner_calibration.v1.json / v2.json (per object)" if (r["task_id"], r["manner_axis"]) in CALIBRATED else "none yet (calibration required before the request can be scored)"),
                      **s,
                      "when_not_executable": ("return {status:'unsupported', code:'primitive_missing:%s'}" % ",".join(s["missing_primitives"])) if s["missing_primitives"] else
                                             ("return {status:'unverified', code:'no_calibrated_band'} if scoring is requested" if s["class"] != "executable_verified" else "n/a")})
    classes = {}
    for x in table:
        classes[x["class"]] = classes.get(x["class"], 0) + 1
    matrix = {}
    for t in sorted(CFG):
        prog = progs[t]
        matrix[str(t)] = {"language": TASKS[t]["language"], "phrase_registry": {k: {"bddl_object": prog.objects[k], "source": v} for k, v in CFG[t]["phr"].items()},
                          "primitives": sorted({it.primitive for it in prog.instances}), "missing_primitives": sorted({it.primitive for it in prog.instances} - impl),
                          "stage_instances": [it.instance_id for it in prog.instances],
                          "manner_dims_per_primitive": {p: list(PRIMITIVES[p]["dims"]) for p in sorted({it.primitive for it in prog.instances})},
                          "pick_place_verified_objects": [o for o, *_ in CFG[t]["pp"] if (t, o) in VERIFIED_SUBTASKS],
                          "structural_errors": prog_errs[t]}
    res = {"implemented_primitives": sorted(impl), "n_requests": len(table), "class_counts": classes, "allowed_contacts": ALLOWED_CONTACTS}
    (OUT / "requirements_table.json").write_text(json.dumps({"summary": res, "requests": table}, indent=1, ensure_ascii=False))
    (OUT / "task_primitive_manner_matrix.json").write_text(json.dumps(matrix, indent=1, ensure_ascii=False))
    print("implemented", sorted(impl), "classes", classes, "program structural errors", {t: e for t, e in prog_errs.items() if e})

    cases = [("request", r["request_id"], r["task_id"], r["request_text"], sem_of(r["gold"], r["request_text"]), True) for r in R]
    cases += [("probe", f"P{i}", p["task_id"], p["text"], sem_of(p["gold"], p["text"]), p["expected"] == "resolved") for i, p in enumerate(PR)]
    score = []
    client = None
    if a.live:
        from synthesis_v2.llm_interface import LLMClient, LLMConfig
        client = LLMClient(LLMConfig.from_env(cache_dir=str(ROOT / "reports/llm_cache/step6_semantics"), ledger_path=str(ROOT / "reports/llm_cache/ledger.jsonl"), max_calls=64))
    for parser in (("lexicon", "llm") if a.live else ("lexicon",)):
        for kind, cid, t, text, gold, complete_expected in cases:
            rec = {"parser": parser, "kind": kind, "id": cid, "task_id": t, "text": text}
            try:
                got = lexicon_semantics(text, progs[t]) if parser == "lexicon" else llm_parse(client, text, progs[t])
            except Exception as exc:
                rec.update(status=f"error:{type(exc).__name__}", detail=str(exc)[:200]); score.append(rec); continue
            errs = check_semantics(got, progs[t])
            expected_kind = None
            if kind == "probe":
                expected_kind = PR[int(cid[1:])]["expected"]
            rec.update(structural_errors=errs, bindings=[b.to_dict() for b in got.bindings], ambiguous=list(got.ambiguous), unsupported=list(got.unsupported),
                       **clause_coverage(got, gold))
            if kind == "probe":
                rec["returned_explicitly"] = bool(got.ambiguous or got.unsupported) if expected_kind != "resolved" else got.complete()
                rec["expected"] = expected_kind
                rec["silent_execution"] = expected_kind != "resolved" and not got.ambiguous and not got.unsupported and bool(got.bindings)
            score.append(rec)
    summ = {}
    for parser in {s["parser"] for s in score}:
        for kind in ("request", "probe"):
            sub = [s for s in score if s["parser"] == parser and s["kind"] == kind]
            d = {"n": len(sub), "exact": sum(bool(s.get("exact")) for s in sub), "structural_ok": sum(s.get("status") is None and not s.get("structural_errors") for s in sub),
                 "errors": sum(str(s.get("status", "")).startswith("error") for s in sub)}
            if kind == "probe":
                d["returned_as_expected"] = sum(bool(s.get("returned_explicitly")) for s in sub)
                d["silent_execution"] = sum(bool(s.get("silent_execution")) for s in sub)
            if kind == "request":
                d["wrong_scope"] = sum(int(s.get("wrong_scope", 0)) for s in sub)
                d["missing_clauses"] = sum(len(s.get("missing", [])) for s in sub)
                d["extra_bindings"] = sum(len(s.get("extra", [])) for s in sub)
            summ[f"{parser}/{kind}"] = d
    spend = client.spend() if client else None
    (OUT / "parse_scores.json").write_text(json.dumps({"summary": summ, "spend": spend, "records": score}, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1)); print("spend", spend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
