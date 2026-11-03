"""Step 31: generalisation of the manner intake to Chinese and paraphrased English instructions.

Test set = 17 items x 5 LIBERO tasks (T0/T7/T1/T6/T8), written BEFORE any model call; gold = the request the
step-30 pipeline should produce (grip: role -> level; speed: set of (role, stage primitive, level)) or the expected
non-accept outcome.  Every item goes through exactly the step-30 path (s30_manner_gen.intake_one: LLM intake ->
admit -> request_from_semantics), nothing is tuned on these sentences.

  build  -> testset.json (frozen, sha256 printed)
  run    -> results.json (one LLM answer per item, +1 gate-feedback retry as in step 30)
  score  -> score.json
  run_v2 / score_v2 -> results_v2.json / score_v2.json: same frozen test set, pipeline after the step-31 fixes
  run_v3 / score_v3 -> v2 + object-layer v2 (post-hoc fix of the one remaining v2 failure), full re-run
  addendum -> testset_addendum_t6.json: the 8 T6 Chinese accept-items rebuilt on a base WITHOUT the order word 再
              (test-construction error: 再 states an order mug->pudding that the English base does not state and that
              contradicts the registered T6 program, so 'conflict' was the correct answer there)
"""
import hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault("S30_OUT", "reports/step31_paraphrase")
import s30_manner_gen as G

OUT = Path(os.environ.get("S31_PARA_OUT", "reports/step31_paraphrase"))
ZH = {
    0: dict(base="把字母汤和番茄酱都放进篮子里", X="番茄酱", Y="字母汤"),
    7: dict(base="把字母汤和奶油奶酪盒都放进篮子里", X="奶油奶酪盒", Y="字母汤"),
    1: dict(base="把奶油奶酪盒和黄油都放进篮子里", X="奶油奶酪盒", Y="黄油"),
    6: dict(base="把白色马克杯放到盘子上，再把巧克力布丁放到盘子右边", X="白色马克杯", Y="巧克力布丁"),
    8: dict(base="把两个摩卡壶都放到炉子上", X="右边的摩卡壶", Y="左边的摩卡壶"),
}
ALL = "*"   # every speed-capable stage of every object
# (id, language, template, gold).  gold: {"status": "accept", "grip": {X|Y: lvl}, "speed": [(X|Y|*, stage|*, lvl)]}
#                                     or {"status": "not_accept", "why": ...}
ITEMS = [
    ("en_gentle_para", "en", "{Base}, taking care not to squeeze the {X} hard", {"grip": {"X": "gentle"}}),
    ("en_firm_para", "en", "{Base}, and keep a tight hold on the {X}", {"grip": {"X": "firm"}}),
    ("en_slow_para", "en", "{Base}; bring the {X} over at a leisurely pace", {"speed": [("X", "transport", "slow")]}),
    ("en_fast_para", "en", "{Base}, and hurry while carrying the {X}", {"speed": [("X", "transport", "fast")]}),
    ("en_binding_reorder", "en", "Grip the {Y} firmly but handle the {X} with a light touch; {base}",
     {"grip": {"X": "gentle", "Y": "firm"}}),
    ("en_fragile_combo", "en", "{Base}. The {X} is fragile, so hold it softly and carry it slowly",
     {"grip": {"X": "gentle"}, "speed": [("X", "transport", "slow")]}),
    ("zh_neutral", "zh", "{zbase}", {}),
    ("zh_gentle", "zh", "{zbase}，抓{zX}的时候轻一点", {"grip": {"X": "gentle"}}),
    ("zh_firm", "zh", "{zbase}，{zX}要握紧", {"grip": {"X": "firm"}}),
    ("zh_slowcarry", "zh", "{zbase}，慢慢地搬运{zX}", {"speed": [("X", "transport", "slow")]}),
    ("zh_fastcarry", "zh", "{zbase}，快速搬运{zX}", {"speed": [("X", "transport", "fast")]}),
    ("zh_binding", "zh", "{zbase}，{zX}轻轻地拿，{zY}要抓牢", {"grip": {"X": "gentle", "Y": "firm"}}),
    ("zh_slow_all", "zh", "{zbase}，全程慢一些", {"speed": [(ALL, ALL, "slow")]}),
    ("zh_fragile_combo", "zh", "{zbase}，{zX}容易碎，要轻拿并且慢慢搬运",
     {"grip": {"X": "gentle"}, "speed": [("X", "transport", "slow")]}),
    ("zh_scope_undetermined", "zh", "{zbase}，慢一点", {"status": "not_accept", "why": "speed scope undetermined (R4)"}),
    ("en_numeric", "en", "{Base}, gripping the {X} with exactly 12 newtons", {"status": "not_accept", "why": "numeric level"}),
    ("zh_conflict", "zh", "{zbase}，{zX}要轻拿，但{zX}也要握紧", {"status": "not_accept", "why": "conflicting grip levels"}),
]


def build():
    rows = []
    for t, c in G.TASKS.items():
        z = ZH[t]
        for iid, lang, tpl, gold in ITEMS:
            text = tpl.format(Base=c["lang"].capitalize(), base=c["lang"], X=c["X"][0], Y=c["Y"][0],
                              zbase=z["base"], zX=z["X"], zY=z["Y"])
            g = {"status": gold.get("status", "accept")}
            if g["status"] == "accept":
                role = {"X": c["X"][1], "Y": c["Y"][1]}
                g["grip"] = {role[k]: v for k, v in gold.get("grip", {}).items()}
                g["speed"] = [[role.get(r, r), s, l] for r, s, l in gold.get("speed", [])]
            else:
                g["why"] = gold["why"]
            rows.append({"key": f"t{t}:{iid}", "task": t, "id": iid, "lang": lang, "instruction": text, "gold": g})
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.dumps(rows, indent=1, ensure_ascii=False)
    (OUT / "testset.json").write_text(s)
    print(len(rows), "items sha256", hashlib.sha256(s.encode()).hexdigest())


def run(testset="testset.json", res_name="results.json"):
    rows = json.loads((OUT / testset).read_text())
    f = OUT / res_name
    res = json.loads(f.read_text()) if f.exists() else {}
    cl = G.client()
    progs = {}
    for r in rows:
        if r["key"] in res:
            continue
        t = r["task"]
        if t not in progs:
            progs[t] = G.program_of(t)[0]
        try:
            res[r["key"]] = G.intake_one(r["instruction"], progs[t], cl)
        except Exception as e:      # parse errors are a result, not a crash
            res[r["key"]] = {"instruction": r["instruction"], "status": f"error:{type(e).__name__}", "error": str(e)[:300]}
        q = res[r["key"]].get("request") or {}
        print(r["key"], res[r["key"]]["status"], q.get("grip"), sorted(set((k.split(".")[0], k.split(".")[-1], v) for k, v in (q.get("speed") or {}).items()))[:6], flush=True)
        f.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def _speed_set(prog, speed_map):
    by = {it.instance_id: it for it in prog.instances}
    return {(by[i].object_role, by[i].primitive, l) for i, l in speed_map.items() if l != "normal"}


def _gold_speed(prog, g):
    out = set()
    for r, s, l in g:
        for it in prog.instances:
            if "speed" in G_dims(it) and (r == ALL or it.object_role == r) and (s == ALL or it.primitive == s):
                out.add((it.object_role, it.primitive, l))
    return out


def G_dims(it):
    from synthesis_v2.language_compiler import PRIMITIVES
    return PRIMITIVES[it.primitive]["dims"]


ZH6_FIXED = "把白色马克杯放到盘子上，把巧克力布丁放到盘子右边"


def addendum():
    rows = [r for r in json.loads((OUT / "testset.json").read_text()) if r["task"] == 6 and r["lang"] == "zh"
            and r["gold"]["status"] == "accept"]
    for r in rows:
        assert r["instruction"].startswith(ZH[6]["base"])
        r["instruction"] = ZH6_FIXED + r["instruction"][len(ZH[6]["base"]):]
        r["key"] += ":fixed_base"
    s = json.dumps(rows, indent=1, ensure_ascii=False)
    (OUT / "testset_addendum_t6.json").write_text(s)
    print(len(rows), "items sha256", hashlib.sha256(s.encode()).hexdigest())


def score(res_name="results.json", out_name="score.json", testset="testset.json"):
    rows = json.loads((OUT / testset).read_text())
    res = json.loads((OUT / res_name).read_text())
    progs, items, agg = {}, [], {}
    for r in rows:
        t, g, a = r["task"], r["gold"], res.get(r["key"])
        if a is None:
            continue
        if t not in progs:
            progs[t] = G.program_of(t)[0]
        prog = progs[t]
        ok_status = (a["status"] == "accept") == (g["status"] == "accept")
        detail = {}
        if g["status"] == "accept" and a["status"] == "accept":
            q = a["request"]
            gs, ps = _gold_speed(prog, g["speed"]), _speed_set(prog, q["speed"])
            grip_ok, speed_ok = q["grip"] == g["grip"], gs == ps
            detail = {"grip_ok": grip_ok, "speed_ok": speed_ok, "grip": q["grip"], "gold_grip": g["grip"],
                      "speed_missing": sorted(gs - ps), "speed_extra": sorted(ps - gs)}
            ok = grip_ok and speed_ok
        else:
            ok = ok_status
        retry = len(a.get("attempts", [])) > 1
        codes = (a.get("decision") or {}).get("codes")
        items.append({"key": r["key"], "lang": r["lang"], "id": r["id"], "ok": ok, "status": a["status"],
                      "gold_status": g["status"], "codes": codes, "retry": retry, **detail})
        for k in ("all", r["lang"], r["id"], f"T{t}", "expect_" + g["status"]):
            s = agg.setdefault(k, [0, 0, 0])
            s[0] += ok; s[1] += 1; s[2] += retry
    (OUT / out_name).write_text(json.dumps({"summary": {k: {"ok": v[0], "n": v[1], "needed_retry": v[2]} for k, v in agg.items()},
                                            "items": items}, indent=1, ensure_ascii=False))
    for k, v in agg.items():
        print(f"{k:24s} {v[0]:3d}/{v[1]:3d} retry={v[2]}")
    for it in items:
        if not it["ok"]:
            print("FAIL", it["key"], it["status"], it["codes"], {k: it.get(k) for k in ("grip", "gold_grip", "speed_missing", "speed_extra") if k in it})


if __name__ == "__main__":
    {"build": build, "run": run, "score": lambda: score(*sys.argv[2:]), "addendum": addendum,
     "run_v2": lambda: run("testset.json", "results_v2.json"),
     "run_add": lambda: run("testset_addendum_t6.json", "results_addendum_t6.json"),
     "score_v2": lambda: score("results_v2.json", "score_v2.json"),
     "score_add": lambda: score("results_addendum_t6.json", "score_addendum_t6.json", "testset_addendum_t6.json"),
     # v3 = v2 + object layer v2 (shared head noun), a POST-HOC fix of the one remaining v2 failure (t8:zh_fragile_combo)
     "run_v3": lambda: (run("testset.json", "results_v3.json"), run("testset_addendum_t6.json", "results_addendum_t6_v3.json")),
     "score_v3": lambda: (score("results_v3.json", "score_v3.json"),
                          score("results_addendum_t6_v3.json", "score_addendum_t6_v3.json", "testset_addendum_t6.json"))}[sys.argv[1]]()
