"""Step 30: manner instruction -> per-stage speed / per-object grip commands -> executed, audited, repaired episodes.

  s30_manner_gen.py intake                         LLM intake + admit for the instruction set -> intake.json
  s30_manner_gen.py run TASK INIT [IID ...]        execute (with bounded repair) -> runs/t{T}_i{I}_{IID}.json + npz
Protocol: reports/step30_manner_gen/protocol.json (written before any execution).
"""
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from s7_task_config import ROOT  # noqa: E402

OUT = Path(os.environ.get("S30_OUT", str(ROOT / "reports" / "step30_manner_gen")))
VERSION = "s30_manner_gen.v6"   # v6 (step 31): carry-clearance repair + clearance experience
SPEED_SWITCH_MARGIN = 0.10  # posterior gain needed to move an unrequested carry speed
PROP = {6: (ROOT / "reports" / "step27_t6_repair" / "frozen_proposal.json", "proposal"),
        8: (ROOT / "reports" / "step29_t8_contact" / "selected_round3_proposal.json", None)}
TASKS = {
    0: dict(lang="put both the alphabet soup and the tomato sauce in the basket",
            X=("tomato sauce", "tomato_sauce"), Y=("alphabet soup", "alphabet_soup")),
    7: dict(lang="put both the alphabet soup and the cream cheese box in the basket",
            X=("cream cheese box", "cream_cheese"), Y=("alphabet soup", "alphabet_soup")),
    1: dict(lang="put both the cream cheese box and the butter in the basket",
            X=("cream cheese box", "cream_cheese"), Y=("butter", "butter")),
    6: dict(lang="put the white mug on the plate and put the chocolate pudding to the right of the plate",
            X=("white mug", "obj1"), Y=("chocolate pudding", "obj0")),
    8: dict(lang="put both moka pots on the stove", X=("right moka pot", "obj1"), Y=("left moka pot", "obj0"),
            glossary=" (roles: obj0 = the left moka pot, obj1 = the right moka pot)"),
}
TEMPLATES = {
    "neutral": "{base}",
    "gentle_X": "{base}, holding the {X} gently",
    "firm_X": "{base}, holding the {X} firmly",
    "slowcarry_X": "{base}, carrying the {X} slowly",
    "fastcarry_X": "{base}, carrying the {X} quickly",
    "gentle_slow_X": "{base}; hold the {X} gently and carry it slowly",
    "firm_fast_X": "{base}; grip the {X} firmly and carry it quickly",
    "Xgentle_Yfirm": "{base}; hold the {X} gently and the {Y} firmly",
    "Xfirm_Ygentle": "{base}; hold the {X} firmly and the {Y} gently",
    "slow_approach": "{base}, approaching each object slowly",
    "slow_all": "{base}, and move slowly",
}


def instructions(t):
    c = TASKS[t]
    return {k: v.format(base=c["lang"].capitalize(), X=c["X"][0], Y=c["Y"][0]) for k, v in TEMPLATES.items()}


def program_of(t):
    """-> (program, roles, names, session factory).  T6/T8 come from their frozen onboarding proposals."""
    import s8_task_force as S8
    import s7_task_config as C
    c = TASKS[t]
    if t in PROP:
        import s26_onboard as S26
        from synthesis_v2.libero_backend import LiberoSession
        f, key = PROP[t]
        p = json.loads(f.read_text())
        p = p[key] if key else p
        prog, refused = S26.compile_program(t, p)
        assert not refused, refused
        names = [o["object"] for o in p["objects"]]
        roles = [f"obj{k}" for k in range(len(names))]
        S8.SPECS = {**S8.SPECS, t: SimpleNamespace(objects=tuple(zip(roles, names)))}  # per_object reads S8.SPECS[t]
        make = lambda init: LiberoSession(t, init, render=False, default_target=names[0])  # noqa: E731
    else:
        prog = C.make_program(t)
        roles = [r for r, _ in C.SPECS[t].objects]
        names = [n for _, n in C.SPECS[t].objects]
        make = lambda init: C.make_session(t, init, prog)  # noqa: E731
    prog = dataclasses.replace(prog, language=c["lang"] + c.get("glossary", ""))
    return prog, roles, names, make


def client():
    from synthesis_v2.llm_interface import LLMClient, LLMConfig
    return LLMClient(LLMConfig.from_env(cache_dir=str(OUT / "llm_cache"), ledger_path=str(OUT / "llm_ledger.jsonl"), max_calls=200),
                     offline=os.environ.get("S30_OFFLINE") == "1")


def intake_one(text, prog, cl):
    from synthesis_v2.language_plan import _attempt, _feedback_from
    from synthesis_v2.manner_libero import MannerRefused, request_from_semantics
    obj, rec, intake, dec, pv = _attempt(text, prog, cl)
    att = [{"prompt_version": pv, "status": dec["status"], "request_hash": rec.request_hash, "llm_status": rec.status}]
    fb = _feedback_from(dec) if dec["status"] != "accept" else None
    if fb is not None:
        obj, rec, intake, dec, pv = _attempt(text, prog, cl, feedback=fb, previous=obj)
        att.append({"prompt_version": pv, "status": dec["status"], "request_hash": rec.request_hash,
                    "llm_status": rec.status, "feedback": fb})
    out = {"instruction": text, "status": dec["status"], "attempts": att, "decision": dec,
           "bindings": [dataclasses.asdict(b) if dataclasses.is_dataclass(b) else b.__dict__
                        for b in intake.semantics.bindings], "request": None}
    if dec["status"] == "accept":
        try:
            out["request"] = request_from_semantics(prog, intake.semantics).to_dict()
        except MannerRefused as e:
            out["status"] = f"refused:{e}"
    return out


def cmd_intake():
    cl = client()
    res = {}
    for t in TASKS:
        prog, *_ = program_of(t)
        for iid, text in instructions(t).items():
            r = intake_one(text, prog, cl)
            res[f"t{t}:{iid}"] = r
            rq = r["request"] or {}
            print(f"T{t} {iid:14s} {r['status']:10s} speed={rq.get('speed')} grip={rq.get('grip')}", flush=True)
    (OUT / "intake.json").write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))


# ---------------------------------------------------------------- experience library (grid rows + write-back)
GRIDS = [(ROOT / "reports" / "step25_response_grid", "t{t}_init*.json"),
         (ROOT / "reports" / "step28_t6_manner" / "grid", "t{t}_init*.json"),
         (ROOT / "reports" / "step28_t6_manner" / "grid_fresh", "t{t}_init*.json")]
WB = OUT / "library_rows.jsonl"


def library_for(t, init):
    """Rows of every grid (requested object only, the one whose force was varied) + step-30 write-back rows.
    Rows of the SAME (task, init) are hidden, so a decision never sees the outcome of its own scene."""
    import s25_policies as P
    lib, rows = P.Library(), []
    for d0, pat in GRIDS:
        for tt in (0, 1, 7, 6):
            for f in sorted(d0.glob(pat.format(t=tt))):
                d = json.loads(f.read_text())
                for e in d["episodes"].values():
                    if "error" in e:
                        continue
                    o = e["objects"][e["requested_role"]]
                    rows.append({"task": tt, "init": d["init"], "object": o.get("target"), "force": tuple(e["force"]),
                                 "speed": e["speed"], "hold_min": o.get("hold_min"), "hold_max": o.get("hold_max"),
                                 "hold": o.get("hold_mean"),
                                 "stable": bool(o.get("grasp_stable")) and o.get("hold_mean") is not None,
                                 "task_ok": bool(e["task_ok"]), "v": e.get("carry_speed_med_mps"), "src": "grid"})
    # S30_WB_READ (':'-separated files) freezes the cross-case library for a formal run (parallel cases then do not
    # depend on finishing order); default = this run's own write-back file.
    for wb in ([Path(x) for x in os.environ["S30_WB_READ"].split(":") if x] if "S30_WB_READ" in os.environ else [WB]):
        if wb.exists():
            for ln in wb.read_text().splitlines():
                r = json.loads(ln)
                r["force"] = tuple(r["force"])
                rows.append(r)
    if os.environ.get("S30_NOLIB") == "1":   # ablation: rule prior + cause-direction repair only
        rows = []
    lib.rows = [r for r in rows if not (r["task"] == t and r["init"] == init)]
    return lib


def stage_speed(ep, label):
    st = [str(x) for x in ep.arrays["stage"]]
    pos = np.asarray(ep.arrays["state"])[:, :3]
    idx = [k for k, x in enumerate(st) if x == label]
    if len(idx) < 3:
        return None
    return float(np.median(np.linalg.norm(np.diff(pos[idx], axis=0), axis=1) * 20.0))


def role_carry_speed(ep, role):
    st = [str(x) for x in ep.arrays["stage"]]
    pos = np.asarray(ep.arrays["state"])[:, :3]
    idx = [k for k, x in enumerate(st) if x.startswith(role + ".") and x.split(".")[-1] in ("lift", "transport")]
    if len(idx) < 3:
        return None
    return float(np.median(np.linalg.norm(np.diff(pos[idx], axis=0), axis=1) * 20.0))


def clearance_prior(lib, t, obj):
    """Experience for the carry height of one object in one task: among the clearance levels already executed in
    OTHER scenes, the one with the highest posterior rate of carries without failure evidence (Beta(1,1);
    ties -> the lower carry).  No experience -> 0 (the task's default heights)."""
    by = {0.0: []}
    for r in lib.rows:
        if r.get("task") == t and r.get("object") == obj and "carry_ev" in r:
            by.setdefault(round(float(r.get("clear", 0.0)), 3), []).append(not r["carry_ev"])
    return max(sorted(by), key=lambda d: ((sum(by[d]) + 1) / (len(by[d]) + 2), -d))


def apply_clearance(prog, clear):
    from synthesis_v2.program_executor import LIFT_DZ
    if not clear:
        return prog
    inst = []
    for it in prog.instances:
        if clear.get(it.object_role) and it.primitive in ("lift", "transport"):
            k = it.primitive + "_dz"
            it = dataclasses.replace(it, params={**it.params, k: float(it.params.get(k, LIFT_DZ)) + clear[it.object_role]})
        inst.append(it)
    return dataclasses.replace(prog, instances=tuple(inst))


PUSH_M = 0.02   # another tracked object moving more than this while an object is carried = carry collision


def carry_evidence(ep, roles, names, events, v1, st):
    """role -> evidence that the execution failed during THIS object's carry (lift/transport stages):
    an unexcused / unexplained robot contact in those steps, or another tracked object displaced > PUSH_M."""
    from synthesis_v2.contact_policy_v1 import _steps_of
    bad = [(c["robot_body"], c["other"]) for c in v1["unexcused_contacts"]] + \
          [(c["robot_body"], c["other"]) for c in v1["support_contacts_unexplained"]]
    pos = {k[4:]: np.asarray(v, dtype=np.float64) for k, v in ep.arrays.items() if k.startswith("pos_")}
    ev = {}
    for r, n in zip(roles, names):
        idx = [k for k, x in enumerate(st) if x.startswith(r + ".") and x.split(".")[-1] in ("lift", "transport")]
        if not idx:
            continue
        S, e = set(idx), []
        for key in bad:
            hit = S & _steps_of(events, key)
            if hit:
                e.append(f"contact:{key[0]}|{key[1]}@{min(hit)}")
        for o, pp in pos.items():
            if o == n or len(pp) != len(st):
                continue
            d = float(np.max(np.linalg.norm(pp[idx] - pp[idx[0]], axis=1)))
            if d > PUSH_M:
                e.append(f"pushed:{o}:{d:.3f}")
        if e:
            ev[r] = e
    return ev


def execute(t, init, prog, roles, names, make, cmd, npz_path):
    import s8_task_force as S8
    from s11_guarded_case import GUARD, windows_from
    from synthesis_v2.contact_audit import ContactRecorder
    from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1
    from synthesis_v2.native_producer import StagePlan
    from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
    prog = apply_clearance(prog, cmd.get("clear"))
    s = make(init)
    sub = S8.install_recorder(s, names)
    box, t0 = {}, time.time()

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))
    settle = S8.EXECUTOR["settle_after_open"]
    grips = {it.instance_id: S8.grip_for(tuple(cmd["grip"][it.object_role]), settle)
             for it in prog.instances if it.primitive == "grasp"}
    stages = tuple(StagePlan(it.label, float(cmd["speed"].get(it.instance_id, 0.12))) for it in prog.instances)
    ep = produce_program_episode(prog, init, ProgramPlan(prog, stages, grips, approach_check=True), keep_images=False,
                                 session=s, on_session=on_session,
                                 execution_guard={**GUARD, "on_refusal": "restore", "pre_rejected": {}})
    crec, det = box["c"], box["det"]
    s.close()
    ctrl, rows = sub.formal_arrays()
    arr = np.asarray(rows, dtype=np.float64).reshape(len(ctrl), 2 * len(names))
    st = [str(x) for x in ep.arrays["stage"]]
    rec = {"ctrl": ctrl, "L": {n: arr[:, 2 * i] for i, n in enumerate(names)},
           "R": {n: arr[:, 2 * i + 1] for i, n in enumerate(names)},
           "substep_cost": sub.cost(), "contact_audit": crec.report(st)}
    po = S8.per_object(ep, prog, t, rec)
    v1 = eval_v1(rec["contact_audit"], crec.events, det.detail, windows_from(ep, t), st)
    save = {}
    for k, v in ep.arrays.items():
        a = np.asarray(v)
        save[k] = a.astype(str) if a.dtype == object else a
    save["force_ctrl_steps"] = np.asarray(ctrl)
    for i, n in enumerate(names):
        save[f"force_L__{n}"], save[f"force_R__{n}"] = arr[:, 2 * i], arr[:, 2 * i + 1]
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **save)
    measured = {"task_ok": bool(ep.task_ok), "contact_clean": bool(v1["clean_v1"]),
                "stage_speed": {it.instance_id: stage_speed(ep, it.label) for it in prog.instances
                                if it.primitive != "grasp"},
                "objects": {r: {"hold_min": po[r].get("hold_min"), "hold_max": po[r].get("hold_max"),
                                "hold_mean": po[r].get("hold_mean"), "stable": bool(po[r].get("grasp_stable")),
                                "valid": bool(po[r].get("measurement_valid")), "target": po[r].get("target"),
                                "carry_v": role_carry_speed(ep, r)} for r in roles if r in po}}
    g = ep.meta.get("execution_guard", {})
    info = {"status": ep.status, "n_steps": int(ep.n_steps), "wall_s": round(time.time() - t0, 1),
            "unexcused": [{"robot_body": c["robot_body"], "other": c["other"], "max_n": round(float(c["max_n"]), 2)}
                          for c in v1["unexcused_contacts"]][:6],
            "support_unexplained": [{"robot_body": c["robot_body"], "other": c["other"],
                                     "max_n": round(float(c.get("max_n", 0) or 0), 2), "stages": c.get("stages")}
                                    for c in v1["support_contacts_unexplained"]][:6],
            "guard_retries": len(g.get("retries", [])), "stage_failure": ep.meta.get("stage_failure"),
            "grasp_failure": ep.meta.get("grasp_failure"), "npz": str(npz_path.relative_to(OUT)),
            "carry_evidence": carry_evidence(ep, roles, names, crec.events, v1, st)}
    return measured, info


def run_case(t, init, iid, budget=3):
    import s25_policies as P
    from s28_t6_replay import member_repair
    from synthesis_v2 import manner_libero as M
    intake = json.loads((OUT / "intake.json").read_text())[f"t{t}:{iid}"]
    out = {"version": VERSION, "module": M.MANNER_LIBERO_VERSION, "task": t, "init": init, "instruction_id": iid,
           "instruction": intake["instruction"], "intake_status": intake["status"], "request": intake["request"],
           "attempts": [], "final": None}
    if intake["status"] != "accept":
        out["final"] = {"joint_pass": False, "stop": "not_admitted"}
        return out
    prog, roles, names, make = program_of(t)
    req = M.MannerRequest(intake["request"]["speed"], intake["request"]["grip"], tuple(intake["request"]["clauses"]))
    lib = library_for(t, init)
    n_lib0 = len(lib.rows)
    obj_of = {r: n for r, n in zip(roles, names)}
    big = (0.0, 1e9)

    def sband_name(role):
        lv = M.carry_level(prog, req, role)
        return lv if lv else "any"

    def choose_grip(role, lvl, v0):
        """Library posterior over (force candidate, carry speed); carry speeds limited to the bound carry band.
        Tie-break: keep the current carry speed, then the force order."""
        fb, sb = M.GRIP_BAND[lvl], P.SPEED_BANDS[sband_name(role)]
        per_v = {}
        for v in sorted(set(M.allowed_carry(prog, req, role, tuple(P.SPEEDS))) | {v0}):
            sc = []
            for f in P.FORCES:
                if f[0] > fb[1]:
                    continue
                pr = P.rule_prior(f, v, fb, sb)
                p, n, basis = lib.p_pass(t, obj_of[role], f, v, fb, sb, pr)
                sc.append((-p, P.FORCES.index(f), f, basis))
            per_v[v] = sorted(sc)[0]
        # Minimal deviation: keep the current carry speed unless OBJECT-SPECIFIC evidence shows a clear gain
        # (pooled any-object evidence says nothing about this object's speed sensitivity).
        f_keep, p_keep = per_v[v0][2], -per_v[v0][0]
        alt = [(-b[0], v, b[2]) for v, b in per_v.items()
               if v != v0 and b[3] == "obj+cfg" and -b[0] >= p_keep + SPEED_SWITCH_MARGIN]
        if alt:
            _, v, f = max(alt)
            return f, v
        return f_keep, v0

    tried = []

    def repair_force(role, lvl, prev, cause, v0, _tf):
        P.OBJECT[t] = obj_of[role]
        rq = {"force": M.GRIP_BAND_NAME[lvl], "speed": sband_name(role)}
        sp = sorted(set(M.allowed_carry(prog, req, role, tuple(P.SPEEDS))) | {v0})
        rank = [(f, s) for f in P.FORCES for s in sp]
        tr = [(tuple(c["grip"][role]), M.carry_v(prog, c["speed"], role)) for c in tried]
        tr += redundant(role, lvl, rank)
        nx = member_repair((tuple(prev), v0), cause, rq, tr, rank, lib, t, obj_of[role])
        return None if nx is None else (nx[0], nx[1])

    seen = []   # (role, force, v, hold_max) of executed attempts in this episode

    def redundant(role, lvl, rank):
        """Candidates whose outcome is already known in THIS scene: same closure stop `lo` and carry speed as an
        executed attempt whose hold never reached its cap `hi` -> any cap above that hold gives the same closure.
        (Pruning on other scenes' rows was tried in pilot_v4 and dropped: it saved no execution and lost a pass.)"""
        out_ = []
        for f, s_ in rank:
            if any(r_ == role and v_ == s_ and hmax is not None and f[0] == f0[0] and min(f[1], f0[1]) > hmax
                   for r_, f0, v_, hmax in seen):
                out_.append((f, s_))
        return out_

    # S31_CLEAR_READ (':'-separated write-back files): extra experience used ONLY by the clearance prior (the force
    # library is unchanged), rows of the evaluated (task, init) hidden like everywhere else
    clib = SimpleNamespace(rows=list(lib.rows) + [
        r for f in os.environ.get("S31_CLEAR_READ", "").split(":") if f
        for r in map(json.loads, Path(f).read_text().splitlines()) if not (r["task"] == t and r["init"] == init)])
    clear0 = {} if os.environ.get("S31_NOCLEARPRIOR") == "1" else \
        {r: clearance_prior(clib, t, obj_of[r]) for r in roles}
    out["clearance_prior"] = clear0
    cmd = M.initial_commands(prog, req, choose_grip, clear0)
    for k in range(budget):
        tried.append(cmd)
        npz = OUT / "npz" / f"t{t}_i{init}_{iid}_a{k}.npz"
        try:
            meas, info = execute(t, init, prog, roles, names, make, cmd, npz)
        except Exception as e:
            out["attempts"].append({"k": k, "cmd": _j(cmd), "error": repr(e)[:400]})
            out["final"] = {"joint_pass": False, "stop": "exception"}
            return out
        aud = M.audit(req, meas)
        wb = []
        for r, o in meas["objects"].items():
            row = {"task": t, "init": init, "object": o["target"], "force": list(cmd["grip"][r]),
                   "speed": M.carry_v(prog, cmd["speed"], r), "hold_min": o["hold_min"], "hold_max": o["hold_max"],
                   "hold": o["hold_mean"], "stable": o["stable"] and o["hold_mean"] is not None,
                   "task_ok": meas["task_ok"], "v": o["carry_v"], "src": f"s30:{iid}:a{k}",
                   "clear": float(cmd.get("clear", {}).get(r, 0.0)), "carry_ev": bool(info["carry_evidence"].get(r))}
            wb.append(row)
        with open(WB, "a") as fh:
            for row in wb:
                fh.write(json.dumps(row) + "\n")
        out["attempts"].append({"k": k, "cmd": _j(cmd), "audit": aud, "measured": meas, "info": info})
        for row in wb:     # in-episode write-back BEFORE the repair decision: it must see this attempt
            row["force"] = tuple(row["force"])
            lib.rows.append(row)
        for r, o in meas["objects"].items():
            seen.append((r, tuple(cmd["grip"][r]), M.carry_v(prog, cmd["speed"], r), o["hold_max"]))
        nxt, acts = M.next_commands(prog, req, cmd, aud, tried, repair_force, info["carry_evidence"])
        out["attempts"][-1]["repair"] = acts
        if aud["joint_pass"]:
            out["final"] = {"joint_pass": True, "attempts": k + 1}
            break
        if nxt is None:
            out["final"] = {"joint_pass": False, "attempts": k + 1, "stop": acts[-1].get("why")}
            break
        cmd = nxt
    if out["final"] is None:
        out["final"] = {"joint_pass": False, "attempts": budget, "stop": "budget"}
    out["library_rows_at_start"] = n_lib0
    return out


def _j(cmd):
    return {"speed": cmd["speed"], "grip": {k: list(v) for k, v in cmd["grip"].items()}, "clear": dict(cmd.get("clear") or {})}


def cmd_run(t, init, iids):
    d = OUT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for iid in iids:
        f = d / f"t{t}_i{init}_{iid}.json"
        if f.exists():
            continue
        r = run_case(t, init, iid)
        f.write_text(json.dumps(r, indent=1, default=str))
        a = r["attempts"]
        fails = [(c["target"], c["cause"]) for x in a[-1:] for c in x.get("audit", {}).get("clauses", []) if not c["pass"]]
        print(f"T{t} i{init} {iid:14s} final={r['final']} n={len(a)} last_fail={fails} "
              f"task_ok={a[-1].get('measured', {}).get('task_ok') if a else None} "
              f"clean={a[-1].get('measured', {}).get('contact_clean') if a else None}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "intake":
        cmd_intake()
    elif a[0] == "run":
        cmd_run(int(a[1]), int(a[2]), a[3:] or list(TEMPLATES))
    else:
        raise SystemExit(f"unknown command {a[0]}")
