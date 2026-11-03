"""Step 26 (P1): LLM-in-the-loop onboarding of a NEW LIBERO task onto the stage executor.

The program skeleton is compiled from the official BDDL goal (one pick-place per goal predicate, in goal order); the
proposer (template | LLM) only fills per-object instance parameters from the capability registry, plus the object
order.  Every proposal is validated against the registry (settable keys, ranges, enums, registered reference ids)
before anything runs; refused keys are reported back, never silently clipped.

Modes
  exec  PROPOSAL.json TASK INIT OUT.json   one guarded episode + diagnosis (worker; run in parallel by `loop`)
  loop  TASK ARM TAG [--rounds R] [--dev 0,1,2,3] [--heldout 10-29] [--procs N]
        ARM in {template, llm_open, llm_loop, human}
"""
import argparse, json, os, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "step26_llm_onboard"
PY = "/workdata/cap-x/.venv-libero/bin/python"
OBJ_KEYS = ("lift_dz", "transport_dz", "place_dz", "place_tol_m", "target_offset_site", "target_align", "target_yaw_deg",
            "target_yaw_symmetry_deg", "grasp_reference_id", "grasp_candidate", "grasp_grid")
PROMPT_VERSION = "s26_onboard.v2"
EXECUTOR_SEMANTICS = """Executor (fixed, not configurable here): per object, stages approach -> descend -> grasp -> lift -> transport -> place -> retreat.
The hand keeps the grasp offset, so every height below is the height of the CARRIED OBJECT'S ORIGIN:
 * approach/descend/grasp: the executor picks a top-down grasp itself (planner over a candidate grid, collision-checked); you may
   restrict/extend it with grasp_grid / grasp_candidate, or select a registered demonstration grasp with grasp_reference_id.
 * lift: object origin raised to (its start position) + lift_dz (default 0.15).
 * transport: object origin moved to target + (0,0,transport_dz) (default 0.15), carried at that height.
 * place: object origin servoed to target + (0,0,place_dz) within place_tol_m (default 0.008), then the gripper opens and the
   object drops freely from there.  Default place_dz is 0.041 for an object target.
 * retreat: hand up 0.15.
Target: for an OBJECT target (e.g. a plate) = that object's ORIGIN position when the pick starts (target_offset_site, yaw and
align keys are ignored for object targets).  For a REGION target "@site:<name>" = site_pos + site_R @ target_offset_site;
target_align="collision_center" puts the object's collision centre (not origin) there; target_yaw_deg (+symmetry) rotates the
object about z relative to the site during transport.
An execution guard refuses motions whose swept hand path would hit any non-target object and retries from a snapshot.
Gripper force: fixed ordinary grip (1-100 N band) - not part of this problem.
Success = the OFFICIAL goal predicates at the end of the episode (semantics given)."""
SYSTEM = """You onboard a new tabletop task onto a fixed stage-based robot executor.  You do not write code: you return ONE JSON
object that configures per-object parameters of a program compiled from the official goal.  Derive every number from the
geometry in the context (object sizes, origin heights, target/site poses, and each goal's precomputed
accepted_region_for_obj_origin), state the derivation briefly, and keep values inside the registry ranges.
supported=false is ONLY for goals whose predicate needs a capability the executor does not have (open/close/turn on/off of an
articulated fixture, or a predicate other than on/in between a movable object and a target).  Geometric doubts are NOT a reason
to refuse: propose your best configuration and let the execution results decide.
JSON schema:
{"supported": bool, "unsupported_reason": str|null,
 "objects": [ {"object": <movable object name from the goal>, "target": <object name or "@site:<region site name>">,
               "params": {<key>: <value>, ...},           # only keys from the allowed list
               "derivation": str} ... ]                     # in EXECUTION ORDER (you choose the order)
 "hypothesis": str}                                         # what you expect / what you changed and why
Allowed param keys: """ + ", ".join(OBJ_KEYS)
CONTACT_SEMANTICS = """Acceptance ALSO requires contact_v1_clean=true (frozen contact policy v1.1).  A robot contact is excused
only if it is a finger-link contact on the object currently being carried, between gripper close and release end, and every
contact point lies on the finger's INNER (pad) face.  Everything else is listed in unexcused_contacts: palm/hand
(right_gripper, gripper0_hand_collision) on any object, finger contacts on a non-carried object (e.g. the target plate or an
already placed object), and 'finger_link_on_target' = the carried object touched by the finger link OUTSIDE its inner face
(side/edge/back of the finger, typically because the grasp straddles a wall or edge, or the object rotates in the hand).
Grasp geometry (yaw, lateral, dz of the candidate), placement height/order and target offsets are the levers you control."""


# ---------------------------------------------------------------- program compile + validation
def compile_program(t, proposal, suite="libero_10"):
    from synthesis_v2 import stage_capabilities as SC
    from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances
    reg = SC.describe()["params"]
    eff = {k: set(v["effective_on"]) for k, v in reg.items()}
    refused, roles, inst = [], {}, []
    for k, o in enumerate(proposal["objects"]):
        role, trole = f"obj{k}", f"tgt{k}"
        roles[role], roles[trole] = o["object"], o["target"]
        seq = list(pick_place_instances(role, trole))
        prm = {}
        for key, val in (o.get("params") or {}).items():
            if key == "__inline_grasp_reference":   # human arm only (legacy inline reference, not proposer-settable)
                prm["grasp_reference"] = val
                continue
            spec = reg.get(key)
            why = None
            if spec is None or not spec.get("proposer_settable"):
                why = "not a proposer-settable registry key"
            elif spec["kind"] == "float" and not (spec["lo"] <= float(val) <= spec["hi"]):
                why = f"out of range [{spec['lo']}, {spec['hi']}]"
            elif spec["kind"] == "vec3" and (len(val) != 3 or not all(spec["lo"] <= float(x) <= spec["hi"] for x in val)):
                why = f"vec3 out of range [{spec['lo']}, {spec['hi']}]"
            elif spec["kind"] == "enum" and val not in spec["enum"]:
                why = f"not in {spec['enum']}"
            if why:
                refused.append({"object": o["object"], "key": key, "value": val, "reason": why})
            else:
                prm[key] = tuple(val) if spec["kind"] == "vec3" else val
        eff["grasp_reference"] = {"approach"}
        seq = [replace(i, params={**i.params, **{kk: vv for kk, vv in prm.items() if i.primitive in eff[kk]}}) for i in seq]
        inst.extend(seq)
    prog = TaskProgram(f"libero_t{t}_onboard", "libero", f"libero:{suite}:{t}", roles, tuple(inst), source="composed")
    return prog, refused


# ---------------------------------------------------------------- one episode + diagnosis
def _approach_summary(ap):
    """Planner record -> what a proposer can act on: feasibility, chosen candidate, and the 6 least-blocked candidates."""
    if not isinstance(ap, dict):
        return None
    c = ap.get("candidates") or []
    best = sorted(c, key=lambda x: -(x.get("min_clearance_m") if x.get("min_clearance_m") is not None else -9))[:6]
    keys = ("yaw_deg", "lateral_m", "dz_m", "longitudinal_m", "min_clearance_m", "at_grasp_clearance_m", "blocking", "phase", "feasible")
    return {"feasible": ap.get("feasible"), "grid_source": ap.get("grid_source"), "n_candidates": len(c),
            "n_feasible": sum(bool(x.get("feasible")) for x in c), "chosen": ap.get("chosen"),
            "least_blocked_candidates": [{k: x.get(k) for k in keys if k in x} for x in best]}


def exec_one(proposal, t, init):
    import mujoco
    import s8_task_force as S8
    from s11_guarded_case import GUARD, windows_from
    from synthesis_v2.contact_audit import ContactRecorder
    from synthesis_v2.contact_policy_v1 import DetailRecorder, evaluate as eval_v1
    from synthesis_v2.libero_backend import LiberoSession
    from synthesis_v2.native_producer import StagePlan
    from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
    from synthesis_v2.failure_report import from_episode
    prog, refused = compile_program(t, proposal)
    names = [o["object"] for o in proposal["objects"]]
    s = LiberoSession(t, init, render=False, default_target=names[0])
    box, t0 = {}, time.time()

    def on_session(ss):
        c = ContactRecorder(ss).install()
        box.update(c=c, det=DetailRecorder(c))
    grip = S8.grip_for((1.0, 100.0), S8.EXECUTOR["settle_after_open"])
    plan = ProgramPlan(prog, tuple(StagePlan(it.label, 0.12) for it in prog.instances),
                       {f"obj{k}.pick_1.grasp": grip for k in range(len(names))}, approach_check=True)
    err = None
    try:
        ep = produce_program_episode(prog, init, plan, keep_images=False, session=s, on_session=on_session,
                                     execution_guard={**GUARD, "on_refusal": "restore", "pre_rejected": {}})
    except Exception as e:  # a refused / crashing configuration is an outcome the proposer must see
        s.close()
        return {"task": t, "init": init, "success": False, "status": f"exception:{type(e).__name__}", "error": repr(e)[:600],
                "refused_params": refused}
    env = s.env
    goals = [{"predicate": g[0], "args": list(g[1:]), "true": bool(env._eval_predicate(g))} for g in env.parsed_problem["goal_state"]]
    m, d = s._mj()
    st = [str(x) for x in ep.arrays["stage"]]
    final = ep.meta.get("final_obj_pos_by_role", {})
    per = []
    for k, o in enumerate(proposal["objects"]):
        role, name = f"obj{k}", o["object"]
        labs = [x for x in dict.fromkeys(st) if x.startswith(role + ".")]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name + "_main")
        fpos = d.xpos[bid].copy()
        sub = (ep.meta.get("subtasks") or {}).get(f"{role}.pick_1", {})
        tp = sub.get("target_pos0")
        tgt = o["target"]
        if tgt.startswith("@site:"):
            sp, sR = s.site_pose(tgt.split(":", 1)[1])
            rel = (sR.T @ (fpos - sp)).round(4).tolist()
            tref = {"site_frame_final_origin": rel}
        else:
            tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tgt + "_main")
            tref = {"target_origin_final": d.xpos[tb].round(4).tolist(),
                    "xy_dist_to_target_origin": round(float(np.linalg.norm(fpos[:2] - d.xpos[tb][:2])), 4),
                    "dz_to_target_origin": round(float(fpos[2] - d.xpos[tb][2]), 4)}
        g = (ep.meta.get("grasp_policy_by_instance") or {}).get(f"{role}.pick_1.grasp", {})
        ap = _approach_summary((ep.meta.get("approach_plan") or {}).get(f"{role}.pick_1"))
        per.append({"object": name, "target": tgt, "stages_entered": [x.split(".")[-1] for x in labs],
                    "grasp_reached_band": g.get("reached_band"), "obj_start_origin": (sub.get("obj_pos0") or [None])[:3],
                    "planned_object_target_origin": tp, "final_origin": fpos.round(4).tolist(), **tref,
                    "approach_plan": ap})
    crec, det = box["c"], box["det"]
    try:
        v1 = eval_v1(crec.report(st), crec.events, det.detail, windows_from(ep, t), st)
        clean = bool(v1["clean_v1"]); unexc = [{"robot_body": c["robot_body"], "other": c["other"], "max_n": round(float(c["max_n"]), 2)}
                                               for c in v1["unexcused_contacts"]][:6]
    except Exception as e:
        clean, unexc = None, [repr(e)[:200]]
    try:
        fr = from_episode(prog, ep)
        ff = dict(fr.get("first_failure") or {}); ff.pop("detail", None)
        fr = {"status_code": fr.get("status_code"), "classes_present": fr.get("classes_present"), "first_failure": ff}
    except Exception as e:
        fr = {"error": repr(e)[:300]}
    g = ep.meta.get("execution_guard", {})
    s.close()
    return {"task": t, "init": init, "success": bool(ep.task_ok), "status": ep.status, "n_steps": int(ep.n_steps),
            "wall_s": round(time.time() - t0, 1), "goals": goals, "objects": per, "contact_v1_clean": clean,
            "unexcused_contacts": unexc, "guard_retries": len(g.get("retries", [])),
            "guard_refusals": json.dumps(g.get("refusals", g.get("retries", [])), default=str)[:800],
            "stage_failure": ep.meta.get("stage_failure"), "grasp_failure": ep.meta.get("grasp_failure"),
            "place_failure": ep.meta.get("place_failure"), "failure_report": json.dumps(fr, default=str)[:1500],
            "refused_params": refused}


# ---------------------------------------------------------------- proposers
def template_proposal(ctx):
    """No proposer: goal-compiled program, executor defaults for every parameter."""
    objs = []
    for g in ctx["goal"]:
        a = g["args"]
        if len(a) != 2:
            continue
        objs.append({"object": a[0], "target": ("@site:" + a[1]) if a[1] in ctx["regions"] else a[1], "params": {}})
    unsup = [g["predicate"] for g in ctx["goal"] if g["predicate"] in ctx["unsupported_predicates_by_executor"] or len(g["args"]) != 2]
    return {"supported": not unsup, "unsupported_reason": f"predicates {unsup}" if unsup else None, "objects": objs,
            "hypothesis": "template"}


def human_proposal(t):
    """The hand-engineered TaskSpec of s7_task_config (reference arm), in the same proposal format."""
    from s7_task_config import SPECS
    sp = SPECS[t]
    tg = sp.targets or tuple(sp.target for _ in sp.objects)
    objs = []
    for k, (r, n) in enumerate(sp.objects):
        p = {}
        if sp.container_dz:
            p.update(place_dz=0.14, transport_dz=0.175)
        p.update({kk: (list(v) if isinstance(v, tuple) else v) for kk, v in (sp.target_params or {}).items()})
        if sp.object_target_params:
            p.update({kk: (list(v) if isinstance(v, tuple) else v) for kk, v in sp.object_target_params[k].items()})
        if sp.grasp_grid:
            p["grasp_grid"] = {"yaws_deg": list(sp.grasp_grid[0]), "laterals_m": list(sp.grasp_grid[1]), "dzs_m": list(sp.grasp_grid[2])}
        if sp.grasp_reference and sp.grasp_reference[k]:
            p["__inline_grasp_reference"] = sp.grasp_reference[k]
        objs.append({"object": n, "target": tg[k][1], "params": p})
    return {"supported": True, "objects": objs, "hypothesis": "human TaskSpec (s7_task_config.SPECS)"}


def _ctx_for_llm(ctx):
    caps = {k: {kk: v[kk] for kk in ("doc", "kind", "lo", "hi", "enum", "fields") if kk in v}
            for k, v in ctx["capabilities"]["params"].items() if k in OBJ_KEYS}
    c = {k: v for k, v in ctx.items() if k not in ("capabilities",)}
    c["allowed_params"] = caps
    return c


def _compact(r):
    keep = ("init", "success", "status", "goals", "objects", "contact_v1_clean", "unexcused_contacts", "guard_retries",
            "guard_refusals", "stage_failure", "grasp_failure", "place_failure", "failure_report", "refused_params", "error")
    return {k: r.get(k) for k in keep if r.get(k) not in (None, [], {})}


def llm_proposal(client, ctx, history, criterion="task"):
    user = {"executor": EXECUTOR_SEMANTICS, "task_context": _ctx_for_llm(ctx)}
    if criterion == "clean":
        user["acceptance"] = CONTACT_SEMANTICS
    if history:
        user["previous_rounds"] = [{"round": h["round"], "proposal": h["proposal"], "dev_accepted": f"{h['n_ok']}/{h['n']}",
                                    "dev_results": [_compact(r) for r in h["results"]] if j >= len(history) - 2
                                    else "omitted (older round)"} for j, h in enumerate(history)]
        user["instruction"] = ("The previous configuration did not succeed on every development init.  Diagnose from the results "
                               "which stage/predicate (or contact rule) failed and why (geometry, heights, order, grasp), then return a corrected full "
                               "proposal.  Change only what the evidence supports; say what you changed in 'hypothesis'.")
    else:
        user["instruction"] = "Return the initial proposal."
    obj, rec = client.complete_json(SYSTEM, json.dumps(user, default=str), prompt_version=PROMPT_VERSION)
    return obj, rec.to_dict()


# ---------------------------------------------------------------- parallel execution
def run_inits(proposal, t, inits, wdir, procs):
    wdir.mkdir(parents=True, exist_ok=True)
    pf = wdir / "proposal.json"; pf.write_text(json.dumps(proposal, indent=1))
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": ".:scripts/mt", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    todo, running, res = list(inits), [], {}
    while todo or running:
        while todo and len(running) < procs:
            i = todo.pop(0); of = wdir / f"init{i}.json"
            if of.exists():
                res[i] = json.loads(of.read_text()); continue
            lf = open(wdir / f"init{i}.log", "w")
            running.append((i, of, subprocess.Popen([PY, __file__, "exec", str(pf), str(t), str(i), str(of)], cwd=ROOT, env=env,
                                                    stdout=lf, stderr=subprocess.STDOUT)))
        time.sleep(2)
        for x in list(running):
            if x[2].poll() is not None:
                running.remove(x)
                res[x[0]] = json.loads(x[1].read_text()) if x[1].exists() else {"init": x[0], "success": False,
                                                                                   "status": f"worker_crash:{x[2].returncode}"}
    return [res[i] for i in inits]


def _range(s):
    out = []
    for part in [x for x in s.split(",") if x.strip()]:
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _ok(x, criterion):
    return bool(x.get("success")) and (criterion == "task" or bool(x.get("contact_v1_clean")))


def loop(args):
    from s26_context import task_context
    from synthesis_v2.llm_interface import LLMClient, LLMConfig
    t = args.task
    base = OUT / f"t{t}" / args.tag / args.arm
    base.mkdir(parents=True, exist_ok=True)
    cf = OUT / "context" / f"t{t}_init0.json"
    if not cf.exists():
        cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text(json.dumps(task_context(t, 0), indent=1, default=str))
    ctx = json.loads(cf.read_text())
    dev, held = _range(args.dev), _range(args.heldout)
    client = None
    if args.arm.startswith("llm"):
        client = LLMClient(LLMConfig.from_env(max_output_tokens=8000, max_calls=args.rounds + 1, timeout_s=300.0,
                                              cache_dir=str(OUT / "llm_cache"), ledger_path=str(base / "llm_ledger.jsonl")))
    history, rounds = [], (args.rounds if args.arm == "llm_loop" else 1)
    for r in range(rounds):
        pf, dd = base / f"round{r}_proposal.json", base / f"round{r}_dev"
        if pf.exists() and dd.exists() and all((dd / f"init{i}.json").exists() for i in dev):
            prop = json.loads(pf.read_text())["proposal"]
            res = [json.loads((dd / f"init{i}.json").read_text()) for i in dev]
            n_ok = sum(_ok(x, args.criterion) for x in res)
            history.append({"round": r, "proposal": prop, "results": res, "n_ok": n_ok, "n": len(res)})
            print(f"T{t} {args.arm} round{r}: resumed dev {n_ok}/{len(res)}", flush=True)
            if n_ok == len(res):
                break
            continue
        if args.arm == "template":
            prop, call = template_proposal(ctx), None
        elif args.arm == "human":
            prop, call = human_proposal(t), None
        else:
            prop, call = llm_proposal(client, ctx, history, args.criterion)
        (base / f"round{r}_proposal.json").write_text(json.dumps({"proposal": prop, "llm_call": call}, indent=1, default=str))
        if not prop.get("supported", True):
            print(f"T{t} {args.arm} round{r}: proposer declares UNSUPPORTED: {prop.get('unsupported_reason')}", flush=True)
            history.append({"round": r, "proposal": prop, "results": [], "n_ok": 0, "n": 0, "unsupported": True})
            break
        res = run_inits(prop, t, dev, base / f"round{r}_dev", args.procs)
        n_ok = sum(_ok(x, args.criterion) for x in res)
        history.append({"round": r, "proposal": prop, "results": res, "n_ok": n_ok, "n": len(res)})
        print(f"T{t} {args.arm} round{r}: dev {n_ok}/{len(res)} " + " ".join(f"i{x['init']}:{x.get('status')}" for x in res), flush=True)
        if n_ok == len(res):
            break
    best = max(history, key=lambda h: (h["n_ok"], h["round"]))
    summary = {"task": t, "arm": args.arm, "tag": args.tag, "criterion": args.criterion, "dev_inits": dev, "heldout_inits": held,
               "rounds": [{"round": h["round"], "dev_ok": h["n_ok"], "dev_n": h["n"], "hypothesis": h["proposal"].get("hypothesis"),
                           "unsupported": h.get("unsupported", False)} for h in history],
               "selected_round": best["round"], "llm_spend": client.spend() if client else None}
    if not best.get("unsupported") and held:
        hres = run_inits(best["proposal"], t, held, base / "heldout", args.procs)
        summary["heldout_success"] = sum(bool(x.get("success")) for x in hres)
        summary["heldout_n"] = len(hres)
        summary["heldout_success_and_clean"] = sum(bool(x.get("success")) and bool(x.get("contact_v1_clean")) for x in hres)
        summary["heldout_status"] = {}
        for x in hres:
            summary["heldout_status"][x.get("status")] = summary["heldout_status"].get(x.get("status"), 0) + 1
    (base / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str), flush=True)


def main():
    if sys.argv[1] == "exec":
        prop = json.loads(Path(sys.argv[2]).read_text())
        r = exec_one(prop, int(sys.argv[3]), int(sys.argv[4]))
        Path(sys.argv[5]).write_text(json.dumps(r, indent=1, default=str))
        print(f"init{sys.argv[4]} success={r['success']} status={r['status']}", flush=True)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("task", type=int); ap.add_argument("arm"); ap.add_argument("tag")
    ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--dev", default="0-3")
    ap.add_argument("--criterion", choices=("task", "clean"), default="task")
    ap.add_argument("--heldout", default="10-29"); ap.add_argument("--procs", type=int, default=8)
    loop(ap.parse_args())


if __name__ == "__main__":
    main()
