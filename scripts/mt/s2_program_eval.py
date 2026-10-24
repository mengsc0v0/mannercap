"""Step 2 acceptance, part 2: an LLM-composed typed program must pass structural checks AND run.

Structural validity is cheap and says nothing about physics, so the program is also executed at the
neutral plan and the episode result is reported.  A hand-written reference program is run on the same
seed so the model's program is compared against a known-feasible one, not against an absolute bar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis_v2.language_compiler import (  # noqa: E402
    PROGRAM_PROMPT_VERSION, build_program_prompt, check_program, parse_program,
)
from synthesis_v2.llm_interface import LLMClient, LLMConfig  # noqa: E402
from synthesis_v2.program_executor import neutral_program_plan, produce_program_episode  # noqa: E402
from synthesis_v2.task_program import TaskProgram, legacy_program  # noqa: E402

GOAL = ("Pick up cubeA and put it back down where it started, then do the same for cubeB. "
        "Use one pick-and-put-down subtask per cube.")
ENV_TASK = "stack"
SCENE_OBJECTS = {"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"}


def reference_program() -> TaskProgram:
    from synthesis_v2.task_program import chain, pick_place_instances
    return TaskProgram("two_cube_putback_ref", "robosuite_native", ENV_TASK, dict(SCENE_OBJECTS),
                       chain(pick_place_instances("cubeA", "spotA"), pick_place_instances("cubeB", "spotB")),
                       source="hand_written")


def execute(program: TaskProgram, seed: int, keep_images: bool) -> dict:
    ep = produce_program_episode(program, seed, neutral_program_plan(program), keep_images=keep_images)
    moved = {}
    for role, nm in program.objects.items():
        if nm.startswith("@spot:"):
            continue
        sub = next((r for r in ep.meta.get("subtasks", {}).values() if r.get("obj_role") == role), {})
        p0 = sub.get("obj_pos0")
        p1 = (ep.meta.get("final_obj_pos_by_role") or {}).get(role)
        if p0 and p1:
            moved[role] = round(float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))), 4)
    return {"status": ep.status, "n_steps": int(np.asarray(ep.arrays["actions"]).shape[0]),
            "n_instances": len(program.instances), "n_grasps": len(program.of_primitive("grasp")),
            "displacement_m": moved, "program_hash": program.program_hash(),
            "_actions": np.asarray(ep.arrays["actions"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--cache", default="reports/llm_cache/program")
    ap.add_argument("--ledger", default="reports/llm_cache/ledger.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    keep = not a.no_images

    recs = [{"source": "hand_written", "structural_errors": [], **execute(reference_program(), a.seed, keep)}]
    ref_act = recs[0]["_actions"]
    spend = None
    if a.live:
        client = LLMClient(LLMConfig.from_env(cache_dir=a.cache, ledger_path=a.ledger, max_calls=8))
        system, user = build_program_prompt(GOAL, ENV_TASK, SCENE_OBJECTS, example=legacy_program("stack"))
        rec: dict = {"source": "llm"}
        try:
            obj, _ = client.complete_json(system, user, prompt_version=PROGRAM_PROMPT_VERSION)
            prog = parse_program(obj, language=GOAL)
            rec["structural_errors"] = check_program(prog)
            rec["raw_instances"] = [i.instance_id for i in prog.instances]
            if not rec["structural_errors"]:
                rec.update(execute(prog, a.seed, keep))
                act = rec["_actions"]
                # physical equivalence: differently-named programs of the same shape must drive the
                # arm identically, otherwise "no new execution branch" would be an empty claim
                rec["max_action_diff_vs_reference"] = (
                    float(np.abs(act - ref_act).max()) if act.shape == ref_act.shape else None)
            else:
                rec["status"] = "structural_error"
        except Exception as exc:
            rec.update(status=f"error:{type(exc).__name__}", detail=str(exc)[:400])
        recs.append(rec)
        spend = client.spend()

    for r in recs:
        r.pop("_actions", None)
        print(json.dumps({k: v for k, v in r.items() if k != "raw_instances"}))
    if spend:
        print("SPEND " + json.dumps(spend))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"goal": GOAL, "records": recs, "spend": spend,
                   "prompt_version": PROGRAM_PROMPT_VERSION}, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
