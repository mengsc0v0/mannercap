"""Step 1 acceptance: two grasp instances, each with its own grip band and its own force target.

Runs a two-object pick-and-put-back program on the existing ``stack`` environment.  The program is
built from the shared ``pick_place_instances`` template only -- no new task execution branch -- and
the two grasps carry different bands.  Running the swapped assignment on the same seed shows the
band follows the *object binding*, not the stage order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthesis_v2.native_producer import GripPlan, StagePlan
from synthesis_v2.program_executor import ProgramPlan, produce_program_episode
from synthesis_v2.task_program import TaskProgram, chain, pick_place_instances

GENTLE = GripPlan(2.0, 6.0)
FIRM = GripPlan(24.0, 34.0)


def two_cube_program() -> TaskProgram:
    inst = chain(
        pick_place_instances("cubeA", "spotA", k=1),
        pick_place_instances("cubeB", "spotB", k=1),
    )
    return TaskProgram(
        name="two_cube_putback", backend="robosuite_native", env_task="stack",
        objects={"cubeA": "cubeA", "cubeB": "cubeB", "spotA": "@spot:cubeA", "spotB": "@spot:cubeB"},
        instances=inst, goal=("AtStart(cubeA)", "AtStart(cubeB)"), source="composed",
        language="pick up both cubes and put each one back down",
    )


def plan_for(program: TaskProgram, grips: dict[str, GripPlan], v: float = 0.12) -> ProgramPlan:
    stages = tuple(StagePlan(it.label, v) for it in program.instances)
    return ProgramPlan(program, stages, grips)


def grasp_windows(ep, program: TaskProgram) -> dict[str, dict]:
    stage = np.asarray(ep.arrays["stage"])
    left, right = np.asarray(ep.arrays["left_mean_n"]), np.asarray(ep.arrays["right_mean_n"])
    valid = np.asarray(ep.arrays["valid_fraction"])
    out = {}
    for it in program.of_primitive("grasp"):
        m = (stage == it.label) & (valid > 0.5) & np.isfinite(left) & np.isfinite(right)
        tot = (left + right)[m]
        out[it.label] = {
            "n_valid": int(m.sum()),
            "total_force_n_mean": round(float(tot.mean()), 3) if tot.size else None,
            "total_force_n_max": round(float(tot.max()), 3) if tot.size else None,
        }
    return out


def run(assignment: dict[str, GripPlan], seed: int, keep_images: bool) -> dict:
    program = two_cube_program()
    errs = program.validate()
    if errs:
        raise SystemExit(f"invalid program: {errs}")
    grips = {f"{role}.pick_1.grasp": g for role, g in assignment.items()}
    ep = produce_program_episode(program, seed, plan_for(program, grips), keep_images=keep_images)
    idx = np.asarray(ep.arrays["stage_instance_index"])
    return {
        "seed": seed,
        "assignment": {k: [v.target_lo_n, v.target_hi_n] for k, v in assignment.items()},
        "status": ep.status,
        "n_steps": int(np.asarray(ep.arrays["actions"]).shape[0]),
        "n_grasp_instances": len(program.of_primitive("grasp")),
        "instance_index_monotonic": bool(np.all(np.diff(idx) >= 0)),
        "force_target_by_instance": ep.meta.get("force_target_by_instance"),
        "grasp_windows": grasp_windows(ep, program),
        "final_obj_pos_by_role": ep.meta.get("final_obj_pos_by_role"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    keep = not a.no_images
    recs = [
        run({"cubeA": GENTLE, "cubeB": FIRM}, a.seed, keep),
        run({"cubeA": FIRM, "cubeB": GENTLE}, a.seed, keep),
    ]
    for r in recs:
        print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(recs, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
