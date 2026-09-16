"""Real Lift/Stack startup check; never treats legacy manner flags as v2 proof.

Run from any cwd with the isolated Linux environment. No LLM calls or outer
candidate retries. Transition records are at control frequency, not substeps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def audit_transitions(path):
    """Offline recomputation of observed timing and task status, not physics QA."""
    count = 0
    bad_time = 0
    gaps = 0
    previous = None
    elapsed = 0.0
    last_task = None
    integration_bad = 0
    integration_checked = 0
    transition_shape_bad = 0
    initial_file = path.parent / "initial_state.json"
    physics_dt = json.loads(initial_file.read_text())["physics_timestep_s"] if initial_file.exists() else None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            before, after = row["state_t"], row["state_t_plus_1"]
            required_vectors = (before.get("qpos"), before.get("qvel"),
                                after.get("qpos"), after.get("qvel"), row.get("action_t"))
            transition_shape_bad += int(any(
                not isinstance(vector, list) or not vector or
                any(not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) for value in vector)
                for vector in required_vectors
            ))
            dt = after["time_s"] - before["time_s"]
            finite_time = math.isfinite(before["time_s"]) and math.isfinite(after["time_s"])
            bad_time += int(not finite_time or dt <= 0)
            gaps += int(previous is not None and abs(previous - before["time_s"]) > 1e-9)
            previous = after["time_s"]
            if finite_time:
                elapsed += dt
            if physics_dt is not None:
                substeps = row.get("physics_substeps")
                valid_count = isinstance(substeps, int) and not isinstance(substeps, bool) and substeps > 0
                integration_checked += 1
                integration_bad += int(not valid_count or not finite_time or abs(dt - substeps * physics_dt) > 1e-8)
            last_task = row["task_success_after"]
            count += 1
    return {"control_transitions": count, "nonpositive_dt": bad_time,
            "time_discontinuities": gaps, "sim_elapsed_s": elapsed,
            "last_task_success": last_task, "physics_qualification": "unknown",
            "integration_check": ("unknown" if not integration_checked else "fail" if integration_bad else "pass"),
            "integration_mismatches": integration_bad,
            "invalid_transition_vectors": transition_shape_bad,
            "timing_check_passed": count > 0 and bad_time == 0 and gaps == 0}


def run_task(task, seed, out, ik_backend="pyroki"):
    target = out / task
    target.mkdir(exist_ok=False)
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = {"task": task, "seed": seed, "split": "development",
              "proposal_source": "fixed_unconstrained_legacy_baseline",
              "llm_calls": 0, "outer_candidate_attempts": 1,
              "manner_status": "unknown", "safety_status": "unknown",
              "jpr": None, "status": "started", "ik_backend": ik_backend}
    env = None
    physics_steps = [0]
    try:
        import numpy as np
        import run_manner_ablation as old
        from capx.envs.manner_validator import MannerSpec
        from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
        from synthesis_v2.mujoco_contacts import capture_object_contacts, resolve_task_object_registry
        from synthesis_v2.simulator_state import capture_simulator_snapshot

        np.random.seed(seed)
        entry = old.TASK_REGISTRY[task]
        env = old.load_sim_cls(entry["sim_cls_path"])(
            privileged=True, enable_render=False, max_steps=entry["max_steps"], seed=seed)
        env.reset(seed=seed)
        sim = env.robosuite_env.sim
        # Hard reset replaces the MjSim instance. Bind only after reset.
        # Setup integration precedes this hook and is explicitly uncounted.
        for step_name in ("step", "step2"):
            original_step = getattr(sim, step_name)

            def counted_step(_original=original_step):
                value = _original()
                physics_steps[0] += 1
                return value

            setattr(sim, step_name, counted_step)
        result["reset_physics_substeps"] = None
        result["constructor_physics_substeps"] = None

        def state():
            d = sim.data
            return {"time_s": float(d.time), "qpos": np.asarray(d.qpos).tolist(),
                    "qvel": np.asarray(d.qvel).tolist(), "actuator_ctrl": np.asarray(d.ctrl).tolist()}

        xml = sim.model.get_xml()
        object_registry = resolve_task_object_registry(env.robosuite_env, task)
        (target / "model.xml").write_text(xml, encoding="utf-8")
        write_json(target / "initial_state.json", {
            "state": state(), "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "physics_timestep_s": float(sim.model.opt.timestep),
            "object_registry": object_registry,
            "snapshot": capture_simulator_snapshot(env),
            "restoration_verified": False,
            "limitation": "Controller caches are serialized; action replay must still validate restoration."})
        params = old.spec_to_exec_params(MannerSpec(label="gate0_unconstrained"))
        write_json(target / "executed_parameters.json", params)
        original = env._do_robosuite_step
        with (target / "transitions.jsonl").open("x", encoding="utf-8") as stream:
            def record(action):
                before = state()
                substeps_before = physics_steps[0]
                original(action)
                contact_evidence = capture_object_contacts(sim, object_registry)
                row = {"state_t": before, "action_t": np.asarray(action).tolist(),
                       "action_slice": env._ACTION_SLICE,
                       "phase_program_marker": env._manner_phase,
                       "physics_substeps": physics_steps[0] - substeps_before,
                       "state_t_plus_1": state(),
                       "contact_evidence_t_plus_1": contact_evidence,
                       "task_success_after": bool(env.task_completed())}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            env._do_robosuite_step = record
            api = FrankaControlPrivilegedApi(env)
            if ik_backend == "mujoco-local":
                from gate0_local_ik import install_local_ik
                result["ik_diagnostics"] = []
                install_local_ik(api, env, result["ik_diagnostics"])
            result["execution"] = entry["runner"](env, api, params)
        result["status"] = "executed"
    except Exception:
        result["status"] = "execution_error"
        result["error"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.perf_counter() - started
        result["process_cpu_s"] = time.process_time() - cpu_started
        result["observed_execution_physics_substeps"] = physics_steps[0]
        if (target / "transitions.jsonl").exists():
            result["offline_recomputation"] = audit_transitions(target / "transitions.jsonl")
        result["artifact_bytes_before_result"] = sum(p.stat().st_size for p in target.iterdir() if p.is_file())
        write_json(target / "result.json", result)
        if env is not None:
            env.robosuite_env.close()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return (result["status"] == "executed"
            and result["offline_recomputation"]["timing_check_passed"]
            and result["offline_recomputation"]["integration_check"] == "pass"
            and result["offline_recomputation"]["invalid_transition_vectors"] == 0
            and result["execution"]["task_completed"] is True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--tasks", nargs="+", choices=["lift", "stack"], default=["lift", "stack"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ik-backend", choices=["pyroki", "mujoco-local"], default="pyroki")
    p.add_argument("--audit", type=Path, help="Only recompute a transitions.jsonl file; no simulation imports")
    args = p.parse_args()
    if args.audit:
        print(json.dumps(audit_transitions(args.audit), indent=2))
        return 0
    if args.output is None:
        p.error("--output is required for a new run")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "stagecap_robosuite_libero_repro_20260714"))
    os.chdir(ROOT / "stagecap_robosuite_libero_repro_20260714")
    write_json(out / "run_manifest.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "python": sys.version, "tasks": args.tasks, "seed": args.seed, "ik_backend": args.ik_backend,
        "stage": "Gate 0 development only", "api_budget": 0,
        "limitations": ["No calibrated force", "No images", "No replay qualification",
                        "Program phases are not physical-event gold labels",
                        "One candidate per task; legacy within-candidate grasp controller retries remain"]})
    results = [run_task(task, args.seed, out, args.ik_backend) for task in args.tasks]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
