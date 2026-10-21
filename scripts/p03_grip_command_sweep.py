"""P0-3 step 1: same init / same motion, sweep ONLY the normalized grip command
and measure target-object per-finger force (P0-1 channel) + hold stability.

Run from the nested capx dir:  PYTHONPATH=$PWD/.. python ../scripts/p03_grip_command_sweep.py
"""
import argparse, json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, ".."))
sys.path.insert(0, os.path.join(BASE, "..", "stagecap_robosuite_libero_repro_20260714"))
import run_manner_ablation as old  # noqa: E402


def phase_stat(tele, phases, fn):
    vals = [s.grasp_left_force_n + s.grasp_right_force_n
            for s in tele if s.phase in phases and s.grasp_force_valid]
    return float(fn(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lift", choices=["lift", "stack"])
    ap.add_argument("--grips", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="/tmp/p03_grip_sweep.jsonl")
    a = ap.parse_args()

    entry = old.TASK_REGISTRY[a.task]
    env = old.load_sim_cls(entry["sim_cls_path"])(
        privileged=True, enable_render=False, max_steps=entry["max_steps"], seed=a.seeds[0])
    from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
    api = FrankaControlPrivilegedApi(env)
    rows = []
    for seed in a.seeds:
        for g in a.grips:
            env.reset(seed=seed)
            env.begin_manner_recording(clear=True)
            params = dict(v_max=0.15, a_max=0.4, jerk_limit=None, kp_pos=1000.0,
                          grip=g, grasp_tries=1, phase_v_max={}, phase_a_max={})
            err = None
            try:
                res = entry["runner"](env, api, params)
            except Exception as e:  # noqa: BLE001
                res, err = {}, repr(e)[:200]
            env.end_manner_recording()
            tele = env.get_manner_telemetry()
            grip_act = 2.0 * g - 1.0
            row = dict(task=a.task, seed=seed, grip_cmd=g, gripper_action=grip_act,
                       task_ok=bool(res.get("task_completed", False)),
                       cube_dz=res.get("cube_delta_z"), error=err, n=len(tele),
                       valid_frac=float(np.mean([s.grasp_force_valid for s in tele])) if tele else 0.0,
                       peak_total_n=phase_stat(tele, {"grip", "stiff", "lift", "transport", "final", "other", "descend"}, np.max),
                       hold_median_n=phase_stat(tele, {"lift", "transport", "final"}, np.median),
                       hold_min_n=phase_stat(tele, {"lift", "transport", "final"}, np.min),
                       finger_sep_end=float(abs(tele[-1].gripper_qpos[0] - tele[-1].gripper_qpos[1])) if tele else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
