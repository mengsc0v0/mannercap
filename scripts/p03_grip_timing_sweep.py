"""P0-3 step 2: 7D-compatible gripper timing. After descend, close (action +1)
for k control steps, then HOLD (action 0 -> sign 0 -> target frozen), then lift.
Measures per-finger target-object force vs k, lift success, slip.
"""
import json, os, sys
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, ".."))
sys.path.insert(0, os.path.join(BASE, "..", "stagecap_robosuite_libero_repro_20260714"))
import run_manner_ablation as old  # noqa: E402

def main():
    ks = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2,4,5,6,7,8,9,10,12,15]
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42]
    entry = old.TASK_REGISTRY["lift"]
    env = old.load_sim_cls(entry["sim_cls_path"])(privileged=True, enable_render=False,
                                                  max_steps=entry["max_steps"], seed=seeds[0])
    from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
    api = FrankaControlPrivilegedApi(env)
    quat = np.array([0.0, 0.0, 1.0, 0.0])
    out = []
    for seed in seeds:
        for k in ks:
            env.reset(seed=seed)
            env.begin_manner_recording(clear=True)
            cube, _, _ = api.get_object_pose("red cube")
            ap = cube.copy(); ap[2] += 0.20
            api.goto_pose_with_profile(ap, quat, v_max=0.15, a_max=0.4, jerk_limit=None, phase="approach")
            old._pump(env, 8)
            api.set_impedance_stiffness(kp_pos=1000.0)
            gp = cube.copy(); gp[2] -= 0.005
            api.goto_pose_with_profile(gp, quat, v_max=0.09, a_max=0.2, jerk_limit=None, phase="descend")
            old._pump(env, 8)
            env._manner_phase = "grip"
            env._set_gripper(0.0)            # action +1 : close
            for _ in range(k):
                env._step_once()
            env._set_gripper(0.5)            # action 0 : hold target
            for _ in range(40):
                env._step_once()
            gs = env._read_grasp_sample()
            hold_pre = (gs.left_force_n, gs.right_force_n)
            env._manner_phase = "other"
            api.set_impedance_stiffness(kp_pos=1000.0)
            lp = gp.copy(); lp[2] += 0.18
            api.goto_pose_with_profile(lp, quat, v_max=0.15, a_max=0.4, jerk_limit=None, phase="lift")
            old._pump(env, 10)
            c2, _, _ = api.get_object_pose("red cube")
            env.end_manner_recording()
            tele = env.get_manner_telemetry()
            lift = [s.grasp_left_force_n + s.grasp_right_force_n for s in tele if s.phase == "lift" and s.grasp_force_valid]
            row = dict(seed=seed, k=k, hold_L=round(hold_pre[0],2), hold_R=round(hold_pre[1],2),
                       lift_med_total=round(float(np.median(lift)),2) if lift else None,
                       lift_min_total=round(float(np.min(lift)),2) if lift else None,
                       cube_dz=round(float(c2[2]-cube[2]),4), ok=bool(env.task_completed()))
            print(json.dumps(row), flush=True)
            out.append(row)
    json.dump(out, open("/tmp/p03_timing.json", "w"))

main()
