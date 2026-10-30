"""Step 15: does the OFFICIAL success predicate still hold after the book has been released and has come to rest?
Replays official T5 demos; after the demo's last action: open the gripper (if the demo has not) and hold still for 120 control
steps; log env success and the book pose (origin, tilt) at the demo end and at rest.  usage: s15_demo_rest_check.py DEMO_FILE OUT DEMO_INDEX..."""
import json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.append(os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO")); os.environ.setdefault("MUJOCO_GL", "egl")
import h5py, mujoco  # noqa: E402
from libero import benchmark  # noqa: E402
from libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.utils import get_libero_path  # noqa: E402
FILE, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
task = benchmark.get_benchmark_dict()["libero_10"]().get_task(5)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file), camera_heights=128, camera_widths=128,
                         controller="OSC_POSE", control_freq=20)
BOOK, SITE = "black_book_1", "desk_caddy_1_back_contain_region"
res = {}
with h5py.File(FILE, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for di in [int(x) for x in sys.argv[3:]]:
        g = f["data"][keys[di]]; acts, states = g["actions"][:], g["states"][:]
        env.reset(); obs = env.set_init_state(states[0]); sim = env.env.sim
        for a in acts: obs, *_ = env.step(a)
        def snap(tag):
            q = np.asarray(obs[BOOK + "_quat"]); e = R.from_quat(q).as_euler("xyz", degrees=True)
            return {"tag": tag, "env_success": bool(env.env._check_success()), "book_origin": np.round(obs[BOOK + "_pos"], 4).tolist(), "book_rpy_deg": np.round(e, 1).tolist()}
        rows = [snap("demo_end")]
        open_a = np.zeros(7); open_a[-1] = -1.0
        for k in range(120):
            obs, *_ = env.step(open_a)
            if k in (9, 39, 79, 119): rows.append(snap(f"after_open_{k+1}"))
        # the official predicate's bounds for this site, to compare with the resting origin
        sid = sim.model.site_name2id(SITE); sp = sim.data.site_xpos[sid]; sz = env.env.object_sites_dict[SITE].size
        rows.append({"in_box_lower_z": float(sp[2] - abs(sz[2]) - 0.01), "in_box_upper_z": float(sp[2] + abs(sz[2]))})
        res[keys[di]] = {"opens_in_demo": bool((acts[:, -1] <= 0).any() and (acts[:, -1] > 0).any()), "rows": rows}
        print("RC", keys[di], json.dumps(rows)[:900], flush=True)
json.dump(res, open(os.path.join(OUT, "demo_rest_check.json"), "w"), indent=1)
