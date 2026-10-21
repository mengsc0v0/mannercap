"""Fixed-budget grid for the wipe teacher: speed x target force x seed (one process per seed)."""
import json, sys, time, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2.wipe_producer import WipePlan, PressPlan, produce_wipe_episode

seed = int(sys.argv[1]); out = sys.argv[2]
speeds = [float(x) for x in sys.argv[3].split(",")]; forces = [float(x) for x in sys.argv[4].split(",")]
BUDGET = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
for v in speeds:
    for F in forces:
        plan = WipePlan("wipe", (StagePlan("approach", 0.3, 1.0, 250), StagePlan("descend", 0.3, 1.0, 250),
                                 StagePlan("sweep", v, 1.0, BUDGET), StagePlan("retreat", 0.3, 1.0, 250)), PressPlan(F))
        t = time.time()
        ep = produce_wipe_episode("wipe", seed, plan, keep_images=False)
        a = ep.arrays
        rec = dict(seed=seed, v=v, F=F, status=ep.status, ok=ep.task_ok, steps=ep.n_steps, stage_steps=ep.stage_steps,
                   wall=round(time.time() - t, 1), meta_sweep=ep.meta.get("sweep"))
        if len(a.get("stage", [])):
            m = a["stage"] == "sweep"
            f = a["tool_mean_n"][m]
            sp = np.linalg.norm(a["next_state"][m, :3] - a["state"][m, :3], axis=1) / 0.05
            rec.update(f_med=float(np.median(f)), f_p10=float(np.percentile(f, 10)), f_p90=float(np.percentile(f, 90)),
                       f_max=float(a["tool_max_n"][m].max()), sp_med=float(np.median(sp)), sp_p90=float(np.percentile(sp, 90)))
        open(out, "a").write(json.dumps(rec) + "\n")
