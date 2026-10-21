"""Log the unwiped markers at the start of each extra pass (tool position, xy distance, height above the dirt centre)."""
import sys, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
from synthesis_v2.native_osc_env import NativeSession
v = float(sys.argv[1]); F = float(sys.argv[2]); seed = int(sys.argv[3])
plan = wp.WipePlan("wipe", (StagePlan("approach", 0.3, 1.0, 250), StagePlan("descend", 0.3, 1.0, 250),
                            StagePlan("sweep", v, 1.0, 1000), StagePlan("retreat", 0.3, 1.0, 250)), wp.PressPlan(F))
s = NativeSession("wipe", seed, render=False)
orig = wp._remaining_idx
def spy(sess):
    idx = orig(sess)
    if idx:
        mk = wp._marker_world(sess)
        tc = sess.env.sim.data.geom_xpos[[g for g in range(0)]] if False else None
        print("remaining", idx, "xy", np.round(mk[idx, :2], 3).tolist(), "z", np.round(mk[idx, 2], 4).tolist(), flush=True)
    return idx
wp._remaining_idx = spy
ep = wp.produce_wipe_episode("wipe", seed, plan, keep_images=False, session=s)
print("status", ep.status, ep.meta["sweep"])
