import sys, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
v=float(sys.argv[1]); F=float(sys.argv[2]); seed=int(sys.argv[3])
plan=wp.WipePlan("wipe",(StagePlan("approach",0.3,1.0,250),StagePlan("descend",0.3,1.0,250),StagePlan("sweep",v,1.0,1000),StagePlan("retreat",0.3,1.0,250)),wp.PressPlan(F))
from synthesis_v2.native_osc_env import NativeSession
s=NativeSession("wipe",seed,render=False)
ep=wp.produce_wipe_episode("wipe",seed,plan,keep_images=False,session=s)
mk=wp._marker_world(s); rem=wp._remaining_idx(s)
print("status",ep.status,"wiped",ep.meta["final_wiped"],ep.meta["sweep"])
print("remaining idx",rem)
print("remaining xy",np.round(mk[rem,:2],3).tolist()[:12])
print("table half sizes", s.env.table_full_size if hasattr(s.env,'table_full_size') else None, "bbox all", mk[:,:2].min(0).round(3), mk[:,:2].max(0).round(3))
a=ep.arrays; m=a["stage"]=="sweep"; st=a["state"][m,:2]
print("tool-eef xy range in sweep", st.min(0).round(3), st.max(0).round(3))
