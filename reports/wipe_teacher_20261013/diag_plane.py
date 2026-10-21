import sys, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
v=float(sys.argv[1]); F=float(sys.argv[2]); seed=int(sys.argv[3])
plan=wp.WipePlan("wipe",(StagePlan("approach",0.3,1.0,250),StagePlan("descend",0.3,1.0,250),StagePlan("sweep",v,1.0,1000),StagePlan("retreat",0.3,1.0,250)),wp.PressPlan(F))
log=[]
orig=wp._WipeRunner._step
def patched(self, action, stage):
    orig(self, action, stage)
    if stage!="sweep": return
    sim=self.s.env.sim; c=sim.data.geom_xpos[self._corner_ids]
    v1=c[0]-c[1]; v1/=np.linalg.norm(v1); v2=c[3]-c[1]; v2/=np.linalg.norm(v2); n=np.cross(v1,v2); n/=np.linalg.norm(n)
    mk=wp._marker_world(self.s); rem=wp._remaining_idx(self.s)
    tc=c.mean(0)
    near=[i for i in rem if np.linalg.norm(mk[i,:2]-tc[:2])<0.02]
    d=[float(np.dot(mk[i]-c[1],n)) for i in near]
    log.append((n[2], tc[2], len(near), min(d) if d else np.nan, max(d) if d else np.nan, self._force(), len(rem)))
wp._WipeRunner._step=patched
ep=wp.produce_wipe_episode("wipe",seed,plan,keep_images=False)
L=np.array(log)
print("ok",ep.task_ok,"steps",len(L),"final wiped",ep.meta.get("final_wiped"))
for k in range(0,len(L),max(1,len(L)//25)):
    r=L[k]; print(k,"nz %.4f tool_z %.4f near %d dmin %.4f dmax %.4f F %.1f rem %d"%tuple(r))
corn=wp.np
