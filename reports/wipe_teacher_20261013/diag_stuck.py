import sys, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
v=float(sys.argv[1]); F=float(sys.argv[2]); seed=int(sys.argv[3])
plan=wp.WipePlan("wipe",(StagePlan("approach",0.3,1.0,250),StagePlan("descend",0.3,1.0,250),StagePlan("sweep",v,1.0,1000),StagePlan("retreat",0.3,1.0,250)),wp.PressPlan(F))
log=[]; orig=wp._WipeRunner._step
def patched(self, action, stage):
    orig(self, action, stage)
    if stage!="sweep": return
    env=self.s.env; r=env.robots[0]; sim=env.sim
    qpos=np.asarray(sim.data.qpos[r._ref_joint_pos_indexes]); lo=r.sim.model.jnt_range[r._ref_joint_indexes][:,0]; hi=r.sim.model.jnt_range[r._ref_joint_indexes][:,1]
    marg=np.minimum(qpos-lo,hi-qpos)
    log.append((*self.tool_center(), action[0],action[1],action[2], self._force(), marg.min(), int(marg.argmin()), len(env.wiped_markers)))
wp._WipeRunner._step=patched
ep=wp.produce_wipe_episode("wipe",seed,plan,keep_images=False)
L=np.array(log); print("ok",ep.task_ok,"n",len(L),"wiped",ep.meta.get("final_wiped"),ep.meta.get("sweep"))
for k in range(0,len(L),max(1,len(L)//30)):
    r=L[k]; print(k,"tool %.3f %.3f %.4f act %.2f %.2f %.2f F %.1f jmargin %.3f j%d wiped %d"%tuple(r))
