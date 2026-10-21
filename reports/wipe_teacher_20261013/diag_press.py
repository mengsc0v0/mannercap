import sys, numpy as np
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
from synthesis_v2.native_osc_env import NativeSession
plan=wp.WipePlan("wipe",(StagePlan("approach",0.3,1.0,250),StagePlan("descend",0.3,1.0,250),StagePlan("sweep",0.1,1.0,100),StagePlan("retreat",0.3,1.0,250)),wp.PressPlan(1.0))
s=NativeSession("wipe",42,render=False); obs=s.reset()
rn=wp._WipeRunner(s,plan,False)
mk=wp._marker_world(s); off=rn.tool_offset(); q=np.asarray(obs["robot0_eef_quat"],float)
rn.servo("approach",mk[0]+[0,0,0.06]-off,q,0.0)
rn.servo("descend",mk[0]+[0,0,0.02]-off,q,0.0)
print("contact",rn.seek_contact("descend",q,200),"tool_z",rn.tool_center()[2])
for az in (-0.3,-0.5,-0.7,-1.0):
    Fs=[];Z=[];N=[]
    for k in range(40):
        a=np.zeros(7,np.float32); a[2]=az; a[3:6]=rn._rot(q); rn._step(a,"sweep")
        Fs.append(rn._force()); Z.append(rn.tool_center()[2])
        c=s.env.sim.data.geom_xpos[rn._corner_ids]; 
    near=[i for i in range(100) if np.linalg.norm(mk[i,:2]-rn.tool_center()[:2])<0.02]
    print("az",az,"F",round(np.mean(Fs[-10:]),1),"tool_z %.4f"%np.mean(Z[-10:]),"wiped",len(s.env.wiped_markers),"table",round(float(mk[:,2].mean()),4))
