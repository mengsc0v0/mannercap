import numpy as np, mujoco
from synthesis_v2.native_producer import StagePlan
from synthesis_v2 import wipe_producer as wp
from synthesis_v2.native_osc_env import NativeSession
plan=wp.WipePlan("wipe",(StagePlan("approach",0.3,1.0,250),StagePlan("descend",0.3,1.0,250),StagePlan("sweep",0.1,1.0,100),StagePlan("retreat",0.3,1.0,250)),wp.PressPlan(1.0))
s=NativeSession("wipe",42,render=False); obs=s.reset()
rn=wp._WipeRunner(s,plan,False); m=s.env.sim.model._model; d=s.env.sim.data._data
mk=wp._marker_world(s); off=rn.tool_offset(); q=np.asarray(obs["robot0_eef_quat"],float)
tool=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("gripper0")]
for i in tool:
    if m.geom_contype[i] or m.geom_conaffinity[i]:
        print("tool geom",mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),"fric",m.geom_friction[i].round(4),"size",m.geom_size[i].round(4))
tb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"table_collision"); print("table fric",m.geom_friction[tb])
rn.servo("approach",mk[0]+[0,0,0.06]-off,q,0.0); rn.servo("descend",mk[0]+[0,0,0.02]-off,q,0.0); rn.seek_contact("descend",q,200)
print("tool z at first contact",round(rn.tool_center()[2],4),"marker z",round(float(mk[0,2]),4))
for F in (5,10,20,30,40):
    rn.integ=0.0
    for k in range(60):  # settle force
        a=np.zeros(7,np.float32); a[2]=rn._az(F,rn._force(),rn.integ); rn.integ=np.clip(rn.integ+(F-rn._force())*0.05,-20,20); a[3:6]=rn._rot(q); rn._step(a,"sweep")
    p0=rn.tool_center().copy(); fz=[]
    for k in range(20):
        a=np.zeros(7,np.float32); a[0]=1.0; a[2]=rn._az(F,rn._force(),rn.integ); rn.integ=np.clip(rn.integ+(F-rn._force())*0.05,-20,20); a[3:6]=rn._rot(q); rn._step(a,"sweep"); fz.append(rn._force())
    p1=rn.tool_center(); print("F",F,"force during push",round(np.mean(fz),1),"tool_z %.4f"%p1[2],"dx over 20 steps %.4f m (%.3f m/s)"%(p1[0]-p0[0],(p1[0]-p0[0])/1.0))
    a=np.zeros(7,np.float32); a[0]=-1.0
    for k in range(20): a2=a.copy(); a2[2]=rn._az(F,rn._force(),rn.integ); a2[3:6]=rn._rot(q); rn._step(a2,"sweep")
