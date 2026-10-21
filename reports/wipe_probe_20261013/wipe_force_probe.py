import os, sys, numpy as np
ROOT="/workdata/stage_cap_privileged_data_generation_repro_20260629/robosuite/stagecap_robosuite_libero_repro_20260714/new/stagecap_robosuite_libero_repro_20260714"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel

env = FrankaRobosuiteSpillWipeLowLevel(privileged=True, enable_render=False, max_steps=6000, seed=42)
env.reset(seed=42)
rs = env.robosuite_env
sim = rs.sim
model = sim.model._model
scratch = mujoco.MjData(model)
joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1,8)]
qadr = model.jnt_qposadr[joints]
lower, upper = model.jnt_range[joints].T
eef = env.gripper_link_idx; base = env.base_link_idx
arm = rs.robots[0].arms[0]
corner_names = rs.robots[0].gripper[arm].important_geoms["corners"]
corner_ids = np.asarray([sim.model.geom_name2id(n) for n in corner_names], dtype=int)
print("corners:", corner_names)

def goto_tool_center(position, quat_wxyz, v_max=0.12, a_max=0.10, phase="wipe"):
    q_before = sim.data.qpos.copy()
    scratch.qpos[:] = q_before; scratch.qvel[:] = sim.data.qvel.copy()
    base_rot = Rotation.from_matrix(sim.data.xmat[base].reshape(3,3))
    tgt = sim.data.xpos[base] + base_rot.apply(np.asarray(position,float).reshape(3))
    quat = np.asarray(quat_wxyz,float).reshape(4)
    tgt_rot = base_rot * Rotation.from_quat(quat[[1,2,3,0]])
    q0 = np.clip(q_before[qadr], lower+1e-6, upper-1e-6)
    def residual(q):
        scratch.qpos[qadr]=q; mujoco.mj_forward(model, scratch)
        center = scratch.geom_xpos[corner_ids].mean(axis=0)
        rerr = (tgt_rot * Rotation.from_matrix(scratch.xmat[eef].reshape(3,3)).inv()).as_rotvec()
        return np.r_[center-tgt, 0.2*rerr, 1e-4*(q-q0)]
    sol = least_squares(residual, q0, bounds=(lower+1e-6,upper-1e-6), max_nfev=300, ftol=1e-10,xtol=1e-10,gtol=1e-10)
    perr = float(np.linalg.norm(residual(sol.x)[:3]))
    env.move_to_joints_profiled(sol.x, v_max=v_max, a_max=a_max, phase=phase)
    return perr

def markers_base(remaining_only=True):
    markers = list(rs.model.mujoco_arena.markers)
    if remaining_only:
        wiped={id(m) for m in rs.wiped_markers}; markers=[m for m in markers if id(m) not in wiped]
    if not markers: return np.zeros((0,3))
    world=np.stack([sim.data.body_xpos[sim.model.body_name2id(m.root_body)].copy() for m in markers])
    b=np.asarray(env.base_link_wxyz_xyz,float); qw,qx,qy,qz=b[:4]; t=b[4:7]
    R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
    return (world-t)@R


down=np.array([0.0,0.0,1.0,0.0])
corner_set=set(int(i) for i in corner_ids)
gripper_geoms=set(i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("gripper0"))
def tool_force(detail=False):
    buf=np.zeros(6); tot=0.0; n=0; pairs={}
    d=sim.data._data
    for ci in range(d.ncon):
        c=d.contact[ci]; g1,g2=int(c.geom1),int(c.geom2)
        if (g1 in gripper_geoms) != (g2 in gripper_geoms):
            mujoco.mj_contactForce(model,d,ci,buf); f=abs(float(buf[0])); tot+=f; n+=1
            o=g2 if g1 in gripper_geoms else g1; t=g1 if g1 in gripper_geoms else g2
            key=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,t),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,o))
            pairs[key]=pairs.get(key,0.0)+f
    return tot,n,pairs
mk=markers_base(); tgt=mk[0].copy()
print("rest force before:", tool_force()[:2])
goto_tool_center(tgt+[0,0,0.08], down, 0.12,0.10,"approach")
print("above table:", tool_force()[:2])
import sys
PEN=float(sys.argv[1]) if len(sys.argv)>1 else 0.005
goto_tool_center(tgt-[0,0,PEN], down, 0.05,0.05,"descend")
fs=[]
for k in range(12):
    env._step_once(); fs.append(tool_force())
print("PEN",PEN,"force per step:",[round(f[0],2) for f in fs],"ncon",[f[1] for f in fs][:3])
print("pairs:",{k:round(v,2) for k,v in fs[-1][2].items()})
print("ee_force",np.round(rs.robots[0].ee_force[arm],3) if hasattr(rs.robots[0],"ee_force") else None)
print("wiped",len(rs.wiped_markers),"done",env.task_completed())
