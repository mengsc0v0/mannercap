import os, sys, json, time, numpy as np
V=float(sys.argv[1]); PEN=float(sys.argv[2]); SEED=int(sys.argv[3]); OUT=sys.argv[4]
ROOT="/workdata/stage_cap_privileged_data_generation_repro_20260629/robosuite/stagecap_robosuite_libero_repro_20260714/new/stagecap_robosuite_libero_repro_20260714"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel

env = FrankaRobosuiteSpillWipeLowLevel(privileged=True, enable_render=False, max_steps=6000, seed=SEED)
env.reset(seed=SEED)
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
gripper_geoms=set(i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("gripper0"))
buf=np.zeros(6)
def tool_force():
    d=sim.data._data; tot=0.0
    for ci in range(d.ncon):
        c=d.contact[ci]
        if (int(c.geom1) in gripper_geoms)!=(int(c.geom2) in gripper_geoms):
            mujoco.mj_contactForce(model,d,ci,buf); tot+=abs(float(buf[0]))
    return tot
log=dict(f=[],phase=[])
orig=env._step_once
def wrapped(*a,**k):
    r=orig(*a,**k)
    log["f"].append(tool_force()); log["phase"].append(getattr(env,"_manner_phase",""))
    if "trig" not in log:
        c=bool(rs.check_contact(rs.robots[0].robot_model)); q=bool(rs.robots[0].check_q_limits())
        if c or q:
            qp=sim.data.qpos[qadr]; marg=np.minimum(qp-lower,upper-qp)
            log["trig"]=dict(step=len(log["f"]),contact=c,qlimit=q,wiped=len(rs.wiped_markers),closest_joint=int(np.argmin(marg))+1,margin=float(marg.min()),force=log["f"][-1])
    return r
env._step_once=wrapped
def nn(pts,start):
    rem=list(range(len(pts))); order=[]; cur=np.asarray(start)
    while rem:
        d=[np.linalg.norm(pts[i][:2]-cur) for i in rem]; k=rem[int(np.argmin(d))]
        order.append(k); cur=pts[k][:2]; rem.remove(k)
    return pts[order]
t0=time.time(); status="ok"; err=""
mk=markers_base(); tgt=mk[0].copy()
try:
    goto_tool_center(tgt+[0,0,0.08], down, 0.12,0.10,"approach")
    goto_tool_center(tgt-[0,0,PEN], down, 0.05,0.05,"descend")
    for it in range(6):
        rem=markers_base()
        if len(rem)==0 or env.task_completed(): break
        path=nn(rem, markers_base(False)[0][:2])
        for w in path:
            t2=w.copy(); t2[2]-=PEN
            goto_tool_center(t2, down, V, V*0.75, "wipe")
            if env.task_completed(): break
        if env.task_completed(): break
except Exception as e:
    status="terminated:"+type(e).__name__; err=str(e)[:80]
f=np.array(log["f"]); ph=np.array(log["phase"])
w=f[ph=="wipe"] if (ph=="wipe").any() else f
rec=dict(v=V,pen=PEN,seed=SEED,wiped=len(rs.wiped_markers),total=int(rs.num_markers),completed=bool(env.task_completed()),
         status=status,err=err,steps=len(f),wipe_steps=int((ph=="wipe").sum()),
         force_med=float(np.median(w)) if len(w) else None,force_p10=float(np.percentile(w,10)) if len(w) else None,
         force_p90=float(np.percentile(w,90)) if len(w) else None,force_max=float(w.max()) if len(w) else None,
         wall_s=round(time.time()-t0,1),trig=log.get("trig"))
open(OUT,"a").write(json.dumps(rec)+"\n"); print(json.dumps(rec))
