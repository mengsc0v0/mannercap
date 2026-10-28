import sys, json, numpy as np
sys.path.insert(0,"scripts/mt"); sys.path.insert(0,".")
from s7_task_config import make_program, make_session
from synthesis_v2.program_executor import _Geometry, _subtask_roles, APPROACH_DZ
from synthesis_v2.approach_planner import plan_approach, scene_from_libero, DEFAULT_YAWS_DEG
lat=(0.0,0.03,-0.03,0.04,-0.04,0.05,-0.05,0.06,-0.06)
dz=(0.0,0.02,0.04,0.06,0.08,0.10)
grid=[(y,l,z,0.0) for y in DEFAULT_YAWS_DEG for l in lat for z in dz]
out={}
for init in [int(sys.argv[1])]:
    prog=make_program(4); s=make_session(4,init,prog); obs=s.reset()
    meta={"down_quat_xyzw":np.asarray(obs["robot0_eef_quat"],float).tolist(),"subtasks":{}}
    geo=_Geometry(s,prog,True,meta)
    for sub in ("white_mug.pick_1","yellow_mug.pick_1"):
        rec=geo.capture(sub,*_subtask_roles(prog,sub))
        nm=[i for i in prog.instances if i.subtask==sub][0].object_role; nm=prog.objects[nm]
        m,d,site,env,ob,tg=scene_from_libero(s,nm,settled=True)
        pl=plan_approach(m,d,site,env,ob,start_pos=np.asarray(s.obs["robot0_eef_pos"],float),start_quat_xyzw=np.asarray(s.obs["robot0_eef_quat"],float),
            grasp_pos=np.asarray(rec["obj_pos0"],float),grasp_quat_xyzw=np.asarray(rec["grasp_quat_xyzw"],float),hover_dz=APPROACH_DZ,target=tg,grid=grid,evaluate_all=True)
        f=[c for c in pl.candidates if c.feasible]
        out[f"{init}:{sub}"]=[(c.yaw_deg,c.lateral_m,c.dz_m,round(c.min_clearance_m,4),round(c.at_grasp_clearance_m,4)) for c in f]
        best=max(f,key=lambda c:c.at_grasp_clearance_m) if f else None
        print(init,sub,"n_cand",len(pl.candidates),"feasible",len(f),"best_at_grasp",best and (best.yaw_deg,best.lateral_m,best.dz_m,round(best.at_grasp_clearance_m,4)),flush=True)
    s.close()
json.dump(out,open(f"reports/step9_t4_onboard/rim_grid_init{sys.argv[1]}.json","w"))
