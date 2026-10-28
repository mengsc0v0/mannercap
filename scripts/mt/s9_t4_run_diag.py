import sys, json, numpy as np
sys.path.insert(0,"scripts/mt"); sys.path.insert(0,".")
from s7_task_config import make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode
i=int(sys.argv[1])
prog=make_program(4); plan=make_plan(4,prog,approach_check=True); s=make_session(4,i,prog)
o0=s.reset()
ks=["porcelain_mug_1_pos","white_yellow_mug_1_pos","plate_1_pos","plate_2_pos"]
init={k:np.asarray(o0[k]).round(3).tolist() for k in ks if k in o0}
ep=produce_program_episode(prog,i,plan,keep_images=False,session=s)
o=s.obs
fin={k:np.asarray(o[k]).round(3).tolist() for k in ks if k in o}
gp=ep.meta.get("grasp_policy_by_instance")
print("RES",i,json.dumps({"status":ep.status,"ok":bool(ep.task_ok),"init":init,"final":fin,"grasp":gp},default=str)[:1800])
