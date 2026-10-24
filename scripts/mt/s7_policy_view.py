"""What the grasp policy saw/sent per call (dev, read-only). Usage: s7_policy_view.py TASK INIT ROLE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import make_program, make_plan, make_session
from synthesis_v2.program_executor import produce_program_episode
T, I, ROLE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
prog = make_program(T); s = make_session(T, I, prog)
ep = produce_program_episode(prog, I, make_plan(T, prog, approach_check=True), keep_images=False, session=s)
pol = ep.meta["grasp_policy_by_instance"][f"{ROLE}.pick_1.grasp"]
print(f"V T{T} i{I} {ROLE} ok={ep.task_ok} reverse={pol['reverse_pulses']}")
for k, r in enumerate(pol["trace"]):
    print(f"V   call{k}: saw L={r['left_n']:5.1f} R={r['right_n']:5.1f} total={r['total_n']:5.1f} -> cmd {r['cmd']:+.0f}")
