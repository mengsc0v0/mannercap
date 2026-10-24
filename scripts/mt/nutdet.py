import sys, json, hashlib, numpy as np
sys.path.insert(0, ".")
from synthesis_v2.native_osc_env import NativeSession, GRIP_OPEN
s = NativeSession("nut_assembly", 0, render=False)
obs = s.reset()
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
out = {"nut_id": int(s.env.nut_id), "nut_pos": np.round(obs["SquareNut_pos"], 6).tolist(),
       "qpos": h(s.env.sim.data.qpos), "qvel": h(s.env.sim.data.qvel)}
for _ in range(15):
    obs, _ = s.step(np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN], np.float32))
out["after_settle_qpos"] = h(s.env.sim.data.qpos)
out["initial"] = {k: h(np.asarray(v)) for k, v in s.initial.items()} if isinstance(s.initial, dict) else None
print("RES", json.dumps(out))
