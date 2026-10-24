import json, sys
sys.path.insert(0, "scripts/mt"); sys.path.insert(0, ".")
from s0_baseline_snapshot import run
from synthesis_v2.manner_request import default_candidate
base = {(r["task"], r["variant"], r["seed"]): r for r in json.load(open("reports/baseline_20261023/episode_digests.json"))}
for task in sys.argv[1:]:
    r = run(task, 0, default_candidate(task, ()), keep_images=False)
    b = base[(task, "neutral", 0)]
    print(task, r["status"], r["n_steps"], "EQUAL" if r["digest"] == b["digest"] else "DIFF",
          [k for k in b["digest"] if b["digest"][k] != r["digest"].get(k)], flush=True)
