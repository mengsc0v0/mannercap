"""E4 supplement: execute EVERY frozen candidate on every trial init, so that the outcome of ANY candidate order can be
computed exactly without re-simulating.  This separates "retrieval found the right parameter" from "the fixed baseline
order happened to point the wrong way".  Usage: s10_candidate_grid.py TASK INIT...
-> reports/step10_experience/grid/t{T}_init{I}.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT  # noqa: E402
import s10_retrieval_trial as R  # noqa: E402

OUT = ROOT / "reports" / "step10_experience" / "grid"


def main() -> int:
    task = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for i in [int(x) for x in sys.argv[2:]]:
        f = OUT / f"t{task}_init{i}.json"
        have = json.loads(f.read_text())["episodes"] if f.exists() else {}
        for cand in R.CAND:
            k = f"{cand[0]}_{cand[1]}"
            if k in have:
                continue
            have[k] = R.episode(task, i, cand)
            print(f"T{task} i{i} cand={cand} hold={have[k]['measured']['hold_mean']} "
                  f"stable={have[k]['validity']['grasp_stable']} task_ok={have[k]['task_ok']}", flush=True)
        f.write_text(json.dumps({"task": task, "init": i, "e1_hash": R.E1["content_hash"], "episodes": have}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
