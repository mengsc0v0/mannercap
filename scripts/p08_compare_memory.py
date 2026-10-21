"""P1-8 equal-budget comparison: no memory vs command-median vs measured-response.

usage: p08_compare_memory.py WARM_RUN_DIR CAL.json OUT_DIR SEEDS [ARMS]
The memory is built once from WARM_RUN_DIR (audited attempts on warm-up seeds) and frozen.  Every arm
gets the same requests, the same initial states (seeds) and the same attempt budget.  Eval seeds must
be disjoint from the warm-up seeds and from the calibration seeds.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from synthesis_v2.manner_calibration import load_calibration
from synthesis_v2.production_pipeline import RequestSpec, RuleProposer, run_requests, summarize
from synthesis_v2.response_memory import CommandMedianProposer, MeasuredResponseProposer, ResponseMemory

BASE = {"lift": "Lift the cube", "stack": "Stack the red cube on the green cube"}
PH = {"neutral": "", "slow": " slowly", "fast": " quickly", "gentle": " gently", "firm": " firmly"}
BUDGET = 3


def main():
    warm, cal_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    seeds = [int(x) for x in sys.argv[4].split(",")]
    arms = sys.argv[5].split(",") if len(sys.argv) > 5 else ["none", "command_median", "measured"]
    cal = load_calibration(cal_path)
    mem = ResponseMemory.from_run(warm, cal["content_sha256"]).freeze()
    warm_seeds = {o.seed for o in mem.obs}
    assert not warm_seeds & set(seeds), f"eval seeds overlap warm-up seeds: {sorted(warm_seeds & set(seeds))}"
    specs = [RequestSpec(t, s, BASE[t] + PH[m] + ".", base_instruction=BASE[t] + ".", pair_key=f"{t}:{s}",
                         extra={"manner_name": m}) for t in ("lift", "stack") for s in seeds for m in PH]
    mk = {"none": RuleProposer, "command_median": lambda: CommandMedianProposer(mem),
          "measured": lambda: MeasuredResponseProposer(mem, online=False)}
    report = {"warm_run": str(warm), "n_memory_observations": len(mem.obs), "eval_seeds": seeds, "budget": BUDGET, "arms": {}}
    for arm in arms:
        d = out / arm
        res = run_requests(specs, cal, d, max_attempts=BUDGET, proposer=mk[arm](), log=lambda m: None)
        atts = [json.loads(x) for x in (d / "attempts.jsonl").read_text().splitlines()]
        first = {}
        for a in atts:
            first.setdefault(a["request_id"], a)
        fails = Counter()
        for a in atts:
            if a["outcome"] in ("validation_failure", "execution_failure"):
                fails[a["reason"].split(";")[0][:60]] += 1
        n = len(res)
        report["arms"][arm] = {
            "n_requests": n,
            "first_attempt_accept": sum(a["outcome"] == "accepted" for a in first.values()),
            "final_accept": sum(r.state == "accepted" for r in res),
            "mean_attempts": sum(r.n_attempts for r in res) / n,
            "physics_substeps": sum(a["resources"]["physics_substeps"] for a in atts),
            "outcomes": dict(Counter(a["outcome"] for a in atts)),
            "failure_reasons": dict(fails.most_common(8)),
            "unresolved": [(r.spec.task, r.spec.seed, r.spec.extra.get("manner_name")) for r in res if r.state != "accepted"],
        }
        print(arm, json.dumps(report["arms"][arm]), flush=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
