import json, collections
paths={"prod_v2":"/tmp/prod_v2/attempts.jsonl","prod_v3":"/tmp/prod_v3/attempts.jsonl",
"p08b/none":"/tmp/p08b/none/none/attempts.jsonl","p08b/command_median":"/tmp/p08b/command_median/command_median/attempts.jsonl","p08b/measured":"/tmp/p08b/measured/measured/attempts.jsonl"}
for name,p in paths.items():
    rows=[json.loads(l) for l in open(p)]
    reqs=len({r["request_id"] for r in rows})
    n=len(rows); retry=[r for r in rows if r["attempt_number"]>1]
    res=[r for r in rows if r["metadata"].get("resumed_from_stage")]
    ns=sum(r["metadata"].get("n_steps",0) for r in rows); pref=sum(r["metadata"].get("prefix_steps_reused",0) for r in rows)
    led=sum(r["resources"]["physics_substeps"] for r in rows)
    bad=[r for r in rows if r["metadata"].get("n_steps") is not None and r["resources"]["physics_substeps"]!=(r["metadata"]["n_steps"]-r["metadata"].get("prefix_steps_reused",0))*25]
    out=collections.Counter(r["outcome"] for r in rows)
    st=collections.Counter(r["metadata"].get("resumed_from_stage") for r in res)
    tasks=collections.Counter(r["request_snapshot"]["task"] for r in rows)
    print(f"{name}: requests {reqs} attempts {n} retries(attempt>1) {len(retry)} resumed {len(res)} outcomes {dict(out)} tasks {dict(tasks)}")
    print(f"   sum n_steps {ns} sum prefix_reused {pref} ({pref/ns:.2%} of steps); ledger physics_substeps {led} (= {led/25:.0f} steps); full-rerun equivalent {ns*25}; ledger!=(n-prefix)*25 in {len(bad)} rows; resumed stages {dict(st)}")
    # retry attempts only
    rn=sum(r["metadata"].get("n_steps",0) for r in retry); rp=sum(r["metadata"].get("prefix_steps_reused",0) for r in retry)
    if retry: print(f"   retry attempts only: steps {rn}, prefix reused {rp} ({rp/rn:.1%}); retries not resumed: {sum(1 for r in retry if not r['metadata'].get('resumed_from_stage'))}")
    ledger_total_wall=sum(r["resources"]["wall_time_s"] for r in rows); print(f"   wall sum {ledger_total_wall:.0f}s")
