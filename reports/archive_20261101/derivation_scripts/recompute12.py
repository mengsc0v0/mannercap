import json, glob, collections, statistics as st
cells=[]
for f in sorted(glob.glob("reports/step12_ablation/t*_init*.json")):
    cells+=json.load(open(f))["cells"]
print("cells",len(cells))
ARMS=[("rule","rerun"),("rule","recover"),("retrieval","rerun"),("retrieval","recover")]
def ab(e): return str(e["status"]).startswith("guard_refused_abort")
def comp(e,band):
    m,v=e["measured"],e["validity"]
    task=bool(e["task_ok"])
    man=bool(v["measurement_valid"] and v["grasp_stable"] and m["hold_min"] is not None and m["hold_min"]>=band[0] and m["hold_max"]<=band[1])
    con=bool(e["contact_v1_clean"])
    return task,man,con
rows=collections.defaultdict(dict)
for c in cells:
    rows[(c["selection"],c["recovery"])][(c["task"],c["init"],c["request"]["id"])]=c
res={}
print("\n== per arm (n=32), recomputed from raw cells")
for arm in ARMS:
    R=rows[arm]; n=len(R)
    fin_t=fin_m=fin_c=fin_j=0; first_t=first_m=first_c=first_j=0; ever_t=ever_m=ever_c=0
    first_pass_attempt=collections.Counter()
    for k,c in R.items():
        band=c["request"]["band_n"]
        done=[e for e in c["executions"] if not ab(e)]
        fin=c["executions"][-1]
        if not ab(fin):
            t,m,cc=comp(fin,band); fin_t+=t; fin_m+=m; fin_c+=cc; fin_j+=(t and m and cc)
        a1=[e for e in done if e["force_attempt"]==1]
        if a1:
            t,m,cc=comp(a1[-1],band); first_t+=t; first_m+=m; first_c+=cc; first_j+=(t and m and cc)
        ev=[comp(e,band) for e in done]
        ever_t+=any(x[0] for x in ev); ever_m+=any(x[1] for x in ev); ever_c+=any(x[2] for x in ev)
        first_pass_attempt[c["joint_passed_at_force_attempt"]]+=1
        assert c["joint_passed"]==(c["joint_passed_at_force_attempt"] is not None)
    jp=sum(c["joint_passed"] for c in R.values())
    sub_m=sum(sum(e["executed_substeps"] for e in c["executions"]) for c in R.values())
    sub_s=sum(sum(e["executed_substeps"] for e in c["executions"] if not e["reused_cached_episode"]) for c in R.values())
    assert sub_m==sum(c["substeps_method"] for c in R.values()) and sub_s==sum(c["substeps_spent"] for c in R.values())
    nex=sum(c["n_physical_executions"] for c in R.values()); nreal=sum(c["n_actually_executed"] for c in R.values())
    disc=sum(c["discarded_substeps"] for c in R.values()); ctrl=sum(c["control_steps_used"] for c in R.values())
    wm=sum(c["wall_method_s"] for c in R.values()); ws=sum(c["wall_spent_s"] for c in R.values())
    # cost on the requests that eventually passed vs failed
    cost_pass=sum(sum(e["executed_substeps"] for e in c["executions"]) for c in R.values() if c["joint_passed"])
    cost_fail=sub_m-cost_pass
    res[arm]=dict(n=n,joint=jp,sub_m=sub_m,sub_s=sub_s)
    print(f"{arm[0]:9s}/{arm[1]:7s} joint(final-pass)={jp}/{n} | last-exec completed: task {fin_t} manner {fin_m} contact {fin_c} joint {fin_j} | "
          f"first-choice(force attempt1 completed): task {first_t} manner {first_m} contact {first_c} joint {first_j} | ever(any completed exec): task {ever_t} manner {ever_m} contact {ever_c}")
    print(f"      first joint pass at force attempt: {dict(first_pass_attempt)}; phys execs {nex} (real {nreal}, cached {nex-nreal}); control steps {ctrl}; discarded substeps {disc}")
    print(f"      substeps method {sub_m} spent {sub_s}; wall method {wm:.0f}s spent {ws:.0f}s; method substeps per qualified {sub_m/jp:.0f}; cost on eventually-passed {cost_pass}, on never-passed {cost_fail}")
print("\nsum spent over four arms (actual physical resource of the whole 2x2):",sum(v['sub_s'] for v in res.values()),"substeps; sum method:",sum(v['sub_m'] for v in res.values()))
# substeps per control step ratio
rat=[e["executed_substeps"]/e["n_steps"] for c in cells for e in c["executions"] if not e["reused_cached_episode"] and e["n_steps"]]
print("executed_substeps per control step (uncached execs): median %.1f min %.1f max %.1f n=%d"%(st.median(rat),min(rat),max(rat),len(rat)))
# first execution refused?
cnt=collections.Counter()
for c in cells:
    e=c["executions"][0]; cnt[(bool(e["guard_refused"]),ab(e),c["recovery"])]+=1
print("first execution (refused?, aborted?, arm):",dict(cnt))
# paired selection effect
print("\n== paired rule vs retrieval (joint pass, final), per recovery")
for rec in ("rerun","recover"):
    A=rows[("rule",rec)]; B=rows[("retrieval",rec)]
    g=[k for k in A if (not A[k]["joint_passed"]) and B[k]["joint_passed"]]
    r=[k for k in A if A[k]["joint_passed"] and not B[k]["joint_passed"]]
    both=[k for k in A if A[k]["joint_passed"] and B[k]["joint_passed"]]
    neither=[k for k in A if not A[k]["joint_passed"] and not B[k]["joint_passed"]]
    print(rec,"gain(retrieval only)",len(g),g,"regress(rule only)",len(r),r,"both",len(both),"neither",len(neither))
    # first-choice difference
    def first_ok(c):
        band=c["request"]["band_n"]; done=[e for e in c["executions"] if not ab(e) and e["force_attempt"]==1]
        if not done: return None
        t,m,cc=comp(done[-1],band); return (t,m,cc)
    mm=collections.Counter()
    for k in A:
        a=first_ok(A[k]); b=first_ok(B[k]); mm[(a[1],b[1])]+=1
    print("   first-choice manner (rule,retrieval) counts:",dict(mm))
    # cost on commonly-passed requests, method book
    ca=[sum(e["executed_substeps"] for e in A[k]["executions"]) for k in both]; cb=[sum(e["executed_substeps"] for e in B[k]["executions"]) for k in both]
    print("   method cost on both-pass requests: rule",sum(ca),"retrieval",sum(cb),"n",len(both))
    # all 32 incl failures
    print("   method cost on all 32: rule",res[("rule",rec)]['sub_m'],"retrieval",res[("retrieval",rec)]['sub_m'])
# per task / per request breakdown
print("\n== by task and by request id, joint pass (final)")
for dim,fn in (("task",lambda k:k[0]),("request",lambda k:k[2]),("init",lambda k:(k[0],k[1]))):
    for arm in ARMS:
        cnt=collections.Counter(); tot=collections.Counter()
        for k,c in rows[arm].items():
            tot[fn(k)]+=1; cnt[fn(k)]+=c["joint_passed"]
        print(dim,f"{arm[0]}/{arm[1]}",{str(x):f"{cnt[x]}/{tot[x]}" for x in sorted(tot)})
# recovery effect, same selection
print("\n== recovery effect (rerun -> recover), method book")
for sel in ("rule","retrieval"):
    a=res[(sel,"rerun")]['sub_m']; b=res[(sel,"recover")]['sub_m']
    print(sel,"rerun",a,"recover",b,"saved",f"{1-b/a:.1%}","joint",res[(sel,'rerun')]['joint'],res[(sel,'recover')]['joint'])
    cnt=0; sa=0; sb=0
    for k in rows[(sel,"rerun")]:
        A=rows[(sel,"rerun")][k]; B=rows[(sel,"recover")][k]
        ca=sum(e["executed_substeps"] for e in A["executions"]); cb=sum(e["executed_substeps"] for e in B["executions"])
        cnt+= cb<ca; sa+=ca; sb+=cb
    print("   recover cheaper in",cnt,"of 32; recover not cheaper (equal or more):",32-cnt)
