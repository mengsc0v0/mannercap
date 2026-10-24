import json, sys
from pathlib import Path
import numpy as np
D = Path("reports/step5_diag_20261025")
for c in sys.argv[1:]:
    r = json.loads((D / f"diag_{c}.json").read_text())
    rows = r["substeps"]
    ks = sorted({x["k"] for x in rows})
    print(f"== {c}  transport_last={r['transport_last']} place_first={r['place_first']}")
    print(" k-tl | audit_mean(L+R) | lastsub(L+R) | substep L+R min/max | n_contact_sub(L/R=0) | other touch(max N) | |dEef|mm/step(speed m/s) | obj-in-eef dz,dxy mm | ang deg | basket-obj dist(xy,z) mm")
    prev = None
    for k in ks:
        g = [x for x in rows if x["k"] == k]
        tot = np.array([x["fl"] + x["fr"] for x in g])
        zero = sum(1 for x in g if x["nl"] == 0 or x["nr"] == 0)
        touch = {}
        for x in g:
            for b, f in x["touch"].items():
                touch[b] = max(touch.get(b, 0), f)
        e0, e1 = np.array(g[0]["eef"]), np.array(g[-1]["eef"])
        sp = np.linalg.norm(e1 - e0) / ((g[-1]["t"] - g[0]["t"]) or 1)
        rel = np.array(g[-1]["rel_eef_frame"]); rel0 = np.array(g[0]["rel_eef_frame"])
        o = np.array(g[-1]["obj"]); b = np.array(g[-1]["basket"])
        t = ", ".join(f"{n[-14:]}:{f:.1f}" for n, f in touch.items()) or "-"
        print(f" {k - r['transport_last']:+3d} | {r['audit_mean_total'][str(k)]:6.2f} | {r['recorded_last_substep_total'][str(k)]:6.2f} | {tot.min():5.1f}/{tot.max():5.1f} | {zero:2d} | {t} | {sp:.3f} | {1000*(rel[2]-rel0[2]):+.2f},{1000*np.linalg.norm(rel[:2]-rel0[:2]):.2f} | {g[-1]['rel_ang_deg']:.1f} | {1000*np.linalg.norm((o-b)[:2]):.0f},{1000*(o-b)[2]:.0f}")
