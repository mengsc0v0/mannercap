import sys, json, glob
for f in sorted(glob.glob(sys.argv[1])):
    for l in open(f):
        r = json.loads(l); m = r.get("meta_sweep") or {}
        print(r["seed"], r["v"], r["F"], "ok" if r["ok"] else "FAIL", r["status"][:20], "sw", r["stage_steps"].get("sweep"),
              "wiped", m.get("wiped"), "f", round(r.get("f_med", -1), 1), round(r.get("f_p10", -1), 1), round(r.get("f_p90", -1), 1),
              "max", round(r.get("f_max", -1), 1), "sp", round(r.get("sp_med", -1), 3))
