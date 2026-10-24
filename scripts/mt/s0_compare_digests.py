"""Compare two episode_digests.json files (baseline vs. candidate) record by record."""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
key = lambda r: (r["task"], r["variant"], r.get("seed"))
B = {key(r): r for r in b}
same = diff = missing = 0
for r in a:
    o = B.get(key(r))
    if o is None:
        missing += 1; print("MISSING", key(r)); continue
    da, db = r.get("digest", {}), o.get("digest", {})
    if da.get("_combined") == db.get("_combined") and da.get("actions") == db.get("actions") and da.get("images") == db.get("images"):
        same += 1
    else:
        diff += 1
        print("DIFF", key(r), [k for k in da if da.get(k) != db.get(k)][:8], r.get("status"), o.get("status"))
print(f"SUMMARY identical={same} different={diff} missing={missing} of {len(a)}")
sys.exit(1 if diff or missing else 0)
