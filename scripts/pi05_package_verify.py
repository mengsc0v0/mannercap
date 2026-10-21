"""Verify a delivered package: SHA256SUMS, per-episode hashes vs episodes.jsonl,
format check (tools/pi05_format_check.py), split-by-seed integrity.  numpy only.

usage: verify_package.py PACKAGE_ROOT
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(root):
    errs = []
    for line in open(os.path.join(root, "SHA256SUMS")):
        if not line.strip():
            continue
        digest, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            errs.append(f"missing {rel}")
        elif sha(p) != digest:
            errs.append(f"sha mismatch {rel}")
    ds = os.path.join(root, "dataset")
    recs = [json.loads(x) for x in open(os.path.join(ds, "episodes.jsonl")) if x.strip()]
    from pi05_format_check import check_episode
    split_of = {}
    for r in recs:
        d = os.path.join(ds, r["episode"])
        if sha(os.path.join(d, "images.npy")) != r["images_sha256"]:
            errs.append(f"{r['episode']}: images_sha256 mismatch")
        if sha(os.path.join(d, "transitions.npz")) != r["transitions_sha256"]:
            errs.append(f"{r['episode']}: transitions_sha256 mismatch")
        for f in ("metadata.json", "model.xml", "replay_state.npz"):
            if not os.path.exists(os.path.join(d, f)):
                errs.append(f"{r['episode']}: missing {f}")
        errs.extend(check_episode(d, r["episode"]))
        if split_of.setdefault(r["seed"], r["split"]) != r["split"]:
            errs.append(f"seed {r['seed']} appears in more than one split")
    rep = json.load(open(os.path.join(ds, "replay_audit.json")))
    if rep["n_pass"] != rep["n_replayed"]:
        errs.append("replay_audit records failing replays inside the exported set")
    print(json.dumps({"episodes": len(recs), "errors": len(errs), "passed": not errs}))
    for e in errs[:30]:
        print(e, file=sys.stderr)
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
