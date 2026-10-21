"""End-to-end: requests -> audited production -> pi0.5 package -> tarball -> re-verify.

usage: produce_pi05_dataset.py --out DIR --seeds 1001,1002 [--tasks lift,stack]
       [--manners neutral,slow,fast,gentle,firm] [--max-attempts 3]
       [--calibration configs/manner_calibration.v1.json] [--val-frac .15 --test-frac .15]

wipe: ``--tasks wipe`` builds the 6 speed x press combinations per seed with outcome-first prompts
(``synthesis_v2/wipe_language.py``; ``--manners`` does not apply).  Production can be sharded:
``--produce-only`` stops after the audited production; ``--merge DIR1,DIR2 --out OUT`` merges the
``<DIR>/pipeline`` directories of the shards and exports the merged package.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, tarfile
from pathlib import Path

from synthesis_v2.manner_calibration import load_calibration
from synthesis_v2.pi05_export import export_package, make_tarball
from synthesis_v2.production_pipeline import RequestSpec, run_requests, summarize

REPO = Path(__file__).resolve().parents[1]
BASE = {"lift": "Lift the cube", "stack": "Stack the red cube on the green cube"}
PHRASE = {
    "neutral": "", "slow": " slowly", "fast": " quickly", "gentle": " gently", "firm": " firmly",
    "slow_gentle": " slowly and gently",
    "fast_approach_slow_place": ", approach quickly and place slowly",
}
STACK_ONLY = {"fast_approach_slow_place"}


def wipe_specs(seed: int) -> list:
    from synthesis_v2.wipe_language import COMBOS, base_for_seed, build_prompt, combo_intents
    out = []
    for speed, press in COMBOS:
        prompt, style = build_prompt(seed, speed, press)
        out.append(RequestSpec("wipe", seed, prompt, base_instruction=base_for_seed(seed), pair_key=f"wipe:{seed}",
                               intents=combo_intents(speed, press),
                               extra={"manner_name": f"speed_{speed}_press_{press}", "language_style": style}))
    return out


def merge_pipelines(shards: list, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "raw").mkdir()
    for name in ("requests.jsonl", "attempts.jsonl"):
        lines = []
        for d in shards:
            lines += [x for x in (Path(d) / "pipeline" / name).read_text(encoding="utf-8").splitlines() if x.strip()]
        (dest / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for d in shards:
        for r in sorted((Path(d) / "pipeline" / "raw").iterdir()):
            if (dest / "raw" / r.name).exists():
                raise FileExistsError(r.name)
            shutil.move(str(r), str(dest / "raw" / r.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--tasks", default="lift,stack")
    ap.add_argument("--manners", default="neutral,slow,fast,gentle,firm")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--calibration", default=str(REPO / "configs/manner_calibration.v1.json"))
    ap.add_argument("--produce-only", action="store_true")
    ap.add_argument("--merge", default="", help="comma-separated shard output dirs to merge instead of producing")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    a = ap.parse_args()
    out = Path(a.out)
    cal = load_calibration(a.calibration)
    specs = []
    for task in a.tasks.split(","):
        if task == "wipe":
            for seed in (int(x) for x in a.seeds.split(",")):
                specs += wipe_specs(seed)
            continue
        for seed in (int(x) for x in a.seeds.split(",")):
            for m in a.manners.split(","):
                if m in STACK_ONLY and task != "stack":
                    continue
                specs.append(RequestSpec(task, seed, BASE[task] + PHRASE[m] + ".", base_instruction=BASE[task] + ".",
                                         pair_key=f"{task}:{seed}", extra={"manner_name": m}))
    if a.merge:
        merge_pipelines(a.merge.split(","), out / "pipeline")
    else:
        results = run_requests(specs, cal, out / "pipeline", max_attempts=a.max_attempts)
    summ = summarize(out / "pipeline")
    (out / "pipeline_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    if a.produce_only:
        return
    from scripts.pi05_replay_native import replay_episode
    root = out / "package" / "pi05_manner_dataset"
    man = export_package(out / "pipeline", root, cal=cal, replay=replay_episode, repo_root=REPO,
                         val_frac=a.val_frac, test_frac=a.test_frac)
    print(json.dumps({k: man[k] for k in ("n_episodes", "n_failed_attempts", "split_counts", "pairs")}))
    tgz = out / "package" / "pi05_manner_dataset.tar.gz"
    digest = make_tarball(root, tgz)
    (out / "package" / "pi05_manner_dataset.tar.gz.sha256").write_text(f"{digest}  {tgz.name}\n")
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp)
        ex = Path(tmp) / root.name
        r1 = subprocess.run([sys.executable, str(ex / "tools/verify_package.py"), str(ex)], capture_output=True, text=True)
        print("extracted verify:", r1.stdout.strip(), r1.stderr.strip()[:500])
        env = dict(os.environ, PYTHONPATH=str(ex / "tools"))
        r2 = subprocess.run([sys.executable, str(ex / "tools/pi05_replay_native.py"), str(ex / "dataset"), str(Path(tmp) / "r.json"),
                             "--sample-n", "2"], capture_output=True, text=True, env=env, cwd=tmp)
        print("extracted replay rc:", r2.returncode, [l for l in r2.stdout.splitlines() if l.startswith("{")])
        if r1.returncode or r2.returncode:
            print(r2.stderr[-800:])
            sys.exit(1)
    print("DONE", tgz, digest)


if __name__ == "__main__":
    main()
