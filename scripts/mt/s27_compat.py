"""Step 27 compatibility check: re-run fixed (task, init, config) cells of the step-25 grid on the OLD env wrapper
(OffScreenRenderEnv, render=True) and compare with the stored grid record (ControlEnv, render=False).
Usage: s27_compat.py OUT_JSON T:I:CFG [T:I:CFG ...]      CFG like 10_100@0.12
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25_grid as G  # noqa: E402
import s7_task_config as C  # noqa: E402

FIELDS = ("task_ok", "status", "n_steps", "contact_v1_clean", "guard_retries", "carry_speed_med_mps",
          "carry_speed_max_mps", "substeps_executed", "stage_failure", "grasp_failure")
OBJ_FIELDS = ("status", "grasp_stable", "hold_n", "hold_mean", "hold_min", "hold_max")


def main() -> int:
    out = Path(sys.argv[1])
    G.make_session = lambda t, i, prog: C.make_session(t, i, prog, render=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["task"], r["init"], r["cfg"]) for r in rows}
    for spec in sys.argv[2:]:
        t, i, cfg = spec.split(":"); t, i = int(t), int(i)
        if (t, i, cfg) in done:
            continue
        f, sp = cfg.split("@"); lo, hi = f.split("_")
        old = G.run_one(t, i, (float(lo), float(hi)), float(sp))
        new = json.loads((G.ROOT / f"reports/step25_response_grid/t{t}_init{i}.json").read_text())["episodes"][cfg]
        diff = {k: (new.get(k), old.get(k)) for k in FIELDS if new.get(k) != old.get(k)}
        for role in new["objects"]:
            for k in OBJ_FIELDS:
                a, b = new["objects"][role].get(k), old["objects"][role].get(k)
                if a != b:
                    diff[f"{role}.{k}"] = (a, b)
        unexc = (sorted((c["robot_body"], c["other"]) for c in new["v1_unexcused"]),
                 sorted((c["robot_body"], c["other"]) for c in old["v1_unexcused"]))
        if unexc[0] != unexc[1]:
            diff["v1_unexcused_pairs"] = unexc
        rows.append({"task": t, "init": i, "cfg": cfg, "new_controlenv": new, "old_offscreen": old,
                     "diff_new_vs_old": diff, "identical": not diff})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, default=str))
        print(f"T{t} i{i} {cfg} identical={not diff} diff={json.dumps(diff, default=str)[:400]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
