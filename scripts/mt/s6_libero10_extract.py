"""Step 6a: read the ten LIBERO-10 tasks from the benchmark's own BDDL files (no hand-copied descriptions)."""
import json
import re
import sys
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(os.environ.get("LIBERO_ROOT", "/workdata/cap-x/capx/third_party/LIBERO-PRO"))
from libero import benchmark  # noqa: E402
from libero.utils import get_libero_path  # noqa: E402


def block(txt: str, key: str) -> str:
    i = txt.find("(:" + key)
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(txt):
        depth += txt[j] == "("
        depth -= txt[j] == ")"
        j += 1
        if depth == 0:
            break
    return txt[i:j]


def main() -> int:
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    out = []
    for i in range(bm.get_num_tasks()):
        t = bm.get_task(i)
        f = Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
        txt = f.read_text()
        objs = block(txt, "objects")
        fx = block(txt, "fixtures")
        goal = block(txt, "goal")
        regions = re.findall(r"\(:regions?\b", txt)
        out.append({"id": i, "name": t.name, "language": t.language, "bddl": str(f.relative_to(get_libero_path("bddl_files"))),
                    "objects": re.sub(r"\s+", " ", objs), "fixtures": re.sub(r"\s+", " ", fx), "goal": re.sub(r"\s+", " ", goal), "init": re.sub(r"\s+", " ", block(txt, "init")),
                    "n_init_states": len(bm.get_task_init_states(i))})
    (ROOT / "reports" / "step6_libero10_spec_20261025" / "libero10_tasks.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r["id"], r["language"], "|", r["goal"], "|", r["objects"], "| init", r["n_init_states"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
