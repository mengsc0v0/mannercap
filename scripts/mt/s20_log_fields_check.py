"""Shows the new retry-log fields on one recover execution (T0 i26, candidate (1,100)); asserts the identities they claim."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_recovery_ablation as S12  # noqa: E402

orig = S12.run_guarded
box = {}


def wrapped(*a):
    out = orig(*a)
    box["rec"], box["ep"] = out[2], out[1]
    return out


S12.run_guarded = wrapped
e = S12.execute(0, 26, (1.0, 100.0), "recover", {})
ev = box["ep"].meta["execution_guard"]["retries"]
sc, rb = box["rec"]["substep_cost"], box["rec"]["rolled_back_cost"]
print(json.dumps(ev, indent=1, default=float))
print("executed", sc["executed_substeps"], "formal", sc["formal_rows"], "discarded", rb["substeps"])
assert sc["executed_substeps"] == sc["formal_rows"] + rb["substeps"]
assert ev and all(r["state_identity_match"] is True for r in ev), "restored state must equal the snapshot state"
guard_disc = sum(r["discarded_substeps_by_this_restore"] for r in ev)
print("guard-discarded", guard_disc, "planner look-ahead discarded", rb["substeps"] - guard_disc)
assert 0 <= guard_disc <= rb["substeps"]
