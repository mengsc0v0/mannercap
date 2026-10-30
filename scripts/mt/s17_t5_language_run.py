"""Step 17 execution check (<=4 runs): sentence -> cached LLM intake -> admit -> plan -> T5 trajectory -> v1.1 audit.
Reuses the frozen step-16 runner (run_one: GUARD, GripPlan, windows, M1-M4 fields) unchanged; only the plan comes from language.
Trajectories go to reports/step17_t5_language/traj (step-16 files are never written) and are compared bit for bit with the
step-16 run of the same init and the structured condition that this sentence is expected to equal."""
import json, sys
from pathlib import Path
import numpy as np
_argv = sys.argv; sys.argv = [_argv[0], "new"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
import s16_t5_speed_pilot as s16
from s7_task_config import ROOT, make_program
from synthesis_v2.llm_interface import LLMClient, LLMConfig
from synthesis_v2.language_plan import compile_request
from synthesis_v2.manner_request import MannerIntent
sys.argv = _argv
OUT = ROOT / "reports" / "step17_t5_language"; (OUT / "traj").mkdir(exist_ok=True)
s16.OUT = OUT                                   # run_one writes traj to s16.OUT/"traj": redirect away from step-16 files
SENT = {"S0": ("把书放入后部隔间，松开并退出。", "N"), "S1": ("把书搬过去，缓慢放入后部隔间。", "S")}
INITS = {0: ROOT / "reports/step16_t5_speed_pilot/dev/traj", 10: ROOT / "reports/step16_t5_speed_pilot/new/traj"}
COLS = s16.PREFIX_COLS + ("stage", "eef_speed_mps")


def main():
    prog = make_program(5); base = s16.make_plan(5, prog, grip=s16.GRIP, approach_check=True, stop_on_grasp_failure=True, stop_on_place_failure=True)
    client = LLMClient(LLMConfig.from_env(cache_dir=str(OUT / "llm_cache"), ledger_path=str(OUT / "llm_ledger.jsonl")), offline=True)
    plans = {}
    for sid, (text, cond) in SENT.items():
        r = compile_request(text, prog, base, client)
        assert r["status"] == "accept" and r["plan"] is not None, (sid, r["status"])
        plans[sid] = r
    recs = []
    for i in INITS:
        for sid, (text, cond) in SENT.items():
            r = plans[sid]
            s16.REQUEST[sid] = tuple(MannerIntent(iid.split(".")[-1], b["dimension"], b["level"])
                                     for b in r["intake"]["semantics"]["bindings"] for iid in b["instance_ids"] if b["level"] != "normal")
            s16.plan_for = (lambda c, p, _pl=r["plan"]: _pl)
            rec, arr, st = s16.run_one(i, sid)               # cond label = sentence id -> traj/init{i}_{sid}.npz
            ref = np.load(INITS[i] / f"init{i}_{cond}.npz", allow_pickle=True)
            mine = np.load(OUT / "traj" / f"init{i}_{sid}.npz", allow_pickle=True)
            same = {k: bool(np.array_equal(ref[k], mine[k], equal_nan=True)) if ref[k].dtype.kind in "fiub" else bool(np.array_equal(ref[k], mine[k])) for k in COLS}
            full = {"sentence_id": sid, "instruction": text, "init": i, "llm": r["llm"], "versions": r["versions"], "status_admit": r["status"],
                    "bindings": [{k: b[k] for k in ("instance_ids", "dimension", "level", "clause")} for b in r["intake"]["semantics"]["bindings"]],
                    "bases": r["intake"]["bases"], "guards": r["decision"]["guards"], "plan_trace": r["trace"], "plan_hash": r["plan_hash"],
                    "structured_reference": {"step16_cond": cond, "step16_file": str((INITS[i] / f"init{i}_{cond}.npz").relative_to(ROOT))},
                    "trajectory_bitwise_equal_to_step16": same, "trajectory_identical": all(same.values()), "run": rec}
            recs.append(full)
            d = rec.get("descent") or {}
            print(f"RUN {sid} init{i} gate={rec['ordinary_gate']} official_ever={rec['official_success_ever']} instr={rec['instruction_completed']} "
                  f"clean_v1.1={rec['contact_clean_v1_1']} descent_n={d.get('n_steps')} mean={d.get('mean_speed_mps')} identical_to_step16={full['trajectory_identical']}", flush=True)
    (OUT / "execution_records.json").write_text(json.dumps({"schema": "mannercap.t5_language_execution.v1", "n_runs": len(recs), "records": recs},
                                                           indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("DONE", len(recs))


if __name__ == "__main__":
    main()
