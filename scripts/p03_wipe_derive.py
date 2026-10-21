"""Derive the wipe bands from dev records and merge them into a copy of an existing calibration.

usage: p03_wipe_derive.py BASE_CAL.json OUT_CAL.json RECORDS.jsonl [RECORDS2.jsonl ...]
The lift/stack sections of BASE_CAL are copied unchanged; wipe speed/press/safety are added.
"""
import json, sys

from synthesis_v2.manner_calibration import (
    DEFAULT_POLICY, derive_wipe_calibration, load_calibration, merge_wipe_calibration, save_calibration,
)
from synthesis_v2.manner_request import semantics_hash
from synthesis_v2.native_osc_env import CONTROLLER_CONFIG_SHA256
from synthesis_v2.stage_table import stage_table_hash
from synthesis_v2.wipe_producer import WIPE_PRODUCER_VERSION

base = load_calibration(sys.argv[1])
recs = [json.loads(x) for p in sys.argv[3:] for x in open(p) if x.strip()]
seeds = sorted({r["seed"] for r in recs})
w = derive_wipe_calibration(recs, dev_seeds=seeds, meta={})
cal = merge_wipe_calibration(base, w, {
    "stage_table_hash": stage_table_hash(), "semantics_hash": semantics_hash(),
    "lift_stack_calibrated_under_stage_table_hash": base["meta"].get("stage_table_hash"),
    "wipe_producer_version": WIPE_PRODUCER_VERSION, "wipe_controller_sha256": CONTROLLER_CONFIG_SHA256})
save_calibration(cal, sys.argv[2])
json.dump(recs, open(sys.argv[2] + ".wipe_records.json", "w"))
print(json.dumps({"speed": cal["speed"]["wipe"], "press": cal["press"]["wipe"], "safety": cal["safety"]["wipe"],
                  "n_dev": w["n_dev_episodes"], "n_failed": w["n_dev_failed"]}, indent=1))
print("written", sys.argv[2], cal["content_sha256"][:12])
