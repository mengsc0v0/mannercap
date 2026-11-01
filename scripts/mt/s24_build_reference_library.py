"""Step 24 (interface item 1): build configs/grasp_references.v1.json from the references the task configs already use.

No simulation, no new demonstration processing: every number is the one frozen in ``scripts/mt/s7_task_config.SPECS`` and is
cross-checked against the committed derivation record (rounding only) before the entry is written.  Provenance = demonstration
file name + sha256, demo / close step, derivation record + extraction script hashes.
usage: s24_build_reference_library.py [DATASET_DIR]   (default: the LIBERO libero_10 dataset directory)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, ".")
from s7_task_config import ROOT, SPECS  # noqa: E402
from synthesis_v2.grasp_references import DEFAULT_PATH, FRAME_DEFINITION, LIBRARY_VERSION, entry_digest, file_sha256  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/workdata/cap-x/capx/third_party/LIBERO-PRO/libero/datasets/libero_10")
FILES = {8: "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
         5: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
         4: "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"}
HASHED_AT_FREEZE = {8: "reports/step22_t8_onboard/round1/identity.json"}   # demo_file_sha256 recorded before the first T8 run
BDDL_NOTE = {8: "bddl KITCHEN_SCENE8_put_both_moka_pots_on_the_stove: 'moka_pot_1 moka_pot_2 - moka_pot' (same object class)"}


def rec(path):
    p = ROOT / path
    return {"path": path, "sha256": file_sha256(p)}


def near(a, b, nd):
    return all(abs(round(x, nd) - y) < 1e-9 for x, y in zip(a, b)) if isinstance(a, (list, tuple)) else abs(round(a, nd) - b) < 1e-9


def dataset(t):
    f = DATA / FILES[t]
    d = {"name": FILES[t], "sha256": file_sha256(f), "hash_taken": "library build (2026-11-01)"}
    fz = HASHED_AT_FREEZE.get(t)
    if fz:
        d["hash_recorded_at_freeze"] = {"path": fz, "equal": json.load(open(ROOT / fz))["demo_file_sha256"] == d["sha256"]}
        assert d["hash_recorded_at_freeze"]["equal"], "dataset file differs from the one recorded at the freeze"
    else:
        d["hash_recorded_at_freeze"] = None   # not recorded when the reference was extracted
    return d


entries = []


def add(**kw):
    kw["frame"] = {k: v for k, v in FRAME_DEFINITION.items() if k in kw["value"]}
    kw["content_sha256"] = entry_digest(kw)
    entries.append(kw)


# ---- T8: two pots, demo_0, the close that lifted each pot -------------------------------------------------------------
T = 8
demo = json.load(open(ROOT / "reports/step22_t8_onboard/demo_replay_A/demo_0.json"))
clr = json.load(open(ROOT / "reports/step23_shared_fixes/demo_close_clearance.json"))
ds = dataset(T)
for k, (role, name) in enumerate(SPECS[T].objects):
    val = SPECS[T].grasp_reference[k]
    ev = next(e for e in demo["close_events"] if e["lifted"] == name)
    assert near(ev["rel_obj"], val["rel_obj"], 4) and near(ev["hand_yaw_rel_deg"], val["hand_yaw_rel_deg"], 1)
    assert clr[name]["close_step"] == ev["close_step"]
    add(reference_id=f"libero_10.t8.demo_0.close_{ev['close_step']}.{name}", scene=f"libero:libero_10:{T}", object_name=name,
        object_class="moka_pot", compatible_objects=["moka_pot_1", "moka_pot_2"], compatibility_basis=BDDL_NOTE[T] +
        "; geometry-equal asset assumed; compatibility is declared, NOT evidence that a grasp from this reference succeeds",
        kind="single_demo_close", value=json.loads(json.dumps(val)),
        source={"dataset_file": ds, "demo": "demo_0", "close_step": ev["close_step"],
                "capture": "control step (replay of the recorded actions) of the gripper close that lifted this object",
                "selection_rule": "lowest-index demo replay-successful in two independent replays (T5 rule), used as is",
                "derivation_records": [rec("reports/step22_t8_onboard/demo_replay_A/demo_0.json"),
                                       rec("reports/step22_t8_onboard/demo_selection.json"),
                                       rec("reports/step23_shared_fixes/t8_full_pose_check.json")],
                "extraction_scripts": [rec("scripts/mt/s22_t8_demo_extract.py"), rec("scripts/mt/s23_full_pose_ref.py")]},
        demo_contact_at_close={"status": "recorded", "record": rec("reports/step23_shared_fixes/demo_close_clearance.json"),
                               "contacts_at_close_start": clr[name]["contacts_at_close_start"],
                               "note": "dist_m <= 0: the demonstration hand already touches the object at the close"})
# ---- T5: one book, demo_0 ---------------------------------------------------------------------------------------------
T = 5
d5 = json.load(open(ROOT / "reports/step14_t5_onboard/demo_replay/demo_0.json"))
val = SPECS[T].grasp_reference[0]
assert near(d5["grasp"]["rel_obj"], val["rel_obj"], 4) and near(d5["grasp"]["hand_yaw_rel_deg"], val["hand_yaw_rel_deg"], 1)
name = SPECS[T].objects[0][1]
add(reference_id=f"libero_10.t5.demo_0.close_{d5['grasp']['close_step']}.{name}", scene=f"libero:libero_10:{T}", object_name=name,
    object_class="black_book", compatible_objects=[name], compatibility_basis="only the object it was extracted from",
    kind="single_demo_close", value=json.loads(json.dumps(val)),
    source={"dataset_file": dataset(T), "demo": "demo_0", "close_step": d5["grasp"]["close_step"],
            "capture": "control step of the LAST gripper close of the demo",
            "selection_rule": "lowest-index demo replay-successful in two independent replays, used as is",
            "derivation_records": [rec("reports/step14_t5_onboard/demo_replay/demo_0.json")],
            "extraction_scripts": [rec("scripts/mt/s14_t5_demo_extract.py")]},
    demo_contact_at_close={"status": "not_recorded"})
# ---- T4: two mugs, median over replay-successful demos ----------------------------------------------------------------
T = 4
r4 = json.load(open(ROOT / "reports/step9_t4_onboard/ref_grasp.json"))
ds = dataset(T)
for k, (role, name) in enumerate(SPECS[T].objects):
    val, pm = SPECS[T].grasp_reference[k], r4["per_mug"][name]
    assert near(pm["median_rel_obj"], val["rel_obj"], 4) and near(pm["median_hand_yaw_rel_deg"], val["hand_yaw_rel_deg"], 1)
    add(reference_id=f"libero_10.t4.median_n{pm['n']}.{name}", scene=f"libero:libero_10:{T}", object_name=name,
        object_class=name.rsplit("_", 1)[0], compatible_objects=[name], compatibility_basis="only the object it was extracted from",
        kind="median_over_demos", value=json.loads(json.dumps(val)),
        source={"dataset_file": ds, "demo": None, "close_step": None, "n_demos": pm["n"],
                "n_demos_in_dataset": r4["replay"]["n_demos"], "n_replay_success": r4["replay"]["n_replay_success"],
                "capture": "per replay-successful demo, the LAST close event that lifted this object; component-wise median",
                "per_demo_rows": "per_mug.<object>.rows in the derivation record (demo, close_step, rel_obj, hand_yaw_rel_deg)",
                "derivation_records": [rec("reports/step9_t4_onboard/ref_grasp.json")],
                "extraction_scripts": [rec("scripts/mt/s9_ref_grasp.py")]},
        demo_contact_at_close={"status": "not_recorded"})

lib = {"version": LIBRARY_VERSION, "note": "append-only: an id is never reused for different numbers; content_sha256 pins each entry",
       "frame_definitions": FRAME_DEFINITION, "entries": entries}
DEFAULT_PATH.write_text(json.dumps(lib, indent=1, ensure_ascii=False) + "\n")
print("wrote", DEFAULT_PATH, [e["reference_id"] for e in entries])
