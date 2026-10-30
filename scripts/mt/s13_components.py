"""Offline re-derivation of the three component verdicts (task / manner / contact) from the raw step12 logs.

Why this exists: the step12 logs store ``force_band_passed = passed(...)``, which is task AND manner, and it stores
False for a branch the guard aborted -- but an aborted branch has no hold window and no measurement, so its task and its
manner are UNDETERMINED, not failed.  Nothing is re-simulated; every value below is derived from fields already logged.

Tri-state: "pass" | "fail" | "und".  A component is "und" when the branch never got far enough to produce evidence.
  task    : ep.task_ok of a completed branch; und for a guard-aborted branch.
  manner  : measurement_valid AND grasp_stable AND hold in [lo, hi]; und when the measurement is not valid because the
            branch was aborted before a hold window existed.
  contact : completed branch -> clean_v1.  Aborted branch -> "fail" only when the log holds a violation that no later
            step could excuse (a palm/non-finger-link contact, or an unexplained robot<->support contact); otherwise "und"
            (a window-based reason on a truncated branch is an artefact of the truncation, not evidence).
  joint   : pass only if all three pass; fail if any is fail; otherwise und.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

SRC = Path("reports/step12_ablation")


def is_aborted(e: dict) -> bool:
    return str(e["status"]).startswith("guard_refused_abort")


def unconditional_contact_violation(e: dict) -> bool:
    if e["v1_support_unexplained"]:
        return True
    return any("not_a_finger_link_contact" in u["reasons"] for u in e["v1_unexcused"])


def components(e: dict, band) -> dict:
    if is_aborted(e):
        task = manner = "und"
        contact = "fail" if unconditional_contact_violation(e) else "und"
    else:
        task = "pass" if e["task_ok"] else "fail"
        m, v = e["measured"], e["validity"]
        ok = bool(v["measurement_valid"] and v["grasp_stable"] and m["hold_min"] is not None
                  and m["hold_min"] >= band[0] and m["hold_max"] <= band[1])
        manner = "pass" if ok else ("fail" if v["measurement_valid"] else "und")
        contact = "pass" if e["contact_v1_clean"] else "fail"
    vals = (task, manner, contact)
    joint = "pass" if all(x == "pass" for x in vals) else ("fail" if "fail" in vals else "und")
    return {"task": task, "manner": manner, "contact": contact, "joint": joint}


def load_cells():
    cells = []
    for f in sorted(glob.glob(str(SRC / "t*_init*.json"))):
        cells += json.load(open(f))["cells"]
    return cells


def key(c):
    return (c["task"], c["init"], c["request"]["id"])
