"""Step 11.1c: attribution summary for the 0/32 contact-policy failures of E3.

Combines (a) the contact COUNTS that are already stored for every executed episode (reports/step10_experience/grid,
64 episodes = 8 candidates x 8 trial inits, no new simulation), with (b) the full contact IDENTITIES of the two
representative cases re-executed by s11_contact_attribution.py, and (c) the static geom identity dump of
s11_gripper_geoms.py which decides whether an audit label is a classification error or a real body.

Usage: s11_attribution_summary.py -> reports/step11_contact/attribution_summary.json
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s7_task_config import ROOT  # noqa: E402

D = ROOT / "reports" / "step11_contact"


def main() -> int:
    eps = {}
    for f in glob.glob(str(ROOT / "reports" / "step10_experience" / "grid" / "t*_init*.json")):
        r = json.loads(Path(f).read_text())
        for k, e in r["episodes"].items():
            eps[(r["task"], r["init"], k)] = e
    counts = {}
    for t in sorted({k[0] for k in eps}):
        sub = [e for k, e in eps.items() if k[0] == t]
        counts[f"T{t}"] = {
            "n_episodes": len(sub),
            "contact_policy_passed": sum(e["contact_policy_passed"] for e in sub),
            "n_forbidden_0p1N_histogram": dict(sorted(Counter(e["n_forbidden_contacts_0p1N"] for e in sub).items())),
            "n_forbidden_ge5N_histogram": dict(sorted(Counter(e["n_forbidden_ge5N"] for e in sub).items())),
        }
    geoms = {}
    for f in glob.glob(str(D / "gripper_geoms_t*.json")):
        g = json.loads(Path(f).read_text())
        geoms[f"T{g['task']}"] = [r for r in g["geoms"] if r["contype"] or r["conaffinity"]]
    cases = {}
    for f in sorted(glob.glob(str(D / "case_t*_init*.json"))):
        c = json.loads(Path(f).read_text())
        ca = c["contact_audit"]
        cases[f"T{c['task']}_init{c['init']}_{c['request']}"] = {
            "candidate": c["candidate"], "requested_object": c["requested_object"], "task_ok": c["task_ok"],
            "development_criterion_passed": c["development_criterion_passed"],
            "clean": ca["clean"], "max_forbidden_n": ca["max_forbidden_n"],
            "forbidden_by_class": ca["forbidden_by_class"],
            "clean_if_finger_link_on_target_allowed": ca["clean_if_finger_link_on_target_allowed__preview_not_verdict"],
            "forbidden_contacts": [{k: v for k, v in x.items()} for x in ca["forbidden_contacts"]],
        }
    out = {
        "schema": "mannercap.contact_attribution_summary.v1",
        "scope": "E3 retrieval comparison (configs/experience_retrieval_e1.json); groups A and B execute the same "
                 "episodes, so a contact failure is a property of the episode, not of the group.",
        "counts_from_existing_records_no_new_simulation": counts,
        "representative_cases_re_executed": cases,
        "gripper_collision_geoms": geoms,
        "findings": {
            "common_to_all": "0 of 64 executed episodes pass the frozen contact policy.",
            "not_one_cause": "the failure is NOT a single cause: the two tasks fail in different classes.",
            "T7_cream_cheese": "every forbidden contact in the representative case is class finger_link_on_target "
                               "(the finger-link collision geom adjacent to the pad, on the grasp target itself, "
                               "during grasp/lift). forbidden_by_class.other = 0, so "
                               "clean_if_finger_link_on_target_allowed = True. This is a POLICY-DEFINITION question: "
                               "the frozen policy allows only the pad to touch the target.",
            "T0_tomato_sauce": "the representative case has 2 contacts of class 'other': body gripper0_right_gripper "
                               "touching tomato_sauce_1_main at 12.8 N over 64 control steps "
                               "(grasp/lift/transport/place) and alphabet_soup_1_main at 8.7 N over 67 control steps "
                               "(grasp/lift/transport). The static geom dump shows gripper0_right_gripper carries "
                               "exactly one collision geom, gripper0_hand_collision (the palm mesh), and no finger "
                               "geom, so this is NOT a classification error: it is a real non-finger body pressing on "
                               "the objects for most of the carry. It affects BOTH the requested object and the "
                               "non-requested object held at the fixed plan, so it is not caused by the force request.",
            "consistency_check": "across all 64 stored episodes, every T0 episode has >=1 forbidden contact at >=5 N "
                                 "and every T7 episode has none, which is consistent with the two causes above but "
                                 "does not by itself prove the class composition of the other 62 episodes.",
        },
        "limits": "class composition is established for 2 episodes (1 per task), re-executed deterministically. The "
                  "other 62 executed episodes have counts only; the E3 trial records did not store contact "
                  "identities. No threshold was changed and no failing init was individually repaired.",
    }
    (D / "attribution_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"counts": counts, "findings": out["findings"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
