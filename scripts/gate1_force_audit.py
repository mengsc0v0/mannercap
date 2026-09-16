"""Summarize recorded object wrench/contact evidence without claiming calibration."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def norm3(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def audit_episode(directory: Path, representative_count: int = 6) -> dict:
    rows = [json.loads(line) for line in (directory / "transitions.jsonl").open()]
    if not rows:
        raise ValueError(f"empty transition log: {directory}")
    chosen = sorted({round(index * (len(rows) - 1) / (representative_count - 1))
                     for index in range(representative_count)})
    summaries = {}
    unique_pairs = set()
    transitions_with_contact = 0
    for row in rows:
        evidence = row.get("contact_evidence_t_plus_1")
        if not evidence or evidence.get("sensor_valid") is not True:
            raise ValueError("missing or invalid contact evidence")
        contacts = evidence["object_contacts"]
        transitions_with_contact += bool(contacts)
        for contact in contacts:
            unique_pairs.add((contact["geom1_name"], contact["geom2_name"]))
        for item in evidence["body_aggregates"]:
            state = summaries.setdefault(item["object_id"], {
                "samples": 0,
                "body_force_norm_n_min": float("inf"),
                "body_force_norm_n_max": 0.0,
                "body_torque_norm_nm_max": 0.0,
                "contact_pair_force_norm_n_max": 0.0,
            })
            state["samples"] += 1
            force = norm3(item["force_n"])
            torque = norm3(item["torque_nm"])
            state["body_force_norm_n_min"] = min(state["body_force_norm_n_min"], force)
            state["body_force_norm_n_max"] = max(state["body_force_norm_n_max"], force)
            state["body_torque_norm_nm_max"] = max(state["body_torque_norm_nm_max"], torque)
        for contact in contacts:
            force = norm3(contact["force_n"])
            for object_id in contact["object_ids"]:
                summaries[object_id]["contact_pair_force_norm_n_max"] = max(
                    summaries[object_id]["contact_pair_force_norm_n_max"], force)

    samples = []
    for index in chosen:
        row = rows[index]
        evidence = row["contact_evidence_t_plus_1"]
        samples.append({
            "transition": index + 1,
            "sim_time_s": evidence["sim_time_s"],
            "phase": row.get("phase_program_marker"),
            "body_aggregates": evidence["body_aggregates"],
            "object_contacts": evidence["object_contacts"],
        })
    return {
        "task": directory.name,
        "transitions": len(rows),
        "transitions_with_object_contact": transitions_with_contact,
        "object_summaries": summaries,
        "unique_geom_pairs": [list(pair) for pair in sorted(unique_pairs)],
        "representative_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema": "mannercap-force-diagnostic-v1",
        "source_run": str(args.run_dir.resolve()),
        "body_wrench_layout": "torque_nm_then_force_n",
        "contact_wrench_layout": "force_n_then_torque_nm",
        "qualification": "unknown",
        "calibration_verified": False,
        "reason": "observational task traces are not controlled-load calibration",
        "episodes": [audit_episode(args.run_dir / task) for task in ("lift", "stack")],
    }
    output = args.output or args.run_dir / "force_diagnostic.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "representative_samples": sum(len(ep["representative_samples"]) for ep in result["episodes"]),
        "calibration_verified": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
