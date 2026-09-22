"""Run 12 controlled MuJoCo wrench-channel checks on a restored Lift scene."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def subtract(left, right):
    return [float(a) - float(b) for a, b in zip(left, right)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    directory = args.episode_dir.resolve()
    initial = json.loads((directory / "initial_state.json").read_text())
    result = json.loads((directory / "result.json").read_text())

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "stagecap_robosuite_libero_repro_20260714")]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(root / "stagecap_robosuite_libero_repro_20260714")

    import mujoco
    import numpy as np
    import run_manner_ablation as old
    from synthesis_v2.mujoco_contacts import capture_object_contacts, resolve_task_object_registry
    from synthesis_v2.simulator_state import restore_simulator_snapshot

    if result["task"] != "lift":
        raise ValueError("controlled calibration currently requires a Lift episode")
    entry = old.TASK_REGISTRY["lift"]
    env = old.load_sim_cls(entry["sim_cls_path"])(
        privileged=True, enable_render=False, max_steps=entry["max_steps"], seed=result["seed"])
    scenarios = [
        ("table_support_zero", False, (0, 0, 0), (0, 0, 0)),
        ("airborne_zero", True, (0, 0, 0), (0, 0, 0)),
        ("airborne_fx_1n", True, (1, 0, 0), (0, 0, 0)),
        ("airborne_fx_2n", True, (2, 0, 0), (0, 0, 0)),
        ("airborne_fx_4n", True, (4, 0, 0), (0, 0, 0)),
        ("airborne_fz_minus1n", True, (0, 0, -1), (0, 0, 0)),
        ("airborne_fz_minus2n", True, (0, 0, -2), (0, 0, 0)),
        ("airborne_fz_minus4n", True, (0, 0, -4), (0, 0, 0)),
        ("airborne_tz_0p05nm", True, (0, 0, 0), (0, 0, 0.05)),
        ("airborne_tz_0p1nm", True, (0, 0, 0), (0, 0, 0.1)),
        ("airborne_tz_0p2nm", True, (0, 0, 0), (0, 0, 0.2)),
        ("airborne_combined", True, (1, 2, 3), (0.1, 0.2, 0.3)),
    ]
    samples = []
    try:
        env.robosuite_env.reset_from_xml_string((directory / "model.xml").read_text())
        registry = resolve_task_object_registry(env.robosuite_env, "lift")
        binding = registry[0]
        cube_joint = env.robosuite_env.cube.joints[0]
        joint_id = int(env.robosuite_env.sim.model.joint_name2id(cube_joint))
        qpos_address = int(env.robosuite_env.sim.model.jnt_qposadr[joint_id])
        for case_id, airborne, applied_force, applied_torque in scenarios:
            restore_simulator_snapshot(env, initial["snapshot"])
            data = env.robosuite_env.sim.data
            if airborne:
                data.qpos[qpos_address + 2] += 0.30
                data.qvel[:] = 0
            data.xfrc_applied[:] = 0
            data.xfrc_applied[binding["body_id"], :3] = applied_force
            data.xfrc_applied[binding["body_id"], 3:] = applied_torque
            mujoco.mj_forward(env.robosuite_env.sim.model._model, data._data)
            snapshot = capture_object_contacts(env.robosuite_env.sim, registry)
            body = snapshot["body_aggregates"][0]
            samples.append({
                "case_id": case_id,
                "airborne": airborne,
                "applied_force_n": list(applied_force),
                "applied_torque_nm": list(applied_torque),
                "measured_body_force_n": body["force_n"],
                "measured_body_torque_nm": body["torque_nm"],
                "object_contact_count": len(snapshot["object_contacts"]),
                "sensor_valid": True,
            })

        airborne_zero = next(item for item in samples if item["case_id"] == "airborne_zero")
        for item in samples:
            if not item["airborne"]:
                item["channel_response_checked"] = False
                continue
            force_delta = subtract(item["measured_body_force_n"], airborne_zero["measured_body_force_n"])
            torque_delta = subtract(item["measured_body_torque_nm"], airborne_zero["measured_body_torque_nm"])
            item["force_response_delta_n"] = force_delta
            item["torque_response_delta_nm"] = torque_delta
            item["force_response_error_n"] = norm(subtract(force_delta, item["applied_force_n"]))
            item["torque_response_error_nm"] = norm(subtract(torque_delta, item["applied_torque_nm"]))
            item["channel_response_checked"] = True
        checked = [item for item in samples if item["channel_response_checked"]]
        channel_passed = all(
            item["object_contact_count"] == 0
            and item["force_response_error_n"] <= args.tolerance
            and item["torque_response_error_nm"] <= args.tolerance
            for item in checked
        )
        contact_samples = []
        for load_n in (0.0, 1.0, 2.0, 4.0):
            restore_simulator_snapshot(env, initial["snapshot"])
            data = env.robosuite_env.sim.data
            data.xfrc_applied[:] = 0
            data.xfrc_applied[binding["body_id"], :3] = (0.0, 0.0, -load_n)
            mujoco.mj_forward(env.robosuite_env.sim.model._model, data._data)
            snapshot = capture_object_contacts(env.robosuite_env.sim, registry)
            table_contacts = [
                contact for contact in snapshot["object_contacts"]
                if "table_collision" in (contact["geom1_name"], contact["geom2_name"])
            ]
            contact_samples.append({
                "case_id": f"table_downward_load_{load_n:g}n",
                "applied_downward_force_n": load_n,
                "table_contact_count": len(table_contacts),
                "summed_normal_force_n": sum(abs(contact["force_n"][0]) for contact in table_contacts),
                "contact_force_frame": "contact_local",
                "normal_component_index": 0,
            })
        baseline_normal = contact_samples[0]["summed_normal_force_n"]
        for item in contact_samples:
            response = item["summed_normal_force_n"] - baseline_normal
            item["normal_force_response_delta_n"] = response
            item["normal_force_response_error_n"] = abs(response - item["applied_downward_force_n"])
            item["normal_force_response_gain"] = (
                None if item["applied_downward_force_n"] == 0
                else response / item["applied_downward_force_n"]
            )
        contact_characterized = all(item["table_contact_count"] > 0 for item in contact_samples)
        contact_passed = all(
            item["table_contact_count"] > 0
            and item["normal_force_response_error_n"] <= args.tolerance
            for item in contact_samples
        )
        report = {
            "schema": "mannercap-controlled-wrench-calibration-v1",
            "source_episode": str(directory),
            "sample_count": len(samples) + len(contact_samples),
            "body_wrench_layout": "torque_nm_then_force_n",
            "applied_wrench_layout": "force_n_then_torque_nm",
            "tolerance": args.tolerance,
            "body_channel_response_verified": channel_passed,
            "table_contact_normal_response_characterized": contact_characterized,
            "table_contact_normal_strict_tolerance_verified": contact_passed,
            "contact_pair_calibration_verified": contact_passed,
            "gripper_contact_calibration_verified": False,
            "real_sensor_calibration_verified": False,
            "physical_qualification": "unknown",
            "samples": samples,
            "contact_normal_samples": contact_samples,
        }
        output = args.output or directory.parent / "controlled_wrench_calibration.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({
            "output": str(output),
            "sample_count": len(samples) + len(contact_samples),
            "body_channel_response_verified": channel_passed,
            "table_contact_normal_response_characterized": contact_characterized,
            "table_contact_normal_strict_tolerance_verified": contact_passed,
            "maximum_force_response_error_n": max(item["force_response_error_n"] for item in checked),
            "maximum_torque_response_error_nm": max(item["torque_response_error_nm"] for item in checked),
            "maximum_contact_normal_response_error_n": max(item["normal_force_response_error_n"] for item in contact_samples),
            "maximum_contact_normal_relative_error": max(
                item["normal_force_response_error_n"] / item["applied_downward_force_n"]
                for item in contact_samples if item["applied_downward_force_n"] > 0
            ),
        }))
        return 0 if channel_passed and contact_passed else 1
    finally:
        env.robosuite_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
