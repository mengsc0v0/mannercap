"""Build dataset/manifest.json and dataset/source_manifest.json from the
per-episode records in episodes.jsonl, per agent_handoff.md sec 6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_stack_common import (  # noqa: E402
    OSC_CONTROLLER_CFG, TASK_PROMPT, CONTROL_FREQ, CAM_MAIN, CAM_WRIST, IMG_SIZE,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_versions():
    import numpy, robosuite, scipy
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except Exception:  # noqa: BLE001
        mujoco_version = "unknown"
    return dict(
        python=platform.python_version(),
        robosuite=robosuite.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        mujoco=mujoco_version,
    )


def get_git_commit(repo_dir: str):
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-root", required=True,
                     help="repo root used to look up git commit for source_manifest")
    args = ap.parse_args()

    episodes_jsonl = os.path.join(args.dataset_root, "episodes.jsonl")
    records = [json.loads(l) for l in open(episodes_jsonl) if l.strip()]

    n_train = sum(1 for r in records if r["split"] == "train")
    n_valid = sum(1 for r in records if r["split"] == "valid")
    total_frames = sum(r["frames"] for r in records)

    failed_path = os.path.join(args.dataset_root, "..", "failed_attempts.jsonl")
    failed_path = os.path.normpath(failed_path)
    n_failed = 0
    if os.path.exists(failed_path):
        n_failed = sum(1 for l in open(failed_path) if l.strip())

    # --- source_manifest.json (write first, no self-reference) ---
    source_manifest = dict(
        generator_script="scripts/pi05_capture_batch.py + pi05_capture_episode.py + pi05_stack_common.py",
        git_commit=get_git_commit(args.repo_root),
        software_versions=get_versions(),
        task="stack",
        manner_tag="phased_5stage",
        prompt=TASK_PROMPT,
        prompt_zh="先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开",
        controller_configs=OSC_CONTROLLER_CFG,
        control_freq=CONTROL_FREQ,
        cameras=[CAM_MAIN, CAM_WRIST],
        image_size=IMG_SIZE,
        randomization=dict(
            sampler="UniformRandomSampler",
            x_range=[-0.18, 0.18], y_range=[-0.12, 0.12],
            z_offset=0.01, ensure_valid_placement=True,
            rng_source="np.random.default_rng(seed); env.placement_initializer.rng reassigned after construction",
        ),
        seed_range=dict(
            seed_start=1,
            n_success=len(records),
            n_failed_attempts=n_failed,
            success_rate=f"{len(records)}/{len(records) + n_failed}",
        ),
        gripper_sign_convention="+1.0 closes gripper (qpos separation shrinks toward 0), -1.0 opens (grows); empirically verified via qpos before/after action.",
        image_orientation_note=(
            "obs[cam+'_image'] via use_camera_obs observable path equals raw "
            "env.sim.render() output bit-for-bit under default macros.IMAGE_CONVENTION="
            "'opengl' (no flip applied by the sensor wrapper). This raw output is "
            "vertically upside-down (confirmed against a known-correct reference "
            "frame from this repo's legacy direct-render+flip video pipeline via "
            "row-brightness-profile comparison). We apply [::-1] (vertical flip) "
            "in EpisodeRecorder.record_step() before saving to images.npy, so the "
            "delivered images ARE upright/positive per handoff sec 3."
        ),
        manner_realization_note=(
            "The fixed-kp=150 OSC_POSE interface does not expose variable stiffness "
            "or grip force in the 7D action space. The demo narrative "
            "(fast-approach/gentle-lift/steady-transport/slow-place/fast-retreat) is "
            "realized purely via per-phase Cartesian speed caps (v_max, w_max) on "
            "the position/orientation servo, which stays within the fixed 7D "
            "interface. manner_ok in metadata/manifest is a DESIGN-INTENT claim "
            "(phase order + speed caps matching the narrative), not an independent "
            "quantitative manner-classifier score computed against this OSC "
            "re-execution."
        ),
        not_yet_verified=[
            "No independent quantitative manner-similarity metric computed against "
            "this OSC re-execution (manner_ok reflects design intent only).",
            "Replay audit was NOT run as part of this delivery; it is left to the "
            "downstream integrator (LA4VLA side) to run tools/pi05_replay_audit.py "
            "and produce dataset/replay_audit.json before claiming actions are "
            "verified-replayable.",
        ],
    )
    source_manifest_path = os.path.join(args.dataset_root, "source_manifest.json")
    with open(source_manifest_path, "w") as f:
        json.dump(source_manifest, f, indent=2, ensure_ascii=False)
    source_sha256 = sha256_file(source_manifest_path)

    # --- manifest.json ---
    manifest = dict(
        schema="pi05-stack-rgb-v1",
        complete=True,
        source_sha256=source_sha256,
        task=TASK_PROMPT,
        fps=CONTROL_FREQ,
        cameras=[CAM_MAIN, CAM_WRIST],
        image_convention="rgb_hwc_uint8_0_255_upright",
        state="eef_xyz, eef_axisangle, two_gripper_qpos",
        action="world_delta_osc7",
        n_episodes=len(records),
        n_train=n_train,
        n_valid=n_valid,
        total_frames=total_frames,
        n_failed_attempts=n_failed,
        success_rate_str=f"{len(records)}/{len(records) + n_failed}",
        per_task_manner_counts={"stack__phased_5stage": len(records)},
        episodes=records,
    )
    manifest_path = os.path.join(args.dataset_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "n_episodes": len(records), "n_train": n_train, "n_valid": n_valid,
        "total_frames": total_frames, "n_failed_attempts": n_failed,
        "source_sha256": source_sha256,
    }))


if __name__ == "__main__":
    main()
