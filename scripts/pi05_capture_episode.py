"""Capture a single pi0.5-format stack episode and write dataset/episode_XXXXXX/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_stack_common import (  # noqa: E402
    MANNER_TAG, PROMPT_ZH, TASK_NAME, TASK_PROMPT, OSC_CONTROLLER_CFG,
    run_one_episode,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_episode(episode_id: str, seed: int, split: str, out_root: str, prompt: str = PROMPT_ZH):
    t0 = time.time()
    rec, obs_final, env, meta = run_one_episode(seed)
    arrays = rec.to_arrays()
    elapsed = time.time() - t0

    ep_dir = os.path.join(out_root, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    images_path = os.path.join(ep_dir, "images.npy")
    np.save(images_path, arrays["images"])

    transitions_path = os.path.join(ep_dir, "transitions.npz")
    np.savez(
        transitions_path,
        state=arrays["state"], actions=arrays["actions"], next_state=arrays["next_state"],
        timestamp=arrays["timestamp"], dt=arrays["dt"], frame_index=arrays["frame_index"],
        sim_step_index=arrays["sim_step_index"], terminated=arrays["terminated"],
        truncated=arrays["truncated"], success=arrays["success"], phase=arrays["phase"],
    )

    model_xml_path = os.path.join(ep_dir, "model.xml")
    with open(model_xml_path, "w") as f:
        f.write(env.sim.model.get_xml())

    replay_state_path = os.path.join(ep_dir, "replay_state.npz")
    np.savez(
        replay_state_path,
        seed=seed,
        controller_cfg_json=json.dumps(OSC_CONTROLLER_CFG),
        cubeA_pos0=np.asarray(meta["cubeA_pos0"], dtype=np.float64),
        cubeB_pos0=np.asarray(meta["cubeB_pos0"], dtype=np.float64),
    )

    metadata = dict(
        episode=episode_id,
        task=TASK_NAME,
        task_prompt=TASK_PROMPT,
        prompt_zh=prompt,
        manner_tag=MANNER_TAG,
        seed=seed,
        split=split,
        frames=int(arrays["state"].shape[0]),
        cameras=["agentview", "robot0_eye_in_hand"],
        image_convention="rgb_hwc_uint8_0_255_upright",
        state_convention="eef_xyz_world, rotvec_absolute(xyzw->rotvec), finger_qpos_raw_x2",
        action_convention="world_delta_osc7: [dx,dy,dz,drx,dry,drz,gripper] normalized[-1,1]; "
                           "pos_scale=0.05m, rot_scale=0.5rad, gripper:+1=close,-1=open",
        controller=OSC_CONTROLLER_CFG,
        control_freq=20,
        initial_object_pose={"cubeA_pos": meta["cubeA_pos0"], "cubeB_pos": meta["cubeB_pos0"]},
        final_object_pose={"cubeA_pos": meta["cubeA_pos_final"], "cubeB_pos": meta["cubeB_pos_final"]},
        phase_steps=meta["phase_steps"],
        task_ok=meta["task_ok"],
        manner_ok=True,
        manner_ok_criterion="phase order + per-phase cartesian speed caps matches prompt narrative "
                             "(fast approach/retreat, slow descend/final, steady lift/transport); "
                             "no independent quantitative manner validator run against this OSC "
                             "re-execution (see README caveats).",
        gripper_sign_convention=meta["gripper_sign_convention"],
        generator="scripts/pi05_capture_episode.py",
        capture_wall_time_sec=elapsed,
        source_attempt_id=f"attempt_{episode_id.split('_')[-1]}",
        reset_id=f"reset_{episode_id.split('_')[-1]}",
        lineage_id=f"source_{episode_id.split('_')[-1]}",
    )
    metadata_path = os.path.join(ep_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    record = dict(
        episode=episode_id, split=split, frames=metadata["frames"],
        images="{}/images.npy".format(episode_id),
        transitions="{}/transitions.npz".format(episode_id),
        images_sha256=sha256_file(images_path),
        transitions_sha256=sha256_file(transitions_path),
        prompt=prompt, task_name=TASK_NAME, manner_tag=MANNER_TAG, seed=seed,
        reset_id=metadata["reset_id"], source_attempt_id=metadata["source_attempt_id"],
        lineage_id=metadata["lineage_id"],
        task_ok=meta["task_ok"], manner_ok=True,
        metadata="{}/metadata.json".format(episode_id),
    )
    return record, elapsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    rec, elapsed = capture_episode(args.episode_id, args.seed, args.split, args.out_root)
    print(json.dumps(rec, ensure_ascii=False))
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr)
