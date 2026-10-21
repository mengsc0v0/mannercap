#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
NEW_ROOT = os.path.abspath(os.path.join(BASE, ".."))
STAGECAP_ROOT = os.path.join(NEW_ROOT, "stagecap_robosuite_libero_repro_20260714")
sys.path.insert(0, NEW_ROOT)
sys.path.insert(0, STAGECAP_ROOT)

import run_manner_ablation as rma  # noqa: E402
from capx.envs.manner_validator import MannerValidator, StageStatus  # noqa: E402
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi  # noqa: E402
from synthesis_v2.la4vla_osc_adapter import adapt_pose_transition, quaternion_to_rotvec_wxyz  # noqa: E402


TASK = "stack"
DEFAULT_TAG = "phased_5stage"
DEFAULT_PROMPT = "先快速靠近，轻轻捧起，稳稳当当地端过去，慢慢放下，最后快速松手离开"
FPS = 20
IMG_SIZE = 224


@dataclasses.dataclass
class StepRow:
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    next_state: np.ndarray
    action7: np.ndarray
    timestamp: float
    dt: float
    frame_index: int
    sim_step_index: int
    terminated: bool
    truncated: bool
    success: bool
    clipped_dims: int


class EpisodeRecorder:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.rows: list[StepRow] = []
        self.clipped_steps = 0
        self._orig_do = env._do_robosuite_step

    def _eef_pose_wxyz_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self.env.robosuite_env.sim.data.xquat[self.env.gripper_link_idx], dtype=np.float64)
        p = np.asarray(self.env.robosuite_env.sim.data.xpos[self.env.gripper_link_idx], dtype=np.float64)
        return q, p

    def _state8_from_sim(self) -> np.ndarray:
        quat_wxyz, pos = self._eef_pose_wxyz_xyz()
        rotvec = np.asarray(quaternion_to_rotvec_wxyz(quat_wxyz.tolist()), dtype=np.float64)
        robs = self.env.robosuite_env._get_observations()
        gq = np.asarray(robs["robot0_gripper_qpos"], dtype=np.float64)
        return np.concatenate([pos, rotvec, gq], axis=0).astype(np.float32)

    def _render_rgb(self, camera: str) -> np.ndarray:
        frame = self.env.robosuite_env.sim.render(
            camera_name=camera,
            width=IMG_SIZE,
            height=IMG_SIZE,
            depth=False,
        )
        arr = np.asarray(frame, dtype=np.uint8)
        return arr[::-1].copy()

    def __enter__(self) -> "EpisodeRecorder":
        def patched(action: np.ndarray) -> None:
            sim = self.env.robosuite_env.sim
            t0 = float(sim.data.time)
            sim_idx = int(self.env._sim_step_count)

            # action-before observation contract
            image = self._render_rgb("agentview")
            wrist = self._render_rgb("robot0_eye_in_hand")
            state = self._state8_from_sim()
            quat0, pos0 = self._eef_pose_wxyz_xyz()

            self._orig_do(action)

            t1 = float(sim.data.time)
            quat1, pos1 = self._eef_pose_wxyz_xyz()
            next_state = self._state8_from_sim()

            # Record OSC-like 7D command adapted from measured EEF transition.
            # Last dim keeps the actual gripper command sent to robosuite.
            sliced = action[: self.env._ACTION_SLICE] if self.env._ACTION_SLICE != 0 else action
            grip_cmd = float(np.asarray(sliced, dtype=np.float64)[-1])
            adapted = adapt_pose_transition(
                position_before_m=pos0.tolist(),
                quaternion_before_wxyz=quat0.tolist(),
                position_after_m=pos1.tolist(),
                quaternion_after_wxyz=quat1.tolist(),
                gripper_command=grip_cmd,
                allow_clipping=True,
            )
            if adapted.clipped:
                self.clipped_steps += 1

            self.rows.append(
                StepRow(
                    image=image,
                    wrist_image=wrist,
                    state=state,
                    next_state=next_state,
                    action7=np.asarray(adapted.normalized_action, dtype=np.float32),
                    timestamp=t0,
                    dt=t1 - t0,
                    frame_index=len(self.rows),
                    sim_step_index=sim_idx,
                    terminated=False,
                    truncated=False,
                    success=bool(self.env.task_completed()),
                    clipped_dims=len(adapted.clipped_dimensions),
                )
            )

        self.env._do_robosuite_step = patched
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.env._do_robosuite_step = self._orig_do


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def try_git_rev(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def save_episode_files(ep_dir: Path, rows: list[StepRow], *, model_xml: str, replay_state: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, str, str]:
    ep_dir.mkdir(parents=True, exist_ok=True)

    images = np.stack([
        np.stack([row.image, row.wrist_image], axis=0) for row in rows
    ], axis=0).astype(np.uint8)
    states = np.stack([row.state for row in rows], axis=0).astype(np.float32)
    actions = np.stack([row.action7 for row in rows], axis=0).astype(np.float32)
    next_states = np.stack([row.next_state for row in rows], axis=0).astype(np.float32)

    timestamps = np.asarray([row.timestamp for row in rows], dtype=np.float64)
    dts = np.asarray([row.dt for row in rows], dtype=np.float64)
    frame_idx = np.asarray([row.frame_index for row in rows], dtype=np.int64)
    sim_idx = np.asarray([row.sim_step_index for row in rows], dtype=np.int64)
    terminated = np.asarray([row.terminated for row in rows], dtype=np.bool_)
    truncated = np.asarray([row.truncated for row in rows], dtype=np.bool_)
    success = np.asarray([row.success for row in rows], dtype=np.bool_)

    img_path = ep_dir / "images.npy"
    np.save(img_path, images, allow_pickle=False)

    trans_path = ep_dir / "transitions.npz"
    np.savez_compressed(
        trans_path,
        state=states,
        actions=actions,
        next_state=next_states,
        timestamp=timestamps,
        dt=dts,
        frame_index=frame_idx,
        sim_step_index=sim_idx,
        terminated=terminated,
        truncated=truncated,
        success=success,
    )

    with (ep_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    (ep_dir / "model.xml").write_text(model_xml, encoding="utf-8")
    np.savez_compressed(ep_dir / "replay_state.npz", **replay_state)

    return len(rows), sha256_file(img_path), sha256_file(trans_path)


def run_one_episode(*, env: Any, api: Any, seed: int, tag: str, prompt: str, max_repairs: int) -> tuple[list[StepRow], dict[str, Any], dict[str, Any]]:
    validator = MannerValidator()
    spec = rma.make_mock_spec(0, tag)
    params = rma.spec_to_exec_params(spec, task=TASK, tag=tag)

    env.reset(seed=seed)
    env.begin_manner_recording(clear=True)

    # replay initial state bundle
    sim = env.robosuite_env.sim
    replay_state = {
        "qpos": np.asarray(sim.data.qpos, dtype=np.float64),
        "qvel": np.asarray(sim.data.qvel, dtype=np.float64),
        "act": np.asarray(sim.data.act, dtype=np.float64),
        "time_s": np.asarray([float(sim.data.time)], dtype=np.float64),
        "qacc_warmstart": np.asarray(sim.data.qacc_warmstart, dtype=np.float64),
    }

    stages = rma.STAGE_BUILDERS[TASK](env, api)
    phase_list = [p for p, _ in stages]
    entry_ckpts: dict[str, Any] = {}
    cur_params = rma._copy_params(params)
    st: dict[str, Any] = {}
    repairs_used = 0
    stage_log: list[dict[str, Any]] = []

    with EpisodeRecorder(env) as rec:
        i = 0
        while i < len(stages):
            phase, fn = stages[i]
            entry_ckpts[phase] = env.capture_checkpoint()
            tele_start = len(env._manner_telemetry)
            fn(cur_params, st)

            seg = env._manner_telemetry[tele_start:]
            manner_ok = rma._validate_stage_manner(seg, phase, spec, validator, StageStatus)
            task_ok = rma._stage_task_predicate(env, api, TASK, phase, st)

            if manner_ok and task_ok:
                stage_log.append({"stage": phase, "manner_ok": True, "task_ok": True, "repaired": False})
                i += 1
                continue

            cause = "manner" if not manner_ok else "task"
            plan = None if repairs_used >= max_repairs else rma._stage_repair_plan(TASK, phase, cause, cur_params, spec)
            if plan is None:
                stage_log.append({"stage": phase, "manner_ok": bool(manner_ok), "task_ok": bool(task_ok), "repaired": False, "gave_up": True, "cause": cause})
                i += 1
                continue

            repairs_used += 1
            stage_log.append(
                {
                    "stage": phase,
                    "manner_ok": bool(manner_ok),
                    "task_ok": bool(task_ok),
                    "repaired": True,
                    "cause": cause,
                    "repair": plan["name"],
                    "rollback_to": plan["rollback_to"],
                }
            )
            target = plan["rollback_to"]
            env.restore_checkpoint(entry_ckpts[target])
            cur_params = plan["params"]
            i = phase_list.index(target)

    env.end_manner_recording()
    result = rma._finalize_staged_result(TASK, env, api, st)

    exec_summary = {
        "spec_label": spec.label,
        "prompt": prompt,
        "seed": seed,
        "tag": tag,
        "params": {
            "v_max": float(cur_params.get("v_max", 0.0)),
            "a_max": float(cur_params.get("a_max", 0.0)),
            "jerk_limit": None if cur_params.get("jerk_limit") is None else float(cur_params.get("jerk_limit")),
            "kp_pos": float(cur_params.get("kp_pos", 0.0)),
            "grip": float(cur_params.get("grip", 0.0)),
            "phase_v_max": {k: float(v) for k, v in (cur_params.get("phase_v_max") or {}).items()},
            "phase_a_max": {k: float(v) for k, v in (cur_params.get("phase_a_max") or {}).items()},
        },
        "result": {
            "task_completed": bool(result.get("task_completed", False)),
            "reward": float(result.get("reward", 0.0)),
            "z_gap": None if result.get("z_gap") is None else float(result.get("z_gap")),
            "red_delta_z": None if result.get("red_delta_z") is None else float(result.get("red_delta_z")),
            "repairs_used": repairs_used,
            "stage_log": stage_log,
        },
        "episode_metrics": {
            "steps": len(rec.rows),
            "clipped_steps": rec.clipped_steps,
            "task_ok": bool(result.get("task_completed", False)),
            "manner_ok": bool(result.get("task_completed", False) and repairs_used <= max_repairs),
        },
    }

    model_xml = env.robosuite_env.sim.model.get_xml()
    return rec.rows, exec_summary, {"model_xml": model_xml, "replay_state": replay_state}


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture stack demo trajectories in pi05 dataset format")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    dataset_dir = out_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    entry = rma.TASK_REGISTRY[TASK]
    sim_cls = rma.load_sim_cls(entry["sim_cls_path"])
    env = sim_cls(privileged=True, enable_render=True, max_steps=entry["max_steps"], seed=args.seed_start)
    api = FrankaControlPrivilegedApi(env)

    run_stamp = subprocess.check_output(["date", "+%Y%m%dT%H%M%S"], text=True).strip()
    source_manifest = {
        "schema": "pi05-stack-rgb-v1",
        "generator": "scripts/capture_stack_pi05_dataset.py",
        "generated_at": run_stamp,
        "git_commit": try_git_rev(Path(NEW_ROOT)),
        "task": TASK,
        "tag": args.tag,
        "prompt": args.prompt,
        "episodes_requested": int(args.episodes),
        "seed_start": int(args.seed_start),
        "controller": "panda_joint_ctrl (source), adapted-to-osc7 labels via measured EEF transition",
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "image_size": IMG_SIZE,
        "fps": FPS,
        "notes": [
            "state/images are action-before snapshots",
            "actions are 7D normalized OSC-like labels from pose transition algebra with clipping allowed",
        ],
    }

    source_manifest_path = dataset_dir / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    source_sha = sha256_file(source_manifest_path)

    episodes_manifest: list[dict[str, Any]] = []
    episodes_jsonl_path = dataset_dir / "episodes.jsonl"
    if episodes_jsonl_path.exists():
        episodes_jsonl_path.unlink()

    replay_rows: list[dict[str, Any]] = []

    try:
        for i in range(args.episodes):
            ep_idx = i + 1
            seed = args.seed_start + i
            ep_name = f"episode_{ep_idx:06d}"
            ep_dir = dataset_dir / ep_name
            split = "train" if i < int(args.episodes * args.train_ratio) else "valid"

            rows, summary, runtime = run_one_episode(
                env=env,
                api=api,
                seed=seed,
                tag=args.tag,
                prompt=args.prompt,
                max_repairs=args.max_repairs,
            )

            metadata = {
                "episode": ep_name,
                "task": TASK,
                "manner_tag": args.tag,
                "prompt": args.prompt,
                "seed": seed,
                "task_ok": summary["episode_metrics"]["task_ok"],
                "manner_ok": summary["episode_metrics"]["manner_ok"],
                "repairs_used": summary["result"]["repairs_used"],
                "stage_log": summary["result"]["stage_log"],
                "clipped_steps": summary["episode_metrics"]["clipped_steps"],
                "step_count": summary["episode_metrics"]["steps"],
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "state_layout": ["eef_x", "eef_y", "eef_z", "rotvec_x", "rotvec_y", "rotvec_z", "finger_qpos_0", "finger_qpos_1"],
                "action_layout": ["u_dx", "u_dy", "u_dz", "u_drx", "u_dry", "u_drz", "u_gripper"],
                "control_freq_hz": FPS,
                "dt_s": 1.0 / FPS,
                "source": summary,
            }

            frames, img_sha, trans_sha = save_episode_files(
                ep_dir,
                rows,
                model_xml=runtime["model_xml"],
                replay_state=runtime["replay_state"],
                metadata=metadata,
            )

            item = {
                "episode": ep_name,
                "split": split,
                "frames": frames,
                "images": f"{ep_name}/images.npy",
                "transitions": f"{ep_name}/transitions.npz",
                "images_sha256": img_sha,
                "transitions_sha256": trans_sha,
                "prompt": args.prompt,
                "task_name": TASK,
                "manner_tag": args.tag,
                "seed": seed,
                "reset_id": f"reset_{ep_idx:06d}",
                "source_attempt_id": f"attempt_{ep_idx:06d}",
                "lineage_id": f"source_{ep_idx:06d}",
                "task_ok": bool(summary["episode_metrics"]["task_ok"]),
                "manner_ok": bool(summary["episode_metrics"]["manner_ok"]),
                "metadata": f"{ep_name}/metadata.json",
            }
            episodes_manifest.append(item)
            with episodes_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            replay_rows.append(
                {
                    "episode": ep_name,
                    "seed": seed,
                    "task_ok": bool(summary["episode_metrics"]["task_ok"]),
                    "manner_ok": bool(summary["episode_metrics"]["manner_ok"]),
                    "frames": frames,
                    "clipped_steps": int(summary["episode_metrics"]["clipped_steps"]),
                    "repairs_used": int(summary["result"]["repairs_used"]),
                }
            )

            print(f"[{ep_idx:03d}/{args.episodes}] {ep_name} seed={seed} frames={frames} task_ok={item['task_ok']} manner_ok={item['manner_ok']} repairs={summary['result']['repairs_used']}")

    finally:
        env.close()

    manifest = {
        "schema": "pi05-stack-rgb-v1",
        "complete": True,
        "source_sha256": source_sha,
        "task": args.prompt,
        "fps": FPS,
        "cameras": ["agentview", "robot0_eye_in_hand"],
        "image_convention": "top_down_rgb",
        "state": "eef_xyz, eef_axisangle, two_gripper_qpos",
        "action": "world_delta_osc7",
        "episodes": episodes_manifest,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(episodes_manifest)
    task_ok = sum(1 for x in replay_rows if x["task_ok"])
    manner_ok = sum(1 for x in replay_rows if x["manner_ok"])
    replay_audit = {
        "schema": "pi05-replay-audit-v1",
        "source_sha256": source_sha,
        "episodes": replay_rows,
        "summary": {
            "episodes": total,
            "task_ok": task_ok,
            "manner_ok": manner_ok,
            "task_ok_rate": 0.0 if total == 0 else task_ok / total,
            "manner_ok_rate": 0.0 if total == 0 else manner_ok / total,
        },
        "passed": bool(total > 0),
        "notes": [
            "This audit summarizes executed collection runs under the stack staged policy.",
            "For strict physical replay from saved initial state + action sequence, run an external replay checker on this dataset.",
        ],
    }
    (dataset_dir / "replay_audit.json").write_text(json.dumps(replay_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_root / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file():
                f.write(f"{sha256_file(p)}  {p.relative_to(out_root).as_posix()}\n")

    print("[done] dataset root:", str(dataset_dir))
    print("[done] episodes:", len(episodes_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
