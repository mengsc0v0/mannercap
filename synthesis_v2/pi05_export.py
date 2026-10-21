"""Package audited accepted episodes as a pi0.5 student dataset.

Reads only the on-disk output of ``production_pipeline`` (requests.jsonl,
attempts.jsonl, raw/<attempt>/...).  An episode is exported only if (a) the
ledger says ``accepted``, (b) its stored audit says overall ``pass`` and (c) a
real action replay in the deployment controller matches the stored states
(5 mm position threshold).  Everything else is written to ``failures/`` with
its reason; nothing is silently dropped.

Layout (see agent_handoff.md):
  <root>/dataset/{manifest.json, episodes.jsonl, source_manifest.json,
                  replay_audit.json, episode_XXXXXX/{images.npy, transitions.npz,
                  metadata.json, model.xml, replay_state.npz}}
  <root>/failures/{failures.jsonl, <attempt>/...}
  <root>/tools/, requirements.txt, README.md, SHA256SUMS
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from synthesis_v2.episode_audit import AUDIT_VERSION
from synthesis_v2.manner_request import semantics_hash
from synthesis_v2.native_osc_env import CONTROLLER_CONFIG, CONTROLLER_CONFIG_SHA256, environment_config
from synthesis_v2.native_producer import PRODUCER_VERSION
from synthesis_v2.stage_table import stage_table_hash

PACKAGE_SCHEMA = "pi05_dataset.v2"
GRIPPER_SIGN = {"close": 1.0, "open": -1.0, "hold": 0.0}
DTYPES = {"state": np.float32, "actions": np.float32, "next_state": np.float32, "timestamp": np.float64,
          "dt": np.float64, "frame_index": np.int64, "sim_step_index": np.int64,
          "terminated": np.bool_, "truncated": np.bool_, "success": np.bool_}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def assign_splits(seeds: Sequence[int], val_frac: float = 0.15, test_frac: float = 0.15) -> dict[int, str]:
    """Seed-group split: all prompts/tasks of one seed share a split."""
    uniq = sorted(set(int(s) for s in seeds), key=lambda s: hashlib.sha256(f"split:{s}".encode()).hexdigest())
    n = len(uniq)
    n_test = max(1, round(n * test_frac)) if n >= 4 else 0
    n_val = max(1, round(n * val_frac)) if n >= 4 else 0
    out: dict[int, str] = {}
    for i, s in enumerate(uniq):
        out[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def _cast_transitions(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(np.asarray(arrays[k]).astype(dt)) for k, dt in DTYPES.items()}


def export_package(
    pipeline_dir: str | Path,
    package_root: str | Path,
    *,
    cal: Mapping[str, Any],
    replay: Callable[..., dict],
    repo_root: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    src, root = Path(pipeline_dir), Path(package_root)
    if root.exists():
        raise FileExistsError(f"{root} exists; refuse to overwrite a delivery package")
    ds, fail = root / "dataset", root / "failures"
    ds.mkdir(parents=True)
    fail.mkdir(parents=True)
    reqs = {r["request_id"]: r for r in _jl(src / "requests.jsonl")}
    atts = _jl(src / "attempts.jsonl")
    accepted = {a["request_id"]: a for a in atts if a["outcome"] == "accepted"}
    seeds = [reqs[r]["initial_state"]["seed"] for r in accepted]
    split_of = assign_splits(seeds, val_frac, test_frac)

    episodes: list[dict] = []
    replay_rows: list[dict] = []
    failures: list[dict] = []
    src_rows: list[dict] = []

    def add_failure(att: Mapping[str, Any], reason: str) -> None:
        raw = src / att["metadata"]["raw_dir"] if att.get("metadata", {}).get("raw_dir") else None
        d = fail / att["attempt_id"].replace(":", "_")
        if raw is not None and raw.exists():
            d.mkdir(parents=True, exist_ok=True)
            for f in ("trajectory.npz", "audit.json", "plan.json", "episode_meta.json"):
                if (raw / f).exists():
                    shutil.copy2(raw / f, d / f)
        failures.append({"attempt_id": att["attempt_id"], "request_id": att["request_id"], "outcome": att["outcome"],
                         "reason": reason, "audit_overall": att.get("metadata", {}).get("audit_overall"),
                         "dir": d.name if d.exists() else None})

    for att in atts:
        if att["outcome"] != "accepted":
            add_failure(att, att.get("reason", ""))

    for rid, att in accepted.items():
        req = reqs[rid]
        raw = src / att["metadata"]["raw_dir"]
        audit = json.loads((raw / "audit.json").read_text())
        emeta = json.loads((raw / "episode_meta.json").read_text())
        arrays = dict(np.load(raw / "trajectory.npz", allow_pickle=False))
        tr = _cast_transitions(arrays)
        task, seed = req["task"], int(req["initial_state"]["seed"])
        if audit["overall"] != "pass":
            add_failure(att, "accepted_in_ledger_but_stored_audit_not_pass")
            continue
        rep = replay(task, seed, tr["actions"], tr["next_state"], bool(tr["success"][-1]), want_assets=True)
        states = rep.pop("_states")
        xml, cams, init = rep.pop("model_xml"), rep.pop("camera_parameters"), rep.pop("initial")
        rep_row = {"request_id": rid, "attempt_id": att["attempt_id"], **rep}
        replay_rows.append(rep_row)
        if not rep["pass"]:
            add_failure(att, f"replay_mismatch:pos={rep['max_pos_err_mm']:.2f}mm:success_match={rep['success_match']}")
            log(f"replay FAILED {att['attempt_id']} {rep['max_pos_err_mm']:.2f} mm")
            continue
        eid = f"episode_{len(episodes):06d}"
        ed = ds / eid
        ed.mkdir()
        imgs = np.load(raw / "images.npy", allow_pickle=False)
        np.save(ed / "images.npy", imgs)
        np.savez(ed / "transitions.npz", **tr)
        (ed / "model.xml").write_text(xml, encoding="utf-8")
        np.savez(ed / "replay_state.npz", replayed_next_state=states,
                 initial_qpos=init["qpos"], initial_qvel=init["qvel"], initial_time=np.float64(init["time"]))
        md = req["metadata"]
        manner_tag = req["manner_key"]
        meta = {
            "episode": eid, "prompt": req["raw_language"], "base_instruction": md.get("base_instruction", ""),
            "task_name": task, "manner_tag": manner_tag, "seed": seed, "reset_id": req["scene_id"],
            "source_attempt_id": att["attempt_id"], "lineage_id": rid, "request_id": rid,
            "plan_hash": att["metadata"]["plan_hash"], "candidate": emeta["candidate"],
            "environment_config": environment_config(task, False), "controller_sha256": CONTROLLER_CONFIG_SHA256,
            "gripper_sign": GRIPPER_SIGN, "camera_parameters": cams, "audit": audit,
            "audit_version": AUDIT_VERSION, "calibration_sha256": cal["content_sha256"],
            "replay": {k: v for k, v in rep.items()}, "producer_version": PRODUCER_VERSION,
            "intents": audit["request"]["intents"],
        }
        (ed / "metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True, default=_jd), encoding="utf-8")
        rec = {"episode": eid, "prompt": req["raw_language"], "task_name": task, "manner_tag": manner_tag, "seed": seed,
               "reset_id": req["scene_id"], "source_attempt_id": att["attempt_id"], "lineage_id": rid,
               "task_ok": audit["task"] == "pass", "manner_ok": audit["manner"] in ("pass", "not_applicable"),
               "images_sha256": sha256_file(ed / "images.npy"), "transitions_sha256": sha256_file(ed / "transitions.npz"),
               "split": split_of[seed], "frames": int(len(tr["actions"])),
               "pair_key": md.get("pair_key"), "base_instruction": md.get("base_instruction", "")}
        for k in ("language_style", "label_source"):  # present only for registered-label requests
            if md.get(k) is not None:
                rec[k] = md[k]
        episodes.append(rec)
        src_rows.append({"episode": eid, "request_id": rid, "attempt_id": att["attempt_id"],
                         "raw_trajectory_sha256": sha256_file(raw / "trajectory.npz"),
                         "n_attempts_for_request": sum(1 for a in atts if a["request_id"] == rid)})
        log(f"exported {eid} {task} seed={seed} '{req['raw_language']}' replay={rep['max_pos_err_mm']:.2f}mm")

    (ds / "episodes.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in episodes) + ("\n" if episodes else ""))
    (ds / "source_manifest.json").write_text(json.dumps(
        {"pipeline_dir": src.name, "n_requests": len(reqs), "n_attempts": len(atts), "episodes": src_rows}, indent=1))
    (ds / "replay_audit.json").write_text(json.dumps({
        "n_replayed": len(replay_rows), "n_pass": sum(1 for r in replay_rows if r["pass"]),
        "pos_tol_mm": 5.0, "episodes": replay_rows}, indent=1, default=_jd))
    (fail / "failures.jsonl").write_text("\n".join(json.dumps(f, sort_keys=True) for f in failures) + ("\n" if failures else ""))

    groups: dict[tuple, list[dict]] = {}
    for e in episodes:
        groups.setdefault((e["task_name"], e["seed"], e["base_instruction"]), []).append(e)
    manifest = {
        "schema": PACKAGE_SCHEMA, "n_episodes": len(episodes), "n_failed_attempts": len(failures),
        "n_requests": len(reqs), "n_requests_without_accepted": len(reqs) - len(accepted),
        "split_counts": {s: sum(1 for e in episodes if e["split"] == s) for s in ("train", "val", "test")},
        "split_seeds": {s: sorted({e["seed"] for e in episodes if e["split"] == s}) for s in ("train", "val", "test")},
        "pairs": {"n_groups": len(groups), "n_groups_with_neutral_and_manner": sum(
            1 for g in groups.values() if any(x["manner_tag"] == "neutral" for x in g) and any(x["manner_tag"] != "neutral" for x in g))},
        "manner_tag_counts": {t: sum(1 for e in episodes if e["manner_tag"] == t) for t in sorted({e["manner_tag"] for e in episodes})},
        "controller_config": CONTROLLER_CONFIG, "controller_sha256": CONTROLLER_CONFIG_SHA256,
        "calibration_sha256": cal["content_sha256"], "audit_version": AUDIT_VERSION, "semantics_hash": semantics_hash(),
        "stage_table_hash": stage_table_hash(), "producer_version": PRODUCER_VERSION,
        "control_hz": 20, "gripper_sign": GRIPPER_SIGN, "image_shape": [2, 224, 224, 3],
        "state_dim": 8, "action_dim": 7,
    }
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=_jd))
    _write_tools(root, Path(repo_root))
    if any(e["task_name"] == "wipe" for e in episodes):
        with open(root / "README.md", "a", encoding="utf-8") as f:
            f.write(_WIPE_README)
    write_sha256sums(root)
    return manifest


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


_README = """# pi0.5 manner dataset ({schema})

* dataset/: episodes (images.npy, transitions.npz, metadata.json, model.xml, replay_state.npz), episodes.jsonl, manifest.json, replay_audit.json
* failures/: every non-exported attempt with reason (kept separately, never mixed into dataset/)
* tools/: verify_package.py (hash + format, numpy only), pi05_replay_native.py (action replay, needs robosuite)

Verify:  python tools/verify_package.py .
Replay:  PYTHONPATH=tools python tools/pi05_replay_native.py dataset out.json --sample-n 3

Action = [dx,dy,dz,drx,dry,drz,grip] in [-1,1] at 20 Hz; grip +1 close / -1 open / 0 hold.
State = eef_xyz + rotvec + 2 gripper qpos.  Images (T,2,224,224,3) uint8: agentview, robot0_eye_in_hand.
Splits are by seed group.  Grip force readings are teacher-side only and are not part of the observation.
"""


_WIPE_README = """
## Task notes: wipe (RoboSuite Wipe)

* The wiping tool has no fingers: action[6] (grip) is always 0 and state[6:8] (gripper qpos) are always 0.  They are
  kept only so that the 7D action / 8D state layout is identical to the lift/stack episodes.
* Success = all 100 dirt particles wiped; the sweep has a fixed 1000-step budget (never raised).
* manner_tag / intents use the dimensions `speed` (stage `sweep`: slow / normal / fast) and `press` (stage `sweep`:
  normal / firm).  Neutral = speed normal + press normal.  There is no `gentle` press level for wipe.
* The prompt of a wipe episode may be a paraphrase that states the desired outcome first and only implies the manner
  (`language_style` in episodes.jsonl: `explicit` | `outcome_hint` | `outcome_implicit`).  The label of every episode
  is the registered intent in metadata.json (`intents`), not a parse of the prompt (`label_source`).
* Tool-vs-table force readings (used for the press audit) are teacher-side only and are not in the observation.
* Realised levels (median over the sweep, audited per episode): sweep speed slow ~0.05 / normal ~0.10 / fast ~0.2 m/s
  (the controller saturates near 0.2 m/s, so `fast` is about 2x `normal`, not more); tool-table force normal ~30 N /
  firm ~45 N (medians; instantaneous peaks are higher).  `normal` press is itself moderately firm: it is the weakest
  press at which the wipe reliably succeeds.
* Instruction language is English only; the `outcome_hint` / `outcome_implicit` cues are tied to the label by
  construction and have not been validated by human readers.
"""


def _write_tools(root: Path, repo: Path) -> None:
    t = root / "tools"
    (t / "synthesis_v2").mkdir(parents=True)
    for f in ("native_osc_env.py", "grasp_force_contract.py"):
        shutil.copy2(repo / "synthesis_v2" / f, t / "synthesis_v2" / f)
    shutil.copy2(repo / "scripts" / "pi05_replay_native.py", t / "pi05_replay_native.py")
    shutil.copy2(repo / "scripts" / "pi05_format_check.py", t / "pi05_format_check.py")
    shutil.copy2(repo / "scripts" / "pi05_package_verify.py", t / "verify_package.py")
    (root / "README.md").write_text(_README.format(schema=PACKAGE_SCHEMA))
    try:
        import mujoco, robosuite  # type: ignore
        rv = f"robosuite=={robosuite.__version__}\nmujoco=={mujoco.__version__}\n"
    except Exception:  # noqa: BLE001
        rv = "robosuite\nmujoco\n"
    (root / "requirements.txt").write_text(f"numpy=={np.__version__}\nscipy\n{rv}# python {platform.python_version()}\n")


def write_sha256sums(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            rows.append(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def make_tarball(root: str | Path, out_tgz: str | Path) -> str:
    root = Path(root)
    with tarfile.open(out_tgz, "w:gz") as tf:
        tf.add(root, arcname=root.name)
    return sha256_file(out_tgz)
