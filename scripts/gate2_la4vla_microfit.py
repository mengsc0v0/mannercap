"""Overfit a fixed LA4VLA micro-batch and emit metrics without a checkpoint.

This is a 32--64 sample learning-path diagnostic.  It is deliberately not a
generalization or closed-loop task evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import hashlib
import subprocess
import sys
import time


def masked_flow_loss(prediction, target, action_mask):
    mask = action_mask.reshape(action_mask.shape[0], -1).to(dtype=prediction.dtype)
    error = (prediction.float() - target.float()).pow(2) * mask.float()
    return error.mean() * (mask.numel() / mask.sum().clamp_min(1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la4vla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--sample-indices", default="0,1,32,33")
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        parser.error("--steps and --lr must be positive")

    root = args.la4vla_root.resolve()
    sys.path.insert(0, str(root / "LA4VLA_1B"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import yaml
    from config import LA4VLAConfig
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    from scripts.LA4VLA_1B import LA4VLA_1B

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    payload = torch.load(args.checkpoint.resolve(), map_location="cpu")
    config_dict = dict(payload["config"])
    config_dict.update({
        "device": "cuda",
        "vlm_name": str(root / "InternVL3-1B"),
        "finetune_vlm": False,
        "finetune_language_model": False,
        "finetune_text_embedder": False,
        "finetune_vision_model": False,
        "finetune_action_head": True,
    })
    config = LA4VLAConfig.from_dict(config_dict)

    with args.dataset_config.resolve().open(encoding="utf-8") as stream:
        dataset_config = yaml.safe_load(stream)
    dataset = LeRobotDataset(
        config=dataset_config,
        image_size=config.image_size,
        max_samples_per_file=50,
        action_horizon=config.horizon,
        video_backend=config.video_backend,
        cache_dir=args.cache_dir.resolve(),
        use_augmentation=False,
    )
    indices = [int(value) for value in args.sample_indices.split(",") if value.strip()]
    if not indices or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"sample indices must lie within [0, {len(dataset) - 1}]")
    items = [dataset[index] for index in indices]
    prompts = [item["prompt"] for item in items]
    images = [item["images"].to(device) for item in items]
    image_mask = torch.stack([item["image_mask"] for item in items]).to(device)
    states = torch.stack([item["state"] for item in items]).float().to(device)
    actions = torch.stack([item["action"] for item in items]).float().to(device)
    action_mask = torch.stack([item["action_mask"] for item in items]).to(device)

    started = time.perf_counter()
    model = LA4VLA_1B(config)
    model.set_finetune_flags()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    model.eval()
    with torch.no_grad():
        fused = model.get_vl_embeddings(
            images=images, image_mask=image_mask, prompt=prompts, return_cls_only=False
        ).detach()

    def evaluate(seed: int) -> float:
        model.eval()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            prediction, noise = model(
                fused, state=states, actions_gt=actions, action_mask=action_mask
            )
            target = (actions - noise).reshape(actions.shape[0], -1)
            return float(masked_flow_loss(prediction, target, action_mask).item())

    initial_loss = evaluate(args.seed + 1)
    history: list[float] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, noise = model(
            fused, state=states, actions_gt=actions, action_mask=action_mask
        )
        target = (actions - noise).reshape(actions.shape[0], -1)
        loss = masked_flow_loss(prediction, target, action_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(loss.item()))
    final_loss = evaluate(args.seed + 1)
    report = {
        "schema": "mannercap-la4vla-microfit-v1",
        "purpose": "fixed_microbatch_learning_pipeline_diagnostic",
        "dataset_samples_available": len(dataset),
        "fixed_sample_indices": indices,
        "fixed_sample_prompts": prompts,
        "training_steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "code_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "source_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.resolve().stat().st_size,
        },
        "dataset_config": {
            "path": str(args.dataset_config.resolve()),
            "sha256": hashlib.sha256(args.dataset_config.resolve().read_bytes()).hexdigest(),
        },
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "initial_fixed_noise_loss": initial_loss,
        "final_fixed_noise_loss": final_loss,
        "fixed_noise_loss_ratio": final_loss / initial_loss,
        "training_loss_first": history[0],
        "training_loss_last": history[-1],
        "training_loss_minimum": min(history),
        "all_losses_finite": True,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_written": False,
        "api_calls": 0,
        "generalization_claim_allowed": False,
        "closed_loop_claim_allowed": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if final_loss < initial_loss else 1


if __name__ == "__main__":
    raise SystemExit(main())
