# -*- coding: utf-8 -*-
"""
LIBERO-90 pretraining for Diffusion Policy.

Trains on all 90 tasks with uniform per-task sampling. The resulting
checkpoint can be used as initialization for sequential CL fine-tuning
on LIBERO-Object via train_sequential.py --pretrain-ckpt.

Usage (from repo root, or inside container with cwd /workspace):
  python -m scripts.train_pretrain \
      --config configs/pretrain_libero90.yaml

To run in background:
  nohup python -m scripts.train_pretrain \
      --config configs/pretrain_libero90.yaml > pretrain.log 2>&1 &
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from scripts.datasets import LiberoUniformDataset, create_libero_dataloader
from scripts.model import DiffusionPolicy, EMAModel


def main(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pretrain_cfg = cfg["pretrain"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ──
    print("=" * 70)
    print(" LIBERO-90 Pretraining: Diffusion Policy")
    print("=" * 70)
    print(f"  data_dir:         {pretrain_cfg['data_dir']}")
    print(f"  epochs:           {pretrain_cfg['epochs']}")
    print(f"  samples_per_epoch:{pretrain_cfg.get('samples_per_epoch', 'full dataset')}")
    print(f"  batch_size:       {data_cfg['batch_size']}")
    print(f"  checkpoint_dir:   {ckpt_dir}")
    print()

    # Load dataset first to know its size, then build loader
    dataset = LiberoUniformDataset(
        data_dir=pretrain_cfg["data_dir"],
        obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        obs_keys=data_cfg["obs_keys"],
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
        normalize_action=data_cfg.get("normalize_action", True),
        max_episodes_per_task=pretrain_cfg.get("max_episodes_per_task"),
    )

    samples_per_epoch = pretrain_cfg.get("samples_per_epoch") or len(dataset)
    print(f"Dataset size: {len(dataset)} total samples → {samples_per_epoch} samples/epoch")

    from torch.utils.data import DataLoader
    from scripts.datasets.libero_dataset import TaskUniformSampler
    sampler = TaskUniformSampler(dataset, num_samples=samples_per_epoch)
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )

    action_mean = dataset.action_mean if data_cfg.get("normalize_action", True) else None
    action_std  = dataset.action_std  if data_cfg.get("normalize_action", True) else None

    # ── 2. Build model ──
    print("\nBuilding Diffusion Policy model...")
    model = DiffusionPolicy(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}\n")

    # ── 3. Optimizer / scheduler ──
    epochs = pretrain_cfg["epochs"]
    total_steps = epochs * len(loader)
    warmup_steps = train_cfg.get("lr_warmup_steps", 1000)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMAModel(model, decay=train_cfg.get("ema_decay", 0.995))

    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler(enabled=use_amp)

    use_wandb = log_cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=log_cfg.get("project", "pretrain"), config=cfg)
        except ImportError:
            print("wandb not available, skipping")
            use_wandb = False

    # ── 4. Training loop ──
    print(f"Starting pretraining: {epochs} epochs × {len(loader)} steps = {total_steps} total steps\n")

    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=use_amp):
                loss = model.compute_loss(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if train_cfg.get("gradient_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["gradient_clip"]
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if use_wandb and global_step % log_cfg.get("log_interval", 100) == 0:
                import wandb
                wandb.log({"loss": loss_val, "lr": scheduler.get_last_lr()[0],
                           "epoch": epoch, "global_step": global_step})

        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | elapsed={elapsed/60:.1f}m"
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(
                ckpt_dir / "best.pt", model, ema, cfg,
                action_mean, action_std, epoch, global_step, avg_loss,
            )

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:04d}.pt", model, ema, cfg,
                action_mean, action_std, epoch, global_step, avg_loss,
            )

    # ── 5. Save final checkpoint ──
    _save_checkpoint(
        ckpt_dir / "final.pt", model, ema, cfg,
        action_mean, action_std, epochs - 1, global_step, avg_loss,
    )

    total_time = time.time() - start_time
    print(f"\nPretraining complete in {total_time/60:.1f}m")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"\nTo fine-tune on LIBERO-Object:")
    print(f"  python -m scripts.train_sequential \\")
    print(f"      --config configs/continual_learning_libero_object.yaml \\")
    print(f"      --pretrain-ckpt {ckpt_dir}/final.pt")


def _save_checkpoint(path, model, ema, cfg, action_mean, action_std,
                     epoch, global_step, loss):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "config": cfg,
            "action_mean": action_mean,
            "action_std": action_std,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain Diffusion Policy on LIBERO-90"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/configs/pretrain_libero90.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
