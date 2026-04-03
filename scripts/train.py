# -*- coding: utf-8 -*-
"""
Diffusion Policy training on LIBERO-90 with per-task uniform sampling.

Usage:
  python -m scripts.train --config configs/diffusion_policy_libero90.yaml
"""

import math
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from scripts.model import DiffusionPolicy, EMAModel
from scripts.datasets.libero_dataset import create_dataloader


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building dataset with per-task uniform sampling...")
    print("=" * 60)

    loader, dataset = create_dataloader(
        data_dir=data_cfg["data_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        samples_per_epoch=data_cfg.get("samples_per_epoch", 50000),
        normalize_action=data_cfg.get("normalize_action", True),
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
    )

    print("=" * 60)
    print("Building Diffusion Policy model...")
    print("=" * 60)

    model = DiffusionPolicy(cfg).to(device)
    ema = EMAModel(model, decay=train_cfg.get("ema_decay", 0.995))

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    total_steps = train_cfg["num_epochs"] * len(loader)
    warmup_steps = train_cfg.get("lr_warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = train_cfg.get("mixed_precision", True)
    scaler = GradScaler(enabled=use_amp)

    use_wandb = log_cfg.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=log_cfg["project"], config=cfg)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=use_amp):
                loss = model.compute_loss(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if train_cfg.get("gradient_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema.update(model)

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if use_wandb and global_step % log_cfg.get("log_interval", 100) == 0:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % log_cfg.get("save_interval", 10) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "action_mean": dataset.action_mean if hasattr(dataset, "action_mean") else None,
                "action_std": dataset.action_std if hasattr(dataset, "action_std") else None,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "config": cfg,
                "action_mean": dataset.action_mean if hasattr(dataset, "action_mean") else None,
                "action_std": dataset.action_std if hasattr(dataset, "action_std") else None,
            }, best_path)

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_policy_libero90.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
