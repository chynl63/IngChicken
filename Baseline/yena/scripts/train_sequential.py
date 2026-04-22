# -*- coding: utf-8 -*-
"""
Sequential Continual Learning pipeline for Diffusion Policy on LIBERO.

Trains a single diffusion policy model across N tasks sequentially.
After each task, evaluates on all previously seen tasks to measure forgetting.

Usage (from repo root, or inside container with cwd /workspace):
  python -m scripts.train_sequential \
      --config configs/continual_learning_libero_object.yaml
"""

import os
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from scripts.model import DiffusionPolicy, EMAModel
from scripts.datasets import (
    SingleTaskDataset,
    create_single_task_dataloader,
    compute_global_action_stats,
)
from scripts.evaluation import (
    evaluate_checkpoint_on_all_tasks,
    compute_nbt,
    compute_average_sr,
    compute_average_sr_per_stage,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
    plot_forgetting_summary,
)

from libero.libero.benchmark import get_benchmark


OFFICIAL_LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]


def verify_task_names(benchmark, benchmark_name: str):
    """Verify that resolved task names match the official list."""
    task_names = benchmark.get_task_names()
    n = benchmark.get_num_tasks()

    print("\n" + "=" * 70)
    print("TASK VERIFICATION")
    print("=" * 70)
    print(f"Benchmark: {benchmark_name}")
    print(f"Number of tasks: {n}")
    print(f"Task order index: {benchmark.task_order_index}")
    print("-" * 70)

    if benchmark_name == "libero_object":
        official = OFFICIAL_LIBERO_OBJECT_TASKS
        ordered_official = [official[i] for i in range(len(official))]

        all_match = True
        for i, name in enumerate(task_names):
            status = "OK" if name in official else "MISMATCH"
            if name not in official:
                all_match = False
            print(f"  Task {i:2d}: {name}")
            print(f"          Status: {status}")

        print("-" * 70)
        if all_match and set(task_names) == set(official):
            print("VERIFIED: All task names match the official LIBERO-Object list.")
        else:
            print("WARNING: Task names do NOT match the official list!")
            print(f"  Expected: {official}")
            print(f"  Got:      {task_names}")
    else:
        for i, name in enumerate(task_names):
            print(f"  Task {i:2d}: {name}")

    print("=" * 70 + "\n")
    return task_names


def verify_data_files(data_root: str, benchmark):
    """Verify all demo HDF5 files exist before starting training."""
    print("Verifying data files...")
    missing = []
    for i in range(benchmark.get_num_tasks()):
        demo_rel = benchmark.get_task_demonstration(i)
        demo_path = os.path.join(data_root, demo_rel)
        exists = os.path.exists(demo_path)
        status = "OK" if exists else "MISSING"
        print(f"  Task {i}: {demo_rel} [{status}]")
        if not exists:
            missing.append(demo_path)

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} demo file(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nPlease download the LIBERO-Object dataset first."
        )
    print("All data files verified.\n")


def train_on_task(
    model: DiffusionPolicy,
    task_idx: int,
    task_name: str,
    demo_path: str,
    cfg: dict,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
) -> tuple:
    """Train the model on a single task.

    Returns:
        (model, ema, epoch_loss_history)
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cl_cfg = cfg["continual_learning"]
    log_cfg = cfg["logging"]

    epochs = cl_cfg["epochs_per_task"]

    print(f"  Loading dataset: {demo_path}")
    loader, dataset = create_single_task_dataloader(
        hdf5_path=demo_path,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        obs_horizon=data_cfg["obs_horizon"],
        action_horizon=data_cfg["action_horizon"],
        action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
        action_std=action_std if data_cfg.get("normalize_action", True) else None,
        obs_keys=data_cfg["obs_keys"],
        use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
        image_size=tuple(data_cfg.get("image_size", [128, 128])),
    )
    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    lr = train_cfg.get("_effective_lr", train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    total_steps = epochs * len(loader)
    warmup_steps = train_cfg.get("lr_warmup_steps", 500)

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
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb
        except ImportError:
            print("  wandb not available, skipping wandb logging")
            use_wandb = False

    epoch_losses = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        pbar = tqdm(
            loader,
            desc=f"  Task {task_idx} | Epoch {epoch+1}/{epochs}",
            leave=False,
        )
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
            batch_losses.append(loss_val)
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if use_wandb and global_step % log_cfg.get("log_interval", 50) == 0:
                wandb_run.log({
                    f"task_{task_idx}/loss": loss_val,
                    f"task_{task_idx}/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                })

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(
            f"  Task {task_idx} | Epoch {epoch+1:3d}/{epochs} | "
            f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

    return model, ema, epoch_losses


def main(cfg, skip_eval=False, pretrain_ckpt=None):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_cfg = cfg["benchmark"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]

    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    results_dir = Path(log_cfg["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Set up LIBERO benchmark ──
    benchmark = get_benchmark(benchmark_cfg["name"])(
        task_order_index=benchmark_cfg.get("task_order_index", 0)
    )
    n_tasks = benchmark.get_num_tasks()
    task_names = verify_task_names(benchmark, benchmark_cfg["name"])

    data_root = benchmark_cfg["data_root"]
    verify_data_files(data_root, benchmark)

    # ── 2. Compute global action normalization stats ──
    print("Computing global action normalization stats...")
    action_mean, action_std = compute_global_action_stats(data_root, benchmark)

    # ── 3. Build model ──
    print("\nBuilding Diffusion Policy model...")
    model = DiffusionPolicy(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    if pretrain_ckpt is not None:
        print(f"\nLoading pretrained weights from: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if missing:
            print(f"  WARNING: missing keys: {missing}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected}")
        print("  Pretrained weights loaded successfully.")
        # Use finetune LR if specified, otherwise fall back to training.learning_rate
        if "finetune_learning_rate" in cfg.get("training", {}):
            cfg["training"]["_effective_lr"] = cfg["training"]["finetune_learning_rate"]
            print(f"  Using finetune LR: {cfg['training']['finetune_learning_rate']}")
        else:
            cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    else:
        cfg["training"]["_effective_lr"] = cfg["training"]["learning_rate"]
    print()

    # ── 4. Performance matrix ──
    perf_matrix = np.full((n_tasks, n_tasks), np.nan)
    training_log = []

    run_meta = {
        "start_time": datetime.now().isoformat(),
        "config": cfg,
        "n_tasks": n_tasks,
        "task_names": task_names,
        "param_count": param_count,
    }

    # ── 5. Sequential training loop ──
    for task_k in range(n_tasks):
        print("\n" + "=" * 70)
        print(f"STAGE {task_k + 1}/{n_tasks}: Training on Task {task_k}")
        print(f"  Name: {task_names[task_k]}")
        print("=" * 70)

        demo_rel = benchmark.get_task_demonstration(task_k)
        demo_path = os.path.join(data_root, demo_rel)

        t_start = time.time()

        model, ema, epoch_losses = train_on_task(
            model=model,
            task_idx=task_k,
            task_name=task_names[task_k],
            demo_path=demo_path,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
        )

        train_time = time.time() - t_start
        print(f"\n  Training time: {train_time:.1f}s")
        print(f"  Final loss: {epoch_losses[-1]:.4f}")

        # ── Save checkpoint ──
        ckpt_path = ckpt_dir / f"after_task_{task_k:02d}.pt"
        torch.save(
            {
                "task_idx": task_k,
                "task_name": task_names[task_k],
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "config": cfg,
                "action_mean": action_mean,
                "action_std": action_std,
                "epoch_losses": epoch_losses,
            },
            ckpt_path,
        )
        print(f"  Checkpoint saved: {ckpt_path}")

        stage_log = {
            "task_idx": task_k,
            "task_name": task_names[task_k],
            "train_time_s": train_time,
            "final_train_loss": float(epoch_losses[-1]),
        }

        if not skip_eval:
            # ── Evaluate on all seen tasks ──
            print(f"\n  Evaluating on tasks 0..{task_k} ({task_k + 1} tasks):")
            eval_model = ema.model
            eval_model.eval()

            low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

            t_eval_start = time.time()
            eval_results = evaluate_checkpoint_on_all_tasks(
                model=eval_model,
                benchmark=benchmark,
                task_indices=list(range(task_k + 1)),
                num_episodes=eval_cfg.get("num_episodes", 20),
                max_steps=eval_cfg.get("max_steps_per_episode", 600),
                action_execution_horizon=eval_cfg.get("action_execution_horizon", 8),
                action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                action_std=action_std if data_cfg.get("normalize_action", True) else None,
                obs_horizon=data_cfg["obs_horizon"],
                image_size=tuple(data_cfg.get("image_size", [128, 128])),
                use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                low_dim_keys=low_dim_keys,
                device=device,
                use_ddim=eval_cfg.get("use_ddim", True),
                ddim_steps=eval_cfg.get("ddim_steps", 16),
                seed=seed,
            )
            eval_time = time.time() - t_eval_start
            print(f"  Evaluation time: {eval_time:.1f}s")

            for task_j, sr in eval_results.items():
                perf_matrix[task_k, task_j] = sr

            avg_sr_stage = np.nanmean(perf_matrix[task_k, : task_k + 1])
            nbt_so_far = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])

            stage_log["eval_time_s"] = eval_time
            stage_log["avg_sr"] = float(avg_sr_stage)
            stage_log["nbt"] = float(nbt_so_far)
            stage_log["eval_results"] = {str(k): float(v) for k, v in eval_results.items()}

            print(f"\n  --- Stage {task_k + 1} Summary ---")
            print(f"  Avg SR (tasks 0..{task_k}): {avg_sr_stage:.4f}")
            print(f"  NBT so far: {nbt_so_far:.4f}")
        else:
            print(f"\n  [skip-eval] Skipping evaluation (checkpoint saved)")

        training_log.append(stage_log)

        # ── Save intermediate results ──
        _save_intermediate(
            perf_matrix, task_names, training_log, cfg, results_dir, task_k
        )

    # ── 6. Final metrics ──
    run_meta["end_time"] = datetime.now().isoformat()
    run_meta["training_log"] = training_log

    if skip_eval:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE (evaluation skipped)")
        print("=" * 70)
        print(f"  {n_tasks} checkpoints saved to: {ckpt_dir}")
        print(f"  Run evaluate_checkpoints.py to compute metrics.\n")

        with open(results_dir / "run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2, default=str)

        print(f"All results saved to: {results_dir}")
        print("Done!")
        return

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    nbt_final = compute_nbt(perf_matrix)
    avg_sr_final = compute_average_sr(perf_matrix)
    avg_sr_stages = compute_average_sr_per_stage(perf_matrix)

    print(f"  Average SR (final): {avg_sr_final:.4f}")
    print(f"  NBT: {nbt_final:.4f}")
    print(f"  Avg SR per stage: {avg_sr_stages}")
    print()

    print("  Performance Matrix:")
    for i in range(n_tasks):
        row_str = "  "
        for j in range(n_tasks):
            if np.isnan(perf_matrix[i, j]):
                row_str += "  --  "
            else:
                row_str += f" {perf_matrix[i, j]:.2f} "
        print(row_str)
    print()

    # ── 7. Save final results ──
    run_meta["nbt"] = float(nbt_final)
    run_meta["avg_sr_final"] = float(avg_sr_final)

    save_results_json(
        perf_matrix, task_names, nbt_final, avg_sr_final, cfg,
        str(results_dir / "results.json"),
    )
    save_results_csv(perf_matrix, task_names, str(results_dir / "perf_matrix.csv"))

    with open(results_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    np.save(results_dir / "perf_matrix.npy", perf_matrix)

    # ── 8. Generate plots ──
    plot_performance_matrix(
        perf_matrix, task_names, str(results_dir / "heatmap.png")
    )
    plot_forgetting_summary(
        perf_matrix, task_names, str(results_dir / "forgetting_summary.png")
    )

    print(f"\nAll results saved to: {results_dir}")
    print("Done!")


def _save_intermediate(perf_matrix, task_names, training_log, cfg, results_dir, task_k):
    """Save intermediate results after each task for crash recovery."""
    np.save(results_dir / "perf_matrix_intermediate.npy", perf_matrix)

    nbt = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])
    avg_sr = np.nanmean(perf_matrix[task_k, : task_k + 1])

    save_results_json(
        perf_matrix, task_names, nbt, avg_sr, cfg,
        str(results_dir / "results_intermediate.json"),
    )

    intermediate = {
        "completed_tasks": task_k + 1,
        "training_log": training_log,
    }
    with open(results_dir / "training_log.json", "w") as f:
        json.dump(intermediate, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential Continual Learning for Diffusion Policy on LIBERO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/configs/continual_learning_libero_object.yaml",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after each task (train only, evaluate later).",
    )
    parser.add_argument(
        "--pretrain-ckpt",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (e.g. from train_pretrain.py). "
             "Model weights will be loaded before sequential fine-tuning.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, skip_eval=args.skip_eval, pretrain_ckpt=args.pretrain_ckpt)
