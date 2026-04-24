# -*- coding: utf-8 -*-
"""
Sequential Continual Learning pipeline for Diffusion Policy on LIBERO.

Trains a single diffusion policy model across N tasks sequentially.
After each task, evaluates on all previously seen tasks to measure forgetting.

Usage (from repo root, or inside container with cwd /workspace):
  python -m merge.scripts.train.train_sequential \
      --config merge/configs/continual_learning_libero_object.yaml
"""

import os
import copy
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

from merge.model import DiffusionPolicy, EMAModel
from merge.datasets import (
    SingleTaskDataset,
    create_single_task_dataloader,
    compute_global_action_stats,
)
from merge.datasets.utils_er import ReplayMemory, cycle, merge_batches, split_batch_size
from merge.scripts.eval import (
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

from merge.SDFT.MSE.sdft import (
    collect_onpolicy_observations,
    stack_obs_batches,
    compute_sdft_loss,
)


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


def _checkpoint_step(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def _prepare_run_dirs(cfg: dict) -> tuple:
    log_cfg = cfg["logging"]
    # Prefer the current config layout under logging, while keeping backward
    # compatibility with older top-level exp_name configs.
    exp_name = log_cfg.get("exp_name") or cfg.get("exp_name")

    if exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("output") / f"{exp_name}_{timestamp}"
        ckpt_dir = run_dir / "checkpoints"
        results_dir = run_dir / "results"

        cfg["run_name"] = run_dir.name
        cfg["run_dir"] = str(run_dir.resolve())
        cfg["logging"]["checkpoint_dir"] = str(ckpt_dir.resolve())
        cfg["logging"]["results_dir"] = str(results_dir.resolve())

        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        run_dir = None
        ckpt_dir = Path(log_cfg["checkpoint_dir"])
        results_dir = Path(log_cfg["results_dir"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, results_dir


def _init_tensorboard_writer(cfg: dict, results_dir: Path):
    log_cfg = cfg["logging"]
    if not log_cfg.get("use_tensorboard", False):
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not available, skipping tensorboard logging")
        return None

    tb_dir = log_cfg.get("tensorboard_dir")
    if tb_dir:
        tb_dir = Path(tb_dir)
    elif cfg.get("run_dir"):
        tb_dir = Path(cfg["run_dir"]) / "tensorboard"
    else:
        tb_dir = results_dir / "tensorboard"

    tb_dir.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["tensorboard_dir"] = str(tb_dir.resolve())
    print(f"TensorBoard logs will be saved to: {tb_dir}")
    return SummaryWriter(log_dir=str(tb_dir))


def _init_wandb_run(cfg: dict):
    log_cfg = cfg["logging"]
    if not log_cfg.get("use_wandb", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not available, skipping wandb logging")
        return None

    run_name = cfg.get("run_name") or log_cfg.get("exp_name")
    return wandb.init(
        project=log_cfg.get("project", "diffusion_policy_cl"),
        name=run_name,
        config=cfg,
    )


def _resolve_weights_path(weights_dir: str) -> Path:
    if not weights_dir:
        return None

    path = Path(weights_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"weights_dir does not exist: {path}")

    if path.is_file():
        return path

    prioritized = [
        path / "checkpoints" / "best_ema.pt",
        path / "checkpoints" / "best.pt",
        path / "checkpoints" / "after_task_09_ema.pt",
        path / "checkpoints" / "after_task_09.pt",
        path / "best_ema.pt",
        path / "best.pt",
    ]
    for candidate in prioritized:
        if candidate.exists():
            return candidate

    ema_candidates = sorted(path.rglob("*_ema.pt"), key=_checkpoint_step)
    if ema_candidates:
        return ema_candidates[-1]

    ckpt_candidates = sorted(path.rglob("*.pt"), key=_checkpoint_step)
    if ckpt_candidates:
        return ckpt_candidates[-1]

    raise FileNotFoundError(f"No checkpoint file found under weights_dir: {path}")


def _load_initial_weights(model: DiffusionPolicy, cfg: dict, device: torch.device):
    weights_path = _resolve_weights_path(cfg.get("weights_dir"))
    if weights_path is None:
        print("Training mode: scratch")
        return None

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    if not compatible:
        raise ValueError(f"No compatible parameters found in checkpoint: {weights_path}")

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print(
        f"Training mode: finetune from {weights_path} "
        f"({len(compatible)} tensors loaded, {len(skipped)} skipped)"
    )
    return weights_path


def _save_checkpoint(path: Path, payload: dict, model: DiffusionPolicy, ema: EMAModel, use_ema_weights: bool):
    checkpoint = dict(payload)
    checkpoint["checkpoint_kind"] = "ema" if use_ema_weights else "raw"
    checkpoint["model_state_dict"] = ema.state_dict() if use_ema_weights else model.state_dict()
    checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, path)


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
    tb_writer=None,
    tb_global_step_offset: int = 0,
    wandb_run=None,
    replay_memory: ReplayMemory = None,
    benchmark=None,
) -> tuple:
    """Train the model on a single task.

    Returns:
        (model, ema, epoch_loss_history)
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cl_cfg = cfg["continual_learning"]
    log_cfg = cfg["logging"]
    replay_cfg = cfg.get("replay", {})
    eval_cfg = cfg.get("evaluation", {})
    sdft_cfg = cfg.get("sdft", {})

    epochs = cl_cfg["epochs_per_task"]
    lambda_sdft = float(sdft_cfg.get("lambda_sdft", 0.0))
    use_sdft = bool(sdft_cfg.get("enabled", sdft_cfg.get("use_sdft", False))) and lambda_sdft > 0.0
    sdft_rollout_interval = int(sdft_cfg.get("sdft_rollout_interval", sdft_cfg.get("rollout_interval", 500)))
    log_sdft_debug = bool(sdft_cfg.get("log_sdft_debug", False))
    sdft_ddim_steps = int(sdft_cfg.get("sdft_ddim_steps", sdft_cfg.get("ddim_steps", eval_cfg.get("ddim_steps", 16))))

    if use_sdft and benchmark is None:
        raise ValueError("SDFT is enabled but benchmark was not provided to train_on_task.")

    teacher = None
    if use_sdft:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.to(device)
        print(
            f"  [sdft] enabled: lambda={lambda_sdft}, rollout_every={sdft_rollout_interval}, "
            f"teacher_sync={sdft_cfg.get('teacher_sync_interval', 500)}, ddim_steps={sdft_ddim_steps}"
        )

    current_batch_size = data_cfg["batch_size"]
    replay_batch_size = 0
    replay_loader = None
    replay_iterator = None

    if replay_memory is not None and replay_memory.has_samples():
        if data_cfg["batch_size"] < 2:
            raise ValueError("Replay requires data.batch_size >= 2")

        mix_ratio = float(replay_cfg.get("mix_ratio", 0.5))
        if not 0.0 < mix_ratio < 1.0:
            raise ValueError("replay.mix_ratio must be in the open interval (0, 1)")

        current_batch_size, replay_batch_size = split_batch_size(
            data_cfg["batch_size"], mix_ratio
        )
        replay_loader = replay_memory.build_loader(
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
            batch_size=replay_batch_size,
        )
        if replay_loader is not None:
            replay_iterator = cycle(replay_loader)
            print(
                f"  Replay enabled: {replay_memory.num_samples()} samples from "
                f"{replay_memory.num_tasks()} previous task(s)"
            )
            print(
                f"  Mixed batch sizes -> current: {current_batch_size}, "
                f"replay: {replay_batch_size}"
            )

    print(f"  Loading dataset: {demo_path}")
    loader, dataset = create_single_task_dataloader(
        hdf5_path=demo_path,
        batch_size=current_batch_size,
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
    if replay_memory is not None and replay_iterator is None:
        print("  Replay buffer empty for this stage, training on current task only")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
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

    use_wandb = wandb_run is not None

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
            if replay_iterator is not None:
                replay_batch = next(replay_iterator)
                replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                batch = merge_batches(batch, replay_batch)

            obs_batches = None
            sdft_rollout_states = 0
            absolute_step = tb_global_step_offset + global_step
            if use_sdft and (absolute_step + 1) % sdft_rollout_interval == 0:
                roll_seed = int(cfg.get("seed", 42) + task_idx * 7919 + absolute_step)
                obs_batches, sdft_rollout_states = collect_onpolicy_observations(
                    model=model,
                    benchmark=benchmark,
                    task_idx=task_idx,
                    num_episodes=int(sdft_cfg.get("sdft_num_episodes", sdft_cfg.get("num_episodes", 2))),
                    max_steps=int(
                        sdft_cfg.get(
                            "sdft_rollout_horizon",
                            sdft_cfg.get("rollout_horizon", eval_cfg.get("max_steps_per_episode", 600)),
                        )
                    ),
                    action_execution_horizon=int(
                        sdft_cfg.get(
                            "action_execution_horizon",
                            eval_cfg.get("action_execution_horizon", 8),
                        )
                    ),
                    action_mean=action_mean if data_cfg.get("normalize_action", True) else None,
                    action_std=action_std if data_cfg.get("normalize_action", True) else None,
                    obs_horizon=data_cfg["obs_horizon"],
                    image_size=tuple(data_cfg.get("image_size", [128, 128])),
                    use_eye_in_hand=data_cfg.get("use_eye_in_hand", True),
                    low_dim_keys=[k for k in data_cfg["obs_keys"] if "image" not in k],
                    device=device,
                    use_ddim=eval_cfg.get("use_ddim", True),
                    ddim_steps=sdft_ddim_steps,
                    max_states=int(sdft_cfg.get("sdft_max_states_per_batch", sdft_cfg.get("max_states_per_batch", 64))),
                    seed=roll_seed,
                    log_debug=log_sdft_debug,
                )

            with autocast(enabled=use_amp):
                loss_main = model.compute_loss(batch)
                loss_sdft = None
                if use_sdft and obs_batches and len(obs_batches) > 0:
                    stacked = stack_obs_batches(obs_batches)
                    if log_sdft_debug:
                        for k, v in stacked.items():
                            print(f"    [sdft] stacked {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                    loss_sdft = compute_sdft_loss(
                        student=model,
                        teacher=teacher,
                        stacked_batch=stacked,
                        ddim_steps=sdft_ddim_steps,
                    )
                    loss = loss_main + lambda_sdft * loss_sdft
                else:
                    loss = loss_main

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

            if use_sdft and teacher is not None:
                ts = int(sdft_cfg.get("teacher_sync_interval", 500))
                if ts > 0 and (tb_global_step_offset + global_step + 1) % ts == 0:
                    teacher.load_state_dict(model.state_dict())

            loss_val = loss.item()
            batch_losses.append(loss_val)
            global_step += 1

            postfix = {
                "loss": f"{loss_val:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if use_sdft and loss_sdft is not None:
                postfix["sdft"] = f"{loss_sdft.item():.4f}"
            pbar.set_postfix(**postfix)

            if use_wandb and global_step % log_cfg.get("log_interval", 50) == 0:
                log_payload = {
                    f"task_{task_idx}/loss": loss_val,
                    f"task_{task_idx}/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }
                if use_sdft:
                    log_payload[f"task_{task_idx}/use_sdft"] = True
                    log_payload[f"task_{task_idx}/lambda_sdft"] = lambda_sdft
                    log_payload[f"task_{task_idx}/sdft_rollout_states"] = float(sdft_rollout_states)
                    if loss_sdft is not None:
                        log_payload[f"task_{task_idx}/loss_sdft"] = loss_sdft.item()
                wandb_run.log(log_payload)
            if tb_writer and global_step % log_cfg.get("log_interval", 50) == 0:
                tb_step = tb_global_step_offset + global_step
                tb_writer.add_scalar(f"task_{task_idx}/train/loss", loss_val, tb_step)
                tb_writer.add_scalar(f"task_{task_idx}/train/lr", scheduler.get_last_lr()[0], tb_step)
                tb_writer.add_scalar(f"task_{task_idx}/train/epoch", epoch, tb_step)

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(
            f"  Task {task_idx} | Epoch {epoch+1:3d}/{epochs} | "
            f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if tb_writer:
            tb_writer.add_scalar(f"task_{task_idx}/epoch/avg_loss", avg_loss, epoch + 1)
            tb_writer.add_scalar(f"task_{task_idx}/epoch/lr", scheduler.get_last_lr()[0], epoch + 1)

    return model, ema, epoch_losses, global_step, dataset


def main(cfg, skip_eval=False):
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark_cfg = cfg["benchmark"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    replay_cfg = cfg.get("replay", {})
    run_dir, ckpt_dir, results_dir = _prepare_run_dirs(cfg)
    tb_writer = _init_tensorboard_writer(cfg, results_dir)
    wandb_run = _init_wandb_run(cfg)

    replay_memory = None
    if replay_cfg.get("enabled", False):
        buffer_size = int(replay_cfg.get("buffer_size", 0))
        if buffer_size <= 0:
            print("Replay requested but replay.buffer_size <= 0, disabling replay.\n")
        else:
            mix_ratio = float(replay_cfg.get("mix_ratio", 0.5))
            if not 0.0 < mix_ratio < 1.0:
                raise ValueError("replay.mix_ratio must be in the open interval (0, 1)")
            replay_memory = ReplayMemory(capacity=buffer_size, seed=seed)
            print(
                f"Replay is enabled with buffer_size={buffer_size} "
                f"and mix_ratio={mix_ratio:.2f}\n"
            )
    else:
        print("Replay is disabled.\n")

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
    init_weights_path = _load_initial_weights(model, cfg, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}\n")

    # ── 4. Performance matrix ──
    perf_matrix = np.full((n_tasks, n_tasks), np.nan)
    training_log = []
    tb_global_step = 0

    run_meta = {
        "start_time": datetime.now().isoformat(),
        "config": cfg,
        "run_dir": str(run_dir.resolve()) if run_dir else None,
        "n_tasks": n_tasks,
        "task_names": task_names,
        "param_count": param_count,
        "init_weights_path": str(init_weights_path) if init_weights_path else None,
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

        model, ema, epoch_losses, task_steps, task_dataset = train_on_task(
            model=model,
            task_idx=task_k,
            task_name=task_names[task_k],
            demo_path=demo_path,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
            device=device,
            tb_writer=tb_writer,
            tb_global_step_offset=tb_global_step,
            wandb_run=wandb_run,
            replay_memory=replay_memory,
            benchmark=benchmark,
        )
        tb_global_step += task_steps

        if replay_memory is not None:
            replay_memory.add_task(demo_path, task_dataset.index)
            print(
                f"  Replay buffer updated: {replay_memory.num_samples()} samples "
                f"across {replay_memory.num_tasks()} task(s)"
            )
        del task_dataset

        train_time = time.time() - t_start
        print(f"\n  Training time: {train_time:.1f}s")
        print(f"  Final loss: {epoch_losses[-1]:.4f}")
        if tb_writer:
            tb_writer.add_scalar(f"task_{task_k}/summary/train_time_s", train_time, task_k + 1)
            tb_writer.add_scalar(f"task_{task_k}/summary/final_train_loss", epoch_losses[-1], task_k + 1)

        # ── Save checkpoint ──
        ckpt_path = ckpt_dir / f"after_task_{task_k:02d}.pt"
        ema_ckpt_path = ckpt_dir / f"after_task_{task_k:02d}_ema.pt"
        checkpoint_payload = {
            "task_idx": task_k,
            "task_name": task_names[task_k],
            "config": cfg,
            "action_mean": action_mean,
            "action_std": action_std,
            "epoch_losses": epoch_losses,
            "init_weights_path": str(init_weights_path) if init_weights_path else None,
        }
        _save_checkpoint(ckpt_path, checkpoint_payload, model, ema, use_ema_weights=False)
        _save_checkpoint(ema_ckpt_path, checkpoint_payload, model, ema, use_ema_weights=True)
        print(f"  Checkpoint saved: {ckpt_path}")
        print(f"  EMA checkpoint saved: {ema_ckpt_path}")

        stage_log = {
            "task_idx": task_k,
            "task_name": task_names[task_k],
            "train_time_s": train_time,
            "final_train_loss": float(epoch_losses[-1]),
        }
        if replay_memory is not None:
            stage_log["replay_buffer_samples"] = replay_memory.num_samples()
            stage_log["replay_buffer_tasks"] = replay_memory.num_tasks()

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
                save_video=eval_cfg.get("save_video", False),
                video_root=results_dir / "videos" / f"stage_{task_k:02d}",
                video_fps=eval_cfg.get("video_fps", 10),
                num_videos_per_task=eval_cfg.get("num_videos_per_task", 2),
                video_rotate_180=eval_cfg.get("video_rotate_180", False),
                video_crop_bottom_frac=eval_cfg.get("video_crop_bottom_frac", 0.0),
                video_episode_policy=eval_cfg.get("video_episode_policy", "first_k"),
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
            if tb_writer:
                tb_writer.add_scalar("eval/avg_sr", avg_sr_stage, task_k + 1)
                tb_writer.add_scalar("eval/nbt", nbt_so_far, task_k + 1)
                tb_writer.add_scalar(f"task_{task_k}/summary/eval_time_s", eval_time, task_k + 1)
                for eval_task_idx, sr in eval_results.items():
                    tb_writer.add_scalar(
                        f"task_{task_k}/eval/task_{eval_task_idx}_sr",
                        sr,
                        task_k + 1,
                    )

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
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()
        if wandb_run is not None:
            wandb_run.finish()
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
        perf_matrix,
        task_names,
        str(results_dir / "heatmap.png"),
        benchmark_name=cfg.get("benchmark", {}).get("name"),
    )
    plot_forgetting_summary(
        perf_matrix, task_names, str(results_dir / "forgetting_summary.png")
    )

    print(f"\nAll results saved to: {results_dir}")
    print("Done!")
    if tb_writer:
        tb_writer.add_scalar("final/avg_sr", avg_sr_final, n_tasks)
        tb_writer.add_scalar("final/nbt", nbt_final, n_tasks)
        tb_writer.flush()
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()


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
        default="merge/configs/continual_learning_libero_object.yaml",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after each task (train only, evaluate later).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, skip_eval=args.skip_eval)
