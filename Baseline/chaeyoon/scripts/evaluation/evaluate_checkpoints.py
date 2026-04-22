# -*- coding: utf-8 -*-
"""
Standalone evaluation of saved checkpoints from sequential CL training.

Loads each checkpoint (after_task_00.pt .. after_task_09.pt), evaluates on
all tasks seen up to that point, and produces the full performance matrix,
metrics, and visualizations.

Usage:
    python -m scripts.evaluation.evaluate_checkpoints --config /path/to/config.yaml
    python -m scripts.evaluation.evaluate_checkpoints --config /path/to/config.yaml --ckpt-dir /path/to/ckpts
    python -m scripts.evaluation.evaluate_checkpoints --config cfg.yaml --run-tag-auto --no-plots
      # ->
      #   results → logging.results_dir/<timestamp>/  (tables + optional PNGs)
      #   logs    → <repo>/logs/<timestamp>/evaluate_checkpoints.log
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.model import DiffusionPolicy
from scripts.datasets import compute_global_action_stats
from libero.libero.benchmark import get_benchmark
from scripts.evaluation.rollout_evaluator import evaluate_policy_on_task
from scripts.evaluation.cl_metrics import (
    compute_nbt,
    compute_average_sr,
    compute_average_sr_per_stage,
    save_results_json,
    save_results_csv,
    plot_performance_matrix,
    plot_forgetting_summary,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class _StdoutTee:
    """Mirror stdout to multiple text streams (e.g. console + log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def evaluate_all_checkpoints(
    cfg,
    ckpt_dir=None,
    results_dir=None,
    ckpt_pattern=None,
    save_plots=True,
):
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "[evaluate_checkpoints] torch.cuda.is_available() is False in this process; "
            "using CPU. On clusters, use sbatch/salloc + `singularity exec --nv` with a GPU."
        )
        device = torch.device("cpu")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]

    if ckpt_dir is None:
        ckpt_dir = Path(log_cfg["checkpoint_dir"])
    else:
        ckpt_dir = Path(ckpt_dir)

    if results_dir is None:
        results_dir = Path(log_cfg["results_dir"])
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark = get_benchmark(cfg["benchmark"]["name"])(
        task_order_index=cfg["benchmark"].get("task_order_index", 0)
    )
    n_tasks = benchmark.get_num_tasks()
    task_names = benchmark.get_task_names()
    data_root = cfg["benchmark"]["data_root"]

    print("Computing global action normalization stats...")
    action_mean, action_std = compute_global_action_stats(data_root, benchmark)

    low_dim_keys = [k for k in data_cfg["obs_keys"] if "image" not in k]

    # Allow restricting to a subset / single checkpoint via pattern
    pattern = ckpt_pattern or "after_task_*.pt"
    ckpt_files = sorted(ckpt_dir.glob(pattern))
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    n_found = len(ckpt_files)
    print(f"\nFound {n_found} checkpoint(s) in {ckpt_dir}")
    for f in ckpt_files:
        print(f"  {f.name}")

    perf_matrix = np.full((n_tasks, n_tasks), np.nan)
    eval_log = []

    total_start = time.time()
    max_task_k_seen = -1

    for ckpt_path in ckpt_files:
        # Always unpickle to CPU first: ckpt files may contain GPU tensors; using
        # map_location=cuda here fails if this process does not see CUDA yet.
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        task_k = ckpt["task_idx"]
        max_task_k_seen = max(max_task_k_seen, task_k)

        print("\n" + "=" * 70)
        print(f"Evaluating checkpoint: {ckpt_path.name} (trained through task {task_k})")
        print(f"  Task: {task_names[task_k]}")
        print("=" * 70)

        model = DiffusionPolicy(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        task_indices = list(range(task_k + 1))
        stage_results = {}

        save_video = bool(eval_cfg.get("save_video", False))
        video_fps = float(eval_cfg.get("video_fps", 10.0))
        num_videos_per_task = int(eval_cfg.get("num_videos_per_task", 2))
        video_rotate_180 = bool(eval_cfg.get("video_rotate_180", False))
        video_crop_bottom_frac = float(eval_cfg.get("video_crop_bottom_frac", 0.0))
        video_episode_policy = str(eval_cfg.get("video_episode_policy", "first_k"))
        video_root = None
        if save_video:
            video_root = results_dir / "videos" / ckpt_path.stem
            video_root.mkdir(parents=True, exist_ok=True)

        for task_j in task_indices:
            t_start = time.time()
            sr, ep_results = evaluate_policy_on_task(
                model=model,
                benchmark=benchmark,
                task_idx=task_j,
                num_episodes=eval_cfg.get("num_episodes", 20),
                max_steps=eval_cfg.get("max_steps_per_episode", 300),
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
                save_video=save_video,
                video_root=video_root,
                checkpoint_stem=ckpt_path.stem,
                video_fps=video_fps,
                num_videos_per_task=num_videos_per_task,
                video_rotate_180=video_rotate_180,
                video_crop_bottom_frac=video_crop_bottom_frac,
                video_episode_policy=video_episode_policy,
            )
            t_elapsed = time.time() - t_start

            perf_matrix[task_k, task_j] = sr

            n_success = sum(ep_results)
            n_total = len(ep_results)
            print(f"  Task {task_j} [{task_names[task_j][:45]}]: "
                  f"SR={sr:.2f} ({n_success}/{n_total}) | {t_elapsed:.0f}s")

            stage_results[task_j] = {
                "success_rate": float(sr),
                "episode_results": [int(e) for e in ep_results],
                "eval_time_s": round(t_elapsed, 1),
            }

        cols = perf_matrix[task_k, : task_k + 1]
        avg_sr = float(np.nanmean(cols))
        nbt_so_far = compute_nbt(perf_matrix[: task_k + 1, : task_k + 1])

        print(f"\n  --- After task {task_k} (checkpoint row {task_k}) ---")
        print("  Per-task SR (each task evaluated separately; not pooled episodes):")
        for j in range(task_k + 1):
            print(f"    task {j}: {perf_matrix[task_k, j]:.4f}")
        print(
            f"  Mean SR (macro avg over tasks 0..{task_k}): {avg_sr:.4f} | "
            f"NBT (submatrix 0..{task_k}): {nbt_so_far:.4f}"
        )

        eval_log.append({
            "checkpoint": ckpt_path.name,
            "task_idx": task_k,
            "task_name": task_names[task_k],
            "avg_sr": float(avg_sr),
            "nbt": float(nbt_so_far),
            "task_results": stage_results,
        })

        # Save intermediate progress
        np.save(results_dir / "perf_matrix_intermediate.npy", perf_matrix)
        with open(results_dir / "eval_log.json", "w") as f:
            json.dump(eval_log, f, indent=2)

        del model
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Final metrics ──
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    sub_n = max_task_k_seen + 1
    nbt_final = compute_nbt(perf_matrix[:sub_n, :sub_n])
    final_stage = perf_matrix[max_task_k_seen, :sub_n]
    mask = ~np.isnan(final_stage)
    avg_sr_final = (
        float(np.mean(final_stage[mask])) if np.any(mask) else 0.0
    )

    print(f"  Average SR (final stage, macro over tasks 0..{max_task_k_seen}): {avg_sr_final:.4f}")
    print(f"  NBT (submatrix 0..{max_task_k_seen}): {nbt_final:.4f}")
    print(f"  Total eval time: {total_time / 60:.1f} min")
    print()

    print("  Performance Matrix (submatrix through last evaluated task)")
    for i in range(sub_n):
        row_str = "  "
        for j in range(sub_n):
            if np.isnan(perf_matrix[i, j]):
                row_str += "  --  "
            else:
                row_str += f" {perf_matrix[i, j]:.2f} "
        print(row_str)
    print()

    save_results_json(
        perf_matrix, task_names, nbt_final, avg_sr_final, cfg,
        str(results_dir / "results.json"),
    )
    save_results_csv(perf_matrix, task_names, str(results_dir / "perf_matrix.csv"))
    np.save(results_dir / "perf_matrix.npy", perf_matrix)

    with open(results_dir / "eval_log.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    if save_plots:
        plot_performance_matrix(
            perf_matrix, task_names, str(results_dir / "heatmap.png")
        )
        plot_forgetting_summary(
            perf_matrix, task_names, str(results_dir / "forgetting_summary.png")
        )
    else:
        print("\n(save_plots=False: skipped heatmap.png and forgetting_summary.png)")

    print(f"\nAll results saved to: {results_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate saved CL checkpoints on LIBERO tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "continual_learning_libero_spatial.yaml"),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Override checkpoint directory from config.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory from config.",
    )
    parser.add_argument(
        "--ckpt-pattern",
        type=str,
        default=None,
        help=(
            "Optional glob pattern to select a subset of checkpoints inside ckpt-dir, "
            'e.g. "after_task_05.pt" or "after_task_0[0-4].pt". '
            "If not provided, all 'after_task_*.pt' are evaluated."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip heatmap/forgetting PNGs (for sharded eval; plot after merge_perf_matrices).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Subdirectory under config logging.results_dir for this run. "
            "Also writes logs/<run-tag>/evaluate_checkpoints.log under the repo. "
            "Mutually exclusive with --results-dir."
        ),
    )
    parser.add_argument(
        "--run-tag-auto",
        action="store_true",
        help="Same as --run-tag but uses timestamp %%Y%%m%%d_%%H%%M%%S. Mutually exclusive with --results-dir.",
    )
    args = parser.parse_args()

    if args.run_tag and args.run_tag_auto:
        print("Error: use only one of --run-tag or --run-tag-auto.", file=sys.stderr)
        sys.exit(2)
    if args.results_dir and (args.run_tag or args.run_tag_auto):
        print(
            "Error: do not combine --results-dir with --run-tag/--run-tag-auto. "
            "Use --results-dir alone for a fixed path, or run-tag with config results_dir.",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg["logging"]
    eval_cfg = cfg.get("evaluation", {})
    save_plots = bool(eval_cfg.get("save_plots", True)) and not args.no_plots

    run_tag = args.run_tag
    if args.run_tag_auto:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = args.results_dir
    if results_dir is None and run_tag is not None:
        results_dir = str(Path(log_cfg["results_dir"]) / run_tag)

    log_f = None
    saved_stdout = sys.stdout
    try:
        if run_tag is not None:
            log_dir = _REPO_ROOT / "logs" / run_tag
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "evaluate_checkpoints.log"
            log_f = open(log_path, "w", encoding="utf-8")
            sys.stdout = _StdoutTee(saved_stdout, log_f)
            print(f"[run] run_tag={run_tag}")
            print(
                f"[run] results_dir will be: "
                f"{results_dir or log_cfg['results_dir']}"
            )
            print(f"[run] log file: {log_path}")
            print(f"[run] save_plots={save_plots}")

        evaluate_all_checkpoints(
            cfg,
            ckpt_dir=args.ckpt_dir,
            results_dir=results_dir,
            ckpt_pattern=args.ckpt_pattern,
            save_plots=save_plots,
        )
    finally:
        if log_f is not None:
            sys.stdout = saved_stdout
            log_f.close()
