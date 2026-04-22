# -*- coding: utf-8 -*-
"""
On-policy SDFT (Self-Distillation Fine-Tuning) helpers for sequential CL.

Collects observation batches from student rollouts in the current task env,
then computes MSE between student and teacher DDIM action predictions.
"""

from __future__ import annotations

import os
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.evaluation.rollout_evaluator import (
    process_env_obs,
    obs_buffer_to_batch,
    predict_action_ddim,
    _predict_action_ddim_core,
)


def clone_obs_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-copy tensor values for storage (avoid in-place mutation)."""
    return {k: v.clone() for k, v in batch.items()}


def stack_obs_batches(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack B=1 rollout batches into one batch along dim 0."""
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    return {k: torch.cat([b[k] for b in batch_list], dim=0) for k in keys}


def subsample_obs_batches(
    batch_list: List[Dict[str, torch.Tensor]],
    max_states: int,
    rng: np.random.Generator,
) -> List[Dict[str, torch.Tensor]]:
    if len(batch_list) <= max_states:
        return batch_list
    idx = rng.choice(len(batch_list), size=max_states, replace=False)
    idx = np.sort(idx)
    return [batch_list[i] for i in idx]


def collect_onpolicy_observations(
    model: torch.nn.Module,
    benchmark,
    task_idx: int,
    *,
    num_episodes: int,
    max_steps: int,
    action_execution_horizon: int,
    action_mean: Optional[np.ndarray],
    action_std: Optional[np.ndarray],
    obs_horizon: int,
    image_size: Tuple[int, int],
    use_eye_in_hand: bool,
    low_dim_keys: Optional[List[str]],
    device: torch.device,
    use_ddim: bool,
    ddim_steps: int,
    max_states: int,
    seed: int,
    log_debug: bool = False,
) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    """Run student policy in the current task env; collect on-policy obs batches.

    Each stored batch matches ``obs_buffer_to_batch`` output (typically B=1).

    Returns:
        (list of batch dicts, number of states used after optional subsampling)
    """
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = "/tmp/numba_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir

    from libero.libero.envs import OffScreenRenderEnv

    if device is None:
        device = next(model.parameters()).device

    bddl_file = benchmark.get_task_bddl_file_path(task_idx)
    init_states = benchmark.get_task_init_states(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size[0],
        camera_widths=image_size[1],
    )
    env.seed(int(seed))

    model.eval()
    collected: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for ep in range(num_episodes):
            env.reset()
            init_state = init_states[ep % len(init_states)]
            obs = env.set_init_state(init_state)

            obs_buffer = collections.deque(maxlen=obs_horizon)
            obs_buffer.append(process_env_obs(obs))

            success = False
            steps = 0

            while steps < max_steps and not success:
                if len(collected) >= max_states:
                    break

                batch = obs_buffer_to_batch(
                    obs_buffer,
                    obs_horizon,
                    use_eye_in_hand,
                    device,
                    low_dim_keys=low_dim_keys,
                )

                collected.append(clone_obs_batch(batch))

                if use_ddim:
                    actions = predict_action_ddim(model, batch, ddim_steps)
                else:
                    actions = model.predict_action(batch)

                actions = actions[0].cpu().numpy()

                if action_mean is not None and action_std is not None:
                    actions = actions * action_std + action_mean

                n_exec = min(action_execution_horizon, len(actions))
                for a_idx in range(n_exec):
                    obs, reward, done, info = env.step(actions[a_idx])
                    steps += 1
                    obs_buffer.append(process_env_obs(obs))

                    if env.check_success():
                        success = True
                        break
                    if steps >= max_steps:
                        break

                if len(collected) >= max_states:
                    break

            if len(collected) >= max_states:
                break

    env.close()

    rng = np.random.default_rng(int(seed) + 17)
    n_before = len(collected)
    collected = subsample_obs_batches(collected, max_states, rng)
    n_after = len(collected)

    if log_debug:
        print(
            f"    [sdft] collected on-policy states: {n_before} (used {n_after}, cap={max_states})",
            flush=True,
        )

    return collected, n_after


def compute_sdft_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    stacked_batch: Dict[str, torch.Tensor],
    ddim_steps: int,
) -> torch.Tensor:
    """L_sdft = MSE(a_student, a_teacher) over full action horizon.

    Uses one shared DDIM initial noise for teacher and student so the
    surrogate targets are aligned (same deterministic denoising chain).
    """
    obs_cond = student.encode_obs(stacked_batch)
    B = obs_cond.shape[0]
    device = obs_cond.device
    x = torch.randn(
        B,
        student.action_horizon,
        student.action_dim,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        a_teacher = _predict_action_ddim_core(
            teacher, stacked_batch, ddim_steps, x_init=x
        )
    a_student = _predict_action_ddim_core(
        student, stacked_batch, ddim_steps, x_init=x
    )
    return F.mse_loss(a_student, a_teacher.detach())
