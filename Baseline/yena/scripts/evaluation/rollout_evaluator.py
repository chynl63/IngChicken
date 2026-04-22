# -*- coding: utf-8 -*-
"""
Simulation-based rollout evaluation for LIBERO tasks.

Runs the policy in the LIBERO/robosuite environment and computes
task success rates via actual rollouts.
"""

import os
import collections
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# DDIM Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_action_ddim(model, batch, num_inference_steps=16):
    """DDIM-style accelerated inference (deterministic, eta=0).

    Uses evenly spaced timestep subsequence that avoids the terminal noise
    level where alphas_cumprod ≈ 0 (which causes numerical explosion in
    the x0 prediction formula).

    Args:
        model: DiffusionPolicy instance (or EMA copy).
        batch: dict of observation tensors on the correct device.
        num_inference_steps: number of denoising steps (< num_diffusion_steps).

    Returns:
        Predicted actions tensor of shape (B, action_horizon, action_dim).
    """
    obs_cond = model.encode_obs(batch)
    B = obs_cond.shape[0]
    device = obs_cond.device
    T = model.num_diffusion_steps

    x = torch.randn(B, model.action_horizon, model.action_dim, device=device)

    step_ratio = T // num_inference_steps
    ddim_timesteps = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int64)
    ddim_timesteps = np.flip(ddim_timesteps).copy()

    for i in range(len(ddim_timesteps)):
        t = int(ddim_timesteps[i])
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model.noise_pred_net(x, ts, obs_cond)

        alpha_cumprod_t = model.alphas_cumprod[t]

        if i + 1 < len(ddim_timesteps):
            alpha_cumprod_prev = model.alphas_cumprod[int(ddim_timesteps[i + 1])]
        else:
            alpha_cumprod_prev = torch.tensor(1.0, device=device)

        pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(
            alpha_cumprod_t
        )
        pred_dir = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir

    return x


# ─────────────────────────────────────────────
# Observation Processing
# ─────────────────────────────────────────────

def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) from robosuite to axis-angle (3D).

    Uses the same math as robosuite.utils.transform_utils.quat2axisangle
    to match the convention used in LIBERO demo data collection.
    """
    import math
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(w)
    return (quat_xyzw[:3] * angle / den).astype(np.float32)


def process_env_obs(obs: dict, obs_keys: list = None) -> dict:
    """Convert raw environment observation to processed format.

    Maps robosuite observation keys to match training data key names:
      agentview_image         -> agentview_image
      robot0_eye_in_hand_image -> eye_in_hand_image
      robot0_eef_pos          -> ee_pos
      robot0_eef_quat (4D)    -> ee_ori (3D axis-angle)
      robot0_gripper_qpos     -> gripper_states

    Images: uint8 (H,W,C) -> float32 (C,H,W) in [0,1]
    Low-dim: float64 -> float32
    """
    processed = {}

    img = obs.get("agentview_image")
    if img is not None:
        img = img.astype(np.float32) / 255.0
        processed["agentview_image"] = np.transpose(img, (2, 0, 1))

    eye_img = obs.get("robot0_eye_in_hand_image")
    if eye_img is not None:
        eye_img = eye_img.astype(np.float32) / 255.0
        processed["eye_in_hand_image"] = np.transpose(eye_img, (2, 0, 1))

    robosuite_to_demo_key = {
        "robot0_eef_pos": "ee_pos",
        "robot0_eef_quat": "ee_ori",
        "robot0_gripper_qpos": "gripper_states",
    }

    for rs_key, demo_key in robosuite_to_demo_key.items():
        if rs_key in obs:
            if rs_key == "robot0_eef_quat":
                processed[demo_key] = _quat_to_axis_angle(obs[rs_key])
            else:
                processed[demo_key] = obs[rs_key].astype(np.float32)

    return processed


def obs_buffer_to_batch(
    obs_buffer: collections.deque,
    obs_horizon: int,
    use_eye_in_hand: bool,
    device: torch.device,
    low_dim_keys: list = None,
) -> dict:
    """Stack observation buffer into a batched model input dict.

    low_dim_keys should match the model's config obs_keys (excluding images),
    e.g. ["ee_pos", "ee_ori", "gripper_states"] for LIBERO-Object.
    """
    if low_dim_keys is None:
        low_dim_keys = ["ee_pos", "ee_ori", "gripper_states"]

    buf_list = list(obs_buffer)
    while len(buf_list) < obs_horizon:
        buf_list.insert(0, buf_list[0])
    buf_list = buf_list[-obs_horizon:]

    batch = {}

    imgs = np.stack([o["agentview_image"] for o in buf_list])
    batch["obs_agentview_image"] = (
        torch.from_numpy(imgs).unsqueeze(0).to(device)
    )

    if use_eye_in_hand and "eye_in_hand_image" in buf_list[0]:
        imgs = np.stack([o["eye_in_hand_image"] for o in buf_list])
        batch["obs_eye_in_hand_image"] = (
            torch.from_numpy(imgs).unsqueeze(0).to(device)
        )

    for key in low_dim_keys:
        if key in buf_list[0]:
            data = np.stack([o[key] for o in buf_list])
            batch[f"obs_{key}"] = (
                torch.from_numpy(data).unsqueeze(0).to(device)
            )

    return batch


# ─────────────────────────────────────────────
# Rollout Evaluation
# ─────────────────────────────────────────────

def evaluate_policy_on_task(
    model: nn.Module,
    benchmark,
    task_idx: int,
    num_episodes: int = 20,
    max_steps: int = 600,
    action_execution_horizon: int = 8,
    action_mean: np.ndarray = None,
    action_std: np.ndarray = None,
    obs_horizon: int = 2,
    image_size: tuple = (128, 128),
    use_eye_in_hand: bool = True,
    low_dim_keys: list = None,
    device: torch.device = None,
    use_ddim: bool = True,
    ddim_steps: int = 16,
    seed: int = 42,
) -> tuple:
    """Evaluate a policy on a single LIBERO task via simulation rollouts.

    Args:
        model: DiffusionPolicy model in eval mode.
        benchmark: LIBERO benchmark instance.
        task_idx: index of the task to evaluate.
        num_episodes: number of rollout episodes.
        max_steps: maximum environment steps per episode.
        action_execution_horizon: how many predicted actions to execute before replanning.
        action_mean, action_std: for denormalizing predicted actions.
        obs_horizon: number of past observation frames.
        image_size: (H, W) for camera resolution.
        use_eye_in_hand: whether eye-in-hand camera is used.
        device: torch device.
        use_ddim: use DDIM inference (faster).
        ddim_steps: number of DDIM denoising steps.
        seed: random seed for environment.

    Returns:
        (success_rate, episode_results) where episode_results is a list of bools.
    """
    # Ensure MuJoCo uses a software renderer in headless environments that
    # do not support EGL device contexts (common on many clusters).
    # Using OSMesa avoids reliance on /dev/dri and EGL PLATFORM_DEVICE.
    # Force software rendering to avoid EGL initialization errors on clusters.
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # Make numba cache writable inside Singularity (robosuite uses numba).
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
    env.seed(seed)

    model.eval()
    successes = []

    for ep in range(num_episodes):
        env.reset()
        init_state = init_states[ep % len(init_states)]
        obs = env.set_init_state(init_state)

        obs_buffer = collections.deque(maxlen=obs_horizon)
        obs_buffer.append(process_env_obs(obs))

        success = False
        steps = 0

        while steps < max_steps and not success:
            batch = obs_buffer_to_batch(
                obs_buffer, obs_horizon, use_eye_in_hand, device,
                low_dim_keys=low_dim_keys,
            )

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

        successes.append(success)
        n_done = len(successes)
        n_succ = sum(successes)
        print(f"    Episode {n_done}/{num_episodes}: "
              f"{'SUCCESS' if success else 'FAIL'} "
              f"(steps={steps}) | "
              f"Running SR: {n_succ}/{n_done} = {n_succ/n_done:.2f}",
              flush=True)

    env.close()

    success_rate = float(np.mean(successes))
    return success_rate, successes


def evaluate_checkpoint_on_all_tasks(
    model: nn.Module,
    benchmark,
    task_indices: list,
    num_episodes: int = 20,
    max_steps: int = 600,
    action_execution_horizon: int = 8,
    action_mean: np.ndarray = None,
    action_std: np.ndarray = None,
    obs_horizon: int = 2,
    image_size: tuple = (128, 128),
    use_eye_in_hand: bool = True,
    low_dim_keys: list = None,
    device: torch.device = None,
    use_ddim: bool = True,
    ddim_steps: int = 16,
    seed: int = 42,
) -> dict:
    """Evaluate a checkpoint on multiple tasks.

    Returns:
        dict mapping task_idx -> success_rate
    """
    results = {}
    task_names = benchmark.get_task_names()
    for task_idx in task_indices:
        sr, ep_results = evaluate_policy_on_task(
            model=model,
            benchmark=benchmark,
            task_idx=task_idx,
            num_episodes=num_episodes,
            max_steps=max_steps,
            action_execution_horizon=action_execution_horizon,
            action_mean=action_mean,
            action_std=action_std,
            obs_horizon=obs_horizon,
            image_size=image_size,
            use_eye_in_hand=use_eye_in_hand,
            low_dim_keys=low_dim_keys,
            device=device,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            seed=seed,
        )
        results[task_idx] = sr
        print(
            f"    Task {task_idx} [{task_names[task_idx][:50]}]: "
            f"SR = {sr:.2f} ({sum(ep_results)}/{len(ep_results)})"
        )
    return results
