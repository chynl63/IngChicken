# -*- coding: utf-8 -*-
"""
Simulation-based rollout evaluation for LIBERO tasks.

Runs the policy in the LIBERO/robosuite environment and computes
task success rates via actual rollouts.
"""

import os
import collections
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# DDIM Inference
# ─────────────────────────────────────────────

def _predict_action_ddim_core(
    model, batch, num_inference_steps=16, x_init: Optional[torch.Tensor] = None
):
    """DDIM-style accelerated inference (deterministic, eta=0).

    Shared implementation for ``predict_action_ddim`` (no grad) and
    ``predict_action_ddim_grad`` (gradients enabled for student SDFT).

    Uses evenly spaced timestep subsequence that avoids the terminal noise
    level where alphas_cumprod ≈ 0 (which causes numerical explosion in
    the x0 prediction formula).

    Args:
        model: DiffusionPolicy instance (or EMA copy).
        batch: dict of observation tensors on the correct device.
        num_inference_steps: number of denoising steps (< num_diffusion_steps).
        x_init: optional initial noise (B, action_horizon, action_dim). If set,
            the same tensor should be used for student and teacher in SDFT so
            DDIM trajectories are comparable (only the first denoising step reads
            this buffer; the loop rebinds ``x`` afterward).

    Returns:
        Predicted actions tensor of shape (B, action_horizon, action_dim).
    """
    obs_cond = model.encode_obs(batch)
    B = obs_cond.shape[0]
    device = obs_cond.device
    T = model.num_diffusion_steps

    if x_init is None:
        x = torch.randn(B, model.action_horizon, model.action_dim, device=device)
    else:
        x = x_init

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


@torch.no_grad()
def predict_action_ddim(model, batch, num_inference_steps=16, x_init=None):
    """DDIM inference without gradients (rollout / teacher)."""
    return _predict_action_ddim_core(model, batch, num_inference_steps, x_init=x_init)


def predict_action_ddim_grad(model, batch, num_inference_steps=16, x_init=None):
    """DDIM inference with gradients enabled (SDFT student path)."""
    return _predict_action_ddim_core(model, batch, num_inference_steps, x_init=x_init)


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


def _resize_hwc_u8(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize (H,W,3) uint8 → same dtype/shape layout."""
    t = (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
    )
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return (
        t.squeeze(0)
        .permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _postprocess_agentview_for_video(
    img: np.ndarray,
    rotate_180: bool,
    crop_bottom_frac: float,
) -> np.ndarray:
    """Adjust agentview for human viewing only (policy still uses raw obs)."""
    out = np.asarray(img, dtype=np.uint8)
    if rotate_180:
        out = np.ascontiguousarray(out[::-1, ::-1, :])
    frac = float(crop_bottom_frac)
    if frac > 0.0 and frac < 0.95:
        h, w = out.shape[:2]
        cut = max(1, int(round(h * frac)))
        keep = h - cut
        if keep >= 2:
            out = _resize_hwc_u8(out[:keep, :, :], h, w)
    return out


def _append_agentview_frame(
    frames: List[np.ndarray],
    obs: dict,
    *,
    rotate_180: bool = False,
    crop_bottom_frac: float = 0.0,
) -> None:
    """Append one RGB uint8 frame for video (optional rotate / reframe)."""
    img = obs.get("agentview_image")
    if img is None:
        return
    frames.append(
        _postprocess_agentview_for_video(
            img, rotate_180=rotate_180, crop_bottom_frac=crop_bottom_frac
        )
    )


def _save_episode_video_mp4(
    frames: List[np.ndarray],
    out_path: Path,
    fps: float,
) -> None:
    """Write frames to mp4 using imageio (requires imageio-ffmpeg for H.264)."""
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    imageio.mimsave(out_path, frames, fps=fps)


def _save_rollout_mp4(
    frames: List[np.ndarray],
    *,
    checkpoint_stem: str,
    task_idx: int,
    ep: int,
    success: bool,
    video_root: Path,
    video_fps: float,
) -> None:
    fname = (
        f"{checkpoint_stem}__task_{task_idx:02d}__ep_{ep:02d}__"
        f"success_{1 if success else 0}.mp4"
    )
    out_path = Path(video_root) / fname
    try:
        _save_episode_video_mp4(frames, out_path, video_fps)
    except Exception as e:
        print(
            f"    [video] Failed to save {out_path}: {e}",
            flush=True,
        )


def _video_balanced_two_debug(message: str) -> None:
    print(f"    [video balanced_two] {message}", flush=True)


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
    save_video: bool = False,
    video_root: Optional[Path] = None,
    checkpoint_stem: str = "checkpoint",
    video_fps: float = 10.0,
    num_videos_per_task: int = 2,
    video_rotate_180: bool = False,
    video_crop_bottom_frac: float = 0.0,
    video_episode_policy: str = "first_k",
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
        save_video: if True, save rollout videos for the first num_videos_per_task episodes.
        video_root: directory to write mp4 files (e.g. results_dir/videos/after_task_03).
        checkpoint_stem: prefix used in output filenames.
        video_fps: frames per second for saved videos.
        num_videos_per_task: max number of episodes per task to record (first episodes first);
            ignored when video_episode_policy is ``balanced_two``.
        video_rotate_180: if True, rotate saved video frames 180° (agentview upright for viewing).
        video_crop_bottom_frac: fraction of height to crop from bottom (floor), then rescale;
            only affects saved video.
        video_episode_policy: ``first_k`` — save video for the first ``num_videos_per_task``
            episodes (legacy). ``balanced_two`` — ep0 always saved; ep1 pending in RAM until
            task end if all outcomes match ep0, else flush mixed episode and stop capture.

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

    if video_episode_policy not in ("first_k", "balanced_two"):
        raise ValueError(
            "video_episode_policy must be 'first_k' or 'balanced_two', "
            f"got {video_episode_policy!r}"
        )

    balanced_two = (
        bool(save_video)
        and video_root is not None
        and video_episode_policy == "balanced_two"
    )
    first_k_video = (
        bool(save_video)
        and video_root is not None
        and not balanced_two
        and num_videos_per_task > 0
    )

    first_label: Optional[bool] = None
    pending_ep1: Optional[List[np.ndarray]] = None
    mixed_seen = False

    if balanced_two:
        _video_balanced_two_debug(
            f"task {task_idx} start, num_episodes={num_episodes} "
            f"(num_videos_per_task={num_videos_per_task} 무시됨)"
        )

    for ep in range(num_episodes):
        env.reset()
        init_state = init_states[ep % len(init_states)]
        obs = env.set_init_state(init_state)

        if balanced_two:
            capture_this_ep = ep == 0 or ep == 1 or (not mixed_seen)
        else:
            capture_this_ep = first_k_video and ep < num_videos_per_task

        video_frames: Optional[List[np.ndarray]]
        if capture_this_ep:
            video_frames = []
            _append_agentview_frame(
                video_frames,
                obs,
                rotate_180=video_rotate_180,
                crop_bottom_frac=video_crop_bottom_frac,
            )
        else:
            video_frames = None

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
                if capture_this_ep:
                    _append_agentview_frame(
                        video_frames,
                        obs,
                        rotate_180=video_rotate_180,
                        crop_bottom_frac=video_crop_bottom_frac,
                    )

                if env.check_success():
                    success = True
                    break
                if steps >= max_steps:
                    break

        if balanced_two and capture_this_ep and video_frames is not None:
            nfr = len(video_frames)
            if nfr == 0:
                _video_balanced_two_debug(
                    f"ep{ep:02d}: 프레임 없음 → 버퍼/디스크 모두 없음 (success={success})"
                )
                if ep == 0:
                    first_label = success
            elif ep == 0:
                _video_balanced_two_debug(
                    f"ep{ep:02d} is flushed to the disk "
                    f"(success={int(success)}, frames={nfr})"
                )
                _save_rollout_mp4(
                    video_frames,
                    checkpoint_stem=checkpoint_stem,
                    task_idx=task_idx,
                    ep=ep,
                    success=success,
                    video_root=Path(video_root),
                    video_fps=video_fps,
                )
                first_label = success
            elif ep == 1:
                if first_label is None:
                    raise RuntimeError("balanced_two: ep=1 done but first_label unset")
                if success == first_label:
                    pending_ep1 = video_frames
                    _video_balanced_two_debug(
                        f"ep{ep:02d}: 버퍼에 유지 (ep00과 라벨 동일, 디스크 미저장, "
                        f"frames={nfr})"
                    )
                else:
                    _video_balanced_two_debug(
                        f"ep{ep:02d} is flushed to the disk "
                        f"(mixed; success={int(success)}, ep00_label={int(first_label)}, "
                        f"frames={nfr})"
                    )
                    _save_rollout_mp4(
                        video_frames,
                        checkpoint_stem=checkpoint_stem,
                        task_idx=task_idx,
                        ep=ep,
                        success=success,
                        video_root=Path(video_root),
                        video_fps=video_fps,
                    )
                    pending_ep1 = None
                    mixed_seen = True
                    _video_balanced_two_debug(
                        "이후 에피소드: 버퍼에도 올리지 않음 (mixed 확정)"
                    )
            elif not mixed_seen:
                if success == first_label:
                    _video_balanced_two_debug(
                        f"ep{ep:02d}: 버퍼에서 제거, 디스크 저장 안 됨 "
                        f"(ep00과 라벨 동일·scratch 폐기, frames={nfr})"
                    )
                else:
                    _video_balanced_two_debug(
                        f"ep{ep:02d} is flushed to the disk "
                        f"(mixed; success={int(success)}, ep00_label={int(first_label)}, "
                        f"frames={nfr}) · ep01 pending은 버퍼에서 폐기, 디스크 저장 안 됨"
                    )
                    _save_rollout_mp4(
                        video_frames,
                        checkpoint_stem=checkpoint_stem,
                        task_idx=task_idx,
                        ep=ep,
                        success=success,
                        video_root=Path(video_root),
                        video_fps=video_fps,
                    )
                    pending_ep1 = None
                    mixed_seen = True
                    _video_balanced_two_debug(
                        "이후 에피소드: 버퍼에도 올리지 않음 (mixed 확정)"
                    )
        elif (
            first_k_video
            and capture_this_ep
            and video_frames is not None
            and len(video_frames) > 0
        ):
            _save_rollout_mp4(
                video_frames,
                checkpoint_stem=checkpoint_stem,
                task_idx=task_idx,
                ep=ep,
                success=success,
                video_root=Path(video_root),
                video_fps=video_fps,
            )

        successes.append(success)
        n_done = len(successes)
        n_succ = sum(successes)
        print(f"    Episode {n_done}/{num_episodes}: "
              f"{'SUCCESS' if success else 'FAIL'} "
              f"(steps={steps}) | "
              f"Running SR: {n_succ}/{n_done} = {n_succ/n_done:.2f}",
              flush=True)

    if (
        balanced_two
        and (not mixed_seen)
        and pending_ep1 is not None
        and len(pending_ep1) > 0
        and first_label is not None
    ):
        nfr = len(pending_ep1)
        _video_balanced_two_debug(
            f"ep01 is flushed to the disk "
            f"(태스크 종료·전 에피 라벨이 ep00과 동일, success={int(first_label)}, "
            f"frames={nfr})"
        )
        _save_rollout_mp4(
            pending_ep1,
            checkpoint_stem=checkpoint_stem,
            task_idx=task_idx,
            ep=1,
            success=first_label,
            video_root=Path(video_root),
            video_fps=video_fps,
        )

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
    save_video: bool = False,
    video_root: Optional[Path] = None,
    checkpoint_stem: str = "checkpoint",
    video_fps: float = 10.0,
    num_videos_per_task: int = 2,
    video_rotate_180: bool = False,
    video_crop_bottom_frac: float = 0.0,
    video_episode_policy: str = "first_k",
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
            save_video=save_video,
            video_root=video_root,
            checkpoint_stem=checkpoint_stem,
            video_fps=video_fps,
            num_videos_per_task=num_videos_per_task,
            video_rotate_180=video_rotate_180,
            video_crop_bottom_frac=video_crop_bottom_frac,
            video_episode_policy=video_episode_policy,
        )
        results[task_idx] = sr
        print(
            f"    Task {task_idx} [{task_names[task_idx][:50]}]: "
            f"SR = {sr:.2f} ({sum(ep_results)}/{len(ep_results)})"
        )
    return results
