"""
Lazy-loading LIBERO-90 dataset with full-dataset shuffled epochs.

Unlike the eager dataset loader, this version keeps only metadata and flat
sample indices in memory. Action / observation arrays are read from HDF5 on
demand inside __getitem__. Each epoch iterates over every valid sample window
once in shuffled order.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LiberoUniformDataset(Dataset):
    """
    Lazy-loading variant of the LIBERO-90 dataset.

    Keeps file paths, episode metadata, and sample indices in memory while
    reading action / observation windows on demand from HDF5.
    """

    def __init__(
        self,
        data_dir: str,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        obs_keys: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (128, 128),
        use_eye_in_hand: bool = True,
        normalize_action: bool = True,
        max_episodes_per_task: Optional[int] = None,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_eye_in_hand = use_eye_in_hand
        self.normalize_action = normalize_action

        if obs_keys is None:
            obs_keys = [
                "agentview_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
        self.obs_keys = obs_keys

        hdf5_files = sorted(glob.glob(os.path.join(data_dir, "**/*.hdf5"), recursive=True))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        print(f"Found {len(hdf5_files)} HDF5 files")

        self.task_data: List[Dict] = []
        self.task_names: List[str] = []
        self._file_handles: Dict[str, h5py.File] = {}
        self.action_mean: Optional[np.ndarray] = None
        self.action_std: Optional[np.ndarray] = None

        self._scan_all_tasks(hdf5_files, max_episodes_per_task)

        if self.normalize_action:
            self._compute_action_stats_streaming()

        self._build_index()
        print(f"Dataset ready: {len(self.task_data)} tasks, {self.total_samples} total samples")

    def _scan_all_tasks(self, hdf5_files: List[str], max_episodes: Optional[int]):
        for fpath in tqdm(hdf5_files, desc="Scanning tasks"):
            task_name = os.path.splitext(os.path.basename(fpath))[0]

            with h5py.File(fpath, "r") as f:
                if "data" not in f:
                    print(f"  Skipping {task_name}: no 'data' group")
                    continue

                demos = sorted(
                    f["data"].keys(),
                    key=lambda x: int(x.replace("demo_", "")),
                )
                if max_episodes:
                    demos = demos[:max_episodes]

                episodes = []
                for demo_key in demos:
                    demo = f["data"][demo_key]
                    obs_group = demo["obs"]

                    obs_sources = {}
                    for key in self.obs_keys:
                        if key in obs_group:
                            obs_sources[key] = key
                        elif key == "agentview_image" and "agentview_rgb" in obs_group:
                            obs_sources[key] = "agentview_rgb"

                    eye_in_hand_source = None
                    if self.use_eye_in_hand:
                        for eye_key in ["eye_in_hand_image", "eye_in_hand_rgb"]:
                            if eye_key in obs_group:
                                eye_in_hand_source = eye_key
                                break

                    episodes.append(
                        {
                            "demo_key": demo_key,
                            "length": int(demo["actions"].shape[0]),
                            "obs_sources": obs_sources,
                            "eye_in_hand_source": eye_in_hand_source,
                        }
                    )

                if episodes:
                    self.task_data.append(
                        {
                            "name": task_name,
                            "file_path": fpath,
                            "episodes": episodes,
                        }
                    )
                    self.task_names.append(task_name)
                    print(
                        f"  {task_name}: {len(episodes)} episodes, "
                        f"avg len={np.mean([e['length'] for e in episodes]):.0f}"
                    )

    def _compute_action_stats_streaming(self):
        sum_actions = None
        sumsq_actions = None
        total_steps = 0

        for task in tqdm(self.task_data, desc="Computing action stats"):
            with h5py.File(task["file_path"], "r") as f:
                for episode in task["episodes"]:
                    actions = f["data"][episode["demo_key"]]["actions"][:].astype(np.float64)
                    if sum_actions is None:
                        action_dim = actions.shape[-1]
                        sum_actions = np.zeros(action_dim, dtype=np.float64)
                        sumsq_actions = np.zeros(action_dim, dtype=np.float64)

                    sum_actions += actions.sum(axis=0)
                    sumsq_actions += np.square(actions).sum(axis=0)
                    total_steps += actions.shape[0]

        if total_steps == 0 or sum_actions is None or sumsq_actions is None:
            raise ValueError("Could not compute action statistics from an empty dataset")

        mean = sum_actions / total_steps
        var = np.maximum(sumsq_actions / total_steps - np.square(mean), 1e-12)
        std = np.sqrt(var)

        self.action_mean = mean.astype(np.float32)
        self.action_std = np.clip(std.astype(np.float32), 1e-6, None)
        print(f"Action stats computed: mean={self.action_mean}, std={self.action_std}")

    def _build_index(self):
        """Build a flat index: (task_idx, episode_idx, start_t) for all valid windows."""
        self.index = []
        for task_idx, task in enumerate(self.task_data):
            for ep_idx, ep in enumerate(task["episodes"]):
                max_start = ep["length"] - self.action_horizon + 1
                for start_t in range(max(0, max_start)):
                    self.index.append((task_idx, ep_idx, start_t))

        self.total_samples = len(self.index)

    def _get_file_handle(self, file_path: str) -> h5py.File:
        handle = self._file_handles.get(file_path)
        if handle is None:
            handle = h5py.File(file_path, "r")
            self._file_handles[file_path] = handle
        return handle

    def close(self):
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._file_handles = {}

    def __getstate__(self):
        # File handles must stay worker-local. Drop them when the dataset is
        # copied into DataLoader workers.
        state = self.__dict__.copy()
        state["_file_handles"] = {}
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, ep_idx, start_t = self.index[idx]
        task = self.task_data[task_idx]
        episode = task["episodes"][ep_idx]

        file_handle = self._get_file_handle(task["file_path"])
        demo = file_handle["data"][episode["demo_key"]]
        obs_group = demo["obs"]
        episode_length = episode["length"]

        obs_start = max(0, start_t - self.obs_horizon + 1)
        obs_end = start_t + 1
        act_end = min(start_t + self.action_horizon, episode_length)

        actions = demo["actions"][start_t:act_end].astype(np.float32)
        if len(actions) < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.normalize_action:
            actions = (actions - self.action_mean) / self.action_std

        result = {"action": torch.from_numpy(actions)}

        for key, source_key in episode["obs_sources"].items():
            obs_data = obs_group[source_key][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)

            if obs_data.ndim == 4:  # image: (T, H, W, C)
                obs_data = obs_data.astype(np.float32) / 255.0
                obs_data = np.transpose(obs_data, (0, 3, 1, 2))

            result[f"obs_{key}"] = torch.from_numpy(obs_data.astype(np.float32))

        eye_in_hand_source = episode["eye_in_hand_source"]
        if eye_in_hand_source is not None:
            obs_data = obs_group[eye_in_hand_source][obs_start:obs_end]
            if len(obs_data) < self.obs_horizon:
                pad = np.repeat(obs_data[:1], self.obs_horizon - len(obs_data), axis=0)
                obs_data = np.concatenate([pad, obs_data], axis=0)
            obs_data = obs_data.astype(np.float32) / 255.0
            obs_data = np.transpose(obs_data, (0, 3, 1, 2))
            result["obs_eye_in_hand_image"] = torch.from_numpy(obs_data)

        result["task_id"] = torch.tensor(task_idx, dtype=torch.long)
        return result


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    **dataset_kwargs,
) -> Tuple[DataLoader, LiberoUniformDataset]:
    dataset = LiberoUniformDataset(
        data_dir=data_dir,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        **dataset_kwargs,
    )

    try:
        torch.zeros(1).pin_memory()
        can_pin = True
    except RuntimeError:
        can_pin = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=can_pin,
        persistent_workers=num_workers > 0,
    )

    return loader, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    loader, dataset = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nDataset stats:")
    print(f"  Tasks: {len(dataset.task_data)}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Samples per epoch: {len(dataset)}")
    print(f"  Batches per epoch: {len(loader)}")

    batch = next(iter(loader))
    print("\nBatch sample:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape} ({value.dtype})")
