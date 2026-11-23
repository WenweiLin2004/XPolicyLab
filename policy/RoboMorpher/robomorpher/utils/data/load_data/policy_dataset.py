"""Unified LeRobot-v3 dataset adapter for VLA training."""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler




def normalize_action_type(value):
    value = str("joint" if value is None else value).lower()
    if value != "joint":
        raise ValueError("action_type must be joint")
    return value


class DistributedReplacementSampler(Sampler):
    def __init__(self, dataset, *, num_replicas=None, rank=None, num_samples=None, seed=0):
        if num_replicas is None:
            num_replicas = (torch.distributed.get_world_size()
                            if torch.distributed.is_available() and torch.distributed.is_initialized()
                            else 1)
        if rank is None:
            rank = (torch.distributed.get_rank()
                    if torch.distributed.is_available() and torch.distributed.is_initialized()
                    else 0)
        self.dataset = dataset
        self.num_replicas, self.rank, self.seed = int(num_replicas), int(rank), int(seed)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size")
        if len(dataset) < 1:
            raise ValueError("Replacement sampling requires a nonempty dataset")
        requested = len(dataset) if num_samples is None else int(num_samples)
        if requested < 1:
            raise ValueError("num_samples must be positive")
        self.requested_num_samples = requested
        self.num_samples = math.ceil(requested / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.global_num_samples = self.total_size
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randint(0, len(self.dataset), (self.total_size,),
                               generator=generator, dtype=torch.int64).tolist()
        begin = self.rank * self.num_samples
        return iter(indices[begin:begin + self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def collate_policy_samples(samples):
    return samples


def _numpy(value, dtype=np.float32):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _to_pil(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 0.0
        array = np.clip(array * 255.0 if maximum <= 1.0 else array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


class LeRobotPolicyDataset(Dataset):
    """Read RobotWin LeRobot v3 into the shared VLA sample contract.

    Each item uses observation.state[t] and action[t:t+H]. An episode
    with N stored action rows contributes max(N-H+1, 1) windows when N>0.
    Short episodes contribute one window padded by repeating their final action.
    Padded timesteps are marked invalid; other incomplete tails are discarded.
    No next-state shift is applied. All arrays remain lazy; only the selected row and video frames
    are decoded in __getitem__.
    """

    def __init__(self, data_root, *, num_arms=2, max_num_joints=7, action_chunk_size=50,
                 camera_names=("cam_high",), episode_files=None,
                 action_type="joint", action_representation=None, joint_auxiliary=False):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobot 0.4.4 is required for the unified loader."
            ) from exc
        if episode_files is not None:
            raise ValueError("episode_files filtering is not supported for a LeRobot root")
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.num_arms = int(num_arms)
        self.max_num_joints = int(max_num_joints)
        self.action_type = normalize_action_type(
            action_representation if action_type == "joint" and action_representation is not None
            else action_type)
        self.action_representation = self.action_type
        if not isinstance(joint_auxiliary, bool):
            raise ValueError("joint_auxiliary must be an explicit boolean")
        if joint_auxiliary or self.action_type != "joint":
            raise ValueError("end-effector action datasets are not supported")
        self.joint_auxiliary = False
        self.action_chunk_size = int(action_chunk_size)
        self.camera_names = tuple(camera_names)
        if self.num_arms < 1 or self.max_num_joints < 1 or self.action_chunk_size < 1:
            raise ValueError("Joint count and action chunk size must be positive")
        if not self.camera_names:
            raise ValueError("camera_names cannot be empty")
        self.dataset = LeRobotDataset(repo_id=Path(self.data_root).name, root=self.data_root,
                                      download_videos=False, video_backend="pyav")
        self.metadata = self.dataset.meta
        self.hf_dataset = self.dataset.hf_dataset
        columns = set(self.hf_dataset.column_names)
        required = {"observation.state", "action", "episode_index", "frame_index",
                    "observation.urdf_features", "observation.joint_valid_mask"}
        missing = required - columns
        if missing:
            raise ValueError(f"LeRobot dataset is missing columns: {sorted(missing)}")
        a = self.num_arms
        state_shape = tuple(self.metadata.features["observation.state"]["shape"])
        action_shape = tuple(self.metadata.features["action"]["shape"])
        if len(state_shape) != 1 or len(action_shape) != 1 or state_shape != action_shape:
            raise ValueError("LeRobot state/action features must be equal flat vectors")
        if state_shape[0] % a != 0 or state_shape[0] // a < 2:
            raise ValueError(
                f"LeRobot state length {state_shape[0]} does not match tokenizer "
                f"layout of {a} arms padded to {self.max_num_joints} joints"
            )
        self.source_num_joints = state_shape[0] // a - 1
        j = self.source_num_joints
        if j != self.max_num_joints:
            raise ValueError(
                f"Stored joints={j} does not match tokenizer max_num_joints={self.max_num_joints}. "
                "Actions are padded to 7 joints; a single arm is the last 7 joint_valid_mask values all false."
            )
        for name, shape in {
            "observation.state": (a * (j + 1),), "action": (a * (j + 1),),
            "observation.urdf_features": (a * j * 6,),
            "observation.joint_valid_mask": (a * j,),
        }.items():
            if tuple(self.metadata.features[name]["shape"]) != shape:
                raise ValueError(f"{name} shape differs from tokenizer layout {shape}")
        for camera in self.camera_names:
            if f"observation.images.{camera}" not in self.metadata.features:
                raise ValueError(f"Unknown camera {camera}")
        source_meta = Path(self.data_root) / "meta" / "dataset_metadata.json"
        dataset_source_metadata = json.loads(source_meta.read_text()) if source_meta.exists() else {}
        sources = dataset_source_metadata.get("sources", [])
        self.episodes, self.skipped_episodes, self._ends = [], [], []
        total = 0
        episode_table = self.metadata.episodes
        for row in episode_table:
            episode_id = int(row["episode_index"])
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            source_length = end - start
            if source_length <= 0:
                continue
            length = max(source_length - self.action_chunk_size + 1, 1)
            first = self.hf_dataset[start]
            urdf = _numpy(first["observation.urdf_features"]).reshape(a, j, 6)
            mask = _numpy(first["observation.joint_valid_mask"]).reshape(a, j).astype(bool)
            tasks = row.get("tasks", [])
            task = str(tasks[0] if tasks else first.get("task", f"episode_{episode_id}"))
            if int(first["episode_index"]) != episode_id or end > len(self.hf_dataset):
                raise ValueError(f"Invalid episode bounds: {episode_id}")
            source = sources[episode_id] if episode_id < len(sources) else {}
            episode = {
                "episode_index": episode_id, "start": start, "end": end,
                "path": f"{self.data_root}#episode={episode_id}",
                "urdf_paths": source.get("urdf_paths", [None] * a),
                "dimensions": mask.sum(-1).tolist(),
                "embodiment": Path(source["path"]).parent.parent.name if source.get("path") else None,
                "source_length": source_length, "length": length, "task": task,
                "urdf_features": urdf.copy(), "joint_valid_mask": mask.copy(),
            }
            if length <= 0:
                self.skipped_episodes.append(episode)
                continue
            self.episodes.append(episode)
            total += length
            self._ends.append(total)
        if not self.episodes:
            raise ValueError("No complete action chunks in the LeRobot dataset")

    def checkpoint_contract(self):
        contract = {
            "kind": "lerobot_action_chunks", "schema_version": 2,
            "format": "lerobot_v3", "state_source": "observation.state[t]",
            "target_source": "action[t:t+H]", "target_offset": 0,
            "action_chunk_size": self.action_chunk_size, "windows_per_episode": "max(N-H+1,1) for N>0",
            "normalization": "raw_physical_values; codec owns stats",
            "tail_policy": "repeat_last_for_short_episodes; mask_padded_timesteps; drop_other_incomplete_tails", "max_num_joints": self.max_num_joints,
            "camera_names": list(self.camera_names),
            "action_type": self.action_type,
            "action_representation": self.action_representation,
            "source_num_joints": self.source_num_joints,
            "state_layout": "physical_joint_state",
            "mask_semantics": "joint_valid_mask is physical-state padding",
        }
        # Keep the historical two-arm contract byte-compatible. Non-default
        # single-arm datasets carry their explicit arm count for resume checks.
        if self.num_arms != 2:
            contract["num_arms"] = self.num_arms
        return contract

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_i = bisect.bisect_right(self._ends, index)
        episode = self.episodes[episode_i]
        previous = self._ends[episode_i - 1] if episode_i else 0
        start = episode["start"] + index - previous
        stop = min(start + self.action_chunk_size, episode["end"])
        valid_length = stop - start
        first = self.hf_dataset[start]
        future = self.hf_dataset[start:stop]
        # Arrow slicing returns column lists/tensors; stack without loading
        # the whole dataset at construction time.
        a, j = self.num_arms, self.source_num_joints
        action = _numpy(future["action"]).reshape(valid_length, a, j + 1).copy()
        if valid_length < self.action_chunk_size:
            action = np.concatenate([action, np.repeat(action[-1:], self.action_chunk_size - valid_length, axis=0)], axis=0)
        state = _numpy(first["observation.state"]).reshape(a, j + 1)
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"Nonfinite action/state in episode {episode['episode_index']}")
        urdf = _numpy(first["observation.urdf_features"]).reshape(a, j, 6)
        mask = _numpy(first["observation.joint_valid_mask"]).reshape(a, j).astype(bool)
        if not np.array_equal(mask, episode["joint_valid_mask"]):
            raise ValueError("Joint layout changed inside an episode")
        # A missing arm is all mask bits false. Zero that arm so it cannot
        # enter normalization, loss, or outputs.
        action[:, :, :j] = np.where(mask[None, :, :], action[:, :, :j], 0.0)
        action[:, :, j] = np.where(mask.any(-1)[None], action[:, :, j], 0.0)
        state[:, :j] = np.where(mask, state[:, :j], 0.0)
        state[:, j] = np.where(mask.any(-1), state[:, j], 0.0)
        urdf = np.where(mask[..., None], urdf, 0.0)
        action_values = action[:, :, :j].copy()
        state_values = state[:, :j].copy()
        output_urdf = urdf.copy()
        output_mask = mask.copy()
        decoded = self.dataset[start]
        images = self._images_from_decoded(decoded)
        sample = {
            "image": images, "lang": str(decoded["task"]),
            "action_angle": action_values,
            "gripper": action[:, :, j].copy(),
            "state_angle": state_values,
            "state_gripper": state[:, j].copy(),
            "urdf_features": output_urdf,
            "joint_valid_mask": output_mask,
            "action_valid_mask": np.arange(self.action_chunk_size) < valid_length,
            "arm_valid_mask": output_mask.any(-1),
            "action_type": self.action_type,
            "action_representation": self.action_representation,
            "urdf_paths": episode["urdf_paths"], "joint_names": [[] for _ in range(self.num_arms)],
            "embodiment": episode["embodiment"], "episode_path": episode["path"],
            "step_id": int(first["frame_index"]), "episode_index": episode["episode_index"],
        }
        return sample


    def _images_from_decoded(self, decoded):
        images = []
        for camera in self.camera_names:
            key = f"observation.images.{camera}"
            if key not in decoded:
                raise KeyError(f"LeRobot image column is missing: {key}")
            images.append(_to_pil(decoded[key]))
        return images


__all__ = ["DistributedReplacementSampler", "LeRobotPolicyDataset",
           "collate_policy_samples"]
