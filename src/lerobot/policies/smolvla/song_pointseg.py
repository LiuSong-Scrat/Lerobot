#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

ROLE_BACKGROUND = 0
ROLE_FOREGROUND = 1
ROLE_IGNORE = -100
ROLE_NAMES = ("background", "foreground")
ROLE_COLORS = np.array(
    [
        [128, 128, 128],
        [44, 202, 124],
    ],
    dtype=np.uint8,
)

DEFAULT_FUTURE_OFFSETS = (1, 2, 4, 8, 16, 31)
MOTION_PRIOR_DIM = 8
POINTSEG_CACHE_VERSION = 1
POINTSEG_CACHE_FIELDS = (
    "point_cloud",
    "priors",
    "labels",
    "weights",
    "class_scores",
    "role_scores",
    "foreground_score",
    "episode_index",
    "frame_index",
    "dataset_index",
)


@dataclass(frozen=True)
class PseudoLabelConfig:
    held_sigma: float = 0.025
    static_sigma: float = 0.025
    held_margin: float = 0.006
    motion_tau: float = 0.012
    gripper_sigma: float = 0.045
    trajectory_sigma: float = 0.12
    approach_margin: float = 0.012
    approach_tau: float = 0.035
    min_confidence: float = 0.20
    background_min_confidence: float = 0.55
    background_foreground_max: float = 0.12
    background_label_weight: float = 0.12
    min_foreground_fraction: float = 0.01
    forced_foreground_min_score: float = 0.05
    teacher_confidence: float = 0.86
    teacher_geometry_gate: float = 0.18
    teacher_background_min_score: float = 0.65
    teacher_background_foreground_max: float = 0.08
    ignore_index: int = ROLE_IGNORE
    nn_chunk_size: int = 1024


@dataclass(frozen=True)
class SongPointSegLossConfig:
    ce_weight: float = 1.0
    foreground_bce_weight: float = 0.35
    smoothness_weight: float = 0.04
    motion_consistency_weight: float = 0.20
    smooth_voxel_size: float = 0.025
    class_weights: tuple[float, float] = (0.50, 2.00)
    ignore_index: int = ROLE_IGNORE


def rot6d_to_matrix(d6: Tensor) -> Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = functional.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(rot: Tensor) -> Tensor:
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def pose9_to_matrix(pose9: Tensor) -> Tensor:
    pose9 = pose9.to(dtype=torch.float32)
    transform = torch.zeros(*pose9.shape[:-1], 4, 4, dtype=pose9.dtype, device=pose9.device)
    transform[..., 3, 3] = 1.0
    transform[..., :3, :3] = rot6d_to_matrix(pose9[..., 3:9])
    transform[..., :3, 3] = pose9[..., :3]
    return transform


def matrix_to_pose9(transform: Tensor) -> Tensor:
    return torch.cat([transform[..., :3, 3], matrix_to_rot6d(transform[..., :3, :3])], dim=-1)


def invert_transform(transform: Tensor) -> Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -(rot_inv @ trans.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = trans_inv
    out[..., 3, 3] = 1.0
    return out


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    return torch.matmul(points, rot.transpose(-1, -2)) + trans.unsqueeze(-2)


def relative_poses_to_first(pose9: Tensor) -> Tensor:
    """Convert a pose sequence in one common frame into poses relative to the first pose."""
    transforms = pose9_to_matrix(pose9)
    first_inv = invert_transform(transforms[..., :1, :, :])
    relative = first_inv @ transforms
    return matrix_to_pose9(relative)


def _identity_pose9(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    pose = torch.zeros(9, device=device, dtype=dtype)
    pose[3] = 1.0
    pose[7] = 1.0
    return pose


def _sample_rows(array: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("Point sample count must be positive.")
    n_points = array.shape[0]
    if n_points == 0:
        return np.zeros((count, array.shape[-1]), dtype=np.float32)
    replace = n_points < count
    indices = rng.choice(n_points, count, replace=replace)
    return np.ascontiguousarray(array[indices], dtype=np.float32)


class SongTemporalPointCloudDataset(torch.utils.data.Dataset):
    """Adds current/future point clouds and current-frame relative future poses to a LeRobot dataset."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        point_cloud_dir: str | Path,
        *,
        future_offsets: tuple[int, ...] | list[int] = DEFAULT_FUTURE_OFFSETS,
        current_points: int = 8192,
        future_points: int = 16384,
        seed: int = 1000,
        return_full_point_cloud: bool = False,
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.point_cloud_dir = Path(point_cloud_dir)
        self.future_offsets = tuple(int(offset) for offset in future_offsets)
        if any(offset <= 0 for offset in self.future_offsets):
            raise ValueError("future_offsets should contain positive frame offsets; current frame is added automatically.")
        self.temporal_offsets = (0, *self.future_offsets)
        self.current_points = int(current_points)
        self.future_points = int(future_points)
        self.seed = int(seed)
        self.return_full_point_cloud = return_full_point_cloud
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[int, np.ndarray] = {}

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_int(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, episode_index: int) -> np.ndarray:
        point_clouds = self._point_cloud_cache.get(episode_index)
        if point_clouds is None:
            path = self.point_cloud_dir / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Point cloud memmap file is missing: {path}")
            point_clouds = np.load(path, mmap_mode=self.mmap_mode)
            self._point_cloud_cache[episode_index] = point_clouds
        return point_clouds

    def _relative_future_poses(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.ndim != 2 or action_chunk.shape[-1] < 9:
            raise ValueError(
                "SongTemporalPointCloudDataset expects the wrapped dataset to return an action chunk "
                f"with shape (T, >=9), got {tuple(action_chunk.shape)}."
            )
        max_offset = max(self.temporal_offsets)
        if action_chunk.shape[0] <= max_offset:
            raise ValueError(
                f"Action chunk length {action_chunk.shape[0]} is too short for max temporal offset {max_offset}. "
                "Create the base LeRobotDataset with action delta timestamps covering every requested offset."
            )
        poses = action_chunk[list(self.temporal_offsets), :9].to(dtype=torch.float32)
        rel = relative_poses_to_first(poses.unsqueeze(0)).squeeze(0)
        rel[0] = _identity_pose9(device=rel.device, dtype=rel.dtype)
        return rel

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[idx])
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        point_clouds = self._episode_point_clouds(episode_index)
        episode_len = int(point_clouds.shape[0])
        rng = np.random.default_rng(self.seed + idx)

        current_full = np.asarray(point_clouds[frame_index], dtype=np.float32)
        item["observation.point_cloud"] = torch.from_numpy(
            _sample_rows(current_full, self.current_points, rng)
        )

        future_samples = []
        future_is_pad = []
        for offset in self.temporal_offsets:
            raw_index = frame_index + offset
            clamped_index = min(max(raw_index, 0), episode_len - 1)
            future_is_pad.append(raw_index >= episode_len)
            future_samples.append(_sample_rows(np.asarray(point_clouds[clamped_index]), self.future_points, rng))

        item["observation.point_cloud_future"] = torch.from_numpy(np.stack(future_samples, axis=0))
        item["future_is_pad"] = torch.tensor(future_is_pad, dtype=torch.bool)
        item["future_offsets"] = torch.tensor(self.temporal_offsets, dtype=torch.long)
        item["future_ee_poses"] = self._relative_future_poses(item["action"])

        if self.return_full_point_cloud:
            item["observation.point_cloud_full"] = torch.from_numpy(np.ascontiguousarray(current_full))

        return item


class SongPointSegCachedDataset(torch.utils.data.Dataset):
    """Reads offline Song pointseg training samples from sharded mmap arrays."""

    def __init__(self, cache_dir: str | Path, *, mmap_mode: str = "r"):
        self.cache_dir = Path(cache_dir)
        self.mmap_mode = mmap_mode
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Song pointseg cache manifest is missing: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        version = int(self.manifest.get("version", -1))
        if version != POINTSEG_CACHE_VERSION:
            raise ValueError(
                f"Unsupported Song pointseg cache version {version}; expected {POINTSEG_CACHE_VERSION}."
            )

        fields = tuple(self.manifest.get("fields", ()))
        missing_fields = [field for field in POINTSEG_CACHE_FIELDS if field not in fields]
        if missing_fields:
            raise ValueError(f"Song pointseg cache is missing fields: {missing_fields}")
        self.fields = fields

        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"Song pointseg cache has no shards: {manifest_path}")
        self.shard_lengths = [int(shard["length"]) for shard in self.shards]
        self._cumulative_lengths = np.cumsum(self.shard_lengths).tolist()
        self._array_cache: dict[int, dict[str, np.ndarray]] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array_cache"] = {}
        return state

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def _locate_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_index = bisect_right(self._cumulative_lengths, idx)
        previous = self._cumulative_lengths[shard_index - 1] if shard_index > 0 else 0
        return shard_index, idx - previous

    def _open_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        arrays = self._array_cache.get(shard_index)
        if arrays is not None:
            return arrays

        shard_dir = self.cache_dir / self.shards[shard_index]["path"]
        arrays = {}
        for field in self.fields:
            path = shard_dir / f"{field}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Song pointseg cache array is missing: {path}")
            arrays[field] = np.load(path, mmap_mode=self.mmap_mode)
        self._array_cache[shard_index] = arrays
        return arrays

    @staticmethod
    def _float_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())

    @staticmethod
    def _long_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.int64).copy())

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        shard_index, local_index = self._locate_index(idx)
        arrays = self._open_shard(shard_index)
        return {
            "observation.point_cloud": self._float_tensor(arrays["point_cloud"][local_index]),
            "pointseg.priors": self._float_tensor(arrays["priors"][local_index]),
            "pointseg.labels": self._long_tensor(arrays["labels"][local_index]),
            "pointseg.weights": self._float_tensor(arrays["weights"][local_index]),
            "pointseg.class_scores": self._float_tensor(arrays["class_scores"][local_index]),
            "pointseg.role_scores": self._float_tensor(arrays["role_scores"][local_index]),
            "pointseg.foreground_score": self._float_tensor(arrays["foreground_score"][local_index]),
            "episode_index": torch.tensor(int(arrays["episode_index"][local_index]), dtype=torch.long),
            "frame_index": torch.tensor(int(arrays["frame_index"][local_index]), dtype=torch.long),
            "dataset_index": torch.tensor(int(arrays["dataset_index"][local_index]), dtype=torch.long),
        }


def _nearest_distances(query: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    chunks = []
    for start in range(0, query.shape[0], chunk_size):
        query_chunk = query[start : start + chunk_size]
        chunks.append(torch.cdist(query_chunk.unsqueeze(0), target.unsqueeze(0)).squeeze(0).amin(dim=-1))
    return torch.cat(chunks, dim=0)


def _motion_residuals_single(
    current_xyz: Tensor,
    future_xyz: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    held_residuals = []
    static_residuals = []
    transforms = pose9_to_matrix(future_poses)
    inverse_transforms = invert_transform(transforms)

    for k in range(1, future_xyz.shape[0]):
        if bool(future_is_pad[k].item()):
            continue
        target_xyz = future_xyz[k]
        held_residuals.append(_nearest_distances(current_xyz, target_xyz, chunk_size))
        static_query = transform_points(current_xyz.unsqueeze(0), inverse_transforms[k].unsqueeze(0)).squeeze(0)
        static_residuals.append(_nearest_distances(static_query, target_xyz, chunk_size))

    if not held_residuals:
        inf = torch.full((current_xyz.shape[0],), 1.0, dtype=current_xyz.dtype, device=current_xyz.device)
        return inf, inf

    return torch.stack(held_residuals, dim=0).amin(dim=0), torch.stack(static_residuals, dim=0).amin(dim=0)


def compute_motion_priors(
    current_pc: Tensor,
    future_pc: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor | None = None,
    *,
    nn_chunk_size: int = 1024,
) -> dict[str, Tensor]:
    """Compute geometry/motion priors used by pseudo labels and the segmentation network."""
    if current_pc.ndim != 3 or future_pc.ndim != 4 or future_poses.ndim != 3:
        raise ValueError("Expected current_pc (B,N,6), future_pc (B,K,M,6), future_poses (B,K,9).")
    bsize, n_points = current_pc.shape[:2]
    if future_is_pad is None:
        future_is_pad = torch.zeros(
            bsize, future_pc.shape[1], dtype=torch.bool, device=current_pc.device
        )
    future_is_pad = future_is_pad.to(device=current_pc.device, dtype=torch.bool)
    current_xyz = current_pc[..., :3].to(dtype=torch.float32)
    future_xyz = future_pc[..., :3].to(device=current_pc.device, dtype=torch.float32)
    future_poses = future_poses.to(device=current_pc.device, dtype=torch.float32)

    held = []
    static = []
    for bidx in range(bsize):
        held_res, static_res = _motion_residuals_single(
            current_xyz[bidx],
            future_xyz[bidx],
            future_poses[bidx],
            future_is_pad[bidx],
            chunk_size=nn_chunk_size,
        )
        held.append(held_res)
        static.append(static_res)
    held_residual = torch.stack(held, dim=0)
    static_residual = torch.stack(static, dim=0)

    trajectory_xyz = future_poses[..., :3]
    valid_traj = ~future_is_pad
    point_to_traj = torch.linalg.norm(current_xyz[:, :, None, :] - trajectory_xyz[:, None, :, :], dim=-1)
    point_to_traj = point_to_traj.masked_fill(~valid_traj[:, None, :], float("inf"))
    min_traj_dist = point_to_traj.amin(dim=-1)
    min_traj_dist = torch.where(torch.isfinite(min_traj_dist), min_traj_dist, torch.zeros_like(min_traj_dist))

    ee_dist = torch.linalg.norm(current_xyz, dim=-1)
    start_dist = point_to_traj[..., 0]
    future_min_dist = point_to_traj[..., 1:].amin(dim=-1) if point_to_traj.shape[-1] > 1 else start_dist
    future_min_dist = torch.where(torch.isfinite(future_min_dist), future_min_dist, start_dist)
    approach_delta = start_dist - future_min_dist
    residual_gap = static_residual - held_residual

    priors = torch.stack(
        [
            ee_dist,
            min_traj_dist,
            approach_delta,
            held_residual,
            static_residual,
            residual_gap,
            torch.exp(-held_residual / 0.05),
            torch.exp(-static_residual / 0.05),
        ],
        dim=-1,
    )

    return {
        "priors": priors,
        "ee_dist": ee_dist,
        "min_traj_dist": min_traj_dist,
        "approach_delta": approach_delta,
        "held_residual": held_residual,
        "static_residual": static_residual,
        "residual_gap": residual_gap,
    }


def generate_pseudo_labels_from_priors(
    priors_or_dict: Tensor | dict[str, Tensor],
    *,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    if torch.is_tensor(priors_or_dict):
        priors = priors_or_dict
        if priors.shape[-1] != MOTION_PRIOR_DIM:
            raise ValueError(f"Expected priors last dim={MOTION_PRIOR_DIM}, got {priors.shape[-1]}.")
        prior_dict = {
            "priors": priors,
            "ee_dist": priors[..., 0],
            "min_traj_dist": priors[..., 1],
            "approach_delta": priors[..., 2],
            "held_residual": priors[..., 3],
            "static_residual": priors[..., 4],
            "residual_gap": priors[..., 5],
        }
    else:
        prior_dict = dict(priors_or_dict)

    ee_dist = prior_dict["ee_dist"]
    min_traj_dist = prior_dict["min_traj_dist"]
    approach_delta = prior_dict["approach_delta"]
    held_residual = prior_dict["held_residual"]
    static_residual = prior_dict["static_residual"]
    residual_gap = prior_dict["residual_gap"]

    held_score = torch.exp(-held_residual / config.held_sigma)
    static_score = torch.exp(-static_residual / config.static_sigma)
    motion_score = torch.sigmoid((residual_gap - config.held_margin) / config.motion_tau)
    gripper_near = torch.exp(-ee_dist / config.gripper_sigma)
    trajectory_near = torch.exp(-min_traj_dist / config.trajectory_sigma)
    approach_score = torch.sigmoid((approach_delta - config.approach_margin) / config.approach_tau)

    gripper_score = held_score * gripper_near
    condition_score = held_score * motion_score * trajectory_near * (1.0 - gripper_score).clamp_min(0.0)
    target_score = static_score * approach_score * trajectory_near * (1.0 - gripper_score).clamp_min(0.0)
    role_scores = torch.stack([gripper_score, condition_score, target_score], dim=-1)
    foreground_score = role_scores.amax(dim=-1)
    background_score = (1.0 - foreground_score).clamp_min(0.0) * (0.5 + 0.5 * static_score) * (
        1.0 - 0.7 * approach_score
    ).clamp_min(0.0)

    class_scores = torch.stack([background_score, foreground_score], dim=-1)
    confidence, labels = class_scores.max(dim=-1)
    labels = labels.to(dtype=torch.long)
    foreground_valid = (labels == ROLE_FOREGROUND) & (confidence >= config.min_confidence)
    background_valid = (
        (labels == ROLE_BACKGROUND)
        & (background_score >= config.background_min_confidence)
        & (foreground_score <= config.background_foreground_max)
    )
    valid = foreground_valid | background_valid
    labels = torch.where(valid, labels, torch.full_like(labels, config.ignore_index))
    weights = confidence.detach().clamp(0.05, 1.0)
    weights = torch.where(labels == ROLE_BACKGROUND, weights * config.background_label_weight, weights)
    weights = torch.where(labels == config.ignore_index, torch.zeros_like(weights), weights)
    labels, weights = _promote_minimum_foreground(labels, weights, class_scores.detach(), config)

    return {
        **prior_dict,
        "labels": labels,
        "weights": weights,
        "class_scores": class_scores.detach(),
        "role_scores": role_scores.detach(),
        "foreground_score": foreground_score.detach(),
        "held_score": held_score.detach(),
        "static_score": static_score.detach(),
        "approach_score": approach_score.detach(),
    }


def generate_pseudo_labels(
    current_pc: Tensor,
    future_pc: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor | None = None,
    *,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    prior_dict = compute_motion_priors(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        nn_chunk_size=config.nn_chunk_size,
    )
    return generate_pseudo_labels_from_priors(prior_dict, config=config)


def _promote_minimum_foreground(
    labels: Tensor, weights: Tensor, class_scores: Tensor, config: PseudoLabelConfig
) -> tuple[Tensor, Tensor]:
    if config.min_foreground_fraction <= 0:
        return labels, weights
    if labels.ndim != 2:
        return labels, weights

    promoted_labels = labels.clone()
    promoted_weights = weights.clone()
    foreground_scores = class_scores[..., ROLE_FOREGROUND]
    min_count = max(1, math.ceil(labels.shape[1] * config.min_foreground_fraction))

    for bidx in range(labels.shape[0]):
        eligible = foreground_scores[bidx] >= config.forced_foreground_min_score
        if not bool(eligible.any()):
            continue
        count = min(min_count, int(eligible.sum().item()))
        masked_scores = foreground_scores[bidx].masked_fill(~eligible, -torch.inf)
        top_indices = torch.topk(masked_scores, k=count).indices
        promoted_labels[bidx, top_indices] = ROLE_FOREGROUND
        promoted_weights[bidx, top_indices] = torch.maximum(
            promoted_weights[bidx, top_indices],
            foreground_scores[bidx, top_indices].clamp(0.05, 1.0),
        )

    return promoted_labels, promoted_weights


def refine_pseudo_labels_with_teacher(
    pseudo: dict[str, Tensor],
    teacher_logits: Tensor,
    *,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    refined = dict(pseudo)
    probs = teacher_logits.softmax(dim=-1).detach()
    teacher_conf, teacher_labels = probs.max(dim=-1)
    class_scores = pseudo["class_scores"]
    geometry_gate = class_scores.gather(-1, teacher_labels.unsqueeze(-1)).squeeze(-1)
    foreground_accept = (
        (teacher_labels == ROLE_FOREGROUND)
        & (teacher_conf >= config.teacher_confidence)
        & (geometry_gate >= config.teacher_geometry_gate)
    )
    background_accept = (
        (teacher_labels == ROLE_BACKGROUND)
        & (teacher_conf >= config.teacher_confidence)
        & (class_scores[..., ROLE_BACKGROUND] >= config.teacher_background_min_score)
        & (pseudo["foreground_score"] <= config.teacher_background_foreground_max)
        & (pseudo["labels"] == ROLE_BACKGROUND)
    )
    accept = foreground_accept | background_accept

    labels = pseudo["labels"].clone()
    weights = pseudo["weights"].clone()
    labels = torch.where(accept, teacher_labels, labels)
    weights = torch.where(accept, torch.maximum(weights, teacher_conf), weights)
    refined["labels"] = labels
    refined["weights"] = weights
    refined["teacher_accept_mask"] = accept
    return refined


class SongPointSegNet(nn.Module):
    """LitePT-based role segmentation for Song manipulation point clouds."""

    def __init__(
        self,
        *,
        backbone_type: str = "litept",
        hidden_dim: int = 128,
        grid_size: float = 0.01,
        in_channels: int = 6 + MOTION_PRIOR_DIM,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.grid_size = float(grid_size)
        self.in_channels = int(in_channels)

        if backbone_type == "litept":
            try:
                from lerobot.policies.smolvla.litept.model import LitePT
            except Exception as exc:  # pragma: no cover - exercised only when optional deps are missing
                raise ImportError(
                    "SongPointSegNet(backbone_type='litept') requires LitePT optional dependencies "
                    "(flash_attn, spconv, torch_scatter, pointrope). Use backbone_type='mlp' for CPU smoke tests."
                ) from exc
            self.backbone = LitePT(
                in_channels=self.in_channels,
                enc_mode=False,
                dec_depths=(1, 1, 1, 1),
                dec_conv=(True, True, False, False),
                dec_attn=(False, False, False, False),
            )
            self.head = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(ROLE_NAMES)),
            )
        elif backbone_type == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(self.in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.head = nn.Linear(hidden_dim, len(ROLE_NAMES))
        else:
            raise ValueError(f"Unknown backbone_type={backbone_type!r}. Expected 'litept' or 'mlp'.")

    def _make_features(self, current_pc: Tensor, priors: Tensor) -> Tensor:
        xyz = current_pc[..., :3].to(dtype=torch.float32)
        rgb = current_pc[..., 3:6].to(dtype=torch.float32) / 255.0
        return torch.cat([xyz, rgb, priors.to(dtype=torch.float32)], dim=-1)

    def _forward_litept(self, current_pc: Tensor, features: Tensor) -> Tensor:
        bsize, n_points = current_pc.shape[:2]
        coord = current_pc[..., :3].reshape(-1, 3).contiguous().to(dtype=torch.float32)
        feat = features.reshape(-1, features.shape[-1]).contiguous().to(dtype=torch.float32)
        counts = torch.full((bsize,), n_points, dtype=torch.long, device=current_pc.device)
        offset = torch.cumsum(counts, dim=0)
        point = self.backbone(
            {
                "coord": coord,
                "feat": feat,
                "offset": offset,
                "grid_size": self.grid_size,
            }
        )
        point_feat = point.feat.reshape(bsize, n_points, -1)
        return self.head(point_feat)

    def forward(
        self,
        current_pc: Tensor,
        future_pc: Tensor | None = None,
        future_poses: Tensor | None = None,
        future_is_pad: Tensor | None = None,
        *,
        priors: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if priors is None:
            if future_pc is None or future_poses is None:
                raise ValueError("future_pc and future_poses are required when priors are not provided.")
            prior_dict = compute_motion_priors(current_pc, future_pc, future_poses, future_is_pad)
            priors = prior_dict["priors"]
        else:
            prior_dict = {"priors": priors}

        features = self._make_features(current_pc, priors)
        if self.backbone_type == "litept":
            role_logits = self._forward_litept(current_pc, features)
        else:
            role_logits = self.head(self.backbone(features))
        role_probs = role_logits.softmax(dim=-1)
        return {
            **prior_dict,
            "role_logits": role_logits,
            "role_probs": role_probs,
            "operation_prob": role_probs[..., ROLE_FOREGROUND],
        }


def _weighted_cross_entropy(logits: Tensor, labels: Tensor, weights: Tensor, config: SongPointSegLossConfig) -> Tensor:
    valid = labels != config.ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    class_weights = torch.tensor(config.class_weights, dtype=logits.dtype, device=logits.device)
    losses = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=config.ignore_index,
        reduction="none",
    ).reshape_as(labels)
    weighted = losses * weights.to(dtype=losses.dtype)
    class_losses = []
    for role_index in range(len(ROLE_NAMES)):
        class_mask = valid & (labels == role_index)
        if bool(class_mask.any()):
            class_losses.append(weighted[class_mask].sum() / weights[class_mask].sum().clamp_min(1e-6))
    if not class_losses:
        return logits.sum() * 0.0
    return torch.stack(class_losses).mean()


def _foreground_bce(logits: Tensor, labels: Tensor, weights: Tensor, ignore_index: int) -> Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    operation_logits = logits[..., ROLE_FOREGROUND] - logits[..., ROLE_BACKGROUND]
    target = ((labels == ROLE_FOREGROUND) & valid).to(dtype=logits.dtype)
    losses = functional.binary_cross_entropy_with_logits(operation_logits, target, reduction="none")
    pos = valid & (labels == ROLE_FOREGROUND)
    neg = valid & (labels == ROLE_BACKGROUND)
    parts = []
    if bool(pos.any()):
        parts.append((losses[pos] * weights[pos]).sum() / weights[pos].sum().clamp_min(1e-6))
    if bool(neg.any()):
        parts.append((losses[neg] * weights[neg]).sum() / weights[neg].sum().clamp_min(1e-6))
    if not parts:
        return logits.sum() * 0.0
    return torch.stack(parts).mean()


def _voxel_smoothness_loss(logits: Tensor, xyz: Tensor, voxel_size: float) -> Tensor:
    probs = logits.softmax(dim=-1)
    total = logits.sum() * 0.0
    used = 0
    for bidx in range(logits.shape[0]):
        voxel = torch.floor(xyz[bidx] / voxel_size).to(dtype=torch.long)
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
        groups = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
        if groups <= 1:
            continue
        sums = torch.zeros(groups, probs.shape[-1], device=probs.device, dtype=probs.dtype)
        counts = torch.zeros(groups, 1, device=probs.device, dtype=probs.dtype)
        sums.scatter_add_(0, inverse[:, None].expand(-1, probs.shape[-1]), probs[bidx])
        counts.scatter_add_(0, inverse[:, None], torch.ones_like(counts[inverse]))
        means = sums / counts.clamp_min(1.0)
        total = total + (probs[bidx] - means[inverse]).square().mean()
        used += 1
    if used == 0:
        return total
    return total / used


def _motion_consistency_loss(logits: Tensor, pseudo: dict[str, Tensor]) -> Tensor:
    probs = logits.softmax(dim=-1)
    class_scores = pseudo["class_scores"].to(device=logits.device, dtype=logits.dtype)
    foreground_scores = class_scores[..., ROLE_FOREGROUND]
    if not bool((foreground_scores > 0).any()):
        return logits.sum() * 0.0
    losses = (probs[..., ROLE_FOREGROUND] - foreground_scores).square()
    return (losses * foreground_scores).sum() / foreground_scores.sum().clamp_min(1e-6)


class SongPointSegLoss(nn.Module):
    def __init__(self, config: SongPointSegLossConfig | None = None):
        super().__init__()
        self.config = config or SongPointSegLossConfig()

    def forward(self, outputs: dict[str, Tensor], pseudo: dict[str, Tensor], current_pc: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = outputs["role_logits"]
        labels = pseudo["labels"].to(device=logits.device)
        weights = pseudo["weights"].to(device=logits.device, dtype=logits.dtype)

        ce = _weighted_cross_entropy(logits, labels, weights, self.config)
        fg_bce = _foreground_bce(logits, labels, weights, self.config.ignore_index)
        smoothness = _voxel_smoothness_loss(logits, current_pc[..., :3], self.config.smooth_voxel_size)
        motion = _motion_consistency_loss(logits, pseudo)
        loss = (
            self.config.ce_weight * ce
            + self.config.foreground_bce_weight * fg_bce
            + self.config.smoothness_weight * smoothness
            + self.config.motion_consistency_weight * motion
        )
        metrics = {
            "loss": loss.detach(),
            "loss_ce": ce.detach(),
            "loss_foreground_bce": fg_bce.detach(),
            "loss_smoothness": smoothness.detach(),
            "loss_motion": motion.detach(),
            "pseudo_valid_ratio": (labels != self.config.ignore_index).to(dtype=torch.float32).mean().detach(),
            "pseudo_valid_foreground_ratio": (
                (labels == ROLE_FOREGROUND)
                .to(dtype=torch.float32)
                .sum()
                / (labels != self.config.ignore_index).to(dtype=torch.float32).sum().clamp_min(1.0)
            ).detach(),
            "pseudo_background_ratio": (labels == ROLE_BACKGROUND).to(dtype=torch.float32).mean().detach(),
            "pseudo_foreground_ratio": (labels == ROLE_FOREGROUND)
            .to(dtype=torch.float32)
            .mean()
            .detach(),
            "pred_foreground_ratio": (outputs["role_probs"].argmax(dim=-1) == ROLE_FOREGROUND)
            .to(dtype=torch.float32)
            .mean()
            .detach(),
            "pred_operation_prob": outputs["operation_prob"].mean().detach(),
        }
        if "teacher_accept_mask" in pseudo:
            metrics["teacher_accept_ratio"] = pseudo["teacher_accept_mask"].to(dtype=torch.float32).mean().detach()
        return loss, metrics


class EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        teacher_state = self.model.state_dict()
        student_state = model.state_dict()
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)


def write_role_ply(path: str | Path, points_xyzrgb: np.ndarray, labels: np.ndarray, operation_prob: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    colors = ROLE_COLORS[np.clip(labels, 0, len(ROLE_COLORS) - 1)]
    if operation_prob is not None:
        operation_prob = np.asarray(operation_prob, dtype=np.float32)
        colors = np.clip(colors.astype(np.float32) * (0.35 + 0.65 * operation_prob[:, None]), 0, 255).astype(
            np.uint8
        )
    xyz = np.asarray(points_xyzrgb[:, :3], dtype=np.float32)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, colors, strict=False):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def save_pointseg_npz(
    path: str | Path,
    current_pc: Tensor,
    outputs: dict[str, Tensor],
    pseudo: dict[str, Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "point_cloud": current_pc.detach().cpu().numpy(),
        "role_logits": outputs["role_logits"].detach().cpu().numpy(),
        "operation_prob": outputs["operation_prob"].detach().cpu().numpy(),
    }
    if pseudo is not None:
        data["pseudo_labels"] = pseudo["labels"].detach().cpu().numpy()
        data["pseudo_weights"] = pseudo["weights"].detach().cpu().numpy()
        if "role_scores" in pseudo:
            data["pseudo_role_scores_gripper_condition_target"] = pseudo["role_scores"].detach().cpu().numpy()
        if "foreground_score" in pseudo:
            data["pseudo_foreground_score"] = pseudo["foreground_score"].detach().cpu().numpy()
    np.savez_compressed(path, **data)


def save_pointseg_config(path: str | Path, args: Any, pseudo_cfg: PseudoLabelConfig, loss_cfg: SongPointSegLossConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple | list):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        return value

    payload = {
        "args": jsonable(vars(args) if hasattr(args, "__dict__") else args),
        "pseudo_label_config": asdict(pseudo_cfg),
        "loss_config": asdict(loss_cfg),
        "role_names": ROLE_NAMES,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def parse_future_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise ValueError("At least one future offset is required.")
    if any(offset <= 0 for offset in offsets):
        raise ValueError("Future offsets must be positive integers.")
    return offsets


def pretty_metrics(metrics: dict[str, Tensor | float], step: int) -> str:
    values = []
    for key, value in metrics.items():
        scalar = float(value.item()) if torch.is_tensor(value) else float(value)
        values.append(f"{key}={scalar:.4f}")
    return f"step={step} " + " ".join(values)
