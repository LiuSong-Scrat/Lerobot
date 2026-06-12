#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import math
import warnings
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data._utils.collate import default_collate

try:
    from pointops import knn_query as _pointops_knn_query
except Exception:  # pragma: no cover - optional CUDA extension.
    _pointops_knn_query = None

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
POINTSEG_CACHE_VERSION = 2
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
POINTSEG_CACHE_LABEL_FIELDS = (
    "point_indices",
    "labels",
    "weights",
    "class_scores",
    "foreground_score",
    "episode_index",
    "frame_index",
    "dataset_index",
)

_POINTOPS_KNN_FAILED = False


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
    n_points = array.shape[0]
    if n_points == 0:
        channels = array.shape[-1] if array.ndim >= 2 else 6
        return np.zeros((0, channels), dtype=np.float32)
    if count <= 0 or n_points <= count:
        return np.ascontiguousarray(array, dtype=np.float32)
    indices = rng.choice(n_points, count, replace=False)
    return np.ascontiguousarray(array[indices], dtype=np.float32)


def _sample_rows_with_indices(
    array: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_points = array.shape[0]
    if n_points == 0:
        channels = array.shape[-1] if array.ndim >= 2 else 6
        return np.zeros((0, channels), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if count <= 0 or n_points <= count:
        indices = np.arange(n_points, dtype=np.int64)
        return np.ascontiguousarray(array, dtype=np.float32), indices
    indices = rng.choice(n_points, count, replace=False).astype(np.int64, copy=False)
    return np.ascontiguousarray(array[indices], dtype=np.float32), indices


def _require_zarr():
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        raise ImportError(
            "Compressed point-cloud storage requires `zarr` and `numcodecs`. "
            "Install them in the active environment, e.g. `conda install -n reap -c conda-forge zarr numcodecs`."
        ) from exc
    return zarr, Blosc


class PackedPointCloudZarr:
    """Read packed xyz=float16, rgb=uint8 zarr point clouds as float32 XYZRGB."""

    def __init__(self, group: Any):
        self.group = group
        self.xyz = group["xyz"]
        self.rgb = group["rgb"]
        self.shape = (*self.xyz.shape[:-1], 6)
        self.dtype = np.dtype(np.float32)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, item):
        xyz = np.asarray(self.xyz[item], dtype=np.float32)
        rgb = np.asarray(self.rgb[item], dtype=np.float32)
        return np.concatenate([xyz, rgb], axis=-1)

    def __array__(self, dtype=None):
        array = self[:]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def episode_point_cloud_npy_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    return Path(point_cloud_dir) / f"episode_{int(episode_index):06d}.npy"


def episode_point_cloud_zarr_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    return Path(point_cloud_dir) / f"episode_{int(episode_index):06d}.zarr"


def find_episode_point_cloud_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    zarr_path = episode_point_cloud_zarr_path(point_cloud_dir, episode_index)
    if zarr_path.exists():
        return zarr_path
    npy_path = episode_point_cloud_npy_path(point_cloud_dir, episode_index)
    if npy_path.exists():
        return npy_path
    raise FileNotFoundError(
        f"Point cloud episode file is missing: expected {zarr_path} or {npy_path}"
    )


def open_episode_point_clouds(
    point_cloud_dir: str | Path,
    episode_index: int,
    *,
    mmap_mode: str = "r",
) -> Any:
    path = find_episode_point_cloud_path(point_cloud_dir, episode_index)
    if path.suffix == ".zarr":
        zarr, _ = _require_zarr()
        zarr_obj = zarr.open(str(path), mode="r")
        if hasattr(zarr_obj, "array_keys") and "xyz" in zarr_obj and "rgb" in zarr_obj:
            return PackedPointCloudZarr(zarr_obj)
        return zarr_obj
    return np.load(path, mmap_mode=mmap_mode)


def _pack_point_cloud_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    finite_rgb = rgb[np.isfinite(rgb)]
    if finite_rgb.size > 0 and float(finite_rgb.max(initial=0.0)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8, copy=False)


def save_point_clouds_zarr(
    path: str | Path,
    point_clouds: np.ndarray,
    *,
    chunks: tuple[int, int, int] | None = None,
    compressor_name: str = "zstd",
    compression_level: int = 3,
    packed: bool = True,
) -> Path:
    zarr, Blosc = _require_zarr()
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 6:
        raise ValueError(f"Expected point clouds shape (T,N,6), got {point_clouds.shape}")
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname=compressor_name, clevel=int(compression_level), shuffle=Blosc.BITSHUFFLE)
    if packed:
        if chunks is None:
            chunks = (1, min(int(point_clouds.shape[1]), 16384), 3)
        group = zarr.open_group(str(path), mode="w")
        group.attrs.update(
            {
                "format": "song_point_cloud_packed_v1",
                "shape": list(point_clouds.shape),
                "xyz_dtype": "float16",
                "rgb_dtype": "uint8",
                "rgb_scale": "0_255",
            }
        )
        xyz = group.create_dataset(
            "xyz",
            shape=point_clouds[..., :3].shape,
            chunks=chunks,
            dtype=np.float16,
            compressor=compressor,
        )
        rgb = group.create_dataset(
            "rgb",
            shape=point_clouds[..., 3:6].shape,
            chunks=chunks,
            dtype=np.uint8,
            compressor=compressor,
        )
        xyz[:] = point_clouds[..., :3].astype(np.float16)
        rgb[:] = _pack_point_cloud_rgb(point_clouds[..., 3:6])
        return path

    if chunks is None:
        chunks = (1, min(int(point_clouds.shape[1]), 16384), 6)
    array = zarr.open(
        str(path),
        mode="w",
        shape=point_clouds.shape,
        chunks=chunks,
        dtype=point_clouds.dtype,
        compressor=compressor,
    )
    array[:] = point_clouds
    return path


def save_episode_point_clouds_zarr(
    point_cloud_dir: str | Path,
    episode_index: int,
    point_clouds: np.ndarray,
    *,
    chunks: tuple[int, int, int] | None = None,
    compressor_name: str = "zstd",
    compression_level: int = 3,
    packed: bool = True,
) -> Path:
    return save_point_clouds_zarr(
        episode_point_cloud_zarr_path(point_cloud_dir, episode_index),
        point_clouds,
        chunks=chunks,
        compressor_name=compressor_name,
        compression_level=compression_level,
        packed=packed,
    )


def _pad_point_tensors(tensors: list[Tensor], pad_value: float | int = 0) -> tuple[Tensor, Tensor]:
    if not tensors:
        raise ValueError("Cannot collate an empty point tensor list.")
    max_points = max(int(tensor.shape[0]) for tensor in tensors)
    trailing_shape = tuple(tensors[0].shape[1:])
    if max_points <= 0:
        raise ValueError("Cannot collate empty point clouds.")
    padded = tensors[0].new_full((len(tensors), max_points, *trailing_shape), pad_value)
    is_pad = torch.ones(len(tensors), max_points, dtype=torch.bool)
    for idx, tensor in enumerate(tensors):
        n_points = int(tensor.shape[0])
        if n_points <= 0:
            continue
        padded[idx, :n_points] = tensor
        is_pad[idx, :n_points] = False
    return padded, is_pad


def _pad_single_axis_tensors(tensors: list[Tensor], axis: int, pad_value: float | int = 0) -> tuple[Tensor, Tensor]:
    if axis < 0:
        axis += tensors[0].ndim
    max_points = max(int(tensor.shape[axis]) for tensor in tensors)
    if max_points <= 0:
        raise ValueError("Cannot collate tensors with an empty point axis.")
    out_shape = [len(tensors), *tensors[0].shape]
    out_shape[axis + 1] = max_points
    padded = tensors[0].new_full(tuple(out_shape), pad_value)
    mask_shape = [len(tensors), *tensors[0].shape[:axis], max_points]
    is_pad = torch.ones(tuple(mask_shape), dtype=torch.bool)
    for batch_idx, tensor in enumerate(tensors):
        n_points = int(tensor.shape[axis])
        slices = [batch_idx, *([slice(None)] * tensor.ndim)]
        slices[axis + 1] = slice(0, n_points)
        padded[tuple(slices)] = tensor
        mask_slices = [batch_idx, *([slice(None)] * axis), slice(0, n_points)]
        is_pad[tuple(mask_slices)] = False
    return padded, is_pad


def song_pointseg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    out: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in batch))
    current_point_fields = {
        "observation.point_cloud": 0,
        "observation.point_cloud_world": 0,
        "pointseg.priors": 0,
        "pointseg.labels": ROLE_IGNORE,
        "pointseg.weights": 0,
        "pointseg.class_scores": 0,
        "pointseg.role_scores": 0,
        "pointseg.foreground_score": 0,
        "observation.point_cloud_indices": -1,
    }
    for key in sorted(keys):
        values = [item[key] for item in batch if key in item]
        if len(values) != len(batch):
            continue
        if key == "observation.point_cloud":
            first = values[0]
            if torch.is_tensor(first) and first.ndim == 2:
                out[key], out["observation.point_cloud_is_pad"] = _pad_point_tensors(values, 0)
                continue
            if torch.is_tensor(first) and first.ndim == 3 and first.shape[0] == 1:
                out[key], out["observation.point_cloud_is_pad"] = _pad_single_axis_tensors(values, 1, 0)
                continue
        if key == "observation.point_cloud_world":
            first = values[0]
            if torch.is_tensor(first) and first.ndim == 2:
                out[key], out["observation.point_cloud_world_is_pad"] = _pad_point_tensors(values, 0)
                continue
        if key in current_point_fields and torch.is_tensor(values[0]) and values[0].ndim >= 1:
            out[key], _ = _pad_point_tensors(values, current_point_fields[key])
            continue
        if key == "observation.point_cloud_future" and torch.is_tensor(values[0]):
            out[key], out["observation.point_cloud_future_is_pad"] = _pad_single_axis_tensors(values, 1, 0)
            continue
        if key == "observation.point_cloud_future_is_pad" and torch.is_tensor(values[0]):
            out[key], _ = _pad_single_axis_tensors(values, 1, True)
            continue
        out[key] = default_collate(values)
    return out


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
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dir,
                episode_index,
                mmap_mode=self.mmap_mode,
            )
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
        current_sample, current_indices = _sample_rows_with_indices(current_full, self.current_points, rng)
        item["observation.point_cloud"] = torch.from_numpy(current_sample)
        item["observation.point_cloud_indices"] = torch.from_numpy(current_indices)

        future_samples = []
        future_point_masks = []
        future_is_pad = []
        for offset in self.temporal_offsets:
            raw_index = frame_index + offset
            clamped_index = min(max(raw_index, 0), episode_len - 1)
            future_is_pad.append(raw_index >= episode_len)
            future_samples.append(_sample_rows(np.asarray(point_clouds[clamped_index]), self.future_points, rng))

        max_future_points = max(sample.shape[0] for sample in future_samples)
        if max_future_points <= 0:
            raise ValueError(f"Future point clouds are empty for dataset index {idx}.")
        future_array = np.zeros((len(future_samples), max_future_points, future_samples[0].shape[-1]), dtype=np.float32)
        future_mask = np.ones((len(future_samples), max_future_points), dtype=bool)
        for future_idx, sample in enumerate(future_samples):
            n_points = sample.shape[0]
            future_array[future_idx, :n_points] = sample
            future_mask[future_idx, :n_points] = False
        item["observation.point_cloud_future"] = torch.from_numpy(future_array)
        item["observation.point_cloud_future_is_pad"] = torch.from_numpy(future_mask)
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
        if version not in {1, POINTSEG_CACHE_VERSION}:
            raise ValueError(
                f"Unsupported Song pointseg cache version {version}; expected 1 or {POINTSEG_CACHE_VERSION}."
            )

        fields = tuple(self.manifest.get("fields", ()))
        self.cache_mode = str(
            self.manifest.get(
                "cache_mode",
                "embedded_point_cloud" if "point_cloud" in fields else "indices",
            )
        )
        expected_fields = POINTSEG_CACHE_FIELDS if self.cache_mode == "embedded_point_cloud" else POINTSEG_CACHE_LABEL_FIELDS
        missing_fields = [field for field in expected_fields if field not in fields]
        if missing_fields:
            raise ValueError(f"Song pointseg cache is missing fields: {missing_fields}")
        self.fields = fields
        self.variable_num_points = bool(self.manifest.get("variable_num_points", False))

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
        offsets_path = shard_dir / "sample_offsets.npy"
        if offsets_path.exists():
            arrays["sample_offsets"] = np.load(offsets_path, mmap_mode=self.mmap_mode)
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
        if "sample_offsets" in arrays:
            offsets = arrays["sample_offsets"]
            point_slice = slice(int(offsets[local_index]), int(offsets[local_index + 1]))
        else:
            point_slice = local_index
        item = {
            "pointseg.labels": self._long_tensor(arrays["labels"][point_slice]),
            "pointseg.weights": self._float_tensor(arrays["weights"][point_slice]),
            "pointseg.class_scores": self._float_tensor(arrays["class_scores"][point_slice]),
            "pointseg.foreground_score": self._float_tensor(arrays["foreground_score"][point_slice]),
            "episode_index": torch.tensor(int(arrays["episode_index"][local_index]), dtype=torch.long),
            "frame_index": torch.tensor(int(arrays["frame_index"][local_index]), dtype=torch.long),
            "dataset_index": torch.tensor(int(arrays["dataset_index"][local_index]), dtype=torch.long),
        }
        if "priors" in arrays:
            item["pointseg.priors"] = self._float_tensor(arrays["priors"][point_slice])
        if "role_scores" in arrays:
            item["pointseg.role_scores"] = self._float_tensor(arrays["role_scores"][point_slice])
        if self.cache_mode == "embedded_point_cloud":
            item["observation.point_cloud"] = self._float_tensor(arrays["point_cloud"][point_slice])
        else:
            item["observation.point_cloud_indices"] = self._long_tensor(arrays["point_indices"][point_slice])
        return item


def _use_pointops_knn(device: torch.device) -> bool:
    return (
        _pointops_knn_query is not None
        and not _POINTOPS_KNN_FAILED
        and device.type == "cuda"
    )


def _nearest_distances_pointops(query: Tensor, target: Tensor) -> Tensor:
    """Return exact nearest L2 distance using the pointops CUDA KNN kernel."""
    group_count, query_count = query.shape[:2]
    target_count = target.shape[1]
    query_flat = query.reshape(group_count * query_count, 3).contiguous()
    target_flat = target.reshape(group_count * target_count, 3).contiguous()
    query_offset = (
        torch.arange(1, group_count + 1, device=query.device, dtype=torch.int32) * query_count
    ).contiguous()
    target_offset = (
        torch.arange(1, group_count + 1, device=query.device, dtype=torch.int32) * target_count
    ).contiguous()
    _, dist = _pointops_knn_query(1, target_flat, target_offset, query_flat, query_offset)
    return dist.reshape(group_count, query_count)


def _nearest_distances_from_grouped_queries(
    query: Tensor, target: Tensor, target_is_pad: Tensor | None = None
) -> Tensor:
    """Return nearest L2 distance for each grouped query point.

    query: (G, N, 3), target: (G, M, 3)
    """
    global _POINTOPS_KNN_FAILED
    has_target_mask = target_is_pad is not None and bool(target_is_pad.any().item())
    if _use_pointops_knn(query.device) and not has_target_mask:
        try:
            return _nearest_distances_pointops(query, target)
        except Exception as exc:
            _POINTOPS_KNN_FAILED = True
            warnings.warn(
                f"pointops KNN failed; falling back to torch GEMM nearest-neighbor. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    target_t = target.transpose(1, 2).contiguous()
    dist_sq = torch.bmm(query, target_t)
    dist_sq.mul_(-2.0)
    dist_sq.add_(query.square().sum(dim=-1, keepdim=True))
    dist_sq.add_(target.square().sum(dim=-1).unsqueeze(-2))
    if target_is_pad is not None:
        dist_sq = dist_sq.masked_fill(target_is_pad[:, None, :].to(device=dist_sq.device), float("inf"))
    min_sq = dist_sq.clamp_min_(0.0).amin(dim=-1)
    return min_sq.sqrt_()


def _nearest_distances(query: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    chunks = []
    target = target.contiguous()
    target_t = target.transpose(0, 1).contiguous()
    target_norm = target.square().sum(dim=-1).unsqueeze(0)
    for start in range(0, query.shape[0], chunk_size):
        query_chunk = query[start : start + chunk_size]
        dist_sq = query_chunk @ target_t
        dist_sq.mul_(-2.0)
        dist_sq.add_(query_chunk.square().sum(dim=-1, keepdim=True))
        dist_sq.add_(target_norm)
        chunks.append(dist_sq.clamp_min_(0.0).amin(dim=-1).sqrt_())
    return torch.cat(chunks, dim=0)


def _future_group_size(chunk_len: int, target_points: int, valid_count: int, device: torch.device) -> int:
    if valid_count <= 1:
        return 1
    if _use_pointops_knn(device):
        return valid_count
    elements_per_future = max(1, 2 * chunk_len * target_points)
    max_elements = 128_000_000
    if device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            max_bytes = min(int(total_bytes * 0.25), int(free_bytes * 0.60), 1_750_000_000)
            max_elements = max(32_000_000, max_bytes // 4)
        except RuntimeError:
            pass
    return max(1, min(valid_count, max_elements // elements_per_future))


def _scatter_amin_by_batch(dst: Tensor, batch_indices: Tensor, values: Tensor) -> None:
    try:
        index = batch_indices[:, None].expand_as(values)
        dst.scatter_reduce_(0, index, values, reduce="amin", include_self=True)
    except (AttributeError, RuntimeError):
        batch_list = batch_indices.detach().cpu().tolist()
        for local_index, bidx in enumerate(batch_list):
            dst[bidx] = torch.minimum(dst[bidx], values[local_index])


def _motion_residuals_single(
    current_xyz: Tensor,
    future_xyz: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    transforms = pose9_to_matrix(future_poses)
    inverse_transforms = invert_transform(transforms)
    valid_indices = torch.nonzero(~future_is_pad[1:], as_tuple=False).flatten() + 1

    if valid_indices.numel() == 0:
        inf = torch.full((current_xyz.shape[0],), 1.0, dtype=current_xyz.dtype, device=current_xyz.device)
        return inf, inf

    held_min = torch.full(
        (current_xyz.shape[0],), float("inf"), dtype=current_xyz.dtype, device=current_xyz.device
    )
    static_min = torch.full_like(held_min, float("inf"))
    valid_targets = future_xyz.index_select(0, valid_indices).contiguous()
    valid_inverse = inverse_transforms.index_select(0, valid_indices)

    for start in range(0, current_xyz.shape[0], chunk_size):
        current_chunk = current_xyz[start : start + chunk_size].contiguous()
        chunk_len = current_chunk.shape[0]
        group_size = _future_group_size(
            chunk_len, valid_targets.shape[1], valid_targets.shape[0], current_xyz.device
        )
        held_chunks = []
        static_chunks = []
        for group_start in range(0, valid_targets.shape[0], group_size):
            group_end = min(group_start + group_size, valid_targets.shape[0])
            targets = valid_targets[group_start:group_end]
            inverse = valid_inverse[group_start:group_end]
            held_query = current_chunk.unsqueeze(0).expand(targets.shape[0], -1, -1)
            static_query = transform_points(current_chunk.unsqueeze(0), inverse)
            grouped_query = torch.cat([held_query, static_query], dim=1).contiguous()
            nearest = _nearest_distances_from_grouped_queries(grouped_query, targets)
            held_chunks.append(nearest[:, :chunk_len])
            static_chunks.append(nearest[:, chunk_len:])
        held_min[start : start + chunk_len] = torch.cat(held_chunks, dim=0).amin(dim=0)
        static_min[start : start + chunk_len] = torch.cat(static_chunks, dim=0).amin(dim=0)

    return held_min, static_min


def _motion_residuals_batched(
    current_xyz: Tensor,
    future_xyz: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor,
    future_point_is_pad: Tensor | None = None,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    transforms = pose9_to_matrix(future_poses)
    inverse_transforms = invert_transform(transforms)
    valid_future = ~future_is_pad[:, 1:]
    if future_point_is_pad is not None:
        valid_future = valid_future & (~future_point_is_pad[:, 1:].all(dim=-1))
    pair_batch, pair_future_offset = torch.nonzero(valid_future, as_tuple=True)
    pair_future = pair_future_offset + 1

    held_min = torch.full(
        current_xyz.shape[:2], float("inf"), dtype=current_xyz.dtype, device=current_xyz.device
    )
    static_min = torch.full_like(held_min, float("inf"))

    if pair_batch.numel() == 0:
        held_min.fill_(1.0)
        static_min.fill_(1.0)
        return held_min, static_min

    valid_counts = valid_future.sum(dim=1)
    for start in range(0, current_xyz.shape[1], chunk_size):
        current_chunk_len = min(chunk_size, current_xyz.shape[1] - start)
        group_size = _future_group_size(
            current_chunk_len, future_xyz.shape[2], int(pair_batch.numel()), current_xyz.device
        )
        for group_start in range(0, int(pair_batch.numel()), group_size):
            group_end = min(group_start + group_size, int(pair_batch.numel()))
            batch_group = pair_batch[group_start:group_end]
            future_group = pair_future[group_start:group_end]
            current_group = current_xyz[batch_group, start : start + current_chunk_len].contiguous()
            targets = future_xyz[batch_group, future_group].contiguous()
            target_is_pad = (
                future_point_is_pad[batch_group, future_group].contiguous()
                if future_point_is_pad is not None
                else None
            )
            inverse = inverse_transforms[batch_group, future_group]
            static_query = transform_points(current_group, inverse)
            grouped_query = torch.cat([current_group, static_query], dim=1).contiguous()
            nearest = _nearest_distances_from_grouped_queries(grouped_query, targets, target_is_pad)
            held_group = nearest[:, :current_chunk_len]
            static_group = nearest[:, current_chunk_len:]
            _scatter_amin_by_batch(
                held_min[:, start : start + current_chunk_len], batch_group, held_group
            )
            _scatter_amin_by_batch(
                static_min[:, start : start + current_chunk_len], batch_group, static_group
            )

    no_valid = valid_counts == 0
    if bool(no_valid.any().item()):
        held_min[no_valid] = 1.0
        static_min[no_valid] = 1.0
    return held_min, static_min


def compute_motion_priors(
    current_pc: Tensor,
    future_pc: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor | None = None,
    *,
    current_is_pad: Tensor | None = None,
    future_point_is_pad: Tensor | None = None,
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
    if current_is_pad is None:
        current_is_pad = torch.zeros(bsize, n_points, dtype=torch.bool, device=current_pc.device)
    else:
        current_is_pad = current_is_pad.to(device=current_pc.device, dtype=torch.bool)
        if current_is_pad.shape != current_pc.shape[:2]:
            raise ValueError(f"Expected current_is_pad shape {current_pc.shape[:2]}, got {current_is_pad.shape}.")
    if future_point_is_pad is not None:
        future_point_is_pad = future_point_is_pad.to(device=current_pc.device, dtype=torch.bool)
        if future_point_is_pad.shape != future_pc.shape[:3]:
            raise ValueError(
                f"Expected future_point_is_pad shape {future_pc.shape[:3]}, got {future_point_is_pad.shape}."
            )
    current_xyz = current_pc[..., :3].to(dtype=torch.float32)
    future_xyz = future_pc[..., :3].to(device=current_pc.device, dtype=torch.float32)
    future_poses = future_poses.to(device=current_pc.device, dtype=torch.float32)

    held_residual, static_residual = _motion_residuals_batched(
        current_xyz,
        future_xyz,
        future_poses,
        future_is_pad,
        future_point_is_pad,
        chunk_size=nn_chunk_size,
    )
    held_residual = torch.where(current_is_pad, torch.ones_like(held_residual), held_residual)
    static_residual = torch.where(current_is_pad, torch.ones_like(static_residual), static_residual)

    trajectory_xyz = future_poses[..., :3]
    valid_traj = ~future_is_pad
    point_to_traj = torch.linalg.norm(current_xyz[:, :, None, :] - trajectory_xyz[:, None, :, :], dim=-1)
    point_to_traj = point_to_traj.masked_fill(~valid_traj[:, None, :], float("inf"))
    point_to_traj = point_to_traj.masked_fill(current_is_pad[:, :, None], float("inf"))
    min_traj_dist = point_to_traj.amin(dim=-1)
    min_traj_dist = torch.where(torch.isfinite(min_traj_dist), min_traj_dist, torch.zeros_like(min_traj_dist))

    ee_dist = torch.linalg.norm(current_xyz, dim=-1)
    ee_dist = torch.where(current_is_pad, torch.zeros_like(ee_dist), ee_dist)
    start_dist = point_to_traj[..., 0]
    future_min_dist = point_to_traj[..., 1:].amin(dim=-1) if point_to_traj.shape[-1] > 1 else start_dist
    future_min_dist = torch.where(torch.isfinite(future_min_dist), future_min_dist, start_dist)
    approach_delta = start_dist - future_min_dist
    residual_gap = static_residual - held_residual
    min_traj_dist = torch.where(current_is_pad, torch.zeros_like(min_traj_dist), min_traj_dist)
    approach_delta = torch.where(current_is_pad, torch.zeros_like(approach_delta), approach_delta)
    residual_gap = torch.where(current_is_pad, torch.zeros_like(residual_gap), residual_gap)

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
        "point_is_pad": current_is_pad,
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
    point_is_pad = prior_dict.get("point_is_pad")
    if point_is_pad is not None:
        point_is_pad = point_is_pad.to(device=labels.device, dtype=torch.bool)
        labels = torch.where(point_is_pad, torch.full_like(labels, config.ignore_index), labels)
        weights = torch.where(point_is_pad, torch.zeros_like(weights), weights)
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
    current_is_pad: Tensor | None = None,
    future_point_is_pad: Tensor | None = None,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    prior_dict = compute_motion_priors(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        current_is_pad=current_is_pad,
        future_point_is_pad=future_point_is_pad,
        nn_chunk_size=config.nn_chunk_size,
    )
    return generate_pseudo_labels_from_priors(prior_dict, config=config)


def force_small_current_clouds_foreground(
    pseudo: dict[str, Tensor],
    current_pc: Tensor,
    configured_current_points: int,
    point_is_pad: Tensor | None = None,
) -> dict[str, Tensor]:
    if configured_current_points <= 0:
        return pseudo
    if current_pc.ndim != 3:
        raise ValueError(f"Expected current_pc shape (B,N,C), got {tuple(current_pc.shape)}.")

    if point_is_pad is None:
        valid = torch.ones(current_pc.shape[:2], dtype=torch.bool, device=current_pc.device)
    else:
        point_is_pad = point_is_pad.to(device=current_pc.device, dtype=torch.bool)
        if point_is_pad.ndim == 3 and point_is_pad.shape[1] == 1:
            point_is_pad = point_is_pad.squeeze(1)
        if point_is_pad.shape != current_pc.shape[:2]:
            raise ValueError(f"Expected point_is_pad shape {current_pc.shape[:2]}, got {point_is_pad.shape}.")
        valid = ~point_is_pad

    valid_counts = valid.sum(dim=1)
    small_cloud = valid_counts < int(configured_current_points)
    if not bool(small_cloud.any().item()):
        return pseudo

    out = dict(pseudo)
    for key in ("labels", "weights", "class_scores", "role_scores", "foreground_score"):
        out[key] = out[key].clone()

    force_mask = small_cloud[:, None] & valid
    pad_mask = ~valid
    out["labels"] = torch.where(force_mask, torch.full_like(out["labels"], ROLE_FOREGROUND), out["labels"])
    out["labels"] = torch.where(pad_mask, torch.full_like(out["labels"], ROLE_IGNORE), out["labels"])
    out["weights"] = torch.where(force_mask, torch.ones_like(out["weights"]), out["weights"])
    out["weights"] = torch.where(pad_mask, torch.zeros_like(out["weights"]), out["weights"])

    out["class_scores"][..., ROLE_BACKGROUND] = torch.where(
        force_mask,
        torch.zeros_like(out["class_scores"][..., ROLE_BACKGROUND]),
        out["class_scores"][..., ROLE_BACKGROUND],
    )
    out["class_scores"][..., ROLE_FOREGROUND] = torch.where(
        force_mask,
        torch.ones_like(out["class_scores"][..., ROLE_FOREGROUND]),
        out["class_scores"][..., ROLE_FOREGROUND],
    )
    out["class_scores"] = torch.where(pad_mask[..., None], torch.zeros_like(out["class_scores"]), out["class_scores"])

    target_role_index = out["role_scores"].shape[-1] - 1
    out["role_scores"] = torch.where(force_mask[..., None], torch.zeros_like(out["role_scores"]), out["role_scores"])
    out["role_scores"][..., target_role_index] = torch.where(
        force_mask,
        torch.ones_like(out["role_scores"][..., target_role_index]),
        out["role_scores"][..., target_role_index],
    )
    out["role_scores"] = torch.where(pad_mask[..., None], torch.zeros_like(out["role_scores"]), out["role_scores"])
    out["foreground_score"] = torch.where(force_mask, torch.ones_like(out["foreground_score"]), out["foreground_score"])
    out["foreground_score"] = torch.where(pad_mask, torch.zeros_like(out["foreground_score"]), out["foreground_score"])
    out["small_cloud_forced_foreground"] = small_cloud
    if point_is_pad is not None:
        out["point_is_pad"] = point_is_pad
    return out


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
        eligible = (foreground_scores[bidx] >= config.forced_foreground_min_score) & (
            labels[bidx] != config.ignore_index
        )
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
        in_channels: int = 6,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.grid_size = float(grid_size)
        self.in_channels = int(in_channels)
        if self.in_channels != 6:
            raise ValueError("SongPointSegNet now uses only raw XYZRGB point features, so in_channels must be 6.")

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

    def _make_features(self, current_pc: Tensor) -> Tensor:
        xyz = current_pc[..., :3].to(dtype=torch.float32)
        rgb = current_pc[..., 3:6].to(dtype=torch.float32) / 255.0
        return torch.cat([xyz, rgb], dim=-1)

    def _forward_litept(self, current_pc: Tensor, features: Tensor, point_is_pad: Tensor | None = None) -> Tensor:
        bsize, n_points = current_pc.shape[:2]
        if point_is_pad is None:
            point_is_pad = torch.zeros(bsize, n_points, dtype=torch.bool, device=current_pc.device)
        valid = ~point_is_pad
        counts = valid.sum(dim=1).to(dtype=torch.long)
        valid_batch = counts > 0
        if not bool(valid_batch.any().item()):
            return current_pc.new_zeros(bsize, n_points, len(ROLE_NAMES))

        valid_local = torch.nonzero(valid[valid_batch], as_tuple=False)
        flat_valid = valid[valid_batch].reshape(-1)
        coord = current_pc[valid_batch, :, :3].reshape(-1, 3)[flat_valid].contiguous().to(dtype=torch.float32)
        feat = features[valid_batch].reshape(-1, features.shape[-1])[flat_valid].contiguous().to(dtype=torch.float32)
        counts = counts[valid_batch]
        offset = torch.cumsum(counts, dim=0)
        point = self.backbone(
            {
                "coord": coord,
                "feat": feat,
                "offset": offset,
                "grid_size": self.grid_size,
            }
        )
        valid_logits = self.head(point.feat)
        logits = valid_logits.new_zeros(bsize, n_points, valid_logits.shape[-1])
        valid_batch_indices = torch.nonzero(valid_batch, as_tuple=False).flatten()
        logits[valid_batch_indices[valid_local[:, 0]], valid_local[:, 1]] = valid_logits
        return logits

    def forward(
        self,
        current_pc: Tensor,
        future_pc: Tensor | None = None,
        future_poses: Tensor | None = None,
        future_is_pad: Tensor | None = None,
        *,
        priors: Tensor | None = None,
        point_is_pad: Tensor | None = None,
        future_point_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if priors is None:
            if future_pc is not None and future_poses is not None:
                prior_dict = compute_motion_priors(
                    current_pc,
                    future_pc,
                    future_poses,
                    future_is_pad,
                    current_is_pad=point_is_pad,
                    future_point_is_pad=future_point_is_pad,
                )
            else:
                prior_dict = {}
                if point_is_pad is not None:
                    prior_dict["point_is_pad"] = point_is_pad
        else:
            prior_dict = {"priors": priors}
            if point_is_pad is not None:
                prior_dict["point_is_pad"] = point_is_pad

        features = self._make_features(current_pc)
        if self.backbone_type == "litept":
            role_logits = self._forward_litept(current_pc, features, point_is_pad)
        else:
            role_logits = self.head(self.backbone(features))
            if point_is_pad is not None:
                role_logits = role_logits.masked_fill(point_is_pad[..., None].to(device=role_logits.device), 0.0)
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


def _voxel_smoothness_loss(
    logits: Tensor, xyz: Tensor, voxel_size: float, point_is_pad: Tensor | None = None
) -> Tensor:
    probs = logits.softmax(dim=-1)
    total = logits.sum() * 0.0
    used = 0
    for bidx in range(logits.shape[0]):
        valid = ~point_is_pad[bidx] if point_is_pad is not None else torch.ones(
            xyz.shape[1], dtype=torch.bool, device=xyz.device
        )
        if int(valid.sum().item()) <= 1:
            continue
        xyz_b = xyz[bidx, valid]
        probs_b = probs[bidx, valid]
        voxel = torch.floor(xyz_b / voxel_size).to(dtype=torch.long)
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
        groups = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
        if groups <= 1:
            continue
        sums = torch.zeros(groups, probs.shape[-1], device=probs.device, dtype=probs.dtype)
        counts = torch.zeros(groups, 1, device=probs.device, dtype=probs.dtype)
        sums.scatter_add_(0, inverse[:, None].expand(-1, probs.shape[-1]), probs_b)
        counts.scatter_add_(0, inverse[:, None], torch.ones_like(counts[inverse]))
        means = sums / counts.clamp_min(1.0)
        total = total + (probs_b - means[inverse]).square().mean()
        used += 1
    if used == 0:
        return total
    return total / used


def _motion_consistency_loss(logits: Tensor, pseudo: dict[str, Tensor]) -> Tensor:
    probs = logits.softmax(dim=-1)
    class_scores = pseudo["class_scores"].to(device=logits.device, dtype=logits.dtype)
    foreground_scores = class_scores[..., ROLE_FOREGROUND]
    point_is_pad = pseudo.get("point_is_pad")
    if point_is_pad is not None:
        foreground_scores = foreground_scores.masked_fill(point_is_pad.to(device=logits.device, dtype=torch.bool), 0.0)
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
        point_is_pad = pseudo.get("point_is_pad", outputs.get("point_is_pad"))
        if point_is_pad is not None:
            point_is_pad = point_is_pad.to(device=logits.device, dtype=torch.bool)
            labels = torch.where(point_is_pad, torch.full_like(labels, self.config.ignore_index), labels)
            weights = torch.where(point_is_pad, torch.zeros_like(weights), weights)
        valid = labels != self.config.ignore_index
        valid_f = valid.to(dtype=torch.float32)

        ce = _weighted_cross_entropy(logits, labels, weights, self.config)
        fg_bce = _foreground_bce(logits, labels, weights, self.config.ignore_index)
        smoothness = _voxel_smoothness_loss(
            logits, current_pc[..., :3], self.config.smooth_voxel_size, point_is_pad
        )
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
            "pseudo_valid_ratio": valid_f.mean().detach(),
            "pseudo_valid_foreground_ratio": (
                (labels == ROLE_FOREGROUND)
                .to(dtype=torch.float32)
                .sum()
                / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pseudo_background_ratio": (
                (labels == ROLE_BACKGROUND).to(dtype=torch.float32).sum() / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pseudo_foreground_ratio": (labels == ROLE_FOREGROUND)
            .to(dtype=torch.float32)
            .sum()
            .div(valid_f.sum().clamp_min(1.0))
            .detach(),
            "pred_foreground_ratio": (
                ((outputs["role_probs"].argmax(dim=-1) == ROLE_FOREGROUND) & valid)
                .to(dtype=torch.float32)
                .sum()
                / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pred_operation_prob": (
                (outputs["operation_prob"] * valid_f.to(dtype=outputs["operation_prob"].dtype)).sum()
                / valid_f.to(dtype=outputs["operation_prob"].dtype).sum().clamp_min(1.0)
            ).detach(),
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
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
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
