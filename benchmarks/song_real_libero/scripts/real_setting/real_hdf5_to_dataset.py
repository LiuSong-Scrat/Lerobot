#!/usr/bin/env python
"""Convert real-robot HDF5 episodes with XYZRGB clouds into a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.song_pointseg import (
    episode_point_cloud_npy_path,
    episode_point_cloud_zarr_path,
    open_episode_point_clouds,
    save_point_clouds_zarr,
)

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import REAL_DATA_ROOT
    from ..libero_setting.libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        fast_inverse_homogeneous,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        sample_or_repeat_points,
        traj6_to_pose9,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT
    from libero_setting.libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        fast_inverse_homogeneous,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        sample_or_repeat_points,
        traj6_to_pose9,
    )


POINT_CLOUD_DIR_NAME = "point_clouds"
WORLD_EE_POSE_DIR_NAME = "world_ee_poses"
POINT_CLOUD_CHANNELS = 6

DATASET_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"],
    },
}


def image_feature_key(camera: str) -> str:
    return f"observation.images.{str(camera).strip()}"


def resolve_image_key(h5_file: h5py.File, requested_key: str | None, camera: str) -> str | None:
    if requested_key:
        if str(requested_key).lower() in {"0", "false", "none", "off"}:
            return None
        if requested_key not in h5_file:
            raise KeyError(f"Requested image key is missing: {requested_key}")
        return requested_key
    image_group_key = "observations/images"
    if image_group_key not in h5_file:
        return None
    image_group = h5_file[image_group_key]
    if camera in image_group:
        return f"{image_group_key}/{camera}"
    names = list(image_group.keys())
    if len(names) == 1:
        return f"{image_group_key}/{names[0]}"
    return None


def infer_image_shape(source_files: list[Path], requested_key: str | None, camera: str) -> tuple[int, int, int] | None:
    for path in source_files:
        with h5py.File(path, "r") as h5_file:
            image_key = resolve_image_key(h5_file, requested_key, camera)
            if image_key is None:
                continue
            image_ds = h5_file[image_key]
            if image_ds.ndim != 4 or image_ds.shape[-1] != 3:
                raise ValueError(f"{image_key} must have shape (T,H,W,3), got {image_ds.shape}")
            return tuple(int(v) for v in image_ds.shape[1:])
    return None


def dataset_features_with_optional_image(image_shape: tuple[int, int, int] | None, camera: str) -> dict[str, Any]:
    features = dict(DATASET_FEATURES)
    if image_shape is not None:
        features[image_feature_key(camera)] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert real HDF5 episodes that already contain XYZRGB point clouds. "
            "No dataset download, simulator replay, depth rendering, or depth back-projection is performed."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--hdf5-dir",
        dest="input_dir",
        type=Path,
        default=REAL_DATA_ROOT / "hdf5_raw",
    )
    parser.add_argument("--input-file", type=Path, action="append", default=None)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=REAL_DATA_ROOT / "lerobot_dataset")
    parser.add_argument("--repo-id", default="song_real_pointcloud")
    parser.add_argument("--robot-type", default="human_hand")
    parser.add_argument("--task", default=None, help="Override the task text stored in each HDF5 file.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--point-cloud-key", default=None, help="Full HDF5 key. Defaults to observations/cloud_rgb/<camera>.")
    parser.add_argument("--image-key", default=None, help="Full HDF5 RGB key. Defaults to observations/images/<camera> when available. Use 'none' to disable.")
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--pose-key", default="observations/pose_eular")
    parser.add_argument("--gripper-key", default="observations/eff_angular")
    parser.add_argument("--timestamp-key", default="timestamp_ms")
    parser.add_argument("--pose-format", choices=("auto", "euler_zyx", "pose9"), default="auto")
    parser.add_argument(
        "--cloud-frame",
        choices=("camera", "auto"),
        default="auto",
        help=(
            "Coordinate frame shared by source clouds and end-effector poses. "
            "'camera' treats the fixed overhead camera as the dataset world/reference frame and never uses "
            "a real-robot camera extrinsic. 'auto' accepts only HDF5 data marked pose_frame='camera'."
        ),
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("source", "fps"),
        default="source",
        help="Use normalized source timestamps when valid, otherwise fixed frame_index/fps.",
    )
    parser.add_argument("--add-gripper-cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--input-has-gripper-cloud",
        action="store_true",
        help="Source clouds already contain gripper points; do not add them again.",
    )
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--gripper-len", type=float, default=0.06)
    parser.add_argument("--gripper-template", choices=("reap", "panda"), default="reap")
    parser.add_argument("--gripper-drop-strategy", choices=("tail", "random", "near_gripper"), default="random")
    parser.add_argument("--gripper-shuffle-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-widths-are-normalized", action="store_true")
    parser.add_argument("--gripper-max-width", type=float, default=0.08)
    parser.add_argument("--point-cloud-storage", choices=("zarr", "npy"), default="zarr")
    parser.add_argument("--zarr-compression-level", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--vis-count", type=int, default=0)
    parser.add_argument("--vis-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def natural_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def discover_hdf5_files(args: argparse.Namespace) -> list[Path]:
    if args.input_file:
        files = [path.expanduser().resolve() for path in args.input_file]
    else:
        root = args.input_dir.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Real HDF5 input directory does not exist: {root}")
        glob_pattern = f"**/{args.pattern}" if args.recursive else args.pattern
        files = sorted(root.glob(glob_pattern), key=natural_key)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input HDF5 file(s) do not exist: {missing}")
    if not files:
        raise FileNotFoundError(f"No HDF5 files matched {args.pattern!r} under {args.input_dir}")
    return files


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_attr(value.item())
    return value


def source_task(h5_file: h5py.File, override: str | None, source_path: Path) -> str:
    if override:
        return str(override)
    for key in ("task", "task_name", "language_instruction"):
        if key in h5_file.attrs:
            value = str(decode_attr(h5_file.attrs[key])).strip()
            if value:
                return value
        if key in h5_file:
            value = str(decode_attr(h5_file[key][()])).strip()
            if value:
                return value
    return source_path.stem


def resolve_cloud_key(h5_file: h5py.File, requested_key: str | None, camera: str) -> str:
    if requested_key:
        if requested_key not in h5_file:
            raise KeyError(f"Point-cloud key {requested_key!r} is missing.")
        return requested_key
    cloud_group_key = "observations/cloud_rgb"
    if cloud_group_key not in h5_file:
        raise KeyError(f"Missing HDF5 group {cloud_group_key!r}.")
    cloud_group = h5_file[cloud_group_key]
    if camera in cloud_group:
        return f"{cloud_group_key}/{camera}"
    names = list(cloud_group.keys())
    if len(names) == 1:
        return f"{cloud_group_key}/{names[0]}"
    raise KeyError(f"Camera {camera!r} is unavailable. Point-cloud cameras: {names}")


def resolve_cloud_frame(h5_file: h5py.File, requested: str) -> str:
    if requested != "auto":
        return requested
    value = str(decode_attr(h5_file.attrs.get("pose_frame", "camera"))).lower()
    if value == "camera":
        return value
    if value in {"world", "base", "current_eff"}:
        raise ValueError(
            "Only camera-frame real HDF5 data is accepted in camera-reference mode: "
            f"pose_frame={value!r}. Regenerate the HDF5 with pose_frame='camera', or explicitly provide "
            "--cloud-frame camera only when both cloud_rgb and pose_eular are already expressed in the "
            "same overhead-camera frame."
        )
    raise ValueError(f"Unsupported source pose_frame attribute: {value!r}")


def load_pose9(h5_file: h5py.File, key: str, pose_format: str, frame_count: int) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Missing end-effector pose key: {key}")
    poses = np.asarray(h5_file[key], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[0] < frame_count:
        raise ValueError(f"{key} must have shape (T,D) with T >= {frame_count}, got {poses.shape}")
    poses = poses[:frame_count]
    resolved_format = pose_format
    if resolved_format == "auto":
        resolved_format = "pose9" if poses.shape[1] >= 9 else "euler_zyx"
    if resolved_format == "pose9":
        if poses.shape[1] < 9:
            raise ValueError(f"pose9 requires at least 9 values per frame, got {poses.shape}")
        return poses[:, :9].astype(np.float32, copy=False)
    if poses.shape[1] < 6:
        raise ValueError(f"euler_zyx requires at least 6 values per frame, got {poses.shape}")
    return traj6_to_pose9(poses[:, :6])


def matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate(
        [matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]],
        axis=-1,
    ).astype(np.float32)


def load_gripper_widths(h5_file: h5py.File, key: str, frame_count: int) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Missing gripper key: {key}")
    widths = np.asarray(h5_file[key], dtype=np.float32).reshape(-1)
    if len(widths) < frame_count:
        raise ValueError(f"{key} has {len(widths)} frames, expected at least {frame_count}")
    widths = widths[:frame_count]
    if not np.isfinite(widths).all():
        raise ValueError(f"{key} contains non-finite gripper widths.")
    return widths


def make_timestamps(
    h5_file: h5py.File,
    key: str,
    frame_count: int,
    fps: int,
    mode: str,
) -> tuple[np.ndarray, str]:
    fallback = np.arange(frame_count, dtype=np.float32) / float(fps)
    if mode == "fps" or key not in h5_file:
        return fallback, "fps"
    raw = np.asarray(h5_file[key], dtype=np.float64).reshape(-1)[:frame_count]
    if len(raw) != frame_count or not np.isfinite(raw).all():
        return fallback, "fps_fallback"
    scale = 1e-3 if key.lower().endswith("_ms") or np.nanmedian(np.diff(raw)) > 1.0 else 1.0
    timestamps = (raw - raw[0]) * scale
    if frame_count > 1 and not np.all(np.diff(timestamps) > 0):
        return fallback, "fps_fallback"
    return timestamps.astype(np.float32), "source"


def reference_to_umi_actions(reference_ee_poses: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    reference_h = pose9_to_homo_np(reference_ee_poses)
    initial_eff_to_reference = reference_h[0]
    reference_to_eff = fast_inverse_homogeneous(reference_h)
    initial_eff_to_current_eff = reference_to_eff @ initial_eff_to_reference
    current_eff_to_initial_eff = fast_inverse_homogeneous(initial_eff_to_current_eff)
    umi_pose9 = matrix_to_pose9(current_eff_to_initial_eff)
    return np.concatenate([umi_pose9, grippers[:, None]], axis=-1).astype(np.float32)


def sanitize_cloud(cloud: np.ndarray) -> np.ndarray:
    cloud = np.asarray(cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < POINT_CLOUD_CHANNELS:
        raise ValueError(f"Expected source cloud shape (N,>=6), got {cloud.shape}")
    cloud = cloud[:, :POINT_CLOUD_CHANNELS]
    valid = np.isfinite(cloud).all(axis=1)
    cloud = cloud[valid]
    if len(cloud) == 0:
        raise ValueError("Point cloud has no finite XYZRGB points.")
    cloud[:, 3:6] = np.clip(cloud[:, 3:6], 0, 255)
    return cloud


def prepare_point_clouds(
    cloud_dataset: h5py.Dataset,
    reference_ee_poses: np.ndarray,
    grippers: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    frame_count = len(reference_ee_poses)
    total_points = int(cfg["num_points"])
    source_clouds = [sanitize_cloud(cloud_dataset[frame_idx]) for frame_idx in range(frame_count)]

    reference_clouds = np.stack(
        [sample_or_repeat_points(cloud, total_points, seed=int(cfg["seed"]) + idx) for idx, cloud in enumerate(source_clouds)]
    )
    if bool(cfg["add_gripper_cloud"]) and not bool(cfg["input_has_gripper_cloud"]):
        reference_clouds = add_reference_gripper_clouds_to_episode(
            reference_clouds,
            reference_ee_poses,
            grippers,
            total_points=total_points,
            gripper_points=int(cfg["gripper_points"]),
            gripper_len=float(cfg["gripper_len"]),
            gripper_template=str(cfg["gripper_template"]),
            seed=int(cfg["seed"]),
            drop_strategy=str(cfg["gripper_drop_strategy"]),
            shuffle_points=bool(cfg["gripper_shuffle_points"]),
            widths_are_normalized=bool(cfg["gripper_widths_are_normalized"]),
            gripper_max_width=float(cfg["gripper_max_width"]),
        )
    return reference_point_cloud_to_current_eff(reference_clouds, reference_ee_poses)


def convert_hdf5_episode(source_path: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(source_path, "r") as h5_file:
        cloud_key = resolve_cloud_key(h5_file, cfg.get("point_cloud_key"), str(cfg["camera"]))
        image_key = resolve_image_key(h5_file, cfg.get("image_key"), str(cfg["camera"]))
        if cfg.get("image_feature_key") is not None and image_key is None:
            raise KeyError(
                f"Dataset was created with RGB feature {cfg['image_feature_key']!r}, "
                f"but no image dataset was found in {source_path}."
            )
        cloud_dataset = h5_file[cloud_key]
        if cloud_dataset.ndim != 3 or cloud_dataset.shape[-1] < POINT_CLOUD_CHANNELS:
            raise ValueError(f"{cloud_key} must have shape (T,N,>=6), got {cloud_dataset.shape}")
        frame_count = int(cloud_dataset.shape[0])
        max_frames = cfg.get("max_frames")
        if max_frames is not None and int(max_frames) > 0:
            frame_count = min(frame_count, int(max_frames))
        if frame_count < 2:
            raise ValueError(f"Episode needs at least 2 frames, got {frame_count}: {source_path}")

        cloud_frame = resolve_cloud_frame(h5_file, str(cfg["cloud_frame"]))
        reference_ee_poses = load_pose9(
            h5_file,
            str(cfg["pose_key"]),
            str(cfg["pose_format"]),
            frame_count,
        )
        grippers = load_gripper_widths(h5_file, str(cfg["gripper_key"]), frame_count)
        timestamps, timestamp_source = make_timestamps(
            h5_file,
            str(cfg["timestamp_key"]),
            frame_count,
            int(cfg["fps"]),
            str(cfg["timestamp_mode"]),
        )
        point_clouds = prepare_point_clouds(
            cloud_dataset,
            reference_ee_poses,
            grippers,
            cfg,
        )
        actions = reference_to_umi_actions(reference_ee_poses, grippers)
        task = source_task(h5_file, cfg.get("task"), source_path)
        images = None
        if image_key is not None:
            image_dataset = h5_file[image_key]
            if image_dataset.ndim != 4 or image_dataset.shape[-1] != 3:
                raise ValueError(f"{image_key} must have shape (T,H,W,3), got {image_dataset.shape}")
            if int(image_dataset.shape[0]) < frame_count:
                raise ValueError(f"{image_key} has {image_dataset.shape[0]} frames but needs {frame_count}.")
            images = np.asarray(image_dataset[:frame_count], dtype=np.uint8)

    episode = {
        "task": task,
        "actions": actions,
        "point_clouds": point_clouds,
        # Legacy key/path retained for the existing WorldFlow dataset wrapper.
        # Values are expressed in the fixed overhead-camera reference frame.
        "world_ee_poses": reference_ee_poses,
        "timestamps": timestamps,
    }
    if images is not None:
        episode["images"] = images
    record = {
        "source_hdf5": str(source_path),
        "task": task,
        "frames": frame_count,
        "source_point_cloud_key": cloud_key,
        "source_image_key": image_key,
        "source_cloud_frame": cloud_frame,
        "reference_frame": "overhead_camera",
        "uses_real_camera_extrinsic": False,
        "timestamp_source": timestamp_source,
        "add_gripper_cloud": bool(cfg["add_gripper_cloud"]) and not bool(cfg["input_has_gripper_cloud"]),
    }
    return episode, record


def save_worker_artifact(artifact_dir: Path, episode: dict[str, Any], record: dict[str, Any], cfg: dict[str, Any]) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    if cfg["point_cloud_storage"] == "zarr":
        save_point_clouds_zarr(
            artifact_dir / "point_clouds.zarr",
            episode["point_clouds"],
            compression_level=int(cfg["zarr_compression_level"]),
        )
    else:
        np.save(artifact_dir / "point_clouds.npy", np.ascontiguousarray(episode["point_clouds"], dtype=np.float32))
    np.save(artifact_dir / "world_ee_poses.npy", np.ascontiguousarray(episode["world_ee_poses"], dtype=np.float32))
    np.save(artifact_dir / "actions.npy", np.ascontiguousarray(episode["actions"], dtype=np.float32))
    np.save(artifact_dir / "timestamps.npy", np.ascontiguousarray(episode["timestamps"], dtype=np.float32))
    if "images" in episode:
        np.save(artifact_dir / "images.npy", np.ascontiguousarray(episode["images"], dtype=np.uint8))
    (artifact_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def convert_worker(job: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(job["source_path"])
    artifact_dir = Path(job["artifact_dir"])
    episode, record = convert_hdf5_episode(source_path, dict(job["cfg"]))
    save_worker_artifact(artifact_dir, episode, record, dict(job["cfg"]))
    return {"job_index": int(job["job_index"]), "artifact_dir": str(artifact_dir)}


def point_cloud_path(root: Path, episode_index: int, storage: str) -> Path:
    point_cloud_dir = root / POINT_CLOUD_DIR_NAME
    if storage == "zarr":
        return episode_point_cloud_zarr_path(point_cloud_dir, episode_index)
    return episode_point_cloud_npy_path(point_cloud_dir, episode_index)


def world_pose_path(root: Path, episode_index: int) -> Path:
    return root / WORLD_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def move_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def save_episode_images_to_paths(
    dataset: LeRobotDataset,
    images: np.ndarray,
    image_key: str,
    episode_index: int,
) -> list[str]:
    images = np.asarray(images, dtype=np.uint8)
    paths: list[str] = []
    for frame_index, image in enumerate(images):
        image_path = dataset._get_image_file_path(  # noqa: SLF001 - use LeRobot standard image layout.
            episode_index=episode_index,
            image_key=image_key,
            frame_index=frame_index,
        )
        if frame_index == 0:
            image_path.parent.mkdir(parents=True, exist_ok=True)
        dataset._save_image(image, image_path, compress_level=6)  # noqa: SLF001
        paths.append(str(image_path))
    return paths


def make_episode_buffer(
    dataset: LeRobotDataset,
    task: str,
    actions: np.ndarray,
    timestamps: np.ndarray,
    images: np.ndarray | None = None,
    image_key: str | None = None,
) -> dict[str, Any]:
    episode_buffer = dataset.create_episode_buffer()
    frame_count = len(actions)
    episode_buffer["size"] = frame_count
    episode_buffer["task"] = [task] * frame_count
    episode_buffer["frame_index"] = np.arange(frame_count, dtype=np.int64)
    episode_buffer["timestamp"] = np.asarray(timestamps, dtype=np.float32)
    episode_buffer["action"] = np.asarray(actions, dtype=np.float32)
    episode_buffer["observation.state"] = np.asarray(actions, dtype=np.float32)
    if images is not None and image_key is not None:
        if len(images) != frame_count:
            raise ValueError(f"Image frame count {len(images)} does not match actions {frame_count}.")
        episode_buffer[image_key] = save_episode_images_to_paths(
            dataset,
            images,
            image_key,
            int(episode_buffer["episode_index"]),
        )
    return episode_buffer


def save_artifact_to_dataset(
    dataset: LeRobotDataset,
    artifact_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    record = json.loads((artifact_dir / "record.json").read_text(encoding="utf-8"))
    actions = np.load(artifact_dir / "actions.npy")
    timestamps = np.load(artifact_dir / "timestamps.npy")
    images_path = artifact_dir / "images.npy"
    images = np.load(images_path, mmap_mode="r") if images_path.exists() else None
    episode_index = int(dataset.meta.total_episodes)
    storage = str(cfg["point_cloud_storage"])
    move_artifact(
        artifact_dir / ("point_clouds.zarr" if storage == "zarr" else "point_clouds.npy"),
        point_cloud_path(dataset.root, episode_index, storage),
    )
    move_artifact(artifact_dir / "world_ee_poses.npy", world_pose_path(dataset.root, episode_index))
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            str(record["task"]),
            actions,
            timestamps,
            images=images,
            image_key=cfg.get("image_feature_key"),
        )
    )
    record["episode_index"] = episode_index
    shutil.rmtree(artifact_dir, ignore_errors=True)
    return record


def write_dataset_sidecar_meta(root: Path, storage: str) -> None:
    point_cloud_dir = root / POINT_CLOUD_DIR_NAME
    point_cloud_dir.mkdir(parents=True, exist_ok=True)
    suffix = "zarr" if storage == "zarr" else "npy"
    point_meta = {
        "key": "observation.point_cloud",
        "dtype": "float32",
        "shape": [None, POINT_CLOUD_CHANNELS],
        "variable_num_points": True,
        "layout": "episode_array",
        "storage_format": storage,
        "path_format": f"{POINT_CLOUD_DIR_NAME}/episode_{{episode_index:06d}}.{suffix}",
        "coordinate_frame": "current_eff",
        "source_reference_frame": "overhead_camera",
    }
    if storage == "zarr":
        point_meta["zarr_encoding"] = "packed_xyz_float16_rgb_uint8"
    (point_cloud_dir / "meta.json").write_text(json.dumps(point_meta, indent=2), encoding="utf-8")

    pose_dir = root / WORLD_EE_POSE_DIR_NAME
    pose_dir.mkdir(parents=True, exist_ok=True)
    pose_meta = {
        "key": "worldflow.ee_poses",
        "dtype": "float32",
        "shape": [9],
        "layout": "episode_npy",
        "path_format": f"{WORLD_EE_POSE_DIR_NAME}/episode_{{episode_index:06d}}.npy",
        "coordinate_frame": "overhead_camera",
        "legacy_directory_name": True,
        "uses_real_camera_extrinsic": False,
    }
    (pose_dir / "meta.json").write_text(json.dumps(pose_meta, indent=2), encoding="utf-8")


def write_ascii_ply(path: Path, cloud: np.ndarray) -> None:
    cloud = np.asarray(cloud, dtype=np.float32)
    colors = np.clip(cloud[:, 3:6], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as ply_file:
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write(f"element vertex {len(cloud)}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(cloud[:, :3], colors, strict=True):
            ply_file.write(
                f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def export_preview(dataset_root: Path, record: dict[str, Any], cfg: dict[str, Any]) -> None:
    vis_count = int(cfg["vis_count"])
    if vis_count <= 0:
        return
    episode_index = int(record["episode_index"])
    clouds = open_episode_point_clouds(dataset_root / POINT_CLOUD_DIR_NAME, episode_index, mmap_mode="r")
    candidate_indices = list(range(0, len(clouds), max(1, int(cfg["vis_stride"]))))
    if len(candidate_indices) > vis_count:
        selected = np.linspace(0, len(candidate_indices) - 1, vis_count).round().astype(int)
        frame_indices = [candidate_indices[int(index)] for index in selected]
    else:
        frame_indices = candidate_indices
    output_dir = dataset_root / "visualizations" / f"episode_{episode_index:06d}"
    for frame_index in frame_indices:
        write_ascii_ply(output_dir / f"frame_{frame_index:04d}_point_cloud_eff.ply", clouds[frame_index])
    (output_dir / "preview.json").write_text(
        json.dumps({**record, "frame_indices": frame_indices}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "camera": args.camera,
        "point_cloud_key": args.point_cloud_key,
        "pose_key": args.pose_key,
        "gripper_key": args.gripper_key,
        "timestamp_key": args.timestamp_key,
        "pose_format": args.pose_format,
        "cloud_frame": args.cloud_frame,
        "timestamp_mode": args.timestamp_mode,
        "task": args.task,
        "fps": int(args.fps),
        "max_frames": args.max_frames,
        "num_points": int(args.num_points),
        "add_gripper_cloud": bool(args.add_gripper_cloud),
        "input_has_gripper_cloud": bool(args.input_has_gripper_cloud),
        "gripper_points": int(args.gripper_points),
        "gripper_len": float(args.gripper_len),
        "gripper_template": args.gripper_template,
        "gripper_drop_strategy": args.gripper_drop_strategy,
        "gripper_shuffle_points": bool(args.gripper_shuffle_points),
        "gripper_widths_are_normalized": bool(args.gripper_widths_are_normalized),
        "gripper_max_width": float(args.gripper_max_width),
        "point_cloud_storage": args.point_cloud_storage,
        "zarr_compression_level": int(args.zarr_compression_level),
        "seed": int(args.seed),
        "vis_count": int(args.vis_count),
        "vis_stride": int(args.vis_stride),
    }


def main() -> None:
    args = parse_args()
    source_files = discover_hdf5_files(args)
    output_root = args.output_root.expanduser().resolve()
    tmp_dir = (
        args.tmp_dir.expanduser().resolve()
        if args.tmp_dir is not None
        else output_root.parent / f".{output_root.name}_real_hdf5_tmp"
    )
    cfg = build_config(args)
    image_shape = infer_image_shape(source_files, args.image_key, args.camera)
    cfg["image_key"] = args.image_key if image_shape is not None else "none"
    cfg["image_feature_key"] = image_feature_key(args.camera) if image_shape is not None else None
    dataset_features = dataset_features_with_optional_image(image_shape, args.camera)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),
        features=dataset_features,
        robot_type=args.robot_type,
        root=output_root,
        use_videos=False,
    )
    write_dataset_sidecar_meta(dataset.root, args.point_cloud_storage)

    jobs = [
        {
            "job_index": index,
            "source_path": str(source_path),
            "artifact_dir": str(tmp_dir / f"episode_{index:06d}"),
            "cfg": {**cfg, "seed": int(args.seed) + index * 1_000_000},
        }
        for index, source_path in enumerate(source_files)
    ]
    results: dict[int, Path] = {}
    start_time = time.time()
    try:
        if int(args.workers) <= 1:
            for job in tqdm(jobs, desc="Converting real HDF5", unit="episode"):
                result = convert_worker(job)
                results[int(result["job_index"])] = Path(result["artifact_dir"])
        else:
            with ProcessPoolExecutor(
                max_workers=min(int(args.workers), len(jobs)),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [executor.submit(convert_worker, job) for job in jobs]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Converting real HDF5",
                    unit="episode",
                ):
                    result = future.result()
                    results[int(result["job_index"])] = Path(result["artifact_dir"])

        records = []
        for job in jobs:
            record = save_artifact_to_dataset(dataset, results[int(job["job_index"])], cfg)
            records.append(record)
            export_preview(dataset.root, record, cfg)
            print(
                f"[OK] episode={record['episode_index']} frames={record['frames']} "
                f"task={record['task']!r} source={record['source_hdf5']}"
            )
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - start_time,
        "input_files": [str(path) for path in source_files],
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "fps": int(args.fps),
        "num_points": int(args.num_points),
        "point_cloud_storage": args.point_cloud_storage,
        "reference_frame": "overhead_camera",
        "uses_real_camera_extrinsic": False,
        "episodes": records,
    }
    summary_path = output_root / "real_hdf5_conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] converted {len(records)} real HDF5 episode(s) to {output_root}")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
