#!/usr/bin/env python
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.song_pointseg import (
    episode_point_cloud_npy_path,
    episode_point_cloud_zarr_path,
    open_episode_point_clouds,
    save_episode_point_clouds_zarr,
    save_point_clouds_zarr,
)

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import DEFAULT_LIBERO_CONFIG, LIBERO_DATA_ROOT, load_json_config
    from .libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        ensure_libero_config,
        fast_inverse_homogeneous,
        make_libero_env,
        normalize_camera_name,
        observation_to_camera_point_cloud,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        render_camera_names_from_config,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import DEFAULT_LIBERO_CONFIG, LIBERO_DATA_ROOT, load_json_config
    from libero_setting.libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        ensure_libero_config,
        fast_inverse_homogeneous,
        make_libero_env,
        normalize_camera_name,
        observation_to_camera_point_cloud,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        render_camera_names_from_config,
    )

from tqdm import tqdm

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


def image_feature_camera(cfg: dict[str, Any]) -> str:
    value = cfg.get("image_camera")
    return str(value) if value is not None else pointcloud_camera_names_from_config(cfg)[0]


def ensure_image_camera_rendered(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("save_rgb_images", True)):
        return
    image_camera = normalize_camera_name(image_feature_camera(cfg))
    camera_names = [normalize_camera_name(name) for name in list(cfg.get("camera_names") or [])]
    pointcloud_cameras = pointcloud_camera_names_from_config(cfg)
    if image_camera not in camera_names and image_camera not in pointcloud_cameras:
        camera_names.append(image_camera)
        cfg["camera_names"] = camera_names


def dataset_features_with_image(cfg: dict[str, Any]) -> dict[str, Any]:
    features = dict(DATASET_FEATURES)
    if bool(cfg.get("save_rgb_images", True)):
        features[image_feature_key(image_feature_camera(cfg))] = {
            "dtype": "image",
            "shape": (int(cfg.get("observation_height", 128)), int(cfg.get("observation_width", 128)), 3),
            "names": ["height", "width", "channels"],
        }
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO demonstrations into the Song point-cloud LeRobot format.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--demo-root", type=Path, default=None)
    parser.add_argument("--demo-file", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-demo", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--point-cloud-storage", choices=("zarr", "npy"), default=None)
    parser.add_argument("--zarr-compression-level", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--replay-mode", choices=("states", "step"), default=None)
    parser.add_argument("--download-demos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-use-huggingface", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--vis-count", type=int, default=None)
    parser.add_argument("--vis-stride", type=int, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-rgb-images", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--image-camera", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_json_config(
        path,
        path_keys=("dataset_output_root", "libero_config_path", "demo_root", "vis_dir", "output_dir"),
    )


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


def resolve_suite_names(cli_suites: list[str] | None, cfg: dict[str, Any]) -> list[str]:
    suites = cli_suites if cli_suites is not None else cfg.get("suites", cfg.get("suite", "libero_object"))
    if isinstance(suites, str):
        suites = [suites]
    suites = [str(suite) for suite in suites]
    if not suites:
        raise ValueError("At least one LIBERO suite must be configured.")
    return suites


def resolve_task_ids_for_suite(
    *,
    suite_name: str,
    task_count: int,
    cli_task_ids: list[int] | None,
    cfg: dict[str, Any],
) -> list[int]:
    if bool(cfg.get("all_tasks", False)):
        return list(range(task_count))
    if cli_task_ids is not None:
        task_ids = cli_task_ids
    else:
        suite_task_ids = cfg.get("suite_task_ids", {})
        if isinstance(suite_task_ids, dict) and suite_name in suite_task_ids:
            task_ids = suite_task_ids[suite_name]
        else:
            task_ids = cfg.get("task_ids")
    if task_ids is None:
        return list(range(task_count))
    resolved = [int(task_id) for task_id in task_ids]
    invalid = [task_id for task_id in resolved if task_id < 0 or task_id >= task_count]
    if invalid:
        raise ValueError(f"Invalid task id(s) for {suite_name}: {invalid}; valid range is [0, {task_count - 1}]")
    return resolved


def camera_image_keys(camera_names: list[str]) -> list[str]:
    keys = []
    for camera_name in camera_names:
        name = str(camera_name)
        keys.append(name if name.endswith("_image") else f"{name}_image")
    return keys


def image_from_raw_obs(raw_obs: dict[str, Any], image_key: str) -> np.ndarray | None:
    if image_key not in raw_obs:
        return None
    image = np.asarray(raw_obs[image_key])
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dataset_image_from_raw_obs(raw_obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    image_key = camera_name if str(camera_name).endswith("_image") else f"{camera_name}_image"
    image = image_from_raw_obs(raw_obs, image_key)
    if image is None:
        return None
    if image_key == "agentview_image":
        image = np.ascontiguousarray(image[:, ::-1])
    return image.copy()


def append_video_frames(video_frames: dict[str, list[np.ndarray]], raw_obs: dict[str, Any], camera_names: list[str]) -> None:
    for image_key in camera_image_keys(camera_names):
        image = image_from_raw_obs(raw_obs, image_key)
        if image is not None:
            if image_key == "agentview_image":
                image = np.ascontiguousarray(image[:, ::-1])
            video_frames.setdefault(image_key, []).append(image.copy())


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps)
        return
    except Exception:
        pass

    import cv2

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            frame = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def export_episode_videos(episode: dict[str, Any], video_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    if not cfg.get("save_video", False):
        return []
    video_frames = episode.get("video_frames") or {}
    if not video_frames:
        return []

    video_dir_name = record.get("video_dir_name")
    if video_dir_name is None:
        video_dir_name = f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"

    episode_dir = video_dir / video_dir_name

    written = []
    for image_key, frames in video_frames.items():
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key)
        path = episode_dir / f"{safe_key}.mp4"
        write_video(path, frames, int(cfg["fps"]))
        written.append(str(path))
    return written


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def point_cloud_file(root: Path, episode_index: int) -> Path:
    return episode_point_cloud_npy_path(root / POINT_CLOUD_DIR_NAME, episode_index)


def point_cloud_storage_path(root: Path, episode_index: int, storage: str) -> Path:
    if storage == "zarr":
        return episode_point_cloud_zarr_path(root / POINT_CLOUD_DIR_NAME, episode_index)
    return point_cloud_file(root, episode_index)


def world_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / WORLD_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def write_point_cloud_meta(root: Path, storage: str = "zarr") -> None:
    pc_dir = root / POINT_CLOUD_DIR_NAME
    pc_dir.mkdir(parents=True, exist_ok=True)
    suffix = "zarr" if storage == "zarr" else "npy"
    meta = {
        "key": "observation.point_cloud",
        "dtype": "float32",
        "shape": [None, POINT_CLOUD_CHANNELS],
        "variable_num_points": True,
        "layout": "episode_array",
        "storage_format": storage,
        "path_format": f"{POINT_CLOUD_DIR_NAME}/episode_{{episode_index:06d}}.{suffix}",
        "coordinate_frame": "current_eff",
        "source_reference_frame": "overview_camera",
    }
    if storage == "zarr":
        meta["zarr_encoding"] = "packed_xyz_float16_rgb_uint8"
    with open(pc_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_worldflow_meta(root: Path) -> None:
    pose_dir = root / WORLD_EE_POSE_DIR_NAME
    pose_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": "worldflow.ee_poses",
        "dtype": "float32",
        "shape": [9],
        "layout": "episode_npy",
        "path_format": f"{WORLD_EE_POSE_DIR_NAME}/episode_{{episode_index:06d}}.npy",
        "coordinate_frame": "overview_camera",
        "legacy_directory_name": True,
        "sim_extrinsic_usage": "eef_world_to_overview_camera_only",
    }
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_episode_point_clouds(
    root: Path,
    episode_index: int,
    point_clouds: np.ndarray,
    *,
    storage: str = "zarr",
    zarr_compression_level: int = 3,
) -> None:
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != POINT_CLOUD_CHANNELS:
        raise ValueError(f"Expected point clouds shape (T, N, 6), got {point_clouds.shape}")
    if storage == "zarr":
        save_episode_point_clouds_zarr(
            root / POINT_CLOUD_DIR_NAME,
            episode_index,
            point_clouds,
            compression_level=int(zarr_compression_level),
        )
    else:
        path = point_cloud_file(root, episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, point_clouds)


def save_episode_worldflow(root: Path, episode_index: int, reference_ee_poses: np.ndarray) -> None:
    reference_ee_poses = np.ascontiguousarray(reference_ee_poses, dtype=np.float32)
    if reference_ee_poses.ndim != 2 or reference_ee_poses.shape[-1] != 9:
        raise ValueError(f"Expected reference-frame ee poses shape (T, 9), got {reference_ee_poses.shape}")
    path = world_ee_pose_file(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reference_ee_poses)


def save_episode_images_to_paths(
    dataset: LeRobotDataset,
    images: np.ndarray,
    image_key: str,
    episode_index: int,
) -> list[str]:
    paths = []
    for frame_index, image in enumerate(np.asarray(images, dtype=np.uint8)):
        image_path = dataset._get_image_file_path(  # noqa: SLF001
            episode_index=episode_index,
            image_key=image_key,
            frame_index=frame_index,
        )
        if frame_index == 0:
            image_path.parent.mkdir(parents=True, exist_ok=True)
        dataset._save_image(image, image_path, compress_level=6)  # noqa: SLF001
        paths.append(str(image_path))
    return paths


def homo_to_pose9(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float32)
    return np.concatenate([H[..., :3, 3], H[..., :3, 0], H[..., :3, 1]], axis=-1).astype(np.float32)


def from_reference_to_umi_tra_pose9(obs_pose9_eff_to_reference: np.ndarray) -> np.ndarray:
    reference_transforms = pose9_to_homo_np(obs_pose9_eff_to_reference)
    eff0_to_reference = reference_transforms[0]
    reference_to_eff = fast_inverse_homogeneous(reference_transforms)
    eff0_to_eff = reference_to_eff @ eff0_to_reference
    eff_to_eff0 = fast_inverse_homogeneous(eff0_to_eff)
    return homo_to_pose9(eff_to_eff0)


def make_episode_buffer(
    dataset: LeRobotDataset,
    task: str,
    actions: np.ndarray,
    timestamps: np.ndarray,
    images: np.ndarray | None = None,
    image_key: str | None = None,
) -> dict[str, Any]:
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = len(actions)
    episode_buffer["task"] = [task] * len(actions)
    episode_buffer["frame_index"] = np.arange(len(actions), dtype=np.int64)
    episode_buffer["timestamp"] = timestamps
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = actions
    if images is not None and image_key is not None:
        if len(images) != len(actions):
            raise ValueError(f"Image frame count {len(images)} does not match actions {len(actions)}.")
        episode_buffer[image_key] = save_episode_images_to_paths(
            dataset,
            images,
            image_key,
            int(episode_buffer["episode_index"]),
        )
    return episode_buffer


def save_converted_episode(dataset: LeRobotDataset, episode: dict[str, Any]) -> None:
    episode_index = dataset.meta.total_episodes
    save_episode_point_clouds(dataset.root, episode_index, episode["point_clouds"])
    save_episode_worldflow(dataset.root, episode_index, episode["world_ee_poses"])
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            episode["task"],
            episode["actions"],
            episode["timestamps"],
        )
    )


def move_episode_array(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def save_episode_artifact(artifact_dir: Path, episode: dict[str, Any], record: dict[str, Any], cfg: dict[str, Any]) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if str(cfg.get("point_cloud_storage", "zarr")) == "zarr":
        save_point_clouds_zarr(
            artifact_dir / "point_clouds.zarr",
            episode["point_clouds"],
            compression_level=int(cfg.get("zarr_compression_level", 3)),
        )
    else:
        np.save(artifact_dir / "point_clouds.npy", np.ascontiguousarray(episode["point_clouds"], dtype=np.float32))
    np.save(artifact_dir / "world_ee_poses.npy", np.ascontiguousarray(episode["world_ee_poses"], dtype=np.float32))
    np.save(artifact_dir / "actions.npy", np.ascontiguousarray(episode["actions"], dtype=np.float32))
    np.save(artifact_dir / "timestamps.npy", np.asarray(episode["timestamps"], dtype=np.float32))
    if "images" in episode:
        np.save(artifact_dir / "images.npy", np.ascontiguousarray(episode["images"], dtype=np.uint8))
    with open(artifact_dir / "record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    if cfg.get("save_video", False):
        video_record = {**record, "video_dir_name": "videos"}
        export_episode_videos(episode, artifact_dir, video_record, cfg)


def move_episode_videos(artifact_dir: Path, vis_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    if not cfg.get("save_video", False):
        return []
    source_dir = artifact_dir / "videos"
    if not source_dir.exists():
        return []
    dest_dir = vis_dir / f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not source_path.is_file():
            continue
        dest_path = dest_dir / source_path.name
        if dest_path.exists():
            dest_path.unlink()
        shutil.move(str(source_path), str(dest_path))
        written.append(str(dest_path))
    return written


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-6, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-6, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def write_ascii_ply_points(path: Path, xyzrgb: np.ndarray) -> None:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[1] != 6:
        raise ValueError(f"Expected xyzrgb shape (N, 6), got {xyzrgb.shape}")
    colors = np.clip(xyzrgb[:, 3:6], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyzrgb.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(xyzrgb[:, :3], colors):
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def write_ascii_ply_lines(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected line points shape (N, 3), got {points.shape}")
    lines = [(idx, idx + 1) for idx in range(max(0, len(points) - 1))]
    if colors is None:
        colors = np.tile(np.asarray([[0, 180, 255]], dtype=np.uint8), (len(lines), 1))
    colors = np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f}\n")
        for (start, end), rgb in zip(lines, colors):
            f.write(f"{start} {end} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def write_ascii_ply_frame(path: Path, pose9: np.ndarray, scale: float = 0.04) -> None:
    points = make_frame_points(pose9, scale=scale)
    lines = [(0, 1), (0, 2), (0, 3)]
    colors = np.asarray([[255, 40, 40], [40, 220, 40], [60, 120, 255]], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f}\n")
        for (start, end), rgb in zip(lines, colors):
            f.write(f"{start} {end} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def make_frame_points(pose9: np.ndarray, scale: float = 0.04) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    origin = pose9[:3]
    rot = rot6d_to_matrix(pose9[3:9])
    endpoints = origin[None] + rot.T * scale
    return np.concatenate([origin[None], endpoints], axis=0)


def export_episode_preview(episode: dict[str, Any], vis_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> None:
    vis_count = int(cfg.get("vis_count", 0) or 0)
    if vis_count <= 0:
        return
    stride = max(1, int(cfg.get("vis_stride", 1) or 1))
    episode_dir = vis_dir / f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    point_clouds = np.asarray(episode["point_clouds"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    reference_ee_poses = np.asarray(episode["world_ee_poses"], dtype=np.float32)
    candidate_indices = list(range(0, len(point_clouds), stride))
    if len(candidate_indices) > vis_count:
        pick = np.linspace(0, len(candidate_indices) - 1, vis_count).round().astype(int)
        frame_indices = [candidate_indices[int(i)] for i in pick]
    else:
        frame_indices = candidate_indices

    for frame_idx in frame_indices:
        write_ascii_ply_points(episode_dir / f"frame_{frame_idx:04d}_point_cloud_eff.ply", point_clouds[frame_idx])
        write_ascii_ply_frame(episode_dir / f"frame_{frame_idx:04d}_umi_action_frame.ply", actions[frame_idx, :9])

    write_ascii_ply_lines(episode_dir / "umi_action_trajectory.ply", actions[:, :3])
    write_ascii_ply_lines(episode_dir / "reference_ee_trajectory.ply", reference_ee_poses[:, :3])
    preview = {
        **record,
        "frame_indices": [int(idx) for idx in frame_indices],
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "render_camera_names": render_camera_names_from_config(cfg),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_template": str(cfg.get("gripper_template", "reap")),
        "files": {
            "umi_action_trajectory": "umi_action_trajectory.ply",
            "reference_ee_trajectory": "reference_ee_trajectory.ply",
            "point_cloud_pattern": "frame_XXXX_point_cloud_eff.ply",
            "frame_pattern": "frame_XXXX_umi_action_frame.ply",
        },
    }
    with open(episode_dir / "preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)


def get_libero_dataset_root(cli_demo_root: Path | None, cfg: dict[str, Any]) -> Path:
    if cli_demo_root is not None:
        return cli_demo_root.expanduser().resolve()
    if cfg.get("demo_root"):
        return Path(cfg["demo_root"]).expanduser().resolve()
    from libero.libero import get_libero_path

    return Path(get_libero_path("datasets")).expanduser().resolve()


def downloadable_suite_name(suite_name: str) -> str | None:
    if suite_name in {"libero_object", "libero_goal", "libero_spatial"}:
        return suite_name
    if suite_name in {"libero_10", "libero_90", "libero_100"}:
        return "libero_100"
    return None


def maybe_download_demos(cfg: dict[str, Any], demo_root: Path) -> None:
    if not cfg.get("download_demos", False):
        return
    dataset_name = downloadable_suite_name(str(cfg["suite"]))
    if dataset_name is None:
        raise ValueError(f"Do not know which LIBERO demo package to download for suite {cfg['suite']!r}.")
    from libero.libero.utils.download_utils import libero_dataset_download


    print(f"No local demos found. Downloading {dataset_name} demos to {demo_root} ...")
    libero_dataset_download(
        datasets=dataset_name,
        download_dir=str(demo_root),
        check_overwrite=False,
        use_huggingface=bool(cfg.get("download_use_huggingface", True)),
    )


def find_demo_file(task: Any, demo_root: Path, explicit_files: list[Path] | None, cfg: dict[str, Any]) -> Path:
    if explicit_files:
        if len(explicit_files) == 1:
            path = explicit_files[0].expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Explicit demo file does not exist: {path}")
            return path
        task_key = normalized_name(task.name)
        for file_path in explicit_files:
            path = file_path.expanduser().resolve()
            if task_key in normalized_name(path.as_posix()):
                return path
        raise FileNotFoundError(f"No explicit --demo-file matches task {task.name!r}.")

    candidates = sorted(list(demo_root.rglob("*.hdf5")) + list(demo_root.rglob("*.h5")), key=natural_key)
    if not candidates:
        maybe_download_demos(cfg, demo_root)
        candidates = sorted(list(demo_root.rglob("*.hdf5")) + list(demo_root.rglob("*.h5")), key=natural_key)
    if not candidates:
        download_hint = (
            f"No .hdf5/.h5 demo files found under {demo_root}.\n"
            "Download LIBERO demos first, or rerun with --download-demos. Example:\n"
            f"  python {Path(__file__).name} --config <config> --suite {cfg['suite']} --task-id <id> --episodes 1 --download-demos\n"
            f"Or pass --demo-root /path/to/libero/datasets / --demo-file /path/to/demo.hdf5."
        )
        raise FileNotFoundError(download_hint)

    task_keys = [
        normalized_name(task.name),
        normalized_name(getattr(task, "problem_folder", "")),
        normalized_name(Path(getattr(task, "bddl_file", "")).stem),
    ]
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        key = normalized_name(candidate.as_posix())
        score = sum(1 for task_key in task_keys if task_key and task_key in key)
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        raise FileNotFoundError(
            f"Could not find a demo file for task {task.name!r} under {demo_root}. "
            "Pass --demo-file explicitly, or verify that the downloaded suite matches --suite."
        )
    scored.sort(key=lambda item: (-item[0], natural_key(item[1])))
    return scored[0][1]


def iter_demo_groups(demo_file: Path):
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        names = sorted(
            [name for name in root.keys() if isinstance(root[name], h5py.Group)],
            key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)],
        )
        for name in names:
            group = root[name]
            if "states" not in group:
                continue
            yield name, group["states"][:], group["actions"][:] if "actions" in group else None


def iter_demo_group_names(demo_file: Path) -> list[str]:
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        return sorted(
            [
                name
                for name in root.keys()
                if isinstance(root[name], h5py.Group) and "states" in root[name]
            ],
            key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)],
        )


def load_demo_group(demo_file: Path, demo_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        if demo_name not in root:
            raise KeyError(f"Demo group {demo_name!r} not found in {demo_file}")
        group = root[demo_name]
        if "states" not in group:
            raise KeyError(f"Demo group {demo_name!r} in {demo_file} is missing 'states'.")
        states = group["states"][:]
        actions = group["actions"][:] if "actions" in group else None
        return states, actions


def set_env_state_and_get_obs(env: Any, state: np.ndarray) -> dict[str, Any]:
    try:
        return env.set_init_state(state)
    except Exception:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        if hasattr(env, "_get_observations"):
            return env._get_observations()
        return env.env._get_observations()


def collect_demo_episode(
    *,
    env: Any,
    states: np.ndarray,
    actions: np.ndarray | None,
    task_language: str,
    cfg: dict[str, Any],
    max_frames: int | None,
    episode_seed: int,
) -> dict[str, Any]:
    if states.ndim != 2:
        raise ValueError(f"Expected demo states shape (T, D), got {states.shape}")
    frame_count = len(states)
    if cfg["replay_mode"] == "step" and actions is not None:
        frame_count = min(frame_count, len(actions) + 1)
    if max_frames is not None and max_frames > 0:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 1:
        raise ValueError("A collected LIBERO episode needs at least two frames.")

    num_points = int(cfg["num_points"])
    observation_height = int(cfg["observation_height"])
    observation_width = int(cfg["observation_width"])
    save_video = bool(cfg.get("save_video", False))
    point_clouds_reference = np.empty((frame_count, num_points, POINT_CLOUD_CHANNELS), dtype=np.float32)
    reference_ee_poses = np.empty((frame_count, 9), dtype=np.float32)
    grippers = np.empty((frame_count, 1), dtype=np.float32)
    video_frames: dict[str, list[np.ndarray]] = {} if save_video else {}
    image_frames: list[np.ndarray] = []
    save_rgb_images = bool(cfg.get("save_rgb_images", True))
    image_camera = image_feature_camera(cfg) if save_rgb_images else None
    pc_camera_names = pointcloud_camera_names_from_config(cfg)

    if cfg["replay_mode"] == "step" and actions is not None:
        raw_obs = set_env_state_and_get_obs(env, states[0])
        for frame_idx in range(frame_count):
            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            if save_rgb_images and image_camera is not None:
                image = dataset_image_from_raw_obs(raw_obs, image_camera)
                if image is None:
                    raise KeyError(f"Missing RGB image for camera {image_camera!r} in LIBERO observation.")
                image_frames.append(image)
            point_cloud_reference, pose9_gripper_reference, _pose9_gripper_sim_world = (
                observation_to_camera_point_cloud(
                    env,
                    raw_obs,
                    pc_camera_names,
                    observation_height,
                    observation_width,
                    num_points,
                    seed=episode_seed + frame_idx,
                )
            )
            point_clouds_reference[frame_idx] = point_cloud_reference
            reference_ee_poses[frame_idx] = pose9_gripper_reference[:9]
            grippers[frame_idx, 0] = pose9_gripper_reference[-1]
            if frame_idx < frame_count - 1:
                raw_obs, _, _, _ = env.step(actions[frame_idx])
    else:
        for frame_idx in range(frame_count):
            raw_obs = set_env_state_and_get_obs(env, states[frame_idx])
            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            if save_rgb_images and image_camera is not None:
                image = dataset_image_from_raw_obs(raw_obs, image_camera)
                if image is None:
                    raise KeyError(f"Missing RGB image for camera {image_camera!r} in LIBERO observation.")
                image_frames.append(image)
            point_cloud_reference, pose9_gripper_reference, _pose9_gripper_sim_world = (
                observation_to_camera_point_cloud(
                    env,
                    raw_obs,
                    pc_camera_names,
                    observation_height,
                    observation_width,
                    num_points,
                    seed=episode_seed + frame_idx,
                )
            )
            point_clouds_reference[frame_idx] = point_cloud_reference
            reference_ee_poses[frame_idx] = pose9_gripper_reference[:9]
            grippers[frame_idx, 0] = pose9_gripper_reference[-1]

    if cfg.get("add_gripper_cloud", True):
        point_clouds_reference = add_reference_gripper_clouds_to_episode(
            point_clouds_reference,
            reference_ee_poses,
            grippers.reshape(-1),
            total_points=int(cfg["num_points"]),
            gripper_points=int(cfg.get("gripper_points", 500)),
            gripper_len=float(cfg.get("gripper_len", 0.06)),
            gripper_template=str(cfg.get("gripper_template", "reap")),
            seed=episode_seed,
            drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
            shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
            widths_are_normalized=False,
            gripper_max_width=float(cfg.get("gripper_qpos_max_width", 0.08)),
        )
    point_clouds = reference_point_cloud_to_current_eff(point_clouds_reference, reference_ee_poses)
    umi_poses = from_reference_to_umi_tra_pose9(reference_ee_poses)
    episode_actions = np.concatenate([umi_poses, grippers], axis=-1).astype(np.float32)
    timestamps = np.arange(len(episode_actions), dtype=np.float32) / float(cfg["fps"])

    episode = {
        "task": task_language,
        "actions": episode_actions,
        "point_clouds": point_clouds,
        # Legacy key/path retained for the existing WorldFlow dataset wrapper.
        # Values are expressed in the fixed Overview-camera reference frame.
        "world_ee_poses": reference_ee_poses,
        "timestamps": timestamps,
        "video_frames": video_frames,
    }
    if save_rgb_images:
        if len(image_frames) != len(episode_actions):
            raise ValueError(f"Collected {len(image_frames)} RGB frames for {len(episode_actions)} actions.")
        episode["images"] = np.asarray(image_frames, dtype=np.uint8)
    return episode


def make_episode_record(
    *,
    suite_name: str,
    task_id: int,
    task: Any,
    demo_file: Path,
    demo_name: str,
    frames: int,
) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "task_id": int(task_id),
        "task_name": task.name,
        "task_language": task.language,
        "problem_folder": getattr(task, "problem_folder", ""),
        "bddl_file": getattr(task, "bddl_file", ""),
        "demo_file": str(demo_file),
        "demo_name": demo_name,
        "frames": int(frames),
    }


def collect_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(job["cfg"])
    suite_name = str(job["suite"])
    cfg["suite"] = suite_name
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    task_id = int(job["task_id"])
    demo_file = Path(job["demo_file"])
    demo_name = str(job["demo_name"])
    states, actions = load_demo_group(demo_file, demo_name)

    env, task = make_libero_env(
        suite,
        task_id,
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        render_camera_names_from_config(cfg),
    )
    try:
        episode = collect_demo_episode(
            env=env,
            states=states,
            actions=actions,
            task_language=task.language,
            cfg=cfg,
            max_frames=job.get("max_frames"),
            episode_seed=int(job["episode_seed"]),
        )
    finally:
        env.close()

    record = make_episode_record(
        suite_name=suite_name,
        task_id=task_id,
        task=task,
        demo_file=demo_file,
        demo_name=demo_name,
        frames=len(episode["actions"]),
    )
    artifact_dir = Path(job["tmp_path"])
    save_episode_artifact(artifact_dir, episode, record, cfg)
    return {"job_index": int(job["job_index"]), "tmp_path": str(artifact_dir)}


def collect_task_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect multiple demos for one LIBERO task while reusing a single environment."""
    cfg = dict(job["cfg"])
    suite_name = str(job["suite"])
    cfg["suite"] = suite_name
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    task_id = int(job["task_id"])
    env, task = make_libero_env(
        suite,
        task_id,
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        render_camera_names_from_config(cfg),
    )

    results: list[dict[str, Any]] = []
    try:
        for episode_job in job["episodes"]:
            demo_file = Path(episode_job["demo_file"])
            demo_name = str(episode_job["demo_name"])
            states, actions = load_demo_group(demo_file, demo_name)
            episode = collect_demo_episode(
                env=env,
                states=states,
                actions=actions,
                task_language=task.language,
                cfg=cfg,
                max_frames=episode_job.get("max_frames"),
                episode_seed=int(episode_job["episode_seed"]),
            )
            record = make_episode_record(
                suite_name=suite_name,
                task_id=task_id,
                task=task,
                demo_file=demo_file,
                demo_name=demo_name,
                frames=len(episode["actions"]),
            )
            artifact_dir = Path(episode_job["tmp_path"])
            save_episode_artifact(artifact_dir, episode, record, cfg)
            results.append({"job_index": int(episode_job["job_index"]), "tmp_path": str(artifact_dir)})
    finally:
        env.close()

    return results


def save_collected_temp_episode(
    *,
    temp_path: Path,
    dataset: LeRobotDataset,
    vis_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = temp_path
    with open(artifact_dir / "record.json", "r", encoding="utf-8") as f:
        record = json.load(f)
    actions = np.load(artifact_dir / "actions.npy")
    timestamps = np.load(artifact_dir / "timestamps.npy")
    images_path = artifact_dir / "images.npy"
    images = np.load(images_path, mmap_mode="r") if images_path.exists() else None
    episode_index = int(dataset.meta.total_episodes)

    point_cloud_storage = str(cfg.get("point_cloud_storage", "zarr"))
    final_point_cloud_path = point_cloud_storage_path(dataset.root, episode_index, point_cloud_storage)
    final_world_ee_pose_path = world_ee_pose_file(dataset.root, episode_index)
    artifact_point_cloud_path = artifact_dir / "point_clouds.npy"
    artifact_point_cloud_zarr_path = artifact_dir / "point_clouds.zarr"
    if point_cloud_storage == "zarr":
        if artifact_point_cloud_zarr_path.exists():
            move_episode_array(artifact_point_cloud_zarr_path, final_point_cloud_path)
        elif artifact_point_cloud_path.exists():
            save_episode_point_clouds_zarr(
                dataset.root / POINT_CLOUD_DIR_NAME,
                episode_index,
                np.load(artifact_point_cloud_path, mmap_mode="r"),
                compression_level=int(cfg.get("zarr_compression_level", 3)),
            )
            artifact_point_cloud_path.unlink(missing_ok=True)
        else:
            raise FileNotFoundError(f"Missing point cloud artifact under {artifact_dir}")
    else:
        move_episode_array(artifact_point_cloud_path, final_point_cloud_path)
    move_episode_array(artifact_dir / "world_ee_poses.npy", final_world_ee_pose_path)
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            str(record["task_language"]),
            actions,
            timestamps,
            images=images,
            image_key=cfg.get("image_feature_key"),
        )
    )
    record["episode_index"] = episode_index
    if int(cfg.get("vis_count", 0) or 0) > 0:
        preview_episode = {
            "actions": actions,
            "point_clouds": open_episode_point_clouds(dataset.root / POINT_CLOUD_DIR_NAME, episode_index),
            "world_ee_poses": np.load(final_world_ee_pose_path, mmap_mode="r"),
        }
        export_episode_preview(preview_episode, vis_dir, record, cfg)
    record["videos"] = move_episode_videos(artifact_dir, vis_dir, record, cfg)
    return record


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    suite_names = resolve_suite_names(args.suite, cfg)
    cfg["suites"] = suite_names
    cfg["all_tasks"] = bool(cfg_get(cfg, args.all_tasks, "all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes", 1))
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 10000))
    cfg["point_cloud_storage"] = str(cfg_get(cfg, args.point_cloud_storage, "point_cloud_storage", "zarr"))
    cfg["zarr_compression_level"] = int(cfg_get(cfg, args.zarr_compression_level, "zarr_compression_level", 3))
    cfg["fps"] = int(cfg_get(cfg, args.fps, "fps", 30))
    cfg["replay_mode"] = cfg_get(cfg, args.replay_mode, "replay_mode", "states")
    cfg["download_demos"] = bool(cfg_get(cfg, args.download_demos, "download_demos", True))
    cfg["download_use_huggingface"] = bool(cfg_get(cfg, args.download_use_huggingface, "download_use_huggingface", True))
    cfg["overwrite_dataset"] = bool(cfg_get(cfg, args.overwrite, "overwrite_dataset", True))
    cfg["vis_count"] = int(cfg_get(cfg, args.vis_count, "vis_count", 0) or 0)
    cfg["vis_stride"] = int(cfg_get(cfg, args.vis_stride, "vis_stride", 1) or 1)
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video", False))
    cfg["save_rgb_images"] = bool(cfg_get(cfg, args.save_rgb_images, "save_rgb_images", True))
    image_camera_value = cfg_get(cfg, args.image_camera, "image_camera")
    if image_camera_value is not None:
        cfg["image_camera"] = str(image_camera_value)
    ensure_image_camera_rendered(cfg)
    cfg["num_workers"] = int(cfg_get(cfg, args.num_workers, "num_workers", 1) or 1)
    max_frames = cfg_get(cfg, args.max_frames_per_demo, "max_frames_per_demo")
    max_frames = int(max_frames) if max_frames is not None else None
    ensure_libero_config(cfg.get("libero_config_path"), args.demo_root or cfg.get("demo_root"))

    output_root = Path(
        cfg_get(cfg, args.output_root, "dataset_output_root", LIBERO_DATA_ROOT / "libero_lerobot_dataset")
    ).expanduser().resolve()
    repo_id = cfg_get(cfg, args.repo_id, "dataset_repo_id", "song_libero_pointcloud")
    demo_root = get_libero_dataset_root(args.demo_root, cfg)
    cfg["demo_root"] = str(demo_root)
    cfg["image_feature_key"] = (
        image_feature_key(image_feature_camera(cfg)) if bool(cfg.get("save_rgb_images", True)) else None
    )
    vis_dir_value = cfg_get(cfg, args.vis_dir, "vis_dir")
    vis_dir = Path(vis_dir_value).expanduser().resolve() if vis_dir_value else output_root / "visualizations"
    tmp_dir = (
        args.tmp_dir.expanduser().resolve()
        if args.tmp_dir is not None
        else output_root.parent / f".{output_root.name}_tmp"
    )

    from libero.libero import benchmark

    if output_root.exists():
        if not cfg["overwrite_dataset"]:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(cfg["fps"]),
        features=dataset_features_with_image(cfg),
        robot_type="libero",
        root=output_root,
        use_videos=False,
    )
    write_point_cloud_meta(dataset.root, storage=str(cfg["point_cloud_storage"]))
    write_worldflow_meta(dataset.root)

    summary: dict[str, Any] = {
        "created_unix_s": time.time(),
        "suites": suite_names,
        "demo_root": str(demo_root),
        "output_root": str(output_root),
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "reference_frame": "overview_camera",
        "reference_camera": pointcloud_camera_names_from_config(cfg)[0],
        "sim_extrinsic_usage": "eef_world_to_overview_camera_only",
        "render_camera_names": render_camera_names_from_config(cfg),
        "image_camera": image_feature_camera(cfg) if bool(cfg.get("save_rgb_images", True)) else None,
        "image_feature_key": cfg.get("image_feature_key"),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_template": str(cfg.get("gripper_template", "reap")),
        "point_cloud_storage": str(cfg.get("point_cloud_storage", "zarr")),
        "point_cloud_zarr_encoding": "packed_xyz_float16_rgb_uint8",
        "zarr_compression_level": int(cfg.get("zarr_compression_level", 3)),
        "num_workers": int(cfg["num_workers"]),
        "episodes": [],
    }

    episode_jobs: list[dict[str, Any]] = []
    task_jobs: list[dict[str, Any]] = []
    job_index = 0
    task_job_index = 0
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_index, suite_name in enumerate(suite_names):
        suite_cls = benchmark_dict[suite_name]
        suite = suite_cls()
        task_ids = resolve_task_ids_for_suite(
            suite_name=suite_name,
            task_count=len(suite.tasks),
            cli_task_ids=args.task_id,
            cfg=cfg,
        )
        suite_cfg = dict(cfg)
        suite_cfg["suite"] = suite_name
        for task_id_value in task_ids:
            task_id = int(task_id_value)
            task = suite.get_task(task_id)
            demo_file = find_demo_file(task, demo_root, args.demo_file, suite_cfg)
            demo_names = iter_demo_group_names(demo_file)[: int(cfg["episodes"])]
            if not demo_names:
                raise RuntimeError(f"No usable demos with states found in {demo_file}")
            task_episode_jobs = []
            for converted_for_task, demo_name in enumerate(demo_names):
                episode_job = {
                    "job_index": job_index,
                    "suite": suite_name,
                    "task_id": task_id,
                    "demo_file": str(demo_file),
                    "demo_name": demo_name,
                    "episode_seed": suite_index * 10_000_000 + task_id * 100000 + converted_for_task * 1000,
                    "max_frames": max_frames,
                    "cfg": suite_cfg,
                    "tmp_path": str(tmp_dir / f"episode_job_{job_index:06d}_{suite_name}_task_{task_id:03d}"),
                }
                episode_jobs.append(episode_job)
                task_episode_jobs.append(episode_job)
                job_index += 1
            task_jobs.append(
                {
                    "task_job_index": task_job_index,
                    "suite": suite_name,
                    "task_id": task_id,
                    "cfg": suite_cfg,
                    "episodes": task_episode_jobs,
                }
            )
            task_job_index += 1

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, Path] = {}
    if int(cfg["num_workers"]) <= 1:
        for task_job in tqdm(task_jobs, desc="Collecting LIBERO tasks", unit="task"):
            for result in collect_task_worker(task_job):
                results[int(result["job_index"])] = Path(result["tmp_path"])
    else:
        print(
            f"[info] collecting {len(episode_jobs)} LIBERO episode(s) across {len(task_jobs)} task job(s) "
            f"with {cfg['num_workers']} worker(s)"
        )
        with ProcessPoolExecutor(
            max_workers=int(cfg["num_workers"]),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(collect_task_worker, task_job) for task_job in task_jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting LIBERO tasks", unit="task"):
                for result in future.result():
                    results[int(result["job_index"])] = Path(result["tmp_path"])

    for episode_job in episode_jobs:
        temp_path = results[int(episode_job["job_index"])]
        record = save_collected_temp_episode(
            temp_path=temp_path,
            dataset=dataset,
            vis_dir=vis_dir,
            cfg=cfg,
        )
        summary["episodes"].append(record)
        print(
            f"[OK] suite={record['suite']} task={record['task_id']} demo={record['demo_name']} "
            f"frames={record['frames']} episode_index={record['episode_index']}"
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    dataset.finalize()
    summary["num_episodes"] = len(summary["episodes"])
    summary_path = output_root / "libero_collect_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] collected {summary['num_episodes']} episode(s) at {output_root}")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
