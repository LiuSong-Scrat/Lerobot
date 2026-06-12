#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from libero_pointcloud_utils import (
    add_world_gripper_clouds_to_episode,
    ensure_libero_config,
    fast_inverse_homogeneous,
    make_libero_env,
    observation_to_world_point_cloud,
    pointcloud_camera_names_from_config,
    pose9_to_homo_np,
    render_camera_names_from_config,
    world_point_cloud_to_current_eff,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO demonstrations into the Song point-cloud LeRobot format.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "libero.json")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--demo-root", type=Path, default=None)
    parser.add_argument("--demo-file", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-frames-per-demo", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--replay-mode", choices=("states", "step"), default=None)
    parser.add_argument("--download-demos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-use-huggingface", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--vis-count", type=int, default=None)
    parser.add_argument("--vis-stride", type=int, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


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
    return root / POINT_CLOUD_DIR_NAME / f"episode_{episode_index:06d}.npy"


def world_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / WORLD_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def write_point_cloud_meta(root: Path) -> None:
    pc_dir = root / POINT_CLOUD_DIR_NAME
    pc_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": "observation.point_cloud",
        "dtype": "float32",
        "shape": [None, POINT_CLOUD_CHANNELS],
        "variable_num_points": True,
        "layout": "episode_npy",
        "path_format": f"{POINT_CLOUD_DIR_NAME}/episode_{{episode_index:06d}}.npy",
    }
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
    }
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_episode_point_clouds(root: Path, episode_index: int, point_clouds: np.ndarray) -> None:
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != POINT_CLOUD_CHANNELS:
        raise ValueError(f"Expected point clouds shape (T, N, 6), got {point_clouds.shape}")
    path = point_cloud_file(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, point_clouds)


def save_episode_worldflow(root: Path, episode_index: int, world_ee_poses: np.ndarray) -> None:
    world_ee_poses = np.ascontiguousarray(world_ee_poses, dtype=np.float32)
    if world_ee_poses.ndim != 2 or world_ee_poses.shape[-1] != 9:
        raise ValueError(f"Expected world ee poses shape (T, 9), got {world_ee_poses.shape}")
    path = world_ee_pose_file(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, world_ee_poses)


def homo_to_pose9(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float32)
    return np.concatenate([H[..., :3, 3], H[..., :3, 0], H[..., :3, 1]], axis=-1).astype(np.float32)


def from_world_to_umi_tra_pose9(obs_pose9_eff_to_world: np.ndarray) -> np.ndarray:
    T_world = pose9_to_homo_np(obs_pose9_eff_to_world)
    T_eff0_to_world = T_world[0]
    T_inv = fast_inverse_homogeneous(T_world)
    T_eff0_to_eff = T_inv @ T_eff0_to_world
    T_eff_to_eff0 = fast_inverse_homogeneous(T_eff0_to_eff)
    return homo_to_pose9(T_eff_to_eff0)


def make_episode_buffer(dataset: LeRobotDataset, task: str, actions: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = len(actions)
    episode_buffer["task"] = [task] * len(actions)
    episode_buffer["frame_index"] = np.arange(len(actions), dtype=np.int64)
    episode_buffer["timestamp"] = timestamps
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = actions
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
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def save_episode_artifact(artifact_dir: Path, episode: dict[str, Any], record: dict[str, Any], cfg: dict[str, Any]) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "point_clouds.npy", np.ascontiguousarray(episode["point_clouds"], dtype=np.float32))
    np.save(artifact_dir / "world_ee_poses.npy", np.ascontiguousarray(episode["world_ee_poses"], dtype=np.float32))
    np.save(artifact_dir / "actions.npy", np.ascontiguousarray(episode["actions"], dtype=np.float32))
    np.save(artifact_dir / "timestamps.npy", np.asarray(episode["timestamps"], dtype=np.float32))
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
    world_ee_poses = np.asarray(episode["world_ee_poses"], dtype=np.float32)
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
    write_ascii_ply_lines(episode_dir / "world_ee_trajectory.ply", world_ee_poses[:, :3])
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
            "world_ee_trajectory": "world_ee_trajectory.ply",
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
    point_clouds_world = np.empty((frame_count, num_points, POINT_CLOUD_CHANNELS), dtype=np.float32)
    world_ee_poses = np.empty((frame_count, 9), dtype=np.float32)
    grippers = np.empty((frame_count, 1), dtype=np.float32)
    video_frames: dict[str, list[np.ndarray]] = {} if save_video else {}
    pc_camera_names = pointcloud_camera_names_from_config(cfg)

    if cfg["replay_mode"] == "step" and actions is not None:
        raw_obs = set_env_state_and_get_obs(env, states[0])
        for frame_idx in range(frame_count):
            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            point_cloud_world, pose9_gripper = observation_to_world_point_cloud(
                env,
                raw_obs,
                pc_camera_names,
                observation_height,
                observation_width,
                num_points,
                seed=episode_seed + frame_idx,
            )
            point_clouds_world[frame_idx] = point_cloud_world
            world_ee_poses[frame_idx] = pose9_gripper[:9]
            grippers[frame_idx, 0] = pose9_gripper[-1]
            if frame_idx < frame_count - 1:
                raw_obs, _, _, _ = env.step(actions[frame_idx])
    else:
        for frame_idx in range(frame_count):
            raw_obs = set_env_state_and_get_obs(env, states[frame_idx])
            if save_video:
                append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            point_cloud_world, pose9_gripper = observation_to_world_point_cloud(
                env,
                raw_obs,
                pc_camera_names,
                observation_height,
                observation_width,
                num_points,
                seed=episode_seed + frame_idx,
            )
            point_clouds_world[frame_idx] = point_cloud_world
            world_ee_poses[frame_idx] = pose9_gripper[:9]
            grippers[frame_idx, 0] = pose9_gripper[-1]

    if cfg.get("add_gripper_cloud", True):
        point_clouds = add_world_gripper_clouds_to_episode(
            point_clouds_world,
            world_ee_poses,
            grippers.reshape(-1),
            total_points=int(cfg["num_points"]),
            gripper_points=int(cfg.get("gripper_points", 500)),
            gripper_len=float(cfg.get("gripper_len", 0.06)),
            gripper_template=str(cfg.get("gripper_template", "reap")),
            seed=episode_seed,
            drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
            shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
            widths_are_normalized=False,
        )
    else:
        point_clouds = world_point_cloud_to_current_eff(point_clouds_world, world_ee_poses)
    umi_poses = from_world_to_umi_tra_pose9(world_ee_poses)
    episode_actions = np.concatenate([umi_poses, grippers], axis=-1).astype(np.float32)
    timestamps = np.arange(len(episode_actions), dtype=np.float32) / float(cfg["fps"])

    return {
        "task": task_language,
        "actions": episode_actions,
        "point_clouds": point_clouds,
        "world_ee_poses": world_ee_poses,
        "timestamps": timestamps,
        "video_frames": video_frames,
    }


def make_episode_record(
    *,
    task_id: int,
    task: Any,
    demo_file: Path,
    demo_name: str,
    frames: int,
) -> dict[str, Any]:
    return {
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
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
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
        task_id=task_id,
        task=task,
        demo_file=demo_file,
        demo_name=demo_name,
        frames=len(episode["actions"]),
    )
    artifact_dir = Path(job["tmp_path"])
    save_episode_artifact(artifact_dir, episode, record, cfg)
    return {"job_index": int(job["job_index"]), "tmp_path": str(artifact_dir)}


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
    episode_index = int(dataset.meta.total_episodes)

    final_point_cloud_path = point_cloud_file(dataset.root, episode_index)
    final_world_ee_pose_path = world_ee_pose_file(dataset.root, episode_index)
    move_episode_array(artifact_dir / "point_clouds.npy", final_point_cloud_path)
    move_episode_array(artifact_dir / "world_ee_poses.npy", final_world_ee_pose_path)
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            str(record["task_language"]),
            actions,
            timestamps,
        )
    )
    record["episode_index"] = episode_index
    if int(cfg.get("vis_count", 0) or 0) > 0:
        preview_episode = {
            "actions": actions,
            "point_clouds": np.load(final_point_cloud_path, mmap_mode="r"),
            "world_ee_poses": np.load(final_world_ee_pose_path, mmap_mode="r"),
        }
        export_episode_preview(preview_episode, vis_dir, record, cfg)
    record["videos"] = move_episode_videos(artifact_dir, vis_dir, record, cfg)
    return record


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["suite"] = cfg_get(cfg, args.suite, "suite")
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes", 1))
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 50000))
    cfg["fps"] = int(cfg_get(cfg, args.fps, "fps", 30))
    cfg["replay_mode"] = cfg_get(cfg, args.replay_mode, "replay_mode", "states")
    cfg["download_demos"] = bool(cfg_get(cfg, args.download_demos, "download_demos", True))
    cfg["download_use_huggingface"] = bool(cfg_get(cfg, args.download_use_huggingface, "download_use_huggingface", True))
    cfg["overwrite_dataset"] = bool(cfg_get(cfg, args.overwrite, "overwrite_dataset", True))
    cfg["vis_count"] = int(cfg_get(cfg, args.vis_count, "vis_count", 0) or 0)
    cfg["vis_stride"] = int(cfg_get(cfg, args.vis_stride, "vis_stride", 1) or 1)
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video", False))
    cfg["num_workers"] = int(cfg_get(cfg, args.num_workers, "num_workers", 1) or 1)
    max_frames = cfg_get(cfg, args.max_frames_per_demo, "max_frames_per_demo")
    max_frames = int(max_frames) if max_frames is not None else None
    ensure_libero_config(cfg.get("libero_config_path"), args.demo_root or cfg.get("demo_root"))

    output_root = Path(
        cfg_get(cfg, args.output_root, "dataset_output_root", Path(__file__).resolve().parents[1] / "data" / "libero_lerobot_dataset")
    ).expanduser().resolve()
    repo_id = cfg_get(cfg, args.repo_id, "dataset_repo_id", "song_libero_pointcloud")
    demo_root = get_libero_dataset_root(args.demo_root, cfg)
    cfg["demo_root"] = str(demo_root)
    vis_dir_value = cfg_get(cfg, args.vis_dir, "vis_dir")
    vis_dir = Path(vis_dir_value).expanduser().resolve() if vis_dir_value else output_root / "visualizations"
    tmp_dir = (
        args.tmp_dir.expanduser().resolve()
        if args.tmp_dir is not None
        else output_root.parent / f".{output_root.name}_tmp"
    )

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[cfg["suite"]]
    suite = suite_cls()
    task_ids = cfg["task_ids"] if cfg["task_ids"] is not None else list(range(len(suite.tasks)))

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
        features=DATASET_FEATURES,
        robot_type="libero",
        root=output_root,
        use_videos=False,
    )
    write_point_cloud_meta(dataset.root)
    write_worldflow_meta(dataset.root)

    summary: dict[str, Any] = {
        "created_unix_s": time.time(),
        "suite": cfg["suite"],
        "demo_root": str(demo_root),
        "output_root": str(output_root),
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "render_camera_names": render_camera_names_from_config(cfg),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_template": str(cfg.get("gripper_template", "reap")),
        "num_workers": int(cfg["num_workers"]),
        "episodes": [],
    }

    jobs: list[dict[str, Any]] = []
    job_index = 0
    for task_id_value in task_ids:
        task_id = int(task_id_value)
        task = suite.get_task(task_id)
        demo_file = find_demo_file(task, demo_root, args.demo_file, cfg)
        demo_names = iter_demo_group_names(demo_file)[: int(cfg["episodes"])]
        if not demo_names:
            raise RuntimeError(f"No usable demos with states found in {demo_file}")
        for converted_for_task, demo_name in enumerate(demo_names):
            jobs.append(
                {
                    "job_index": job_index,
                    "task_id": task_id,
                    "demo_file": str(demo_file),
                    "demo_name": demo_name,
                    "episode_seed": task_id * 100000 + converted_for_task * 1000,
                    "max_frames": max_frames,
                    "cfg": cfg,
                    "tmp_path": str(tmp_dir / f"episode_job_{job_index:06d}"),
                }
            )
            job_index += 1

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if int(cfg["num_workers"]) <= 1:
        for job in tqdm(jobs, desc="Collecting LIBERO episodes", unit="episode"):
            result = collect_episode_worker(job)
            record = save_collected_temp_episode(
                temp_path=Path(result["tmp_path"]),
                dataset=dataset,
                vis_dir=vis_dir,
                cfg=cfg,
            )
            summary["episodes"].append(record)
            print(
                f"[OK] task={record['task_id']} demo={record['demo_name']} frames={record['frames']} "
                f"episode_index={record['episode_index']}"
            )
    else:
        print(f"[info] collecting {len(jobs)} LIBERO episode(s) with {cfg['num_workers']} worker(s)")
        results: dict[int, Path] = {}
        with ProcessPoolExecutor(
            max_workers=int(cfg["num_workers"]),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(collect_episode_worker, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting LIBERO episodes", unit="episode"):
                result = future.result()
                results[int(result["job_index"])] = Path(result["tmp_path"])

        for job in jobs:
            temp_path = results[int(job["job_index"])]
            record = save_collected_temp_episode(
                temp_path=temp_path,
                dataset=dataset,
                vis_dir=vis_dir,
                cfg=cfg,
            )
            summary["episodes"].append(record)
            print(
                f"[OK] task={record['task_id']} demo={record['demo_name']} frames={record['frames']} "
                f"episode_index={record['episode_index']}"
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
