#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def ensure_libero_config(config_path: str | Path | None = None, dataset_root: str | Path | None = None) -> Path:
    """Create LIBERO's config.yaml non-interactively before importing libero.libero."""
    project_root = Path(__file__).resolve().parents[1]
    config_dir = Path(
        config_path
        or os.environ.get("LIBERO_CONFIG_PATH")
        or project_root / "data" / "libero_config"
    ).expanduser().resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    requested_dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    if config_file.exists() and requested_dataset_root is None:
        return config_file

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("LIBERO is not installed. Install it before running LIBERO benchmark scripts.")
    package_root = Path(list(spec.submodule_search_locations)[0]).resolve()
    benchmark_root = package_root / "libero"
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Could not locate LIBERO benchmark root under {package_root}")

    datasets = requested_dataset_root if requested_dataset_root is not None else benchmark_root.parent / "datasets"
    if config_file.exists():
        existing = config_file.read_text(encoding="utf-8")
        if f"datasets: {datasets}" in existing:
            return config_file
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {datasets}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_file


def normalize_camera_name(camera_name: str) -> str:
    camera_name = str(camera_name)
    return camera_name[: -len("_image")] if camera_name.endswith("_image") else camera_name


def pointcloud_camera_names_from_config(cfg: dict[str, Any]) -> list[str]:
    camera_names = cfg.get("pointcloud_camera_names") or cfg.get("camera_names") or []
    camera_names = [normalize_camera_name(camera_name) for camera_name in camera_names]
    if not camera_names:
        raise ValueError("At least one LIBERO point-cloud camera must be configured.")
    return camera_names


def render_camera_names_from_config(cfg: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for camera_name in list(cfg.get("camera_names") or []) + pointcloud_camera_names_from_config(cfg):
        camera_name = normalize_camera_name(camera_name)
        if camera_name not in merged:
            merged.append(camera_name)
    if not merged:
        raise ValueError("At least one LIBERO render camera must be configured.")
    return merged


def sample_or_repeat_points(xyzrgb: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[1] != 6:
        raise ValueError(f"Expected point cloud shape (N, 6), got {xyzrgb.shape}")
    if num_points <= 0 or xyzrgb.shape[0] == num_points:
        return xyzrgb
    if xyzrgb.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    rng = np.random.default_rng(seed)
    replace = xyzrgb.shape[0] < num_points
    indices = rng.choice(xyzrgb.shape[0], num_points, replace=replace)
    return xyzrgb[indices].astype(np.float32, copy=False)


def normalize_gripper_widths(widths: np.ndarray, already_normalized: bool = False) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32).reshape(-1).copy()
    if widths.size == 0:
        return widths
    if already_normalized:
        return np.clip(widths, 0.0, 1.0)

    width_range = widths.max() - widths.min()
    if width_range != 0:
        widths = (widths - widths.min()) / width_range
    else:
        widths[widths >= 0] = 1.0
    return np.clip(widths, 0.0, 1.0)


def gripper_width_percent_from_scalar(width: float, max_physical_width: float = 0.04) -> float:
    width = float(width)
    if not np.isfinite(width):
        return 0.0
    if 0.0 <= width <= 0.08:
        return float(np.clip(width / max(max_physical_width, 1e-6), 0.0, 1.0))
    return float(np.clip(width, 0.0, 1.0))


def allocate_counts(total: int, weights: list[float] | np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    if weights.sum() <= 0:
        counts = np.zeros(len(weights), dtype=np.int64)
        counts[:total] = 1
        return counts

    expected = total * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def box_faces(min_corner: np.ndarray, size: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    sx, sy, sz = size
    x0, y0, z0 = min_corner
    x1, y1, z1 = min_corner + size
    return [
        (sy * sz, np.array([x0, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sy * sz, np.array([x1, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y1, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sy, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
        (sx * sy, np.array([x0, y0, z1]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
    ]


def sample_box_surface(min_corner: np.ndarray, size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    faces = box_faces(np.asarray(min_corner, dtype=np.float64), np.asarray(size, dtype=np.float64))
    counts = allocate_counts(count, [face[0] for face in faces])
    samples = []
    for face_count, (_, origin, axis_a, axis_b) in zip(counts, faces):
        if face_count == 0:
            continue
        uv = rng.random((face_count, 2), dtype=np.float64)
        samples.append(origin + uv[:, :1] * axis_a + uv[:, 1:] * axis_b)
    if not samples:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(samples)


def create_gripper_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    gripper_len: float = 0.06,
    max_width: float = 0.06,
    finger_length: float = 0.08,
    finger_thickness: float = 0.01,
    base_thickness: float = 0.01,
    handle_length: float = 0.05,
) -> np.ndarray:
    width = float(np.clip(width_percent, 0.0, 1.0)) * max_width
    boxes = [
        (
            np.array([width / 2.0, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-width / 2.0 - finger_thickness, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-max_width / 2.0 - finger_thickness / 2.0, 0.0, 0.0]),
            np.array([max_width + finger_thickness, base_thickness, finger_thickness]),
        ),
        (
            np.array([-base_thickness / 2.0, -handle_length, 0.0]),
            np.array([base_thickness, handle_length, base_thickness]),
        ),
    ]

    box_areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    box_counts = allocate_counts(count, box_areas)
    points = []
    for box_count, (min_corner, size) in zip(box_counts, boxes):
        points.append(sample_box_surface(min_corner, size, box_count, rng))
    points = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)

    static_rot = R.from_euler("zyx", [np.pi / 2.0, np.pi / 2.0, 0.0]).as_matrix()
    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    # Keep the REAP gripper template aligned with the real-data HDF5 path:
    # `gripper_len` is a positive length, while the template offset direction
    # is fixed by the end-effector frame convention.
    points = points @ static_rot.T + np.array([0.0, 0.0, -gripper_len])
    points = points @ pose_rot.T + pose[:3]
    return points


def create_panda_gripper_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    max_width: float = 0.08,
    finger_length: float = 0.065,
    finger_thickness: float = 0.012,
    finger_depth: float = 0.018,
    palm_depth: float = 0.05,
) -> np.ndarray:
    """Approximate robosuite PandaGripper geometry in the grip-site frame."""
    width = float(np.clip(width_percent, 0.0, 1.0)) * max_width
    half_width = width / 2.0
    boxes = [
        (
            np.array([half_width, -finger_depth / 2.0, -0.045]),
            np.array([finger_thickness, finger_depth, finger_length]),
        ),
        (
            np.array([-half_width - finger_thickness, -finger_depth / 2.0, -0.045]),
            np.array([finger_thickness, finger_depth, finger_length]),
        ),
        (
            np.array([-max_width / 2.0 - finger_thickness, -0.035, -0.095]),
            np.array([max_width + 2.0 * finger_thickness, 0.07, palm_depth]),
        ),
    ]

    box_areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    box_counts = allocate_counts(count, box_areas)
    points = []
    for box_count, (min_corner, size) in zip(box_counts, boxes):
        points.append(sample_box_surface(min_corner, size, box_count, rng))
    points = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)

    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    points = points @ pose_rot.T + pose[:3]
    return points


def create_gripper_cloud_rgb(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    gripper_len: float,
    gripper_template: str = "reap",
) -> np.ndarray:
    if gripper_template == "panda":
        points = create_panda_gripper_points(width_percent, pose, count, rng)
    else:
        points = create_gripper_points(
            width_percent,
            pose,
            count,
            rng,
            gripper_len=gripper_len,
        )
    colors = np.tile(np.array([[204.0, 51.0, 51.0]], dtype=np.float32), (points.shape[0], 1))
    return np.hstack((points.astype(np.float32), colors))


def merge_cloud_with_gripper(
    original_cloud: np.ndarray,
    gripper_cloud: np.ndarray,
    rng: np.random.Generator,
    drop_strategy: str = "tail",
    shuffle_points: bool = False,
) -> np.ndarray:
    total_points = original_cloud.shape[0]
    gripper_points = min(gripper_cloud.shape[0], total_points)
    keep_points = total_points - gripper_points

    if gripper_points == 0:
        merged = original_cloud.copy()
    elif keep_points == 0:
        idx = rng.choice(gripper_cloud.shape[0], total_points, replace=gripper_cloud.shape[0] < total_points)
        merged = gripper_cloud[idx]
    elif drop_strategy == "tail":
        merged = np.vstack((original_cloud[:keep_points], gripper_cloud[:gripper_points]))
    elif drop_strategy == "near_gripper":
        center = gripper_cloud[:gripper_points, :3].mean(axis=0)
        dist = np.linalg.norm(original_cloud[:, :3] - center, axis=1)
        keep_idx = np.argpartition(dist, gripper_points)[gripper_points:]
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))
    else:
        keep_idx = rng.choice(total_points, keep_points, replace=False)
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))

    if shuffle_points and merged.shape[0] > 1:
        rng.shuffle(merged, axis=0)
    return merged.astype(original_cloud.dtype, copy=False)


def add_local_gripper_cloud_to_point_cloud(
    point_cloud_eff: np.ndarray,
    width_percent: float,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "tail",
    shuffle_points: bool = False,
) -> np.ndarray:
    total_points = int(total_points)
    gripper_points = max(0, min(int(gripper_points), total_points))
    rng = np.random.default_rng(seed)
    point_cloud_eff = sample_or_repeat_points(point_cloud_eff, total_points, seed=seed)
    if gripper_points == 0:
        return point_cloud_eff
    gripper_pose_eff = np.zeros(6, dtype=np.float32)
    gripper_cloud = create_gripper_cloud_rgb(
        width_percent,
        gripper_pose_eff,
        gripper_points,
        rng,
        gripper_len=float(gripper_len),
        gripper_template=str(gripper_template),
    )
    return merge_cloud_with_gripper(
        point_cloud_eff,
        gripper_cloud,
        rng,
        drop_strategy=drop_strategy,
        shuffle_points=shuffle_points,
    )


def add_local_gripper_clouds_to_episode(
    point_clouds_eff: np.ndarray,
    gripper_widths: np.ndarray,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "tail",
    shuffle_points: bool = False,
    widths_are_normalized: bool = False,
) -> np.ndarray:
    point_clouds_eff = np.asarray(point_clouds_eff, dtype=np.float32)
    widths = normalize_gripper_widths(gripper_widths, already_normalized=widths_are_normalized)
    if len(point_clouds_eff) != len(widths):
        raise ValueError(f"Point cloud frames {len(point_clouds_eff)} != gripper widths {len(widths)}")
    merged = np.empty((len(point_clouds_eff), int(total_points), 6), dtype=np.float32)
    for frame_idx, (cloud, width) in enumerate(zip(point_clouds_eff, widths)):
        merged[frame_idx] = add_local_gripper_cloud_to_point_cloud(
            cloud,
            float(width),
            total_points=total_points,
            gripper_points=gripper_points,
            gripper_len=gripper_len,
            gripper_template=gripper_template,
            seed=seed + frame_idx,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged


def rot6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-6, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-6, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pose9_to_homo_np(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    homo = np.zeros((*pose9.shape[:-1], 4, 4), dtype=np.float32)
    homo[..., 3, 3] = 1.0
    homo[..., :3, :3] = rot6d_to_matrix_np(pose9[..., 3:9])
    homo[..., :3, 3] = pose9[..., :3]
    return homo


def pose9_to_traj6_np(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)[..., :9]
    rot = rot6d_to_matrix_np(pose9[..., 3:9])
    flat_rot = rot.reshape(-1, 3, 3)
    euler = R.from_matrix(flat_rot).as_euler("zyx", degrees=False).astype(np.float32)
    euler = euler.reshape(*pose9.shape[:-1], 3)
    return np.concatenate([pose9[..., :3], euler], axis=-1).astype(np.float32)


def fast_inverse_homogeneous(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    out = np.zeros_like(T)
    rot = T[..., :3, :3]
    trans = T[..., :3, 3]
    rot_inv = np.swapaxes(rot, -1, -2)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -(rot_inv @ trans[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def traj6_to_pose9(traj6: np.ndarray) -> np.ndarray:
    traj6 = np.asarray(traj6, dtype=np.float32)
    if traj6.ndim == 1:
        rot = R.from_euler("zyx", traj6[3:6], degrees=False).as_matrix().astype(np.float32)
        return np.concatenate([traj6[:3], rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)
    rot = R.from_euler("zyx", traj6[..., 3:6], degrees=False).as_matrix().astype(np.float32)
    rot6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    return np.concatenate([traj6[..., :3], rot6d], axis=-1).astype(np.float32)


def pose9_from_pos_quat(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    rot = R.from_quat(np.asarray(quat_xyzw, dtype=np.float32)).as_matrix().astype(np.float32)
    rot6d = np.concatenate([rot[:, 0], rot[:, 1]], axis=0)
    return np.concatenate([np.asarray(pos, dtype=np.float32), rot6d, np.asarray([gripper], dtype=np.float32)])


def gripper_scalar(raw_obs: dict[str, Any]) -> float:
    qpos = np.asarray(raw_obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    qpos = np.abs(qpos)  ### absolute pos
    if qpos.size == 0:
        return 0.0
    
    return float(np.sum(qpos))


def eef_pose9_gripper_from_obs(raw_obs: dict[str, Any]) -> np.ndarray:
    return pose9_from_pos_quat(
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32),
        gripper_scalar(raw_obs),
    )


def world_point_cloud_to_current_eff(point_cloud_world: np.ndarray, current_pose9_gripper: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud_world, dtype=np.float32)
    squeeze = pc.ndim == 2
    if squeeze:
        pc = pc[None]
    if pc.ndim != 3 or pc.shape[-1] != 6:
        raise ValueError(f"Expected point_cloud shape (N, 6) or (B, N, 6), got {point_cloud_world.shape}")

    pose = np.asarray(current_pose9_gripper, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None]
    if pose.shape[0] == 1 and pc.shape[0] > 1:
        pose = np.repeat(pose, pc.shape[0], axis=0)
    if pose.shape[0] != pc.shape[0]:
        raise ValueError(f"Pose batch {pose.shape[0]} does not match point cloud batch {pc.shape[0]}.")

    eff_to_world = pose9_to_homo_np(pose)
    world_to_eff = fast_inverse_homogeneous(eff_to_world)
    xyz_h = np.concatenate([pc[..., :3], np.ones((*pc.shape[:2], 1), dtype=np.float32)], axis=-1)
    eff_xyz_h = np.einsum("bij,bnj->bni", world_to_eff, xyz_h)
    pc_eff = np.concatenate([eff_xyz_h[..., :3], pc[..., 3:]], axis=-1).astype(np.float32)
    return pc_eff[0] if squeeze else pc_eff


def add_world_gripper_cloud_to_point_cloud(
    point_cloud_world: np.ndarray,
    current_pose9_gripper: np.ndarray,
    width_percent: float,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "tail",
    shuffle_points: bool = False,
) -> np.ndarray:
    total_points = int(total_points)
    gripper_points = max(0, min(int(gripper_points), total_points))
    rng = np.random.default_rng(seed)
    point_cloud_world = sample_or_repeat_points(point_cloud_world, total_points, seed=seed)
    if gripper_points == 0:
        return world_point_cloud_to_current_eff(point_cloud_world, current_pose9_gripper)

    current_traj6 = pose9_to_traj6_np(np.asarray(current_pose9_gripper, dtype=np.float32))[..., :6]
    gripper_cloud_world = create_gripper_cloud_rgb(
        width_percent,
        current_traj6,
        gripper_points,
        rng,
        gripper_len=float(gripper_len),
        gripper_template=str(gripper_template),
    )
    merged_world = merge_cloud_with_gripper(
        point_cloud_world,
        gripper_cloud_world,
        rng,
        drop_strategy=drop_strategy,
        shuffle_points=shuffle_points,
    )
    return world_point_cloud_to_current_eff(merged_world, current_pose9_gripper)


def add_world_gripper_clouds_to_episode(
    point_clouds_world: np.ndarray,
    current_pose9_grippers: np.ndarray,
    gripper_widths: np.ndarray,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "tail",
    shuffle_points: bool = False,
    widths_are_normalized: bool = False,
) -> np.ndarray:
    point_clouds_world = np.asarray(point_clouds_world, dtype=np.float32)
    current_pose9_grippers = np.asarray(current_pose9_grippers, dtype=np.float32)
    widths = normalize_gripper_widths(gripper_widths, already_normalized=widths_are_normalized)
    if len(point_clouds_world) != len(current_pose9_grippers):
        raise ValueError(
            f"Point cloud frames {len(point_clouds_world)} != pose frames {len(current_pose9_grippers)}"
        )
    if len(point_clouds_world) != len(widths):
        raise ValueError(f"Point cloud frames {len(point_clouds_world)} != gripper widths {len(widths)}")

    merged_eff = np.empty((len(point_clouds_world), int(total_points), 6), dtype=np.float32)
    for frame_idx, (cloud_world, pose9, width) in enumerate(
        zip(point_clouds_world, current_pose9_grippers, widths)
    ):
        merged_eff[frame_idx] = add_world_gripper_cloud_to_point_cloud(
            cloud_world,
            pose9,
            float(width),
            total_points=total_points,
            gripper_points=gripper_points,
            gripper_len=gripper_len,
            gripper_template=gripper_template,
            seed=seed + frame_idx,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged_eff


def backproject_camera(env: Any, raw_obs: dict[str, Any], camera_name: str, height: int, width: int) -> np.ndarray:
    try:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
    except Exception as exc:  # pragma: no cover - optional LIBERO dependency path
        raise RuntimeError("robosuite camera utilities are required for LIBERO point-cloud generation.") from exc

    camera_name = normalize_camera_name(camera_name)
    image_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    if image_key not in raw_obs or depth_key not in raw_obs:
        available = ", ".join(sorted(raw_obs.keys()))
        raise KeyError(
            f"LIBERO observation is missing {image_key!r} or {depth_key!r}. "
            f"Available keys: {available}. Ensure OffScreenRenderEnv was created with camera_depths=True."
        )

    rgb = np.asarray(raw_obs[image_key], dtype=np.float32)
    depth = np.asarray(raw_obs[depth_key])
    depth = get_real_depth_map(env.sim, depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = depth.reshape(height, width)
    vv, uu = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, height, width).astype(np.float32)
    camera_to_world = get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32)
    z = depth
    x = (uu - intrinsic[0, 2]) * z / intrinsic[0, 0]
    # robosuite image arrays are indexed top-to-bottom, while the camera projection
    # matrix uses the image-plane v coordinate. Flip rows before camera->world.
    pixel_v = float(height - 1) - vv
    y = (pixel_v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points_h = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    world = (camera_to_world @ camera_points_h.T).T[:, :3]

    colors = rgb.reshape(-1, 3)
    valid = np.isfinite(world).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    return np.concatenate([world[valid], colors[valid]], axis=1).astype(np.float32, copy=False)


def observation_to_point_clouds(
    env: Any,
    raw_obs: dict[str, Any],
    camera_names: list[str],
    height: int,
    width: int,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clouds = [backproject_camera(env, raw_obs, camera_name, height, width) for camera_name in camera_names]
    point_cloud_world = np.concatenate(clouds, axis=0)
    eef_pose = eef_pose9_gripper_from_obs(raw_obs)
    point_cloud_eff = world_point_cloud_to_current_eff(point_cloud_world, eef_pose)
    point_cloud_eff = sample_or_repeat_points(point_cloud_eff, num_points, seed=seed)
    point_cloud_world = sample_or_repeat_points(point_cloud_world, num_points, seed=seed)
    return point_cloud_eff, point_cloud_world, eef_pose


def observation_to_world_point_cloud(
    env: Any,
    raw_obs: dict[str, Any],
    camera_names: list[str],
    height: int,
    width: int,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    clouds = [backproject_camera(env, raw_obs, camera_name, height, width) for camera_name in camera_names]
    point_cloud_world = np.concatenate(clouds, axis=0)
    eef_pose = eef_pose9_gripper_from_obs(raw_obs)
    point_cloud_world = sample_or_repeat_points(point_cloud_world, num_points, seed=seed)
    return point_cloud_world, eef_pose


def action_pose9_to_libero(action_pose9_gripper: np.ndarray, trans_scale: float, rot_scale: float, gripper_threshold: float) -> np.ndarray:
    action = np.asarray(action_pose9_gripper, dtype=np.float32).reshape(-1)
    trans = np.clip(action[:3] / max(trans_scale, 1e-6), -1.0, 1.0)
    rot = rot6d_to_matrix_np(action[3:9])
    rotvec = R.from_matrix(rot).as_rotvec().astype(np.float32)
    rotvec = np.clip(rotvec / max(rot_scale, 1e-6), -1.0, 1.0)
    gripper = 1.0 if float(action[-1]) >= gripper_threshold else -1.0
    return np.concatenate([trans, rotvec, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)


def get_task_init_states(task_suite: Any, task_id: int) -> np.ndarray:
    from libero.libero import get_libero_path

    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[task_id].problem_folder
        / task_suite.tasks[task_id].init_states_file
    )
    return torch.load(init_states_path, weights_only=False)


def make_libero_env(
    task_suite: Any,
    task_id: int,
    height: int,
    width: int,
    camera_names: list[str],
):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    cameras = [normalize_camera_name(camera_name) for camera_name in camera_names]
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=height,
        camera_widths=width,
        camera_names=cameras,
        camera_depths=True,
    )
    env.reset()
    for robot in env.robots:
        robot.controller.use_delta = True
    return env, task
