#!/usr/bin/env python3
"""Convert the cfg7 RH20T recording to the current LeRobot format.

This converter intentionally handles only the RH20T real-robot layout.  It
does not replay scenes, contact a simulator, or modify the source recording.
The cfg7 camera selection is fixed by scripts/camera.md.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_LEROBOT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.song_pointseg import (  # noqa: E402
    save_episode_point_clouds_zarr,
)


CFG = 7
CAMERA_SERIAL = "cam_037522061512"
IMAGE_KEY = "observation.images.front"
IMAGE_SHAPE = (256, 256, 3)
ACTION_NAMES = ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]
SYNC_TOLERANCE_MS = 15.0
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 5.0
NUM_POINTS = 10000


@dataclass
class SceneConversion:
    scene: Path
    task: str
    images: list[np.ndarray]
    states: np.ndarray
    actions: np.ndarray
    timestamps_s: np.ndarray
    point_clouds: np.ndarray
    actual_tcp_base: np.ndarray
    command_raw: np.ndarray
    command_indices: np.ndarray
    command_errors_ms: np.ndarray
    depth_indices: np.ndarray
    depth_errors_ms: np.ndarray
    state_interpolated: np.ndarray
    gripper_raw: np.ndarray
    point_is_pad: np.ndarray
    point_source_index: np.ndarray
    first_eef0: np.ndarray
    calibration: dict[str, Any]
    sync_records: list[dict[str, Any]]
    source_frame_count: int


def load_npy(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    if value.shape == () and value.dtype == object:
        return value.item()
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def infer_timestamp_unit(values: np.ndarray) -> tuple[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        raise ValueError("At least two finite timestamps are required.")
    magnitude = float(np.median(np.abs(finite)))
    if magnitude >= 1e17:
        return "ns", 1e-9
    if magnitude >= 1e14:
        return "us", 1e-6
    if magnitude >= 1e11:
        return "ms", 1e-3
    if magnitude < 1e8:
        return "s", 1.0
    raise ValueError(f"Timestamp unit is ambiguous for values around {magnitude:g}.")


def normalize_timestamps(values: Any, label: str) -> tuple[np.ndarray, str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    unit, scale = infer_timestamp_unit(values)
    timestamps = values * scale
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{label} timestamps are not finite and strictly increasing.")
    return timestamps, unit, scale


def video_info(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata for {path}: {count=} {fps=} {width=} {height=}")
    return count, fps, width, height


def read_video_frame(cap: cv2.VideoCapture, expected_index: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {expected_index}.")
    return frame


def unpack_depth(depth_frame: np.ndarray, color_shape: tuple[int, int], *, multiplier: int) -> np.ndarray:
    color_height, color_width = color_shape
    if depth_frame.ndim == 3:
        depth_frame = depth_frame[:, :, 0]
    depth_height, depth_width = depth_frame.shape
    if (depth_height, depth_width) == (color_height * 2, color_width):
        low_byte = depth_frame[:color_height].astype(np.uint16)
        high_byte = depth_frame[color_height:].astype(np.uint16)
        depth_mm = low_byte | (high_byte << 8)
    elif (depth_height, depth_width) == (color_height, color_width):
        depth_mm = depth_frame.astype(np.uint16)
    else:
        raise ValueError(
            f"Unsupported RGB-D sizes: color={color_width}x{color_height}, "
            f"depth={depth_width}x{depth_height}"
        )
    if multiplier != 1:
        depth_mm = (depth_mm.astype(np.uint32) * multiplier).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm


def rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def command_rotation_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    return rx(float(euler_xyz[0])) @ ry(float(euler_xyz[1])) @ rz(float(euler_xyz[2]))


def rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Invalid wxyz quaternion: {quaternion}")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def pose_to_10d(transform: np.ndarray, gripper_width_m: float) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3].astype(np.float32), matrix_to_rot6d(transform[:3, :3]), [float(gripper_width_m)]]
    ).astype(np.float32)


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def interpolate_pose(records: list[dict[str, Any]], timestamp_s: float, tolerance_s: float) -> tuple[np.ndarray, bool, float]:
    timestamps = np.asarray([float(record["timestamp"]) for record in records], dtype=np.float64)
    positions = np.asarray([record["tcp"][:3] for record in records], dtype=np.float64)
    rotations = np.asarray([rotation_from_wxyz(record["tcp"][3:]) for record in records], dtype=np.float64)
    right = int(np.searchsorted(timestamps, timestamp_s, side="left"))
    if right == 0:
        error = abs(timestamp_s - timestamps[0])
        if error > tolerance_s:
            raise ValueError(f"TCP timestamp is outside tolerance: {error * 1000:.3f} ms")
        return make_transform(positions[0], rotations[0]), False, error
    if right >= len(timestamps):
        error = abs(timestamp_s - timestamps[-1])
        if error > tolerance_s:
            raise ValueError(f"TCP timestamp is outside tolerance: {error * 1000:.3f} ms")
        return make_transform(positions[-1], rotations[-1]), False, error
    left = right - 1
    left_error = abs(timestamp_s - timestamps[left])
    right_error = abs(timestamps[right] - timestamp_s)
    if left_error <= 1e-12:
        return make_transform(positions[left], rotations[left]), False, left_error
    if right_error <= 1e-12:
        return make_transform(positions[right], rotations[right]), False, right_error
    if min(left_error, right_error) > tolerance_s:
        raise ValueError(f"TCP timestamp is outside tolerance: {min(left_error, right_error) * 1000:.3f} ms")
    alpha = (timestamp_s - timestamps[left]) / (timestamps[right] - timestamps[left])
    position = positions[left] * (1.0 - alpha) + positions[right] * alpha
    # Quaternion SLERP without depending on scipy, using the matrix logarithm-free form.
    q0 = matrix_to_xyzw(rotations[left])
    q1 = matrix_to_xyzw(rotations[right])
    q = slerp_xyzw(q0, q1, float(alpha))
    return make_transform(position, xyzw_to_matrix(q)), True, min(left_error, right_error)


def matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return np.asarray([x, y, z, w], dtype=np.float64)


def xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / sin_theta


def nearest_index(values: np.ndarray, query: float) -> tuple[int, float]:
    index = int(np.searchsorted(values, query, side="left"))
    candidates = []
    if index < len(values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(float(values[item]) - query))
    return best, abs(float(values[best]) - query)


def parse_scene_name(scene: Path) -> dict[str, str]:
    match = re.match(r"task_(\d+)_user_(\d+)_scene_(\d+)_cfg_(\d+)$", scene.name)
    if match is None:
        raise ValueError(f"Unexpected RH20T scene name: {scene.name}")
    return {
        "task_id": match.group(1),
        "user_id": match.group(2),
        "scene_id": match.group(3),
        "cfg": match.group(4),
    }


def discover_scenes(input_root: Path) -> list[Path]:
    cfg_root = input_root / f"RH20T_cfg{CFG}"
    if not cfg_root.is_dir():
        raise FileNotFoundError(f"cfg7 root is missing: {cfg_root}")
    scenes = sorted(
        path
        for path in cfg_root.glob("task_*_user_*_scene_*_cfg_*")
        if path.is_dir() and not path.name.endswith("_human")
    )
    for scene in scenes:
        parsed = parse_scene_name(scene)
        if int(parsed["cfg"]) != CFG:
            raise ValueError(f"Non-cfg7 scene matched: {scene}")
    return scenes


def calibration_for_scene(cfg_root: Path, metadata: dict[str, Any], serial: str) -> dict[str, Any]:
    calib_id = str(metadata["calib"])
    calib_root = cfg_root / "calib" / calib_id
    intrinsics = load_npy(calib_root / "intrinsics.npy")
    extrinsics = load_npy(calib_root / "extrinsics.npy")
    if serial not in intrinsics or serial not in extrinsics:
        raise KeyError(f"Calibration {calib_id} does not contain camera serial {serial}.")
    intrinsic = np.asarray(intrinsics[serial], dtype=np.float64)
    extrinsic_value = extrinsics[serial]
    extrinsic = np.asarray(extrinsic_value[0] if isinstance(extrinsic_value, list) else extrinsic_value, dtype=np.float64)
    if intrinsic.shape[0] < 3 or intrinsic.shape[1] < 3:
        raise ValueError(f"Invalid intrinsics for {serial}: {intrinsic.shape}")
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError(f"Invalid extrinsics for {serial}: {extrinsic.shape}")
    if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"Extrinsic bottom row is invalid for {serial}: {extrinsic[3]}")
    return {
        "calibration_id": calib_id,
        "calibration_root": str(calib_root),
        "intrinsic": intrinsic,
        "stored_extrinsic": extrinsic,
        "transform_direction": "stored_world_to_camera; inverse_used_as_base_from_camera",
    }


def resolve_actual_records(scene: Path, serial: str) -> list[dict[str, Any]]:
    values = load_npy(scene / "transformed" / "tcp_base.npy")
    key = serial.removeprefix("cam_")
    if key not in values:
        raise KeyError(f"TCP-base data is missing camera/device {key}.")
    records = list(values[key])
    if not records:
        raise ValueError(f"TCP-base data is empty for {key}.")
    records.sort(key=lambda record: int(record["timestamp"]))
    return records


def resolve_gripper_records(scene: Path, serial: str) -> dict[int, dict[str, Any]]:
    values = load_npy(scene / "transformed" / "gripper.npy")
    key = serial.removeprefix("cam_")
    if key not in values:
        raise KeyError(f"Gripper data is missing camera/device {key}.")
    records = values[key]
    if not records:
        raise ValueError(f"Gripper data is empty for {key}.")
    return {int(timestamp): value for timestamp, value in records.items()}


def parse_command(scene: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(load_npy(scene / "robot_command" / "tcpcommand_timestamp.npy"), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 7:
        raise ValueError(f"Expected command array (N,7+), got {raw.shape}")
    timestamps = raw[:, -1]
    unit, scale = infer_timestamp_unit(timestamps)
    timestamps_s = timestamps * scale
    if np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("Command timestamps are not strictly increasing.")
    command = raw[:, :6].copy()
    command[:, :3] /= 1000.0
    return np.column_stack([command, timestamps_s]), raw


def make_command_transform(command_row: np.ndarray) -> np.ndarray:
    position_m = np.asarray(command_row[:3], dtype=np.float64)
    rotation = command_rotation_xyz(command_row[3:6])
    return make_transform(position_m, rotation)


def letterbox_rgb(image_rgb: np.ndarray, size: int = 256) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    output[top : top + new_height, left : left + new_width] = resized
    return output


def project_point_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    base_from_camera: np.ndarray,
    current_eef_from_base: np.ndarray,
    frame_index: int,
    episode_index: int,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_height, color_width = color_bgr.shape[:2]
    depth_height, depth_width = depth_mm.shape
    if (color_height, color_width) != (depth_height, depth_width):
        raise ValueError(f"RGB/depth projection shape mismatch: {color_bgr.shape} vs {depth_mm.shape}")
    k = np.asarray(intrinsic[:3, :3], dtype=np.float64).copy()
    k[0] *= depth_width / 640.0
    k[1] *= depth_height / 360.0
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if min(fx, fy) <= 0:
        raise ValueError(f"Invalid scaled intrinsics: {k}")
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M)
    v, u = np.nonzero(valid)
    if len(v) == 0:
        raise ValueError(f"No valid depth points at frame {frame_index}.")
    rng = np.random.default_rng(seed + episode_index * 1_000_003 + frame_index)
    if len(v) >= num_points:
        selected = np.sort(rng.choice(len(v), size=num_points, replace=False))
        sample_v, sample_u = v[selected], u[selected]
        is_pad = np.zeros(num_points, dtype=bool)
    else:
        pad_indices = rng.choice(len(v), size=num_points - len(v), replace=True)
        sample_v = np.concatenate([v, v[pad_indices]])
        sample_u = np.concatenate([u, u[pad_indices]])
        is_pad = np.zeros(num_points, dtype=bool)
        is_pad[len(v):] = True
    z = depth_m[sample_v, sample_u]
    x = (sample_u.astype(np.float32) - float(cx)) * z / float(fx)
    y = (sample_v.astype(np.float32) - float(cy)) * z / float(fy)
    camera_xyz = np.column_stack([x, y, z]).astype(np.float64)
    base_xyz = (base_from_camera[:3, :3] @ camera_xyz.T).T + base_from_camera[:3, 3]
    eef_xyz = (current_eef_from_base[:3, :3] @ base_xyz.T).T + current_eef_from_base[:3, 3]
    colors = color_bgr[sample_v, sample_u][:, ::-1].astype(np.uint8)
    source = (sample_v * depth_width + sample_u).astype(np.int64)
    cloud = np.concatenate([eef_xyz.astype(np.float32), colors.astype(np.float32)], axis=-1)
    if not np.isfinite(cloud[:, :3]).all() or np.allclose(cloud[:, :3], 0):
        raise ValueError(f"Invalid point cloud at frame {frame_index}.")
    return cloud.astype(np.float32), is_pad, source


def write_binary_ply(path: Path, points: np.ndarray, scores: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(points[:, 3:6], 0, 255).astype(np.uint8)
    scores = np.zeros(len(points), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("foreground_score", "<f4"),
        ]
    )
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    data["foreground_score"] = scores
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float foreground_score\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        data.tofile(f)


def nearest_record(records: dict[int, dict[str, Any]], timestamp_raw: int) -> tuple[dict[str, Any], float]:
    keys = np.asarray(sorted(records), dtype=np.int64)
    index, error = nearest_index(keys.astype(np.float64), float(timestamp_raw))
    return records[int(keys[index])], error


def resize_depth_intrinsic(intrinsic: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    height, width = depth_shape
    k = np.asarray(intrinsic[:3, :3], dtype=np.float64).copy()
    k[0] *= width / 640.0
    k[1] *= height / 360.0
    return k


def convert_scene(
    scene: Path,
    cfg_root: Path,
    episode_index: int,
    args: argparse.Namespace,
) -> SceneConversion:
    parsed = parse_scene_name(scene)
    metadata = read_json(scene / "metadata.json")
    camera = scene / CAMERA_SERIAL
    if not camera.is_dir():
        raise FileNotFoundError(f"Required cfg7 camera is missing: {camera}")
    color_path, depth_path = camera / "color.mp4", camera / "depth.mp4"
    if not color_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Required RGB-D video is missing under {camera}")

    color_count, color_fps, color_width, color_height = video_info(color_path)
    depth_count, depth_fps, depth_width, depth_height = video_info(depth_path)
    camera_timestamps = load_npy(camera / "timestamps.npy")
    color_raw = np.asarray(camera_timestamps["color"], dtype=np.float64)
    depth_raw = np.asarray(camera_timestamps["depth"], dtype=np.float64)
    color_ts, timestamp_unit, timestamp_scale = normalize_timestamps(color_raw, "RGB")
    depth_ts, depth_unit, _ = normalize_timestamps(depth_raw, "depth")
    if color_count != len(color_ts) or depth_count != len(depth_ts):
        raise ValueError(
            f"Video/timestamp count mismatch: color video={color_count}, timestamps={len(color_ts)}, "
            f"depth video={depth_count}, timestamps={len(depth_ts)}"
        )
    if abs(color_fps - depth_fps) > 0.2 or color_count != depth_count:
        raise ValueError(f"RGB/depth video mismatch: {(color_count, color_fps)} vs {(depth_count, depth_fps)}")
    if not np.allclose(color_ts, depth_ts, atol=SYNC_TOLERANCE_MS / 1000.0):
        raise ValueError("RGB/depth timestamp streams are not aligned within 15 ms.")

    calibration = calibration_for_scene(cfg_root, metadata, CAMERA_SERIAL.removeprefix("cam_"))
    stored_extrinsic = calibration["stored_extrinsic"]
    base_from_camera = inverse_transform(stored_extrinsic)
    actual_records = resolve_actual_records(scene, CAMERA_SERIAL)
    actual_raw_ts = np.asarray([int(record["timestamp"]) for record in actual_records], dtype=np.float64)
    actual_unit, actual_scale = infer_timestamp_unit(actual_raw_ts)
    actual_records = [dict(record, timestamp=float(record["timestamp"]) * actual_scale) for record in actual_records]
    command, command_raw = parse_command(scene)
    command_ts = command[:, -1]
    command_match_errors = np.asarray([nearest_index(command_ts, float(timestamp))[1] for timestamp in color_ts], dtype=np.float64)
    valid_frame_indices = np.flatnonzero(command_match_errors <= args.sync_tolerance_ms / 1000.0)
    if args.max_frames is not None:
        valid_frame_indices = valid_frame_indices[: args.max_frames]
    if len(valid_frame_indices) == 0:
        raise ValueError("No RGB frames overlap the command timestamp range within the sync tolerance.")
    valid_frame_set = set(int(index) for index in valid_frame_indices)
    gripper_records = resolve_gripper_records(scene, CAMERA_SERIAL)
    gripper_keys = np.asarray(sorted(gripper_records), dtype=np.float64) * timestamp_scale
    gripper_by_seconds = {float(key * timestamp_scale): value for key, value in gripper_records.items()}

    first_actual = actual_records[0]
    first_tcp_base = make_transform(first_actual["tcp"][:3], rotation_from_wxyz(first_actual["tcp"][3:]))
    base_to_eef0 = inverse_transform(first_tcp_base)
    eef0_poses = []
    states = []
    actions = []
    images = []
    clouds = []
    pad_masks = []
    source_indices = []
    command_indices = []
    command_errors_ms = []
    depth_indices = []
    depth_errors_ms = []
    state_interpolated = []
    gripper_raw_values = []
    sync_records = []

    color_cap = cv2.VideoCapture(str(color_path))
    depth_cap = cv2.VideoCapture(str(depth_path))
    if not color_cap.isOpened() or not depth_cap.isOpened():
        raise RuntimeError(f"Could not open selected RGB-D camera: {camera}")
    try:
        for frame_index in range(int(valid_frame_indices[-1]) + 1):
            color_bgr = read_video_frame(color_cap, frame_index)
            depth_frame = read_video_frame(depth_cap, frame_index)
            if frame_index not in valid_frame_set:
                continue
            timestamp_s = color_ts[frame_index]
            if color_bgr.shape[:2] != (color_height, color_width):
                raise ValueError(f"Unexpected RGB frame shape: {color_bgr.shape}")
            depth_mm = unpack_depth(
                depth_frame,
                (color_height, color_width),
                multiplier=4 if CAMERA_SERIAL.startswith("cam_f") else 1,
            )
            depth_index, depth_error_s = nearest_index(depth_ts, float(timestamp_s))
            actual_pose_base, interpolated, actual_error_s = interpolate_pose(
                actual_records, float(timestamp_s), SYNC_TOLERANCE_MS / 1000.0
            )
            command_index, command_error_s = nearest_index(command_ts, float(timestamp_s))
            if command_error_s > SYNC_TOLERANCE_MS / 1000.0:
                raise ValueError(f"Command synchronization exceeds 15 ms at frame {frame_index}: {command_error_s * 1000:.3f} ms")
            command_base = make_command_transform(command[command_index])
            actual_eef0 = base_to_eef0 @ actual_pose_base
            command_eef0 = base_to_eef0 @ command_base
            gripper_record, gripper_error_raw = nearest_record(gripper_records, int(color_raw[frame_index]))
            if gripper_error_raw * timestamp_scale > SYNC_TOLERANCE_MS / 1000.0:
                raise ValueError(f"Gripper synchronization exceeds 15 ms at frame {frame_index}.")
            info = gripper_record.get("gripper_info", [])
            if len(info) == 0:
                raise ValueError(f"gripper_info is empty at frame {frame_index}.")
            width_raw = float(info[0])
            width_m = width_raw / 1000.0
            if not np.isfinite(width_m) or not (-0.001 <= width_m <= 0.2):
                raise ValueError(f"Invalid gripper width at frame {frame_index}: raw={width_raw}")
            actual_eef0_10d = pose_to_10d(actual_eef0, width_m)
            command_eef0_10d = pose_to_10d(command_eef0, width_m)
            cloud, is_pad, source = project_point_cloud(
                color_bgr,
                depth_mm,
                calibration["intrinsic"],
                base_from_camera,
                inverse_transform(actual_pose_base),
                frame_index,
                episode_index,
                int(args.num_points),
                int(args.point_seed),
            )
            images.append(letterbox_rgb(color_bgr[:, :, ::-1]))
            states.append(actual_eef0_10d)
            actions.append(command_eef0_10d)
            clouds.append(cloud)
            pad_masks.append(is_pad)
            source_indices.append(source)
            command_indices.append(command_index)
            command_errors_ms.append(command_error_s * 1000.0)
            depth_indices.append(depth_index)
            depth_errors_ms.append(depth_error_s * 1000.0)
            state_interpolated.append(interpolated)
            gripper_raw_values.append(width_raw)
            eef0_poses.append(actual_eef0)
            sync_records.append(
                {
                    "frame_index": frame_index,
                    "rgb_timestamp_raw": int(color_raw[frame_index]),
                    "timestamp_s": float(timestamp_s),
                    "depth_index": int(depth_index),
                    "depth_error_ms": float(depth_error_s * 1000.0),
                    "command_index": int(command_index),
                    "command_error_ms": float(command_error_s * 1000.0),
                    "tcp_interpolated": bool(interpolated),
                    "tcp_error_ms": float(actual_error_s * 1000.0),
                    "gripper_raw_width": width_raw,
                    "point_count": int(np.count_nonzero(~is_pad)),
                    "point_pad_count": int(np.count_nonzero(is_pad)),
                }
            )
    finally:
        color_cap.release()
        depth_cap.release()

    return SceneConversion(
        scene=scene,
        task=f"task_{parsed['task_id']}",
        images=images,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        timestamps_s=np.asarray(color_ts[valid_frame_indices], dtype=np.float32),
        point_clouds=np.asarray(clouds, dtype=np.float32),
        actual_tcp_base=np.asarray(eef0_poses, dtype=np.float64),
        command_raw=command_raw,
        command_indices=np.asarray(command_indices, dtype=np.int64),
        command_errors_ms=np.asarray(command_errors_ms, dtype=np.float32),
        depth_indices=np.asarray(depth_indices, dtype=np.int64),
        depth_errors_ms=np.asarray(depth_errors_ms, dtype=np.float32),
        state_interpolated=np.asarray(state_interpolated, dtype=bool),
        gripper_raw=np.asarray(gripper_raw_values, dtype=np.float32),
        point_is_pad=np.asarray(pad_masks, dtype=bool),
        point_source_index=np.asarray(source_indices, dtype=np.int64),
        first_eef0=base_to_eef0,
        calibration={
            "calibration_id": calibration["calibration_id"],
            "calibration_root": calibration["calibration_root"],
            "camera_serial": CAMERA_SERIAL,
            "camera_mount_type": "fixed_external",
            "stored_extrinsic": calibration["stored_extrinsic"],
            "transform_direction": calibration["transform_direction"],
            "intrinsic": calibration["intrinsic"],
            "depth_encoding": "stacked_low_byte_then_high_byte",
            "depth_unit": "millimeter",
            "depth_scale": 1000.0,
            "timestamp_unit": timestamp_unit,
            "actual_timestamp_unit": actual_unit,
            "command_timestamp_unit": infer_timestamp_unit(command_raw[:, -1])[0],
            "command_position_unit": "millimeter_to_meter",
            "command_rotation": "xyz_euler_radians",
            "actual_quaternion_order": "wxyz",
        },
        sync_records=sync_records,
        source_frame_count=color_count,
    )


def convert_scene_worker(payload: tuple[Path, Path, int, argparse.Namespace]) -> tuple[bool, SceneConversion | None, str | None]:
    scene, cfg_root, episode_index, args = payload
    try:
        return True, convert_scene(scene, cfg_root, episode_index, args), None
    except Exception:
        return False, None, traceback.format_exc()


def iter_converted_scenes(scenes: list[Path], cfg_root: Path, args: argparse.Namespace):
    if int(args.workers) <= 1:
        for scene_index, scene in enumerate(scenes):
            try:
                yield scene_index, scene, convert_scene(scene, cfg_root, scene_index, args), None
            except Exception:
                yield scene_index, scene, None, traceback.format_exc()
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(int(args.workers), len(scenes)), mp_context=context) as executor:
        pending = {}
        next_submit = 0
        for _ in range(min(int(args.workers), len(scenes))):
            pending[next_submit] = executor.submit(
                convert_scene_worker, (scenes[next_submit], cfg_root, next_submit, args)
            )
            next_submit += 1
        for scene_index, scene in enumerate(scenes):
            future = pending.pop(scene_index)
            ok, converted, error = future.result()
            yield scene_index, scene, converted, error if not ok else None
            if next_submit < len(scenes):
                pending[next_submit] = executor.submit(
                    convert_scene_worker, (scenes[next_submit], cfg_root, next_submit, args)
                )
                next_submit += 1



def save_scene_artifacts(output_root: Path, episode_index: int, converted: SceneConversion, *, save_pointcloud: bool) -> None:
    for name in ("raw_actions", "raw_actions_full", "raw_tcp", "world_ee_poses", "initial_states", "point_clouds", "calibration_refs", "visualizations"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    arrays = output_root / "raw_actions"
    arrays.mkdir(parents=True, exist_ok=True)
    np.save(arrays / f"episode_{episode_index:06d}.npy", converted.actions)
    np.save(output_root / "raw_actions_full" / f"episode_{episode_index:06d}.npy", converted.command_raw)
    np.save(output_root / "raw_tcp" / f"episode_{episode_index:06d}.npy", converted.actual_tcp_base)
    np.save(output_root / "world_ee_poses" / f"episode_{episode_index:06d}.npy", converted.actual_tcp_base)
    np.save(output_root / "initial_states" / f"episode_{episode_index:06d}.npy", converted.states[0])
    np.save(output_root / "point_clouds" / f"episode_{episode_index:06d}_is_pad.npy", converted.point_is_pad)
    np.save(output_root / "point_clouds" / f"episode_{episode_index:06d}_source_index.npy", converted.point_source_index)
    write_json(output_root / "calibration_refs" / f"episode_{episode_index:06d}.json", converted.calibration)
    if save_pointcloud:
        point_dir = output_root / "point_clouds"
        temporary = point_dir / f".episode_{episode_index:06d}.zarr.tmp"
        final = point_dir / f"episode_{episode_index:06d}.zarr"
        if temporary.exists():
            shutil.rmtree(temporary)
        save_episode_point_clouds_zarr(point_dir, episode_index, converted.point_clouds, packed=True)
        if not final.exists():
            raise RuntimeError(f"Point cloud writer did not create {final}")


def write_scene_visualizations(output_root: Path, episode_index: int, converted: SceneConversion) -> None:
    for label, index in (("first", 0), ("middle", len(converted.images) // 2), ("last", len(converted.images) - 1)):
        write_binary_ply(output_root / "visualizations" / f"episode_{episode_index:06d}_{label}.ply", converted.point_clouds[index])


def feature_spec() -> dict[str, Any]:
    return {
        "action": {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES},
        IMAGE_KEY: {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]},
    }


def append_episode(dataset: LeRobotDataset, converted: SceneConversion) -> None:
    episode_buffer = dataset.create_episode_buffer()
    frame_count = len(converted.images)
    episode_index = int(episode_buffer["episode_index"])
    image_paths = []
    for frame_index, image in enumerate(converted.images):
        image_path = dataset._get_image_file_path(episode_index, IMAGE_KEY, frame_index)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        dataset._save_image(image, image_path, compress_level=6)
        image_paths.append(str(image_path))
    episode_buffer["size"] = frame_count
    episode_buffer["task"] = [converted.task] * frame_count
    episode_buffer["frame_index"] = np.arange(frame_count, dtype=np.int64)
    episode_buffer["timestamp"] = np.asarray(converted.timestamps_s, dtype=np.float32)
    episode_buffer["action"] = np.asarray(converted.actions, dtype=np.float32)
    episode_buffer["observation.state"] = np.asarray(converted.states, dtype=np.float32)
    episode_buffer[IMAGE_KEY] = image_paths
    dataset.save_episode(episode_data=episode_buffer, parallel_encoding=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cfgs", default="7", help="Only cfg7 is supported by this entry point.")
    parser.add_argument("--camera-serial", default=CAMERA_SERIAL)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scene", default=None, help="Convert one scene basename for smoke testing.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit scene count for smoke tests.")
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--point-seed", type=int, default=1000)
    parser.add_argument("--sync-tolerance-ms", type=float, default=SYNC_TOLERANCE_MS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-pointcloud", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[1] / "lerobot_dataset" / stamp


def run_dry_run(input_root: Path, scenes: list[Path]) -> None:
    by_task: dict[str, int] = {}
    for scene in scenes:
        parsed = parse_scene_name(scene)
        by_task[parsed["task_id"]] = by_task.get(parsed["task_id"], 0) + 1
    print(json.dumps({"cfg": CFG, "camera_serial": CAMERA_SERIAL, "robot_scenes": len(scenes), "tasks": by_task}, indent=2))
    for scene in scenes[:5]:
        camera = scene / CAMERA_SERIAL
        if not camera.is_dir():
            print(json.dumps({"scene": scene.name, "status": "missing_camera"}))
            continue
        color = video_info(camera / "color.mp4")
        depth = video_info(camera / "depth.mp4")
        print(json.dumps({"scene": scene.name, "color": color, "depth": depth, "status": "ready"}))
    if len(scenes) > 5:
        print(f"... {len(scenes) - 5} additional scenes omitted from dry-run detail")


def main() -> None:
    args = parse_args()
    if str(args.cfgs).replace("cfg", "") not in {"7", "7,"}:
        raise ValueError("This converter intentionally supports cfg7 only.")
    if args.camera_serial != CAMERA_SERIAL:
        raise ValueError(f"cfg7 camera is fixed by camera.md: use {CAMERA_SERIAL}")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    input_root = args.input_root.expanduser().resolve()
    cfg_root = input_root / f"RH20T_cfg{CFG}"
    scenes = discover_scenes(input_root)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if args.scene:
        scenes = [scene for scene in scenes if scene.name == args.scene]
        if not scenes:
            raise FileNotFoundError(f"Requested scene is not a cfg7 robot scene: {args.scene}")
    if args.dry_run:
        run_dry_run(input_root, scenes)
        return

    output_root = (args.output_root or default_output_root()).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; choose a new timestamp or use --overwrite")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    records = []
    started = time.time()
    try:
        for episode_index, scene, converted, worker_error in tqdm(iter_converted_scenes(scenes, cfg_root, args), total=len(scenes), desc="RH20T cfg7", unit="scene"):
            try:
                if worker_error:
                    raise RuntimeError(worker_error)
                if dataset is None:
                    fps = max(1, int(round(video_info(scene / CAMERA_SERIAL / "color.mp4")[1])))

                    dataset = LeRobotDataset.create(
                        repo_id=f"rh20t_cfg7_{output_root.name}",
                        fps=fps,
                        features=feature_spec(),
                        root=output_root,
                        robot_type="Kuka",
                        use_videos=False,
                    )
                    (output_root / "failed_scenes.jsonl").touch()
                    (output_root / "synchronization_report.jsonl").touch()
                saved_episode_index = int(dataset.meta.total_episodes)
                append_episode(dataset, converted)
                save_scene_artifacts(output_root, saved_episode_index, converted, save_pointcloud=not args.skip_pointcloud)
                if not args.no_visualizations:
                    write_scene_visualizations(output_root, saved_episode_index, converted)
                for sync in converted.sync_records:
                    sync["scene"] = scene.name
                    sync["episode_index"] = saved_episode_index
                    with (output_root / "synchronization_report.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(sync, ensure_ascii=False) + "\n")
                record = {
                    "episode_index": saved_episode_index,
                    "scene": scene.name,
                    "task": converted.task,
                    "source_frame_count": converted.source_frame_count,
                    "output_frame_count": len(converted.images),
                    "camera_serial": CAMERA_SERIAL,
                    "command_action": True,
                    "action_alignment": "observation/current_command",
                    "command_error_ms_max": float(converted.command_errors_ms.max(initial=0.0)),
                    "depth_error_ms_max": float(converted.depth_errors_ms.max(initial=0.0)),
                    "state_interpolated": int(converted.state_interpolated.sum()),
                    "point_pad_fraction": float(converted.point_is_pad.mean()),
                    "calibration": converted.calibration,
                }
                records.append(record)
                write_json(output_root / "meta" / "rh20t_episodes" / f"episode_{saved_episode_index:06d}.json", record)
                print(f"[OK] episode={saved_episode_index} frames={len(converted.images)} scene={scene.name}")
            except Exception as exc:
                failure = {
                    "scene": scene.name,
                    "source": str(scene),
                    "stage": "convert_scene",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                output_root.mkdir(parents=True, exist_ok=True)
                with (output_root / "failed_scenes.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                if dataset is not None:
                    dataset.clear_episode_buffer(delete_images=True)
                print(f"[FAILED] {scene.name}: {exc}", file=sys.stderr)
        if dataset is None or not records:
            raise RuntimeError("No scene was converted successfully.")
        dataset.finalize()
    finally:
        if dataset is not None and dataset.image_writer is not None:
            dataset.stop_image_writer()

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "elapsed_s": time.time() - started,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "cfg": CFG,
        "camera_serial": CAMERA_SERIAL,
        "camera_mount_type": "fixed_external",
        "action_source": "robot_command/tcpcommand_timestamp.npy",
        "action_alignment": "observation/current_command",
        "action_pose_frame": "eef0",
        "command_position_unit": "millimeter_to_meter",
        "command_rotation": "xyz_euler_radians",
        "observation_state_source": "transformed/tcp_base.npy",
        "actual_quaternion_order": "wxyz",
        "depth_encoding": "stacked_low_byte_then_high_byte",
        "depth_unit": "millimeter",
        "depth_scale": 1000.0,
        "num_points": int(args.num_points),
        "episodes": records,
        "failed_scene_count": sum(1 for _ in (output_root / "failed_scenes.jsonl").open()),
    }
    write_json(output_root / "meta" / "rh20t_conversion.json", summary)
    write_json(
        output_root / "meta" / "manifest.json",
        {
            "version": 1,
            "source": "RH20T_cfg7",
            "camera_serial": CAMERA_SERIAL,
            "image_key": IMAGE_KEY,
            "image_shape": IMAGE_SHAPE,
            "action_shape": [10],
            "action_source": "robot_command/tcpcommand_timestamp.npy",
            "action_alignment": "observation/current_command",
            "point_cloud_frame": "current_eef",
            "point_cloud_storage": "point_clouds/episode_*.zarr",
            "episodes": len(records),
            "frames": sum(record["output_frame_count"] for record in records),
        },
    )
    print(f"[DONE] output={output_root}")
    print(f"[DONE] episodes={len(records)} frames={sum(record['output_frame_count'] for record in records)}")


if __name__ == "__main__":
    main()
