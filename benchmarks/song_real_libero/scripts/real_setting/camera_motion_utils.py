#!/usr/bin/env python
"""Camera-motion geometry shared by real-robot recording and conversion.

Transform convention
--------------------
``camera_to_tracking[t]`` is :math:`T_{T<-C_t}`: it maps a point expressed in
the camera frame at time ``t`` into the arbitrary frame of the VIO/SLAM
tracking system.  The model's stable world frame is the first overview-camera
frame of the episode:

``camera_to_model_world[t] = inv(camera_to_tracking[0]) @ camera_to_tracking[t]``

Therefore ``camera_to_model_world[t]`` maps points from ``C_t`` to ``C_0``.
``world`` below always means this model/dataset world, never the arbitrary
tracking frame and never a calibrated robot-base frame. ``overhead`` remains
only a camera name / storage key. This module intentionally has no model or
simulator dependencies.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CAMERA_POSE_DIRECTION_CAMERA_TO_TRACKING = "camera_to_tracking"
CAMERA_POSE_DIRECTION_TRACKING_TO_CAMERA = "tracking_to_camera"
# Input-only aliases for old files and common external VIO terminology. An
# external key named "camera_to_world" is canonicalized as camera_to_tracking;
# it does not define the model world.
CAMERA_POSE_DIRECTION_CAMERA_TO_WORLD = "camera_to_world"
CAMERA_POSE_DIRECTION_WORLD_TO_CAMERA = "world_to_camera"


def _normalize(vector: np.ndarray, *, name: str, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"{name} has near-zero or non-finite norm.")
    return vector / norm


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 rotation matrix onto SO(3)."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"Rotation must be a finite (3,3) matrix, got {rotation.shape}.")
    u, _singular_values, vh = np.linalg.svd(rotation)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def quaternion_xyzw_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize(
        np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4),
        name="quaternion",
    )
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose9_to_matrix(pose9: np.ndarray) -> np.ndarray:
    """Convert [xyz, x_axis, y_axis] pose vectors to homogeneous matrices."""

    poses = np.asarray(pose9, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None]
    if poses.ndim != 2 or poses.shape[1] < 9:
        raise ValueError(f"pose9 must have shape (T,>=9), got {poses.shape}.")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    for index, pose in enumerate(poses):
        x_axis = _normalize(pose[3:6], name=f"pose9[{index}] x-axis")
        y_raw = pose[6:9] - x_axis * float(np.dot(x_axis, pose[6:9]))
        y_axis = _normalize(y_raw, name=f"pose9[{index}] y-axis")
        z_axis = _normalize(np.cross(x_axis, y_axis), name=f"pose9[{index}] z-axis")
        matrices[index, :3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        matrices[index, :3, 3] = pose[:3]
    return matrices


def matrix_to_pose9(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None]
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected homogeneous matrices with shape (T,4,4), got {matrices.shape}.")
    return np.concatenate(
        [matrices[:, :3, 3], matrices[:, :3, 0], matrices[:, :3, 1]],
        axis=-1,
    ).astype(np.float32)


def camera_pose_values_to_matrices(
    values: np.ndarray,
    *,
    pose_format: str = "auto",
    direction: str = CAMERA_POSE_DIRECTION_CAMERA_TO_TRACKING,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Parse full-SE(3) camera poses into ``T_tracking_camera``."""

    values = np.asarray(values)
    resolved_format = str(pose_format)
    if resolved_format == "auto":
        if (values.ndim == 3 and values.shape[1:] == (4, 4)) or (values.ndim == 2 and values.shape[1] == 16):
            resolved_format = "matrix"
        elif values.ndim == 2 and values.shape[1] == 7:
            resolved_format = "pose7_xyzw"
        elif values.ndim == 2 and values.shape[1] >= 9:
            resolved_format = "pose9"
        else:
            raise ValueError(f"Cannot infer camera-pose format from shape {values.shape}.")

    if resolved_format == "matrix":
        if values.ndim == 2 and values.shape[1] == 16:
            matrices = values.reshape(-1, 4, 4).astype(np.float64)
        elif values.ndim == 3 and values.shape[1:] == (4, 4):
            matrices = values.astype(np.float64)
        else:
            raise ValueError(f"matrix camera poses require (T,4,4) or (T,16), got {values.shape}.")
    elif resolved_format in {"pose7_xyzw", "pose7_wxyz"}:
        if values.ndim != 2 or values.shape[1] < 7:
            raise ValueError(f"{resolved_format} requires shape (T,>=7), got {values.shape}.")
        matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
        matrices[:, :3, 3] = np.asarray(values[:, :3], dtype=np.float64)
        for index, value in enumerate(values):
            quaternion = np.asarray(value[3:7], dtype=np.float64)
            if resolved_format == "pose7_wxyz":
                quaternion = quaternion[[1, 2, 3, 0]]
            matrices[index, :3, :3] = quaternion_xyzw_to_rotation(quaternion)
    elif resolved_format == "pose9":
        matrices = pose9_to_matrix(values)
    else:
        raise ValueError(f"Unsupported camera-pose format: {resolved_format!r}.")

    if not np.isfinite(float(translation_scale)) or float(translation_scale) <= 0.0:
        raise ValueError(f"translation_scale must be finite and positive, got {translation_scale}.")
    matrices[:, :3, 3] *= float(translation_scale)
    matrices = validate_transform_sequence(matrices)
    if direction in {
        CAMERA_POSE_DIRECTION_TRACKING_TO_CAMERA,
        CAMERA_POSE_DIRECTION_WORLD_TO_CAMERA,
    }:
        matrices = invert_transform_sequence(matrices)
    elif direction not in {
        CAMERA_POSE_DIRECTION_CAMERA_TO_TRACKING,
        CAMERA_POSE_DIRECTION_CAMERA_TO_WORLD,
    }:
        raise ValueError(f"Unsupported camera-pose direction: {direction!r}.")
    return validate_transform_sequence(matrices)


def validate_transform_sequence(matrices: np.ndarray) -> np.ndarray:
    """Validate SE(3) matrices and project small rotation noise onto SO(3)."""

    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None]
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Camera poses must have shape (T,4,4), got {matrices.shape}.")
    if len(matrices) == 0 or not np.isfinite(matrices).all():
        raise ValueError("Camera-pose sequence is empty or contains non-finite values.")
    expected_last_row = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(matrices[:, 3, :], expected_last_row, atol=1e-5, rtol=0.0):
        raise ValueError("Camera-pose matrices must have homogeneous last row [0,0,0,1].")
    output = matrices.copy()
    for index in range(len(output)):
        output[index, :3, :3] = project_rotation_to_so3(output[index, :3, :3])
        output[index, 3, :] = expected_last_row
    return output


def invert_transform_sequence(matrices: np.ndarray) -> np.ndarray:
    matrices = validate_transform_sequence(matrices)
    inverse = np.repeat(np.eye(4, dtype=np.float64)[None], len(matrices), axis=0)
    rotation_t = np.swapaxes(matrices[:, :3, :3], -1, -2)
    inverse[:, :3, :3] = rotation_t
    inverse[:, :3, 3] = -(rotation_t @ matrices[:, :3, 3, None])[..., 0]
    return inverse


def camera_to_model_world_transforms(
    camera_to_tracking: np.ndarray,
    reference_camera_to_tracking: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``T_modelWorld_currentCamera`` for every frame.

    Without an explicit reference, model world is the first camera frame of
    this episode. With ``reference_camera_to_tracking``, model world is that
    canonical camera frame expressed in the same persistent tracking frame.
    """

    camera_to_tracking = validate_transform_sequence(camera_to_tracking)
    if reference_camera_to_tracking is None:
        reference = camera_to_tracking[:1]
    else:
        reference = validate_transform_sequence(reference_camera_to_tracking)
        if len(reference) != 1:
            raise ValueError("reference_camera_to_tracking must contain exactly one SE(3) transform.")
    tracking_to_model_world = invert_transform_sequence(reference)[0]
    transforms = tracking_to_model_world[None] @ camera_to_tracking
    transforms = validate_transform_sequence(transforms)
    if reference_camera_to_tracking is None:
        # Avoid tiny numerical drift in the episode reference frame.
        transforms[0] = np.eye(4, dtype=np.float64)
    return transforms


def camera_to_overhead_transforms(camera_to_tracking: np.ndarray) -> np.ndarray:
    """Deprecated alias; overhead is a camera name, not a coordinate frame."""

    return camera_to_model_world_transforms(camera_to_tracking)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    transform = validate_transform_sequence(np.asarray(transform))[0]
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_xyzrgb_cloud(cloud: np.ndarray, transform: np.ndarray) -> np.ndarray:
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape (N,>=3), got {cloud.shape}.")
    output = cloud.copy()
    output[:, :3] = transform_points(output[:, :3], transform)
    return output


def transform_pose9_sequence(poses_camera: np.ndarray, camera_to_model_world: np.ndarray) -> np.ndarray:
    """Transform ``T_camera_ee`` poses into the episode model-world frame."""

    pose_matrices = pose9_to_matrix(poses_camera)
    camera_to_model_world = validate_transform_sequence(camera_to_model_world)
    if len(pose_matrices) != len(camera_to_model_world):
        raise ValueError(
            f"Pose count {len(pose_matrices)} does not match "
            f"camera-transform count {len(camera_to_model_world)}."
        )
    return matrix_to_pose9(camera_to_model_world @ pose_matrices)


def rotation_angle_rad(rotation: np.ndarray) -> float:
    rotation = project_rotation_to_so3(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def camera_motion_metrics(camera_to_model_world: np.ndarray) -> dict[str, Any]:
    camera_to_model_world = validate_transform_sequence(camera_to_model_world)
    translations = camera_to_model_world[:, :3, 3]
    angles = np.asarray([rotation_angle_rad(matrix[:3, :3]) for matrix in camera_to_model_world])
    increments = (
        np.linalg.norm(np.diff(translations, axis=0), axis=-1) if len(translations) > 1 else np.zeros(0)
    )
    return {
        "frame_count": int(len(camera_to_model_world)),
        "translation_span_m": float(np.max(np.linalg.norm(translations, axis=-1))),
        "rotation_span_deg": float(np.rad2deg(np.max(angles))),
        "max_translation_step_m": float(np.max(increments)) if len(increments) else 0.0,
        "final_translation_m": translations[-1].tolist(),
        "final_rotation_deg": float(np.rad2deg(angles[-1])),
    }


def stationary_pose_report(
    camera_to_model_world: np.ndarray,
    *,
    min_frames: int,
    max_translation_drift_m: float,
    max_rotation_drift_deg: float,
    accepted: np.ndarray | None = None,
    min_accepted_ratio: float = 0.8,
) -> dict[str, Any]:
    """Summarize whether an estimated camera trajectory is stationary.

    This function only evaluates a supplied full-SE(3) trajectory. It does not
    infer translation from raw IMU samples. ``camera_to_model_world`` is
    expected to be relative to the first camera frame.
    """

    transforms = validate_transform_sequence(camera_to_model_world)
    if int(min_frames) <= 0:
        raise ValueError("min_frames must be positive.")
    if float(max_translation_drift_m) < 0.0:
        raise ValueError("max_translation_drift_m must be non-negative.")
    if float(max_rotation_drift_deg) < 0.0:
        raise ValueError("max_rotation_drift_deg must be non-negative.")
    if not 0.0 <= float(min_accepted_ratio) <= 1.0:
        raise ValueError("min_accepted_ratio must be in [0, 1].")

    translation_drift_m = np.linalg.norm(transforms[:, :3, 3], axis=-1)
    rotation_drift_deg = np.rad2deg(np.asarray([rotation_angle_rad(matrix[:3, :3]) for matrix in transforms]))
    translation_step_m = (
        np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=-1)
        if len(transforms) > 1
        else np.zeros(0, dtype=np.float64)
    )

    if accepted is None:
        accepted_mask = np.ones(len(transforms), dtype=bool)
    else:
        accepted_mask = np.asarray(accepted, dtype=bool).reshape(-1)
        if len(accepted_mask) != len(transforms):
            raise ValueError(f"accepted has {len(accepted_mask)} entries for {len(transforms)} camera poses.")
    accepted_count = int(accepted_mask.sum())
    accepted_ratio = float(accepted_count / len(transforms))
    enough_frames = len(transforms) >= int(min_frames)
    translation_ok = float(np.max(translation_drift_m)) <= float(max_translation_drift_m)
    rotation_ok = float(np.max(rotation_drift_deg)) <= float(max_rotation_drift_deg)
    accepted_ratio_ok = accepted_ratio >= float(min_accepted_ratio)

    return {
        "passed": bool(enough_frames and translation_ok and rotation_ok and accepted_ratio_ok),
        "frame_count": int(len(transforms)),
        "min_frames": int(min_frames),
        "enough_frames": bool(enough_frames),
        "accepted_count": accepted_count,
        "accepted_ratio": accepted_ratio,
        "min_accepted_ratio": float(min_accepted_ratio),
        "accepted_ratio_ok": bool(accepted_ratio_ok),
        "translation": {
            "max_drift_m": float(np.max(translation_drift_m)),
            "p95_drift_m": float(np.percentile(translation_drift_m, 95.0)),
            "final_drift_m": float(translation_drift_m[-1]),
            "max_step_m": (float(np.max(translation_step_m)) if len(translation_step_m) else 0.0),
            "threshold_m": float(max_translation_drift_m),
            "passed": bool(translation_ok),
        },
        "rotation": {
            "max_drift_deg": float(np.max(rotation_drift_deg)),
            "p95_drift_deg": float(np.percentile(rotation_drift_deg, 95.0)),
            "final_drift_deg": float(rotation_drift_deg[-1]),
            "threshold_deg": float(max_rotation_drift_deg),
            "passed": bool(rotation_ok),
        },
    }


def matrix_from_json_record(record: dict[str, Any]) -> np.ndarray:
    """Parse one external VIO/SLAM record into a camera-to-tracking matrix."""

    for key in (
        "camera_to_tracking",
        "T_tracking_camera",
        # Input compatibility with common VIO exports and old project files.
        "camera_to_world",
        "T_world_camera",
        "transform_matrix",
    ):
        if key in record:
            matrix = np.asarray(record[key], dtype=np.float64)
            if matrix.size != 16:
                raise ValueError(f"{key} must contain 16 matrix values, got shape {matrix.shape}.")
            return validate_transform_sequence(matrix.reshape(4, 4))[0]
    if "translation_m" in record:
        translation = np.asarray(record["translation_m"], dtype=np.float64).reshape(3)
        if "quaternion_xyzw" in record:
            quaternion = np.asarray(record["quaternion_xyzw"], dtype=np.float64).reshape(4)
        elif "quaternion_wxyz" in record:
            quaternion = np.asarray(record["quaternion_wxyz"], dtype=np.float64).reshape(4)[[1, 2, 3, 0]]
        else:
            raise KeyError(
                "Camera-pose record with translation_m also needs quaternion_xyzw or quaternion_wxyz."
            )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
        matrix[:3, 3] = translation
        return matrix
    raise KeyError(
        "Camera-pose record needs camera_to_tracking/T_tracking_camera/transform_matrix, "
        "or translation_m plus a quaternion."
    )
