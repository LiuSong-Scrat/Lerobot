#!/usr/bin/env python3
"""Canonical RH20T v3 virtual parallel-gripper geometry.

The model-facing opening is the physical clear gap, in metres, between the
two inner finger faces.  Geometry must never infer a per-episode scale or turn
that value into a percentage.  This module is shared by dataset conversion,
real-data preprocessing and evaluation so those paths cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRIPPER_CONTRACT = "rh20t_canonical_parallel_gripper_v3"
GRIPPER_FRAME_CONVENTION = "x_up_normal_y_right_z_forward_origin_finger_five_eighths_v2"
GRIPPER_ORIGIN_FINGER_FRACTION_FROM_BASE = 5.0 / 8.0
GRIPPER_COLOR_RGB = np.array([255.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_GRIPPER_POINTS = 500

# The converted LIBERO dataset defines the model-facing virtual EEF from the
# original robosuite / controller EEF by right-multiplying this transform:
#
#   T_world_virtual_eef = T_world_previous_eef @ T_previous_eef_virtual_eef
#
# Keep this next to the canonical geometry so collection, training and
# evaluation cannot silently use different tool-frame origins.
T_PREVIOUS_EEF_VIRTUAL_EEF = np.eye(4, dtype=np.float32)
T_PREVIOUS_EEF_VIRTUAL_EEF[:3, 3] = np.array([0.005, 0.0, -0.03], dtype=np.float32)


@dataclass(frozen=True)
class CanonicalGripperParameters:
    max_width_m: float = 0.08
    finger_length_m: float = 0.08
    finger_thickness_m: float = 0.01
    palm_width_m: float = 0.10
    palm_depth_m: float = 0.01
    handle_length_m: float = 0.05


DEFAULT_PARAMETERS = CanonicalGripperParameters()


def physical_opening_widths(
    values: np.ndarray,
    *,
    already_normalized: bool = False,
    max_width_m: float | None = DEFAULT_PARAMETERS.max_width_m,
) -> np.ndarray:
    """Return physical two-finger clear-gap widths in metres.

    Normalized source values are supported only as an explicit ingestion
    option and are converted once here.  Metre-valued data is never divided by
    an episode maximum or device maximum.
    """

    widths = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if not np.all(np.isfinite(widths)):
        invalid = np.flatnonzero(~np.isfinite(widths))
        raise ValueError(f"gripper widths contain non-finite values at {invalid[:20].tolist()}")
    if max_width_m is not None:
        max_width_m = float(max_width_m)
        if not np.isfinite(max_width_m) or max_width_m <= 0.0:
            raise ValueError(f"max_width_m must be positive and finite, got {max_width_m}")
    if already_normalized:
        if max_width_m is None:
            raise ValueError("max_width_m is required when source widths are normalized")
        widths = np.clip(widths, 0.0, 1.0) * max_width_m
    elif np.any(widths < 0.0):
        invalid = np.flatnonzero(widths < 0.0)
        raise ValueError(f"physical gripper widths must be non-negative; bad frames {invalid[:20].tolist()}")
    if max_width_m is not None:
        widths = np.clip(widths, 0.0, max_width_m)
    return widths.astype(np.float32, copy=False)


def physical_opening_width(
    value: float,
    *,
    max_width_m: float | None = DEFAULT_PARAMETERS.max_width_m,
) -> float:
    width = float(value)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError(f"physical gripper width must be finite and non-negative, got {value}")
    if max_width_m is None:
        return width
    limit = float(max_width_m)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"max_width_m must be positive and finite, got {limit}")
    return float(np.clip(width, 0.0, limit))


def allocate_surface_counts(total: int, sizes: list[np.ndarray]) -> np.ndarray:
    """RH20T-compatible surface-area allocation over the four boxes."""

    areas = np.asarray(
        [2.0 * (s[0] * s[1] + s[0] * s[2] + s[1] * s[2]) for s in sizes],
        dtype=np.float64,
    )
    expected = int(total) * areas / areas.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def sample_box_surface(
    min_corner: np.ndarray,
    size: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one box surface exactly as the RH20T template does."""

    size = np.asarray(size, dtype=np.float64)
    min_corner = np.asarray(min_corner, dtype=np.float64)
    sx, sy, sz = size
    x0, y0, z0 = min_corner
    faces = (
        (sy * sz, [x0, y0, z0], [0.0, sy, 0.0], [0.0, 0.0, sz]),
        (sy * sz, [x0 + sx, y0, z0], [0.0, sy, 0.0], [0.0, 0.0, sz]),
        (sx * sz, [x0, y0, z0], [sx, 0.0, 0.0], [0.0, 0.0, sz]),
        (sx * sz, [x0, y0 + sy, z0], [sx, 0.0, 0.0], [0.0, 0.0, sz]),
        (sx * sy, [x0, y0, z0], [sx, 0.0, 0.0], [0.0, sy, 0.0]),
        (sx * sy, [x0, y0, z0 + sz], [sx, 0.0, 0.0], [0.0, sy, 0.0]),
    )
    weights = np.asarray([face[0] for face in faces], dtype=np.float64)
    expected = int(count) * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = int(count) - int(counts.sum())
    if remainder:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    samples = []
    for face_count, (_, origin, axis_a, axis_b) in zip(counts, faces, strict=True):
        if not face_count:
            continue
        uv = rng.random((int(face_count), 2))
        samples.append(
            np.asarray(origin)
            + uv[:, :1] * np.asarray(axis_a)
            + uv[:, 1:] * np.asarray(axis_b)
        )
    return np.vstack(samples) if samples else np.empty((0, 3), dtype=np.float64)


def sample_gripper_width(
    opening_width_m: float,
    count: int,
    rng: np.random.Generator,
    *,
    parameters: CanonicalGripperParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    """Sample the canonical local-EFF template from a physical width in metres."""

    width = physical_opening_width(opening_width_m, max_width_m=parameters.max_width_m)
    origin_from_finger_base = (
        parameters.finger_length_m * GRIPPER_ORIGIN_FINGER_FRACTION_FROM_BASE
    )
    finger_base_z = -origin_from_finger_base
    thickness = parameters.finger_thickness_m
    palm_depth = parameters.palm_depth_m
    sizes = [
        np.array([thickness, thickness, parameters.finger_length_m]),
        np.array([thickness, thickness, parameters.finger_length_m]),
        np.array([thickness, parameters.palm_width_m, palm_depth]),
        np.array([palm_depth, palm_depth, parameters.handle_length_m]),
    ]
    min_corners = [
        np.array([-thickness / 2.0, -width / 2.0 - thickness, finger_base_z]),
        np.array([-thickness / 2.0, width / 2.0, finger_base_z]),
        np.array([-thickness / 2.0, -parameters.palm_width_m / 2.0, finger_base_z - palm_depth]),
        np.array(
            [
                -palm_depth / 2.0,
                -palm_depth / 2.0,
                finger_base_z - palm_depth - parameters.handle_length_m,
            ]
        ),
    ]
    counts = allocate_surface_counts(count, sizes)
    return np.concatenate(
        [
            sample_box_surface(corner, size, int(box_count), rng)
            for corner, size, box_count in zip(min_corners, sizes, counts, strict=True)
        ],
        axis=0,
    )


def gripper_cloud_rgb(
    opening_width_m: float,
    count: int,
    rng: np.random.Generator,
    *,
    parameters: CanonicalGripperParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    xyz = sample_gripper_width(opening_width_m, count, rng, parameters=parameters)
    rgb = np.broadcast_to(GRIPPER_COLOR_RGB, (len(xyz), 3))
    return np.concatenate((xyz.astype(np.float32), rgb.astype(np.float32)), axis=1)

