#!/usr/bin/env python
"""Torch-free point-cloud view composition for LIBERO environment workers.

These functions intentionally mirror the NumPy-only public helpers in
``lerobot.policies.smolvla.song_pointseg``. Importing that policy module from a
spawned MuJoCo worker also imports Torch, Open3D, and CUDA libraries, adding
several GiB of private memory per environment process.
"""

from __future__ import annotations

from typing import Any

import numpy as np

POINT_CLOUD_VIEW_DIRS = {
    "agentview": "point_clouds",
    "robot0_eye_in_hand": "point_clouds_robot0_eye_in_hand",
}


def parse_camera_views(value: Any = None) -> tuple[str, ...]:
    if value is None:
        return ("agentview",)
    if isinstance(value, list | tuple):
        parts = [str(part).strip() for part in value]
    else:
        text = str(value).strip().strip("[]")
        parts = [part.strip().strip("\"'") for part in text.split(",")]
    views = tuple(part for part in parts if part) or ("agentview",)
    unknown = [view for view in views if view not in POINT_CLOUD_VIEW_DIRS]
    if unknown:
        raise ValueError(
            f"Unsupported camera view(s) {unknown}; supported views are "
            f"{sorted(POINT_CLOUD_VIEW_DIRS)}."
        )
    if len(set(views)) != len(views):
        raise ValueError(f"camera views contain duplicates: {views}.")
    return views


def parse_camera_view_weights(
    value: Any = None,
    *,
    num_views: int | None = None,
) -> tuple[float, ...] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, list | tuple):
        parts = list(value)
    else:
        text = str(value).strip().strip("[]")
        parts = [part.strip().strip("\"'") for part in text.split(",")]
    weights = tuple(float(part) for part in parts if str(part).strip())
    if not weights:
        return None
    if num_views is not None and len(weights) != int(num_views):
        raise ValueError(
            f"camera view weights must contain one value per view: got {weights} "
            f"for {int(num_views)} views."
        )
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError(f"camera view weights must be finite and positive, got {weights}.")
    return weights


def parse_camera_view_fusion(value: Any = None) -> str:
    if value is None or not str(value).strip():
        return "legacy_budget"
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy_budget",
        "budget": "legacy_budget",
        "legacy_budget": "legacy_budget",
        "fps": "fps",
        "voxel_fps": "voxel_fps",
        "voxelized_fps": "voxel_fps",
        "voxel_cover_fps": "voxel_cover_fps",
        "voxel_coverage_fps": "voxel_cover_fps",
        "novelty_union": "novelty_union",
        "novelty": "novelty_union",
        "multiscale_novelty_union": "multiscale_novelty_union",
        "multiscale_novelty": "multiscale_novelty_union",
        "consensus_multiscale_novelty_union": "consensus_multiscale_novelty_union",
        "consensus_multiscale_novelty": "consensus_multiscale_novelty_union",
        "transport_novelty_union": "transport_novelty_union",
        "transport_novelty": "transport_novelty_union",
        "uniform_union": "uniform_union",
        "random_union": "uniform_union",
        "union_random": "uniform_union",
        "full_union": "full_union",
        "union_all": "full_union",
        "primary_residual": "primary_residual",
        "residual": "primary_residual",
    }
    if mode not in aliases:
        raise ValueError(
            "camera view fusion must be 'legacy_budget', 'fps', 'voxel_fps', "
            "'voxel_cover_fps', 'novelty_union', 'multiscale_novelty_union', "
            "'consensus_multiscale_novelty_union', 'transport_novelty_union', "
            "'uniform_union', 'full_union', or 'primary_residual'; "
            f"got {value!r}."
        )
    return aliases[mode]


def compose_point_cloud_views(
    view_clouds: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    gripper_points: int = 500,
    seed: int = 0,
    view_weights: Any = None,
    fusion: Any = "legacy_budget",
) -> np.ndarray:
    clouds = [np.asarray(cloud, dtype=np.float32) for cloud in view_clouds]
    if not clouds:
        raise ValueError("At least one camera cloud is required.")
    for cloud in clouds:
        if cloud.ndim != 2 or cloud.shape[-1] != 6:
            raise ValueError(f"Expected point cloud shape (N,6), got {cloud.shape}.")
    if len(clouds) == 1:
        return np.ascontiguousarray(clouds[0], dtype=np.float32)
    fusion = parse_camera_view_fusion(fusion)
    total_points = int(clouds[0].shape[0])
    if any(int(cloud.shape[0]) != total_points for cloud in clouds[1:]):
        raise ValueError(
            "All camera clouds must contain the same number of points, got "
            f"{[cloud.shape[0] for cloud in clouds]}."
        )
    gripper_points = int(gripper_points)
    if gripper_points < 0 or gripper_points >= total_points:
        raise ValueError(
            f"gripper_points must be in [0, {total_points - 1}], got {gripper_points}."
        )
    if fusion == "uniform_union":
        if parse_camera_view_weights(view_weights, num_views=len(clouds)) is not None:
            raise ValueError("camera view weights cannot be combined with uniform_union fusion.")
        scene_budget = total_points - gripper_points
        scene_union = np.concatenate(
            [cloud[:-gripper_points] if gripper_points else cloud for cloud in clouds],
            axis=0,
        )
        if scene_union.shape[0] < scene_budget:
            raise ValueError(
                f"Scene union has {scene_union.shape[0]} points but requires {scene_budget}."
            )
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(scene_union.shape[0], scene_budget, replace=False)
        parts = [np.ascontiguousarray(scene_union[indices], dtype=np.float32)]
        if gripper_points:
            parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    if fusion in {
        "fps",
        "voxel_fps",
        "voxel_cover_fps",
        "novelty_union",
        "multiscale_novelty_union",
        "consensus_multiscale_novelty_union",
        "transport_novelty_union",
        "full_union",
        "primary_residual",
    }:
        if parse_camera_view_weights(view_weights, num_views=len(clouds)) is not None:
            raise ValueError(f"camera view weights cannot be combined with {fusion} fusion.")
        parts = [
            np.ascontiguousarray(
                cloud[:-gripper_points] if gripper_points else cloud,
                dtype=np.float32,
            )
            for cloud in clouds
        ]
        if gripper_points:
            parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)

    scene_budget = total_points - gripper_points
    parsed_weights = parse_camera_view_weights(view_weights, num_views=len(clouds))
    weights = np.ones(len(clouds), dtype=np.float64)
    if parsed_weights is not None:
        weights = np.asarray(parsed_weights, dtype=np.float64)
    expected = scene_budget * weights / weights.sum()
    allocations_array = np.floor(expected).astype(np.int64)
    remainder = int(scene_budget - allocations_array.sum())
    if remainder:
        fractional_order = np.argsort(-(expected - allocations_array), kind="stable")
        allocations_array[fractional_order[:remainder]] += 1
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for cloud, count in zip(clouds, allocations_array.tolist(), strict=True):
        scene = cloud[:-gripper_points] if gripper_points else cloud
        if len(scene) == 0:
            raise ValueError("A camera cloud contains no non-gripper scene points.")
        if len(scene) == count:
            sample = scene
        else:
            indices = rng.choice(len(scene), count, replace=len(scene) < count)
            sample = scene[indices]
        parts.append(np.ascontiguousarray(sample, dtype=np.float32))
    if gripper_points:
        parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
    composed = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    if composed.shape != clouds[0].shape:
        raise RuntimeError(
            f"Composed cloud shape {composed.shape} does not match stored shape {clouds[0].shape}."
        )
    return composed
