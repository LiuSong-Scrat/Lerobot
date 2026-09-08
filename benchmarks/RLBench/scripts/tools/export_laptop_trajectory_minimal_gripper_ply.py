#!/usr/bin/env python3
"""Export one close_laptop_lid trajectory with the U-shaped virtual gripper."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import zarr

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from libero_setting.libero_pointcloud_utils import (  # noqa: E402
    create_rlbench_minimal_two_finger_points,
)
from rlbench_reap_gripper import (  # noqa: E402
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    canonical_reap_metadata,
    create_rlbench_reap_points_from_physical_width,
    rlbench_physical_width_from_open_fraction,
)


DEFAULT_DATASET = ROOT / "benchmarks/RLBench/datasets/laptop/rlbench_close_laptop_lid_100traj_lerobot_raw_50000_20260812"
DEFAULT_OUTPUT = ROOT / "benchmarks/RLBench/outputs/laptop_episode_000000_minimal_gripper_ply"
DEFAULT_CACHE = ROOT / "benchmarks/RLBench/datasets/laptop/rlbench_close_laptop_lid_100traj_lerobot_raw_50000_20260812_pointseg_cache_10000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointseg-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument(
        "--gripper-template",
        choices=["rlbench_minimal", "libero"],
        default="libero",
        help="Use the canonical RLBench-aligned REAP template or the legacy Panda U template.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Export exactly the cached point selection for each frame instead of all scene points.",
    )
    return parser.parse_args()


def foreground_colors(scores: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    """Blue-to-cyan-to-yellow-to-red colors for cached foreground scores."""
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    color = np.asarray(base_rgb, dtype=np.float32).copy()
    valid = np.isfinite(score)
    value = np.clip(score[valid], 0.0, 1.0)
    heat = np.stack(
        (np.clip(2.0 * value, 0.0, 1.0), np.clip(1.0 - np.abs(2.0 * value - 1.0), 0.0, 1.0), np.clip(1.0 - 2.0 * value, 0.0, 1.0)),
        axis=1,
    )
    color[valid] = heat * 255.0
    return np.clip(color, 0.0, 255.0).astype(np.uint8)


def load_episode_cache_scores(cache_root: Path, episode: int, frame_count: int, total_points: int) -> list[dict]:
    samples: list[dict | None] = [None] * frame_count
    matched = np.zeros(frame_count, dtype=np.int32)
    for shard in sorted(cache_root.glob("rank_*/shard_*")):
        episode_indices = np.load(shard / "episode_index.npy", mmap_mode="r")
        rows = np.flatnonzero(episode_indices == episode)
        if not len(rows):
            continue
        frame_indices = np.load(shard / "frame_index.npy", mmap_mode="r")
        sample_offsets = np.load(shard / "sample_offsets.npy", mmap_mode="r")
        point_indices = np.load(shard / "point_indices.npy", mmap_mode="r")
        foreground_score = np.load(shard / "foreground_score.npy", mmap_mode="r")
        for row in rows:
            frame = int(frame_indices[row])
            if not 0 <= frame < frame_count:
                continue
            start, stop = int(sample_offsets[row]), int(sample_offsets[row + 1])
            indices = np.asarray(point_indices[start:stop], dtype=np.int64)
            values = np.asarray(foreground_score[start:stop], dtype=np.float32)
            valid = (indices >= 0) & (indices < total_points) & np.isfinite(values)
            samples[frame] = {"indices": indices[valid], "scores": values[valid]}
            matched[frame] += 1
    missing = np.flatnonzero(matched == 0)
    repeated = np.flatnonzero(matched > 1)
    if len(missing) or len(repeated):
        raise RuntimeError(
            f"PointSeg cache mapping is not one-to-one: missing={missing.tolist()} repeated={repeated.tolist()}"
        )
    return [sample for sample in samples if sample is not None]


def write_ply(
    path: Path,
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    scene_scores: np.ndarray,
    gripper_xyz: np.ndarray,
) -> None:
    xyz = np.vstack((scene_xyz, gripper_xyz)).astype(np.float32)
    rgb = np.vstack((foreground_colors(scene_scores, scene_rgb), np.tile([235, 70, 35], (len(gripper_xyz), 1)))).astype(np.uint8)
    kinds = np.concatenate((np.zeros(len(scene_xyz), dtype=np.uint8), np.ones(len(gripper_xyz), dtype=np.uint8)))
    scores = np.concatenate((scene_scores, np.full(len(gripper_xyz), np.nan, dtype=np.float32)))
    selected = np.concatenate((np.isfinite(scene_scores), np.zeros(len(gripper_xyz), dtype=bool)))
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment coordinate_frame current_eef\n")
        file.write("comment point_kind 0=scene 1=rlbench_minimal_two_finger\n")
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property float foreground_score\nproperty uchar cache_point_selected\n")
        file.write("property uchar point_kind\nend_header\n")
        for point, color, score, is_selected, kind in zip(xyz, rgb, scores, selected, kinds, strict=True):
            score_text = "nan" if not np.isfinite(score) else f"{score:.7f}"
            file.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {score_text} "
                f"{int(is_selected)} {int(kind)}\n"
            )


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    episode = int(args.episode)
    point_path = root / "point_clouds" / f"episode_{episode:06d}.zarr"
    actions_path = root / "raw_expert_actions" / f"episode_{episode:06d}.npy"
    point_group = zarr.open(str(point_path), mode="r")
    xyz = np.asarray(point_group["xyz"], dtype=np.float32)
    rgb = np.asarray(point_group["rgb"], dtype=np.uint8)
    actions = np.asarray(np.load(actions_path), dtype=np.float32)
    frame_count = min(len(xyz), len(rgb), len(actions))
    if frame_count == 0:
        raise RuntimeError("The selected episode contains no aligned point-cloud frames.")
    if xyz.shape[1] <= int(args.gripper_points):
        raise RuntimeError(f"Expected scene + gripper points, got {xyz.shape[1]} total points.")

    output = args.output_dir.expanduser().resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    scene_count = xyz.shape[1] - int(args.gripper_points)
    cache_samples = load_episode_cache_scores(
        args.pointseg_cache.expanduser().resolve(), episode, frame_count, xyz.shape[1]
    )
    records = []
    for frame in range(frame_count):
        width_percent = float(np.clip(actions[frame, 7], 0.0, 1.0))
        physical_width = float(rlbench_physical_width_from_open_fraction(width_percent))
        def make_gripper(count: int) -> np.ndarray:
            rng = np.random.default_rng(episode * 100000 + frame)
            if args.gripper_template == "libero":
                return create_rlbench_reap_points_from_physical_width(
                    physical_width,
                    np.zeros(6, dtype=np.float32),
                    count,
                    rng,
                ).astype(np.float32)
            return create_rlbench_minimal_two_finger_points(
                width_percent,
                np.zeros(6, dtype=np.float32),
                count,
                rng,
            ).astype(np.float32)

        gripper = make_gripper(int(args.gripper_points))
        filename = f"frame_{frame:06d}.ply"
        cache_indices = cache_samples[frame]["indices"]
        cache_scores = cache_samples[frame]["scores"]
        full_scores = np.full(xyz.shape[1], np.nan, dtype=np.float32)
        full_scores[cache_indices] = cache_scores
        scene_scores = full_scores[:scene_count]
        if args.cache_only:
            old_gripper_mask = cache_indices >= scene_count
            scene_indices = cache_indices[~old_gripper_mask]
            scene_xyz = xyz[frame, scene_indices]
            scene_rgb = rgb[frame, scene_indices]
            selected_scores = cache_scores[~old_gripper_mask]
            replacement_count = int(old_gripper_mask.sum())
            if replacement_count:
                replacement = make_gripper(replacement_count)
            else:
                replacement = np.empty((0, 3), dtype=np.float32)
            write_ply(
                frames_dir / filename,
                scene_xyz,
                scene_rgb,
                selected_scores,
                replacement,
            )
            exported_points = int(len(scene_xyz) + len(replacement))
        else:
            write_ply(
                frames_dir / filename,
                xyz[frame, :scene_count],
                rgb[frame, :scene_count],
                scene_scores,
                gripper,
            )
            exported_points = int(scene_count + len(gripper))
        records.append({
            "frame": frame,
            "file": f"frames/{filename}",
            "gripper_width_percent": width_percent,
            "gripper_physical_width_m": physical_width,
            "cached_foreground_points": int(np.isfinite(scene_scores).sum()),
            "cached_foreground_score_mean": (
                float(np.nanmean(scene_scores)) if np.isfinite(scene_scores).any() else None
            ),
            "exported_points": exported_points,
        })
    manifest = {
        "dataset_root": str(root),
        "episode": episode,
        "frame_count": frame_count,
        "scene_points_per_frame": int(scene_count),
        "cache_only": bool(args.cache_only),
        "virtual_gripper_points_per_frame": int(args.gripper_points),
        "pointseg_cache": str(args.pointseg_cache.expanduser().resolve()),
        "foreground_score_semantics": "cached PointSeg pseudo foreground score; NaN means not selected by the 10000-point cache sample",
        "gripper_template": (
            LIBERO_GRIPPER_TEMPLATE_VERSION
            if args.gripper_template == "libero"
            else "rlbench_minimal_two_finger_u_bridge_v1"
        ),
        "coordinate_frame": "current_eef",
        "frames": records,
    }
    if args.gripper_template == "libero":
        manifest.update(canonical_reap_metadata())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": frame_count, "points_per_frame": int(scene_count + args.gripper_points)}))


if __name__ == "__main__":
    main()
