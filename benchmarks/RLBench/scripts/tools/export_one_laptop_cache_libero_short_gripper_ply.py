#!/usr/bin/env python3
"""Export one cached laptop point cloud with a shortened LIBERO gripper."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import zarr

from _rlbench_tool_paths import LEROBOT_ROOT as ROOT
sys.path.insert(0, str(ROOT / "benchmarks" / "song_real_libero" / "scripts"))

from rlbench_reap_gripper import (  # noqa: E402
    canonical_reap_metadata,
    create_rlbench_reap_points_from_physical_width,
    rlbench_physical_width_from_open_fraction,
)

DEFAULT_DATASET = ROOT / "benchmarks/RLBench/datasets/laptop/rlbench_close_laptop_lid_100traj_lerobot_raw_50000_20260812"
DEFAULT_CACHE = ROOT / "benchmarks/RLBench/datasets/laptop/rlbench_close_laptop_lid_100traj_lerobot_raw_50000_20260812_pointseg_cache_10000"
DEFAULT_OUTPUT = ROOT / "benchmarks/RLBench/outputs/laptop_frame_000000_cache10000_libero_short_gripper.ply"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pointseg-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument("--finger-length", type=float, default=0.05)
    return parser.parse_args()


def load_cache_sample(cache_root: Path, episode: int, frame: int):
    matches = []
    for shard in sorted(cache_root.glob("rank_*/shard_*")):
        episodes = np.load(shard / "episode_index.npy", mmap_mode="r")
        frames = np.load(shard / "frame_index.npy", mmap_mode="r")
        rows = np.flatnonzero((episodes == episode) & (frames == frame))
        if not len(rows):
            continue
        offsets = np.load(shard / "sample_offsets.npy", mmap_mode="r")
        indices = np.load(shard / "point_indices.npy", mmap_mode="r")
        scores = np.load(shard / "foreground_score.npy", mmap_mode="r")
        row = int(rows[0])
        start, stop = int(offsets[row]), int(offsets[row + 1])
        matches.append((
            np.asarray(indices[start:stop], dtype=np.int64),
            np.asarray(scores[start:stop], dtype=np.float32),
        ))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cache sample for episode={episode}, frame={frame}; found {len(matches)}")
    return matches[0]


def score_color(scores: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    color = np.asarray(base_rgb, dtype=np.float32).copy()
    valid = np.isfinite(scores)
    value = np.clip(scores[valid], 0.0, 1.0)
    color[valid] = np.stack(
        (np.clip(2 * value, 0, 1), np.clip(1 - np.abs(2 * value - 1), 0, 1), np.clip(1 - 2 * value, 0, 1)),
        axis=1,
    ) * 255.0
    return np.clip(color, 0, 255).astype(np.uint8)


def write_ply(path: Path, xyz, rgb, scores, selected, kinds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write("comment coordinate_frame current_eef\n")
        file.write("comment gripper_template rlbench_aligned_reap_short_two_finger_v4\n")
        file.write("element vertex 10000\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("property float foreground_score\nproperty uchar cache_point_selected\nproperty uchar point_kind\n")
        file.write("end_header\n")
        for p, c, s, selected_flag, kind in zip(xyz, rgb, scores, selected, kinds, strict=True):
            score_text = "nan" if not np.isfinite(s) else f"{s:.7f}"
            file.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])} {score_text} {int(selected_flag)} {int(kind)}\n")


def main():
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    episode, frame = int(args.episode), int(args.frame)
    point_group = zarr.open(str(root / "point_clouds" / f"episode_{episode:06d}.zarr"), mode="r")
    xyz = np.asarray(point_group["xyz"][frame], dtype=np.float32)
    rgb = np.asarray(point_group["rgb"][frame], dtype=np.uint8)
    cache_indices, cache_scores = load_cache_sample(args.pointseg_cache.expanduser().resolve(), episode, frame)
    if len(cache_indices) != 10000:
        raise RuntimeError(f"Expected 10000 cached points, got {len(cache_indices)}")
    scene_count = xyz.shape[0] - int(args.gripper_points)
    old_gripper = cache_indices >= scene_count
    scene_indices = cache_indices[~old_gripper]
    scene_xyz = xyz[scene_indices]
    scene_rgb = score_color(cache_scores[~old_gripper], rgb[scene_indices])
    scene_scores = cache_scores[~old_gripper]
    replacement_count = int(old_gripper.sum())
    if replacement_count:
        # LIBERO's simplified convention uses width_percent in [0, 1].
        # The original raw action stores 0=closed and 1=open in column 7.
        action = np.load(root / "raw_expert_actions" / f"episode_{episode:06d}.npy", mmap_mode="r")
        width_percent = float(np.clip(action[frame, 7], 0.0, 1.0))
        physical_width = float(rlbench_physical_width_from_open_fraction(width_percent))
        gripper = create_rlbench_reap_points_from_physical_width(
            physical_width,
            np.zeros(6, dtype=np.float32),
            replacement_count,
            np.random.default_rng(episode * 100000 + frame),
            finger_length=float(args.finger_length),
        ).astype(np.float32)
    else:
        width_percent = None
        physical_width = None
        gripper = np.empty((0, 3), dtype=np.float32)
    out_xyz = np.vstack((scene_xyz, gripper))
    out_rgb = np.vstack((scene_rgb, np.tile([235, 70, 35], (len(gripper), 1)))).astype(np.uint8)
    out_scores = np.concatenate((scene_scores, np.full(len(gripper), np.nan, dtype=np.float32)))
    out_selected = np.concatenate((np.ones(len(scene_xyz), dtype=bool), np.zeros(len(gripper), dtype=bool)))
    out_kinds = np.concatenate((np.zeros(len(scene_xyz), dtype=np.uint8), np.ones(len(gripper), dtype=np.uint8)))
    if len(out_xyz) != 10000:
        raise RuntimeError(f"Output point count is {len(out_xyz)}, expected 10000")
    write_ply(args.output.expanduser().resolve(), out_xyz, out_rgb, out_scores, out_selected, out_kinds)
    manifest = {
        "output": str(args.output.expanduser().resolve()),
        "episode": episode,
        "frame": frame,
        "point_count": len(out_xyz),
        "cached_scene_points": len(scene_xyz),
        "replacement_gripper_points": len(gripper),
        "gripper_template": "rlbench_aligned_reap_short_two_finger_v4",
        "finger_length_m": float(args.finger_length),
        "gripper_width_percent": width_percent,
        "gripper_physical_width_m": physical_width,
        "foreground_score_source": str(args.pointseg_cache.expanduser().resolve()),
        "coordinate_frame": "current_eef",
    }
    manifest.update(canonical_reap_metadata())
    manifest["gripper_variant"] = "short_finger"
    manifest_path = args.output.expanduser().resolve().with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
