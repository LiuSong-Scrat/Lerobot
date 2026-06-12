#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from libero_collect_dataset import write_ascii_ply_lines, write_ascii_ply_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PLY previews from a Song point-cloud LeRobot dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", default="", help="Comma-separated frame ids. If empty, use stride/count.")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def parse_frame_ids(frames: str, total: int, stride: int, count: int) -> list[int]:
    if frames.strip():
        ids = [int(item.strip()) for item in frames.split(",") if item.strip()]
        return [idx for idx in ids if 0 <= idx < total]
    candidates = list(range(0, total, max(1, stride)))
    if len(candidates) > count:
        pick = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        candidates = [candidates[int(i)] for i in pick]
    return candidates


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (args.output_dir or dataset_root / "visualizations").expanduser().resolve()
    episode = int(args.episode)

    pc_path = dataset_root / "point_clouds" / f"episode_{episode:06d}.npy"
    pose_path = dataset_root / "world_ee_poses" / f"episode_{episode:06d}.npy"
    if not pc_path.exists():
        raise FileNotFoundError(f"Missing point cloud episode file: {pc_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing world pose episode file: {pose_path}")

    point_clouds = np.load(pc_path, mmap_mode="r")
    world_poses = np.load(pose_path, mmap_mode="r")
    frame_ids = parse_frame_ids(args.frames, len(point_clouds), args.stride, args.count)
    episode_dir = output_dir / f"episode_{episode:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_ids:
        write_ascii_ply_points(episode_dir / f"frame_{frame_idx:04d}_point_cloud_eff.ply", np.asarray(point_clouds[frame_idx]))
    write_ascii_ply_lines(episode_dir / "world_ee_trajectory.ply", np.asarray(world_poses[:, :3]))
    with open(episode_dir / "preview.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "episode": episode,
                "frames": frame_ids,
                "point_cloud_file": str(pc_path),
                "world_pose_file": str(pose_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote preview to {episode_dir}")


if __name__ == "__main__":
    main()
