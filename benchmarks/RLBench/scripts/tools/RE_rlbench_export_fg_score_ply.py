#!/usr/bin/env python3
"""Export cached RLBench point clouds as standalone PLY diagnostics.

The cache already contains the candidate point indices and their PointSeg
foreground scores. ``fg_score`` selects the highest-scoring points from that
candidate set; it does not invent scores for the 40k points omitted by a
10k-point cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-cloud-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", required=True, help="Comma-separated episode IDs.")
    parser.add_argument("--selection", choices=("random", "fg_score"), default="random")
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def heatmap(scores: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    return np.rint(
        255.0
        * np.stack(
            (
                np.clip(2.0 * score, 0.0, 1.0),
                np.clip(1.0 - np.abs(2.0 * score - 1.0), 0.0, 1.0),
                np.clip(1.0 - 2.0 * score, 0.0, 1.0),
            ),
            axis=1,
        )
    ).astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, scores: np.ndarray) -> None:
    colors = heatmap(scores)
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("foreground_score", "<f4"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    vertices["foreground_score"] = scores
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment fg score heatmap: blue=0 yellow=0.5 red=1\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float foreground_score\nend_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header)
        vertices.tofile(file)


def load_samples(cache_dir: Path, episode_ids: set[int]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    samples = {}
    for shard in sorted((cache_dir / "rank_000").glob("shard_*")):
        episodes = np.load(shard / "episode_index.npy")
        matching_rows = np.flatnonzero(np.isin(episodes, tuple(episode_ids)))
        if matching_rows.size == 0:
            continue
        offsets = np.load(shard / "sample_offsets.npy")
        frames = np.load(shard / "frame_index.npy")
        indices = np.load(shard / "point_indices.npy")
        scores = np.load(shard / "foreground_score.npy").astype(np.float32)
        for row in matching_rows:
            episode, frame = episodes[row], frames[row]
            key = (int(episode), int(frame))
            start, end = int(offsets[row]), int(offsets[row + 1])
            if key in samples:
                raise ValueError(f"Duplicate cache sample: {key}")
            samples[key] = (indices[start:end], scores[start:end])
    return samples


def main() -> None:
    args = parse_args()
    episodes = {int(value) for value in args.episode_indices.split(",") if value.strip()}
    if not episodes or args.points <= 0:
        raise ValueError("episode-indices must be non-empty and points must be positive")
    samples = load_samples(args.cache_dir, episodes)
    rng = np.random.default_rng(args.seed)
    manifest = []
    for episode in sorted(episodes):
        cloud = zarr.open(str(args.point_cloud_dir / f"episode_{episode:06d}.zarr"), mode="r")
        frame_count = int(cloud["xyz"].shape[0])
        for frame in range(frame_count):
            key = (episode, frame)
            if key not in samples:
                raise ValueError(f"Missing cached frame: episode={episode}, frame={frame}")
            indices, scores = samples[key]
            count = min(args.points, len(indices))
            if args.selection == "fg_score":
                order = np.argsort(scores, kind="stable")[-count:]
                order = order[::-1]
            else:
                order = rng.choice(len(indices), size=count, replace=False)
            selected_indices = indices[order]
            selected_scores = scores[order]
            xyz = np.asarray(cloud["xyz"][frame], dtype=np.float32)[selected_indices]
            path = args.output_dir / f"episode_{episode:06d}" / f"frame_{frame:06d}_fg_score_{args.selection}.ply"
            write_ply(path, xyz, selected_scores)
        manifest.append({"episode_index": episode, "frames": frame_count, "points_per_frame": min(args.points, len(samples[(episode, 0)][0]))})
    (args.output_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "point_cloud_dir": str(args.point_cloud_dir),
                "cache_dir": str(args.cache_dir),
                "selection": args.selection,
                "selection_scope": "cached candidate points only",
                "seed": args.seed,
                "episodes": manifest,
                "coordinate_frame": "world",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {sum(item['frames'] for item in manifest)} PLY files to {args.output_dir}")


if __name__ == "__main__":
    main()
