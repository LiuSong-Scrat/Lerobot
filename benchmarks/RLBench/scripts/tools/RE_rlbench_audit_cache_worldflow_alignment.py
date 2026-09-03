#!/usr/bin/env python3
"""Read-only audit of LeRobot, WorldFlow sidecars, and PointSeg cache indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def episode_bounds(root: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "dataset_from_index", "dataset_to_index", "length"],
        )
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"])
            length = int(row["length"])
            if stop - start != length:
                raise RuntimeError(f"episode {episode}: bounds length mismatch")
            result[episode] = (start, stop)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    bounds = episode_bounds(root)

    sidecar_frames = 0
    sidecar_errors: list[str] = []
    episode_lengths: dict[int, int] = {}
    for episode, (start, stop) in sorted(bounds.items()):
        expected = stop - start
        achieved_path = root / "world_base_ee_poses" / f"episode_{episode:06d}.npy"
        target_path = root / "world_base_action_target_ee_poses" / f"episode_{episode:06d}.npy"
        world_path = root / "world_ee_poses" / f"episode_{episode:06d}.npy"
        shapes = []
        for path in (achieved_path, target_path, world_path):
            if not path.is_file():
                sidecar_errors.append(f"missing:{path}")
                continue
            shapes.append(tuple(np.load(path, mmap_mode="r").shape))
        if len(shapes) == 3 and any(shape != (expected, 9) for shape in shapes):
            sidecar_errors.append(f"episode={episode}:expected={(expected, 9)}:actual={shapes}")
        sidecar_frames += expected
        episode_lengths[episode] = expected

    cache_samples = 0
    expected_shard_start = 0
    index_errors: list[str] = []
    point_count_min: int | None = None
    point_count_max: int | None = None
    seen_episodes: set[int] = set()
    previous_dataset_index = -1
    for shard_number, shard in enumerate(manifest["shards"]):
        shard_dir = cache / shard["path"]
        declared_start = int(shard["start"])
        declared_length = int(shard["length"])
        if declared_start != expected_shard_start:
            index_errors.append(
                f"shard={shard_number}:start={declared_start}:expected={expected_shard_start}"
            )
        episodes = np.load(shard_dir / "episode_index.npy", mmap_mode="r")
        frames = np.load(shard_dir / "frame_index.npy", mmap_mode="r")
        indices = np.load(shard_dir / "dataset_index.npy", mmap_mode="r")
        offsets = np.load(shard_dir / "sample_offsets.npy", mmap_mode="r")
        if not (len(episodes) == len(frames) == len(indices) == declared_length):
            index_errors.append(f"shard={shard_number}:index_array_length_mismatch")
        if len(offsets) != declared_length + 1:
            index_errors.append(f"shard={shard_number}:offset_length_mismatch")
        else:
            counts = np.diff(offsets)
            local_min = int(np.min(counts)) if len(counts) else 0
            local_max = int(np.max(counts)) if len(counts) else 0
            point_count_min = local_min if point_count_min is None else min(point_count_min, local_min)
            point_count_max = local_max if point_count_max is None else max(point_count_max, local_max)
        for episode_value, frame_value, index_value in zip(episodes, frames, indices, strict=True):
            episode = int(episode_value)
            frame = int(frame_value)
            index = int(index_value)
            if episode not in bounds:
                index_errors.append(f"unknown_episode={episode}:index={index}")
                continue
            start, stop = bounds[episode]
            expected_index = start + frame
            if frame < 0 or frame >= stop - start:
                index_errors.append(f"episode={episode}:invalid_frame={frame}")
            if index != expected_index:
                index_errors.append(
                    f"episode={episode}:frame={frame}:index={index}:expected={expected_index}"
                )
            if index != previous_dataset_index + 1:
                index_errors.append(
                    f"noncontiguous_index={index}:expected={previous_dataset_index + 1}"
                )
            previous_dataset_index = index
            seen_episodes.add(episode)
        cache_samples += declared_length
        expected_shard_start += declared_length

    result = {
        "dataset_root": str(root),
        "cache_dir": str(cache),
        "dataset": {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
        },
        "worldflow": {
            "episodes": len(bounds),
            "frames": sidecar_frames,
            "errors": sidecar_errors[:20],
            "error_count": len(sidecar_errors),
        },
        "cache": {
            "manifest_num_samples": int(manifest["num_samples"]),
            "physical_shard_samples": cache_samples,
            "shards": len(manifest["shards"]),
            "last_shard_length": int(manifest["shards"][-1]["length"]),
            "episodes_seen": len(seen_episodes),
            "point_count_per_sample_min": point_count_min,
            "point_count_per_sample_max": point_count_max,
            "index_errors": index_errors[:20],
            "index_error_count": len(index_errors),
        },
    }
    expected_frames = int(info["total_frames"])
    result["pass"] = bool(
        sidecar_frames == expected_frames
        and cache_samples == expected_frames
        and int(manifest["num_samples"]) == expected_frames
        and len(bounds) == int(info["total_episodes"])
        and len(seen_episodes) == int(info["total_episodes"])
        and not sidecar_errors
        and not index_errors
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
