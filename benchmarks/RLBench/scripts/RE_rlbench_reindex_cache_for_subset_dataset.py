#!/usr/bin/env python3
"""Reindex an RLBench PointSeg cache for a standalone task-subset dataset.

The pseudo-label payloads are frame-content dependent, so a subset dataset that
reuses the exact source frames only needs its cache index arrays and provenance
rewritten.  Small changed files are backed up before atomic replacement.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def hardlink_backup(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def load_episode_bounds(dataset_root: Path) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No episode metadata Parquet files found")
    for path in paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "dataset_from_index", "dataset_to_index"],
        )
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            if episode in bounds:
                raise RuntimeError(f"Duplicate episode metadata for {episode}")
            bounds[episode] = (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
            )
    return bounds


def patch_episode_index_fields(value: object, mapping: dict[int, int]) -> object:
    if isinstance(value, list):
        return [patch_episode_index_fields(item, mapping) for item in value]
    if not isinstance(value, dict):
        return value
    patched = {}
    for key, item in value.items():
        if key == "episode_index":
            old_episode = int(item)
            if old_episode not in mapping:
                raise RuntimeError(f"Unknown episode_index={old_episode} in JSON metadata")
            patched[key] = mapping[old_episode]
        else:
            patched[key] = patch_episode_index_fields(item, mapping)
    return patched


def validate_cache(
    cache_dir: Path,
    target_root: Path,
    bounds: dict[int, tuple[int, int]],
    expected_samples: int,
) -> dict[str, int]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    target_text = str(target_root)
    args = manifest["args"]
    if args.get("dataset_root") != target_text:
        raise RuntimeError("Manifest dataset_root does not reference target dataset")
    if args.get("dataset_repo_id") != target_text:
        raise RuntimeError("Manifest dataset_repo_id does not reference target dataset")
    if args.get("point_cloud_dir") != str(target_root / "point_clouds"):
        raise RuntimeError("Manifest point_cloud_dir does not reference target dataset")

    selected = [int(value) for value in manifest["selected_episode_indices"]]
    if selected != sorted(bounds):
        raise RuntimeError("Manifest selected episodes do not match target dataset")

    sample_count = 0
    seen_episodes: set[int] = set()
    all_dataset_indices: list[np.ndarray] = []
    for shard in manifest["shards"]:
        shard_dir = cache_dir / shard["path"]
        episodes = np.load(shard_dir / "episode_index.npy", mmap_mode="r")
        frames = np.load(shard_dir / "frame_index.npy", mmap_mode="r")
        indices = np.load(shard_dir / "dataset_index.npy", mmap_mode="r")
        offsets = np.load(shard_dir / "sample_offsets.npy", mmap_mode="r")
        if not (len(episodes) == len(frames) == len(indices) == int(shard["length"])):
            raise RuntimeError(f"Index-array length mismatch in {shard_dir}")
        if len(offsets) != len(episodes) + 1 or not np.all(np.diff(offsets) == 20000):
            raise RuntimeError(f"Unexpected current-point count in {shard_dir}")
        for episode, frame, index in zip(episodes, frames, indices, strict=True):
            episode = int(episode)
            frame = int(frame)
            start, stop = bounds[episode]
            if frame < 0 or start + frame >= stop:
                raise RuntimeError(f"Invalid frame {frame} for episode {episode}")
            if int(index) != start + frame:
                raise RuntimeError(
                    f"dataset_index mismatch: episode={episode}, frame={frame}, index={index}"
                )
            seen_episodes.add(episode)
        all_dataset_indices.append(np.asarray(indices))
        sample_count += len(episodes)

    if sample_count != expected_samples or sample_count != int(manifest["num_samples"]):
        raise RuntimeError(f"Unexpected cache sample count: {sample_count}")
    concatenated = np.concatenate(all_dataset_indices)
    if not np.array_equal(concatenated, np.arange(expected_samples, dtype=concatenated.dtype)):
        raise RuntimeError("Cache dataset_index is not globally contiguous")
    if seen_episodes != set(bounds):
        raise RuntimeError("Cache does not cover every target episode")
    return {
        "episodes": len(seen_episodes),
        "samples": sample_count,
        "shards": len(manifest["shards"]),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    mapping_path = dataset_root / "meta" / "rlbench_subset_mapping.json"
    mapping_meta = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping = {
        int(old): int(new)
        for old, new in mapping_meta["old_to_new_episode_index"].items()
    }
    bounds = load_episode_bounds(dataset_root)
    if set(mapping.values()) != set(bounds):
        raise RuntimeError("Subset mapping does not match target episode metadata")
    expected_samples = max(stop for _, stop in bounds.values())

    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("args", {}).get("dataset_root") == str(dataset_root):
        result = validate_cache(cache_dir, dataset_root, bounds, expected_samples)
        print(json.dumps({"status": "already_reindexed", **result}, indent=2))
        return

    selected_old = [int(value) for value in manifest["selected_episode_indices"]]
    if set(selected_old) != set(mapping):
        raise RuntimeError("Cache selected episodes do not match subset source mapping")
    source_root = Path(mapping_meta["source_dataset"]).expanduser().resolve()
    cached_root = Path(manifest["args"]["dataset_root"]).expanduser().resolve()
    if cached_root != source_root:
        raise RuntimeError(f"Cache source {cached_root} does not match mapping source {source_root}")

    backup_root = cache_dir / "reindex_backup_old_indices"
    backup_root.mkdir(exist_ok=True)
    hardlink_backup(manifest_path, backup_root / "manifest.before_standalone_reindex.json")

    for shard in manifest["shards"]:
        shard_dir = cache_dir / shard["path"]
        episode_path = shard_dir / "episode_index.npy"
        dataset_path = shard_dir / "dataset_index.npy"
        frame_path = shard_dir / "frame_index.npy"
        backup_shard = backup_root / shard["path"]
        hardlink_backup(episode_path, backup_shard / episode_path.name)
        hardlink_backup(dataset_path, backup_shard / dataset_path.name)

        old_episodes = np.load(episode_path)
        frames = np.load(frame_path)
        if not set(np.unique(old_episodes).tolist()).issubset(mapping):
            raise RuntimeError(f"Unexpected old episode index in {shard_dir}")
        new_episodes = np.fromiter(
            (mapping[int(value)] for value in old_episodes),
            dtype=old_episodes.dtype,
            count=len(old_episodes),
        )
        new_indices = np.fromiter(
            (bounds[int(episode)][0] + int(frame) for episode, frame in zip(new_episodes, frames, strict=True)),
            dtype=np.int64,
            count=len(frames),
        )
        atomic_npy(episode_path, new_episodes)
        atomic_npy(dataset_path, new_indices)

    json_paths = [cache_dir / "terminal_soft_continuity_summary.json"]
    json_paths.extend(sorted((cache_dir / "visualizations").rglob("*_stats.json")))
    for path in json_paths:
        relative = path.relative_to(cache_dir)
        hardlink_backup(path, backup_root / "json" / relative)
        value = json.loads(path.read_text(encoding="utf-8"))
        atomic_json(path, patch_episode_index_fields(value, mapping))

    visualization_dirs = sorted(
        (path for path in (cache_dir / "visualizations").glob("*/episode_*") if path.is_dir()),
        reverse=True,
    )
    for path in visualization_dirs:
        old_episode = int(path.name.removeprefix("episode_"))
        if old_episode not in mapping:
            raise RuntimeError(f"Unknown visualization episode directory: {path}")
        destination = path.with_name(f"episode_{mapping[old_episode]:06d}")
        if destination.exists():
            raise FileExistsError(destination)
        path.rename(destination)

    new_selected = sorted(mapping.values())
    manifest["selected_episode_indices"] = new_selected
    manifest["args"]["dataset_repo_id"] = str(dataset_root)
    manifest["args"]["dataset_root"] = str(dataset_root)
    manifest["args"]["point_cloud_dir"] = str(dataset_root / "point_clouds")
    manifest["args"]["episode_indices"] = ",".join(map(str, new_selected))
    manifest["standalone_subset_reindex"] = {
        "source_dataset": str(source_root),
        "target_dataset": str(dataset_root),
        "mapping_file": str(mapping_path),
        "pseudo_label_payload": "unchanged; source and target frames are identical",
        "index_arrays": "episode_index and dataset_index rewritten",
        "backup": str(backup_root),
    }
    atomic_json(manifest_path, manifest)

    result = validate_cache(cache_dir, dataset_root, bounds, expected_samples)
    print(json.dumps({"status": "reindexed", **result}, indent=2))


if __name__ == "__main__":
    main()
