#!/usr/bin/env python3
"""Merge LeRobot v3 datasets together with SONG variable-length sidecars.

The upstream LeRobot aggregation utility merges Parquet/image metadata, but it
does not know about the per-episode point-cloud and world-pose sidecars used by
song_real_libero.  This wrapper merges the core dataset, reindexes those
sidecars, and optionally aliases semantically identical task strings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import load_stats, write_stats, write_tasks

EPISODIC_SIDECARS = {
    "point_clouds": ".zarr",
    "world_ee_poses": ".npy",
    "camera_motion": ".npy",
}


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Task alias must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"Task alias must have non-empty OLD and NEW: {value!r}")
        aliases[old] = new
    return aliases


def canonical_task(task: str, aliases: dict[str, str]) -> str:
    return aliases.get(str(task), str(task))


def replace_column(table: pa.Table, name: str, values: list | np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing Parquet column {name!r}")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def atomic_write_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    temporary.replace(path)


def scalar_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    return get_feature_stats(values, axis=0, keepdims=True)


def set_episode_scalar_stats(
    table: pa.Table,
    row_values: dict[str, list[np.ndarray]],
) -> pa.Table:
    for feature, values_per_row in row_values.items():
        stats_per_row = [scalar_stats(values) for values in values_per_row]
        for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            column = f"stats/{feature}/{stat_name}"
            if column not in table.column_names:
                continue
            values = [stats[stat_name].tolist() for stats in stats_per_row]
            table = replace_column(table, column, values)
    return table


def normalize_merged_tasks(root: Path, aliases: dict[str, str]) -> dict:
    meta = LeRobotDatasetMetadata("local/merged", root=root)
    old_tasks = [str(task) for task in meta.tasks.index.tolist()]
    canonical_tasks: list[str] = []
    for task in old_tasks:
        canonical = canonical_task(task, aliases)
        if canonical not in canonical_tasks:
            canonical_tasks.append(canonical)
    task_to_index = {task: index for index, task in enumerate(canonical_tasks)}
    old_to_new = {
        old_index: task_to_index[canonical_task(task, aliases)]
        for old_index, task in enumerate(old_tasks)
    }

    data_counts: Counter[int] = Counter()
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        old = np.asarray(table["task_index"], dtype=np.int64)
        new = np.fromiter((old_to_new[int(value)] for value in old), dtype=np.int64, count=len(old))
        data_counts.update(new.tolist())
        if not np.array_equal(old, new):
            table = replace_column(table, "task_index", new)
            atomic_write_parquet(table, path)

    episode_lengths: list[int] = []
    episode_indices: list[int] = []
    episode_task_indices: list[int] = []
    all_frame_indices: list[np.ndarray] = []
    all_global_indices: list[np.ndarray] = []
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        tasks = table["tasks"].to_pylist()
        normalized_tasks = [
            [canonical_task(task, aliases) for task in episode_tasks]
            for episode_tasks in tasks
        ]
        table = replace_column(table, "tasks", normalized_tasks)

        indices = np.asarray(table["episode_index"], dtype=np.int64)
        lengths = np.asarray(table["length"], dtype=np.int64)
        from_indices = np.asarray(table["dataset_from_index"], dtype=np.int64)
        to_indices = np.asarray(table["dataset_to_index"], dtype=np.int64)
        task_indices = np.asarray(
            [task_to_index[episode_tasks[0]] for episode_tasks in normalized_tasks],
            dtype=np.int64,
        )
        table = set_episode_scalar_stats(
            table,
            {
                "episode_index": [np.full(length, index, dtype=np.int64) for index, length in zip(indices, lengths, strict=True)],
                "index": [np.arange(start, stop, dtype=np.int64) for start, stop in zip(from_indices, to_indices, strict=True)],
                "frame_index": [np.arange(length, dtype=np.int64) for length in lengths],
                "task_index": [np.full(length, task, dtype=np.int64) for task, length in zip(task_indices, lengths, strict=True)],
            },
        )
        atomic_write_parquet(table, path)
        episode_lengths.extend(lengths.tolist())
        episode_indices.extend(indices.tolist())
        episode_task_indices.extend(task_indices.tolist())
        all_frame_indices.extend(np.arange(length, dtype=np.int64) for length in lengths)
        all_global_indices.extend(
            np.arange(start, stop, dtype=np.int64)
            for start, stop in zip(from_indices, to_indices, strict=True)
        )

    write_tasks(
        pd.DataFrame({"task_index": range(len(canonical_tasks))}, index=canonical_tasks),
        root,
    )
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = len(canonical_tasks)
    atomic_write_json(info_path, info)

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    episodes = np.asarray(episode_indices, dtype=np.int64)
    episode_tasks = np.asarray(episode_task_indices, dtype=np.int64)
    repeated_episode_indices = np.repeat(episodes, lengths)
    repeated_task_indices = np.repeat(episode_tasks, lengths)
    frame_indices = np.concatenate(all_frame_indices)
    global_indices = np.concatenate(all_global_indices)
    stats = load_stats(root)
    if stats is None:
        raise FileNotFoundError(root / "meta/stats.json")
    stats["episode_index"] = scalar_stats(repeated_episode_indices)
    stats["task_index"] = scalar_stats(repeated_task_indices)
    stats["frame_index"] = scalar_stats(frame_indices)
    stats["index"] = scalar_stats(global_indices)
    write_stats(stats, root)

    return {
        "old_tasks": old_tasks,
        "tasks": canonical_tasks,
        "old_to_new_task_index": {str(k): v for k, v in old_to_new.items()},
        "frames_per_task": {canonical_tasks[k]: int(v) for k, v in sorted(data_counts.items())},
    }


def link_or_copy_file(source: Path, destination: Path, copy_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def link_or_copy_tree(source: Path, destination: Path, copy_mode: str, modes: Counter[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for directory, subdirs, files in os.walk(source):
        relative = Path(directory).relative_to(source)
        target_directory = destination / relative
        for subdir in subdirs:
            (target_directory / subdir).mkdir(exist_ok=True)
        for filename in files:
            mode = link_or_copy_file(Path(directory) / filename, target_directory / filename, copy_mode)
            modes[mode] += 1


def merge_sidecars(
    roots: list[Path],
    episode_counts: list[int],
    destination: Path,
    copy_mode: str,
) -> dict:
    offsets = np.cumsum([0, *episode_counts[:-1]]).tolist()
    report: dict[str, dict] = {}
    point_counts: Counter[int] = Counter()
    for sidecar, suffix in EPISODIC_SIDECARS.items():
        target = destination / sidecar
        target.mkdir(parents=True, exist_ok=True)
        modes: Counter[str] = Counter()
        copied = 0
        source_meta: dict | None = None
        for root, count, offset in zip(roots, episode_counts, offsets, strict=True):
            source_dir = root / sidecar
            if not source_dir.is_dir():
                continue
            meta_path = source_dir / "meta.json"
            if meta_path.is_file() and source_meta is None:
                source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for old_index in range(count):
                source_path = source_dir / f"episode_{old_index:06d}{suffix}"
                if not source_path.exists():
                    if sidecar == "camera_motion":
                        continue
                    raise FileNotFoundError(source_path)
                new_index = offset + old_index
                target_path = target / f"episode_{new_index:06d}{suffix}"
                if source_path.is_dir():
                    link_or_copy_tree(source_path, target_path, copy_mode, modes)
                    if sidecar == "point_clouds":
                        group = zarr.open_group(str(source_path), mode="r")
                        point_counts[int(group["xyz"].shape[1])] += 1
                else:
                    modes[link_or_copy_file(source_path, target_path, copy_mode)] += 1
                copied += 1
        if source_meta is not None:
            if sidecar == "point_clouds":
                source_meta.update(
                    variable_num_points=True,
                    shape=[None, 6],
                    observed_num_points_per_episode={str(k): v for k, v in sorted(point_counts.items())},
                    merged_episode_storage="one packed Zarr group per episode; N may differ between episodes",
                )
            atomic_write_json(target / "meta.json", source_meta)
        report[sidecar] = {"episodes": copied, "file_modes": dict(modes)}
    report["point_clouds"]["num_points_distribution"] = {
        str(k): v for k, v in sorted(point_counts.items())
    }
    return report


def validate_inputs(roots: list[Path]) -> list[LeRobotDatasetMetadata]:
    metadata = [LeRobotDatasetMetadata(f"local/input_{i}", root=root) for i, root in enumerate(roots)]
    first = metadata[0]
    for root, meta in zip(roots[1:], metadata[1:], strict=True):
        if meta.fps != first.fps:
            raise ValueError(f"FPS mismatch: {root}: {meta.fps} != {first.fps}")
        if meta.features != first.features:
            raise ValueError(f"Feature schema mismatch: {root}")
        point_meta = json.loads((root / "point_clouds/meta.json").read_text(encoding="utf-8"))
        if point_meta.get("virtual_gripper_contract") != "rh20t_canonical_parallel_gripper_v3":
            raise ValueError(f"Input is not RH20T v3: {root}")
    first_point_meta = json.loads((roots[0] / "point_clouds/meta.json").read_text(encoding="utf-8"))
    if first_point_meta.get("virtual_gripper_contract") != "rh20t_canonical_parallel_gripper_v3":
        raise ValueError(f"Input is not RH20T v3: {roots[0]}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Repeat for each LeRobot root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="song_real_pointcloud_src_with_stagegen")
    parser.add_argument("--task-alias", action="append", default=[], help="Exact OLD=NEW task mapping.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    roots = [path.resolve() for path in args.input]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".incomplete-{uuid.uuid4().hex[:8]}")
    aliases = parse_aliases(args.task_alias)
    started = time.monotonic()
    try:
        metadata = validate_inputs(roots)
        aggregate_datasets(
            repo_ids=[f"local/input_{i}" for i in range(len(roots))],
            aggr_repo_id=args.repo_id,
            roots=roots,
            aggr_root=temporary,
        )
        task_report = normalize_merged_tasks(temporary, aliases)
        sidecar_report = merge_sidecars(
            roots,
            [meta.total_episodes for meta in metadata],
            temporary,
            args.copy_mode,
        )
        manifest = {
            "status": "complete",
            "repo_id": args.repo_id,
            "sources": [
                {
                    "root": str(root),
                    "episode_offset": int(offset),
                    "episodes": int(meta.total_episodes),
                    "frames": int(meta.total_frames),
                }
                for root, offset, meta in zip(
                    roots,
                    np.cumsum([0, *[m.total_episodes for m in metadata[:-1]]]),
                    metadata,
                    strict=True,
                )
            ],
            "tasks": task_report,
            "sidecars": sidecar_report,
            "point_cloud_storage": "variable length by episode",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(temporary / "merge_manifest.json", manifest)
        temporary.replace(output)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
